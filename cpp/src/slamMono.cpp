#include "isaeslam/slamCore.h"

namespace isae {

bool SLAMMono::init() {

    // get first frame with images and set keyframe
    _frame = _slam_param->getDataProvider()->next();
    while (_frame->getSensors().empty()) {
        _frame = _slam_param->getDataProvider()->next();
    }
    _frame->setKeyFrame();
    std::shared_ptr<Frame> kf_init = _frame;

    // Prior on the first frame, it is set as the origin or the last kf's pose
    if (getLastKF()) {
        _frame->setWorld2FrameTransform(getLastKF()->getWorld2FrameTransform());
        _frame->setPrior(getLastKF()->getWorld2FrameTransform(), 100 * Vector6d::Ones());
    } else {
        _frame->setWorld2FrameTransform(Eigen::Affine3d::Identity());
        _frame->setPrior(Eigen::Affine3d::Identity(), 100 * Vector6d::Ones());
    }

    // detect all features on all sensors
    detectFeatures(_frame->getSensors().at(0));

    // Init with essential matrix
    EpipolarPoseEstimator essential_ransac;
    bool ready_to_init = false;

    // Track features until enough parallax
    while (!ready_to_init) {

        // Get next frames with images
        _frame = _slam_param->getDataProvider()->next();
        if (_frame->getSensors().empty())
            continue;
        _nframes++;

        trackFeatures(kf_init->getSensors().at(0),
                      _frame->getSensors().at(0),
                      _matches_in_time,
                      _matches_in_time_lmk,
                      kf_init->getSensors().at(0)->getFeatures());

        // Return false and delete KF if there is not enough matches
        if (_matches_in_time["pointxd"].size() < 20) {
            std::cout << "Not enough matches and / or parallax, restarting init" << std::endl;
            return false;
        }

        // Essential matrix filtering
        Eigen::Affine3d T_last_curr;
        Eigen::MatrixXd cov;
        essential_ransac.estimateTransformBetween(kf_init, _frame, _matches_in_time["pointxd"], T_last_curr, cov);

        // Set the scale to 10cm (arbitrary)
        T_last_curr.translation() /= 10;
        _frame->setWorld2FrameTransform((kf_init->getFrame2WorldTransform() * T_last_curr).inverse());
        outlierRemoval();

        // Parallax computation
        double n_matches = 0;
        for (auto tmatch : _matches_in_time) {
            n_matches += tmatch.second.size();
        }

        double avg_parallax = 0;
        for (auto tmatch : _matches_in_time) {
            for (auto match : tmatch.second) {
                avg_parallax += std::acos(match.first->getBearingVectors().at(0).transpose() *
                                          match.second->getBearingVectors().at(0)) /
                                n_matches;
            }
        }
        avg_parallax = avg_parallax * 180 / M_PI;

        if (avg_parallax > 3)
            ready_to_init = true;

        // Send the frame to the viewer
        _frame_to_display = _frame;
    }

    // Add kf init in the local map
    _local_map->addFrame(kf_init);

    // Compute velocity
    double dt = (_frame->getTimestamp() - kf_init->getTimestamp()) * 1e-9;
    _6d_velocity =
        (geometry::se3_RTtoVec6d(kf_init->getWorld2FrameTransform() * _frame->getFrame2WorldTransform())) / dt;

    // init first landmarks
    initLandmarks(_frame);
    _slam_param->getOptimizerFront()->landmarkOptimization(_frame);

    // Create the 3D mesh
    if (_slam_param->_config.mesh3D) {
        _mesher->addNewKF(_frame);
    }

    // Ignore features that were not triangulated
    cleanFeatures(_frame);
    detectFeatures(_frame->getSensors().at(0));

    // Send frame to optimizer
    profiling();
    _frame_to_optim = _frame;
    _is_init        = true;
    _nkeyframes++;

    return true;
}

bool SLAMMono::frontEndStep() {

    // Get next frame with images
    _frame = _slam_param->getDataProvider()->next();
    if (_frame->getSensors().empty())
        return true;
    _nframes++;

    // Predict pose with constant velocity model
    double dt = (_frame->getTimestamp() - getLastKF()->getTimestamp()) * 1e-9;
    Eigen::Affine3d T_f_w =
        geometry::se3_Vec6dtoRT(_6d_velocity * dt).inverse() * getLastKF()->getWorld2FrameTransform();
    _frame->setWorld2FrameTransform(T_f_w);

    // Detect all features (only if we use the matcher)
    isae::timer::tic();
    if (_slam_param->_config.tracker == "matcher") {
        detectFeatures(_frame->getSensors().at(0));
    }
    _avg_detect_t = (_avg_detect_t * (_nframes - 1) + isae::timer::silentToc()) / _nframes;

    // Match or track the features in time
    uint nmatches_in_time;
    isae::timer::tic();
    if (_slam_param->_config.tracker == "klt") {
        nmatches_in_time = trackFeatures(getLastKF()->getSensors().at(0),
                                         _frame->getSensors().at(0),
                                         _matches_in_time,
                                         _matches_in_time_lmk,
                                         getLastKF()->getSensors().at(0)->getFeatures());
    } else {
        nmatches_in_time = matchFeatures(getLastKF()->getSensors().at(0),
                                         _frame->getSensors().at(0),
                                         _matches_in_time,
                                         _matches_in_time_lmk,
                                         getLastKF()->getSensors().at(0)->getFeatures());
    }

    _avg_matches_time = (_avg_matches_time * (_nframes - 1) + nmatches_in_time) / _nframes;
    _avg_match_time_t = (_avg_match_time_t * (_nframes - 1) + isae::timer::silentToc()) / _nframes;

    // Get P3d from n-1 matched features and estimate 3D pose from 2D (n)/3D (n-1) matchings
    // to predict pose. Also remove outliers from tracks_in_time vector
    isae::timer::tic();
    bool good_it   = predict(_frame);
    _avg_predict_t = (_avg_predict_t * (_nframes - 1) + isae::timer::silentToc()) / _nframes;

    if (good_it) {
        _successive_fails = 0;

        // Epipolar Filtering for matches in time
        isae::timer::tic();
        int removed_matching_nb = _matches_in_time["pointxd"].size() + _matches_in_time_lmk["pointxd"].size();
        _matches_in_time =
            epipolarFiltering(getLastKF()->getSensors().at(0), _frame->getSensors().at(0), _matches_in_time);
        _matches_in_time_lmk =
            epipolarFiltering(getLastKF()->getSensors().at(0), _frame->getSensors().at(0), _matches_in_time_lmk);
        removed_matching_nb -= _matches_in_time["pointxd"].size() + _matches_in_time_lmk["pointxd"].size();
        _removed_feat = (_removed_feat * (_nframes - 1) + removed_matching_nb) / _nframes;
        _avg_filter_t = (_avg_filter_t * (_nframes - 1) + isae::timer::silentToc()) / _nframes;

        // Remove Outliers in case of klt
        if (_slam_param->_config.tracker == "klt") {
            isae::timer::tic();
            outlierRemoval();
            _avg_clean_t = (_avg_clean_t * (_nframes - 1) + isae::timer::silentToc()) / _nframes;
        }

        // Update tracked landmarks
        updateLandmarks(_matches_in_time_lmk);

       // Single Frame ESKF Update
        isae::timer::tic();

        Eigen::MatrixXd cov;
        Eigen::Affine3d T_last_curr, T_w_f;
        T_last_curr = getLastKF()->getWorld2FrameTransform() * _frame->getFrame2WorldTransform();
        ESKFEstimator eskf;
        eskf.estimateTransformBetween(getLastKF(), _frame, _matches_in_time_lmk["pointxd"], T_last_curr, cov);
        T_w_f = getLastKF()->getFrame2WorldTransform() * T_last_curr;
        _frame->setdTCov(cov);
        _frame->setWorld2FrameTransform(T_w_f.inverse());

        _avg_frame_opt_t = (_avg_frame_opt_t * (_nframes - 1) + isae::timer::silentToc()) / _nframes;
        _lmk_inmap       = (_lmk_inmap * (_nframes - 1) + _frame->getLandmarks()["pointxd"].size()) / _nframes;

        // Compute velocity and motion model
        _6d_velocity =
            (geometry::se3_RTtoVec6d(getLastKF()->getWorld2FrameTransform() * _frame->getFrame2WorldTransform())) / dt;
    } else {

        // If the prediction is wrong, we reinitialize the odometry from the last KF:
        // - A KF is voted
        // - All matches in time are removed
        // Can be improved: redetect new points, retrack old features....

        _successive_fails++;
        outlierRemoval();
        _frame->setKeyFrame();
    }

    if (shouldInsertKeyframe(_frame)) {

        // Keyrame is added
        _nkeyframes++;

        // Landmark Initialization:
        // - Triangulate new points : LR + (n-1) / n
        // - Optimize points only because optimal mid-point is not optimal for LM
        // - Reject outliers with reprojection error
        isae::timer::tic();
        initLandmarks(_frame);
        _slam_param->getOptimizerFront()->landmarkOptimization(_frame);
        _avg_lmk_init_t = (_avg_lmk_init_t * (_nkeyframes - 1) + isae::timer::silentToc()) / _nkeyframes;

        // Repopulate in the case of klt tracking
        typed_vec_features new_features;
        if (_slam_param->_config.tracker == "klt") {
            isae::timer::tic();
            new_features  = detectFeatures(_frame->getSensors().at(0));
            _avg_detect_t = (_avg_detect_t * (_nkeyframes - 1) + isae::timer::silentToc()) / _nkeyframes;
        }

        // Recover Map Landmark
        isae::timer::tic();
        _map_mutex.lock();
        uint resu = recoverFeatureFromMapLandmarks(_local_map, _frame);
        _map_mutex.unlock();

        _avg_lmk_resur_t = (_avg_lmk_resur_t * (_nkeyframes - 1) + isae::timer::silentToc()) / _nkeyframes;
        _avg_resur_lmk   = (_avg_lmk_resur_t * (_nkeyframes - 1) + resu) / _nkeyframes;

        // Wait the end of optim
        while (_frame_to_optim != nullptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        _frame_to_optim = _frame;

    } else {

        // If no KF is voted, the frame is discarded and the landmarks are cleaned
        _frame->cleanLandmarks();
    }

    // Init the SLAM again in case of successive failures or if the frame is too far from the last KF
    if ((getLastKF()->getWorld2FrameTransform() * _frame->getFrame2WorldTransform()).translation().norm() > 10 ||
        (_successive_fails > 5)) {

        _is_init = false;

        return true;
    }

    // Send the frame to the viewer
    _frame_to_display = _frame;

    return true;
}

bool SLAMMono::backEndStep() {

    if (_frame_to_optim) {

        // Add frame to local map
        _local_map->addFrame(_frame_to_optim);

        // 3D Mesh update
        if (_slam_param->_config.mesh3D) {
            _mesher->addNewKF(_frame_to_optim);
            _mesh_to_display = _mesher->_mesh_3d;
        }

        // Marginalization (+ sparsification) of the last frame
        if (_local_map->getMarginalizationFlag()) {
            isae::timer::tic();
            if (_slam_param->_config.marginalization == 1)
                _slam_param->getOptimizerBack()->marginalize(_local_map->getFrames().at(0),
                                                             _local_map->getFrames().at(1),
                                                             _slam_param->_config.sparsification == 1);
            _avg_marg_t = (_avg_marg_t * (_nkeyframes - 1) + isae::timer::silentToc()) / _nkeyframes;
            _global_map->addFrame(_local_map->getFrames().at(0));

            _map_mutex.lock();
            _local_map->discardLastFrame();
            _map_mutex.unlock();
        }

        // Optimize Local Map
        isae::timer::tic();
        _slam_param->getOptimizerBack()->localMapBA(_local_map, _local_map->getFixedFrameNumber());
        _avg_wdw_opt_t = (_avg_wdw_opt_t * (_nkeyframes - 1) + isae::timer::silentToc()) / _nkeyframes;
        profiling();

        // Reset frame to optim
        _frame_to_optim = nullptr;

        // Send the local map to the viewer
        _local_map_to_display = _local_map;
    }

    return true;
}

} // namespace isae