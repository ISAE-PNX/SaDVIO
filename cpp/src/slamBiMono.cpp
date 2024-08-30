#include "isaeslam/optimizers/AngularAdjustmentCERESAnalytic.h"
#include "isaeslam/slamCore.h"

namespace isae {

bool SLAMBiMono::init() {

    // get first frame and set keyframe
    _frame = _slam_param->getDataProvider()->next();
    while (_frame->getSensors().empty()) {
        _frame = _slam_param->getDataProvider()->next();
    }

    // Prior on the first frame, it is set as the origin
    _frame->setWorld2FrameTransform(Eigen::Affine3d::Identity());
    _frame->setPrior(Eigen::Affine3d::Identity(), 100 * Vector6d::Ones());

    // detect all features on all sensors
    detectFeatures(_frame->getSensors().at(0));

    // Track features in frame
    trackFeatures(_frame->getSensors().at(0),
                  _frame->getSensors().at(1),
                  _matches_in_frame,
                  _matches_in_frame_lmk,
                  _frame->getSensors().at(0)->getFeatures());

    // Filter matches in frame
    _matches_in_frame = epipolarFiltering(_frame->getSensors().at(0), _frame->getSensors().at(1), _matches_in_frame);

    // init the velocity
    _6d_velocity = 0.00001 * Vector6d::Ones();

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

    profiling();

    // Send frame to optimizer
    _frame_to_optim = _frame;
    _is_init        = true;
    _nkeyframes++;

    return true;
}

bool SLAMBiMono::frontEndStep() {

    // Get next frame
    _frame = _slam_param->getDataProvider()->next();

    // Ignore frames without images
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

        // Single Frame Bundle Adjustment
        isae::timer::tic();
        _slam_param->getOptimizerFront()->singleFrameOptimization(_frame);
        _avg_frame_opt_t = (_avg_frame_opt_t * (_nframes - 1) + isae::timer::silentToc()) / _nframes;
        _lmk_inmap       = (_lmk_inmap * (_nframes - 1) + _frame->getInMapLandmarksNumber()) / _nframes;

        // Compute velocity and motion model
        _6d_velocity =
            (geometry::se3_RTtoVec6d(getLastKF()->getWorld2FrameTransform() * _frame->getFrame2WorldTransform())) / dt;
    } else {

        // If the prediction is wrong, we reinitialize the odometry from the last KF:
        // - A KF is voted
        // - All matches in time are removed
        // Can be improved: redetect new points, retrack old features....

        outlierRemoval();
        _frame->setKeyFrame();
    }

    if (shouldInsertKeyframe(_frame)) {
        
        // Frame is added
        _nkeyframes++;

        // Repopulate in the case of klt tracking
        typed_vec_features new_features;
        if (_slam_param->_config.tracker == "klt") {
            isae::timer::tic();
            cleanFeatures(_frame);
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

        // Track features in frame
        isae::timer::tic();
        uint nmatches_in_frame = trackFeatures(_frame->getSensors().at(0),
                                               _frame->getSensors().at(1),
                                               _matches_in_frame,
                                               _matches_in_frame_lmk,
                                               _frame->getSensors().at(0)->getFeatures());

        // Epipolar Filtering for matches in frame
        _matches_in_frame =
            epipolarFiltering(_frame->getSensors().at(0), _frame->getSensors().at(1), _matches_in_frame);
        _matches_in_frame_lmk =
            epipolarFiltering(_frame->getSensors().at(0), _frame->getSensors().at(1), _matches_in_frame_lmk);

        // Update tracked landmarks
        updateLandmarks(_matches_in_frame_lmk);

        _avg_matches_frame = (_avg_matches_frame * (_nkeyframes - 1) + nmatches_in_frame) / _nkeyframes;
        _avg_match_frame_t = (_avg_match_frame_t * (_nkeyframes - 1) + isae::timer::silentToc()) / _nkeyframes;

        // Landmark Initialization:
        // - Triangulate new points : LR + (n-1) / n
        // - Optimize points only because optimal mid-point is not optimal for LM
        // - Reject outliers with reprojection error
        isae::timer::tic();
        initLandmarks(_frame);
        _slam_param->getOptimizerFront()->landmarkOptimization(_frame);
        _avg_lmk_init_t = (_avg_lmk_init_t * (_nkeyframes - 1) + isae::timer::silentToc()) / _nkeyframes;

        // Wait the end of optim
        while (_frame_to_optim != nullptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        _frame_to_optim = _frame;

    } else {
        // If no KF is voted, the frame is discarded and the landmarks are cleaned
        _frame->cleanLandmarks();
    }

    // Send the frame to the viewer
    _frame_to_display = _frame;

    return true;
}

bool SLAMBiMono::backEndStep() {

    // Optimize when a frame is declared as optimizable
    if (_frame_to_optim) {

        // Add frame to local map
        _local_map->addFrame(_frame_to_optim);
        _frame_to_optim->setKeyFrame();

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