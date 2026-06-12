
#include "isaeslam/slamCore.h"
#include <opencv2/core.hpp>

namespace isae {

bool SLAMBiMono::init() {
    std::cout << "Initialize BIMONO..." << std::endl; 
    if (_successive_fails) {
        std::cerr << "Re-initialization ... " << _successive_fails << std::endl;
        // throw std::runtime_error("THROW RE-INIT");
    }

    std::cout << "Create frame..." << std::endl; 
    // get first frame and set keyframe
    _frame = _slam_param->getDataProvider()->next();
    while (_frame->getSensors().empty()) {
        _frame = _slam_param->getDataProvider()->next();
    }

    std::cout << "Create prior..." << std::endl; 
    // Prior on the first frame, it is set as the origin
    _frame->setWorld2FrameTransform(Eigen::Affine3d::Identity());
    _frame->setPrior(Eigen::Affine3d::Identity(), 100 * Vector6d::Ones());

    std::cout << "Create features..." << std::endl; 
    // detect all features on all sensors
    detectFeatures(_frame->getSensors().at(0));

    // Track features in frame
    trackFeatures(_frame->getSensors().at(0),
                  _frame->getSensors().at(1),
                  _matches_in_frame,
                  _matches_in_frame_lmk,
                  _frame->getSensors().at(0)->getFeatures());

    std::cout << "Create matches..." << std::endl; 
    // Filter matches in frame
    _matches_in_frame = epipolarFiltering(_frame->getSensors().at(0), _frame->getSensors().at(1), _matches_in_frame);

    // init the velocity
    _6d_velocity = 0.00001 * Vector6d::Ones();

    std::cout << "Create landmarks..." << std::endl; 
    // init first landmarks
    initLandmarks(_frame);
    std::cout << "Optimize landmarks..." << std::endl; 
    _slam_param->getOptimizerFront()->landmarkOptimization(_frame);
    _local_map_to_display.reset();

    std::cout << "Create mesh..." << std::endl; 
    // Create the 3D mesh
    if (_slam_param->_config.mesh3D) {
        _mesher->addNewKF(_frame);
    }

    std::cout << "Clean..." << std::endl; 
    // Ignore features that were not triangulated
    cleanFeatures(_frame);
    detectFeatures(_frame->getSensors().at(0));

    profiling();

    // Send frame to optimizer
    _map_mutex.lock();
    _frame_to_optim_queue.push(_frame);
    _map_mutex.unlock();
    _is_init        = true;
    _successive_fails = 0;
    _nkeyframes++;
    dispMAll();

    std::cout << "Initialized BIMONO!" << std::endl; 
    return true;
}

bool SLAMBiMono::frontEndStep() {

    std::cout << "Frontend step" << std::endl;
    // Get next frame
    _frame = _slam_param->getDataProvider()->next();
    std::cout << "## # # # ## NEXT frame ## # # # ##" << std::endl;

    // Ignore frames without images
    if (_frame->getSensors().empty())
        return true;

    if (!getLastKF())
        return true;

    _nframes++;

    // std::cout << "DEBUG: frame " << _nframes << "/" << _nkeyframes << std::endl;
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
    // std::cout << "SLAMCORE DEBUG: Features in img0: " << _frame->getSensors().at(0)->getFeatures()["pointxd"].size() << std::endl;
    // std::cout << "SLAMCORE DEBUG: track w.r.t. prior KF"
    //                 << " [frontEndStep]" << std::endl;
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
    // dispMiTl();
    // std::cout << "Features in img0: " << _frame->getSensors().at(0)->getFeatures()["pointxd"].size() << std::endl;

    _avg_matches_time = (_avg_matches_time * (_nframes - 1) + nmatches_in_time) / _nframes;
    _avg_match_time_t = (_avg_match_time_t * (_nframes - 1) + isae::timer::silentToc()) / _nframes;

    // Get P3d from n-1 matched features and estimate 3D pose from 2D (n)/3D (n-1) matchings
    // to predict pose. Also remove outliers from tracks_in_time vector
    isae::timer::tic();
    bool good_it   = predict(_frame);
    _avg_predict_t = (_avg_predict_t * (_nframes - 1) + isae::timer::silentToc()) / _nframes;
    // dispMAll();
    // std::cout << "MiT-L:   " << _matches_in_time_lmk["pointxd"].size() << std::endl;

    // std::cout << "Predict is: " << good_it << std::endl;
    
    if (good_it) {
        _successive_fails = 0;

        // Epipolar Filtering for matches in time
        isae::timer::tic();
        int removed_matching_nb = _matches_in_time["pointxd"].size();
        
        _matches_in_time =
            epipolarFiltering(getLastKF()->getSensors().at(0), _frame->getSensors().at(0), _matches_in_time);
        removed_matching_nb -= _matches_in_time["pointxd"].size();
        _removed_feat = (_removed_feat * (_nframes - 1) + removed_matching_nb) / _nframes;
        _avg_filter_t = (_avg_filter_t * (_nframes - 1) + isae::timer::silentToc()) / _nframes;

        // dispMiT();

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

        // dispMiTl();

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


    // _frame->setKeyFrame(); // DEBUG
    if (shouldInsertKeyframe(_frame)) {

        std::stringstream  msg_vote;
        msg_vote << "Voted KeyFrame " << _frame->_id << std::endl;
        std::cout << msg_vote.str();

        // Frame is added
        _nkeyframes++;

        // std::cout << "Features in img0: " << _frame->getSensors().at(0)->getFeatures()["pointxd"].size() << std::endl;
        // Repopulate in the case of klt tracking
        typed_vec_features new_features;
        if (_slam_param->_config.tracker == "klt") {
            isae::timer::tic();
            new_features  = detectFeatures(_frame->getSensors().at(0));
            _avg_detect_t = (_avg_detect_t * (_nkeyframes - 1) + isae::timer::silentToc()) / _nkeyframes;
        }
        // std::cout << "Features in img0: " << _frame->getSensors().at(0)->getFeatures()["pointxd"].size() << std::endl;
        // Recover Map Landmark
        isae::timer::tic();
        _map_mutex.lock();
        uint resu = recoverFeatureFromMapLandmarks(_local_map, _frame);
        _map_mutex.unlock();
        
        // std::cout << "SLAMCORE DEBUG: Resurrected lmks   " << resu << std::endl;

        _avg_lmk_resur_t = (_avg_lmk_resur_t * (_nkeyframes - 1) + isae::timer::silentToc()) / _nkeyframes;
        _avg_resur_lmk   = (_avg_lmk_resur_t * (_nkeyframes - 1) + resu) / _nkeyframes;

        // Track features in frame
        isae::timer::tic();
        std::cout << "SLAMCORE DEBUG: match w.r.t. 2nd cam"
                    << " [frontEndStep]" << std::endl;        
        _map_mutex.lock();
        uint nmatches_in_frame = trackFeatures(_frame->getSensors().at(0),
                                               _frame->getSensors().at(1),
                                               _matches_in_frame,
                                               _matches_in_frame_lmk,
                                               _frame->getSensors().at(0)->getFeatures());                                               
        _map_mutex.unlock();
        // dispMAll();
        // _feature_evolution->
         std::cout << "Features in img0: " << _frame->getSensors().at(0)->getFeatures()["pointxd"].size() << std::endl;
        //  std::cout << "Features in img1: " << _frame->getSensors().at(1)->getFeatures()["pointxd"].size() << std::endl;

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
        _map_mutex.lock();
        initLandmarks(_frame);        
        _map_mutex.unlock();
        _map_mutex.lock();
        _slam_param->getOptimizerFront()->landmarkOptimization(_frame);
        _map_mutex.unlock();
        _avg_lmk_init_t = (_avg_lmk_init_t * (_nkeyframes - 1) + isae::timer::silentToc()) / _nkeyframes;

        // Wait the end of optim if frontend is too fast
        bool warning_sent = false;
        while (_frame_to_optim_queue.size() >= _slam_param->_config.max_kf_number) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (!warning_sent) {
                std::cerr   << "WARNING: SLAM Backend cannot keep up! " << _frame_to_optim_queue.size() 
                            << " frames waiting to be processed." << std::endl;
                warning_sent = true;
            }
        }
        _map_mutex.lock();
        _frame_to_optim_queue.push(_frame);
        _map_mutex.unlock();
        // std::stringstream  msg;
        // msg << "Features in img0: " << _frame->getSensors().at(0)->getFeatures()["pointxd"].size() << std::endl;
        // std::cout << msg.str();

    } else {
        // If no KF is voted, the frame is discarded and the landmarks are cleaned
        _map_mutex.lock();
        _frame->cleanLandmarks();
        _map_mutex.unlock();
    }

    // dispMAll();
    // Send the frame to the viewer
    _frame_to_display = _frame;

    // Init the SLAM again in case of successive failures or if the frame is too far from the last KF
    if ((getLastKF()->getWorld2FrameTransform() * _frame->getFrame2WorldTransform()).translation().norm() > 10 ||
        (_successive_fails > 5)) {

        std::cout << "Detected 5 or more successive fails: Reset local map (" << _successive_fails << ")" << std::endl;

        _is_init = false;
        _local_map->reset();
        resetLandmarks();
        _slam_param->getOptimizerBack()->resetMarginalization();
        std::cout << "Detected 5 or more successive fails: Reset local map (" << _successive_fails << ")" << std::endl;
        return true;
    }
    return true;
}

bool SLAMBiMono::backEndStep() {

    // Optimize when a frame is declared as optimizable
    _map_mutex.lock();
    for(; !_frame_to_optim_queue.empty(); _frame_to_optim_queue.pop()) {
        _frame_to_optim = _frame_to_optim_queue.front();

        std::stringstream msg;
        msg << "BACKEND: Add frame " << _frame_to_optim->_id << " to map ..." << std::endl;
        std::cout << msg.str();

        // Add frame to local map
        _local_map->addFrame(_frame_to_optim);
        _frame_to_optim->setKeyFrame();

        std::stringstream().swap(msg);
        msg << "BACKEND: Added Keyframe " << _frame_to_optim->_id << " to map!" << std::endl;
        std::cout << msg.str();

        // 3D Mesh update
        if (_slam_param->_config.mesh3D) {
            _mesher->addNewKF(_frame_to_optim);
            _mesh_to_display = _mesher->_mesh_3d;
        }
    }
    _map_mutex.unlock();
    if (_frame_to_optim) {
        // Reset frame to optim
        _frame_to_optim = nullptr;

        // Marginalization (+ sparsification) of the last frame
        isae::timer::tic();
        while (_local_map->getMarginalizationFlag() && _local_map->getFrames().size() >= 2) {
            if (_slam_param->_config.marginalization == 1)
                _slam_param->getOptimizerBack()->marginalize(_local_map->getFrames().at(0),
                                                             _local_map->getFrames().at(1),
                                                             _slam_param->_config.sparsification == 1);

            // Uncomment below to enable global map
            // _global_map->addFrame(_local_map->getFrames().at(0));

            _map_mutex.lock();
            std::cout << "BACKEND: Remove Frame..." << std::endl;
            _local_map->discardLastFrame();
            std::cout << "BACKEND: Removed Frame!" << std::endl;
            _map_mutex.unlock();
        }
        _avg_marg_t = (_avg_marg_t * (_nkeyframes - 1) + isae::timer::silentToc()) / _nkeyframes;
        // isae::timer::toc("BackEnd Marginalize");

        // Optimize Local Map
        isae::timer::tic();
        _slam_param->getOptimizerBack()->localMapBA(_local_map, _local_map->getFixedFrameNumber());
        _avg_wdw_opt_t = (_avg_wdw_opt_t * (_nkeyframes - 1) + isae::timer::silentToc()) / _nkeyframes;
        profiling();
        // isae::timer::toc("BackEnd Map");

        if (_local_map_to_display)
            std::cout << "LMTD: " << _local_map_to_display.use_count() << std::endl;

        // std::cout << "Show map" << std::endl;
        // Send the local map to the viewer

        // TODO: Figure out good way for thread-safe map access!

        // std::cout << "Show map ..." << std::endl;
        _local_map_to_display = _local_map;
        // std::cout << "Show map ... ..." << std::endl;
    }
    std::cout << "BACKEND: Slam is init: " << _is_init << std::endl;
    std::cout << "BACKEND: Done" << std::endl;

    return true;
}

} // namespace isae