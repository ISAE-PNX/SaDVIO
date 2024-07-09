#include "isaeslam/slamCore.h"

namespace isae {

bool SLAMBiMonoVIO::init() {

    // get first frame that must contain IMU
    _frame = _slam_param->getDataProvider()->next();
    if (!_frame || !_frame->getIMU()) {
        return false;
    }

    // Align the global frame with the gravity direction (if IMU available)
    Eigen::Affine3d T_i_f = Eigen::Affine3d::Identity();
    Eigen::Vector3d acc_mean(0.0, 0.0, 0.0);
    Eigen::Vector3d gyr_mean(0.0, 0.0, 0.0);
    _last_IMU = _frame->getIMU();
    int count = 0;

    // Average IMU measurement to reduce noise influence
    while (count < 10 || _frame->getSensors().size() == 0) {
        if (!_frame->getIMU()) {
            _frame = _slam_param->getDataProvider()->next();
            continue;
        }
        acc_mean += _frame->getIMU()->getAcc();
        gyr_mean += _frame->getIMU()->getGyr();
        _frame = _slam_param->getDataProvider()->next();
        count++;
    }
    acc_mean /= count;
    gyr_mean /= count;

    // Compute normalized direction vectors
    Eigen::Vector3d acc_mean_norm = acc_mean.normalized();
    Eigen::Vector3d g_norm        = (-g).normalized();

    // We use the rodriguez formula to compute the rotation matrix (TODO implement it in geometry)
    Eigen::Vector3d k = acc_mean_norm.cross(g_norm);
    Eigen::Matrix3d K = geometry::skewMatrix(k.normalized());
    T_i_f.affine().block(0, 0, 3, 3) =
        (Eigen::Matrix3d::Identity() + k.norm() * K + (1 - acc_mean_norm.dot(g_norm)) * K * K);
    // Prior on the first frame, it is set as the origin
    _frame->setWorld2FrameTransform(T_i_f.inverse());
    // _frame->setPrior(T_i_f.inverse(), 100 * Vector6d::Ones());

    // init the velocity
    _6d_velocity = 0.00001 * Vector6d::Ones();

    // Set Keyframe and initialize preintegration
    _frame->setKeyFrame();
    std::shared_ptr<IMU> imu_kf =
        std::make_shared<IMU>(_slam_param->getDataProvider()->getIMUConfig(), _last_IMU->getAcc(), _last_IMU->getGyr());
    imu_kf->setLastIMU(_last_IMU);
    imu_kf->setLastKF(_frame);
    imu_kf->processIMU();
    _frame->setIMU(imu_kf, _last_IMU->getFrame2SensorTransform());
    _last_IMU = imu_kf;
    _local_map->addFrame(_frame);

    // Init bias (assuming that we are not moving)
    Eigen::Vector3d ba = acc_mean + T_i_f.rotation().transpose() * g;
    Eigen::Vector3d bg = gyr_mean;
    _frame->getIMU()->setBa(ba);
    _frame->getIMU()->setBg(bg);
    std::cout << "Bias accel " << ba.transpose() << std::endl;
    std::cout << "Bias gyro " << bg.transpose() << std::endl;

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

    // init first landmarks
    initLandmarks(_frame);
    _slam_param->getOptimizerFront()->landmarkOptimization(_frame);

    // Ignore features that were not triangulated
    cleanFeatures(_frame);
    detectFeatures(_frame->getSensors().at(0));

    // Perform Local BA on 10 KF before optimizing inertial variables
    while (_local_map->getFrames().size() < 10)
        step_init();

    // Launch optimization of the inertial variables
    Eigen::Matrix3d dRi;
    _slam_param->getOptimizerFront()->VIInit(_local_map, dRi);
    _slam_param->getOptimizerFront()->localMapVIOptimization(_local_map, 1);

    std::cout << "Orientation update : " << dRi << std::endl;
    std::cout << "Gyro bias : " << getLastKF()->getIMU()->getBg().transpose() << std::endl;
    std::cout << "Acc bias : " << getLastKF()->getIMU()->getBa().transpose() << std::endl;

    std::cout << "IMU velo : "
              << (_last_IMU->getFrame()->getFrame2WorldTransform().rotation().transpose() * _last_IMU->getVelocity())
                     .transpose()
              << std::endl;

    profiling();

    // Set pb init
    _nkeyframes++;
    _is_init = true;

    return true;
}

bool SLAMBiMonoVIO::step_init() {
    // Get next frame
    _frame = _slam_param->getDataProvider()->next();

    // Process IMU data
    if (_frame->getIMU()) {
        _frame->getIMU()->setLastIMU(_last_IMU);
        _frame->getIMU()->setLastKF(getLastKF());

        // Ignore measurement if it failed
        if (_frame->getIMU()->processIMU()) {
            _last_IMU = _frame->getIMU();
        }
    }

    // Skip if there is no images in the frame
    if (_frame->getSensors().empty())
        return true;

    _nframes++;

    // Estimate the transformation between frames using constant velocity
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
        detectFeatures(_frame->getSensors().at(0));
        _frame->setKeyFrame();
    }

    // Force a KF to prevent the IMU to drift
    if (dt > 0.05)
        _frame->setKeyFrame();

    if (shouldInsertKeyframe(_frame)) {

        // An IMU is set to this KF if there is not one already
        if (!_frame->getIMU()) {

            // Check if the _last_imu is in the future
            while (_last_IMU->getFrame()->getTimestamp() > _frame->getTimestamp()) {
                _last_IMU = _last_IMU->getLastIMU();
            }

            std::shared_ptr<IMU> imu_kf = std::make_shared<IMU>(
                _slam_param->getDataProvider()->getIMUConfig(), _last_IMU->getAcc(), _last_IMU->getGyr());
            imu_kf->setLastIMU(_last_IMU);
            imu_kf->setLastKF(getLastKF());
            _frame->setIMU(imu_kf, _last_IMU->getFrame2SensorTransform());

            // Process IMU but without updating the frame pose
            Eigen::Affine3d T_f_w = _frame->getWorld2FrameTransform();
            _frame->getIMU()->processIMU();
            _frame->setWorld2FrameTransform(T_f_w);

            // Set last imu
            _last_IMU = _frame->getIMU();
        }

        // Frame is added, marginalization flag is raised if necessary
        _nkeyframes++;
        _local_map->addFrame(_frame);

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
        uint resu = recoverFeatureFromMapLandmarks(_local_map, _frame);

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

        // Perform local BA
        isae::timer::tic();
        _slam_param->getOptimizerFront()->localMapBA(_local_map, _local_map->getFixedFrameNumber());
        _avg_wdw_opt_t = (_avg_wdw_opt_t * (_nkeyframes - 1) + isae::timer::silentToc()) / _nkeyframes;

    } else {

        // If no KF is voted, the frame is discarded and the landmarks are cleaned
        _frame->cleanLandmarks();
    }

    return true;
}

bool SLAMBiMonoVIO::frontEndStep() {

    // Get next frame
    _frame = _slam_param->getDataProvider()->next();

    // Timing vector frate
    std::vector<float> timing_fe;

    // Process IMU data
    if (_frame->getIMU()) {
        _frame->getIMU()->setLastIMU(_last_IMU);
        _frame->getIMU()->setLastKF(getLastKF());

        // Ignore measurement if it failed
        if (_frame->getIMU()->processIMU()) {
            _last_IMU = _frame->getIMU();
        }
    }

    // Skip if there is no images in the frame
    if (_frame->getSensors().empty())
        return true;

    _nframes++;

    // Estimate the transformation between frames using IMU for the rotation and cst velocity for translation
    double dt          = (_frame->getTimestamp() - getLastKF()->getTimestamp()) * 1e-9;
    Eigen::Affine3d dT = getLastKF()->getWorld2FrameTransform() * _last_IMU->getFrame()->getFrame2WorldTransform();
    Eigen::Affine3d T_f_w =
        geometry::se3_Vec6dtoRT(_6d_velocity * dt).inverse() * getLastKF()->getWorld2FrameTransform();
    _frame->setWorld2FrameTransform(T_f_w);

    // Detect all features (only if we use the matcher)
    if (_slam_param->_config.tracker == "matcher") {
        isae::timer::tic();
        detectFeatures(_frame->getSensors().at(0));
        _avg_detect_t = (_avg_detect_t * (_nframes - 1) + isae::timer::silentToc()) / _nframes;
    }

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
    float matching_dt = isae::timer::silentToc();
    timing_fe.push_back(matching_dt);
    _avg_match_time_t = (_avg_match_time_t * (_nframes - 1) + matching_dt) / _nframes;

    // Get P3d from n-1 matched features and estimate 3D pose from 2D (n)/3D (n-1) matchings
    // to predict pose. Also remove outliers from tracks_in_time vector
    isae::timer::tic();
    bool good_it   = predict(_frame);
    float pnp_dt   = isae::timer::silentToc();
    _avg_predict_t = (_avg_predict_t * (_nframes - 1) + pnp_dt) / _nframes;
    timing_fe.push_back(pnp_dt);

    if (good_it) {

        // Epipolar Filtering for matches in time
        isae::timer::tic();
        int removed_matching_nb = _matches_in_time["pointxd"].size() + _matches_in_time_lmk["pointxd"].size();
        _matches_in_time =
            epipolarFiltering(getLastKF()->getSensors().at(0), _frame->getSensors().at(0), _matches_in_time);
        _matches_in_time_lmk =
            epipolarFiltering(getLastKF()->getSensors().at(0), _frame->getSensors().at(0), _matches_in_time_lmk);
        removed_matching_nb -= _matches_in_time["pointxd"].size() + _matches_in_time_lmk["pointxd"].size();
        float epi_dt  = isae::timer::silentToc();
        _removed_feat = (_removed_feat * (_nframes - 1) + removed_matching_nb) / _nframes;
        _avg_filter_t = (_avg_filter_t * (_nframes - 1) + epi_dt) / _nframes;
        timing_fe.push_back(epi_dt);

        // Remove Outliers in case of klt
        if (_slam_param->_config.tracker == "klt") {
            isae::timer::tic();
            outlierRemoval();
            double dt_filter = isae::timer::silentToc();
            _avg_clean_t = (_avg_clean_t * (_nframes - 1) + dt_filter) / _nframes;
            timing_fe.push_back(dt_filter);
        }

        // Update tracked landmarks
        updateLandmarks(_matches_in_time_lmk);

        // Single Frame Bundle Adjustment
        isae::timer::tic();
        _slam_param->getOptimizerFront()->singleFrameVIOptimization(_frame);
        float optim_dt   = isae::timer::silentToc();
        _avg_frame_opt_t = (_avg_frame_opt_t * (_nframes - 1) + optim_dt) / _nframes;
        _lmk_inmap       = (_lmk_inmap * (_nframes - 1) + _frame->getInMapLandmarksNumber()) / _nframes;
        timing_fe.push_back(optim_dt);

        // Compute velocity and motion model
        _6d_velocity =
            (geometry::se3_RTtoVec6d(getLastKF()->getWorld2FrameTransform() * _frame->getFrame2WorldTransform())) / dt;

        // Update timing list
        _timings_frate.push_back(timing_fe);

    } else {

        // If the prediction is wrong, we reinitialize the odometry from the last KF:
        // - The pose is estimated using IMU only
        // - A KF is voted
        // - All matches in time are removed
        // Can be improved: redetect new points, retrack old features....

        T_f_w = dT.inverse() * getLastKF()->getWorld2FrameTransform();
        _frame->setWorld2FrameTransform(T_f_w);
        outlierRemoval();
        detectFeatures(_frame->getSensors().at(0));
        _frame->setKeyFrame();
    }

    // Force a KF to prevent the IMU to drift
    if (dt > 1.0)
        _frame->setKeyFrame();

    if (shouldInsertKeyframe(_frame)) {
        _frame->unsetKeyFrame(); // Dirty as hell, to fix
        _nkeyframes++;

        // An IMU is set to this KF if there is not one already
        if (!_frame->getIMU()) {

            // Check if the _last_imu is in the future
            while (_last_IMU->getFrame()->getTimestamp() > _frame->getTimestamp()) {
                _last_IMU = _last_IMU->getLastIMU();
            }

            std::shared_ptr<IMU> imu_kf = std::make_shared<IMU>(
                _slam_param->getDataProvider()->getIMUConfig(), _last_IMU->getAcc(), _last_IMU->getGyr());
            imu_kf->setLastIMU(_last_IMU);
            imu_kf->setLastKF(getLastKF());
            _frame->setIMU(imu_kf, _last_IMU->getFrame2SensorTransform());

            // Process IMU but without updating the frame pose
            Eigen::Affine3d T_f_w = _frame->getWorld2FrameTransform();
            _frame->getIMU()->processIMU();
            _frame->setWorld2FrameTransform(T_f_w);
        }

        _frame->getIMU()->estimateTransform(_last_IMU->getLastKF(), _frame, dT);

        if (!good_it) {
            std::cout << "IMU dT : \n" << dT.matrix() << std::endl;
            std::cout << "pnp dT : \n"
                      << (getLastKF()->getWorld2FrameTransform() * _frame->getFrame2WorldTransform()).matrix()
                      << "\n ---" << std::endl;
        }

        // Repopulate in the case of klt tracking
        typed_vec_features new_features;
        if (_slam_param->_config.tracker == "klt") {
            isae::timer::tic();
            cleanFeatures(_frame);
            new_features    = detectFeatures(_frame->getSensors().at(0));
            float detect_dt = isae::timer::silentToc();
            _avg_detect_t   = (_avg_detect_t * (_nkeyframes - 1) + detect_dt) / _nkeyframes;
            timing_fe.push_back(detect_dt);
        }

        // Recover Map Landmark
        isae::timer::tic();
        _map_mutex.lock();
        uint resu = recoverFeatureFromMapLandmarks(_local_map, _frame);
        _map_mutex.unlock();

        float recover_dt = isae::timer::silentToc();
        _avg_lmk_resur_t = (_avg_lmk_resur_t * (_nkeyframes - 1) + recover_dt) / _nkeyframes;
        _avg_resur_lmk   = (_avg_lmk_resur_t * (_nkeyframes - 1) + resu) / _nkeyframes;
        timing_fe.push_back(recover_dt);

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

        float track_dt     = isae::timer::silentToc();
        _avg_matches_frame = (_avg_matches_frame * (_nkeyframes - 1) + nmatches_in_frame) / _nkeyframes;
        _avg_match_frame_t = (_avg_match_frame_t * (_nkeyframes - 1) + track_dt) / _nkeyframes;
        timing_fe.push_back(track_dt);

        // Wait the end of optim
        while (_frame_to_optim != nullptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Landmark Initialization:
        // - Triangulate new points : LR + (n-1) / n
        // - Optimize points only because optimal mid-point is not optimal for LM
        // - Reject outliers with reprojection error
        isae::timer::tic();
        initLandmarks(_frame);
        _slam_param->getOptimizerFront()->landmarkOptimization(_frame);
        float lmk_dt    = isae::timer::silentToc();
        _avg_lmk_init_t = (_avg_lmk_init_t * (_nkeyframes - 1) + lmk_dt) / _nkeyframes;
        timing_fe.push_back(lmk_dt);
        _timings_kfrate_fe.push_back(timing_fe);

        // Send frame to optim to optimizer
        _frame_to_optim = _frame;
        _last_IMU       = _frame_to_optim->getIMU();

    } else {
        // If no KF is voted, the frame is discarded and the landmarks are cleaned
        _frame->cleanLandmarks();
    }

    // Send the frame to the viewer
    _frame_to_display = _frame;

    return true;
}

bool SLAMBiMonoVIO::backEndStep() {

    if (_frame_to_optim) {

        std::vector<float> timing_be;

        // Add frame to local map
        _local_map->addFrame(_frame_to_optim);
        _frame_to_optim->setKeyFrame();

        // Marginalization (+ sparsification) of the last frame
        isae::timer::tic();
        if (_local_map->getMarginalizationFlag()) {
            if (_slam_param->_config.marginalization == 1)
                _slam_param->getOptimizerBack()->marginalize(_local_map->getFrames().at(0),
                                                             _local_map->getFrames().at(1),
                                                             _slam_param->_config.sparsification == 1);

            _global_map->addFrame(_local_map->getFrames().at(0));
            _map_mutex.lock();
            _local_map->discardLastFrame();
            _map_mutex.unlock();
        }

        float marg_dt = isae::timer::silentToc();
        _avg_marg_t   = (_avg_marg_t * (_nkeyframes - 1) + marg_dt) / _nkeyframes;
        timing_be.push_back(marg_dt);

        // Optimize Local Map
        isae::timer::tic();
        _slam_param->getOptimizerBack()->localMapVIOptimization(_local_map, _local_map->getFixedFrameNumber());

        // Update current IMU biases after optimization
        _frame_to_optim->getIMU()->updateBiases();

        float optim_dt = isae::timer::silentToc();
        _avg_wdw_opt_t = (_avg_wdw_opt_t * (_nkeyframes - 1) + optim_dt) / _nkeyframes;
        timing_be.push_back(optim_dt);
        _timings_kfrate_be.push_back(timing_be);

        // profiling
        profiling();

        // 3D Mesh update
        if (_slam_param->_config.mesh3D) {
            _mesher->addNewKF(_frame_to_optim);
            _mesh_to_display = _mesher->_mesh_3d;
        }

        // Reset frame to optim
        _frame_to_optim = nullptr;

        // Send the local map to the viewer
        _local_map_to_display = _local_map;
    }

    return true;
}

void SLAMBiMonoVIO::IMUprofiling() {

    if (!_is_init) {

        // Write header if not init
        std::ofstream fw_res("log_slam/vio_poses.csv", std::ofstream::out | std::ofstream::trunc);
        fw_res << "timestamp (ns), T_wf(00), T_wf(01), T_wf(02), T_wf(03), T_wf(10), T_wf(11), T_wf(12), "
               << "T_wf(13), T_wf(20), T_wf(21), T_wf(22), T_wf(23), v_w(0), v_w(1), v_w(2)\n";
        fw_res.close();

    } else {

        // Write in a csv file for evaluation
        std::ofstream fw_res("log_slam/vio_poses.csv", std::ofstream::out | std::ofstream::app);
        const Eigen::Matrix3d R = _frame->getFrame2WorldTransform().linear();
        Eigen::Vector3d twc     = _frame->getFrame2WorldTransform().translation();
        Eigen::Vector3d vw      = _frame->getIMU()->getVelocity();
        fw_res << _frame->getTimestamp() << "," << R(0, 0) << "," << R(0, 1) << "," << R(0, 2) << "," << twc.x() << ","
               << R(1, 0) << "," << R(1, 1) << "," << R(1, 2) << "," << twc.y() << "," << R(2, 0) << "," << R(2, 1)
               << "," << R(2, 2) << "," << twc.z() << "," << vw.x() << "," << vw.y() << "," << vw.z() << "\n";
        fw_res.close();
    }
}

} // namespace isae