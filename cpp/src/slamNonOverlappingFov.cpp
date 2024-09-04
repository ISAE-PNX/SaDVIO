#include "isaeslam/slamCore.h"

namespace isae {

Eigen::Affine3d pose_from_line(std::string line) {
    std::istringstream s(line);

    std::vector<double> values;
    std::copy(std::istream_iterator<double>(s), std::istream_iterator<double>(), std::back_inserter(values));

    Eigen::Affine3d pose;
    pose.matrix() << values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8], values[9],
        values[10], values[11], values[12], 0, 0, 0, 1;

    return pose;
}

bool SLAMNonOverlappingFov::init() {
    // get first frame and set keyframe
    _frame = _slam_param->getDataProvider()->next();
    if (!_frame) {
        sleep(1);
        return false;
    }
    if (_frame->getSensors().size() == 0)
        return false;

    _frame->setKeyFrame();
    _local_map->addFrame(_frame);
    std::cout << "t0: " << _frame->getTimestamp() << std::endl;

    // Get the txt file for the groundtruth
    std::string gt_path = "/home/deos/ce.debeunne/Documents/phd/developpement/SLAM_eval/dataset/gazebo/chariot4.txt";
    std::fstream gt_file;
    Eigen::Affine3d T_0, T_1;

    gt_file.open(gt_path, std::ios::in);
    if (gt_file.is_open()) { // checking whether the file is open
        std::string tp;
        while (getline(gt_file, tp)) { // read data from file object and put it into string.
            if (std::abs((double)_frame->getTimestamp() - std::stod(tp.substr(0, 19))) < 5e+8) {
                T_0 = pose_from_line(tp);
                break;
            }
        }
        gt_file.close(); // close the file object.
    }

    // Prior on the first frame, it is set as the origin
    _frame->setWorld2FrameTransform(Eigen::Affine3d::Identity());
    _frame->setPrior(Eigen::Affine3d::Identity(), 100 * Vector6d::Ones());

    // detect all features on all sensors
    detectFeatures(_frame->getSensors().at(0));
    detectFeatures(_frame->getSensors().at(1));

    // Init with essential matrix
    EpipolarPoseEstimator essential_ransac;
    bool cam0_ready              = false;
    Eigen::Affine3d T_cam1_cam1p = Eigen::Affine3d::Identity();
    Eigen::Affine3d T_cam0_cam0p = Eigen::Affine3d::Identity();

    Eigen::Affine3d T_cam0_cam1 = getLastKF()->getSensors().at(0)->getFrame2SensorTransform() *
                                  getLastKF()->getSensors().at(1)->getFrame2SensorTransform().inverse();

    // Track features until enough parallax
    while (!cam0_ready) {
        _frame = _slam_param->getDataProvider()->next();
        if (_frame->getSensors().size() == 0)
            continue;

        _nframes++;

        // track features on the first camera
        trackFeatures(getLastKF()->getSensors().at(0),
                      _frame->getSensors().at(0),
                      _matches_in_time,
                      _matches_in_time_lmk,
                      getLastKF()->getSensors().at(0)->getFeatures());

        // Track features on the second camera
        trackFeatures(getLastKF()->getSensors().at(1),
                      _frame->getSensors().at(1),
                      _matches_in_time_cam1,
                      _matches_in_time_cam1_lmk,
                      getLastKF()->getSensors().at(1)->getFeatures());

        // Essential matrix filtering for both cameras
        Eigen::MatrixXd cov;
        essential_ransac.estimateTransformSensors(getLastKF()->getSensors().at(0),
                                                  _frame->getSensors().at(0),
                                                  _matches_in_time["pointxd"],
                                                  T_cam0_cam0p,
                                                  cov);
        essential_ransac.estimateTransformSensors(getLastKF()->getSensors().at(1),
                                                  _frame->getSensors().at(1),
                                                  _matches_in_time_cam1["pointxd"],
                                                  T_cam1_cam1p,
                                                  cov);
        outlierRemoval();

        cam0_ready = !isDegenerativeMotion(T_cam0_cam0p, T_cam0_cam1, _matches_in_time);

        if (_matches_in_time["pointxd"].size() < 25 && _matches_in_time_cam1["pointxd"].size() < 25)
            return false;
    }

    // Compute the groundtruth scale
    gt_file.open(gt_path, std::ios::in);
    if (gt_file.is_open()) { // checking whether the file is open
        std::string tp;
        while (getline(gt_file, tp)) { // read data from file object and put it into string.
            if (std::abs((double)_frame->getTimestamp() - std::stod(tp.substr(0, 19))) < 5e+8) {
                T_1 = pose_from_line(tp);
                break;
            }
        }
        gt_file.close(); // close the file object.
    }
    double gt_scale = (T_0 * T_1.inverse()).translation().norm();
    std::cout << "gt scale : " << gt_scale << std::endl;

    // Compute the scale of the first camera motion
    double lambda_0 = 0;
    scaleEstimationRANSAC(T_cam0_cam1, T_cam0_cam0p, _matches_in_time_cam1, lambda_0);
    if (lambda_0 < 0.01)
        return false;
    Eigen::Affine3d T_cam0_w = getLastKF()->getSensors().at(0)->getWorld2SensorTransform();
    T_cam0_cam0p.translation() *= lambda_0;
    Eigen::Affine3d T_cam0p_w = T_cam0_cam0p.inverse() * T_cam0_w;
    _frame->setWorld2FrameTransform(_frame->getSensors().at(0)->getFrame2SensorTransform().inverse() * T_cam0p_w);

    std::cout << "Estimated scale : " << _frame->getWorld2FrameTransform().translation().norm() << std::endl;
    // Eigen::Affine3d T_f_w = _frame->getWorld2FrameTransform();
    // T_f_w.translation().normalize();
    // T_f_w.translation() *= gt_scale;
    // _frame->setWorld2FrameTransform(T_f_w);

    // Compute velocity
    double dt = (_frame->getTimestamp() - getLastKF()->getTimestamp()) * 1e-9;
    _6d_velocity =
        (geometry::se3_RTtoVec6d(getLastKF()->getWorld2FrameTransform() * _frame->getFrame2WorldTransform())) / dt;
    profiling();

    // Set KF and Triangulate landmarks
    _frame->setKeyFrame();
    _local_map->addFrame(_frame);
    _nkeyframes++;

    // Init first landmarks
    initLandmarks(_frame);

    // Optimize the landmarks and the scale
    std::vector<std::shared_ptr<Frame>> last_two_frames;
    _local_map->getLastNFramesIn(2, last_two_frames);
    _slam_param->getOptimizerBack()->landmarkOptimizationNoFov(
        last_two_frames.at(1), last_two_frames.at(0), T_cam0_cam0p, 0.05);

    // Create the 3D mesh
    if (_slam_param->_config.mesh3D) {
        _mesher->addNewKF(_frame);
    }

    // Ignore features that were not triangulated
    cleanFeatures(_frame);
    detectFeatures(_frame->getSensors().at(0));
    detectFeatures(_frame->getSensors().at(1));

    // Set init
    _is_init = true;

    return true;
}

bool SLAMNonOverlappingFov::frontEndStep() {

    // Get next frame
    isae::timer::tic();
    _frame = _slam_param->getDataProvider()->next();
    if (_frame->getSensors().size() == 0)
        return true;
    _nframes++;

    // Predict pose with constant velocity model
    double dt = (_frame->getTimestamp() - getLastKF()->getTimestamp()) * 1e-9;
    Eigen::Affine3d T_f_w =
        geometry::se3_Vec6dtoRT(_6d_velocity * dt).inverse() * getLastKF()->getWorld2FrameTransform();
    _frame->setWorld2FrameTransform(T_f_w);

    // Track features in time on both cameras
    uint nmatches_in_time;

    nmatches_in_time = trackFeatures(getLastKF()->getSensors().at(0),
                                     _frame->getSensors().at(0),
                                     _matches_in_time,
                                     _matches_in_time_lmk,
                                     getLastKF()->getSensors().at(0)->getFeatures());

    nmatches_in_time += trackFeatures(getLastKF()->getSensors().at(1),
                                      _frame->getSensors().at(1),
                                      _matches_in_time_cam1,
                                      _matches_in_time_cam1_lmk,
                                      getLastKF()->getSensors().at(1)->getFeatures());

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
        _matches_in_time_cam1 =
            epipolarFiltering(getLastKF()->getSensors().at(1), _frame->getSensors().at(1), _matches_in_time_cam1);
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
        isae::timer::tic();
        updateLandmarks(_matches_in_time_lmk);
        updateLandmarks(_matches_in_time_cam1_lmk);

        // Single Frame Bundle Adjustment
        _slam_param->getOptimizerFront()->singleFrameOptimization(_frame);
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

        outlierRemoval();
        _frame->setKeyFrame();
    }

    if (shouldInsertKeyframe(_frame)) {
        _nkeyframes++;

        // Landmark Initialization:
        // - Triangulate new points : LR + (n-1) / n
        // - Optimize points only + scale
        // - Reject outliers with reprojection error
        isae::timer::tic();
        initLandmarks(_frame);
        std::vector<std::shared_ptr<Frame>> last_two_frames;
        _local_map->getLastNFramesIn(2, last_two_frames);

        isae::timer::tic();
        Eigen::Affine3d T_cam0_cam0p = last_two_frames.at(1)->getSensors().at(0)->getFrame2SensorTransform() *
                                       last_two_frames.at(1)->getWorld2FrameTransform() *
                                       last_two_frames.at(0)->getFrame2WorldTransform() *
                                       last_two_frames.at(1)->getSensors().at(0)->getFrame2SensorTransform().inverse();

        _slam_param->getOptimizerFront()->landmarkOptimizationNoFov(
            last_two_frames.at(1), last_two_frames.at(0), T_cam0_cam0p, 0.05);

        _avg_lmk_init_t = (_avg_lmk_init_t * (_nkeyframes - 1) + isae::timer::silentToc()) / _nkeyframes;

        // Repopulate in the case of klt tracking
        if (_slam_param->_config.tracker == "klt") {
            isae::timer::tic();
            cleanFeatures(_frame);
            detectFeatures(_frame->getSensors().at(0));
            detectFeatures(_frame->getSensors().at(1));
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

    // Send the frame to the viewer
    _frame_to_display = _frame;

    return true;
}

bool SLAMNonOverlappingFov::backEndStep() {
    
    if (_frame_to_optim) {

        // Frame is added, marginalization flag is raised if necessary
        _local_map->addFrame(_frame_to_optim);

        // 3D Mesh update
        if (_slam_param->_config.mesh3D) {
            _mesher->addNewKF(_frame);
            _mesh_to_display = _mesher->_mesh_3d;
        }

        // Discard the last frame
        if (_local_map->getMarginalizationFlag()) {
            if (_slam_param->_config.marginalization == 1)
                _slam_param->getOptimizerBack()->marginalize(_local_map->getFrames().at(0),
                                                             _local_map->getFrames().at(1),
                                                             _slam_param->_config.sparsification == 1);
            _global_map->addFrame(_local_map->getFrames().at(0));
            _local_map->discardLastFrame();
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

void SLAMNonOverlappingFov::outlierRemoval() {

    // Remove outliers from the current kf
    for (auto tf : _frame->getSensors().at(0)->getFeatures()) {
        for (auto f : tf.second) {
            if (f->isOutlier()) {
                _frame->getSensors().at(0)->removeFeature(f);
                continue;
            }

            bool is_outlier = true;

            for (auto m : _matches_in_time[tf.first]) {
                if (f == m.second) {
                    is_outlier = false;
                    break;
                }
            }

            if (!is_outlier)
                continue;

            for (auto m : _matches_in_time_lmk[tf.first]) {
                if (f == m.second) {
                    is_outlier = false;
                    break;
                }
            }

            if (is_outlier) {
                _frame->getSensors().at(0)->removeFeature(f);
            }
        }
    }

    for (auto tf : _frame->getSensors().at(1)->getFeatures()) {
        for (auto f : tf.second) {
            if (f->isOutlier()) {
                _frame->getSensors().at(1)->removeFeature(f);
                continue;
            }

            bool is_outlier = true;

            for (auto m : _matches_in_time_cam1[tf.first]) {
                if (f == m.second) {
                    is_outlier = false;
                    break;
                }
            }

            if (!is_outlier)
                continue;

            for (auto m : _matches_in_time_cam1_lmk[tf.first]) {
                if (f == m.second) {
                    is_outlier = false;
                    break;
                }
            }

            if (is_outlier) {
                _frame->getSensors().at(1)->removeFeature(f);
            }
        }
    }
}

int SLAMNonOverlappingFov::scaleEstimationRANSAC(const Eigen::Affine3d T_cam0_cam1,
                                                 const Eigen::Affine3d T_cam0_cam0p,
                                                 typed_vec_match matches_cam1,
                                                 double &lambda) {

    Eigen::Vector3d xA = T_cam0_cam0p.rotation() * T_cam0_cam1.translation() - T_cam0_cam1.translation();
    Eigen::Matrix3d A  = T_cam0_cam1.rotation().transpose() * geometry::skewMatrix(xA) * T_cam0_cam0p.rotation() *
                        T_cam0_cam1.rotation();
    Eigen::Matrix3d B = T_cam0_cam1.rotation().transpose() * geometry::skewMatrix(T_cam0_cam0p.translation()) *
                        T_cam0_cam0p.rotation() * T_cam0_cam1.rotation();

    // Compute degenerate plane normal formed by the rotated cam1 and the direction of the translation
    // cf. Clipp et. al 2008
    Eigen::Vector3d n = ((T_cam0_cam0p.rotation() * T_cam0_cam1.translation() - T_cam0_cam1.translation())
                             .cross(T_cam0_cam0p.translation()))
                            .normalized();

    // RANSAC implementation
    uint max_iter        = matches_cam1["pointxd"].size();
    int min_matches      = 10;
    double err_threshold = 0.2; // 10 % of error
    int best_n_inliers   = 0;

    // Build a vector of lambda
    std::vector<double> lambda_vec;
    for (uint k = 0; k < matches_cam1["pointxd"].size(); k++) {
        // Compute the scale factor
        Eigen::Vector3d bk  = matches_cam1["pointxd"].at(k).first->getBearingVectors().at(0);
        Eigen::Vector3d bkp = matches_cam1["pointxd"].at(k).second->getBearingVectors().at(0);
        double lambdak      = (bk.transpose() * A * bkp);
        lambdak /= (bk.transpose() * B * bkp);
        lambdak *= -1;

        // Check if it is not degenerate
        Eigen::Vector3d u = T_cam0_cam1.rotation() * matches_cam1["pointxd"].at(k).first->getBearingVectors().at(0);
        double check      = u.dot(n);
        if (std::abs(check) < 0.05)
            lambdak = 0;

        lambda_vec.push_back(lambdak);
    }

    // RANSAC loop
    std::vector<double> inlier_vector;
    for (uint i = 0; i < max_iter; i++) {
        // Pick a random point (in fact no we go through all the matches)
        uint j = i;
        inlier_vector.clear();

        // Get the scale factor
        double lambdaj = lambda_vec.at(j);
        inlier_vector.push_back(lambdaj);

        if (lambdaj <= 0)
            continue;

        // Check the parallax in pixels
        double parallax = (matches_cam1["pointxd"].at(i).first->getPoints().at(0) -
                           matches_cam1["pointxd"].at(i).second->getPoints().at(0))
                              .norm();
        if (parallax < 4)
            continue;

        // Build the set of inliers
        int n_inliers = 0;

        for (uint k = 0; k < matches_cam1["pointxd"].size(); k++) {

            if (k == j)
                continue;

            // Get the scale factor
            double lambdak = lambda_vec.at(k);

            // Check if inlier
            if (std::abs((lambdaj - lambdak) / lambdaj) < err_threshold) {
                inlier_vector.push_back(lambdak);
                n_inliers++;
            }
        }

        // Check if it has the most inliers
        if (n_inliers < best_n_inliers || n_inliers < min_matches)
            continue;

        best_n_inliers = n_inliers;

        // Compute the best lambda
        lambda = std::accumulate(inlier_vector.begin(), inlier_vector.end(), 0.0) / inlier_vector.size();
    }

    std::cout << lambda << " found with : " << best_n_inliers << " inliers" << std::endl;

    return best_n_inliers;
}

void SLAMNonOverlappingFov::initLandmarks(std::shared_ptr<Frame> &f) {

    // Init unitialized landmarks for cam0
    for (auto &ttracks_in_time : _matches_in_time_lmk) {

        // Init all tracked feature in frame
        for (auto &ttime : ttracks_in_time.second) {

            // Check if the landmark is not initialized
            if (ttime.first->getLandmark().lock()) {
                if (ttime.first->getLandmark().lock()->isInitialized())
                    continue;
            }

            // Build the feature vector
            std::vector<std::shared_ptr<AFeature>> features;
            for (auto feat : ttime.first->getLandmark().lock()->getFeatures()) {
                features.push_back(feat.lock());
            }
            _slam_param->getLandmarksInitializer()[ttracks_in_time.first]->initFromFeatures(features);
        }
    }

    // Init unitialized landmarks for cam1
    for (auto &ttracks_in_time : _matches_in_time_cam1_lmk) {

        // Init all tracked feature in frame
        for (auto &ttime : ttracks_in_time.second) {

            // Check if the landmark is not initialized
            if (ttime.first->getLandmark().lock()) {
                if (ttime.first->getLandmark().lock()->isInitialized())
                    continue;
            }

            // Build the feature vector
            std::vector<std::shared_ptr<AFeature>> features;
            for (auto feat : ttime.first->getLandmark().lock()->getFeatures()) {
                features.push_back(feat.lock());
            }
            _slam_param->getLandmarksInitializer()[ttracks_in_time.first]->initFromFeatures(features);
        }
    }

    // Init landmarks with tracks in time on cam0
    for (auto &ttracks_in_time : _matches_in_time) {

        for (auto &ttime : ttracks_in_time.second) {
            std::vector<std::shared_ptr<AFeature>> feats;
            feats.push_back(ttime.first);
            feats.push_back(ttime.second);

            _slam_param->getLandmarksInitializer()[ttracks_in_time.first]->initFromFeatures(feats);
        }
    }

    // Init landmarks with tracks in time on cam1
    for (auto &ttracks_in_time : _matches_in_time_cam1) {

        for (auto &ttime : ttracks_in_time.second) {
            std::vector<std::shared_ptr<AFeature>> feats;
            feats.push_back(ttime.first);
            feats.push_back(ttime.second);

            _slam_param->getLandmarksInitializer()[ttracks_in_time.first]->initFromFeatures(feats);
        }
    }
}

bool SLAMNonOverlappingFov::predict(std::shared_ptr<Frame> &f) {
    Eigen::MatrixXd cov0 = 10000 * Eigen::MatrixXd::Identity(6, 6);
    Eigen::MatrixXd cov1 = cov0;

    // Predict pose with constant velocity model dT =
    Eigen::Affine3d T_f0_f0p = (getLastKF()->getWorld2FrameTransform() * f->getFrame2WorldTransform());
    Eigen::Affine3d T_f1_f1p = T_f0_f0p;
    Eigen::Affine3d T_const  = T_f0_f0p;

    // Filter on cam 0 + check constant velocity
    bool cam0_ok = _slam_param->getPoseEstimator()->estimateTransformBetween(
        getLastKF(), f, _matches_in_time_lmk["pointxd"], T_f0_f0p, cov0);
    cam0_ok = (cam0_ok && ((T_f0_f0p.translation() - T_const.translation()).norm() < 1));

    // Filter also on cam1 + check constant velocity
    bool cam1_ok = _slam_param->getPoseEstimator()->estimateTransformBetween(
        getLastKF(), f, _matches_in_time_cam1_lmk["pointxd"], T_f1_f1p, cov1);
    cam1_ok = (cam1_ok && ((T_f1_f1p.translation() - T_const.translation()).norm() < 1));

    // False if the prediction failed and constant velocity applies
    if (!cam0_ok && !cam1_ok) {
        std::cerr << "Predict fails" << std::endl;

        return false;
    } else {

        // Update the pose only for pnp
        if (_slam_param->_config.pose_estimator != "pnp")
            T_f0_f0p = T_const;

        if (cam0_ok && !cam1_ok) {
            T_f0_f0p = T_f0_f0p;
        } else if (cam1_ok && !cam0_ok) {
            T_f0_f0p = T_f1_f1p;
        } else {
            // Weighted average of both prediction
            Eigen::MatrixXd W = cov0 * (cov0 + cov1).inverse();
            Vector6d dT       = geometry::se3_RTtoVec6d(T_f0_f0p.inverse() * T_f1_f1p);
            T_f0_f0p          = T_f0_f0p * geometry::se3_Vec6dtoRT(W * dT);
        }

        f->setWorld2FrameTransform(T_f0_f0p.inverse() * getLastKF()->getWorld2FrameTransform());
        return true;
    }
}

bool SLAMNonOverlappingFov::isDegenerativeMotion(Eigen::Affine3d T_cam0_cam0p,
                                                 Eigen::Affine3d T_cam0_cam1,
                                                 typed_vec_match matches) {
    double n_matches = matches["pointxd"].size();

    // Check the critical condition (Clipp et al)
    Eigen::Vector3d check = (T_cam0_cam0p.rotation() * T_cam0_cam1.translation() - T_cam0_cam1.translation())
                                .cross(T_cam0_cam0p.translation().normalized());

    if (check.norm() < 0.015) {
        return true;
    }

    // First check if there is enough rotation
    if (geometry::log_so3(T_cam0_cam0p.rotation()).norm() < 0.1) {
        return true;
    }

    // Second check if there is enough parallax
    double avg_parallax = 0;
    for (auto tmatch : matches) {
        for (auto match : tmatch.second) {
            avg_parallax += std::acos(match.first->getBearingVectors().at(0).transpose() *
                                      match.second->getBearingVectors().at(0)) /
                            n_matches;
        }
    }
    avg_parallax *= 180 / M_PI;

    if (avg_parallax < _max_movement_parallax) {
        return true;
    }

    return false;
}

} // namespace isae