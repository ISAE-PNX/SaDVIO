#include "isaeslam/slamCore.h"

namespace isae {

SLAMCore::SLAMCore(std::shared_ptr<isae::SLAMParameters> slam_param) : _slam_param(slam_param) {
    std::cout << _slam_param->_config.min_kf_number << "," << _slam_param->_config.max_kf_number << ","
              << _slam_param->_config.min_lmk_number << "," << _slam_param->_config.fixed_frame_number << ","
              << _slam_param->_config.min_movement_parallax << "," << _slam_param->_config.max_movement_parallax
              << std::endl;

    _max_movement_parallax = _slam_param->_config.max_movement_parallax;
    _min_movement_parallax = _slam_param->_config.min_movement_parallax;
    _min_lmk_number        = _slam_param->_config.min_lmk_number;

    _local_map  = std::make_shared<LocalMap>(_slam_param->_config.min_kf_number,
                                            _slam_param->_config.max_kf_number,
                                            _slam_param->_config.fixed_frame_number);
    _global_map = std::make_shared<GlobalMap>();
    _mesher     = std::make_shared<Mesher>(
        _slam_param->_config.slam_mode, _slam_param->_config.ZNCC_tsh, _slam_param->_config.max_length_tsh);

    _avg_detect_t      = 0;
    _avg_lmk_init_t    = 0;
    _avg_lmk_resur_t   = 0;
    _avg_matches_frame = 0;
    _avg_match_frame_t = 0;
    _avg_matches_time  = 0;
    _avg_match_time_t  = 0;
    _avg_predict_t     = 0;
    _avg_filter_t      = 0;
    _avg_processing_t  = 0;
    _avg_frame_opt_t   = 0;
    _avg_wdw_opt_t     = 0;
    _avg_clean_t       = 0;
    _avg_marg_t        = 0;
    _removed_feat      = 0;
    _lmk_inmap         = 0;
    _nframes           = 0;
    _nkeyframes        = 0;
};

void SLAMCore::outlierRemoval() {

    // Remove outliers from the current kf after tracking / matching
    isae::typed_vec_features clean_features;

    for (auto tf : _frame->getSensors().at(0)->getFeatures()) {

        for (auto m : _matches_in_time[tf.first]) {
            if (!m.second->isOutlier()) {
                clean_features[tf.first].push_back(m.second);
            }
        }

        for (auto m : _matches_in_time_lmk[tf.first]) {
            if (!m.second->isOutlier()) {
                clean_features[tf.first].push_back(m.second);
            }
        }

        _frame->getSensors().at(0)->purgeFeatures(tf.first);
        _frame->getSensors().at(0)->addFeatures(tf.first, clean_features[tf.first]);
    }
}

void SLAMCore::cleanFeatures(std::shared_ptr<Frame> &f) {
    // remove feature with outlier ldmk and feature

    isae::typed_vec_features clean_features;
    for (size_t i = 0; i < 1; i++) {
        for (auto tfeat : f->getSensors().at(i)->getFeatures()) {
            for (auto feat : tfeat.second) {
                if (feat->getLandmark().lock()) {
                    if (!feat->getLandmark().lock()->isOutlier() && !feat->isOutlier())
                        clean_features[tfeat.first].push_back(feat);
                }
            }
            f->getSensors().at(i)->purgeFeatures(tfeat.first);
            f->getSensors().at(i)->addFeatures(tfeat.first, clean_features[tfeat.first]);
        }
    }
}

void SLAMCore::updateLandmarks(typed_vec_match matches_lmk) {

    // Update all existing landmarks tracked in time
    for (auto &tmatches_lmk : matches_lmk) {
        // Init all tracked feature with existing landmarks
        for (auto &match_lmk : tmatches_lmk.second) {
            _slam_param->getLandmarksInitializer()[tmatches_lmk.first]->initFromMatch(match_lmk);
        }
    }
}

void SLAMCore::initLandmarks(std::shared_ptr<Frame> &f) {

    // Get number of landmarks requested per type (defined in the tracker and provided in yaml)
    std::map<std::string, int> N;
    for (auto &tracker : _slam_param->getLandmarksInitializer()) {
        N[tracker.first] = tracker.second->getNbRequieredLdmk();

        // Check number of missing landmarks after map update
        N[tracker.first] = N[tracker.first] - f->getLandmarks()[tracker.first].size();
    }

    // Init unitialized landmarks
    for (auto &ttracks_in_time : _matches_in_time_lmk) {

        // Init all tracked feature in frame
        int nb_created = 0;
        for (auto &ttime : ttracks_in_time.second) {
            // Check if we have enough landmarks
            if (N[ttracks_in_time.first] - nb_created <= 0)
                break;

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
            nb_created++;
        }
        N[ttracks_in_time.first] = N[ttracks_in_time.first] - nb_created;
    }

    // Init landmarks with tracks in time (+ seek for matches in frame if in stereo)
    for (auto &ttracks_in_time : _matches_in_time) {

        // Init all tracked feature with track in frame
        vec_match to_init;
        int nb_created = 0;
        for (auto &ttime : ttracks_in_time.second) {
            std::vector<std::shared_ptr<AFeature>> feats;

            // Check if we have enougth landmarks
            if (N[ttracks_in_time.first] - nb_created <= 0)
                break;

            to_init.push_back(ttime);
            feats.push_back(ttime.first);
            feats.push_back(ttime.second);
            nb_created++;

            // Add a new feature for triangulation if it is also matched in frame (bimono case)
            if (_slam_param->getDataProvider()->getNCam() == 2) {
                for (auto &tframe : _matches_in_frame[ttracks_in_time.first]) {
                    if (tframe.first->getLandmark().lock())
                        continue;

                    if (ttime.second == tframe.first) {
                        // Check if the feat has enough parallax
                        if ((tframe.first->getPoints().at(0) - tframe.second->getPoints().at(0)).norm() < 4) {
                            break;
                        } else {
                            to_init.push_back(tframe);
                            feats.push_back(tframe.second);
                            break;
                        }
                    }
                }
            }

            _slam_param->getLandmarksInitializer()[ttracks_in_time.first]->initFromFeatures(feats);
        }

        // Check number of missing landmarks after tracks_in_time_lmk
        N[ttracks_in_time.first] = N[ttracks_in_time.first] - nb_created;
    }

    // Initializing landmarks with L / R matches only in the worst case
    // Need to initialize the remaining N landmarks with in frame matches
    for (auto &ttracks_in_frame : _matches_in_frame) {

        // Init all tracked feature in frame
        vec_match to_init;
        int nb_created = 0;
        for (auto &tframe : ttracks_in_frame.second) {
            // Check if we have enough landmarks
            if (N[ttracks_in_frame.first] - nb_created <= 0)
                break;

            // Check if the feat has enough parallax
            if ((tframe.first->getPoints().at(0) - tframe.second->getPoints().at(0)).norm() < 4) {
                continue;
            }

            // If this frame match has already been used before (should be associated to a ldmk) continue
            if (tframe.first->getLandmark().lock())
                continue;
            else {
                to_init.push_back(tframe);
                nb_created++;
            }
        }
        _slam_param->getLandmarksInitializer()[ttracks_in_frame.first]->initFromMatches(to_init);
        N[ttracks_in_frame.first] = N[ttracks_in_frame.first] - nb_created;
    }
}

typed_vec_features SLAMCore::detectFeatures(std::shared_ptr<ImageSensor> &sensor) {

    typed_vec_features new_features;

    // For each detector launch an adaptive detector with bucketting
    for (auto &typed_detector : _slam_param->getFeatureDetectors()) {
        std::vector<std::shared_ptr<AFeature>> features;

        features = typed_detector.second->detectAndComputeGrid(
            sensor->getRawData(), sensor->getMask(), sensor->getFeatures()[typed_detector.first]);
        sensor->addFeatures(typed_detector.first, features);
        new_features[typed_detector.first] = features;
    }

    return new_features;
}

typed_vec_match SLAMCore::epipolarFiltering(std::shared_ptr<ImageSensor> &cam0,
                                            std::shared_ptr<ImageSensor> &cam1,
                                            typed_vec_match matches) {
    typed_vec_match valid_matches;
    Eigen::Affine3d T_c0_c1  = cam0->getWorld2SensorTransform() * cam1->getSensor2WorldTransform();
    Eigen::Vector3d epi_line = -T_c0_c1.translation();
    epi_line /= epi_line.norm();

    // Epipolar filtering for tracks_in_time, only for punctual landmarks
    for (auto &m : matches["pointxd"]) {

        // Check the angle with the epipolar plane
        Eigen::Vector3d ray1 = m.first->getBearingVectors().at(0);
        Eigen::Vector3d ray2 = T_c0_c1.rotation() * m.second->getBearingVectors().at(0);

        Eigen::Vector3d epiplane_normal = ray1.cross(epi_line);
        epiplane_normal /= epiplane_normal.norm();
        double residual = std::abs(epiplane_normal.dot(ray2));

        // The angular threshold is set to 1 degree
        if (90 - std::acos(residual) * 180 / M_PI < 0.5)
            valid_matches["pointxd"].push_back(m);
        else
            m.second->setOutlier();
    }

    return valid_matches;
}

uint SLAMCore::recoverFeatureFromMapLandmarks(std::shared_ptr<isae::AMap> localmap, std::shared_ptr<Frame> &f) {
    uint nb_resurected = 0;

    for (auto typed_ldmk : localmap->getLandmarks()) {
        nb_resurected += _slam_param->getFeatureMatchers()[typed_ldmk.first].feature_matcher->ldmk_match(
            f->getSensors().at(0), typed_ldmk.second, 5, 5);
    }

    return nb_resurected;
}

void SLAMCore::predictFeature(std::vector<std::shared_ptr<AFeature>> features,
                              std::shared_ptr<ImageSensor> sensor,
                              std::vector<std::shared_ptr<AFeature>> &features_init,
                              vec_match previous_matches = {}) {
    bool is_init;

    for (auto feature : features) {

        is_init = false;

        if (feature->getLandmark().lock()) {

            // Let's project the landmark with the predicted frame pose
            Eigen::Affine3d T_w_lmk = feature->getLandmark().lock()->getPose();
            std::vector<Eigen::Vector2d> predicted_p2ds;

            bool success;
            success = sensor->project(
                T_w_lmk, feature->getLandmark().lock()->getModel(), Eigen::Vector3d::Ones(), predicted_p2ds);

            if (success && std::isfinite(predicted_p2ds.at(0).x()) && std::isfinite(predicted_p2ds.at(0).y())) {
                features_init.push_back(std::make_shared<AFeature>(predicted_p2ds));
                is_init = true;
            }
        }

        if (is_init)
            continue;

        for (auto match : previous_matches) {

            if (feature == match.first) {
                features_init.push_back(match.second);
                is_init = true;
                break;
            }
        }

        if (is_init)
            continue;
        else
            features_init.push_back(feature);
    }
}

uint SLAMCore::matchFeatures(std::shared_ptr<ImageSensor> &sensor0,
                             std::shared_ptr<ImageSensor> &sensor1,
                             typed_vec_match &matches,
                             typed_vec_match &matches_lmk,
                             typed_vec_features features_to_track) {

    uint nb_matches = 0;

    for (const auto &typed_matcher : _slam_param->getFeatureMatchers()) {

        // Build features_init vector
        std::vector<std::shared_ptr<AFeature>> features_init;
        predictFeature(
            sensor0->getFeatures(typed_matcher.first), sensor1, features_init, _matches_in_time[typed_matcher.first]);

        matches[typed_matcher.first].clear();
        matches_lmk[typed_matcher.first].clear();

        nb_matches += typed_matcher.second.feature_matcher->match(sensor0->getFeatures(typed_matcher.first),
                                                                  sensor1->getFeatures(typed_matcher.first),
                                                                  features_init,
                                                                  matches[typed_matcher.first],
                                                                  matches_lmk[typed_matcher.first],
                                                                  typed_matcher.second.matcher_width,
                                                                  typed_matcher.second.matcher_height);
    }

    return nb_matches;
}

uint SLAMCore::trackFeatures(std::shared_ptr<ImageSensor> &sensor0,
                             std::shared_ptr<ImageSensor> &sensor1,
                             typed_vec_match &matches,
                             typed_vec_match &matches_lmk,
                             typed_vec_features features_to_track) {

    uint nb_tracks = 0;
    for (const auto &typed_tracker : _slam_param->getFeatureTrackers()) {

        // Build features_init vector
        std::vector<std::shared_ptr<AFeature>> features_init;
        predictFeature(features_to_track[typed_tracker.first], sensor1, features_init, matches[typed_tracker.first]);

        // Clear matches typed vec for update
        matches[typed_tracker.first].clear();
        matches_lmk[typed_tracker.first].clear();

        nb_tracks += typed_tracker.second.feature_tracker->track(sensor0,
                                                                 sensor1,
                                                                 features_to_track[typed_tracker.first],
                                                                 features_init,
                                                                 matches[typed_tracker.first],
                                                                 matches_lmk[typed_tracker.first],
                                                                 typed_tracker.second.tracker_width,
                                                                 typed_tracker.second.tracker_height,
                                                                 typed_tracker.second.tracker_nlvls_pyramids,
                                                                 typed_tracker.second.tracker_max_err,
                                                                 true);

        // std::cout << "SLAMCORE DEBUG = track type : " << typed_tracker.first << std::endl;
        // std::cout << "SLAMCORE DEBUG = total tracks : " << nb_tracks << std::endl;
    }

    return nb_tracks;
}

bool SLAMCore::shouldInsertKeyframe(std::shared_ptr<Frame> &f) {

    // Compute average translationnal parallax of matches
    double avg_parallax  = 0;
    double n_matches     = 0;
    double n_matches_lmk = 0;

    // Compute parallax
    for (auto tmatch : _matches_in_time_lmk) {
        n_matches += (_matches_in_time[tmatch.first].size() + _matches_in_time_lmk[tmatch.first].size());
        n_matches_lmk += _matches_in_time_lmk[tmatch.first].size();
    }

    for (auto tmatch : _matches_in_time) {
        for (auto match : tmatch.second) {
            avg_parallax +=
                std::acos(match.first->getBearingVectors().at(0).transpose() * match.second->getBearingVectors().at(0));
        }
    }

    for (auto tmatch : _matches_in_time_lmk) {
        for (auto match : tmatch.second) {
            avg_parallax +=
                std::acos(match.first->getBearingVectors().at(0).transpose() * match.second->getBearingVectors().at(0));
        }
    }

    avg_parallax /= (n_matches + n_matches_lmk);
    avg_parallax *= 180 / M_PI;

    // Check conditions to vote for a KF or not

    // Case when it is already a KF
    if (f->isKeyFrame()) {
        return true;
    }

    // Case when the parallax fall under the parallax noise condition => KF not voted
    if (avg_parallax < _min_movement_parallax) {
        return false;
    }

    // Case when the parallax in degree is over the threshold => KF voted
    if (avg_parallax > _max_movement_parallax) {
        f->setKeyFrame();
        return true;
    }

    // Case when many landmarks has been lost => KF voted
    if (n_matches_lmk < _min_lmk_number) {
        f->setKeyFrame();
        return true;
    }

    return false;
}

bool SLAMCore::predict(std::shared_ptr<Frame> &f) {
    Eigen::MatrixXd covdT = 100 * Eigen::MatrixXd::Identity(6, 6);

    // Predict pose with constant velocity model dT
    Eigen::Affine3d T_last_curr = (getLastKF()->getWorld2FrameTransform() * f->getFrame2WorldTransform());
    Eigen::Affine3d T_const     = T_last_curr;

    // False if the prediction failed and constant velocity applies
    if (!_slam_param->getPoseEstimator()->estimateTransformBetween(
            getLastKF(), f, _matches_in_time_lmk["pointxd"], T_last_curr, covdT)) {
        std::cerr << "Predict fails" << std::endl;

        f->setWorld2FrameTransform(T_last_curr.inverse() * getLastKF()->getWorld2FrameTransform());
        return false;
    } else {

        // Update the pose only for pnp
        if (_slam_param->_config.pose_estimator != "pnp")
            T_last_curr = T_const;

        // Check constant translation velocity assumption at 1000% (only if there is enough motion)
        if ((T_const.translation().norm() > 0.01 && T_last_curr.translation().norm() > 0.01) &&
            (T_const.translation() - T_last_curr.translation()).norm() / T_last_curr.translation().norm() > 10) {
            std::cout << "Constant velocity model failed FORCE IT " << std::endl;
            std::cout << "previous = " << std::endl << T_const.matrix() << std::endl;
            T_last_curr = T_const;
            f->setWorld2FrameTransform(T_last_curr.inverse() * getLastKF()->getWorld2FrameTransform());
            return false;
        }

        f->setWorld2FrameTransform(T_last_curr.inverse() * getLastKF()->getWorld2FrameTransform());

        return true;
    }
}

void SLAMCore::profiling() {

    if (!std::filesystem::is_directory("log_slam"))
        std::filesystem::create_directory("log_slam");

    if (!_is_init) {

        // Clean the result file
        std::ofstream fw_res("log_slam/results.csv", std::ofstream::out | std::ofstream::trunc);
        fw_res << "timestamp (ns), nframes, T_wf(00), T_wf(01), T_wf(02), T_wf(03), T_wf(10), T_wf(11), T_wf(12), "
               << "T_wf(13), T_wf(20), T_wf(21), T_wf(22), T_wf(23)\n";
        fw_res.close();

        // std::ofstream fw_res1("log_slam/info_mat.csv", std::ofstream::out | std::ofstream::trunc);
        // fw_res1 << "Im(00), Im(11), Im(22), Im(33), Im(44), Im(55), "
        //        << "If(00), If(11), If(22), If(33), If(44), If(55), "
        //        << "t_norm, r_norm, n_lmk\n";
        // fw_res1.close();

        // For timing statistics
        // // Clean profiling file
        // std::ofstream fw_prof_fefr("log_slam/timing_fe_fr.csv",
        //                            std::ofstream::out | std::ofstream::trunc);
        // fw_prof_fefr << "tracking_dt, predict_dt, epi_dt, filter_dt, optim_dt\n";
        // fw_prof_fefr.close();

        // // Clean profiling file
        // std::ofstream fw_prof_fekfr("log_slam/timing_fe_kfr.csv",
        //                             std::ofstream::out | std::ofstream::trunc);
        // fw_prof_fekfr
        //     << "tracking_dt, predict_dt, epi_dt, filter_dt, optim_dt, detect_dt, reover_dt, track_f_dt, init_dt\n";
        // fw_prof_fekfr.close();

        // // Clean profiling file
        // std::ofstream fw_prof_be("log_slam/timing_be.csv",
        //                             std::ofstream::out | std::ofstream::trunc);
        // fw_prof_be
        //     << "marg_dt, optim_dt, epi_dt\n";
        // fw_prof_be.close();
    } else {

        // Write in a txt file for evaluation
        if (getLastKF()) {
            std::shared_ptr<Frame> f = getLastKF();
            std::ofstream fw_res("log_slam/results.csv", std::ofstream::out | std::ofstream::app);
            Eigen::Affine3d T_w_f   = f->getFrame2WorldTransform();
            const Eigen::Matrix3d R = T_w_f.linear();
            Eigen::Vector3d twc     = T_w_f.translation();
            fw_res << f->getTimestamp() << "," << _nframes << "," << R(0, 0) << "," << R(0, 1) << "," << R(0, 2) << ","
                   << twc.x() << "," << R(1, 0) << "," << R(1, 1) << "," << R(1, 2) << "," << twc.y() << "," << R(2, 0)
                   << "," << R(2, 1) << "," << R(2, 2) << "," << twc.z() << "\n";
            fw_res.close();
        }

        // For timing statistics
        // if (!_timings_frate.empty()) {

        //     if (!_frame->isKeyFrame()) {
        //         if (_timings_frate.back().empty())
        //             return;
        //         std::ofstream fw_prof_fefr("log_slam/timing_fe_fr.csv",
        //                                    std::ofstream::out | std::ofstream::app);
        //         fw_prof_fefr << _timings_frate.back().at(0) << "," << _timings_frate.back().at(1) << ","
        //                      << _timings_frate.back().at(2) << "," << _timings_frate.back().at(3) << ","
        //                      << _timings_frate.back().at(4) << "\n";
        //         fw_prof_fefr.close();
        //     } else {
        //         if (_timings_kfrate_fe.back().size() < 9)
        //             return;
        //         std::ofstream fw_prof_fekfr("log_slam/timing_fe_kfr.csv",
        //                                     std::ofstream::out | std::ofstream::app);
        //         fw_prof_fekfr << _timings_kfrate_fe.back().at(0) << "," << _timings_kfrate_fe.back().at(1) << ","
        //                       << _timings_kfrate_fe.back().at(2) << "," << _timings_kfrate_fe.back().at(3) << ","
        //                       << _timings_kfrate_fe.back().at(4) << "," << _timings_kfrate_fe.back().at(5) << ","
        //                       << _timings_kfrate_fe.back().at(6) << "," << _timings_kfrate_fe.back().at(7) << ","
        //                       << _timings_kfrate_fe.back().at(8) << "\n";
        //         fw_prof_fekfr.close();

        //         std::ofstream fw_prof_be("log_slam/timing_be.csv",
        //                                     std::ofstream::out | std::ofstream::app);
        //         fw_prof_be << _timings_kfrate_be.back().at(0) << "," << _timings_kfrate_be.back().at(1) << "\n";
        //         fw_prof_be.close();
        //     }
        // }
    }

    // Write a txt file for profiling
    std::ofstream fw("log_slam/slam_profiler.txt", std::ofstream::out);
    fw << "===== SLAM profiler ======= \n";
    fw << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << "\n";
    fw << "Dataset: " << _slam_param->_config.dataset_id << "\n";
    fw << "Number of frames: " << _nframes << "\n";
    fw << "Number of keyframes: " << _nkeyframes << "\n";
    fw << "Img process dt: " << _avg_processing_t << "\n";
    fw << "Avg number of matches in frame: " << _avg_matches_frame << "\n";
    fw << "Avg number of matches in time: " << _avg_matches_time << "\n";
    fw << "Removed matches: " << _removed_feat << "\n";
    fw << "Avg number of lmks resurected: " << _avg_resur_lmk << "\n";
    fw << "Avg landmarks matched in map: " << _lmk_inmap << "\n";
    fw << "Detection dt: " << _avg_detect_t << "\n";
    fw << "Prediction " + _slam_param->_config.pose_estimator + "RANSAC dt: " << _avg_predict_t << "\n";
    fw << "Matching in frame dt: " << _avg_match_frame_t << "\n";
    fw << "Matching in time dt: " << _avg_match_time_t << "\n";
    fw << "Average filter time dt: " << _avg_filter_t << "\n";
    fw << "Average cleaning time dt: " << _avg_clean_t << "\n";
    fw << "Landmark init dt: " << _avg_lmk_init_t << "\n";
    fw << "Optimize frame dt: " << _avg_frame_opt_t << "\n";
    fw << "Marginalization dt: " << _avg_marg_t << "\n";
    if (_slam_param->_config.mesh3D)
        fw << "Mesh dt: " << _mesher->_avg_mesh_t << "\n";
    fw << "Optimize window dt: " << _avg_wdw_opt_t << "\n";
    float frontend_dt = _avg_detect_t + _avg_predict_t + _avg_match_frame_t * _nkeyframes / _nframes +
                        _avg_match_time_t + _avg_filter_t + _avg_clean_t + _avg_lmk_init_t * _nkeyframes / _nframes +
                        _avg_frame_opt_t + _avg_lmk_resur_t * _nkeyframes / _nframes;
    fw << "Front end dt: " << frontend_dt << "\n";
    float backend_dt = _avg_wdw_opt_t + _avg_marg_t;
    fw << "Back end dt: " << backend_dt << "\n";
}

void SLAMCore::runFrontEnd() {

    while (true) {

        if (!_is_init) {
            bool init_success = this->init();
            while (!init_success)
                init_success = this->init();
        } else
            this->frontEndStep();
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

void SLAMCore::runBackEnd() {

    while (true) {

        this->backEndStep();
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

void SLAMCore::runFullOdom() {

    while (true) {

        if (!_is_init) {
            bool init_success = this->init();
            while (!init_success)
                init_success = this->init();
        } else
            this->frontEndStep();

        this->backEndStep();
    }
}

} // namespace isae