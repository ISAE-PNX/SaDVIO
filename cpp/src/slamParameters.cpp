#include "isaeslam/slamParameters.h"

#include "isaeslam/data/landmarks/BBox3d.h"
#include "isaeslam/data/landmarks/Line3D.h"
#include "isaeslam/data/landmarks/Point3D.h"

#include "isaeslam/featuredetectors/custom_detectors/Line2DFeatureDetector.h"
#include "isaeslam/featuredetectors/custom_detectors/csvKeypointDetector.h"
#include "isaeslam/featuredetectors/custom_detectors/semanticBBoxFeatureDetector.h"
#include "isaeslam/featuredetectors/opencv_detectors/cvBRISKFeatureDetector.h"
#include "isaeslam/featuredetectors/opencv_detectors/cvFASTFeatureDetector.h"
#include "isaeslam/featuredetectors/opencv_detectors/cvGFTTFeatureDetector.h"
#include "isaeslam/featuredetectors/opencv_detectors/cvKAZEFeatureDetector.h"
#include "isaeslam/featuredetectors/opencv_detectors/cvORBFeatureDetector.h"

#include "isaeslam/featurematchers/Point2DFeatureMatcher.h"
#include "isaeslam/featurematchers/Point2DFeatureTracker.h"
#include "isaeslam/featurematchers/Line2DFeatureMatcher.h"
#include "isaeslam/featurematchers/Line2DFeatureTracker.h"
#include "isaeslam/featurematchers/semanticBBoxFeatureMatcher.h"
#include "isaeslam/featurematchers/semanticBBoxFeatureTracker.h"

#include "isaeslam/landmarkinitializer/Line3DlandmarkInitializer.h"
#include "isaeslam/landmarkinitializer/Point3DlandmarkInitializer.h"
#include "isaeslam/landmarkinitializer/semanticBBoxlandmarkInitializer.h"

#include "isaeslam/estimator/EpipolarPoseEstimator.h"
#include "isaeslam/estimator/PnPPoseEstimator.h"

#include "isaeslam/optimizers/AngularAdjustmentCERESAnalytic.h"
#include "isaeslam/optimizers/BundleAdjustmentCERESAnalytic.h"
#include "isaeslam/optimizers/BundleAdjustmentCERESNumeric.h"
#include "isaeslam/dataproviders/adataprovider.h"

isae::SLAMParameters::SLAMParameters(const std::string config_folder_path) {
    std::cout << "------------------------------------" << std::endl;
    readConfigFile(config_folder_path);
    createProvider();
    createDetectors();
    createMatchers();
    createTrackers();
    createLandmarkInitializers();
    createPoseEstimator();
    createOptimizer();
    std::cout << "------------------------------------" << std::endl;
}

void isae::SLAMParameters::readConfigFile(const std::string &path_config_folder) {
    YAML::Node yaml_file = YAML::LoadFile(path_config_folder + "/config.yaml");

    // Dataset ID
    _config.dataset_id     = yaml_file["dataset_id"].as<std::string>();
    _config.dataset_path   = path_config_folder + "/dataset/" + _config.dataset_id + ".yaml";
    _config.slam_mode      = yaml_file["slam_mode"].as<std::string>();
    _config.enable_visu    = yaml_file["enable_visu"].as<int>();
    _config.multithreading = yaml_file["multithreading"].as<int>();

    // Image processing
    _config.contrast_enhancer = yaml_file["contrast_enhancer"].as<int>();
    _config.clahe_clip        = yaml_file["clahe_clip"].as<float>();
    _config.downsampling      = yaml_file["downsampling"].as<float>();

    // SLAM parameters
    _config.pose_estimator        = yaml_file["pose_estimator"].as<std::string>();
    _config.rel_pose_estimator    = yaml_file["rel_pose_estimator"].as<std::string>();
    _config.optimizer             = yaml_file["optimizer"].as<std::string>();
    _config.tracker               = yaml_file["tracker"].as<std::string>();
    _config.min_kf_number         = yaml_file["min_kf_number"].as<int>();
    _config.max_kf_number         = yaml_file["max_kf_number"].as<int>();
    _config.fixed_frame_number    = yaml_file["fixed_frame_number"].as<int>();
    _config.min_lmk_number        = yaml_file["min_lmk_number"].as<float>();
    _config.min_movement_parallax = yaml_file["min_movement_parallax"].as<float>();
    _config.max_movement_parallax = yaml_file["max_movement_parallax"].as<float>();
    _config.marginalization       = yaml_file["marginalization"].as<int>();
    _config.sparsification        = yaml_file["sparsification"].as<int>();
    _config.mesh3D                = yaml_file["mesh3d"].as<int>();
    _config.ZNCC_tsh              = yaml_file["ZNCC_tsh"].as<double>();
    _config.max_length_tsh        = yaml_file["max_length_tsh"].as<double>();
    _config.eskf_r                = yaml_file["eskf_r"].as<double>();
    _config.delta_norm            = yaml_file["delta_norm"].as<double>();
    _config.translation_norm      = yaml_file["translation_norm"].as<double>();

    // Features type
    YAML::Node features_node = yaml_file["features_handled"];

    for (YAML::iterator it = features_node.begin(); it != features_node.end(); ++it) {
        FeatureStruct feature_struct;
        feature_struct.label_feature            = (*it)["label_feature"].as<std::string>();
        feature_struct.detector_label           = (*it)["detector_label"].as<std::string>();
        feature_struct.number_detected_features = (*it)["number_detected_features"].as<int>();
        feature_struct.n_features_per_cell      = (*it)["n_features_per_cell"].as<int>();
        feature_struct.tracker_label            = (*it)["tracker_label"].as<std::string>();
        feature_struct.tracker_height           = (*it)["tracker_height"].as<int>();
        feature_struct.tracker_nlvls_pyramids   = (*it)["tracker_nlvls_pyramids"].as<int>();
        feature_struct.tracker_max_err          = (*it)["tracker_max_err"].as<double>();
        feature_struct.tracker_width            = (*it)["tracker_width"].as<int>();
        feature_struct.matcher_label            = (*it)["matcher_label"].as<std::string>();
        feature_struct.max_matching_dist        = (*it)["max_matching_dist"].as<double>();
        feature_struct.matcher_height           = (*it)["matcher_height"].as<int>();
        feature_struct.matcher_width            = (*it)["matcher_width"].as<int>();
        feature_struct.lmk_triangulator         = (*it)["lmk_triangulator"].as<std::string>();

        _config.features_handled.push_back(feature_struct);
    }
}

void isae::SLAMParameters::createProvider() {
    std::cout << "Create Data Provider" << std::endl;
    this->_data_provider = std::make_shared<ADataProvider>(_config.dataset_path, _config);
}

void isae::SLAMParameters::createDetectors() {

    std::cout << "Create Feature detectors" << std::endl;
    for (auto config_line : _config.features_handled) {
        if (config_line.detector_label == "cvORBFeatureDetector") {
            std::cout << "+ Adding cvORBFeatureDetector" << std::endl;
            isae::cvORBFeatureDetector orb_detector = isae::cvORBFeatureDetector(
                config_line.number_detected_features, config_line.n_features_per_cell, config_line.max_matching_dist);
            _detector_map[config_line.label_feature] = std::make_shared<isae::cvORBFeatureDetector>(orb_detector);
        } else if (config_line.detector_label == "cvKAZEFeatureDetector") {
            std::cout << "+ Adding cvKAZEFeatureDetector" << std::endl;
            isae::cvKAZEFeatureDetector kaze_detector = isae::cvKAZEFeatureDetector(
                config_line.number_detected_features, config_line.n_features_per_cell, config_line.max_matching_dist);
            _detector_map[config_line.label_feature] = std::make_shared<isae::cvKAZEFeatureDetector>(kaze_detector);
        } else if (config_line.detector_label == "cvBRISKFeatureDetector") {
            std::cout << "+ Adding cvBRISKFeatureDetector" << std::endl;
            isae::cvBRISKFeatureDetector brisk_detector = isae::cvBRISKFeatureDetector(
                config_line.number_detected_features, config_line.n_features_per_cell, config_line.max_matching_dist);
            _detector_map[config_line.label_feature] = std::make_shared<isae::cvBRISKFeatureDetector>(brisk_detector);
        } else if (config_line.detector_label == "cvFASTFeatureDetector") {
            std::cout << "+ Adding cvFASTFeatureDetector" << std::endl;
            isae::cvFASTFeatureDetector fast_detector = isae::cvFASTFeatureDetector(
                config_line.number_detected_features, config_line.n_features_per_cell, config_line.max_matching_dist);
            _detector_map[config_line.label_feature] = std::make_shared<isae::cvFASTFeatureDetector>(fast_detector);
        } else if (config_line.detector_label == "cvGFTTFeatureDetector") {
            std::cout << "+ Adding cvGFTTFeatureDetector" << std::endl;
            isae::cvGFTTFeatureDetector gftt_detector = isae::cvGFTTFeatureDetector(
                config_line.number_detected_features, config_line.n_features_per_cell, config_line.max_matching_dist);
            _detector_map[config_line.label_feature] = std::make_shared<isae::cvGFTTFeatureDetector>(gftt_detector);
        } else if (config_line.detector_label == "cvCSVFeatureDetector") {
            std::cout << "+ Adding cvCSVFeatureDetector" << std::endl;
            isae::CsvKeypointDetector SIFT_detector = isae::CsvKeypointDetector(
                config_line.number_detected_features, config_line.n_features_per_cell, config_line.max_matching_dist);
            _detector_map[config_line.label_feature] = std::make_shared<isae::CsvKeypointDetector>(SIFT_detector);
        } else if (config_line.detector_label == "Line2DFeatureDetector") {
            std::cout << "+ Adding Line2DFeatureDetector" << std::endl;
            isae::Line2DFeatureDetector lineDetector = isae::Line2DFeatureDetector(
                config_line.number_detected_features, config_line.n_features_per_cell, config_line.max_matching_dist);
            _detector_map[config_line.label_feature] = std::make_shared<isae::Line2DFeatureDetector>(lineDetector);
        } else if (config_line.detector_label == "semanticBBoxFeatureDetector") {
            std::cout << "+ Adding semanticBBoxFeatureDetector" << std::endl;
            isae::semanticBBoxFeatureDetector bboxFeatureDetector = isae::semanticBBoxFeatureDetector(
                config_line.number_detected_features, config_line.n_features_per_cell);
            _detector_map[config_line.label_feature] =
                std::make_shared<isae::semanticBBoxFeatureDetector>(bboxFeatureDetector);
        }
    }
}

void isae::SLAMParameters::createMatchers() {
    std::cout << "Create Feature Matchers" << std::endl;
    for (auto config_line : _config.features_handled) {

        isae::FeatureMatcherStruct matcher;
        matcher.matcher_height = config_line.matcher_height;
        matcher.matcher_width  = config_line.matcher_width;
        // Get the associated detector
        std::shared_ptr<AFeatureDetector> detector = _detector_map[config_line.label_feature];

        if (config_line.matcher_label == "Point2DFeatureMatcher") {
            std::cout << "+ Adding Point2DFeatureMatcher" << std::endl;
            isae::Point2DFeatureMatcher p2dMatcher(detector);
            matcher.feature_matcher                 = std::make_shared<Point2DFeatureMatcher>(p2dMatcher);
            _matcher_map[config_line.label_feature] = matcher;

        } else if (config_line.matcher_label == "Line2DFeatureMatcher") {
            std::cout << "+ Adding LineFeatureMatcher" << std::endl;
            isae::Line2DFeatureMatcher line2DMatcher(detector);
            matcher.feature_matcher                 = std::make_shared<Line2DFeatureMatcher>(line2DMatcher);
            _matcher_map[config_line.label_feature] = matcher;

        } else if (config_line.matcher_label == "semanticBBoxFeatureMatcher") {
            std::cout << "+ Adding semanticBBoxFeatureMatcher" << std::endl;
            isae::semanticBBoxFeatureMatcher BBoxMatcher(detector);
            matcher.feature_matcher                 = std::make_shared<semanticBBoxFeatureMatcher>(BBoxMatcher);
            _matcher_map[config_line.label_feature] = matcher;
        }
    }
}

void isae::SLAMParameters::createTrackers() {
    std::cout << "Create Feature Trackers" << std::endl;
    for (auto config_line : _config.features_handled) {

        FeatureTrackerStruct tracker;
        tracker.tracker_height         = config_line.tracker_height;
        tracker.tracker_width          = config_line.tracker_width;
        tracker.tracker_nlvls_pyramids = config_line.tracker_nlvls_pyramids;
        tracker.tracker_max_err        = config_line.tracker_max_err;

        // Get the associated detector
        std::shared_ptr<AFeatureDetector> detector = _detector_map[config_line.label_feature];

        if (config_line.tracker_label == "Point2DFeatureTracker") {
            std::cout << "+ Adding Point2DFeatureTracker" << std::endl;
            isae::Point2DFeatureTracker p2dTracker(detector);
            tracker.feature_tracker                 = std::make_shared<Point2DFeatureTracker>(p2dTracker);
            _tracker_map[config_line.label_feature] = tracker;
            
        } else if (config_line.tracker_label == "Line2DFeatureTracker") {
            std::cout << "+ Adding LineFeatureTracker" << std::endl;
            isae::Line2DFeatureTracker line2DTracker(detector);
            tracker.feature_tracker                 = std::make_shared<Line2DFeatureTracker>(line2DTracker);
            _tracker_map[config_line.label_feature] = tracker;

        } else if (config_line.tracker_label == "semanticBBoxFeatureTracker") {
            std::cout << "+ Adding semanticBBoxFeatureTracker" << std::endl;
            isae::semanticBBoxFeatureTracker BBoxTracker(detector);
            tracker.feature_tracker                 = std::make_shared<semanticBBoxFeatureTracker>(BBoxTracker);
            _tracker_map[config_line.label_feature] = tracker;
        }
    }
}

void isae::SLAMParameters::createLandmarkInitializers() {
    std::cout << "Create Landmarks Initializers" << std::endl;
    for (auto config_line : _config.features_handled) {

        if (config_line.lmk_triangulator == "Point3DLandmarkInitializer") {
            std::cout << "+ Adding Point3DLandmarkInitializer" << std::endl;
            _lmk_init_map[config_line.label_feature] = std::make_shared<Point3DLandmarkInitializer>();
        } else if (config_line.lmk_triangulator == "Line3DLandmarkInitializer") {
            std::cout << "+ Adding Line3DLandmarkInitializer " << std::endl;
            _lmk_init_map[config_line.label_feature] = std::make_shared<Line3DLandmarkInitializer>();
        } else if (config_line.lmk_triangulator == "semanticBBoxLandmarkInitializer") {
            std::cout << "+ Adding semanticBBoxLandmarkInitializer" << std::endl;
            _lmk_init_map[config_line.label_feature] = std::make_shared<semanticBBoxLandmarkInitializer>();
        }
    }
}

void isae::SLAMParameters::createPoseEstimator() {
    std::cout << "Create Interframe Pose Estimator" << std::endl;
    if (_config.pose_estimator == "epipolar") {
        std::cout << "+ Adding EpipolarPoseEstimator" << std::endl;
        isae::EpipolarPoseEstimator pose_estimator;
        _pose_estimator = std::make_shared<isae::EpipolarPoseEstimator>(pose_estimator);
    } else if (_config.pose_estimator == "imu") {
        std::cout << "+ Adding IMUPredictor -- TODO" << std::endl;

    } else if (_config.pose_estimator == "pnp") {
        std::cout << "+ Adding PnPPoseEstimator" << std::endl;
        isae::PnPPoseEstimator pose_estimator;
        _pose_estimator = std::make_shared<isae::PnPPoseEstimator>(pose_estimator);

    }
}

void isae::SLAMParameters::createOptimizer() {

    std::cout << "Create Optimizer" << std::endl;
    if (_config.optimizer == "Numeric") {
        std::cout << "+ Adding CERES optimizer with numerical jacobians" << std::endl;
        isae::BundleAdjustmentCERESNumeric ceres_ba;
        _optimizer_frontend = std::make_shared<isae::BundleAdjustmentCERESNumeric>(ceres_ba);
        _optimizer_backend  = std::make_shared<isae::BundleAdjustmentCERESNumeric>(ceres_ba);
    } else if (_config.optimizer == "Analytic") {
        std::cout << "+ Adding CERES optimizer with anlytical jacobians" << std::endl;
        isae::BundleAdjustmentCERESAnalytic ceres_ba;
        _optimizer_frontend = std::make_shared<isae::BundleAdjustmentCERESAnalytic>(ceres_ba);
        _optimizer_backend  = std::make_shared<isae::BundleAdjustmentCERESAnalytic>(ceres_ba);
    } else if (_config.optimizer == "AngularAnalytic") {
        std::cout << "+ Adding Angular error CERES optimizer with analytical jacobians" << std::endl;
        isae::AngularAdjustmentCERESAnalytic ceres_ba;
        _optimizer_frontend = std::make_shared<isae::AngularAdjustmentCERESAnalytic>(ceres_ba);
        _optimizer_backend  = std::make_shared<isae::AngularAdjustmentCERESAnalytic>(ceres_ba);
    }
}