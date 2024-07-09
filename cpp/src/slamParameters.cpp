#include "isaeslam/slamParameters.h"

#include "isaeslam/data/landmarks/BBox3d.h"
#include "isaeslam/data/landmarks/Edgelet3D.h"
#include "isaeslam/data/landmarks/Line3D.h"
#include "isaeslam/data/landmarks/Point3D.h"

#include "isaeslam/featuredetectors/custom_detectors/Edgelet2DFeatureDetector.h"
#include "isaeslam/featuredetectors/custom_detectors/EllipsePatternFeatureDetector.h"
#include "isaeslam/featuredetectors/custom_detectors/Line2DFeatureDetector.h"
#include "isaeslam/featuredetectors/custom_detectors/csvKeypointDetector.h"
#include "isaeslam/featuredetectors/custom_detectors/semanticBBoxFeatureDetector.h"
#include "isaeslam/featuredetectors/opencv_detectors/cvBRISKFeatureDetector.h"
#include "isaeslam/featuredetectors/opencv_detectors/cvFASTFeatureDetector.h"
#include "isaeslam/featuredetectors/opencv_detectors/cvGFTTFeatureDetector.h"
#include "isaeslam/featuredetectors/opencv_detectors/cvKAZEFeatureDetector.h"
#include "isaeslam/featuredetectors/opencv_detectors/cvORBFeatureDetector.h"
#include "isaeslam/featuredetectors/opencv_detectors/cvSTFeatureDetector.h"
#include "isaeslam/featurematchers/Point2DFeatureMatcher.h"
#include "isaeslam/featurematchers/Point2DFeatureTracker.h"

#include "isaeslam/featurematchers/EdgeletFeatureMatcher.h"
#include "isaeslam/featurematchers/EdgeletFeatureTracker.h"
#include "isaeslam/featurematchers/EllipsePatternFeatureMatcher.h"
#include "isaeslam/featurematchers/EllipsePatternFeatureTracker.h"
#include "isaeslam/featurematchers/Line2DFeatureMatcher.h"
#include "isaeslam/featurematchers/Line2DFeatureTracker.h"
#include "isaeslam/featurematchers/semanticBBoxFeatureMatcher.h"
#include "isaeslam/featurematchers/semanticBBoxFeatureTracker.h"

#include "isaeslam/landmarkinitializer/Edgelet3DlandmarkInitializer.h"
#include "isaeslam/landmarkinitializer/Line3DlandmarkInitializer.h"
#include "isaeslam/landmarkinitializer/Point3DlandmarkInitializer.h"
#include "isaeslam/landmarkinitializer/semanticBBoxlandmarkInitializer.h"

#include "isaeslam/estimator/EpipolarPoseEstimator.h"
#include "isaeslam/estimator/EpipolarPoseEstimatorCustom.h"
#include "isaeslam/estimator/PnPPoseEstimator.h"

#include "isaeslam/optimizers/AngularAdjustmentCERESAnalytic.h"
#include "isaeslam/optimizers/BundleAdjustmentCERESAnalytic.h"
#include "isaeslam/optimizers/BundleAdjustmentCERESNumeric.h"

isae::SLAMParameters::SLAMParameters(const std::string config_file) {
    std::cout << "------------------------------------" << std::endl;
    isae::ConfigFileReader configurator(config_file);
    _config = configurator._config;
    createProvider();
    createDetectors();
    createMatchers();
    createTrackers();
    createLandmarkInitializers();
    createPoseEstimator();
    createOptimizer();
    std::cout << "------------------------------------" << std::endl;
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
        } else if (config_line.detector_label == "Edgelet2DFeatureDetector") {
            std::cout << "+ Adding Edgelet2DFeatureDetector" << std::endl;
            isae::EdgeletFeatureDetector edgelet_detector = isae::EdgeletFeatureDetector(
                config_line.number_detected_features, config_line.n_features_per_cell, config_line.max_matching_dist);
            _detector_map[config_line.label_feature] = std::make_shared<isae::EdgeletFeatureDetector>(edgelet_detector);
        } else if (config_line.detector_label == "Line2DFeatureDetector") {
            std::cout << "+ Adding Line2DFeatureDetector" << std::endl;
            isae::Line2DFeatureDetector lineDetector = isae::Line2DFeatureDetector(
                config_line.number_detected_features, config_line.n_features_per_cell, config_line.max_matching_dist);
            _detector_map[config_line.label_feature] = std::make_shared<isae::Line2DFeatureDetector>(lineDetector);
        } else if (config_line.detector_label == "EllipsePatternDetector") {
            std::cout << "+ Adding EllipsePatternDetector" << std::endl;
            isae::EllipsePatternFeatureDetector ellipsePatternDetector = isae::EllipsePatternFeatureDetector(
                config_line.number_detected_features, config_line.n_features_per_cell);
            _detector_map[config_line.label_feature] =
                std::make_shared<isae::EllipsePatternFeatureDetector>(ellipsePatternDetector);
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

        } else if (config_line.matcher_label == "EdgeletFeatureMatcher") {
            std::cout << "+ Adding EdgeletFeatureMatcher" << std::endl;
            isae::EdgeletFeatureMatcher edgeletMatcher(detector);
            matcher.feature_matcher                 = std::make_shared<EdgeletFeatureMatcher>(edgeletMatcher);
            _matcher_map[config_line.label_feature] = matcher;

        } else if (config_line.matcher_label == "Line2DFeatureMatcher") {
            std::cout << "+ Adding LineFeatureMatcher" << std::endl;
            isae::Line2DFeatureMatcher line2DMatcher(detector);
            matcher.feature_matcher                 = std::make_shared<Line2DFeatureMatcher>(line2DMatcher);
            _matcher_map[config_line.label_feature] = matcher;

        } else if (config_line.matcher_label == "EllipsePatternFeatureMatcher") {
            std::cout << "+ Adding EllipsePatternFeatureMatcher" << std::endl;
            isae::EllipsePatternFeatureMatcher ellipseMatcher(detector);
            matcher.feature_matcher                 = std::make_shared<EllipsePatternFeatureMatcher>(ellipseMatcher);
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

        } else if (config_line.tracker_label == "EdgeletFeatureTracker") {
            std::cout << "+ Adding EdgeletFeatureTracker" << std::endl;
            isae::EdgeletFeatureTracker edgeletTracker(detector);
            tracker.feature_tracker                 = std::make_shared<EdgeletFeatureTracker>(edgeletTracker);
            _tracker_map[config_line.label_feature] = tracker;

        } else if (config_line.tracker_label == "Line2DFeatureTracker") {
            std::cout << "+ Adding LineFeatureTracker" << std::endl;
            isae::Line2DFeatureTracker line2DTracker(detector);
            tracker.feature_tracker                 = std::make_shared<Line2DFeatureTracker>(line2DTracker);
            _tracker_map[config_line.label_feature] = tracker;

        } else if (config_line.tracker_label == "EllipsePatternFeatureTracker") {
            std::cout << "+ Adding EllipsePatternFeatureTracker" << std::endl;
            isae::EllipsePatternFeatureTracker ellipseTracker(detector);
            tracker.feature_tracker                 = std::make_shared<EllipsePatternFeatureTracker>(ellipseTracker);
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
            isae::Point3DLandmarkInitializer p3dInit(config_line.number_kept_features);
            _lmk_init_map[config_line.label_feature] = std::make_shared<Point3DLandmarkInitializer>(p3dInit);

        } else if (config_line.lmk_triangulator == "Edgelet3DLandmarkInitializer") {
            std::cout << "+ Adding Edgelet3DLandmarkInitializer" << std::endl;
            isae::Edgelet3DLandmarkInitializer edgeletInit(config_line.number_kept_features);
            _lmk_init_map[config_line.label_feature] = std::make_shared<Edgelet3DLandmarkInitializer>(edgeletInit);

        } else if (config_line.lmk_triangulator == "Line3DLandmarkInitializer") {
            std::cout << "+ Adding Line3DLandmarkInitializer " << std::endl;
            isae::Line3DLandmarkInitializer lineInit(config_line.number_kept_features);
            _lmk_init_map[config_line.label_feature] = std::make_shared<Line3DLandmarkInitializer>(lineInit);

        } else if (config_line.lmk_triangulator == "EllipsePatternLandmarkInitializer") {
            std::cout << "+ Adding EllipsePatternLandmarkInitializer -- TODO" << std::endl;

        } else if (config_line.lmk_triangulator == "semanticBBoxLandmarkInitializer") {
            std::cout << "+ Adding semanticBBoxLandmarkInitializer" << std::endl;
            isae::semanticBBoxLandmarkInitializer bboxInit(config_line.number_kept_features);
            _lmk_init_map[config_line.label_feature] = std::make_shared<semanticBBoxLandmarkInitializer>(bboxInit);
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

    } else if (_config.pose_estimator == "epipolar_custom") {
        std::cout << "+ Adding EpipolarPoseEstimatorCustom" << std::endl;
        isae::EpipolarPoseEstimatorCustom pose_estimator;
        _pose_estimator = std::make_shared<isae::EpipolarPoseEstimatorCustom>(pose_estimator);
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