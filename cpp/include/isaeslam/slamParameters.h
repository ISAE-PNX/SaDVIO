#ifndef SLAMPARAMETERS_H
#define SLAMPARAMETERS_H

#include <iostream>
#include <string>
#include <unordered_map>

#include "isaeslam/dataproviders/adataprovider.h"
#include "isaeslam/optimizers/AOptimizer.h"
#include "utilities/ConfigFileReader.h"

namespace isae {

class AFeatureDetector;
class AFeatureMatcher;
class AFeatureTracker;
class APoseEstimator;
class ALandmarkInitializer;
class BundleAdjustmentCERES;
class loopdetector;
class LocalMap;

struct FeatureMatcherStruct {
    int matcher_width;
    int matcher_height;
    std::shared_ptr<isae::AFeatureMatcher> feature_matcher;
};

struct FeatureTrackerStruct {
    int tracker_width;
    int tracker_height;
    int tracker_nlvls_pyramids;
    double tracker_max_err;
    std::shared_ptr<isae::AFeatureTracker> feature_tracker;
};

class SLAMParameters {
  public:
    SLAMParameters(const std::string config_file);

    std::shared_ptr<ADataProvider> getDataProvider() { return _data_provider; }
    std::unordered_map<std::string, std::shared_ptr<AFeatureDetector>> getFeatureDetectors() { return _detector_map; }
    std::unordered_map<std::string, FeatureTrackerStruct> getFeatureTrackers() { return _tracker_map; }
    std::unordered_map<std::string, FeatureMatcherStruct> getFeatureMatchers() { return _matcher_map; }
    std::unordered_map<std::string, std::shared_ptr<ALandmarkInitializer>> getLandmarksInitializer() {
        return _lmk_init_map;
    };

    std::shared_ptr<APoseEstimator> getPoseEstimator() { return _pose_estimator; }
    std::shared_ptr<AOptimizer> getOptimizerFront() { return _optimizer_frontend; }
    std::shared_ptr<AOptimizer> getOptimizerBack() { return _optimizer_backend; }
    isae::Config _config;

  private:

    std::shared_ptr<ADataProvider> _data_provider;
    std::unordered_map<std::string, std::shared_ptr<AFeatureDetector>> _detector_map;
    std::unordered_map<std::string, FeatureTrackerStruct> _tracker_map;
    std::unordered_map<std::string, FeatureMatcherStruct> _matcher_map;
    std::unordered_map<std::string, std::shared_ptr<ALandmarkInitializer>> _lmk_init_map;

    std::shared_ptr<APoseEstimator> _pose_estimator;
    std::shared_ptr<isae::AOptimizer> _optimizer_frontend, _optimizer_backend;

    std::shared_ptr<loopdetector> _loop_detect;
    std::shared_ptr<LocalMap> _local_map;

    void createProvider();
    void createDetectors();
    void createTrackers();
    void createMatchers();
    void createPoseEstimator();
    void createLandmarkInitializers();
    void createOptimizer();
};

} // namespace isae

#endif
