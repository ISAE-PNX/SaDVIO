#ifndef SLAMPARAMETERS_H
#define SLAMPARAMETERS_H

#include "isaeslam/optimizers/AOptimizer.h"
#include <iostream>
#include <string>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

namespace isae {

class AFeatureDetector;
class AFeatureMatcher;
class AFeatureTracker;
class ADataProvider;
class APoseEstimator;
class ALandmarkInitializer;
class BundleAdjustmentCERES;
class LocalMap;

/*!
 * @brief A struct that contains a feature matcher and its parameters
 */
struct FeatureMatcherStruct {
    int matcher_width;
    int matcher_height;
    std::shared_ptr<AFeatureMatcher> feature_matcher;
};

/*!
 * @brief A struct that contains a feature tracker and its parameters
 */
struct FeatureTrackerStruct {
    int tracker_width;
    int tracker_height;
    int tracker_nlvls_pyramids;
    double tracker_max_err;
    std::shared_ptr<AFeatureTracker> feature_tracker;
};

/*!
 * @brief A struct that gathers all the parameters for a feature
 */
struct FeatureStruct {
    std::string label_feature;    //!< Label of the feature
    std::string detector_label;   //!< label of the feature detector
    int number_detected_features; //!< number of features to be detected by the detector
    int n_features_per_cell;      //!< number of features per cell for bucketting
    std::string tracker_label;    //!< class name of the tracker we will use in our SLAM
    int tracker_height;           //!< searchAreaHeight of tracker
    int tracker_width;            //!< searchAreaWidth of tracker
    int tracker_nlvls_pyramids;   //!< nlevels of pyramids for klt tracking
    double tracker_max_err;       //!< error threshold for klt tracking
    std::string matcher_label;    //!< class name of the matcher we will use in our SLAM
    double max_matching_dist;     //!< distance for matching
    int matcher_height;           //!< searchAreaHeight of tracker
    int matcher_width;            //!< searchAreaWidth of tracker
    std::string
        lmk_triangulator; //!< landmarkTriangulation class we will use to triangulate landmark of label_feature type
};

/*!
 * @brief This structure contains the configuration parameters located in the config file.
 */
struct Config {
    std::string dataset_path;    //!< Path to the dataset
    std::string dataset_id;      //!< Id of the dataset
    std::string slam_mode;       //!< SLAM mode (mono, bimono, monovio...)
    bool multithreading;         //!< Allow to run front-end and back-end on different threads (unstable...)
    bool enable_visu;            //!< Allow visualization
    std::string optimizer;       //!< Optimizer type (ReprojectionError, AngularError...)
    int contrast_enhancer;       //!< integer to choose the contrast enhancement algorithm
    float clahe_clip;            //!< Clip of CLAHE (useful only if it is chosen for contrast enhancement)
    float downsampling;          //!< Float to reduce the size of the image (0,5 = half the size of the img)
    int marginalization;         //!< 0 no marginalization, 1 marginalization
    bool sparsification;         //!< 0 no sparsification, 1 sparsification
    std::string pose_estimator;  //!< Type of pose estimator
    std::string rel_pose_estimator;  //!< Type of relative pose estimator (NFR or ESKF)
    float eskf_r;                    //!< Measurement covariance R of ESKF (pixels)
    std::string tracker;         //!< Type of tracking (matcher or klt)
    int min_kf_number;           //!< Minimum KF for optimization
    int max_kf_number;           //!< Size maximum of the sliding windown
    int fixed_frame_number;      //!< Number of fixed frame for gauge fixing
    float min_lmk_number;        //!< Below this number of landmark, a KF is voted
    float min_movement_parallax; //!< Below this parallax, no motion is considered
    float max_movement_parallax; //!< Over this parallax, a KF is voted
    bool mesh3D;                 //!< 0 no 3D mesh, 1 3D mesh
    double ZNCC_tsh;             //!< Threshold on ZNCC for triangle filtering
    double max_length_tsh;       //!< Threshold on maximum length for triangle filtering

    std::vector<FeatureStruct> features_handled; //!< types of features the slam will work on separated with commas (,)
};

/*!
 * @brief A class that gathers most of the algorithmic blocks of the SLAM system that can be setup in the config file
 *
 * Some attributes are sets as unordered map because these depends on the feature type (e.g. matcher, detector...). Then
 * the proper blocks can be called using the feature label.
 */
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
    void readConfigFile(const std::string &path_config_folder);
    Config _config;

  private:
    std::shared_ptr<ADataProvider> _data_provider;
    std::unordered_map<std::string, std::shared_ptr<AFeatureDetector>> _detector_map;
    std::unordered_map<std::string, FeatureTrackerStruct> _tracker_map;
    std::unordered_map<std::string, FeatureMatcherStruct> _matcher_map;
    std::unordered_map<std::string, std::shared_ptr<ALandmarkInitializer>> _lmk_init_map;

    std::shared_ptr<APoseEstimator> _pose_estimator;
    std::shared_ptr<AOptimizer> _optimizer_frontend, _optimizer_backend;
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
