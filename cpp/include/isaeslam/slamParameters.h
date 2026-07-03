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
    double delta_norm;           //!< Threshold on total delta change (rotation + translation) betweeen two frames
    double translation_norm;     //!< Threshold on total delta translation betweeen two frames

    std::vector<FeatureStruct> features_handled; //!< types of features the slam will work on separated with commas (,)
};

struct FeatureEvolution {
    unsigned long long _timestamp;      //!< Timestamp of the frame in nanoseconds
    uint _nframes;
    uint provided_by_prior_frame;       //!< Number of features detected in preceding frame's image
    uint tracked_in_new_frame;          //!< Number of features re-detected in new frame's image
        // NOTE: provided_by_prior_frame = found_in_new_frame + (features not detected again)
    uint tracked_lmk;                   //!< Number of features already associated to a landmark
    uint tracked_lmk_init;              //!< Subset of features associated to an INITIALIZED landmark
        // LET: tracked_lmk_noninit = (features whose landmark is not declared init.)
        // NOTE: tracked_lmk = tracked_lmk_init + tracked_lmk_noninit
    uint tracked_no_lmk;                //!< Number of features not yet associated to a landmark
        // LET: outliers = (features declared as outliers in prior step)
        // NOTE: tracked_in_new_frame = tracked_lmk + tracked_no_lmk + outliers
    uint used_matches;                  //!< Number of feature pairs actually used for the frame-to-frame pose estimate
        // LET: ransac_rejections = (feature pairs rejected by RANSAC / APoseEstimator)
        // NOTE: tracked_lmk_init = used_matches + ransac_rejections
    uint matches_in_time_lmk;           //!< Total number of matches in time, associated with a landmark (init or not), after pose estimation
        // NOTE: matches_in_time_lmk = used_matches + tracked_lmk_noninit
    uint matches_in_time_lmk_new_init;  //!< Number of non-initialized landmarks newly initialized
        // LET: matches_in_time_lmk_failed_init = (Features attempted to init unsuccessfully)
        // NOTE: tracked_lmk_noninit = matches_in_time_lmk_new_init + matches_in_time_lmk_failed_init
    uint matches_in_time;               //!< Number of filtered features not yet associated to a landmark
        // LET: EPIPOLAR_matches_in_time = (features rejected by epipolar filtering)
        // NOTE: found_tracks_no_lmk = matches_in_time + EPIPOLAR_matches_in_time
    uint matches_in_time_new_lmk;       //!< Number of features --shared between frames-- that were assigned a new, initialized landmark
    uint found_in_new_frame;            //!< Number of entirely new features detected in the new frame
    uint feat_provided_by_primarycam;   //!< Total number of features in the primary camera (typically cam0)
        // NOTE: feat_provided_by_primarycam = matches_in_time_lmk + matches_in_time + found_in_new_frame
    uint tracked_in_secondarycam;       //!< Number of features re-detected in 2nd camera (if stereo)
    uint tracked_in_secondarycam_lmk;   //!< Subset of re-detected features already associated to a landmark
    uint tracked_in_secondarycam_no_lmk;//!< Subset of re-detected features not yet associated to a landmark
        // NOTE: tracked_in_secondarycam = tracked_in_secondarycam_lmk + tracked_in_secondarycam_no_lmk;
    uint matches_in_frame;              //!< [tracked_in_secondarycam_no_lmk] after epipolar filtering
    uint matches_in_frame_lmk;          //!< [tracked_in_secondarycam_lmk] after epipolar filtering
    uint matches_in_frame_new_lmk;      //!< Number of features --shared within the frame-- that were assigned a new, initialized landmark
    uint ft_resur_new;                  //!< Number of new resurrected landmarks
        // LET: total_lmk_created = (all new landmarks / newly initialized landmarks in this frame)
        // NOTE: total_lmk_created = matches_in_frame_new_lmk + matches_in_time_new_lmk + matches_in_time_lmk_new_init
        // LET: ft_with_lmk = (Total number of features in primary camera, associated to an initialized landmark)
        // NOTE: ft_with_lmk = ft_resur_new + total_lmk_created
    uint ft_resur_total;                //!< Total number of features in current frame associated to any resurrected landmark
    uint ft_lmk_init_total;             //!< Total number of features in current frame associated to an initialized, non-resurrected landmark
    uint ft_lmk_noninit_total;          //!< Total number of features in current frame associated to a non-initialized, non-resurrected landmark
    uint ft_no_lmk_total;               //!< Total number of features in current frame not (yet) associated to any landmark
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
