#ifndef SLAMCORE_H
#define SLAMCORE_H

#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

#include "isaeslam/data/features/AFeature2D.h"
#include "isaeslam/data/maps/globalmap.h"
#include "isaeslam/data/maps/localmap.h"
#include "isaeslam/data/mesh/mesh.h"
#include "isaeslam/data/mesh/mesher.h"
#include "isaeslam/data/sensors/ASensor.h"
#include "isaeslam/data/sensors/DoubleSphere.h"
#include "isaeslam/estimator/EpipolarPoseEstimator.h"
#include "isaeslam/featuredetectors/custom_detectors/Edgelet2DFeatureDetector.h"
#include "isaeslam/featuredetectors/custom_detectors/EllipsePatternFeatureDetector.h"
#include "isaeslam/featuredetectors/opencv_detectors/cvORBFeatureDetector.h"
#include "isaeslam/featurematchers/EdgeletFeatureMatcher.h"
#include "isaeslam/featurematchers/EdgeletFeatureTracker.h"
#include "isaeslam/featurematchers/EllipsePatternFeatureMatcher.h"
#include "isaeslam/featurematchers/EllipsePatternFeatureTracker.h"
#include "isaeslam/featurematchers/Point2DFeatureMatcher.h"
#include "isaeslam/featurematchers/Point2DFeatureTracker.h"
#include "isaeslam/landmarkinitializer/Edgelet3DlandmarkInitializer.h"
#include "isaeslam/landmarkinitializer/Point3DlandmarkInitializer.h"
#include "isaeslam/slamParameters.h"
#include "isaeslam/typedefs.h"
#include "utilities/timer.h"

namespace isae {

class SLAMCore {
  public:
    SLAMCore(){};
    SLAMCore(std::shared_ptr<isae::SLAMParameters> slam_param);

    // Initialization step : create the first 3D landmarks and keyframe(s)
    virtual bool init() = 0;

    // Front End: detection, tracking, pose estimation and landmark triangulation
    virtual bool frontEndStep() = 0;

    // Back End: marginalization, local map optimization
    virtual bool backEndStep() = 0;

    // Threads front and back
    void runBackEnd();
    void runFrontEnd();
    void runFullOdom();

    // Flag for init
    bool _is_init = false;

    // Public variables for display 
    std::shared_ptr<isae::SLAMParameters> _slam_param;
    std::shared_ptr<Frame> _frame_to_display;
    std::shared_ptr<isae::LocalMap> _local_map_to_display;
    std::shared_ptr<isae::GlobalMap> _global_map_to_display;
    std::shared_ptr<Mesh3D> _mesh_to_display;

    // Feature detection
    typed_vec_features detectFeatures(std::shared_ptr<ImageSensor> &sensor);
    void cleanFeatures(std::shared_ptr<Frame> &f);

    // Trackers and Matchers
    uint matchFeatures(std::shared_ptr<ImageSensor> &sensor0,
                       std::shared_ptr<ImageSensor> &sensor1,
                       typed_vec_match &matches,
                       typed_vec_match &matches_lmk,
                       typed_vec_features features_to_track);
    uint trackFeatures(std::shared_ptr<ImageSensor> &sensor0,
                       std::shared_ptr<ImageSensor> &sensor1,
                       typed_vec_match &matches,
                       typed_vec_match &matches_lmk,
                       typed_vec_features features_to_track);

    // Predict feature position in a given sensor
    void predictFeature(std::vector<std::shared_ptr<AFeature>> features,
                        std::shared_ptr<ImageSensor> sensor,
                        std::vector<std::shared_ptr<AFeature>> &features_init,
                        vec_match previous_matches);

    // Filtering
    typed_vec_match
    epipolarFiltering(std::shared_ptr<ImageSensor> &cam0, std::shared_ptr<ImageSensor> &cam1, typed_vec_match matches);
    void outlierRemoval();

    // Pose estimation
    bool predict(std::shared_ptr<Frame> &f);

    // Deal with mapping
    void initLandmarks(std::shared_ptr<Frame> &f);
    void updateLandmarks(typed_vec_match matches_lmk);
    uint recoverFeatureFromMapLandmarks(std::shared_ptr<isae::AMap> localmap, std::shared_ptr<Frame> &f);
    bool shouldInsertKeyframe(std::shared_ptr<Frame> &f);
    std::shared_ptr<Frame> getLastKF() { return _local_map->getLastFrame(); }

    // Profiling
    void profiling();

  protected:
    std::shared_ptr<Frame> _frame;

    // Typed vector for matches
    typed_vec_match _matches_in_time;
    typed_vec_match _matches_in_time_lmk;
    typed_vec_match _matches_in_frame;
    typed_vec_match _matches_in_frame_lmk;

    // Local Map, Mesh and Keyframe voting policy
    std::shared_ptr<isae::LocalMap> _local_map;
    std::shared_ptr<isae::GlobalMap> _global_map;
    std::shared_ptr<Mesher> _mesher;
    double _max_movement_parallax;
    double _min_movement_parallax;
    double _min_lmk_number;

    // To ensure safe communication between threads
    std::mutex _map_mutex;
    std::shared_ptr<Frame> _frame_to_optim;

    // Constant velocity model
    Vector6d _6d_velocity;

    // Profiling variables
    uint _nframes;
    uint _nkeyframes;
    float _avg_detect_t;
    float _avg_processing_t;
    float _avg_match_frame_t;
    float _avg_match_time_t;
    float _avg_filter_t;
    float _avg_lmk_init_t;
    float _avg_lmk_resur_t;
    float _avg_predict_t;
    float _avg_frame_opt_t;
    float _avg_clean_t;
    float _avg_marg_t;
    float _avg_wdw_opt_t;
    float _removed_lmk;
    float _removed_feat;
    float _lmk_inmap;
    float _avg_matches_time;
    float _avg_matches_frame;
    float _avg_resur_lmk;

    // For timing statistics
    std::vector<std::vector<float>> _timings_frate;
    std::vector<std::vector<float>> _timings_kfrate_fe;
    std::vector<std::vector<float>> _timings_kfrate_be;
};

class SLAMBiMono : public SLAMCore {

  public:
    SLAMBiMono(std::shared_ptr<SLAMParameters> slam_param) : SLAMCore(slam_param) {}

    bool init() override;
    bool frontEndStep() override;
    bool backEndStep() override;
};

class SLAMBiMonoVIO : public SLAMCore {

  public:
    SLAMBiMonoVIO(std::shared_ptr<SLAMParameters> slam_param) : SLAMCore(slam_param) {}

    bool init() override;
    bool frontEndStep() override;
    bool backEndStep() override;

    // Necessary to perform initialization of inertial variables
    bool step_init();

    // For a profiling at IMU rate
    void IMUprofiling();

  private:
    std::shared_ptr<IMU> _last_IMU;
};

class SLAMMonoVIO : public SLAMCore {

  public:
    SLAMMonoVIO(std::shared_ptr<SLAMParameters> slam_param) : SLAMCore(slam_param) {}

    bool init() override;
    bool frontEndStep() override;
    bool backEndStep() override;

    // Necessary to perform initialization of inertial variables
    bool step_init();

  private:
    std::shared_ptr<IMU> _last_IMU;
};

class SLAMMono : public SLAMCore {

  public:
    SLAMMono(std::shared_ptr<SLAMParameters> slam_param) : SLAMCore(slam_param) {}

    bool init() override;
    bool frontEndStep() override;
    bool backEndStep() override;
};

class SLAMNonOverlappingFov : public SLAMCore {

  public:
    SLAMNonOverlappingFov(){};
    SLAMNonOverlappingFov(std::shared_ptr<SLAMParameters> slam_param) : SLAMCore(slam_param) {}

    bool init() override;
    bool frontEndStep() override;
    bool backEndStep() override;

    // To remove outliers on both cameras
    void outlierRemoval();

    // To init landmarks on both cameras
    void initLandmarks(std::shared_ptr<Frame> &f);

    // To compute the scale using a single point in a RANSAC fashion
    int scaleEstimationRANSAC(const Eigen::Affine3d T_cam0_cam1,
                              const Eigen::Affine3d T_cam0_cam0p,
                              typed_vec_match matches_cam1,
                              double &lambda);

    // To check if the scale can't be recovered
    bool isDegenerativeMotion(Eigen::Affine3d T_cam0_cam0p, Eigen::Affine3d T_cam0_cam1, typed_vec_match matches);

    // Prediction that takes into account both camera motions
    bool predict(std::shared_ptr<Frame> &f);

  private:
    typed_vec_match _matches_in_time_cam1;
    typed_vec_match _matches_in_time_cam1_lmk;
};

} // namespace isae

#endif // SLAMCORE_H