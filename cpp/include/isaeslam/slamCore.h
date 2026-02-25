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
#include "isaeslam/estimator/ESKFEstimator.h"
#include "isaeslam/estimator/EpipolarPoseEstimator.h"
#include "isaeslam/featuredetectors/opencv_detectors/cvORBFeatureDetector.h"
#include "isaeslam/featurematchers/Point2DFeatureMatcher.h"
#include "isaeslam/featurematchers/Point2DFeatureTracker.h"
#include "isaeslam/landmarkinitializer/Point3DlandmarkInitializer.h"
#include "isaeslam/optimizers/AngularAdjustmentCERESAnalytic.h"
#include "isaeslam/slamParameters.h"
#include "isaeslam/dataproviders/adataprovider.h"
#include "isaeslam/typedefs.h"
#include "utilities/timer.h"

namespace isae {

/*!
 * @brief The core abstract class of the SLAM system. It handles front-end, back-end and initialization.
 *
 * This is the skeleton of the SLAM system that will call all the subsystems (sensor processing, optimizer, map
 * processing...) It loads all the parameters of the SLAM and implements frontEnd and backEnds that must be called in a
 * separated thread. A few important methods relative to feature processing and pose estimation are implemented as well
 * to make the frontEndStep() and backEndStep() methods readable. Also, a few profiling variables and methods are
 * implemented to monitor the performances.
 */
class SLAMCore {
  public:
    SLAMCore(){};
    SLAMCore(std::shared_ptr<isae::SLAMParameters> slam_param);

    /*!
     * @brief Initialization step : create the first 3D landmarks and keyframe(s)
     */
    virtual bool init() = 0;

    /*!
     * @brief Front End: detection, tracking, pose estimation and landmark triangulation
     */
    virtual bool frontEndStep() = 0;

    /*!
     * @brief brief description Back End: marginalization, local map optimization
     */
    virtual bool backEndStep() = 0;

    /*!
     * @brief Thread for the backend
     */
    void runBackEnd();

    /*!
     * @brief Thread for the frontend
     */
    void runFrontEnd();

    /*!
     * @brief Thread for the backend
     */
    void runFullOdom();

    bool _is_init         = false; //!< Flag for initialization
    int _successive_fails = 0;     //!< Number of successive failure to trigger reinitialization

    // Public variables for display
    std::shared_ptr<isae::SLAMParameters> _slam_param;
    std::shared_ptr<Frame> _frame_to_display;
    std::shared_ptr<isae::LocalMap> _local_map_to_display;
    std::shared_ptr<isae::GlobalMap> _global_map_to_display;
    std::shared_ptr<Mesh3D> _mesh_to_display;

    /*!
     * @brief Detect all types of features for a given sensor with bucketting
     */
    typed_vec_features detectFeatures(std::shared_ptr<ImageSensor> &sensor);

    /*!
     * @brief Clean all the features that are outliers or are linked to outlier landmark
     */
    void cleanFeatures(std::shared_ptr<Frame> &f);

    /*!
     * @brief Predicts the position of the features and matches all the features between sensors
     *
     * @param sensor0 First sensor
     * @param sensor1 Second sensor
     * @param matches Typed vector of matches without landmarks passed as reference
     * @param matches_lmk Typed vector of matches with landmarks passed as reference
     * @param features_to_track Typed vector of the features on sensor0 that will be matched for prediction
     *
     * @return The number of tracks
     */
    uint matchFeatures(std::shared_ptr<ImageSensor> &sensor0,
                       std::shared_ptr<ImageSensor> &sensor1,
                       typed_vec_match &matches,
                       typed_vec_match &matches_lmk,
                       typed_vec_features features_to_track);

    /*!
     * @brief Predicts the position of the features and tracks all the features
     *
     * @param sensor0 First sensor
     * @param sensor1 Second sensor
     * @param matches Typed vector of matches without landmarks passed as reference
     * @param matches_lmk Typed vector of matches with landmarks passed as reference
     * @param features_to_track Typed vector of the features on sensor0 that will be tracked for prediction
     *
     * @return The number of tracks
     */
    uint trackFeatures(std::shared_ptr<ImageSensor> &sensor0,
                       std::shared_ptr<ImageSensor> &sensor1,
                       typed_vec_match &matches,
                       typed_vec_match &matches_lmk,
                       typed_vec_features features_to_track);

    /*!
     * @brief Predicts the position of a set of features on a given sensor
     *
     * @param features Set of features to be predicted
     * @param sensor Sensor on which the features has to be predicted
     * @param features_init The set of features predicted passed as a reference
     * @param previous_matches A prior on features_init that can eventually be used default is an empty set
     */
    void predictFeature(std::vector<std::shared_ptr<AFeature>> features,
                        std::shared_ptr<ImageSensor> sensor,
                        std::vector<std::shared_ptr<AFeature>> &features_init,
                        vec_match previous_matches);

    /*!
     * @brief Filters the matches between two sensors (with consistent pose estimates) using epipolar plane check
     *
     * @param cam0 First sensor
     * @param cam1 Second sensor
     * @param matches Typed vector of matches to be filtered
     *
     * @return A Typed vector of the valid matches
     */
    typed_vec_match
    epipolarFiltering(std::shared_ptr<ImageSensor> &cam0, std::shared_ptr<ImageSensor> &cam1, typed_vec_match matches);

    /*!
     * @brief Remove all the outlier features of _frame
     */
    void outlierRemoval();

    /*!
     * @brief Estimates the pose of a given frame
     *
     * @param f The frame whose pose is estimated passed as a reference
     *
     * @return True if the prediction is a success, False otherwise
     */
    bool predict(std::shared_ptr<Frame> &f);

    /*!
     * @brief Initialize all the landmarks of the current frame using _matches_in_time and _matches_in_frame
     */
    void initLandmarks(std::shared_ptr<Frame> &f);
    void updateLandmarks(typed_vec_match matches_lmk);

    /*!
     * @brief Performs landmark resurection
     *
     * @param localmap Local maps that contains the landmark to project
     * @param f Frame on which the landmarks are resurected
     *
     * @return The number of resurected landmarks
     */
    uint recoverFeatureFromMapLandmarks(std::shared_ptr<isae::AMap> localmap, std::shared_ptr<Frame> &f);

    /*!
     * @brief Determines if the frame in argument is a KF
     */
    bool shouldInsertKeyframe(std::shared_ptr<Frame> &f);
    std::shared_ptr<Frame> getLastKF() { return _local_map->getLastFrame(); }


    /*!
     * @brief TODO 
     */
    void initProfiling(const std::filesystem::path& p);
    void initProfiling() {
        // Create an empty path object to use default initialization
        initProfiling(std::filesystem::path("log_slam"));
    }

    /*!
     * @brief A function to monitor the SLAM behaviour
     */
    void profiling();

  protected:
    std::shared_ptr<Frame> _frame; //!< Current frame

    // Typed vector for matches
    typed_vec_match _matches_in_time;     //!< Typed vector of the matches between the last KF and _frame
    typed_vec_match _matches_in_time_lmk; //!< Typed vector of the matches with landmarks between the last KF and _frame
    typed_vec_match _matches_in_frame;    //!< Typed vector of the matches between the sensors of _frame
    typed_vec_match _matches_in_frame_lmk; //!< Typed vector of the matches with landmarks between the sensors of _frame

    // Local Map, Mesh and Keyframe voting policy
    std::shared_ptr<isae::LocalMap> _local_map;   //!< Current local map
    std::shared_ptr<isae::GlobalMap> _global_map; //!< Current global map
    std::shared_ptr<Mesher> _mesher;              //!< Mesher of the SLAM
    double _max_movement_parallax;                //!< Max parallax until a KF is voted
    double _min_movement_parallax;                //!< Under this parallax, no motion is considered
    double _min_lmk_number;                       //!< Under this number of landmark in _frame a KF is voted
    double _parallax;                             //!< Parallax between the last KF and _frame
    Vector6d _6d_velocity;                        //!< Current Velocity as a Twist vector

    // To ensure safe communication between threads
    std::mutex _map_mutex;
    std::shared_ptr<Frame> _frame_to_optim; //!< For communication between front-end and back-end

    // Profiling variables
    std::filesystem::path profiling_path;
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

/*!
 * @brief A SLAM class for bi-monocular setups
 */
class SLAMBiMono : public SLAMCore {

  public:
    SLAMBiMono(std::shared_ptr<SLAMParameters> slam_param) : SLAMCore(slam_param) {}

    bool init() override;
    bool frontEndStep() override;
    bool backEndStep() override;
};

/*!
 * @brief A SLAM class for bi-monocular + IMU setups
 */
class SLAMBiMonoVIO : public SLAMCore {

  public:
    SLAMBiMonoVIO(std::shared_ptr<SLAMParameters> slam_param) : SLAMCore(slam_param) {}

    bool init() override;
    bool frontEndStep() override;
    bool backEndStep() override;

    /*!
     * @brief Initialization steps with bi-monocular only for IMU initialization
     */
    bool step_init();

    /*!
     * @brief For profiling at IMU rate
     */
    void IMUprofiling();

  private:
    std::shared_ptr<IMU> _last_IMU; //!< Last IMU processed by the SLAM
};

/*!
 * @brief A SLAM class for monocular + IMU setups
 */
class SLAMMonoVIO : public SLAMCore {

  public:
    SLAMMonoVIO(std::shared_ptr<SLAMParameters> slam_param) : SLAMCore(slam_param) {}

    bool init() override;
    bool frontEndStep() override;
    bool backEndStep() override;

    /*!
     * @brief Initialization steps with bi-monocular only for IMU initialization
     */
    bool step_init();

  private:
    std::shared_ptr<IMU> _last_IMU; //!< Last IMU processed by the SLAM
};

/*!
 * @brief A SLAM class for monocular setups
 */
class SLAMMono : public SLAMCore {

  public:
    SLAMMono(std::shared_ptr<SLAMParameters> slam_param) : SLAMCore(slam_param) {}

    bool init() override;
    bool frontEndStep() override;
    bool backEndStep() override;
};

/*!
 * @brief A SLAM class for non overlapping FoV setups
 */
class SLAMNonOverlappingFov : public SLAMCore {

  public:
    SLAMNonOverlappingFov(){};
    SLAMNonOverlappingFov(std::shared_ptr<SLAMParameters> slam_param) : SLAMCore(slam_param) {}

    bool init() override;
    bool frontEndStep() override;
    bool backEndStep() override;

    /*!
     * @brief To remove outliers on both cameras
     */
    void outlierRemoval();

    /*!
     * @brief To init landmarks on both cameras
     */
    void initLandmarks(std::shared_ptr<Frame> &f);

    /*!
     * @brief To compute the scale using a single point in a RANSAC fashion
     *
     * @param T_cam0_cam1 Extrinsic between cameras
     * @param T_cam0_cam0p Up to scale estimation of motion of cam0
     * @param matches_cam1 Matches in time of cam1
     * @param lambda Scale estimate passed as reference
     *
     * @return Number of inliers of the RANSAC
     */
    int scaleEstimationRANSAC(const Eigen::Affine3d T_cam0_cam1,
                              const Eigen::Affine3d T_cam0_cam0p,
                              typed_vec_match matches_cam1,
                              double &lambda);

    /*!
     * @brief To check if the scale can't be recovered
     */
    bool isDegenerativeMotion(Eigen::Affine3d T_cam0_cam0p, Eigen::Affine3d T_cam0_cam1, typed_vec_match matches);

    /*!
     * @brief Prediction that takes into account both camera motions
     */
    bool predict(std::shared_ptr<Frame> &f);

  private:
    typed_vec_match _matches_in_time_cam1; //!< Typed vector of the matches between the last KF and _frame on cam1
    typed_vec_match
        _matches_in_time_cam1_lmk; //!< Typed vector of the matches with lmks between the last KF and _frame on cam1
};

} // namespace isae

#endif // SLAMCORE_H