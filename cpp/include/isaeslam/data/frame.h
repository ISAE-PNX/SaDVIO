#ifndef FRAME_H
#define FRAME_H

#include "isaeslam/data/features/AFeature2D.h"
#include "isaeslam/data/landmarks/ALandmark.h"
#include "isaeslam/typedefs.h"

// #include <iostream>

namespace isae {

class ImageSensor;
class ASensor;
class IMU;

/*!
 * @brief A Frame class representing a set of sensors and landmarks at a specific timestamp.
 *
 * A frame is the core component of a SLAM system. It gathers measurements for one or more sensors (e.g., cameras, IMUs)
 * at a specific timestamp. It contains pointers to sensor objects as well as pointer to landmarks that serves for
 * mapping and localization. A frame will be located in space with its pose and eventually, a covariance will
 * be assoiated to it. If a frame is considered as a keyframe, it will be incorporated in a map.
 */
class Frame : public std::enable_shared_from_this<Frame> {
  public:
    Frame() = default;

    static int _frame_count; //!< Static counter for frame IDs
    int _id;                 //!< Unique ID for the frame, assigned at initialization

    /*!
     * @brief Initialization of a frame with images
     */
    void init(const std::vector<std::shared_ptr<ImageSensor>> &sensors, unsigned long long timestamp);

    /*!
     * @brief Initialization of a frame with IMU
     */
    void init(std::shared_ptr<IMU> &imu, unsigned long long timestamp);

    /*!
     * @brief Initialization of a frame with a set of Sensors
     */
    void init(const std::vector<std::shared_ptr<ASensor>> &sensors, unsigned long long timestamp);

    std::vector<std::shared_ptr<ImageSensor>> getSensors() const { return _sensors; }
    std::shared_ptr<IMU> getIMU() const { return _imu; }
    void setIMU(std::shared_ptr<IMU> &imu, Eigen::Affine3d T_s_f);

    /*!
     * @brief free all the pointers related to sensors
     */
    void cleanSensors() {
        // std::cout << "Cleaning Sensors" << " (Frame " << _id << "/" << _frame_count << ")" << std::endl;
        _imu = nullptr;
        _sensors.clear();
    }

    /*!
     * @brief Set the pose of the world wrt the frame
     */
    void setWorld2FrameTransform(Eigen::Affine3d T_f_w) {
        std::lock_guard<std::mutex> lock(_frame_mtx);
        _T_f_w = T_f_w;
    }

    /*!
     * @brief Get the pose of the world wrt to the frame
     */
    Eigen::Affine3d getWorld2FrameTransform() const {
        std::lock_guard<std::mutex> lock(_frame_mtx);
        return _T_f_w;
    }

    /*!
     * @brief Get the pose of the frame in the world frame
     */
    Eigen::Affine3d getFrame2WorldTransform() const {
        std::lock_guard<std::mutex> lock(_frame_mtx);
        return _T_f_w.inverse();
    }

    void addLandmark(std::shared_ptr<ALandmark> ldmk) { _landmarks[ldmk->_label].push_back(ldmk); }
    void addLandmarks(isae::typed_vec_landmarks ldmks) {
        for (auto typed_ldmks : ldmks) {
            for (auto l : typed_ldmks.second)
                _landmarks[typed_ldmks.first].push_back(l);
        }
    }
    typed_vec_landmarks getLandmarks() const { return _landmarks; }

    uint getInMapLandmarksNumber() const;
    uint getLandmarksNumber() const {
        uint N = 0;
        for (const auto &typeldmk : _landmarks)
            N += typeldmk.second.size();
        return N;
    }
    void cleanLandmarks();

    unsigned long long getTimestamp() const { return _timestamp; }

    void setKeyFrame() { _is_kf = true; }
    void unsetKeyFrame() { _is_kf = false; }

    /*!
     * @brief Return true if the frame is a keyframe, false otherwise.
     */
    bool isKeyFrame() const { return _is_kf; }

    // Handles a prior on T_f_w
    bool hasPrior() const { return _has_prior; }
    void setPrior(Eigen::Affine3d T_prior, Vector6d inf_prior) {
        _has_prior   = true;
        _T_prior     = T_prior;
        _inf_T_prior = inf_prior;
    }
    Eigen::Affine3d getPrior() const { return _T_prior; }
    Vector6d getInfPrior() const { return _inf_T_prior; }

    // Handles relative pose covariance
    // The convention is wrt the previous frame
    void setdTCov(Eigen::MatrixXd dT_cov) { _dT_cov = dT_cov; }
    Eigen::MatrixXd getdTCov() const { return _dT_cov; }

  private:
    Eigen::Affine3d _T_f_w = Eigen::Affine3d::Identity(); //!< Pose of the frame
    typed_vec_landmarks _landmarks;                       //!< Landmarks associated to the frame as a typed vector

    unsigned long long _timestamp;                      //!< Timestamp of the frame in nanoseconds
    std::vector<std::shared_ptr<ImageSensor>> _sensors; //!< Image sensors associated to the frame
    std::shared_ptr<IMU> _imu;                          //!< IMU sensor associated to the frame

    bool _is_kf     = false; //!< Flag indicating if the frame is a keyframe
    bool _has_prior = false; //!< Flag indicating if the frame has a prior on its pose

    Eigen::Affine3d _T_prior = Eigen::Affine3d::Identity(); //!< Prior on the frame pose
    Vector6d _inf_T_prior    = Vector6d::Zero();            //!< Information matrix of the prior on the frame pose
    Eigen::MatrixXd _dT_cov =
        Eigen::MatrixXd::Identity(6, 6); //!< Covariance of the relative pose wrt the previous frame

    mutable std::mutex _frame_mtx;
};

} // namespace isae

#endif // FRAME_H
