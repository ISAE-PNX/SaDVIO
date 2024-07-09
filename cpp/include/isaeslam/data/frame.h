#ifndef FRAME_H
#define FRAME_H

#include "isaeslam/data/features/AFeature2D.h"
#include "isaeslam/data/landmarks/ALandmark.h"
#include "isaeslam/typedefs.h"

namespace isae {

class ImageSensor;
class ASensor;
class IMU;

class Frame : public std::enable_shared_from_this<Frame> {
  public:
    Frame() { _T_f_w = Eigen::Affine3d::Identity(); }

    // Frame id for bookeeping
    static int _frame_count;
    int _id;

    // Initialization of a frame with images
    void init(const std::vector<std::shared_ptr<ImageSensor>> &sensors, unsigned long long timestamp);

    // Initialization of a frame with IMU
    void init(std::shared_ptr<IMU> &imu, unsigned long long timestamp);

    // Initialization of a frame with a set of Sensors
    void init(const std::vector<std::shared_ptr<ASensor>> &sensors, unsigned long long timestamp);

    // get sensors list
    std::vector<std::shared_ptr<ImageSensor>> getSensors() const { return _sensors; }
    std::shared_ptr<IMU> getIMU() const { return _imu; }
    void setIMU(std::shared_ptr<IMU> &imu, Eigen::Affine3d T_s_f);

    // Frame pose
    void setWorld2FrameTransform(Eigen::Affine3d T_f_w) {
        std::lock_guard<std::mutex> lock(_frame_mtx);
        _T_f_w = T_f_w;
    }
    Eigen::Affine3d getWorld2FrameTransform() const {
        std::lock_guard<std::mutex> lock(_frame_mtx);
        return _T_f_w;
    }
    Eigen::Affine3d getFrame2WorldTransform() const {
        std::lock_guard<std::mutex> lock(_frame_mtx);
        return _T_f_w.inverse();
    }

    // add landmark
    void addLandmark(std::shared_ptr<ALandmark> ldmk) { _landmarks[ldmk->getLandmarkLabel()].push_back(ldmk); }
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
    bool isKeyFrame() const { return _is_kf; }

    // Handles a prior on T_f_w
    bool hasPrior() const { return _has_prior; }
    void setPrior(Eigen::Affine3d T_prior, Vector6d inf_prior) {
        _has_prior = true;
        _T_prior   = T_prior;
        _inf_prior = inf_prior;
    }
    Eigen::Affine3d getPrior() const { return _T_prior; }
    Vector6d getInfPrior() const { return _inf_prior; }

  private:
    Eigen::Affine3d _T_f_w;
    typed_vec_landmarks _landmarks;

    unsigned long long _timestamp;
    std::vector<std::shared_ptr<ImageSensor>> _sensors;
    std::shared_ptr<IMU> _imu;

    bool _is_kf     = false;
    bool _has_prior = false;

    Eigen::Affine3d _T_prior;
    Vector6d _inf_prior;

    mutable std::mutex _frame_mtx;
};

} // namespace isae

#endif // FRAME_H
