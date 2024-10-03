#ifndef IMU_H
#define IMU_H

#include "isaeslam/data/sensors/ASensor.h"

namespace isae {

static Eigen::Vector3d g(0, 0, -9.81);

struct imu_config : sensor_config {
    double gyr_noise;
    double bgyr_noise;
    double acc_noise;
    double bacc_noise;
    double rate_hz;
};

class IMU : public ASensor {
  public:
    IMU(Eigen::Vector3d acc, Eigen::Vector3d gyr) : ASensor("imu"), _acc(acc), _gyr(gyr) {
        _v     = Eigen::Vector3d::Zero();
        _ba    = Eigen::Vector3d::Zero();
        _bg    = Eigen::Vector3d::Zero();
        _Sigma = Eigen::Matrix<double, 9, 9>::Zero();
    }
    IMU(std::shared_ptr<imu_config> config, Eigen::Vector3d acc, Eigen::Vector3d gyr)
        : ASensor("imu"), _acc(acc), _gyr(gyr) {
        _v          = Eigen::Vector3d::Zero();
        _gyr_noise  = config->gyr_noise;
        _acc_noise  = config->acc_noise;
        _bgyr_noise = config->bgyr_noise;
        _bacc_noise = config->bacc_noise;
        _rate_hz    = config->rate_hz;
        _Sigma      = Eigen::Matrix<double, 9, 9>::Zero();
        _ba         = Eigen::Vector3d::Zero();
        _bg         = Eigen::Vector3d::Zero();
        _delta_R    = Eigen::Matrix3d::Identity();

        _eta << _gyr_noise, _gyr_noise, _gyr_noise, _acc_noise, _acc_noise, _acc_noise;
        _eta = _eta.cwiseAbs2();
        _eta = _eta * _rate_hz;
    }
    ~IMU() {}

    Eigen::Vector3d getAcc() { return _acc; }
    Eigen::Vector3d getGyr() { return _gyr; }

    // This is a dirty trick to avoid that the shared_ptr of the frame desapear in the recursive process
    // TO DO: is there a better way? Maybe storing the variable of interest in an IMU object
    void setCurFrame(std::shared_ptr<Frame> frame) {
        _cur_frame = frame;
    }
    void setBa(Eigen::Vector3d ba) {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        _ba = ba;
    }
    void setBg(Eigen::Vector3d bg) {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        _bg = bg;
    }
    void setDeltaP(const Eigen::Vector3d delta_p) {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        _delta_p = delta_p;
    }
    void setDeltaV(const Eigen::Vector3d delta_v) {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        _delta_v = delta_v;
    }
    void setDeltaR(const Eigen::Matrix3d delta_R) {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        _delta_R = delta_R;
    }
    void setVelocity(const Eigen::Vector3d v) {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        _v = v;
    }
    Eigen::Vector3d getBa() {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _ba;
    }
    Eigen::Vector3d getBg() {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _bg;
    }
    Eigen::Vector3d getDeltaP() const {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _delta_p;
    }
    Eigen::Vector3d getDeltaV() const {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _delta_v;
    }
    Eigen::Matrix3d getDeltaR() const {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _delta_R;
    }
    Eigen::Vector3d getVelocity() const {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _v;
    }
    Eigen::MatrixXd getCov() const {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _Sigma;
    }
    double getGyrNoise() const { return _gyr_noise; }
    double getAccNoise() const { return _acc_noise; }
    double getbGyrNoise() const { return _bgyr_noise; }
    double getbAccNoise() const { return _bacc_noise; }

    void setLastKF(std::shared_ptr<Frame> frame) {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        _last_kf = frame;
    }
    std::shared_ptr<Frame> getLastKF() {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _last_kf;
    }
    void setLastIMU(std::shared_ptr<IMU> imu) {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        _last_IMU = imu;
    }
    std::shared_ptr<IMU> getLastIMU() {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _last_IMU;
    }

    // Compute the deltas for preintegration factor
    bool processIMU();

    // Compute the transformation between frames using the deltas
    void
    estimateTransform(const std::shared_ptr<Frame> &frame1, const std::shared_ptr<Frame> &frame2, Eigen::Affine3d &dT);

    // Update deltas with biases variations
    void biasDeltaCorrection(Eigen::Vector3d d_ba, Eigen::Vector3d d_bg);

    // Update biases (e.g. after optimization)
    void updateBiases();

    // Jacobians of deltas w.r.t. the bias
    Eigen::Matrix3d _J_dR_bg;
    Eigen::Matrix3d _J_dv_ba;
    Eigen::Matrix3d _J_dv_bg;
    Eigen::Matrix3d _J_dp_ba;
    Eigen::Matrix3d _J_dp_bg;

  private:
    // Measurements
    Eigen::Vector3d _acc;
    Eigen::Vector3d _gyr;

    // IMU noise
    double _gyr_noise;
    double _bgyr_noise;
    double _acc_noise;
    double _bacc_noise;
    double _rate_hz;
    Vector6d _eta;

    // States computed by processIMU()
    Eigen::Vector3d _delta_p;
    Eigen::Vector3d _delta_v;
    Eigen::Matrix3d _delta_R;
    Eigen::Vector3d _ba;
    Eigen::Vector3d _bg;
    Eigen::Vector3d _v;

    // Covariance computation
    Eigen::MatrixXd _Sigma;

    std::shared_ptr<IMU> _last_IMU;
    std::shared_ptr<Frame> _last_kf;
    std::shared_ptr<Frame> _cur_frame; 

    // Mutex
    mutable std::mutex _imu_mtx;
};

} // namespace isae

#endif