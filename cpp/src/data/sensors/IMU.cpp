#include "isaeslam/data/sensors/IMU.h"

namespace isae {

bool IMU::processIMU() {

    if (!_last_kf || !_last_IMU || !_frame) {
        return false;
    }

    // Case of wrong sync, return false
    if (_frame->getTimestamp() < _last_IMU->getFrame()->getTimestamp()) {
        return false;
    }

    // Bias propagation
    _ba = _last_IMU->getBa();
    _bg = _last_IMU->getBg();

    // Compute increments
    double dt = (_frame->getTimestamp() - _last_IMU->getFrame()->getTimestamp()) * 1e-9;

    if (dt > 1) {
        dt = 1 / _rate_hz;
    }

    double dt22         = 0.5 * dt * dt;
    Eigen::Vector3d dv  = (_last_IMU->getAcc() - _last_IMU->getBa()) * dt;
    Eigen::Vector3d dp  = (_last_IMU->getAcc() - _last_IMU->getBa()) * dt22;
    Eigen::Matrix3d dR  = geometry::exp_so3((_last_IMU->getGyr() - _last_IMU->getBg()) * dt);
    Eigen::Matrix3d Jrk = geometry::so3_rightJacobian((_last_IMU->getGyr() - _last_kf->getIMU()->getBg()) * dt);

    // Velocity update
    Eigen::Matrix3d R_w_fp = _last_IMU->getFrame()->getWorld2FrameTransform().rotation().transpose();
    _v                     = _last_IMU->getVelocity() + R_w_fp * dv + g * dt;

    // Pose update
    Eigen::Affine3d T_w_f            = _last_IMU->getFrame()->getWorld2FrameTransform().inverse();
    T_w_f.affine().block(0, 0, 3, 3) = R_w_fp * dR;
    T_w_f.affine().block(0, 3, 3, 1) += _last_IMU->getVelocity() * dt + R_w_fp * dp + g * dt22;
    _frame->setWorld2FrameTransform(T_w_f.inverse());

    // For covariance computation
    Eigen::MatrixXd B   = Eigen::Matrix<double, 9, 6>::Zero();
    B.block(0, 0, 3, 3) = Jrk * dt;
    B.block(3, 3, 3, 3) = _last_IMU->getDeltaR() * dt;
    B.block(6, 3, 3, 3) = _last_IMU->getDeltaR() * dt22;

    // Restart the pre integration if the last measurement is in a KF
    if (_last_IMU->getFrame()->isKeyFrame()) {
        _delta_p = dp;
        _delta_v = dv;
        _delta_R = dR;
        _Sigma   = B * _eta.asDiagonal() * B.transpose();
        _Sigma.block(6, 6, 3, 3) += 0.0001 * Eigen::Matrix3d::Identity() * dt; // integration cov
        _J_dR_bg = -Jrk * dt;
        _J_dv_ba = -Eigen::Matrix3d::Identity() * dt;
        _J_dv_bg = Eigen::Matrix3d::Zero();
        _J_dp_ba = -dt22 * Eigen::Matrix3d::Identity();
        _J_dp_bg = Eigen::Matrix3d::Zero();

    } else {

        // Compute the deltas
        _delta_R = _last_IMU->getDeltaR() * dR;
        _delta_v = _last_IMU->getDeltaV() + _last_IMU->getDeltaR() * dv;
        _delta_p = _last_IMU->getDeltaP() + _last_IMU->getDeltaV() * dt + _last_IMU->getDeltaR() * dp;

        // For covariance computation
        Eigen::MatrixXd A = Eigen::Matrix<double, 9, 9>::Identity();
        Eigen::Matrix3d dR_dA =
            _last_IMU->getDeltaR() * geometry::skewMatrix(_last_IMU->getAcc() - _last_kf->getIMU()->getBa());
        A.block(0, 0, 3, 3) = dR.transpose();
        A.block(3, 0, 3, 3) = -dR_dA * dt;
        A.block(6, 0, 3, 3) = -dR_dA * dt22;
        A.block(6, 3, 3, 3) = Eigen::Matrix3d::Identity() * dt;

        // Compute the covariance
        _Sigma = A * _last_IMU->getCov() * A.transpose() + B * _eta.asDiagonal() * B.transpose();
        _Sigma.block(6, 6, 3, 3) += 0.0001 * Eigen::Matrix3d::Identity() * dt; // integration cov

        // Compute the Jacobians w.r.t the bias
        _J_dR_bg = dR.transpose() * _last_IMU->_J_dR_bg - Jrk * dt;
        _J_dv_ba = _last_IMU->_J_dv_ba - _last_IMU->getDeltaR() * dt;
        _J_dv_bg = _last_IMU->_J_dv_bg - dR_dA * _last_IMU->_J_dR_bg * dt;
        _J_dp_ba = _last_IMU->_J_dp_ba + _last_IMU->_J_dv_ba * dt - dt22 * _last_IMU->getDeltaR();
        _J_dp_bg = _last_IMU->_J_dp_bg + _last_IMU->_J_dv_bg * dt - dt22 * dR_dA * _last_IMU->_J_dR_bg;
    }

    return true;
}

void IMU::estimateTransform(const std::shared_ptr<Frame> &frame1,
                            const std::shared_ptr<Frame> &frame2,
                            Eigen::Affine3d &dT) {

    double dt                     = (frame2->getTimestamp() - frame1->getTimestamp()) * 1e-9;
    dT                            = Eigen::Affine3d::Identity();
    Eigen::Matrix3d R1            = frame1->getWorld2FrameTransform().rotation();
    dT.translation()              = _delta_p + R1 * frame1->getIMU()->getVelocity() * dt + 0.5 * R1 * g * dt * dt;
    dT.affine().block(0, 0, 3, 3) = _delta_R;
}

void IMU::biasDeltaCorrection(Eigen::Vector3d d_ba, Eigen::Vector3d d_bg) {
    _delta_p = _delta_p + _J_dp_ba * d_ba + _J_dp_bg * d_bg;
    _delta_v = _delta_v + _J_dv_ba * d_ba + _J_dv_bg * d_bg;
    _delta_R = _delta_R * geometry::exp_so3(_J_dR_bg * d_bg);
}

void IMU::updateBiases() {
    if (!_last_kf->getIMU())
        return;

    _ba = _last_kf->getIMU()->getBa();
    _bg = _last_kf->getIMU()->getBg();
}

} // namespace isae