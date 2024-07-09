#ifndef RESIDUALS_H
#define RESIDUALS_H

#include "isaeslam/data/sensors/Camera.h"
#include "isaeslam/data/sensors/IMU.h"
#include <ceres/ceres.h>

namespace isae {

class Motion2DFactor : public ceres::SizedCostFunction<6, 6, 6> {
  public:
    Motion2DFactor(const Eigen::Vector2d motion_2d, const Eigen::MatrixXd sqrt_inf, const double dt)
        : _motion_2d(motion_2d), _sqrt_inf(sqrt_inf), _dt(dt) {
        _dx     = _motion_2d(0) * _dt;
        _dtheta = _motion_2d(1) * _dt;
    }
    Motion2DFactor() {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Affine3d T_fk_w   = geometry::se3_doubleVec6dtoRT(parameters[0]);
        Eigen::Affine3d T_fkp1_w = geometry::se3_doubleVec6dtoRT(parameters[1]);

        Eigen::Affine3d T_delta = Eigen::Affine3d::Identity();
        T_delta.linear() << std::cos(_dtheta), -std::sin(_dtheta), 0, std::sin(_dtheta), std::cos(_dtheta), 0, 0, 0, 1;
        T_delta.translation() << _dx, 0, 0;

        Eigen::Map<Vector6d> err(residuals);
        err = _sqrt_inf * geometry::se3_RTtoVec6d(T_fk_w.inverse() * T_delta * T_fkp1_w);

        if (jacobians != NULL) {
            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J_k(jacobians[0]);
                J_k.setIdentity();

                Eigen::Affine3d T_gamma = T_delta * T_fkp1_w;
                Eigen::Vector3d w       = geometry::log_so3(T_fk_w.linear().transpose() * T_gamma.linear());
                J_k.block<3, 3>(0, 0)   = -geometry::so3_rightJacobian(w).inverse() * T_gamma.linear().transpose() *
                                        T_fk_w.linear() *
                                        geometry::so3_rightJacobian(geometry::log_so3(T_fk_w.linear()));
                J_k.block<3, 3>(3, 0) =
                    T_fk_w.linear().transpose() * geometry::skewMatrix(T_gamma.translation() - T_fk_w.translation()) *
                    T_fk_w.linear() * geometry::so3_rightJacobian(geometry::log_so3(T_fk_w.linear()));
                J_k.block<3, 3>(3, 3) = -T_fk_w.linear().transpose();
                J_k                   = _sqrt_inf * J_k;
            }

            if (jacobians[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J_kp1(jacobians[1]);
                J_kp1.setIdentity();

                Eigen::Affine3d T_beta  = T_fk_w.inverse() * T_delta;
                Eigen::Vector3d w       = geometry::log_so3(T_beta.linear() * T_fkp1_w.linear());
                J_kp1.block<3, 3>(0, 0) = geometry::so3_rightJacobian(w).inverse() *
                                          geometry::so3_rightJacobian(geometry::log_so3(T_fkp1_w.linear()));
                J_kp1.block<3, 3>(3, 3) = T_beta.linear();
                J_kp1                   = _sqrt_inf * J_kp1;
            }
        }

        return true;
    }

    Eigen::Vector2d _motion_2d;
    Eigen::MatrixXd _sqrt_inf;
    double _dt;
    double _dx;
    double _dtheta;
};

class Relative6DPose : public ceres::SizedCostFunction<6, 6, 6> {
  public:
    Relative6DPose(const Eigen::Affine3d T_w_a,
                   const Eigen::Affine3d T_w_b,
                   const Eigen::Affine3d T_a_b_prior,
                   const Eigen::MatrixXd sqrt_inf)
        : _T_w_a(T_w_a), _T_w_b(T_w_b), _T_a_b_prior(T_a_b_prior), _sqrt_inf(sqrt_inf) {}
    Relative6DPose() {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Map<Vector6d> err(residuals);
        Eigen::Affine3d T_w_a_up    = _T_w_a * geometry::se3_doubleVec6dtoRT(parameters[0]);
        Eigen::Affine3d T_w_b_up    = _T_w_b * geometry::se3_doubleVec6dtoRT(parameters[1]);
        Eigen::Affine3d T_b_a_prior = _T_a_b_prior.inverse();
        Eigen::Affine3d T           = T_b_a_prior * T_w_a_up.inverse() * T_w_b_up;
        err                         = _sqrt_inf * geometry::se3_RTtoVec6d(T);

        if (jacobians != NULL) {

            Eigen::Vector3d tb = T_w_b_up.translation();
            Eigen::Vector3d ta = T_w_a_up.translation();
            Eigen::Vector3d w  = geometry::log_so3(T.rotation());

            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J(jacobians[0]);
                J.setIdentity();
                Eigen::Vector3d dw = Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]);

                // d(log(dr)) / d(taua)
                J.block(0, 0, 3, 3) = -geometry::so3_rightJacobian(w).inverse() * T_w_b_up.rotation().transpose() *
                                      T_w_a_up.rotation() * geometry::so3_rightJacobian(dw);

                // d(dt) / d(taua)
                J.block(3, 0, 3, 3) = T_b_a_prior.rotation() * T_w_a_up.rotation().transpose() *
                                      geometry::skewMatrix(tb - ta) * T_w_a_up.rotation() *
                                      geometry::so3_rightJacobian(dw);

                // d(dt) / d(ta)
                J.block(3, 3, 3, 3) = -T_b_a_prior.rotation();
                J                   = _sqrt_inf * J;
            }

            if (jacobians[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J(jacobians[1]);
                J.setIdentity();
                Eigen::Vector3d dw = Eigen::Vector3d(parameters[1][0], parameters[1][1], parameters[1][2]);

                // d(log(dr)) / d(taub)
                J.block(0, 0, 3, 3) = geometry::so3_rightJacobian(w).inverse() * geometry::so3_rightJacobian(dw);

                // d(dt) / d(tb)
                J.block(3, 3, 3, 3) = T_b_a_prior.rotation() * T_w_a_up.rotation().transpose() * T_w_b_up.rotation();
                J                   = _sqrt_inf * J;
            }
        }

        return true;
    }

    Eigen::Affine3d _T_w_a, _T_w_b, _T_a_b_prior;
    Eigen::MatrixXd _sqrt_inf;
};

class IMUFactor : public ceres::SizedCostFunction<9, 6, 6, 3, 3, 3, 3> {
  public:
    IMUFactor(const std::shared_ptr<IMU> imu_i, const std::shared_ptr<IMU> imu_j) : _imu_i(imu_i), _imu_j(imu_j) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {

        // Beware: the assumption here is that the IMU frame is the main frame
        Eigen::Affine3d T_fi_w =
            _imu_i->getFrame()->getWorld2FrameTransform() * geometry::se3_doubleVec6dtoRT(parameters[0]);
        Eigen::Affine3d T_fj_w =
            _imu_j->getFrame()->getWorld2FrameTransform() * geometry::se3_doubleVec6dtoRT(parameters[1]);
        Eigen::Vector3d v_i  = _imu_i->getVelocity() + Eigen::Map<const Eigen::Vector3d>(parameters[2]);
        Eigen::Vector3d v_j  = _imu_j->getVelocity() + Eigen::Map<const Eigen::Vector3d>(parameters[3]);
        Eigen::Vector3d d_ba = Eigen::Map<const Eigen::Vector3d>(parameters[4]);
        Eigen::Vector3d d_bg = Eigen::Map<const Eigen::Vector3d>(parameters[5]);
        double dtij          = (_imu_j->getFrame()->getTimestamp() - _imu_i->getFrame()->getTimestamp()) * 1e-9;

        // Get the information matrix
        Eigen::MatrixXd cov = _imu_j->getCov();
        // cov                 = cov / cov(0, 0); // Normalize the covariance to have weights ~ 1
        Eigen::Matrix<double, 9, 9> inf_sqrt =
            Eigen::LLT<Eigen::Matrix<double, 9, 9>>(cov.inverse()).matrixL().transpose();

        // Derive the residuals
        Eigen::Matrix3d dR = (_imu_j->getDeltaR() * geometry::exp_so3(_imu_j->_J_dR_bg * d_bg)).transpose() *
                             T_fi_w.rotation() * T_fj_w.rotation().transpose();
        Eigen::Vector3d r_dr = geometry::log_so3(dR);
        Eigen::Vector3d r_dv = T_fi_w.rotation() * (v_j - v_i - g * dtij) -
                               (_imu_j->getDeltaV() + _imu_j->_J_dv_bg * d_bg + _imu_j->_J_dv_ba * d_ba);
        Eigen::Vector3d r_dp = T_fi_w.rotation() * (T_fj_w.inverse().translation() - T_fi_w.inverse().translation() -
                                                    v_i * dtij - 0.5 * g * dtij * dtij) -
                               (_imu_j->getDeltaP() + _imu_j->_J_dp_bg * d_bg + _imu_j->_J_dp_ba * d_ba);
        Eigen::Map<Eigen::Matrix<double, 9, 1>> err(residuals);
        err.block(0, 0, 3, 1) = r_dr;
        err.block(3, 0, 3, 1) = r_dv;
        err.block(6, 0, 3, 1) = r_dp;
        err                   = inf_sqrt * err;

        if (jacobians != NULL) {

            // Jacobian wrt the pose i
            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 6, Eigen::RowMajor>> J_dTfi(jacobians[0]);
                J_dTfi                   = Eigen::Matrix<double, 9, 6, Eigen::RowMajor>::Zero();
                Eigen::Vector3d w_dfi    = Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]);
                Eigen::Matrix3d J_r_wdfi = geometry::so3_rightJacobian(w_dfi);
                J_dTfi.block(0, 0, 3, 3) = geometry::so3_rightJacobian(r_dr).inverse() * T_fj_w.rotation() * J_r_wdfi;
                J_dTfi.block(3, 0, 3, 3) = -T_fi_w.rotation() * geometry::skewMatrix(v_j - v_i - g * dtij) * J_r_wdfi;
                J_dTfi.block(6, 0, 3, 3) =
                    -T_fi_w.rotation() *
                    geometry::skewMatrix(T_fj_w.inverse().translation() - v_i * dtij - 0.5 * g * dtij * dtij) *
                    J_r_wdfi;
                J_dTfi.block(6, 3, 3, 3) = _imu_i->getFrame()->getWorld2FrameTransform().rotation();
                J_dTfi                   = inf_sqrt * J_dTfi;
            }

            // Jacobian wrt the pose j
            if (jacobians[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 6, Eigen::RowMajor>> J_dTfj(jacobians[1]);
                J_dTfj                   = Eigen::Matrix<double, 9, 6, Eigen::RowMajor>::Zero();
                Eigen::Vector3d w_dfj    = Eigen::Vector3d(parameters[1][0], parameters[1][1], parameters[1][2]);
                Eigen::Matrix3d J_r_wdfj = geometry::so3_rightJacobian(w_dfj);
                J_dTfj.block(0, 0, 3, 3) = -geometry::so3_rightJacobian(r_dr).inverse() * T_fj_w.rotation() * J_r_wdfj;
                J_dTfj.block(6, 0, 3, 3) = -T_fi_w.rotation() * T_fj_w.rotation().transpose() *
                                           geometry::skewMatrix(T_fj_w.translation()) * T_fj_w.rotation() * J_r_wdfj;
                J_dTfj.block(6, 3, 3, 3) = -T_fi_w.rotation() * geometry::exp_so3(w_dfj).transpose();
                J_dTfj                   = inf_sqrt * J_dTfj;
            }

            // Jacobian wrt the velocity i
            if (jacobians[2] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J_dvi(jacobians[2]);
                J_dvi                   = Eigen::Matrix<double, 9, 3, Eigen::RowMajor>::Zero();
                J_dvi.block(3, 0, 3, 3) = -T_fi_w.rotation();
                J_dvi.block(6, 0, 3, 3) = -T_fi_w.rotation() * dtij;
                J_dvi                   = inf_sqrt * J_dvi;
            }

            // Jacobian wrt the velocity j
            if (jacobians[3] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J_dvj(jacobians[3]);
                J_dvj                   = Eigen::Matrix<double, 9, 3, Eigen::RowMajor>::Zero();
                J_dvj.block(3, 0, 3, 3) = T_fi_w.rotation();
                J_dvj                   = inf_sqrt * J_dvj;
            }

            // Jacobian wrt the bias of the accelerometer
            if (jacobians[4] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J_dba(jacobians[4]);
                J_dba                   = Eigen::Matrix<double, 9, 3, Eigen::RowMajor>::Zero();
                J_dba.block(3, 0, 3, 3) = -_imu_j->_J_dv_ba;
                J_dba.block(6, 0, 3, 3) = -_imu_j->_J_dp_ba;
                J_dba                   = inf_sqrt * J_dba;
            }

            // Jacobian wrt the bias of the gyroscope
            if (jacobians[5] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J_dbg(jacobians[5]);
                J_dbg                   = Eigen::Matrix<double, 9, 3, Eigen::RowMajor>::Zero();
                J_dbg.block(0, 0, 3, 3) = -geometry::so3_rightJacobian(r_dr).inverse() * dR.transpose() *
                                          geometry::so3_rightJacobian(_imu_j->_J_dR_bg * d_bg) * _imu_j->_J_dR_bg;
                J_dbg.block(3, 0, 3, 3) = -_imu_j->_J_dv_bg;
                J_dbg.block(6, 0, 3, 3) = -_imu_j->_J_dp_bg;
                J_dbg                   = inf_sqrt * J_dbg;
            }
        }

        return true;
    }

    std::shared_ptr<IMU> _imu_i;
    std::shared_ptr<IMU> _imu_j;
};

class IMUBiasFactor : public ceres::SizedCostFunction<6, 3, 3, 3, 3> {

  public:
    IMUBiasFactor(const std::shared_ptr<IMU> imu_i, const std::shared_ptr<IMU> imu_j) : _imu_i(imu_i), _imu_j(imu_j) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Vector3d d_bai = Eigen::Map<const Eigen::Vector3d>(parameters[0]);
        Eigen::Vector3d d_bgi = Eigen::Map<const Eigen::Vector3d>(parameters[1]);
        Eigen::Vector3d d_baj = Eigen::Map<const Eigen::Vector3d>(parameters[2]);
        Eigen::Vector3d d_bgj = Eigen::Map<const Eigen::Vector3d>(parameters[3]);
        double dtij           = (_imu_j->getFrame()->getTimestamp() - _imu_i->getFrame()->getTimestamp()) * 1e-9;

        double sigma2_dba = dtij * _imu_i->getbAccNoise() * _imu_i->getbAccNoise(); // / _imu_j->getCov()(0, 0);
        Eigen::Matrix3d sqrt_inf_ba = Eigen::Matrix3d::Identity() * (1 / std::sqrt(sigma2_dba));
        double sigma2_dbg = dtij * _imu_i->getbGyrNoise() * _imu_i->getbGyrNoise(); // / _imu_j->getCov()(0, 0);
        Eigen::Matrix3d sqrt_inf_bg = Eigen::Matrix3d::Identity() * (1 / std::sqrt(sigma2_dbg));

        Eigen::Map<Eigen::Matrix<double, 6, 1>> err(residuals);
        err.block(0, 0, 3, 1) = sqrt_inf_ba * (_imu_j->getBa() + d_baj - _imu_i->getBa() - d_bai);
        err.block(3, 0, 3, 1) = sqrt_inf_bg * (_imu_j->getBg() + d_bgj - _imu_i->getBg() - d_bgi);

        if (jacobians != NULL) {

            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J_dba(jacobians[0]);
                J_dba                   = Eigen::Matrix<double, 6, 3, Eigen::RowMajor>::Zero();
                J_dba.block(0, 0, 3, 3) = -sqrt_inf_ba;
            }

            if (jacobians[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J_dbg(jacobians[1]);
                J_dbg                   = Eigen::Matrix<double, 6, 3, Eigen::RowMajor>::Zero();
                J_dbg.block(3, 0, 3, 3) = -sqrt_inf_bg;
            }

            if (jacobians[2] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J_dba(jacobians[2]);
                J_dba                   = Eigen::Matrix<double, 6, 3, Eigen::RowMajor>::Zero();
                J_dba.block(0, 0, 3, 3) = sqrt_inf_ba;
            }

            if (jacobians[3] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J_dbg(jacobians[3]);
                J_dbg                   = Eigen::Matrix<double, 6, 3, Eigen::RowMajor>::Zero();
                J_dbg.block(3, 0, 3, 3) = sqrt_inf_bg;
            }
        }

        return true;
    }

    std::shared_ptr<IMU> _imu_i;
    std::shared_ptr<IMU> _imu_j;
};

class IMUFactorInit : public ceres::SizedCostFunction<9, 2, 3, 3, 3, 3, 1> {
  public:
    IMUFactorInit(const std::shared_ptr<IMU> imu_i, const std::shared_ptr<IMU> imu_j) : _imu_i(imu_i), _imu_j(imu_j) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {

        // Beware: the assumption here is that the IMU frame is the main frame
        Eigen::Vector3d w_w_i  = Eigen::Vector3d(parameters[0][0], parameters[0][1], 0);
        Eigen::Matrix3d R_w_i  = geometry::exp_so3(w_w_i);
        Eigen::Vector3d v_i    = _imu_i->getVelocity() + Eigen::Map<const Eigen::Vector3d>(parameters[1]);
        Eigen::Vector3d v_j    = _imu_j->getVelocity() + Eigen::Map<const Eigen::Vector3d>(parameters[2]);
        Eigen::Vector3d d_ba   = Eigen::Map<const Eigen::Vector3d>(parameters[3]);
        Eigen::Vector3d d_bg   = Eigen::Map<const Eigen::Vector3d>(parameters[4]);
        double lambda          = *parameters[5];
        Eigen::Affine3d T_fi_w = _imu_i->getFrame()->getWorld2FrameTransform();
        Eigen::Affine3d T_fj_w = _imu_j->getFrame()->getWorld2FrameTransform();
        double dtij            = (_imu_j->getFrame()->getTimestamp() - _imu_i->getFrame()->getTimestamp()) * 1e-9;

        // Get the information matrix
        Eigen::MatrixXd cov = _imu_j->getCov();
        Eigen::Matrix<double, 9, 9> inf_sqrt =
            Eigen::LLT<Eigen::Matrix<double, 9, 9>>(cov.inverse()).matrixL().transpose();

        // Derive the residuals
        Eigen::Matrix3d dR = (_imu_j->getDeltaR() * geometry::exp_so3(_imu_j->_J_dR_bg * d_bg)).transpose() *
                             T_fi_w.rotation() * T_fj_w.rotation().transpose();
        Eigen::Vector3d r_dr = geometry::log_so3(dR);
        Eigen::Vector3d r_dv = T_fi_w.rotation() * R_w_i * ((v_j - v_i) - g * dtij) -
                               (_imu_j->getDeltaV() + _imu_j->_J_dv_bg * d_bg + _imu_j->_J_dv_ba * d_ba);
        Eigen::Vector3d r_dp =
            T_fi_w.rotation() * R_w_i *
                (std::exp(lambda) * (T_fj_w.inverse().translation() - T_fi_w.inverse().translation()) - v_i * dtij -
                 0.5 * g * dtij * dtij) -
            (_imu_j->getDeltaP() + _imu_j->_J_dp_bg * d_bg + _imu_j->_J_dp_ba * d_ba);
        Eigen::Map<Eigen::Matrix<double, 9, 1>> err(residuals);
        err.block(0, 0, 3, 1) = r_dr;
        err.block(3, 0, 3, 1) = r_dv;
        err.block(6, 0, 3, 1) = r_dp;
        err                   = inf_sqrt * err;

        if (jacobians != NULL) {

            // Jacobian wrt the gravity orientation
            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 2, Eigen::RowMajor>> J_Rwi(jacobians[0]);
                J_Rwi                   = Eigen::Matrix<double, 9, 2, Eigen::RowMajor>::Zero();
                J_Rwi.block(3, 0, 3, 2) = -T_fi_w.rotation() * R_w_i * geometry::skewMatrix((v_j - v_i) - g * dtij) *
                                          geometry::so3_rightJacobian(w_w_i).block(0, 0, 3, 2);
                J_Rwi.block(6, 0, 3, 2) = -T_fi_w.rotation() * R_w_i *
                                          geometry::skewMatrix(std::exp(lambda) * (T_fj_w.inverse().translation() -
                                                                                   T_fi_w.inverse().translation()) -
                                                               v_i * dtij - 0.5 * g * dtij * dtij) *
                                          geometry::so3_rightJacobian(w_w_i).block(0, 0, 3, 2);
                J_Rwi = inf_sqrt * J_Rwi;
            }

            // Jacobian wrt the velocity i
            if (jacobians[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J_vi(jacobians[1]);
                J_vi                   = Eigen::Matrix<double, 9, 3, Eigen::RowMajor>::Zero();
                J_vi.block(3, 0, 3, 3) = -(T_fi_w.rotation() * R_w_i);
                J_vi.block(6, 0, 3, 3) = -(T_fi_w.rotation() * R_w_i) * dtij;
                J_vi                   = inf_sqrt * J_vi;
            }

            // Jacobian wrt the velocity j
            if (jacobians[2] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J_vj(jacobians[2]);
                J_vj                   = Eigen::Matrix<double, 9, 3, Eigen::RowMajor>::Zero();
                J_vj.block(3, 0, 3, 3) = T_fi_w.rotation() * R_w_i;
                J_vj                   = inf_sqrt * J_vj;
            }

            // Jacobian wrt the bias of the accelerometer
            if (jacobians[3] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J_dba(jacobians[3]);
                J_dba                   = Eigen::Matrix<double, 9, 3, Eigen::RowMajor>::Zero();
                J_dba.block(3, 0, 3, 3) = -_imu_j->_J_dv_ba;
                J_dba.block(6, 0, 3, 3) = -_imu_j->_J_dp_ba;
                J_dba                   = inf_sqrt * J_dba;
            }

            // Jacobian wrt the bias of the gyroscope
            if (jacobians[4] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J_dbg(jacobians[4]);
                J_dbg                   = Eigen::Matrix<double, 9, 3, Eigen::RowMajor>::Zero();
                J_dbg.block(0, 0, 3, 3) = -geometry::so3_rightJacobian(r_dr).inverse() * dR.transpose() *
                                          geometry::so3_rightJacobian(_imu_j->_J_dR_bg * d_bg) * _imu_j->_J_dR_bg;
                J_dbg.block(3, 0, 3, 3) = -_imu_j->_J_dv_bg;
                J_dbg.block(6, 0, 3, 3) = -_imu_j->_J_dp_bg;
                J_dbg                   = inf_sqrt * J_dbg;
            }

            // Jacobian wrt the scale
            if (jacobians[5] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 1>> J_scale(jacobians[5]);
                J_scale                   = Eigen::Matrix<double, 9, 1>::Zero();
                J_scale.block(6, 0, 3, 1) = T_fi_w.rotation() * R_w_i *
                                            (T_fj_w.inverse().translation() - T_fi_w.inverse().translation()) *
                                            std::exp(lambda);
                J_scale = inf_sqrt * J_scale;
            }
        }

        return true;
    }

    std::shared_ptr<IMU> _imu_i;
    std::shared_ptr<IMU> _imu_j;
};

class Landmark3DPrior : public ceres::SizedCostFunction<3, 3> {
  public:
    Landmark3DPrior(const Eigen::Vector3d prior, const Eigen::Vector3d lmk, const Eigen::Matrix3d sqrt_inf)
        : _prior(prior), _lmk(lmk), _sqrt_inf(sqrt_inf) {}
    Landmark3DPrior() {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Map<Eigen::Vector3d> err(residuals);
        err = _sqrt_inf * (_lmk + Eigen::Map<const Eigen::Vector3d>(parameters[0]) - _prior);

        if (jacobians != NULL) {
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[0]);
            J = _sqrt_inf;
        }

        return true;
    }

    Eigen::Vector3d _prior, _lmk;
    Eigen::Matrix3d _sqrt_inf;
};

class LandmarkToLandmarkFactor : public ceres::SizedCostFunction<3, 3, 3> {
  public:
    LandmarkToLandmarkFactor(const Eigen::Vector3d delta,
                             const Eigen::Vector3d lmk0,
                             const Eigen::Vector3d lmk1,
                             const Eigen::Matrix3d sqrt_inf)
        : _delta(delta), _lmk0(lmk0), _lmk1(lmk1), _sqrt_inf(sqrt_inf) {}
    LandmarkToLandmarkFactor() {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Map<Eigen::Vector3d> err(residuals);
        err = _sqrt_inf * ((_lmk0 + Eigen::Map<const Eigen::Vector3d>(parameters[0])) -
                           (_lmk1 + Eigen::Map<const Eigen::Vector3d>(parameters[1])) - _delta);

        if (jacobians != NULL) {

            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J_l0(jacobians[0]);
                J_l0 = _sqrt_inf;
            }

            if (jacobians[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J_l1(jacobians[1]);
                J_l1 = -_sqrt_inf;
            }
        }

        return true;
    }
    Eigen::Vector3d _delta, _lmk0, _lmk1;
    Eigen::Matrix3d _sqrt_inf;
};

class PoseToLandmarkFactor : public ceres::SizedCostFunction<3, 6, 3> {
  public:
    PoseToLandmarkFactor(const Eigen::Vector3d delta,
                         const Eigen::Affine3d T_f_w,
                         const Eigen::Vector3d t_w_lmk,
                         const Eigen::Matrix3d sqrt_inf)
        : _delta(delta), _t_w_lmk(t_w_lmk), _T_f_w(T_f_w), _sqrt_inf(sqrt_inf) {}
    PoseToLandmarkFactor() {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Map<Eigen::Vector3d> err(residuals);
        Eigen::Affine3d T_f_w   = _T_f_w * geometry::se3_doubleVec6dtoRT(parameters[0]);
        Eigen::Vector3d t_w_lmk = _t_w_lmk + Eigen::Map<const Eigen::Vector3d>(parameters[1]);

        err = _sqrt_inf * (T_f_w * t_w_lmk - _delta);

        if (jacobians != NULL) {

            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> J_f(jacobians[0]);
                Eigen::Vector3d w_df  = Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]);
                Eigen::Matrix3d dR    = geometry::exp_so3(w_df);
                J_f.block(0, 0, 3, 3) = _sqrt_inf * _T_f_w.rotation() *
                                        (-dR * geometry::skewMatrix(t_w_lmk) * geometry::so3_rightJacobian(w_df));
                J_f.block(0, 3, 3, 3) = _sqrt_inf * _T_f_w.rotation();
            }

            if (jacobians[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J_l(jacobians[1]);
                J_l = _sqrt_inf * T_f_w.rotation();
            }
        }

        return true;
    }
    Eigen::Vector3d _delta, _t_w_lmk;
    Eigen::Affine3d _T_f_w;
    Eigen::Matrix3d _sqrt_inf;
};

class PosePriordx : public ceres::SizedCostFunction<6, 6> {
  public:
    PosePriordx(const Eigen::Affine3d T, const Eigen::Affine3d T_prior, const Eigen::MatrixXd sqrt_inf)
        : _T(T), _T_prior(T_prior), _sqrt_inf(sqrt_inf) {}
    PosePriordx() {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Map<Vector6d> err(residuals);
        Eigen::Affine3d T = _T * geometry::se3_doubleVec6dtoRT(parameters[0]);
        err               = _sqrt_inf * geometry::se3_RTtoVec6d(T * _T_prior.inverse());

        if (jacobians != NULL) {
            Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J(jacobians[0]);
            J.setIdentity();
            Eigen::Vector3d dw = Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]);

            Eigen::Vector3d w = geometry::log_so3(T.rotation() * _T_prior.rotation().transpose());
            J.block(0, 0, 3, 3) =
                geometry::so3_rightJacobian(w).inverse() * _T_prior.rotation() * geometry::so3_rightJacobian(dw);
            J.block(3, 0, 3, 3) = T.rotation() *
                                  geometry::skewMatrix(_T_prior.rotation().transpose() * _T_prior.translation()) *
                                  geometry::so3_rightJacobian(dw);
            J.block(3, 3, 3, 3) = _T.rotation();
            J                   = _sqrt_inf * J;
        }

        return true;
    }

    Eigen::Affine3d _T, _T_prior;
    Eigen::MatrixXd _sqrt_inf;
};

class IMUPriordx : public ceres::SizedCostFunction<15, 6, 3, 3, 3> {
  public:
    IMUPriordx(const Eigen::Affine3d T,
               const Eigen::Affine3d T_prior,
               const Eigen::Vector3d v,
               const Eigen::Vector3d v_prior,
               const Eigen::Vector3d ba,
               const Eigen::Vector3d ba_prior,
               const Eigen::Vector3d bg,
               const Eigen::Vector3d bg_prior,
               const Eigen::MatrixXd sqrt_inf)
        : _T(T), _T_prior(T_prior), _v(v), _v_prior(v_prior), _ba(ba), _ba_prior(ba_prior), _bg(bg),
          _bg_prior(bg_prior), _sqrt_inf(sqrt_inf) {}
    IMUPriordx() {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Map<Eigen::Matrix<double, 15, 1>> err(residuals);
        Eigen::Affine3d T      = _T * geometry::se3_doubleVec6dtoRT(parameters[0]);
        err.block(0, 0, 6, 1)  = geometry::se3_RTtoVec6d(T * _T_prior.inverse());
        err.block(6, 0, 3, 1)  = _v + Eigen::Map<const Eigen::Vector3d>(parameters[1]) - _v_prior;
        err.block(9, 0, 3, 1)  = _ba + Eigen::Map<const Eigen::Vector3d>(parameters[2]) - _ba_prior;
        err.block(12, 0, 3, 1) = _bg + Eigen::Map<const Eigen::Vector3d>(parameters[3]) - _bg_prior;
        err                    = _sqrt_inf * err;

        if (jacobians != NULL) {

            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> J(jacobians[0]);
                J.setZero();
                Eigen::Vector3d dw = Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]);

                Eigen::Vector3d w = geometry::log_so3(T.rotation() * _T_prior.rotation().transpose());
                J.block(0, 0, 3, 3) =
                    geometry::so3_rightJacobian(w).inverse() * _T_prior.rotation() * geometry::so3_rightJacobian(dw);
                J.block(3, 0, 3, 3) = T.rotation() *
                                      geometry::skewMatrix(_T_prior.rotation().transpose() * _T_prior.translation()) *
                                      geometry::so3_rightJacobian(dw);
                J.block(3, 3, 3, 3) = _T.rotation();
                J                   = _sqrt_inf * J;
            }

            if (jacobians[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J_v(jacobians[1]);
                J_v.setZero();
                J_v.block(6, 0, 3, 3) = Eigen::Matrix3d::Identity();
            }

            if (jacobians[2] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J_ba(jacobians[2]);
                J_ba.setZero();
                J_ba.block(9, 0, 3, 3) = Eigen::Matrix3d::Identity();
            }

            if (jacobians[3] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J_bg(jacobians[3]);
                J_bg.setZero();
                J_bg.block(12, 0, 3, 3) = Eigen::Matrix3d::Identity();
            }
        }

        return true;
    }

    Eigen::Affine3d _T, _T_prior;
    Eigen::Vector3d _v, _v_prior, _ba, _ba_prior, _bg, _bg_prior;
    Eigen::MatrixXd _sqrt_inf;
};

class scalePrior : public ceres::SizedCostFunction<1, 1> {
  public:
    scalePrior(const double sqrt_inf) : _sqrt_inf(sqrt_inf) {}
    scalePrior() {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        residuals[0] = _sqrt_inf * (1 - parameters[0][0]);

        if (jacobians != NULL) {
            jacobians[0][0] = -_sqrt_inf;
        }

        return true;
    }
    double _sqrt_inf;
};

//============================================================================================ linexd
// "linexd" reprojection error
inline bool linexdReprojectionError(const Eigen::Affine3d T_w_lmk,
                                    const std::shared_ptr<AModel3d> model3d,
                                    const Eigen::Vector3d &scale,
                                    const Eigen::Affine3d &T_s_w,
                                    const std::shared_ptr<ImageSensor> cam,
                                    const std::vector<Eigen::Vector2d> &p2d,
                                    double *residual) {

    Eigen::Vector3d b0, b1;
    b0 = cam->getRayCamera(p2d.at(0));
    b1 = cam->getRayCamera(p2d.at(1));

    Eigen::Affine3d T_s_lmk = T_s_w * T_w_lmk;

    // Check plane coplanarity
    Eigen::Vector3d n_obs, n_lmk;
    n_obs = b0.cross(b1);
    n_obs.normalize();
    n_lmk = (T_s_lmk.translation().normalized()).cross(geometry::Rotation2directionVector(T_s_lmk.rotation()));

    Eigen::Map<Eigen::Matrix<double, 1, 1>> res(residual);
    res = (n_obs.transpose() * n_lmk);
    return true;
}

// struct scaleFactor {

//     scaleFactor(const Eigen::Vector3d b,
//                 const Eigen::Vector3d bp,
//                 const Eigen::Affine3d T_cam0_f,
//                 const Eigen::Affine3d T_cam0_cam1)
//         : _b(b), _bp(b), _T_cam0_f(T_cam0_f), _T_cam0_cam1(T_cam0_cam1) {}

//     // Constant parameters used to process the residual
//     const Eigen::Vector3d _b, _bp;
//     const Eigen::Affine3d _T_cam0_f, _T_cam0_cam1;

//     template <typename T> bool operator()(const T *const f_pose, const T *const fp_pose, T *residual) const {

//         Eigen::Affine3d T_cam0_cam0p = _T_cam0_f * geometry::se3_doubleVec6dtoRT(f_pose) *
//                                        geometry::se3_doubleVec6dtoRT(fp_pose).inverse() * _T_cam0_f.inverse();

//         Eigen::Vector3d xA = T_cam0_cam0p.rotation() * _T_cam0_cam1.translation() - _T_cam0_cam1.translation();
//         Eigen::Matrix3d A  = _T_cam0_cam1.rotation().transpose() * geometry::skewMatrix(xA) * T_cam0_cam0p.rotation()
//         *
//                             _T_cam0_cam1.rotation();
//         Eigen::Matrix3d B = _T_cam0_cam1.rotation().transpose() * geometry::skewMatrix(T_cam0_cam0p.translation()) *
//                             T_cam0_cam0p.rotation() * _T_cam0_cam1.rotation();

//         double err = (_b.transpose() * A * _bp);
//         err /= (_b.transpose() * B * _bp);
//         err += 1;
//         residual[0] = T(err);

//         return true;
//     }

//     // Factory to hide the construction of the CostFunction object from the client code.
//     static ceres::CostFunction *Create(const Eigen::Vector3d b,
//                                        const Eigen::Vector3d bp,
//                                        const Eigen::Affine3d T_cam0_f,
//                                        const Eigen::Affine3d T_cam0_cam1) {
//         return (new ceres::NumericDiffCostFunction<scaleFactor, ceres::FORWARD, 1, 6, 6>(
//             new scaleFactor(b, bp, T_cam0_f, T_cam0_cam1)));
//         // 6: dof first argument (f_pose), 6:dof second argument (fp_pose), 1: size of error
//         // residual (residual)
//     }
// };

} // namespace isae

#endif // RESIDUALS_H
