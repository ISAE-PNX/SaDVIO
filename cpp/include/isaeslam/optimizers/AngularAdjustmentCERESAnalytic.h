#ifndef ANGULARADJUSTMENTCERESANALYTIC_H
#define ANGULARADJUSTMENTCERESANALYTIC_H

#include "isaeslam/optimizers/AOptimizer.h"

namespace isae {

class AngularAdjustmentCERESAnalytic : public AOptimizer {
  public:
    AngularAdjustmentCERESAnalytic()  = default;
    ~AngularAdjustmentCERESAnalytic() = default;

    virtual bool
    marginalize(std::shared_ptr<Frame> &frame0, std::shared_ptr<Frame> &frame1, bool enable_sparsif) override;

    virtual Eigen::MatrixXd marginalizeRelative(std::shared_ptr<Frame> &frame0, std::shared_ptr<Frame> &frame1) override;

    virtual bool landmarkOptimizationNoFov(std::shared_ptr<Frame> &f,
                                           std::shared_ptr<Frame> &fp,
                                           Eigen::Affine3d &T_cam0_cam0p,
                                           double info_scale) override;

  protected:
    uint addResidualsLocalMap(ceres::Problem &problem,
                              ceres::LossFunction *loss_function,
                              ceres::ParameterBlockOrdering *ordering,
                              std::vector<std::shared_ptr<Frame>> &frame_vector,
                              size_t fixed_frame_number,
                              std::shared_ptr<isae::LocalMap> &local_map) override;

    uint addSingleFrameResiduals(ceres::Problem &problem,
                                 ceres::LossFunction *loss_function,
                                 std::shared_ptr<Frame> &frame,
                                 typed_vec_landmarks &cloud_to_optimize) override;

    uint addLandmarkResiduals(ceres::Problem &problem,
                              ceres::LossFunction *loss_function,
                              typed_vec_landmarks &cloud_to_optimize) override;

    uint addMarginalizationResiduals(ceres::Problem &problem,
                                     ceres::LossFunction *loss_function,
                                     ceres::ParameterBlockOrdering *ordering) override;

  public:
    class AngularErrCeres_pointxd_dx : public ceres::SizedCostFunction<2, 6, 3> {
      public:
        AngularErrCeres_pointxd_dx(const Eigen::Vector3d &bearing_vector,
                                   const Eigen::Affine3d &T_s_f,
                                   const Eigen::Affine3d &T_f_w,
                                   const Eigen::Vector3d &t_w_lmk,
                                   const double sigma = 1)
            : _bearing_vector(bearing_vector), _T_s_f(T_s_f), _T_f_w(T_f_w), _t_w_lmk(t_w_lmk), _sigma(sigma) {}
        ~AngularErrCeres_pointxd_dx() {}

        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
            // Get World to sensor transform
            Eigen::Affine3d dT = geometry::se3_doubleVec6dtoRT(parameters[0]);
            Eigen::Vector3d dt = Eigen::Map<const Eigen::Vector3d>(parameters[1]);
            double weight      = 1 / (_sigma);

            // Get Landmark P3D pose
            Eigen::Vector3d t_s_lmk = _T_s_f * _T_f_w * dT * (_t_w_lmk + dt);
            double t_s_lmk_norm     = t_s_lmk.norm();
            Eigen::Vector3d b_s_lmk = t_s_lmk / t_s_lmk_norm;

            // Project on tangent plane
            Eigen::Vector3d b1;
            if ((_bearing_vector - Eigen::Vector3d(1, 0, 0)).norm() > 1e-5) {
                b1 = _bearing_vector.cross(Eigen::Vector3d(1, 0, 0));
                b1.normalize();
            } else {
                b1 = _bearing_vector.cross(Eigen::Vector3d(0, 0, 1));
                b1.normalize();
            }

            Eigen::Vector3d b2 = b1.cross(_bearing_vector);
            b2.normalize();

            Eigen::MatrixXd P  = Eigen::MatrixXd::Zero(3, 2);
            P.col(0)           = b1;
            P.col(1)           = b2;
            Eigen::MatrixXd Pt = P.transpose();

            Eigen::Map<Eigen::Vector2d> res(residuals);
            res = weight * Pt * (b_s_lmk - _bearing_vector);

            if (jacobians != NULL) {

                Eigen::MatrixXd J_e_lmk = Eigen::MatrixXd::Zero(2, 3);
                J_e_lmk += Pt * (Eigen::Matrix3d::Identity() - b_s_lmk * b_s_lmk.transpose()) *
                           _T_s_f.linear() * _T_f_w.linear() / t_s_lmk_norm;

                if (jacobians[0] != NULL) {
                    Eigen::MatrixXd J_bear_frame = Eigen::MatrixXd::Zero(3, 6);
                    J_bear_frame.block(0, 0, 3, 3) =
                        -dT.linear() * isae::geometry::skewMatrix(_t_w_lmk + dt) *
                        geometry::so3_rightJacobian(isae::geometry::se3_RTtoVec6d(dT).block<3, 1>(0, 0));
                    J_bear_frame.block(0, 3, 3, 3) = Eigen::Matrix3d::Identity();

                    Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J_frame(jacobians[0]);
                    J_frame = weight * J_e_lmk * J_bear_frame;
                }

                if (jacobians[1] != NULL) {
                    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_lmk(jacobians[1]);
                    J_lmk = weight * J_e_lmk * dT.linear();
                }
            }

            return true;
        }

      protected:
        // Constant parameters used to process the residual
        const Eigen::Vector3d _bearing_vector;
        const Eigen::Affine3d _T_s_f;
        const Eigen::Affine3d _T_f_w;
        const Eigen::Vector3d _t_w_lmk;
        const double _sigma;
    };

    class AngularErrorScaleCam0 : public ceres::SizedCostFunction<2, 1, 3> {
      public:
        AngularErrorScaleCam0(const Eigen::Vector3d &bearing_vector,
                              const Eigen::Vector3d &t_w_lmk,
                              const Eigen::Affine3d &T_cam0_w,
                              const Eigen::Affine3d &T_cam0_cam0p,
                              const Eigen::Affine3d &T_cam_cam0,
                              const double sigma)
            : _bearing_vector(bearing_vector), _t_w_lmk(t_w_lmk), _T_cam0_w(T_cam0_w), _T_cam0_cam0p(T_cam0_cam0p),
              _T_cam_cam0(T_cam_cam0), _sigma(sigma) {}
        AngularErrorScaleCam0() {}

        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {

            // Parameters
            Eigen::Vector3d t_w_lmk = _t_w_lmk + Eigen::Map<const Eigen::Vector3d>(parameters[1]);
            double weight           = 1 / (_sigma * _sigma);

            // Compute the camera pose
            Eigen::Affine3d T_cam0_cam0p_scaled = _T_cam0_cam0p;
            T_cam0_cam0p_scaled.translation() *= *parameters[0];
            Eigen::Affine3d T_cam_w = _T_cam_cam0 * T_cam0_cam0p_scaled.inverse() * _T_cam0_w;

            // Get Landmark P3D pose
            Eigen::Vector3d t_s_lmk = T_cam_w * t_w_lmk;
            Eigen::Vector3d b_s_lmk = t_s_lmk / t_s_lmk.norm();

            // Project on tangent plane
            Eigen::Vector3d b1;
            if ((_bearing_vector - Eigen::Vector3d(1, 0, 0)).norm() > 1e-5) {
                b1 = _bearing_vector.cross(Eigen::Vector3d(1, 0, 0));
                b1.normalize();
            } else {
                b1 = _bearing_vector.cross(Eigen::Vector3d(0, 0, 1));
                b1.normalize();
            }

            Eigen::Vector3d b2 = b1.cross(_bearing_vector);
            b2.normalize();

            Eigen::MatrixXd P = Eigen::MatrixXd::Zero(3, 2);
            P.col(0)          = b1;
            P.col(1)          = b2;

            Eigen::Map<Eigen::Vector2d> res(residuals);
            res = weight * P.transpose() * (b_s_lmk - _bearing_vector);

            if (jacobians != NULL) {

                Eigen::MatrixXd J_e_lmk = Eigen::MatrixXd::Zero(2, 3);
                J_e_lmk +=
                    P.transpose() * (Eigen::Matrix3d::Identity() - b_s_lmk * b_s_lmk.transpose()) / t_s_lmk.norm();

                if (jacobians[0] != NULL) {
                    Eigen::Map<Eigen::Matrix<double, 2, 1>> J(jacobians[0]);
                    Eigen::Vector3d J_bear_lambda =
                        -_T_cam_cam0.rotation() * _T_cam0_cam0p.rotation().transpose() * _T_cam0_cam0p.translation();
                    J = weight * J_e_lmk * J_bear_lambda;
                }

                if (jacobians[1] != NULL) {
                    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[1]);
                    Eigen::Matrix3d J_bear_lmk =
                        _T_cam_cam0.rotation() * _T_cam0_cam0p.rotation().transpose() * _T_cam0_w.rotation();
                    J = weight * J_e_lmk * J_bear_lmk;
                }
            }

            return true;
        }

        Eigen::Vector3d _bearing_vector, _t_w_lmk;
        Eigen::Affine3d _T_cam0_w, _T_cam0_cam0p, _T_cam_cam0;
        double _sigma;
    };

    class AngularErrorScaleDepth : public ceres::SizedCostFunction<2, 1, 1> {
      public:
        AngularErrorScaleDepth(const Eigen::Vector3d &bearing_vector,
                               const Eigen::Vector3d &bearing_vector_cam,
                               const Eigen::Affine3d &T_cam0_cam0p,
                               const Eigen::Affine3d &T_cam_cam0,
                               const double depth,
                               const double sigma)
            : _bearing_vector(bearing_vector), _bearing_vector_cam(bearing_vector_cam), _T_cam0_cam0p(T_cam0_cam0p),
              _T_cam_cam0(T_cam_cam0), _depth(depth), _sigma(sigma) {}
        AngularErrorScaleDepth() {}

        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {

            // Compute the camera pose
            Eigen::Affine3d T_cam0p_cam0_scaled = _T_cam0_cam0p.inverse();
            T_cam0p_cam0_scaled.translation() *= *parameters[0];
            Eigen::Affine3d T_camp_cam = _T_cam_cam0 * T_cam0p_cam0_scaled * _T_cam_cam0.inverse();

            // Compute the landmark bearing vector
            Eigen::Vector3d t_cam_lmk = _bearing_vector_cam * (_depth + *parameters[1]);
            Eigen::Vector3d t_s_lmk   = T_camp_cam * t_cam_lmk;
            Eigen::Vector3d b_s_lmk   = t_s_lmk / t_s_lmk.norm();

            // Compute weight
            double weight = 1 / _sigma;

            // Project on tangent plane
            Eigen::Vector3d b1;
            if ((_bearing_vector - Eigen::Vector3d(1, 0, 0)).norm() > 1e-5) {
                b1 = _bearing_vector.cross(Eigen::Vector3d(1, 0, 0));
                b1.normalize();
            } else {
                b1 = _bearing_vector.cross(Eigen::Vector3d(0, 0, 1));
                b1.normalize();
            }

            Eigen::Vector3d b2 = b1.cross(_bearing_vector);
            b2.normalize();

            Eigen::MatrixXd P = Eigen::MatrixXd::Zero(3, 2);
            P.col(0)          = b1;
            P.col(1)          = b2;

            Eigen::Map<Eigen::Vector2d> res(residuals);
            res = weight * P.transpose() * (b_s_lmk - _bearing_vector);

            if (jacobians != NULL) {

                Eigen::MatrixXd J_e_lmk = Eigen::MatrixXd::Zero(2, 3);
                J_e_lmk +=
                    P.transpose() * (Eigen::Matrix3d::Identity() - b_s_lmk * b_s_lmk.transpose()) / t_s_lmk.norm();

                if (jacobians[0] != NULL) {
                    Eigen::Map<Eigen::Matrix<double, 2, 1>> J(jacobians[0]);
                    Eigen::Vector3d J_lmk_lambda = _T_cam_cam0.rotation() * _T_cam0_cam0p.inverse().translation();
                    J                            = weight * J_e_lmk * J_lmk_lambda;
                }

                if (jacobians[1] != NULL) {
                    Eigen::Map<Eigen::Matrix<double, 2, 1>> J(jacobians[1]);
                    Eigen::Vector3d J_lmk_depth = T_camp_cam.rotation() * _bearing_vector_cam;
                    J                           = weight * J_e_lmk * J_lmk_depth;
                }
            }

            return true;
        }

        Eigen::Vector3d _bearing_vector, _bearing_vector_cam;
        Eigen::Affine3d _T_cam0_cam0p, _T_cam_cam0;
        double _depth, _sigma;
    };

    class AngularErrCeres_pointxd_depth : public ceres::SizedCostFunction<2, 6, 6, 1> {
      public:
        AngularErrCeres_pointxd_depth(const Eigen::Vector3d &bearing_vector,
                                      const Eigen::Vector3d &bearing_vector_cam,
                                      const Eigen::Affine3d &T_s_f,
                                      const Eigen::Affine3d &T_fa_w,
                                      const Eigen::Affine3d &T_f_w,
                                      const double depth,
                                      const double sigma = 1)
            : _bearing_vector(bearing_vector), _bearing_vector_cam(bearing_vector_cam), _T_s_f(T_s_f), _T_f_w(T_f_w),
              _T_fa_w(T_fa_w), _depth(depth), _sigma(sigma) {}
        ~AngularErrCeres_pointxd_depth() {}

        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
            // Get World to sensor transform
            Eigen::Affine3d dT     = geometry::se3_doubleVec6dtoRT(parameters[0]);
            Eigen::Affine3d dTa    = geometry::se3_doubleVec6dtoRT(parameters[1]);
            Eigen::Affine3d T_s_sa = _T_s_f * _T_f_w * dT * dTa.inverse() * _T_fa_w.inverse() * _T_s_f.inverse();
            double weight          = 1 / _sigma;

            // Get Landmark P3D pose
            Eigen::Vector3d t_sa_lmk = _bearing_vector_cam * (_depth + *parameters[2]);
            Eigen::Vector3d t_s_lmk  = T_s_sa * t_sa_lmk;
            Eigen::Vector3d b_s_lmk  = t_s_lmk / t_s_lmk.norm();

            // Project on tangent plane
            Eigen::Vector3d b1;
            if ((_bearing_vector - Eigen::Vector3d(1, 0, 0)).norm() > 1e-5) {
                b1 = _bearing_vector.cross(Eigen::Vector3d(1, 0, 0));
                b1.normalize();
            } else {
                b1 = _bearing_vector.cross(Eigen::Vector3d(0, 0, 1));
                b1.normalize();
            }

            Eigen::Vector3d b2 = b1.cross(_bearing_vector);
            b2.normalize();

            Eigen::MatrixXd P = Eigen::MatrixXd::Zero(3, 2);
            P.col(0)          = b1;
            P.col(1)          = b2;

            Eigen::Map<Eigen::Vector2d> res(residuals);
            res = weight * P.transpose() * (b_s_lmk - _bearing_vector);

            if (jacobians != NULL) {

                Eigen::MatrixXd J_e_lmk = Eigen::MatrixXd::Zero(2, 3);
                J_e_lmk +=
                    P.transpose() * (Eigen::Matrix3d::Identity() - b_s_lmk * b_s_lmk.transpose()) / t_s_lmk.norm();

                if (jacobians[0] != NULL) {
                    Eigen::MatrixXd J_lmk_frame = Eigen::MatrixXd::Zero(3, 6);
                    Eigen::Matrix3d R_s_w       = (_T_s_f * _T_f_w).rotation();
                    Eigen::Vector3d t_w_lmk     = dTa.inverse() * _T_fa_w.inverse() * _T_s_f.inverse() * t_sa_lmk;
                    J_lmk_frame.block(0, 0, 3, 3) =
                        -R_s_w * dT.linear() * isae::geometry::skewMatrix(t_w_lmk) *
                        geometry::so3_rightJacobian(isae::geometry::se3_RTtoVec6d(dT).block<3, 1>(0, 0));
                    J_lmk_frame.block(0, 3, 3, 3) = R_s_w;

                    Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J_frame(jacobians[0]);
                    J_frame = weight * J_e_lmk * J_lmk_frame;
                }

                if (jacobians[1] != NULL) {
                    Eigen::MatrixXd J_lmk_framea = Eigen::MatrixXd::Zero(3, 6);
                    Eigen::Matrix3d R_s_w        = (_T_s_f * _T_f_w * dT).rotation();
                    Eigen::Vector3d t_w_lmk      = _T_fa_w.inverse() * _T_s_f.inverse() * t_sa_lmk;
                    J_lmk_framea.block(0, 0, 3, 3) =
                        R_s_w * dTa.linear().transpose() * isae::geometry::skewMatrix(t_w_lmk - dTa.translation()) *
                        dTa.linear() * geometry::so3_rightJacobian(isae::geometry::se3_RTtoVec6d(dT).block<3, 1>(0, 0));
                    J_lmk_framea.block(0, 3, 3, 3) = -R_s_w * dTa.linear().transpose();

                    Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J_framea(jacobians[1]);
                    J_framea = weight * J_e_lmk * J_lmk_framea;
                }

                if (jacobians[2] != NULL) {
                    Eigen::Map<Eigen::Matrix<double, 2, 1>> J_depth(jacobians[2]);
                    J_depth = weight * J_e_lmk * T_s_sa.rotation() * _bearing_vector_cam;
                }
            }

            return true;
        }

      protected:
        // Constant parameters used to process the residual
        Eigen::Vector3d _bearing_vector, _bearing_vector_cam;
        const Eigen::Affine3d _T_s_f, _T_f_w, _T_fa_w;
        const double _depth, _sigma;
    };

    //=================================================================================================================
    // For dealing with linexd landmarks

    class AngularErrCeres_linexd_dx : public ceres::SizedCostFunction<2, 6, 6> {
      public:
        AngularErrCeres_linexd_dx(const std::vector<Eigen::Vector3d> &bearing_vectors,
                                  const Eigen::Affine3d &T_s_f,
                                  const Eigen::Affine3d &T_f_w,
                                  const Eigen::Affine3d &T_w_lmk,
                                  const double sigma = 1)
            : _bearing_vectors(bearing_vectors), _T_s_f(T_s_f), _T_f_w(T_f_w), _T_w_lmk(T_w_lmk), _sigma(sigma) {}
        ~AngularErrCeres_linexd_dx() {}

        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
            // Get World to sensor transform
            Eigen::Affine3d dT    = geometry::se3_doubleVec6dtoRT(parameters[0]);
            Eigen::Affine3d dTlmk = geometry::se3_doubleVec6dtoRT(parameters[1]);
            double weight         = 1 / (_sigma * _sigma);

            Eigen::Affine3d T_s_lmk = _T_s_f * _T_f_w * dT * _T_w_lmk * dTlmk;

            // Check plane coplanarity (normals has to be parallel)
            Eigen::Vector3d n_obs, n_ldmk, n_ldmk_normed;
            n_obs = _bearing_vectors.at(0).cross(_bearing_vectors.at(1));
            n_obs.normalize();

            // landmark center "bearing vector"
            Eigen::Vector3d b_ldmk = T_s_lmk.translation() / T_s_lmk.translation().norm();
            n_ldmk                 = b_ldmk.cross(geometry::Rotation2directionVector(T_s_lmk.rotation()));
            n_ldmk_normed          = n_ldmk / n_ldmk.norm();

            // Process residuals :
            // (1) Check 3D line, 2D line parallelism : are the normals colinears ?
            // (2) Distance btw Landmark pt and observation plane : is bearing vector orthogonal to normal to 2d line ?
            Eigen::Map<Eigen::Vector2d> res(residuals);
            res(0) = weight * (n_obs.cross(n_ldmk_normed)).norm();
            res(1) = weight * n_obs.transpose() * b_ldmk;

            // Process jacobians
            if (jacobians != NULL) {

                Eigen::MatrixXd J_e0_n_lmk = Eigen::MatrixXd::Zero(1, 3);
                J_e0_n_lmk                 = geometry::J_norm(n_obs.cross(n_ldmk_normed)) * geometry::J_AcrossX(n_obs) *
                             geometry::J_normalization(n_ldmk);

                Eigen::MatrixXd J_e1_t_lmk = Eigen::MatrixXd::Zero(1, 3);
                J_e1_t_lmk                 = n_obs.transpose() * geometry::J_normalization(T_s_lmk.translation());

                Eigen::MatrixXd J_t_lmk_dT = Eigen::MatrixXd::Zero(3, 6);
                J_t_lmk_dT.block(0, 0, 3, 3) =
                    -(_T_s_f * _T_f_w).rotation() * dT.rotation() *
                        geometry::skewMatrix(_T_w_lmk.rotation() * dTlmk.translation()) *
                        geometry::so3_rightJacobian(geometry::log_so3(dT.rotation())) +
                    -(_T_s_f * _T_f_w).rotation() * dT.rotation() * geometry::skewMatrix(_T_w_lmk.translation());
                J_t_lmk_dT.block(0, 3, 3, 3) = (_T_s_f * _T_f_w).rotation();

                Eigen::MatrixXd J_R_lmk_dT = Eigen::MatrixXd::Zero(3, 6);
                J_R_lmk_dT.block(0, 0, 3, 3) =
                    -(_T_s_f * _T_f_w).rotation() * dT.rotation() *
                    geometry::skewMatrix(_T_w_lmk.rotation() * dTlmk.rotation() * Eigen::Vector3d(1, 0, 0)) *
                    geometry::so3_rightJacobian(geometry::log_so3(dT.rotation()));

                Eigen::MatrixXd J_t_lmk_dTlmk   = Eigen::MatrixXd::Zero(3, 6);
                J_t_lmk_dTlmk.block(0, 3, 3, 3) = (_T_s_f * _T_f_w * dT * _T_w_lmk).rotation();

                Eigen::MatrixXd J_R_lmk_dTlmk   = Eigen::MatrixXd::Zero(3, 6);
                J_R_lmk_dTlmk.block(0, 0, 3, 3) = -(_T_s_f * _T_f_w * dT * _T_w_lmk).rotation() * dTlmk.rotation() *
                                                  geometry::skewMatrix(Eigen::Vector3d(1, 0, 0)) *
                                                  geometry::so3_rightJacobian(geometry::log_so3(dTlmk.rotation()));

                // wrt Frame params dT 2x6
                if (jacobians[0] != NULL) {

                    Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J_frame(jacobians[0]);
                    J_frame.block(0, 0, 1, 6) =
                        weight * J_e0_n_lmk *
                        (geometry::skewMatrix(T_s_lmk.rotation() * Eigen::Vector3d(1, 0, 0)).transpose() *
                             (geometry::J_normalization(T_s_lmk.translation()) * J_t_lmk_dT) +
                         geometry::skewMatrix(n_ldmk_normed) * J_R_lmk_dT);
                    J_frame.block(1, 0, 1, 6) = weight * J_e1_t_lmk * J_t_lmk_dT;
                }

                // wrt landmark params dTlmk 2x6
                if (jacobians[1] != NULL) {
                    Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J_lmk(jacobians[1]);
                    J_lmk.block(0, 0, 1, 6) =
                        weight * J_e0_n_lmk *
                        (geometry::skewMatrix(T_s_lmk.rotation() * Eigen::Vector3d(1, 0, 0)).transpose() *
                             (geometry::J_normalization(T_s_lmk.translation()) * J_t_lmk_dTlmk) +
                         geometry::skewMatrix(n_ldmk_normed) * J_R_lmk_dTlmk);
                    J_lmk.block(1, 0, 1, 6) = weight * J_e1_t_lmk * J_t_lmk_dTlmk;
                }
            }

            return true;
        }

      protected:
        // Constant parameters used to process the residual
        const std::vector<Eigen::Vector3d> _bearing_vectors;
        const Eigen::Affine3d _T_s_f;
        const Eigen::Affine3d _T_f_w;
        const Eigen::Affine3d _T_w_lmk;
        const double _sigma;
    };

    // class AngularErrCeres_linexd : public ceres::SizedCostFunction<2, 6, 6> {
    // public:
    //     AngularErrCeres_linexd(const std::vector<Eigen::Vector3d> &bearing_vectors,
    //                            const Eigen::Affine3d T_s_f,
    //                            const double sigma = 1)
    //             : _bearing_vectors(bearing_vectors), _T_s_f(T_s_f), _sigma(sigma) {}
    //     ~AngularErrCeres_linexd() {}

    //     virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
    //         // Get World to sensor transform
    //         Eigen::Affine3d T_f_w   = geometry::se3_doubleVec6dtoRT(parameters[0]);
    //         Eigen::Affine3d T_w_lmk = geometry::se3_doubleVec6dtoRT(parameters[1]);
    //         Eigen::Vector3d t_w_lmk = T_w_lmk.translation();
    //         double weight = 1 / (_sigma * _sigma);

    //         // Get Landmark P3D pose
    //         Eigen::Affine3d T_s_w = _T_s_f * T_f_w;
    //         Eigen::Affine3d T_s_lmk = T_s_w * T_w_lmk;

    //         // Check plane coplanarity (normals has to be parallel)
    //         Eigen::Vector3d n_obs, n_ldmk;
    //         n_obs = _bearing_vectors.at(0).cross(_bearing_vectors.at(1));
    //         n_obs.normalize();

    //         Eigen::Vector3d t_ldmk_tilde = T_s_lmk.translation().normalized();
    //         n_ldmk = t_ldmk_tilde.cross(geometry::Rotation2directionVector(T_s_lmk.rotation()));

    //         Eigen::Map<Eigen::Vector2d> res(residuals);

    //         // Check 3D line, 2D line parallelism
    //         res.x() = (n_obs.transpose()*n_ldmk);

    //         // Distance btw Landmark pt and observation plane
    //         res.y() = n_obs.transpose()*(T_w_lmk.translation()-T_s_w.translation());

    //         if (jacobians != NULL) {

    //             Eigen::MatrixXd J_e_bear = Eigen::MatrixXd::Zero(1, 3);
    //             J_e_bear = ((T_s_lmk.rotation()*Eigen::Vector3d(1,0,0)).cross(n_obs)).transpose()*
    //                         (Eigen::Matrix3d::Identity() - t_ldmk_tilde *
    //                         t_ldmk_tilde.transpose()/T_s_lmk.translation().norm());
    //             Eigen::MatrixXd J_e_tw = Eigen::MatrixXd::Zero(1, 3);
    //             J_e_tw = J_e_bear*_T_s_f.linear()*T_f_w.linear();

    //             Eigen::MatrixXd J_e_Rw = Eigen::MatrixXd::Zero(3, 3);
    //             J_e_Rw = (n_obs.cross(t_ldmk_tilde)).transpose()*Eigen::Matrix3d::Identity()*Eigen::Vector3d(1,0,0)*
    //                      _T_s_f.linear()*T_f_w.linear();

    //             // J_RW_rot =   SE3 TO CHECK
    //             Eigen::MatrixXd J_e_rot = Eigen::MatrixXd::Zero(1, 3);
    //             J_e_rot = J_e_bear*J_e_Rw*geometry::so3_rightJacobian(isae::geometry::se3_RTtoVec6d(T_w_lmk).block<3,
    //             1>(0, 0));

    //             if (jacobians[0] != NULL) {
    //                 Eigen::MatrixXd J_bear_frame = Eigen::MatrixXd::Zero(3, 6);
    //                 J_bear_frame.block(0, 0, 3, 3) =
    //                         -T_f_w.linear() * isae::geometry::skewMatrix(t_w_lmk) *
    //                         geometry::so3_rightJacobian(isae::geometry::se3_RTtoVec6d(T_f_w).block<3, 1>(0, 0));
    //                 J_bear_frame.block(0, 3, 3, 3) = Eigen::Matrix3d::Identity();

    //                 Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J_frame(jacobians[0]);
    //                 J_frame = Eigen::MatrixXd::Zero(2, 6);
    //                 J_frame.block(0,0,1,6) = weight * J_e_bear * J_bear_frame;

    //                 J_frame.block(1,0,1,3) = Eigen::MatrixXd::Zero(1, 3);
    //                 J_frame.block(1,3,1,3) = -n_obs.transpose();

    //             }

    //             if (jacobians[1] != NULL) {
    //                 Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J_lmk(jacobians[1]);
    //                 J_lmk = Eigen::MatrixXd::Zero(2, 6);
    //                 J_lmk.block(0,0,1,3) = J_e_rot;
    //                 J_lmk.block(0,3,1,3) = J_e_tw;

    //                 J_lmk.block(1,0,1,3) = Eigen::MatrixXd::Zero(1, 3);
    //                 J_lmk.block(1,3,1,3) = n_obs.transpose()*_T_s_f.rotation()*T_f_w.rotation();
    //             }

    //         }

    //         return true;
    //     }

    // protected:
    //     // Constant parameters used to process the residual
    //     const std::vector<Eigen::Vector3d> _bearing_vectors;
    //     const Eigen::Affine3d _T_s_f;
    //     const double _sigma;
    // };

    // class AngularErrCeresStaticFrame_linexd : public ceres::SizedCostFunction<2, 6> {
    // public:
    //     AngularErrCeresStaticFrame_linexd(const std::vector<Eigen::Vector3d> &bearing_vectors,
    //                                        const Eigen::Affine3d &T_f_w,
    //                                        const Eigen::Affine3d &T_s_f)
    //             : _bearing_vectors(bearing_vectors), _T_f_w(T_f_w), _T_s_f(T_s_f) {}
    //     ~AngularErrCeresStaticFrame_linexd() {}

    //     virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {

    //         // Get Landmark P3D pose
    //         Eigen::Affine3d T_w_lmk = geometry::se3_doubleVec6dtoRT(parameters[0]);

    //         // Get Landmark P3D pose
    //         Eigen::Affine3d T_s_w = _T_s_f * _T_f_w;
    //         Eigen::Affine3d T_s_lmk = T_s_w * T_w_lmk;

    //         // Check plane coplanarity
    //         Eigen::Vector3d n_obs, n_ldmk;
    //         n_obs = _bearing_vectors.at(0).cross(_bearing_vectors.at(1));
    //         n_obs.normalize();

    //         Eigen::Vector3d t_ldmk_tilde = T_s_lmk.translation().normalized();
    //         n_ldmk = t_ldmk_tilde.cross(geometry::Rotation2directionVector(T_s_lmk.linear()));

    //         Eigen::Map<Eigen::Vector2d> res(residuals);

    //         // Check 3D line, 2D line parallelism
    //         res.x() = (n_obs.transpose()*n_ldmk);

    //         // Distance btw Landmark pt and observation plane
    //         res.y() = n_obs.transpose()*(T_w_lmk.translation()-T_s_w.translation());

    //         if (jacobians != NULL) {
    //             if (jacobians[0] != NULL) {
    //                 Eigen::MatrixXd J_e_bear = Eigen::MatrixXd::Zero(1, 3);
    //                 J_e_bear = ((T_s_lmk.rotation()*Eigen::Vector3d(1,0,0)).cross(n_obs)).transpose()*
    //                         (Eigen::Matrix3d::Identity() - t_ldmk_tilde *
    //                         t_ldmk_tilde.transpose()/T_s_lmk.translation().norm());

    //                 Eigen::MatrixXd J_e_tw = Eigen::MatrixXd::Zero(1, 3);
    //                 J_e_tw = J_e_bear*_T_s_f.linear()*_T_f_w.linear();

    //                 Eigen::MatrixXd J_e_Rw = Eigen::MatrixXd::Zero(3, 3);
    //                 J_e_Rw =
    //                 (n_obs.cross(t_ldmk_tilde)).transpose()*Eigen::Matrix3d::Identity()*Eigen::Vector3d(1,0,0)*
    //                         _T_s_f.rotation()*_T_f_w.rotation();

    //                 // J_RW_rot =   SE3 TO CHECK
    //                 Eigen::MatrixXd J_Rw_wrot = Eigen::MatrixXd::Zero(3, 3);
    //                 J_Rw_wrot =
    //                 T_s_w.rotation()*geometry::so3_rightJacobian(isae::geometry::se3_RTtoVec6d(T_w_lmk).block<3,
    //                 1>(0, 0));

    //                 Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J_lmk(jacobians[0]);
    //                 J_lmk = Eigen::MatrixXd::Zero(2, 6);
    //                 J_lmk.block(0,0,1,3) = J_e_Rw*J_Rw_wrot;
    //                 J_lmk.block(0,3,1,3) = J_e_tw;

    //                 J_lmk.block(1,0,1,3) = Eigen::MatrixXd::Zero(1, 3);
    //                 J_lmk.block(1,3,1,3) = n_obs.transpose()*_T_s_f.rotation()*_T_f_w.rotation();
    //             }
    //         }

    //         return true;
    //     }

    // protected:
    //     // Constant parameters used to process the residual
    //     const std::vector<Eigen::Vector3d> _bearing_vectors;
    //     const Eigen::Affine3d _T_f_w;
    //     const Eigen::Affine3d _T_s_f;
    // };

    // class AngularErrCeresStaticLandmark_linexd : public ceres::SizedCostFunction<2, 6> {
    // public:
    //     AngularErrCeresStaticLandmark_linexd(const std::vector<Eigen::Vector3d> &bearing_vectors,
    //                                           const Eigen::Affine3d &T_w_lmk,
    //                                           const Eigen::Affine3d &T_s_f)
    //             : _bearing_vectors(bearing_vectors), _T_w_lmk(T_w_lmk), _T_s_f(T_s_f) {}
    //     ~AngularErrCeresStaticLandmark_linexd() {}

    //     virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
    //         // Get World to sensor transform
    //         Eigen::Affine3d T_f_w = geometry::se3_doubleVec6dtoRT(parameters[0]);
    //         Eigen::Affine3d T_s_w = _T_s_f * T_f_w;

    //         // Get Landmark P3D pose
    //         Eigen::Affine3d T_s_lmk = T_s_w * _T_w_lmk;

    //         // Check plane coplanarity (normals has to be parallel)
    //         Eigen::Vector3d n_obs, n_ldmk;
    //         n_obs = _bearing_vectors.at(0).cross(_bearing_vectors.at(1));
    //         n_obs.normalize();

    //         Eigen::Vector3d t_ldmk_tilde = T_s_lmk.translation().normalized();
    //         n_ldmk = t_ldmk_tilde.cross(geometry::Rotation2directionVector(T_s_lmk.rotation()));

    //         Eigen::Map<Eigen::Vector2d> res(residuals);

    //         // Check 3D line, 2D line parallelism
    //         res.x() = (n_obs.transpose()*n_ldmk);

    //         // Distance btw Landmark pt and observation plane
    //         res.y() = n_obs.transpose()*(_T_w_lmk.translation()-T_s_w.translation());

    //         if (jacobians != NULL) {
    //             Eigen::MatrixXd J_e_bear = Eigen::MatrixXd::Zero(1, 3);
    //             J_e_bear = ((T_s_lmk.rotation()*Eigen::Vector3d(1,0,0)).cross(n_obs)).transpose()*
    //                         (Eigen::Matrix3d::Identity() - t_ldmk_tilde *
    //                         t_ldmk_tilde.transpose()/T_s_lmk.translation().norm());

    //             Eigen::MatrixXd J_e_frame = Eigen::MatrixXd::Zero(1, 6);

    //             Eigen::MatrixXd J_bear_frame = Eigen::MatrixXd::Zero(3, 6);
    //             J_bear_frame.block(0, 0, 3, 3) =
    //                     -T_f_w.linear() * isae::geometry::skewMatrix(_T_w_lmk.translation()) *
    //                     geometry::so3_rightJacobian(isae::geometry::se3_RTtoVec6d(T_f_w).block<3, 1>(0, 0));
    //             J_bear_frame.block(0, 3, 3, 3) = Eigen::Matrix3d::Identity();

    //             Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J_frame(jacobians[0]);
    //             J_frame = Eigen::MatrixXd::Zero(2, 6);
    //             J_frame.block(0,0,1,6) = J_e_bear * J_bear_frame;

    //             J_frame.block(1,0,1,3) = Eigen::MatrixXd::Zero(1, 3);
    //             J_frame.block(1,3,1,3) = -n_obs.transpose();
    //         }

    //         return true;
    //     }

    // protected:
    //     // Constant parameters used to process the residual
    //     const std::vector<Eigen::Vector3d> _bearing_vectors;
    //     const Eigen::Affine3d _T_w_lmk;
    //     const Eigen::Affine3d _T_s_f;
    // };
};

} // namespace isae

#endif // ANGULARADJUSTMENTCERESANALYTIC_H
