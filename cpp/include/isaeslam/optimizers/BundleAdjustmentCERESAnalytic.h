#ifndef BUNDLEADJUSTMENTCERESANALYTIC_H
#define BUNDLEADJUSTMENTCERESANALYTIC_H

#include "isaeslam/optimizers/AOptimizer.h"

namespace isae {

/*!
 * @brief An optimizer that uses Angular Cost function on the unit sphere for Bundle Adjustment and analytical jacobian
 * derivation
 */
class BundleAdjustmentCERESAnalytic : public AOptimizer {
  public:
    BundleAdjustmentCERESAnalytic()  = default;
    ~BundleAdjustmentCERESAnalytic() = default;

    virtual bool
    marginalize(std::shared_ptr<Frame> &frame0, std::shared_ptr<Frame> &frame1, bool enable_sparsif) override;

    bool localMapVIOptimizationTd(std::shared_ptr<isae::LocalMap> &local_map,
                                  double &td,
                                  const size_t fixed_frame_number = 0) override;

    virtual Eigen::MatrixXd marginalizeRelative(std::shared_ptr<Frame> &frame0,
                                                std::shared_ptr<Frame> &frame1) override;

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
};

/*!
 * @brief Reprojection cost function for a point landmark in the sensor frame, with time delta.
 * Parameters are delta update of the frame pose, the landmark position and the time delta.
 *
 * The cost function uses 2D feature velocity to estimate the time delay, as proposed in "Online temporal calibration
 * for monocular visual-inertial systems" by Qin et al Source: https://arxiv.org/abs/1808.00692
 */
class ReprojectionErrCeres_pointxd_dx_td_velo : public ceres::SizedCostFunction<2, 6, 3, 1> {
  public:
    ReprojectionErrCeres_pointxd_dx_td_velo(const Eigen::Vector2d &p2d,
                                       const Eigen::Vector2d &velocity,
                                       const std::shared_ptr<ImageSensor> &cam,
                                       const Eigen::Affine3d &T_w_lmk,
                                       const double sigma = 1.0)
        : p2d_(p2d), cam_(cam), _T_w_lmk(T_w_lmk) {
        info_sqrt_ = (1 / sigma) * Eigen::Matrix2d::Identity();
    }
    ~ReprojectionErrCeres_pointxd_dx_td_velo() {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        // Get World to sensor transform
        Eigen::Affine3d T_f_w =
            cam_->getFrame()->getWorld2FrameTransform() * geometry::se3_doubleVec6dtoRT(parameters[0]);

        double td = *parameters[2];

        // Get Landmark P3D pose
        Eigen::Affine3d T_w_lmk = _T_w_lmk * geometry::se3_doubleVec3dtoRT(parameters[1]);
        Eigen::Vector2d projection;
        Eigen::Map<Eigen::Vector2d> res(residuals);
        res = Eigen::Vector2d::Zero();

        if (jacobians != NULL) {
            if (cam_->project(T_w_lmk, T_f_w, info_sqrt_, projection, jacobians[0], jacobians[1])) {
                res = info_sqrt_ * (projection - (p2d_ + velocity * td));
            }

            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J_proj_f(jacobians[0]);
                Eigen::MatrixXd J_lf_dlf = Eigen::MatrixXd::Zero(6, 6);
                J_lf_dlf.block(0, 0, 3, 3) =
                    geometry::so3_rightJacobian(geometry::log_so3(T_f_w.rotation())).inverse() *
                    geometry::so3_rightJacobian(Eigen::Vector3d(parameters[0][0], parameters[0][1],
                    parameters[0][2]));
                J_lf_dlf.block(3, 3, 3, 3) = cam_->getFrame()->getWorld2FrameTransform().rotation();
                J_proj_f                   = J_proj_f * J_lf_dlf;
            }

            if (jacobians[2] != NULL) {
                Eigen::Map<Eigen::Vector2d> J_td(jacobians[2]);
                J_td = -info_sqrt_ * velocity;
            }

        } else {
            if (cam_->project(T_w_lmk, T_f_w, info_sqrt_, projection, NULL, NULL)) {
                res = info_sqrt_ * (projection - (p2d_ + velocity * td));
            }
        }

        return true;
    }

  protected:
    const Eigen::Vector2d p2d_;              //!< 2D point observation of the landmark
    const Eigen::Vector2d velocity;          //!< 2D velocity of the point observation
    const std::shared_ptr<ImageSensor> cam_; //!< Camera sensor used to project the landmark
    const Eigen::Affine3d _T_w_lmk;          //!< Transform of the landmark in the world frame
    Eigen::Matrix2d info_sqrt_;              //!< Square root of the information matrix for the residuals
};

/*!
 * @brief Reprojection cost function for a point landmark in the sensor frame, with time delta.
 * Parameters are delta update of the frame pose, the landmark position and the time delta.
 *
 * The cost function uses 2D feature velocity to estimate the time delay, as proposed in "Online temporal calibration
 * for monocular visual-inertial systems" by Qin et al Source: https://arxiv.org/abs/1808.00692
 */
class ReprojectionErrCeres_pointxd_dx_td : public ceres::SizedCostFunction<2, 6, 3, 1> {
  public:
    ReprojectionErrCeres_pointxd_dx_td(const Eigen::Vector2d &p2d,
                                       const std::shared_ptr<ImageSensor> &cam,
                                       const std::shared_ptr<IMU> &imu,
                                       const Eigen::Affine3d &T_w_lmk,
                                       const double sigma = 1.0)
        : p2d_(p2d), cam_(cam), _T_w_lmk(T_w_lmk), imu_(imu) {
        info_sqrt_ = (1 / sigma) * Eigen::Matrix2d::Identity();
    }
    ~ReprojectionErrCeres_pointxd_dx_td() {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        // Update due to td
        double td                = *parameters[2];
        Eigen::Vector3d velocity = imu_->getVelocity();
        Eigen::Vector3d gyr      = imu_->getGyr();
        Eigen::Matrix3d dR_td    = geometry::exp_so3(gyr * td).transpose();
        Eigen::Vector3d dt_td    = -dR_td * velocity * td;
        Eigen::Affine3d dT_td    = Eigen::Affine3d::Identity();
        dT_td.linear()           = dR_td;
        dT_td.translation()      = dt_td;

        // Get World to sensor transform
        Eigen::Affine3d T_f_w_unfixed =
            cam_->getFrame()->getWorld2FrameTransform() * geometry::se3_doubleVec6dtoRT(parameters[0]);
        Eigen::Affine3d T_f_w = dT_td * T_f_w_unfixed;

        // Get Landmark P3D pose
        Eigen::Affine3d T_w_lmk = _T_w_lmk * geometry::se3_doubleVec3dtoRT(parameters[1]);
        Eigen::Vector2d projection;
        Eigen::Map<Eigen::Vector2d> res(residuals);
        res = Eigen::Vector2d::Zero();

        if (jacobians != NULL) {
            double j0[12];
            if (cam_->project(T_w_lmk, T_f_w, info_sqrt_, projection, j0, jacobians[1])) {
                res = info_sqrt_ * (projection - p2d_);
            }
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J_proj_f(j0);

            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J_proj_dlf(jacobians[0]);
                Eigen::MatrixXd J_f_dlf = Eigen::MatrixXd::Zero(6, 6);
                J_f_dlf.block(0, 0, 3, 3) =
                    geometry::so3_rightJacobian(geometry::log_so3(T_f_w.rotation())).inverse() *
                    geometry::so3_rightJacobian(Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]));
                J_f_dlf.block(3, 3, 3, 3) = dR_td * cam_->getFrame()->getWorld2FrameTransform().rotation();
                J_proj_dlf                = J_proj_f * J_f_dlf;
            }

            if (jacobians[2] != NULL) {
                Eigen::Map<Eigen::Vector2d> J_proj_td(jacobians[2]);
                Eigen::MatrixXd J_f_td = Eigen::MatrixXd::Zero(6, 1);
                Eigen::Matrix3d Jr      = geometry::so3_rightJacobian(gyr * td);
                J_f_td.block(0, 0, 3, 1) =
                    -T_f_w_unfixed.rotation().transpose() * geometry::exp_so3(gyr * td) * Jr * gyr;
                J_f_td.block(3, 0, 3, 1) = geometry::exp_so3(gyr * td).transpose() *
                                               geometry::skewMatrix(T_f_w_unfixed.translation()) *
                                               geometry::exp_so3(gyr * td) * Jr * gyr -
                                           dR_td * geometry::skewMatrix(velocity) * dR_td.transpose() * Jr * gyr * td -
                                           dR_td.transpose() * velocity;
                J_proj_td = J_proj_f * J_f_td;
            }

        } else {
            if (cam_->project(T_w_lmk, T_f_w, info_sqrt_, projection, NULL, NULL)) {
                res = info_sqrt_ * (projection - p2d_);
            }
        }

        return true;
    }

  protected:
    const Eigen::Vector2d p2d_;              //!< 2D point observation of the landmark
    const std::shared_ptr<ImageSensor> cam_; //!< Camera sensor used to project the landmark
    const std::shared_ptr<IMU> imu_;
    const Eigen::Affine3d _T_w_lmk; //!< Transform of the landmark in the world frame
    Eigen::Matrix2d info_sqrt_;     //!< Square root of the information matrix for the residuals
};

/*!
 * @brief Reprojection errro cost function for a punctual landmark. The parameters are the delta update of the frame and
 * the delta update of the 3D position of the landmark.
 *
 * BEWARE: This funciton requires that the jacobians are derived in the projection function of the camera model.
 */
class ReprojectionErrCeres_pointxd_dx : public ceres::SizedCostFunction<2, 6, 3> {
  public:
    ReprojectionErrCeres_pointxd_dx(const Eigen::Vector2d &p2d,
                                    const std::shared_ptr<ImageSensor> &cam,
                                    const Eigen::Affine3d &T_w_lmk,
                                    const double sigma = 1.0)
        : p2d_(p2d), cam_(cam), _T_w_lmk(T_w_lmk) {
        info_sqrt_ = (1 / sigma) * Eigen::Matrix2d::Identity();
    }
    ~ReprojectionErrCeres_pointxd_dx() {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        // Get World to sensor transform
        Eigen::Affine3d T_f_w =
            cam_->getFrame()->getWorld2FrameTransform() * geometry::se3_doubleVec6dtoRT(parameters[0]);

        // Get Landmark P3D pose
        Eigen::Affine3d T_w_lmk = _T_w_lmk * geometry::se3_doubleVec3dtoRT(parameters[1]);
        Eigen::Vector2d projection;
        Eigen::Map<Eigen::Vector2d> res(residuals);

        if (jacobians != NULL) {
            if (!cam_->project(T_w_lmk, T_f_w, info_sqrt_, projection, jacobians[0], jacobians[1])) {
                res = Eigen::Vector2d::Zero();

            } else {
                res = info_sqrt_ * (projection - p2d_);
            }

            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J_proj_f(jacobians[0]);
                Eigen::MatrixXd J_lf_dlf = Eigen::MatrixXd::Zero(6, 6);
                J_lf_dlf.block(0, 0, 3, 3) =
                    geometry::so3_rightJacobian(geometry::log_so3(T_f_w.rotation())).inverse() *
                    geometry::so3_rightJacobian(Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]));
                J_lf_dlf.block(3, 3, 3, 3) = cam_->getFrame()->getWorld2FrameTransform().rotation();
                J_proj_f                   = J_proj_f * J_lf_dlf;
            }
        } else {
            if (!cam_->project(T_w_lmk, T_f_w, info_sqrt_, projection, NULL, NULL)) {
                res = Eigen::Vector2d::Zero();

            } else {
                res = info_sqrt_ * (projection - p2d_);
            }
        }

        return true;
    }

  protected:
    const Eigen::Vector2d p2d_;              //!< 2D point observation of the landmark
    const std::shared_ptr<ImageSensor> cam_; //!< Camera sensor used to project the landmark
    const Eigen::Affine3d _T_w_lmk;          //!< Transform of the landmark in the world frame
    Eigen::Matrix2d info_sqrt_;              //!< Square root of the information matrix for the residuals
};

/*!
 * @brief Reprojection error cost function for a line landmark. The parameters are the delta update of the frame and the
 * delta update of the line 3D parametrization.
 *
 * BEWARE: This funciton requires that the jacobians are derived in the projection function of the camera model.
 */
class ReprojectionErrCeres_linexd_dx : public ceres::SizedCostFunction<4, 6, 6> {
  public:
    ReprojectionErrCeres_linexd_dx(const std::vector<Eigen::Vector2d> &p2ds,
                                   const std::shared_ptr<ImageSensor> &cam,
                                   const Eigen::Affine3d &T_w_lmk,
                                   const std::shared_ptr<AModel3d> &model3d,
                                   const double sigma = 1.0)
        : p2ds_(p2ds), cam_(cam), T_w_lmk_(T_w_lmk), model3d_(model3d) {
        info_sqrt_ = (1 / sigma) * Eigen::Matrix2d::Identity();
    }
    ~ReprojectionErrCeres_linexd_dx() {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        // Get World to sensor transform
        Eigen::Affine3d T_f_w =
            cam_->getFrame()->getWorld2FrameTransform() * geometry::se3_doubleVec6dtoRT(parameters[0]);

        // Get Landmark P3D pose
        Eigen::Affine3d T_w_lmk = T_w_lmk_ * geometry::se3_doubleVec3dtoRT(parameters[1]);
        std::vector<Eigen::Vector2d> projections(2);
        Eigen::Map<Eigen::Vector4d> res(residuals);

        // get the transform for each point of the landmark model
        std::vector<Eigen::Affine3d> T_w_lmk_pts;
        for (auto pt : model3d_->getModel()) {
            Eigen::Affine3d Tpt = Eigen::Affine3d::Identity();
            Tpt.translation()   = pt;
            T_w_lmk_pts.push_back(T_w_lmk * Tpt);
        }

        if (jacobians != NULL) {
            // For each reprojection (N points in the model)
            for (uint i = 0; i < T_w_lmk_pts.size(); ++i) {
                double j0[12];
                double j1[6];
                if (!cam_->project(T_w_lmk_pts.at(i), T_f_w, info_sqrt_, projections.at(i), j0, j1)) {
                    res.block(2 * i, 0, 2, 1) = Eigen::Vector2d::Zero();
                } else {
                    res.block(2 * i, 0, 2, 1) = info_sqrt_ * (projections.at(i) - p2ds_.at(i));
                }

                if (jacobians[0] != NULL) {
                    Eigen::Map<Eigen::Matrix<double, 4, 6, Eigen::RowMajor>> J_frame(jacobians[0]);
                    Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jac0(j0);

                    Eigen::MatrixXd J_lf_dlf = Eigen::MatrixXd::Zero(6, 6);
                    J_lf_dlf.block(0, 0, 3, 3) =
                        geometry::so3_rightJacobian(geometry::log_so3(T_f_w.rotation())).inverse() *
                        geometry::so3_rightJacobian(
                            Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]));
                    J_lf_dlf.block(3, 3, 3, 3)    = cam_->getFrame()->getWorld2FrameTransform().rotation();
                    J_frame.block(2 * i, 0, 2, 6) = jac0 * J_lf_dlf;
                }

                if (jacobians[1] != NULL) {
                    Eigen::Map<Eigen::Matrix<double, 4, 6, Eigen::RowMajor>> J_lmk(jacobians[1]);
                    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jac1(j1);

                    Eigen::MatrixXd J_P3_Tldmk(3, 6);
                    J_P3_Tldmk.block(0, 0, 3, 3) =
                        -T_w_lmk.linear() * isae::geometry::skewMatrix(model3d_->getModel().at(i));
                    J_P3_Tldmk.block(0, 3, 3, 3) = Eigen::Matrix3d::Identity();

                    J_lmk.block(2 * i, 0, 2, 6) = jac1 * J_P3_Tldmk;
                }
            }

        } else {

            for (uint i = 0; i < T_w_lmk_pts.size(); ++i) {
                if (!cam_->project(T_w_lmk, T_f_w, info_sqrt_, projections.at(i), NULL, NULL)) {
                    res.block(2 * i, 0, 2, 1) = Eigen::Vector2d::Zero();

                } else {
                    res.block(2 * i, 0, 2, 1) = info_sqrt_ * (projections.at(1) - p2ds_.at(1));
                }
            }
        }

        return true;
    }

  protected:
    const std::vector<Eigen::Vector2d> p2ds_; //!< 2D point observations of the line landmark
    const std::shared_ptr<ImageSensor> cam_;  //!< Camera sensor used to project the landmark
    const Eigen::Affine3d T_w_lmk_;           //!< Transform of the landmark in the world frame
    const std::shared_ptr<AModel3d> model3d_; //!< 3D model of the line landmark
    Eigen::Matrix2d info_sqrt_;               //!< Square root of the information matrix for the residuals
};

} // namespace isae

#endif // BUNDLEADJUSTMENTCERESANALYTIC_H
