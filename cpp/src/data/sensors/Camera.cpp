#include "isaeslam/data/sensors/Camera.h"
#include "utilities/geometry.h"

namespace isae {

Eigen::Vector3d Camera::getRay(Eigen::Vector2d f) {
    Eigen::Vector3d ray_world;
    Eigen::Vector3d ray_cam = this->getRayCamera(f);

    ray_world = this->getSensor2WorldTransform().rotation() * ray_cam;
    ray_world.normalize();
    return ray_world;
}

Eigen::Vector3d Camera::getRayCamera(Eigen::Vector2d f) {
    Eigen::Vector3d ray_cam;

    ray_cam.segment<2>(0) = (f - _calibration.block<2, 1>(0, 2));
    ray_cam[0] /= _calibration(0, 0);
    ray_cam[1] /= _calibration(1, 1);
    ray_cam[2] = 1;

    ray_cam.normalize();
    return ray_cam;
}

bool Camera::project(const Eigen::Affine3d &T_w_lmk,
                     const std::shared_ptr<AModel3d> ldmk_model,
                     const Eigen::Vector3d &scale,
                     std::vector<Eigen::Vector2d> &p2d_vector) {

    for (const auto &p3d_model : ldmk_model->getModel()) {
        // conversion to the camera coordinate system
        Eigen::Vector3d t_w_lmk   = T_w_lmk * p3d_model.cwiseProduct(scale);
        Eigen::Vector3d t_cam_lmk = this->getWorld2SensorTransform() * t_w_lmk;

        // to image homogeneous coordinates
        Eigen::Vector3d pt = this->getCalibration() * t_cam_lmk;
        pt /= pt(2, 0);
        Eigen::Vector2d p2d = pt.block<2, 1>(0, 0);
        p2d_vector.push_back(p2d);

        if (t_cam_lmk[2] < 0.1) // point behind the camera
            return false;

        if (p2d[0] < 0 || p2d[1] < 0 || p2d[0] > _raw_data.cols || p2d[1] > _raw_data.rows) // out of image
            return false;
        
        if (!std::isfinite(p2d[0]) || !std::isfinite(p2d[1])) // Nan returned
            return false;
    }
    return true;
}

bool Camera::project(const Eigen::Affine3d &T_w_lmk,
                     const std::shared_ptr<AModel3d> ldmk_model,
                     const Eigen::Vector3d &scale,
                     const Eigen::Affine3d &T_f_w,
                     std::vector<Eigen::Vector2d> &p2d_vector) {

    for (auto &p3d_model : ldmk_model->getModel()) {
        // conversion to the camera coordinate system
        Eigen::Vector3d t_w_lmk   = T_w_lmk * p3d_model.cwiseProduct(scale);
        Eigen::Vector3d t_cam_lmk = this->getFrame2SensorTransform() * T_f_w * t_w_lmk;

        if (t_cam_lmk[2] < 0.1) // point behind the camera
            return false;

        // to image homogeneous coordinates
        Eigen::Vector3d pt = this->getCalibration() * t_cam_lmk;
        pt /= pt(2, 0);
        Eigen::Vector2d p2d = pt.block<2, 1>(0, 0);
        p2d_vector.push_back(p2d);

        if (p2d[0] < 0 || p2d[1] < 0 || p2d[0] > _raw_data.cols || p2d[1] > _raw_data.rows) // out of image
            return false;
        
        if (!std::isfinite(p2d[0]) || !std::isfinite(p2d[1])) // Nan returned
            return false;
    }
    return true;
}

bool Camera::project(const Eigen::Affine3d &T_w_lmk,
                     const Eigen::Affine3d &T_f_w,
                     const Eigen::Matrix2d sqrt_info,
                     Eigen::Vector2d &p2d,
                     double *J_proj_frame,
                     double *J_proj_lmk) {

    // conversion to the camera coordinate system
    Eigen::Vector3d t_cam_lmk =
        this->getFrame2SensorTransform() * T_f_w * T_w_lmk.translation();

    // to image homogeneous coordinates
    Eigen::Vector3d pt = this->getCalibration() * t_cam_lmk;
    Eigen::MatrixXd J_h(2, 3);
    J_h << 1 / pt(2, 0), 0.0, -pt(0, 0) / (pt(2, 0) * pt(2, 0)), //
        0.0, 1 / pt(2, 0), -pt(1, 0) / (pt(2, 0) * pt(2, 0));    //

    pt /= pt(2, 0);
    p2d = pt.block<2, 1>(0, 0);

    if (J_proj_frame != NULL) {
        // Jacobian wrt to frame pose vector
        Eigen::MatrixXd J_int = Eigen::MatrixXd::Zero(3, 6);

        J_int.block(0, 0, 3, 3) = -T_f_w.linear() * isae::geometry::skewMatrix(T_w_lmk.translation()) *
                                  geometry::so3_rightJacobian(isae::geometry::se3_RTtoVec6d(T_f_w).block<3, 1>(0, 0));
        J_int.block(0, 3, 3, 3) = Eigen::Matrix3d::Identity();

        Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J_frame(J_proj_frame);
        J_frame = J_h * this->getCalibration() * this->getFrame2SensorTransform().linear() *
                  J_int;
        J_frame = sqrt_info * J_frame;
    }

    if (J_proj_lmk != NULL) {
        // Jacobian wrt lmk p3d vector
        Eigen::Matrix3d J_aug = Eigen::Matrix3d::Identity();
        J_aug                 = this->getFrame2SensorTransform().linear() * T_f_w.linear() * J_aug;

        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_lmk(J_proj_lmk);
        J_lmk = J_h * this->getCalibration() * J_aug;
        J_lmk = sqrt_info * J_lmk;
    }

    if (t_cam_lmk[2] < 0.1) // point behind the camera
        return false;

    if (p2d[0] < 0 || p2d[1] < 0 || p2d[0] > 2 * this->getCalibration()(0, 2) ||
        p2d[1] > 2 * this->getCalibration()(1, 2)) // out of the image (suppose principal point at the center)
        return false;
    
    if (!std::isfinite(p2d[0]) || !std::isfinite(p2d[1])) // Nan returned
            return false;

    return true;
}

} // namespace isae