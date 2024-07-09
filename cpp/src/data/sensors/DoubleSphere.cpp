#include "isaeslam/data/sensors/DoubleSphere.h"

namespace isae {

Eigen::Vector3d DoubleSphere::getRay(Eigen::Vector2d f) {
    Eigen::Vector3d ray_world;
    Eigen::Vector3d ray_cam = this->getRayCamera(f);

    ray_world = this->getSensor2WorldTransform().rotation() * ray_cam;
    ray_world.normalize();
    return ray_world;
}

Eigen::Vector3d DoubleSphere::getRayCamera(Eigen::Vector2d f) {
    Eigen::Vector3d ray_cam;

    // double sphere model as defined by Usenko et. al.
    double mx = (f(0) - _calibration(0, 2)) / _calibration(0, 0);
    double my = (f(1) - _calibration(1, 2)) / _calibration(1, 1);
    double r2 = mx * mx + my * my;

    double mz  = (1 - _alpha * _alpha * r2) / (_alpha * std::sqrt(1 - (2 * _alpha - 1) * r2) + 1 - _alpha);
    double mz2 = mz * mz;
    double k   = (mz * _xi + std::sqrt(mz2 + (1 - _xi * _xi) * r2)) / (mz2 + r2);

    ray_cam[0] = k * mx;
    ray_cam[1] = k * my;
    ray_cam[2] = k * mz - _xi;
    
    return ray_cam;
}

bool DoubleSphere::project(const Eigen::Affine3d &T_w_lmk,
                           const std::shared_ptr<AModel3d> ldmk_model,
                           const Eigen::Vector3d &scale,
                           std::vector<Eigen::Vector2d> &p2d_vector) {

    for (const auto &p3d_model : ldmk_model->getModel()) {
        // conversion to the camera coordinate system
        Eigen::Vector3d t_w_lmk   = T_w_lmk * p3d_model.cwiseProduct(scale);
        Eigen::Vector3d t_cam_lmk = this->getWorld2SensorTransform() * t_w_lmk;

        if (t_cam_lmk[2] < 0.1) // point behind the camera
            return false;

        // double sphere model as defined by Usenko et. al.
        double d1 =
            std::sqrt(t_cam_lmk.x() * t_cam_lmk.x() + t_cam_lmk.y() * t_cam_lmk.y() + t_cam_lmk.z() * t_cam_lmk.z());
        double d2 = std::sqrt(t_cam_lmk.x() * t_cam_lmk.x() + t_cam_lmk.y() * t_cam_lmk.y() +
                              (_xi * d1 + t_cam_lmk.z()) * (_xi * d1 + t_cam_lmk.z()));
        Eigen::Vector3d sph_pix(t_cam_lmk.x() / (_alpha * d2 + (1 - _alpha) * (_xi * d1 + t_cam_lmk.z())),
                                t_cam_lmk.y() / (_alpha * d2 + (1 - _alpha) * (_xi * d1 + t_cam_lmk.z())),
                                1);
        Eigen::Vector2d p2d = (this->getCalibration() * sph_pix).block<2, 1>(0, 0);

        p2d_vector.push_back(p2d);

        // Check validity
        double w1;
        if (_alpha <= 0.5)
            w1 = _alpha / (1 - _alpha);
        else
            w1 = (1 - _alpha) / _alpha;

        double w2 = (w1 + _xi) / std::sqrt(2 * w1 * _xi + _xi * _xi + 1);
        if (t_cam_lmk.z() <= -w2 * d1)
            return false;

        if (p2d[0] < 0 || p2d[1] < 0 || p2d[0] > _raw_data.cols || p2d[1] > _raw_data.rows) // out of image
            return false;
        
        if (!std::isfinite(p2d[0]) || !std::isfinite(p2d[1])) // Nan returned
            return false;
    }

    return true;
}

bool DoubleSphere::project(const Eigen::Affine3d &T_w_lmk,
                           const std::shared_ptr<AModel3d> ldmk_model,
                           const Eigen::Vector3d &scale,
                           const Eigen::Affine3d &T_f_w,
                           std::vector<Eigen::Vector2d> &p2d_vector) {

    for (const auto &p3d_model : ldmk_model->getModel()) {
        // conversion to the camera coordinate system
        Eigen::Vector3d t_w_lmk   = T_w_lmk * p3d_model.cwiseProduct(scale);
        Eigen::Vector3d t_cam_lmk = this->getFrame2SensorTransform() * T_f_w * t_w_lmk;

        if (t_cam_lmk[2] < 0.1) // point behind the camera
            return false;

        // double sphere model as defined by Usenko et. al.
        double d1 =
            std::sqrt(t_cam_lmk.x() * t_cam_lmk.x() + t_cam_lmk.y() * t_cam_lmk.y() + t_cam_lmk.z() * t_cam_lmk.z());
        double d2 = std::sqrt(t_cam_lmk.x() * t_cam_lmk.x() + t_cam_lmk.y() * t_cam_lmk.y() +
                              (_xi * d1 + t_cam_lmk.z()) * (_xi * d1 + t_cam_lmk.z()));
        Eigen::Vector3d sph_pix(t_cam_lmk.x() / (_alpha * d2 + (1 - _alpha) * (_xi * d1 + t_cam_lmk.z())),
                                t_cam_lmk.y() / (_alpha * d2 + (1 - _alpha) * (_xi * d1 + t_cam_lmk.z())),
                                1);
        Eigen::Vector2d p2d = (this->getCalibration() * sph_pix).block<2, 1>(0, 0);

        // Check validity
        double w1;
        if (_alpha <= 0.5)
            w1 = _alpha / (1 - _alpha);
        else
            w1 = (1 - _alpha) / _alpha;

        double w2 = (w1 + _xi) / std::sqrt(2 * w1 * _xi + _xi * _xi + 1);
        if (t_cam_lmk.z() <= -w2 * d1)
            return false;

        if (p2d[0] < 0 || p2d[1] < 0 || p2d[0] > _raw_data.cols || p2d[1] > _raw_data.rows) // out of image
            return false;
        
        if (!std::isfinite(p2d[0]) || !std::isfinite(p2d[1])) // Nan returned
            return false;

        p2d_vector.push_back(p2d);
    }

    return true;
}
bool DoubleSphere::project(const Eigen::Affine3d &T_w_lmk,
                           const Eigen::Affine3d &T_f_w,
                           const Eigen::Matrix2d sqrt_info,
                           Eigen::Vector2d &p2d,
                           double *J_proj_frame,
                           double *J_proj_lmk) {
    return false;
}

} // namespace isae