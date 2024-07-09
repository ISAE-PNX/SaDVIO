#include "isaeslam/data/sensors/Fisheye.h"
#include <iostream>

namespace isae {

Eigen::Vector3d Fisheye::getRay(Eigen::Vector2d f) {
    Eigen::Vector3d ray_world;
    Eigen::Vector3d ray_cam;

    // compute radius
    double xd = (f(0) - _calibration(0, 2)) / _rmax;
    double yd = (f(1) - _calibration(1, 2)) / _rmax;
    double rd = std::sqrt(xd * xd + yd * yd);
    double theta;

    // convert it to pinhole pixel
    if (_model == "equidistant") {
        theta = rd / _calibration(0, 0);
    } else if (_model == "equisolid") {
        theta = 2 * std::asin(rd / (2 * _calibration(0, 0)));
    } else if (_model == "stereographic") {
        theta = 2 * std::atan2(rd, 2 * _calibration(0, 0));
    } else {
        std::cout << "Unknown fisheye lens model" << std::endl;
        return ray_world;
    }

    ray_cam[0] = xd;
    ray_cam[1] = yd;
    ray_cam[2] = rd / std::tan(theta);
    ray_world  = this->getSensor2WorldTransform().rotation() * ray_cam;
    ray_world.normalize();
    return ray_world;
}

Eigen::Vector3d Fisheye::getRayCamera(Eigen::Vector2d f) {
    Eigen::Vector3d ray_cam;

    // compute radius
    double xd = (f(0) - _calibration(0, 2)) / _rmax;
    double yd = (f(1) - _calibration(1, 2)) / _rmax;
    double rd = std::sqrt(xd * xd + yd * yd);
    double theta;

    // convert it to pinhole pixel
    if (_model == "equidistant") {
        theta = rd / _calibration(0, 0);
    } else if (_model == "equisolid") {
        theta = 2 * std::asin(rd / (2 * _calibration(0, 0)));
    } else if (_model == "stereographic") {
        theta = 2 * std::atan2(rd, 2 * _calibration(0, 0));
    } else {
        std::cout << "Unknown fisheye lens model" << std::endl;
        return ray_cam;
    }

    ray_cam[0] = xd;
    ray_cam[1] = yd;
    ray_cam[2] = rd / std::tan(theta);
    ray_cam.normalize();
    return ray_cam;
}

bool Fisheye::project(const Eigen::Affine3d &T_w_lmk,
                      const std::shared_ptr<AModel3d> ldmk_model,
                      const Eigen::Vector3d &scale,
                      std::vector<Eigen::Vector2d> &p2d_vector) {

    for (const auto &p3d_model : ldmk_model->getModel()) {

        // conversion to the camera coordinate system
        Eigen::Vector3d t_w_lmk   = T_w_lmk * p3d_model.cwiseProduct(scale);
        Eigen::Vector3d t_cam_lmk = this->getWorld2SensorTransform() * t_w_lmk;

        // conversion to spherical coordinates
        double r = std::sqrt(t_cam_lmk(0) * t_cam_lmk(0) + t_cam_lmk(1) * t_cam_lmk(1) + t_cam_lmk(2) * t_cam_lmk(2));
        double theta = std::acos(t_cam_lmk(2) / r);
        double alpha = std::atan2(t_cam_lmk(1), t_cam_lmk(0));

        // mapping on the fisheye image plane
        double rd;

        if (_model == "equidistant") {
            rd = _calibration(0, 0) * theta;
        } else if (_model == "equisolid") {
            rd = 2 * _calibration(0, 0) * std::sin(theta / 2);
        } else if (_model == "stereographic") {
            rd = 2 * _calibration(0, 0) * std::tan(theta / 2);
        } else {
            std::cout << "Unknown fisheye lens model" << std::endl;
            return false;
        }

        // converting to pixel coordinates
        double u = rd * std::cos(alpha) * _rmax + _calibration(0, 2);
        double v = rd * std::sin(alpha) * _rmax + _calibration(1, 2);
        Eigen::Vector2d p2d(u, v);
        p2d_vector.push_back(p2d);

        if (t_cam_lmk[2] < 0.01) // point behind the camera
            return false;

        if (p2d[0] < 0 || p2d[1] < 0 || p2d[0] > _raw_data.cols || p2d[1] > _raw_data.rows) // out of image
            return false;

        if (!std::isfinite(p2d[0]) || !std::isfinite(p2d[1])) // Nan returned
            return false;
    }
    return true;
}

bool Fisheye::project(const Eigen::Affine3d &T_w_lmk,
                      const std::shared_ptr<AModel3d> ldmk_model,
                      const Eigen::Vector3d &scale,
                      const Eigen::Affine3d &T_f_w,
                      std::vector<Eigen::Vector2d> &p2d_vector) {
    for (const auto &p3d_model : ldmk_model->getModel()) {
        // conversion to the camera coordinate system
        Eigen::Vector3d t_w_lmk   = T_w_lmk * p3d_model.cwiseProduct(scale);
        Eigen::Vector3d t_cam_lmk = this->getFrame2SensorTransform() * T_f_w * t_w_lmk;

        // conversion to spherical coordinates
        double r = std::sqrt(t_cam_lmk(0) * t_cam_lmk(0) + t_cam_lmk(1) * t_cam_lmk(1) + t_cam_lmk(2) * t_cam_lmk(2));
        double theta = std::acos(t_cam_lmk(2) / r);
        double alpha = std::atan2(t_cam_lmk(1), t_cam_lmk(0));

        // mapping on the fisheye image plane
        double rd;

        if (_model == "equidistant") {
            rd = _calibration(0, 0) * theta;
        } else if (_model == "equisolid") {
            rd = 2 * _calibration(0, 0) * std::sin(theta / 2);
        } else if (_model == "stereographic") {
            rd = 2 * _calibration(0, 0) * std::tan(theta / 2);
        } else {
            std::cout << "Unknown fisheye lens model" << std::endl;
            return false;
        }

        // converting to pixel coordinates
        double u = rd * std::cos(alpha) * _rmax + _calibration(0, 2);
        double v = rd * std::sin(alpha) * _rmax + _calibration(1, 2);
        Eigen::Vector2d p2d(u, v);
        p2d_vector.push_back(p2d);

        if (t_cam_lmk[2] < 0.01) // point behind the camera
            return false;

        if (p2d[0] < 0 || p2d[1] < 0 || p2d[0] > _raw_data.cols || p2d[1] > _raw_data.rows) // out of image
            return false;

        if (!std::isfinite(p2d[0]) || !std::isfinite(p2d[1])) // Nan returned
            return false;
    }
    return true;
}
bool Fisheye::project(const Eigen::Affine3d &T_w_lmk,
                      const Eigen::Affine3d &T_f_w,
                      const Eigen::Matrix2d sqrt_info,
                      Eigen::Vector2d &p2d,
                      double *J_proj_frame,
                      double *J_proj_lmk) {

    // conversion to the camera coordinate system
    Eigen::Vector3d t_w_lmk   = T_w_lmk.translation();
    Eigen::Vector3d t_cam_lmk = this->getFrame2SensorTransform() * T_f_w * t_w_lmk;

    // conversion to spherical coordinates
    double r     = std::sqrt(t_cam_lmk(0) * t_cam_lmk(0) + t_cam_lmk(1) * t_cam_lmk(1) + t_cam_lmk(2) * t_cam_lmk(2));
    double theta = std::acos(t_cam_lmk(2) / r);
    double alpha = std::atan2(t_cam_lmk(1), t_cam_lmk(0));

    // mapping on the fisheye image plane
    double rd;

    if (_model == "equidistant") {
        rd = _calibration(0, 0) * theta;
    } else if (_model == "equisolid") {
        rd = 2 * _calibration(0, 0) * std::sin(theta / 2);
    } else if (_model == "stereographic") {
        rd = 2 * _calibration(0, 0) * std::tan(theta / 2);
    } else {
        std::cout << "Unknown fisheye lens model" << std::endl;
        return false;
    }

    // converting to pixel coordinates
    double u = rd * std::cos(alpha) * _rmax + _calibration(0, 2);
    double v = rd * std::sin(alpha) * _rmax + _calibration(1, 2);
    p2d << u, v;

    if (t_cam_lmk[2] < 0.01) // point behind the camera
        return false;

    if (p2d[0] < 0 || p2d[1] < 0 || p2d[0] > _raw_data.cols || p2d[1] > _raw_data.rows) // out of image
        return false;

    if (!std::isfinite(p2d[0]) || !std::isfinite(p2d[1])) // Nan returned
        return false;

    return true;
}

} // namespace isae