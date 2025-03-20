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

Eigen::Vector3d Omni::getRay(Eigen::Vector2d f) {
    Eigen::Vector3d ray_world;
    Eigen::Vector3d ray_cam;

    ray_cam   = this->getRayCamera(f);
    ray_world = this->getSensor2WorldTransform().rotation() * ray_cam;
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

// This function is inspired by VINS-Mono implementation
// Source : https://github.com/HKUST-Aerial-Robotics/VINS-Mono/blob/master/camera_model/src/camera_models/CataCamera.cc
Eigen::Vector3d Omni::getRayCamera(Eigen::Vector2d f) {

    double mx_d, my_d, mx2_d, mxy_d, my2_d, mx_u, my_u;
    double rho2_d, rho4_d, radDist_d, Dx_d, Dy_d, inv_denom_d;
    double lambda;
    Eigen::Vector3d ray_cam;

    // Lift points to normalised plane
    mx_d = ((f(0) - _calibration(0, 2)) * (1 - _alpha)) / _calibration(0, 0);
    my_d = ((f(1) - _calibration(1, 2)) * (1 - _alpha)) / _calibration(1, 1);

    if (!_distortion) {
        mx_u = mx_d;
        my_u = my_d;
    } else {
        // Apply inverse distortion model

        double k1 = _D(0);
        double k2 = _D(1);
        double p1 = _D(2);
        double p2 = _D(3);

        // Inverse distortion model
        // proposed by Heikkila
        mx2_d       = mx_d * mx_d;
        my2_d       = my_d * my_d;
        mxy_d       = mx_d * my_d;
        rho2_d      = mx2_d + my2_d;
        rho4_d      = rho2_d * rho2_d;
        radDist_d   = k1 * rho2_d + k2 * rho4_d;
        Dx_d        = mx_d * radDist_d + p2 * (rho2_d + 2 * mx2_d) + 2 * p1 * mxy_d;
        Dy_d        = my_d * radDist_d + p1 * (rho2_d + 2 * my2_d) + 2 * p2 * mxy_d;
        inv_denom_d = 1 / (1 + 4 * k1 * rho2_d + 6 * k2 * rho4_d + 8 * p1 * my_d + 8 * p2 * mx_d);

        mx_u = mx_d - inv_denom_d * Dx_d;
        my_u = my_d - inv_denom_d * Dy_d;
    }

    // Lift normalised points to the sphere (inv_hslash)
    if (_xi == 1.0) {
        lambda = 2.0 / (mx_u * mx_u + my_u * my_u + 1.0);
        ray_cam << lambda * mx_u, lambda * my_u, lambda - 1.0;
    } else {
        lambda =
            (_xi + sqrt(1.0 + (1.0 - _xi * _xi) * (mx_u * mx_u + my_u * my_u))) / (1.0 + mx_u * mx_u + my_u * my_u);
        ray_cam << lambda * mx_u, lambda * my_u, lambda - _xi;
    }

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

// This function is inspired by VINS-Mono implementation
// Source : https://github.com/HKUST-Aerial-Robotics/VINS-Mono/blob/master/camera_model/src/camera_models/CataCamera.cc
Eigen::Vector2d Omni::distort(const Eigen::Vector2d &p) {
    
    double k1 = _D(0);
    double k2 = _D(1);
    double p1 = _D(2);
    double p2 = _D(3);

    double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

    mx2_u      = p(0) * p(0);
    my2_u      = p(1) * p(1);
    mxy_u      = p(0) * p(1);
    rho2_u     = mx2_u + my2_u;
    rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
    Eigen::Vector2d d;
    d << p(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u),
        p(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);

    return p + d;
}

// This function is inspired by VINS-Mono implementation
// Source : https://github.com/HKUST-Aerial-Robotics/VINS-Mono/blob/master/camera_model/src/camera_models/CataCamera.cc
bool Omni::project(const Eigen::Affine3d &T_w_lmk,
                   const std::shared_ptr<AModel3d> ldmk_model,
                   const Eigen::Vector3d &scale,
                   std::vector<Eigen::Vector2d> &p2d_vector) {

    for (const auto &p3d_model : ldmk_model->getModel()) {
        // conversion to the camera coordinate system
        Eigen::Vector3d t_w_lmk   = T_w_lmk * p3d_model.cwiseProduct(scale);
        Eigen::Vector3d t_cam_lmk = this->getWorld2SensorTransform() * t_w_lmk;

        if (t_cam_lmk[2] < 0.1) // point behind the camera
            return false;

        // Project on the normalized plane
        double z             = t_cam_lmk[2] + _xi * t_cam_lmk.norm();
        Eigen::Vector2d p2dn = t_cam_lmk.block<2, 1>(0, 0) / z;

        // Distort if necessary
        if (_distortion) {
            p2dn = this->distort(p2dn);
        }

        // Omni model as defined by Mei et al.
        Eigen::Vector2d p2d;
        p2d << _calibration(0, 0) * p2dn(0) / (1 - _alpha) + _calibration(0, 2),
            _calibration(1, 1) * p2dn(1) / (1 - _alpha) + _calibration(1, 2);
        p2d_vector.push_back(p2d);

        // Check validity
        double w;
        double d =
            std::sqrt(t_cam_lmk.x() * t_cam_lmk.x() + t_cam_lmk.y() * t_cam_lmk.y() + t_cam_lmk.z() * t_cam_lmk.z());
        if (_alpha <= 0.5)
            w = _alpha / (1 - _alpha);
        else
            w = (1 - _alpha) / _alpha;

        if (t_cam_lmk.z() <= -w * d)
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

// This function is inspired by VINS-Mono implementation
// Source : https://github.com/HKUST-Aerial-Robotics/VINS-Mono/blob/master/camera_model/src/camera_models/CataCamera.cc
bool Omni::project(const Eigen::Affine3d &T_w_lmk,
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

        // Project on the normalized plane
        double z             = t_cam_lmk[2] + _xi * t_cam_lmk.norm();
        Eigen::Vector2d p2dn = t_cam_lmk.block<2, 1>(0, 0) / z;
        
        // Distort if necessary
        if (_distortion) {
            p2dn = p2dn + this->distort(p2dn);
        }

        // Omni model as defined by Mei et al.
        Eigen::Vector2d p2d;
        p2d << _calibration(0, 0) * p2dn(0) / (1 - _alpha) + _calibration(0, 2),
            _calibration(1, 1) * p2dn(1) / (1 - _alpha) + _calibration(1, 2);

        p2d_vector.push_back(p2d);

        // Check validity
        double w;
        double d =
            std::sqrt(t_cam_lmk.x() * t_cam_lmk.x() + t_cam_lmk.y() * t_cam_lmk.y() + t_cam_lmk.z() * t_cam_lmk.z());
        if (_alpha <= 0.5)
            w = _alpha / (1 - _alpha);
        else
            w = (1 - _alpha) / _alpha;

        if (t_cam_lmk.z() <= -w * d)
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

bool Omni::project(const Eigen::Affine3d &T_w_lmk,
                   const Eigen::Affine3d &T_f_w,
                   const Eigen::Matrix2d sqrt_info,
                   Eigen::Vector2d &p2d,
                   double *J_proj_frame,
                   double *J_proj_lmk) {
    return false;
}

} // namespace isae