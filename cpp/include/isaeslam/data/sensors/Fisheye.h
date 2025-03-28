#ifndef FISHEYE_H
#define FISHEYE_H

#include "isaeslam/data/sensors/ASensor.h"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/types.hpp>

namespace isae {

class Fisheye : public ImageSensor {

  public:
    Fisheye(const cv::Mat &image, Eigen::Matrix3d K, std::string model, float rmax)
        : ImageSensor(), _model(model), _rmax(rmax) {
        _calibration = K;
        _raw_data    = image.clone();
        _mask        = cv::Mat(_raw_data.rows, _raw_data.cols, CV_8UC1, cv::Scalar(255));
        _has_depth   = false;
    }

    bool project(const Eigen::Affine3d &T_w_lmk,
                 const std::shared_ptr<AModel3d> ldmk_model,
                 const Eigen::Vector3d &scale,
                 std::vector<Eigen::Vector2d> &p2ds) override;
    bool project(const Eigen::Affine3d &T_w_lmk,
                 const std::shared_ptr<AModel3d> ldmk_model,
                 const Eigen::Vector3d &scale,
                 const Eigen::Affine3d &T_f_w,
                 std::vector<Eigen::Vector2d> &p2ds) override;
    bool project(const Eigen::Affine3d &T_w_lmk,
                 const Eigen::Affine3d &T_f_w,
                 const Eigen::Matrix2d sqrt_info,
                 Eigen::Vector2d &p2d,
                 double *J_proj_frame,
                 double *J_proj_lmk) override;

    Eigen::Vector3d getRayCamera(Eigen::Vector2d f);
    Eigen::Vector3d getRay(Eigen::Vector2d f);
    double getFocal() override { return _rmax; }
    const cv::Mat &getDepthMat() override { return _raw_data; }
    std::vector<Eigen::Vector3d> getP3Dcam(const std::shared_ptr<AFeature> &feature) override {
        std::vector<Eigen::Vector3d> p3ds;
        return p3ds;
    }

  private:
    std::string _model;
    float _rmax;
};

class Omni : public ImageSensor {

  public:
    Omni(const cv::Mat &image, Eigen::Matrix3d K, double xi) : ImageSensor(), _xi(xi) {
        _calibration = K;
        _alpha       = _xi / (1 + _xi);
        _raw_data    = image.clone();
        _mask        = cv::Mat(_raw_data.rows, _raw_data.cols, CV_8UC1, cv::Scalar(255));
        _has_depth   = false;
        _distortion  = false; 
    }

    Omni(const cv::Mat &image, Eigen::Matrix3d K, double xi, Eigen::Vector4d D) : ImageSensor(), _xi(xi) {
        _calibration = K;
        _alpha       = _xi / (1 + _xi);
        _raw_data    = image.clone();
        _mask        = cv::Mat(_raw_data.rows, _raw_data.cols, CV_8UC1, cv::Scalar(255));
        _has_depth   = false;
        _distortion  = true;
        _D           = D;
    }

    bool project(const Eigen::Affine3d &T_w_lmk,
                 const std::shared_ptr<AModel3d> ldmk_model,
                 const Eigen::Vector3d &scale,
                 std::vector<Eigen::Vector2d> &p2ds) override;
    bool project(const Eigen::Affine3d &T_w_lmk,
                 const std::shared_ptr<AModel3d> ldmk_model,
                 const Eigen::Vector3d &scale,
                 const Eigen::Affine3d &T_f_w,
                 std::vector<Eigen::Vector2d> &p2ds) override;
    bool project(const Eigen::Affine3d &T_w_lmk,
                 const Eigen::Affine3d &T_f_w,
                 const Eigen::Matrix2d sqrt_info,
                 Eigen::Vector2d &p2d,
                 double *J_proj_frame,
                 double *J_proj_lmk) override;

    Eigen::Vector3d getRayCamera(Eigen::Vector2d f);
    Eigen::Vector3d getRay(Eigen::Vector2d f);
    double getFocal() override { return (_calibration(0, 0) + _calibration(1, 1)) / 2; }
    const cv::Mat &getDepthMat() override { return _raw_data; }
    std::vector<Eigen::Vector3d> getP3Dcam(const std::shared_ptr<AFeature> &feature) override {
        std::vector<Eigen::Vector3d> p3ds;
        return p3ds;
    }
    Eigen::Vector2d distort(const Eigen::Vector2d &p);

  private:
    double _xi, _alpha;
    bool _distortion;
    Eigen::Vector4d _D;
};

} // namespace isae

#endif
