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
        _has_depth = false;
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

} // namespace isae

#endif
