#ifndef RGBDCAMERA_H
#define RGBDCAMERA_H

#include "isaeslam/data/sensors/Camera.h"

namespace isae {

class RGBDCamera : public Camera {
  public:
    RGBDCamera(const cv::Mat &image, const cv::Mat &depth, Eigen::Matrix3d K) : Camera(image, K) {
        _calibration = K;
        _raw_data     = image.clone();
        _depth       = depth.clone();
        _has_depth    = true;
    }

    Eigen::Vector3d getRayCamera(Eigen::Vector2d f);
    Eigen::Vector3d getRay(Eigen::Vector2d f);
    const cv::Mat &getDepthMat() override { return _depth; }
    std::vector<Eigen::Vector3d> getP3Dcam(const std::shared_ptr<AFeature> &feature) override;

  private:
    std::vector<double> getDepth(const std::shared_ptr<AFeature> &feature);
    cv::Mat _depth;
};

} // namespace isae

#endif // RGBDCAMERA_H