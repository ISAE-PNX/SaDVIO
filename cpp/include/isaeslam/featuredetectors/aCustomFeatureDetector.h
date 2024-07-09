#ifndef ACUSTOMFEATUREDETECTOR_H
#define ACUSTOMFEATUREDETECTOR_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "isaeslam/data/features/Point2D.h"
#include "isaeslam/data/sensors/ASensor.h"
#include "isaeslam/featuredetectors/aFeatureDetector.h"
#include "isaeslam/typedefs.h"

namespace isae {

class ACustomFeatureDetector : public AFeatureDetector {
  public:
    ACustomFeatureDetector(int n, int n_per_cell) : AFeatureDetector(n, n_per_cell) {}

    void detectAndCompute(const cv::Mat &img,
                          const cv::Mat &mask,
                          std::vector<cv::KeyPoint> &keypoints,
                          cv::Mat &descriptors,
                          int n_points = 0) {}
    std::vector<std::shared_ptr<AFeature>> detectAndComputeGrid(
        const cv::Mat &img,
        const cv::Mat &mask,
        std::vector<std::shared_ptr<AFeature>> existing_features = std::vector<std::shared_ptr<AFeature>>());

    virtual void customDetectAndCompute(const cv::Mat &img,
                                        const cv::Mat &mask,
                                        std::vector<std::shared_ptr<AFeature>> &features)                = 0;
    virtual void computeDescriptor(const cv::Mat &img, std::vector<std::shared_ptr<AFeature>> &features) = 0;

  protected:
    const int getDefaultNorm() const { return _defaultNorm; }
    int _defaultNorm;
};

} // namespace isae

#endif // ACUSTOMFEATUREDETECTOR_H
