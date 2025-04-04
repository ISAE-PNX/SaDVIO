#ifndef CVGFTTFEATUREDETECTOR_H
#define CVGFTTFEATUREDETECTOR_H

#include "isaeslam/featuredetectors/aOpenCVFeatureDetector.h"

namespace isae {

class cvGFTTFeatureDetector : public AOpenCVFeatureDetector {
  public:
    cvGFTTFeatureDetector(int n, int n_per_cell, double max_matching_dist = 64)
        : AOpenCVFeatureDetector(n, n_per_cell) {
        _max_matching_dist = max_matching_dist;
        _detector          = cv::GFTTDetector::create(n_per_cell);
        _descriptor        = cv::ORB::create();
    }
    void init() override {}
    double getDist(const cv::Mat &desc1, const cv::Mat &desc2) const override {
        return cv::norm(desc1, desc2, _descriptor->defaultNorm());
    }
    std::vector<std::shared_ptr<AFeature>> detectAndComputeGrid(
        const cv::Mat &img,
        const cv::Mat &mask,
        std::vector<std::shared_ptr<AFeature>> existing_features) override;
};

} // namespace isae

#endif // CVGFTTFEATUREDETECTOR_H