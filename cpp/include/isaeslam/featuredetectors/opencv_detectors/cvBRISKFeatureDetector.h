#ifndef CVBRISKFEATUREDETECTOR_H
#define CVBRISKFEATUREDETECTOR_H

#include "isaeslam/featuredetectors/aOpenCVFeatureDetector.h"

namespace isae {

class cvBRISKFeatureDetector : public AOpenCVFeatureDetector {
  public:
    cvBRISKFeatureDetector(int n, int n_per_cell, double max_matching_dist = 64)
        : AOpenCVFeatureDetector(n, n_per_cell) {
        _max_matching_dist = max_matching_dist;
        this->init();
    }

    void init() override;
    double getDist(const cv::Mat &desc1, const cv::Mat &desc2) const override;
};

} // namespace isae

#endif // CVBRISKFEATUREDETECTOR_H