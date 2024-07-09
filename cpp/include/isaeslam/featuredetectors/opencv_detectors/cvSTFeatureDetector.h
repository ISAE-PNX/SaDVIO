#ifndef CVSTFEATUREDETECTOR_H
#define CVSTFEATUREDETECTOR_H

#include "isaeslam/featuredetectors/aOpenCVFeatureDetector.h"

namespace isae {

class cvSTFeatureDetector : public AOpenCVFeatureDetector {
  public:
    cvSTFeatureDetector(int n, int n_per_cell, double max_matching_dist = 0.5) : AOpenCVFeatureDetector(n, n_per_cell) {
        _max_matching_dist = max_matching_dist;
        this->init();
    }

    void init() override;
    double getDist(const cv::Mat &desc1, const cv::Mat &desc2) const override;
};

} // namespace isae

#endif // CVSTFEATUREDETECTOR_H