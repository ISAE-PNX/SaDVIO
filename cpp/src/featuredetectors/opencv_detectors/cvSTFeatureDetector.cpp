#include "isaeslam/featuredetectors/opencv_detectors/cvSTFeatureDetector.h"

namespace isae {

void cvSTFeatureDetector::init() {
    _detector   = cv::GFTTDetector::create(250, 0.001, 20, 3, false, 0.04);
    _descriptor = cv::ORB::create();
}

double cvSTFeatureDetector::getDist(const cv::Mat &desc1, const cv::Mat &desc2) const {

    return cv::norm(desc1, desc2, _descriptor->defaultNorm());
}

} // namespace isae