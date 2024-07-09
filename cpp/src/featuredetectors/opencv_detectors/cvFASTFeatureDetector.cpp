#include "isaeslam/featuredetectors/opencv_detectors/cvFASTFeatureDetector.h"
#include <mutex>
#include <thread>

namespace isae {

void cvFASTFeatureDetector::init() {
    _detector   = cv::FastFeatureDetector::create(10, true, cv::FastFeatureDetector::TYPE_9_16);
    _descriptor = cv::ORB::create();
}

double cvFASTFeatureDetector::getDist(const cv::Mat &desc1, const cv::Mat &desc2) const {

    return cv::norm(desc1, desc2, _descriptor->defaultNorm());
}

} // namespace isae