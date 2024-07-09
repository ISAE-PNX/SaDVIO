#include "isaeslam/featuredetectors/opencv_detectors/cvORBFeatureDetector.h"

namespace isae {



void cvORBFeatureDetector::init()
{
    _detector = cv::ORB::create(_n_total, 1.2f, 8, 0, 0, 2, cv::ORB::FAST_SCORE, 31, 5);
    _descriptor = _detector;
}

double cvORBFeatureDetector::getDist(const cv::Mat &desc1, const cv::Mat &desc2) const
{

    return cv::norm(desc1, desc2, _detector->defaultNorm());
}


}// namespace isae
