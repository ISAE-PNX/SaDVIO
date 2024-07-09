#include "isaeslam/featuredetectors/opencv_detectors/cvKAZEFeatureDetector.h"

#include <mutex>
#include <thread>

namespace isae {

void cvKAZEFeatureDetector::init()
{
    _detector = cv::KAZE::create(false,
                                 false,
                                 0.001,
                                 3,
                                 3);
    _descriptor = _detector;
}

double cvKAZEFeatureDetector::getDist(const cv::Mat &desc1, const cv::Mat &desc2) const
{

    return cv::norm(desc1, desc2, _detector->defaultNorm());
}


}// namespace isae