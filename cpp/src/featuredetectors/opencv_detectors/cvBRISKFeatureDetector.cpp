#include "isaeslam/featuredetectors/opencv_detectors/cvBRISKFeatureDetector.h"

#include <mutex>
#include <thread>

namespace isae {

void cvBRISKFeatureDetector::init()
{
    _detector = cv::BRISK::create(30,
                                  3,
                                  1.0);
    _descriptor = _detector;
}

double cvBRISKFeatureDetector::getDist(const cv::Mat &desc1, const cv::Mat &desc2) const
{

    return cv::norm(desc1, desc2, _detector->defaultNorm());
}

}