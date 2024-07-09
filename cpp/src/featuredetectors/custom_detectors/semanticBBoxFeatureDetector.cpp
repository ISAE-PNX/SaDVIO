#include "isaeslam/featuredetectors/custom_detectors/semanticBBoxFeatureDetector.h"

namespace isae {



    void semanticBBoxFeatureDetector::init()
    {
        _defaultNorm = cv::NORM_L1;
        _max_matching_dist = 0.1;
    }

    double semanticBBoxFeatureDetector::getDist(const cv::Mat &desc1, const cv::Mat &desc2) const
    {
        return cv::norm(desc1, desc2, this->getDefaultNorm());
    }

    void semanticBBoxFeatureDetector::customDetectAndCompute(const cv::Mat &img, const cv::Mat &mask, std::vector<std::shared_ptr<AFeature> > &features)
    {
        // GT detection already set in frame by provider
    }

    void semanticBBoxFeatureDetector::computeDescriptor(const cv::Mat &img, std::vector<std::shared_ptr<AFeature> > &features)
    {

    }


}// namespace isae


