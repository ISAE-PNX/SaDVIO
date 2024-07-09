#include "isaeslam/featuredetectors/custom_detectors/EllipsePatternFeatureDetector.h"


namespace isae {



    void EllipsePatternFeatureDetector::init()
    {
        _defaultNorm = cv::NORM_HAMMING2;
        _max_matching_dist = 0.1;
    }

    double EllipsePatternFeatureDetector::getDist(const cv::Mat &desc1, const cv::Mat &desc2) const
    {
        return cv::norm(desc1, desc2, cv::NORM_L1);
    }


    void EllipsePatternFeatureDetector::customDetectAndCompute(const cv::Mat &img, const cv::Mat &mask, std::vector<std::shared_ptr<AFeature> > &features)
    {
        cv::Mat descriptors;
        std::vector<EllipsePattern> ellipses;
        EllipseExtractor->extract(img, ellipses, descriptors);

        for (uint i = 0; i < ellipses.size(); ++i) {
            cv::Mat desc = descriptors.row(i);
            if (mask.at<int>(ellipses.at(i).p2ds().at(0)(0),ellipses.at(i).p2ds().at(0)(1)) == 255){
                features.push_back(std::make_shared<isae::EllipsePattern2D>(ellipses.at(i).p2ds(), desc));
            }
        }

    }

    void EllipsePatternFeatureDetector::computeDescriptor(const cv::Mat &img, std::vector<std::shared_ptr<AFeature> > &features){

        std::cerr << "EllipsePatternFeatureDetector::computeDescriptor NOT IMPLEMENTED YET" << std::endl;


        // remove not described features
        deleteUndescribedFeatures(features);
    }





}// namespace isae
