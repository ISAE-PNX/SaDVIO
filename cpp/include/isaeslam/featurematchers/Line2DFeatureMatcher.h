#ifndef LINE2DFEATUREMATCHER_H
#define LINE2DFEATUREMATCHER_H

#include <type_traits>
#include "isaeslam/typedefs.h"
#include "isaeslam/featurematchers/afeaturematcher.h"

#include <opencv2/line_descriptor.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>

namespace isae {



class Line2DFeatureMatcher : public AFeatureMatcher{
public:

    Line2DFeatureMatcher(){}
    Line2DFeatureMatcher(std::shared_ptr<AFeatureDetector> detector) : AFeatureMatcher(detector) {
        _feature_label = "linexd";
        _bd_line_matcher = cv::line_descriptor::BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    }

    // Process 2D-2D cross-check matching and sort matches by score
    uint match(std::vector<std::shared_ptr<AFeature>> &features1,
                       std::vector<std::shared_ptr<AFeature>> &features2,
                       std::vector<std::shared_ptr<AFeature>> &features_init,
                       vec_match &matches,
                       vec_match &matches_with_ldmks,
                       int searchAreaWidth = 51,
                       int searchAreaHeight = 51) override;

    uint ldmk_match(std::shared_ptr<ImageSensor> &sensor1,
                    vec_shared<ALandmark> &ldmks,
                    int searchAreaWidth  = 51,
                    int searchAreaHeight  = 51) override;

private:
    void getPossibleMatchesBetween(const std::vector<std::shared_ptr<AFeature>> &features1,
                                   const std::vector<std::shared_ptr<AFeature>> &features2,
                                   const std::vector<std::shared_ptr<AFeature>> &features_init,
                                   const uint &searchAreaWidth,
                                   const uint &searchAreaHeight,
                                   vec_feat_matches &matches,
                                   vec_feat_matches_scores &all_scores) override;

    cv::Ptr<cv::line_descriptor::BinaryDescriptorMatcher> _bd_line_matcher;

};

}// namespace isae

#endif // LINE2DFEATUREMATCHER_H
