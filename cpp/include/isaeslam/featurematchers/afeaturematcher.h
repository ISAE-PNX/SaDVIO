#ifndef AFEATUREMATCHER_H
#define AFEATUREMATCHER_H

#include "isaeslam/typedefs.h"
#include <type_traits>

#include "isaeslam/data/features/AFeature2D.h"
#include "isaeslam/data/sensors/Camera.h"
#include "isaeslam/featuredetectors/aFeatureDetector.h"

namespace isae {

/**
 * @brief Interface for components matching one feature list with another
 * @author Damien Vivet
 */
using vec_feat_matches        = std::unordered_map<std::shared_ptr<AFeature>, std::vector<std::shared_ptr<AFeature>>>;
using vec_feat_matches_scores = std::unordered_map<std::shared_ptr<AFeature>, std::vector<double>>;

class AFeatureMatcher {
  public:
    AFeatureMatcher() {}
    AFeatureMatcher(std::shared_ptr<AFeatureDetector> detector) : _detector(detector) {}

    // Process 2D-2D cross-check matching and sort matches by score
    virtual uint match(std::vector<std::shared_ptr<AFeature>> &features1,
                       std::vector<std::shared_ptr<AFeature>> &features2,
                       std::vector<std::shared_ptr<AFeature>> &features_init,
                       vec_match &matches,
                       vec_match &matches_with_ldmks,
                       int searchAreaWidth  = 51,
                       int searchAreaHeight = 51);

    virtual uint ldmk_match(std::shared_ptr<ImageSensor> &sensor1,
                    vec_shared<ALandmark> &ldmks,
                    int searchAreaWidth  = 51,
                    int searchAreaHeight = 51);

  protected:
    virtual void getPossibleMatchesBetween(const std::vector<std::shared_ptr<AFeature>> &features1,
                                   const std::vector<std::shared_ptr<AFeature>> &features2,
                                   const std::vector<std::shared_ptr<AFeature>> &features_init,
                                   const uint &searchAreaWidth,
                                   const uint &searchAreaHeight,
                                   vec_feat_matches &matches,
                                   vec_feat_matches_scores &all_scores);

    vec_match filterMatches(vec_feat_matches &matches12,
                            vec_feat_matches &matches21,
                            vec_feat_matches_scores &all_scores12,
                            vec_feat_matches_scores &all_scores21);

    std::shared_ptr<AFeatureDetector> _detector; // feature detector for distance measurement
    double _first_second_match_score_ratio = 0.9;
    std::string _feature_label;
};

} // namespace isae
#endif // AFEATUREMATCHER_H
