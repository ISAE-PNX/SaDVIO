#include "isaeslam/featurematchers/semanticBBoxFeatureTracker.h"
#include "isaeslam/data/features/BBox2d.h"
#include <map>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

namespace isae {

uint semanticBBoxFeatureTracker::track(std::shared_ptr<ImageSensor> &sensor1,
                                       std::shared_ptr<ImageSensor> &sensor2,
                                       std::vector<std::shared_ptr<AFeature>> &features_to_track,
                                       std::vector<std::shared_ptr<AFeature>> &features_init,
                                       vec_match &matches,
                                       vec_match &tracks_with_ldmk,
                                       int search_width,
                                       int search_height,
                                       int nlvls_pyramids,
                                       double max_err,
                                       bool backward) {

    if (features_to_track.size() < 1 || sensor2->getFeatures("bboxxd").size() < 1)
        return 0;

    // detect features on sensor2
    std::vector<std::shared_ptr<AFeature>> features2;
    features2 = sensor2->getFeatures("bboxxd");
    // sensor2->addFeatures(feature_label, features2);
    _matcher->match(features_to_track, features2, features_to_track, matches, tracks_with_ldmk, 1000, 1000);

    return matches.size();
}

} // namespace isae
