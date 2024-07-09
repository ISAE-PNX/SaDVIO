#include "isaeslam/featurematchers/EllipsePatternFeatureTracker.h"

namespace isae {

uint EllipsePatternFeatureTracker::track(std::shared_ptr<isae::ImageSensor> &sensor1,
                                         std::shared_ptr<isae::ImageSensor> &sensor2,
                                         std::vector<std::shared_ptr<AFeature>> &features_to_track,
                                         std::vector<std::shared_ptr<AFeature>> &features_init,
                                         vec_match &tracks,
                                         vec_match &tracks_with_ldmk,
                                         int search_width,
                                         int search_height,
                                         int nlvls_pyramids,
                                         double max_err,
                                         bool backward) {

    if (features_to_track.size() < 1)
        return 0;

    // detect features on sensor2
    std::vector<std::shared_ptr<AFeature>> features2;
    features2 = _detector->detectAndComputeGrid(sensor2->getRawData(), sensor2->getMask(), features2);
    sensor2->addFeatures(_feature_label, features2);
    _matcher->match(
        features_to_track, features2, features_to_track, tracks, tracks_with_ldmk, search_width, search_height);

    return tracks.size();
}

} // namespace isae