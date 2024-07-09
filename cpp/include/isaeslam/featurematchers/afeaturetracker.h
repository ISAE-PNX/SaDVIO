#ifndef AFEATURETRACKER_H
#define AFEATURETRACKER_H

#include <memory>
#include <utility>
#include <vector>

#include "isaeslam/data/sensors/Camera.h"
#include "isaeslam/featuredetectors/aFeatureDetector.h"
#include "isaeslam/typedefs.h"

namespace isae {

/**
 * @brief Feature tracker that requires the two FeatureSets to have small differences
 */

class AFeatureTracker {
  public:
    AFeatureTracker() {}
    AFeatureTracker(std::shared_ptr<AFeatureDetector> detector) : _detector(detector) {}

    virtual uint track(std::shared_ptr<isae::ImageSensor> &sensor1,
                       std::shared_ptr<isae::ImageSensor> &sensor2,
                       std::vector<std::shared_ptr<AFeature>> &features_to_track,
                       std::vector<std::shared_ptr<AFeature>> &features_init,
                       vec_match &tracks,
                       vec_match &tracks_with_ldmk,
                       int search_width   = 21,
                       int search_height  = 21,
                       int nlvls_pyramids = 3,
                       double max_err     = 10,
                       bool backward      = false) = 0;

  protected:
    std::shared_ptr<AFeatureDetector> _detector; //< feature detector for feature init
    std::string _feature_label;
};

} // namespace isae

#endif // AFEATURETRACKER_H
