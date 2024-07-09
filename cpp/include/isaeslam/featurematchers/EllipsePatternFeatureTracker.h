#ifndef ELLIPSEPATTERNFEATURETRACKER_H
#define ELLIPSEPATTERNFEATURETRACKER_H

#include "isaeslam/data/features/EllipsePattern2D.h"
#include "isaeslam/featurematchers/EllipsePatternFeatureMatcher.h"
#include "isaeslam/featurematchers/afeaturetracker.h"
#include "isaeslam/typedefs.h"
#include <type_traits>

namespace isae {

class EllipsePatternFeatureTracker : public AFeatureTracker {
  public:
    EllipsePatternFeatureTracker() {}
    EllipsePatternFeatureTracker(std::shared_ptr<AFeatureDetector> detector) : AFeatureTracker(detector) {
        _matcher = std::make_shared<EllipsePatternFeatureMatcher>(_detector);
    }

    uint track(std::shared_ptr<ImageSensor> &sensor1,
               std::shared_ptr<ImageSensor> &sensor2,
               std::vector<std::shared_ptr<AFeature>> &features_to_track,
               std::vector<std::shared_ptr<AFeature>> &features_init,
               vec_match &tracks,
               vec_match &tracksWithLdmk,
               int search_width   = 21,
               int search_height  = 21,
               int nlvls_pyramids = 3,
               double max_err     = 10,
               bool backward      = false) override;

  private:
    std::shared_ptr<EllipsePatternFeatureMatcher> _matcher;
};

} // namespace isae

#endif // ELLIPSEPATTERNFEATURETRACKER_H
