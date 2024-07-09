#ifndef LINE2DFEATURETRACKER_H
#define LINE2DFEATURETRACKER_H

#include "isaeslam/typedefs.h"
#include <type_traits>

#include "isaeslam/featurematchers/Line2DFeatureMatcher.h"
#include "isaeslam/featurematchers/afeaturetracker.h"
#include "isaeslam/featuredetectors/custom_detectors/Line2DFeatureDetector.h"
#include "isaeslam/data/features/Line2D.h"

namespace isae {

class Line2DFeatureTracker : public AFeatureTracker {
  public:
    Line2DFeatureTracker() {}
    Line2DFeatureTracker(std::shared_ptr<AFeatureDetector> detector)
        : AFeatureTracker(detector), _termCrit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.01) {}

    uint track(std::shared_ptr<isae::ImageSensor> &sensor1,
               std::shared_ptr<isae::ImageSensor> &sensor2,
               std::vector<std::shared_ptr<AFeature>> &features_to_track,
               std::vector<std::shared_ptr<AFeature>> &features_init,
               vec_match &tracks,
               vec_match &tracks_with_ldmk,
               int search_width   = 21,
               int search_height  = 21,
               int nlvls_pyramids = 3,
               double max_err     = 10,
               bool backward      = false) override;

  private:
    cv::TermCriteria _termCrit; //< termination criteria for the optical flow algorithm
    double _klt_max_err       = 50.;
    double _max_backward_dist = 5;
};

} // namespace isae

#endif // LINE2DFEATURETRACKER_H
