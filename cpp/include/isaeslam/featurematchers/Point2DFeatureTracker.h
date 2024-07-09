#ifndef POINT2DFEATURETRACKER_H
#define POINT2DFEATURETRACKER_H

#include "isaeslam/typedefs.h"
#include <type_traits>

#include "isaeslam/featurematchers/afeaturetracker.h"

namespace isae {

class Point2DFeatureTracker : public AFeatureTracker {
  public:
    Point2DFeatureTracker() {}
    Point2DFeatureTracker(std::shared_ptr<AFeatureDetector> detector)
        : AFeatureTracker(detector), _termCrit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01) {}

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
};

} // namespace isae

#endif // POINT2DFEATURETRACKER_H
