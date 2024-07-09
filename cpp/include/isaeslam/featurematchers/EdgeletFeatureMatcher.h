#ifndef EDGELETFEATUREMATCHER_H
#define EDGELETFEATUREMATCHER_H

#include <type_traits>
#include "isaeslam/typedefs.h"

#include "isaeslam/featurematchers/afeaturematcher.h"
#include "isaeslam/featurematchers/EdgeletFeatureTracker.h"


namespace isae {



class EdgeletFeatureMatcher : public AFeatureMatcher{
public:

    EdgeletFeatureMatcher(){}
    EdgeletFeatureMatcher(std::shared_ptr<AFeatureDetector> detector) : AFeatureMatcher(detector) {
        _feature_label = "edgeletxd";
        isae::EdgeletFeatureTracker trackerObj(detector);
        _tracker = std::make_shared<isae::EdgeletFeatureTracker>(trackerObj);
    }

    uint match(std::shared_ptr<ImageSensor> &sensor1, std::shared_ptr<ImageSensor> &sensor2, vec_match &matches, vec_match &matches_with_ldmks,
                                int searchAreaWidth = 51, int searchAreaHeight = 51);

    private:
        std::shared_ptr<EdgeletFeatureTracker> _tracker;
};

}// namespace isae

#endif // EDGELETFEATUREMATCHER_H
