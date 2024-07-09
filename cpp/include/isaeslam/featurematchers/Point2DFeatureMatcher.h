#ifndef POINT2DFEATUREMATCHER_H
#define POINT2DFEATUREMATCHER_H

#include <type_traits>
#include "isaeslam/typedefs.h"

#include "isaeslam/featurematchers/afeaturematcher.h"

namespace isae {



class Point2DFeatureMatcher : public AFeatureMatcher{
public:

    Point2DFeatureMatcher(){}
    Point2DFeatureMatcher(std::shared_ptr<AFeatureDetector> detector) : AFeatureMatcher(detector) {_feature_label="pointxd";}
};

}// namespace isae

#endif // POINT2DFEATUREMATCHER_H
