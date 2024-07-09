#ifndef ELLIPSEPATTERNFEATUREMATCHER_H
#define ELLIPSEPATTERNFEATUREMATCHER_H

#include <type_traits>
#include "isaeslam/typedefs.h"

#include "isaeslam/featurematchers/afeaturematcher.h"

namespace isae {



class EllipsePatternFeatureMatcher : public AFeatureMatcher{
public:

    EllipsePatternFeatureMatcher(){}
    EllipsePatternFeatureMatcher(std::shared_ptr<AFeatureDetector> detector) : AFeatureMatcher(detector) {_feature_label = "ellipsepatternxd";}

};

}// namespace isae

#endif // ELLIPSEPATTERNFEATUREMATCHER_H
