#ifndef SEMANTICBBOXFEATUREMATCHER_H
#define SEMANTICBBOXFEATUREMATCHER_H

#include <type_traits>
#include "isaeslam/typedefs.h"

#include "isaeslam/featurematchers/afeaturematcher.h"

namespace isae {



class semanticBBoxFeatureMatcher : public AFeatureMatcher{
public:

    semanticBBoxFeatureMatcher(){}
    semanticBBoxFeatureMatcher(std::shared_ptr<AFeatureDetector> detector) : AFeatureMatcher(detector) {_feature_label="bboxxd";}
};

}// namespace isae

#endif // SEMANTICBBOXFEATUREMATCHER_H
