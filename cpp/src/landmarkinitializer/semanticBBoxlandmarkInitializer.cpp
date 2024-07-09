#include "isaeslam/landmarkinitializer/semanticBBoxlandmarkInitializer.h"
#include "isaeslam/data/features/AFeature2D.h"
#include "isaeslam/data/landmarks/Point3D.h"
#include "utilities/geometry.h"

namespace isae {


bool semanticBBoxLandmarkInitializer::initLandmark(std::vector<std::shared_ptr<isae::AFeature> > features, std::shared_ptr<isae::ALandmark> &landmark)
{
    // Get the 3D landmark pose initialy read from the GT file
    std::shared_ptr<ALandmark> ldmk = features.at(0)->getLandmark().lock();
    ldmk->setPose(features.at(0)->getSensor()->getSensor2WorldTransform()*ldmk->getPose());
    ldmk->setInlier();
    return true;
}

bool semanticBBoxLandmarkInitializer::initLandmarkWithDepth(std::vector<std::shared_ptr<AFeature> > features, std::shared_ptr<ALandmark> &landmark){

    return true;
}


std::shared_ptr<ALandmark> semanticBBoxLandmarkInitializer::createNewLandmark(std::shared_ptr<isae::AFeature> f)
{
    // can't init BBox3d with only one feature
    return nullptr;
}

} // namespace isae
