#include "isaeslam/landmarkinitializer/alandmarkinitializer.h"
#include "isaeslam/data/features/AFeature2D.h"
#include <isaeslam/data/sensors/Camera.h>

namespace isae {

uint ALandmarkInitializer::initFromMatches(vec_match matches) {
    uint nb_valid_ldmk = 0;

    for (auto match : matches)
        nb_valid_ldmk += this->initFromMatch(match);
    return nb_valid_ldmk;
}

uint ALandmarkInitializer::initFromFeatures(std::vector<std::shared_ptr<AFeature>> feats) {
    std::shared_ptr<ALandmark> l;

    // Check if there is already a landmark
    for (auto feat : feats) {
        if (feat->getLandmark().lock()) {
            l = feat->getLandmark().lock();
            break;
        }
    }

    // In the case of an initialized lmk, associate the other features to it
    if (l) {

        if (l->isInitialized()) {
            for (auto feat : feats) {
                feat->linkLandmark(l);
                feat->getSensor()->getFrame()->addLandmark(l);
            }
            return 1;
        }
    }

    // Perform multiview triangulation if there is no initialized lmk
    if (!initLandmark(feats, l))
        return 0;

    // Associate this landmark to its frames (check if already added first)
    std::vector<std::shared_ptr<Frame>> addedframes;
    for (auto feat : feats) {
        if (std::find(addedframes.begin(), addedframes.end(), feat->getSensor()->getFrame()) == addedframes.end()) {
            feat->getSensor()->getFrame()->addLandmark(l);
            addedframes.push_back(feat->getSensor()->getFrame());
        }
    }

    if (l)
        return 1;
    return 0;
}

uint ALandmarkInitializer::initFromMatch(feature_pair match) {
    std::shared_ptr<AFeature> f1, f2;
    f1 = match.first;
    f2 = match.second;

    // Don't if there is an outlier
    if (f1->isOutlier() || f2->isOutlier())
        return 0;

    // Does an attached landmark already exist for f1 or f2 ?
    std::shared_ptr<ALandmark> l1 = f1->getLandmark().lock();
    std::shared_ptr<ALandmark> l2 = f2->getLandmark().lock();

    // Case 1 : no landmark exist
    if (!l1 && !l2) {
        std::shared_ptr<ALandmark> l = createNewLandmark(f1, f2);
        if (!l)
            return 0;
        // A new landmark has been created at this frame, add it
        _initialized_landmarks[f1->getFeatureLabel()].push_back(l);
        l->setDescriptor(f1->getDescriptor());

        f2->getSensor()->getFrame()->addLandmark(l);
        // if f1 & f2 are not from the same frame, add landmark to frame 2 also
        if (f1->getSensor()->getFrame() != f2->getSensor()->getFrame())
            f2->getSensor()->getFrame()->addLandmark(l);
    }

    // Case 2 : feature 1 only is already associated to a ldmk
    else if (l1 && !l2) {

        // add the feature to the landmark
        l1->addFeature(f2);

        // update the descriptor
        l1->setDescriptor(f2->getDescriptor());

        _initialized_landmarks[f1->getFeatureLabel()].push_back(l1);

        // if f1 & f2 are not from the same frame, add landmark to frame 2 also
        if (f1->getSensor()->getFrame() != f2->getSensor()->getFrame())
            f2->getSensor()->getFrame()->addLandmark(l1);
    }

    // Case 3 : feature 2 only is already associated to a ldmk // POSSIBLE ???
    else if (!l1 && l2) {

        // add the feature to the landmark, reverse link is done by addFeature
        l2->addFeature(f1);

        // update the descriptor
        l2->setDescriptor(f1->getDescriptor());

        _initialized_landmarks[f1->getFeatureLabel()].push_back(l1);
        // if f1 & f2 are not from the same frame, add landmark to frame 2 also
        if (f1->getSensor()->getFrame() != f2->getSensor()->getFrame())
            f1->getSensor()->getFrame()->addLandmark(l2);
    }

    // Case 4 :: the two features already have a landmark need to fuse
    else {
        // fuse the landmarks
        if (!l1->fuseWithLandmark(l2))
            return 0;

        l1->setDescriptor(f2->getDescriptor());

        // if f1 & f2 are not from the same frame, add landmark to frame 2 also
        if (f1->getSensor()->getFrame() != f2->getSensor()->getFrame())
            f2->getSensor()->getFrame()->addLandmark(l1);
    }

    return 1;
}

std::shared_ptr<ALandmark> ALandmarkInitializer::createNewLandmark(std::shared_ptr<AFeature> f1,
                                                                   std::shared_ptr<AFeature> f2) {
    std::vector<std::shared_ptr<isae::AFeature>> features;
    features.push_back(f1);
    features.push_back(f2);

    std::shared_ptr<isae::ALandmark> landmark;

    if (f1->getSensor()->hasDepth()) {
        if (initLandmarkWithDepth(features, landmark))
            return landmark;
    } else {
        if (initLandmark(features, landmark))
            return landmark;
    }
    return nullptr;
}

} // namespace isae
