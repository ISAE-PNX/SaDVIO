#include "isaeslam/data/maps/globalmap.h"

namespace isae {

void GlobalMap::addFrame(std::shared_ptr<isae::Frame> &frame) {
    // A KF has been voted, the frame is added to the local map
    _frames.push_back(frame);

    // Add landmarks to the map
    this->pushLandmarks(frame);
}

void GlobalMap::pushLandmarks(std::shared_ptr<isae::Frame> &frame) {
    typed_vec_landmarks all_ldmks = frame->getLandmarks();

    // For all type of landmarks to add
    for (auto &typed_ldmks : all_ldmks) {
        for (auto &ldmk : typed_ldmks.second) {
            if (!(!ldmk->isInitialized() || ldmk->getFeatures().empty())) {
                ldmk->setInMap();
                _landmarks[ldmk->getLandmarkLabel()].push_back(ldmk);
            }
        }
    }
}

} // namespace isae