#include "isaeslam/data/maps/localmap.h"

namespace isae {

LocalMap::LocalMap(size_t min_kf_number, size_t max_kf_number, size_t fixed_frames_number)
    : _min_kf_number(min_kf_number), _max_kf_number(max_kf_number), _fixed_frames_number(fixed_frames_number) {}

void LocalMap::addFrame(std::shared_ptr<isae::Frame> &frame) {

    // A KF has been voted, the frame is added to the local map
    _localmap_mtx.lock();
    _frames.push_back(frame);
    _localmap_mtx.unlock();

    // Add landmarks to the map
    this->pushLandmarks(frame);

    // If we have too much frames, raise the marginalization flag
    if (_frames.size() > _max_kf_number) {
        _margin_flag = true;
    } else {
        _margin_flag = false;
    }
}

void LocalMap::discardLastFrame() {

    // Discard features from the marginalized frame
    _frames.at(0)->cleanLandmarks();
    _frames.at(0)->cleanSensors();

    _localmap_mtx.lock();
    _removed_frame_poses.push_back(_frames.at(0)->getFrame2WorldTransform());
    _frames.pop_front();
    _localmap_mtx.unlock();

    // remove landmarks in the map without any feature
    this->removeEmptyLandmarks();
}

void LocalMap::removeEmptyLandmarks() {
    // Remove map empty landmarks
    for (auto &tlmks : _landmarks) {
        for(std::vector<std::shared_ptr<isae::ALandmark>>::iterator it = tlmks.second.begin(); it!=tlmks.second.end();) {
            if (it->get()->getFeatures().empty()){ 
                it->get()->setMarg();
                
                _localmap_mtx.lock();
                it = tlmks.second.erase(it);
                 _localmap_mtx.unlock();
            } else {
                it++;
            }
        }
    }
}

void LocalMap::pushLandmarks(std::shared_ptr<isae::Frame> &frame) {
    typed_vec_landmarks all_ldmks = frame->getLandmarks();

    // For all type of landmarks to add
    for (auto &typed_ldmks : all_ldmks) {
        for (auto &ldmk : typed_ldmks.second) {
            if (!(!ldmk->isInitialized() || ldmk->isInMap() || ldmk->getFeatures().empty())) {
                ldmk->setInMap();

                _localmap_mtx.lock();
                _landmarks[ldmk->getLandmarkLabel()].push_back(ldmk);
                _localmap_mtx.unlock();
            }
        }
    }
}

} // namespace isae
