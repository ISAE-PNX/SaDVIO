#include "isaeslam/data/maps/localmap.h"
#include <iostream>

namespace isae {

LocalMap::LocalMap(size_t min_kf_number, size_t max_kf_number, size_t fixed_frames_number)
    : _min_kf_number(min_kf_number), _max_kf_number(max_kf_number), _fixed_frames_number(fixed_frames_number) {}

void LocalMap::addFrame(std::shared_ptr<isae::Frame> &frame) {

    // A KF has been voted, the frame is added to the local map
    // The frames are ordered from the oldest to the newest
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
    _margin_flag = false;

    // // If we have not enough landmarks in common with the last frame, raise the marginalization flag
    // int lmk_counter = 0;
    // for (auto &feat_last : _frames.front()->getSensors().at(0)->getFeatures()["pointxd"]) {
    //     if (!feat_last->getLandmark().lock())
    //         continue;
    //     for (auto &feat_curr : _frames.back()->getSensors().at(0)->getFeatures()["pointxd"]) {
    //         if (!feat_curr->getLandmark().lock())
    //             continue;
    //         if (feat_last->getLandmark().lock()->_id == feat_curr->getLandmark().lock()->_id) {
    //             lmk_counter++;
    //         }
    //     }

    //     if (lmk_counter > 5) {
    //         break;
    //     }
    // }

    // if (lmk_counter < 5) {
    //     _margin_flag = true;
    // } else {
    //     _margin_flag = false;
    // }
}

void LocalMap::removeEmptyLandmarks() {
    // Remove map empty landmarks
    for (auto &tlmks : _landmarks) {
        for (std::vector<std::shared_ptr<isae::ALandmark>>::iterator it = tlmks.second.begin();
             it != tlmks.second.end();) {
            if (it->get()->getFeatures().empty()) {
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

void LocalMap::reset() {
    _localmap_mtx.lock();
    _frames.clear();
    _landmarks.clear();
    _localmap_mtx.unlock();
}

bool LocalMap::computeRelativePose(std::shared_ptr<isae::Frame> &frame1,
                                   std::shared_ptr<isae::Frame> &frame2,
                                   Eigen::Affine3d &T_f1_f2,
                                   Eigen::MatrixXd &cov) {
    // Compute the relative pose between two frames
    T_f1_f2 = frame1->getFrame2WorldTransform().inverse() * frame2->getFrame2WorldTransform();

    // Select all the frames included between the two frames
    std::vector<std::shared_ptr<isae::Frame>> frames_to_add;
    for (auto &frame : _frames) {
        if (frame->getTimestamp() >= frame1->getTimestamp() && frame->getTimestamp() <= frame2->getTimestamp()) {
            frames_to_add.push_back(frame);
        }
    }

    // If we haven't found at least 2 KF, return false 
    if (frames_to_add.size() < 2) {
        return false;
    }

    // Propagate the covariance
    cov = frames_to_add.at(1)->getdTCov();
    Eigen::Affine3d T_f1_fi =
        frame1->getFrame2WorldTransform().inverse() * frames_to_add.at(1)->getFrame2WorldTransform();

    for (uint i = 2; i < frames_to_add.size(); i++) {

        Eigen::Affine3d T_fim1_fi = frames_to_add.at(i - 1)->getFrame2WorldTransform().inverse() *
                                    frames_to_add.at(i)->getFrame2WorldTransform();

        Eigen::MatrixXd J_f1   = Eigen::MatrixXd::Identity(6, 6);
        J_f1.block<3, 3>(0, 0) = T_fim1_fi.rotation().transpose();
        J_f1.block<3, 3>(3, 0) = T_f1_fi.rotation();

        Eigen::MatrixXd J_dt   = Eigen::MatrixXd::Identity(6, 6);
        J_dt.block<3, 3>(3, 3) = T_f1_fi.rotation().transpose();

        T_f1_fi = frame1->getFrame2WorldTransform().inverse() * frames_to_add.at(i)->getFrame2WorldTransform();
        cov     = J_f1 * cov * J_f1.transpose() + J_dt * frames_to_add.at(i)->getdTCov() * J_dt.transpose();
    }
    
    return true;
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
