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

void LocalMap::removeFrame(std::shared_ptr<isae::Frame> &frame) {
    // Remove the frame from the local map
    _localmap_mtx.lock();
    for (auto it = _frames.begin(); it != _frames.end(); ++it) {
        if (*it == frame) {
            it->get()->cleanLandmarks();
            it->get()->cleanSensors();
            _frames.erase(it);
            break;
        }
    }
    _localmap_mtx.unlock();
    _margin_flag = false;
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
    for (auto &frame : _frames) {
        frame->cleanLandmarks();
        frame->cleanSensors();
    }
    _frames.clear();
    for (auto &tlmks : _landmarks) {
        tlmks.second.clear();
    }
    _localmap_mtx.unlock();
}

bool LocalMap::computeRelativePose(std::shared_ptr<isae::Frame> &frame1,
                                   std::shared_ptr<isae::Frame> &frame2,
                                   Eigen::Affine3d &T_f1_f2,
                                   Eigen::MatrixXd &cov) {
    // Check if the local map is not empty
    if (_frames.size() < 2) {
        std::cout << "Local map is empty! No relative pose. " << "(" << _frames.size() << ")" << std::endl;
        return false;
    }

    // Compute the relative pose between two frames
    T_f1_f2 = frame1->getFrame2WorldTransform().inverse() * frame2->getFrame2WorldTransform();

    // Select all the frames included between the two frames
    std::vector<std::shared_ptr<isae::Frame>> frames_to_add;
    for (auto &frame : _frames) {
        if (frame->getTimestamp() >= frame1->getTimestamp() && frame->getTimestamp() <= frame2->getTimestamp()) {
            frames_to_add.push_back(frame);
        }
    }
    std::cout << "Relative frame dt [s]: " << (frame2->getTimestamp() - frame1->getTimestamp())*1e-9 << std::endl;

    // If we haven't found at least 2 KF, return false
    if (frames_to_add.size() < 2) {
        std::cout << "Local map has insufficient KeyFrames! No relative pose. " << "(" << frames_to_add.size() << ")" << std::endl;
        return false;
    }

    // Propagate the covariance
    cov = frames_to_add.at(1)->getdTCov();
    Eigen::Affine3d T_f1_fim1 =
        frame1->getFrame2WorldTransform().inverse() * frames_to_add.at(1)->getFrame2WorldTransform();

    for (uint i = 2; i < frames_to_add.size(); i++) {
        Eigen::Affine3d T_fim1_fi = frames_to_add.at(i - 1)->getFrame2WorldTransform().inverse() *
                                    frames_to_add.at(i)->getFrame2WorldTransform();

        // Jacobian formulas come from "A micro Lie Theory for state estimation in robotics" by Solà et al.
        Eigen::MatrixXd J_f1   = Eigen::MatrixXd::Identity(6, 6);
        J_f1.block<3, 3>(0, 0) = T_fim1_fi.rotation().transpose();
        J_f1.block<3, 3>(0, 3) = -T_fim1_fi.rotation() * geometry::skewMatrix(T_fim1_fi.translation());
        J_f1.block<3, 3>(3, 3) = T_fim1_fi.rotation().transpose();

        Eigen::MatrixXd J_dt = Eigen::MatrixXd::Identity(6, 6);

        T_f1_fim1 = frame1->getFrame2WorldTransform().inverse() * frames_to_add.at(i)->getFrame2WorldTransform();
        if (cov.rows() != 6 && cov.cols() != 6) {
            cov = Eigen::MatrixXd::Identity(6, 6);
            std::cout << "Covariance rows and cols do not match: " << cov.rows() << " rows, " << cov.cols() << " cols." << std::endl;
            return false;
        }
        cov = J_f1 * cov * J_f1.transpose() + J_dt * frames_to_add.at(i)->getdTCov() * J_dt.transpose();
    
        bool cov_err = false;
        // debug (check for identity matrix)
        if (abs(cov.matrix().trace() - 6.) < 0.00001)
        {
            std::cout << "Covariance is identity matrix." << std::endl;
            cov_err = true;
        }
        // debug (check for NaN values)
        if (cov.matrix().hasNaN())
        {
            std::cout << "Covariance contains NaN values." << std::endl;
            cov_err = true;
        }    
        if (cov_err)
        {            
            std::cout << cov.matrix() << std::endl;
            std::cout << frames_to_add.at(i)->getdTCov().matrix() << std::endl;
            std::cout << J_f1.matrix() << std::endl;
            std::cout << J_dt.matrix() << std::endl;
        }
    }

    if (cov.rows() != 6 && cov.cols() != 6) {
        cov = Eigen::MatrixXd::Identity(6, 6);
        std::cout << "Covariance rows and cols do not match: " << cov.rows() << " rows, " << cov.cols() << " cols." << std::endl;
        return false;
    }

    return true;
}

} // namespace isae
