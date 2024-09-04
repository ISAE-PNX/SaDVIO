#include "isaeslam/data/frame.h"
#include "isaeslam/data/sensors/ASensor.h"
#include "isaeslam/data/sensors/IMU.h"

namespace isae {

int Frame::_frame_count = 0;

void Frame::init(const std::vector<std::shared_ptr<isae::ImageSensor>> &sensors, unsigned long long timestamp) {
    _timestamp = timestamp;
    _sensors   = sensors;
    _T_f_w     = Eigen::Affine3d::Identity();

    // increases everytime an object is created
    _frame_count++;
    _id = _frame_count;

    // Set frame to sensors
    for (auto &sensor : sensors) {
        sensor->setFrame(this->shared_from_this());
    }
}

void Frame::init(std::shared_ptr<IMU> &imu, unsigned long long timestamp) {
    _timestamp = timestamp;
    _T_f_w     = Eigen::Affine3d::Identity();

    // increases everytime an object is created
    _frame_count++;
    _id = _frame_count;

    _imu       = imu;
    _imu->setFrame(this->shared_from_this());
}

void Frame::init(const std::vector<std::shared_ptr<ASensor>> &sensors, unsigned long long timestamp) {
    _timestamp = timestamp;
    _T_f_w     = Eigen::Affine3d::Identity();

    // increases everytime an object is created
    _frame_count++;
    _id = _frame_count;

    for (auto sensor : sensors) {
        if (sensor->getType() == "imu") {
            _imu = std::static_pointer_cast<IMU>(sensor);
        }
        if (sensor->getType() == "image") {
            _sensors.push_back(std::static_pointer_cast<ImageSensor>(sensor));
        }
        sensor->setFrame(this->shared_from_this());
    }
}

void Frame::setIMU(std::shared_ptr<IMU> &imu, Eigen::Affine3d T_s_f) {
    _imu = imu;
    _imu->setFrame(this->shared_from_this());
    _imu->setFrame2SensorTransform(T_s_f);
}

uint Frame::getInMapLandmarksNumber() const {
    uint in_map = 0;
    std::vector<std::shared_ptr<ALandmark>> ldmks;
    for (auto const &sensor : _sensors) {
        for (auto const &typed_feat : sensor->getFeatures()) {
            for (auto const &f : typed_feat.second) {
                if (f->getLandmark().lock()) {
                    if (std::find(ldmks.begin(), ldmks.end(), f->getLandmark().lock()) == ldmks.end()) {
                        if (f->getLandmark().lock()->isInitialized() && !f->getLandmark().lock()->isOutlier() &&
                            f->getLandmark().lock()->isInMap()) {
                            in_map++;
                            ldmks.push_back(f->getLandmark().lock());
                        }
                    }
                }
            }
        }
    }
    return in_map;
}

void Frame::cleanLandmarks() {

    // We have to unlink the deleted features and the remaining landmarks

    // For all sensors
    for (auto &sensor : _sensors) {
        // For all type of features
        for (auto &typed_fs : sensor->getFeatures()) {
            // For each feature
            for (auto &f : typed_fs.second) {
                // if the feature is linked to a ldmk, unlink and clean
                if (f->getLandmark().lock()) {
                    f->getLandmark().lock()->removeFeature(f);
                    f->getLandmark().lock()->removeExpiredFeatures();
                }
            }
        }
    }
}

} // namespace isae
