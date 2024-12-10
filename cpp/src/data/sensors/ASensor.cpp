#include "isaeslam/data/sensors/ASensor.h"

namespace isae {

Eigen::Affine3d ASensor::getWorld2SensorTransform() { return _T_s_f * this->_frame.lock()->getWorld2FrameTransform(); }

Eigen::Affine3d ASensor::getSensor2WorldTransform() { return this->getWorld2SensorTransform().inverse(); }

void ImageSensor::applyCLAHE(float clahe_clip) { imgproc::histogramEqualizationCLAHE(_raw_data, clahe_clip); }

void ImageSensor::histogramEqualization() { cv::equalizeHist(_raw_data, _raw_data); }

void ImageSensor::imageNormalization() { cv::normalize(_raw_data, _raw_data, 0, 255, cv::NORM_MINMAX); }

void ImageSensor::applyAGCWD(float alpha) { imgproc::AGCWD(_raw_data, _raw_data, alpha); }

Eigen::Matrix3d ImageSensor::getCalibration() { return _calibration; }

void ImageSensor::addFeature(std::string feature_label, std::shared_ptr<AFeature> f) {
    f->setSensor(shared_from_this());
    f->computeBearingVectors();

    std::lock_guard<std::mutex> lock(_cam_mtx);
    _features[feature_label].push_back(f);
}

void ImageSensor::addFeatures(std::string feature_label, std::vector<std::shared_ptr<AFeature>> features) {
    for (auto f : features) {
        this->addFeature(feature_label, f);
    }
}

void ImageSensor::removeFeature(std::shared_ptr<AFeature> f) {
    std::lock_guard<std::mutex> lock(_cam_mtx);

    for (int i = _features[f->getFeatureLabel()].size() - 1; i >= 0; i--) {
        if (_features[f->getFeatureLabel()].at(i) == f) {
            _features[f->getFeatureLabel()].erase(_features[f->getFeatureLabel()].begin() + i);
        }
    }
}

std::vector<std::shared_ptr<AFeature>> &ImageSensor::getFeatures(std::string feature_label) {
    std::lock_guard<std::mutex> lock(_sensor_mtx);
    return _features[feature_label];
}

} // namespace isae
