#ifndef AFEATURE_H
#define AFEATURE_H

#include "isaeslam/data/landmarks/ALandmark.h"
#include "isaeslam/data/sensors/ASensor.h"
#include "utilities/geometry.h"

namespace isae {

class ImageSensor;
class ALandmark;

class AFeature : public std::enable_shared_from_this<AFeature> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    AFeature() {}
    AFeature(std::vector<Eigen::Vector2d> poses2d, cv::Mat desc = cv::Mat(), int octave = 0, double sigma = 1.0)
        : _poses2d(poses2d), _descriptor(desc), _octave(octave), _sigma(sigma) {}

    const std::vector<Eigen::Vector2d> getPoints() const {
        std::lock_guard<std::mutex> lock(_feat_mtx);
        return _poses2d;
    }
    void setPoints(std::vector<Eigen::Vector2d> poses2d) {
        std::lock_guard<std::mutex> lock(_feat_mtx);
        _poses2d = poses2d;
    }

    const cv::Mat &getDescriptor() const {
        std::lock_guard<std::mutex> lock(_feat_mtx);
        return _descriptor;
    }
    void setDescriptor(cv::Mat descriptor) {
        std::lock_guard<std::mutex> lock(_feat_mtx);
        _descriptor = descriptor;
    }

    const int &getOctave() const {
        std::lock_guard<std::mutex> lock(_feat_mtx);
        return _octave;
    }
    void setOctave(int octave) {
        std::lock_guard<std::mutex> lock(_feat_mtx);
        _octave = octave;
    }

    const float getResponse() {
        std::lock_guard<std::mutex> lock(_feat_mtx);
        return _response;
    }

    const double getSigma() {
        std::lock_guard<std::mutex> lock(_feat_mtx);
        return _sigma;
    }

    const std::string getFeatureLabel() const {
        std::lock_guard<std::mutex> lock(_feat_mtx);
        return _feature_label;
    }

    std::weak_ptr<ALandmark> getLandmark() {
        std::lock_guard<std::mutex> lock(_feat_mtx);
        return _landmark;
    }
    void linkLandmark(std::shared_ptr<ALandmark> landmark) {
        std::lock_guard<std::mutex> lock(_feat_mtx);
        _landmark = landmark;
    }
    void unlinkLandmark() {
        std::lock_guard<std::mutex> lock(_feat_mtx);
        _landmark.reset();
    }

    void setSensor(std::shared_ptr<ImageSensor> sensor) {
        std::lock_guard<std::mutex> lock(_feat_mtx);
        _sensor = sensor;
    }
    std::shared_ptr<ImageSensor> getSensor() {
        std::lock_guard<std::mutex> lock(_feat_mtx);
        return _sensor.lock();
    }

    void computeBearingVectors();
    std::vector<Eigen::Vector3d> getBearingVectors() {
        std::lock_guard<std::mutex> lock(_feat_mtx);
        return _bearing_vectors;
    }
    std::vector<Eigen::Vector3d> getRays();

    void setOutlier() {
        std::lock_guard<std::mutex> lock(_feat_mtx);
        _outlier = true;
    }
    bool isOutlier() {
        std::lock_guard<std::mutex> lock(_feat_mtx);
        return _outlier;
    }

  protected:
    std::string _feature_label;
    std::vector<Eigen::Vector2d> _poses2d;
    std::vector<Eigen::Vector3d> _bearing_vectors;
    cv::Mat _descriptor;
    int _octave;
    float _response;
    double _sigma;
    std::weak_ptr<ImageSensor> _sensor; //< link to parent sensor
    std::weak_ptr<ALandmark> _landmark; //< link to landmark
    bool _outlier = false;

    mutable std::mutex _feat_mtx;
};

} // namespace isae
#endif
