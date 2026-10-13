#ifndef AFEATURE_H
#define AFEATURE_H

#include "isaeslam/data/landmarks/ALandmark.h"
#include "isaeslam/data/sensors/ASensor.h"
#include "utilities/geometry.h"

namespace isae {

class ImageSensor;
class ALandmark;

/*!
 * @brief Abstract class for 2D features.
 *
 * Its main member is a vector of 2D points representing the feature in the image.
 * It can be used to represent points, lines, bounding boxes, etc.
 */
class AFeature : public std::enable_shared_from_this<AFeature> {
  public:
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

    void setVelocity(std::vector<Eigen::Vector3d> velocity) {
        // std::lock_guard<std::mutex> lock(_feat_mtx);
        _velocity = velocity;
    }

    std::vector<Eigen::Vector3d> getVelocity() {
        std::lock_guard<std::mutex> lock(_feat_mtx);
        return _velocity;
    }

    /**
     * @brief Compute the bearing vectors from the 2D points, usually inverting the camera projection model
     */
    void computeBearingVectors();

    std::vector<Eigen::Vector3d> getBearingVectors() {
        std::lock_guard<std::mutex> lock(_feat_mtx);
        return _bearing_vectors;
    }
    std::vector<Eigen::Vector3d> getRays();

    /**
     * @brief Set the 2D feature as an outlier
     */
    void setOutlier() {
        std::lock_guard<std::mutex> lock(_feat_mtx);
        _outlier = true;
    }

    /**
     *  @brief True if the feature is an outlier
     */
    bool isOutlier() {
        std::lock_guard<std::mutex> lock(_feat_mtx);
        return _outlier;
    }

  protected:
    std::string _feature_label;                    //!< Feature label, e.g. point, line, bouding box, etc.
    std::vector<Eigen::Vector2d> _poses2d;         //!< 2D coordinates of the points of the feature in the image
    std::vector<Eigen::Vector3d> _bearing_vectors; //!< Bearing vectors corresponding to the 2D points
    std::vector<Eigen::Vector3d> _velocity;        //!< Velocity of the 2D points 
    cv::Mat _descriptor;                           //!< Descriptor of the feature, e.g. SIFT, ORB, etc.
    int _octave;                                   //!< Octave of the feature, used in multi-scale feature detection
    float _response;                               //!< Response of the feature, i.e. strength of the feature
    double _sigma;                                 //!< Noise level of the feature
    std::weak_ptr<ImageSensor> _sensor;            //!< Link to the parent sensor
    std::weak_ptr<ALandmark> _landmark;            //!< Link to the landmark associated with the feature
    bool _outlier = false;                         //!< Flag to indicate if the feature is an outlier

    mutable std::mutex _feat_mtx;
};

} // namespace isae
#endif
