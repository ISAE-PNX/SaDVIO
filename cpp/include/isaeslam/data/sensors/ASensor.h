#ifndef ASENSOR_H
#define ASENSOR_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include "isaeslam/data/features/AFeature2D.h"
#include "isaeslam/data/frame.h"
#include "isaeslam/data/landmarks/Model3D.h"
#include "isaeslam/typedefs.h"
#include "utilities/imgProcessing.h"

#include <iostream>

namespace isae {

class Frame;
class AFeature;

/*!
* @brief Abstract struct of a sensor
*/
struct sensor_config {
    std::string sensor_type;
    std::string ros_topic;
    Eigen::Affine3d T_s_f;
};

/*!
* @brief Abstract class for all sensors
*
* This class provides a common interface for all sensors in the SLAM system.
* It contains a reference to a frame, its extrinsic and its type.
*/
class ASensor {
  public:
    ASensor(std::string type) { _type = type; }
    ~ASensor() {}

    std::string getType() { return _type; }

    // Associated frame
    void setFrame(std::shared_ptr<Frame> frame) {
        std::lock_guard<std::mutex> lock(_sensor_mtx);
        _frame = frame;
    }
    std::shared_ptr<Frame> getFrame() {
        std::lock_guard<std::mutex> lock(_sensor_mtx);
        return _frame.lock();
    }

    // Sensor pose in the frame coordinate
    void setFrame2SensorTransform(Eigen::Affine3d T_s_f) { _T_s_f = T_s_f; }
    Eigen::Affine3d getFrame2SensorTransform() { return _T_s_f; }

    // Sensor pose in the world coordinate
    Eigen::Affine3d getWorld2SensorTransform();
    Eigen::Affine3d getSensor2WorldTransform();

  protected:
    std::weak_ptr<Frame> _frame;
    Eigen::Affine3d _T_s_f;
    std::string _type;

    std::mutex _sensor_mtx;
};

/*!
* @brief Camera configuration
*
* This struct contains the intrinsic parameters of the camera, its projection model,
* its width and height, and parameters for distortion and fisheye models.
*/
struct cam_config : sensor_config {
    Eigen::Matrix3d K;
    std::string proj_model;
    int width;
    int height;

    // For distortion
    bool undistort;
    Eigen::Vector4d d;
    cv::Mat undist_map_x;
    cv::Mat undist_map_y;

    // For fisheye
    double rmax;

    // For ds
    double alpha;
    double xi;
};

/*!
* @brief Abstract class for image sensors
*
* This class inherits from ASensor and provides additional functionalities specific to image sensors.
* It includes methods for handling depth data, image pyramids, feature extraction, and projection.
* It has also all the methods to apply image processing techniques such as CLAHE, histogram equalization etc...
*/
class ImageSensor : public ASensor, public std::enable_shared_from_this<ImageSensor> {

  public:
    ImageSensor() : ASensor("image") {}
    ~ImageSensor() {}

    // For RGBD sensors TODO
    bool hasDepth() { return _has_depth; }

    /*!
    * @brief Get the ray in camera coordinates
    */
    virtual Eigen::Vector3d getRayCamera(Eigen::Vector2d f) = 0;

    /*!
    * @brief Get the ray in world coordinates
    */
    virtual Eigen::Vector3d getRay(Eigen::Vector2d f)       = 0;

    /*!
    * @brief Compute the focal length of the camera
    *
    * Virtual function because it depends on the camera model.
    */
    virtual double getFocal()                               = 0;

    cv::Mat getRawData() { return _raw_data; }

    void setPyr(const std::vector<cv::Mat> &img_pyr) { _img_pyr = img_pyr; }
    const std::vector<cv::Mat> getPyr() { return _img_pyr; }

    /*!
    * @brief Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the image
    */
    void applyCLAHE(float clahe_clip);

    /*!
    * @brief Apply histogram equalization to the image
    */
    void histogramEqualization();

    /*!
    * @brief Apply Image normalization to the image
    */
    void imageNormalization();

    /*!
    * @brief Apply adaptive gamme correction
    */
    void applyAGCWD(float alpha);

    void setMask(cv::Mat mask) { _mask = mask; }
    cv::Mat getMask() { return _mask; }
    Eigen::Matrix3d getCalibration() { return _calibration; }

    /*!
    * @brief Add a single feature and compute its bearing vector
    */
    void addFeature(std::string feature_label, std::shared_ptr<AFeature> f);

    /*!
    * @brief Add a vector of features and compute their bearing vectors
    */
    void addFeatures(std::string feature_label, std::vector<std::shared_ptr<AFeature>> features);

    void removeFeature(std::shared_ptr<AFeature> f);

    typed_vec_features &getFeatures() {
        std::lock_guard<std::mutex> lock(_cam_mtx);
        return _features;
    }


    uint countInitFeatures();

    /*!
    * @brief Clear all features of a specific type
    */
    void purgeFeatures(std::string feature_label) { _features[feature_label].clear(); }

    /*!
    * @brief Get features of a specific type
    */
    std::vector<std::shared_ptr<AFeature>> &getFeatures(std::string feature_label);

    /*!
    * @brief Virtual function to project a landmark in the image plane
    */
    virtual bool project(const Eigen::Affine3d &T_w_lmk,
                         const std::shared_ptr<AModel3d> ldmk_model,
                         std::vector<Eigen::Vector2d> &p2d_vector) = 0;

    /*!
    * @brief Virtual function to project a landmark in the image plane with the pose of the frame
    */
    virtual bool project(const Eigen::Affine3d &T_w_lmk,
                         const std::shared_ptr<AModel3d> ldmk_model,
                         const Eigen::Affine3d &T_f_w,
                         std::vector<Eigen::Vector2d> &p2d_vector) = 0;
    
    /*!
    * @brief Virtual function to project a landmark and compute the Jacobian of the projection
    */
    virtual bool project(const Eigen::Affine3d &T_w_lmk,
                         const Eigen::Affine3d &T_f_w,
                         const Eigen::Matrix2d sqrt_info,
                         Eigen::Vector2d &p2d,
                         double *J_proj_frame,
                         double *J_proj_lmk)                       = 0;

  protected:
    Eigen::Matrix3d _calibration;  //!< intrinsic matrix of the camera (sensor ?)
    cv::Mat _raw_data;             //!< Raw image data
    std::vector<cv::Mat> _img_pyr; //!< Image pyramid for multi-scale processing
    cv::Mat _mask;                 //!< Mask to ignore
    typed_vec_features _features;  //!< Typed vector of features
    bool _has_depth;               //!< Is it a RGBD ?

    std::mutex _cam_mtx;
};

} // namespace isae

#endif
