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

namespace isae {

class Frame;
class AFeature;

struct sensor_config {
    std::string sensor_type;
    std::string ros_topic;
    Eigen::Affine3d T_s_f;
};

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

class ImageSensor : public ASensor, public std::enable_shared_from_this<ImageSensor> {

  public:
    ImageSensor() : ASensor("image") {}
    ~ImageSensor() {}

    // For RGBD sensors
    bool hasDepth() { return _has_depth; }
    virtual const cv::Mat &getDepthMat()                                                     = 0;
    virtual std::vector<Eigen::Vector3d> getP3Dcam(const std::shared_ptr<AFeature> &feature) = 0;

    // To do: unify these functions
    virtual Eigen::Vector3d getRayCamera(Eigen::Vector2d f) = 0;
    virtual Eigen::Vector3d getRay(Eigen::Vector2d f)       = 0;
    virtual double getFocal()                               = 0;

    // return sensor rawData
    cv::Mat getRawData() { return _raw_data; }

    // Get and Set img pyramide
    void setPyr(const std::vector<cv::Mat> &img_pyr) { _img_pyr = img_pyr; }
    const std::vector<cv::Mat> getPyr() { return _img_pyr; }

    // apply CLAHE
    void applyCLAHE(float clahe_clip);

    // Histo equalization
    void histogramEqualization();

    // image normalization
    void imageNormalization();

    // adaptive gamma correction
    void applyAGCWD(float alpha);

    // return sensor mask for feature extraction
    void setMask(cv::Mat mask) { _mask = mask; }
    cv::Mat getMask() { return _mask; }

    // return calibration
    Eigen::Matrix3d getCalibration();

    // add features of the given type to the camera
    void addFeature(std::string feature_label, std::shared_ptr<AFeature> f);
    void addFeatures(std::string feature_label, std::vector<std::shared_ptr<AFeature>> features);
    void removeFeature(std::shared_ptr<AFeature> f);

    // return all detected features
    typed_vec_features &getFeatures() {
        std::lock_guard<std::mutex> lock(_cam_mtx);
        return _features;
    }
    void purgeFeatures(std::string feature_label) { _features[feature_label].clear(); }

    // return desired type of detected features
    std::vector<std::shared_ptr<AFeature>> &getFeatures(std::string feature_label);

    // Projection function with or without jacobian computation
    virtual bool project(const Eigen::Affine3d &T_w_lmk,
                         const std::shared_ptr<AModel3d> ldmk_model,
                         const Eigen::Vector3d &scale,
                         std::vector<Eigen::Vector2d> &p2d_vector) = 0;
    virtual bool project(const Eigen::Affine3d &T_w_lmk,
                         const std::shared_ptr<AModel3d> ldmk_model,
                         const Eigen::Vector3d &scale,
                         const Eigen::Affine3d &T_f_w,
                         std::vector<Eigen::Vector2d> &p2d_vector) = 0;
    virtual bool project(const Eigen::Affine3d &T_w_lmk,
                         const Eigen::Affine3d &T_f_w,
                         const Eigen::Matrix2d sqrt_info,
                         Eigen::Vector2d &p2d,
                         double *J_proj_frame,
                         double *J_proj_lmk)                       = 0;

  protected:
    Eigen::Matrix3d _calibration;  // intrinsic matrix of the camera (sensor ?)
    cv::Mat _raw_data;             // Raw img data
    std::vector<cv::Mat> _img_pyr; // Img pyramid for KLT tracking
    cv::Mat _mask;                 // mask to remove
    typed_vec_features _features;  // Vector of features
    bool _has_depth;               // Is it a RGBD ?

    std::mutex _cam_mtx;
};

} // namespace isae

#endif
