#ifndef ALANDMARK_H
#define ALANDMARK_H

#include "isaeslam/data/landmarks/Model3D.h"
#include "utilities/geometry.h"

namespace isae {

class AFeature;

class ALandmark : public std::enable_shared_from_this<ALandmark> {
  public:
    ALandmark() {}
    ALandmark(Eigen::Affine3d T_w_l, std::vector<std::shared_ptr<isae::AFeature>> features);
    ~ALandmark() {}

    // Landmark id for bookeeping
    static int _landmark_count;
    int _id;

    virtual void init(Eigen::Affine3d T_w_l, std::vector<std::shared_ptr<isae::AFeature>> features);

    std::string getLandmarkLabel() const { return _label; }

    void addFeature(std::shared_ptr<AFeature> feature);
    std::vector<std::weak_ptr<AFeature>> getFeatures() {
        removeExpiredFeatures();
        return _features;
    }

    void removeExpiredFeatures();
    void removeFeature(std::shared_ptr<AFeature> f);
    bool fuseWithLandmark(std::shared_ptr<ALandmark> landmark);

    void setPose(Eigen::Affine3d T_w_l) {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        _T_w_l = T_w_l;
    }
    Eigen::Affine3d getPose() const {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        return _T_w_l;
    }

    std::vector<Eigen::Vector3d> getModelPoints() { return _model->getModel(); }
    std::shared_ptr<AModel3d> getModel() const { return _model; }

    void setScale(Eigen::Vector3d scale) { _scale = scale; }
    Eigen::Vector3d getScale() const { return _scale; }

    cv::Mat getDescriptor() const {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        return _descriptor;
    }
    void setDescriptor(cv::Mat descriptor) {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        _descriptor = descriptor;
    }

    bool isInitialized() const {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        return _initialized;
    }
    void setUninitialized() {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        _initialized = false;
    }

    void setInMap() {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        _in_map = true;
    }
    bool isInMap() {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        return _in_map;
    }

    bool sanityCheck();
    virtual double chi2err(std::shared_ptr<AFeature> f);
    double avgChi2err();

    bool isOutlier() const {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        return _outlier;
    }
    void setOutlier() {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        _outlier = true;
    }
    void setInlier() {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        _outlier = false;
    }

    bool isResurected() const {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        return _is_resurected;
    }
    void setResurected() {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        _is_resurected = true;
    }

    bool hasPrior() const {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        return _has_prior;
    }
    void setPrior() {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        _has_prior = true;
    }

    bool isMarg() const {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        return _is_marg;
    }
    void setMarg() {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        _is_marg = true;
    }

  protected:
    bool _initialized   = false;
    bool _in_map        = false;
    bool _outlier       = false;
    bool _is_resurected = false;
    bool _has_prior     = false;
    bool _is_marg       = false;

    std::string _label;
    Eigen::Affine3d _T_w_l;
    cv::Mat _descriptor;

    Eigen::Vector3d _scale = Eigen::Vector3d(1, 1, 1);
    std::shared_ptr<AModel3d> _model;

    std::vector<std::weak_ptr<AFeature>> _features;

    mutable std::mutex _lmk_mtx;
};

} // namespace isae

#endif
