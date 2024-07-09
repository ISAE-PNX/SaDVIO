#ifndef BBOX3D_H
#define BBOX3D_H

#include "isaeslam/data/features/AFeature2D.h"
#include "isaeslam/data/landmarks/ALandmark.h"

namespace isae {

class BBox3D : public ALandmark {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    BBox3D() {
        _label = "bboxxd";
        _model = std::make_shared<ModelBBox3D>();
    }
    BBox3D(std::vector<Eigen::Vector3d> T_w_l_vector, cv::Mat desc = cv::Mat()) : ALandmark() {
        _label = "bboxxd";
        _model = std::make_shared<ModelBBox3D>();
    }

    BBox3D(Eigen::Affine3d T_w_l, std::vector<std::shared_ptr<isae::AFeature>> features) : ALandmark(T_w_l, features) {
        _label = "bboxxd";
        _model = std::make_shared<ModelBBox3D>();
    }

    BBox3D(Eigen::Affine3d T_w_l, Eigen::Vector3d scale, std::vector<std::shared_ptr<isae::AFeature>> features)
        : ALandmark(T_w_l, features) {
        _label = "bboxxd";
        _model = std::make_shared<ModelBBox3D>();
        _scale = scale;
    }
};

} // namespace isae

#endif // BBOX3D_H
