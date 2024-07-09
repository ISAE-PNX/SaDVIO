#ifndef POINT3D_H
#define POINT3D_H

#include "isaeslam/data/features/AFeature2D.h"
#include "isaeslam/data/landmarks/ALandmark.h"

namespace isae {

class Point3D : public ALandmark {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Point3D() {
        _label = "pointxd";
        _model = std::make_shared<ModelPoint3D>();
    }
    Point3D(std::vector<Eigen::Vector3d> T_w_l_vector, cv::Mat desc = cv::Mat()) : ALandmark() {
        _label = "pointxd";
        _model = std::make_shared<ModelPoint3D>();
    }

    Point3D(Eigen::Affine3d T_w_l, std::vector<std::shared_ptr<isae::AFeature>> features)
        : ALandmark(T_w_l, features) {
        _label = "pointxd";
        _model = std::make_shared<ModelPoint3D>();
    }

};

} // namespace isae

#endif // POINT3D_H
