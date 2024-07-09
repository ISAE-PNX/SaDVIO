#ifndef EDGELET3D_H
#define EDGELET3D_H

#include "isaeslam/data/landmarks/ALandmark.h"

namespace isae {

class Edgelet3D : public ALandmark {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Edgelet3D() {
        _label            = "edgeletxd";
        _model            = std::make_shared<ModelEdgelet3D>();
        _scale *= 100;
    }

    Edgelet3D(std::vector<Eigen::Vector3d> T_w_l_vector, cv::Mat desc = cv::Mat()) : ALandmark() {
        _label            = "edgeletxd";
        _model            = std::make_shared<ModelEdgelet3D>();
        _scale *= 100;
    }

    Edgelet3D(Eigen::Affine3d T_w_l, std::vector<std::shared_ptr<isae::AFeature>> features)
        : ALandmark(T_w_l, features) {
        _label            = "edgeletxd";
        _model            = std::make_shared<ModelEdgelet3D>();
        _scale *= 100;
    }

};

} // namespace isae

#endif // EDGELET3D_H
