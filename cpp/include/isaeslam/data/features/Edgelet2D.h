#ifndef EDGELET2D_H
#define EDGELET2D_H

#include "isaeslam/data/features/AFeature2D.h"

namespace isae {

    class Edgelet2D : public AFeature {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Edgelet2D(){
            _feature_label = "edgeletxd";
        }
        Edgelet2D(std::vector<Eigen::Vector2d> poses2d, cv::Mat desc = cv::Mat()):AFeature(poses2d, desc){
            _feature_label = "edgeletxd";
        }
    };

} // namespace isae

#endif //EDGELET2D_H
