#ifndef ELLIPSEPATEERN2D_H
#define ELLIPSEPATEERN2D_H

#include "isaeslam/data/features/AFeature2D.h"

namespace isae {

    class EllipsePattern2D : public AFeature {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        EllipsePattern2D(){
            _feature_label = "ellipsepatternxd";
        }
        EllipsePattern2D(std::vector<Eigen::Vector2d> poses2d, cv::Mat desc = cv::Mat()):AFeature(poses2d, desc){
            _feature_label = "ellipsepatternxd";
        }
    };

} // namespace isae

#endif //ELLIPSEPATEERN2D_H
