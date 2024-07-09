#ifndef LINE2D_H
#define LINE2D_H

#include "isaeslam/data/features/AFeature2D.h"

namespace isae {

    class Line2D : public AFeature {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Line2D(){
            _feature_label = "linexd";
        }
        Line2D(std::vector<Eigen::Vector2d> poses2d, cv::Mat desc = cv::Mat()):AFeature(poses2d, desc){
            _feature_label = "linexd";
        }
    };

} // namespace isae

#endif //LINE2D_H
