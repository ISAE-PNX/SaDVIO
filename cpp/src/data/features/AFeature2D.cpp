#include "isaeslam/data/features/AFeature2D.h"

namespace isae {

std::vector<Eigen::Vector3d> AFeature::getRays() {
    std::vector<Eigen::Vector3d> rays;

    for (Eigen::Vector2d pt : _poses2d) {
        rays.push_back(_sensor.lock()->getRay(pt));
    }
    return rays;
}

void AFeature::computeBearingVectors() { 

    for (Eigen::Vector2d pt : _poses2d) {
        _bearing_vectors.push_back(_sensor.lock()->getRayCamera(pt));
    }
}

} // namespace isae
