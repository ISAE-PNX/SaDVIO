#include "isaeslam/data/sensors/RGBDCamera.h"

namespace isae {

Eigen::Vector3d RGBDCamera::getRay(Eigen::Vector2d f) {
    Eigen::Vector3d ray_world;
    Eigen::Vector3d ray_cam;
    ray_cam.segment<2>(0) = (f - _calibration.block<2, 1>(0, 2));
    ray_cam[0] /= _calibration(0, 0);
    ray_cam[1] /= _calibration(1, 1);
    ray_cam[2] = 1;
    ray_world = this->getSensor2WorldTransform().rotation() * ray_cam;
    ray_world.normalize();
    return ray_world;
}

Eigen::Vector3d RGBDCamera::getRayCamera(Eigen::Vector2d f) {
    Eigen::Vector3d ray_cam;
    ray_cam.segment<2>(0) = (f - _calibration.block<2, 1>(0, 2));
    ray_cam[0] /= _calibration(0, 0);
    ray_cam[1] /= _calibration(1, 1);
    ray_cam[2] = 1;
    return ray_cam;
}

std::vector<double> RGBDCamera::getDepth(const std::shared_ptr<AFeature> &feature) {
    std::vector<double> depths;
    for (auto &pt: feature->getPoints()){
        Eigen::Vector3d colors = geometry::getColorSubpix(_depth, pt);
        depths.push_back(colors.x());
    }
    return depths;
}

std::vector<Eigen::Vector3d> RGBDCamera::getP3Dcam(const std::shared_ptr<AFeature> &feature){
    std::vector<double> depths = getDepth(feature);
    std::vector<Eigen::Vector3d> p3ds;
    for(uint i=0; i < feature->getPoints().size(); ++i){
        Eigen::Vector3d p2dh = Eigen::Vector3d(feature->getPoints().at(i).x(), feature->getPoints().at(i).y(), 1.);
        double z = depths.at(i);

        Eigen::Vector3d p3d;
        p3d = z*_calibration.inverse()*p2dh;
        p3ds.push_back(p3d);
    }
    return p3ds;
}

} // namespace isae
