#include "isaeslam/landmarkinitializer/Point3DlandmarkInitializer.h"
#include "isaeslam/data/features/AFeature2D.h"
#include "isaeslam/data/landmarks/Point3D.h"
#include "utilities/geometry.h"

namespace isae {

bool Point3DLandmarkInitializer::initLandmark(std::vector<std::shared_ptr<isae::AFeature>> features,
                                              std::shared_ptr<isae::ALandmark> &landmark) {
    Eigen::Vector3d position;
    landmark = std::shared_ptr<Point3D>(new Point3D());

    // Point2D triangulation required at least 2 features
    if (features.size() < 2)
        return false;

    // Get ray and optical centers of cameras in world coordinates
    Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
    Eigen::Vector3d C(0, 0, 0);
    vec3d rays;

    for (const std::shared_ptr<AFeature> &f : features) {

        std::shared_ptr<ImageSensor> cam = f->getSensor();
        Eigen::Vector3d ray              = f->getRays().at(0);

        Eigen::Vector3d o;
        Eigen::Matrix3d A;

        rays.push_back(ray);
        o = cam->getSensor2WorldTransform().translation();

        A << ray[0] * ray[0] - 1, ray[0] * ray[1], ray[0] * ray[2], ray[0] * ray[1], ray[1] * ray[1] - 1,
            ray[1] * ray[2], ray[0] * ray[2], ray[1] * ray[2], ray[2] * ray[2] - 1;

        S += A;

        C += A * o;
    }

    // TODO: which metric is the best to ensure good triangulation? 

    // Test compute the global parallax
    // Eigen::Affine3d T_c0_w = features.at(0)->getSensor()->getWorld2SensorTransform();
    // Eigen::Vector3d u0     = features.at(0)->getBearingVectors().at(0);
    // Eigen::Affine3d T_cn_w;
    // Eigen::Vector3d un;
    // for (auto &f : features) {
    //     T_cn_w = f->getSensor()->getWorld2SensorTransform();
    //     un     = f->getBearingVectors().at(0);
    // }

    // Eigen::Affine3d T_cn_c0 = T_cn_w * T_c0_w.inverse();
    // u0 = T_cn_c0.rotation() * u0;
    // double parallax    = std::acos(u0.normalized().dot(un.normalized())) * 180 / M_PI;
    // if (std::abs(parallax) < 1) {
    //     landmark->init(Eigen::Affine3d::Identity(), features);
    //     landmark->setUninitialized();
    //     return true;
    // }


    // Compute SVD and check the smallest value 
    // Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(S);
    // if (std::abs(saes.eigenvalues().z()) / features.size() < 1e-5) {
    //     landmark->init(Eigen::Affine3d::Identity(), features);
    //     landmark->setUninitialized();
    //     return true;
    // }
    
    // Compute the determinant
    if (std::abs(S.determinant()) < 1e-5) {
        landmark->init(Eigen::Affine3d::Identity(), features);
        landmark->setUninitialized();
        return true;
    }

    // Process landmark pose in world frame
    position = S.inverse() * C;

    // Process orientation
    // no orientation for Point3D
    Eigen::Matrix3d orientation = Eigen::Matrix3d::Identity();

    // Create landmark
    Eigen::Affine3d T_w_lmk;
    T_w_lmk.translation() = position;
    T_w_lmk.linear()      = orientation;

    // Test if the landmark is in the front of the cam or if it is not too far
    Eigen::Affine3d T_cam_lmk = features.at(0)->getSensor()->getWorld2SensorTransform() * T_w_lmk;
    if (T_cam_lmk.translation()(2) < 0 || T_cam_lmk.translation().norm() > 20)
        return false;

    // Set Landmark state
    landmark->init(T_w_lmk, features);

    return true;
}

bool Point3DLandmarkInitializer::initLandmarkWithDepth(std::vector<std::shared_ptr<AFeature>> features,
                                                       std::shared_ptr<ALandmark> &landmark) {

    std::vector<std::vector<Eigen::Vector3d>> all_p3ds;
    for (auto f : features) {
        std::vector<Eigen::Vector3d> p3ds = f->getSensor()->getP3Dcam(f);
        all_p3ds.push_back(p3ds);
    }

    Eigen::Vector3d position;
    for (auto p3ds : all_p3ds)
        position += p3ds.at(0);
    position = position / all_p3ds.size();

    Eigen::Affine3d T_w_lmk;
    T_w_lmk.translation() = position;
    T_w_lmk.linear()      = Eigen::Matrix3d::Identity();

    landmark = std::shared_ptr<Point3D>(new Point3D());
    landmark->init(T_w_lmk, features);

    return true;
}

std::shared_ptr<ALandmark> Point3DLandmarkInitializer::createNewLandmark(std::shared_ptr<isae::AFeature> f) {
    // can't init Point3D with only one feature
    return nullptr;
}

} // namespace isae
