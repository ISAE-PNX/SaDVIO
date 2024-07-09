#include "isaeslam/landmarkinitializer/Edgelet3DlandmarkInitializer.h"
#include "isaeslam/data/features/AFeature2D.h"
#include "isaeslam/data/landmarks/Edgelet3D.h"
#include "utilities/geometry.h"

#include <Eigen/Dense>

namespace isae {


bool Edgelet3DLandmarkInitializer::initLandmark(std::vector<std::shared_ptr<isae::AFeature> > features, std::shared_ptr<isae::ALandmark> &landmark)
{
    Eigen::Vector3d position;

    // Point2D triangulation requiered at least 2 features
    uint N = features.size();
    if(N < 2)
        return false;

    // Get ray and optical centers of cameras in world coordinates
    Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
    Eigen::Vector3d C(0,0,0);
    std::vector<Eigen::Vector3d> Ns;

    vec3d rays;
    for(const std::shared_ptr<AFeature> &f : features){

        std::shared_ptr<ImageSensor> cam = f->getSensor();
        Eigen::Vector3d ray = f->getRays().at(0);
        Eigen::Vector3d ray2 = f->getRays().at(1);

        Eigen::Vector3d o;
        Eigen::Matrix3d A;

        rays.push_back(ray);
        o = cam->getSensor2WorldTransform().translation();

        A <<
             ray[0]*ray[0] - 1,ray[0]*ray[1],ray[0]*ray[2],
                ray[0]*ray[1],ray[1]*ray[1] - 1,ray[1]*ray[2],
                ray[0]*ray[2],ray[1]*ray[2],ray[2]*ray[2] - 1;

        Ns.push_back((ray.cross(ray2)).normalized());
        S+=A;
        C+=A*o;
    }

    // Process landmark pose in camera frame z in front !
    position = S.inverse()*C;

    // Process orientation
    // Process 3D edgelet orientation : the line is the intersection of all planes defined by the normals Ns
    Eigen::Matrix3d orientation = processOrientation(Ns);


    // Create landmark
    Eigen::Affine3d T_w_lmk;
    T_w_lmk.translation() = position;
    T_w_lmk.linear() = orientation;

    // Add landmark reference to frames (so can get the linked cloud)
    landmark = std::shared_ptr<Edgelet3D>(new Edgelet3D());
//    std::vector<std::shared_ptr<Frame>> allframe;
//    for(const std::shared_ptr<AFeature> &f : features) {
//        std::shared_ptr<Frame> frame = f->getSensor()->getFrame();
//        bool new_frame = true;
//        for (auto f : allframe) {
//            if (frame == f)
//                new_frame = false;
//        }
//
//        if (new_frame) {
//            //frame->addLandmark(landmark);
//            allframe.push_back(frame);
//        }
//    }

    // Set Landmark state
    landmark->init(T_w_lmk, features);
//    landmark = std::shared_ptr<Edgelet3D>(new Edgelet3D(T_w_lmk, features));
//    landmark->setParallax(angle);
//    for(const std::shared_ptr<AFeature> &f : features){
//        landmark->addFeature(f);
//        f->linkLandmark(landmark);
//    }

    // Check for landmark reprojection error
    landmark->sanityCheck();
    return true;
}

bool Edgelet3DLandmarkInitializer::initLandmarkWithDepth(std::vector<std::shared_ptr<AFeature> > features, std::shared_ptr<ALandmark> &landmark){

    std::vector<std::vector<Eigen::Vector3d>> all_p3ds;
    for(auto f: features) {
        std::vector<Eigen::Vector3d> p3ds = f->getSensor()->getP3Dcam(f);
        all_p3ds.push_back(p3ds);
    }

    Eigen::Vector3d position, direction;
    for(auto p3ds : all_p3ds) {
        position += p3ds.at(0);
        direction += p3ds.at(1);
    }
    position = position/all_p3ds.size();
    direction = direction/all_p3ds.size();

    Eigen::Affine3d T_w_lmk;
    T_w_lmk.translation() = position;
    T_w_lmk.linear() = geometry::directionVector2Rotation(direction-position);

    landmark = std::shared_ptr<Edgelet3D>(new Edgelet3D());
    landmark->init(T_w_lmk, features);

    return true;
}

std::shared_ptr<ALandmark> Edgelet3DLandmarkInitializer::createNewLandmark(std::shared_ptr<isae::AFeature> f)
{
    // can't init Edgelet3D with only one feature
    return nullptr;
}

Eigen::Matrix3d Edgelet3DLandmarkInitializer::processOrientation(std::vector<Eigen::Vector3d> Ns){

    Eigen::MatrixXd A(Ns.size(),3);
    for(uint r=0; r< Ns.size(); ++r){
        A.row(r) << Ns.at(r).x(), Ns.at(r).y(), Ns.at(r).z();
    }

    // Solve Ax=0
    Eigen::MatrixXd Ab = A.adjoint()*A; // |M * x|^2 subject to |x|=1
    Eigen::Vector3d direction = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(Ab).eigenvectors().col(0);
    direction.normalize();

    return geometry::directionVector2Rotation(direction, Eigen::Vector3d(1,0,0)); // rotation to take (1,0,0) to actual direction

}

} // namespace isae
