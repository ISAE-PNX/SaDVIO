#include "isaeslam/landmarkinitializer/Line3DlandmarkInitializer.h"
#include "isaeslam/data/features/AFeature2D.h"
#include "isaeslam/data/landmarks/Line3D.h"
#include "utilities/geometry.h"

#include <Eigen/Dense>

namespace isae {


bool Line3DLandmarkInitializer::initLandmark(std::vector<std::shared_ptr<isae::AFeature> > features, std::shared_ptr<isae::ALandmark> &landmark)
{
    // Point2D triangulation required at least 2 features
    uint N = features.size();
    if(N < 2)
        return false;

    // Get ray and optical centers of cameras in world coordinates
    std::vector<Eigen::Vector3d> Ns, Os;
    std::vector<Eigen::Vector3d> rays_s, rays_e;

    for(const std::shared_ptr<AFeature> &f : features){

        std::shared_ptr<ImageSensor> cam = f->getSensor();
        Eigen::Vector3d ray_s = f->getRays().at(0);
        Eigen::Vector3d ray_e = f->getRays().at(1);

        rays_s.push_back(ray_s);
        rays_e.push_back(ray_e);

        Eigen::Vector3d n = (ray_s.cross(ray_e)).normalized();
        Ns.push_back(n);
        Os.push_back(cam->getSensor2WorldTransform().translation());

    }

    // Process 3D line orientation : the line is the intersection of all planes defined by the normals Ns
    Eigen::Vector3d direction = processOrientation(Ns);
    Eigen::Matrix3d orientation = geometry::directionVector2Rotation(direction, Eigen::Vector3d(1,0,0)); // rotation to take (1,0,0) to actual direction

    // Process line position
    // AX=B -- n'*X = n'*o
    Eigen::Vector3d point_on_line = processPosition(Ns, Os);

    // Process segment points
    Eigen::Vector3d start, end;
    processSegmentPoints(point_on_line, direction, Os, rays_s, rays_e, start, end);
    Eigen::Vector3d position = 0.5*(start+end);
    // std::cout << "DEBUG TRIANGULATOR = start = " << start.transpose() << std::endl;
    // std::cout << "DEBUG TRIANGULATOR = end = " << end.transpose() << std::endl;


    // Create landmark
    Eigen::Affine3d T_w_lmk;
    T_w_lmk.translation() = position;
    T_w_lmk.linear() = orientation;

    // Add landmark reference to frames (so can get the linked cloud)
    landmark = std::shared_ptr<Line3D>(new Line3D());

    // Set Landmark state
    double scale = (start-end).norm();
    landmark->init(T_w_lmk, features);
    landmark->setScale(Eigen::Vector3d(scale,scale,scale));

    // Check for landmark reprojection error
    landmark->sanityCheck();

    //  std::cout << "DEBUG TRIANGULATOR = Ldmk initialized ! " << std::endl;
    return true;
}

bool Line3DLandmarkInitializer::initLandmarkWithDepth(std::vector<std::shared_ptr<AFeature> > features, std::shared_ptr<ALandmark> &landmark){

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

    landmark = std::shared_ptr<Line3D>(new Line3D());
    landmark->init(T_w_lmk, features);

    double scale = 0.;
    for (auto f : features)
        scale = std::max(scale, (f->getPoints().at(0)-f->getPoints().at(1)).norm());
    landmark->setScale(Eigen::Vector3d(scale,scale,scale));

    return true;
}

std::shared_ptr<ALandmark> Line3DLandmarkInitializer::createNewLandmark(std::shared_ptr<isae::AFeature> f)
{
    // can't init Line3D with only one feature
    return nullptr;
}

Eigen::Vector3d Line3DLandmarkInitializer::processOrientation(std::vector<Eigen::Vector3d> Ns){

    Eigen::MatrixXd A(Ns.size(),3);
    for(uint r=0; r< Ns.size(); ++r){
        A.row(r) << Ns.at(r).x(), Ns.at(r).y(), Ns.at(r).z();
    }

    // Solve Ax=0
    Eigen::MatrixXd Ab = A.adjoint()*A; // |M * x|^2 subject to |x|=1
    Eigen::Vector3d direction = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(Ab).eigenvectors().col(0);
    direction.normalize();

    return direction;
}

Eigen::Vector3d Line3DLandmarkInitializer::processPosition(std::vector<Eigen::Vector3d> Ns, std::vector<Eigen::Vector3d> Os){
    Eigen::MatrixXd A(Ns.size(),3);
    Eigen::MatrixXd B(Ns.size(),1);
    for(uint r=0; r< Ns.size(); ++r){
        A.row(r) << Ns.at(r).x(), Ns.at(r).y(), Ns.at(r).z();
        B.row(r) = Ns.at(r).transpose()*Os.at(r);
    }
    Eigen::Vector3d pt_on_line = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B);
    return pt_on_line;
}


void Line3DLandmarkInitializer::processSegmentPoints(Eigen::Vector3d position,
                                                     Eigen::Vector3d direction,
                                                     std::vector<Eigen::Vector3d> Os,
                                                     std::vector<Eigen::Vector3d> rays_s,
                                                     std::vector<Eigen::Vector3d> rays_e,
                                                     Eigen::Vector3d &start,
                                                     Eigen::Vector3d &end){

    std::vector<double> ts;
    for(uint i=0; i < Os.size(); ++i){
        Eigen::Vector2d C;
        Eigen::Matrix2d CV;
        CV << 1., rays_s.at(i).transpose()*direction, rays_s.at(i).transpose()*direction, 1.;
        C << rays_s.at(i).transpose()*(position-Os.at(i)), -direction.transpose()*(position-Os.at(i));
        Eigen::Vector2d txy = 1./(1.- rays_s.at(i).transpose()*direction*rays_s.at(i).transpose()*direction)*CV*C;
        ts.push_back(txy.y());

        CV << 1., rays_e.at(i).transpose()*direction, rays_e.at(i).transpose()*direction, 1.;
        C << rays_e.at(i).transpose()*(position-Os.at(i)), -direction.transpose()*(position-Os.at(i));
        txy = 1./(1.- rays_e.at(i).transpose()*direction*rays_e.at(i).transpose()*direction)*CV*C;
        ts.push_back(txy.y());
    }

    double tmin = *std::max_element(ts.begin(), ts.end());
    double tmax = *std::min_element(ts.begin(), ts.end());

    start = position+tmin*direction;
    end = position+tmax*direction;
}


} // namespace isae
