#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <cmath>
#include <limits>

#include <Eigen/Eigenvalues>
#include <opencv2/core.hpp>

#include "isaeslam/typedefs.h"

namespace isae {

//< 3D geometry utilities
namespace geometry {

template <typename Derived> inline Eigen::Matrix3d skewMatrix(const Eigen::MatrixBase<Derived> &w) {
    Eigen::Matrix3d skew;
    skew << 0, -w[2], w[1], //
        w[2], 0, -w[0],     //
        -w[1], w[0], 0;     //
    return skew;
}

inline Eigen::Vector3d FromskewMatrix(const Eigen::Matrix3d &skew) {
    Eigen::Vector3d w(skew(2, 1), skew(0, 2), skew(1, 0));
    return w;
}

inline Eigen::Matrix3d so3_rightJacobian(const Eigen::Vector3d &w) {
    double w_norm = w.norm();
    Eigen::Matrix3d w_skew = skewMatrix(w);
    if (w_norm < 1e-5)
        return Eigen::Matrix3d::Identity();
    return Eigen::Matrix3d::Identity() - ((1 - cos(w_norm)) / (w_norm * w_norm)) * w_skew +
           ((w_norm - sin(w_norm)) / (w_norm * w_norm * w_norm)) * w_skew * w_skew;
}

inline Eigen::Matrix3d so3_leftJacobian(const Eigen::Vector3d &w) {
    Eigen::Matrix3d Jl;
    double w_norm     = w.norm();
    Eigen::Vector3d a = w / w_norm;
    Jl = sin(w_norm) / w_norm * Eigen::Matrix3d::Identity() + (1 - sin(w_norm) / w_norm) * a * a.transpose() +
         (1 - cos(w_norm)) / w_norm * skewMatrix(a);
    return Jl;
}

inline Eigen::Vector3d rotationMatrixToEulerAnglesEigen(Eigen::Matrix3d R) {

    float sy = sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular) {
        x = atan2(R(2, 1), R(2, 2));
        y = atan2(-R(2, 0), sy);
        z = atan2(R(1, 0), R(0, 0));
    } else {
        x = atan2(-R(1, 2), R(1, 1));
        y = atan2(-R(2, 0), sy);
        z = 0;
    }
    return Eigen::Vector3d(x * 180 / 3.1416, y * 180 / 3.1416, z * 180 / 3.1416);
}

inline Eigen::Matrix3d xFront2zFront() {
    Eigen::Matrix3d xF2zF;
    xF2zF << 0., -1., 0., //
        0., 0., -1.,      //
        1., 0., 0.;       //
    return xF2zF;
}

inline Eigen::Matrix3d zFront2xFront() {
    Eigen::Matrix3d zF2xF;
    zF2xF << 0, 0, 1, //
        -1, 0, 0,     //
        0, -1, 0;     //
    return zF2xF;
}

inline Eigen::Matrix3d xRight2xFront() {
    Eigen::Matrix3d xR2xF;
    xR2xF << 0., 1., 0., //
        -1., 0., 0.,     //
        0., 0., 1.;      //
    return xR2xF;
}

inline double dist2EpipolarLine(const Eigen::Vector3d &epipolarline, const Eigen::Vector2d &pt_to_test) {
    // inverse x and y for image coordinate system
    return fabs(epipolarline.dot(Eigen::Vector3d(pt_to_test(1), pt_to_test(0), 1.)));
}

inline double getMaxAngleBetweenVectors(vec3d vecs) {
    double max = 0;
    if (vecs.empty())
        return -1;
    for (size_t i = 0; i < vecs.size() - 1; i++) {
        for (size_t j = i + 1; j < vecs.size(); j++) {
            double angle = std::atan2(vecs[i].cross(vecs[j]).norm(), vecs[i].dot(vecs[j]));
            angle        = std::abs(angle);
            if (angle > max)
                max = angle;
        }
    }
    return max;
}

inline Eigen::Matrix3d directionVector2Rotation(Eigen::Vector3d v,
                                                const Eigen::Vector3d &reference_vector = Eigen::Vector3d(1, 0, 0)) {

    // v = R*reference
    // Rodrigues's rotation formula gives the result of a rotation of a unit vector a about an axis of
    // rotation k through the angle θ. We can make use of this by realizing that, in order to bring
    // a normalized vector a into coincidence with another normalized vector b, we simply need to
    // rotate a about k=(a+b)/2 by the angle pi.
    v                   = v.normalized(); // assure v is normalized
    Eigen::Vector3d sum = v + reference_vector;
    Eigen::Matrix3d R   = 2.0 * (sum * sum.transpose()) / (sum.transpose() * sum) - Eigen::Matrix3d::Identity();
    return R;
}

inline Eigen::Vector3d Rotation2directionVector(const Eigen::Matrix3d &R,
                                                const Eigen::Vector3d &reference_vector = Eigen::Vector3d(1, 0, 0)) {
    Eigen::Vector3d v = R * reference_vector;
    return v.normalized(); // assure v is normalized (should already be)
}

inline Eigen::Matrix3d exp_so3(const Eigen::Vector3d &v) {

    double tolerance = 1e-9;
    double angle     = v.norm();

    Eigen::Matrix3d Rot;
    if (angle < tolerance) {
        // Near |phi|==0, use first order Taylor expansion
        Rot = Eigen::Matrix3d::Identity() + geometry::skewMatrix(v);
    } else {
        Eigen::Vector3d axis = v / angle;
        Eigen::Matrix3d skew = geometry::skewMatrix(axis);
        Rot                  = Eigen::Matrix3d::Identity() + (1. - cos(angle)) * skew * skew +
              sin(angle) * skew;
    }
    return Rot;
}

inline Eigen::Vector3d log_so3(const Eigen::Matrix3d &M) {
    double tolerance = 1e-9;
    double cos_angle = 0.5 * M.trace() - 0.5;

    // Clip cos(angle) to its proper domain to avoid NaNs from rounding errors
    cos_angle    = std::min(std::max(cos_angle, -1.), 1.);
    double angle = acos(cos_angle);

    Eigen::Vector3d phi;
    // If angle is close to zero, use first-order Taylor expansion
    if (std::fabs(sin(angle)) < tolerance || angle < tolerance)
        phi = 0.5 * geometry::FromskewMatrix(M - M.transpose());
    else {
        // Otherwise take the matrix logarithm and return the rotation vector
        phi = (0.5 * angle / sin(angle)) * geometry::FromskewMatrix(M - M.transpose());
    }
    return phi;
}

inline Vector6d se3_RTtoVec6d(Eigen::Affine3d RT) {
    Vector6d pose;
    Eigen::Vector3d w = log_so3(RT.linear());
    pose(0)           = w(0);
    pose(1)           = w(1);
    pose(2)           = w(2);
    Eigen::Vector3d t = RT.translation();
    pose(3)           = t(0);
    pose(4)           = t(1);
    pose(5)           = t(2);
    return pose;
}

inline Eigen::Affine3d se3_Vec6dtoRT(Vector6d pose) {
    Eigen::Affine3d RT;
    Eigen::Vector3d w, t;
    w << pose(0), pose(1), pose(2);
    t << pose(3), pose(4), pose(5);
    RT.linear()      = exp_so3(w);
    RT.translation() = t;
    return RT;
}

inline Eigen::Affine3d se3_Vec3dtoRT(Eigen::Vector3d p3d) {
    Eigen::Affine3d RT;
    RT.linear()      = Eigen::Matrix3d::Identity();
    RT.translation() = p3d;
    return RT;
}

inline Eigen::Affine3d se3_doubleVec6dtoRT(const double *pose) {
    Eigen::Affine3d RT;
    RT.linear() = exp_so3(Eigen::Vector3d(pose[0], pose[1], pose[2]));
    RT.translation() << pose[3], pose[4], pose[5];
    return RT;
}

inline Eigen::Affine3d se3_doubleVec3dtoRT(const double *p3d) {
    Eigen::Affine3d RT;
    RT.linear() = Eigen::Matrix3d::Identity();
    RT.translation() << p3d[0], p3d[1], p3d[2];
    return RT;
}

template <typename T> inline Eigen::Transform<T, 3, 4> se3_doubleVec6dtoRT(const T *pose) {
    Eigen::Transform<T, 3, 4> RT;
    // Eigen::Affine3d RT;
    Eigen::Matrix<T, 3, 1> w, t;
    // Eigen::Vector3d w, t;
    w << pose[0], pose[1], pose[2];
    t << pose[3], pose[4], pose[5];
    RT.linear()      = exp_so3(w);
    RT.translation() = t;
    return RT;
}

inline double getVectorAngle(Eigen::Vector2d pt_end, Eigen::Vector2d pt_start) {
    double angle = fmod(atan2((pt_end(1) - pt_start(1)), (pt_end(0) - pt_start(0))), 2 * M_PI);
    if (angle < 0.0)
        angle += 2.0 * M_PI;
    return angle;
}

inline double getDeltaAngle(double orientation, double orientation_reproj) {

    double residu = fmod(orientation - orientation_reproj, M_PI);
    if (residu > M_PI / 2.)
        residu += -M_PI;
    if (residu < -M_PI / 2.)
        residu += M_PI;
    return residu;
}

inline Eigen::Vector3d getColorSubpix(const cv::Mat &img, Eigen::Vector2d pt) {

    // bilinear interpolation
    double dx = pt.x() - (int)pt.x();
    double dy = pt.y() - (int)pt.y();

    double weight_tl = (1.0 - dx) * (1.0 - dy);
    double weight_tr = (dx) * (1.0 - dy);
    double weight_bl = (1.0 - dx) * (dy);
    double weight_br = (dx) * (dy);

    std::vector<double> interp_cols(3, 0);
    for (uint i = 0; i < (uint)(img.channels()); ++i)
        interp_cols.at(i) = weight_tl * img.at<uint>((int)pt.x(), (int)pt.y()) +
                            weight_tr * img.at<uint>((int)pt.x() + 1, (int)pt.y()) +
                            weight_bl * img.at<uint>((int)pt.x(), (int)pt.y() + 1) +
                            weight_br * img.at<uint>((int)pt.x() + 1, (int)pt.y() + 1);

    // if grayscale image only output(0) is set, other are 0
    return Eigen::Vector3d(interp_cols.at(0), interp_cols.at(1), interp_cols.at(2));
}

inline double getAngle(Eigen::Vector3d p, Eigen::Vector3d p1, Eigen::Vector3d p2) {
    Eigen::Vector3d u1 = p - p1;
    Eigen::Vector3d u2 = p - p2;

    return std::acos(u1.dot(u2) / (u1.norm() * u2.norm()));
}

// Check if a point is in a triangle with the barycentric technique
// cf. https://blackpawn.com/texts/pointinpoly/
template <typename Derived = Eigen::VectorXd> inline bool pointInTriangle(Derived pt, std::vector<Derived> triangle) {
    // Compute vectors
    Derived v0 = triangle.at(2) - triangle.at(0);
    Derived v1 = triangle.at(1) - triangle.at(0);
    Derived v2 = pt - triangle.at(0);

    // Compute dot products
    double dot00 = v0.dot(v0);
    double dot01 = v0.dot(v1);
    double dot02 = v0.dot(v2);
    double dot11 = v1.dot(v1);
    double dot12 = v1.dot(v2);

    // Compute barycentric coordinates
    double inv_denom = 1 / (dot00 * dot11 - dot01 * dot01);
    double u         = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    double v         = (dot00 * dot12 - dot01 * dot02) * inv_denom;

    // Check if point is in triangle
    return (u >= 0) && (v >= 0) && (u + v < 1);
}

// According to CGAL library 
// Cf. Principal Component Analysis in CGAL, Gupta et al
inline Eigen::Matrix2d cov2dTriangle(std::vector<Eigen::Vector2d> triangle) {

    // Compute transformation
    Eigen::Vector2d x0 = triangle.at(0);
    Eigen::Matrix2d At;
    At << triangle.at(1).x() - x0.x(), triangle.at(2).x() - x0.x(), triangle.at(1).y() - x0.y(),
        triangle.at(2).y() - x0.y();
    Eigen::Vector2d barycenter = (1.0 / 3.0) * (x0 + triangle.at(1) + triangle.at(2));

    // Primitive covariance
    Eigen::Matrix2d Mot;
    Mot << 1.0 / 12, 1.0 / 24, 1.0 / 24, 1.0 / 12;
    Eigen::Matrix2d M = 2 * At * Mot * At.transpose() +
                        (barycenter * x0.transpose() + x0 * barycenter.transpose() - x0 * x0.transpose()) -
                        barycenter * barycenter.transpose();

    return M;
}

inline double areaTriangle(std::vector<Eigen::Vector2d> triangle) {

    Eigen::Vector2d x0 = triangle.at(0);
    Eigen::Matrix2d At;
    At << triangle.at(1).x() - x0.x(), triangle.at(2).x() - x0.x(), triangle.at(1).y() - x0.y(),
        triangle.at(2).y() - x0.y();

    return At.determinant() / 2;

}



inline Eigen::MatrixXd J_norm(Eigen::Vector3d X){
    return X.transpose()/X.norm();
}

inline Eigen::Matrix3d J_normalization(Eigen::Vector3d X){
    return (Eigen::Matrix3d::Identity() - X*X.transpose())/X.norm();
}

inline Eigen::Matrix3d J_XcrossA(Eigen::Vector3d A){
    return geometry::skewMatrix(A);
}

inline Eigen::Matrix3d J_AcrossX(Eigen::Vector3d A){
    return -geometry::skewMatrix(A);
}

inline Eigen::Matrix3d J_Rexpwt(Eigen::Matrix3d R, Eigen::Matrix3d expw, Eigen::Vector3d t){    
    return -R*expw*geometry::skewMatrix(t)*Eigen::Matrix3d::Identity()*geometry::so3_rightJacobian(geometry::log_so3(expw));
}


} // namespace geometry
} // namespace isae

#endif // GEOMETRY_H
