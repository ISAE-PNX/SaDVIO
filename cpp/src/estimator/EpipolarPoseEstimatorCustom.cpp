#include "isaeslam/estimator/EpipolarPoseEstimatorCustom.h"
#include "utilities/geometry.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>

namespace isae {

std::vector<int> random_index(int size) {
    std::vector<int> index_list;
    index_list.resize(size, 0);
    std::iota(index_list.begin(), index_list.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(index_list.begin(), index_list.end(), g);
    return index_list;
}

Eigen::Vector3d triangulate(Eigen::Vector3d ray0, Eigen::Vector3d ray1, Eigen::Vector3d t) {
    // triangulate point with mid point method

    // Get ray and optical centers of cameras
    Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
    Eigen::Vector3d C(0, 0, 0);

    // Process the rays
    Eigen::Matrix3d A;
    Eigen::Vector3d o;

    // ray cam 0
    A << ray0[0] * ray0[0] - 1, ray0[0] * ray0[1], ray0[0] * ray0[2], ray0[0] * ray0[1], ray0[1] * ray0[1] - 1,
        ray0[1] * ray0[2], ray0[0] * ray0[2], ray0[1] * ray0[2], ray0[2] * ray0[2] - 1;
    o << 0, 0, 0;
    S += A;
    C += A * o;

    // ray cam 1
    A << ray1[0] * ray1[0] - 1, ray1[0] * ray1[1], ray1[0] * ray1[2], ray1[0] * ray1[1], ray1[1] * ray1[1] - 1,
        ray1[1] * ray1[2], ray1[0] * ray1[2], ray1[1] * ray1[2], ray1[2] * ray1[2] - 1;
    o = t;
    S += A;
    C += A * o;

    // Process landmark pose in camera frame
    Eigen::Vector3d position = S.inverse() * C;
    return position;
}

int checkRT(const Eigen::Matrix3d &R,
            const Eigen::Vector3d &t,
            std::vector<Eigen::Vector2d> rays1,
            std::vector<Eigen::Vector2d> rays2,
            std::vector<int> &inliers) {

    int inliers_number = 0;
    inliers.clear();

    // We check if the depth is positive
    for (size_t i = 0; i < rays1.size(); i++) {
        Eigen::Vector3d normal1(rays1.at(i).x(), rays1.at(i).y(), 1);
        Eigen::Vector3d normal2(rays2.at(i).x(), rays2.at(i).y(), 1);
        Eigen::Vector3d lmk_C1 = triangulate(normal1, normal2, t);

        // check parallax
        double dist1 = normal1.norm();
        double dist2 = normal2.norm();

        double cosParallax = normal1.dot(normal2) / (dist1 * dist2);

        if (!std::isfinite(lmk_C1(0)) || !std::isfinite(lmk_C1(1)) || !std::isfinite(lmk_C1(2))) {
            inliers.push_back(0);
            continue;
        }

        // check depth wrt C1 only if enough parallax as infinite point can have negative depth
        if (lmk_C1(2) <= 0 || cosParallax > 0.99998) {
            inliers.push_back(0);
            continue;
        }

        // check depth wrt C2 as well
        Eigen::Vector3d lmk_C2 = R * lmk_C1 + t;
        if (lmk_C2(2) <= 0 || cosParallax > 0.99998) {
            inliers.push_back(0);
            continue;
        }
        inliers.push_back(1);
        inliers_number++;
    }

    return inliers_number;
}

bool recoverPoseEssential(Eigen::Matrix3d E,
                          std::vector<Eigen::Vector2d> rays1,
                          std::vector<Eigen::Vector2d> rays2,
                          Eigen::Vector3d &t,
                          Eigen::Matrix3d &R,
                          std::vector<int> &inliers) {
    // recover displacement from E
    // We then have x2 = Rx1 + t
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U  = svd.matrixU();
    Eigen::Matrix3d Vt = svd.matrixV().transpose();

    // Let's compute the possible rotation and translation
    Eigen::Matrix3d W;
    W << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    std::vector<int> new_inliers;

    Eigen::Matrix3d R1 = U * W * Vt;
    if (R1.determinant() < 0)
        R1 = -R1;
    Eigen::Vector3d t1 = U.col(2);
    t1                 = t1 / t1.norm();

    Eigen::Matrix3d R2 = U * W.transpose() * Vt;
    if (R2.determinant() < 0)
        R2 = -R2;
    Eigen::Vector3d t2 = -t1;
    int nInliers1, nInliers2, nInliers3, nInliers4;
    std::vector<int> inliers1, inliers2, inliers3, inliers4;

    Eigen::Vector3d rpy_R1 = geometry::rotationMatrixToEulerAnglesEigen(R1);
    // Discard too important rotations
    if (std::abs(rpy_R1(0)) < 90 && std::abs(rpy_R1(1)) < 90 && std::abs(rpy_R1(2)) < 90) {
        nInliers1 = checkRT(R1, t1, rays1, rays2, inliers1);
        nInliers2 = checkRT(R1, t2, rays1, rays2, inliers2);
    } else {
        nInliers1 = 0;
        nInliers2 = 0;
    }

    Eigen::Vector3d rpy_R2 = geometry::rotationMatrixToEulerAnglesEigen(R2);
    if (std::abs(rpy_R2(0)) < 90 && std::abs(rpy_R2(1)) < 90 && std::abs(rpy_R2(2)) < 90) {
        nInliers3 = checkRT(R2, t1, rays1, rays2, inliers3);
        nInliers4 = checkRT(R2, t2, rays1, rays2, inliers4);
    } else {
        nInliers3 = 0;
        nInliers4 = 0;
    }

    int maxInliers = std::max(nInliers1, std::max(nInliers2, std::max(nInliers3, nInliers4)));

    if (maxInliers == 0)
        return false;

    // Select the transformation with the biggest nInliers
    if (maxInliers == nInliers1) {
        R       = R1;
        t       = t1;
        inliers = inliers1;
    } else if (maxInliers == nInliers2) {
        R       = R1;
        t       = t2;
        inliers = inliers2;
    } else if (maxInliers == nInliers3) {
        R       = R2;
        t       = t1;
        inliers = inliers3;
    } else if (maxInliers == nInliers4) {
        R       = R2;
        t       = t2;
        inliers = inliers4;
    }
    return true;
}

Eigen::Matrix3d
computeEssential(std::vector<Eigen::Vector2d> rays1, std::vector<Eigen::Vector2d> rays2, int NPoints = 8) {
    // Let's find the essential matrix with the 8 points algorithm
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(NPoints, 9);

    for (size_t i = 0; i < rays1.size(); i++) {
        Eigen::Vector2d x1v = rays1.at(i);
        Eigen::Vector2d x2v = rays2.at(i);

        A(i, 0) = x2v.x() * x1v.x();
        A(i, 1) = x2v.x() * x1v.y();
        A(i, 2) = x2v.x() * 1;
        A(i, 3) = x2v.y() * x1v.x();
        A(i, 4) = x2v.y() * x1v.y();
        A(i, 5) = x2v.y() * 1;
        A(i, 6) = 1 * x1v.x();
        A(i, 7) = 1 * x1v.y();
        A(i, 8) = 1;
    }

    // Compute eigen values of A
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_0(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    // Compute the approximated Essential matrix
    Eigen::VectorXd e = svd_0.matrixV().col(8);
    Eigen::Matrix3d E_init;
    E_init << e(0), e(1), e(2), e(3), e(4), e(5), e(6), e(7), e(8);

    // Step 2: project it into the essential space
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_1(E_init, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d SIGMA;
    SIGMA << 1, 0, 0, 0, 1, 0, 0, 0, 0;

    Eigen::Matrix3d E = svd_1.matrixU() * SIGMA * svd_1.matrixV().transpose();
    return E;
}

void EssentialRANSAC(std::vector<Eigen::Vector2d> rays1,
                     std::vector<Eigen::Vector2d> rays2,
                     Eigen::Matrix3d &best_E,
                     std::vector<int> &inliers,
                     int Niter   = 500,
                     int Npoints = 8) {
    double best_score = 0;
    std::vector<int> inliers_iter; // 1 if in, 0 if out
    double threshold = 0.0087;

    for (int k = 0; k < Niter; k++) {

        std::vector<int> index_list = random_index((int)rays1.size());

        std::vector<Eigen::Vector2d> rays1_iter, rays2_iter;
        for (int i = 0; i < Npoints; i++) {
            rays1_iter.push_back(rays1.at(index_list[i]));
            rays2_iter.push_back(rays2.at(index_list[i]));
        }

        // Step 1: compute a first approximation of E
        Eigen::Matrix3d E = computeEssential(rays1_iter, rays2_iter);

        // Step 2: Check inliers
        double score = 0;
        inliers_iter.clear();
        for (size_t i = 0; i < rays1.size(); i++) {
            Eigen::Vector3d x1v(rays1.at(i).x(), rays1.at(i).y(), 1);
            Eigen::Vector3d x2v(rays2.at(i).x(), rays2.at(i).y(), 1);

            // Residuals computed with the angle wrt to epiplanes
            Eigen::Vector3d epiplane_1 = E * x1v;
            double residual_1          = std::abs(epiplane_1.dot(x2v)) / epiplane_1.norm();

            if (threshold < residual_1) {
                inliers_iter.push_back(0);
                continue;
            } else
                score += (threshold - residual_1) * (threshold - residual_1);

            Eigen::Vector3d epiplane_2 = E.transpose() * x2v;
            double residual_2          = std::abs(epiplane_2.dot(x1v)) / epiplane_2.norm();

            if (threshold < residual_2) {
                inliers_iter.push_back(0);
                continue;
            } else {
                score += (threshold - residual_2) * (threshold - residual_2);
                inliers_iter.push_back(1);
            }
        }

        // Step 3: Update the Essential Matrix
        if (score > best_score) {
            best_score = score;
            best_E     = E;
            inliers    = inliers_iter;
        }
    }
}

bool EpipolarPoseEstimatorCustom::estimateTransformBetween(const std::shared_ptr<Frame> &frame1,
                                                           const std::shared_ptr<Frame> &frame2,
                                                           vec_match &matches,
                                                           Eigen::Affine3d &dT,
                                                           Eigen::MatrixXd &covdT) {

    if (matches.size() < 4) { // changed from 10 to match 3d pattern
        std::cerr << "Not enough matches for Epipolar geometry" << std::endl;
        return false;
    }

    // Get matched points in opencv format
    std::vector<Eigen::Vector2d> rays_prev, rays_curr;
    for (const auto &match : matches) {
        Eigen::Vector3d ray1 = match.first->getSensor()->getRayCamera(match.first->getPoints().at(0));
        Eigen::Vector3d ray2 = match.second->getSensor()->getRayCamera(match.second->getPoints().at(0));
        rays_prev.push_back({ray1.x() / ray1.z(), ray1.y() / ray1.z()});
        rays_curr.push_back({ray2.x() / ray2.z(), ray2.y() / ray2.z()});
    }

    // Process Essential matrix RANSAC
    Eigen::Matrix3d best_E;
    std::vector<int> inliers;
    EssentialRANSAC(rays_prev, rays_curr, best_E, inliers);

    // Retrieve inliers
    std::vector<Eigen::Vector2d> rays_prev_filtered;
    std::vector<Eigen::Vector2d> rays_curr_filtered;
    for (size_t i = 0; i < rays_curr.size(); i++) {
        if (inliers.at(i) == 1) {
            rays_prev_filtered.push_back(rays_prev.at(i));
            rays_curr_filtered.push_back(rays_curr.at(i));
        }
    }

    // Process Essential matrix with all inliers
    best_E = computeEssential(rays_prev_filtered, rays_curr_filtered, rays_prev_filtered.size());
    best_E /= best_E(2, 2);

    // keep matches that are inliers
    vec_match correct_matches;
    int n_inliers = 0;
    for (size_t i = 0; i < matches.size(); i++) {
        if (inliers.at(i) == 1) {
            correct_matches.push_back(matches.at(i));
            n_inliers++;
        }
    }
    matches = correct_matches;

    if (n_inliers < 3) {
        return false;
    }

    return true;
}

bool EpipolarPoseEstimatorCustom::estimateTransformBetween(const std::shared_ptr<Frame> &frame1,
                                                           const std::shared_ptr<Frame> &frame2,
                                                           typed_vec_match &typed_matches,
                                                           Eigen::Affine3d &dT,
                                                           Eigen::MatrixXd &covdT) {
    return false;
}

} // namespace isae
