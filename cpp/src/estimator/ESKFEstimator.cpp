#include "isaeslam/estimator/ESKFEstimator.h"
#include "isaeslam/data/sensors/Camera.h"
#include "utilities/geometry.h"
#include <iostream>

namespace isae {

// Function to compute the Jacobian of the switch from homogeneous point to 2D point
Eigen::MatrixXd jac_homogeneous(const Eigen::Vector3d& point) {
    Eigen::MatrixXd jac(2, 3);
    double inv_z = 1.0 / point(2);
    double inv_z_sq = inv_z * inv_z;

    jac(0, 0) = inv_z;
    jac(0, 1) = 0;
    jac(0, 2) = -point(0) * inv_z_sq;

    jac(1, 0) = 0;
    jac(1, 1) = inv_z;
    jac(1, 2) = -point(1) * inv_z_sq;

    return jac;
}

// Function to compute the Jacobian of the update of the pose T \delta \tau w.r.t the delta
Eigen::MatrixXd jac_delta_update(const Eigen::Vector3d& dtheta, const Eigen::Matrix3d& rotation) {
    Eigen::MatrixXd jr = geometry::so3_rightJacobian(dtheta); 
    Eigen::MatrixXd jac(6, 6);

    jac.block<3, 3>(0, 0) = jr;
    jac.block<3, 3>(0, 3) = Eigen::MatrixXd::Zero(3, 3);
    jac.block<3, 3>(3, 0) = Eigen::MatrixXd::Zero(3, 3);
    jac.block<3, 3>(3, 3) = rotation;

    return jac;
}

std::tuple<Eigen::Vector2d, Eigen::MatrixXd, Eigen::MatrixXd> jac_projection(
    const Eigen::Matrix3d& camera_matrix,
    const Eigen::Vector3d& point_3d,
    const Eigen::Matrix3d& rotation,
    const Eigen::Vector3d& translation,
    const Eigen::Vector3d& dtheta
) {
    // Project point
    Eigen::MatrixXd extrinsic_matrix(3, 4);
    extrinsic_matrix.block<3, 3>(0, 0) = rotation;
    extrinsic_matrix.col(3) = translation;

    Eigen::Vector4d homogeneous_point(point_3d(0), point_3d(1), point_3d(2), 1);
    Eigen::Vector3d proj = camera_matrix * extrinsic_matrix * homogeneous_point;

    // Compute jacs
    Eigen::MatrixXd jac_proj_R = -camera_matrix * rotation * geometry::skewMatrix(point_3d);
    Eigen::MatrixXd jac_proj_T(3, 6);
    jac_proj_T.block<3, 3>(0, 0) = jac_proj_R;
    jac_proj_T.block<3, 3>(0, 3) = Eigen::MatrixXd::Identity(3, 3);

    Eigen::MatrixXd J_T = jac_homogeneous(proj) * jac_proj_T * jac_delta_update(dtheta, rotation);
    Eigen::MatrixXd J_p = jac_homogeneous(proj) * camera_matrix * rotation;

    // Compute 2D projection
    Eigen::Vector2d proj_2d(proj(0) / proj(2), proj(1) / proj(2));

    return {proj_2d, J_T, J_p};
}

bool isae::ESKFEstimator::estimateTransformBetween(const std::shared_ptr<Frame> &frame1,
                                                   const std::shared_ptr<Frame> &frame2,
                                                   vec_match &matches,
                                                   Eigen::Affine3d &dT,
                                                   Eigen::MatrixXd &covdT) {
    if (matches.size() < 5)
        return false;

    std::vector<int> outliersidx;

    // Get matched features from frame 1 with existing 3D landmarks
    std::vector<Eigen::Vector3d> p3d_vector;
    p3d_vector.reserve(matches.size());
    std::vector<Eigen::Vector2d> p2d_vector;
    p2d_vector.reserve(matches.size());

    vec_match init_matches, noninit_matches;
    init_matches.reserve(matches.size());
    noninit_matches.reserve(matches.size());

    Eigen::Affine3d T_cam1_w = matches.at(0).first->getSensor()->getWorld2SensorTransform();
    for (auto &m : matches) {
        if (m.first->getLandmark().lock()) {

            // Ignore non initialized landmarks to keep them in track
            if (!m.first->getLandmark().lock()->isInitialized()) {
                noninit_matches.push_back(m);
                continue;
            }
            init_matches.push_back(m);

            // Get p3d in camera one frame
            Eigen::Vector3d t_w_lmk    = m.first->getLandmark().lock()->getPose().translation();
            Eigen::Vector3d t_cam1_lmk = T_cam1_w * t_w_lmk;
            p3d_vector.push_back(t_cam1_lmk);

            // Get corresponding detection in homogeneous coordinates in frame 2
            Eigen::Vector3d ray_cam2 = m.second->getBearingVectors().at(0);
            p2d_vector.push_back(Eigen::Vector2d(0 + (ray_cam2.x() / ray_cam2.z()), 0 + (ray_cam2.y() / ray_cam2.z())));
        }
    }

    // Init the transformation
    Eigen::Affine3d T_cam1_f1 = matches.at(0).first->getSensor()->getFrame2SensorTransform();
    Eigen::Affine3d T_cam2_f2 = matches.at(0).second->getSensor()->getFrame2SensorTransform();
    Eigen::Affine3d T_cam2_cam1   = T_cam1_f1 * dT.inverse() * T_cam2_f2.inverse();

    Eigen::Matrix3d K      = Eigen::Matrix3d::Identity();
    // K(0,2) = 4;
    // K(1,2) = 4;
    Eigen::Matrix2d R      = 0.01 * Eigen::Matrix2d::Identity();
    Eigen::Matrix<double, 6, 6> P = Eigen::Matrix<double, 6, 6>::Identity();
    P.block(0, 0, 3, 3) = 0.01 * Eigen::Matrix3d::Identity();
    // std::shared_ptr<Camera> sensor = std::shared_ptr<Camera>(new Camera(cv::Mat::zeros(8, 8, CV_16F), K));

    // We estimate T_cam2_cam1 using the ESKF
    for (uint i = 0; i < p3d_vector.size(); ++i) {

        // Projection
        // Eigen::Affine3d T_w_lmk = Eigen::Affine3d::Identity();
        // T_w_lmk.translation() = p3d_vector.at(i);
        // double* J_proj_frame = new double[12];
        Eigen::Vector2d proj;
        Eigen::MatrixXd J_T, J_p;
        
        // if(sensor->project(T_w_lmk, dT, R, proj, J_proj_frame, nullptr)) {
        std::tie(proj, J_T, J_p) = jac_projection(K, p3d_vector.at(i), T_cam2_cam1.linear(), T_cam2_cam1.translation(), Eigen::Vector3d::Zero());
        // Eigen::Matrix<double, 2, 6> J = Eigen::Map<const Eigen::Matrix<double, 2, 6>>(J_proj_frame);
        Eigen::Vector2d err = p2d_vector[i] - proj;

        // Kalman Equations
        Eigen::Matrix<double, 6, 2> K = P * J_T.transpose() * (J_T * P * J_T.transpose() + R).inverse();
        Eigen::Matrix<double, 6, 1> dx = K * err;
        P = (Eigen::Matrix<double, 6, 6>::Identity() - K * J_T) * P;

        // Update
        Eigen::Affine3d dtau = Eigen::Affine3d::Identity();
        dtau.translation() = dx.bottomRows<3>();
        dtau.affine().block(0,0,3,3) = geometry::exp_so3(dx.topRows<3>());
        T_cam2_cam1 = T_cam2_cam1 * dtau;
        // }
    }

    covdT = P;
    dT = T_cam1_f1.inverse() * T_cam2_cam1.inverse() * T_cam2_f2;

    return true;
                                                   
}

bool isae::ESKFEstimator::estimateTransformBetween(const std::shared_ptr<Frame> &frame1,
                                                   const std::shared_ptr<Frame> &frame2,
                                                   typed_vec_match &typed_matches,
                                                   Eigen::Affine3d &dT,
                                                   Eigen::MatrixXd &covdT) {
    return false;
                                                   
}

} // namespace isae