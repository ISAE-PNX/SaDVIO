#include "isaeslam/estimator/PnPPoseEstimator.h"
#include "utilities/geometry.h"

namespace isae {

bool PnPPoseEstimator::estimateTransformBetween(const std::shared_ptr<Frame> &frame1,
                                                const std::shared_ptr<Frame> &frame2,
                                                vec_match &matches,
                                                Eigen::Affine3d &dT,
                                                Eigen::MatrixXd &covdT) {
    
    if (matches.size() < 5)
        return false;

    std::vector<int> outliersidx;

    // Get matched features from frame 1 with existing 3D landmarks
    std::vector<cv::Point3d> p3d_vector;
    p3d_vector.reserve(matches.size());
    std::vector<cv::Point2d> p2d_vector;
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
            p3d_vector.push_back({t_cam1_lmk.x(), t_cam1_lmk.y(), t_cam1_lmk.z()});

            // Get corresponding detection in homogeneous coordinates in frame 2
            Eigen::Vector3d ray_cam2 = m.second->getBearingVectors().at(0);
            p2d_vector.push_back({ray_cam2.x() / ray_cam2.z(), ray_cam2.y() / ray_cam2.z()});
        }
    }

    // Init the transformation for pnp
    Eigen::Affine3d T_cam1_f1 = matches.at(0).first->getSensor()->getFrame2SensorTransform();
    Eigen::Affine3d T_cam2_f2 = matches.at(0).first->getSensor()->getFrame2SensorTransform();
    Eigen::Affine3d dT_init   = (T_cam1_f1 * dT * T_cam2_f2.inverse()).inverse();
    cv::Mat tvec, rvec, Rinit;
    Eigen::Vector3d tinit = dT_init.translation();
    cv::eigen2cv(dT_init.rotation(), Rinit);
    cv::eigen2cv(tinit, tvec);
    cv::Rodrigues(Rinit, rvec);

    // Prepare solvepnp variables
    cv::Mat D;
    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat inliers;
    bool use_extrinsic_guess = true;
    float confidence         = 0.99;
    uint nmaxiter            = 50;
    double errth             = 1.0;
    double focal             = matches.at(0).first->getSensor()->getFocal();

    if (p3d_vector.size() < 5)
        return false;

    cv::solvePnPRansac(p3d_vector,
                       p2d_vector,
                       K,
                       D,
                       rvec,
                       tvec,
                       use_extrinsic_guess,
                       nmaxiter,
                       errth / focal,
                       confidence,
                       inliers,
                       cv::SOLVEPNP_P3P);

    if (inliers.rows < 5)
        return false;

    // Relaunch solvepnp with the inliers only using first approx
    use_extrinsic_guess = true;
    std::vector<cv::Point2d> in_p2d_vector;
    in_p2d_vector.reserve(p2d_vector.size());
    std::vector<cv::Point3d> in_p3d_vector;
    in_p3d_vector.reserve(p3d_vector.size());
    vec_match inliers_matches;
    inliers_matches.reserve(matches.size());
    
    for (int i = 0; i < inliers.rows; i++) {
        in_p2d_vector.push_back(p2d_vector.at(inliers.at<int>(i)));
        in_p3d_vector.push_back(p3d_vector.at(inliers.at<int>(i)));
        inliers_matches.push_back(init_matches.at(inliers.at<int>(i)));
    }
    cv::solvePnP(in_p3d_vector, in_p2d_vector, K, D, rvec, tvec, use_extrinsic_guess, cv::SOLVEPNP_ITERATIVE);

    // Get covariance matrix of rotation and translation
    cv::Mat J;
    std::vector<cv::Point2d> p;
    cv::projectPoints(p3d_vector, rvec, tvec, K, D, p, J);
    cv::Mat Sigma = cv::Mat(J.t() * J, cv::Rect(0,0,6,6)).inv();
    cv::cv2eigen(Sigma, covdT);

    // Store the resulting pose of F2-Cam1 in F1-Cam1 frame
    cv::Mat Rcv;
    cv::Rodrigues(rvec, Rcv);
    Eigen::Vector3d t_cam2_cam1;
    Eigen::Matrix3d R_cam2_cam1;
    cv::cv2eigen(Rcv, R_cam2_cam1);
    cv::cv2eigen(tvec, t_cam2_cam1);

    // Compute the transformation between frame 1 and frame 2
    Eigen::Affine3d T_cam2_cam1;

    T_cam2_cam1.linear()      = R_cam2_cam1;
    T_cam2_cam1.translation() = t_cam2_cam1;
    dT                        = T_cam1_f1.inverse() * T_cam2_cam1.inverse() * T_cam2_f2;

    // Update the matches passed as reference
    matches = inliers_matches;
    for (auto &m : noninit_matches) {
        matches.push_back(m);
    }
    return true;
}

bool PnPPoseEstimator::estimateTransformBetween(const std::shared_ptr<Frame> &frame1,
                                                const std::shared_ptr<Frame> &frame2,
                                                typed_vec_match &typed_matches,
                                                Eigen::Affine3d &dT,
                                                Eigen::MatrixXd &covdT) {

    // Get all matches of all types
    vec_match matches;
    for (auto &tmatch : typed_matches) {
        for (auto m : tmatch.second)
            matches.push_back(m);
    }
    if (estimateTransformBetween(frame1, frame2, matches, dT, covdT)) {
        typed_vec_match inliers_typed_matches;
        for (auto &m : matches)
            inliers_typed_matches[m.first->getFeatureLabel()].push_back(m);
        typed_matches = inliers_typed_matches;
        return true;
    }
    return false;
}

} // namespace isae