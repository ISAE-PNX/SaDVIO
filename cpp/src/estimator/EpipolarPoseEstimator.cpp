#include "isaeslam/estimator/EpipolarPoseEstimator.h"
#include "utilities/geometry.h"

#include <iostream>

namespace isae {

bool estimateMotionWithHomography(std::vector<cv::Point2d> p_prev,
                                  std::vector<cv::Point2d> p_curr,
                                  cv::Mat K,
                                  cv::Mat H,
                                  std::vector<cv::Mat> Rs,
                                  std::vector<cv::Mat> ts,
                                  std::vector<cv::Mat> normals,
                                  std::vector<int> inliers) {

    // Homography matrix
    int method                   = cv::RANSAC;
    double ransacReprojThreshold = 3;
    cv::Mat cvMask;
    H = cv::findHomography(p_prev, p_curr, method, ransacReprojThreshold, cvMask);
    H /= H.at<double>(2, 2);

    // Get inliers
    inliers.clear();
    for (int i = 0; i < cvMask.rows; i++)
        if ((int)cvMask.at<unsigned char>(i, 0) == 1)
            inliers.push_back(i);

    // Recover R,t from Homograph matrix
    cv::decomposeHomographyMat(H, K, Rs, ts, normals);
    // Normalize t
    for (auto &t : ts) {
        t = t / sqrt(t.at<double>(1, 0) * t.at<double>(1, 0) + t.at<double>(2, 0) * t.at<double>(2, 0) +
                     t.at<double>(0, 0) * t.at<double>(0, 0));
    }

    // Remove wrong RT
    // If for a (R,t), a point's pos is behind the camera, then this is wrong.
    std::vector<cv::Mat> res_Rs, res_ts, res_normals;
    cv::Mat possibleSolutions; // Use print_MatProperty to know its type: 32SC1
    std::vector<cv::Point2f> p_prev_np, p_curr_np;
    for (int idx : inliers) {
        p_prev_np.push_back(cv::Point2f((p_prev[idx].x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                                        (p_prev[idx].y - K.at<double>(1, 2)) / K.at<double>(1, 1)));
        p_curr_np.push_back(cv::Point2f((p_curr[idx].x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                                        (p_curr[idx].y - K.at<double>(1, 2)) / K.at<double>(1, 1)));
    }

    cv::filterHomographyDecompByVisibleRefpoints(Rs, normals, p_prev_np, p_curr_np, possibleSolutions);
    for (int i = 0; i < possibleSolutions.rows; i++) {
        int idx = possibleSolutions.at<int>(i, 0);
        res_Rs.push_back(Rs[idx]);
        res_ts.push_back(ts[idx]);
        res_normals.push_back(normals[idx]);
    }

    // return
    Rs      = res_Rs;
    ts      = res_ts;
    normals = res_normals;

    return true;
}

bool EpipolarPoseEstimator::estimateTransformBetween(const std::shared_ptr<Frame> &frame1,
                                                     const std::shared_ptr<Frame> &frame2,
                                                     vec_match &matches,
                                                     Eigen::Affine3d &dT,
                                                     Eigen::MatrixXd &covdT) {

    if (matches.size() < 5) { // 5pts minimum for Essential Matrix
        std::cerr << "Not enough matches for Epipolar geometry" << std::endl;
        return false;
    }

    // Get homogeneous matched points in opencv format
    std::vector<cv::Point2f> p_prev, p_curr;
    double avg_flow = 0;
    for (const auto &match : matches) {
        Eigen::Vector3d ray1 = match.first->getSensor()->getRayCamera(match.first->getPoints().at(0));
        Eigen::Vector3d ray2 = match.second->getSensor()->getRayCamera(match.second->getPoints().at(0));
        avg_flow += (match.first->getPoints().at(0) - match.second->getPoints().at(0)).norm();
        p_prev.push_back(cv::Point2f(ray1.x() / ray1.z(), ray1.y() / ray1.z()));
        p_curr.push_back(cv::Point2f(ray2.x() / ray2.z(), ray2.y() / ray2.z()));
    }

    // Don't use epipolar for small motion
    avg_flow /= p_prev.size();
    if (avg_flow < 5) {
        return true;
    }

    // Prepare variables
    cv::Mat K   = cv::Mat::eye(3, 3, CV_32F);
    float focal = matches.at(0).first->getSensor()->getFocal();

    cv::Mat inliers;
    cv::Mat E = cv::findEssentialMat(p_prev, p_curr, K, cv::RANSAC, 0.99, 1.0 / focal, inliers);

    // Remove matches that are not inliers for E processing
    vec_match inliermatches;
    for (uint i = 0; i < matches.size(); i++) {
        if (inliers.at<char>(i)) {
            inliermatches.push_back(matches.at(i));
        }
    }

    cv::Mat Rcv, tcv;
    cv::recoverPose(E, p_prev, p_curr, K, Rcv, tcv, inliers);

    // set Eigen transform dT from Rcv and tcv, WARNING, translation is normalized
    Eigen::Vector3d t_cam2_cam1;
    Eigen::Matrix3d R_cam2_cam1;
    cv::cv2eigen(Rcv, R_cam2_cam1);
    cv::cv2eigen(tcv, t_cam2_cam1);

    // Compute the transformation between frame 1 and frame 2
    Eigen::Affine3d T_cam2_cam1, T_cam1_f1, T_cam2_f2;
    T_cam2_cam1.setIdentity();

    T_cam1_f1                 = matches.at(0).first->getSensor()->getFrame2SensorTransform();
    T_cam2_f2                 = matches.at(0).first->getSensor()->getFrame2SensorTransform();
    T_cam2_cam1.linear()      = R_cam2_cam1;
    T_cam2_cam1.translation() = t_cam2_cam1;

    // The transform has an arbitrary scale
    dT = T_cam1_f1.inverse() * T_cam2_cam1.inverse() * T_cam2_f2;
    dT.translation().normalize();

    return true;
}

bool EpipolarPoseEstimator::estimateTransformBetween(const std::shared_ptr<Frame> &frame1,
                                                     const std::shared_ptr<Frame> &frame2,
                                                     typed_vec_match &typed_matches,
                                                     Eigen::Affine3d &dT,
                                                     Eigen::MatrixXd &covdT) {
    dT = Eigen::Affine3d::Identity();

    // Get all matches of all types
    vec_match matches;
    for (auto &tmatch : typed_matches) {
        for (auto m : tmatch.second)
            matches.push_back(m);
    }

    if (matches.size() < 4) { // changed from 10 to amtch 3d pattern
        std::cerr << "Not enought matches for Epipolar geometry" << std::endl;
        return false;
    }

    // Get matches in opencv format
    std::vector<cv::Point2d> p_prev, p_curr;
    for (const auto &match : matches) {
        const std::shared_ptr<AFeature> f1 = match.first;
        const std::shared_ptr<AFeature> f2 = match.second;
        p_prev.push_back({f1->getPoints().at(0)[0], f1->getPoints().at(0)[1]});
        p_curr.push_back({f2->getPoints().at(0)[0], f2->getPoints().at(0)[1]});
    }

    // Process Essential matrix
    Eigen::Vector2d principal_pt_ = frame2->getSensors().at(0)->getCalibration().block<2, 1>(0, 2);
    cv::Point2d principal_pt(principal_pt_(0), principal_pt_(1));
    cv::Mat cvMask;
    cv::Mat E = cv::findEssentialMat(p_prev,
                                     p_curr,
                                     frame2->getSensors().at(0)->getCalibration()(0, 0),
                                     principal_pt,
                                     cv::RANSAC,
                                     0.99,
                                     1.0,
                                     cvMask);

    // Remove matches that are not inliers for E processing
    typed_vec_match typed_inliermatches;
    uint idx = 0;
    for (auto &tmatch : typed_matches) {
        for (auto m : tmatch.second) {
            if (cvMask.at<char>(idx))
                typed_inliermatches[tmatch.first].push_back(m);
            idx++;
        }
    }

    if (E.empty() || E.cols != 3 || E.rows != 3)
        return false;

    cv::Mat Rcv, tcv;
    int nInliers = cv::recoverPose(
        E, p_prev, p_curr, Rcv, tcv, frame2->getSensors().at(0)->getCalibration()(0, 0), principal_pt, cvMask);

    typed_matches = typed_inliermatches;

    if (nInliers < 3)
        return false;

    // set Eigen transform dT from Rcv and tcv, WARNING, translation is normalized
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    cv::cv2eigen(Rcv.clone(), R);
    cv::cv2eigen(tcv.clone(), t);

    // we need dT for the frame not the camera !
    Eigen::Matrix3d Rframe = frame1->getFrame2WorldTransform().linear();

    R = R.transpose();
    t = -t.normalized();

    dT.linear()      = Rframe.inverse() * R * Rframe;
    dT.translation() = t.normalized();
    dT               = dT.inverse();

    dT.translation().normalize();
    return true;
}

bool EpipolarPoseEstimator::estimateTransformSensors(const std::shared_ptr<ImageSensor> &sensor1,
                                                     const std::shared_ptr<ImageSensor> &sensor2,
                                                     vec_match &matches,
                                                     Eigen::Affine3d &dT,
                                                     Eigen::MatrixXd &covdT) {

    if (matches.size() < 8) { // 5pts minimum for Essential Matrix
        std::cerr << "Not enough matches for Epipolar geometry" << std::endl;
        return false;
    }

    // Get homogeneous matched points in opencv format
    std::vector<cv::Point2f> p_prev, p_curr;
    double avg_flow = 0;
    for (const auto &match : matches) {
        Eigen::Vector3d ray1 = sensor1->getRayCamera(match.first->getPoints().at(0));
        Eigen::Vector3d ray2 = sensor2->getRayCamera(match.second->getPoints().at(0));
        avg_flow += (match.first->getPoints().at(0) - match.second->getPoints().at(0)).norm();
        p_prev.push_back(cv::Point2f(ray1.x() / ray1.z(), ray1.y() / ray1.z()));
        p_curr.push_back(cv::Point2f(ray2.x() / ray2.z(), ray2.y() / ray2.z()));
    }

    // Don't use epipolar for small motion
    avg_flow /= p_prev.size();
    if (avg_flow < 5) {
        return true;
    }

    // Prepare variables
    cv::Mat K   = cv::Mat::eye(3, 3, CV_32F);
    float focal = sensor2->getFocal();

    cv::Mat inliers;
    cv::Mat E = cv::findEssentialMat(p_prev, p_curr, K, cv::RANSAC, 0.99, 3.0 / focal, inliers);

    // Remove matches that are not inliers for E processing
    vec_match inliermatches;
    for (uint i = 0; i < matches.size(); i++) {
        if (inliers.at<char>(i)) {
            inliermatches.push_back(matches.at(i));
        }
    }
    matches = inliermatches;

    cv::Mat Rcv, tcv;
    cv::recoverPose(E, p_prev, p_curr, K, Rcv, tcv, inliers);

    // set Eigen transform dT from Rcv and tcv, WARNING, translation is normalized
    Eigen::Vector3d t_cam2_cam1;
    Eigen::Matrix3d R_cam2_cam1;
    cv::cv2eigen(Rcv, R_cam2_cam1);
    cv::cv2eigen(tcv, t_cam2_cam1);

    // Compute the transformation between frame 1 and frame 2
    Eigen::Affine3d T_cam2_cam1, T_cam1_f1, T_cam2_f2;
    T_cam2_cam1.setIdentity();

    T_cam2_cam1.linear()      = R_cam2_cam1;
    T_cam2_cam1.translation() = t_cam2_cam1;

    // The transform has a unitary scale
    dT = T_cam2_cam1.inverse();
    dT.translation().normalize();

    return true;
}

} // namespace isae
