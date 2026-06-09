#ifndef ROSVISUALIZER_H
#define ROSVISUALIZER_H

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/core.hpp>
#include <thread>

#include "sensor_msgs/point_cloud2_iterator.hpp"
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker.hpp>
#include <sadvio_msgs/msg/slam_param.hpp>

#include "isaeslam/data/mesh/mesh.h"
#include "isaeslam/slamCore.h"

// namespace isae {

sensor_msgs::msg::PointCloud2::SharedPtr convertToPointCloud2(const std::vector<Eigen::Vector3d> &points) {
    // Create a PointCloud2 message
    auto point_cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();

    // Set the header
    point_cloud_msg->header.frame_id = "world"; // Set the frame ID as needed
    point_cloud_msg->header.stamp    = rclcpp::Clock().now();

    // Set the point step and row step
    point_cloud_msg->point_step = sizeof(float) * 3;
    point_cloud_msg->row_step   = point_cloud_msg->point_step * point_cloud_msg->width;

    // Set the is_dense flag
    point_cloud_msg->is_dense = true;

    // Set the height and width of the point cloud
    point_cloud_msg->height = 1;
    point_cloud_msg->width  = points.size();

    // Set the fields of the point cloud
    sensor_msgs::PointCloud2Modifier modifier(*point_cloud_msg);
    modifier.setPointCloud2Fields(3,
                                  "x",
                                  1,
                                  sensor_msgs::msg::PointField::FLOAT32,
                                  "y",
                                  1,
                                  sensor_msgs::msg::PointField::FLOAT32,
                                  "z",
                                  1,
                                  sensor_msgs::msg::PointField::FLOAT32);
    // Resize the data vector
    point_cloud_msg->data.resize(points.size() * sizeof(float) * 3);

    // Copy the data from the Eigen vectors to the PointCloud2 message
    sensor_msgs::PointCloud2Iterator<float> iter_x(*point_cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(*point_cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(*point_cloud_msg, "z");

    for (const auto &point : points) {
        *iter_x = point.x();
        *iter_y = point.y();
        *iter_z = point.z();
        ++iter_x;
        ++iter_y;
        ++iter_z;
    }

    return point_cloud_msg;
}

class RosVisualizer : public rclcpp::Node {

  public:
    RosVisualizer() : Node("slam_publisher") {
        std::cout << "\n Creation of ROS vizualizer" << std::endl;

        _pub_image_matches_in_time  = this->create_publisher<sensor_msgs::msg::Image>("image_matches_in_time", 1);
        _pub_image_matches_in_frame = this->create_publisher<sensor_msgs::msg::Image>("image_matches_in_frame", 1);
        _pub_image_kps              = this->create_publisher<sensor_msgs::msg::Image>("image_kps", 1);
        _pub_vo_traj                = this->create_publisher<visualization_msgs::msg::Marker>("vo_traj", 100);
        _pub_vo_pose                = this->create_publisher<geometry_msgs::msg::PoseStamped>("vo_pose", 100);
        _pub_local_map_cloud        = this->create_publisher<visualization_msgs::msg::Marker>("map_local_cloud", 1);
        _pub_local_map_cloud1       = this->create_publisher<visualization_msgs::msg::Marker>("map_local_cloud1", 1);
        _pub_global_map_cloud       = this->create_publisher<visualization_msgs::msg::Marker>("map_global_cloud", 1);
        _pub_local_map_lines        = this->create_publisher<visualization_msgs::msg::Marker>("map_local_lines", 1);
        _pub_global_map_lines       = this->create_publisher<visualization_msgs::msg::Marker>("map_global_lines", 1);
        _pub_marker                 = this->create_publisher<visualization_msgs::msg::Marker>("mesh", 1);
        _pub_cloud                  = this->create_publisher<sensor_msgs::msg::PointCloud2>("point_cloud", 1);
        _pub_param                  = this->create_publisher<sadvio_msgs::msg::SLAMParam>("param", 1);
        _pub_tf_rel                 = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("tf_rel", 100);
        _tf_broadcaster             = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        _vo_traj_msg.type    = visualization_msgs::msg::Marker::LINE_STRIP;
        _vo_traj_msg.color.a = 1.0;
        _vo_traj_msg.color.r = 0.0;
        _vo_traj_msg.color.g = 0.0;
        _vo_traj_msg.color.b = 1.0;
        _vo_traj_msg.scale.x = 0.05;

        // map points design
        _points_local.type    = visualization_msgs::msg::Marker::POINTS;
        _points_local.id      = 0;
        _points_local.scale.x = 0.05;
        _points_local.scale.y = 0.05;
        _points_local.color.a = 1.0;
        _points_local.color.r = 0.0;
        _points_local.color.g = 1.0;
        _points_local.color.b = 0.0;

        _points_local1.type    = visualization_msgs::msg::Marker::POINTS;
        _points_local1.id      = 0;
        _points_local1.scale.x = 0.05;
        _points_local1.scale.y = 0.05;
        _points_local1.color.a = 1.0;
        _points_local1.color.r = 1.0;
        _points_local1.color.g = 0.0;
        _points_local1.color.b = 0.0;

        _points_global.type    = visualization_msgs::msg::Marker::POINTS;
        _points_global.id      = 0;
        _points_global.scale.x = 0.05;
        _points_global.scale.y = 0.05;
        _points_global.color.a = 1.0;
        _points_global.color.r = 0.0;
        _points_global.color.g = 0.0;
        _points_global.color.b = 0.0;

        // map lines design
        _lines_local.type    = visualization_msgs::msg::Marker::LINE_LIST;
        _lines_local.id      = 4;
        _lines_local.scale.x = 0.05;
        _lines_local.scale.y = 0.05;
        _lines_local.color.a = 1.0;
        _lines_local.color.r = 1.0;
        _lines_local.color.g = 0.0;
        _lines_local.color.b = 0.0;

        _lines_global.type    = visualization_msgs::msg::Marker::LINE_LIST;
        _lines_global.id      = 4;
        _lines_global.scale.x = 0.05;
        _lines_global.scale.y = 0.05;
        _lines_global.color.a = 1.0;
        _lines_global.color.r = 0.5;
        _lines_global.color.g = 0.5;
        _lines_global.color.b = 0.5;

        // Mesh line design
        _mesh_line_list.type    = visualization_msgs::msg::Marker::TRIANGLE_LIST;
        _mesh_line_list.id      = 2;
        _mesh_line_list.color.a = 0.5;
        _mesh_line_list.scale.x = 1.0;
        _mesh_line_list.scale.y = 1.0;
        _mesh_line_list.scale.z = 1.0;
    }

    void drawMatchesTopBottom(cv::Mat Itop,
                              std::vector<cv::KeyPoint> kp_top,
                              cv::Mat Ibottom,
                              std::vector<cv::KeyPoint> kp_bottom,
                              std::vector<cv::DMatch> m,
                              cv::Mat &resultImg) {

        uint H = Itop.rows;

        // rotate images 90°
        cv::rotate(Itop, Itop, cv::ROTATE_90_CLOCKWISE);
        cv::rotate(Ibottom, Ibottom, cv::ROTATE_90_CLOCKWISE);

        // change kp coords
        std::vector<cv::KeyPoint> kp_top2, kp_bottom2;
        for (auto &k : kp_top)
            kp_top2.push_back(cv::KeyPoint(H - k.pt.y, k.pt.x, 1));
        for (auto &k : kp_bottom)
            kp_bottom2.push_back(cv::KeyPoint(H - k.pt.y, k.pt.x, 1));

        drawMatches(Itop,
                    kp_top2,
                    Ibottom,
                    kp_bottom2,
                    m,
                    resultImg,
                    cv::Scalar::all(-1),
                    cv::Scalar::all(-1),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        cv::rotate(resultImg, resultImg, cv::ROTATE_90_COUNTERCLOCKWISE);
    }

    void publishImage(const std::shared_ptr<isae::Frame> frame) {
        if (!_pub_image_kps->get_subscription_count()) 
            return;
        std_msgs::msg::Header header;
        header.frame_id = "world";
        header.stamp    = rclcpp::Node::now();
        // Display keypoints
        cv::Mat img_2_pub;
        cv::cvtColor(frame->getSensors().at(0)->getRawData(), img_2_pub, CV_GRAY2RGB);

        int nftrn = 0;
        int nresr = 0;
        int ninit = 0;
        int nlmkninit = 0;
        for (const auto &feat : frame->getSensors().at(0)->getFeatures()["pointxd"]) {
            cv::Scalar col;

            if (feat->getLandmark().lock() == nullptr) {
                col = cv::Scalar(0, 0, 255);    // feature was detected, but has no landmark assigned
                nftrn++;
            } else if (feat->getLandmark().lock()->isResurected()) {
                col = cv::Scalar(0, 255, 0);    // feature was detected _AND_ resurrected (discontinuous tracking)
                nresr++;
            } else {
                if (feat->getLandmark().lock()->isInitialized()) {
                    if (feat->getLandmark().lock()->isOutlier()) {
                        col = cv::Scalar(137, 255, 255); // feature has initialized feature - but was detected as outlier
                    } else {
                        if (feat->getLandmark().lock()->getFeatures().size() == 2) {
                            col = cv::Scalar(255, 255, 0); // feature has landmark assigned -- landmark is linked to only 2 feature
                        } else if (feat->getLandmark().lock()->getFeatures().size() == 3) {
                            col = cv::Scalar(255, 0, 255); // feature has landmark assigned -- landmark is linked to only 3 feature
                        } else {
                            col = cv::Scalar(255, 0, 0); // feature has landmark assigned
                        }
                    }
                    ninit++;
                } else {
                    nlmkninit++;
                }
            }
            Eigen::Vector2d pt2d = feat->getPoints().at(0);

            cv::circle(img_2_pub, cv::Point(pt2d.x(), pt2d.y()), 4, col, -1);
        }
        // std::cout   << "kps: " << nftrn << "/" << nresr << "/" << ninit << "/" << nlmkninit
        //             << " (" << nftrn+nresr+ninit+nlmkninit << ")" << std::endl;

        for (const auto &feat : frame->getSensors().at(0)->getFeatures()["linexd"]) {
            cv::Scalar col;

            if (feat->getLandmark().lock() == nullptr) {
                col = cv::Scalar(255, 0, 0);
            } else {
                col = cv::Scalar(255, 0, 255);
            }
            Eigen::Vector2d pt2d  = feat->getPoints().at(0);
            Eigen::Vector2d pt2d2 = feat->getPoints().at(1);
            cv::line(img_2_pub, cv::Point(pt2d.x(), pt2d.y()), cv::Point(pt2d2.x(), pt2d2.y()), col, 2);
        }

        auto img_kps_msg = cv_bridge::CvImage(header, "rgb8", img_2_pub).toImageMsg();

        _pub_image_kps->publish(*img_kps_msg.get());
    }

    void publishMatches(const isae::typed_vec_match matches, bool in_time) {
        if (!_pub_image_matches_in_time->get_subscription_count() && 
            !_pub_image_matches_in_frame->get_subscription_count()) {
                return;
        }

        std_msgs::msg::Header header;
        header.frame_id = "world";
        header.stamp    = rclcpp::Node::now();

        // Display keypoints
        cv::Mat img_matches;
        cv::Mat img_2_pub_line, img_2_pub_line_2, img_2_pub_pts;

        for (const auto &tmatches : matches) {
            if (tmatches.second.size() == 0)
                continue;

            if (img_matches.empty())
                cv::cvtColor(tmatches.second.at(0).first->getSensor()->getRawData(), img_matches, CV_GRAY2RGB);

            if (tmatches.first == "pointxd") {

                for (const auto &match : tmatches.second) {
                    cv::Scalar colpt   = cv::Scalar(255, 0, 0);
                    cv::Scalar colline = cv::Scalar(0, 255, 0);

                    Eigen::Vector2d pt2d1 = match.first->getPoints().at(0);
                    cv::circle(img_matches, cv::Point(pt2d1.x(), pt2d1.y()), 4, colpt, -1);
                    Eigen::Vector2d pt2d2 = match.second->getPoints().at(0);
                    cv::circle(img_matches, cv::Point(pt2d2.x(), pt2d2.y()), 4, colpt, -1);
                    cv::line(img_matches, cv::Point(pt2d1.x(), pt2d1.y()), cv::Point(pt2d2.x(), pt2d2.y()), colline, 2);
                }
            }

            if (tmatches.first == "linexd") {
                for (const auto &match : tmatches.second) {
                    // Display first feature
                    cv::Scalar colpt        = cv::Scalar(0, 0, 255);
                    cv::Scalar colline      = cv::Scalar(0, 0, 255);
                    cv::Scalar colline2     = cv::Scalar(0, 255, 255);
                    cv::Scalar collinematch = cv::Scalar(255, 0, 255);

                    Eigen::Vector2d pt2d  = match.first->getPoints().at(0);
                    Eigen::Vector2d pt2d2 = match.first->getPoints().at(1);
                    cv::circle(img_matches, cv::Point(pt2d.x(), pt2d.y()), 4, colpt, -1);
                    cv::circle(img_matches, cv::Point(pt2d2.x(), pt2d2.y()), 4, colpt, -1);
                    cv::line(img_matches, cv::Point(pt2d.x(), pt2d.y()), cv::Point(pt2d2.x(), pt2d2.y()), colline, 2);

                    // Display second feature
                    Eigen::Vector2d pt2d_2  = match.second->getPoints().at(0);
                    Eigen::Vector2d pt2d2_2 = match.second->getPoints().at(1);
                    cv::circle(img_matches, cv::Point(pt2d_2.x(), pt2d_2.y()), 4, colpt, -1);
                    cv::circle(img_matches, cv::Point(pt2d2_2.x(), pt2d2_2.y()), 4, colpt, -1);
                    cv::line(img_matches,
                             cv::Point(pt2d_2.x(), pt2d_2.y()),
                             cv::Point(pt2d2_2.x(), pt2d2_2.y()),
                             colline2,
                             2);

                    // Display matching line between the centers
                    cv::line(img_matches,
                             0.5 * (cv::Point(pt2d_2.x(), pt2d_2.y()) + cv::Point(pt2d2_2.x(), pt2d2_2.y())),
                             0.5 * (cv::Point(pt2d.x(), pt2d.y()) + cv::Point(pt2d2.x(), pt2d2.y())),
                             collinematch,
                             2);
                }
            }
        }

        auto imgTrackMsg = cv_bridge::CvImage(header, "rgb8", img_matches).toImageMsg();

        // Choose the good publisher if it is tracked in frame or in time
        if (in_time) {
            _pub_image_matches_in_time->publish(*imgTrackMsg.get());
        } else {
            _pub_image_matches_in_frame->publish(*imgTrackMsg.get());
        }
    }

    void publishFrame(const std::shared_ptr<isae::Frame> frame) {
        if (!_pub_vo_pose->get_subscription_count()) 
            return;

        geometry_msgs::msg::PoseStamped Twc_msg;
        Twc_msg.header.stamp    = rclcpp::Time(frame->getTimestamp());
        Twc_msg.header.frame_id = "world";

        // Deal with position
        geometry_msgs::msg::Point p;
        const Eigen::Vector3d twc = frame->getFrame2WorldTransform().translation();
        p.x                       = twc.x();
        p.y                       = twc.y();
        p.z                       = twc.z();
        Twc_msg.pose.position     = p;

        // Deal with orientation
        geometry_msgs::msg::Quaternion q;
        const Eigen::Quaterniond eigen_q = (Eigen::Quaterniond)frame->getFrame2WorldTransform().linear();
        q.x                              = eigen_q.x();
        q.y                              = eigen_q.y();
        q.z                              = eigen_q.z();
        q.w                              = eigen_q.w();
        Twc_msg.pose.orientation         = q;

        // Publish transform
        geometry_msgs::msg::TransformStamped Twc_tf;
        Twc_tf.header.stamp            = rclcpp::Time(frame->getTimestamp());
        Twc_tf.header.frame_id         = "world";
        Twc_tf.child_frame_id          = "robot";
        Twc_tf.transform.translation.x = twc.x();
        Twc_tf.transform.translation.y = twc.y();
        Twc_tf.transform.translation.z = twc.z();
        Twc_tf.transform.rotation      = Twc_msg.pose.orientation;
        _tf_broadcaster->sendTransform(Twc_tf);

        // publish messages
        _pub_vo_pose->publish(Twc_msg);
    }

    void publishLocalMap(const std::shared_ptr<isae::LocalMap> map) {
        if (!_pub_vo_traj->get_subscription_count()) 
            return;

        _vo_traj_msg.header.stamp    = rclcpp::Node::now();
        _vo_traj_msg.header.frame_id = "world";
        _vo_traj_msg.points.clear();
        geometry_msgs::msg::Point p;

        for (auto &frame : map->getOldFramesPoses()) {
            const Eigen::Vector3d twc = frame.translation();
            p.x                       = twc.x();
            p.y                       = twc.y();
            p.z                       = twc.z();
            _vo_traj_msg.points.push_back(p);
        }

        for (auto &frame : map->getFrames()) {
            const Eigen::Vector3d twc = frame->getFrame2WorldTransform().translation();
            p.x                       = twc.x();
            p.y                       = twc.y();
            p.z                       = twc.z();
            _vo_traj_msg.points.push_back(p);
        }

        // publish message
        _pub_vo_traj->publish(_vo_traj_msg);
    }

    void publishLocalMapCloud(const std::shared_ptr<isae::LocalMap> map, const bool no_fov_mode = false) {
        if (!_pub_local_map_cloud->get_subscription_count() &&
            !_pub_local_map_cloud1->get_subscription_count() &&
            !_pub_local_map_lines->get_subscription_count()) {
               return;
        }
        isae::typed_vec_landmarks ldmks = map->getLandmarks();

        if (_pub_local_map_cloud->get_subscription_count() ||
            _pub_local_map_cloud1->get_subscription_count()) 
        {
            _points_local.header.frame_id    = "world";
            _points_local.header.stamp       = rclcpp::Node::now();
            _points_local.action             = visualization_msgs::msg::Marker::ADD;
            _points_local.pose.orientation.w = 1.0;

            _points_local1.header.frame_id    = "world";
            _points_local1.header.stamp       = rclcpp::Node::now();
            _points_local1.action             = visualization_msgs::msg::Marker::ADD;
            _points_local1.pose.orientation.w = 1.0;

            // build the point cloud from point3D lmks
            _points_local.points.clear();
            _points_local1.points.clear();

            for (auto &l : ldmks["pointxd"]) {
                if (l->isOutlier())
                    continue;
                Eigen::Vector3d pt3d = l->getPose().translation();

                geometry_msgs::msg::Point pt;
                pt.x = pt3d.x();
                pt.y = pt3d.y();
                pt.z = pt3d.z();

                if (no_fov_mode) {
                    if (l->getFeatures().at(0).lock()->getSensor() ==
                        l->getFeatures().at(0).lock()->getSensor()->getFrame()->getSensors().at(1))
                        _points_local1.points.push_back(pt);
                    else
                        _points_local.points.push_back(pt);
                } else
                    _points_local.points.push_back(pt);
            }

            _pub_local_map_cloud1->publish(_points_local1);
            _pub_local_map_cloud->publish(_points_local);
        }

        if (_pub_local_map_lines->get_subscription_count()) 
        {
            _lines_local.header.frame_id    = "world";
            _lines_local.header.stamp       = rclcpp::Node::now();
            _lines_local.action             = visualization_msgs::msg::Marker::ADD;
            _lines_local.pose.orientation.w = 1.0;

            // build the point cloud from line3D lmks
            _lines_local.points.clear();
            for (auto &l : ldmks["linexd"]) {
                if (l->isOutlier())
                    continue;
                Eigen::Affine3d T_w_ldmk                = l->getPose();
                std::vector<Eigen::Vector3d> ldmk_model = l->getModelPoints();
                for (const auto &p3d_model : ldmk_model) {
                    // conversion to the world coordinate system
                    Eigen::Vector3d t_w_lmk = T_w_ldmk * p3d_model.cwiseProduct(Eigen::Vector3d::Ones());
                    geometry_msgs::msg::Point pt;
                    pt.x = t_w_lmk.x();
                    pt.y = t_w_lmk.y();
                    pt.z = t_w_lmk.z();
                    _lines_local.points.push_back(pt);
                }
            }

            _pub_local_map_lines->publish(_lines_local);
        }
    }

    void publishGlobalMapCloud(const std::shared_ptr<isae::GlobalMap> map) {
        if (!_pub_global_map_cloud->get_subscription_count() &&
            !_pub_global_map_lines->get_subscription_count()) {
                return;
        }
        isae::typed_vec_landmarks ldmks = map->getLandmarks();

        _points_global.header.frame_id    = "world";
        _points_global.header.stamp       = rclcpp::Node::now();
        _points_global.action             = visualization_msgs::msg::Marker::ADD;
        _points_global.pose.orientation.w = 1.0;

        // build the point cloud from point3D lmks
        _points_global.points.clear();
        for (auto &l : ldmks["pointxd"]) {
            if (l->isOutlier())
                continue;
            Eigen::Vector3d pt3d = l->getPose().translation();
            geometry_msgs::msg::Point pt;
            pt.x = pt3d.x();
            pt.y = pt3d.y();
            pt.z = pt3d.z();
            _points_global.points.push_back(pt);
        }

        _pub_global_map_cloud->publish(_points_global);

        _lines_global.header.frame_id    = "world";
        _lines_global.header.stamp       = rclcpp::Node::now();
        _lines_global.action             = visualization_msgs::msg::Marker::ADD;
        _lines_global.pose.orientation.w = 1.0;

        // build the point cloud from line3D lmks
        _lines_global.points.clear();
        for (auto &l : ldmks["linexd"]) {
            if (l->isOutlier())
                continue;
            Eigen::Affine3d T_w_ldmk                = l->getPose();
            std::vector<Eigen::Vector3d> ldmk_model = l->getModelPoints();
            for (const auto &p3d_model : ldmk_model) {
                // conversion to the world coordinate system
                Eigen::Vector3d t_w_lmk = T_w_ldmk * p3d_model.cwiseProduct(Eigen::Vector3d::Ones());
                geometry_msgs::msg::Point pt;
                pt.x = t_w_lmk.x();
                pt.y = t_w_lmk.y();
                pt.z = t_w_lmk.z();
                _lines_global.points.push_back(pt);
            }
        }

        _pub_global_map_lines->publish(_lines_global);
    }

    void publishMesh(const std::shared_ptr<isae::Mesh3D> mesh) {
        if (!_pub_marker->get_subscription_count() &&
            !_pub_cloud->get_subscription_count()) {
                return;
        }

        _mesh_line_list.points.clear();
        _mesh_line_list.colors.clear();

        _mesh_line_list.header.frame_id = "relative";
        _mesh_line_list.header.stamp    = rclcpp::Node::now();

        for (auto &polygon : mesh->getPolygonVector()) {

            // Handles the points of the polygon
            std::vector<geometry_msgs::msg::Point> p_vector;
            std::vector<std_msgs::msg::ColorRGBA> c_vector;
            for (auto &vertex : polygon->getVertices()) {
                geometry_msgs::msg::Point p;
                std_msgs::msg::ColorRGBA color;
                Eigen::Vector3d lmk_coord = vertex->getVertexPosition();

                p.x = lmk_coord.x();
                p.y = lmk_coord.y();
                p.z = lmk_coord.z();
                p_vector.push_back(p);

                // Color triangle with its slope
                double trav_score = polygon->getPolygonNormal().dot(Eigen::Vector3d(0, 0, 1));
                color.r           = (1 - trav_score);
                color.g           = trav_score;
                color.b           = 0;
                color.a           = 1.0;

                c_vector.push_back(color);
            }

            // Set the lines of the polygon
            _mesh_line_list.points.push_back(p_vector.at(0));
            _mesh_line_list.colors.push_back(c_vector.at(0));
            _mesh_line_list.points.push_back(p_vector.at(1));
            _mesh_line_list.colors.push_back(c_vector.at(1));
            _mesh_line_list.points.push_back(p_vector.at(2));
            _mesh_line_list.colors.push_back(c_vector.at(2));
        }

        _pub_marker->publish(_mesh_line_list);

        // Publish the dense point cloud from the mesh 3D
        std::vector<Eigen::Vector3d> pt_cloud = mesh->getPointCloud();
        if (pt_cloud.empty())
            return;

        sensor_msgs::msg::PointCloud2::SharedPtr pc2_msg_ = std::make_shared<sensor_msgs::msg::PointCloud2>();
        pc2_msg_                                          = convertToPointCloud2(pt_cloud);

        _pub_cloud->publish(*pc2_msg_);
    }

    sadvio_msgs::msg::SLAMParam fillParamMsg(std::shared_ptr<isae::SLAMParameters> _slam_param) {
        sadvio_msgs::msg::SLAMParam param_msg;
        param_msg.dataset_path      = _slam_param->_config.dataset_path;
        param_msg.dataset_id        = _slam_param->_config.dataset_id;
        param_msg.slam_mode         = _slam_param->_config.slam_mode;
        param_msg.multithreading    = _slam_param->_config.multithreading;
        param_msg.enable_visu       = _slam_param->_config.enable_visu;
        param_msg.optimizer         = _slam_param->_config.optimizer;
        param_msg.contrast_enhancer = _slam_param->_config.contrast_enhancer;
        param_msg.clahe_clip        = _slam_param->_config.clahe_clip;
        param_msg.downsampling      = _slam_param->_config.downsampling;
        param_msg.marginalization   = _slam_param->_config.marginalization;
        param_msg.sparsification    = _slam_param->_config.sparsification;
        param_msg.pose_estimator    = _slam_param->_config.pose_estimator;
        param_msg.rel_pose_estimator = _slam_param->_config.rel_pose_estimator;
        param_msg.eskf_r            = _slam_param->_config.eskf_r;
        param_msg.tracker           = _slam_param->_config.tracker;
        param_msg.min_kf_number     = _slam_param->_config.min_kf_number;
        param_msg.max_kf_number     = _slam_param->_config.max_kf_number;
        param_msg.fixed_frame_number = _slam_param->_config.fixed_frame_number;
        param_msg.min_lmk_number    = _slam_param->_config.min_lmk_number;
        param_msg.min_movement_parallax = _slam_param->_config.min_movement_parallax;
        param_msg.max_movement_parallax = _slam_param->_config.max_movement_parallax;
        param_msg.mesh3d            = _slam_param->_config.mesh3D;
        param_msg.zncc_tsh          = _slam_param->_config.ZNCC_tsh;
        param_msg.max_length_tsh    = _slam_param->_config.max_length_tsh;

        for (auto ft : _slam_param->_config.features_handled) {
            sadvio_msgs::msg::SLAMFeatures feature_hndl_msg;
            feature_hndl_msg.label_feature          = ft.label_feature;
            feature_hndl_msg.detector_label         = ft.detector_label;
            feature_hndl_msg.number_detected_features = ft.number_detected_features;
            feature_hndl_msg.n_features_per_cell    = ft.n_features_per_cell;
            feature_hndl_msg.tracker_label          = ft.tracker_label;
            feature_hndl_msg.tracker_height         = ft.tracker_height;
            feature_hndl_msg.tracker_width          = ft.tracker_width;
            feature_hndl_msg.tracker_nlvls_pyramids = ft.tracker_nlvls_pyramids;
            feature_hndl_msg.tracker_max_err        = ft.tracker_max_err;
            feature_hndl_msg.matcher_label          = ft.matcher_label;
            feature_hndl_msg.max_matching_dist      = ft.max_matching_dist;
            feature_hndl_msg.matcher_height         = ft.matcher_height;
            feature_hndl_msg.matcher_width          = ft.matcher_width;
            feature_hndl_msg.lmk_triangulator       = ft.lmk_triangulator;
            param_msg.features_handled.push_back(feature_hndl_msg);
        }
        return param_msg;
    }

    void publishParam(std::shared_ptr<isae::SLAMParameters> _slam_param) {
        if (!_pub_param->get_subscription_count()) 
            return;
        sadvio_msgs::msg::SLAMParam param_msg = fillParamMsg(_slam_param);
        _pub_param->publish(param_msg);
    }
    
    uint publishParam(std::shared_ptr<isae::SLAMParameters> _slam_param, const std::shared_ptr<isae::Frame> frame) {
        if (!_pub_param->get_subscription_count()) 
            return 0;
        sadvio_msgs::msg::SLAMParam param_msg = fillParamMsg(_slam_param);
        unsigned long long ts = frame->getTimestamp();
        param_msg.header.stamp.sec = ts/1000000000u;
        param_msg.header.stamp.nanosec = ts % 1000000000u;
        _pub_param->publish(param_msg);
        return frame->_id;
    }

    void publishRelativeTF(const std::shared_ptr<isae::LocalMap> map) {
        if (!_pub_tf_rel->get_subscription_count()) 
            return;
        
        if (map->getFrames().size() > 1) {
            std::shared_ptr<isae::Frame> f = map->getFrames().back(); // newest
            std::shared_ptr<isae::Frame> f_prev = map->getFrames().at(map->getFrames().size()-2); // second-to-last (second-newest)
            
            Eigen::Affine3d T_f1_f2;
            Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(6, 6);

            map->computeRelativePose(f_prev, f, T_f1_f2, cov);
            Eigen::Vector3d r = isae::geometry::log_so3(T_f1_f2.rotation());
            Eigen::Vector3d t = T_f1_f2.translation();

            unsigned long long ts = f->getTimestamp();
            _pose_rel_tf.header.stamp.sec = ts/1000000000u;
            _pose_rel_tf.header.stamp.nanosec = ts % 1000000000u;
            _pose_rel_tf.header.frame_id = "relative";
            _pose_rel_tf.pose.pose.position.x = t.x();
            _pose_rel_tf.pose.pose.position.y = t.y();
            _pose_rel_tf.pose.pose.position.z = t.z();

            Eigen::Quaterniond q = Eigen::Quaterniond(T_f1_f2.rotation());
            _pose_rel_tf.pose.pose.orientation.x = q.x();
            _pose_rel_tf.pose.pose.orientation.y = q.y();
            _pose_rel_tf.pose.pose.orientation.z = q.z();
            _pose_rel_tf.pose.pose.orientation.w = q.w();

            std::array<double, 36UL> cov_pub;
            Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> C(cov_pub.data());
            C = cov;
            _pose_rel_tf.pose.covariance = cov_pub;
            _pub_tf_rel->publish(_pose_rel_tf);
        }
    }

    void runVisualizer(std::shared_ptr<isae::SLAMCore> SLAM) {

        uint old_frame_count = 0;
        while (true) {

            if (SLAM->_frame_to_display) {
                publishImage(SLAM->_frame_to_display);
                publishFrame(SLAM->_frame_to_display);

                if (SLAM->_frame_to_display->_id - old_frame_count > 10)
                    old_frame_count = publishParam(SLAM->_slam_param, SLAM->_frame_to_display);

                SLAM->_frame_to_display.reset();
            }

            if (SLAM->_local_map_to_display) {
                publishLocalMap(SLAM->_local_map_to_display);
                publishLocalMapCloud(SLAM->_local_map_to_display);
                publishRelativeTF(SLAM->_local_map_to_display);
                SLAM->_local_map_to_display.reset();
            }

            if (SLAM->_mesh_to_display) {
                publishMesh(SLAM->_mesh_to_display);
                SLAM->_mesh_to_display.reset();
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr _pub_marker, _pub_vo_traj, _pub_global_map_cloud,
        _pub_local_map_cloud, _pub_local_map_cloud1, _pub_global_map_lines, _pub_local_map_lines;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _pub_image_kps, _pub_image_matches_in_time,
        _pub_image_matches_in_frame;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _pub_cloud;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _pub_vo_pose;
    rclcpp::Publisher<sadvio_msgs::msg::SLAMParam>::SharedPtr _pub_param;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr _pub_tf_rel;
    std::shared_ptr<tf2_ros::TransformBroadcaster> _tf_broadcaster;

    visualization_msgs::msg::Marker _vo_traj_msg;
    visualization_msgs::msg::Marker _points_local, _points_global, _points_local1, _lines_local, _lines_global;
    visualization_msgs::msg::Marker _mesh_line_list;
    
    geometry_msgs::msg::PoseWithCovarianceStamped _pose_rel_tf;
};

// } // namespace isae

#endif // ROSVISUALIZER_H