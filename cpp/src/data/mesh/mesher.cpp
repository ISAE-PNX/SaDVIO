#include "isaeslam/data/mesh/mesher.h"

#include <opencv2/highgui/highgui.hpp>

namespace isae {

// Draw delaunay triangles
static void draw_delaunay(cv::Mat &img, std::vector<FeatPolygon> tri_feat_vector) {

    std::vector<cv::Point> pt(3);
    cv::Size size = img.size();
    cv::Rect rect(0, 0, size.width, size.height);
    cv::Scalar delaunay_color(0, 255, 0);
    cv::Scalar kp_color(255, 0, 0);

    for (auto feat : tri_feat_vector) {
        pt[0] = cv::Point(feat.at(0)->getPoints().at(0)(0), feat.at(0)->getPoints().at(0)(1));
        pt[1] = cv::Point(feat.at(1)->getPoints().at(0)(0), feat.at(1)->getPoints().at(0)(1));
        pt[2] = cv::Point(feat.at(2)->getPoints().at(0)(0), feat.at(2)->getPoints().at(0)(1));

        // Draw rectangles completely inside the image.
        cv::line(img, pt[0], pt[1], delaunay_color, 2, cv::LINE_8, 0);
        cv::line(img, pt[1], pt[2], delaunay_color, 2, cv::LINE_8, 0);
        cv::line(img, pt[2], pt[0], delaunay_color, 2, cv::LINE_8, 0);

        // Draw keypoints
        cv::circle(img, cv::Point(pt[0].x, pt[0].y), 4, kp_color, -1);
        cv::circle(img, cv::Point(pt[1].x, pt[1].y), 4, kp_color, -1);
        cv::circle(img, cv::Point(pt[2].x, pt[2].y), 4, kp_color, -1);
    }
}

Mesher::Mesher(std::string slam_mode, double ZNCC_tsh, double max_length_tsh) : _slam_mode(slam_mode) {

    // Init mesh 3D
    _mesh_3d    = std::make_shared<Mesh3D>(ZNCC_tsh, max_length_tsh);
    _avg_mesh_t = 0;
    _n_kf       = 0;

    // For timing statistics
    // std::ofstream fw_prof_mesh("log_slam/timing_mesh.csv",
    //                            std::ofstream::out | std::ofstream::trunc);
    // fw_prof_mesh << "mesh_dt\n";
    // fw_prof_mesh.close();

    // Launch mesher thread
    std::thread mesher_thread(&Mesher::run, this);
    mesher_thread.detach();
}

void Mesher::addNewKF(std::shared_ptr<Frame> frame) {
    std::lock_guard<std::mutex> lock(_mesher_mtx);
    _kf_queue.push(frame);
}

bool Mesher::getNewKf() {

    // Check if a KF is available
    if (_kf_queue.empty()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(_mesher_mtx);
    // Add the KF in the queue
    _curr_kf = _kf_queue.front();
    _kf_queue.pop();

    return true;
}

void Mesher::run() {

    while (true) {

        if (getNewKf()) {

            // For profiling
            _n_kf++;
            auto t0 = std::chrono::high_resolution_clock::now();

            _mesh_3d->updateMesh(this->createMesh2D(_curr_kf->getSensors().at(0)), _curr_kf);
            if (_slam_mode == "nofov")
                _mesh_3d->updateMesh(this->createMesh2D(_curr_kf->getSensors().at(1)), _curr_kf);

            auto t1     = std::chrono::high_resolution_clock::now();
            double dt   = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            _avg_mesh_t = (_avg_mesh_t * ((double)_n_kf - 1) + dt) / (double)_n_kf;

            // std::ofstream fw_mesh("log_slam/timing_mesh.csv",
            //                       std::ofstream::out | std::ofstream::app);
            // fw_mesh << dt << "\n";
            // fw_mesh.close();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

std::vector<cv::Vec6f> Mesher::computeMesh2D(const cv::Size img_size,
                                             const std::vector<cv::Point2f> p2f_to_triangulate) {

    // Nothing to triangulate.
    if (p2f_to_triangulate.size() == 0)
        return std::vector<cv::Vec6f>();

    // Rectangle to be used with Subdiv2D.
    // https://answers.opencv.org/question/180984/out-of-range-error-in-delaunay-triangulation/
    static const cv::Rect2f rect(0.0, 0.0, img_size.width, img_size.height);
    // subdiv has the delaunay triangulation function
    cv::Subdiv2D subdiv(rect);
    subdiv.initDelaunay(rect);

    // Maybe a check for p2f inside the image check needed here
    std::vector<cv::Point2f> p2f_inside;
    p2f_inside.reserve(p2f_to_triangulate.size());
    for (const cv::Point2f p2f : p2f_to_triangulate) {
        if (rect.contains(p2f)) {
            p2f_inside.push_back(p2f);
        }
    }

    // Perform 2D Delaunay triangulation.
    subdiv.insert(p2f_inside);

    std::vector<cv::Vec6f> tri_2d;
    subdiv.getTriangleList(tri_2d);

    // TODO: link between features and triangles
    // Maybe another check for vertices inside the image

    return tri_2d;
}

std::vector<FeatPolygon> Mesher::createMesh2D(std::shared_ptr<ImageSensor> sensor) {

    // Select Pointxd features with lmk for mesh2D
    std::vector<std::shared_ptr<AFeature>> features_to_triangulate;
    for (auto feat : sensor->getFeatures()["pointxd"]) {

        // If the feature has an inlier landmark, it will be in the 2D mesh
        if (feat->getLandmark().lock()) {

            // If the feature is too far away, it is ignored
            Eigen::Vector3d t_c_l =
                sensor->getWorld2SensorTransform() * feat->getLandmark().lock()->getPose().translation();

            if (t_c_l.norm() > 10)
                continue;

            if (feat->getLandmark().lock()->isOutlier() || !feat->getLandmark().lock()->isInMap() ||
                !feat->getLandmark().lock()->isInitialized())
                continue;

            features_to_triangulate.push_back(feat);
        }
    }

    cv::Size img_size = sensor->getRawData().size();

    // Convert features to cv point2f
    std::vector<cv::Point2f> p2f_to_triangulate;
    AFeatureDetector::FeatureToP2f(features_to_triangulate, p2f_to_triangulate);

    std::vector<cv::Vec6f> tri_2d;
    tri_2d = computeMesh2D(img_size, p2f_to_triangulate);

    // Build the triangle feature vector
    std::vector<FeatPolygon> tri_feat_vector;
    for (auto tri : tri_2d) {
        FeatPolygon tri_feat;
        std::vector<std::shared_ptr<AFeature>>::iterator feat_it;

        Eigen::Vector2d p0;
        p0 << tri[0], tri[1];
        for (feat_it = features_to_triangulate.begin(); feat_it != features_to_triangulate.end(); feat_it++) {

            if ((feat_it->get()->getPoints().at(0) - p0).norm() < 1e-4) {
                tri_feat.push_back(feat_it->get()->shared_from_this());
                break;
            }
        }

        Eigen::Vector2d p1;
        p1 << tri[2], tri[3];
        for (feat_it = features_to_triangulate.begin(); feat_it != features_to_triangulate.end(); feat_it++) {

            if ((feat_it->get()->getPoints().at(0) - p1).norm() < 1e-4) {
                tri_feat.push_back(feat_it->get()->shared_from_this());
                break;
            }
        }

        Eigen::Vector2d p2;
        p2 << tri[4], tri[5];
        for (feat_it = features_to_triangulate.begin(); feat_it != features_to_triangulate.end(); feat_it++) {

            if ((feat_it->get()->getPoints().at(0) - p2).norm() < 1e-4) {
                tri_feat.push_back(feat_it->get()->shared_from_this());
                break;
            }
        }
        tri_feat_vector.push_back(tri_feat);
    }

    // Draw for debug
    bool debug = false;
    if (debug) {
        cv::Mat img_copy = sensor->getRawData().clone();
        cv::Mat img_color;
        cv::cvtColor(img_copy, img_color, cv::COLOR_GRAY2RGB);
        draw_delaunay(img_color, tri_feat_vector);
        std::string title = "delaunay.png";
        cv::imwrite(title, img_color);
    }

    return tri_feat_vector;
}

} // namespace isae
