#include "isaeslam/data/mesh/mesh.h"

namespace isae {

void Mesh3D::updateMesh(std::vector<FeatPolygon> feats_polygon, std::shared_ptr<Frame> frame) {

    // Set the reference frame and a copy to all important variables
    _reference_frame = frame;
    _cam0            = _reference_frame->getSensors().at(0);
    _img0            = _cam0->getRawData().clone();
    if (_reference_frame->getSensors().at(1)) {
        _cam1 = _reference_frame->getSensors().at(1);
        _img1 = _cam1->getRawData().clone();
    } else {
        _cam1 = nullptr;
    }
    _T_w_cam0 = _cam0->getSensor2WorldTransform();

    // Clear every triangle with a marginalized / discarded lmk
    for (auto polygon : _polygons) {

        bool to_remove = false;
        for (auto vertex : polygon->getVertices()) {
            if (vertex->getLmk()->isMarg() || vertex->getLmk()->isOutlier()) {
                to_remove = true;
                if (_map_lmk_vertex.find(vertex->getLmk()) != _map_lmk_vertex.end()) {
                    _map_lmk_vertex.erase(vertex->getLmk());
                }
            }
        }

        // Remove the polygon on every vertex
        if (to_remove) {
            for (auto vertex : polygon->getVertices()) {
                vertex->removePolygon(polygon);
            }
            polygon->setOutlier();
        }
    }

    // Remove outliers from _polygons
    _polygons.erase(std::remove_if(_polygons.begin(),
                                   _polygons.end(),
                                   [](std::shared_ptr<Polygon> pol) { return pol->isOutlier(); }),
                    _polygons.end());

    // Add every new triangles from the current mesh2D
    for (auto feat_polygon : feats_polygon) {

        // Fill a lmk polygon with all the lmk of the features
        std::vector<std::shared_ptr<Vertex>> vertices;

        for (auto feat : feat_polygon) {

            // Add vertex if it is not already in the mesh
            if (_map_lmk_vertex.find(feat->getLandmark().lock()) == _map_lmk_vertex.end()) {
                _map_lmk_vertex.emplace(feat->getLandmark().lock(),
                                        std::make_shared<Vertex>(feat->getLandmark().lock()));
            }
            vertices.push_back(_map_lmk_vertex.at(feat->getLandmark().lock()));
        }

        // Check if it is already in the mesh3D TODO
        bool is_in_mesh3D = false;
        for (auto polygon : _polygons) {
            if (polygon->getVertices() == vertices) {
                is_in_mesh3D = true;
                break;
            }
        }

        if (!is_in_mesh3D && checkTriangle(vertices)) {

            // Create and analyse the polygon
            std::shared_ptr<Polygon> polygon = std::make_shared<Polygon>(vertices);

            // Compute Area for checkPolygonArea
            // std::vector<Eigen::Vector2d> tri2d;
            // for (auto feat : feat_polygon)
            //     tri2d.push_back(feat->getPoints().at(0));
            // double area2d = geometry::areaTriangle(tri2d);

            analysePolygon(polygon);

            if (checkPolygon(polygon)) {
                _polygons.push_back(polygon);

                // Add it to every vertex
                for (auto vertex : vertices) {
                    vertex->addPolygon(_polygons.back());
                }
            }
        }
    }

    this->projectMesh();
    this->generatePointCloud();
}

void Mesh3D::filterMesh() {

    for (auto polygon : _polygons) {
        Eigen::Vector3d avg_normal = Eigen::Vector3d::Zero();
        int n_polygons             = 0;

        for (auto vertex : polygon->getVertices()) {

            for (auto poly_adj : vertex->getPolygons()) {
                if (!poly_adj->isOutlier()) {
                    avg_normal += poly_adj->getPolygonNormal();
                    n_polygons++;
                }
            }

            // outlier if one of the vertices is an outlier
            if (vertex->getLmk()->isOutlier())
                polygon->setOutlier();
        }

        // Average normal of all the surroundings polygons
        avg_normal /= (double)n_polygons;

        // If the polygon has a very different normal wrt it neighbours it is removed
        // TODO: would be better with a chi2 test
        if (polygon->getPolygonNormal().dot(avg_normal) * 180 / M_PI > 60)
            polygon->setOutlier();

        // In that case the polygon is lonely
        if (n_polygons == 0)
            polygon->setOutlier();
    }

    // Remove outliers from _polygons
    _polygons.erase(std::remove_if(_polygons.begin(),
                                   _polygons.end(),
                                   [](std::shared_ptr<Polygon> pol) { return pol->isOutlier(); }),
                    _polygons.end());
}

void Mesh3D::analysePolygon(std::shared_ptr<Polygon> polygon) {

    // Compute Normal
    Eigen::Vector3d b1 =
        polygon->getVertices().at(1)->getVertexPosition() - polygon->getVertices().at(0)->getVertexPosition();
    Eigen::Vector3d b2 =
        polygon->getVertices().at(2)->getVertexPosition() - polygon->getVertices().at(0)->getVertexPosition();
    Eigen::Vector3d normal = b2.cross(b1);

    // Compute barycenter
    Eigen::Vector3d barycenter =
        (polygon->getVertices().at(0)->getVertexPosition() + polygon->getVertices().at(1)->getVertexPosition() +
         polygon->getVertices().at(0)->getVertexPosition()) /
        3;
    polygon->setBarycenter(barycenter);

    // Put the normal in the direction of the robot
    Eigen::Vector3d barycenter_f = _reference_frame->getWorld2FrameTransform() * barycenter;
    Eigen::Vector3d normal_f     = _reference_frame->getWorld2FrameTransform().rotation() * normal;
    if (barycenter_f.dot(normal_f) > 0)
        normal = -normal;
    polygon->setNormal(normal.normalized());

    // Find 2D coordinates of the points
    // Eigen::Vector3d b1_norm = b1.normalized();
    // Eigen::Vector2d u1(b1.norm(), 0);
    // Eigen::Vector2d u2(b1_norm.dot(b2), (b2 - b1_norm.dot(b2) * b1_norm).norm());
    // std::vector<Eigen::Vector2d> tri_2d;
    // tri_2d.push_back(Eigen::Vector2d::Zero());
    // tri_2d.push_back(u1);
    // tri_2d.push_back(u2);

    // // Compute the transformation matrix
    // Eigen::Matrix2d M = geometry::cov2dTriangle(tri_2d);
    // polygon->setCovariance(M);

    // // Compute score (now it gives the quality of the triangle)
    // Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // double ratio = M.trace() * (svd.singularValues()(1) / svd.singularValues()(0));
    // // Eigen::Vector3d ground_normal(0.0, 0.0, 1.0);
    // // double normal_dot_ground = std::abs(ground_normal.dot(normal.normalized()));
    // // if (normal_dot_ground < 0.0)
    // //     polygon->setScore(0);
    // // else
    // //     polygon->setScore(normal_dot_ground);
    // polygon->setScore(ratio);
}

bool Mesh3D::checkTriangle(std::vector<std::shared_ptr<Vertex>> vertices) {

    if (vertices.size() != 3)
        return false;

    if (vertices.at(0) == vertices.at(1) || vertices.at(1) == vertices.at(2) || vertices.at(2) == vertices.at(0))
        return false;

    Eigen::Vector3d p0 = vertices.at(0)->getVertexPosition();
    Eigen::Vector3d p1 = vertices.at(1)->getVertexPosition();
    Eigen::Vector3d p2 = vertices.at(2)->getVertexPosition();

    // Check if there is no acute angles
    double angle0 = geometry::getAngle(p0, p1, p2) * 180 / M_PI;
    if (angle0 > 160 || angle0 < 20)
        return false;

    double angle1 = geometry::getAngle(p1, p0, p2) * 180 / M_PI;
    if (angle1 > 160 || angle1 < 20)
        return false;

    double angle2 = geometry::getAngle(p2, p0, p1) * 180 / M_PI;
    if (angle2 > 160 || angle2 < 20)
        return false;

    // Check if there is not too long side
    Eigen::Vector3d c0 = p1 - p0;
    Eigen::Vector3d c1 = p2 - p0;
    Eigen::Vector3d c2 = p2 - p1;

    if (c0.norm() > _max_length_tsh || c1.norm() > _max_length_tsh || c2.norm() > _max_length_tsh)
        return false;
    return true;
}

void Mesh3D::projectMesh() {

    _map_poly_tri2d.clear();

    // Project every polygon in sensor 0 of the current frame
    for (auto polygon : _polygons) {

        if (polygon->isOutlier())
            continue;

        std::vector<Eigen::Vector2d> triangle_2d;
        for (auto &vtx : polygon->getVertices()) {


            // Prepare variables for projection
            std::vector<Eigen::Vector2d> p2ds;
            Eigen::Affine3d T_w_vtx = Eigen::Affine3d::Identity();
            T_w_vtx.translation()   = vtx->getVertexPosition();


            // Check only infinite values (we can use triangles outside of the image)
            if (_cam0->project(T_w_vtx, vtx->getLmk()->getModel(), Eigen::Vector3d::Ones(), p2ds)) {

                triangle_2d.push_back(p2ds.at(0));
            }
        }

        // Only consider triangles (TODO also use triangles with vertices out of the image)
        if (triangle_2d.size() == 3)
            _map_poly_tri2d.emplace(polygon, triangle_2d);
        else
            continue;
    }
}

bool Mesh3D::checkPolygonTri(std::shared_ptr<Polygon> polygon3d, FeatPolygon polygon2d) {

    // Available only in stereo mode
    if (_reference_frame->getSensors().size() == 1)
        return true;

    Eigen::Vector3d t_w_cam0 = _T_w_cam0.translation();

    // Let's get info about the triangle
    Eigen::Vector3d n     = polygon3d->getPolygonNormal();
    Eigen::Vector3d t_w_a = polygon3d->getVertices().at(0)->getVertexPosition();

    // Get all the pixels in the triangle
    std::vector<Eigen::Vector2d> tri2d, pixels_in_tri2d;
    std::vector<uint8_t> pixels_img0, pixels_img1;

    // Get Umin, Vmin, Umax, Vmax
    int umin = 10000;
    int umax = 0;
    int vmin = 10000;
    int vmax = 0;
    for (auto feat : polygon2d) {
        Eigen::Vector2d pt = feat->getPoints().at(0);
        tri2d.push_back(pt);

        if (pt(0) < umin)
            umin = (int)pt(0);
        if (pt(1) < vmin)
            vmin = (int)pt(1);
        if (pt(0) > umax)
            umax = (int)pt(0);
        if (pt(1) > vmax)
            vmax = (int)pt(1);
    }

    // parse the square [[Umin, Vmin], [Umax, Vmax]]
    cv::Mat patch0 = cv::Mat::zeros(cv::Size(umax - umin, vmax - vmin), CV_8U);
    cv::Mat patch1 = cv::Mat::zeros(cv::Size(umax - umin, vmax - vmin), CV_8U);
    for (int u = umin; u < umax; u++) {
        for (int v = vmin; v < vmax; v++) {
            Eigen::Vector2d pt(u, v);

            if (geometry::pointInTriangle(pt, tri2d)) {

                Eigen::Vector3d bearing_vec = _cam0->getRay(pt);
                double depth                = (1 / bearing_vec.dot(n)) * (t_w_a.dot(n) - t_w_cam0.dot(n));
                Eigen::Vector3d p3d         = t_w_cam0 + depth * bearing_vec;
                std::vector<Eigen::Vector2d> p2ds1;

                // Project the pixel in cam1
                Eigen::Affine3d T_w_p3d = Eigen::Affine3d::Identity();
                T_w_p3d.translation()   = p3d;

                if (_cam1->project(T_w_p3d,
                                   polygon3d->getVertices().at(0)->getLmk()->getModel(),
                                   Eigen::Vector3d::Ones(),
                                   p2ds1)) {
                    pixels_img1.push_back(_img1.at<uint8_t>(p2ds1.at(0)(1), p2ds1.at(0)(0)));
                    pixels_img0.push_back(_img0.at<uint8_t>(pt(1), pt(0)));
                    patch0.at<uint8_t>(v - vmin, u - umin) = _img0.at<uint8_t>(pt(1), pt(0));
                    patch1.at<uint8_t>(v - vmin, u - umin) = _img1.at<uint8_t>(p2ds1.at(0)(1), p2ds1.at(0)(0));
                }
            }
        }
    }

    // Compute ZNCC
    cv::Mat pix_img0 = cv::Mat::zeros(pixels_img0.size(), 1, CV_8U);
    cv::Mat pix_img1 = cv::Mat::zeros(pixels_img0.size(), 1, CV_8U);
    for (uint i = 0; i < pixels_img0.size(); i++) {
        pix_img0.at<uint8_t>(i, 0) = pixels_img0.at(i);
        pix_img1.at<uint8_t>(i, 0) = pixels_img1.at(i);
    }

    // Let's compute the distance between the patches
    double errorZNCC = isae::imgproc::ZNCC(pix_img0, pix_img1);
    polygon3d->setScore(errorZNCC);
    bool debug = false;
    if (debug) {
        cv::imwrite(std::to_string(errorZNCC) + "_0.png", patch0);
        cv::imwrite(std::to_string(errorZNCC) + "_1.png", patch1);
    }

    if (errorZNCC < _ZNCC_tsh || !std::isfinite(errorZNCC))
        return false;
    else
        return true;

    return true;
}

bool Mesh3D::checkPolygonArea(std::shared_ptr<Polygon> polygon, double area2d) {

    // Available only in stereo mode
    if (_reference_frame->getSensors().size() == 1)
        return true;

    // Project every polygon in sensor 0 of the current frame
    Eigen::Vector3d t_w_cam0 = _T_w_cam0.translation();

    // Let's get info about the triangle
    Eigen::Vector3d n     = polygon->getPolygonNormal();
    Eigen::Vector3d t_w_a = polygon->getVertices().at(0)->getVertexPosition();

    // Let's compute the two barycenters
    Eigen::Affine3d T_w_b = Eigen::Affine3d::Identity();
    T_w_b.translation()   = polygon->getBarycenter();
    std::vector<Eigen::Vector2d> p2ds;
    Eigen::Vector2d b0, b1;

    if (!_cam0->project(T_w_b, polygon->getVertices().at(0)->getLmk()->getModel(), Eigen::Vector3d::Ones(), p2ds))
        return false;
    b0 = p2ds.at(0);

    // Let's compute the patch and the depths around the barycenter
    int patch_size = (int)(std::sqrt(area2d));
    patch_size     = 15;
    if (patch_size % 2 == 0) // The patch size must be odd
        patch_size++;
    int half_patch_size = (patch_size - 1) / 2;
    std::vector<Eigen::Vector3d> pixels_3d;

    // false if the patch goes out of the image
    if ((int)b0(0) - half_patch_size < 1 || (int)b0(1) - half_patch_size < 1 ||
        (int)b0(0) + half_patch_size > _img0.cols - 1 || (int)b0(1) + half_patch_size > _img0.rows - 1)
        return false;

    cv::Mat patch0, patch1;
    patch0 =
        cv::Mat(_img0, cv::Rect((int)b0(0) - half_patch_size, (int)b0(1) - half_patch_size, patch_size, patch_size));
    patch1 = cv::Mat::zeros(cv::Size(patch_size, patch_size), _img0.type());

    for (int i = -half_patch_size; i < half_patch_size + 1; i++) {
        for (int j = -half_patch_size; j < half_patch_size + 1; j++) {
            Eigen::Vector2d p   = Eigen::Vector2d((int)b0(0) + i, (int)b0(1) + j);
            Eigen::Vector3d v   = _cam0->getRay(p);
            double depth        = (1 / v.dot(n)) * (t_w_a.dot(n) - t_w_cam0.dot(n));
            Eigen::Vector3d p3d = t_w_cam0 + depth * v;
            std::vector<Eigen::Vector2d> p2ds1;

            // Project the pixel in cam1
            Eigen::Affine3d T_w_p3d = Eigen::Affine3d::Identity();
            T_w_p3d.translation()   = p3d;

            if (_cam1->project(
                    T_w_p3d, polygon->getVertices().at(0)->getLmk()->getModel(), Eigen::Vector3d::Ones(), p2ds1))
                patch1.at<uint8_t>(half_patch_size + j, half_patch_size + i) =
                    _img1.at<uint8_t>((int)p2ds1.at(0)(1), (int)p2ds1.at(0)(0));
            else
                return false;

            if (i == 0 && j == 0) {
                b1 = p2ds1.at(0);
            }
        }
    }

    // Let's compute the distance between the patches
    double errorZNCC = isae::imgproc::ZNCC(patch0, patch1);
    polygon->setScore(errorZNCC);

    bool debug = false;
    if (debug) {
        cv::Mat img_color0, img_color1;
        cv::cvtColor(_cam1->getRawData().clone(), img_color1, cv::COLOR_GRAY2RGB);
        cv::circle(img_color1, cv::Point2d(b1(0), b1(1)), 11, cv::Scalar(0, 255, 0));
        cv::imwrite(std::to_string(errorZNCC) + "_0.png", patch0);
        cv::imwrite(std::to_string(errorZNCC) + "_1.png", patch1);
    }

    if (errorZNCC < _ZNCC_tsh || !std::isfinite(errorZNCC))
        return false;
    else
        return true;
}

bool Mesh3D::checkPolygon(std::shared_ptr<Polygon> polygon) {

    // Available only in stereo mode
    if (_reference_frame->getSensors().size() == 1)
        return true;

    // Project every polygon in sensor 0 of the current frame
    Eigen::Vector3d t_w_cam0 = _T_w_cam0.translation();

    // Let's get info about the triangle
    Eigen::Vector3d n     = polygon->getPolygonNormal();
    Eigen::Vector3d t_w_a = polygon->getVertices().at(0)->getVertexPosition();

    // Let's compute the two barycenters
    Eigen::Affine3d T_w_b = Eigen::Affine3d::Identity();
    T_w_b.translation()   = polygon->getBarycenter();
    std::vector<Eigen::Vector2d> p2ds;
    Eigen::Vector2d b0, b1;

    if (!_cam0->project(T_w_b, polygon->getVertices().at(0)->getLmk()->getModel(), Eigen::Vector3d::Ones(), p2ds))
        return false;
    b0 = p2ds.at(0);

    // Let's compute the patch and the depths around the barycenter
    int patch_size      = 15;
    int half_patch_size = (patch_size - 1) / 2;
    std::vector<Eigen::Vector3d> pixels_3d;

    // false if the patch goes out of the image
    if ((int)b0(0) - half_patch_size < 1 || (int)b0(1) - half_patch_size < 1 ||
        (int)b0(0) + half_patch_size > _img0.cols - 1 || (int)b0(1) + half_patch_size > _img0.rows - 1)
        return true;

    cv::Mat patch0, patch1;
    patch0 =
        cv::Mat(_img0, cv::Rect((int)b0(0) - half_patch_size, (int)b0(1) - half_patch_size, patch_size, patch_size));
    patch1 = cv::Mat::zeros(cv::Size(patch_size, patch_size), _img0.type());

    for (int i = -half_patch_size; i < half_patch_size + 1; i++) {
        for (int j = -half_patch_size; j < half_patch_size + 1; j++) {
            Eigen::Vector2d p   = Eigen::Vector2d((int)b0(0) + i, (int)b0(1) + j);
            Eigen::Vector3d v   = _cam0->getRay(p);
            double depth        = (1 / v.dot(n)) * (t_w_a.dot(n) - t_w_cam0.dot(n));
            Eigen::Vector3d p3d = t_w_cam0 + depth * v;
            std::vector<Eigen::Vector2d> p2ds1;

            // Project the pixel in cam1
            Eigen::Affine3d T_w_p3d = Eigen::Affine3d::Identity();
            T_w_p3d.translation()   = p3d;

            if (_cam1->project(
                    T_w_p3d, polygon->getVertices().at(0)->getLmk()->getModel(), Eigen::Vector3d::Ones(), p2ds1))
                patch1.at<uint8_t>(half_patch_size + j, half_patch_size + i) =
                    _img1.at<uint8_t>((int)p2ds1.at(0)(1), (int)p2ds1.at(0)(0));
            else
                return true;

            if (i == 0 && j == 0) {
                b1 = p2ds1.at(0);
            }
        }
    }

    // Let's compute the distance between the patches
    double errorZNCC = isae::imgproc::ZNCC(patch0, patch1);
    polygon->setScore(errorZNCC);

    bool debug = false;
    if (debug) {
        cv::Mat img_color0, img_color1;
        cv::cvtColor(_cam1->getRawData().clone(), img_color1, cv::COLOR_GRAY2RGB);
        cv::circle(img_color1, cv::Point2d(b1(0), b1(1)), 11, cv::Scalar(0, 255, 0));
        cv::imwrite(std::to_string(errorZNCC) + "_0.png", patch0);
        cv::imwrite(std::to_string(errorZNCC) + "_1.png", patch1);
    }

    if (errorZNCC < _ZNCC_tsh || !std::isfinite(errorZNCC))
        return false;
    else
        return true;
}

inline std::vector<Eigen::Vector3d> sampleTriangle(std::vector<Eigen::Vector3d> triangle) {

    Eigen::Vector3d u = (triangle.at(1) - triangle.at(0));
    Eigen::Vector3d v = (triangle.at(2) - triangle.at(0));
    double area       = (u.cross(v)).norm() / 2;

    std::vector<Eigen::Vector3d> sample;
    int n_points = std::round(area * 1000);

    while (sample.size() < n_points) {
        Eigen::Vector3d pt = triangle.at(0) + ((double)rand() / (RAND_MAX)) * u + ((double)rand() / (RAND_MAX)) * v;

        if (geometry::pointInTriangle(pt, triangle))
            sample.push_back(pt);
    }

    return sample;
}

void Mesh3D::generatePointCloud() {

    std::lock_guard<std::mutex> lock(_pc_mtx);
    _pcl_cloud.points.clear();
    _pcl_cloud.header.seq = _reference_frame->_id;

    Eigen::Affine3d T_cam0_w = _T_w_cam0.inverse();
    int height               = _img0.rows;
    int width                = _img0.cols;

    // Function to cast rays and fill the point cloud on a subset of the image
    std::mutex mtx;
    auto generate_pts = [height, T_cam0_w, &mtx, this](int col_start, int col_end) {
        for (int i = 0; i < height; i++) {
            if (i % 4 != 0)
                continue;
            for (int j = col_start; j < col_end + 1; j++) {
                if (j % 4 != 0)
                    continue;

                Eigen::Vector2d p((double)i, (double)j);
                std::vector<std::shared_ptr<Polygon>> crossed_triangles;

                // Parse all 2d triangles
                for (auto &poly_tri : _map_poly_tri2d) {
                    if (geometry::pointInTriangle(p, poly_tri.second))
                        crossed_triangles.push_back(poly_tri.first);
                }

                // Retain the closest triangle
                std::shared_ptr<Polygon> closest_triangle;
                Eigen::Vector3d v = _cam0->getRayCamera(p);
                double min_depth  = 1000;
                double min_score  = 100000;

                for (auto &triangle : crossed_triangles) {

                    Eigen::Vector3d n = T_cam0_w.rotation() * triangle->getPolygonNormal();

                    Eigen::Vector3d t_cam0_a = T_cam0_w * triangle->getVertices().at(0)->getVertexPosition();
                    double depth             = (t_cam0_a.dot(n) / v.dot(n));
                    double score             = triangle->getScore();

                    if ((v * depth - t_cam0_a).norm() < 1e-2) {
                        t_cam0_a = T_cam0_w * triangle->getVertices().at(1)->getVertexPosition();
                        depth    = (t_cam0_a.dot(n) / v.dot(n));
                    }

                    // Ignore negative depth
                    if (depth < 0.25 || depth > 5)
                        continue;

                    if (score < min_score) {
                        min_score        = score;
                        min_depth        = depth;
                        closest_triangle = triangle;
                    }
                }

                // Save the point if valid
                if (min_depth < 1000) {
                    pcl::PointNormal pt;
                    Eigen::Vector3d normal = closest_triangle->getPolygonNormal();
                    Eigen::Vector3d t_w_p  = _T_w_cam0 * (min_depth * v);

                    pt.x        = t_w_p.x();
                    pt.y        = t_w_p.y();
                    pt.z        = t_w_p.z();
                    pt.normal_x = normal.x();
                    pt.normal_y = normal.y();
                    pt.normal_z = normal.z();

                    mtx.lock();
                    _pcl_cloud.points.push_back(pt);
                    mtx.unlock();
                }
            }
        }
    };

    // Launch threads
    int n_threads = 4;
    int chunk     = (int)(width / n_threads);
    std::vector<std::thread> threads;
    for (int k = 0; k < n_threads - 1; k++) {
        threads.push_back(std::thread(generate_pts, k * chunk, (k + 1) * chunk));
    }
    threads.push_back(std::thread(generate_pts, chunk * (n_threads - 1), width));

    for (auto &th : threads) {
        th.join();
    }

    // for (auto polygon : _polygons) {

    //     std::vector<Eigen::Vector3d> triangle_3d, samples;
    //     for (auto vertex : polygon->getVertices()) {
    //         triangle_3d.push_back(vertex->getLmk()->getPose().translation());
    //     }

    //     samples = sampleTriangle(triangle_3d);

    //     for (auto t_w_p : samples) {
    //         pcl::PointNormal pt;
    //         Eigen::Vector3d normal(1,0,0);

    //         pt.x        = t_w_p.x();
    //         pt.y        = t_w_p.y();
    //         pt.z        = t_w_p.z();
    //         pt.normal_x = normal.x();
    //         pt.normal_y = normal.y();
    //         pt.normal_z = normal.z();

    //         _pcl_cloud.points.push_back(pt);
    //     }
    // }

    bool write_clouds = false;
    if (write_clouds && _pcl_cloud.points.size() > 0) {
        _pcl_cloud.width  = _pcl_cloud.points.size();
        _pcl_cloud.height = 1;
        pcl::PointCloud<pcl::PointNormal> pc_frame;
        std::string title = "cloud/" + std::to_string(_reference_frame->getTimestamp()) + ".pcd";
        pcl::transformPointCloud(_pcl_cloud, pc_frame, T_cam0_w.matrix());
        pcl::io::savePCDFileASCII(title, pc_frame);
    }
}

} // namespace isae