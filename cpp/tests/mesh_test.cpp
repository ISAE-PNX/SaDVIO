#include <gtest/gtest.h>

#include "isaeslam/data/features/AFeature2D.h"
#include "isaeslam/data/features/Point2D.h"
#include "isaeslam/data/frame.h"
#include "isaeslam/data/landmarks/ALandmark.h"
#include "isaeslam/data/landmarks/Point3D.h"
#include "isaeslam/data/mesh/mesh.h"
#include "isaeslam/data/sensors/Camera.h"
#include "utilities/geometry.h"

namespace isae {

class MeshTest : public testing::Test {

    /* Toy example of a mesh with a single triangular face
     *       lmk_0 --- lmk_1
     *         \       /
     *          \     /
     *           lmk_2
     */

  public:
    void SetUp() override {
        srand((unsigned int)time(0));

        // Set Frame
        _frame0 = std::shared_ptr<Frame>(new Frame());
        Eigen::Affine3d T_f0_w = Eigen::Affine3d::Identity();
        T_f0_w.translation() << 0.05, 0.0, 3.0;
        Eigen::Affine3d T_f_camright = Eigen::Affine3d::Identity();
        T_f_camright.translation() << 0.1, 0.0, 0.0;

        // Set Sensors
        _K        = Eigen::Matrix3d::Identity();
        _K(0, 0)  = 320;
        _K(1, 1)  = 320;
        _K(0, 2)  = 320;
        _K(1, 2)  = 240;
        cv::Mat img_left = cv::imread("../tests/img_left.png", cv::IMREAD_GRAYSCALE);
        cv::Mat img_right = cv::imread("../tests/img_right.png", cv::IMREAD_GRAYSCALE);
        _sensor0l = std::shared_ptr<Camera>(new Camera(img_left, _K));
        _sensor0l->setFrame2SensorTransform(Eigen::Affine3d::Identity());
        _sensor0r = std::shared_ptr<Camera>(new Camera(img_right, _K));
        _sensor0r->setFrame2SensorTransform(T_f_camright.inverse());
        std::vector<std::shared_ptr<ImageSensor>> sensors_frame0;
        sensors_frame0.push_back(_sensor0l);
        sensors_frame0.push_back(_sensor0r);
        _frame0->init(sensors_frame0, 0);
        _frame0->setWorld2FrameTransform(T_f0_w);

        // Set landmark 0
        Eigen::Affine3d T_w_l0 = Eigen::Affine3d::Identity();
        T_w_l0.translation()   = Eigen::Vector3d(1, 1, -1);
        std::vector<std::shared_ptr<AFeature>> feat_vec_0;
        _lmk_0 = std::shared_ptr<Point3D>(new Point3D(T_w_l0, feat_vec_0));
        std::vector<Eigen::Vector2d> projection_0_0_l, projection_0_0_r;
        _sensor0l->project(_lmk_0->getPose(), _lmk_0->getModel(), Eigen::Vector3d(1, 1, 1), projection_0_0_l);
        _sensor0r->project(_lmk_0->getPose(), _lmk_0->getModel(), Eigen::Vector3d(1, 1, 1), projection_0_0_r);
        _feat_0 = std::shared_ptr<Point2D>(new Point2D(projection_0_0_l));
        _lmk_0->addFeature(_feat_0);
        _sensor0l->addFeature("pointxd", _feat_0);

        // Set landmark 1
        Eigen::Affine3d T_w_l1 = Eigen::Affine3d::Identity();
        T_w_l1.translation()   = Eigen::Vector3d(-1, -1, -1);
        std::vector<std::shared_ptr<AFeature>> feat_vec_1;
        _lmk_1 = std::shared_ptr<Point3D>(new Point3D(T_w_l1, feat_vec_1));
        std::vector<Eigen::Vector2d> projection_0_1_l, projection_0_1_r;
        _sensor0l->project(_lmk_1->getPose(), _lmk_1->getModel(), Eigen::Vector3d(1, 1, 1), projection_0_1_l);
        _sensor0r->project(_lmk_1->getPose(), _lmk_1->getModel(), Eigen::Vector3d(1, 1, 1), projection_0_1_r);
        _feat_1 = std::shared_ptr<Point2D>(new Point2D(projection_0_1_l));
        _lmk_1->addFeature(_feat_1);
        _sensor0l->addFeature("pointxd", _feat_1);

        // Set landmark 2
        Eigen::Affine3d T_w_l2 = Eigen::Affine3d::Identity();
        T_w_l2.translation()   = Eigen::Vector3d(1, -1, -1);
        std::vector<std::shared_ptr<AFeature>> feat_vec_2;
        _lmk_2 = std::shared_ptr<Point3D>(new Point3D(T_w_l2, feat_vec_2));
        std::vector<Eigen::Vector2d> projection_0_2_l, projection_0_2_r;
        _sensor0l->project(_lmk_2->getPose(), _lmk_2->getModel(), Eigen::Vector3d(1, 1, 1), projection_0_2_l);
        _sensor0r->project(_lmk_2->getPose(), _lmk_2->getModel(), Eigen::Vector3d(1, 1, 1), projection_0_2_r);
        _feat_2 = std::shared_ptr<Point2D>(new Point2D(projection_0_2_l));
        _lmk_2->addFeature(_feat_2);
        _sensor0l->addFeature("pointxd", _feat_2);

        // Set landmark 3
        Eigen::Affine3d T_w_l3 = Eigen::Affine3d::Identity();
        T_w_l3.translation()   = Eigen::Vector3d(-1, 1, -1);
        std::vector<std::shared_ptr<AFeature>> feat_vec_3;
        _lmk_3 = std::shared_ptr<Point3D>(new Point3D(T_w_l3, feat_vec_3));
        std::vector<Eigen::Vector2d> projection_0_3_l, projection_0_3_r;
        _sensor0l->project(_lmk_3->getPose(), _lmk_3->getModel(), Eigen::Vector3d(1, 1, 1), projection_0_3_l);
        _sensor0r->project(_lmk_2->getPose(), _lmk_2->getModel(), Eigen::Vector3d(1, 1, 1), projection_0_3_r);
        _feat_3 = std::shared_ptr<Point2D>(new Point2D(projection_0_3_l));
        _lmk_3->addFeature(_feat_3);
        _sensor0l->addFeature("pointxd", _feat_3);

        // Init the 2D mesh
        _2D_triangle.push_back(_feat_0);
        _2D_triangle.push_back(_feat_1);
        _2D_triangle.push_back(_feat_2);

        // Init the new 2D mesh for update
        _new_2D_triangle.push_back(_feat_0);
        _new_2D_triangle.push_back(_feat_1);
        _new_2D_triangle.push_back(_feat_3);
        
        // Uncomment to check if the triangle is well projected
        // cv::Mat img_colorl, img_colorr;
        // cv::cvtColor(img_left.clone(), img_colorl, cv::COLOR_GRAY2RGB);
        // cv::cvtColor(img_right.clone(), img_colorr, cv::COLOR_GRAY2RGB);
        // cv::circle(img_colorl, cv::Point2d(projection_0_1_l.at(0)(0), projection_0_1_l.at(0)(1)), 11, cv::Scalar(0, 255, 0));
        // cv::circle(img_colorr, cv::Point2d(projection_0_1_r.at(0)(0), projection_0_1_r.at(0)(1)), 11, cv::Scalar(0, 255, 0));
        // cv::circle(img_colorl, cv::Point2d(projection_0_0_l.at(0)(0), projection_0_0_l.at(0)(1)), 11, cv::Scalar(255, 0, 0));
        // cv::circle(img_colorr, cv::Point2d(projection_0_0_r.at(0)(0), projection_0_0_r.at(0)(1)), 11, cv::Scalar(255, 0, 0));
        // cv::circle(img_colorl, cv::Point2d(projection_0_2_l.at(0)(0), projection_0_2_l.at(0)(1)), 11, cv::Scalar(0, 0, 255));
        // cv::circle(img_colorr, cv::Point2d(projection_0_2_r.at(0)(0), projection_0_2_r.at(0)(1)), 11, cv::Scalar(0, 0, 255));
        // cv::imwrite("_0.png", img_colorl);
        // cv::imwrite("_1.png", img_colorr);
    }

    std::shared_ptr<Frame> _frame0;
    std::shared_ptr<Point2D> _feat_0;
    std::shared_ptr<Point3D> _lmk_0;
    std::shared_ptr<Point2D> _feat_1;
    std::shared_ptr<Point3D> _lmk_1;
    std::shared_ptr<Point2D> _feat_2;
    std::shared_ptr<Point3D> _lmk_2;
    std::shared_ptr<Point2D> _feat_3;
    std::shared_ptr<Point3D> _lmk_3;

    FeatPolygon _2D_triangle;
    FeatPolygon _new_2D_triangle;

    Eigen::Matrix3d _K;
    std::shared_ptr<ImageSensor> _sensor0l;
    std::shared_ptr<ImageSensor> _sensor0r;
};

TEST_F(MeshTest, MeshTestBase) {

    std::vector<FeatPolygon> mesh_2d;
    mesh_2d.push_back(_2D_triangle);
    Mesh3D mesh_3d;
    mesh_3d.updateMesh(mesh_2d, _frame0);

    // The size of the mesh is correct
    ASSERT_EQ(mesh_3d.getPolygonVector().size(), 1);

    // Try checkPolygonTri
    FeatPolygon feat_tri2d;
    feat_tri2d.push_back(_feat_0);
    feat_tri2d.push_back(_feat_1);
    feat_tri2d.push_back(_feat_2);
    bool tri_valid = mesh_3d.checkPolygonTri(mesh_3d.getPolygonVector().at(0), feat_tri2d);

    ASSERT_EQ(tri_valid, true);

    // The normal is correct
    Eigen::Vector3d normal_gt(0, 0, -1);
    Eigen::Vector3d normal_mesh = mesh_3d.getPolygonVector().at(0)->getPolygonNormal();
    ASSERT_EQ((normal_gt.cross(normal_mesh)).norm(), 0);

    // Update of the mesh
    // Test if we have one new polygon
    std::vector<FeatPolygon> new_mesh_2d;
    new_mesh_2d.push_back(_new_2D_triangle);
    mesh_3d.updateMesh(new_mesh_2d, _frame0);
    ASSERT_EQ(mesh_3d.getPolygonVector().size(), 2);
    ASSERT_EQ(mesh_3d.getMap().at(_lmk_1)->getPolygons().size(), 2);
    ASSERT_EQ(mesh_3d.getMap().at(_lmk_2)->getPolygons().size(), 1);

    // Test if we have a marginalized landmark and no new polygons
    _lmk_2->setMarg();
    new_mesh_2d.clear();
    mesh_3d.updateMesh(new_mesh_2d, _frame0);

    // Test if there is only one polygon left
    ASSERT_EQ(mesh_3d.getPolygonVector().size(), 1);

    // Test if the good polygon is remaining
    std::vector<std::shared_ptr<Vertex>> vertices;
    ASSERT_EQ(mesh_3d.getPolygonVector().at(0)->getVertices().at(0)->getLmk(), _lmk_0);
    ASSERT_EQ(mesh_3d.getPolygonVector().at(0)->getVertices().at(1)->getLmk(), _lmk_1);
    ASSERT_EQ(mesh_3d.getPolygonVector().at(0)->getVertices().at(2)->getLmk(), _lmk_3);

    // Test if the vertices have only one remaining polygon
    ASSERT_EQ(mesh_3d.getMap().at(_lmk_1)->getPolygons().size(), 1);
    ASSERT_EQ(mesh_3d.getMap().at(_lmk_1)->getPolygons().at(0), mesh_3d.getPolygonVector().at(0));
}

TEST_F(MeshTest, pointInTriangleTest) {
    // Build a random triangle
    Eigen::Vector3d A = Eigen::Vector3d::Random();
    Eigen::Vector3d B = Eigen::Vector3d::Random();
    Eigen::Vector3d C = Eigen::Vector3d::Random();
    std::vector<Eigen::Vector3d> triangle;
    triangle.push_back(A);
    triangle.push_back(B);
    triangle.push_back(C);

    // Build a point inside the triangle
    Eigen::Vector3d P = A + 0.4 * (B - A) + 0.5 * (C - A);
    ASSERT_EQ(geometry::pointInTriangle(P, triangle), true);

    // Build a point outside the triangle
    Eigen::Vector3d Pp = A + 0.8 * (B - A) + 0.5 * (C - A);
    ASSERT_EQ(geometry::pointInTriangle(Pp, triangle), false);
}

TEST_F(MeshTest, covTest) {

    // Build a random triangle
    Eigen::Vector3d A = Eigen::Vector3d::Random();
    Eigen::Vector3d B = Eigen::Vector3d::Random();
    Eigen::Vector3d C = Eigen::Vector3d::Random();
    std::vector<Eigen::Vector3d> triangle;
    triangle.push_back(A);
    triangle.push_back(B);
    triangle.push_back(C);

    // Turn it into a 2D triangle
    Eigen::Vector3d b1 = B - A;
    Eigen::Vector3d b2 = C - A;

    // Find 2D coordinates of the points
    Eigen::Vector3d b1_norm = b1.normalized();
    Eigen::Vector2d u1(b1.norm(), 0);
    Eigen::Vector2d u2(b1_norm.dot(b2), (b2 - b1_norm.dot(b2) * b1_norm).norm());

    // Check if the area of both triangle is the same
    double a_3d = 0.5 * (b1.cross(b2)).norm();
    Eigen::Matrix2d tri;
    tri << u1.x(), u2.x(), u1.y(), u2.y();
    double a_2d = 0.5 * tri.determinant();

    ASSERT_NEAR(a_3d, a_2d, 1e-6);

    // Now compute the numerical covariance of the triangle
    std::vector<Eigen::Vector2d> tri_2d;
    tri_2d.push_back(Eigen::Vector2d(0, 0));
    tri_2d.push_back(u1);
    tri_2d.push_back(u2);
    tri << u1.x(), u2.x(), u1.y(), u2.y();
    Eigen::Vector2d barycenter = (1.0 / 3.0) * (u1 + u2);

    Eigen::Matrix2d cov_num  = Eigen::Matrix2d::Zero();
    Eigen::Vector2d bary_num = Eigen::Vector2d::Zero();
    int n_pts                = 1000000;

    for (int i = 0; i < n_pts; i++) {
        Eigen::Vector2d rand_pt = 2 * Eigen::Vector2d::Random();
        while (!geometry::pointInTriangle(rand_pt, tri_2d))
            rand_pt = 2 * Eigen::Vector2d::Random();

        cov_num = cov_num + (rand_pt - barycenter) * (rand_pt - barycenter).transpose();
        bary_num += rand_pt;
    }

    // Try to compute the barycenter
    bary_num /= n_pts;
    cov_num = cov_num / n_pts;

    ASSERT_NEAR((bary_num - barycenter).norm(), 0, 1e-2);

    // Compute covariance
    Eigen::Matrix2d M = geometry::cov2dTriangle(tri_2d);
    std::cout << cov_num << "\n" << std::endl;
    std::cout << M << std::endl;
    ASSERT_NEAR((M - cov_num).sum(), 0, 1e-3);

    // Try equi triangle (the eigen values must be equal)
    u1 << std::cos(M_PI / 3), std::sin(M_PI / 3);
    u2 << 1, 0;
    tri_2d.clear();
    tri_2d.push_back(Eigen::Vector2d(0, 0));
    tri_2d.push_back(u1);
    tri_2d.push_back(u2);
    M = geometry::cov2dTriangle(tri_2d);
    std::cout << geometry::cov2dTriangle(tri_2d) << std::endl;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    double ratio = svd.singularValues()(0) / svd.singularValues()(1);
    ASSERT_NEAR(ratio,1, 1e-3);
}

} // namespace isae