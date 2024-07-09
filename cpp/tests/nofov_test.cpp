#include <gtest/gtest.h>

#include "Eigen/Geometry"
#include "isaeslam/data/features/AFeature2D.h"
#include "isaeslam/data/features/Point2D.h"
#include "isaeslam/data/frame.h"
#include "isaeslam/data/landmarks/ALandmark.h"
#include "isaeslam/data/landmarks/Point3D.h"
#include "isaeslam/data/mesh/mesh.h"
#include "isaeslam/data/sensors/Camera.h"
#include "isaeslam/estimator/EpipolarPoseEstimator.h"
#include "isaeslam/landmarkinitializer/Point3DlandmarkInitializer.h"
#include "isaeslam/optimizers/AngularAdjustmentCERESAnalytic.h"
#include "isaeslam/optimizers/BundleAdjustmentCERESAnalytic.h"
#include "isaeslam/slamCore.h"

namespace isae {

class NoFOVTest : public testing::Test {

    /* Simulation to test the non overlapping FOV SLAM:
     *
     *           .  .      .
     *      . . .   .
     *    .                .  .  .
     *     .    c1 ---- c2   .  .  .
     *       . .          .  . .
     *           .  .  . .
     *
     * A setup of two non overlapping cameras is simulated, a thousands of random 3D points are populated around it and
     * a random motion is applied to the bench.
     */

  public:
    void SetUp() override {

        srand((unsigned int)time(0));

        // Intrinsic
        _K       = Eigen::Matrix3d::Identity();
        _K(0, 0) = 100;
        _K(1, 1) = 100;
        _K(0, 2) = 400;
        _K(1, 2) = 400;

        // Generates random 3D points in a 10m * 10m box
        for (uint i = 0; i < 10000; i++) {
            _3d_points.push_back(10 * Eigen::Vector3d::Random());
        }
    }

    Eigen::Matrix3d _K;
    std::vector<Eigen::Vector3d> _3d_points;
    std::shared_ptr<ImageSensor> _sensor1, _sensor2, _sensor1p, _sensor2p;
    typed_vec_match _matches_in_time_cam1, _matches_cam2;
    std::shared_ptr<Frame> _frame, _framep;
};

TEST_F(NoFOVTest, scaleTest) {

    // Generates a random motion
    Eigen::Quaterniond q_rand         = Eigen::Quaterniond::UnitRandom();
    Eigen::Vector3d t_rand            = Eigen::Vector3d::Random();
    Eigen::Affine3d T_f_fp            = Eigen::Affine3d::Identity();
    T_f_fp.affine().block(0, 0, 3, 3) = q_rand.toRotationMatrix();
    T_f_fp.translation()              = t_rand;

    // Example of motion from ISAE dataset (chariot1)
    T_f_fp.matrix() << 0.997565, 0.0696928, 0.00270891, 0.0768125, -0.0697074, 0.997551, 0.00570372, -0.158637,
        -0.00230477, -0.00587867, 0.99998, -0.0178483, 0, 0, 0, 1;

    // Set the frames
    _frame  = std::shared_ptr<Frame>(new Frame());
    _framep = std::shared_ptr<Frame>(new Frame());

    // Set transforms sensors (from ISAE bench)
    Eigen::Affine3d T_f_s1, T_f_s2, T_s1_s2;
    T_f_s1.matrix() << -0.01404322, 0.00230685, 0.99989873, 0.06684756, -0.99986816, -0.00818516, -0.01402391,
        0.23005136, 0.00815198, -0.99996384, 0.00242149, 0.01394674, 0., 0., 0., 1.;

    T_f_s2.matrix() << 0.0279097, 0.00437207, -0.99960089, -0.06755216, 0.99961045, -0.00016776, 0.02790923, -0.2074177,
        -0.00004568, -0.99999043, -0.00437504, 0.0111566, 0., 0., 0., 1.;
    
    T_s1_s2 = T_f_s1.inverse() * T_f_s2;
    
    
    // Get groundtruth transformation
    Eigen::Affine3d T_s1_s1p = T_f_s1.inverse() * T_f_fp * T_f_s1;

    // Creates two cameras with non overlapping FoV
    _sensor1 = std::shared_ptr<Camera>(new Camera(cv::Mat::zeros(800, 800, CV_16F), _K));
    _sensor1->setFrame2SensorTransform(T_f_s1.inverse());
    _sensor2 = std::shared_ptr<Camera>(new Camera(cv::Mat::zeros(800, 800, CV_16F), _K));
    _sensor2->setFrame2SensorTransform(T_f_s2.inverse());
    std::vector<std::shared_ptr<ImageSensor>> sensors_frame;
    sensors_frame.push_back(_sensor1);
    sensors_frame.push_back(_sensor2);
    _frame->init(sensors_frame, 0);
    _frame->setWorld2FrameTransform(Eigen::Affine3d::Identity());
    _frame->setKeyFrame();

    _sensor1p = std::shared_ptr<Camera>(new Camera(cv::Mat::zeros(800, 800, CV_16F), _K));
    _sensor1p->setFrame2SensorTransform(T_f_s1.inverse());
    _sensor2p = std::shared_ptr<Camera>(new Camera(cv::Mat::zeros(800, 800, CV_16F), _K));
    _sensor2p->setFrame2SensorTransform(T_f_s2.inverse());
    std::vector<std::shared_ptr<ImageSensor>> sensors_framep;
    sensors_framep.push_back(_sensor1p);
    sensors_framep.push_back(_sensor2p);
    _framep->init(sensors_framep, 0);
    _framep->setWorld2FrameTransform(T_f_fp.inverse());
    _framep->setKeyFrame();

    // Add keypoints on the cameras
    Point3DLandmarkInitializer p3d_init(100);
    for (auto pt : _3d_points) {
        Eigen::Affine3d T_w_lmk = Eigen::Affine3d::Identity();
        T_w_lmk.translation()   = pt;

        Eigen::Vector2d p2d, p2dp;
        bool is_proj = _sensor1->project(
            T_w_lmk, _frame->getWorld2FrameTransform(), Eigen::Matrix2d::Identity(), p2d, nullptr, nullptr);
        bool is_projp = _sensor1p->project(
            T_w_lmk, _framep->getWorld2FrameTransform(), Eigen::Matrix2d::Identity(), p2dp, nullptr, nullptr);

        if (is_proj && is_projp) {
            std::vector<Eigen::Vector2d> p2ds, p2dsp;
            p2ds.push_back(p2d);
            p2dsp.push_back(p2dp);
            std::shared_ptr<Point2D> feat  = std::shared_ptr<Point2D>(new Point2D(p2ds));
            std::shared_ptr<Point2D> featp = std::shared_ptr<Point2D>(new Point2D(p2dsp));
            _sensor1->addFeature("pointxd", feat);
            _sensor1p->addFeature("pointxd", featp);
            _matches_in_time_cam1["pointxd"].push_back({feat, featp});

            std::vector<std::shared_ptr<AFeature>> feat_vec;
            feat_vec.push_back(feat);
            feat_vec.push_back(featp);
            p3d_init.initFromFeatures(feat_vec);
            continue;
        }

        is_proj = _sensor2->project(
            T_w_lmk, _frame->getWorld2FrameTransform(), Eigen::Matrix2d::Identity(), p2d, nullptr, nullptr);
        is_projp = _sensor2p->project(
            T_w_lmk, _framep->getWorld2FrameTransform(), Eigen::Matrix2d::Identity(), p2dp, nullptr, nullptr);

        if (is_proj && is_projp) {
            std::vector<Eigen::Vector2d> p2ds, p2dsp;
            p2ds.push_back(p2d);
            p2dsp.push_back(p2dp);
            std::shared_ptr<Point2D> feat  = std::shared_ptr<Point2D>(new Point2D(p2ds));
            std::shared_ptr<Point2D> featp = std::shared_ptr<Point2D>(new Point2D(p2dsp));
            _sensor2->addFeature("pointxd", feat);
            _sensor2p->addFeature("pointxd", featp);
            _matches_cam2["pointxd"].push_back({feat, featp});

            std::vector<std::shared_ptr<AFeature>> feat_vec;
            feat_vec.push_back(feat);
            feat_vec.push_back(featp);
            p3d_init.initFromFeatures(feat_vec);
            continue;
        }
    }

    // We have enough matches
    ASSERT_GT(_matches_in_time_cam1["pointxd"].size(), 5);
    ASSERT_GT(_matches_cam2["pointxd"].size(), 1);

    // The rotation is well estimated via epipolar geometry
    Eigen::Affine3d T_last_curr;
    Eigen::MatrixXd cov;
    EpipolarPoseEstimator essential_ransac;
    essential_ransac.estimateTransformSensors(_sensor1, _sensor1p, _matches_in_time_cam1["pointxd"], T_last_curr, cov);
    ASSERT_NEAR((T_last_curr.rotation().transpose() * T_s1_s1p.rotation()).sum(), 3, 1e-5);

    // The scale is well recovered by the one point RANSAC
    SLAMNonOverlappingFov slam;
    double lambda;
    slam.scaleEstimationRANSAC(T_s1_s2, T_last_curr, _matches_cam2, lambda);
    T_last_curr.translation() *= lambda;
    std::cout << "gt : \n" << T_s1_s1p.matrix() << std::endl;
    std::cout << "that is a : " << T_s1_s1p.translation().norm() << " scale " << std::endl;
    std::cout << "est ransac : \n" << T_last_curr.matrix() << std::endl;
    ASSERT_NEAR(lambda, T_s1_s1p.translation().norm(), 1e-3);

    // Recover scale with optimizer
    isae::AngularAdjustmentCERESAnalytic ceres_ba;
    T_last_curr.translation() *= 1.2; // change the scale
    ceres_ba.landmarkOptimizationNoFov(_frame, _framep, T_last_curr, 0.0);
    std::cout << "est optim : \n" << T_last_curr.matrix() << std::endl;
    ASSERT_NEAR((T_last_curr.matrix() - T_s1_s1p.matrix()).sum(), 0, 1e-3);
}

TEST_F(NoFOVTest, degenerativeCase) {

    // Generates a degenerative motion (translation only)
    Eigen::Vector3d t_rand = Eigen::Vector3d::Random();
    Eigen::Affine3d T_f_fp = Eigen::Affine3d::Identity();
    T_f_fp.translation()   = t_rand;

    // Set the frames
    _frame = std::shared_ptr<Frame>(new Frame());
    _frame->setWorld2FrameTransform(Eigen::Affine3d::Identity());
    _framep = std::shared_ptr<Frame>(new Frame());
    _framep->setWorld2FrameTransform(T_f_fp.inverse());

    // Creates two cameras with non overlapping FoV
    _sensor1 = std::shared_ptr<Camera>(new Camera(cv::Mat::zeros(800, 800, CV_16F), _K));
    _sensor1->setFrame2SensorTransform(Eigen::Affine3d::Identity());
    _sensor2 = std::shared_ptr<Camera>(new Camera(cv::Mat::zeros(800, 800, CV_16F), _K));
    Eigen::Affine3d T_f_s2;
    T_f_s2.matrix() << -0.8660254037844387, 1.0605752387249069e-16, -0.49999999999999994, -0.8660254037844387,
        -1.2246467991473532e-16, -1.0, 0.0, -1.2246467991473532e-16, -0.49999999999999994, 6.123233995736766e-17,
        0.8660254037844387, -0.49999999999999994, 0.0, 0.0, 0.0, 1.0;
    _sensor2->setFrame2SensorTransform(T_f_s2.inverse());

    _sensor1p = std::shared_ptr<Camera>(new Camera(cv::Mat::zeros(800, 800, CV_16F), _K));
    _sensor1p->setFrame2SensorTransform(Eigen::Affine3d::Identity());
    _sensor2p = std::shared_ptr<Camera>(new Camera(cv::Mat::zeros(800, 800, CV_16F), _K));
    _sensor2p->setFrame2SensorTransform(T_f_s2.inverse());

    // Add keypoints on the cameras
    for (auto pt : _3d_points) {
        Eigen::Affine3d T_w_lmk = Eigen::Affine3d::Identity();
        T_w_lmk.translation()   = pt;

        Eigen::Vector2d p2d, p2dp;
        bool is_proj = _sensor1->project(
            T_w_lmk, _frame->getWorld2FrameTransform(), Eigen::Matrix2d::Identity(), p2d, nullptr, nullptr);
        bool is_projp = _sensor1p->project(
            T_w_lmk, _framep->getWorld2FrameTransform(), Eigen::Matrix2d::Identity(), p2dp, nullptr, nullptr);

        if (is_proj && is_projp) {
            std::vector<Eigen::Vector2d> p2ds, p2dsp;
            p2ds.push_back(p2d);
            p2dsp.push_back(p2dp);
            std::shared_ptr<Point2D> feat  = std::shared_ptr<Point2D>(new Point2D(p2ds));
            std::shared_ptr<Point2D> featp = std::shared_ptr<Point2D>(new Point2D(p2dsp));
            _sensor1->addFeature("pointxd", feat);
            _sensor1p->addFeature("pointxd", featp);
            _matches_in_time_cam1["pointxd"].push_back({feat, featp});
            continue;
        }

        is_proj = _sensor2->project(
            T_w_lmk, _frame->getWorld2FrameTransform(), Eigen::Matrix2d::Identity(), p2d, nullptr, nullptr);
        is_projp = _sensor2p->project(
            T_w_lmk, _framep->getWorld2FrameTransform(), Eigen::Matrix2d::Identity(), p2dp, nullptr, nullptr);

        if (is_proj && is_projp) {
            std::vector<Eigen::Vector2d> p2ds, p2dsp;
            p2ds.push_back(p2d);
            p2dsp.push_back(p2dp);
            std::shared_ptr<Point2D> feat  = std::shared_ptr<Point2D>(new Point2D(p2ds));
            std::shared_ptr<Point2D> featp = std::shared_ptr<Point2D>(new Point2D(p2dsp));
            _sensor2->addFeature("pointxd", feat);
            _sensor2p->addFeature("pointxd", featp);
            _matches_cam2["pointxd"].push_back({feat, featp});
            continue;
        }
    }

    // The rotation is well estimated via epipolar geometry
    Eigen::Affine3d T_last_curr;
    Eigen::MatrixXd cov;
    EpipolarPoseEstimator essential_ransac;
    essential_ransac.estimateTransformSensors(_sensor1, _sensor1p, _matches_in_time_cam1["pointxd"], T_last_curr, cov);
    ASSERT_NEAR((T_last_curr.rotation().transpose() * _framep->getFrame2WorldTransform().rotation()).sum(), 3, 1e-5);

    // The scale is well recovered by the one point RANSAC
    SLAMNonOverlappingFov slam;
    double lambda;
    slam.scaleEstimationRANSAC(_sensor2->getFrame2SensorTransform().inverse(), T_last_curr, _matches_cam2, lambda);

    // The motion is not degenerative
    ASSERT_EQ(slam.isDegenerativeMotion(T_last_curr, T_f_s2, _matches_cam2), true);
}

} // namespace isae