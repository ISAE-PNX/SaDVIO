#include "isaeslam/data/frame.h"
#include "isaeslam/data/sensors/Camera.h"
#include "isaeslam/optimizers/AngularAdjustmentCERESAnalytic.h"
#include "isaeslam/optimizers/BundleAdjustmentCERESAnalytic.h"
#include "isaeslam/optimizers/residuals.hpp"
#include <gtest/gtest.h>

namespace isae {

class ResidualTest : public testing::Test {
  public:
    void SetUp() override {
        _frame0 = std::shared_ptr<Frame>(new Frame());
        _frame1 = std::shared_ptr<Frame>(new Frame());

        // Generates a random pose + a small displacement
        Eigen::Affine3d T_f0_w            = Eigen::Affine3d::Identity();
        Eigen::Quaterniond q_rand         = Eigen::Quaterniond::UnitRandom();
        Eigen::Vector3d t_rand            = Eigen::Vector3d::Random();
        T_f0_w.affine().block(0, 0, 3, 3) = q_rand.toRotationMatrix();
        T_f0_w.translation()              = t_rand;
        _dT                               = Eigen::Affine3d::Identity();
        _dT.translation()                 = Eigen::Vector3d(0.05, 0.05, 0);

        // Set Intrinsics
        _K       = Eigen::Matrix3d::Identity();
        _K(0, 0) = 100;
        _K(1, 1) = 100;
        _K(0, 2) = 400;
        _K(1, 2) = 400;
        _sensor0 = std::shared_ptr<Camera>(new Camera(cv::Mat::zeros(800, 800, CV_16F), _K));
        _sensor1 = std::shared_ptr<Camera>(new Camera(cv::Mat::zeros(800, 800, CV_16F), _K));

        // Init frames
        std::vector<std::shared_ptr<ImageSensor>> sensors_frame0;
        sensors_frame0.push_back(_sensor0);
        _frame0->init(sensors_frame0, 0);
        _frame0->setWorld2FrameTransform(T_f0_w);
        _sensor0->setFrame2SensorTransform(Eigen::Affine3d::Identity());

        std::vector<std::shared_ptr<ImageSensor>> sensors_frame1;
        sensors_frame1.push_back(_sensor1);
        _frame1->init(sensors_frame1, 0);
        _frame1->setWorld2FrameTransform(_dT.inverse() * T_f0_w);
        _sensor1->setFrame2SensorTransform(Eigen::Affine3d::Identity());

        // Init a random landmark in the FOV
        _rand_lmk               = Eigen::Affine3d::Identity();
        _rand_lmk.translation() = Eigen::Vector3d::Random();

        while (!_sensor0->project(
                   _rand_lmk, _frame0->getWorld2FrameTransform(), Eigen::Matrix2d::Identity(), _p2d0, NULL, NULL) ||
               !_sensor1->project(
                   _rand_lmk, _frame1->getWorld2FrameTransform(), Eigen::Matrix2d::Identity(), _p2d1, NULL, NULL)) {
            _rand_lmk.translation() = Eigen::Vector3d::Random();
        }
    }

    std::shared_ptr<Frame> _frame0, _frame1;
    std::shared_ptr<Camera> _sensor0, _sensor1;
    Eigen::Affine3d _rand_lmk, _dT;
    Eigen::Matrix3d _K;
    Eigen::Vector2d _p2d0, _p2d1;
};

TEST_F(ResidualTest, PriorResidual) {

    // Generates a random pose
    Eigen::Affine3d T_f0_w            = Eigen::Affine3d::Identity();
    Eigen::Quaterniond q_rand         = Eigen::Quaterniond::UnitRandom();
    Eigen::Vector3d t_rand            = Eigen::Vector3d::Random();
    T_f0_w.affine().block(0, 0, 3, 3) = q_rand.toRotationMatrix();
    T_f0_w.translation()              = t_rand;

    // Create a Cost function
    ceres::CostFunction *cost_fct =
        new PosePriordx(Eigen::Affine3d::Identity(), T_f0_w, 100 * Vector6d::Ones().asDiagonal());

    // Set variables
    PoseParametersBlock dX(Eigen::Affine3d::Identity());

    std::vector<double *> parameters_blocks;
    std::vector<const ceres::Manifold *> *manifs = new std::vector<const ceres::Manifold *>;
    parameters_blocks.push_back(dX.values());
    manifs->push_back(nullptr);

    // Check the jacss
    ceres::NumericDiffOptions numeric_diff_options;
    ceres::GradientChecker gradient_checker(cost_fct, manifs, numeric_diff_options);
    ceres::GradientChecker::ProbeResults results;
    if (!gradient_checker.Probe(parameters_blocks.data(), 1e-5, &results)) {
        LOG(ERROR) << "An error has occurred:\n" << results.error_log;
    }

    ASSERT_NEAR((results.local_jacobians.at(0) - results.jacobians.at(0)).sum(), 0, 1e-5);
}

TEST_F(ResidualTest, reprojTest) {

    // Create a Cost function
    ceres::CostFunction *cost_fct =
        new BundleAdjustmentCERESAnalytic::ReprojectionErrCeres_pointxd_dx(_p2d0, _sensor0, _rand_lmk, 1);

    // Set variables
    // ceres::Manifold *nullptr = new SE3RightParameterization();
    PoseParametersBlock dX(Eigen::Affine3d::Identity());
    PointXYZParametersBlock dlmk(Eigen::Vector3d::Zero());

    std::vector<double *> parameters_blocks;
    std::vector<const ceres::Manifold *> *manifs = new std::vector<const ceres::Manifold *>;
    parameters_blocks.push_back(dX.values());
    parameters_blocks.push_back(dlmk.values());
    manifs->push_back(nullptr);
    manifs->push_back(nullptr);

    // Check the jacss
    ceres::NumericDiffOptions numeric_diff_options;
    ceres::GradientChecker gradient_checker(cost_fct, manifs, numeric_diff_options);
    ceres::GradientChecker::ProbeResults results;
    if (!gradient_checker.Probe(parameters_blocks.data(), 1e-9, &results)) {
        LOG(ERROR) << "An error has occurred:\n" << results.error_log;
    }

    ASSERT_NEAR((results.local_jacobians.at(0) - results.jacobians.at(0)).sum(), 0, 1e-5);
    ASSERT_NEAR((results.local_jacobians.at(1) - results.jacobians.at(1)).sum(), 0, 1e-5);
}

TEST_F(ResidualTest, angularTest) {

    // Get depth
    Eigen::Vector3d t_s0_lmk = _sensor0->getWorld2SensorTransform() * _rand_lmk.translation();
    double depth             = t_s0_lmk.norm();

    // Get bearing vectors
    Eigen::Vector3d b0 = _sensor0->getRayCamera(_p2d0);
    Eigen::Vector3d b1 = _sensor1->getRayCamera(_p2d1);

    // Create a Cost function
    ceres::CostFunction *cost_fct =
        new AngularAdjustmentCERESAnalytic::AngularErrCeres_pointxd_depth(b1,
                                                                          b0,
                                                                          Eigen::Affine3d::Identity(),
                                                                          _frame0->getWorld2FrameTransform(),
                                                                          _frame1->getWorld2FrameTransform(),
                                                                          depth,
                                                                          1);

    // Set variables
    PoseParametersBlock dX(Eigen::Affine3d::Identity());
    PoseParametersBlock dXa(Eigen::Affine3d::Identity());
    double ddepth[1] = {0.0};

    std::vector<double *> parameters_blocks;
    std::vector<const ceres::Manifold *> *manifs = new std::vector<const ceres::Manifold *>;
    parameters_blocks.push_back(dX.values());
    parameters_blocks.push_back(dXa.values());
    parameters_blocks.push_back(ddepth);
    manifs->push_back(nullptr);
    manifs->push_back(nullptr);
    manifs->push_back(nullptr);

    // Check the jacss
    ceres::NumericDiffOptions numeric_diff_options;
    ceres::GradientChecker gradient_checker(cost_fct, manifs, numeric_diff_options);
    ceres::GradientChecker::ProbeResults results;
    if (!gradient_checker.Probe(parameters_blocks.data(), 1e-9, &results)) {
        LOG(ERROR) << "An error has occurred:\n" << results.error_log;
    }

    ASSERT_NEAR((results.local_jacobians.at(0) - results.jacobians.at(0)).sum(), 0, 1e-5);
    ASSERT_NEAR((results.local_jacobians.at(1) - results.jacobians.at(1)).sum(), 0, 1e-5);
    ASSERT_NEAR((results.residuals.norm()), 0, 1e-5);
}

TEST_F(ResidualTest, scaleTest) {

    Eigen::Vector3d b1      = _sensor1->getRayCamera(_p2d1);
    Eigen::Affine3d T_c0_c0 = Eigen::Affine3d::Identity();

    // Create a Cost function
    ceres::CostFunction *cost_fct = new AngularAdjustmentCERESAnalytic::AngularErrorScaleCam0(
        b1, _rand_lmk.translation(), _sensor0->getWorld2SensorTransform(), _dT, T_c0_c0, 1);

    // Set variables
    double lambda[1] = {1.0};
    PointXYZParametersBlock dlmk(Eigen::Vector3d::Zero());

    std::vector<double *> parameters_blocks;
    std::vector<const ceres::Manifold *> *manifs = new std::vector<const ceres::Manifold *>;
    parameters_blocks.push_back(lambda);
    parameters_blocks.push_back(dlmk.values());
    manifs->push_back(nullptr);
    manifs->push_back(nullptr);

    // Check the jacss
    ceres::NumericDiffOptions numeric_diff_options;
    ceres::GradientChecker gradient_checker(cost_fct, manifs, numeric_diff_options);
    ceres::GradientChecker::ProbeResults results;
    if (!gradient_checker.Probe(parameters_blocks.data(), 1e-9, &results)) {
        LOG(ERROR) << "An error has occurred:\n" << results.error_log;
    }

    ASSERT_NEAR((results.local_jacobians.at(0) - results.jacobians.at(0)).sum(), 0, 1e-5);
    ASSERT_NEAR((results.local_jacobians.at(1) - results.jacobians.at(1)).sum(), 0, 1e-5);

    // Check the cost function with scales
    Eigen::Vector3d t_s_lmk = _sensor0->getWorld2SensorTransform() * _rand_lmk.translation();
    double depth            = t_s_lmk.norm();
    Eigen::Vector3d b0      = _sensor0->getRayCamera(_p2d0);

    // Check if the bearing vector * depth gives the actual landmark pose
    ASSERT_NEAR((t_s_lmk - b0 * depth).norm(), 0, 1e-5);

    ceres::CostFunction *cost_fct1 =
        new AngularAdjustmentCERESAnalytic::AngularErrorScaleDepth(b1, b0, _dT, T_c0_c0, depth, 1);

    // Set variables
    double ddepth[1] = {0.0};

    std::vector<double *> parameters_blocks1;
    parameters_blocks1.push_back(lambda);
    parameters_blocks1.push_back(ddepth);

    // Check the jacss
    ceres::GradientChecker gradient_checker1(cost_fct1, manifs, numeric_diff_options);
    ceres::GradientChecker::ProbeResults results1;
    if (!gradient_checker1.Probe(parameters_blocks1.data(), 1e-9, &results1)) {
        LOG(ERROR) << "An error has occurred:\n" << results1.error_log;
    }

    ASSERT_NEAR((results1.local_jacobians.at(0) - results1.jacobians.at(0)).sum(), 0, 1e-5);
    ASSERT_NEAR((results1.local_jacobians.at(1) - results1.jacobians.at(1)).sum(), 0, 1e-5);
    ASSERT_NEAR((results1.residuals.norm()), 0, 1e-5);
}

TEST_F(ResidualTest, PoseToLandmarkResidual) {

    // Generates a random pose
    Eigen::Affine3d T_f0_w            = Eigen::Affine3d::Identity();
    Eigen::Quaterniond q_rand         = Eigen::Quaterniond::UnitRandom();
    Eigen::Vector3d t_rand            = Eigen::Vector3d::Random();
    T_f0_w.affine().block(0, 0, 3, 3) = q_rand.toRotationMatrix();
    T_f0_w.translation()              = t_rand;

    // Generates a random lmk
    Eigen::Vector3d t_w_lmk  = Eigen::Vector3d::Random();
    Eigen::Vector3d t_f0_lmk = T_f0_w * t_w_lmk;

    // Create a Cost function
    ceres::CostFunction *cost_fct = new PoseToLandmarkFactor(
        t_f0_lmk, T_f0_w, t_w_lmk + 0.01 * Eigen::Vector3d::Random(), 100 * Eigen::Vector3d::Ones().asDiagonal());

    // Set variables
    PoseParametersBlock dX(Eigen::Affine3d::Identity());
    PointXYZParametersBlock dl(Eigen::Vector3d::Zero());

    std::vector<double *> parameters_blocks;
    std::vector<const ceres::Manifold *> *manifs = new std::vector<const ceres::Manifold *>;
    parameters_blocks.push_back(dX.values());
    parameters_blocks.push_back(dl.values());
    manifs->push_back(nullptr);
    manifs->push_back(nullptr);

    // Check the jacss
    ceres::NumericDiffOptions numeric_diff_options;
    ceres::GradientChecker gradient_checker(cost_fct, manifs, numeric_diff_options);
    ceres::GradientChecker::ProbeResults results;
    if (!gradient_checker.Probe(parameters_blocks.data(), 1e-5, &results)) {
        LOG(ERROR) << "An error has occurred:\n" << results.error_log;
    }

    ASSERT_NEAR((results.local_jacobians.at(0) - results.jacobians.at(0)).sum(), 0, 1e-5);
    ASSERT_NEAR((results.local_jacobians.at(1) - results.jacobians.at(1)).sum(), 0, 1e-5);
}

TEST_F(ResidualTest, RelativePose6DResidual) {

    // Generate random poses
    Eigen::Affine3d T_w_f0           = Eigen::Affine3d::Identity();
    Eigen::Quaterniond q_rand         = Eigen::Quaterniond::UnitRandom();
    Eigen::Vector3d t_rand            = Eigen::Vector3d::Random();
    T_w_f0.affine().block(0, 0, 3, 3) = q_rand.toRotationMatrix();
    T_w_f0.translation()              = t_rand;

    Eigen::Affine3d T_w_f1            = Eigen::Affine3d::Identity();
    Eigen::Quaterniond q1_rand        = Eigen::Quaterniond::UnitRandom();
    Eigen::Vector3d t1_rand           = Eigen::Vector3d::Random();
    T_w_f1.affine().block(0, 0, 3, 3) = q1_rand.toRotationMatrix();
    T_w_f1.translation()              = t1_rand;

    // Create a Cost function
    Eigen::Affine3d T_f0_f1 = T_w_f0.inverse() * T_w_f1;
    ceres::CostFunction *cost_fct = new Relative6DPose(T_w_f0, T_w_f1, T_f0_f1, Vector6d::Ones().asDiagonal());

    // Set variables
    PoseParametersBlock dX_f0(Eigen::Affine3d::Identity());
    PoseParametersBlock dX_f1(Eigen::Affine3d::Identity());
    std::vector<double *> parameters_blocks;
    std::vector<const ceres::Manifold *> *manifs = new std::vector<const ceres::Manifold *>;
    parameters_blocks.push_back(dX_f0.values());
    parameters_blocks.push_back(dX_f1.values());
    manifs->push_back(nullptr);
    manifs->push_back(nullptr);

    // Check the jacss
    ceres::NumericDiffOptions numeric_diff_options;
    ceres::GradientChecker gradient_checker(cost_fct, manifs, numeric_diff_options);
    ceres::GradientChecker::ProbeResults results;
    if (!gradient_checker.Probe(parameters_blocks.data(), 1e-5, &results)) {
        LOG(ERROR) << "An error has occurred:\n" << results.error_log;
    }

    ASSERT_NEAR((results.local_jacobians.at(0) - results.jacobians.at(0)).sum(), 0, 1e-5);
    ASSERT_NEAR((results.local_jacobians.at(1) - results.jacobians.at(1)).sum(), 0, 1e-5);
}

} // namespace isae