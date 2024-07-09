#include <fstream>
#include <gtest/gtest.h>

#include "isaeslam/data/frame.h"
#include "isaeslam/data/maps/localmap.h"
#include "isaeslam/data/sensors/IMU.h"
#include "isaeslam/optimizers/AngularAdjustmentCERESAnalytic.h"

namespace isae {

void read_line_euroc(std::string line, Eigen::Affine3d &pose, Eigen::Vector3d &v, double &ts) {
    std::istringstream s(line);
    std::string svalue;
    std::string::size_type sz;

    std::vector<double> values;
    while (getline(s, svalue, ',')) {
        values.push_back(std::stod(svalue, &sz)); // convert to double
    }

    // Deal with pose
    Eigen::Quaterniond q(values[4], values[5], values[6], values[7]);
    Eigen::Matrix3d R = q.toRotationMatrix();

    pose                            = Eigen::Affine3d::Identity();
    pose.affine().block(0, 0, 3, 3) = R;
    pose.translation()              = Eigen::Vector3d(values[1], values[2], values[3]);

    // Deal with ts (in seconds)
    ts = values[0] * 1e-9;

    // Deal with velocity
    v = Eigen::Vector3d(values[8], values[9], values[10]);
}

void write_result(std::shared_ptr<Frame> f) {

    // Write in a txt file for evaluation
    std::ofstream fw_res("result.txt", std::ofstream::out | std::ofstream::app);
    const Eigen::Matrix3d R = f->getFrame2WorldTransform().rotation();
    Eigen::Vector3d twc     = f->getFrame2WorldTransform().translation();
    fw_res << f->getTimestamp() << " " << 0 << " " << R(0, 0) << " " << R(0, 1) << " " << R(0, 2) << " " << twc.x()
           << " " << R(1, 0) << " " << R(1, 1) << " " << R(1, 2) << " " << twc.y() << " " << R(2, 0) << " " << R(2, 1)
           << " " << R(2, 2) << " " << twc.z() << "\n";
    fw_res.close();
}

void write_imu_data(double ts, Eigen::Vector3d acc) {

    // Write in a txt file for evaluation
    std::ofstream fw_res("acc.txt", std::ofstream::out | std::ofstream::app);
    fw_res << ts << "," << acc(0) << "," << acc(1) << "," << acc(2) << "\n";
    fw_res.close();
}

class ImuTest : public testing::Test {
  public:
    void SetUp() override {
        // Set Imu Config
        _imu_cfg             = std::shared_ptr<imu_config>(new imu_config());
        _imu_cfg->gyr_noise  = (0.5 * M_PI) / (180 * 60);
        _imu_cfg->bgyr_noise = 1.9393e-05;
        _imu_cfg->acc_noise  = 0.1 / 60;
        _imu_cfg->bacc_noise = 3.0000e-3;
        _imu_cfg->rate_hz    = 200;
        _imu_cfg->T_s_f      = Eigen::Affine3d::Identity();
        _acc                 = Eigen::Vector3d(0.5, 1.0, 10.81);
        _gyr                 = Eigen::Vector3d(0.1, 0.3, 0.1);

        // Set Frames and IMU
        _imu0   = std::shared_ptr<IMU>(new IMU(_imu_cfg, _acc, _gyr));
        _frame0 = std::shared_ptr<Frame>(new Frame());
        _frame0->init(_imu0, 1e9);
        _frame0->setWorld2FrameTransform(Eigen::Affine3d::Identity());
        _frame0->setKeyFrame();
        _imu0->setLastKF(_frame0);

        _imu1 = std::shared_ptr<IMU>(new IMU(_imu_cfg, _acc, _gyr));
        _imu1->setLastIMU(_imu0);
        _imu1->setLastKF(_frame0);
        _frame1 = std::shared_ptr<Frame>(new Frame());
        _frame1->init(_imu1, 1.5e9);

        _imu2 = std::shared_ptr<IMU>(new IMU(_imu_cfg, _acc, _gyr));
        _imu2->setLastIMU(_imu1);
        _imu2->setLastKF(_frame0);
        _frame2 = std::shared_ptr<Frame>(new Frame());
        _frame2->init(_imu2, 2e9);
    }

    std::shared_ptr<imu_config> _imu_cfg;
    Eigen::Vector3d _acc;
    Eigen::Vector3d _gyr;
    std::shared_ptr<Frame> _frame0;
    std::shared_ptr<IMU> _imu0;
    std::shared_ptr<Frame> _frame1;
    std::shared_ptr<IMU> _imu1;
    std::shared_ptr<Frame> _frame2;
    std::shared_ptr<IMU> _imu2;
};

TEST_F(ImuTest, ImuTestBase) {

    // Check measurements
    ASSERT_EQ(_imu0->getAcc(), Eigen::Vector3d(0.5, 1.0, 10.81));
    ASSERT_EQ(_imu0->getGyr(), Eigen::Vector3d(0.1, 0.3, 0.1));

    // Check preintegration measurement (without biases)
    _imu1->processIMU();
    Eigen::Matrix3d dR = geometry::exp_so3(_gyr / 2);
    ASSERT_EQ((dR * _imu1->getDeltaR().transpose()).trace(), 3);

    Eigen::Vector3d dv = _acc / 2;
    ASSERT_EQ((dv - _imu1->getDeltaV()).norm(), 0);

    Eigen::Vector3d dp = 0.5 * _acc * 0.5 * 0.5;
    ASSERT_EQ((dp - _imu1->getDeltaP()).norm(), 0);

    // Check pose estimation
    Eigen::Affine3d T_f0_f1;
    _imu1->estimateTransform(_frame0, _frame1, T_f0_f1);
    ASSERT_EQ((dR * T_f0_f1.rotation().transpose()).trace(), 3);
    Eigen::Vector3d t_f0_f1 = dp + 0.5 * g * 0.5 * 0.5;
    ASSERT_EQ((T_f0_f1.translation() - t_f0_f1).norm(), 0);

    // Check preintegration measurement with biases
    Eigen::Vector3d ba(0.1, 0.2, 0.3);
    Eigen::Vector3d bg(0.2, 0.3, 0.1);
    _imu0->setBa(ba);
    _imu0->setBg(bg);
    _imu1->processIMU();

    // Check preintegration measurement (without biases)
    dR = geometry::exp_so3((_gyr - bg) * 0.5);
    ASSERT_EQ((dR * _imu1->getDeltaR().transpose()).trace(), 3);

    dv = (_acc - ba) / 2;
    ASSERT_EQ((dv - _imu1->getDeltaV()).norm(), 0);

    dp = 0.5 * (_acc - ba) * 0.5 * 0.5;
    ASSERT_EQ((dp - _imu1->getDeltaP()).norm(), 0);
}

TEST_F(ImuTest, ImuNewMeas) {

    // Check with two measurements
    _imu1->processIMU();
    _imu2->processIMU();

    // Check preintegration meas
    Eigen::Matrix3d dR = geometry::exp_so3(_gyr);
    ASSERT_EQ((dR * _imu2->getDeltaR().transpose()).trace(), 3);
    Eigen::Vector3d dv = _acc * 0.5 + _imu1->getDeltaR() * _acc * 0.5;
    ASSERT_EQ((dv - _imu2->getDeltaV()).norm(), 0);
    Eigen::Vector3d dp =
        0.5 * _acc * 0.5 * 0.5 + _imu1->getDeltaV() * 0.5 + 0.5 * _imu1->getDeltaR() * _acc * 0.5 * 0.5;
    ASSERT_EQ((dp - _imu2->getDeltaP()).norm(), 0);
}

// Here we use the same test as in gtsam:
// https://github.com/borglab/gtsam/blob/develop/gtsam/navigation/tests/testImuFactor.cpp

TEST_F(ImuTest, checkCov) {

    // Set Frame and IMU according to gtsam test values
    Eigen::Vector3d acc(0.1, 0.0, 0.0);
    Eigen::Vector3d gyr(M_PI / 100.0, 0.0, 0.0);
    _imu_cfg->rate_hz = 2;
    _imu0             = std::shared_ptr<IMU>(new IMU(_imu_cfg, acc, gyr));
    _frame0           = std::shared_ptr<Frame>(new Frame());
    _frame0->init(_imu0, 1e9);
    _frame0->setWorld2FrameTransform(Eigen::Affine3d::Identity());
    _frame0->setKeyFrame();

    _imu1 = std::shared_ptr<IMU>(new IMU(_imu_cfg, _acc, _gyr));
    _imu1->setLastIMU(_imu0);
    _imu1->setLastKF(_frame0);
    _frame1 = std::shared_ptr<Frame>(new Frame());
    _frame1->init(_imu1, 1.5e9);
    _imu1->processIMU();
    Eigen::MatrixXd expected_cov = Eigen::Matrix<double, 9, 9>::Zero();
    expected_cov << 1.0577e-08, 0, 0, 0, 0, 0, 0, 0, 0, //
        0, 1.0577e-08, 0, 0, 0, 0, 0, 0, 0,             //
        0, 0, 1.0577e-08, 0, 0, 0, 0, 0, 0,             //
        0, 0, 0, 1.38889e-06, 0, 0, 3.47222e-07, 0, 0,  //
        0, 0, 0, 0, 1.38889e-06, 0, 0, 3.47222e-07, 0,  //
        0, 0, 0, 0, 0, 1.38889e-06, 0, 0, 3.47222e-07,  //
        0, 0, 0, 3.47222e-07, 0, 0, 5.00868e-05, 0, 0,  //
        0, 0, 0, 0, 3.47222e-07, 0, 0, 5.00868e-05, 0,  //
        0, 0, 0, 0, 0, 3.47222e-07, 0, 0, 5.00868e-05;
    ASSERT_NEAR((expected_cov - _imu1->getCov()).trace(), 0, 1e-9);
}

TEST_F(ImuTest, checkJacobiansBiasGyr) {

    // First processing
    _imu1->processIMU();
    double dt                         = 0.5;
    Eigen::Matrix3d J_rk              = geometry::so3_rightJacobian((_gyr)*dt);
    Eigen::Matrix3d expected_J_dbg    = -J_rk * dt;
    Eigen::Matrix3d expected_J_dv_dba = -Eigen::Matrix3d::Identity() * dt;
    Eigen::Matrix3d expected_J_dv_dbg = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d expected_J_dp_dba = -0.5 * Eigen::Matrix3d::Identity() * dt * dt;
    Eigen::Matrix3d expected_J_dp_dbg = Eigen::Matrix3d::Zero();

    ASSERT_NEAR((_imu1->_J_dR_bg * expected_J_dbg.inverse()).trace(), 3, 1e-9);
    ASSERT_NEAR((_imu1->_J_dv_ba * expected_J_dv_dba.inverse()).trace(), 3, 1e-9);
    ASSERT_NEAR((_imu1->_J_dv_bg - expected_J_dv_dbg).sum(), 0, 1e-9);
    ASSERT_NEAR((_imu1->_J_dp_ba - expected_J_dp_dba).sum(), 0, 1e-9);
    ASSERT_NEAR((_imu1->_J_dp_bg - expected_J_dp_dbg).sum(), 0, 1e-9);

    // Second processing
    _imu2->processIMU();
    Eigen::Matrix3d dR = geometry::exp_so3(_gyr * dt);
    expected_J_dbg     = dR.transpose() * _imu1->_J_dR_bg - J_rk * dt;
    expected_J_dv_dba  = expected_J_dv_dba - _imu1->getDeltaR() * dt;
    expected_J_dv_dbg  = -_imu1->getDeltaR() * geometry::skewMatrix(_acc) * _imu1->_J_dR_bg * dt;
    expected_J_dp_dba  = expected_J_dp_dba + _imu1->_J_dv_ba * dt - 0.5 * _imu1->getDeltaR() * dt * dt;
    expected_J_dp_dbg =
        _imu1->_J_dv_bg * dt - 0.5 * _imu1->getDeltaR() * geometry::skewMatrix(_acc) * _imu1->_J_dR_bg * dt * dt;

    ASSERT_NEAR((_imu2->_J_dR_bg * expected_J_dbg.inverse()).trace(), 3, 1e-9);
    ASSERT_NEAR((_imu2->_J_dv_ba * expected_J_dv_dba.inverse()).trace(), 3, 1e-9);
    ASSERT_NEAR((_imu2->_J_dv_bg - expected_J_dv_dbg).sum(), 0, 1e-9);
    ASSERT_NEAR((_imu2->_J_dp_ba - expected_J_dp_dba).sum(), 0, 1e-9);
    ASSERT_NEAR((_imu2->_J_dp_bg - expected_J_dp_dbg).sum(), 0, 1e-9);
}

TEST_F(ImuTest, predictionPositionVelocity) {

    // We set a rotated inertial frame
    Eigen::Affine3d T_i_f = Eigen::Affine3d::Identity();
    T_i_f.affine().block(0, 0, 3, 3) << 0.38001193, 0.16469125, 0.91020202, 0.03067918, -0.9857245, 0.16554758,
        0.92447267, -0.0349858, -0.37963966;
    T_i_f.affine().block(0, 3, 3, 1) = Eigen::Vector3d::Ones();

    Eigen::Vector3d gyr, acc, ba, bg;
    gyr << 0, 0, 0;
    acc << 0, 0, 10.81;
    acc = T_i_f.rotation().transpose() * acc;
    ba << 0.0, 0, 0;
    bg << 0, 0, 0;
    double dt         = 0.001;
    _imu_cfg->rate_hz = 1000;

    std::shared_ptr<IMU> cur_imu     = std::shared_ptr<IMU>(new IMU(_imu_cfg, acc, gyr));
    std::shared_ptr<Frame> cur_frame = std::shared_ptr<Frame>(new Frame());
    cur_frame->init(cur_imu, 1e9);
    cur_frame->setWorld2FrameTransform(T_i_f.inverse());
    cur_frame->setKeyFrame();
    cur_imu->setBa(ba);
    cur_imu->setBg(bg);
    std::shared_ptr<IMU> last_imu = cur_imu;

    std::shared_ptr<Frame> lastKF = cur_frame;

    for (int i = 1; i < 1001; i++) {
        cur_imu = std::shared_ptr<IMU>(new IMU(_imu_cfg, acc, gyr));
        cur_imu->setLastIMU(last_imu);
        cur_imu->setLastKF(lastKF);

        // Create frame and process IMU
        cur_frame = std::shared_ptr<Frame>(new Frame());
        cur_frame->init(cur_imu, 1e9 + (dt * i) * 1e9);
        cur_imu->processIMU();

        // Set last imu
        last_imu = cur_imu;
    }

    ASSERT_NEAR((cur_frame->getFrame2WorldTransform().translation() - Eigen::Vector3d(1, 1, 1.5)).norm(), 0, 1e-5);
    ASSERT_NEAR((cur_imu->getVelocity() - Eigen::Vector3d(0, 0, 1)).norm(), 0, 1e-5);
    ASSERT_NEAR((cur_frame->getFrame2WorldTransform().rotation().transpose() * T_i_f.rotation()).trace(), 3, 1e-5);

    // Test IMU Factor

    // Create parameter blocks
    PointXYZParametersBlock dvi(Eigen::Vector3d::Zero());
    PointXYZParametersBlock dvj(Eigen::Vector3d::Zero());
    PointXYZParametersBlock dba(Eigen::Vector3d::Zero());
    PointXYZParametersBlock dbg(Eigen::Vector3d::Zero());
    PoseParametersBlock dX_i(Eigen::Affine3d::Identity());
    PoseParametersBlock dX_j(Eigen::Affine3d::Identity());
    double lambda[1] = {0.0};

    std::vector<double *> parameters_blocks;
    std::vector<const ceres::Manifold *> *manifs = new std::vector<const ceres::Manifold *>;
    manifs->push_back(nullptr);
    manifs->push_back(nullptr);
    manifs->push_back(nullptr);
    manifs->push_back(nullptr);
    manifs->push_back(nullptr);
    manifs->push_back(nullptr);
    parameters_blocks.push_back(dX_i.values());
    parameters_blocks.push_back(dX_j.values());
    parameters_blocks.push_back(dvi.values());
    parameters_blocks.push_back(dvj.values());
    parameters_blocks.push_back(dba.values());
    parameters_blocks.push_back(dbg.values());

    // Create cost fct
    ceres::CostFunction *cost_fct = new IMUFactor(lastKF->getIMU(), cur_frame->getIMU());

    // Create residual and jaocobian objects
    Eigen::VectorXd residuals;
    residuals.resize(cost_fct->num_residuals());

    std::vector<int> block_sizes = cost_fct->parameter_block_sizes();
    double **raw_jacobians       = new double *[block_sizes.size()];
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    jacobians.resize(block_sizes.size());

    for (size_t i = 0; i < block_sizes.size(); i++) {
        jacobians[i].resize(cost_fct->num_residuals(), block_sizes[i]);
        raw_jacobians[i] = jacobians[i].data();
    }

    cost_fct->Evaluate(parameters_blocks.data(), residuals.data(), raw_jacobians);
    ASSERT_NEAR(residuals.norm(), 0, 1e-3);

    // Check the jacss
    ceres::NumericDiffOptions numeric_diff_options;
    ceres::GradientChecker gradient_checker(cost_fct, manifs, numeric_diff_options);
    ceres::GradientChecker::ProbeResults results;
    if (!gradient_checker.Probe(parameters_blocks.data(), 1e-5, &results)) {
        LOG(ERROR) << "An error has occurred:\n" << results.error_log;
    }

    ASSERT_NEAR((results.local_jacobians.at(0) - results.jacobians.at(0)).sum(), 0, 1e-5);
    ASSERT_NEAR((results.local_jacobians.at(1) - results.jacobians.at(1)).sum(), 0, 1e-5);
    ASSERT_NEAR((results.local_jacobians.at(2) - results.jacobians.at(2)).sum(), 0, 1e-5);
    ASSERT_NEAR((results.local_jacobians.at(3) - results.jacobians.at(3)).sum(), 0, 1e-5);
    ASSERT_NEAR((results.local_jacobians.at(4) - results.jacobians.at(4)).sum(), 0, 1e-5);
    ASSERT_NEAR((results.local_jacobians.at(5) - results.jacobians.at(5)).sum(), 0, 1e-5);

    // Test Inertial optimization
    std::shared_ptr<LocalMap> local_map = std::make_shared<LocalMap>(0, 10, 0);
    lastKF->setKeyFrame();
    lastKF->setPrior(T_i_f.inverse(), 100 * Vector6d::Ones());                         // To constrain the problem
    cur_frame->setPrior(cur_frame->getWorld2FrameTransform(), 100 * Vector6d::Ones()); // To constrain the problem
    cur_frame->setKeyFrame();
    local_map->addFrame(lastKF);
    local_map->addFrame(cur_frame);

    // Add error in states
    Vector6d err_pose;
    err_pose << 0.0000, -0.000, 0.000, 0.1, 0.05, -0.01;
    lastKF->getIMU()->setVelocity(lastKF->getIMU()->getVelocity());
    cur_frame->getIMU()->setVelocity(cur_frame->getIMU()->getVelocity() + Eigen::Vector3d(0.04, 0.02, -0.02));
    cur_frame->setWorld2FrameTransform(cur_frame->getWorld2FrameTransform() * geometry::se3_Vec6dtoRT(err_pose));

    // Solve the SLAM problem
    isae::AngularAdjustmentCERESAnalytic ceres_ba;
    ceres_ba.localMapVIOptimization(local_map, 0);

    // Check states
    ASSERT_NEAR((cur_frame->getFrame2WorldTransform().translation() - Eigen::Vector3d(1, 1, 1.5)).norm(), 0, 1e-2);
    ASSERT_NEAR((cur_imu->getVelocity() - Eigen::Vector3d(0, 0, 1)).norm(), 0, 1e-2);
    ASSERT_NEAR((cur_frame->getFrame2WorldTransform().rotation().transpose() * T_i_f.rotation()).trace(), 3, 1e-5);

    // Test IMU Factor INIT
    ceres::CostFunction *cost_fct1 = new IMUFactorInit(lastKF->getIMU(), cur_frame->getIMU());

    double r_w_i[2] = {0.0, 0.0};
    std::vector<double *> parameters_blocks1;
    std::vector<const ceres::Manifold *> *manifs1 = new std::vector<const ceres::Manifold *>;

    manifs1->push_back(nullptr);
    manifs1->push_back(nullptr);
    manifs1->push_back(nullptr);
    manifs1->push_back(nullptr);
    manifs1->push_back(nullptr);
    manifs1->push_back(nullptr);
    parameters_blocks1.push_back(r_w_i);
    parameters_blocks1.push_back(dvi.values());
    parameters_blocks1.push_back(dvj.values());
    parameters_blocks1.push_back(dba.values());
    parameters_blocks1.push_back(dbg.values());
    parameters_blocks1.push_back(lambda);

    // Check the jacss
    ceres::GradientChecker gradient_checker1(cost_fct1, manifs1, numeric_diff_options);
    ceres::GradientChecker::ProbeResults results1;
    if (!gradient_checker1.Probe(parameters_blocks1.data(), 1e-5, &results1)) {
        LOG(ERROR) << "An error has occurred:\n" << results1.error_log;
    }

    ASSERT_NEAR((results.local_jacobians.at(0) - results.jacobians.at(0)).sum(), 0, 1e-5);
    ASSERT_NEAR((results.local_jacobians.at(1) - results.jacobians.at(1)).sum(), 0, 1e-5);
    ASSERT_NEAR((results.local_jacobians.at(2) - results.jacobians.at(2)).sum(), 0, 1e-5);
    ASSERT_NEAR((results.local_jacobians.at(3) - results.jacobians.at(3)).sum(), 0, 1e-5);
    ASSERT_NEAR((results.local_jacobians.at(4) - results.jacobians.at(4)).sum(), 0, 1e-5);
    ASSERT_NEAR((results.local_jacobians.at(5) - results.jacobians.at(5)).sum(), 0, 1e-5);
}

TEST_F(ImuTest, biasEstimation) {

    // Let's see if the bias are estimated with a pose prior
    _frame0->setPrior(Eigen::Affine3d::Identity(), 100 * Vector6d::Ones());
    _imu0->setBa(Eigen::Vector3d(0.5, 1., 1.0));
    _imu0->setBg(Eigen::Vector3d(0.1, 0.3, 0.1));
    _frame1->setWorld2FrameTransform(Eigen::Affine3d::Identity());
    _frame1->setPrior(Eigen::Affine3d::Identity(), 100 * Vector6d::Ones());

    // Build the local map
    std::shared_ptr<LocalMap> local_map = std::make_shared<LocalMap>(0, 10, 0);
    local_map->addFrame(_frame0);
    _imu1->processIMU();
    _frame1->setKeyFrame();
    local_map->addFrame(_frame1);

    // Solve the SLAM problem
    isae::AngularAdjustmentCERESAnalytic ceres_ba;
    ceres_ba.localMapVIOptimization(local_map, 0);

    // check bias
    ASSERT_NEAR((_imu0->getBg() - Eigen::Vector3d(0.1, 0.3, 0.1)).norm(), 0, 1e-5);
    ASSERT_NEAR((_imu0->getBa() - Eigen::Vector3d(0.5, 1, 1)).norm(), 0, 1e-5);
}

// Combination of a rotation and a translation
// The results of the free integration comes from https://github.com/Aceinna/gnss-ins-sim/tree/master

TEST_F(ImuTest, predictionWithRotation) {

    // The IMU is set on the frame 1 from Aceinna
    Eigen::Affine3d T_i_f = Eigen::Affine3d::Identity();
    T_i_f.affine().block(0, 0, 3, 3) << 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0;

    Eigen::Vector3d gyr, acc, ba, bg;
    gyr << 0.5, 0, 0;
    acc << -1, 0, -9.81;
    ba << 0.0, 0, 0;
    bg << 0, 0, 0;
    double dt = 0.005;

    std::shared_ptr<IMU> cur_imu     = std::shared_ptr<IMU>(new IMU(_imu_cfg, acc, gyr));
    std::shared_ptr<Frame> cur_frame = std::shared_ptr<Frame>(new Frame());
    cur_frame->init(cur_imu, 1e9);
    cur_frame->setWorld2FrameTransform(T_i_f.inverse());
    cur_frame->setKeyFrame();
    std::shared_ptr<IMU> last_imu = cur_imu;

    std::shared_ptr<Frame> lastKF = cur_frame;

    for (int i = 1; i < 202; i++) {
        cur_imu = std::shared_ptr<IMU>(new IMU(_imu_cfg, acc, gyr));
        cur_imu->setLastIMU(last_imu);
        cur_imu->setLastKF(lastKF);

        // Create frame and process IMU
        cur_frame = std::shared_ptr<Frame>(new Frame());
        cur_frame->init(cur_imu, 1e9 + (dt * i) * 1e9);
        cur_imu->processIMU();

        // Estimate transform
        Eigen::Affine3d dT;
        cur_frame->getIMU()->estimateTransform(lastKF, cur_frame, dT);
        cur_frame->setWorld2FrameTransform(dT.inverse() * lastKF->getWorld2FrameTransform());

        // Set last imu
        last_imu = cur_imu;
    }

    ASSERT_NEAR(
        (T_i_f.inverse() * cur_frame->getFrame2WorldTransform().translation() - Eigen::Vector3d(-0.505, 0.813, 0.1))
            .norm(),
        0,
        1e-2);
    ASSERT_NEAR((T_i_f.inverse() * cur_imu->getVelocity() - Eigen::Vector3d(-1, 2.41, 0.4)).norm(), 0, 1e-2);

    // Second set of measurements
    cur_frame->setKeyFrame();
    lastKF = cur_frame;
    gyr << 0.5, 0.2, 0.04;
    acc << -1, 0.05, -9.81;

    for (int i = 202; i < 401; i++) {
        cur_imu = std::shared_ptr<IMU>(new IMU(_imu_cfg, acc, gyr));
        cur_imu->setLastIMU(last_imu);
        cur_imu->setLastKF(lastKF);

        // Create frame and process IMU
        cur_frame = std::shared_ptr<Frame>(new Frame());
        cur_frame->init(cur_imu, 1e9 + (dt * i) * 1e9);
        cur_imu->processIMU();

        // Estimate transform
        Eigen::Affine3d dT;
        cur_frame->getIMU()->estimateTransform(lastKF, cur_frame, dT);
        cur_frame->setWorld2FrameTransform(dT.inverse() * lastKF->getWorld2FrameTransform());

        // Set last imu
        last_imu = cur_imu;
    }

    ASSERT_NEAR(
        (T_i_f.inverse() * cur_frame->getFrame2WorldTransform().translation() - Eigen::Vector3d(-2.31, 6.18, 1.62))
            .norm(),
        0,
        1e-2);
    ASSERT_NEAR((T_i_f.inverse() * cur_imu->getVelocity() - Eigen::Vector3d(-2.95, 8.91, 3.24)).norm(), 0, 1e-2);
}

TEST_F(ImuTest, predictionWithRotation2) {

    // The IMU is set on the frame 1 from Aceinna
    Eigen::Affine3d T_i_f = Eigen::Affine3d::Identity();
    T_i_f.affine().block(0, 0, 3, 3) << 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0;

    Eigen::Vector3d gyr, acc, ba, bg;
    gyr << 0.5, 0.2, 0.04;
    acc << -1, 0.05, -9.81;
    ba << 0.0, 0, 0;
    bg << 0, 0, 0;
    double dt = 0.005;

    std::shared_ptr<IMU> cur_imu     = std::shared_ptr<IMU>(new IMU(_imu_cfg, acc, gyr));
    std::shared_ptr<Frame> cur_frame = std::shared_ptr<Frame>(new Frame());
    cur_frame->init(cur_imu, 1e9);
    cur_frame->setWorld2FrameTransform(T_i_f.inverse());
    cur_frame->setKeyFrame();
    std::shared_ptr<IMU> last_imu = cur_imu;

    std::shared_ptr<Frame> lastKF = cur_frame;

    for (int i = 1; i < 201; i++) {
        cur_imu = std::shared_ptr<IMU>(new IMU(_imu_cfg, acc, gyr));
        cur_imu->setLastIMU(last_imu);
        cur_imu->setLastKF(lastKF);

        // Create frame and process IMU
        cur_frame = std::shared_ptr<Frame>(new Frame());
        cur_frame->init(cur_imu, 1e9 + (dt * i) * 1e9);
        cur_imu->processIMU();

        // Estimate transform
        Eigen::Affine3d dT;
        cur_frame->getIMU()->estimateTransform(lastKF, cur_frame, dT);
        cur_frame->setWorld2FrameTransform(dT.inverse() * lastKF->getWorld2FrameTransform());

        // Set last imu
        last_imu = cur_imu;
    }

    ASSERT_NEAR((T_i_f.inverse() * cur_frame->getFrame2WorldTransform().translation() -
                 Eigen::Vector3d(-0.82143062, 0.80412303, 0.15357111))
                    .norm(),
                0,
                1e-2);
    ASSERT_NEAR((T_i_f.inverse() * cur_imu->getVelocity() - Eigen::Vector3d(-1.97810799, 2.38035184, 0.5780088)).norm(),
                0,
                1e-2);
}

TEST_F(ImuTest, simuEuroc) {

    // Get the txt file for the groundtruth
    std::string gt_path = "../tests/euroc_gt.csv";
    std::fstream gt_file;

    // Imu measurement and poses as vectors
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> meas_vec;
    std::vector<Eigen::Affine3d> pose_vec;
    std::vector<Eigen::Vector3d> vel_vec;
    std::vector<double> ts_vec;

    gt_file.open(gt_path, std::ios::in);
    if (gt_file.is_open()) { // checking whether the file is open
        std::string tp;

        double ts, tsp;           // current and previous timestamp
        Eigen::Vector3d v, vp;    // current and previous velocity
        Eigen::Affine3d T, Tp;    // current and previous pose
        Eigen::Vector3d acc, gyr; // accelero and gyro meas

        getline(gt_file, tp); // skip the first line

        // Init the derivation
        getline(gt_file, tp);
        read_line_euroc(tp, Tp, vp, tsp);

        getline(gt_file, tp);
        read_line_euroc(tp, T, v, ts);

        Tp  = T;
        vp  = v;
        tsp = ts;

        while (getline(gt_file, tp)) { // read data from file object and put it into string.
            read_line_euroc(tp, T, v, ts);

            // Compute accelero and gyro measurements
            // v = (1 / (ts - tsp)) * (T.translation() - Tp.translation());
            acc = (1 / (ts - tsp)) * (T.rotation().transpose() * (v - vp)) - T.rotation().transpose() * g;
            gyr = (1 / (ts - tsp)) * geometry::log_so3((Tp.rotation().transpose() * T.rotation()));

            // Fill vectors
            meas_vec.push_back(std::make_pair(acc, gyr));
            vel_vec.push_back(vp);
            pose_vec.push_back(Tp);
            ts_vec.push_back(tsp * 1e9);

            // Set previous values
            Tp  = T;
            vp  = v;
            tsp = ts;
        }

        gt_file.close(); // close the file object.
    }

    // TEST IMU INTEGRATION

    // Set the first frame
    std::shared_ptr<Frame> frame0 = std::shared_ptr<Frame>(new Frame());
    std::shared_ptr<IMU> imu0     = std::make_shared<IMU>(_imu_cfg, meas_vec.at(0).first, meas_vec.at(0).second);
    frame0->init(imu0, ts_vec.at(0));
    frame0->setWorld2FrameTransform(pose_vec.at(0).inverse());
    frame0->setKeyFrame();
    imu0->setLastKF(frame0);
    imu0->setVelocity(vel_vec.at(0));
    write_result(frame0);

    double ts, tsp, dt;    // current and previous timestamp
    Eigen::Affine3d T, Tp; // current and previous pose
    Eigen::Vector3d v, vp; // current and previous velocity
    tsp = ts_vec.at(0);
    vp  = vel_vec.at(0);
    Tp  = pose_vec.at(0);

    for (uint i = 1; i < meas_vec.size(); i++) {
        std::shared_ptr<Frame> frame = std::shared_ptr<Frame>(new Frame());
        ts                           = ts_vec.at(i);
        dt                           = (ts - tsp) * 1e-9;

        // Create and process IMU
        std::shared_ptr<IMU> imu = std::make_shared<IMU>(_imu_cfg, meas_vec.at(i).first, meas_vec.at(i).second);
        frame->init(imu, ts);
        imu->setLastKF(frame0);
        imu->setLastIMU(imu0);
        imu->processIMU();
        imu0 = imu;

        // Simple integration
        Eigen::Matrix3d dR           = geometry::exp_so3(meas_vec.at(i - 1).second * dt);
        Eigen::Matrix3d R_w_f        = Tp.rotation();
        v                            = vp + g * dt + R_w_f * meas_vec.at(i - 1).first * dt;
        T.affine().block(0, 0, 3, 3) = Tp.rotation() * dR;
        T.translation() =
            Tp.translation() + vp * dt + 0.5 * g * dt * dt + 0.5 * R_w_f * meas_vec.at(i - 1).first * dt * dt;
        vp  = v;
        Tp  = T;
        tsp = ts;

        ASSERT_NEAR(
            ((T * frame->getWorld2FrameTransform()).matrix() - Eigen::Affine3d::Identity().matrix()).norm(), 0, 0.1);

        // if (i % 50 == 0) {
        //     write_result(frame);
        // }
    }

    // TEST IMU INITIALIZATION

    int idx_start                       = 2000;                                 // Start after 10 seconds
    double dt_kf                        = 0.5;                                  // t between kf
    std::shared_ptr<LocalMap> local_map = std::make_shared<LocalMap>(0, 10, 0); // The sliding window
    double scale_factor                 = 0.5;                                  // Scale factor to be recovered
    std::unordered_map<double, Eigen::Affine3d> map_ts_gt; // A map to stack the gt poses of the local map

    // Set the first frame
    frame0 = std::shared_ptr<Frame>(new Frame());
    imu0   = std::make_shared<IMU>(_imu_cfg, meas_vec.at(idx_start).first, meas_vec.at(idx_start).second);
    frame0->init(imu0, ts_vec.at(idx_start));
    Eigen::Affine3d T_w_f0 = pose_vec.at(idx_start);
    map_ts_gt.emplace(ts_vec.at(idx_start), T_w_f0);
    T_w_f0.translation() *= scale_factor;
    frame0->setWorld2FrameTransform(T_w_f0.inverse());
    frame0->setKeyFrame();
    imu0->setLastKF(frame0);
    imu0->setVelocity(vel_vec.at(idx_start) * scale_factor);
    local_map->addFrame(frame0);
    tsp = ts_vec.at(idx_start);

    for (uint i = idx_start + 1; i < meas_vec.size(); i++) {
        std::shared_ptr<Frame> frame = std::shared_ptr<Frame>(new Frame());
        ts                           = ts_vec.at(i);
        dt                           = (ts - tsp) * 1e-9;

        // Create and process IMU
        std::shared_ptr<IMU> imu = std::make_shared<IMU>(_imu_cfg, meas_vec.at(i).first, meas_vec.at(i).second);
        frame->init(imu, ts);
        imu->setLastKF(frame0);
        imu->setLastIMU(imu0);
        imu->processIMU();
        imu0 = imu;

        // Compute scaled pose and velocity
        Eigen::Affine3d T_w_f = pose_vec.at(i);
        map_ts_gt.emplace(ts, T_w_f);
        T_w_f.translation() *= scale_factor;
        frame->setWorld2FrameTransform(T_w_f.inverse());
        imu0->setVelocity(vel_vec.at(i) * scale_factor);

        // Vote KF
        if (dt > dt_kf) {
            frame->setKeyFrame();
            local_map->addFrame(frame);
            frame0 = frame;
            tsp    = ts;
        }

        // Break the loop if enough KF
        if (local_map->getMapSize() == 10)
            break;
    }

    // Solve the init problem
    Eigen::Matrix3d R_w_i;
    isae::AngularAdjustmentCERESAnalytic ceres_ba;
    ceres_ba.VIInit(local_map, R_w_i, true);

    std::cout << "Rotation Matrix : \n" << R_w_i << std::endl;

    for (auto frame : local_map->getFrames()) {
        Eigen::Affine3d T_w_f = map_ts_gt.at(frame->getTimestamp());
        ASSERT_NEAR(((T_w_f * frame->getWorld2FrameTransform()).matrix() - Eigen::Affine3d::Identity().matrix()).norm(),
                    0,
                    0.02);
    }
}

TEST_F(ImuTest, TestPreInteg) {
    // Measurements
    const double a = 0.1, w = M_PI / 100.0;
    Eigen::Vector3d measured_acc(a, 0.0, 0.0);
    Eigen::Vector3d measured_gyr(w, 0.0, 0.0);
    double deltaT = 0.5;

    // Expected pre-integrated values
    Eigen::Vector3d expectedDeltaR1(w * deltaT, 0.0, 0.0);
    Eigen::Vector3d expectedDeltaP1(0.5 * a * deltaT*deltaT, 0, 0);
    Eigen::Vector3d expectedDeltaV1(0.05, 0.0, 0.0);

    // Set Frames and IMU
    _imu0   = std::shared_ptr<IMU>(new IMU(_imu_cfg, measured_acc, measured_gyr));
    _frame0 = std::shared_ptr<Frame>(new Frame());
    _frame0->init(_imu0, 1e9);
    _frame0->setWorld2FrameTransform(Eigen::Affine3d::Identity());
    _frame0->setKeyFrame();
    _imu0->setLastKF(_frame0);

    _imu1 = std::shared_ptr<IMU>(new IMU(_imu_cfg, measured_acc, measured_gyr));
    _imu1->setLastIMU(_imu0);
    _imu1->setLastKF(_frame0);
    _frame1 = std::shared_ptr<Frame>(new Frame());
    _frame1->init(_imu1, 1.5e9);

    // Pre integrate
    _imu1->processIMU();
    ASSERT_EQ((_imu1->getDeltaR() - geometry::exp_so3(expectedDeltaR1)).sum(), 0);
    ASSERT_EQ((_imu1->getDeltaP() - expectedDeltaP1).norm(), 0);
    ASSERT_EQ((_imu1->getDeltaV() - expectedDeltaV1).norm(), 0);

    // Integrate again
    Eigen::Vector3d expectedDeltaR2(2.0 * 0.5 * M_PI / 100.0, 0.0, 0.0);
    Eigen::Vector3d expectedDeltaP2(0.025 + expectedDeltaP1(0) + 0.5 * 0.1 * 0.5 * 0.5, 0, 0);
    Eigen::Vector3d expectedDeltaV2 = Eigen::Vector3d(0.05, 0.0, 0.0) +
                                geometry::exp_so3(expectedDeltaR1) * measured_acc * 0.5;

    _imu2 = std::shared_ptr<IMU>(new IMU(_imu_cfg, measured_acc, measured_gyr));
    _imu2->setLastIMU(_imu1);
    _imu2->setLastKF(_frame0);
    _frame2 = std::shared_ptr<Frame>(new Frame());
    _frame2->init(_imu2, 2e9);
    _imu2->processIMU();
    ASSERT_NEAR((_imu2->getDeltaR() - geometry::exp_so3(expectedDeltaR2)).sum(), 0, 0.000001);
    ASSERT_EQ((_imu2->getDeltaP() - expectedDeltaP2).norm(), 0);
    ASSERT_EQ((_imu2->getDeltaV() - expectedDeltaV2).norm(), 0);
}

} // namespace isae