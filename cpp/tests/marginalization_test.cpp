#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "isaeslam/data/features/Point2D.h"
#include "isaeslam/data/frame.h"
#include "isaeslam/data/landmarks/Point3D.h"
#include "isaeslam/data/sensors/Camera.h"
#include "isaeslam/optimizers/BundleAdjustmentCERESAnalytic.h"

namespace isae {

class MarginalizationTest : public testing::Test {
    /*
    * Simple Toy Example for Marginalization

    *   l0   l1
    *   |  /    \
    *   | /      \
    *   x0        x1
    *     \      /
    *      \    /
    *        l2
    * */
  public:
    void SetUp() override {

        // Set Frames
        _frame0 = std::shared_ptr<Frame>(new Frame());
        _frame1 = std::shared_ptr<Frame>(new Frame());

        // Set Sensors
        _K       = Eigen::Matrix3d::Identity();
        _K(0, 0) = 100;
        _K(1, 1) = 100;
        _K(0, 2) = 400;
        _K(1, 2) = 400;

        // We set the two sensors for frame 0 (We only marginalize stereo factors)
        _sensor0l = std::shared_ptr<Camera>(new Camera(cv::Mat::zeros(800, 800, CV_16F), _K));
        _sensor0r = std::shared_ptr<Camera>(new Camera(cv::Mat::zeros(800, 800, CV_16F), _K));
        std::vector<std::shared_ptr<ImageSensor>> sensors_frame0;
        sensors_frame0.push_back(_sensor0l);
        sensors_frame0.push_back(_sensor0r);
        _frame0->init(sensors_frame0, 0);
        _sensor0l->setFrame2SensorTransform(Eigen::Affine3d::Identity());
        Eigen::Affine3d T_s01_f0   = Eigen::Affine3d::Identity();
        T_s01_f0.translation().y() = 0.2;
        _sensor0r->setFrame2SensorTransform(T_s01_f0);

        // We set the two sensors for frame 0 (We only marginalize stereo factors)
        _sensor1l = std::shared_ptr<Camera>(new Camera(cv::Mat::zeros(800, 800, CV_16F), _K));
        _sensor1r = std::shared_ptr<Camera>(new Camera(cv::Mat::zeros(800, 800, CV_16F), _K));
        std::vector<std::shared_ptr<ImageSensor>> sensors_frame1;
        sensors_frame1.push_back(_sensor1l);
        sensors_frame1.push_back(_sensor1r);
        _frame1->init(sensors_frame1, 1.0);
        _sensor1l->setFrame2SensorTransform(Eigen::Affine3d::Identity());
        Eigen::Affine3d T_s11_f1   = Eigen::Affine3d::Identity();
        T_s11_f1.translation().y() = 0.2;
        _sensor1r->setFrame2SensorTransform(T_s11_f1);

        // Set frame pose
        _frame0->setWorld2FrameTransform(Eigen::Affine3d::Identity());
        Eigen::Affine3d T_w_f1 = Eigen::Affine3d::Identity();
        T_w_f1.translation()   = Eigen::Vector3d(0, 0, 1);
        _frame1->setWorld2FrameTransform(T_w_f1.inverse());

        // Set landmark 0
        Eigen::Affine3d T_w_l0 = Eigen::Affine3d::Identity();
        T_w_l0.translation()   = Eigen::Vector3d(0.5, 0, 2);
        std::vector<std::shared_ptr<AFeature>> feat_vec;
        _lmk_0 = std::shared_ptr<Point3D>(new Point3D(T_w_l0, feat_vec));
        _frame0->addLandmark(_lmk_0);
        _lmk_0->setInMap();
        _lmk_0->setInlier();

        // Set landmark 1
        Eigen::Affine3d T_w_l1 = Eigen::Affine3d::Identity();
        T_w_l1.translation()   = Eigen::Vector3d(-1, 0, 2);
        _lmk_1                 = std::shared_ptr<Point3D>(new Point3D(T_w_l1, feat_vec));
        _frame0->addLandmark(_lmk_1);
        _frame1->addLandmark(_lmk_1);
        _lmk_1->setInMap();
        _lmk_1->setInlier();

        // Set landmark 2
        Eigen::Affine3d T_w_l2 = Eigen::Affine3d::Identity();
        T_w_l2.translation()   = Eigen::Vector3d(1, 0, 2);
        _lmk_2                 = std::shared_ptr<Point3D>(new Point3D(T_w_l2, feat_vec));
        _frame0->addLandmark(_lmk_2);
        _frame1->addLandmark(_lmk_2);
        _lmk_2->setInMap();
        _lmk_2->setInlier();

        // Projections and add features for lmk 0
        std::vector<Eigen::Vector2d> projection_0_0_l;
        _sensor0l->project(_lmk_0->getPose(), _lmk_0->getModel(), Eigen::Vector3d(1, 1, 1), projection_0_0_l);

        std::vector<Eigen::Vector2d> projection_0_0_r;
        _sensor0r->project(_lmk_0->getPose(), _lmk_0->getModel(), Eigen::Vector3d(1, 1, 1), projection_0_0_r);

        _feat_0_0_l = std::shared_ptr<Point2D>(new Point2D(projection_0_0_l));
        _lmk_0->addFeature(_feat_0_0_l);
        _sensor0l->addFeature("pointxd", _feat_0_0_l);

        _feat_0_0_r = std::shared_ptr<Point2D>(new Point2D(projection_0_0_r));
        _lmk_0->addFeature(_feat_0_0_r);
        _sensor0r->addFeature("pointxd", _feat_0_0_r);

        // Projections and add features for lmk 1
        std::vector<Eigen::Vector2d> projection_0_1_l;
        _sensor0l->project(_lmk_1->getPose(), _lmk_1->getModel(), Eigen::Vector3d(1, 1, 1), projection_0_1_l);

        std::vector<Eigen::Vector2d> projection_0_1_r;
        _sensor0r->project(_lmk_1->getPose(), _lmk_1->getModel(), Eigen::Vector3d(1, 1, 1), projection_0_1_r);

        std::vector<Eigen::Vector2d> projection_1_1_l;
        _sensor1l->project(_lmk_1->getPose(), _lmk_1->getModel(), Eigen::Vector3d(1, 1, 1), projection_1_1_l);

        std::vector<Eigen::Vector2d> projection_1_1_r;
        _sensor1r->project(_lmk_1->getPose(), _lmk_1->getModel(), Eigen::Vector3d(1, 1, 1), projection_1_1_r);

        _feat_0_1_l = std::shared_ptr<Point2D>(new Point2D(projection_0_1_l));
        _lmk_1->addFeature(_feat_0_1_l);
        _sensor0l->addFeature("pointxd", _feat_0_1_l);

        _feat_0_1_r = std::shared_ptr<Point2D>(new Point2D(projection_0_1_r));
        _lmk_1->addFeature(_feat_0_1_r);
        _sensor0r->addFeature("pointxd", _feat_0_1_r);

        _feat_1_1_l = std::shared_ptr<Point2D>(new Point2D(projection_1_1_l));
        _lmk_1->addFeature(_feat_1_1_l);
        _sensor1l->addFeature("pointxd", _feat_1_1_l);

        _feat_1_1_r = std::shared_ptr<Point2D>(new Point2D(projection_1_1_r));
        _lmk_1->addFeature(_feat_1_1_r);
        _sensor1r->addFeature("pointxd", _feat_1_1_r);

        // Projections and add features for lmk 2
        std::vector<Eigen::Vector2d> projection_0_2_l;
        _sensor0l->project(_lmk_2->getPose(), _lmk_2->getModel(), Eigen::Vector3d(1, 1, 1), projection_0_2_l);

        std::vector<Eigen::Vector2d> projection_0_2_r;
        _sensor0r->project(_lmk_2->getPose(), _lmk_2->getModel(), Eigen::Vector3d(1, 1, 1), projection_0_2_r);

        std::vector<Eigen::Vector2d> projection_1_2_l;
        _sensor1l->project(_lmk_2->getPose(), _lmk_2->getModel(), Eigen::Vector3d(1, 1, 1), projection_1_2_l);

        std::vector<Eigen::Vector2d> projection_1_2_r;
        _sensor1r->project(_lmk_2->getPose(), _lmk_2->getModel(), Eigen::Vector3d(1, 1, 1), projection_1_2_r);

        _feat_0_2_l = std::shared_ptr<Point2D>(new Point2D(projection_0_2_l));
        _lmk_2->addFeature(_feat_0_2_l);
        _sensor0l->addFeature("pointxd", _feat_0_2_l);

        _feat_0_2_r = std::shared_ptr<Point2D>(new Point2D(projection_0_2_r));
        _lmk_2->addFeature(_feat_0_2_r);
        _sensor0r->addFeature("pointxd", _feat_0_2_r);

        _feat_1_2_l = std::shared_ptr<Point2D>(new Point2D(projection_1_2_l));
        _lmk_2->addFeature(_feat_1_2_l);
        _sensor1l->addFeature("pointxd", _feat_1_2_l);

        _feat_1_2_r = std::shared_ptr<Point2D>(new Point2D(projection_1_2_r));
        _lmk_2->addFeature(_feat_1_2_r);
        _sensor1r->addFeature("pointxd", _feat_1_2_r);
    }

    Eigen::Matrix3d _K;
    std::shared_ptr<Frame> _frame0;
    std::shared_ptr<ImageSensor> _sensor0l;
    std::shared_ptr<ImageSensor> _sensor0r;
    std::shared_ptr<Frame> _frame1;
    std::shared_ptr<ImageSensor> _sensor1l;
    std::shared_ptr<ImageSensor> _sensor1r;
    std::shared_ptr<Point3D> _lmk_0;
    std::shared_ptr<Point2D> _feat_0_0_l;
    std::shared_ptr<Point2D> _feat_0_0_r;
    std::shared_ptr<Point3D> _lmk_1;
    std::shared_ptr<Point2D> _feat_0_1_l;
    std::shared_ptr<Point2D> _feat_0_1_r;
    std::shared_ptr<Point2D> _feat_1_1_l;
    std::shared_ptr<Point2D> _feat_1_1_r;
    std::shared_ptr<Point3D> _lmk_2;
    std::shared_ptr<Point2D> _feat_0_2_l;
    std::shared_ptr<Point2D> _feat_0_2_r;
    std::shared_ptr<Point2D> _feat_1_2_l;
    std::shared_ptr<Point2D> _feat_1_2_r;

    std::unordered_map<std::shared_ptr<Frame>, PoseParametersBlock> _map_frame_posepar;
    std::unordered_map<std::shared_ptr<ALandmark>, PointXYZParametersBlock> _map_lmk_ptpar;
    std::unordered_map<std::shared_ptr<ALandmark>, PoseParametersBlock> _map_lmk_posepar;

    Marginalization _marg;
};

// Is the setup all right
TEST_F(MarginalizationTest, setupTest) {

    // Check number of feats
    ASSERT_EQ(_sensor0r->getFeatures()["pointxd"].size(), 3);
    ASSERT_EQ(_sensor0l->getFeatures()["pointxd"].size(), 3);

    ASSERT_EQ(_sensor1l->getFeatures()["pointxd"].size(), 2);
    ASSERT_EQ(_sensor1r->getFeatures()["pointxd"].size(), 2);

    // Check if the map is good
    ASSERT_EQ(_frame0->getInMapLandmarksNumber(), 3);
}

// Pre Marginalization test
TEST_F(MarginalizationTest, preMargTest) {

    std::shared_ptr<Marginalization> marg_last = std::shared_ptr<Marginalization>(new Marginalization());
    _marg.preMarginalize(_frame0, _frame1, marg_last);

    // Check all the variables in _marg
    ASSERT_EQ(_marg._n, 6);
    ASSERT_EQ(_marg._m, 9);
    ASSERT_EQ(_marg._frame_to_marg, _frame0);
    ASSERT_EQ(_marg._lmk_to_marg["pointxd"].at(0), _lmk_0);
    ASSERT_EQ(_marg._lmk_to_keep["pointxd"].size(), 2);
}

// We test the gradient and info mat computation as in the solver
TEST_F(MarginalizationTest, margTest) {

    std::shared_ptr<Marginalization> marg_last = std::shared_ptr<Marginalization>(new Marginalization());
    _marg.preMarginalize(_frame0, _frame1, marg_last);

    // Create pose parameters for the frame to marginalize
    _map_frame_posepar.emplace(_frame0, PoseParametersBlock(_frame0->getWorld2FrameTransform()));

    ASSERT_EQ(_marg._lmk_to_keep["pointxd"].size(), 2);
    ASSERT_EQ(_marg._lmk_to_marg["pointxd"].size(), 1);

    // Create Marginalization Blocks with landmarks to keep
    for (auto tlmk : _marg._lmk_to_keep) {
        for (auto lmk : tlmk.second) {
            _map_lmk_ptpar.emplace(lmk, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
            // For each feature on the frame
            for (auto feature : lmk->getFeatures()) {
                if (feature.lock()->getSensor()->getFrame() == _frame0) {

                    // Compute index and block vectors for reprojection factor
                    std::vector<double *> parameter_blocks;
                    std::vector<int> parameter_idx;

                    // For the frame
                    parameter_idx.push_back(_marg._map_frame_idx.at(feature.lock()->getSensor()->getFrame()));
                    parameter_blocks.push_back(_map_frame_posepar.at(feature.lock()->getSensor()->getFrame()).values());

                    // For the lmk
                    parameter_idx.push_back(_marg._map_lmk_idx.at(lmk));
                    parameter_blocks.push_back(_map_lmk_ptpar.at(lmk).values());

                    // Add the reprojection factor in the marginalization scheme
                    std::cout << feature.lock()->getPoints().size() << std::endl;
                    ceres::CostFunction *cost_fct = new BundleAdjustmentCERESAnalytic::ReprojectionErrCeres_pointxd_dx(
                        feature.lock()->getPoints().at(0), feature.lock()->getSensor(), lmk->getPose());
                    _marg.addMarginalizationBlock(
                        std::make_shared<MarginalizationBlockInfo>(cost_fct, parameter_idx, parameter_blocks));
                }
            }
        }
    }

    // Create Marginalization Blocks with landmark to marginalize
    for (auto tlmk : _marg._lmk_to_marg) {
        for (auto lmk : tlmk.second) {
            _map_lmk_ptpar.emplace(lmk, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
            // For each feature on the frame
            for (auto feature : lmk->getFeatures()) {
                if (feature.lock()->getSensor()->getFrame() == _frame0) {

                    // Compute index and block vectors for reprojection factor
                    std::vector<double *> parameter_blocks;
                    std::vector<int> parameter_idx;

                    // For the frame
                    parameter_idx.push_back(_marg._map_frame_idx.at(feature.lock()->getSensor()->getFrame()));
                    parameter_blocks.push_back(_map_frame_posepar.at(feature.lock()->getSensor()->getFrame()).values());

                    // For the lmk
                    parameter_idx.push_back(_marg._map_lmk_idx.at(lmk));
                    parameter_blocks.push_back(_map_lmk_ptpar.at(lmk).values());

                    // Add the reprojection factor in the marginalization scheme
                    ceres::CostFunction *cost_fct = new BundleAdjustmentCERESAnalytic::ReprojectionErrCeres_pointxd_dx(
                        feature.lock()->getPoints().at(0), feature.lock()->getSensor(), lmk->getPose());
                    _marg.addMarginalizationBlock(
                        std::make_shared<MarginalizationBlockInfo>(cost_fct, parameter_idx, parameter_blocks));
                }
            }
        }
    }

    // Check the indices of the variables
    ASSERT_EQ(_marg._map_frame_idx.at(_frame0), 0);
    ASSERT_EQ(_marg._map_lmk_idx.at(_lmk_0), 6);
    ASSERT_EQ(_marg._map_lmk_idx.at(_lmk_1), 9);
    ASSERT_EQ(_marg._map_lmk_idx.at(_lmk_2), 12);

    _marg.computeSchurComplement();

    // Are indices updated ?
    ASSERT_EQ(_marg._map_lmk_idx.at(_lmk_1), 0);
    ASSERT_EQ(_marg._map_lmk_idx.at(_lmk_2), 3);

    // Info Mat check
    ASSERT_EQ(_marg._Ak.size(), 36);
    ASSERT_NEAR((_marg._Ak - _marg._Ak.transpose()).norm(), 0, 1e-8);

    // We test off diag
    double off = _marg.computeOffDiag(_lmk_1, _lmk_2);
    ASSERT_GT(off, 0);
}

// Test the case where the frame to marginalize is not corelated
TEST_F(MarginalizationTest, margFailTest) {

    std::shared_ptr<Frame> frame2   = std::shared_ptr<Frame>(new Frame());
    std::shared_ptr<Camera> sensor2 = std::shared_ptr<Camera>(new Camera(cv::Mat::zeros(400, 400, CV_16F), _K));

    std::vector<std::shared_ptr<ImageSensor>> sensors_frame2;
    sensors_frame2.push_back(sensor2);
    frame2->init(sensors_frame2, 2.0);
    sensor2->setFrame2SensorTransform(Eigen::Affine3d::Identity());

    std::shared_ptr<Marginalization> marg_last = std::shared_ptr<Marginalization>(new Marginalization());
    _marg.preMarginalize(frame2, frame2, marg_last);

    ASSERT_EQ(_marg.computeSchurComplement(), false);
}

} // namespace isae