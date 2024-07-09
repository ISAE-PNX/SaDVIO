#include <gtest/gtest.h>
#include <random>

#include "isaeslam/data/features/Point2D.h"
#include "isaeslam/data/frame.h"
#include "isaeslam/data/maps/localmap.h"
#include "isaeslam/data/sensors/IMU.h"
#include "isaeslam/optimizers/AngularAdjustmentCERESAnalytic.h"
#include "isaeslam/featuredetectors/opencv_detectors/cvFASTFeatureDetector.h"
#include "isaeslam/featurematchers/Point2DFeatureMatcher.h"
#include "isaeslam/data/landmarks/Point3D.h"

namespace isae {

class AssociationTest : public testing::Test {
  public:
    void SetUp() override {

        std::random_device rd;                          // Only used once to initialise (seed) engine
        std::mt19937 rng(rd());                         // Random-number engine used (Mersenne-Twister in this case)
        std::uniform_int_distribution<int> uni(0, 256); // Guaranteed unbiased

        // Intrinsic
        _K       = Eigen::Matrix3d::Identity();
        _K(0, 0) = 100;
        _K(1, 1) = 100;
        _K(0, 2) = 400;
        _K(1, 2) = 400;

        // Set Frame and Sensor
        _frame0  = std::shared_ptr<Frame>(new Frame());
        _frame1  = std::shared_ptr<Frame>(new Frame());
        _sensor0 = std::shared_ptr<Camera>(new Camera(cv::Mat::zeros(800, 800, CV_16F), _K));
        _sensor1 = std::shared_ptr<Camera>(new Camera(cv::Mat::zeros(800, 800, CV_16F), _K));
        _sensor0->setFrame2SensorTransform(Eigen::Affine3d::Identity());
        _sensor1->setFrame2SensorTransform(Eigen::Affine3d::Identity());
        std::vector<std::shared_ptr<ImageSensor>> sensors_frame;
        sensors_frame.push_back(_sensor0);
        _frame0->init(sensors_frame, 0);
        std::vector<std::shared_ptr<ImageSensor>> sensors_frame1;
        sensors_frame1.push_back(_sensor1);
        _frame1->init(sensors_frame1, 1);
        
        srand((unsigned int)time(0));

        // Generates random 3D points in a 10m * 10m box
        for (uint i = 0; i < 1000; i++) {

            // Random point
            Eigen::Vector3d rand_pt = 10 * Eigen::Vector3d::Random();
            Eigen::Affine3d rand_T  = Eigen::Affine3d::Identity();
            rand_T.translation()    = rand_pt;

            // Random descriptor
            cv::Mat rand_desc = cv::Mat::zeros(cv::Size(32, 1), CV_8U);
            cv::randu(rand_desc, 0, 255);

            // Project point
            std::vector<Eigen::Vector2d> p2ds;
            Eigen::Vector2d p2d;
            if (!_sensor0->project(
                    rand_T, _frame0->getWorld2FrameTransform(), Eigen::Matrix2d::Identity(), p2d, nullptr, nullptr))
                continue;
            p2ds.push_back(p2d);

            // Create feature
            std::shared_ptr<Point2D> feat0 = std::make_shared<Point2D>(p2ds, rand_desc.clone());
            _sensor0->addFeature("pointxd", feat0);
            std::shared_ptr<Point2D> feat1 = std::make_shared<Point2D>(p2ds, rand_desc.clone());
            _sensor1->addFeature("pointxd", feat1);

            // Create landmark
            vec_shared<AFeature> feats;
            feats.push_back(feat0);
            std::shared_ptr<Point3D> lmk = std::shared_ptr<Point3D>(new Point3D());
            lmk->init(rand_T, feats); // NOTE : init is necessary to use make shared from this
            lmk->setInMap();
            _lmks.push_back(lmk);
        }
    }

    std::shared_ptr<Frame> _frame0, _frame1;
    std::shared_ptr<ImageSensor> _sensor0, _sensor1;

    Eigen::Matrix3d _K;
    std::vector<Eigen::Vector3d> _3d_points;
    vec_shared<ALandmark> _lmks;
    std::unordered_map<std::shared_ptr<ALandmark>, cv::Mat> _map_lmk_desc;
};

TEST_F(AssociationTest, matchLmk) {
    std::shared_ptr<cvFASTFeatureDetector> detector = std::make_shared<cvFASTFeatureDetector>(100, 30, 5);
    Point2DFeatureMatcher matcher(detector);

    int n_matched = matcher.ldmk_match(_sensor1, _lmks);
    ASSERT_EQ(n_matched, _lmks.size());
}

} // namespace isae