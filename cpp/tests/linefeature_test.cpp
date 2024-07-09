#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "isaeslam/data/features/Line2D.h"
#include "isaeslam/data/landmarks/Line3D.h"
#include "isaeslam/data/frame.h"
#include "isaeslam/data/sensors/Camera.h"
#include "isaeslam/landmarkinitializer/Line3DlandmarkInitializer.h"
#include "isaeslam/featuredetectors/custom_detectors/Line2DFeatureDetector.h"
#include "isaeslam/featurematchers/Line2DFeatureMatcher.h"
#include "isaeslam/featurematchers/Line2DFeatureTracker.h"


namespace isae {

class LineFeatureTest : public testing::Test {
    public:
        void SetUp() override {

            // Set Camera Config and features detection
            Eigen::Matrix3d K0, K1;
            Eigen::Vector4f D0, D1;
            cv::Mat imgRaw0, imgRaw1;
            imgRaw0 = cv::imread("../tests/cam0.png");
            imgRaw1 = cv::imread("../tests/cam1.png");

            T_BS0.linear() << 0.0148655429818, -0.999880929698, 0.00414029679422,
                    0.999557249008, 0.0149672133247, 0.025715529948,
                    -0.0257744366974, 0.00375618835797, 0.999660727178;

            T_BS0.translation() << -0.0216401454975, -0.064676986768, 0.00981073058949;

            K0 << 458.654, 0., 367.215,
                    0., 457.296, 248.375,
                    0. ,0., 1.;

            T_BS1.linear() << 0.0125552670891, -0.999755099723, 0.0182237714554,
                    0.999598781151, 0.0130119051815, 0.0251588363115,
                    -0.0253898008918, 0.0179005838253, 0.999517347078;

            T_BS1.translation() <<-0.0198435579556, 0.0453689425024, 0.00786212447038;

            K1 << 457.587, 0., 379.999,
                    0., 456.134, 255.238,
                    0. ,0., 1.;

            D0 << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05;
            D1 << -0.28368365,  0.07451284, -0.00010473, -3.55590700e-05;


            Eigen::MatrixXd p0S(2,4), p0E(2,4), p1S(2,4), p1E(2,4);
            p0S << 147.5507,  610.5421,  616.3151,  155.4294, 471.0146,  472.1311,   16.9374,    3.8912;
            p0E << 610.5421,  616.3151,  155.4294 , 147.5507,  472.1311,   16.9374,    3.8912,  471.0146;
            p1S << 107.8754 , 574.7646 , 576.9331,  118.7161,  486.1799 , 486.3055 ,  31.7858 ,  20.0815;
            p1E << 574.7646 , 576.9331 , 118.7161 , 107.8754,  486.3055 ,  31.7858 ,  20.0815 , 486.1799;


            // 4 lines : ps to pe
            for(uint i=0; i < 4; ++i) {
                _ps0.push_back(p0S.block<2, 1>(0, i));
                _pe0.push_back(p0E.block<2, 1>(0, i));
                _ps1.push_back(p1S.block<2, 1>(0, i));
                _pe1.push_back(p1E.block<2, 1>(0, i));
            }

            _frame0 = std::shared_ptr<Frame>(new Frame());
            _frame0->setWorld2FrameTransform(Eigen::Affine3d::Identity());
            _frame0->setKeyFrame();

            _cam0 = std::make_shared<Camera>(imgRaw0, K0);
            _cam0->setFrame2SensorTransform(T_BS0.inverse());
            _cam1 = std::make_shared<Camera>(imgRaw1, K1);
            _cam1->setFrame2SensorTransform(T_BS1.inverse());
            std::vector<std::shared_ptr<ImageSensor>> sensors;
            sensors.push_back(_cam0);
            sensors.push_back(_cam1);

            // Set Frame
            _frame0->init(sensors, 0.);

        }

        Eigen::Affine3d T_BS0, T_BS1;
        std::shared_ptr<Frame> _frame0;
        std::shared_ptr<Camera> _cam0;
        std::shared_ptr<Camera> _cam1;
        std::vector<Eigen::Vector2d> _ps0, _pe0, _ps1, _pe1;
        std::vector<Eigen::Vector3d> ray0s, ray0e, rays1, raye1;
    };


TEST_F(LineFeatureTest, LineFeatureRay) {

    // Check given features points (start and end point of lines) :
    // - the plane C1 C2 ray1 ray2 Normal processing
    //
    // Observed 3D points :
    //    P1s = [-0.5  -0.5  1  1
    //          -0.5  0.5  1  1
    //          0.5  0.5 1 1
    //            0.5  -0.5 1 1];
    //    P2s = [-0.5  0.5  1  1
    //           0.5  0.5 1 1
    //            0.5  -0.5 1 1
    //           -0.5  -0.5  1  1];

    Eigen::MatrixXd N0(3,4), N1(3,4);
    Eigen::MatrixXd rs0s(3,4), re0s(3,4), rs1s(3,4), re1s(3,4);
    N0 <<   0.9004,   -0.0000,   -0.8847,    0.0000,
            0,   -0.8687,         0,    0.9154,
            0.4350,    0.4954,    0.4661,    0.4025;

    N1 <<   0.9001,         0,   -0.8858,   -0.0000,
            0 ,  -0.9091,         0,    0.8763,
            0.4356,    0.4166,    0.4641,    0.4817;

    rs0s <<   -0.4045,   -0.3870,    0.4161  ,  0.4344,
            -0.3681  ,  0.4568 ,   0.4505 ,  -0.3625,
            0.8372  ,  0.8010   , 0.7899  ,  0.8246;

    re0s <<-0.3870  ,  0.4161 ,   0.4344   ,-0.4045,
            0.4568  ,  0.4505   ,-0.3625  , -0.3681,
            0.8010  ,  0.7899  ,  0.8246  ,  0.8372;

    rs1s <<      -0.3904  , -0.4027  ,  0.4300 ,   0.4173,
            -0.4435  ,  0.3813 ,   0.3761 ,  -0.4378,
            0.8068  ,  0.8321 ,   0.8207  ,  0.7964;

    re1s << -0.4027  ,  0.4300  ,  0.4173 ,  -0.3904,
            0.3813 ,   0.3761  , -0.4378  , -0.4435,
            0.8321  ,  0.8207   , 0.7964  ,  0.8068;

    // Check obtained rays
    for(uint i=0; i < 4; ++i){

        Eigen::Vector3d rs0 = _frame0->getSensors().at(0)->getRay(_ps0.at(i));
        ASSERT_NEAR((rs0- rs0s.block<3,1>(0,i)).norm(), 0., 1e-3);

        Eigen::Vector3d re0 = _frame0->getSensors().at(0)->getRay(_pe0.at(i));
        ASSERT_NEAR((re0- re0s.block<3,1>(0,i)).norm(), 0., 1e-3);

        Eigen::Vector3d rs1 = _frame0->getSensors().at(1)->getRay(_ps1.at(i));
        ASSERT_NEAR((rs1- rs1s.block<3,1>(0,i)).norm(), 0., 1e-3);

        Eigen::Vector3d re1 = _frame0->getSensors().at(1)->getRay(_pe1.at(i));
        ASSERT_NEAR((re1- re1s.block<3,1>(0,i)).norm(), 0., 1e-3);

        Eigen::Vector3d normal0 = re0.cross(rs0).normalized();
        Eigen::Vector3d normal1 = re1.cross(rs1).normalized();

        ASSERT_NEAR((normal0 - N0.block<3,1>(0,i)).norm(), 0., 1e-3);
        ASSERT_NEAR((normal1 - N1.block<3,1>(0,i)).norm(), 0., 1e-3);
    }

}



TEST_F(LineFeatureTest, LineFeatureTriangulation) {
    // Check given features points (start and end point of lines) :
    // - the 3D ray in world
    Eigen::Vector2d start0 = _ps0.at(0);
    Eigen::Vector2d end0 = _pe0.at(0);
    Eigen::Vector2d start1 = _ps1.at(0);;
    Eigen::Vector2d end1 = _pe1.at(0);

    std::cout << "start0 = " << start0.transpose() << std::endl;
    std::cout << "end0 = " << end0.transpose() << std::endl;
    std::cout << "start1 = " << start1.transpose() << std::endl;
    std::cout << "end1 = " << end1.transpose() << std::endl;

    std::vector<std::shared_ptr<AFeature>> features;
    std::vector<Eigen::Vector2d> points1, points2;
    points1.push_back( start0 );
    points1.push_back( end0 );
    std::shared_ptr<AFeature> f1 = std::make_shared<Line2D>(points1, cv::Mat());
    f1->setSensor(_cam0);
    features.push_back(f1);

    points2.push_back( start1 );
    points2.push_back( end1 );
    std::shared_ptr<AFeature> f2 = std::make_shared<Line2D>(points2, cv::Mat());
    f2->setSensor(_cam1);
    features.push_back(f2);

    _frame0->getSensors().at(0)->addFeature("linexd", f1);
    _frame0->getSensors().at(1)->addFeature("linexd", f2);

    Line3DLandmarkInitializer ldmkInit(2);
    ldmkInit.initFromFeatures(features);
    typed_vec_landmarks ldmks = _frame0->getLandmarks();

    std::cout << ldmks.size() << std::endl;


    Eigen::Affine3d T = ldmks["linexd"].at(0)->getPose();
    Eigen::Vector3d t = T.translation();
    Eigen::Matrix3d R = T.rotation();
    Eigen::Vector3d v = isae::geometry::Rotation2directionVector(R, Eigen::Vector3d(1, 0, 0));
    std::vector<Eigen::Vector3d> ldmk_model = ldmks["linexd"].at(0)->getModelPoints();
    Eigen::Vector3d scale = ldmks["linexd"].at(0)->getScale();

    std::vector<Eigen::Vector3d> start_end;
    for (const auto &p3d_model : ldmk_model) {
    // conversion to the world coordinate system
        Eigen::Vector3d t_w_lmk = T * p3d_model.cwiseProduct(scale);
        start_end.push_back(t_w_lmk);
    }



    std::cout << "Results triangulation : " << std::endl
              << "center = " << t <<  std::endl
              << "direction = " << v <<  std::endl
              << " rot = " << R <<  std::endl
              << " start = " << start_end.at(0).transpose() <<  std::endl
              << " end = " << start_end.at(1).transpose()  <<  std::endl;

    ASSERT_NEAR( (t - Eigen::Vector3d(-0.5,0,1)).norm() , 0, 1e-3);
    ASSERT_NEAR( (v.normalized() - Eigen::Vector3d(0,1,0)).norm() , 0, 1e-3);

    ASSERT_TRUE (((start_end.at(0)-Eigen::Vector3d(-0.5, 0.5, 1)).norm() < 1e-3) || ((start_end.at(0)-Eigen::Vector3d(-0.5, -0.5, 1)).norm() < 1e-3));
    ASSERT_TRUE (((start_end.at(1)-Eigen::Vector3d(-0.5, 0.5, 1)).norm() < 1e-3) || ((start_end.at(1)-Eigen::Vector3d(-0.5, -0.5, 1)).norm() < 1e-3));
}



TEST_F(LineFeatureTest, LineFeatureDetection) {

    cv::Mat I = cv::imread("../tests/rotated.png");
    Line2DFeatureDetector detector(1, 100, 25.);

    std::vector<std::shared_ptr<AFeature>> features;
    detector.customDetectAndCompute(I, cv::Mat(), features);

    cv::RNG rng(12345);
    for(auto &f : features){
        cv::Scalar color = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));

        Eigen::Vector2d pt2d  = f->getPoints().at(0);
        Eigen::Vector2d pt2d2 = f->getPoints().at(1);
        cv::line(I, cv::Point(pt2d.x(), pt2d.y()), cv::Point(pt2d2.x(), pt2d2.y()), color, 2);
    }

    cv::imshow("Detections", I);
    //cv::waitKey(0);

    // Only lines > 100 pixels (hardcoded in the detector... to be changed !)
    ASSERT_EQ(features.size() ,4);
}


TEST_F(LineFeatureTest, LineFeatureMatching) {

    cv::Mat I1 = cv::imread("../tests/cam0.png");
    cv::Mat I2 = cv::imread("../tests/cam1.png");

    cv::Mat I1t = cv::imread("../tests/cam0.png");
    cv::Mat I2t = cv::imread("../tests/cam1.png");

//    cv::Mat I1 = cv::imread("./benchmark/textureless_corridor/1.png");
//    cv::Mat I2 = cv::imread("./benchmark/textureless_corridor/2.png");

    Line2DFeatureDetector detector(50, 10, 50.);
    std::vector<std::shared_ptr<AFeature>> features1, features2, features_init;
    detector.customDetectAndCompute(I1, cv::Mat(), features1);
    detector.customDetectAndCompute(I2, cv::Mat(), features2);


    _frame0->getSensors().at(0)->addFeatures("linexd", features1);
    _frame0->getSensors().at(1)->addFeatures("linexd", features2);

    Line2DFeatureMatcher matcher(std::make_shared<Line2DFeatureDetector>(detector));
    Line2DFeatureTracker tracker(std::make_shared<Line2DFeatureDetector>(detector));
    vec_match matches, tracks;
    vec_match matches_with_ldmks, tracks_with_ldmks;
    matcher.match(features1, features2, features1, matches, matches_with_ldmks, 1000, 1000);
    tracker.track(_frame0->getSensors().at(0), 
                  _frame0->getSensors().at(1), 
                   features1, 
                   features1, 
                   tracks, 
                   tracks_with_ldmks, 
                   1000, 
                   1000,
                   3,
                   0.5, // max_err
                   true);

    cv::RNG rng(12345);
    for(auto &f : features1){
        cv::Scalar color = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));

        Eigen::Vector2d pt2d  = f->getPoints().at(0);
        Eigen::Vector2d pt2d2 = f->getPoints().at(1);
        cv::line(I1, cv::Point(pt2d.x(), pt2d.y()), cv::Point(pt2d2.x(), pt2d2.y()), color, 2);
    }

    for(auto &f : features2){
        cv::Scalar color = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));

        Eigen::Vector2d pt2d  = f->getPoints().at(0);
        Eigen::Vector2d pt2d2 = f->getPoints().at(1);
        cv::line(I2, cv::Point(pt2d.x(), pt2d.y()), cv::Point(pt2d2.x(), pt2d2.y()), color, 2);
    }


    for(auto &m : matches){
        cv::Scalar color = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        Eigen::Vector2d pt2d  = m.first->getPoints().at(0);
        Eigen::Vector2d pt2d2 = m.first->getPoints().at(1);
        cv::line(I1, cv::Point(pt2d.x(), pt2d.y()), cv::Point(pt2d2.x(), pt2d2.y()), color, 2);

        pt2d  = m.second->getPoints().at(0);
        pt2d2 = m.second->getPoints().at(1);
        cv::line(I2, cv::Point(pt2d.x(), pt2d.y()), cv::Point(pt2d2.x(), pt2d2.y()), color, 2);
    }

    for(auto &m : tracks){
        cv::Scalar color = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        Eigen::Vector2d pt2d  = m.first->getPoints().at(0);
        Eigen::Vector2d pt2d2 = m.first->getPoints().at(1);
        cv::line(I1t, cv::Point(pt2d.x(), pt2d.y()), cv::Point(pt2d2.x(), pt2d2.y()), color, 2);
        std::cout << pt2d.transpose() << "->" << pt2d2.transpose() << " <==> ";
        pt2d  = m.second->getPoints().at(0);
        pt2d2 = m.second->getPoints().at(1);
        cv::line(I2t, cv::Point(pt2d.x(), pt2d.y()), cv::Point(pt2d2.x(), pt2d2.y()), color, 2);
        std::cout << pt2d.transpose() << "->" << pt2d2.transpose() << std::endl;
        
    }

    // cv::imshow("I1", I1);
    // cv::imshow("I2", I2);
    // cv::imshow("I1t", I1t);
    // cv::imshow("I2t", I2t);    
    // cv::waitKey(0);

    Line3DLandmarkInitializer ldmkInit(2);
    ldmkInit.initFromMatches(matches);

    //typed_vec_landmarks ldmks = ldmkInit.getInitializedLandmarks();
    typed_vec_landmarks ldmks = _frame0->getLandmarks();
    std::cout << ldmks["linexd"].size() << std::endl;

    std::cout << "Matches with the same colors on I1 and I2" << std::endl;
    std::cout << features1.size() << " " << features2.size() << " " << matches_with_ldmks.size() << " " << matches.size() << std::endl;

    std::cout << "Tracks with the same colors on I1t and I2t" << std::endl;
    std::cout << features1.size() << " " << features2.size() << " " << tracks_with_ldmks.size() << " " << tracks.size() << std::endl;

    ASSERT_EQ(matches_with_ldmks.size() ,0);
    ASSERT_TRUE(matches.size() > 5); 
    ASSERT_TRUE(ldmks["linexd"].size() > 4);   
}


} // namespace isae