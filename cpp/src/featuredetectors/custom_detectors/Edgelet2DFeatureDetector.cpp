#include "isaeslam/featuredetectors/custom_detectors/Edgelet2DFeatureDetector.h"
#include <opencv2/core/eigen.hpp>
#include <random>
#include <Eigen/Eigenvalues>

namespace isae {



    void EdgeletFeatureDetector::init()
    {
        //_descriptor = cv::xfeatures2d::BEBLID::create(1, cv::xfeatures2d::BEBLID::SIZE_256_BITS);
        //_descriptor = cv::xfeatures2d::BriefDescriptorExtractor::create(32, false);
        _descriptor = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
        _defaultNorm = cv::NORM_HAMMING2;

        _cell_size = 7;
        _half_cell_size = 3;
    }

    double EdgeletFeatureDetector::getDist(const cv::Mat &desc1, const cv::Mat &desc2) const
    {
        //std::cout << "dist = " << cv::norm(desc1, desc2, this->getDefaultNorm()) << std::endl;
        return cv::norm(desc1, desc2, this->getDefaultNorm());
    }



    void EdgeletFeatureDetector::computeLineDescriptor(const cv::Mat &img, const std::vector<cv::KeyPoint> &kps, const std::vector<cv::Point2f> &orientations, cv::Mat &descriptors){
        // convert keypoints to keylines to be described
        std::vector<cv::line_descriptor::KeyLine> keylines;
        for(uint i=0; i < kps.size(); ++i){
            cv::line_descriptor::KeyLine kl;
            kl.pt = kps.at(i).pt;
            kl.octave = kps.at(i).octave;
            kl.angle = kps.at(i).angle;
            kl.startPointX = kps.at(i).pt.x - orientations.at(i).x;
            kl.startPointY = kps.at(i).pt.y - orientations.at(i).y;
            kl.endPointX = kps.at(i).pt.x + orientations.at(i).x;
            kl.endPointY = kps.at(i).pt.y + orientations.at(i).y;
            keylines.push_back(kl);
        }
        _descriptor->compute(img, keylines, descriptors);
    }

    void EdgeletFeatureDetector::customDetectAndCompute(const cv::Mat &img, const cv::Mat &mask, std::vector<std::shared_ptr<AFeature> > &features)
    {

        // Detect edgelets
        std::vector<double> scores;
        std::vector<cv::KeyPoint> kps;
        std::vector<cv::Point2f> orientations;
        detect(img, kps, orientations, scores);

        // Compute descriptors as "lines" in the edgelet direction
        cv::Mat descriptors;
        computeLineDescriptor(img, kps, orientations, descriptors);

        // create edgelet features and add direction as 2nd point
        std::vector<std::shared_ptr<AFeature> > localfeatures;
        KeypointToFeature(kps, descriptors, localfeatures, "edgeletxd");
        for(size_t i=0; i < localfeatures.size(); ++i){
            std::vector<Eigen::Vector2d> pts = localfeatures.at(i)->getPoints();
            pts.push_back(pts.at(0)+Eigen::Vector2d(orientations.at(i).x,orientations.at(i).y));
            localfeatures.at(i)->setPoints(pts);
        }

        // order feature score to select only the best detected features
        std::vector<uint> vx;
        vx.resize(scores.size());
        for( uint i= 0; i < scores.size(); ++i ) vx.at(i) = i;

        // randomly shuffle scores vector as edgelets are detected in order
        std::random_shuffle(scores.begin(), scores.end());
        partial_sort( vx.begin(), vx.begin()+std::min(_n_per_cell,(int)vx.size()), vx.end(), Comp(scores) );

        // Keep only the best features (_n if available)
        for(int i=0; i < std::min(_n_per_cell,(int)vx.size()); ++i)
            features.push_back(localfeatures.at(vx[i]));

    }


    void EdgeletFeatureDetector::computeDescriptor(const cv::Mat &img, std::vector<std::shared_ptr<AFeature> > &features){

        // convert features to keypoints
        std::vector<double> scores;
        std::vector<cv::KeyPoint> kps;
        cv::Mat descriptors;
        FeatureToKeypoint(features, kps, descriptors);


        if(kps.size() == 0)
            return;

        // Process orientation and score for current undescribed features
        std::vector<cv::Point2f> orientations;
        std::vector<double> gradient_norms;
        cv::Mat imgb;
        cv::Mat gradx = cv::Mat::zeros( img.rows, img.cols, CV_32F);
        cv::Mat grady = cv::Mat::zeros(img.rows, img.cols, CV_32F);
        cv::GaussianBlur( img, imgb, cv::Size( 3, 3 ), 0, 0 );
        cv::Scharr(imgb, gradx, CV_32F, 1, 0, 1/32.0);
        cv::Scharr(imgb, grady, CV_32F, 0, 1, 1/32.0);
        cv::Mat canny;
        cv::Canny(imgb, canny, 30, 50);





        for(uint i=0; i < kps.size(); ++i ){
            uint x_start = (uint)(kps.at(i).pt.x - _half_cell_size);
            uint y_start = (uint)(kps.at(i).pt.y - _half_cell_size);
            cv::Rect roi(x_start, y_start, _cell_size, _cell_size);

            double gnorm;
            cv::Point2f orient;

            /// TODO check why there is some features with coordinates outside the image !!!
//            std::cout << "fea = " << features.at(i)->getPoints().at(0)(0) << " " << features.at(i)->getPoints().at(0)(1) << std::endl;
            if(roi.x < 0 || roi.y <0 || roi.x+roi.width >= canny.cols || roi.y+roi.height >= canny.rows){
                gnorm = 0;
                orient = cv::Point2f(0,0);
            }
            else
                orient = getOrientation(canny(roi), gradx(roi), grady(roi), gnorm);

            gradient_norms.push_back(gnorm);
            orientations.push_back(orient);
        }

        // Compute descriptors as "lines" in the edgelet direction
        computeLineDescriptor(img, kps, orientations, descriptors);

        // convert back to feature
        std::vector<std::shared_ptr<AFeature> > localfeatures;
        KeypointToFeature(kps, descriptors, localfeatures, "edgeletxd");
        for(uint i=0; i < kps.size(); ++i ){
            std::vector<Eigen::Vector2d> pts = localfeatures.at(i)->getPoints();
            pts.push_back(pts.at(0)+Eigen::Vector2d(orientations.at(i).x, orientations.at(i).y));
            localfeatures.at(i)->setPoints(pts);
        }

        // remove not described features
        deleteUndescribedFeatures(localfeatures);
        features = localfeatures;
    }


    void EdgeletFeatureDetector::detect(const cv::Mat &img, std::vector<cv::KeyPoint> &kps, std::vector<cv::Point2f> &orientations, std::vector<double> &scores) {
        cv::Mat gradx = cv::Mat::zeros(img.rows, img.cols, CV_32F);
        cv::Mat grady = cv::Mat::zeros(img.rows, img.cols, CV_32F);
        cv::Mat mag = cv::Mat::zeros(img.rows, img.cols, CV_32F);

        cv::Mat imgb;
        cv::GaussianBlur(img, imgb, cv::Size(3, 3), 0, 0);
        cv::Scharr(imgb, gradx, CV_32F, 1, 0, 1 / 32.0);
        cv::Scharr(imgb, grady, CV_32F, 0, 1, 1 / 32.0);
        cv::magnitude(gradx, grady, mag);

        cv::Mat canny;
        cv::Canny(imgb, canny, 30, 50);


        // For all cell in image
        for (uint u = _half_cell_size; u < img.rows - _half_cell_size; u += _cell_size)
            for (uint v = _half_cell_size; v < img.cols - _half_cell_size; v += _cell_size) {
                uint x_start = u - _half_cell_size;
                uint y_start = v - _half_cell_size;
                cv::Rect roi(y_start, x_start, _cell_size, _cell_size);

                float max_grad_2 = 0;
                float max_grad = 0;
                int max_grad_x = 0;
                int max_grad_y = 0;
                float gx = 0;
                float gy = 0;

                // For each pixel of the cell
                for (uint i = 0; i < _cell_size; i++)
                    for (uint j = 0; j < _cell_size; j++) {
                        if (canny(roi).ptr<uchar>(i)[j] == 0)
                            continue;

                        float temp = mag.ptr<float>(x_start + i)[y_start + j];
                        if (temp > max_grad) {
                            max_grad_x = x_start + i;
                            max_grad_y = y_start + j;
                            max_grad = temp;
                            gx = gradx.ptr<float>(max_grad_x)[max_grad_y];
                            gy = grady.ptr<float>(max_grad_x)[max_grad_y];
                        } else if (temp > max_grad_2) {
                            max_grad_2 = temp;
                        }
                    }

                // Threshold magnitude
                int edge_threshold = _edge_threshold;
                Eigen::Vector2d g(gx, gy);
                g = g.normalized();
                if (max_grad > edge_threshold  && max_grad > 0.75*max_grad_2) {
                    kps.push_back(cv::KeyPoint(cv::Point2f(max_grad_y, max_grad_x), 1));
//                    orientations.push_back(cv::Point2f(g.x(), g.y()));
//                    scores.push_back(max_grad);
                    double score;
                    cv::Point2f orient = getOrientation(canny(roi), gradx(roi), grady(roi), score);
                    orientations.push_back(orient);
                    scores.push_back(score);
                }
            }

    }
    cv::Point2f EdgeletFeatureDetector::getOrientation(const cv::Mat &img, const cv::Mat &gradx, const cv::Mat &grady, double &score){

        // Find non zero pixel coord
        std::vector<cv::Point> nonZeroCoordinates;
        cv::findNonZero(img, nonZeroCoordinates);

        if(nonZeroCoordinates.size() <= _cell_size) {
            score = 0;
            return cv::Point2f(0,0);
        }


        Eigen::VectorXd Vx(nonZeroCoordinates.size());
        Eigen::VectorXd Vy(nonZeroCoordinates.size());

        for (uint i = 0; i < nonZeroCoordinates.size(); i++ ) {
            Vx(i) = nonZeroCoordinates.at(i).x;
            Vy(i) = nonZeroCoordinates.at(i).y;
        }

        double mx = Vx.array().mean();
        double my = Vy.array().mean();

        double varx = (Vx.array() - mx).square().sum() / (Vx.array().size());
        double vary = (Vy.array() - my).square().sum() / (Vy.array().size());
        double varxy = ((Vx.array() - mx) * (Vy.array() - mx)).sum() / (Vy.array().size() - 1);

        Eigen::Matrix2d C;
        C << varx, varxy, varxy, vary;
        Eigen::Vector2cd eivals = C.eigenvalues();
        double mgx = 0;
        double mgy = 0;
        for (uint i = 0; i < nonZeroCoordinates.size(); i++) {
            mgx += gradx.at<float>(nonZeroCoordinates.at(i).y, nonZeroCoordinates.at(i).x);
            mgy += grady.at<float>(nonZeroCoordinates.at(i).y, nonZeroCoordinates.at(i).x);
        }

        score = 1 - fmin(fabs(eivals(0).real()), fabs(eivals(1).real()));
        // return normalized orientation
        return cv::Point2f(-mgy, mgx)/sqrt(mgx*mgx+mgy*mgy);
    }

}// namespace isae


