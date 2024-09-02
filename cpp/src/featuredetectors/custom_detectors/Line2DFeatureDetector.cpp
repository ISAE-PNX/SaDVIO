#include "isaeslam/featuredetectors/custom_detectors/Line2DFeatureDetector.h"
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/core/eigen.hpp>
#include <random>
#include <Eigen/Eigenvalues>


#include <opencv2/highgui.hpp>

namespace isae {



    void Line2DFeatureDetector::init()
    {
        _detector = cv::line_descriptor::LSDDetector::createLSDDetector();
        _descriptor = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
        _defaultNorm = cv::NORM_HAMMING2;

        // TODO how to better configure line detector 
        _params.minLineLen = 100;
        _elsed = std::make_shared<upm::ELSED>(_params);
    }


    double Line2DFeatureDetector::getDist(const cv::Mat &desc1, const cv::Mat &desc2) const
    {
        //std::cout << "dist = " << cv::norm(desc1, desc2, this->getDefaultNorm()) << std::endl;
        return cv::norm(desc1, desc2, this->getDefaultNorm());
    }


    bool Line2DFeatureDetector::getFeaturesInBox(int x,
                                            int y,
                                            int w,
                                            int h,
                                            std::vector<int> &indexes,
                                            std::vector<std::shared_ptr<AFeature>> features) const {
        for (size_t i = 0; i < features.size(); i++) {
            const std::shared_ptr<AFeature> &f = features[i];
            Eigen::Vector2d middlePt = (f->getPoints().at(0) + f->getPoints().at(1))/2;
            if (middlePt(0) < x || middlePt(0) > x + w || middlePt(1) < y || middlePt(1) > y + h)
                continue;
            indexes.push_back(i);
        }
        if (indexes.empty())
            return false;
        else
            return true;
    }


    void Line2DFeatureDetector::customDetectAndCompute(const cv::Mat &img, const cv::Mat &mask, std::vector<std::shared_ptr<AFeature> > &features)
    {
        /* ELSED detector + convert segment into keyline*/
        upm::Segments segs = _elsed->detect(img);
        int line_id = 0;

        std::vector<cv::line_descriptor::KeyLine> keyLine;
        for(unsigned int i = 0; i < segs.size(); i++)
        {
            cv::line_descriptor::KeyLine kl = MakeKeyLine(cv::Point2f(segs[i][0], segs[i][1]), cv::Point2f(segs[i][2], segs[i][3]), img.cols);
            kl.class_id = line_id;
            line_id++;
            keyLine.push_back(kl);
        }

        /* compute descriptors */
        cv::Mat descriptors;
        if(keyLine.size() > 0)
            _descriptor->compute(img, keyLine, descriptors);

        /* delete undesired KeyLines, according to input mask and filtering by lenth of a line*/
        /// TODO ???

        /* Convert line to local features for SLAM */
        std::vector<std::shared_ptr<AFeature> > localfeatures;
        KeyLineToFeature(keyLine, descriptors, localfeatures, "linexd");

        std::vector<double> scores;
        for(uint i=0; i < localfeatures.size(); ++i){
            scores.push_back(keyLine.at(i).response);
        }

        // order feature score to select only the best detected features
        std::vector<uint> vx;
        vx.resize(scores.size());

        // randomly shuffle scores vector as edgelets are detected in order
        std::random_shuffle(scores.begin(), scores.end());
        for( uint i= 0; i < scores.size(); ++i )
            vx.at(i) = i;
        partial_sort( vx.begin(), vx.begin()+std::min(_n_per_cell,(int)vx.size()), vx.end(), Comp(scores) );

        // Keep only the best features (_n if available)
        for(int i=0; i < std::min(_n_per_cell,(int)vx.size()); ++i)
            features.push_back(localfeatures.at(vx[i]));

    }

    void Line2DFeatureDetector::computeDescriptor(const cv::Mat &img, std::vector<std::shared_ptr<AFeature> > &features){

        // Convert Features to KeyLines
        std::vector<cv::line_descriptor::KeyLine> keyLines;
        keyLines.reserve(features.size());
        cv::Mat descriptors;
        FeatureToKeyLine(features, keyLines, descriptors, img.cols);


        // Calculate descriptors for the KeyLines
        std::vector<std::shared_ptr<AFeature> > localfeatures;
        _descriptor->compute(img, keyLines, descriptors);
        KeyLineToFeature(keyLines, descriptors, localfeatures, "linexd");

        // remove not described features
        deleteUndescribedFeatures(localfeatures);
        features = localfeatures;
    }



    cv::line_descriptor::KeyLine Line2DFeatureDetector::MakeKeyLine( cv::Point2f start_pts, cv::Point2f end_pts, size_t cols ){
        cv::line_descriptor::KeyLine keyLine;
        //    keyLine.class_id = 0;
        //    keyLine.numOfPixels;

        // Set start point(and octave)
        if(start_pts.x > end_pts.x)
        {
            cv::Point2f tmp_pts;
            tmp_pts = start_pts;
            start_pts = end_pts;
            end_pts = tmp_pts;
        }

        keyLine.startPointX = (int)start_pts.x;
        keyLine.startPointY = (int)start_pts.y;
        keyLine.sPointInOctaveX = start_pts.x;
        keyLine.sPointInOctaveY = start_pts.y;

        // Set end point(and octave)
        keyLine.endPointX = (int)end_pts.x;
        keyLine.endPointY = (int)end_pts.y;
        keyLine.ePointInOctaveX = end_pts.x;
        keyLine.ePointInOctaveY = end_pts.y;

        // Set angle
        keyLine.angle = std::atan2((end_pts.y-start_pts.y),(end_pts.x-start_pts.x));

        // Set line length & response
        keyLine.lineLength = keyLine.numOfPixels = norm( cv::Mat(end_pts), cv::Mat(start_pts));
        keyLine.response = cv::norm( cv::Mat(end_pts), cv::Mat(start_pts))/cols;

        // Set octave
        keyLine.octave = 0;

        // Set pt(mid point)
        keyLine.pt = (start_pts + end_pts)/2;

        // Set size
        keyLine.size = fabs((end_pts.x-start_pts.x) * (end_pts.y-start_pts.y));

        return keyLine;
    }


    void Line2DFeatureDetector::KeyLineToFeature(std::vector<cv::line_descriptor::KeyLine>  &keyLine, cv::Mat &descriptors, std::vector<std::shared_ptr<AFeature> > &localfeatures, const std::string &featurelabel){
        
        // for each line create a feature
        for (uint i = 0; i < keyLine.size(); ++i) {
            std::vector<Eigen::Vector2d> points;
        
            points.push_back( Eigen::Vector2d(keyLine.at(i).startPointX, keyLine.at(i).startPointY) );
            points.push_back( Eigen::Vector2d(keyLine.at(i).endPointX, keyLine.at(i).endPointY) );

            if (descriptors.empty())
                std::cerr << "empty descriptor" << std::endl; // features.push_back(std::make_shared<Point2D>(pt));
            else {
                if (featurelabel == "linexd")
                    localfeatures.push_back(std::make_shared<Line2D>(points, descriptors.row(i)));
            }
        }
    }

    void Line2DFeatureDetector::FeatureToKeyLine(std::vector<std::shared_ptr<AFeature> > &localfeatures, std::vector<cv::line_descriptor::KeyLine>  &keyLine, cv::Mat &descriptors, uint nb_cols){
        
        // for each line create a feature
        for (uint i = 0; i < localfeatures.size(); ++i) {
            descriptors.push_back(localfeatures.at(i)->getDescriptor());
            keyLine.push_back( MakeKeyLine(cv::Point2f(localfeatures.at(i)->getPoints().at(0)(0), localfeatures.at(i)->getPoints().at(0)(1)),
                                           cv::Point2f(localfeatures.at(i)->getPoints().at(1)(0), localfeatures.at(i)->getPoints().at(1)(1)), 
                                           nb_cols) );            
        }
    }


}// namespace isae


