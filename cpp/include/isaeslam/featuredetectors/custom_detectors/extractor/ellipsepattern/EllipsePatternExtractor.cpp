#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "isaeslam/featuredetectors/custom_detectors/extractor/ellipsepattern/EllipsePatternExtractor.h"


namespace isae {


void EllipsePatternExtractor::extract(const cv::Mat &in_image, std::vector<EllipsePattern> &ellipses_patterns, cv::Mat &out_descriptors)
{
    // Detect ellipsoidal blobs
    std::vector<Ellipse> ellipses = detect_ellipses(in_image);

    // Get homographies to get circular
    std::vector<std::vector<cv::Point2d>> ellipses_points;
    std::vector<double> white_val, black_val;
    std::vector<cv::Mat> homographies = get_ellipses_homographies(in_image, ellipses, ellipses_points, white_val, black_val);

    // Read coded message and add ID to ellipses
    read_ellipses(in_image, ellipses, ellipses_points, homographies, black_val, white_val);

    //this->display(in_image, ellipses);

    // Try to get patterns from the detections
    ellipses_patterns = extract_pattern(ellipses);

    // Create descriptors matrix (list of detected pattern IDs)
    out_descriptors = cv::Mat (ellipses_patterns.size(), 1, CV_64FC1);
    for(uint i=0; i < ellipses_patterns.size(); ++i)
        out_descriptors.at<double>(i) = ellipses_patterns.at(i).ID();

}




void EllipsePatternExtractor::image_processing(const cv::Mat image, cv::Mat &binary_image, const int thresholdBlockSize)
{

    //! GrayScale conversion
    cv::Mat gray_image(image.rows, image.cols, CV_8UC1);
    if(image.channels() == 3)
        cv::cvtColor( image, gray_image, cv::COLOR_RGB2GRAY );
    else
        gray_image = image;

    //! Adaptative Thresholding
    cv::adaptiveThreshold(gray_image, binary_image,255,cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, thresholdBlockSize, 0.0);
}




const std::vector<Ellipse> EllipsePatternExtractor::detect_ellipses(const cv::Mat & image)
{
    cv::Mat binary_image;
    image_processing(image, binary_image);

    //! Contour extraction
    std::vector < std::vector<cv::Point> > contours;
    cv::findContours(binary_image, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    //! Blob filtering, keep only ellipses
    std::vector<Ellipse> ellipses;

    for (size_t i = 0; i < contours.size(); ++i)
    {
        //! Reject with area
        double area = cv::contourArea(contours[i]);
        if (area < 10 /*|| area >= 50000*/) continue;

        //! Fit ellipse on data
        cv::RotatedRect fitted_ellipse = cv::fitEllipse(contours[i]);

        //! Ellipsity factor
        double ellipsity_factor = 0.25 * CV_PI * fitted_ellipse.size.width * fitted_ellipse.size.height / area;
        if (ellipsity_factor < 0.95 || ellipsity_factor > 1.05) continue;


        Ellipse e(fitted_ellipse.center,
                  std::max(fitted_ellipse.size.width, fitted_ellipse.size.height)/2.,
                  std::min(fitted_ellipse.size.width, fitted_ellipse.size.height)/2.,
                  fitted_ellipse.angle*M_PI/180.);

        ellipses.push_back(e);
    }

    return ellipses;
}



const std::vector<cv::Mat> EllipsePatternExtractor::get_ellipses_homographies(const cv::Mat &image, const std::vector<Ellipse> &ellipses, std::vector<std::vector<cv::Point2d>> & ellipses_points, std::vector<double> &white_val, std::vector<double> &black_val)
{
    std::vector<cv::Mat> Homographies(ellipses.size());

    //#pragma omp parallel for
    for(uint c =0; c < ellipses.size(); ++c)  //for(auto e : ellipses)
    {
        Ellipse e = ellipses.at(c);
        // detected points
        std::vector<cv::Point2d> detected_points;
        detected_points.push_back(e.center());
        detected_points.push_back(e.center()+cv::Point2d(e.a()*cos(e.theta()+M_PI/2), e.a()*sin(e.theta()+M_PI/2)));
        detected_points.push_back(e.center()+cv::Point2d(e.b()*cos(e.theta()), e.b()*sin(e.theta())));
        detected_points.push_back(e.center()-cv::Point2d(e.a()*cos(e.theta()+M_PI/2), e.a()*sin(e.theta()+M_PI/2)));
        detected_points.push_back(e.center()-cv::Point2d(e.b()*cos(e.theta()), e.b()*sin(e.theta())));

        // desired points
        std::vector<cv::Point2d> desired_points;
        desired_points.push_back(e.center());
        desired_points.push_back(e.center()-cv::Point2d(e.a(),0));
        desired_points.push_back(e.center()+cv::Point2d(0,e.a()));
        desired_points.push_back(e.center()+cv::Point2d(e.a(),0));
        desired_points.push_back(e.center()-cv::Point2d(0,e.a()));

        // find homography
        cv::Mat H = cv::findHomography(detected_points, desired_points);
        Homographies.at(c) = H;


        // Find mean local black & white colors
        // Find location of code
        cv::Mat grayImage;

        if(image.channels() == 3)
            cv::cvtColor(image,grayImage, cv::COLOR_RGB2GRAY);
        else
            grayImage = image;

        std::vector<cv::Point2d> white_pix, white_pix_ell, black_pix, black_pix_ell, circle_pts, ellipse_pts;

        uint nb_points = 128;
        std::vector<double> angs = linspace(2*M_PI, 0., nb_points);

        for(double i : angs){
            white_pix.push_back(cv::Point2d(1.15*e.a()*cos(i), 1.15*e.a()*sin(i))+e.center());
            black_pix.push_back(cv::Point2d(0.75*e.a()*cos(i), 0.75*e.a()*sin(i))+e.center());
            circle_pts.push_back(cv::Point2d(params().radius_ratio()*e.a()*cos(i), params().radius_ratio()*e.a()*sin(i))+e.center());
        }
        cv::perspectiveTransform(white_pix, white_pix_ell, H.inv());
        cv::perspectiveTransform(black_pix, black_pix_ell, H.inv());
        cv::perspectiveTransform(circle_pts, ellipse_pts, H.inv());

        // get back & white mean values
        white_val.push_back(mean_interp2(grayImage, white_pix_ell, grayImage.type()));
        black_val.push_back(mean_interp2(grayImage, black_pix_ell, grayImage.type()));

        // get ellipse points
        ellipses_points.push_back(ellipse_pts);
    }
    return Homographies;
}


void EllipsePatternExtractor::read_ellipses(const cv::Mat &image, std::vector<Ellipse> &ellipses, const std::vector<std::vector<cv::Point2d>> & ellipses_points, const std::vector<cv::Mat> & Homographies, const std::vector<double> &white_val, const std::vector<double> &black_val)
{
    cv::Mat display = image.clone();
    for(uint i =0; i < ellipses.size(); ++i)
    {
        // Get code bar values on the desired ellipse points
        std::vector<double> code_values;
        std::vector<uint> readed_binary_code;

        cv::Mat grayImage;

        if(image.channels() == 3)
            cv::cvtColor(image, grayImage, cv::COLOR_RGB2GRAY);
        else
            grayImage = image;

        for(uint p =0; p < ellipses_points.at(i).size(); ++p)
        {
            double val = interp2(grayImage, ellipses_points.at(i).at(p), grayImage.type());
            code_values.push_back(val);
            readed_binary_code.push_back( (val < (white_val.at(i)+black_val.at(i))/2./*2./3.*/)? 1 : 0); // black = 1, white = 0
        }

        double orientation;
        uint ID = decode_ellipse(readed_binary_code, orientation);
        ellipses.at(i).set_orientation(ellipses.at(i).theta() - orientation - M_PI/2.);
        ellipses.at(i).set_ID(ID);
    }

    // Delete wrong ellipses
    for(uint i = 0; i< ellipses.size(); ++i)
    {
        if (ellipses.at(i).ID() < 6400 || ellipses.at(i).ID() > 6500){
            ellipses.erase(ellipses.begin()+i);
            i = i-1;
        }
    }


}

const uint EllipsePatternExtractor::decode_ellipse(std::vector<uint> binary_code, double &orientation)
{
    std::vector<uint> extended_pattern = extend_pattern(this->params().binary_pattern(), binary_code.size()/2);

    uint begin_offset = circular_correlation(binary_code, extended_pattern);
    std::vector<uint> compressed_pattern = compress_pattern(binary_code, extended_pattern.size()/this->params().binary_pattern().size(), begin_offset);

    orientation = ((double)begin_offset/(double)binary_code.size())*2*M_PI;

    return binary_to_decimal(compressed_pattern, true);
}



const std::vector<uint> EllipsePatternExtractor::extend_pattern(const std::vector<uint> & binary_pattern, const int & nb_value)
{
    if (nb_value%((int)binary_pattern.size()) != 0){
        std::cout << "generate_pattern::nb_value is not a multiple of binary_pattern_size" << std::endl;
        abort();
    }

    std::vector<uint> extended_pattern;
    const size_t rep = nb_value / binary_pattern.size();

    for(size_t i = 0; i < binary_pattern.size(); ++i)
        for(size_t j = 0; j < rep; ++j)
            extended_pattern.push_back((binary_pattern[i]));

    return extended_pattern;
}


const std::vector<uint> EllipsePatternExtractor::compress_pattern(const std::vector<uint> &binary_pattern, const int &nb_value, const uint & begin_offset)
{
    if (binary_pattern.size()%nb_value != 0){
        std::cout << "compress_pattern::nb_value(length code) is not a multiple of binary_pattern_size" << std::endl;
        abort();
    }

    const size_t rep = binary_pattern.size()/nb_value;

    std::vector<uint> compressed_pattern;
    for(size_t i = 0; i < rep; ++i){
        double moy_sum = 0.0;
        for(int j = 0; j < nb_value; ++j){
            size_t data_ind = i*nb_value+j+begin_offset;
            if (data_ind >= binary_pattern.size()) data_ind = data_ind - binary_pattern.size();
            moy_sum += binary_pattern.at(data_ind);
        }

        moy_sum = moy_sum/((double)nb_value);

        if (moy_sum > (double)0.5)
            compressed_pattern.push_back(1);
        else
            compressed_pattern.push_back(0);
    }
    return compressed_pattern;
}




const uint EllipsePatternExtractor::circular_correlation(const std::vector<uint> & readed_pattern, const std::vector<uint> & extended_pattern)
{

    if (extended_pattern.size() > readed_pattern.size()){
        std::cout << "Pattern is bigger than readed code... abort" << std::endl;
        abort();
    }

    //! Circular Augmentation of data
    std::vector<uint> data = readed_pattern;
    for(size_t i = 0; i < extended_pattern.size(); ++i)
        data.push_back(readed_pattern.at(i));

    //! For all shifts, calculate correlation
    int min_score = 100000000;
    int begin_offset = -1;


//    cv::Mat res;
//    cv::matchTemplate(cv::Mat(readed_pattern), cv::Mat(extended_pattern), res, cv::TM_CCORR_NORMED);

//    double minVal;
//    double maxVal;
//    cv::Point minLoc;
//    cv::Point maxLoc;

//    cv::minMaxLoc( res, &minVal, &maxVal, &minLoc, &maxLoc );




    for(uint i = 0; i < readed_pattern.size(); ++i){
        int score = 0;
        for(size_t j = 0; j < extended_pattern.size(); ++j){
            size_t data_ind = i+j;
            score += (data.at(data_ind) - extended_pattern.at(j)) * (data.at(data_ind) - extended_pattern.at(j));
        }

        if (score < min_score){
            min_score = score;
            begin_offset = i;
        }
    }

    return begin_offset;
}




const std::vector<EllipsePattern> EllipsePatternExtractor::extract_pattern(const std::vector<Ellipse> &ellipses)
{
    std::vector<EllipsePattern> patterns;
    if(params_.nb_mires() == 4)
        patterns = extract_pattern_4(ellipses);
    else if(params_.nb_mires() == 16)
        patterns = extract_pattern_16(ellipses);

    return patterns;
}



const std::vector<EllipsePattern> EllipsePatternExtractor::extract_pattern_4(const std::vector<Ellipse> &ellipses)
{
    std::vector<EllipsePattern> patterns;
    std::vector<Ellipse> E1, E2, E3, EN;

    // Sort ellipses wrt ID
    for(auto e : ellipses)
    {
        switch (e.ID()) {
            case 6401: { E1.push_back(e); break;}
            case 6402: { E2.push_back(e); break;}
            case 6403: { E3.push_back(e); break;}
            default:{ EN.push_back(e); break;}
        }
    }

    // For each ellipse 1, search for the complete pattern
    for(auto e1 : E1){
        EllipsePattern p;
        p.push_ellipse(e1);
        // predict other ellipses center
        double scale = 4;
        cv::Point2d c2(e1.center() + cv::Point2d(scale*e1.a()*cos(-e1.orientation()+M_PI/2.), -scale*e1.a()*sin(-e1.orientation()+M_PI/2.)));
        cv::Point2d c3(e1.center() + cv::Point2d(scale*e1.a()*cos(-e1.orientation()), -scale*e1.a()*sin(-e1.orientation())));
        cv::Point2d cn(e1.center() + cv::Point2d(scale*e1.a()*cos(-e1.orientation()), -scale*e1.a()*sin(-e1.orientation())) + cv::Point2d(scale*e1.a()*cos(-e1.orientation()+M_PI/2.), -scale*e1.a()*sin(-e1.orientation()+M_PI/2.)));

        // Get ellipse 2
        for(auto e2 : E2){
            if(cv::norm(e2.center() - c2) < scale*e1.a()){
                p.push_ellipse(e2);
            }
        }

        // Get ellipse 3
        for(auto e3 : E3){
            if(cv::norm(e3.center() - c3) < scale*e1.a()){
                p.push_ellipse(e3);
            }
        }

        // Get ellipse N
        for(auto en : EN){
            if(cv::norm(en.center() - cn) < scale*e1.a()){
                p.push_ellipse(en);
                p.set_ID(en.ID()-6400);
            }
        }
        if(p.ellipses().size() == 4)
            patterns.push_back(p);
    }
    return patterns;
}


const std::vector<EllipsePattern> EllipsePatternExtractor::extract_pattern_16(const std::vector<Ellipse> &ellipses)
{
    std::vector<EllipsePattern> patterns;
    EllipsePattern pattern;

    if(ellipses.size() != 16)
        return patterns;

    // Recherche et classement des ellipses
    for(uint i =1; i <= 16; ++i)
    {
        for(auto e : ellipses)
        {
            if(e.ID()-6400 == i)
            {
                pattern.push_ellipse(e);
                break;
            }
        }
    }
    if(pattern.ellipses().size() == 16)
        patterns.push_back(pattern);

    return patterns;
}


} // namespace isae