//
// Created by d.vivet on 23/04/2021.
//

#ifndef ELLIPSEPATTERNEXTRACTOR_H
#define ELLIPSEPATTERNEXTRACTOR_H

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "isaeslam/typedefs.h"

namespace isae
{

class Ellipse
{
public:
    Ellipse(cv::Point2d center, double a, double b, double theta):center_(center),a_(a), b_(b), theta_(theta),orientation_(theta){ID_=0;}

    // Getters
    const double a() const {return a_;}
    const double b() const {return b_;}
    const double theta() const {return theta_;}
    const double orientation() const {return orientation_;}
    const cv::Point2d center() const {return center_;}
    const uint ID() const {return ID_;}
    const cv::RotatedRect rotatedRect() const {
        cv::RotatedRect rec(center_, cv::Size2d(2*a_,2*b_), theta_*180./M_PI - 90.);
        return rec;
    }


    // Setters
    void set_a(double a){a_ = a;}
    void set_b(double b){b_ = b;}
    void set_theta(double theta){theta_ = theta;}
    void set_orientation(double orientation){orientation_ = orientation;}
    void set_a(cv::Point2d center){center_ = center;}
    void set_ID(uint ID){ID_ = ID;}

private:
    cv::Point2d center_;
    double a_;
    double b_;
    double theta_;
    double orientation_;
    uint ID_;
};



class EllipsePattern
{
public:
    EllipsePattern(){}

    // Getters
    const std::vector<Ellipse> & ellipses() const {return ellipses_;}
    std::vector<Eigen::Vector2d> p2ds() {
        std::vector<Eigen::Vector2d> p2ds;
        for(uint i=0; i < 4; ++i)
            p2ds.push_back(Eigen::Vector2d(ellipses_.at(i).center().x, ellipses_.at(i).center().y));
        return p2ds;
    }

    const uint & ID() const {return ID_;}

    // Setters
    void set_ellipses(std::vector<Ellipse> ellipses){ellipses_ = ellipses;}
    void set_ID(uint ID){ID_ = ID;}
    void push_ellipse(Ellipse e){ellipses_.push_back(e);}

private:
    std::vector<Ellipse> ellipses_;
    uint ID_;
};



class EllipsePatternParameters
{
public:
    EllipsePatternParameters(){}

    // getters
    //* Get the header of the coded message used as validation
    const std::vector<uint> & binary_pattern() const {return binary_pattern_;}

    //* Get the distance between the center of two adjacent ellipses
    const double & distance() const {return distance_;}

    //* Get the radius ratio between the ellipses circle and the middle of coded message circle
    const double & radius_ratio() const {return radius_ratio_;}

    //* Get the number of mire composing the pattern
    const uint & nb_mires() const {return nb_mires_;}


    // setters
    void set_binary_pattern(std::string binary_pattern){
        for(auto c :binary_pattern)
            binary_pattern_.push_back(c - '0');
    }
    void set_distance(const double & distance){distance_=distance;}
    void set_radius_ratio(const double & radius_ratio){radius_ratio_ = radius_ratio;}
    void set_nb_mires(const uint & nb_mires){nb_mires_ = nb_mires;}

 private:

    uint nb_mires_;
    double distance_;
    double radius_ratio_;
    std::vector<uint> binary_pattern_;
};













    class EllipsePatternExtractor {
    public:        
        //! Constructor
        EllipsePatternExtractor(){
            params_.set_radius_ratio(1.68);
            params_.set_distance(10); // 4
            params_.set_nb_mires(4); // 16
            params_.set_binary_pattern("10110010");
        }

        //! Destructor
        ~EllipsePatternExtractor(){}

        //! Extract keypoints and each descriptor of them
        void extract(const cv::Mat &in_image, std::vector<EllipsePattern> &ellipses, cv::Mat &out_descriptors);


        void display(const cv::Mat &image, std::vector<Ellipse> &ellipses)
        {
            cv::Mat disp = image.clone();

            for(uint i=0; i < ellipses.size(); ++i)
            {
                cv::ellipse(disp, ellipses.at(i).rotatedRect(), cv::Scalar(255,0,0), 1);
                cv::putText(disp, std::to_string(ellipses.at(i).ID()-6400), ellipses.at(i).center(), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255,0,0));
                cv::line(disp, ellipses.at(i).center(), ellipses.at(i).center() + 2*cv::Point2d(ellipses.at(i).a()*cos(ellipses.at(i).orientation()), ellipses.at(i).a()*sin(ellipses.at(i).orientation())),cv::Scalar(0,255,0), 1);
            }
            cv::namedWindow("Detection ell", cv::WINDOW_NORMAL);
            cv::imshow("Detection ell", disp);
            cv::waitKey(0);
        }

    private:
        void set_params(EllipsePatternParameters params){params_ = params;}
        EllipsePatternParameters & params(){return params_;}

        const std::vector<Ellipse> detect_ellipses(const cv::Mat &image);                        
        const std::vector<cv::Mat> get_ellipses_homographies(const cv::Mat &image, const std::vector<Ellipse> &ellipsoidal_blobs, std::vector<std::vector<cv::Point2d>> & ellipses_points, std::vector<double> &white_val, std::vector<double> &black_val);
        void read_ellipses(const cv::Mat &image, std::vector<Ellipse> &ellipses, const std::vector<std::vector<cv::Point2d>> & ellipses_points, const std::vector<cv::Mat> & Homographies, const std::vector<double> &white_val, const std::vector<double> &black_val);
        const uint decode_ellipse(std::vector<uint> binary_code, double &angle_offset);
        const std::vector<uint> extend_pattern(const std::vector<uint> &binary_pattern, const int &nb_value);
        const std::vector<uint> compress_pattern(const std::vector<uint> &binary_pattern, const int &nb_value, const uint &begin_offset);
        const uint circular_correlation(const std::vector<uint> & readed_pattern, const std::vector<uint> & extended_pattern);
        const std::vector<EllipsePattern> extract_pattern(const std::vector<Ellipse> & ellipses);
        const uint circular_correlation_fft(const std::vector<uint> & readed_pattern, const std::vector<uint> & extended_pattern);

        const std::vector<EllipsePattern> extract_pattern_4(const std::vector<Ellipse> &ellipses);
        const std::vector<EllipsePattern> extract_pattern_16(const std::vector<Ellipse> &ellipses);

        void image_processing(const cv::Mat image, cv::Mat &binary_image, const int thresholdBlockSize = 21);

        EllipsePatternParameters params_;
    };


} //namespace isae






inline double interp2(const cv::Mat &I, const cv::Point2d &pp, const uint &depth)
{
    // pp.x => y  pp.y => x because of frame image vs xy
    double x1,y1,x2,y2;
    y1 = (double)std::floor(pp.x);
    y2 = (double)std::ceil(pp.x);
    x1 = (double)std::floor(pp.y);
    x2 = (double)std::ceil(pp.y);
    double dfx,dfy,dfxy, res;
    res = 0.0;


    switch ( depth ) {
        case CV_8U:
            dfx = (double)I.at<uchar>(x2,y1) - (double)I.at<uchar>(x1,y1);
            dfy = (double)I.at<uchar>(x1,y2) - (double)I.at<uchar>(x1,y1);
            dfxy = (double)I.at<uchar>(x1,y1) + (double)I.at<uchar>(x2,y2) - (double)I.at<uchar>(x2,y1) - (double)I.at<uchar>(x1,y2);
            res = dfx*(pp.y-x1)/(x2-x1) + dfy*(pp.x-y1)/(y2-y1) + dfxy*(pp.y-x1)/(x2-x1)*(pp.x-y1)/(y2-y1) + (double)I.at<uchar>(x1,y1);
            break;

        case CV_32F:
            dfx = (double)I.at<float>(x2,y1) - (double)I.at<float>(x1,y1);
            dfy = (double)I.at<float>(x1,y2) - (double)I.at<float>(x1,y1);
            dfxy = (double)I.at<float>(x1,y1) + (double)I.at<float>(x2,y2) - (double)I.at<float>(x2,y1) - (double)I.at<float>(x1,y2);
            res = dfx*(pp.y-x1)/(x2-x1) + dfy*(pp.x-y1)/(y2-y1) + dfxy*(pp.y-x1)/(x2-x1)*(pp.x-y1)/(y2-y1) + (double)I.at<float>(x1,y1);
            break;

    }
    return res;
}



inline double mean_interp2(const cv::Mat &I, const std::vector<cv::Point2d> &pps, const uchar &depth)
{
    double mean = 0.0;
    for(auto pt : pps)
        mean += interp2(I, pt, depth);

    return mean/pps.size();
}





template<typename T>
inline T median(std::vector<T> &v)
{
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
}


template<typename T>
inline T mean(std::vector<T> &v)
{
    T mean = 0;
    for(auto val : v)
        mean+=val;

    return mean/v.size();
}



template <typename T>
inline std::vector<T> linspace(T a, T b, size_t N) {
    T h = (b - a) / static_cast<T>(N-1);
    std::vector<T> xs(N);
    typename std::vector<T>::iterator x;
    T val;
    for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
        *x = val;
    return xs;
}


inline uint binary_to_decimal(const std::vector<uint> & binary, const bool & have_parity)
{
    uint dec = 0;
    uint p2  = 1;

    if (have_parity)
    {
        for(uint i = 1; i < (binary.size()-1) ; ++i)
        {
            dec += binary[binary.size()-1-i] * p2;
            p2 = p2 * 2;
        }
    }
    else
    {
        p2 = p2*2;
        for(uint i = 0; i < (binary.size()) ; ++i)
        {
            dec += binary[binary.size()-i] * p2;
            p2 = p2 * 2;
        }
    }

    return dec;
}

#endif //ELLIPSEPATTERNEXTRACTOR_H

