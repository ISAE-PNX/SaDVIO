#ifndef IMGPROCESSING_H
#define IMGPROCESSING_H

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/imgproc.hpp>

namespace isae {
namespace imgproc {

inline void histogramEqualizationCLAHE(cv::Mat &image, float clahe_clip = 2) {
    bool gray = image.channels() == 1;
    if (gray)
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);

    cv::cvtColor(image, image, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    int tile_xsize = image.cols / 50;
    int tile_ysize = image.rows / 50;

    cv::Ptr<cv::CLAHE> clahe =
        cv::createCLAHE((clahe_clip < 1 ? 1 : clahe_clip),
                        cv::Size((tile_xsize < 1 ? 1 : tile_xsize), (tile_ysize < 1 ? 1 : tile_ysize)));
    clahe->apply(channels[0], channels[0]);
    cv::merge(channels, image);
    cv::cvtColor(image, image, cv::COLOR_YCrCb2BGR);

    if (gray)
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
}

inline void AGCWD(const cv::Mat & src, cv::Mat & dst, double alpha)
{
	int rows = src.rows;
	int cols = src.cols;
	int channels = src.channels();
	int total_pixels = rows * cols;

	cv::Mat L;
	cv::Mat HSV;
	std::vector<cv::Mat> HSV_channels;
	if (channels == 1) {
		L = src.clone();
	}
	else {
		cv::cvtColor(src, HSV, cv::COLOR_HSV2BGR_FULL);
		cv::split(HSV, HSV_channels);
		L = HSV_channels[2];
	}

	int histsize = 256;
	float range[] = { 0,256 };
	const float* histRanges = { range };
	cv::Mat hist;
	calcHist(&L, 1, 0, cv::Mat(), hist, 1, &histsize, &histRanges, true, false);

	double total_pixels_inv = 1.0 / total_pixels;
	cv::Mat PDF = cv::Mat::zeros(256, 1, CV_64F);
	for (int i = 0; i < 256; i++) {
		PDF.at<double>(i) = hist.at<float>(i) * total_pixels_inv;
	}

	double pdf_min, pdf_max;
	cv::minMaxLoc(PDF, &pdf_min, &pdf_max);
	cv::Mat PDF_w = PDF.clone();
	for (int i = 0; i < 256; i++) {
		PDF_w.at<double>(i) = pdf_max * std::pow((PDF_w.at<double>(i) - pdf_min) / (pdf_max - pdf_min), alpha);
	}

	cv::Mat CDF_w = PDF_w.clone();
	double culsum = 0;
	for (int i = 0; i < 256; i++) {
		culsum += PDF_w.at<double>(i);
		CDF_w.at<double>(i) = culsum;
	}
	CDF_w /= culsum;

	std::vector<uchar> table(256, 0);
	for (int i = 1; i < 256; i++) {
		table[i] = cv::saturate_cast<uchar>(255.0 * std::pow(i / 255.0, 1 - CDF_w.at<double>(i)));
	}

	cv::LUT(L, table, L);

	if (channels == 1) {
		dst = L.clone();
	}
	else {
		cv::merge(HSV_channels, dst);
		cv::cvtColor(dst, dst, cv::COLOR_HSV2BGR_FULL);
	}

	return;
}

inline void highBoostFiltering(cv::Mat &I, cv::Mat &Ires, float scale) {
    cv::Mat Kernel         = -scale * cv::Mat::ones(3, 3, CV_32FC1);
    Kernel.at<float>(1, 1) = 8 * scale;

    cv::filter2D(I, Ires, -1, Kernel);

    Ires += I;
}

inline void imageImprovment(cv::Mat &image, cv::Mat &result) {
    /** Image improvment */
    cv::Mat img = image;
    cv::GaussianBlur(img, img, cv::Size(5, 5), .7, .7);
    histogramEqualizationCLAHE(img);
    // highBoostFiltering(img,result,0.05);
}

inline double ZNCC(cv::Mat I0, cv::Mat I1) {

    // The patches must have the same size
    if (I0.size() != I1.size()) {
        return 1000;
    }

    cv::Scalar mean0, stdev0, mean1, stdev1;
    cv::meanStdDev(I0, mean0, stdev0);
    cv::meanStdDev(I1, mean1, stdev1);

    double s = 0;
    for (int i = 0; i < I0.rows; i++) {
        for (int j = 0; j < I0.cols; j++) {
            s = s + (I0.at<uint8_t>(i, j) - mean0[0]) * (I1.at<uint8_t>(i, j) - mean1[0]);
        }
    }

    return s / (I0.rows * I0.cols * stdev0[0] * stdev1[0]);
}

inline double ZNCC2(cv::Mat I0, cv::Mat I1) {

    // The patches must have the same size
    if (I0.size() != I1.size()) {
        return 1000;
    }

    double s = 0;
	double s0 = 0;
	double s1 = 0;
    for (int i = 0; i < I0.rows; i++) {
        for (int j = 0; j < I0.cols; j++) {
            s = s + I0.at<uint8_t>(i, j) * I1.at<uint8_t>(i, j);
			s0 = I0.at<uint8_t>(i, j) * I0.at<uint8_t>(i, j);
			s1 = I1.at<uint8_t>(i, j) * I1.at<uint8_t>(i, j);
        }
    }

    return s / std::sqrt(s0 * s1);
}

} // namespace imgproc
} // namespace isae

#endif // IMGPROCESSING_H
