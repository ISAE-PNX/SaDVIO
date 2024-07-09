#include "isaeslam/featuredetectors/custom_detectors/csvKeypointDetector.h"

#include <fstream>
#include <boost/tokenizer.hpp>
#include <algorithm> 

using namespace boost;

namespace isae {

void CsvKeypointDetector::init() {
    _folder_path = "/media/ce.debeunne/HDD/datasets/OIVIO/MN_050_GV_01/husky0/cam0/sift_kp";
}

void CsvKeypointDetector::customDetectAndCompute(const cv::Mat &img,
                                                     const cv::Mat &mask,
                                                     std::vector<std::shared_ptr<AFeature>> &features) {

    double ts = img.at<double>(0,0);
    std::string csv_path = _folder_path + "/" + std::to_string((unsigned long long)ts) + ".csv";
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    // Input stream
    std::ifstream in(csv_path.c_str());
    std::string line;
    std::vector<std::string> vec;

    // Read file
    bool first_line = true;
    while (getline(in, line)){

        if (first_line) {
            first_line = false;
            continue;
        }

        // Get kps and descriptors
        size_t sz;
        size_t start_value = 1;
        double u = std::stod(&line[start_value], &sz);
        start_value = start_value + 2 + sz;
        double v = std::stod(&line[start_value], &sz);

        start_value = start_value + 4 + sz;
        std::vector<int> desc_vec;
        for (size_t k = 0; k < 128; k++){
            int value = std::stoi(&line[start_value], &sz);
            start_value = start_value + 2 + sz;
            desc_vec.push_back(value);
        }
        cv::Mat desc = cv::Mat( desc_vec ).reshape(0, 128);

        // Update vectors / mat
        keypoints.push_back(cv::KeyPoint(cv::Point2d(u, v), 1));
        descriptors.push_back(desc.t());
    }
    
    this->KeypointToFeature(keypoints, descriptors, features);
}

void CsvKeypointDetector::computeDescriptor(const cv::Mat &img, std::vector<std::shared_ptr<AFeature>> &features) {}

double CsvKeypointDetector::getDist(const cv::Mat &desc1, const cv::Mat &desc2) const
{

    return cv::norm(desc1, desc2, cv::NORM_L2);
}

} // namespace isae