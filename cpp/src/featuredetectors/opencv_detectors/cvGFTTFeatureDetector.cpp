#include "isaeslam/featuredetectors/opencv_detectors/cvGFTTFeatureDetector.h"

namespace isae {

std::vector<std::shared_ptr<AFeature>> cvGFTTFeatureDetector::detectAndComputeGrid(
    const cv::Mat &img,
    const cv::Mat &mask,
    std::vector<std::shared_ptr<AFeature>> existing_features = std::vector<std::shared_ptr<AFeature>>()) {
    int cell_size = floor(std::sqrt(img.rows * img.cols * _n_per_cell / _n_total));

    // Compute the number of rows and columns needed
    int n_rows = floor(img.rows / cell_size);
    int n_cols = floor(img.cols / cell_size);

    // Occupied cells and updated mask
    cv::Mat updated_mask;
    mask.copyTo(updated_mask);
    std::vector<std::vector<int>> occupied_cells(n_cols + 1, std::vector<int>(n_rows + 1, 0));

    // Draw a circle on the mask around features
    for (auto feat : existing_features) {
        for (auto pt : feat->getPoints()) {
            occupied_cells[floor((int)pt.x() / cell_size)][floor((int)pt.y() / cell_size)] += 1;
            cv::circle(updated_mask, cv::Point2d(pt.x(), pt.y()), cell_size, cv::Scalar(0), -1);
        }
    }

    std::vector<cv::KeyPoint> new_keypoints;
    std::vector<cv::Point2d> new_keypoints_2d;
    new_keypoints.reserve(_n_total);
    cv::Mat new_descriptors;

    cv::goodFeaturesToTrack(
        img, new_keypoints_2d, _n_total - existing_features.size(), 0.01, cell_size, updated_mask, 3, false, 0.04);

    // Convert to keypoints
    for (uint i = 0; i < new_keypoints_2d.size(); i++) {
        cv::KeyPoint kp;
        kp.pt       = new_keypoints_2d.at(i);
        kp.size     = 1;
        kp.response = 1;
        new_keypoints.push_back(kp);
    }

    // If nothing was detected...
    std::vector<std::shared_ptr<AFeature>> new_features;
    if (new_keypoints.empty())
        return new_features;

    // Compute descriptors
    _descriptor->compute(img, new_keypoints, new_descriptors);

    // build new feature vector
    KeypointToFeature(new_keypoints, new_descriptors, new_features, "pointxd");

    return new_features;
}

} // namespace isae