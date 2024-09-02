#include <mutex>
#include <opencv2/highgui/highgui.hpp>
#include <thread>

#include "isaeslam/featuredetectors/aOpenCVFeatureDetector.h"

namespace isae {

void AOpenCVFeatureDetector::detectAndCompute(const cv::Mat &img,
                                              const cv::Mat &mask,
                                              std::vector<cv::KeyPoint> &keypoints,
                                              cv::Mat &descriptors,
                                              int n_points) {
    // call OpenCV detect function
    _detector->detect(img, keypoints, mask);

    if (!keypoints.empty()) {
        // retain the good number of keypoints
        retainBest(keypoints, n_points);

        // Describe Keypoints
        _descriptor->compute(img, keypoints, descriptors);
    }
}

void subPixelRef(cv::Mat I, std::vector<cv::KeyPoint> &keypoints) {
    std::vector<cv::Point2f> pts;

    for (uint i = 0; i < keypoints.size(); i++) {
        pts.push_back(keypoints.at(i).pt);
    }
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01);
    cv::cornerSubPix(I, pts, cv::Size(3, 3), cv::Size(-1, -1), criteria);

    for (uint i = 0; i < keypoints.size(); i++) {
        keypoints.at(i).pt = pts.at(i);
    }
}

std::vector<std::shared_ptr<AFeature>>
AOpenCVFeatureDetector::detectAndComputeGrid(const cv::Mat &img,
                                             const cv::Mat &mask,
                                             std::vector<std::shared_ptr<AFeature>> existing_features) {
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
            cv::circle(updated_mask, cv::Point2d(pt.x(), pt.y()), 5, cv::Scalar(0), -1);
        }
    }

    std::vector<cv::KeyPoint> new_keypoints;
    new_keypoints.reserve(_n_total);
    cv::Mat new_descriptors;

    // Define local detection function
    std::mutex mtx;
    auto detectComputeSmall =
        [cell_size, img, n_rows, updated_mask, occupied_cells, this, &mtx, &existing_features, &new_keypoints](
            int col_start, int col_end) {
            for (int col = col_start; col < col_end; col++) {
                for (int row = 0; row < n_rows; row++) {
                    int n_to_detect = _n_per_cell - occupied_cells[col][row];

                    if (n_to_detect < 1)
                        continue;

                    int x = col * cell_size;
                    int y = row * cell_size;
                    int width, height;

                    if (x + cell_size > img.cols - 1)
                        width = img.cols - x;
                    else
                        width = cell_size;

                    if (y + cell_size > img.rows - 1)
                        height = img.rows - y;
                    else
                        height = cell_size;

                    cv::Rect roi(x, y, width, height);

                    // Detect the perfect amount of keypoints
                    std::vector<cv::KeyPoint> keypoints_local;
                    keypoints_local.reserve(_n_per_cell);
                    _detector->detect(img(roi), keypoints_local, updated_mask(roi));
                    retainBest(keypoints_local, n_to_detect);

                    // Add local detection to the full detection
                    {
                        mtx.lock();
                        for (uint i = 0; i < keypoints_local.size(); i++) {

                            keypoints_local.at(i).pt += cv::Point2f(x, y);
                            new_keypoints.push_back(keypoints_local.at(i));

                            cv::circle(updated_mask, keypoints_local.at(i).pt, 5, cv::Scalar(0), -1);
                        }
                        mtx.unlock();
                    }
                }
            }
        };

    // If there is not grid, detection on the full image
    if (_n_per_cell == _n_total) {
        n_rows = 0;
        n_cols = 0;

        _detector->detect(img, new_keypoints, updated_mask);
        retainBest(new_keypoints, _n_per_cell);
    }

    // Launch on different thread the local detections
    int n_threads = 2;
    int chunk     = (int)(n_cols / n_threads);
    std::vector<std::thread> threads;
    for (int k = 0; k < n_threads - 1; k++) {
        threads.push_back(std::thread(detectComputeSmall, k * chunk, (k + 1) * chunk));
    }
    threads.push_back(std::thread(detectComputeSmall, chunk * (n_threads - 1), n_cols));

    for (auto &th : threads) {
        th.join();
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

void AOpenCVFeatureDetector::computeDescriptor(const cv::Mat &img, std::vector<std::shared_ptr<AFeature>> &features) {

    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(features.size());

    cv::Mat descriptors;
    FeatureToKeypoint(features, keypoints, descriptors);
    _descriptor->compute(img, keypoints, descriptors);

    // compute can delete or add keypoints, for each features, find the good descriptor
    for (uint i = 0; i < features.size(); ++i) {
        for (uint k = 0; k < keypoints.size(); ++k) {
            cv::KeyPoint kp = keypoints.at(k);
            if (features.at(i)->getPoints().at(0).x() == kp.pt.x && features.at(i)->getPoints().at(0).y() == kp.pt.y) {
                features.at(i)->setDescriptor(descriptors.row(k));
                break;
            }
        }
    }

}

void AOpenCVFeatureDetector::retainBest(std::vector<cv::KeyPoint> &_keypoints, int n) {
    if (_keypoints.size() > size_t(n)) {
        if (n == 0) {
            _keypoints.clear();
            return;
        }
        std::nth_element(_keypoints.begin(),
                         _keypoints.begin() + n,
                         _keypoints.end(),
                         [](cv::KeyPoint &a, cv::KeyPoint &b) { return a.response > b.response; });
        _keypoints.resize(n);
    }
}

} // namespace isae
