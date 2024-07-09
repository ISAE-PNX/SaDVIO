#include <mutex>
#include <thread>

#include "isaeslam/featuredetectors/aCustomFeatureDetector.h"

namespace isae {

std::vector<std::shared_ptr<AFeature>>
ACustomFeatureDetector::detectAndComputeGrid(const cv::Mat &img,
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
            cv::circle(updated_mask, cv::Point2d(pt.x(), pt.y()), 10, cv::Scalar(0));
        }
    }

    // Define local detection function
    std::mutex mtx;
    std::vector<std::shared_ptr<AFeature>> new_features;
    auto detectComputeSmall =
        [cell_size, img, updated_mask, occupied_cells, this, &mtx, &existing_features, &new_features](int row,
                                                                                                      int col) {
            int n_to_detect = _n_per_cell - occupied_cells[col][row];
            if (n_to_detect < 1)
                return;

            int x = col * cell_size;
            int y = row * cell_size;

            cv::Rect roi(x, y, cell_size, cell_size);

            std::vector<std::shared_ptr<AFeature>> features_local;

            this->customDetectAndCompute(img(roi), updated_mask(roi), features_local);

            for (auto f : features_local) {
                std::vector<Eigen::Vector2d> pts;
                if (/*(f->getFeatureLabel() == "edgeletxd")||*/
                    (f->getFeatureLabel() == "pointxd")) {
                    pts = f->getPoints();
                    pts.at(0) += Eigen::Vector2d(x, y);
                } else // multiple points features (pattern)
                {
                    pts = f->getPoints();
                    for (uint i = 0; i < pts.size(); ++i)
                        pts.at(i) += Eigen::Vector2d(x, y);
                }
                f->setPoints(pts);

                mtx.lock();
                new_features.push_back(f);
                mtx.unlock();
            }
        };
    
    // If there is not grid, detection on the full image
    if (_n_per_cell == _n_total) {
        n_rows = 0;
        n_cols = 0;

        this->customDetectAndCompute(img, updated_mask, new_features);
    }

    std::vector<std::thread> threads;
    for (int r = 0; r < n_rows; r++) {
        for (int c = 0; c < n_cols; c++) {
            threads.push_back(std::thread(detectComputeSmall, r, c));
        }
    }
    for (auto &th : threads) {
        th.join();
    }

    return new_features;
}

} // namespace isae
