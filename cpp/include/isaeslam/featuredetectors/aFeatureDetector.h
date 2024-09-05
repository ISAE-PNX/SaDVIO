#ifndef AFEATUREDETECTOR_H
#define AFEATUREDETECTOR_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "isaeslam/data/features/Edgelet2D.h"
#include "isaeslam/data/features/Line2D.h"
#include "isaeslam/data/features/Point2D.h"
#include "isaeslam/data/sensors/ASensor.h"
#include "isaeslam/typedefs.h"
#include "utilities/geometry.h"

namespace isae {

class Point2D;

class AFeatureDetector {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    AFeatureDetector() {}
    AFeatureDetector(int n, int n_per_cell)
        : _n_total(n), _n_per_cell(n_per_cell) {}
    ~AFeatureDetector() {}

    virtual void init()                                                                                  = 0;
    virtual void detectAndCompute(const cv::Mat &img,
                                  const cv::Mat &mask,
                                  std::vector<cv::KeyPoint> &keypoints,
                                  cv::Mat &descriptors,
                                  int n_points = 0)                                                      = 0;
    virtual void computeDescriptor(const cv::Mat &img, std::vector<std::shared_ptr<AFeature>> &features) = 0;

    virtual std::vector<std::shared_ptr<AFeature>> detectAndComputeGrid(
        const cv::Mat &img,
        const cv::Mat &mask,
        std::vector<std::shared_ptr<AFeature>> existing_features = std::vector<std::shared_ptr<AFeature>>()) = 0;

    size_t getNbDesiredFeatures() { return _n_total; }
    virtual double getDist(const cv::Mat &desc1, const cv::Mat &desc2) const = 0;
    double getMaxMatchingDist() const { return _max_matching_dist; }

    bool getFeaturesInBox(int x,
                          int y,
                          int w,
                          int h,
                          const std::vector<std::shared_ptr<AFeature>> &features,
                          std::vector<std::shared_ptr<AFeature>> &features_in_box) const;

    void deleteUndescribedFeatures(std::vector<std::shared_ptr<AFeature>> &features);

    static void KeypointToFeature(std::vector<cv::KeyPoint> keypoints,
                                  cv::Mat descriptors,
                                  std::vector<std::shared_ptr<AFeature>> &features,
                                  const std::string &featurelabel = "pointxd");

    static void FeatureToKeypoint(std::vector<std::shared_ptr<AFeature>> features,
                                  std::vector<cv::KeyPoint> &keypoints,
                                  cv::Mat &descriptors);

    static void FeatureToP2f(std::vector<std::shared_ptr<AFeature>> features, std::vector<cv::Point2f> &p2fs);

  protected:
    int _n_total;              // the maximum amount of features the detector should find for any given image
    int _n_per_cell;           // the number of features per cell
    double _max_matching_dist; // distance threshold for matching
};

} // namespace isae

#endif // AFEATUREDETECTOR_H
