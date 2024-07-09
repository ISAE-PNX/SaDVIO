#ifndef EDGELETFEATUREDETECTOR_H
#define EDGELETFEATUREDETECTOR_H

#include "isaeslam/featuredetectors/aCustomFeatureDetector.h"
#include <opencv2/line_descriptor.hpp>

namespace isae {

class EdgeletFeatureDetector : public ACustomFeatureDetector {
  public:
    EdgeletFeatureDetector(int n, int n_per_cell, double max_matching_dist = 250)
        : ACustomFeatureDetector(n, n_per_cell) {
        this->init();
        _max_matching_dist = max_matching_dist;
    }

    void customDetectAndCompute(const cv::Mat &img,
                                const cv::Mat &mask,
                                std::vector<std::shared_ptr<AFeature>> &features) override;
    void computeDescriptor(const cv::Mat &img, std::vector<std::shared_ptr<AFeature>> &features) override;

    void computeLineDescriptor(const cv::Mat &img,
                               const std::vector<cv::KeyPoint> &kps,
                               const std::vector<cv::Point2f> &orientations,
                               cv::Mat &descriptors);

    void init() override;
    double getDist(const cv::Mat &desc1, const cv::Mat &desc2) const override;

  private:
    double _edge_threshold = 40;
    uint _cell_size        = 7;
    uint _half_cell_size   = 3;
    cv::Ptr<cv::line_descriptor::BinaryDescriptor> _descriptor;
    // cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> _descriptor;
    void detect(const cv::Mat &img,
                std::vector<cv::KeyPoint> &kps,
                std::vector<cv::Point2f> &orientations,
                std::vector<double> &scores);

    // Sort features
    struct Comp {
        Comp(const std::vector<double> &v) : _v(v) {}

        bool operator()(double a, double b) { return _v[a] > _v[b]; }
        const std::vector<double> &_v;
    };

    cv::Point2f getOrientation(const cv::Mat &img, const cv::Mat &gradx, const cv::Mat &grady, double &score);
};

} // namespace isae

#endif // EDGELETFEATUREDETECTOR_H
