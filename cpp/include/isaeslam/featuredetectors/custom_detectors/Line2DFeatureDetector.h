#ifndef LINE2DFEATUREDETECTOR_H
#define LINE2DFEATUREDETECTOR_H

#include "../thirdparty/ELSED/src/ELSED.h"
#include "isaeslam/featuredetectors/aCustomFeatureDetector.h"
#include <opencv2/features2d.hpp>
#include <opencv2/line_descriptor.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/ximgproc.hpp>

namespace isae {

class Line2DFeatureDetector : public ACustomFeatureDetector {
  public:
    Line2DFeatureDetector(int n, int n_per_cell, double max_matching_dist = 25)
        : ACustomFeatureDetector(n, n_per_cell) {
        this->init();
        _max_matching_dist = max_matching_dist;
    };

    void customDetectAndCompute(const cv::Mat &img,
                                const cv::Mat &mask,
                                std::vector<std::shared_ptr<AFeature>> &features) override;

    void computeDescriptor(const cv::Mat &img, std::vector<std::shared_ptr<AFeature>> &features) override;

    void init() override;

    double getDist(const cv::Mat &desc1, const cv::Mat &desc2) const override;

  private:
    cv::line_descriptor::KeyLine MakeKeyLine(cv::Point2f start_pts, cv::Point2f end_pts, size_t cols);

    void KeyLineToFeature(std::vector<cv::line_descriptor::KeyLine> &keyLine,
                          cv::Mat &descriptors,
                          std::vector<std::shared_ptr<AFeature>> &localfeatures,
                          const std::string &featuretype);

    void FeatureToKeyLine(std::vector<std::shared_ptr<AFeature>> &localfeatures,
                          std::vector<cv::line_descriptor::KeyLine> &keyLine,
                          cv::Mat &descriptors,
                          uint nb_cols);

    bool getFeaturesInBox(int x,
                          int y,
                          int w,
                          int h,
                          const std::vector<std::shared_ptr<AFeature>> &features,
                          std::vector<std::shared_ptr<AFeature>> &features_in_box) const;

    std::shared_ptr<upm::ELSED> _elsed;
    upm::ELSEDParams _params;

    cv::Ptr<cv::line_descriptor::LSDDetector> _detector;
    cv::Ptr<cv::line_descriptor::BinaryDescriptor> _descriptor;

    // Sort features
    struct Comp {
        Comp(const std::vector<double> &v) : _v(v) {}

        bool operator()(double a, double b) { return _v[a] > _v[b]; }
        const std::vector<double> &_v;
    };
};

} // namespace isae

#endif // EDGELETFEATUREDETECTOR_H
