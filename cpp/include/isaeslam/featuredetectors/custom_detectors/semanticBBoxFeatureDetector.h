#ifndef SEMANTICBBOXFEATUREDETECTOR_H
#define SEMANTICBBOXFEATUREDETECTOR_H

#include "isaeslam/featuredetectors/aCustomFeatureDetector.h"

namespace isae {

class semanticBBoxFeatureDetector : public ACustomFeatureDetector {
  public:
    semanticBBoxFeatureDetector(int n, int n_per_cell) : ACustomFeatureDetector(n, n_per_cell) { this->init(); }

    void customDetectAndCompute(const cv::Mat &img,
                                const cv::Mat &mask,
                                std::vector<std::shared_ptr<AFeature>> &features) override;
    void computeDescriptor(const cv::Mat &img, std::vector<std::shared_ptr<AFeature>> &features) override;

    void init() override;

    double getDist(const cv::Mat &desc1, const cv::Mat &desc2) const override;

  private:
};

} // namespace isae

#endif // SEMANTICBBOXFEATUREDETECTOR_H
