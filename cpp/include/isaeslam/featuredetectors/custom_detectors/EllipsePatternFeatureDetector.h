#ifndef ELLIPSEPATTERNFEATUREDETECTOR_H
#define ELLIPSEPATTERNFEATUREDETECTOR_H

#include "isaeslam/data/features/EllipsePattern2D.h"
#include "isaeslam/featuredetectors/aCustomFeatureDetector.h"
#include "isaeslam/featuredetectors/custom_detectors/extractor/ellipsepattern/EllipsePatternExtractor.h"

namespace isae {

class EllipsePatternFeatureDetector : public ACustomFeatureDetector {
  public:
    EllipsePatternFeatureDetector(int n, int n_per_cell) : ACustomFeatureDetector(n, n_per_cell) { this->init(); }

    void customDetectAndCompute(const cv::Mat &img,
                                const cv::Mat &mask,
                                std::vector<std::shared_ptr<AFeature>> &features) override;
    void computeDescriptor(const cv::Mat &img, std::vector<std::shared_ptr<AFeature>> &features);

    void init() override;
    double getDist(const cv::Mat &desc1, const cv::Mat &desc2) const override;

  private:
    EllipsePatternExtractor *EllipseExtractor = new EllipsePatternExtractor();
};

} // namespace isae

#endif // ELLIPSEPATTERNFEATUREDETECTOR_H
