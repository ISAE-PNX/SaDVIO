#ifndef EPIPOLARPOSEESTIMATORCUSTOM_H
#define EPIPOLARPOSEESTIMATORCUSTOM_H

#include "isaeslam/estimator/APoseEstimator.h"

namespace isae {

class EpipolarPoseEstimatorCustom : public APoseEstimator {
  public:
    bool estimateTransformBetween(const std::shared_ptr<Frame> &frame1, const std::shared_ptr<Frame> &frame2,
                                  vec_match &matches, Eigen::Affine3d &dT, Eigen::MatrixXd &covdT) override;
    bool estimateTransformBetween(const std::shared_ptr<Frame> &frame1, const std::shared_ptr<Frame> &frame2,
                                  typed_vec_match &typed_matches, Eigen::Affine3d &dT, Eigen::MatrixXd &covdT) override;
};

} // namespace isae

#endif // EPIPOLARPOSEESTIMATORCUSTOM_H