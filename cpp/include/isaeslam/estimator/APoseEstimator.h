#ifndef APOSEESTIMATOR_H
#define APOSEESTIMATOR_H

#include "isaeslam/data/frame.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

namespace isae {

class APoseEstimator {
  public:
    virtual bool estimateTransformBetween(const std::shared_ptr<Frame> &frame1, const std::shared_ptr<Frame> &frame2,
                                          vec_match &matches, Eigen::Affine3d &dT, Eigen::MatrixXd &covdT) = 0;
    virtual bool estimateTransformBetween(const std::shared_ptr<Frame> &frame1, const std::shared_ptr<Frame> &frame2,
                                          typed_vec_match &typed_matches, Eigen::Affine3d &dT,
                                          Eigen::MatrixXd &covdT)                                          = 0;
};

} // namespace isae

#endif // APOSEESTIMATOR_H