#ifndef APOSEESTIMATOR_H
#define APOSEESTIMATOR_H

#include "isaeslam/data/frame.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

namespace isae {

class APoseEstimatorConfig {
  public:
    // whatever parameters the estimator needs
};

/*!
 * @brief Abstract class for relative pose estimation between two frames.
 *
 * This class defines the interface for estimating the transformation between two frames, either using matches or typed
 * matches. It may also return the covariance of the estimated transformation.
 */
class APoseEstimator {
  public:
    virtual bool estimateTransformBetween(const std::shared_ptr<Frame> &frame1,
                                          const std::shared_ptr<Frame> &frame2,
                                          vec_match &matches,
                                          Eigen::Affine3d &dT,
                                          Eigen::MatrixXd &covdT) = 0;
    virtual bool estimateTransformBetween(const std::shared_ptr<Frame> &frame1,
                                          const std::shared_ptr<Frame> &frame2,
                                          typed_vec_match &typed_matches,
                                          Eigen::Affine3d &dT,
                                          Eigen::MatrixXd &covdT) = 0;
};

} // namespace isae

#endif // APOSEESTIMATOR_H