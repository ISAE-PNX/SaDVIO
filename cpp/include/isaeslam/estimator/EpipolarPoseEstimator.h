#ifndef EPIPOLARPOSEESTIMATOR_H
#define EPIPOLARPOSEESTIMATOR_H

#include "isaeslam/estimator/APoseEstimator.h"


namespace isae {

class EpipolarPoseEstimator : public APoseEstimator {
  public:
    bool estimateTransformBetween(const std::shared_ptr<Frame> &frame1,
                                  const std::shared_ptr<Frame> &frame2,
                                  vec_match &matches,
                                  Eigen::Affine3d &dT,
                                  Eigen::MatrixXd &covdT) override;
    bool estimateTransformBetween(const std::shared_ptr<Frame> &frame1,
                                  const std::shared_ptr<Frame> &frame2,
                                  typed_vec_match &typed_matches,
                                  Eigen::Affine3d &dT,
                                  Eigen::MatrixXd &covdT) override;
    bool estimateTransformSensors(const std::shared_ptr<ImageSensor> &sensor1,
                                  const std::shared_ptr<ImageSensor> &sensor2,
                                  vec_match &matches,
                                  Eigen::Affine3d &dT,
                                  Eigen::MatrixXd &covdT);
};

} // namespace isae

#endif // EPIPOLARPOSEESTIMATOR_H