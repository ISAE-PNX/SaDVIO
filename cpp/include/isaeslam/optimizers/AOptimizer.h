#ifndef AOPTIMIZER_H
#define AOPTIMIZER_H

#include "isaeslam/data/frame.h"
#include "isaeslam/data/maps/localmap.h"
#include "isaeslam/optimizers/marginalization.hpp"
#include "isaeslam/optimizers/parametersBlock.hpp"
#include "isaeslam/optimizers/residuals.hpp"
#include <ceres/ceres.h>

namespace isae {

class AOptimizer {
  public:
    AOptimizer() {

        // Init marginalization variables
        _marginalization      = std::make_shared<Marginalization>();
        _marginalization_last = std::make_shared<Marginalization>();
    };

    bool landmarkOptimization(std::shared_ptr<Frame> &frame);

    bool singleFrameOptimization(std::shared_ptr<Frame> &moving_frame);

    bool singleFrameVIOptimization(std::shared_ptr<isae::Frame> &moving_frame);

    bool localMapBA(std::shared_ptr<isae::LocalMap> &local_map, const size_t fixed_frame_number = 0);

    bool localMapVIOptimization(std::shared_ptr<isae::LocalMap> &local_map, const size_t fixed_frame_number = 0);

    bool VIInit(std::shared_ptr<isae::LocalMap> &local_map, Eigen::Matrix3d &R_w_i, bool optim_scale = false);

    virtual bool landmarkOptimizationNoFov(std::shared_ptr<Frame> &f,
                                           std::shared_ptr<Frame> &fp,
                                           Eigen::Affine3d &T_cam0_cam0p,
                                           double info_scale);

    virtual bool marginalize(std::shared_ptr<Frame> &frame0, std::shared_ptr<Frame> &frame1, bool enable_sparsif) {
        return true;
    };

  protected:
    bool isMovingFrame(const std::shared_ptr<isae::Frame> &frame,
                       const std::vector<std::shared_ptr<isae::Frame>> &frame_vector);
    bool isMovingLandmark(const std::shared_ptr<isae::ALandmark> &ldmk,
                          const std::vector<std::shared_ptr<isae::ALandmark>> &cloud_to_optimize);

    virtual uint addResidualsLocalMap(ceres::Problem &problem,
                                      ceres::LossFunction *loss_function,
                                      ceres::ParameterBlockOrdering *ordering,
                                      std::vector<std::shared_ptr<Frame>> &frame_vector,
                                      size_t fixed_frame_number,
                                      std::shared_ptr<isae::LocalMap> &local_map) = 0;

    virtual uint addLandmarkResiduals(ceres::Problem &problem,
                                      ceres::LossFunction *loss_function,
                                      typed_vec_landmarks &cloud_to_optimize) = 0;

    virtual uint addSingleFrameResiduals(ceres::Problem &problem,
                                         ceres::LossFunction *loss_function,
                                         std::shared_ptr<Frame> &frame,
                                         typed_vec_landmarks &cloud_to_optimize) = 0;

    virtual uint addMarginalizationResiduals(ceres::Problem &problem,
                                             ceres::LossFunction *loss_function,
                                             ceres::ParameterBlockOrdering *ordering) {
        return 0;
    }

    uint addIMUResiduals(ceres::Problem &problem,
                         ceres::LossFunction *loss_function,
                         ceres::ParameterBlockOrdering *ordering,
                         std::vector<std::shared_ptr<Frame>> &frame_vector,
                         size_t fixed_frame_number);

    std::unordered_map<std::shared_ptr<Frame>, PoseParametersBlock> _map_frame_posepar;
    std::unordered_map<std::shared_ptr<Frame>, PointXYZParametersBlock> _map_frame_velpar;
    std::unordered_map<std::shared_ptr<Frame>, PointXYZParametersBlock> _map_frame_dbapar;
    std::unordered_map<std::shared_ptr<Frame>, PointXYZParametersBlock> _map_frame_dbgpar;
    std::unordered_map<std::shared_ptr<ALandmark>, PointXYZParametersBlock> _map_lmk_ptpar;
    std::unordered_map<std::shared_ptr<ALandmark>, PoseParametersBlock> _map_lmk_posepar;

    bool _enable_sparsif = false;
    std::shared_ptr<Marginalization> _marginalization;
    std::shared_ptr<Marginalization> _marginalization_last;
};

} // namespace isae

#endif // AOPTIMIZER_H
