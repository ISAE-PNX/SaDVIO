#ifndef AOPTIMIZER_H
#define AOPTIMIZER_H

#include "isaeslam/data/frame.h"
#include "isaeslam/data/maps/localmap.h"
#include "isaeslam/optimizers/marginalization.hpp"
#include "isaeslam/optimizers/parametersBlock.hpp"
#include "isaeslam/optimizers/residuals.hpp"
#include <ceres/ceres.h>

namespace isae {

/*!
 * \brief Abstract class for optimization pipeline based on CERES.
 *
 * This class provides methods for optimizing frames, landmarks, and local maps.
 * It also implements methods involving factors such as initialization and marginalization.
 * This virtual formulation allows for different optimization scheme with different cost functions, different
 * parameterization or different CERES settings (e.g. numerical or analytic Jacobians).
 */
class AOptimizer {
  public:
    AOptimizer() {
        // Init marginalization variables
        _marginalization      = std::make_shared<Marginalization>();
        _marginalization_last = std::make_shared<Marginalization>();
    };

    void resetMarginalization() {
        _marginalization      = std::make_shared<Marginalization>();
        _marginalization_last = std::make_shared<Marginalization>();
    };

    /*!
     * @brief Structure only Bundle Adjustment for a frame.
     * @param frame The frame to optimize.
     *
     * This method optimize the landmarks associated to the frame, setting static the pose parameters.
     * It can be used to refine the triangulation of new landmarks.
     */
    bool landmarkOptimization(std::shared_ptr<Frame> &frame);

    /*!
     * @brief Motion only Bundle Adjustment for a frame.
     * @param moving_frame The frame to optimize.
     *
     * This method optimizes the frame's pose parameters only, setting static the landmarks.
     */
    bool singleFrameOptimization(std::shared_ptr<Frame> &moving_frame);

    /*!
     * @brief Visual-Inertial Motion only Bundle Adjustment for a frame.
     * @param moving_frame The frame to optimize.
     *
     * This method optimizes the frame's pose and velocity parameters.
     */
    bool singleFrameVIOptimization(std::shared_ptr<isae::Frame> &moving_frame);

    /*!
     * @brief Bundle Adjustment for a local map.
     * @param local_map The local map to optimize.
     * @param fixed_frame_number The frame number to fix during optimization (default is 0).
     */
    bool localMapBA(std::shared_ptr<isae::LocalMap> &local_map, const size_t fixed_frame_number = 0);

    /*!
     * @brief Visual-Inertial Bundle Adjustment for a local map.
     * @param local_map The local map to optimize.
     * @param fixed_frame_number The frame number to fix during optimization (default is 0).
     */
    bool localMapVIOptimization(std::shared_ptr<isae::LocalMap> &local_map, const size_t fixed_frame_number = 0);

    /*!
     * @brief Visual-Inertial Bundle Adjustment for a local map with time delay.
     * @param local_map The local map to optimize.
     * @param fixed_frame_number The frame number to fix during optimization (default is 0).
     * @return The time delay used for optimization.
     */
    virtual bool localMapVIOptimizationTd(std::shared_ptr<isae::LocalMap> &local_map,
                                          double &td,
                                          const size_t fixed_frame_number = 0) {
        return true;
    }
    /*!
     * @brief Visual-Inertial Initialization for a local map.
     * @param local_map The local map to initialize.
     * @param R_w_i The initial rotation of the world frame with respect to the inertial frame.
     * @param optim_scale Whether to optimize the scale (default is false).
     * @return The scale factor used for initialization.
     */
    double VIInit(std::shared_ptr<isae::LocalMap> &local_map, Eigen::Matrix3d &R_w_i, bool optim_scale = false);

    /*!
     * @brief Structure only Bundle Adjustment for a frame with non overlapping fields of view sensors.
     * @param f The frame to optimize.
     * @param fp The previous frame to optimize.
     * @param T_cam0_cam0p The transformation from the current frame to the previous frame.
     * @param info_scale The scale factor for the information matrix.
     */
    virtual bool landmarkOptimizationNoFov(std::shared_ptr<Frame> &f,
                                           std::shared_ptr<Frame> &fp,
                                           Eigen::Affine3d &T_cam0_cam0p,
                                           double info_scale);

    /*!
     * @brief Marginalization of a frame and its landmarks.
     * @param frame0 The frame to marginalize.
     * @param frame1 The frame to keep connected to frame0.
     * @param enable_sparsif Whether to enable sparsification (default is false).
     *
     * Perform all the marginalization steps for a given frame. A marginalization object deals with all these steps
     */
    virtual bool marginalize(std::shared_ptr<Frame> &frame0, std::shared_ptr<Frame> &frame1, bool enable_sparsif) {
        return true;
    };

    /*!
     * @brief Marginalization of all the landmarks linked to two frame and computation of the information matrix of the
     * relative pose.
     * @param frame0 The first frame to marginalize.
     * @param frame1 The second frame to marginalize.
     * @return The information matrix of the relative pose between frame0 and frame1.
     *
     * This method computes the information matrix of the relative pose between two frames and marginalizes the
     * landmarks linked to them. It uses NFR to extract a relative pose factor and to derive its information matrix.
     */
    virtual Eigen::MatrixXd marginalizeRelative(std::shared_ptr<Frame> &frame0, std::shared_ptr<Frame> &frame1) {
        return Eigen::MatrixXd::Identity(6, 6);
    }

  protected:
    /*!
     * @brief Add visual residuals to a CERES problem for a local map.
     * @param problem The CERES problem to which the residuals will be added.
     * @param loss_function The loss function to be used for the residuals.
     * @param ordering The parameter block ordering for the problem.
     * @param frame_vector The vector of frames in the local map.
     * @param fixed_frame_number The number of frames to fix during optimization.
     * @param local_map The local map to which the residuals will be added.
     * @return The number of residuals added.
     */
    virtual uint addResidualsLocalMap(ceres::Problem &problem,
                                      ceres::LossFunction *loss_function,
                                      ceres::ParameterBlockOrdering *ordering,
                                      std::vector<std::shared_ptr<Frame>> &frame_vector,
                                      size_t fixed_frame_number,
                                      std::shared_ptr<isae::LocalMap> &local_map) = 0;

    /*!
     * @brief Add visual residuals to a CERES problem for structure only BA.
     * @param problem The CERES problem to which the residuals will be added.
     * @param loss_function The loss function to be used for the residuals.
     * @param cloud_to_optimize The landmarks to be optimized.
     * @return The number of residuals added.
     */
    virtual uint addLandmarkResiduals(ceres::Problem &problem,
                                      ceres::LossFunction *loss_function,
                                      typed_vec_landmarks &cloud_to_optimize) = 0;

    /*!
     * @brief Add visual residuals to a CERES problem for a motion only BA.
     * @param problem The CERES problem to which the residuals will be added.
     * @param loss_function The loss function to be used for the residuals.
     * @param frame The frame to which the residuals will be added.
     * @param cloud_to_optimize The landmarks belonging to the frame.
     * @return The number of residuals added.
     */
    virtual uint addSingleFrameResiduals(ceres::Problem &problem,
                                         ceres::LossFunction *loss_function,
                                         std::shared_ptr<Frame> &frame,
                                         typed_vec_landmarks &cloud_to_optimize) = 0;

    /*!
     * @brief Add marginalization residuals to a CERES problem.
     * @param problem The CERES problem to which the residuals will be added.
     * @param loss_function The loss function to be used for the residuals.
     * @param ordering The parameter block ordering for the problem.
     * @return The number of marginalization residuals added.
     */
    virtual uint addMarginalizationResiduals(ceres::Problem &problem,
                                             ceres::LossFunction *loss_function,
                                             ceres::ParameterBlockOrdering *ordering) {
        return 0;
    }

    /*!
     * @brief Add IMU residuals to a CERES problem.
     * @param problem The CERES problem to which the residuals will be added.
     * @param loss_function The loss function to be used for the residuals.
     * @param ordering The parameter block ordering for the problem.
     * @param frame_vector The vector of frames in the local map.
     * @param fixed_frame_number The number of frames to fix during optimization.
     */
    uint addIMUResiduals(ceres::Problem &problem,
                         ceres::LossFunction *loss_function,
                         ceres::ParameterBlockOrdering *ordering,
                         std::vector<std::shared_ptr<Frame>> &frame_vector,
                         size_t fixed_frame_number);

    std::unordered_map<std::shared_ptr<Frame>, PoseParametersBlock> _map_frame_posepar; //!< map for pose parameters
    std::unordered_map<std::shared_ptr<Frame>, PointXYZParametersBlock>
        _map_frame_velpar; //!< map for velocity parameters
    std::unordered_map<std::shared_ptr<Frame>, PointXYZParametersBlock>
        _map_frame_dbapar; //!< map for accelerometer bias parameters
    std::unordered_map<std::shared_ptr<Frame>, PointXYZParametersBlock>
        _map_frame_dbgpar; //!< map for gyroscope bias parameters
    std::unordered_map<std::shared_ptr<ALandmark>, PointXYZParametersBlock>
        _map_lmk_ptpar; //!< map for landmark point parameters
    std::unordered_map<std::shared_ptr<ALandmark>, PoseParametersBlock>
        _map_lmk_posepar; //!< map for landmark pose parameters

    bool _enable_sparsif = false;                           //!< enable sparsification of the marginalization
    std::shared_ptr<Marginalization> _marginalization;      //!< marginalization object
    std::shared_ptr<Marginalization> _marginalization_last; //!< marginalization object of the last optimization
};

} // namespace isae

#endif // AOPTIMIZER_H
