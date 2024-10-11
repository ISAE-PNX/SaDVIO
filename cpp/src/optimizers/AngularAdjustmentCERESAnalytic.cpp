#include "isaeslam/optimizers/AngularAdjustmentCERESAnalytic.h"
#include "utilities/timer.h"

namespace isae {

uint AngularAdjustmentCERESAnalytic::addSingleFrameResiduals(ceres::Problem &problem,
                                                             ceres::LossFunction *loss_function,
                                                             std::shared_ptr<Frame> &frame,
                                                             typed_vec_landmarks &cloud_to_optimize) {

    _map_frame_posepar.emplace(frame, PoseParametersBlock(Eigen::Affine3d::Identity()));
    problem.AddParameterBlock(_map_frame_posepar.at(frame).values(), 6);

    for (auto &ldmk_list : cloud_to_optimize) {

        // Deal with pointxd landmarks
        if (ldmk_list.first == "pointxd") {
            // For all landmark
            for (auto &landmark : ldmk_list.second) {

                if (!landmark->isInitialized() || landmark->isOutlier())
                    continue;

                // For all feature
                std::vector<std::weak_ptr<AFeature>> featuresAssociatedLandmarks = landmark->getFeatures();

                for (std::weak_ptr<AFeature> &wfeature : featuresAssociatedLandmarks) {
                    std::shared_ptr<AFeature> feature = wfeature.lock();

                    if (!feature) {
                        continue;
                    }

                    std::shared_ptr<ImageSensor> cam  = feature->getSensor();
                    std::shared_ptr<Frame> feat_frame = cam->getFrame();

                    if (frame == feat_frame) {

                        if (_map_lmk_ptpar.find(landmark) == _map_lmk_ptpar.end()) {
                            _map_lmk_ptpar.emplace(landmark, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
                            problem.AddParameterBlock(_map_lmk_ptpar.at(landmark).values(), 3);
                            problem.SetParameterBlockConstant(_map_lmk_ptpar.at(landmark).values());
                        }

                        ceres::CostFunction *cost_fct =
                            new AngularErrCeres_pointxd_dx(feature->getBearingVectors().at(0),
                                                           cam->getFrame2SensorTransform(),
                                                           frame->getWorld2FrameTransform(),
                                                           landmark->getPose().translation(),
                                                           1.0 / cam->getFocal());

                        problem.AddResidualBlock(cost_fct,
                                                 loss_function,
                                                 _map_frame_posepar.at(frame).values(),
                                                 _map_lmk_ptpar.at(landmark).values());
                    }
                }
            }
        }

        // Deal with linexd landmarks
        if (ldmk_list.first == "linexd") {
            // For all landmark
            for (auto &landmark : ldmk_list.second) {

                if (!landmark->isInitialized() || landmark->isOutlier())
                    continue;

                // For all feature
                std::vector<std::weak_ptr<AFeature>> featuresAssociatedLandmarks = landmark->getFeatures();

                for (std::weak_ptr<AFeature> &wfeature : featuresAssociatedLandmarks) {
                    std::shared_ptr<AFeature> feature = wfeature.lock();

                    if (!feature) {
                        continue;
                    }

                    std::shared_ptr<ImageSensor> cam  = feature->getSensor();
                    std::shared_ptr<Frame> feat_frame = cam->getFrame();

                    if (frame == feat_frame) {

                        if (_map_lmk_ptpar.find(landmark) == _map_lmk_ptpar.end()) {
                            _map_lmk_posepar.emplace(landmark, PoseParametersBlock(Eigen::Affine3d::Identity()));
                            problem.AddParameterBlock(_map_lmk_posepar.at(landmark).values(), 6);
                            problem.SetParameterBlockConstant(_map_lmk_posepar.at(landmark).values());
                        }

                        ceres::CostFunction *cost_fct = new AngularErrCeres_linexd_dx(feature->getBearingVectors(),
                                                                                      cam->getFrame2SensorTransform(),
                                                                                      frame->getWorld2FrameTransform(),
                                                                                      landmark->getPose(),
                                                                                      1.0);

                        problem.AddResidualBlock(cost_fct,
                                                 loss_function,
                                                 _map_frame_posepar.at(frame).values(),
                                                 _map_lmk_posepar.at(landmark).values());
                    }
                }
            }
        }
    }

    return 0;
}

uint AngularAdjustmentCERESAnalytic::addLandmarkResiduals(ceres::Problem &problem,
                                                          ceres::LossFunction *loss_function,
                                                          typed_vec_landmarks &cloud_to_optimize) {

    // ceres::Manifold *nullptr = new SE3RightParameterization();

    // For all the landmarks
    for (auto &ldmk_list : cloud_to_optimize) {
        // Deal with pointxd landmarks
        if (ldmk_list.first == "pointxd") {
            // For all landmark
            for (auto &landmark : ldmk_list.second) {

                if (!landmark->isInitialized() || landmark->isOutlier())
                    continue;

                // Add parameter block for each landmark
                _map_lmk_ptpar.emplace(landmark, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
                problem.AddParameterBlock(_map_lmk_ptpar.at(landmark).values(), 3);

                // For all feature
                std::vector<std::weak_ptr<AFeature>> featuresAssociatedLandmarks = landmark->getFeatures();

                for (std::weak_ptr<AFeature> &wfeature : featuresAssociatedLandmarks) {
                    std::shared_ptr<AFeature> feature = wfeature.lock();
                    std::shared_ptr<ImageSensor> cam  = feature->getSensor();
                    std::shared_ptr<Frame> frame      = cam->getFrame();

                    // Check the consistency of the frame
                    if (!feature || !frame->isKeyFrame()) {
                        continue;
                    }

                    if (_map_frame_posepar.find(frame) == _map_frame_posepar.end()) {
                        _map_frame_posepar.emplace(frame, PoseParametersBlock(Eigen::Affine3d::Identity()));
                        problem.AddParameterBlock(_map_frame_posepar.at(frame).values(), 6);
                        problem.SetParameterBlockConstant(_map_frame_posepar.at(frame).values());
                    }

                    ceres::CostFunction *cost_fct = new AngularErrCeres_pointxd_dx(feature->getBearingVectors().at(0),
                                                                                   cam->getFrame2SensorTransform(),
                                                                                   frame->getWorld2FrameTransform(),
                                                                                   landmark->getPose().translation(),
                                                                                   (1.5 / cam->getFocal()));

                    problem.AddResidualBlock(cost_fct,
                                             loss_function,
                                             _map_frame_posepar.at(frame).values(),
                                             _map_lmk_ptpar.at(landmark).values());
                }
            }
        }

        // Deal with linexd landmarks
        if (ldmk_list.first == "linexd") {

            // For all landmark
            for (auto &landmark : ldmk_list.second) {

                if (!landmark->isInitialized() || landmark->isOutlier())
                    continue;

                // Add parameter block for each landmark
                _map_lmk_posepar.emplace(landmark, PoseParametersBlock(Eigen::Affine3d::Identity()));
                problem.AddParameterBlock(_map_lmk_posepar.at(landmark).values(), 6);

                // For all feature
                std::vector<std::weak_ptr<AFeature>> featuresAssociatedLandmarks = landmark->getFeatures();

                for (std::weak_ptr<AFeature> &wfeature : featuresAssociatedLandmarks) {
                    std::shared_ptr<AFeature> feature = wfeature.lock();
                    std::shared_ptr<ImageSensor> cam  = feature->getSensor();
                    std::shared_ptr<Frame> frame      = cam->getFrame();

                    // Check the consistency of the frame
                    if (!feature || !frame->isKeyFrame()) {
                        continue;
                    }

                    if (_map_frame_posepar.find(frame) == _map_frame_posepar.end()) {
                        _map_frame_posepar.emplace(frame, PoseParametersBlock(Eigen::Affine3d::Identity()));
                        problem.AddParameterBlock(_map_frame_posepar.at(frame).values(), 6);
                        problem.SetParameterBlockConstant(_map_frame_posepar.at(frame).values());
                    }

                    ceres::CostFunction *cost_fct = new AngularErrCeres_linexd_dx(feature->getBearingVectors(),
                                                                                  cam->getFrame2SensorTransform(),
                                                                                  frame->getWorld2FrameTransform(),
                                                                                  landmark->getPose(),
                                                                                  1);

                    problem.AddResidualBlock(cost_fct,
                                             loss_function,
                                             _map_frame_posepar.at(frame).values(),
                                             _map_lmk_posepar.at(landmark).values());
                }
            }
        }
    }

    return 0;
}

uint AngularAdjustmentCERESAnalytic::addResidualsLocalMap(ceres::Problem &problem,
                                                          ceres::LossFunction *loss_function,
                                                          ceres::ParameterBlockOrdering *ordering,
                                                          std::vector<std::shared_ptr<Frame>> &frame_vector,
                                                          size_t fixed_frame_number,
                                                          std::shared_ptr<isae::LocalMap> &local_map) {

    uint nb_residuals = 0;

    // Add parameter block and ordering for each frame parameter
    for (size_t i = 0; i < frame_vector.size(); i++) {
        _map_frame_posepar.emplace(frame_vector.at(i), PoseParametersBlock(Eigen::Affine3d::Identity()));
    }

    for (size_t i = 0; i < frame_vector.size(); i++) {

        std::shared_ptr<Frame> frame = frame_vector.at(i);

        problem.AddParameterBlock(_map_frame_posepar.at(frame).values(), 6);
        ordering->AddElementToGroup(_map_frame_posepar.at(frame).values(), 1);

        // Set parameter block constant for fixed frames
        if ((int)i > (int)(frame_vector.size() - fixed_frame_number - 1)) {
            problem.SetParameterBlockConstant(_map_frame_posepar.at(frame).values());
        }

        // Had a prior factor if it exists
        if (frame->hasPrior()) {
            ceres::CostFunction *cost_fct =
                new PosePriordx(frame->getWorld2FrameTransform(), frame->getPrior(), frame->getInfPrior().asDiagonal());
            problem.AddResidualBlock(cost_fct, loss_function, _map_frame_posepar.at(frame).values());
        }
    }

    // For all the landmarks
    for (auto &ldmk_list : local_map->getLandmarks()) {

        // Deal with pointxd landmarks
        if (ldmk_list.first == "pointxd") {
            // For all landmark
            for (auto &landmark : ldmk_list.second) {

                if (!landmark->isInitialized() || landmark->isOutlier())
                    continue;

                // Add parameter block for each landmark
                _map_lmk_ptpar.emplace(landmark, PointXYZParametersBlock(Eigen::Vector3d::Zero()));

                problem.AddParameterBlock(_map_lmk_ptpar.at(landmark).values(), 3);
                ordering->AddElementToGroup(_map_lmk_ptpar.at(landmark).values(), 0);

                // For all feature
                std::vector<std::weak_ptr<AFeature>> featuresAssociatedLandmarks = landmark->getFeatures();

                for (std::weak_ptr<AFeature> &wfeature : featuresAssociatedLandmarks) {
                    std::shared_ptr<AFeature> feature = wfeature.lock();
                    std::shared_ptr<ImageSensor> cam  = feature->getSensor();
                    std::shared_ptr<Frame> frame      = cam->getFrame();

                    // Check the consistency of the frame
                    if (!feature || !frame->isKeyFrame() ||
                        _map_frame_posepar.find(frame) == _map_frame_posepar.end()) {
                        continue;
                    }

                    nb_residuals++;

                    ceres::CostFunction *cost_fct = new AngularErrCeres_pointxd_dx(feature->getBearingVectors().at(0),
                                                                                   cam->getFrame2SensorTransform(),
                                                                                   frame->getWorld2FrameTransform(),
                                                                                   landmark->getPose().translation(),
                                                                                   (1.5 / cam->getFocal()));

                    problem.AddResidualBlock(cost_fct,
                                             loss_function,
                                             _map_frame_posepar.at(frame).values(),
                                             _map_lmk_ptpar.at(landmark).values());
                }
            }
        }

        // Deal with linexd landmarks
        if (ldmk_list.first == "linexd") {
            // For all landmark
            for (auto &landmark : ldmk_list.second) {

                if (!landmark->isInitialized() || landmark->isOutlier())
                    continue;

                // Add parameter block for each landmark
                _map_lmk_posepar.emplace(landmark, PoseParametersBlock(Eigen::Affine3d::Identity()));

                problem.AddParameterBlock(_map_lmk_posepar.at(landmark).values(), 6);
                ordering->AddElementToGroup(_map_lmk_posepar.at(landmark).values(), 0);

                // For all feature
                std::vector<std::weak_ptr<AFeature>> featuresAssociatedLandmarks = landmark->getFeatures();

                for (std::weak_ptr<AFeature> &wfeature : featuresAssociatedLandmarks) {
                    std::shared_ptr<AFeature> feature = wfeature.lock();
                    std::shared_ptr<ImageSensor> cam  = feature->getSensor();
                    std::shared_ptr<Frame> frame      = cam->getFrame();

                    // Check the consistency of the frame
                    if (!feature || !frame->isKeyFrame() ||
                        _map_frame_posepar.find(frame) == _map_frame_posepar.end()) {
                        continue;
                    }

                    nb_residuals++;

                    ceres::CostFunction *cost_fct = new AngularErrCeres_linexd_dx(feature->getBearingVectors(),
                                                                                  cam->getFrame2SensorTransform(),
                                                                                  frame->getWorld2FrameTransform(),
                                                                                  landmark->getPose(),
                                                                                  1);

                    problem.AddResidualBlock(cost_fct,
                                             loss_function,
                                             _map_frame_posepar.at(frame).values(),
                                             _map_lmk_posepar.at(landmark).values());
                }
            }
        }
    }

    return nb_residuals;
}

uint AngularAdjustmentCERESAnalytic::addMarginalizationResiduals(ceres::Problem &problem,
                                                                 ceres::LossFunction *loss_function,
                                                                 ceres::ParameterBlockOrdering *ordering) {

    // Add marginalization factor, dense case
    if (!_marginalization->_lmk_to_keep.empty() && !_enable_sparsif) {
        // Get parameter blocks for marginalization
        std::vector<double *> prior_parameter_blocks;

        // Add frame to keep blocks
        if (_marginalization->_frame_to_keep) {
            ordering->Remove(_map_frame_posepar.at(_marginalization->_frame_to_keep).values());
            prior_parameter_blocks.push_back(_map_frame_posepar.at(_marginalization->_frame_to_keep).values());
            ordering->AddElementToGroup(_map_frame_posepar.at(_marginalization->_frame_to_keep).values(), 2);

            ordering->Remove(_map_frame_velpar.at(_marginalization->_frame_to_keep).values());
            prior_parameter_blocks.push_back(_map_frame_velpar.at(_marginalization->_frame_to_keep).values());
            ordering->AddElementToGroup(_map_frame_velpar.at(_marginalization->_frame_to_keep).values(), 2);

            ordering->Remove(_map_frame_dbapar.at(_marginalization->_frame_to_keep).values());
            prior_parameter_blocks.push_back(_map_frame_dbapar.at(_marginalization->_frame_to_keep).values());
            ordering->AddElementToGroup(_map_frame_dbapar.at(_marginalization->_frame_to_keep).values(), 2);

            ordering->Remove(_map_frame_dbgpar.at(_marginalization->_frame_to_keep).values());
            prior_parameter_blocks.push_back(_map_frame_dbgpar.at(_marginalization->_frame_to_keep).values());
            ordering->AddElementToGroup(_map_frame_dbgpar.at(_marginalization->_frame_to_keep).values(), 2);
        }

        // Add lmk to keep blocks
        for (auto lmk : _marginalization->_lmk_to_keep["pointxd"]) {
            // TODO: fix this, it is supposed to be in the map
            if (_map_lmk_ptpar.find(lmk) == _map_lmk_ptpar.end()) {
                _map_lmk_ptpar.emplace(lmk, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
                problem.AddParameterBlock(_map_lmk_ptpar.at(lmk).values(), 3);
                ordering->AddElementToGroup(_map_lmk_ptpar.at(lmk).values(), 0);
            }
            ordering->Remove(_map_lmk_ptpar.at(lmk).values());
            prior_parameter_blocks.push_back(_map_lmk_ptpar.at(lmk).values());
            ordering->AddElementToGroup(_map_lmk_ptpar.at(lmk).values(), 2);
        }

        ceres::CostFunction *cost_fct = new MarginalizationFactor(_marginalization);
        problem.AddResidualBlock(cost_fct, loss_function, prior_parameter_blocks);
    }

    // Add marginalization factor, sparse case
    if (!_marginalization->_lmk_to_keep.empty() && _enable_sparsif) {

        /// CASE 1 VIO ///
        if (_marginalization->_frame_to_keep) {
            std::shared_ptr<Frame> frame_to_keep = _marginalization->_frame_to_keep;
            Eigen::Affine3d T_f_w                = frame_to_keep->getWorld2FrameTransform();
            Eigen::Vector3d v                    = frame_to_keep->getIMU()->getVelocity();
            Eigen::Vector3d ba                   = frame_to_keep->getIMU()->getBa();
            Eigen::Vector3d bg                   = frame_to_keep->getIMU()->getBg();
            ceres::CostFunction *cost_fct0 =
                new IMUPriordx(T_f_w, T_f_w, v, v, ba, ba, bg, bg, _marginalization->_map_frame_inf.at(frame_to_keep));
            problem.AddResidualBlock(cost_fct0,
                                     loss_function,
                                     _map_frame_posepar.at(frame_to_keep).values(),
                                     _map_frame_velpar.at(frame_to_keep).values(),
                                     _map_frame_dbapar.at(frame_to_keep).values(),
                                     _map_frame_dbgpar.at(frame_to_keep).values());

            // Relative factors for other lmk
            for (auto lmk : _marginalization->_lmk_to_keep["pointxd"]) {

                ceres::CostFunction *cost_fct = new PoseToLandmarkFactor(_marginalization->_map_lmk_prior.at(lmk),
                                                                         T_f_w,
                                                                         lmk->getPose().translation(),
                                                                         _marginalization->_map_lmk_inf.at(lmk));

                // TODO: fix this, it is supposed to be in the map
                if (_map_lmk_ptpar.find(lmk) == _map_lmk_ptpar.end()) {
                    _map_lmk_ptpar.emplace(lmk, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
                    problem.AddParameterBlock(_map_lmk_ptpar.at(lmk).values(), 3);
                    ordering->AddElementToGroup(_map_lmk_ptpar.at(lmk).values(), 0);
                }

                problem.AddResidualBlock(cost_fct,
                                         loss_function,
                                         _map_frame_posepar.at(frame_to_keep).values(),
                                         _map_lmk_ptpar.at(lmk).values());
            }
        }

        /// CASE 2 VO ///
        else {
            // Unary factor for lmk with prior
            ceres::CostFunction *cost_fct_0 =
                new Landmark3DPrior(_marginalization->_prior_lmk,
                                    _marginalization->_lmk_with_prior->getPose().translation(),
                                    _marginalization->_info_lmk);
            problem.AddResidualBlock(
                cost_fct_0, loss_function, _map_lmk_ptpar.at(_marginalization->_lmk_with_prior).values());
            ordering->Remove(_map_lmk_ptpar.at(_marginalization->_lmk_with_prior).values());
            ordering->AddElementToGroup(_map_lmk_ptpar.at(_marginalization->_lmk_with_prior).values(), 2);

            // TODO: fix this, it is supposed to be in the map
            if (_map_lmk_ptpar.find(_marginalization->_lmk_with_prior) == _map_lmk_ptpar.end()) {
                _map_lmk_ptpar.emplace(_marginalization->_lmk_with_prior,
                                       PointXYZParametersBlock(Eigen::Vector3d::Zero()));
                problem.AddParameterBlock(_map_lmk_ptpar.at(_marginalization->_lmk_with_prior).values(), 3);
                ordering->AddElementToGroup(_map_lmk_ptpar.at(_marginalization->_lmk_with_prior).values(), 0);
            }

            // Relative factors for other lmk
            for (uint k = 0; k < _marginalization->_lmk_to_keep["pointxd"].size() - 1; k++) {

                std::shared_ptr<ALandmark> lmk_k   = _marginalization->_lmk_to_keep["pointxd"].at(k);
                std::shared_ptr<ALandmark> lmk_kp1 = _marginalization->_lmk_to_keep["pointxd"].at(k + 1);

                // TODO: fix this, it is supposed to be in the map
                if (_map_lmk_ptpar.find(lmk_k) == _map_lmk_ptpar.end()) {
                    _map_lmk_ptpar.emplace(lmk_k, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
                    problem.AddParameterBlock(_map_lmk_ptpar.at(lmk_k).values(), 3);
                    ordering->AddElementToGroup(_map_lmk_ptpar.at(lmk_k).values(), 0);
                }

                // TODO: fix this, it is supposed to be in the map
                if (_map_lmk_ptpar.find(lmk_kp1) == _map_lmk_ptpar.end()) {
                    _map_lmk_ptpar.emplace(lmk_k, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
                    problem.AddParameterBlock(_map_lmk_ptpar.at(lmk_kp1).values(), 3);
                    ordering->AddElementToGroup(_map_lmk_ptpar.at(lmk_kp1).values(), 0);
                }

                // The landmark must be distinct
                if (lmk_k == lmk_kp1)
                    continue;

                ordering->Remove(_map_lmk_ptpar.at(lmk_kp1).values());
                ordering->AddElementToGroup(_map_lmk_ptpar.at(lmk_kp1).values(), 2);

                ceres::CostFunction *cost_fct =
                    new LandmarkToLandmarkFactor(_marginalization->_map_lmk_prior.at(lmk_kp1),
                                                 lmk_k->getPose().translation(),
                                                 lmk_kp1->getPose().translation(),
                                                 _marginalization->_map_lmk_inf.at(lmk_kp1));
                problem.AddResidualBlock(
                    cost_fct, loss_function, _map_lmk_ptpar.at(lmk_k).values(), _map_lmk_ptpar.at(lmk_kp1).values());
            }
        }
    }

    return 0;
}

bool AngularAdjustmentCERESAnalytic::marginalize(std::shared_ptr<Frame> &frame0,
                                                 std::shared_ptr<Frame> &frame1,
                                                 bool enable_sparsif) {
    _enable_sparsif = enable_sparsif;

    // Setup the maps for memory gestion
    _map_lmk_ptpar.clear();
    _map_frame_posepar.clear();
    _map_frame_velpar.clear();
    _map_frame_dbapar.clear();
    _map_frame_dbgpar.clear();

    // Select the nodes to marginalize / keep
    _marginalization->preMarginalize(frame0, frame1, _marginalization_last);

    // Create pose parameters for the frames
    _map_frame_posepar.emplace(frame0, PoseParametersBlock(Eigen::Affine3d::Identity()));
    _map_frame_posepar.emplace(frame1, PoseParametersBlock(Eigen::Affine3d::Identity()));

    // Create Marginalization Blocks with pre-integration factors in the case of VIO
    if (frame0->getIMU() && frame1->getIMU()) {

        // Create velo and bias parameters for both frames
        _map_frame_velpar.emplace(frame0, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
        _map_frame_dbapar.emplace(frame0, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
        _map_frame_dbgpar.emplace(frame0, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
        _map_frame_velpar.emplace(frame1, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
        _map_frame_dbapar.emplace(frame1, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
        _map_frame_dbgpar.emplace(frame1, PointXYZParametersBlock(Eigen::Vector3d::Zero()));

        // Parameters of marginalization blocks
        std::vector<double *> parameter_blocks;
        std::vector<int> parameter_idx;

        // For the frames
        parameter_idx.push_back(_marginalization->_map_frame_idx.at(frame0));
        parameter_blocks.push_back(_map_frame_posepar.at(frame0).values());
        parameter_idx.push_back(_marginalization->_map_frame_idx.at(frame1));
        parameter_blocks.push_back(_map_frame_posepar.at(frame1).values());

        // For the velocities
        parameter_idx.push_back(_marginalization->_map_frame_idx.at(frame0) + 6);
        parameter_blocks.push_back(_map_frame_velpar.at(frame0).values());
        parameter_idx.push_back(_marginalization->_map_frame_idx.at(frame1) + 6);
        parameter_blocks.push_back(_map_frame_velpar.at(frame1).values());

        // For the biases
        parameter_idx.push_back(_marginalization->_map_frame_idx.at(frame0) + 9);
        parameter_blocks.push_back(_map_frame_dbapar.at(frame0).values());
        parameter_idx.push_back(_marginalization->_map_frame_idx.at(frame0) + 12);
        parameter_blocks.push_back(_map_frame_dbgpar.at(frame0).values());

        // Add the pre integration factor in the marginalization scheme
        ceres::CostFunction *cost_fct = new IMUFactor(frame0->getIMU(), frame1->getIMU());
        _marginalization->addMarginalizationBlock(
            std::make_shared<MarginalizationBlockInfo>(cost_fct, parameter_idx, parameter_blocks));

        // Parameters of marginalization blocks
        std::vector<double *> parameter_blocks_b;
        std::vector<int> parameter_idx_b;

        // For the biases
        parameter_idx_b.push_back(_marginalization->_map_frame_idx.at(frame0) + 9);
        parameter_blocks_b.push_back(_map_frame_dbapar.at(frame0).values());
        parameter_idx_b.push_back(_marginalization->_map_frame_idx.at(frame0) + 12);
        parameter_blocks_b.push_back(_map_frame_dbgpar.at(frame0).values());
        parameter_idx_b.push_back(_marginalization->_map_frame_idx.at(frame1) + 9);
        parameter_blocks_b.push_back(_map_frame_dbapar.at(frame1).values());
        parameter_idx_b.push_back(_marginalization->_map_frame_idx.at(frame1) + 12);
        parameter_blocks_b.push_back(_map_frame_dbgpar.at(frame1).values());

        // Add the bias random walk factor in the marginalization scheme
        ceres::CostFunction *cost_fct_b = new IMUBiasFactor(frame0->getIMU(), frame1->getIMU());
        _marginalization->addMarginalizationBlock(
            std::make_shared<MarginalizationBlockInfo>(cost_fct_b, parameter_idx_b, parameter_blocks_b));
    }

    // Create Marginalization Blocks with landmark to keep
    for (auto tlmk : _marginalization->_lmk_to_keep) {
        for (auto lmk : tlmk.second) {
            _map_lmk_ptpar.emplace(lmk, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
            // For each feature on the frame
            for (auto feature : lmk->getFeatures()) {
                if (feature.lock()->getSensor()->getFrame() == frame0) {

                    // Compute index and block vectors for reprojection factor
                    std::vector<double *> parameter_blocks;
                    std::vector<int> parameter_idx;

                    // For the frame
                    parameter_idx.push_back(_marginalization->_map_frame_idx.at(frame0));
                    parameter_blocks.push_back(_map_frame_posepar.at(frame0).values());

                    // For the lmk
                    parameter_idx.push_back(_marginalization->_map_lmk_idx.at(lmk));
                    parameter_blocks.push_back(_map_lmk_ptpar.at(lmk).values());

                    // Add the angular factor in the marginalization scheme
                    ceres::CostFunction *cost_fct =
                        new AngularErrCeres_pointxd_dx(feature.lock()->getBearingVectors().at(0),
                                                       feature.lock()->getSensor()->getFrame2SensorTransform(),
                                                       frame0->getWorld2FrameTransform(),
                                                       lmk->getPose().translation(),
                                                       (1 / feature.lock()->getSensor()->getFocal()));
                    _marginalization->addMarginalizationBlock(
                        std::make_shared<MarginalizationBlockInfo>(cost_fct, parameter_idx, parameter_blocks));
                }
            }
        }
    }

    // Create Marginalization Blocks with landmark to marginalize
    for (auto tlmk : _marginalization->_lmk_to_marg) {
        for (auto lmk : tlmk.second) {
            _map_lmk_ptpar.emplace(lmk, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
            // For each feature on the frame
            for (auto feature : lmk->getFeatures()) {
                if (feature.lock()->getSensor()->getFrame() == frame0) {

                    // Compute index and block vectors for reprojection factor
                    std::vector<double *> parameter_blocks;
                    std::vector<int> parameter_idx;

                    // For the frame
                    parameter_idx.push_back(_marginalization->_map_frame_idx.at(frame0));
                    parameter_blocks.push_back(_map_frame_posepar.at(frame0).values());

                    // For the lmk
                    parameter_idx.push_back(_marginalization->_map_lmk_idx.at(lmk));
                    parameter_blocks.push_back(_map_lmk_ptpar.at(lmk).values());

                    // Add the angular factor in the marginalization scheme
                    ceres::CostFunction *cost_fct =
                        new AngularErrCeres_pointxd_dx(feature.lock()->getBearingVectors().at(0),
                                                       feature.lock()->getSensor()->getFrame2SensorTransform(),
                                                       frame0->getWorld2FrameTransform(),
                                                       lmk->getPose().translation(),
                                                       (1 / feature.lock()->getSensor()->getFocal()));
                    _marginalization->addMarginalizationBlock(
                        std::make_shared<MarginalizationBlockInfo>(cost_fct, parameter_idx, parameter_blocks));
                }
            }
        }
    }

    // Create a marginalization block with previous prior
    if (!_marginalization_last->_lmk_to_keep.empty()) {

        // Compute index and block vectors for marginalization factor
        std::vector<double *> parameter_blocks;
        std::vector<int> parameter_idx;

        // Fill the parameters with previous frame to keep
        if (_marginalization_last->_frame_to_keep) {
            parameter_blocks.push_back(_map_frame_posepar.at(_marginalization_last->_frame_to_keep).values());
            parameter_idx.push_back(_marginalization->_map_frame_idx.at(_marginalization_last->_frame_to_keep));

            parameter_blocks.push_back(_map_frame_velpar.at(_marginalization_last->_frame_to_keep).values());
            parameter_idx.push_back(_marginalization->_map_frame_idx.at(_marginalization_last->_frame_to_keep) + 6);

            parameter_blocks.push_back(_map_frame_dbapar.at(_marginalization_last->_frame_to_keep).values());
            parameter_idx.push_back(_marginalization->_map_frame_idx.at(_marginalization_last->_frame_to_keep) + 9);

            parameter_blocks.push_back(_map_frame_dbgpar.at(_marginalization_last->_frame_to_keep).values());
            parameter_idx.push_back(_marginalization->_map_frame_idx.at(_marginalization_last->_frame_to_keep) + 12);
        }

        // Fill the parameters with previous landmarks kept
        for (auto tlmk : _marginalization_last->_lmk_to_keep) {
            for (auto lmk : tlmk.second) {
                parameter_blocks.push_back(_map_lmk_ptpar.at(lmk).values());
                parameter_idx.push_back(_marginalization->_map_lmk_idx.at(lmk));
            }
        }
        // Add the last prior in the marginalization scheme
        ceres::CostFunction *cost_fct = new MarginalizationFactor(_marginalization_last);
        _marginalization->addMarginalizationBlock(
            std::make_shared<MarginalizationBlockInfo>(cost_fct, parameter_idx, parameter_blocks));
    }

    // Add frame prior if it exists
    if (frame0->hasPrior()) {
        // Compute index and block vectors for pose prior factor
        std::vector<double *> parameter_blocks;
        std::vector<int> parameter_idx;
        parameter_blocks.push_back(_map_frame_posepar.at(frame0).values());
        parameter_idx.push_back(_marginalization->_map_frame_idx.at(frame0));

        ceres::CostFunction *cost_fct =
            new PosePriordx(frame0->getWorld2FrameTransform(), frame0->getPrior(), frame0->getInfPrior().asDiagonal());
        _marginalization->addMarginalizationBlock(
            std::make_shared<MarginalizationBlockInfo>(cost_fct, parameter_idx, parameter_blocks));
    }

    if (frame1->hasPrior()) {
        // Compute index and block vectors for pose prior factor
        std::vector<double *> parameter_blocks;
        std::vector<int> parameter_idx;
        parameter_blocks.push_back(_map_frame_posepar.at(frame1).values());
        parameter_idx.push_back(_marginalization->_map_frame_idx.at(frame1));

        ceres::CostFunction *cost_fct =
            new PosePriordx(frame1->getWorld2FrameTransform(), frame1->getPrior(), frame1->getInfPrior().asDiagonal());
        _marginalization->addMarginalizationBlock(
            std::make_shared<MarginalizationBlockInfo>(cost_fct, parameter_idx, parameter_blocks));
    }

    // Reset the marginalization scheme if it failed
    if (!_marginalization->computeSchurComplement()) {
        _marginalization->_lmk_to_keep.clear();
        _marginalization->_marginalization_blocks.clear();
        _marginalization_last->_lmk_to_keep.clear();
        return false;
    }

    // Compute the sparse factors
    if (_enable_sparsif) {
        if (_marginalization->_frame_to_keep)
            _marginalization->sparsifyVIO();
        else
            _marginalization->sparsifyVO();
    }

    // The jacobians and residuals of the dense factor are computed in every case for propagation
    _marginalization->computeJacobiansAndResiduals();

    // Update the last marginalization variable
    _marginalization_last->_map_lmk_prior.clear();
    _marginalization_last->_map_lmk_inf.clear();
    _marginalization_last->_lmk_to_keep              = _marginalization->_lmk_to_keep;
    _marginalization_last->_frame_to_keep            = _marginalization->_frame_to_keep;
    _marginalization_last->_map_frame_idx            = _marginalization->_map_frame_idx;
    _marginalization_last->_map_lmk_idx              = _marginalization->_map_lmk_idx;
    _marginalization_last->_map_frame_inf            = _marginalization->_map_frame_inf;
    _marginalization_last->_map_lmk_inf              = _marginalization->_map_lmk_inf;
    _marginalization_last->_map_lmk_prior            = _marginalization->_map_lmk_prior;
    _marginalization_last->_lmk_with_prior           = _marginalization->_lmk_with_prior;
    _marginalization_last->_prior_lmk                = _marginalization->_prior_lmk;
    _marginalization_last->_info_lmk                 = _marginalization->_info_lmk;
    _marginalization_last->_Ak                       = _marginalization->_Ak;
    _marginalization_last->_bk                       = _marginalization->_bk;
    _marginalization_last->_marginalization_jacobian = _marginalization->_marginalization_jacobian;
    _marginalization_last->_marginalization_residual = _marginalization->_marginalization_residual;
    _marginalization_last->_m                        = _marginalization->_m;
    _marginalization_last->_n                        = _marginalization->_n;
    _marginalization_last->_n_full                   = _marginalization->_n_full;
    _marginalization_last->_U                        = _marginalization->_U;
    _marginalization_last->_Lambda                   = _marginalization->_Lambda;
    _marginalization_last->_Sigma                    = _marginalization->_Sigma;

    return true;
}

bool AngularAdjustmentCERESAnalytic::landmarkOptimizationNoFov(std::shared_ptr<Frame> &f,
                                                               std::shared_ptr<Frame> &fp,
                                                               Eigen::Affine3d &T_cam0_cam0p,
                                                               double info_scale) {

    // Set maps
    _map_frame_posepar.clear();
    _map_lmk_ptpar.clear();
    std::unordered_map<std::shared_ptr<isae::ALandmark>, ceres::ResidualBlockId> map_lmk_res;

    // Build the Bundle Adjustement Problem
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(std::sqrt(1.345));

    // Create scale parameter block
    double lambda[1] = {1.0};
    problem.AddParameterBlock(lambda, 1);

    // Create pose parameter block and fix them
    _map_frame_posepar.emplace(f, PoseParametersBlock(Eigen::Affine3d::Identity()));
    problem.AddParameterBlock(_map_frame_posepar.at(f).values(), 6);
    problem.SetParameterBlockConstant(_map_frame_posepar.at(f).values());

    // Fix the scale for degenerated motion
    if (geometry::log_so3(T_cam0_cam0p.rotation()).norm() < 0.05 || T_cam0_cam0p.translation().norm() < 0.01) {
        problem.SetParameterBlockConstant(lambda);
    }

    // Add residuals for every landmarks that were observed on both frames
    std::shared_ptr<ASensor> cam0                = f->getSensors().at(0);
    std::vector<std::shared_ptr<ALandmark>> lmks = fp->getLandmarks()["pointxd"];
    for (auto &lmk : lmks) {

        if (!lmk->isInitialized() || lmk->isOutlier())
            continue;

        std::shared_ptr<AFeature> feat  = nullptr;
        std::shared_ptr<AFeature> featp = nullptr;

        // Test more constraints
        std::vector<std::shared_ptr<AFeature>> feats;

        // For all feature
        for (auto &wfeature : lmk->getFeatures()) {
            std::shared_ptr<AFeature> feature = wfeature.lock();

            if (!feature || !feature->getSensor()->getFrame()->isKeyFrame()) {
                continue;
            }

            if (feature->getSensor()->getFrame() == f) {
                feat = feature;
                continue;
            }

            if (feature->getSensor()->getFrame() == fp) {
                featp = feature;
                continue;
            }

            feats.push_back(feature);
        }

        // Add reprojection factor if there is a match
        if (feat && featp) {

            // Add parameter block for each landmark
            _map_lmk_ptpar.emplace(lmk, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
            problem.AddParameterBlock(_map_lmk_ptpar.at(lmk).values(), 3);

            // The scale on the cam0 is estimated
            Eigen::Affine3d T_cam_cam0 =
                featp->getSensor()->getFrame2SensorTransform() * cam0->getFrame2SensorTransform().inverse();

            ceres::CostFunction *cost_fct = new AngularErrorScaleCam0(featp->getBearingVectors().at(0),
                                                                      lmk->getPose().translation(),
                                                                      cam0->getWorld2SensorTransform(),
                                                                      T_cam0_cam0p,
                                                                      T_cam_cam0,
                                                                      1);
            // problem.AddResidualBlock(cost_fct, loss_function, lambda, _map_lmk_ptpar.at(lmk).values());
            map_lmk_res.emplace(
                lmk, problem.AddResidualBlock(cost_fct, loss_function, lambda, _map_lmk_ptpar.at(lmk).values()));

            ceres::CostFunction *cost_fct1 =
                new AngularErrCeres_pointxd_dx(feat->getBearingVectors().at(0),
                                               feat->getSensor()->getFrame2SensorTransform(),
                                               f->getWorld2FrameTransform(),
                                               lmk->getPose().translation());
            problem.AddResidualBlock(
                cost_fct1, loss_function, _map_frame_posepar.at(f).values(), _map_lmk_ptpar.at(lmk).values());

            for (auto &f : feats) {

                std::shared_ptr<Frame> fother = f->getSensor()->getFrame();
                if (_map_frame_posepar.find(fother) == _map_frame_posepar.end()) {
                    _map_frame_posepar.emplace(fother, PoseParametersBlock(Eigen::Affine3d::Identity()));
                    problem.AddParameterBlock(_map_frame_posepar.at(fother).values(), 6);
                    problem.SetParameterBlockConstant(_map_frame_posepar.at(fother).values());
                }

                ceres::CostFunction *cost_fct2 =
                    new AngularErrCeres_pointxd_dx(f->getBearingVectors().at(0),
                                                   f->getSensor()->getFrame2SensorTransform(),
                                                   fother->getWorld2FrameTransform(),
                                                   lmk->getPose().translation());
                problem.AddResidualBlock(
                    cost_fct2, loss_function, _map_frame_posepar.at(fother).values(), _map_lmk_ptpar.at(lmk).values());
            }
        }
    }

    // Add a prior to prevent scale from diverging
    ceres::CostFunction *cost_fct1 = new scalePrior(info_scale);
    problem.AddResidualBlock(cost_fct1, nullptr, lambda);

    // Solve the problem we just built
    ceres::Solver::Options options;
    options.trust_region_strategy_type         = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type                 = ceres::DENSE_SCHUR;
    options.max_num_iterations                 = 20;
    options.minimizer_progress_to_stdout       = false;
    options.use_explicit_schur_complement      = true;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.function_tolerance                 = 1.e-3;
    options.num_threads                        = 4;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (!summary.IsSolutionUsable() || lambda[0] < 0.5 || lambda[0] > 1.5)
        return false;

    // Update the current pose with the optimized scale
    T_cam0_cam0p.translation() *= lambda[0];
    Eigen::Affine3d T_f0_f0p = f->getSensors().at(0)->getFrame2SensorTransform().inverse() * T_cam0_cam0p *
                               f->getSensors().at(0)->getFrame2SensorTransform();
    fp->setWorld2FrameTransform((f->getFrame2WorldTransform() * T_f0_f0p).inverse());

    // Chi2 test
    for (auto &lmk_ptpar : _map_lmk_ptpar) {
        auto lmk = lmk_ptpar.first;

        if (!lmk->isInitialized() || lmk->isOutlier())
            continue;

        // Check the residuals
        Eigen::Vector2d residuals;
        std::vector<double *> parameters;
        parameters.push_back(lambda);
        parameters.push_back(_map_lmk_ptpar.at(lmk).values());
        problem.GetCostFunctionForResidualBlock(map_lmk_res.at(lmk))
            ->Evaluate(parameters.data(), residuals.data(), nullptr);

        if (residuals.norm() > (double)(2 / f->getSensors().at(0)->getFocal())) {
            lmk->setOutlier();
            continue;
        }

        if (!lmk->isOutlier())
            lmk->setPose(lmk->getPose() * _map_lmk_ptpar.at(lmk).getPose());
    }

    std::cout << "scale : " << lambda[0] << std::endl;

    return true;
}

Eigen::MatrixXd AngularAdjustmentCERESAnalytic::marginalizeRelative(std::shared_ptr<Frame> &frame0,
                                                         std::shared_ptr<Frame> &frame1) {

    // Setup the maps for memory gestion
    _map_lmk_ptpar.clear();
    _map_frame_posepar.clear();
    _map_frame_velpar.clear();
    _map_frame_dbapar.clear();
    _map_frame_dbgpar.clear();

    // Select the nodes to marginalize / keep
    _marginalization->preMarginalizeRelative(frame0, frame1);

    // Create pose parameters for the frames
    _map_frame_posepar.emplace(frame0, PoseParametersBlock(Eigen::Affine3d::Identity()));
    _map_frame_posepar.emplace(frame1, PoseParametersBlock(Eigen::Affine3d::Identity()));

    // Create Marginalization Blocks with pre-integration factors in the case of VIO
    if (frame0->getIMU() && frame1->getIMU()) {

        // Create velo and bias parameters for both frames
        _map_frame_velpar.emplace(frame0, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
        _map_frame_dbapar.emplace(frame0, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
        _map_frame_dbgpar.emplace(frame0, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
        _map_frame_velpar.emplace(frame1, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
        _map_frame_dbapar.emplace(frame1, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
        _map_frame_dbgpar.emplace(frame1, PointXYZParametersBlock(Eigen::Vector3d::Zero()));

        // Parameters of marginalization blocks (the variables are stored in the order v0, v1, ba0, bg0, ba1, bg0)
        std::vector<double *> parameter_blocks;
        std::vector<int> parameter_idx;

        // For the frames
        parameter_idx.push_back(_marginalization->_map_frame_idx.at(frame0));
        parameter_blocks.push_back(_map_frame_posepar.at(frame0).values());
        parameter_idx.push_back(_marginalization->_map_frame_idx.at(frame1));
        parameter_blocks.push_back(_map_frame_posepar.at(frame1).values());

        // For the velocities
        parameter_idx.push_back(_marginalization->_n);
        parameter_blocks.push_back(_map_frame_velpar.at(frame0).values());
        parameter_idx.push_back(_marginalization->_n + 3);
        parameter_blocks.push_back(_map_frame_velpar.at(frame1).values());

        // For the biases
        parameter_idx.push_back(_marginalization->_n + 6);
        parameter_blocks.push_back(_map_frame_dbapar.at(frame0).values());
        parameter_idx.push_back(_marginalization->_n + 9);
        parameter_blocks.push_back(_map_frame_dbgpar.at(frame0).values());

        // Add the pre integration factor in the marginalization scheme
        ceres::CostFunction *cost_fct = new IMUFactor(frame0->getIMU(), frame1->getIMU());
        _marginalization->addMarginalizationBlock(
            std::make_shared<MarginalizationBlockInfo>(cost_fct, parameter_idx, parameter_blocks));

        // Parameters of marginalization blocks
        std::vector<double *> parameter_blocks_b;
        std::vector<int> parameter_idx_b;

        // For the biases
        parameter_idx_b.push_back(_marginalization->_n + 6);
        parameter_blocks_b.push_back(_map_frame_dbapar.at(frame0).values());
        parameter_idx_b.push_back(_marginalization->_n + 9);
        parameter_blocks_b.push_back(_map_frame_dbgpar.at(frame0).values());
        parameter_idx_b.push_back(_marginalization->_n + 12);
        parameter_blocks_b.push_back(_map_frame_dbapar.at(frame1).values());
        parameter_idx_b.push_back(_marginalization->_n + 15);
        parameter_blocks_b.push_back(_map_frame_dbgpar.at(frame1).values());

        // Add the bias random walk factor in the marginalization scheme
        ceres::CostFunction *cost_fct_b = new IMUBiasFactor(frame0->getIMU(), frame1->getIMU());
        _marginalization->addMarginalizationBlock(
            std::make_shared<MarginalizationBlockInfo>(cost_fct_b, parameter_idx_b, parameter_blocks_b));
    }

    // Create Marginalization Blocks with landmark to marginalize
    for (auto tlmk : _marginalization->_lmk_to_marg) {
        for (auto lmk : tlmk.second) {
            _map_lmk_ptpar.emplace(lmk, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
            // For each feature on the frame
            for (auto &feature : lmk->getFeatures()) {
                std::shared_ptr<Frame> frame = feature.lock()->getSensor()->getFrame();
                if (frame == frame0 || frame == frame1) {

                    // Compute index and block vectors for reprojection factor
                    std::vector<double *> parameter_blocks;
                    std::vector<int> parameter_idx;

                    // For the frame
                    parameter_idx.push_back(_marginalization->_map_frame_idx.at(frame));
                    parameter_blocks.push_back(_map_frame_posepar.at(frame).values());

                    // For the lmk
                    parameter_idx.push_back(_marginalization->_map_lmk_idx.at(lmk));
                    parameter_blocks.push_back(_map_lmk_ptpar.at(lmk).values());

                    // Add the angular factor in the marginalization scheme
                    ceres::CostFunction *cost_fct =
                        new AngularErrCeres_pointxd_dx(feature.lock()->getBearingVectors().at(0),
                                                       feature.lock()->getSensor()->getFrame2SensorTransform(),
                                                       frame->getWorld2FrameTransform(),
                                                       lmk->getPose().translation(),
                                                       (1 / feature.lock()->getSensor()->getFocal()));
                    _marginalization->addMarginalizationBlock(
                        std::make_shared<MarginalizationBlockInfo>(cost_fct, parameter_idx, parameter_blocks));
                }
            }
        }
    }

    // Reset the marginalization scheme if it failed
    if (!_marginalization->computeSchurComplement()) {
        _marginalization->_lmk_to_keep.clear();
        _marginalization->_marginalization_blocks.clear();
        _marginalization_last->_lmk_to_keep.clear();
        return Eigen::MatrixXd::Zero(12,12);
    }

    // Compute the relative pose factor with NFR

    // Build a marginalization block to compute the jacobian
    Eigen::Affine3d T_w_a         = frame0->getFrame2WorldTransform();
    Eigen::Affine3d T_w_b         = frame1->getFrame2WorldTransform();
    Eigen::Affine3d T_a_b         = frame0->getWorld2FrameTransform() * T_w_b;
    ceres::CostFunction *cost_fct = new Relative6DPose(T_w_a, T_w_b, T_a_b, Vector6d::Ones().asDiagonal());
    std::vector<double *> parameter_blocks;
    std::vector<int> parameter_idx;
    parameter_blocks.push_back(_map_frame_posepar.at(frame0).values());
    parameter_idx.push_back(0);
    parameter_blocks.push_back(_map_frame_posepar.at(frame1).values());
    parameter_idx.push_back(0);
    MarginalizationBlockInfo block_relpose(cost_fct, parameter_idx, parameter_blocks);

    // Compute the covariance of the non linear factor
    block_relpose.Evaluate();
    Eigen::MatrixXd J   = Eigen::MatrixXd::Zero(6, 12);
    J.block(0, 0, 6, 6) = block_relpose._jacobians.at(0);
    J.block(0, 6, 6, 6) = block_relpose._jacobians.at(1);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(_marginalization->_Ak);
    Eigen::MatrixXd Ak_inv =
        saes.eigenvectors() *
        Eigen::VectorXd((saes.eigenvalues().array() > 1e-12).select(saes.eigenvalues().array().inverse(), 0))
            .asDiagonal() *
        saes.eigenvectors().transpose();
    Eigen::MatrixXd cov = J * _marginalization->_Sigma_k * J.transpose();
    Eigen::MatrixXd inf = cov.inverse();

    return inf;
}

} // namespace isae
