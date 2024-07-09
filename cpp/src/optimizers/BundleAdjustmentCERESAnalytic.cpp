#include "isaeslam/optimizers/BundleAdjustmentCERESAnalytic.h"

namespace isae {

uint BundleAdjustmentCERESAnalytic::addSingleFrameResiduals(ceres::Problem &problem,
                                                            ceres::LossFunction *loss_function,
                                                            std::shared_ptr<Frame> &frame,
                                                            typed_vec_landmarks &cloud_to_optimize) {

    // ceres::Manifold *nullptr = new SE3RightParameterization();
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

                        _map_lmk_ptpar.emplace(landmark, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
                        problem.AddParameterBlock(_map_lmk_ptpar.at(landmark).values(), 3);
                        problem.SetParameterBlockConstant(_map_lmk_ptpar.at(landmark).values());

                        ceres::CostFunction *cost_fct =
                            new ReprojectionErrCeres_pointxd_dx(feature->getPoints().at(0), cam, landmark->getPose());

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

                        _map_lmk_posepar.emplace(landmark, PoseParametersBlock(Eigen::Affine3d::Identity()));
                        problem.AddParameterBlock(_map_lmk_posepar.at(landmark).values(), 6);
                        problem.SetParameterBlockConstant(_map_lmk_posepar.at(landmark).values());

                        ceres::CostFunction *cost_fct = new ReprojectionErrCeres_linexd_dx(feature->getPoints(),
                                                                                           cam,
                                                                                           landmark->getPose(),
                                                                                           landmark->getModel(),
                                                                                           landmark->getScale(),
                                                                                           1);

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

uint BundleAdjustmentCERESAnalytic::addLandmarkResiduals(ceres::Problem &problem,
                                                         ceres::LossFunction *loss_function,
                                                         typed_vec_landmarks &cloud_to_optimize) {

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

                    if (!feature || !feature->getSensor()->getFrame()->isKeyFrame()) {
                        continue;
                    }

                    std::shared_ptr<ImageSensor> cam = feature->getSensor();
                    std::shared_ptr<Frame> frame     = cam->getFrame();

                    if (_map_frame_posepar.find(frame) == _map_frame_posepar.end()) {
                        _map_frame_posepar.emplace(frame, PoseParametersBlock(Eigen::Affine3d::Identity()));
                        problem.AddParameterBlock(_map_frame_posepar.at(frame).values(), 6);
                        problem.SetParameterBlockConstant(_map_frame_posepar.at(frame).values());
                    }

                    ceres::CostFunction *cost_fct =
                        new ReprojectionErrCeres_pointxd_dx(feature->getPoints().at(0), cam, landmark->getPose());
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

                    if (!feature || !feature->getSensor()->getFrame()->isKeyFrame()) {
                        continue;
                    }

                    std::shared_ptr<ImageSensor> cam = feature->getSensor();
                    std::shared_ptr<Frame> frame     = cam->getFrame();

                    if (_map_frame_posepar.find(frame) == _map_frame_posepar.end()) {
                        _map_frame_posepar.emplace(frame, PoseParametersBlock(Eigen::Affine3d::Identity()));
                        problem.AddParameterBlock(_map_frame_posepar.at(frame).values(), 6);
                        problem.SetParameterBlockConstant(_map_frame_posepar.at(frame).values());
                    }

                    ceres::CostFunction *cost_fct = new ReprojectionErrCeres_linexd_dx(
                        feature->getPoints(), cam, landmark->getPose(), landmark->getModel(), landmark->getScale(), 1);
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

uint BundleAdjustmentCERESAnalytic::addResidualsLocalMap(ceres::Problem &problem,
                                                         ceres::LossFunction *loss_function,
                                                         ceres::ParameterBlockOrdering *ordering,
                                                         std::vector<std::shared_ptr<Frame>> &frame_vector,
                                                         size_t fixed_frame_number,
                                                         std::shared_ptr<isae::LocalMap> &local_map) {

    uint nb_residuals = 0;

    // Add parameter block and ordering for each frame parameter

    for (size_t i = 0; i < frame_vector.size(); i++) {

        std::shared_ptr<Frame> frame = frame_vector.at(i);
        _map_frame_posepar.emplace(frame, PoseParametersBlock(Eigen::Affine3d::Identity()));

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

                    if (!feature || !feature->getSensor()->getFrame()->isKeyFrame())
                        continue;

                    std::shared_ptr<ImageSensor> cam = feature->getSensor();
                    std::shared_ptr<Frame> frame     = cam->getFrame();

                    nb_residuals++;

                    ceres::CostFunction *cost_fct =
                        new ReprojectionErrCeres_pointxd_dx(feature->getPoints().at(0), cam, landmark->getPose());

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
                ordering->AddElementToGroup(_map_lmk_posepar.at(landmark).values(), 2);

                // For all feature
                std::vector<std::weak_ptr<AFeature>> featuresAssociatedLandmarks = landmark->getFeatures();
                for (std::weak_ptr<AFeature> &wfeature : featuresAssociatedLandmarks) {
                    std::shared_ptr<AFeature> feature = wfeature.lock();

                    if (!feature || !feature->getSensor()->getFrame()->isKeyFrame())
                        continue;

                    std::shared_ptr<ImageSensor> cam = feature->getSensor();
                    std::shared_ptr<Frame> frame     = cam->getFrame();

                    nb_residuals++;

                    ceres::CostFunction *cost_fct = new ReprojectionErrCeres_linexd_dx(
                        feature->getPoints(), cam, landmark->getPose(), landmark->getModel(), landmark->getScale(), 1);

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

uint BundleAdjustmentCERESAnalytic::addMarginalizationResiduals(ceres::Problem &problem,
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
        for (auto tlmk : _marginalization->_lmk_to_keep) {
            for (auto lmk : tlmk.second) {
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
        }

        ceres::CostFunction *cost_fct = new MarginalizationFactor(_marginalization);
        problem.AddResidualBlock(cost_fct, loss_function, prior_parameter_blocks);
    }

    // Add marginalization factor, sparse case
    if (_marginalization->_lmk_to_keep.size() > 1 && _enable_sparsif) {

        /// CASE 1 VIO ///
        if (_marginalization->_frame_to_keep->getIMU()) {
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
            for (auto &lmk : _marginalization->_lmk_to_keep["pointxd"]) {

                ceres::CostFunction *cost_fct = new PoseToLandmarkFactor(_marginalization->_map_lmk_prior.at(lmk),
                                                                         T_f_w,
                                                                         lmk->getPose().translation(),
                                                                         _marginalization->_map_lmk_inf.at(lmk));
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

            // Relative factors for other lmk
            for (uint k = 0; k < _marginalization->_lmk_to_keep["pointxd"].size() - 1; k++) {

                std::shared_ptr<ALandmark> lmk_k   = _marginalization->_lmk_to_keep["pointxd"].at(k);
                std::shared_ptr<ALandmark> lmk_kp1 = _marginalization->_lmk_to_keep["pointxd"].at(k + 1);

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

bool BundleAdjustmentCERESAnalytic::marginalize(std::shared_ptr<Frame> &frame0,
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

    // Create pose parameters for the frame to marginalize
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

    // Create Marginalization Blocks with landmarks to keep
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

                    // Add the reprojection factor in the marginalization scheme
                    ceres::CostFunction *cost_fct = new ReprojectionErrCeres_pointxd_dx(
                        feature.lock()->getPoints().at(0), feature.lock()->getSensor(), lmk->getPose());

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

                    // Add the reprojection factor in the marginalization scheme
                    ceres::CostFunction *cost_fct = new ReprojectionErrCeres_pointxd_dx(
                        feature.lock()->getPoints().at(0), feature.lock()->getSensor(), lmk->getPose());

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

    // Reset the marginalization scheme if it failed
    if (!_marginalization->computeSchurComplement()) {
        _marginalization->_lmk_to_keep.clear();
        _marginalization->_marginalization_blocks.clear();
        _marginalization_last->_lmk_to_keep.clear();
        return false;
    }

    // Compute the sparse factors
    if (_enable_sparsif) {
        if (_marginalization->_frame_to_keep->getIMU())
            _marginalization->sparsifyVIO();
        else
            _marginalization->sparsifyVO();
    }

    // The jacobians and residuals of the dense factor are computed in every case
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

} // namespace isae
