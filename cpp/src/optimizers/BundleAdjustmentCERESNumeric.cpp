#include "isaeslam/optimizers/BundleAdjustmentCERESNumeric.h"

namespace isae {

uint BundleAdjustmentCERESNumeric::addSingleFrameResiduals(ceres::Problem &problem,
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

                        _map_lmk_ptpar.emplace(landmark, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
                        problem.AddParameterBlock(_map_lmk_ptpar.at(landmark).values(), 3);
                        problem.SetParameterBlockConstant(_map_lmk_ptpar.at(landmark).values());

                        ceres::CostFunction *cost_fct =
                            ReprojectionErrCeres_pointxd::Create(feature->getPoints().at(0), cam, landmark->getPose());

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


                        ceres::CostFunction *cost_fct = 
                            ReprojectionErrCeres_linexd::Create(feature->getPoints(), 
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

uint BundleAdjustmentCERESNumeric::addLandmarkResiduals(ceres::Problem &problem,
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
                        ReprojectionErrCeres_pointxd::Create(feature->getPoints().at(0), cam, landmark->getPose());

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
                
                for (const std::weak_ptr<AFeature> &wfeature : featuresAssociatedLandmarks) {
                    std::shared_ptr<AFeature> feature = wfeature.lock();

                    if (!feature || !feature->getSensor()->getFrame()->isKeyFrame())
                        continue;

                    std::shared_ptr<ImageSensor> cam = feature->getSensor();
                    std::shared_ptr<Frame> frame = cam->getFrame();


                    if (_map_frame_posepar.find(frame) == _map_frame_posepar.end()) {
                        _map_frame_posepar.emplace(frame, PoseParametersBlock(Eigen::Affine3d::Identity()));
                        problem.AddParameterBlock(_map_frame_posepar.at(frame).values(), 6);
                        problem.SetParameterBlockConstant(_map_frame_posepar.at(frame).values());
                    }                    

                    ceres::CostFunction *cost_fct =
                        ReprojectionErrCeres_linexd::Create(feature->getPoints(),
                                                            cam,
                                                            frame->getWorld2FrameTransform(),
                                                            landmark->getModel(),
                                                            landmark->getScale());

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

uint BundleAdjustmentCERESNumeric::addResidualsLocalMap(ceres::Problem &problem,
                                                        ceres::LossFunction *loss_function,
                                                        ceres::ParameterBlockOrdering *ordering,
                                                        std::vector<std::shared_ptr<Frame>> &frame_vector,
                                                        size_t fixed_frame_number,
                                                        std::shared_ptr<isae::LocalMap> &local_map) {

    uint nb_residuals                     = 0;
    typed_vec_landmarks cloud_to_optimize = local_map->getLandmarks();

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
    }

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
                ordering->AddElementToGroup(_map_lmk_ptpar.at(landmark).values(), 0);

                // For all feature
                std::vector<std::weak_ptr<AFeature>> featuresAssociatedLandmarks = landmark->getFeatures();

                for (std::weak_ptr<AFeature> &wfeature : featuresAssociatedLandmarks) {
                    std::shared_ptr<AFeature> feature = wfeature.lock();

                    if (!feature || !feature->getSensor()->getFrame()->isKeyFrame()) {
                        continue;
                    }

                    std::shared_ptr<ImageSensor> cam = feature->getSensor();
                    std::shared_ptr<Frame> frame     = cam->getFrame();

                    nb_residuals++;

                    ceres::CostFunction *cost_fct =
                        ReprojectionErrCeres_pointxd::Create(feature->getPoints().at(0), cam, landmark->getPose());

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
                
                for (const std::weak_ptr<AFeature> &wfeature : featuresAssociatedLandmarks) {
                    std::shared_ptr<AFeature> feature = wfeature.lock();

                    if (!feature || !feature->getSensor()->getFrame()->isKeyFrame())
                        continue;

                    std::shared_ptr<ImageSensor> cam = feature->getSensor();
                    std::shared_ptr<Frame> frame = cam->getFrame();

                    nb_residuals++;

                    ceres::CostFunction *cost_fct = 
                    ReprojectionErrCeres_linexd::Create(feature->getPoints(),
                                                        cam,
                                                        frame->getWorld2FrameTransform(),
                                                        landmark->getModel(),
                                                        landmark->getScale());                    

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

} // namespace isae
