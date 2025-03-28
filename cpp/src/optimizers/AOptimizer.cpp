#include "isaeslam/optimizers/AOptimizer.h"

namespace isae {

bool AOptimizer::isMovingFrame(const std::shared_ptr<isae::Frame> &frame,
                               const std::vector<std::shared_ptr<isae::Frame>> &frame_vector) {

    for (auto &f : frame_vector)
        if (f == frame)
            return true;
    return false;
}

bool AOptimizer::isMovingLandmark(const std::shared_ptr<isae::ALandmark> &ldmk,
                                  const std::vector<std::shared_ptr<isae::ALandmark>> &cloud_to_optimize) {
    for (auto &l : cloud_to_optimize)
        if (l == ldmk)
            return true;
    return false;
}

uint AOptimizer::addIMUResiduals(ceres::Problem &problem,
                                 ceres::LossFunction *loss_function,
                                 ceres::ParameterBlockOrdering *ordering,
                                 std::vector<std::shared_ptr<Frame>> &frame_vector,
                                 size_t fixed_frame_number) {

    // Add parameter blocks specific to IMU (we suppose that the parameter blocks for pose were already added)
    for (size_t i = 0; i < frame_vector.size(); i++) {
        if (frame_vector.at(i)->getIMU()) {
            _map_frame_velpar.emplace(frame_vector.at(i), PointXYZParametersBlock(Eigen::Vector3d::Zero()));
            _map_frame_dbapar.emplace(frame_vector.at(i), PointXYZParametersBlock(Eigen::Vector3d::Zero()));
            _map_frame_dbgpar.emplace(frame_vector.at(i), PointXYZParametersBlock(Eigen::Vector3d::Zero()));

            problem.AddParameterBlock(_map_frame_velpar.at(frame_vector.at(i)).values(), 3);
            ordering->AddElementToGroup(_map_frame_velpar.at(frame_vector.at(i)).values(), 1);

            problem.AddParameterBlock(_map_frame_dbapar.at(frame_vector.at(i)).values(), 3);
            // problem.SetParameterBlockConstant(_map_frame_dbapar.at(frame_vector.at(i)).values());
            ordering->AddElementToGroup(_map_frame_dbapar.at(frame_vector.at(i)).values(), 1);

            problem.AddParameterBlock(_map_frame_dbgpar.at(frame_vector.at(i)).values(), 3);
            // problem.SetParameterBlockConstant(_map_frame_dbgpar.at(frame_vector.at(i)).values());
            ordering->AddElementToGroup(_map_frame_dbgpar.at(frame_vector.at(i)).values(), 1);

            // Set parameter block constant for fixed frames
            if ((int)i > (int)(frame_vector.size() - fixed_frame_number - 1)) {
                problem.SetParameterBlockConstant(_map_frame_velpar.at(frame_vector.at(i)).values());
                problem.SetParameterBlockConstant(_map_frame_dbapar.at(frame_vector.at(i)).values());
                problem.SetParameterBlockConstant(_map_frame_dbgpar.at(frame_vector.at(i)).values());
            }
        }
    }

    for (size_t i = 0; i < frame_vector.size(); i++) {

        std::shared_ptr<Frame> framej = frame_vector.at(i);

        // Skip if no IMU
        if (!framej->getIMU())
            continue;
        std::shared_ptr<Frame> framei = framej->getIMU()->getLastKF();

        // Skip if framei == nullptr
        if (!framei)
            continue;

        // If dt > 1 ignore IMU measurement (TO DO: marginalize IMU measurement only to get a proper relative factor?)
        if ((framej->getTimestamp() - framei->getTimestamp()) * 1e-9 > 1)
            continue;

        if (_map_frame_velpar.find(framei) != _map_frame_velpar.end() && framei != framej) {

            // add IMU factor
            ceres::CostFunction *cost_fct = new IMUFactor(framei->getIMU(), framej->getIMU());
            problem.AddResidualBlock(cost_fct,
                                     loss_function,
                                     _map_frame_posepar.at(framei).values(),
                                     _map_frame_posepar.at(framej).values(),
                                     _map_frame_velpar.at(framei).values(),
                                     _map_frame_velpar.at(framej).values(),
                                     _map_frame_dbapar.at(framei).values(),
                                     _map_frame_dbgpar.at(framei).values());

            // add Bias random walk factor
            ceres::CostFunction *cost_fct_bias = new IMUBiasFactor(framei->getIMU(), framej->getIMU());
            problem.AddResidualBlock(cost_fct_bias,
                                     loss_function,
                                     _map_frame_dbapar.at(framei).values(),
                                     _map_frame_dbgpar.at(framei).values(),
                                     _map_frame_dbapar.at(framej).values(),
                                     _map_frame_dbgpar.at(framej).values());
        }
    }
    return 0;
}

bool AOptimizer::landmarkOptimization(std::shared_ptr<Frame> &frame) {

    // Build the Bundle Adjustement Problem
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(std::sqrt(1.345));

    // Get point cloud to be optimized
    typed_vec_landmarks cloud_to_optimize = frame->getLandmarks();
    addLandmarkResiduals(problem, loss_function, cloud_to_optimize);

    // Solve the problem we just built
    ceres::Solver::Options options;
    options.trust_region_strategy_type         = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type                 = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations                 = 10;
    options.minimizer_progress_to_stdout       = false;
    options.use_explicit_schur_complement      = true;
    options.function_tolerance                 = 1e-3;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.num_threads                        = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::mutex opti_mtx;
    std::lock_guard<std::mutex> lock(opti_mtx);

    // Chi2 test
    for (auto &ldmk_list : cloud_to_optimize) {
        for (auto &ldmk : ldmk_list.second) {

            // Same check as in AddResiduals
            if (!ldmk->isInitialized() || ldmk->isOutlier())
                continue;

            bool can_be_updated = ldmk->sanityCheck(); // Only inliers can be updated

            if (can_be_updated) {
                if (ldmk_list.first == "pointxd") {
                    ldmk->setPose(ldmk->getPose() * _map_lmk_ptpar.at(ldmk).getPose());
                } else {
                    ldmk->setPose(ldmk->getPose() * _map_lmk_posepar.at(ldmk).getPose());
                }
            }
        }
    }

    // Clear maps for bookeeping
    _map_frame_posepar.clear();
    _map_lmk_ptpar.clear();
    _map_lmk_posepar.clear();

    return true;
}

bool AOptimizer::singleFrameOptimization(std::shared_ptr<isae::Frame> &moving_frame) {

    // Build the Bundle Adjustement Problem
    ceres::Problem problem;
    ceres::LossFunction *loss_function = nullptr;

    // Get point cloud to be optimized
    typed_vec_landmarks cloud_to_optimize = moving_frame->getLandmarks();

    // Add residuals
    addSingleFrameResiduals(problem, loss_function, moving_frame, cloud_to_optimize);

    // Solve the problem we just built
    ceres::Solver::Options options;
    options.trust_region_strategy_type         = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type                 = ceres::SPARSE_NORMAL_CHOLESKY; // SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations                 = 5;
    options.minimizer_progress_to_stdout       = false;
    options.use_explicit_schur_complement      = true;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.function_tolerance                 = 1.e-3;
    options.num_threads                        = 1;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Covariance Estimation
    // ceres::Covariance::Options options_cov;
    // ceres::Covariance cov(options_cov);

    // // Select Covariance block
    // std::vector<std::pair<const double*, const double*> > covariance_blocks;
    // for (auto &frame_posepar : _map_frame_posepar) {
    //     covariance_blocks.push_back({frame_posepar.second.values(), frame_posepar.second.values()});
    // }

    // // Compute Covariance
    // cov.Compute(covariance_blocks, &problem);

    // // std::cout << "---" << std::endl;

    // // Display
    // for (auto &frame_posepar : _map_frame_posepar) {
    //     double cov_pp[6 * 6];
    //     cov.GetCovarianceBlock(frame_posepar.second.values(), frame_posepar.second.values(), cov_pp);

    //     Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> cov_mat_map(cov_pp);
    //     // std::cout << cov_mat_map << std::endl;
    // }


    // Update state
    for (auto &frame_posepar : _map_frame_posepar) {
        frame_posepar.first->setWorld2FrameTransform(frame_posepar.first->getWorld2FrameTransform() *
                                                     frame_posepar.second.getPose());
    }

    // Set maps for bookeeping
    _map_frame_posepar.clear();
    _map_lmk_ptpar.clear();
    _map_lmk_posepar.clear();

    // std::cout << summary.FullReport() << std::endl;

    return true;
}

bool AOptimizer::singleFrameVIOptimization(std::shared_ptr<isae::Frame> &moving_frame) {

    // Build the Bundle Adjustement Problem
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(std::sqrt(1.345));

    // Get point cloud to be optimized
    typed_vec_landmarks cloud_to_optimize = moving_frame->getLandmarks();

    // Add visual residuals
    addSingleFrameResiduals(problem, loss_function, moving_frame, cloud_to_optimize);

    // Add IMU residuals
    std::vector<std::shared_ptr<Frame>> frame_vec;
    if (moving_frame->getIMU() && moving_frame->getIMU()->getLastKF()->getIMU()) {
        frame_vec.push_back(moving_frame);
        frame_vec.push_back(moving_frame->getIMU()->getLastKF());
        std::shared_ptr<Frame> frame = moving_frame->getIMU()->getLastKF();

        addSingleFrameResiduals(problem, loss_function, frame, cloud_to_optimize);

        auto ordering = new ceres::ParameterBlockOrdering;
        addIMUResiduals(problem, nullptr, ordering, frame_vec, 0);
    }

    // Solve the problem we just built
    ceres::Solver::Options options;
    options.trust_region_strategy_type         = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type                 = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations                 = 5;
    options.minimizer_progress_to_stdout       = false;
    options.use_explicit_schur_complement      = true;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.function_tolerance                 = 1.e-3;
    options.num_threads                        = 1;
    options.max_solver_time_in_seconds         = 0.005;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (!summary.IsSolutionUsable())
        return false;

    // Update state
    for (auto &frame_posepar : _map_frame_posepar) {
        frame_posepar.first->setWorld2FrameTransform(frame_posepar.first->getWorld2FrameTransform() *
                                                     frame_posepar.second.getPose());
    }

    // For IMU
    if (moving_frame->getIMU() && moving_frame->getIMU()->getLastKF()->getIMU()) {
        for (auto &frame_velpar : _map_frame_velpar) {
            frame_velpar.first->getIMU()->setVelocity(frame_velpar.first->getIMU()->getVelocity() +
                                                      frame_velpar.second.getPose().translation());
        }

        for (auto &frame_dbapar : _map_frame_dbapar) {
            frame_dbapar.first->getIMU()->setBa(frame_dbapar.first->getIMU()->getBa() +
                                                frame_dbapar.second.getPose().translation());
        }

        for (auto &frame_dbgpar : _map_frame_dbgpar) {
            frame_dbgpar.first->getIMU()->setBg(frame_dbgpar.first->getIMU()->getBg() +
                                                frame_dbgpar.second.getPose().translation());
        }
    }

    // Set maps for bookeeping
    _map_frame_posepar.clear();
    _map_lmk_ptpar.clear();
    _map_lmk_posepar.clear();
    _map_frame_velpar.clear();
    _map_frame_dbapar.clear();
    _map_frame_dbgpar.clear();

    // std::cout << summary.FullReport() << std::endl;

    return true;
}

bool AOptimizer::localMapBA(std::shared_ptr<isae::LocalMap> &local_map, const size_t fixed_sized_number) {

    // Build the Bundle Adjustement Problem
    ceres::Problem problem;
    ceres::LossFunction *loss_function = nullptr;

    // Get all moving frames
    std::vector<std::shared_ptr<isae::Frame>> frame_vector;
    local_map->getLastNFramesIn(local_map->getMapSize(), frame_vector);

    // Add residuals
    auto ordering = new ceres::ParameterBlockOrdering;
    addResidualsLocalMap(problem, loss_function, ordering, frame_vector, fixed_sized_number, local_map);
    addMarginalizationResiduals(problem, loss_function, ordering);

    // Solve the problem we just built
    ceres::Solver::Options options;
    options.linear_solver_ordering.reset(ordering);
    options.trust_region_strategy_type         = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type                 = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations                 = 20;
    options.minimizer_progress_to_stdout       = false;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.function_tolerance                 = 1.e-3;
    options.num_threads                        = 4;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Update states
    for (auto &frame_posepar : _map_frame_posepar) {
        frame_posepar.first->setWorld2FrameTransform(frame_posepar.first->getWorld2FrameTransform() *
                                                     frame_posepar.second.getPose());
    }

    for (auto &lmk_posepar : _map_lmk_posepar) {
        lmk_posepar.first->setPose(lmk_posepar.first->getPose() * lmk_posepar.second.getPose());
    }

    for (auto &lmk_ptpar : _map_lmk_ptpar) {
        lmk_ptpar.first->setPose(lmk_ptpar.first->getPose() * lmk_ptpar.second.getPose());
    }

    // Clear maps for bookeeping
    _map_frame_posepar.clear();
    _map_lmk_ptpar.clear();
    _map_lmk_posepar.clear();

    // std::cout << summary.FullReport() << std::endl;

    return true;
}

bool AOptimizer::localMapVIOptimization(std::shared_ptr<isae::LocalMap> &local_map, const size_t fixed_sized_number) {

    // Set maps for bookeeping;
    _map_lmk_ptpar.clear();
    _map_frame_posepar.clear();
    _map_frame_velpar.clear();
    _map_frame_dbapar.clear();
    _map_frame_dbgpar.clear();

    // Build the Bundle Adjustement Problem
    ceres::Problem problem;
    ceres::LossFunction *loss_function = nullptr;

    // Get all moving frames
    std::vector<std::shared_ptr<isae::Frame>> frame_vector;
    local_map->getLastNFramesIn(local_map->getMapSize(), frame_vector);

    // Add residuals
    auto ordering = new ceres::ParameterBlockOrdering;
    addResidualsLocalMap(problem, loss_function, ordering, frame_vector, fixed_sized_number, local_map);
    addIMUResiduals(problem, loss_function, ordering, frame_vector, fixed_sized_number);
    addMarginalizationResiduals(problem, loss_function, ordering);

    // Solve the problem we just built
    ceres::Solver::Options options;
    options.linear_solver_ordering.reset(ordering);
    options.trust_region_strategy_type         = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type                 = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations                 = 20;
    options.minimizer_progress_to_stdout       = false;
    options.use_explicit_schur_complement      = true;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.function_tolerance                 = 1.e-3;
    options.num_threads                        = 4;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Update state
    for (auto &frame_posepar : _map_frame_posepar) {
        frame_posepar.first->setWorld2FrameTransform(frame_posepar.first->getWorld2FrameTransform() *
                                                     frame_posepar.second.getPose());
    }

    for (auto &lmk_posepar : _map_lmk_posepar) {
        lmk_posepar.first->setPose(lmk_posepar.first->getPose() * lmk_posepar.second.getPose());
    }

    for (auto &lmk_ptpar : _map_lmk_ptpar) {
        lmk_ptpar.first->setPose(lmk_ptpar.first->getPose() * lmk_ptpar.second.getPose());
    }

    // For IMU
    for (auto &frame_velpar : _map_frame_velpar) {
        frame_velpar.first->getIMU()->setVelocity(frame_velpar.first->getIMU()->getVelocity() +
                                                  frame_velpar.second.getPose().translation());
    }

    for (auto &frame_dbapar : _map_frame_dbapar) {
        frame_dbapar.first->getIMU()->setBa(frame_dbapar.first->getIMU()->getBa() +
                                            frame_dbapar.second.getPose().translation());
    }

    for (auto &frame_dbgpar : _map_frame_dbgpar) {
        frame_dbgpar.first->getIMU()->setBg(frame_dbgpar.first->getIMU()->getBg() +
                                            frame_dbgpar.second.getPose().translation());
    }

    // Update deltas with IMU biases
    for (auto &frame : frame_vector) {
        if (!frame->getIMU())
            continue;

        if (!frame->getIMU()->getLastKF())
            continue;

        std::shared_ptr<Frame> previous_frame = frame->getIMU()->getLastKF();

        if (_map_frame_dbapar.find(previous_frame) != _map_frame_dbapar.end()) {
            frame->getIMU()->biasDeltaCorrection(_map_frame_dbapar.at(previous_frame).getPose().translation(),
                                                 _map_frame_dbgpar.at(previous_frame).getPose().translation());
        }
    }

    // Set maps for bookeeping;
    _map_lmk_ptpar.clear();
    _map_frame_posepar.clear();
    _map_frame_velpar.clear();
    _map_frame_dbapar.clear();
    _map_frame_dbgpar.clear();

    // std::cout << summary.FullReport() << std::endl;

    return true;
}

double AOptimizer::VIInit(std::shared_ptr<isae::LocalMap> &local_map, Eigen::Matrix3d &R_w_i, bool optim_scale) {
    int steps = 50;

    // Build the Bundle Adjustement Problem
    ceres::Problem problem;
    ceres::LossFunction *loss_function = nullptr;

    // Get all moving frames
    std::vector<std::shared_ptr<isae::Frame>> frame_vector;
    local_map->getLastNFramesIn(local_map->getMapSize(), frame_vector);

    // Add parameter blocks for the velocity of the IMU
    for (auto frame : frame_vector) {
        if (frame->getIMU()) {
            _map_frame_velpar.emplace(frame, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
        }
    }

    // Parameter block of the gravity direction
    double r_wi_par[2] = {0.0, 0.0};
    problem.AddParameterBlock(r_wi_par, 2);
    // problem.SetParameterBlockConstant(r_wi_par);

    // Parameter blocks of the delta bias (assumed constant on this sliding window)
    PointXYZParametersBlock dba_par = PointXYZParametersBlock(Eigen::Vector3d(0, 0, 0));
    PointXYZParametersBlock dbg_par = PointXYZParametersBlock(Eigen::Vector3d(0, 0, 0));
    problem.AddParameterBlock(dba_par.values(), 3);
    problem.SetParameterBlockConstant(dba_par.values());
    problem.AddParameterBlock(dbg_par.values(), 3);
    problem.SetParameterBlockConstant(dbg_par.values());

    // Parameter block of the scale, that is set to 0 as it goes in an exponential
    double lambda[1] = {0.0};
    problem.AddParameterBlock(lambda, 1);
    if (!optim_scale)
        problem.SetParameterBlockConstant(lambda);

    for (auto framej : frame_vector) {

        std::shared_ptr<Frame> framei = framej->getIMU()->getLastKF();

        if (_map_frame_velpar.find(framei) != _map_frame_velpar.end() && framei != framej) {

            // add IMU factor
            ceres::CostFunction *cost_fct = new IMUFactorInit(framei->getIMU(), framej->getIMU());
            problem.AddResidualBlock(cost_fct,
                                     loss_function,
                                     r_wi_par,
                                     _map_frame_velpar.at(framei).values(),
                                     _map_frame_velpar.at(framej).values(),
                                     dba_par.values(),
                                     dbg_par.values(),
                                     lambda);
        }
    }

    // add Bias prior
    double dt        = frame_vector.at(0)->getTimestamp() - frame_vector.at(frame_vector.size() - 1)->getTimestamp();
    double sigma_dba = std::sqrt(dt) * frame_vector.at(0)->getIMU()->getbAccNoise();
    Eigen::Matrix3d sqrt_inf_ba = Eigen::Matrix3d::Identity() * (1 / sigma_dba);
    ceres::CostFunction *cost_fct_ba =
        new Landmark3DPrior(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), sqrt_inf_ba);
    problem.AddResidualBlock(cost_fct_ba, loss_function, dba_par.values());
    double sigma_dbg            = std::sqrt(dt) * frame_vector.at(0)->getIMU()->getbGyrNoise();
    Eigen::Matrix3d sqrt_inf_bg = Eigen::Matrix3d::Identity() * (1 / sigma_dbg);
    ceres::CostFunction *cost_fct_bg =
        new Landmark3DPrior(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), sqrt_inf_bg);
    problem.AddResidualBlock(cost_fct_bg, loss_function, dbg_par.values());

    // Solve the problem we just built
    ceres::Solver::Options options;
    options.trust_region_strategy_type         = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type                 = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations                 = steps;
    options.minimizer_progress_to_stdout       = false;
    options.use_explicit_schur_complement      = true;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.function_tolerance                 = 1.e-3;
    options.num_threads                        = 4;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Update IMU velocity and biases
    for (auto &frame_velpar : _map_frame_velpar) {
        frame_velpar.first->getIMU()->setVelocity(
            (frame_velpar.first->getIMU()->getVelocity() + frame_velpar.second.getPose().translation()));
        frame_velpar.first->getIMU()->setBa(frame_velpar.first->getIMU()->getBa() + dba_par.getPose().translation());
        frame_velpar.first->getIMU()->setBa(frame_velpar.first->getIMU()->getBa() + dba_par.getPose().translation());
    }

    // Update gravity direction with only 2DoF
    R_w_i = geometry::exp_so3(Eigen::Vector3d(r_wi_par[0], r_wi_par[1], 0));

    // Update frame poses
    Eigen::Affine3d T_w_i            = Eigen::Affine3d::Identity();
    T_w_i.affine().block(0, 0, 3, 3) = R_w_i;
    for (auto &frame : frame_vector) {

        // Apply scale + rotate on inertial frame
        Eigen::Affine3d T_f_w = frame->getWorld2FrameTransform();
        T_f_w.translation() *= std::exp(lambda[0]);
        T_f_w = T_f_w * T_w_i;
        frame->setWorld2FrameTransform(T_f_w);

        if (frame->hasPrior()) {
            frame->setPrior(frame->getWorld2FrameTransform(), 100 * Vector6d::Ones());
        }
    }

    // Update landmarks
    for (auto &tlandmark : local_map->getLandmarks()) {
        for (auto landmark : tlandmark.second) {
            if (!landmark->isOutlier()) {
                Eigen::Affine3d T_w_lmk = T_w_i.inverse() * landmark->getPose();
                T_w_lmk.translation() *= std::exp(lambda[0]);
                landmark->setPose(T_w_lmk);
            }
        }
    }

    std::cout << summary.FullReport() << std::endl;
    std::cout << "Scale : " << std::exp(lambda[0]) << std::endl;

    // Set maps for bookeeping
    _map_frame_posepar.clear();
    _map_lmk_ptpar.clear();
    _map_lmk_posepar.clear();
    _map_frame_velpar.clear();
    _map_frame_dbapar.clear();
    _map_frame_dbgpar.clear();

    return std::exp(lambda[0]) ;
}

// To be implemented in dedicated solvers
bool AOptimizer::landmarkOptimizationNoFov(std::shared_ptr<Frame> &f,
                                           std::shared_ptr<Frame> &fp,
                                           Eigen::Affine3d &T_cam0_cam0p,
                                           double info_scale) {
    return true;
}

} // namespace isae