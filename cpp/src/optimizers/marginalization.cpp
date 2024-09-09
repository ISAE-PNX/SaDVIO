#include "isaeslam/optimizers/marginalization.hpp"

#include <mutex>
#include <thread>

namespace isae {

void MarginalizationBlockInfo::Evaluate() {

    _residuals.resize(_cost_function->num_residuals());

    std::vector<int> block_sizes = _cost_function->parameter_block_sizes();
    _raw_jacobians               = new double *[block_sizes.size()];
    _jacobians.resize(block_sizes.size());

    for (size_t i = 0; i < block_sizes.size(); i++) {
        _jacobians[i].resize(_cost_function->num_residuals(), block_sizes[i]);
        _raw_jacobians[i] = _jacobians[i].data();
    }
    _cost_function->Evaluate(_parameter_blocks.data(), _residuals.data(), _raw_jacobians);
}

void Marginalization::preMarginalize(std::shared_ptr<Frame> &frame0,
                                     std::shared_ptr<Frame> &frame1,
                                     std::shared_ptr<Marginalization> &marginalization_last) {
    // Reset variables
    _frame_to_marg = frame0;
    _frame_to_keep = nullptr;
    _lmk_to_keep.clear();
    _lmk_to_marg.clear();
    _map_frame_idx.clear();
    _map_lmk_idx.clear();
    _map_lmk_inf.clear();
    _map_lmk_prior.clear();
    _n = 0;
    _m = 0;

    // We add the frame to marginalize
    _map_frame_idx.emplace(_frame_to_marg, 0);
    _m           = 6;
    int last_idx = 6;

    // We add velocity and bias in the case of VIO
    if (frame0->getIMU()) {
        _m += 9;
        last_idx += 9;
    }

    // Distinguish lmk to marginalize and lmk to keep in landmarks linked to the frame to marginalize
    for (auto tlmks : _frame_to_marg->getLandmarks()) {
        // For all type of landmark
        for (auto lmk : tlmks.second) {

            if (lmk->isOutlier() || !lmk->isInMap() || !lmk->isInitialized())
                continue;

            bool is_lonely = true;
            int num_cam    = 0;
            for (auto f : lmk->getFeatures()) {

                // If the landmark is linked to other frames, it is kept
                if (f.lock()->getSensor()->getFrame() != frame0) {
                    is_lonely = false;

                    // We only include landmarks that have stereo factors (for matrix invertibility)
                } else {
                    num_cam += 1;
                }
            }

            // If the landmark has no prior and doesn't have full 3D information, it is ignored
            if (num_cam != 2 && !lmk->hasPrior()) {
                lmk->setMarg();
                continue;
            }

            // The lonely landmarks are marginalized, the other will have a prior
            if (!is_lonely) {
                lmk->setPrior();
                _lmk_to_keep[tlmks.first].push_back(lmk);
                // Index increment is 3 for pointxd, 6 for other
                (tlmks.first == "pointxd" ? _n += 3 : _n += 6);
            } else {
                _lmk_to_marg[tlmks.first].push_back(lmk);
                lmk->setMarg();
                (tlmks.first == "pointxd" ? _m += 3 : _m += 6);
            }
        }
    }

    // Fill map lmk idx for lmk to marg
    for (auto tlmks : _lmk_to_marg) {
        for (auto lmk : tlmks.second) {
            _map_lmk_idx.emplace(lmk, last_idx);
            (tlmks.first == "pointxd" ? last_idx += 3 : last_idx += 6);
        }
    }

    // We add the frame to keep in the case of VIO
    if (frame1->getIMU()) {
        _frame_to_keep = frame1;
        _map_frame_idx.emplace(_frame_to_keep, last_idx);
        _n += 15;
        last_idx += 15;
    }

    // Fill map lmk idx for lmk to keep
    for (auto tlmks : _lmk_to_keep) {
        for (auto lmk : tlmks.second) {
            _map_lmk_idx.emplace(lmk, last_idx);
            (tlmks.first == "pointxd" ? last_idx += 3 : last_idx += 6);
        }
    }

    // Add Landmarks from the last prior that are not linked to the current KF
    // (This is only for resurected landmarks)
    if (marginalization_last->_lmk_to_keep.empty())
        return;

    bool discard_prior = false;

    for (auto tlmks : marginalization_last->_lmk_to_keep) {
        // For all type of landmarks
        for (auto lmk : tlmks.second) {
            if (_map_lmk_idx.find(lmk) == _map_lmk_idx.end()) {

                // Outlier case: we remove the prior
                if (lmk->isOutlier()) {
                    discard_prior = true;
                    break;
                }

                _lmk_to_keep[tlmks.first].push_back(lmk);
                _map_lmk_idx.emplace(lmk, last_idx);
                (tlmks.first == "pointxd" ? _n += 3 : _n += 6);
                (tlmks.first == "pointxd" ? last_idx += 3 : last_idx += 6);
            }
        }
    }

    if (discard_prior)
        marginalization_last->_lmk_to_keep.clear();
}

void Marginalization::computeInformationAndGradient(std::vector<std::shared_ptr<MarginalizationBlockInfo>> blocks,
                                                    Eigen::MatrixXd &A,
                                                    Eigen::VectorXd &b) {
    std::mutex mtx;
    auto updateInfoMat = [&mtx, &A, &b](std::vector<std::shared_ptr<MarginalizationBlockInfo>> block_vector) {
        for (auto block : block_vector) {
            block->Evaluate();

            for (size_t i = 0; i < block->_parameter_blocks.size(); i++) {
                int size_i                 = block->_cost_function->parameter_block_sizes().at(i);
                int idx_i                  = block->_parameter_idx.at(i);
                Eigen::MatrixXd jacobian_i = block->_jacobians.at(i);

                // Outlier case, how to handle this?
                if (idx_i == -1)
                    continue;

                for (size_t j = i; j < block->_parameter_blocks.size(); j++) {
                    int size_j                 = block->_cost_function->parameter_block_sizes().at(j);
                    int idx_j                  = block->_parameter_idx.at(j);
                    Eigen::MatrixXd jacobian_j = block->_jacobians.at(j);

                    // Outlier case, how to handle this?
                    if (idx_j == -1)
                        continue;

                    {
                        mtx.lock();
                        if (i == j) {
                            A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                        } else {
                            A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                            A.block(idx_j, idx_i, size_j, size_i) =
                                A.block(idx_i, idx_j, size_i, size_j).transpose().eval();
                        }
                        mtx.unlock();
                    }
                }
                {
                    mtx.lock();
                    b.segment(idx_i, size_i) += jacobian_i.transpose() * block->_residuals;
                    mtx.unlock();
                }
            }
        }
    };

    // Split the blocks in chunks
    int n_thread = 4;
    std::vector<std::vector<std::shared_ptr<MarginalizationBlockInfo>>> thread_chunks;
    for (int i = 0; i < n_thread; i++) {
        std::vector<std::shared_ptr<MarginalizationBlockInfo>> block_vector;
        thread_chunks.push_back(block_vector);
    }
    for (uint i = 0; i < blocks.size(); i++) {
        thread_chunks.at(i % n_thread).push_back(blocks.at(i));
    }

    // Launch on different thread the local detections
    std::vector<std::thread> threads;
    for (auto block_vector : thread_chunks) {
        threads.push_back(std::thread(updateInfoMat, block_vector));
    }
    for (auto &th : threads) {
        th.join();
    }
}

bool Marginalization::computeSchurComplement() {

    if (_n < 4)
        return false;

    // Instanciate A and b
    Eigen::MatrixXd A(_m + _n, _m + _n);
    Eigen::VectorXd b(_m + _n);
    A.setZero();
    b.setZero();

    computeInformationAndGradient(_marginalization_blocks, A, b);

    // Free marginalization blocks for ceres
    for (size_t i = 0; i < _marginalization_blocks.size(); i++) {
        delete _marginalization_blocks[i]->_cost_function;
        delete _marginalization_blocks[i]->_raw_jacobians;
    }
    _marginalization_blocks.clear();

    // Schur Complement computation
    Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, _m, _m) + A.block(0, 0, _m, _m).transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);
    Eigen::MatrixXd Amm_inv =
        saes.eigenvectors() *
        Eigen::VectorXd((saes.eigenvalues().array() > _eps).select(saes.eigenvalues().array().inverse(), 0))
            .asDiagonal() *
        saes.eigenvectors().transpose();

    Eigen::VectorXd bmm = b.segment(0, _m);
    Eigen::MatrixXd Arm = A.block(_m, 0, _n, _m);
    Eigen::MatrixXd Arr = A.block(_m, _m, _n, _n);
    Eigen::VectorXd brr = b.segment(_m, _n);

    _Ak = Arr - Arm * Amm_inv * Arm.transpose();
    _bk = brr - Arm * Amm_inv * bmm;

    // Update the map index to apply the reduction
    for (auto lmk_idx : _map_lmk_idx) {
        _map_lmk_idx.at(lmk_idx.first) -= _m;
    }
    if (_frame_to_keep) {
        _map_frame_idx.at(_frame_to_keep) -= _m;
    }

    // Perform the rank revealling decomposition
    rankReveallingDecomposition(_Ak, _U, _Lambda);
    _Sigma   = _Lambda.array().inverse();
    _n_full  = _U.cols();
    _Sigma_k = _U * _Sigma.asDiagonal() * _U.transpose();

    return true;
}

double Marginalization::computeEntropy(std::shared_ptr<ALandmark> lmk) {

    int size;
    lmk->getLandmarkLabel() == "pointxd" ? size = 3 : size = 6;

    Eigen::MatrixXd Sigma = _Sigma_k.block(_map_lmk_idx.at(lmk), _map_lmk_idx.at(lmk), size, size);
    return std::log(std::pow(2 * M_PI * M_E, size / 2) * Sigma.determinant());
}

double Marginalization::computeMutualInformation(std::shared_ptr<ALandmark> lmk_i, std::shared_ptr<ALandmark> lmk_j) {

    // Retrieve the size of the landmarks
    int size_i;
    lmk_i->getLandmarkLabel() == "pointxd" ? size_i = 3 : size_i = 6;

    int size_j;
    lmk_j->getLandmarkLabel() == "pointxd" ? size_j = 3 : size_j = 6;

    // Extract submatrices for MI calculation
    Eigen::MatrixXd Sigma_ii = _Sigma_k.block(_map_lmk_idx.at(lmk_i), _map_lmk_idx.at(lmk_i), size_i, size_i);
    Eigen::MatrixXd Sigma_ij = _Sigma_k.block(_map_lmk_idx.at(lmk_i), _map_lmk_idx.at(lmk_j), size_i, size_j);
    Eigen::MatrixXd Sigma_ji = Sigma_ij.transpose();
    Eigen::MatrixXd Sigma_jj = _Sigma_k.block(_map_lmk_idx.at(lmk_j), _map_lmk_idx.at(lmk_j), size_j, size_j);

    // Build the sigma matrix for the denominator
    Eigen::MatrixXd Sigma(size_i + size_j, size_i + size_j);
    Sigma.topLeftCorner(size_i, size_i)     = Sigma_ii;
    Sigma.topRightCorner(size_i, size_j)    = Sigma_ij;
    Sigma.bottomLeftCorner(size_j, size_i)  = Sigma_ji;
    Sigma.bottomRightCorner(size_j, size_j) = Sigma_jj;

    // Compute MI
    double num   = Sigma_ii.determinant() * Sigma_jj.determinant();
    double denom = Sigma.determinant();
    return std::log(num / denom);
}

double Marginalization::computeOffDiag(std::shared_ptr<ALandmark> lmk_i, std::shared_ptr<ALandmark> lmk_j) {

    // Retrieve the size of the landmarks
    int size_i;
    lmk_i->getLandmarkLabel() == "pointxd" ? size_i = 3 : size_i = 6;

    int size_j;
    lmk_j->getLandmarkLabel() == "pointxd" ? size_j = 3 : size_j = 6;

    Eigen::MatrixXd Lambda_ij = _Ak.block(_map_lmk_idx.at(lmk_i), _map_lmk_idx.at(lmk_j), size_i, size_j);

    return std::abs(Lambda_ij.trace());
}

void Marginalization::rankReveallingDecomposition(Eigen::MatrixXd A, Eigen::MatrixXd &U, Eigen::VectorXd &d) {

    int n = A.rows();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(A);
    Eigen::VectorXd d_full = Eigen::VectorXd((saes.eigenvalues().array() > _eps).select(saes.eigenvalues().array(), 0));

    int j = 0;
    for (int i = 0; i < n; i++) {
        if (d_full(i) > _eps)
            j++;
    }

    U = Eigen::MatrixXd::Zero(n, j);
    d = Eigen::VectorXd::Zero(j);

    int k = 0;
    for (int i = 0; i < n; i++) {

        if (d_full(i) > _eps) {
            U.col(k) = saes.eigenvectors().col(i);
            d(k)     = d_full(i);
            k++;
        }
    }
}

double Marginalization::computeKLD(Eigen::MatrixXd A_p, Eigen::MatrixXd A_q) {

    Eigen::MatrixXd U;
    Eigen::VectorXd d;
    rankReveallingDecomposition(A_p, U, d);

    Eigen::VectorXd Sigma_p = d.array().inverse();
    Eigen::MatrixXd delta   = U.transpose() * A_q * U * Sigma_p.asDiagonal();
    double delta_det = delta.determinant();

    if (delta_det == 0) {
        delta = delta + 0.001 * Eigen::MatrixXd::Identity(delta.rows(), delta.rows());
        return 100;
    }

    return 0.5 * (delta.trace() - std::log(delta_det) - U.cols());
}

bool Marginalization::sparsifyVIO() {

    if (_n == 0)
        return false;

    // For pose to landmark factors
    Eigen::Affine3d T_f_w = _frame_to_keep->getWorld2FrameTransform();
    Eigen::Matrix3d R_f_w = T_f_w.rotation();
    Eigen::Matrix3d t_skew = geometry::skewMatrix(T_f_w.translation());
    for (auto tlmk : _lmk_to_keep) {
        for (auto lmk : tlmk.second) {
            Eigen::MatrixXd J                      = Eigen::MatrixXd::Zero(3, _n);
            J.block(0, _map_lmk_idx.at(lmk), 3, 3) = R_f_w;
            J.block(0, _map_frame_idx.at(_frame_to_keep), 3, 3) =
                -R_f_w * t_skew;
            J.block(0, _map_frame_idx.at(_frame_to_keep) + 3, 3, 3) = R_f_w;
            Eigen::MatrixXd J_tilde                                 = J * _U;

            Eigen::Matrix3d inf = (J_tilde * _Sigma.asDiagonal() * J_tilde.transpose()).inverse();
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(inf);
            Eigen::Vector3d S =
                Eigen::Vector3d((saes.eigenvalues().array() > _eps).select(saes.eigenvalues().array(), 0));
            Eigen::Vector3d S_sqrt   = S.cwiseSqrt();
            Eigen::Matrix3d inf_sqrt = saes.eigenvectors() * S_sqrt.asDiagonal() * saes.eigenvectors().transpose();

            _map_lmk_inf.emplace(lmk, inf_sqrt);
            Eigen::Vector3d t_f_lmk = T_f_w * lmk->getPose().translation();
            _map_lmk_prior.emplace(lmk, t_f_lmk);
        }
    }

    // For absolute frame factor
    Eigen::MatrixXd J                                       = Eigen::MatrixXd::Zero(15, _n);
    J.block(0, _map_frame_idx.at(_frame_to_keep), 15, 15)   = Eigen::MatrixXd::Identity(15, 15);
    J.block(0, _map_frame_idx.at(_frame_to_keep), 3, 3)     = T_f_w.rotation();
    J.block(0, _map_frame_idx.at(_frame_to_keep) + 3, 3, 3) = T_f_w.rotation();
    J.block(3, _map_frame_idx.at(_frame_to_keep) + 3, 3, 3) = T_f_w.rotation();
    Eigen::MatrixXd J_tilde                                 = J * _U;

    Eigen::MatrixXd inf = (J_tilde * _Sigma.asDiagonal() * J_tilde.transpose()).inverse();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(inf);
    Eigen::VectorXd S      = Eigen::VectorXd((saes.eigenvalues().array() > _eps).select(saes.eigenvalues().array(), 0));
    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::MatrixXd inf_sqrt = saes.eigenvectors() * S_sqrt.asDiagonal() * saes.eigenvectors().transpose();
    _map_frame_inf.emplace(_frame_to_keep, inf_sqrt);

    return true;
}

bool Marginalization::sparsifyVO() {

    if (_n == 0)
        return false;

    /// CHOW LIU TREE REORDERING ///

    // Compute a matrix with all MI values
    Eigen::MatrixXd mi_matrix = Eigen::MatrixXd::Zero(_lmk_to_keep["pointxd"].size(), _lmk_to_keep["pointxd"].size());
    for (uint k = 0; k < _lmk_to_keep["pointxd"].size(); k++) {
        for (uint l = 0; l < _lmk_to_keep["pointxd"].size(); l++) {

            // Skip the diagonal elements and if the matrix is already filled
            if (k == l || mi_matrix(k, l) != 0)
                continue;

            std::shared_ptr<ALandmark> lmk_k = _lmk_to_keep["pointxd"].at(k);
            std::shared_ptr<ALandmark> lmk_l = _lmk_to_keep["pointxd"].at(l);
            mi_matrix(k, l)                  = computeOffDiag(lmk_k, lmk_l);
            mi_matrix(l, k)                  = mi_matrix(k, l);
        }
    }

    // Hungarian algorithm to find the best combination
    std::vector<std::shared_ptr<ALandmark>> lmk_to_keep_ordered;
    lmk_to_keep_ordered.reserve(_lmk_to_keep["pointxd"].size());
    int max_row, max_col;

    // First couple
    mi_matrix.maxCoeff(&max_row, &max_col);
    lmk_to_keep_ordered.push_back(_lmk_to_keep["pointxd"].at(max_row));
    lmk_to_keep_ordered.push_back(_lmk_to_keep["pointxd"].at(max_col));

    // Set zero lines and cols of the first lmk and cols of the second
    mi_matrix.col(max_row).setZero();
    mi_matrix.row(max_row).setZero();
    mi_matrix.col(max_col).setZero();

    int current_idx = max_col;

    // Loop until the maximum is zero
    while (mi_matrix.row(current_idx).maxCoeff(&max_row, &max_col) != 0) {
        lmk_to_keep_ordered.push_back(_lmk_to_keep["pointxd"].at(max_col));

        // Set zero line of the first lmk and col of the second
        mi_matrix.row(current_idx).setZero();
        mi_matrix.col(max_col).setZero();

        current_idx = max_col;
    }

    _lmk_to_keep["pointxd"].clear();
    _lmk_to_keep["pointxd"] = lmk_to_keep_ordered;

    // Compute a vector of entropy
    Eigen::VectorXd entropys(_lmk_to_keep["pointxd"].size());
    for (uint k = 0; k < _lmk_to_keep["pointxd"].size(); k++) {
        entropys(k) = computeEntropy(_lmk_to_keep["pointxd"].at(k));
    }
    entropys.minCoeff(&max_row);
    _lmk_with_prior = _lmk_to_keep["pointxd"].at(max_row);

    _lmk_to_keep["pointxd"].clear();
    _lmk_to_keep["pointxd"] = lmk_to_keep_ordered;

    /// LANDMARK CHAIN ///

    // A unary factor for the first landmark and relative pose factors between lmk two by two
    Eigen::MatrixXd J                                  = Eigen::MatrixXd::Zero(3, _n);
    J.block(0, _map_lmk_idx.at(_lmk_with_prior), 3, 3) = Eigen::Matrix3d::Identity();
    Eigen::MatrixXd J_tilde                            = J * _U;

    Eigen::Matrix3d sig = (J_tilde * _Sigma.asDiagonal() * J_tilde.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes0(sig);
    Eigen::Vector3d S =
        Eigen::Vector3d((saes0.eigenvalues().array() > _eps).select(saes0.eigenvalues().array().inverse(), 0));
    Eigen::Vector3d S_sqrt   = S.cwiseSqrt();
    Eigen::Matrix3d inf_sqrt = saes0.eigenvectors() * S_sqrt.asDiagonal() * saes0.eigenvectors().transpose();

    _info_lmk  = inf_sqrt;
    _prior_lmk = _lmk_with_prior->getPose().translation();

    // relative pose factors between lmk two by two
    for (uint k = 0; k < _lmk_to_keep["pointxd"].size() - 1; k++) {
        std::shared_ptr<ALandmark> lmk_k   = _lmk_to_keep["pointxd"].at(k);
        std::shared_ptr<ALandmark> lmk_kp1 = _lmk_to_keep["pointxd"].at(k + 1);

        Eigen::MatrixXd J                          = Eigen::MatrixXd::Zero(3, _n);
        J.block(0, _map_lmk_idx.at(lmk_k), 3, 3)   = Eigen::Matrix3d::Identity();
        J.block(0, _map_lmk_idx.at(lmk_kp1), 3, 3) = -Eigen::Matrix3d::Identity();
        Eigen::MatrixXd J_tilde                    = J * _U;

        Eigen::Matrix3d sig = J_tilde * _Sigma.asDiagonal() * J_tilde.transpose();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes1(sig);
        Eigen::Vector3d S =
            Eigen::Vector3d((saes1.eigenvalues().array() > _eps).select(saes1.eigenvalues().array().inverse(), 0));
        Eigen::Vector3d S_sqrt   = S.cwiseSqrt();
        Eigen::Matrix3d inf_sqrt = saes1.eigenvectors() * S_sqrt.asDiagonal() * saes1.eigenvectors().transpose();

        _map_lmk_inf.emplace(lmk_kp1, inf_sqrt);
        _map_lmk_prior.emplace(lmk_kp1, lmk_k->getPose().translation() - lmk_kp1->getPose().translation());
    }

    return true;
}

bool Marginalization::computeJacobiansAndResiduals() {

    Eigen::VectorXd Lambda_sqrt = _Lambda.cwiseSqrt();
    Eigen::VectorXd Sigma_sqrt  = _Sigma.cwiseSqrt();

    // I = U Lambda U^T = J^T J
    // Gauss Newton: I dx = - J^T r = g
    // => J = Lambda^{1/2} U^T
    // => - U Lambda^{1/2} r = g <=> r = -Lambda{-1/2} U^T g

    _marginalization_jacobian = Lambda_sqrt.asDiagonal() * _U.transpose();
    _marginalization_residual = (-1) * Sigma_sqrt.asDiagonal() * _U.transpose() * _bk;

    return true;
}

void Marginalization::preMarginalizeRelative(std::shared_ptr<Frame> &frame0, std::shared_ptr<Frame> &frame1) {

    // Reset variables
    _frame_to_marg = nullptr;
    _frame_to_keep = nullptr;
    _lmk_to_keep.clear();
    _lmk_to_marg.clear();
    _map_frame_idx.clear();
    _map_lmk_idx.clear();
    _map_lmk_inf.clear();
    _map_lmk_prior.clear();

    // We keep only the two poses 
    _n = 12;
    _map_frame_idx.emplace(frame0, 0);
    _map_frame_idx.emplace(frame1, 6);
    int last_idx = 12;
    _m           = 0;

    // Set to marginalize bias and velocity in VIO case
    if (frame0->getIMU()) {
        _m += 18;
        last_idx += 18;
    }

    // Set to marginalize all the common landamrks between the frames
    for (auto tlmks : frame0->getLandmarks()) {

        // For all type of landmark
        for (auto lmk : tlmks.second) {

            if (lmk->isOutlier() || !lmk->isInMap() || !lmk->isInitialized())
                continue;
            for (auto f : lmk->getFeatures()) {

                // If the landmark is linked to frame1 it is marginalized
                if (f.lock()->getSensor()->getFrame() == frame1) {

                    _lmk_to_marg[tlmks.first].push_back(lmk);
                    lmk->setMarg();
                    (tlmks.first == "pointxd" ? _m += 3 : _m += 6);
                    _map_lmk_idx.emplace(lmk, last_idx);
                    (tlmks.first == "pointxd" ? last_idx += 3 : last_idx += 6);

                } else {
                    continue;
                }
            }
        }
    }
}

} // namespace isae