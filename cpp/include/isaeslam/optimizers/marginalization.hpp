#include <ceres/ceres.h>
#include <unordered_map>

#include "isaeslam/data/frame.h"
#include "isaeslam/data/landmarks/ALandmark.h"
#include "isaeslam/data/maps/localmap.h"
#include "isaeslam/typedefs.h"

namespace isae {

// Marginalization block that stores a factor and the indices of the variables involved
struct MarginalizationBlockInfo {

    MarginalizationBlockInfo(ceres::CostFunction *cost_function,
                             std::vector<int> parameter_idx,
                             std::vector<double *> parameter_blocks)
        : _cost_function(cost_function), _parameter_idx(parameter_idx), _parameter_blocks(parameter_blocks) {}

    void Evaluate();

    ceres::CostFunction *_cost_function;
    std::vector<int> _parameter_idx;
    std::vector<double *> _parameter_blocks;

    double **_raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> _jacobians;
    Eigen::VectorXd _residuals;
};

class Marginalization {
  public:
    // Select all the variables to keep and to marginalize in the Markov Blanket
    void preMarginalize(std::shared_ptr<Frame> &frame0,
                        std::shared_ptr<Frame> &frame1,
                        std::shared_ptr<Marginalization> &marginalization_last);
    
    // Select the variables to marginalize to compute relative pose factors
    void preMarginalizeRelative(std::shared_ptr<Frame> &frame0, std::shared_ptr<Frame> &frame1);

    void addMarginalizationBlock(std::shared_ptr<MarginalizationBlockInfo> marginalization_block) {
        _marginalization_blocks.push_back(marginalization_block);
    }
    bool sparsifyVIO();
    bool sparsifyVO();
    void computeInformationAndGradient(std::vector<std::shared_ptr<MarginalizationBlockInfo>> blocks,
                                       Eigen::MatrixXd &A,
                                       Eigen::VectorXd &b);
    void rankReveallingDecomposition(Eigen::MatrixXd A, Eigen::MatrixXd &U, Eigen::VectorXd &d);
    bool computeSchurComplement();
    bool computeJacobiansAndResiduals();
    double computeEntropy(std::shared_ptr<ALandmark> lmk);
    double computeMutualInformation(std::shared_ptr<ALandmark> lmk_i, std::shared_ptr<ALandmark> lmk_j);
    double computeOffDiag(std::shared_ptr<ALandmark> lmk_i, std::shared_ptr<ALandmark> lmk_j);
    double computeKLD(Eigen::MatrixXd A_p, Eigen::MatrixXd A_q);

    // Marginalization info
    int _m, _n, _n_full;
    const double _eps = 1e-12;

    // Bookeeping of the variables to keep and to marginalize
    std::shared_ptr<Frame> _frame_to_marg;
    std::shared_ptr<Frame> _frame_to_keep;
    typed_vec_landmarks _lmk_to_keep;
    typed_vec_landmarks _lmk_to_marg;
    std::unordered_map<std::shared_ptr<Frame>, int> _map_frame_idx;
    std::unordered_map<std::shared_ptr<ALandmark>, int> _map_lmk_idx;
    std::unordered_map<std::shared_ptr<Frame>, Eigen::MatrixXd> _map_frame_inf;
    std::vector<std::shared_ptr<MarginalizationBlockInfo>> _marginalization_blocks;

    // Sparsification info
    std::unordered_map<std::shared_ptr<ALandmark>, Eigen::Matrix3d> _map_lmk_inf;
    std::unordered_map<std::shared_ptr<ALandmark>, Eigen::Vector3d> _map_lmk_prior;
    std::shared_ptr<ALandmark> _lmk_with_prior;
    Eigen::Matrix3d _info_lmk;
    Eigen::Vector3d _prior_lmk;

    // Matrices and vectors of the dense prior
    Eigen::MatrixXd _Ak;
    Eigen::VectorXd _bk;
    Eigen::MatrixXd _Sigma_k;
    Eigen::MatrixXd _U;
    Eigen::VectorXd _Lambda;
    Eigen::VectorXd _Sigma;
    Eigen::MatrixXd _marginalization_jacobian;
    Eigen::VectorXd _marginalization_residual;
};

class MarginalizationFactor : public ceres::CostFunction {
  public:
    MarginalizationFactor(std::shared_ptr<Marginalization> marginalization_info)
        : _marginalization_info(marginalization_info) {

        // Add frame block size
        if (_marginalization_info->_frame_to_keep) {
            this->mutable_parameter_block_sizes()->push_back(6);
            this->mutable_parameter_block_sizes()->push_back(3);
            this->mutable_parameter_block_sizes()->push_back(3);
            this->mutable_parameter_block_sizes()->push_back(3);
        }

        // Add landmark block size
        for (auto tlmk : _marginalization_info->_lmk_to_keep) {
            for (auto lmk : tlmk.second) {
                (tlmk.first == "pointxd" ? this->mutable_parameter_block_sizes()->push_back(3)
                                         : this->mutable_parameter_block_sizes()->push_back(6));
            }
        }

        // Set the number of residuals
        this->set_num_residuals(_marginalization_info->_n_full);
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        int n = _marginalization_info->_n_full;
        Eigen::VectorXd dx(_marginalization_info->_n);
        dx.setZero();

        // Add frame dx
        int block_id = 0;
        if (_marginalization_info->_frame_to_keep) {
            dx.segment<6>(_marginalization_info->_map_frame_idx.at(_marginalization_info->_frame_to_keep)) =
                Eigen::Map<const Eigen::Matrix<double, 6, 1>>(parameters[block_id]);
            block_id++;
            dx.segment<3>(_marginalization_info->_map_frame_idx.at(_marginalization_info->_frame_to_keep) + 6) =
                Eigen::Map<const Eigen::Vector3d>(parameters[block_id]);
            block_id++;
            dx.segment<3>(_marginalization_info->_map_frame_idx.at(_marginalization_info->_frame_to_keep) + 9) =
                Eigen::Map<const Eigen::Vector3d>(parameters[block_id]);
            block_id++;
            dx.segment<3>(_marginalization_info->_map_frame_idx.at(_marginalization_info->_frame_to_keep) + 12) =
                Eigen::Map<const Eigen::Vector3d>(parameters[block_id]);
            block_id++;
        }

        // Add landmarks dx
        for (auto tlmk : _marginalization_info->_lmk_to_keep) {
            for (auto lmk : tlmk.second) {
                if (_marginalization_info->_map_lmk_idx.at(lmk) == -1)
                    continue;

                dx.segment<3>(_marginalization_info->_map_lmk_idx.at(lmk)) =
                    Eigen::Map<const Eigen::Vector3d>(parameters[block_id]);
                block_id++;
            }
        }

        // Compute the residual
        Eigen::Map<Eigen::VectorXd>(residuals, n) =
            _marginalization_info->_marginalization_residual + _marginalization_info->_marginalization_jacobian * dx;

        // Fill the jacobians
        if (jacobians) {

            block_id = 0;

            // Jacobians on frame
            if (_marginalization_info->_frame_to_keep) {
                if (jacobians[block_id]) {

                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(
                        jacobians[block_id], n, 6);
                    jacobian.setZero();
                    jacobian.leftCols(6) = _marginalization_info->_marginalization_jacobian.middleCols(
                        _marginalization_info->_map_frame_idx.at(_marginalization_info->_frame_to_keep), 6);
                }
                block_id++;
                if (jacobians[block_id]) {

                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(
                        jacobians[block_id], n, 3);
                    jacobian.setZero();
                    jacobian.leftCols(3) = _marginalization_info->_marginalization_jacobian.middleCols(
                        _marginalization_info->_map_frame_idx.at(_marginalization_info->_frame_to_keep) + 6, 3);
                }
                block_id++;
                if (jacobians[block_id]) {

                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(
                        jacobians[block_id], n, 3);
                    jacobian.setZero();
                    jacobian.leftCols(3) = _marginalization_info->_marginalization_jacobian.middleCols(
                        _marginalization_info->_map_frame_idx.at(_marginalization_info->_frame_to_keep) + 9, 3);
                }
                block_id++;
                if (jacobians[block_id]) {

                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(
                        jacobians[block_id], n, 3);
                    jacobian.setZero();
                    jacobian.leftCols(3) = _marginalization_info->_marginalization_jacobian.middleCols(
                        _marginalization_info->_map_frame_idx.at(_marginalization_info->_frame_to_keep) + 12, 3);
                }
                block_id++;
            }

            // Jacobians on landmarks
            for (auto tlmk : _marginalization_info->_lmk_to_keep) {
                for (auto lmk : tlmk.second) {
                    if (_marginalization_info->_map_lmk_idx.at(lmk) == -1)
                        continue;

                    if (jacobians[block_id]) {

                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(
                            jacobians[block_id], n, 3);
                        jacobian.setZero();
                        jacobian.leftCols(3) = _marginalization_info->_marginalization_jacobian.middleCols(
                            _marginalization_info->_map_lmk_idx.at(lmk), 3);
                    }
                    block_id++;
                }
            }
        }
        return true;
    }

    std::shared_ptr<Marginalization> _marginalization_info;
};

} // namespace isae