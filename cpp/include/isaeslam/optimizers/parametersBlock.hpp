#include "isaeslam/typedefs.h"
#include "utilities/geometry.h"

#include <Eigen/Core>

namespace isae {

class PoseParametersBlock {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PoseParametersBlock() {}

    PoseParametersBlock(const Eigen::Affine3d &T_w_f) {
        Eigen::Map<Vector6d> x(values_);
        x = geometry::se3_RTtoVec6d(T_w_f);
    }

    PoseParametersBlock(const PoseParametersBlock &block) {
        for( size_t i = 0 ; i < ndim_ ; i++ ) {
            values_[i] = block.values_[i];
        }
    }

    PoseParametersBlock& operator = (const PoseParametersBlock &block) 
    { 
        for( size_t i = 0 ; i < ndim_ ; i++ ) {
            values_[i] = block.values_[i];
        }
        return *this; 
    }  

    Eigen::Affine3d getPose() {
        Eigen::Map<Vector6d> x(values_);
        return geometry::se3_Vec6dtoRT(x);
    }

    inline double* values() {  
        return values_; 
    }

    static const size_t ndim_ = 6;
    double values_[ndim_] = {0.,0.,0.,0.,0.,0.};
};

class PointXYZParametersBlock {
    
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PointXYZParametersBlock() {}

    PointXYZParametersBlock(const Eigen::Vector3d &X) {
        Eigen::Map<Eigen::Vector3d> (values_, 3, 1) = X;
    }

    PointXYZParametersBlock(const PointXYZParametersBlock &block) {
        for( size_t i = 0 ; i < ndim_ ; i++ ) {
            values_[i] = block.values_[i];
        }
    }

    PointXYZParametersBlock& operator = (const PointXYZParametersBlock &block) 
    { 
        for( size_t i = 0 ; i < ndim_ ; i++ ) {
            values_[i] = block.values_[i];
        }
        return *this; 
    }

    Eigen::Affine3d getPose() {
        Eigen::Affine3d plmk(Eigen::Affine3d::Identity());
        Eigen::Map<Eigen::Vector3d> tlmk(values_);
        plmk.translation() = tlmk;
        return plmk;
    }

    inline double* values() {  
        return values_; 
    }

    static const size_t ndim_ = 3;
    double values_[ndim_] = {0.,0.,0.};
};

}