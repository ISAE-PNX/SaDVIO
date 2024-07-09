#include "isaeslam/data/landmarks/Line3D.h"
#include "isaeslam/data/features/AFeature2D.h"

namespace isae {

double Line3D::chi2err(std::shared_ptr<AFeature> f) {

    std::vector<Eigen::Vector2d> projections;
    double feat_chi2_error    = 0.0;
    Eigen::Matrix2d sqrt_info = (1 / f->getSigma()) * Eigen::Matrix2d::Identity();

    if (!f->getSensor()->project(this->getPose(), this->getModel(), this->getScale(), projections))
        return 1000;


    // distance from projections to 2D line
    std::vector<Eigen::Vector2d>  p2ds = f->getPoints();
    Eigen::Vector3d v_dir, v_proj1, v_proj2;
    v_dir << p2ds.at(0)-p2ds.at(1), 0.;
    v_proj1 << p2ds.at(0)-projections.at(0), 0.;
    v_proj2 << p2ds.at(0)-projections.at(1), 0.;

    Eigen::Vector2d err((v_proj1.cross(v_dir)).norm()/v_dir.norm(), 
                        (v_proj2.cross(v_dir)).norm()/v_dir.norm());

    feat_chi2_error = err.squaredNorm()/(double)p2ds.size();

    return feat_chi2_error;
}

}