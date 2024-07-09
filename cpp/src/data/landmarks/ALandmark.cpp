#include "isaeslam/data/landmarks/ALandmark.h"
#include "isaeslam/data/features/AFeature2D.h"
namespace isae {

int ALandmark::_landmark_count = 0;

ALandmark::ALandmark(Eigen::Affine3d T_w_l, std::vector<std::shared_ptr<isae::AFeature>> features) {
    _T_w_l             = T_w_l;
    this->_initialized = true;

    for (auto f : features) {
        addFeature(f);
    }
}

void ALandmark::init(Eigen::Affine3d T_w_l, std::vector<std::shared_ptr<isae::AFeature>> features) {

    _T_w_l             = T_w_l;
    this->_initialized = true;

    // increases everytime an object is created
    _landmark_count++;
    _id = _landmark_count;

    // Update features
    for (auto f : features)
        this->addFeature(f);
    
    // Add descriptor
    _descriptor = features.at(0)->getDescriptor();

}

void ALandmark::addFeature(std::shared_ptr<AFeature> feature) {
    std::lock_guard<std::mutex> lock(_lmk_mtx);
    
    _features.push_back(feature);
    feature->linkLandmark(shared_from_this());
}

void ALandmark::removeExpiredFeatures() {
    std::lock_guard<std::mutex> lock(_lmk_mtx);

    for (int i = _features.size() - 1; i >= 0; i--) {
        std::weak_ptr<AFeature> wf = _features.at(i);
        if (!wf.lock()) {
            _features.erase(_features.begin() + i);
        }
    }
}

void ALandmark::removeFeature(std::shared_ptr<AFeature> f) {
    std::lock_guard<std::mutex> lock(_lmk_mtx);

    for (int i = _features.size() - 1; i >= 0; i--) {
        std::weak_ptr<AFeature> wf = _features.at(i);
        if (wf.lock() == f) {
            _features.erase(_features.begin() + i);
        }
    }
}

bool ALandmark::fuseWithLandmark(std::shared_ptr<isae::ALandmark> landmark) {

    // for all the attached features to the landmark to be fused, check if there is some conflict
    for (auto &wf : landmark->getFeatures()) {
        std::shared_ptr<AFeature> f = wf.lock();

        // check if a feature from the same sensor already exist => wrong association somewhere
        for (std::weak_ptr<AFeature> &wff : this->_features) {
            const std::shared_ptr<AFeature> ff = wff.lock();

            // TODO : Why is this case happening ?
            if (!ff) {
                this->removeFeature(ff);
                continue;
            }
            if (ff->getSensor() == f->getSensor()) {
                return false;
            }
        }
    }

    // No problem of duplicated feature, merge all features in this landmark !
    std::vector<std::weak_ptr<AFeature>> fs = landmark->getFeatures();
    for (auto &f : fs) {
        if (f.lock())
            this->addFeature(f.lock());
    }

    return true;
}

double ALandmark::chi2err(std::shared_ptr<AFeature> f) {

    std::vector<Eigen::Vector2d> p2ds;
    double feat_chi2_error    = 0.0;
    Eigen::Matrix2d sqrt_info = (1 / f->getSigma()) * Eigen::Matrix2d::Identity();

    if (!f->getSensor()->project(_T_w_l, _model, _scale, p2ds))
        return 1000;

    for (uint i = 0; i < f->getPoints().size(); ++i) {
        Eigen::Vector2d err = (sqrt_info * (p2ds.at(i) - f->getPoints().at(i)));
        feat_chi2_error += std::pow(err(0), 2);
        feat_chi2_error += std::pow(err(1), 2);
    }

    feat_chi2_error /= (double)f->getPoints().size();

    return feat_chi2_error;
}

double ALandmark::avgChi2err() {

    double mean_chi2_error = 0.;

    for (const auto &f : _features) {
        double feat_chi2_error = chi2err(f.lock());
        mean_chi2_error += feat_chi2_error;
    }

    return mean_chi2_error / _features.size();
}

bool ALandmark::sanityCheck() {

    // If the landmark has not enough features, it is an outlier
    if (_features.size() < 2) {
        this->setOutlier();
        return false;
    }

    // We proceed a 95% chi2test on a 2D point detection
    if (this->avgChi2err() > 2.0) {
        this->setOutlier();
        return false;
    } else
        this->setInlier();

    return true;
}

} // namespace isae
