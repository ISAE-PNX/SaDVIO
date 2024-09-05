//
// Created by d.vivet on 26/11/2021.
//

#include "isaeslam/featuredetectors/aFeatureDetector.h"
#include <iostream>

namespace isae {

bool AFeatureDetector::getFeaturesInBox(int x,
                                        int y,
                                        int w,
                                        int h,
                                        const std::vector<std::shared_ptr<AFeature>> &features,
                                        std::vector<std::shared_ptr<AFeature>> &features_in_box) const {

    for (auto &f : features) {

        if (f->getFeatureLabel() == "linexd") {
            Eigen::Vector2d middlePt = (f->getPoints().at(0) + f->getPoints().at(1)) / 2.;
            if (middlePt(0) < x || middlePt(0) > x + w || middlePt(1) < y || middlePt(1) > y + h)
                continue;
            features_in_box.push_back(f);
        } else {
            if (f->getPoints().at(0)(0) < x || f->getPoints().at(0)(0) > x + w || f->getPoints().at(0)(1) < y ||
                f->getPoints().at(0)(1) > y + h)
                continue;
            features_in_box.push_back(f);
        }
    }
    if (features_in_box.empty())
        return false;
    else
        return true;
}

void AFeatureDetector::deleteUndescribedFeatures(std::vector<std::shared_ptr<AFeature>> &features) {
    // remove not described features
    features.erase(std::remove_if(features.begin(),
                                  features.end(),
                                  [](std::shared_ptr<AFeature> &f) { return f->getDescriptor().empty(); }),
                   features.end());
}

void AFeatureDetector::KeypointToFeature(std::vector<cv::KeyPoint> keypoints,
                                         cv::Mat descriptors,
                                         std::vector<std::shared_ptr<AFeature>> &features,
                                         const std::string &featurelabel) {
    // Create Point2D from feature & descriptor
    for (uint i = 0; i < keypoints.size(); ++i) {
        Eigen::Vector2d point(keypoints.at(i).pt.x, keypoints.at(i).pt.y);

        std::vector<Eigen::Vector2d> points;
        points.push_back(point);

        if (descriptors.empty()) {
            std::cerr << "empty descriptor" << std::endl;
        } else {
            if (featurelabel == "pointxd")
                features.push_back(std::make_shared<Point2D>(points, descriptors.row(i), keypoints.at(i).octave));

            if (featurelabel == "edgeletxd")
                features.push_back(std::make_shared<Edgelet2D>(points, descriptors.row(i)));

            if (featurelabel == "linexd")
                features.push_back(std::make_shared<Line2D>(points, descriptors.row(i)));
        }
    }
}

void AFeatureDetector::FeatureToKeypoint(std::vector<std::shared_ptr<AFeature>> features,
                                         std::vector<cv::KeyPoint> &keypoints,
                                         cv::Mat &descriptors) {

    for (uint i = 0; i < features.size(); ++i) {
        descriptors.push_back(features.at(i)->getDescriptor());
        keypoints.push_back(
            cv::KeyPoint(cv::Point2f(features.at(i)->getPoints().at(0)(0), features.at(i)->getPoints().at(0)(1)),
                         1,
                         -1.0,
                         0.0,
                         0,
                         1));
    }
}

void AFeatureDetector::FeatureToP2f(std::vector<std::shared_ptr<AFeature>> features, std::vector<cv::Point2f> &p2fs) {
    for (uint i = 0; i < features.size(); ++i) {
        p2fs.push_back(cv::Point2f(features.at(i)->getPoints().at(0)(0), features.at(i)->getPoints().at(0)(1)));
    }
}

} // namespace isae