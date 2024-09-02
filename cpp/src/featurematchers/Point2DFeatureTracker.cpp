#include "isaeslam/featurematchers/Point2DFeatureTracker.h"
#include "isaeslam/data/features/Point2D.h"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

namespace isae {

uint Point2DFeatureTracker::track(std::shared_ptr<isae::ImageSensor> &sensor1,
                                  std::shared_ptr<isae::ImageSensor> &sensor2,
                                  std::vector<std::shared_ptr<AFeature>> &features_to_track,
                                  std::vector<std::shared_ptr<AFeature>> &features_init,
                                  vec_match &tracks,
                                  vec_match &tracks_with_ldmk,
                                  int search_width,
                                  int search_height,
                                  int nlvls_pyramids,
                                  double max_err,
                                  bool backward) {

    if (features_to_track.size() < 10)
        return false;

    // Construct opencv p2f structure for KLT tracking
    std::vector<cv::Point2f> pts1, pts2, pts1b;
    pts1.reserve(features_to_track.size());
    pts2.reserve(features_to_track.size());
    pts1b.reserve(features_to_track.size());

    for (auto f : features_to_track) {
        pts1.push_back(cv::Point2f((float)f->getPoints().at(0)(0), (float)f->getPoints().at(0)(1)));
        pts1b.push_back(cv::Point2f((float)f->getPoints().at(0)(0), (float)f->getPoints().at(0)(1)));
    }

    // Init with previous keypoints
    for (auto f : features_init) {
        pts2.push_back(cv::Point2f((float)f->getPoints().at(0)(0), (float)f->getPoints().at(0)(1)));
    }

    // Configure and process KLT optical flow research
    std::vector<uchar> status, statusb;
    status.reserve(features_to_track.size());
    statusb.reserve(features_to_track.size());
    std::vector<float> err, errb;
    err.reserve(features_to_track.size());
    errb.reserve(features_to_track.size());

    // Process one way: sensor1->sensor2
    cv::calcOpticalFlowPyrLK(sensor1->getRawData(),
                             sensor2->getRawData(),
                             pts1,
                             pts2,
                             status,
                             err,
                             {search_width, search_height},
                             nlvls_pyramids,
                             _termCrit,
                             (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS));

    if (backward) {
        // Process the other way: sensor2->sensor1
        cv::calcOpticalFlowPyrLK(sensor2->getRawData(),
                                 sensor1->getRawData(),
                                 pts2,
                                 pts1b,
                                 statusb,
                                 errb,
                                 {search_width, search_height},
                                 nlvls_pyramids,
                                 _termCrit,
                                 (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS));

    } else {
        statusb = status;
        errb    = err;
    }

    // Delete point if KLT failed or if point is outside the image
    std::vector<std::shared_ptr<AFeature>> features2;
    features2.reserve(features_to_track.size());

    for (size_t i = 0; i < status.size(); i++) {
        // Invalid match if one of the OF failed or KLT error is too high
        if (status.at(i) == 0 || err.at(i) > max_err || statusb.at(i) == 0 || errb.at(i) > max_err) {
            continue;
        }

        // Check if keypoints are the same in backward case
        if (backward) {
            if (cv::norm(pts1.at(i) - pts1b.at(i)) > 0.5)
                continue; 
        }

        // Create feature
        std::vector<Eigen::Vector2d> poses2d;
        poses2d.push_back(Eigen::Vector2d(pts2.at(i).x, pts2.at(i).y));

        std::shared_ptr<AFeature> new_feat = std::make_shared<Point2D>(poses2d);
        features2.push_back(new_feat);

        if (features_to_track.at(i)->getLandmark().lock()){ 
            if (!features_to_track.at(i)->getLandmark().lock()->isOutlier())
                tracks_with_ldmk.push_back({features_to_track.at(i), new_feat});
        }
        else
            tracks.push_back({features_to_track.at(i), new_feat});
    }

    // Compute descriptors for new tracked features
    _detector->computeDescriptor(sensor2->getRawData(), features2);

    // Set arbitrary descriptors for empty ones 
    for (size_t i = 0; i < features2.size(); i++) {

        // Set the descriptor of 
        if (features2[i]->getDescriptor().empty()) {
            features2[i]->setDescriptor(features_to_track.at(i)->getDescriptor());
        }
        
    }

    // add tracked features to sensor 2
    sensor2->addFeatures("pointxd", features2);

    return tracks.size() + tracks_with_ldmk.size();
}

} // namespace isae
