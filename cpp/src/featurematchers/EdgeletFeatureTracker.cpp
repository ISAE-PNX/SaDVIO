#include "isaeslam/featurematchers/EdgeletFeatureTracker.h"
#include <map>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

namespace isae {

uint EdgeletFeatureTracker::track(std::shared_ptr<isae::ImageSensor> &sensor1,
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
    for (auto f : features_to_track) {
        pts1.push_back(cv::Point2f((float)f->getPoints().at(0)(0), (float)f->getPoints().at(0)(1)));
    }

    // Init with previous keypoints
    for (auto f : features_to_track) {
        pts2.push_back(cv::Point2f((float)f->getPoints().at(0)(0), (float)f->getPoints().at(0)(1)));
    }


//    if (existing_features2.size() == 0) {
//        for (auto f : features_to_track) {
//            pts2.push_back(cv::Point2f((float)f->getPoints().at(0)(0), (float)f->getPoints().at(0)(1)));
//        }
//    } else {
//        for (auto f : existing_features2) {
//            pts2.push_back(cv::Point2f((float)f->getPoints().at(0)(0), (float)f->getPoints().at(0)(1)));
//        }
//    }

    // Configure and process KLT optical flow research
    std::vector<uchar> status, statusb;
    std::vector<float> err, errb;

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

    // Process the other way: sensor2->sensor1
    if (backward){
        cv::calcOpticalFlowPyrLK(sensor2->getRawData(),
                                sensor1->getRawData(),
                                pts2,
                                pts1,
                                statusb,
                                errb,
                                {(int)(search_width * 1.25), (int)(search_height * 1.25)},
                                 nlvls_pyramids,
                                 _termCrit,
                                 (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS));
    } else{
        statusb = status;
        errb = err;
    }

    // Delete point if KLT failed or if point is outside the image
    cv::Size imgsize = sensor2->getRawData().size();
    std::vector<int> valid_matches;
    std::map<int, int> valid_matches_coords_idx;
    for (size_t i = 0; i < status.size(); i++) {
        // Invalid match if one of the OF failed or KLT error is too high
        if ((status.at(i) == 0) || (statusb.at(i) == 0) ||
            (err.at(i) > max_err) || (errb.at(i) > max_err)) {
            continue;
        }

//        // Check if tracked points in the second sensor are in the image
//        uint bound = 0;
//        if ((pts2.at(i).x < (0 + bound)) || (pts2.at(i).y < (0 + bound)) || (pts2.at(i).x > (imgsize.width - bound)) ||
//            (pts2.at(i).y > (imgsize.height - bound))) {
//            continue;
//        }
//
//        // Check if pairing backward is too far from original detection
//        Eigen::Vector2d pt1  = Eigen::Vector2d(pts1.at(i).x, pts1.at(i).y);
//        Eigen::Vector2d pt1b = Eigen::Vector2d(pts1b.at(i).x, pts1b.at(i).y);
//        if ((pt1 - pt1b).norm() > _max_backward_dist) {
//            continue;
//        }

        // Map the matching between coordinates of points of new features and index of matched features_to_track
        valid_matches_coords_idx[pts2.at(i).x + pts2.at(i).y * imgsize.width] = i;
        valid_matches.push_back(i);
    }

    // Create features with the good detected tracks
    std::vector<std::shared_ptr<AFeature>> features2;
    for (uint i : valid_matches) {
        std::vector<Eigen::Vector2d> poses2d;
        poses2d.push_back(Eigen::Vector2d(pts2.at(i).x, pts2.at(i).y));
        std::shared_ptr<AFeature> new_feat = std::make_shared<Edgelet2D>(poses2d);
        features2.push_back(new_feat);
    }

    // Compute descriptors for new tracked features
    _detector->computeDescriptor(sensor2->getRawData(), features2);

    // Keep only matches with described features2
    std::vector<bool> removed(features2.size(), false);
    std::vector<bool> already_matched(features_to_track.size(), false);
    for (size_t i = 0; i < features2.size(); i++) {
        // get the feature1 index for the given feature2 pose
        int j = valid_matches_coords_idx[features2.at(i)->getPoints().at(0)(0) +
                                         features2.at(i)->getPoints().at(0)(1) * imgsize.width];

        // Remove features and matches if descriptor couldn't be calculated
        if (features2[i]->getDescriptor().empty()) {
            removed.at(i) = true;
            continue;
        }

        // Keep matching with valid described feature 2
        if (features_to_track.at(j)->getLandmark().lock())
            tracks_with_ldmk.push_back({features_to_track.at(j), features2.at(i)});
        else
            tracks.push_back({features_to_track.at(j), features2.at(i)});
    }

    // Add new features 
    sensor2->addFeatures("edgeletxd", features2);

    return tracks.size() + tracks_with_ldmk.size();
}

} // namespace isae