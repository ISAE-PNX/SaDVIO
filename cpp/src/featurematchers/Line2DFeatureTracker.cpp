#include "isaeslam/featurematchers/Line2DFeatureTracker.h"

namespace isae {

uint Line2DFeatureTracker::track(std::shared_ptr<isae::ImageSensor> &sensor1,
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


    std::vector<std::shared_ptr<AFeature>> features2;
    Line2DFeatureDetector detector(50, 10, 50.);
    detector.customDetectAndCompute(sensor2->getRawData(), sensor2->getMask(), features2);      
    sensor2->addFeatures("linexd", features2);

    // Failure of the tracker, set a simple matcher instead
    Line2DFeatureMatcher matcher(_detector);
    matcher.match(features_to_track, features2, features_init, tracks, tracks_with_ldmk, search_width, search_height);
    
    // std::cout << "TRACKER DEBUG = features_to_track : " << features_to_track.size() << std::endl;    
    // std::cout << "TRACKER DEBUG = detected img2 : " << features2.size() << std::endl;    
    // std::cout << "TRACKER DEBUG = Lines matched : " << tracks.size() + tracks_with_ldmk.size() << std::endl;
    return  tracks.size() + tracks_with_ldmk.size();

    // if (features_to_track.size() < 1)
    //     return 0;

    // std::cout << "------------------------------------------------------------------";
    
    // // Construct opencv p2f structure for KLT tracking
    // std::vector<cv::Point2f> pts1se, pts2se, pts1seb;

    // for (auto f : features_to_track) {
    //     // get start and end points for each line
    //     pts1se.push_back(cv::Point2f((float)f->getPoints().at(0)(0), (float)f->getPoints().at(0)(1)));
    //     pts1se.push_back(cv::Point2f((float)f->getPoints().at(1)(0), (float)f->getPoints().at(1)(1)));

    //     pts1seb.push_back(cv::Point2f((float)f->getPoints().at(0)(0), (float)f->getPoints().at(0)(1)));
    //     pts1seb.push_back(cv::Point2f((float)f->getPoints().at(1)(0), (float)f->getPoints().at(1)(1)));
    // }

    // // Init with previous keypoints
    // for (auto f : features_init) {
    //     pts2se.push_back(cv::Point2f((float)f->getPoints().at(0)(0), (float)f->getPoints().at(0)(1)));
    //     pts2se.push_back(cv::Point2f((float)f->getPoints().at(1)(0), (float)f->getPoints().at(1)(1)));
    // }

    // // Configure and process KLT optical flow research
    // std::vector<uchar> status, statusb;
    // std::vector<float> err, errb;

    // // Process one way: sensor1->sensor2
    // cv::calcOpticalFlowPyrLK(sensor1->getRawData(),
    //                          sensor2->getRawData(),
    //                          pts1se,
    //                          pts2se,
    //                          status,
    //                          err,
    //                          {search_width, search_height},
    //                          nlvls_pyramids,
    //                          _termCrit,
    //                          (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS));

    // if (backward){
    //     // Process the other way: sensor2->sensor1
    //     cv::calcOpticalFlowPyrLK(sensor2->getRawData(),
    //                              sensor1->getRawData(),
    //                              pts2se,
    //                              pts1seb,
    //                              statusb,
    //                              errb,
    //                              {(int)(search_width * 1.25), (int)(search_height * 1.25)},
    //                              nlvls_pyramids,
    //                              _termCrit,
    //                              (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS));

    // } else{
    //     statusb = status;
    //     errb = err;
    // }


    // for (uint i=0; i < pts1se.size(); i++){
    //     std::cout << pts1se.at(i) << " " << pts2se.at(i) << std::endl;

    // }


    // // Delete point if KLT failed or if point is outside the image
    // cv::Size imgsize = sensor2->getRawData().size();
    // std::vector<int> valid_matches;
    // std::map<int, int> valid_matches_coords_idx;

    
    // for (size_t i = 0; i < status.size()-1; i = i+2) {    
    //     // Invalid match if one of the OF failed or KLT error is too high
    //     if ((status.at(i) == 0) || (statusb.at(i) == 0) ||          // start point
    //         (status.at(i + 1) == 0) || (statusb.at(i + 1) == 0) ||  // end point
    //         (err.at(i) > max_err) || (errb.at(i) > max_err) ||      // start point
    //         (err.at(i + 1) > max_err) || (errb.at(i + 1) > max_err) // end point
    //     ) {
    //         // std::cout << "wrong " << (status.at(i) == 0) << " " << (status.at(i + 1) == 0) << std::endl;
    //         // std::cout << "      " << (statusb.at(i) == 0) << " " << (statusb.at(i + 1) == 0) << std::endl;
    //         // std::cout << "      " << (err.at(i) > max_err) << " " << (err.at(i+1) > max_err) << std::endl;
    //         // std::cout << "      " << (errb.at(i) > max_err) << " " << (errb.at(i+1) > max_err) << std::endl;
    //         // std::cout << "      " << max_err << " " << err.at(i) << " " << errb.at(i) << " " << errb.at(i) << " " <<errb.at(i+1) << std::endl;
    //         continue;
    //     }

    //     // Check if keypoints are the same in backward case
    //     if (backward) {
    //         if (std::abs(pts1se.at(i).x - pts1seb.at(i).x) > 0.5 ||
    //             std::abs(pts1se.at(i).y - pts1seb.at(i).y) > 0.5 ||
    //             std::abs(pts1se.at(i+1).x - pts1seb.at(i+1).x) > 0.5 ||
    //             std::abs(pts1se.at(i+1).y - pts1seb.at(i+1).y) > 0.5)
    //             continue; 
    //     }

    //     // map between unique pose of feature2 with index of feature_to_track
    //     valid_matches_coords_idx[pts2se.at(i).x + pts2se.at(i).y * imgsize.width] = i/2;
    //     valid_matches.push_back(i/2);
    // }

    // //std::cout << "tracks valid_matches size " << valid_matches.size() << std::endl;

    // // Create features with the good detected tracks
    // std::vector<std::shared_ptr<AFeature>> features2;
    // for (uint i : valid_matches) {
    //     std::vector<Eigen::Vector2d> poses2d;
    //     poses2d.push_back(Eigen::Vector2d(pts2se.at(i).x, pts2se.at(i).y));         // start
    //     poses2d.push_back(Eigen::Vector2d(pts2se.at(i + 1).x, pts2se.at(i + 1).y)); // end
    //     std::shared_ptr<AFeature> new_feat = std::make_shared<Line2D>(poses2d);
    //     features2.push_back(new_feat);
    // }

    // // Compute descriptors for new tracked features
    // _detector->computeDescriptor(sensor2->getRawData(), features2);

    // // Keep only matches with described features2
    // std::vector<bool> removed(features2.size(), false);
    // std::vector<bool> already_matched(features_to_track.size(), false);
    // for (size_t i = 0; i < features2.size(); i++) {
        
    //     // get the feature_to_track index for the given feature2 pose
    //     int j = valid_matches_coords_idx[features2.at(i)->getPoints().at(0)(0) +
    //                                      features2.at(i)->getPoints().at(0)(1) * imgsize.width];


    //     // Remove features and matches if descriptor couldn't be calculated
    //     if (features2[i]->getDescriptor().empty()) {
    //         features2[i]->setDescriptor(features_to_track.at(j)->getDescriptor());
    //         continue;
    //     }


    //     // Keep matching with valid described feature 2
    //     if (features_to_track.at(j)->getLandmark().lock())
    //         tracks_with_ldmk.push_back({features_to_track.at(j), features2.at(i)});
    //     else
    //         tracks.push_back({features_to_track.at(j), features2.at(i)});
    // }

    // // Add features
    // sensor2->addFeatures("linexd", features2);

    // return tracks.size() + tracks_with_ldmk.size();
}

} // namespace isae