#include "isaeslam/featurematchers/EdgeletFeatureMatcher.h"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

namespace isae {

uint EdgeletFeatureMatcher::match(std::shared_ptr<ImageSensor> &sensor1,
                                  std::shared_ptr<ImageSensor> &sensor2,
                                  vec_match &matches,
                                  vec_match &matches_with_ldmks,
                                  int searchAreaWidth,
                                  int searchAreaHeight) {
    bool USE_TRACKING = 1;

    if (USE_TRACKING) {
        sensor2->purgeFeatures(_feature_label);
        return _tracker->track(sensor1,
                               sensor2,
                               sensor1->getFeatures("edgeletxd"),
                               sensor1->getFeatures("edgeletxd"),
                               matches,
                               matches_with_ldmks,
                               searchAreaWidth,
                               searchAreaHeight);
    } else {

        vec_feat_matches all_matches;
        vec_feat_matches_scores all_scores;

        // Number max of authorized matches for one feature to be considered
        uint max_nb_match = 5;

        // Get features to track
        std::vector<std::shared_ptr<AFeature>> features1 = sensor1->getFeatures("edgeletxd");
        if (features1.size() < 5)
            return false;

        // Construct opencv p2d structure for KLT tracking
        std::vector<cv::Point2f> pts1, pts2;
        for (auto f : features1) {
            pts1.push_back(cv::Point2f((float)f->getPoints().at(0)(0), (float)f->getPoints().at(0)(1)));
        }
        pts2 = pts1; // estimated pose on next image

        // Configure and process KLT optical flow research
        std::vector<uchar> status, statusb;
        std::vector<float> err, errb;

        // Process one way: sensor1->sensor2
        cv::TermCriteria _termCrit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.01);

        cv::calcOpticalFlowPyrLK(sensor1->getRawData(),
                                 sensor2->getRawData(),
                                 pts1,
                                 pts2,
                                 status,
                                 err,
                                 {searchAreaWidth, searchAreaHeight},
                                 3,
                                 _termCrit,
                                 0,
                                 0.001);

        std::vector<std::shared_ptr<AFeature>> features2 = sensor2->getFeatures("edgeletxd");

        // For all tracked point
        uint pt2_idx = 0;
        for (auto &p : pts2) {

            // Get corresponding feature1
            std::shared_ptr<AFeature> f = features1.at(pt2_idx);

            // If invalid track continue
            pt2_idx++;
            if (status.at(pt2_idx - 1) == 0)
                continue;

            // Get second set's features in the viscinity of predicted feature
            std::vector<std::shared_ptr<AFeature>> feats_in_box;
            if (!_detector->getFeaturesInBox(p.x, p.y, 3, 3, features2, feats_in_box))
                continue;

            // Remove matches that are not possible
            uint possible_match_number = 0;
            for (auto &f2 : feats_in_box) {
                double score = _detector->getDist(f->getDescriptor(), f2->getDescriptor());

                if (score >= _detector->getMaxMatchingDist()) {
                    all_matches[f].push_back(f2);
                    all_scores[f].push_back(score);
                }

                // Stop if too much matching remain possible
                possible_match_number++;
                if (possible_match_number > max_nb_match)
                    break;
            }

            // Filter matches and select valid ones
            std::vector<double> scores;
            vec_match unsorted_matches, matches_with_and_without_ldmk;
            for (auto &m12 : all_matches) {

                // Get best matching 1->2
                double score = std::numeric_limits<double>::infinity();
                std::shared_ptr<AFeature> f2;
                for (uint i = 0; i < m12.second.size(); ++i) {
                    if (all_scores[m12.first].at(i) < score) {
                        score = all_scores[m12.first].at(i);
                        f2    = m12.second.at(i);
                    }
                }
                scores.push_back(score);
                unsorted_matches.push_back({m12.first, f2});
            }
            // Sort the matches based on their scores
            std::vector<std::size_t> index_vec;
            for (std::size_t i = 0; i != unsorted_matches.size(); ++i) {
                index_vec.push_back(i);
            }
            sort(index_vec.begin(), index_vec.end(), [&](std::size_t a, std::size_t b) {
                return scores[a] < scores[b];
            });

            for (auto &idx : index_vec)
                matches_with_and_without_ldmk.push_back(unsorted_matches.at(idx));

            // Get matches with and without ldmk
            for (auto &m : matches_with_and_without_ldmk)
                if (m.first->getLandmark().lock())
                    matches_with_ldmks.push_back(m);
                else
                    matches.push_back(m);
            return matches.size() + matches_with_ldmks.size();
        }
    }
}

} // namespace isae
