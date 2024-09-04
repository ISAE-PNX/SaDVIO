#include "isaeslam/featurematchers/afeaturematcher.h"
#include "isaeslam/data/features/Edgelet2D.h"
#include "utilities/geometry.h"

namespace isae {

void AFeatureMatcher::getPossibleMatchesBetween(const std::vector<std::shared_ptr<AFeature>> &features1,
                                                const std::vector<std::shared_ptr<AFeature>> &features2,
                                                const std::vector<std::shared_ptr<AFeature>> &features_init,
                                                const uint &searchAreaWidth,
                                                const uint &searchAreaHeight,
                                                vec_feat_matches &matches,
                                                vec_feat_matches_scores &all_scores) {
    // For all features from first set
    for (size_t i = 0; i < features1.size(); i++) {

        std::shared_ptr<AFeature> f      = features1.at(i);
        std::shared_ptr<AFeature> f_init = features_init.at(i);

        // Retain the two best scores
        double best_dist1 = _detector->getMaxMatchingDist();
        double best_dist2 = _detector->getMaxMatchingDist();
        int best_idx1     = -1;

        // Get second set's features in the viscinity
        std::vector<int> indexes;
        if (!_detector->getFeaturesInBox(f_init->getPoints().at(0)(0) - searchAreaWidth / 2,
                                         f_init->getPoints().at(0)(1) - searchAreaHeight / 2,
                                         searchAreaWidth,
                                         searchAreaHeight,
                                         indexes,
                                         features2))
            continue; // no features detected

        // Remove matches that are not possible
        for (int idx = indexes.size() - 1; idx >= 0; idx--) {

            // Check if the octave is similar
            if (f->getOctave() < features2.at(indexes.at(idx))->getOctave() - 1 ||
                f->getOctave() > features2.at(indexes.at(idx))->getOctave()) {
                continue;
            }

            // Check the score
            double score = _detector->getDist(f->getDescriptor(), features2.at(indexes.at(idx))->getDescriptor());
            if (score < best_dist1) {
                best_dist2 = best_dist1;
                best_dist1 = score;

                best_idx1 = idx;
            } else if (score < best_dist2) {
                best_dist2 = score;
            }
        }

        // keep match only if the ratio is low enough
        if (best_dist1 / best_dist2 < _first_second_match_score_ratio) {
            matches[f].push_back(features2.at(indexes.at(best_idx1)));
            all_scores[f].push_back(best_dist1);
        }
    }
}

vec_match AFeatureMatcher::filterMatches(vec_feat_matches &matches12,
                                         vec_feat_matches &matches21,
                                         vec_feat_matches_scores &all_scores12,
                                         vec_feat_matches_scores &all_scores21) {

    // Filter matches and select valid ones
    std::vector<double> scores;
    vec_match unsorted_matches;
    vec_match matches;
    for (auto &m12 : matches12) {

        // Get best matching 1->2
        double score = std::numeric_limits<double>::infinity();
        std::shared_ptr<AFeature> f2;
        for (uint i = 0; i < m12.second.size(); ++i) {
            if (all_scores12[m12.first].at(i) < score) {
                score = all_scores12[m12.first].at(i);
                f2    = m12.second.at(i);
            }
        }
        // Check if the best matching 2->1 for f2 is the original f1
        double score21 = std::numeric_limits<double>::infinity();
        std::shared_ptr<AFeature> f1;
        for (uint i = 0; i < matches21[f2].size(); ++i) {
            if (all_scores21[f2].at(i) < score21) {
                score21 = all_scores21[f2].at(i);
                f1      = matches21[f2].at(i);
            }
        }
        if (f1 == m12.first) {
            scores.push_back(score);
            unsorted_matches.push_back({f1, f2});
        }
    }

    // Sort the matches based on their scores
    std::vector<std::size_t> index_vec;
    for (std::size_t i = 0; i != unsorted_matches.size(); ++i) {
        index_vec.push_back(i);
    }
    sort(index_vec.begin(), index_vec.end(), [&](std::size_t a, std::size_t b) { return scores[a] < scores[b]; });

    for (auto &idx : index_vec)
        matches.push_back(unsorted_matches.at(idx));

    return matches;
}

uint AFeatureMatcher::match(std::vector<std::shared_ptr<AFeature>> &features1,
                            std::vector<std::shared_ptr<AFeature>> &features2,
                            std::vector<std::shared_ptr<AFeature>> &features_init,
                            vec_match &matches,
                            vec_match &matches_with_ldmks,
                            int searchAreaWidth,
                            int searchAreaHeight) {

    // Note for possible evolution
    // Matching strategy :
    // - 1) brute force big window
    // - 2) match table
    // - 3) get pure matches : 1 <-> 1
    // - 4) confused matches : 1 <-> N / N <-> 1 :
    //         option 1 : score > threshold, take max
    //         option 2 : graph cut, maximize global matching cost (?)

    // For each feature to match from sensor 1, find possible feature from sensor 2
    vec_feat_matches matches12;
    vec_feat_matches_scores all_scores12;
    getPossibleMatchesBetween(
        features1, features2, features_init, searchAreaWidth, searchAreaHeight, matches12, all_scores12);

    // Get matches with and without ldmk
    for (auto &m : matches12) {
        std::shared_ptr<AFeature> f1 = m.first;
        if (f1->getLandmark().lock()) {
            if (!f1->getLandmark().lock()->isOutlier()) {
                for (uint i = 0; i < m.second.size(); ++i) {
                    matches_with_ldmks.push_back({f1, m.second.at(i)});
                }
            }
        } else {
            for (uint i = 0; i < m.second.size(); ++i) {
                matches.push_back({f1, m.second.at(i)});
            }
        }
    }
    return matches.size() + matches_with_ldmks.size();
}

uint AFeatureMatcher::ldmk_match(std::shared_ptr<ImageSensor> &sensor1,
                                 vec_shared<ALandmark> &ldmks,
                                 int searchAreaWidth,
                                 int searchAreaHeight) {
    // For each landmark to match, try to find a feature close
    uint nb_ldmk_matched = 0;
    for (auto &lmk : ldmks) {

        // Check if the ldmk is already matched in the frame
        bool already_in = false;
        for (auto f : lmk->getFeatures()) {
            if (!f.lock())
                continue;

            if (f.lock()->getSensor()->getFrame() == sensor1->getFrame()) 
                already_in = true;
        }
        if (already_in) 
            continue;

        std::string label = lmk->getLandmarkLabel();

        // If it has prior continue
        if (lmk->hasPrior() || !lmk->isInitialized() || lmk->isOutlier())
            continue;

        // Project the landmark in the current sensor
        std::vector<Eigen::Vector2d> p2ds;
        if (!sensor1->project(lmk->getPose(), lmk->getModel(), lmk->getScale(), p2ds))
            continue;

        // For all features of this type, try to find one close to the reprojection
        double min1 = std::numeric_limits<double>::infinity();
        double min2 = std::numeric_limits<double>::infinity();
        std::shared_ptr<AFeature> matched;
        std::vector<int> indexes;

        // Select the region of interest
        double x = std::max(p2ds.at(0).x() - ((double)searchAreaWidth / 2), 0.0); 

        double y = std::max(p2ds.at(0).y() - ((double)searchAreaHeight / 2), 0.0);

        int width = searchAreaWidth;
        if (x + width > sensor1->getRawData().cols)
            width = sensor1->getRawData().cols - x;
        
        int height = searchAreaHeight;
        if (y + height > sensor1->getRawData().rows)
            height = sensor1->getRawData().cols - y;

        std::vector<std::shared_ptr<AFeature>> features;

        _detector->getFeaturesInBox(x,
                                    y,
                                    width,
                                    height,
                                    indexes,
                                    sensor1->getFeatures(label));
        if (!indexes.empty()) {
            for (auto i : indexes)
                features.push_back(sensor1->getFeatures(label).at(i));

            for (auto &f : features) {
                
                // Ignore features with landmarks
                if (f->getLandmark().lock())
                    continue;

                double score = _detector->getDist(f->getDescriptor(), lmk->getDescriptor());

                if (score < min1) {
                    min2    = min1;
                    min1    = score;
                    matched = f;

                } else if (score < min2)
                    min2 = score;
            }

            if (matched != nullptr && min1 < _detector->getMaxMatchingDist() &&
                min1 / min2 < _first_second_match_score_ratio) {

                // Check reprojection error
                double reproj_err = (p2ds.at(0) - matched->getPoints().at(0)).norm();
                if (reproj_err > 1.0) {
                    continue;
                }

                lmk->addFeature(matched);
                matched->linkLandmark(lmk);
                sensor1->getFrame()->addLandmark(lmk);
                lmk->setResurected();

                nb_ldmk_matched++;
            }
        }
    }

    return nb_ldmk_matched;
}

} // namespace isae
