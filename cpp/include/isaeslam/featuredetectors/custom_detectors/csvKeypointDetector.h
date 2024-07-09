#ifndef CSVKEYPOINTDETECTOR_H
#define CSVKEYPOINTDETECTOR_H

#include "isaeslam/featuredetectors/aCustomFeatureDetector.h"

namespace isae {

/*
This Class is design for CSV reading features, this is how it should be used:

-> The convention is: timestamp.csv for each image
-> The variable _folder_path indicates the folder where the csv are stored
-> in customdetectAndCompute the cv::Mat contains a double with the timestamp of the image

Thus SLAMCore should be slightly edited to create the timestamp cv::Mat and pass it as argument in detectfeatures
Moreover, klt tracking cannot be used because detectAndComputeAdditionnal won't work

Example:

  CsvKeypointDetector csv_detector(1000);
  cv::Mat ts_mat = cv::Mat::zeros(1,1,CV_32F);
  ts_mat.at<double>(0,0) = (double)f->getTimestamp();
  csv_detector.customDetectAndCompute(ts_mat, ts_mat, features);
  f->getSensors().at(i)->addFeatures("pointxd", features);

*/

class CsvKeypointDetector : public ACustomFeatureDetector {
  public:
    CsvKeypointDetector(int n, int n_per_cell, double max_matching_dist = 64)
        : ACustomFeatureDetector(n, n_per_cell) {
        this->init();
        _max_matching_dist = max_matching_dist;
    }

    void customDetectAndCompute(const cv::Mat &img,
                                    const cv::Mat &mask,
                                    std::vector<std::shared_ptr<AFeature>> &features) override;
    void computeDescriptor(const cv::Mat &img, std::vector<std::shared_ptr<AFeature>> &features) override;

    void init() override;
    double getDist(const cv::Mat &desc1, const cv::Mat &desc2) const override;

  private:
    std::string _folder_path;
};

} // namespace isae

#endif // CSVKEYPOINTDETECTOR_H