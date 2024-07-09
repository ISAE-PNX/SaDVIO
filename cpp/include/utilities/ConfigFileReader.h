#ifndef CONFIGFILEREADER_H
#define CONFIGFILEREADER_H

#include <yaml-cpp/yaml.h>

namespace isae {

struct FeatureStruct {
    std::string label_feature;
    std::string detector_label;   // the capital letter should be respected
    int number_detected_features; // number of features to be detected by the detector
    int number_kept_features;     // number of features to be kept in the SLAM estimation
    int n_features_per_cell;      // number of features per cell for bucketting
    std::string tracker_label;    // class name of the tracker we will use in our SLAM
    int tracker_height;           // searchAreaHeight of tracker
    int tracker_width;            // searchAreaWidth of tracker
    int tracker_nlvls_pyramids;   // nlevels of pyramids for klt tracking
    double tracker_max_err;       // error threshold for klt tracking
    std::string matcher_label;    // class name of the matcher we will use in our SLAM
    double max_matching_dist;     // distance for matching
    int matcher_height;           // searchAreaHeight of tracker
    int matcher_width;            // searchAreaWidth of tracker
    std::string
        lmk_triangulator; // landmarkTriangulation class we will use to triangulate landmark of label_feature type
};

// This structure contains the configuration parameters located in the config file.
struct Config {
    std::string dataset_path;
    std::string dataset_id;
    std::string slam_mode;
    bool multithreading;
    bool enable_visu;
    std::string optimizer;
    int contrast_enhancer; // integer to choose the contrast enhancement algorithm
    float clahe_clip;
    float downsampling;  // float to reduce the size of the image
    int marginalization; // 0 no marginalization, 1 marginalization
    bool sparsification; // 0 no sparsification, 1 sparsification
    std::string pose_estimator;
    std::string tracker;
    int min_kf_number;
    int max_kf_number;
    int fixed_frame_number;
    float min_lmk_number;
    float min_movement_parallax;
    float max_movement_parallax;
    bool mesh3D;           // 0 no 3D mesh, 1 3D mesh
    double ZNCC_tsh;       // Threshold on ZNCC for triangle filtering
    double max_length_tsh; // Threshold on maximum length for triangle filtering

    std::vector<isae::FeatureStruct>
        features_handled; // types of features the slam will work on separated with commas (,)
};

class ConfigFileReader {
  public:
    ConfigFileReader(const std::string &path_config_folder);
    Config _config;
};

} // namespace isae

#endif
