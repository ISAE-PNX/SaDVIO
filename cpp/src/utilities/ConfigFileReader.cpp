#include "utilities/ConfigFileReader.h"

namespace isae {

ConfigFileReader::ConfigFileReader(const std::string &path_config_folder) {
    YAML::Node yaml_file = YAML::LoadFile(path_config_folder + "/config.yaml");

    // Dataset ID
    _config.dataset_id     = yaml_file["dataset_id"].as<std::string>();
    _config.dataset_path   = path_config_folder + "/dataset/" + _config.dataset_id + ".yaml";
    _config.slam_mode      = yaml_file["slam_mode"].as<std::string>();
    _config.enable_visu    = yaml_file["enable_visu"].as<int>();
    _config.multithreading = yaml_file["multithreading"].as<int>();

    // Image processing
    _config.contrast_enhancer = yaml_file["contrast_enhancer"].as<int>();
    _config.clahe_clip        = yaml_file["clahe_clip"].as<float>();
    _config.downsampling      = yaml_file["downsampling"].as<float>();

    // SLAM parameters
    _config.pose_estimator        = yaml_file["pose_estimator"].as<std::string>();
    _config.optimizer             = yaml_file["optimizer"].as<std::string>();
    _config.tracker               = yaml_file["tracker"].as<std::string>();
    _config.min_kf_number         = yaml_file["min_kf_number"].as<int>();
    _config.max_kf_number         = yaml_file["max_kf_number"].as<int>();
    _config.fixed_frame_number    = yaml_file["fixed_frame_number"].as<int>();
    _config.min_lmk_number        = yaml_file["min_lmk_number"].as<float>();
    _config.min_movement_parallax = yaml_file["min_movement_parallax"].as<float>();
    _config.max_movement_parallax = yaml_file["max_movement_parallax"].as<float>();
    _config.marginalization       = yaml_file["marginalization"].as<int>();
    _config.sparsification        = yaml_file["sparsification"].as<int>();
    _config.mesh3D                = yaml_file["mesh3d"].as<int>();
    _config.ZNCC_tsh              = yaml_file["ZNCC_tsh"].as<double>();
    _config.max_length_tsh        = yaml_file["max_length_tsh"].as<double>();

    // Features type
    YAML::Node features_node = yaml_file["features_handled"];

    for (YAML::iterator it = features_node.begin(); it != features_node.end(); ++it) {
        FeatureStruct feature_struct;
        feature_struct.label_feature            = (*it)["label_feature"].as<std::string>();
        feature_struct.detector_label           = (*it)["detector_label"].as<std::string>();
        feature_struct.number_detected_features = (*it)["number_detected_features"].as<int>();
        feature_struct.number_kept_features     = (*it)["number_kept_features"].as<int>();
        feature_struct.n_features_per_cell      = (*it)["n_features_per_cell"].as<int>();
        feature_struct.tracker_label            = (*it)["tracker_label"].as<std::string>();
        feature_struct.tracker_height           = (*it)["tracker_height"].as<int>();
        feature_struct.tracker_nlvls_pyramids   = (*it)["tracker_nlvls_pyramids"].as<int>();
        feature_struct.tracker_max_err          = (*it)["tracker_max_err"].as<double>();
        feature_struct.tracker_width            = (*it)["tracker_width"].as<int>();
        feature_struct.matcher_label            = (*it)["matcher_label"].as<std::string>();
        feature_struct.max_matching_dist        = (*it)["max_matching_dist"].as<double>();
        feature_struct.matcher_height           = (*it)["matcher_height"].as<int>();
        feature_struct.matcher_width            = (*it)["matcher_width"].as<int>();
        feature_struct.lmk_triangulator         = (*it)["lmk_triangulator"].as<std::string>();

        _config.features_handled.push_back(feature_struct);
    }
}

} // namespace isae
