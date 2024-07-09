#ifndef AIMAGEPROVIDER_H
#define AIMAGEPROVIDER_H

#include <fstream>
#include <iostream>
#include <queue>

#include <Eigen/Core>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <yaml-cpp/yaml.h>

#include "isaeslam/data/sensors/Camera.h"
#include "isaeslam/data/sensors/DoubleSphere.h"
#include "isaeslam/data/sensors/Fisheye.h"
#include "isaeslam/data/sensors/IMU.h"
#include "utilities/ConfigFileReader.h"

namespace isae {

class ADataProvider {
  public:
    ADataProvider(std::string path, Config slam_config);
    std::shared_ptr<Frame> next();
    std::vector<std::shared_ptr<cam_config>> getCamConfigs() { return _cam_configs; };
    std::shared_ptr<imu_config> getIMUConfig() { return _imu_config; }
    int getNCam() { return _ncam; };
    int getTiming() { return _img_process_dt; }

    // From raw data to Sensors
    std::vector<std::shared_ptr<ImageSensor>> createImageSensors(std::vector<cv::Mat> imgs,
                                                                 std::vector<cv::Mat> masks = {});
    std::shared_ptr<IMU> createImuSensor(Eigen::Vector3d acc, Eigen::Vector3d gyr);

    // Build frame from Sensors
    void addFrameToTheQueue(std::vector<std::shared_ptr<ASensor>> sensors, double time);

  protected:
    void loadSensorsConfiguration(const std::string &path);
    void loadCamConfig(YAML::Node cam_node);
    void loadIMUConfig(YAML::Node imu_node);

    // Sensors configuration
    std::shared_ptr<imu_config> _imu_config;
    std::vector<std::shared_ptr<cam_config>> _cam_configs;
    int _ncam;

    std::queue<std::shared_ptr<Frame>> _frame_queue;

    Config _slam_config;    // SLAM configuration (mainly for contrast)
    int _nframes;           // Frame counter
    double _img_process_dt; // timing of image processing
};

class EUROCGrabber {
  public:
    EUROCGrabber(std::string folder_path, std::shared_ptr<ADataProvider> prov)
        : _folder_path(folder_path), _prov(prov) {}

    void load_filenames();
    bool addNextFrame();

    void addAllFrames() {
        bool not_over = true;
        while (not_over) {
            not_over = addNextFrame();
        }
    }

  private:
    double _time_tolerance = 0.0025;
    std::string _folder_path;
    std::queue<std::string> _cam0_filename_queue, _cam1_filename_queue;
    std::queue<double> _cam0_timestamp_queue, _cam1_timestamp_queue, _imu_timestamp_queue;
    std::queue<std::shared_ptr<IMU>> _imu_queue;

    std::shared_ptr<ADataProvider> _prov;
};

} // namespace isae

#endif // AIMAGEPROVIDER_H
