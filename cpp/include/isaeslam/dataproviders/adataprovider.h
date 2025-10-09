#ifndef AIMAGEPROVIDER_H
#define AIMAGEPROVIDER_H

#include <fstream>
#include <iostream>
#include <queue>

#include <Eigen/Core>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

#include "isaeslam/data/sensors/Camera.h"
#include "isaeslam/data/sensors/DoubleSphere.h"
#include "isaeslam/data/sensors/Fisheye.h"
#include "isaeslam/data/sensors/IMU.h"
#include "isaeslam/slamParameters.h"

namespace isae {

/*!
 * @brief ADataProvider class for managing data from various sensors.
 *
 * This class is responsible for loading sensor configurations, processing frames, and managing the queue of frames.
 */
class ADataProvider {
  public:
    ADataProvider(std::string path, Config slam_config);

    /*!
     * @brief Return the next frame from the queue.
     */
    std::shared_ptr<Frame> next();

    std::vector<std::shared_ptr<cam_config>> getCamConfigs() { return _cam_configs; };
    std::shared_ptr<imu_config> getIMUConfig() { return _imu_config; }
    int getNCam() { return _ncam; };

    /*!
     * @brief From raw image to sensor objects.
     */
    std::vector<std::shared_ptr<ImageSensor>> createImageSensors(const std::vector<cv::Mat> &imgs,
                                                                 const std::vector<cv::Mat> &masks = {});

    /*!
     * @brief From raw IMU measurements to an IMU sensor object.
     */
    std::shared_ptr<IMU> createImuSensor(const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr);

    /*!
     * @brief Create a frame from sensors and timestamp and add it to the queue.
     */
    void addFrameToTheQueue(std::vector<std::shared_ptr<ASensor>> sensors, double time);

    void addFrameToTheQueue(std::shared_ptr<Frame> frame);

  protected:
    void loadSensorsConfiguration(const std::string &path);
    void loadCamConfig(YAML::Node cam_node);
    void loadIMUConfig(YAML::Node imu_node);

    std::shared_ptr<imu_config> _imu_config;               //!< IMU configuration
    std::vector<std::shared_ptr<cam_config>> _cam_configs; //!< Vector of camera configurations
    int _ncam;                                             //!< Number of Image Sensors
    std::queue<std::shared_ptr<Frame>> _frame_queue;       //!< Queue of frames to be processed
    Config _slam_config;                                   //!< SLAM configuration
    int _nframes;                                          //!< Frame counter
};

/*!
 * @brief EUROCGrabber class for loading and processing frames from raw data in the EUROC format
 *
 * This class is responsible for loading filenames, timestamps, and IMU data from raw data files in the EUROC format,
 * and adding frames to the data provider.
 */
class EUROCGrabber {
  public:
    EUROCGrabber(std::string folder_path, std::shared_ptr<ADataProvider> prov)
        : _folder_path(folder_path), _prov(prov) {}

    /*!
     * @brief Load filenames and timestamps from .csv files.
     */
    void load_filenames();

    /*!
     * @brief Add the frame that comes next in the queue of the data provider.
     */
    bool addNextFrame();

    /*!
     * @brief Add all frames from the queue of the data provider until no more frames are available.
     */
    void addAllFrames() {
        bool not_over = true;
        while (not_over) {
            not_over = addNextFrame();
        }
    }

  private:
    double _time_tolerance = 0.0025; //!< Time tolerance in seconds to consider measurements as synchronized
    std::string _folder_path;        //!< Path to the folder containing the dataset
    std::queue<std::string> _cam0_filename_queue, _cam1_filename_queue; //!< Queues for camera filenames
    std::queue<long long> _cam0_timestamp_queue, _cam1_timestamp_queue,
        _imu_timestamp_queue;                    //!< Queues for camera and IMU timestamps
    std::queue<std::shared_ptr<IMU>> _imu_queue; //!< Queue for IMU sensors

    std::shared_ptr<ADataProvider> _prov; //!< Pointer to the data provider
};

} // namespace isae

#endif // AIMAGEPROVIDER_H
