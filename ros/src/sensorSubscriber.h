#include "isaeslam/dataproviders/adataprovider.h"

#include <mutex>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include <cv_bridge/cv_bridge.h>

class SensorSubscriber : public rclcpp::Node {

  public:
    SensorSubscriber(std::shared_ptr<isae::ADataProvider> prov) : Node("sensor_subscriber"), _prov(prov) {

        _subscription_left = this->create_subscription<sensor_msgs::msg::Image>(
            _prov->getCamConfigs().at(0)->ros_topic,
            10,
            std::bind(&SensorSubscriber::subLeftImage, this, std::placeholders::_1));
        if (_prov->getNCam() == 2)
            _subscription_right = this->create_subscription<sensor_msgs::msg::Image>(
                _prov->getCamConfigs().at(1)->ros_topic,
                10,
                std::bind(&SensorSubscriber::subRightImage, this, std::placeholders::_1));
        if (_prov->getIMUConfig())
            _subscription_imu = this->create_subscription<sensor_msgs::msg::Imu>(
                _prov->getIMUConfig()->ros_topic,
                10,
                std::bind(&SensorSubscriber::subIMU, this, std::placeholders::_1));
    }

    void subLeftImage(const sensor_msgs::msg::Image &img_msg) {
        std::lock_guard<std::mutex> lock(_img_mutex);
        _imgs_bufl.push(img_msg);
    }

    void subRightImage(const sensor_msgs::msg::Image &img_msg) {
        std::lock_guard<std::mutex> lock(_img_mutex);
        _imgs_bufr.push(img_msg);
    }

    void subIMU(const sensor_msgs::msg::Imu &imu_msg) {
        std::lock_guard<std::mutex> lock(_imu_mutex);
        _imu_buf.push(imu_msg);
    }

    cv::Mat getGrayImageFromMsg(const sensor_msgs::msg::Image &img_msg) {
        // Get and prepare images
        cv_bridge::CvImagePtr ptr;
        try {
            ptr = cv_bridge::toCvCopy(img_msg, "mono8");
        } catch (cv_bridge::Exception &e) {
            std::cout << "\n\n\ncv_bridge exeception: %s\n\n\n" << e.what() << std::endl;
        }

        return ptr->image;
    }

    void getImuInfoFromMsg(const sensor_msgs::msg::Imu &imu_msg, Eigen::Vector3d &acc, Eigen::Vector3d &gyr) {

        // Extract the acceleration and gyroscope values from the IMU message
        double ax = imu_msg.linear_acceleration.x;
        double ay = imu_msg.linear_acceleration.y;
        double az = imu_msg.linear_acceleration.z;
        double gx = imu_msg.angular_velocity.x;
        double gy = imu_msg.angular_velocity.y;
        double gz = imu_msg.angular_velocity.z;

        // Create an Eigen vector for the acceleration and gyroscope values
        acc << ax, ay, az;
        gyr << gx, gy, gz;
    }

    void sync_process() {
        std::cout << "\nStarting the measurements reader thread!\n";

        std::vector<std::shared_ptr<isae::ASensor>> sensors;
        double time_tolerance = 0.0025; // TODO add this as a parameter of the yaml
        double t_last         = 0;
        double t_curr         = 0;

        while (true) {

            // Image messages
            cv::Mat image0, image1;
            std::vector<cv::Mat> imgs;
            std::vector<std::shared_ptr<isae::ImageSensor>> img_sensors;

            // Case Stereo
            if (_prov->getNCam() == 2) {
                if (!_imgs_bufl.empty() && !_imgs_bufr.empty()) {
                    double time0 = _imgs_bufl.front().header.stamp.sec * 1e9 + _imgs_bufl.front().header.stamp.nanosec;
                    double time1 = _imgs_bufr.front().header.stamp.sec * 1e9 + _imgs_bufr.front().header.stamp.nanosec;
                    t_curr       = time0;

                    // sync tolerance
                    if (time0 < time1 - 20000000) {
                        _imgs_bufl.pop();
                        std::cout << "\n Throw img0 -- Sync error : " << (time0 - time1) << "\n";
                    } else if (time0 > time1 + 20000000) {
                        _imgs_bufr.pop();
                        std::cout << "\n Throw img1 -- Sync error : " << (time0 - time1) << "\n";
                    } else {

                        // Check if this measurement can be added to the current frame
                        if (std::abs(t_curr - t_last) * 1e-9 > time_tolerance && !sensors.empty()) {
                            _prov->addFrameToTheQueue(sensors, t_last);
                            sensors.clear();
                        }

                        _img_mutex.lock();
                        image0 = getGrayImageFromMsg(_imgs_bufl.front());
                        image1 = getGrayImageFromMsg(_imgs_bufr.front());
                        _imgs_bufl.pop();
                        _imgs_bufr.pop();
                        _img_mutex.unlock();

                        imgs.push_back(image0);
                        imgs.push_back(image1);

                        img_sensors = _prov->createImageSensors(imgs);
                        sensors.push_back(img_sensors.at(0));
                        sensors.push_back(img_sensors.at(1));
                    }

                    t_last = t_curr;
                }

                // Case mono
            } else {
                if (!_imgs_bufl.empty()) {
                    std::lock_guard<std::mutex> lock(_img_mutex);
                    t_curr = _imgs_bufl.front().header.stamp.sec * 1e9 + _imgs_bufl.front().header.stamp.nanosec;

                    // Check if this measurement can be added to the current frame
                    if (std::abs(t_curr - t_last) * 1e-9 > time_tolerance && !sensors.empty()) {
                        _prov->addFrameToTheQueue(sensors, t_last);
                        sensors.clear();
                    }

                    image0 = getGrayImageFromMsg(_imgs_bufl.front());
                    _imgs_bufl.pop();
                    imgs.push_back(image0);

                    img_sensors = _prov->createImageSensors(imgs);
                    sensors.push_back(img_sensors.at(0));

                    t_last = t_curr;
                }
            }

            // IMU message
            if (!_imu_buf.empty()) {
                t_curr = _imu_buf.front().header.stamp.sec * 1e9 + _imu_buf.front().header.stamp.nanosec;
                // t_curr += 0.004486636586849766 * 1e9;

                // Check if this measurement can be added to the current frame
                if (std::abs(t_curr - t_last) * 1e-9 > time_tolerance && !sensors.empty()) {
                    _prov->addFrameToTheQueue(sensors, t_last);
                    sensors.clear();
                }

                Eigen::Vector3d acc, gyr;
                _imu_mutex.lock();
                getImuInfoFromMsg(_imu_buf.front(), acc, gyr);
                _imu_buf.pop();
                _imu_mutex.unlock();
                std::shared_ptr<isae::IMU> imu_ptr = _prov->createImuSensor(acc, gyr);
                sensors.push_back(imu_ptr);

                t_last = t_curr;
            }
        }

        std::cout << "\n Bag reader SyncProcess thread is terminating!\n";
    }

    std::shared_ptr<isae::ADataProvider> _prov;

    std::queue<sensor_msgs::msg::Image> _imgs_bufl, _imgs_bufr;
    std::queue<sensor_msgs::msg::Imu> _imu_buf;
    std::mutex _img_mutex, _imu_mutex;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _subscription_left;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _subscription_right;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr _subscription_imu;
};

class SensorSubscriberCompressed : public rclcpp::Node {

  public:
    SensorSubscriberCompressed(std::shared_ptr<isae::ADataProvider> prov) : Node("sensor_subscriber"), _prov(prov) {

        _subscription_left = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            _prov->getCamConfigs().at(0)->ros_topic,
            rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile(),
            std::bind(&SensorSubscriberCompressed::subLeftImage, this, std::placeholders::_1));
        if (_prov->getNCam() == 2)
            _subscription_right = this->create_subscription<sensor_msgs::msg::CompressedImage>(
                _prov->getCamConfigs().at(1)->ros_topic,
                rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile(),
                std::bind(&SensorSubscriberCompressed::subRightImage, this, std::placeholders::_1));
        if (_prov->getIMUConfig())
            _subscription_imu = this->create_subscription<sensor_msgs::msg::Imu>(
                _prov->getIMUConfig()->ros_topic,
                rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile(),
                std::bind(&SensorSubscriberCompressed::subIMU, this, std::placeholders::_1));
    }

    void subLeftImage(const sensor_msgs::msg::CompressedImage &img_msg) {
        std::lock_guard<std::mutex> lock(_img_mutex);
        _imgs_bufl.push(img_msg);
    }

    void subRightImage(const sensor_msgs::msg::CompressedImage &img_msg) {
        std::lock_guard<std::mutex> lock(_img_mutex);
        _imgs_bufr.push(img_msg);
    }

    void subIMU(const sensor_msgs::msg::Imu &imu_msg) {
        std::lock_guard<std::mutex> lock(_imu_mutex);
        _imu_buf.push(imu_msg);
    }

    cv::Mat getGrayImageFromMsg(const sensor_msgs::msg::CompressedImage &img_msg) {
        // Get and prepare images
        cv_bridge::CvImagePtr ptr;
        try {
            ptr = cv_bridge::toCvCopy(img_msg, "mono8");
        } catch (cv_bridge::Exception &e) {
            std::cout << "\n\n\ncv_bridge exeception: %s\n\n\n" << e.what() << std::endl;
        }
        cv::Mat img = ptr->image;
        cv::resize(img, img, cv::Size(1920, 1080));

        return img;
    }

    void getImuInfoFromMsg(const sensor_msgs::msg::Imu &imu_msg, Eigen::Vector3d &acc, Eigen::Vector3d &gyr) {

        // Extract the acceleration and gyroscope values from the IMU message
        double ax = imu_msg.linear_acceleration.x;
        double ay = imu_msg.linear_acceleration.y;
        double az = imu_msg.linear_acceleration.z;
        double gx = imu_msg.angular_velocity.x;
        double gy = imu_msg.angular_velocity.y;
        double gz = imu_msg.angular_velocity.z;

        // Create an Eigen vector for the acceleration and gyroscope values
        acc << ax, ay, az;
        gyr << gx, gy, gz;
    }

    void sync_process() {
        std::cout << "\nStarting the measurements reader thread!\n";

        std::vector<std::shared_ptr<isae::ASensor>> sensors;
        double time_tolerance = 0.0025; // TODO add this as a parameter of the yaml
        double t_last         = 0;
        double t_curr         = 0;

        while (true) {

            // Image messages
            cv::Mat image0, image1;
            std::vector<cv::Mat> imgs;
            std::vector<std::shared_ptr<isae::ImageSensor>> img_sensors;

            // Case Stereo
            if (_prov->getNCam() == 2) {
                if (!_imgs_bufl.empty() && !_imgs_bufr.empty()) {
                    double time0 = _imgs_bufl.front().header.stamp.sec * 1e9 + _imgs_bufl.front().header.stamp.nanosec;
                    double time1 = _imgs_bufr.front().header.stamp.sec * 1e9 + _imgs_bufr.front().header.stamp.nanosec;
                    t_curr       = time0;

                    // sync tolerance
                    if (time0 < time1 - 20000000) {
                        _imgs_bufl.pop();
                        std::cout << "\n Throw img0 -- Sync error : " << (time0 - time1) << "\n";
                    } else if (time0 > time1 + 20000000) {
                        _imgs_bufr.pop();
                        std::cout << "\n Throw img1 -- Sync error : " << (time0 - time1) << "\n";
                    } else {

                        // Check if this measurement can be added to the current frame
                        if (std::abs(t_curr - t_last) * 1e-9 > time_tolerance && !sensors.empty()) {
                            _prov->addFrameToTheQueue(sensors, t_last);
                            sensors.clear();
                        }

                        _img_mutex.lock();
                        image0 = getGrayImageFromMsg(_imgs_bufl.front());
                        image1 = getGrayImageFromMsg(_imgs_bufr.front());
                        _imgs_bufl.pop();
                        _imgs_bufr.pop();
                        _img_mutex.unlock();

                        imgs.push_back(image0);
                        imgs.push_back(image1);

                        img_sensors = _prov->createImageSensors(imgs);
                        sensors.push_back(img_sensors.at(0));
                        sensors.push_back(img_sensors.at(1));
                    }

                    t_last = t_curr;
                }

                // Case mono
            } else {
                if (!_imgs_bufl.empty()) {
                    std::lock_guard<std::mutex> lock(_img_mutex);
                    t_curr = _imgs_bufl.front().header.stamp.sec * 1e9 + _imgs_bufl.front().header.stamp.nanosec;

                    // Check if this measurement can be added to the current frame
                    if (std::abs(t_curr - t_last) * 1e-9 > time_tolerance && !sensors.empty()) {
                        _prov->addFrameToTheQueue(sensors, t_last);
                        sensors.clear();
                    }

                    image0 = getGrayImageFromMsg(_imgs_bufl.front());
                    _imgs_bufl.pop();
                    imgs.push_back(image0);

                    img_sensors = _prov->createImageSensors(imgs);
                    sensors.push_back(img_sensors.at(0));

                    t_last = t_curr;
                }
            }

            // IMU message
            if (!_imu_buf.empty()) {
                t_curr = _imu_buf.front().header.stamp.sec * 1e9 + _imu_buf.front().header.stamp.nanosec;
                // t_curr += 0.004486636586849766 * 1e9;

                // Check if this measurement can be added to the current frame
                if (std::abs(t_curr - t_last) * 1e-9 > time_tolerance && !sensors.empty()) {
                    _prov->addFrameToTheQueue(sensors, t_last);
                    sensors.clear();
                }

                Eigen::Vector3d acc, gyr;
                _imu_mutex.lock();
                getImuInfoFromMsg(_imu_buf.front(), acc, gyr);
                _imu_buf.pop();
                _imu_mutex.unlock();
                std::shared_ptr<isae::IMU> imu_ptr = _prov->createImuSensor(acc, gyr);
                sensors.push_back(imu_ptr);

                t_last = t_curr;
            }
        }

        std::cout << "\n Bag reader SyncProcess thread is terminating!\n";
    }

    std::shared_ptr<isae::ADataProvider> _prov;

    std::queue<sensor_msgs::msg::CompressedImage> _imgs_bufl, _imgs_bufr;
    std::queue<sensor_msgs::msg::Imu> _imu_buf;
    std::mutex _img_mutex, _imu_mutex;

    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr _subscription_left;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr _subscription_right;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr _subscription_imu;
};