#include "isaeslam/dataproviders/adataprovider.h"
#include "utilities/geometry.h"
#include "utilities/timer.h"

namespace isae {

ADataProvider::ADataProvider(std::string path, Config slam_config) {
    _slam_config    = slam_config;
    _nframes        = 0;
    _img_process_dt = 0;
    this->loadSensorsConfiguration(path);
}

std::shared_ptr<Frame> ADataProvider::next() {
    std::mutex img_mutex;
    std::lock_guard<std::mutex> lock(img_mutex);
    std::shared_ptr<Frame> f = std::shared_ptr<Frame>(new Frame());

    while (_frame_queue.empty())
        cv::waitKey(1);

    f = _frame_queue.front();
    _frame_queue.pop();

    return f;
}

void ADataProvider::loadSensorsConfiguration(const std::string &path) {

    YAML::Node config = YAML::LoadFile(path);

    // Load cams
    _ncam = config["ncam"].as<int>();
    for (int i = 0; i < _ncam; i++) {
        this->loadCamConfig(config["camera_" + std::to_string(i)]);
    }

    // Load IMU
    bool vio_mode = (_slam_config.slam_mode == "bimonovio" || _slam_config.slam_mode == "monovio");
    if (config["imu"] && vio_mode) {
        this->loadIMUConfig(config["imu"]);
    }

    std::cout << " ================================================= " << std::endl;
    std::cout << config["dataset"] << std::endl;
    std::cout << "Cameras config" << std::endl;
    for (int i = 0; i < _ncam; i++) {
        std::cout << "K_cam" << i << " = " << _cam_configs.at(i)->K << std::endl;
        std::cout << "T_cam" << i << " = " << _cam_configs.at(i)->T_s_f.matrix() << std::endl;
    }

    if (config["imu"] && vio_mode) {
        std::cout << "IMU Config" << std::endl;
        std::cout << "T_imu = " << _imu_config->T_s_f.matrix() << std::endl;
    }

    std::cout << " ================================================= " << std::endl;
}

void ADataProvider::loadIMUConfig(YAML::Node imu_node) {
    _imu_config              = std::make_shared<imu_config>();
    _imu_config->sensor_type = "imu";

    // Load ROS topic
    _imu_config->ros_topic = imu_node["topic"].as<std::string>();

    // Load extrinsic
    std::vector<double> data_T_s_f(16);
    data_T_s_f         = imu_node["T_BS"]["data"].as<std::vector<double>>();
    _imu_config->T_s_f = Eigen::Map<Eigen::Affine3d::MatrixType>(&data_T_s_f[0], 4, 4).transpose();

    // Load intrinsics
    _imu_config->acc_noise  = imu_node["accelerometer_noise_density"].as<double>();
    _imu_config->gyr_noise  = imu_node["gyroscope_noise_density"].as<double>();
    _imu_config->bacc_noise = imu_node["accelerometer_random_walk"].as<double>();
    _imu_config->bgyr_noise = imu_node["accelerometer_random_walk"].as<double>();
    _imu_config->rate_hz    = imu_node["rate_hz"].as<double>();
}

void ADataProvider::loadCamConfig(YAML::Node cam_node) {
    std::shared_ptr<cam_config> cam_cfg = std::make_shared<cam_config>();

    // Load K
    Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
    std::vector<float> intrinsics(4);
    intrinsics = cam_node["intrinsics"].as<std::vector<float>>();
    K(0, 0)    = intrinsics[0];
    K(1, 1)    = intrinsics[1];
    K(0, 2)    = intrinsics[2];
    K(1, 2)    = intrinsics[3];

    // Load T_s_f
    Eigen::Affine3d T_s_f;
    std::vector<double> data_T(16);
    data_T = cam_node["T_BS"]["data"].as<std::vector<double>>();
    T_s_f  = Eigen::Map<Eigen::Affine3d::MatrixType>(&data_T[0], 4, 4).transpose().inverse();

    // load distortion
    Eigen::Vector4f D;
    std::vector<float> distortion(4);
    distortion = cam_node["distortion_coefficients"].as<std::vector<float>>();
    D << distortion[0], distortion[1], distortion[2], distortion[3];

    // load image size
    std::vector<int> img_size = cam_node["resolution"].as<std::vector<int>>();

    // Create config
    std::string proj_model = cam_node["projection_model"].as<std::string>();
    cam_cfg->proj_model    = proj_model;

    // Fill intrinsics depending on the model
    if (proj_model == "pinhole") {
        cam_cfg->sensor_type = "pinhole_cam";

    } else if (proj_model == "equidistant") {
        cam_cfg->sensor_type = "fisheye_cam";
        cam_cfg->rmax        = cam_node["rmax"].as<double>();

    } else if (proj_model == "double_sphere") {
        cam_cfg->sensor_type = "ds_cam";
        std::vector<double> intrinsics(6);
        intrinsics = cam_node["intrinsics"].as<std::vector<double>>();
        K(0, 0)    = intrinsics[2];
        K(1, 1)    = intrinsics[3];
        K(0, 2)    = intrinsics[4];
        K(1, 2)    = intrinsics[5];

        cam_cfg->xi    = intrinsics[0];
        cam_cfg->alpha = intrinsics[1];
    }

    if (cam_node["distortion_model"].as<std::string>() == "radial-tangential") {
        cam_cfg->undistort = true;

        // Init cv variables
        cv::Mat K_cv, D_cv, new_K_cv, undist_map_x, undist_map_y;
        cv::Size img_size_cv(img_size[0], img_size[1]);
        cv::eigen2cv(K, K_cv);
        cv::eigen2cv(D, D_cv);

        new_K_cv = cv::getOptimalNewCameraMatrix(K_cv, D_cv, img_size_cv, 0., img_size_cv);
        cv::initUndistortRectifyMap(K_cv, D_cv, cv::Mat(), new_K_cv, img_size_cv, CV_32FC1, undist_map_x, undist_map_y);

        // load undist map
        cam_cfg->undist_map_x = undist_map_x;
        cam_cfg->undist_map_y = undist_map_y;
        cv::cv2eigen(new_K_cv, K);

    } else
        cam_cfg->undistort = false;

    cam_cfg->K      = K.cast<double>();
    cam_cfg->d      = D.cast<double>();
    cam_cfg->T_s_f  = T_s_f;
    cam_cfg->width  = img_size[0];
    cam_cfg->height = img_size[1];

    // Load ROS topic
    cam_cfg->ros_topic = cam_node["topic"].as<std::string>();
    _cam_configs.push_back(cam_cfg);
}
std::vector<std::shared_ptr<ImageSensor>> ADataProvider::createImageSensors(std::vector<cv::Mat> imgs,
                                                                            std::vector<cv::Mat> masks) {
    std::vector<std::shared_ptr<ImageSensor>> sensor_vector;

    isae::timer::tic();
    _nframes++;

    double downsampling = _slam_config.downsampling;

    for (int i = 0; i < _ncam; ++i) {

        cv::Mat img_res;
        Eigen::Matrix3d K = downsampling * _cam_configs.at(i)->K;
        K(2, 2)           = 1;

        std::shared_ptr<ImageSensor> cam;
        if (_cam_configs.at(i)->proj_model == "pinhole") {

            cv::Mat img;

            if (_cam_configs.at(i)->undistort)
                cv::remap(imgs.at(i),
                          img,
                          _cam_configs.at(i)->undist_map_x,
                          _cam_configs.at(i)->undist_map_y,
                          cv::INTER_LINEAR);
            else
                img = imgs.at(i);

            cv::resize(img, img_res, cv::Size(), downsampling, downsampling);

            cam = std::shared_ptr<Camera>(new Camera(img_res, K));

        } else if (_cam_configs.at(i)->proj_model == "equidistant") {

            cv::resize(imgs.at(i), img_res, cv::Size(), downsampling, downsampling);

            cam = std::shared_ptr<Fisheye>(
                new Fisheye(img_res, K, _cam_configs.at(i)->proj_model, _cam_configs.at(i)->rmax));

        } else if (_cam_configs.at(i)->proj_model == "double_sphere") {

            cv::resize(imgs.at(i), img_res, cv::Size(), downsampling, downsampling);

            cam = std::shared_ptr<DoubleSphere>(
                new DoubleSphere(img_res, K, _cam_configs.at(i)->alpha, _cam_configs.at(i)->xi));
        }

        // Add mask if available
        if (!masks.empty())
            cam->setMask(masks.at(i));

        // Image processing
        if (_slam_config.contrast_enhancer == 1)
            cam->applyCLAHE(_slam_config.clahe_clip);
        else if (_slam_config.contrast_enhancer == 2)
            cam->histogramEqualization();
        else if (_slam_config.contrast_enhancer == 3)
            cam->imageNormalization();
        else if (_slam_config.contrast_enhancer == 4)
            cam->applyAGCWD(_slam_config.clahe_clip);

        cam->setFrame2SensorTransform(_cam_configs.at(i)->T_s_f);
        sensor_vector.push_back(cam);
    }

    return sensor_vector;
}

std::shared_ptr<IMU> ADataProvider::createImuSensor(Eigen::Vector3d acc, Eigen::Vector3d gyr) {
    std::shared_ptr<IMU> imu = std::shared_ptr<IMU>(new IMU(_imu_config, acc, gyr));
    return imu;
}

void ADataProvider::addFrameToTheQueue(std::vector<std::shared_ptr<ASensor>> sensors, double time) {
    std::shared_ptr<Frame> f = std::shared_ptr<Frame>(new Frame());

    // Init the Frame
    f->init(sensors, time);

    // add to queue
    _frame_queue.push(f);
}

void EUROCGrabber::load_filenames() {
    // Load cam0
    std::string csv_file = _folder_path + "/cam0/data.csv";
    std::ifstream infile(csv_file);

    std::string line;
    int idx_filename  = 1;
    int idx_timestamp = 0;
    while (std::getline(infile, line)) {

        // Split line into tokens
        std::stringstream line_stream(line);
        std::vector<std::string> line_vector;
        std::string cell;

        while (std::getline(line_stream, cell, ',')) {
            line_vector.push_back(cell);
        }

        // Get the idx of the filename key for the first line
        if (line[0] == '#') {

            for (int i = 0; i < line_vector.size(); i++) {
                std::string key = line_vector.at(i);

                if (key == "filename")
                    idx_filename = i;

                if (key == "#timestamp [ns]")
                    idx_timestamp = i;
            }

            continue;
        }

        // Fill the queues
        _cam0_filename_queue.push(line_vector[idx_filename]);
        _cam0_timestamp_queue.push(std::stod(line_vector[idx_timestamp]));
    }

    // Load cam1
    csv_file = _folder_path + "/cam1/data.csv";
    std::ifstream infile_cam1(csv_file);

    while (std::getline(infile_cam1, line)) {

        // Split line into tokens
        std::stringstream line_stream(line);
        std::vector<std::string> line_vector;
        std::string cell;

        while (std::getline(line_stream, cell, ',')) {
            line_vector.push_back(cell);
        }

        // Get the idx of the filename key for the first line
        if (line[0] == '#') {

            for (int i = 0; i < line_vector.size(); i++) {
                std::string key = line_vector.at(i);

                if (key == "filename")
                    idx_filename = i;

                if (key == "#timestamp [ns]")
                    idx_timestamp = i;
            }

            continue;
        }

        // Fill the queues
        _cam1_filename_queue.push(line_vector[idx_filename]);
        _cam1_timestamp_queue.push(std::stod(line_vector[idx_timestamp]));
    }

    // Load imu
    csv_file = _folder_path + "/imu0/data.csv";
    std::ifstream infile_imu(csv_file);

    while (std::getline(infile_imu, line)) {

        // Skip the first line
        if (line[0] == '#')
            continue;

        // Split line into tokens
        std::stringstream line_stream(line);
        std::vector<std::string> line_vector;
        std::string cell;

        while (std::getline(line_stream, cell, ',')) {
            line_vector.push_back(cell);
        }

        // Fill the queues
        _imu_timestamp_queue.push(std::stod(line_vector[0]));
        Eigen::Vector3d gyr(std::stod(line_vector[1]), std::stod(line_vector[2]), std::stod(line_vector[3]));
        Eigen::Vector3d acc(std::stod(line_vector[4]), std::stod(line_vector[5]), std::stod(line_vector[6]));
        if (_prov->getIMUConfig())
            _imu_queue.push(_prov->createImuSensor(acc, gyr));
    }
}

bool EUROCGrabber::addNextFrame() {

    if ((_prov->getIMUConfig() && _imu_queue.empty()) || _cam0_filename_queue.empty() || _cam1_filename_queue.empty() ||
        _cam0_timestamp_queue.empty() || _cam1_timestamp_queue.empty() || _imu_timestamp_queue.empty())
        return false;
    std::vector<std::shared_ptr<ASensor>> sensors;

    // first catch an IMU measurement
    double imu_ts  = _imu_timestamp_queue.front();
    double cam0_ts = _cam0_timestamp_queue.front();
    double cam1_ts = _cam1_timestamp_queue.front();

    // Case 1 : imu is in the future, discard image until it is not
    if (imu_ts > cam0_ts + _time_tolerance * 1e9) {
        _cam0_timestamp_queue.pop();
        _cam0_filename_queue.pop();
        _cam1_timestamp_queue.pop();
        _cam1_filename_queue.pop();
        return true;
    }

    // Case 2 : imu is in the past too far away from image, imu only frame
    else if (imu_ts < cam0_ts - _time_tolerance * 1e9) {
        if (_prov->getIMUConfig()) {
            sensors.push_back(_imu_queue.front());
            _prov->addFrameToTheQueue(sensors, imu_ts);
            _imu_queue.pop();
        }
        _imu_timestamp_queue.pop();
        return true;
    }

    // Case 3 : imu and cam0 are synced, lets goooo
    else {
        // sync tolerance
        if (cam0_ts < cam1_ts - 20000000) {
            _cam0_filename_queue.pop();
            _cam0_timestamp_queue.pop();
            cam0_ts = _cam0_timestamp_queue.front();

            // Don't forget to add IMU
            if (_prov->getIMUConfig()) {
                sensors.push_back(_imu_queue.front());
                _imu_queue.pop();
            }
            _imu_timestamp_queue.pop();
            std::cout << "\n Throw img0 -- Sync error : " << (cam0_ts - cam1_ts) << "\n";
        } else if (cam0_ts > cam1_ts + 20000000) {
            _cam1_filename_queue.pop();
            _cam1_timestamp_queue.pop();
            cam1_ts = _cam1_timestamp_queue.front();

            // Don't forget to add IMU
            if (_prov->getIMUConfig()) {
                sensors.push_back(_imu_queue.front());
                _imu_queue.pop();
            }
            _imu_timestamp_queue.pop();
            std::cout << "\n Throw img1 -- Sync error : " << (cam0_ts - cam1_ts) << "\n";
        } else {

            std::string path_img0 = _folder_path + "/cam0/data/" +  std::to_string((unsigned long long)_cam0_timestamp_queue.front()) + ".png";
            std::string path_img1 = _folder_path + "/cam1/data/" + std::to_string((unsigned long long)_cam1_timestamp_queue.front()) + ".png";
            cv::Mat img_left      = cv::imread(path_img0, cv::IMREAD_GRAYSCALE);
            if (img_left.empty()) {
                std::cerr << path_img0 << " not opened " << std::endl;
                _cam0_timestamp_queue.pop();
                _cam0_filename_queue.pop();
                _cam1_timestamp_queue.pop();
                _cam1_filename_queue.pop();
                return false;
            }
            cv::Mat img_right = cv::imread(path_img1, cv::IMREAD_GRAYSCALE);
            if (img_right.empty()) {
                std::cerr << path_img1 << " not opened " << std::endl;
                _cam0_timestamp_queue.pop();
                _cam0_filename_queue.pop();
                _cam1_timestamp_queue.pop();
                _cam1_filename_queue.pop();
                return false;
            }

            _cam0_filename_queue.pop();
            _cam1_filename_queue.pop();
            _cam0_timestamp_queue.pop();
            _cam1_timestamp_queue.pop();

            std::vector<cv::Mat> imgs;
            imgs.push_back(img_left);
            imgs.push_back(img_right);

            // Add image sensors
            std::vector<std::shared_ptr<isae::ImageSensor>> img_sensors = _prov->createImageSensors(imgs);
            sensors.push_back(img_sensors.at(0));
            sensors.push_back(img_sensors.at(1));

            // Don't forget to add IMU
            if (_prov->getIMUConfig()) {
                sensors.push_back(_imu_queue.front());
                _imu_queue.pop();
            }
            _imu_timestamp_queue.pop();

            _prov->addFrameToTheQueue(sensors, imu_ts);
        }
    }

    return true;
}

} // namespace isae