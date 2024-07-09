#include "isaeslam/slamCore.h"

int main(int argc, char **argv) {

    // Create the SLAM parameter object
    std::string config_folder_path                   = argv[1];
    std::shared_ptr<isae::SLAMParameters> slam_param = std::make_shared<isae::SLAMParameters>(config_folder_path);

    // Create SLAM object
    std::shared_ptr<isae::SLAMCore> SLAM;

    if (slam_param->_config.slam_mode == "bimono")
        SLAM = std::make_shared<isae::SLAMBiMono>(slam_param);
    else if (slam_param->_config.slam_mode == "mono")
        SLAM = std::make_shared<isae::SLAMMono>(slam_param);
    else if (slam_param->_config.slam_mode == "nofov")
        SLAM = std::make_shared<isae::SLAMNonOverlappingFov>(slam_param);
    else if (slam_param->_config.slam_mode == "bimonovio")
        SLAM = std::make_shared<isae::SLAMBiMonoVIO>(slam_param);
    else if (slam_param->_config.slam_mode == "monovio")
        SLAM = std::make_shared<isae::SLAMMonoVIO>(slam_param);

    // Create the data grabber
    std::string data_folder_path = argv[2];
    isae::EUROCGrabber grabber(data_folder_path, slam_param->getDataProvider());

    // Load images
    grabber.load_filenames();

    // Load image thread
    std::thread datagrabber(&isae::EUROCGrabber::addAllFrames, grabber);
    datagrabber.detach();

    if (slam_param->_config.multithreading) {

        // Launch front end thread
        std::thread frontend_thread(&isae::SLAMCore::runFrontEnd, SLAM);
        frontend_thread.detach();

        // Launch back end thread
        std::thread backend_thread(&isae::SLAMCore::runBackEnd, SLAM);
        backend_thread.join();

    } else {

        // Launch full odom thread
        std::thread odom_thread(&isae::SLAMCore::runFullOdom, SLAM);
        odom_thread.join();
    }

    return 0;
}