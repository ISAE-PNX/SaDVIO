# SaDVIO

This library provide a modular C++ framework dedicated on research about Visual Odometry (VO) and Visual Inertial Odometry (VIO). It also contains implementations of research works done at ISAE such as: factor graph sparsification, traversability estimation with 3D mesh and non overlapping field of view VO. This version is compatible with the middleware ROS2. 

<p align='center'>
    <img src="./doc/video.gif" alt="drawing" width="800"/>
</p>


# Installation

Here are the dependencies:
* [OpenCV](https://github.com/opencv/opencv/tree/4.10.0) for image processing
* [CERES](http://ceres-solver.org/installation.html) a non linear solver from google
* [Eigen](https://eigen.tuxfamily.org/dox/GettingStarted.html) a linera algebra library
* [yaml-cpp](https://github.com/jbeder/yaml-cpp) to parse config files 
* [PCL](https://pointclouds.org/documentation/index.html) to deal with point clouds 

You can either run this code with [ROS2](http://docs.ros.org/en/humble/Installation.html), the famous robotic middleware, or without ROS2. For now it has been tested with Galactic and Humble.

## ROS2 install 

Once these are all installed, use the following commands to build:

```
cd ~/your_ws/src
git clone https://github.com/ISAE-PNX/SaDVIO.git
cd ..
colcon build --symlink-install --packages-select isae_slam
```
Then launch the program with:
```
ros2 launch isae_slam isae_slam.xml
```
You can then play a rosbag with the topics specified in the [config](ros/config) files. 

## Classic install

Go in the cpp folder and build it 

```
cd ~/your_ws/src
git clone https://github.com/ISAE-PNX/SaDVIO.git
cd SaDVIO/cpp
mkdir build
cd build
cmake ..
make
```
You can then run this executable, adding the folder of the config files and the folder of your dataset:
```
./isaeslam "/ur/path/SaDVIO/ros/config" "/ur/path/V1_01_easy/mav0"
``` 
 Your dataset must be at the [EUROC dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) format and you must edit properly the files in the [config folder](ros/config).

# Disclaimer

A few functionnalities are currently being tested, their performances are not guaranteed and were not presented in any paper:
* The lines as features and landmarks
* The *mono* and *monovio* modes