FROM osrf/ros:humble-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

ENV DIRPATH /root/
WORKDIR $DIRPATH

# Install all the dependencies (including ROS2 humble)
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    gazebo \
    libglu1-mesa-dev \
    vim \
    python3-pip \
    python3-pydantic \
    ros-humble-gazebo-ros \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-joint-state-publisher \
    ros-humble-robot-localization \
    ros-humble-plotjuggler-ros \
    ros-humble-robot-state-publisher \
    ros-humble-ros2bag \
    ros-humble-rosbag2-storage-default-plugins \
    ros-humble-rqt-tf-tree \
    ros-humble-rmw-fastrtps-cpp \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-slam-toolbox \
    ros-humble-turtlebot3 \
    ros-humble-turtlebot3-msgs \
    ros-humble-twist-mux \
    ros-humble-usb-cam \
    ros-humble-xacro \
    ruby-dev \
    rviz \
    tmux \
    wget \
    xorg-dev \
    zsh \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libeigen3-dev \
    libpcl1-dev

# Install CERES
RUN git clone --depth=1 -b 2.2.0 https://ceres-solver.googlesource.com/ceres-solver && \
    cd ceres-solver && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j 3 && \
    make install

# Install SaDVIO
RUN git clone https://github.com/ISAE-PNX/SaDVIO.git
RUN cd SaDVIO/cpp && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make

# Download and extract EuRoC dataset.
RUN apt-get update && apt-get install -y zip
RUN wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.zip
RUN mkdir -p $DIRPATH/euroc && unzip V1_01_easy.zip -d $DIRPATH/euroc