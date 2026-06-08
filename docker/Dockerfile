FROM ros:jazzy

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake git \
    libeigen3-dev \
    libopencv-dev libopencv-contrib-dev \
    libyaml-cpp-dev \
    libgoogle-glog-dev libgflags-dev \
    libsuitesparse-dev \
    libceres-dev \
    libpcl-dev \
    libboost-system-dev \
    ros-jazzy-rclcpp \
    ros-jazzy-rcutils \
    ros-jazzy-rcl-interfaces \
    ros-jazzy-sensor-msgs \
    ros-jazzy-geometry-msgs \
    ros-jazzy-visualization-msgs \
    ros-jazzy-std-msgs \
    ros-jazzy-tf2-ros \
    ros-jazzy-tf2-geometry-msgs \
    ros-jazzy-cv-bridge \
    ros-jazzy-pcl-conversions \
    ros-jazzy-ament-index-cpp \
    ros-jazzy-rmw-cyclonedds-cpp \
    ros-jazzy-rviz2 \
    ros-jazzy-rosbag2 \
    ros-jazzy-rosbag2-storage-default-plugins \
    && rm -rf /var/lib/apt/lists/*

# Use CycloneDDS to avoid FastCDR version mismatches
# ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

RUN mkdir -p /root/sadvio_ws/src
COPY SaDVIO /root/sadvio_ws/src/SaDVIO

# Build standalone C++ library
RUN cd /root/sadvio_ws/src/SaDVIO/cpp && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DISAESLAM_BUILD_TESTS=OFF && \
    make -j$($(nproc)-2) && \
    make install

RUN bash -c "\
    source /opt/ros/jazzy/setup.bash && \
    cd /root/sadvio_ws && \
    colcon build \
    --symlink-install \
    --packages-select isae_slam_ros \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
    2>&1 | tee /root/sadvio_ws/build.log && \
    echo '[sadvio] build finished'"

RUN echo "export TERM=xterm-256color"                   >> /root/.bashrc && \
    echo "source /opt/ros/jazzy/setup.bash"             >> /root/.bashrc && \
    echo "source /root/sadvio_ws/install/setup.bash"    >> /root/.bashrc
#echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /root/.bashrc
