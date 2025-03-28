# SET(CMAKE_CONFIGURATION_TYPES Debug;Release;MinSizeRel;RelWithDebInfo)
SET(CMAKE_BUILD_TYPE RELEASE) # DEBUG / RELEASE

# set(CMAKE_BUILD_TYPE "DEBUG")
project(isae_slam_ros)
cmake_minimum_required(VERSION 3.14)

# -------------------------------------------------------------------
# Compiler Options
# -------------------------------------------------------------------
add_definitions("-Wall -std=c++17 -O3")

if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  message(STATUS "Debug configuration")
  add_definitions("-g")
else(CMAKE_BUILD_TYPE STREQUAL "RELEASE")
  message(STATUS "Release configuration")
  add_definitions("-O3")
endif(CMAKE_BUILD_TYPE STREQUAL "DEBUG")

add_definitions(-D_USE_MATH_DEFINES)

# -------------------------------------------------------------------
# Dependencies
# -------------------------------------------------------------------

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(rcutils REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Ceres REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
)


# -------------------------------------------------------------------
# Library
# -------------------------------------------------------------------

file(GLOB_RECURSE LIB_SOURCES ../cpp/src/*.cpp ../cpp/include/*.cpp
  ../cpp/thirdparty/ELSED/src/*.cpp)
file(GLOB_RECURSE LIB_HEADERS ../cpp/include/*.h ../cpp/include/*.hpp ../cpp/include/*.inl
  ../cpp/thirdparty/ELSED/src/*.h)
add_library(${PROJECT_NAME} SHARED ${LIB_SOURCES} ${LIB_HEADERS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CERES_LIBRARIES} ${YAML_CPP_LIBRARIES} ${PCL_LIBRARIES} ${Boost_SYSTEM_LIBRARY})

install(
  TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION lib/${PROJECT_NAME}
  ARCHIVE DESTINATION lib/${PROJECT_NAME}
  RUNTIME DESTINATION bin/${PROJECT_NAME}
)

# -------------------------------------------------------------------
# Executables generation
# -------------------------------------------------------------------
add_executable(vio_ros ./src/main.cpp)
ament_target_dependencies(vio_ros rclcpp sensor_msgs pcl_conversions visualization_msgs tf2_ros tf2_geometry_msgs std_msgs cv_bridge)
target_link_libraries(vio_ros ${PROJECT_NAME})

install(TARGETS vio_ros
  DESTINATION lib/${PROJECT_NAME})

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

ament_package()
