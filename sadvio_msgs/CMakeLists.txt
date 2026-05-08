cmake_minimum_required(VERSION 3.5)
project(sadvio_msgs)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_ros REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

set(msg_files
  "msg/SLAMFeatures.msg"
  "msg/SLAMParam.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES
    sensor_msgs
    std_msgs
)

# Export the include directory
install(
  DIRECTORY include/
  DESTINATION "include/${PROJECT_NAME}"
)
ament_export_include_directories("include/${PROJECT_NAME}")

# rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
# target_link_libraries(${PROJECT_NAME}_lib "${cpp_typesupport_target}") 

ament_export_dependencies(rosidl_default_runtime sensor_msgs std_msgs)

ament_package()