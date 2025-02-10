# SET(CMAKE_CONFIGURATION_TYPES Debug;Release;MinSizeRel;RelWithDebInfo)
SET(CMAKE_BUILD_TYPE RELEASE) # DEBUG / RELEASE

# set(CMAKE_BUILD_TYPE "DEBUG")
project(isae_slam)
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
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Ceres REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
)

# -------------------------------------------------------------------
# Library
# -------------------------------------------------------------------
file(GLOB_RECURSE LIB_SOURCES src/*.cpp include/*.cpp
  thirdparty/ELSED/src/*.cpp)
file(GLOB_RECURSE LIB_HEADERS include/*.h include/*.hpp include/*.inl
  thirdparty/ELSED/src/*.h)
add_library(${PROJECT_NAME} SHARED ${LIB_SOURCES} ${LIB_HEADERS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CERES_LIBRARIES} ${YAML_CPP_LIBRARIES} ${PCL_LIBRARIES} ${Boost_SYSTEM_LIBRARY})

# -------------------------------------------------------------------
# Install
# -------------------------------------------------------------------
set(ISAE_SLAM_LIBRARIES "${PROJECT_BINARY_DIR}/libisae_slam.so")

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION lib/${PROJECT_NAME}
  ARCHIVE DESTINATION lib/${PROJECT_NAME}
  RUNTIME DESTINATION bin/${PROJECT_NAME}
)
INSTALL (
    DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h*")
install(
  EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  DESTINATION "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/cmake"
)

# include(CMakePackageConfigHelpers)
# configure_package_config_file(
#     Config.cmake.in
#     "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
#     INSTALL_DESTINATION "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/cmake"
# )

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/cmake"
)

# -------------------------------------------------------------------
# Executables generation
# -------------------------------------------------------------------
add_executable(isaeslam ./main.cpp)
target_link_libraries(isaeslam ${PROJECT_NAME})

# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------
option(ISAESLAM_BUILD_TESTS "Build tests" ON)

if(ISAESLAM_BUILD_TESTS)
  include(CTest)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
  )
  find_package(GTest REQUIRED)
  enable_testing()

  add_executable(isae_test
    ./tests/association_test.cpp
    ./tests/basic_test.cpp
    ./tests/imu_test.cpp
    ./tests/linefeature_test.cpp
    ./tests/marginalization_test.cpp
    ./tests/mesh_test.cpp
    ./tests/nofov_test.cpp
    ./tests/residual_test.cpp)
  target_include_directories(isae_test PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
  target_link_libraries(isae_test
    GTest::gtest_main
    ${PROJECT_NAME}
  )

  include(GoogleTest)
  gtest_discover_tests(isae_test)
endif(ISAESLAM_BUILD_TESTS)
