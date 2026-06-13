cmake_minimum_required(VERSION 3.22)

project(
  feature_extraction_cuda_elm
  VERSION 0.1.0
  DESCRIPTION "GPU-accelerated Extreme Learning Machine feature extraction"
  LANGUAGES CXX CUDA)

include(CTest)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Use standard CUDA 17 for CMake compatibility.
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

add_library(feature_elm_core src/core/version.cpp src/core/elm.cpp)
add_library(feature_elm::core ALIAS feature_elm_core)

# CUDA backend library
add_library(feature_elm_cuda src/cuda/device_buffer.cu src/cuda/simple_kernel.cu src/cuda/elm_gpu.cu)
add_library(feature_elm::cuda ALIAS feature_elm_cuda)

target_link_libraries(feature_elm_cuda PUBLIC CUDA::cudart CUDA::cublas CUDA::cusolver)

target_include_directories(
  feature_elm_core
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
         $<INSTALL_INTERFACE:include>)
target_compile_features(feature_elm_core PUBLIC cxx_std_20)

target_include_directories(
  feature_elm_cuda
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
         $<INSTALL_INTERFACE:include>
         $<TARGET_PROPERTY:CUDA::cudart,INTERFACE_INCLUDE_DIRECTORIES>)
# CUDA targets should use the CMAKE_CUDA_STANDARD setting and avoid explicit C++20 host flags.

if(MSVC)
  target_compile_options(feature_elm_core PRIVATE /W4 /WX)
else()
  target_compile_options(feature_elm_core PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(BUILD_TESTING)
  find_package(GTest REQUIRED)

  add_executable(test_sanity tests/test_sanity.cpp)
  target_link_libraries(test_sanity PRIVATE feature_elm::core GTest::gtest_main)
  target_compile_features(test_sanity PRIVATE cxx_std_20)

  add_executable(test_elm_cpu tests/test_elm_cpu.cpp)
  target_link_libraries(test_elm_cpu PRIVATE feature_elm::core GTest::gtest_main)
  target_compile_features(test_elm_cpu PRIVATE cxx_std_20)

  add_executable(test_cuda_infra tests/test_cuda_infra.cpp)
  target_link_libraries(test_cuda_infra PRIVATE feature_elm::cuda GTest::gtest_main)
  target_compile_features(test_cuda_infra PRIVATE cxx_std_20)

  add_executable(test_elm_gpu tests/test_elm_gpu.cpp)
  target_link_libraries(test_elm_gpu PRIVATE feature_elm::core feature_elm::cuda GTest::gtest_main)
  target_compile_features(test_elm_gpu PRIVATE cxx_std_20)

  include(GoogleTest)
  gtest_discover_tests(test_sanity)
  gtest_discover_tests(test_elm_cpu)
  gtest_discover_tests(test_cuda_infra TEST_PREFIX cuda_infra.)
  gtest_discover_tests(test_elm_gpu TEST_PREFIX elm_gpu.)
endif()
