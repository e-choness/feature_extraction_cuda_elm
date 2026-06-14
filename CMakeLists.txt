cmake_minimum_required(VERSION 3.22)

project(
  feature_extraction_cuda_elm
  VERSION 0.1.0
  DESCRIPTION "GPU-accelerated Extreme Learning Machine feature extraction"
  LANGUAGES CXX CUDA)

include(CTest)
include(FetchContent)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

option(ENABLE_CUDA "Enable CUDA backend support" ON)
find_package(CUDAToolkit QUIET)

add_library(feature_elm_core
  src/core/version.cpp
  src/core/elm.cpp
  src/core/os_elm.cpp
  src/core/os_celm.cpp
  src/core/h_os_elm.cpp
  src/core/rbf_features.cpp)
add_library(feature_elm::core ALIAS feature_elm_core)

target_include_directories(
  feature_elm_core
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
         $<INSTALL_INTERFACE:include>)
target_compile_features(feature_elm_core PUBLIC cxx_std_20)

if(ENABLE_CUDA AND CUDAToolkit_FOUND)
  add_library(feature_elm_cuda
    src/cuda/device_buffer.cu
    src/cuda/simple_kernel.cu
    src/cuda/elm_gpu.cu)
  add_library(feature_elm::cuda ALIAS feature_elm_cuda)

  target_link_libraries(feature_elm_cuda PUBLIC CUDA::cudart CUDA::cublas CUDA::cusolver)

  target_include_directories(
    feature_elm_cuda
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
           $<INSTALL_INTERFACE:include>
           $<TARGET_PROPERTY:CUDA::cudart,INTERFACE_INCLUDE_DIRECTORIES>)

  target_link_libraries(feature_elm_core PUBLIC feature_elm::cuda)
else()
  add_library(feature_elm_cuda src/cuda/elm_gpu_stub.cpp)
  add_library(feature_elm::cuda ALIAS feature_elm_cuda)
  target_include_directories(feature_elm_cuda PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
  target_link_libraries(feature_elm_core PUBLIC feature_elm::cuda)
endif()

if(MSVC)
  target_compile_options(feature_elm_core PRIVATE /W4 /WX)
else()
  target_compile_options(feature_elm_core PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

add_subdirectory(src/app)

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

  add_executable(test_os_elm tests/test_os_elm.cpp)
  target_link_libraries(test_os_elm PRIVATE feature_elm::core feature_elm::cuda GTest::gtest_main)
  target_compile_features(test_os_elm PRIVATE cxx_std_20)

  add_executable(test_os_celm_cpu tests/test_os_celm_cpu.cpp)
  target_link_libraries(test_os_celm_cpu PRIVATE feature_elm::core GTest::gtest_main)
  target_compile_features(test_os_celm_cpu PRIVATE cxx_std_20)

  add_executable(test_h_os_elm_cpu tests/test_h_os_elm_cpu.cpp)
  target_link_libraries(test_h_os_elm_cpu PRIVATE feature_elm::core GTest::gtest_main)
  target_compile_features(test_h_os_elm_cpu PRIVATE cxx_std_20)

  add_executable(test_rbf_features tests/test_rbf_features.cpp)
  target_link_libraries(test_rbf_features PRIVATE feature_elm::core GTest::gtest_main)
  target_compile_features(test_rbf_features PRIVATE cxx_std_20)

  add_executable(test_golden_baseline tests/test_golden_baseline.cpp)
  target_link_libraries(test_golden_baseline PRIVATE feature_elm::core GTest::gtest_main)
  target_compile_features(test_golden_baseline PRIVATE cxx_std_20)

  add_executable(test_demo_api tests/test_demo_api.cpp)
  target_link_libraries(test_demo_api PRIVATE demo_backend GTest::gtest_main)
  target_compile_features(test_demo_api PRIVATE cxx_std_20)

  include(GoogleTest)
  gtest_discover_tests(test_sanity)
  gtest_discover_tests(test_elm_cpu)
  gtest_discover_tests(test_cuda_infra TEST_PREFIX cuda_infra.)
  gtest_discover_tests(test_elm_gpu TEST_PREFIX elm_gpu.)
  gtest_discover_tests(test_os_elm TEST_PREFIX os_elm.)
  gtest_discover_tests(test_os_celm_cpu TEST_PREFIX os_celm.)
  gtest_discover_tests(test_h_os_elm_cpu TEST_PREFIX h_os_elm.)
  gtest_discover_tests(test_rbf_features TEST_PREFIX rbf_features.)
  gtest_discover_tests(test_golden_baseline TEST_PREFIX golden_baseline.)
  gtest_discover_tests(test_demo_api TEST_PREFIX demo_api.)

  add_subdirectory(bench)
endif()