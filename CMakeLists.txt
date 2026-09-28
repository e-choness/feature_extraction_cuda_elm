cmake_minimum_required(VERSION 3.24)

project(
  feature_extraction_cuda_elm
  VERSION 0.2.0
  DESCRIPTION "GPU-accelerated Extreme Learning Machine feature extraction"
  LANGUAGES CXX)

include(CTest)
include(FetchContent)
include(CheckLanguage)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_CUDA "Enable CUDA backend support (falls back to CPU stubs if nvcc is missing)" ON)
# Shared C API (libfeature_elm_capi.so) for dynamic hosts such as the ZeroGPU Gradio Space.
option(FEATURE_ELM_BUILD_CAPI "Build the libfeature_elm_capi shared library" OFF)
if(FEATURE_ELM_BUILD_CAPI)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

set(FEATURE_ELM_HAS_CUDA OFF)
if(ENABLE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    # Default fat binary: Turing (T4), Ampere (A100, A10G), Ada (L4, L40S, RTX 40xx), Hopper and
    # Blackwell, plus PTX for newer parts. Override with -DCMAKE_CUDA_ARCHITECTURES=native for
    # faster local builds.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;86-real;89-real;90-real;120")
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    find_package(CUDAToolkit REQUIRED)
    set(FEATURE_ELM_HAS_CUDA ON)
  else()
    message(STATUS "ENABLE_CUDA is ON but no CUDA compiler was found; building CPU-only stubs")
  endif()
endif()

# Header-only third-party dependencies, pinned by release archive and SHA-256.
FetchContent_Declare(
  httplib
  URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.58.0.tar.gz
  URL_HASH SHA256=31932ce8b33f2905472a987dc1984b9d8a7d338083a066021349587cc5f1cfea
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_SUBDIR _no_cmake_)
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_SUBDIR _no_cmake_)
FetchContent_MakeAvailable(httplib nlohmann_json)

add_library(feature_elm_third_party INTERFACE)
add_library(feature_elm::third_party ALIAS feature_elm_third_party)
target_include_directories(feature_elm_third_party SYSTEM
  INTERFACE ${httplib_SOURCE_DIR} ${nlohmann_json_SOURCE_DIR}/include)
target_link_libraries(feature_elm_third_party INTERFACE Threads::Threads)

add_library(feature_elm_core
    src/core/version.cpp
    src/core/random_additive_map.cpp
    src/core/identity_map.cpp
    src/core/batch_ridge_solver.cpp
    src/core/elm.cpp
    src/core/os_elm.cpp
    src/core/os_celm.cpp
    src/core/rls_solver.cpp
    src/core/h_os_elm.cpp
    src/core/elm_ae.cpp
    src/core/stacked_feature_map.cpp
    src/core/ml_elm.cpp
    src/core/rbf_features.cpp
    src/core/rbf_map.cpp
    src/io/dataset.cpp
    src/io/preprocess.cpp
    src/io/drift_stream.cpp
    src/io/idx_dataset.cpp)
add_library(feature_elm::core ALIAS feature_elm_core)

target_include_directories(
  feature_elm_core
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
         $<INSTALL_INTERFACE:include>)
target_compile_features(feature_elm_core PUBLIC cxx_std_20)

# CPU hot loops are multithreaded with OpenMP (see src/core/parallel.hpp). Results are
# bit-identical to a serial build; OMP_NUM_THREADS controls the thread count.
option(FEATURE_ELM_OPENMP "Multithread CPU hot loops with OpenMP" ON)
if(FEATURE_ELM_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(feature_elm_core PUBLIC OpenMP::OpenMP_CXX)
  else()
    message(STATUS "FEATURE_ELM_OPENMP is ON but OpenMP was not found; CPU loops stay serial")
  endif()
endif()

if(FEATURE_ELM_HAS_CUDA)
  add_library(feature_elm_cuda
    src/cuda/device_buffer.cu
    src/cuda/gpu_ops.cu
    src/cuda/solver_gpu.cu
    src/cuda/rls_gpu.cu)
  add_library(feature_elm::cuda ALIAS feature_elm_cuda)

  # Static CUDA math libraries let the demo image run on the slim nvidia/cuda "base" image.
  option(FEATURE_ELM_STATIC_CUDA "Link cuBLAS/cuSOLVER statically" OFF)
  if(FEATURE_ELM_STATIC_CUDA)
    target_link_libraries(feature_elm_cuda PUBLIC CUDA::cudart_static CUDA::cublas_static
                                                  CUDA::cublasLt_static CUDA::cusolver_static)
  else()
    target_link_libraries(feature_elm_cuda PUBLIC CUDA::cudart CUDA::cublas CUDA::cusolver)
  endif()

  target_include_directories(
    feature_elm_cuda
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
           $<INSTALL_INTERFACE:include>
           $<TARGET_PROPERTY:CUDA::cudart,INTERFACE_INCLUDE_DIRECTORIES>)

  target_link_libraries(feature_elm_core PUBLIC feature_elm::cuda)
else()
  add_library(feature_elm_cuda src/cuda/cuda_stubs.cpp)
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

add_executable(felm_train src/tools/felm_train.cpp)
target_link_libraries(felm_train PRIVATE feature_elm::core feature_elm::cuda)
target_compile_features(felm_train PRIVATE cxx_std_20)
set_target_properties(felm_train PROPERTIES OUTPUT_NAME felm-train)

if(FEATURE_ELM_BUILD_CAPI)
  add_library(feature_elm_capi SHARED src/capi/feature_elm_capi.cpp)
  target_link_libraries(feature_elm_capi PRIVATE demo_backend feature_elm::cuda)
  target_compile_features(feature_elm_capi PRIVATE cxx_std_20)
  # Export only the felm_* entry points; keep bundled C++ runtime and static-library symbols
  # private so they cannot clash with whatever else the host process has loaded.
  set_target_properties(feature_elm_capi PROPERTIES CXX_VISIBILITY_PRESET hidden
                                                    VISIBILITY_INLINES_HIDDEN ON)
  if(NOT MSVC)
    target_link_options(feature_elm_capi PRIVATE -Wl,--exclude-libs,ALL -static-libstdc++
                        -static-libgcc)
  endif()
endif()

if(BUILD_TESTING)
  find_package(GTest REQUIRED)
  include(GoogleTest)

  add_executable(test_sanity tests/test_sanity.cpp)
  target_link_libraries(test_sanity PRIVATE feature_elm::core GTest::gtest_main)
  target_compile_features(test_sanity PRIVATE cxx_std_20)

  add_executable(test_feature_map tests/test_feature_map.cpp)
  target_link_libraries(test_feature_map PRIVATE feature_elm::core GTest::gtest_main)
  target_compile_features(test_feature_map PRIVATE cxx_std_20)

  add_executable(test_elm_cpu tests/test_elm_cpu.cpp)
  target_link_libraries(test_elm_cpu PRIVATE feature_elm::core GTest::gtest_main)
  target_compile_features(test_elm_cpu PRIVATE cxx_std_20)

  add_executable(test_solver tests/test_solver.cpp)
  target_link_libraries(test_solver PRIVATE feature_elm::core GTest::gtest_main)
  target_compile_features(test_solver PRIVATE cxx_std_20)

  if(FEATURE_ELM_HAS_CUDA)
    add_executable(test_cuda_infra tests/test_cuda_infra.cpp)
    target_link_libraries(test_cuda_infra PRIVATE feature_elm::cuda GTest::gtest_main)
    target_compile_features(test_cuda_infra PRIVATE cxx_std_20)
    gtest_discover_tests(test_cuda_infra TEST_PREFIX cuda_infra.)
  endif()

  add_executable(test_elm_gpu tests/test_elm_gpu.cpp)
  target_link_libraries(test_elm_gpu PRIVATE feature_elm::core feature_elm::cuda GTest::gtest_main)
  target_compile_features(test_elm_gpu PRIVATE cxx_std_20)

  add_executable(test_gpu_ops tests/test_gpu_ops.cpp)
  target_link_libraries(test_gpu_ops PRIVATE feature_elm::core feature_elm::cuda GTest::gtest_main)
  target_compile_features(test_gpu_ops PRIVATE cxx_std_20)

  add_executable(test_os_elm tests/test_os_elm.cpp)
  target_link_libraries(test_os_elm PRIVATE feature_elm::core feature_elm::cuda GTest::gtest_main)
  target_compile_features(test_os_elm PRIVATE cxx_std_20)

  add_executable(test_os_celm_cpu tests/test_os_celm_cpu.cpp)
  target_link_libraries(test_os_celm_cpu PRIVATE feature_elm::core GTest::gtest_main)
  target_compile_features(test_os_celm_cpu PRIVATE cxx_std_20)

  add_executable(test_h_os_elm_cpu tests/test_h_os_elm_cpu.cpp)
  target_link_libraries(test_h_os_elm_cpu PRIVATE feature_elm::core GTest::gtest_main)
  target_compile_features(test_h_os_elm_cpu PRIVATE cxx_std_20)

  add_executable(test_elm_ae tests/test_elm_ae.cpp)
  target_link_libraries(test_elm_ae PRIVATE feature_elm::core GTest::gtest_main)
  target_compile_features(test_elm_ae PRIVATE cxx_std_20)

  add_executable(test_ml_elm_cpu tests/test_ml_elm_cpu.cpp)
  target_link_libraries(test_ml_elm_cpu PRIVATE feature_elm::core GTest::gtest_main)
  target_compile_features(test_ml_elm_cpu PRIVATE cxx_std_20)

  add_executable(test_rbf_features tests/test_rbf_features.cpp)
  target_link_libraries(test_rbf_features PRIVATE feature_elm::core GTest::gtest_main)
  target_compile_features(test_rbf_features PRIVATE cxx_std_20)

  add_executable(test_rbf_map tests/test_rbf_map.cpp)
  target_link_libraries(test_rbf_map PRIVATE feature_elm::core GTest::gtest_main)
  target_compile_features(test_rbf_map PRIVATE cxx_std_20)

  add_executable(test_golden_baseline tests/test_golden_baseline.cpp)
  target_link_libraries(test_golden_baseline PRIVATE feature_elm::core GTest::gtest_main)
  target_compile_features(test_golden_baseline PRIVATE cxx_std_20)

  add_executable(test_demo_api tests/test_demo_api.cpp)
  target_link_libraries(test_demo_api PRIVATE demo_backend feature_elm::cuda feature_elm::third_party GTest::gtest_main)
  target_compile_definitions(test_demo_api PRIVATE FEATURE_ELM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_features(test_demo_api PRIVATE cxx_std_20)

  add_executable(test_dataset_io tests/test_dataset_io.cpp)
  target_link_libraries(test_dataset_io PRIVATE feature_elm::core GTest::gtest_main)
  target_compile_definitions(test_dataset_io PRIVATE FEATURE_ELM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_features(test_dataset_io PRIVATE cxx_std_20)

  gtest_discover_tests(test_sanity)
  gtest_discover_tests(test_feature_map TEST_PREFIX feature_map.)
  gtest_discover_tests(test_elm_cpu TEST_PREFIX elm_cpu.)
  gtest_discover_tests(test_solver TEST_PREFIX solver.)
  gtest_discover_tests(test_elm_gpu TEST_PREFIX elm_gpu.)
  gtest_discover_tests(test_gpu_ops TEST_PREFIX gpu_ops.)
  gtest_discover_tests(test_os_elm TEST_PREFIX os_elm.)
  gtest_discover_tests(test_os_celm_cpu TEST_PREFIX os_celm.)
  gtest_discover_tests(test_h_os_elm_cpu TEST_PREFIX h_os_elm.)
  gtest_discover_tests(test_elm_ae TEST_PREFIX elm_ae.)
  gtest_discover_tests(test_ml_elm_cpu TEST_PREFIX ml_elm.)
  gtest_discover_tests(test_rbf_features TEST_PREFIX rbf_features.)
  gtest_discover_tests(test_rbf_map TEST_PREFIX rbf_map.)
  gtest_discover_tests(test_golden_baseline TEST_PREFIX golden_baseline.)
  gtest_discover_tests(test_demo_api TEST_PREFIX demo_api.)
  gtest_discover_tests(test_dataset_io TEST_PREFIX dataset_io.)

  add_subdirectory(bench)
endif()
