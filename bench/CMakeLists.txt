find_package(benchmark QUIET)
if (NOT benchmark_FOUND)
  FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.7.1
  )
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)
endif()

add_executable(bench_elm_core bench_elm_core.cpp)
target_link_libraries(bench_elm_core PRIVATE feature_elm::core benchmark::benchmark)
target_compile_features(bench_elm_core PRIVATE cxx_std_20)

add_executable(bench_elm_cuda bench_elm_cuda.cpp)
target_link_libraries(bench_elm_cuda PRIVATE feature_elm::core feature_elm::cuda benchmark::benchmark)
target_compile_features(bench_elm_cuda PRIVATE cxx_std_20)

add_executable(bench_feature_maps bench_feature_maps.cpp)
target_link_libraries(bench_feature_maps PRIVATE feature_elm::core benchmark::benchmark)
target_compile_features(bench_feature_maps PRIVATE cxx_std_20)

add_executable(bench_solvers bench_solvers.cpp)
target_link_libraries(bench_solvers PRIVATE feature_elm::core feature_elm::cuda benchmark::benchmark)
target_compile_features(bench_solvers PRIVATE cxx_std_20)

add_executable(bench_ml_elm bench_ml_elm.cpp)
target_link_libraries(bench_ml_elm PRIVATE feature_elm::core benchmark::benchmark)
target_compile_features(bench_ml_elm PRIVATE cxx_std_20)
