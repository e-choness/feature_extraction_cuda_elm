add_library(demo_backend STATIC demo_backend.cpp)
target_include_directories(demo_backend PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src> $<INSTALL_INTERFACE:include>)
target_link_libraries(demo_backend PRIVATE feature_elm::core)
if(TARGET feature_elm::cuda)
  target_link_libraries(demo_backend PRIVATE feature_elm::cuda)
endif()
target_compile_features(demo_backend PRIVATE cxx_std_20)

target_compile_options(demo_backend PRIVATE
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>)

target_include_directories(demo_backend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(demo_app demo_main.cpp)
target_include_directories(demo_app PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(demo_app PRIVATE demo_backend)
target_compile_features(demo_app PRIVATE cxx_std_20)

target_compile_options(demo_app PRIVATE
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>)
