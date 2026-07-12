cmake_minimum_required(VERSION 3.24)
project(cpp-segmentlib CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(SEGMENTLIB_BUILD_TESTS "Build the test suite" ON)
option(SEGMENTLIB_BUILD_BENCH "Build the benchmark tools" ON)
# Training is a development tool, physically separated from inference: the
# segmentlib_train target is opt-in and is the only place BLAS is linked
# (design.ja.md 5.9, mlp_module_design.ja.md 0/5節).
option(SEGMENTLIB_BUILD_TRAINING "Build the MLP training components (segmentlib_train)" OFF)

add_subdirectory(src)

if(SEGMENTLIB_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(SEGMENTLIB_BUILD_BENCH)
    add_subdirectory(bench)
endif()
