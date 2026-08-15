# A separate project that consumes segmentlib the way a downstream user would.
# It is NOT part of the main build (tests/CMakeLists.txt lists its sources
# explicitly and does not add this directory); configure it on its own:
#
#   cmake -S tests/consumer -B build-consumer -DSEGMENTLIB_ROOT=$PWD
#   cmake --build build-consumer
#   ./build-consumer/consumer
#
# Deliberately passes no SEGMENTLIB_* options, so it inherits whatever defaults
# a real consumer would get. scripts/check_consumer.sh drives this and asserts
# that the defaults keep our developer targets out of the consumer's build.
cmake_minimum_required(VERSION 3.24)
project(segmentlib_consumer_check CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT SEGMENTLIB_ROOT)
    message(FATAL_ERROR "set -DSEGMENTLIB_ROOT to the segmentlib checkout")
endif()

# add_subdirectory() rather than FetchContent so the check runs against the
# working tree instead of a pushed commit. FetchContent reduces to this once
# the clone is done, so the substance is the same.
add_subdirectory(${SEGMENTLIB_ROOT} segmentlib-build)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE segmentlib)
