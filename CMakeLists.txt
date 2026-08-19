cmake_minimum_required(VERSION 3.24)
# The VERSION here and the SEGMENTLIB_VERSION macros in
# include/segmentlib/types.h are bumped together at release time
# (docs/RELEASING.md); there is no automatic sync.
project(cpp-segmentlib VERSION 0.1.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Only when this is the build being run. Under add_subdirectory()/FetchContent
# the cache is the consuming project's, so forcing a build type there decides
# it for their code too, from a dependency they added for one static library.
if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# MSVC needs /EHsc for exceptions (doctest requires them), /utf-8 for the
# UTF-8 string literals throughout the codebase, and /Zc:__cplusplus so that
# __cplusplus reports the standard actually in use: without it MSVC still
# answers 199711L, and cpp-fstlib refuses to compile on its C++17 check.
# add_compile_options keeps these even when CI overrides CMAKE_CXX_FLAGS (e.g.
# to add /arch:AVX2), which would otherwise drop CMake's default /EHsc and
# disable exceptions.
if(MSVC)
    add_compile_options(/EHsc /utf-8 /Zc:__cplusplus)
endif()

# Developer targets default to on for a standalone build and off for a
# consumer: a project that pulls this in via FetchContent wants one static
# library, not our test suite (which fetches doctest over the network), our
# benchmarks, or our CLI in their `all` target.
option(SEGMENTLIB_BUILD_TESTS "Build the test suite" ${PROJECT_IS_TOP_LEVEL})
option(SEGMENTLIB_BUILD_BENCH "Build the benchmark tools" ${PROJECT_IS_TOP_LEVEL})
option(SEGMENTLIB_BUILD_CLI "Build the segmenter CLI" ${PROJECT_IS_TOP_LEVEL})
option(SEGMENTLIB_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

# Warnings are carried by an interface target rather than add_compile_options so
# they apply to our code only: doctest arrives via FetchContent and is not ours
# to keep warning-clean. Linked PRIVATE everywhere, so consumers never inherit
# them either.
add_library(segmentlib_warnings INTERFACE)
if(MSVC)
    target_compile_options(segmentlib_warnings INTERFACE /W4)
    if(SEGMENTLIB_WARNINGS_AS_ERRORS)
        target_compile_options(segmentlib_warnings INTERFACE /WX)
    endif()
else()
    target_compile_options(segmentlib_warnings INTERFACE -Wall -Wextra)
    if(SEGMENTLIB_WARNINGS_AS_ERRORS)
        target_compile_options(segmentlib_warnings INTERFACE -Werror)
    endif()
endif()
# Training is a development tool, physically separated from inference: the
# segmentlib_train target is opt-in and is the only place BLAS is linked
# (design.ja.md 4.9).
option(SEGMENTLIB_BUILD_TRAINING "Build the MLP training components (segmentlib_train)" OFF)

add_subdirectory(src)

if(SEGMENTLIB_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(SEGMENTLIB_BUILD_BENCH)
    add_subdirectory(bench)
endif()
