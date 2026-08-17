# Inference is header-only, so there is nothing to compile here and the target
# only carries include paths and the language requirement.
add_library(segmentlib INTERFACE)

# PROJECT_SOURCE_DIR, not CMAKE_SOURCE_DIR: the latter is the *top-level*
# source dir, so under add_subdirectory()/FetchContent it resolves to the
# consuming project's include/ and segmentlib fails to compile its own sources.
target_include_directories(segmentlib
    INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

# The dictionary matcher's FST comes from cpp-fstlib, vendored under
# third_party/. INTERFACE rather than PRIVATE now that mlp/dictionary.h holds
# the fst::map itself: a consumer including that header compiles it too.
# SYSTEM because it is not ours to keep clean under our own warning flags.
#
# third_party/ is on the path, not third_party/cpp-fstlib/, so the include in
# dictionary.h reads "cpp-fstlib/fstlib.h". That keeps a consumer's own
# fstlib.h from shadowing the vendored one, and keeps this directory's other
# files (README.md, LICENSE) off the include path entirely.
target_include_directories(segmentlib
    SYSTEM INTERFACE ${PROJECT_SOURCE_DIR}/third_party
)

# C++17 is the floor, not the ceiling: consumers may build at any later
# standard. Keeping inference reachable from a C++17 project is the point of
# the header-only form.
target_compile_features(segmentlib INTERFACE cxx_std_17)

if(SEGMENTLIB_BUILD_TRAINING)
    add_library(segmentlib_train STATIC
        mlp/train/corpus.cpp
        mlp/train/example.cpp
        mlp/train/dataset.cpp
        mlp/train/cpu_blas.cpp
        mlp/train/net.cpp
        mlp/train/adam.cpp
        mlp/train/trainer.cpp
        mlp/train/quantize.cpp
        mlp/train/exporter.cpp
    )
    # Training headers are internal (no public headers);
    # consumers of this target (CLI train, tests) include them as
    # "mlp/train/*.h" relative to src/.
    target_include_directories(segmentlib_train
        PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)
    target_link_libraries(segmentlib_train PUBLIC segmentlib)
    target_link_libraries(segmentlib_train PRIVATE segmentlib_warnings)

    # BLAS is linked here and nowhere else (design.ja.md 4.9).
    if(APPLE)
        set(SEGMENTLIB_BLAS "Accelerate" CACHE STRING
            "BLAS implementation for training (Accelerate|OpenBLAS|MKL)")
    else()
        set(SEGMENTLIB_BLAS "OpenBLAS" CACHE STRING
            "BLAS implementation for training (Accelerate|OpenBLAS|MKL)")
    endif()
    if(SEGMENTLIB_BLAS STREQUAL "Accelerate")
        target_link_libraries(segmentlib_train PRIVATE "-framework Accelerate")
    else()
        find_package(BLAS REQUIRED)
        target_link_libraries(segmentlib_train PRIVATE BLAS::BLAS)
    endif()
endif()

if(SEGMENTLIB_BUILD_CLI)
    add_subdirectory(cli)
endif()
