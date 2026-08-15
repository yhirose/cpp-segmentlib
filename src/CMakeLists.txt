add_library(segmentlib STATIC
    unicode/utf8.cpp
    unicode/egc.cpp
    unicode/normalize.cpp
    kytea/char_table.cpp
    kytea/model.cpp
    kytea/scorer.cpp
    kytea/kytea_backend.cpp
    mlp/vocab.cpp
    mlp/dictionary.cpp
    mlp/precompute.cpp
    mlp/model.cpp
    mlp/scorer.cpp
    mlp/mlp_backend.cpp
    segmenter.cpp
    output.cpp
)

# PROJECT_SOURCE_DIR, not CMAKE_SOURCE_DIR: the latter is the *top-level*
# source dir, so under add_subdirectory()/FetchContent it resolves to the
# consuming project's include/ and segmentlib fails to compile its own sources.
target_include_directories(segmentlib
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

target_compile_features(segmentlib PUBLIC cxx_std_23)
target_link_libraries(segmentlib PRIVATE segmentlib_warnings)

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
    # Training headers are internal (no public headers, mlp_module_design 0節);
    # consumers of this target (CLI train, tests) include them as
    # "mlp/train/*.h" relative to src/.
    target_include_directories(segmentlib_train
        PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)
    target_link_libraries(segmentlib_train PUBLIC segmentlib)
    target_link_libraries(segmentlib_train PRIVATE segmentlib_warnings)

    # BLAS is linked here and nowhere else (design.ja.md 5.9).
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

add_subdirectory(cli)
