include(FetchContent)
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11
    GIT_SHALLOW ON
)
# doctest 2.4.11 ships a cmake_minimum_required below CMake 4's floor; allow it.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_MakeAvailable(doctest)
unset(CMAKE_POLICY_VERSION_MINIMUM)

add_executable(segmentlib_tests
    unit/test_main.cpp
    unit/utf8_test.cpp
    unit/egc_test.cpp
    unit/normalize_test.cpp
    unit/binary_reader_test.cpp
    unit/binary_writer_test.cpp
    unit/aho_corasick_test.cpp
    unit/vocab_test.cpp
    unit/char_table_test.cpp
    unit/automaton_test.cpp
    unit/model_test.cpp
    unit/mlp_model_test.cpp
    unit/mlp_kernels_test.cpp
    golden/golden_test.cpp
)

target_link_libraries(segmentlib_tests PRIVATE segmentlib doctest::doctest segmentlib_warnings)

if(SEGMENTLIB_BUILD_TRAINING)
    target_sources(segmentlib_tests PRIVATE
        unit/train_corpus_test.cpp
        unit/train_example_test.cpp
        unit/train_net_test.cpp
        unit/train_trainer_test.cpp
        unit/train_quantize_test.cpp
        unit/train_exporter_test.cpp
        unit/train_inference_test.cpp
    )
    target_link_libraries(segmentlib_tests PRIVATE segmentlib_train)
endif()

target_compile_definitions(segmentlib_tests PRIVATE
    SEGMENTLIB_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/kytea/jp-0.4.7-5.mod"
    SEGMENTLIB_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden/fixtures"
    SEGMENTLIB_GRAPHEME_BREAK_TEST_PATH="${PROJECT_SOURCE_DIR}/tests/data/GraphemeBreakTest.txt"
)

add_test(NAME segmentlib_tests COMMAND segmentlib_tests)
