#include <doctest/doctest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "segmentlib/output.h"
#include "segmentlib/segmenter.h"

// Paths injected by CMake so the test can find the model and fixtures
// regardless of the working directory.
#ifndef SEGMENTLIB_MODEL_PATH
#define SEGMENTLIB_MODEL_PATH ""
#endif
#ifndef SEGMENTLIB_GOLDEN_DIR
#define SEGMENTLIB_GOLDEN_DIR ""
#endif

using namespace segmentlib;

namespace {

// True when the 128 MB KyTea model is present. Absence normally skips, since
// the model is gitignored and fetched on demand — but a skip that reports
// success is indistinguishable from a pass, and these are the two highest-value
// tests in the suite (byte parity with real KyTea, and parallel-vs-serial
// agreement). Setting SEGMENTLIB_REQUIRE_GOLDEN turns absence into a failure so
// a CI job that is supposed to have fetched the model cannot silently stop
// running them.
bool golden_model_available() {
    const std::filesystem::path model_path{SEGMENTLIB_MODEL_PATH};
    if (!model_path.empty() && std::filesystem::exists(model_path)) {
        return true;
    }
    const bool required = std::getenv("SEGMENTLIB_REQUIRE_GOLDEN") != nullptr;
    REQUIRE_MESSAGE(!required,
                    "SEGMENTLIB_REQUIRE_GOLDEN is set but the model is missing at '"
                        << SEGMENTLIB_MODEL_PATH << "'");
    MESSAGE("golden: model not found at '" << SEGMENTLIB_MODEL_PATH
                                           << "'; run scripts/fetch_kytea_model.sh. Skipping.");
    return false;
}

}  // namespace

// Byte-for-byte parity with KyTea on a fixed corpus. The reference output in
// expected.txt was produced by `kytea -model jp-0.4.7-5.mod -notags`.
TEST_CASE("KyTea byte-exact parity on the golden corpus") {
    if (!golden_model_available()) {
        return;
    }
    const std::filesystem::path model_path{SEGMENTLIB_MODEL_PATH};

    auto segmenter = Segmenter::load(model_path);
    REQUIRE(segmenter.has_value());

    const std::filesystem::path dir{SEGMENTLIB_GOLDEN_DIR};
    std::ifstream input(dir / "input.txt");
    std::ifstream expected(dir / "expected.txt");
    REQUIRE(input.good());
    REQUIRE(expected.good());

    std::string line;
    std::string want;
    std::string got;
    std::size_t line_no = 0;
    while (std::getline(input, line)) {
        ++line_no;
        REQUIRE_MESSAGE(std::getline(expected, want), "expected.txt shorter than input.txt");

        auto segments = segmenter->tokenize(line);
        REQUIRE(segments.has_value());
        got.clear();
        append_full_line(*segments, line, got);

        CHECK_MESSAGE(got == want, "line " << line_no << ": got [" << got << "] want [" << want << "]");
    }
    CHECK_FALSE(std::getline(expected, want));  // no extra expected lines
}

// The parallel batch API must produce exactly the same result as serial
// tokenize() for every input, regardless of thread count.
TEST_CASE("tokenize_all matches serial tokenize") {
    if (!golden_model_available()) {
        return;
    }
    const std::filesystem::path model_path{SEGMENTLIB_MODEL_PATH};
    auto segmenter = Segmenter::load(model_path);
    REQUIRE(segmenter.has_value());

    std::ifstream input(std::filesystem::path{SEGMENTLIB_GOLDEN_DIR} / "input.txt");
    REQUIRE(input.good());
    std::vector<std::string> lines;
    for (std::string line; std::getline(input, line);) {
        lines.push_back(line);
    }
    REQUIRE_FALSE(lines.empty());

    std::vector<std::string_view> views(lines.begin(), lines.end());

    for (const unsigned threads : {1u, 2u, 4u, 8u}) {
        auto batch = segmenter->tokenize_all(views, threads);
        REQUIRE(batch.size() == views.size());
        for (std::size_t i = 0; i < views.size(); ++i) {
            auto serial = segmenter->tokenize(views[i]);
            REQUIRE(serial.has_value());
            REQUIRE(batch[i].has_value());
            CHECK(*batch[i] == *serial);
        }
    }
}
