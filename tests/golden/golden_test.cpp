#include <doctest/doctest.h>

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

// Byte-for-byte parity with KyTea on a fixed corpus. The reference output in
// expected.txt was produced by `kytea -model jp-0.4.7-5.mod -notags`. Requires
// the (gitignored) model; if it is absent the test is skipped rather than
// failing, so CI without the model still passes.
TEST_CASE("KyTea byte-exact parity on the golden corpus") {
    const std::filesystem::path model_path{SEGMENTLIB_MODEL_PATH};
    if (model_path.empty() || !std::filesystem::exists(model_path)) {
        MESSAGE("golden: model not found at '" << SEGMENTLIB_MODEL_PATH
                                               << "'; run scripts/fetch_kytea_model.sh. Skipping.");
        return;
    }

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

        // The boundaries-only path (tokenize_boundaries + append_boundary_line,
        // what `predict --boundaries-only` uses) must produce byte-identical
        // segmentation output — the same words, from the tag-free fast path.
        auto cuts = segmenter->tokenize_boundaries(line);
        REQUIRE(cuts.has_value());
        std::string got_b;
        append_boundary_line(*cuts, line, got_b);
        CHECK_MESSAGE(got_b == want, "boundary line " << line_no << ": got [" << got_b << "]");
    }
    CHECK_FALSE(std::getline(expected, want));  // no extra expected lines
}

// Byte-for-byte parity with KyTea's default (tagged) output — surface,
// part-of-speech, and reading for every word, including unknown-word readings
// estimated by the subword language model (stage B). The reference in
// expected_tags.txt was produced by `kytea -model jp-0.4.7-5.mod`. (The corpus
// avoids astral-plane characters, whose word segmentation intentionally diverges
// from KyTea's buggy findType; see char_table.h.)
TEST_CASE("KyTea tag-prediction parity (surface/POS/reading, byte-exact)") {
    const std::filesystem::path model_path{SEGMENTLIB_MODEL_PATH};
    if (model_path.empty() || !std::filesystem::exists(model_path)) {
        return;  // model absent: covered by the skip message in the WS parity case
    }
    auto segmenter = Segmenter::load(model_path);
    REQUIRE(segmenter.has_value());

    const std::filesystem::path dir{SEGMENTLIB_GOLDEN_DIR};
    std::ifstream input(dir / "input.txt");
    std::ifstream expected(dir / "expected_tags.txt");
    REQUIRE(input.good());
    REQUIRE(expected.good());

    std::string line;
    std::string want;
    std::string got;
    std::size_t line_no = 0;
    while (std::getline(input, line)) {
        ++line_no;
        REQUIRE_MESSAGE(std::getline(expected, want), "expected_tags.txt shorter than input.txt");

        auto segments = segmenter->tokenize(line);
        REQUIRE(segments.has_value());
        got.clear();
        append_tagged_line(*segments, line, got);

        CHECK_MESSAGE(got == want, "line " << line_no << ": got [" << got << "] want [" << want << "]");
    }
    CHECK_FALSE(std::getline(expected, want));  // no extra expected lines
}

// The parallel batch API must produce exactly the same result as serial
// tokenize() for every input, regardless of thread count.
TEST_CASE("tokenize_all matches serial tokenize") {
    const std::filesystem::path model_path{SEGMENTLIB_MODEL_PATH};
    if (model_path.empty() || !std::filesystem::exists(model_path)) {
        return;  // model absent: covered by the skip message above
    }
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
