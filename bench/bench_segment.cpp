// In-process inference microbenchmark for the segmentlib KyTea backend.
//
// Loads the model once, then repeatedly segments an in-memory corpus, timing
// only the tokenize calls (model load and I/O excluded). This isolates the pure
// inference cost of the library — the number a cross-tool CLI comparison cannot
// cleanly obtain for the other tools.
//
// The headline "inference speed" figure uses tokenize_boundaries() (word
// segmentation only, no tag/reading prediction) to stay apples-to-apples with
// bench_kytea.cpp (calculateWS only) and Vaporetto (segmentation only, no tag
// prediction) — see docs/design.ja.md 10.1's "conditions fixed" methodology.
// A second, separate figure reports tokenize() (segmentation + POS + reading,
// stage A-2/B), which the other two tools have no equivalent for, so it is
// not compared against them.
//
//   bench_segment <model> <corpus> [iterations]

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <print>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "segmentlib/segmenter.h"

namespace {

std::vector<std::string> read_lines(const std::string& path) {
    std::ifstream in(path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        lines.push_back(std::move(line));
        line.clear();
    }
    return lines;
}

// Counts Unicode codepoints (UTF-8 lead bytes), the unit Vaporetto reports.
std::size_t count_codepoints(const std::vector<std::string>& lines) {
    std::size_t n = 0;
    for (const auto& l : lines) {
        for (const unsigned char b : l) {
            if ((b & 0xC0) != 0x80) {
                ++n;
            }
        }
    }
    return n;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::println(stderr, "usage: bench_segment <model> <corpus> [iterations]");
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string corpus_path = argv[2];
    const int iterations = (argc > 3) ? std::atoi(argv[3]) : 5;

    const auto load_start = std::chrono::steady_clock::now();
    auto segmenter = segmentlib::Segmenter::load(model_path);
    const auto load_end = std::chrono::steady_clock::now();
    if (!segmenter) {
        std::println(stderr, "failed to load model: {}", segmenter.error().message);
        return 1;
    }
    const double load_ms =
        std::chrono::duration<double, std::milli>(load_end - load_start).count();

    const auto lines = read_lines(corpus_path);
    const std::size_t chars = count_codepoints(lines);

    // Warm up caches and branch predictors.
    std::size_t sink = 0;
    for (const auto& l : lines) {
        auto s = segmenter->tokenize_boundaries(l);
        if (s) sink += s->size();
    }

    double best_ms = 1e300;
    for (int it = 0; it < iterations; ++it) {
        const auto t0 = std::chrono::steady_clock::now();
        for (const auto& l : lines) {
            auto s = segmenter->tokenize_boundaries(l);
            if (s) sink += s->size();
        }
        const auto t1 = std::chrono::steady_clock::now();
        best_ms = std::min(best_ms, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    const double chars_per_sec = static_cast<double>(chars) / (best_ms / 1000.0);
    std::println("model load:        {:.0f} ms", load_ms);
    std::println("corpus:            {} lines, {} chars (codepoints)", lines.size(), chars);
    std::println("best pass (WS only, tokenize_boundaries): {:.1f} ms  ({} iterations)", best_ms,
                 iterations);
    std::println("inference speed:   {:.2f} M chars/sec", chars_per_sec / 1e6);

    // Tag prediction (segmentation + POS + reading, stage A-2/B). Not
    // apples-to-apples with the other tools; reported separately for reference.
    double tag_best_ms = 1e300;
    for (int it = 0; it < iterations; ++it) {
        const auto t0 = std::chrono::steady_clock::now();
        for (const auto& l : lines) {
            auto s = segmenter->tokenize(l);
            if (s) sink += s->size();
        }
        const auto t1 = std::chrono::steady_clock::now();
        tag_best_ms = std::min(tag_best_ms, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    const double tag_chars_per_sec = static_cast<double>(chars) / (tag_best_ms / 1000.0);
    std::println("\n(reference, not comparable to KyTea/Vaporetto below) tag prediction "
                 "(tokenize): {:.1f} ms  {:.2f} M chars/sec",
                 tag_best_ms, tag_chars_per_sec / 1e6);

    // Parallel throughput via the batch API, across a few thread counts.
    // tokenize_all() has no boundaries-only variant, so this includes tag
    // prediction too (segmentlib-only figure; not compared to KyTea/Vaporetto).
    std::vector<std::string_view> views(lines.begin(), lines.end());
    const std::span<const std::string_view> texts{views};
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    std::println("\nparallel (tokenize_all, tag prediction included, {} hw threads):", hw);
    double single_ms = 0.0;
    for (const unsigned t : {1u, 2u, 4u, hw}) {
        double best = 1e300;
        for (int it = 0; it < iterations; ++it) {
            const auto t0 = std::chrono::steady_clock::now();
            auto results = segmenter->tokenize_all(texts, t);
            const auto t1 = std::chrono::steady_clock::now();
            for (const auto& r : results) {
                if (r) sink += r->size();
            }
            best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        if (t == 1u) single_ms = best;
        const double mcps = static_cast<double>(chars) / (best / 1000.0) / 1e6;
        std::println("  {:2} threads: {:6.1f} ms   {:5.2f} M chars/sec   {:4.2f}x",
                     t, best, mcps, single_ms / best);
    }
    std::println("(sink={})", sink);  // defeat dead-code elimination
    return 0;
}
