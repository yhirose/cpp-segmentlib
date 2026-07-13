#include <charconv>
#include <cstdio>
#include <iostream>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "commands.h"
#include "segmentlib/output.h"
#include "segmentlib/segmenter.h"

namespace segmentlib::cli {

namespace {

struct Options {
    std::string_view model_path;
    unsigned threads = 0;  // 0 = hardware_concurrency
    bool ok = true;
};

Options parse(std::span<const std::string_view> args) {
    Options opt;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view a = args[i];
        if (a == "--model" && i + 1 < args.size()) {
            opt.model_path = args[++i];
        } else if (a == "--threads" && i + 1 < args.size()) {
            const std::string_view v = args[++i];
            unsigned n = 0;
            const auto [_, ec] = std::from_chars(v.data(), v.data() + v.size(), n);
            if (ec != std::errc{}) {
                std::println(stderr, "predict: invalid --threads value '{}'", v);
                opt.ok = false;
            } else {
                opt.threads = n;
            }
        } else {
            std::println(stderr, "predict: unexpected argument '{}'", a);
            opt.ok = false;
        }
    }
    if (opt.model_path.empty()) {
        std::println(stderr, "predict: --model <path> is required");
        opt.ok = false;
    }
    return opt;
}

// Writes a segmented line (surface words) plus a newline.
void write_segments(const Segments& segments, std::string_view line, std::string& out) {
    out.clear();
    append_full_line(segments, line, out);
    out.push_back('\n');
}

}  // namespace

int run_predict(std::span<const std::string_view> args) {
    const Options opt = parse(args);
    if (!opt.ok) {
        return 2;
    }

    auto segmenter = Segmenter::load(std::filesystem::path(opt.model_path));
    if (!segmenter) {
        std::println(stderr, "predict: failed to load model: {}", segmenter.error().message);
        return 1;
    }

    std::ios::sync_with_stdio(false);

    // Process the input in blocks: read up to kBlock lines, segment them in
    // parallel, then write the results in input order. Blocking keeps peak
    // memory bounded (unlike buffering the whole stream) while still exposing
    // enough independent work per call to saturate the worker threads.
    constexpr std::size_t kBlock = 65536;
    std::vector<std::string> lines;
    std::vector<std::string_view> views;
    std::string line;
    std::string out;
    for (;;) {
        lines.clear();
        while (lines.size() < kBlock && std::getline(std::cin, line)) {
            lines.push_back(std::move(line));
            line.clear();
        }
        if (lines.empty()) {
            break;
        }

        views.assign(lines.begin(), lines.end());
        const auto results = segmenter->tokenize_all(views, opt.threads);
        for (std::size_t i = 0; i < results.size(); ++i) {
            if (!results[i]) {
                std::fflush(stdout);
                std::println(stderr, "predict: {}", results[i].error().message);
                return 1;
            }
            write_segments(*results[i], lines[i], out);
            std::fwrite(out.data(), 1, out.size(), stdout);
        }
    }
    return 0;
}

}  // namespace segmentlib::cli
