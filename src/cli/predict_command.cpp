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

// One worker per line is already past the point of any gain, and each thread
// costs a stack; without a cap `--threads 20000` really did spawn 20000 of
// them. std::thread's constructor throws once the OS refuses, which run_batch
// does not catch, so an unbounded value turns a typo into a terminate().
constexpr unsigned kMaxThreads = 1024;

Options parse(std::span<const std::string_view> args) {
    Options opt;
    const auto need_value = [&](std::size_t& i) -> std::string_view {
        if (i + 1 >= args.size()) {
            std::println(stderr, "predict: {} requires a value", args[i]);
            opt.ok = false;
            return {};
        }
        return args[++i];
    };
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view a = args[i];
        if (a == "--model") {
            const std::string_view v = need_value(i);
            if (opt.ok) {
                opt.model_path = v;
            }
        } else if (a == "--threads") {
            const std::string_view v = need_value(i);
            if (!opt.ok) {
                continue;
            }
            unsigned n = 0;
            const auto [ptr, ec] = std::from_chars(v.data(), v.data() + v.size(), n);
            // ptr, not just ec: from_chars stops at the first non-digit and
            // reports success for the prefix, so checking ec alone accepted
            // "4abc" as 4. train_command.cpp has always checked both.
            if (ec != std::errc{} || ptr != v.data() + v.size()) {
                std::println(stderr, "predict: invalid --threads value '{}'", v);
                opt.ok = false;
            } else if (n > kMaxThreads) {
                std::println(stderr, "predict: --threads {} exceeds the maximum of {}", n,
                             kMaxThreads);
                opt.ok = false;
            } else {
                opt.threads = n;
            }
        } else {
            std::println(stderr, "predict: unexpected argument '{}'", a);
            opt.ok = false;
        }
    }
    if (opt.ok && opt.model_path.empty()) {
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
    // Lines are taken verbatim, including a trailing '\r' on CRLF input. That
    // looks like a bug (the CR becomes its own token) but is deliberate: real
    // KyTea does exactly the same, and the golden tests hold this CLI to
    // byte-for-byte parity with it. Stripping the CR here would break that.
    constexpr std::size_t kBlock = 65536;
    std::vector<std::string> lines;
    std::vector<std::string_view> views;
    std::string line;
    std::string out;
    std::size_t line_no = 0;  // 1-based index of lines[0] within the whole stream
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
                // Aborting the whole stream on a bad line matches KyTea and
                // Vaporetto, which both stop with a non-zero status, so the
                // contract stays the same. What they also do and this did not
                // is say *where*: without a line number a single bad byte in a
                // million-line corpus is unfindable.
                std::fflush(stdout);
                std::println(stderr, "predict: line {}: {}", line_no + i + 1,
                             results[i].error().message);
                return 1;
            }
            write_segments(*results[i], lines[i], out);
            std::fwrite(out.data(), 1, out.size(), stdout);
        }
        line_no += lines.size();
    }
    return 0;
}

}  // namespace segmentlib::cli
