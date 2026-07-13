// In-process inference microbenchmark for KyTea itself, via libkytea.
// Mirrors bench_segment: load once, then time calculateWS (word segmentation)
// over a corpus.
//
//   bench_kytea <model> <corpus> [iterations]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <kytea/kytea.h>
#include <kytea/kytea-struct.h>
#include <kytea/kytea-string.h>
#include <kytea/string-util.h>

using namespace kytea;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: bench_kytea <model> <corpus> [iterations]\n");
        return 2;
    }
    const int iterations = (argc > 3) ? std::atoi(argv[3]) : 5;

    Kytea analyzer;
    const auto load0 = std::chrono::steady_clock::now();
    analyzer.readModel(argv[1]);
    const auto load1 = std::chrono::steady_clock::now();
    const double load_ms = std::chrono::duration<double, std::milli>(load1 - load0).count();

    StringUtil* util = analyzer.getStringUtil();

    std::vector<std::string> lines;
    {
        std::ifstream in(argv[2]);
        std::string line;
        while (std::getline(in, line)) lines.push_back(line);
    }
    std::size_t chars = 0;
    for (const auto& l : lines) {
        for (const unsigned char b : l) {
            if ((b & 0xC0) != 0x80) ++chars;
        }
    }

    auto run_ws = [&]() -> std::size_t {
        std::size_t sink = 0;
        for (const auto& l : lines) {
            KyteaString surface = util->mapString(l);
            KyteaString norm = util->normalize(surface);
            KyteaSentence sent(surface, norm);
            analyzer.calculateWS(sent);
            sink += sent.words.size();
        }
        return sink;
    };

    std::size_t sink = run_ws();  // warmup

    double ws_best_ms = 1e300;
    for (int it = 0; it < iterations; ++it) {
        const auto t0 = std::chrono::steady_clock::now();
        sink += run_ws();
        const auto t1 = std::chrono::steady_clock::now();
        ws_best_ms = std::min(ws_best_ms, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    const double ws_mcps = static_cast<double>(chars) / (ws_best_ms / 1000.0) / 1e6;
    std::printf("model load:        %.0f ms\n", load_ms);
    std::printf("corpus:            %zu lines, %zu chars (codepoints)\n", lines.size(), chars);
    std::printf("best pass:         %.1f ms  (%d iterations)\n", ws_best_ms, iterations);
    std::printf("inference speed:   %.2f M chars/sec\n", ws_mcps);
    std::printf("(sink=%zu)\n", sink);
    return 0;
}
