#include "segmentlib/segmenter.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <fstream>
#include <string>
#include <thread>
#include <utility>

#include "segmentlib/kytea/model.h"
#include "segmentlib/mlp/model.h"

namespace segmentlib {

std::expected<Segmenter, Error> Segmenter::load_kytea(const std::filesystem::path& model_path) {
    auto model = kytea::Model::load(model_path);
    if (!model) {
        return std::unexpected(model.error());
    }
    return Segmenter(kytea::KyteaBackend(std::move(*model)));
}

std::expected<Segmenter, Error> Segmenter::load_mlp(const std::filesystem::path& model_path) {
    auto model = mlp::Model::load(model_path);
    if (!model) {
        return std::unexpected(model.error());
    }
    return Segmenter(mlp::MlpBackend(std::move(*model)));
}

std::expected<Segmenter, Error> Segmenter::load(const std::filesystem::path& model_path) {
    // Auto-detection by the file's leading bytes: the MLP format opens with
    // its ASCII signature line (design.ja.md 4.7); KyTea models start with
    // "KyTea " and anything unrecognized falls through to the KyTea loader's
    // own diagnostics. (Vaporetto's zstd magic joins here later.)
    std::ifstream in(model_path, std::ios::binary);
    if (!in) {
        return std::unexpected(Error{ErrorCode::IoError, "cannot open model file"});
    }
    std::string head(mlp::kModelSignature.size(), '\0');
    in.read(head.data(), static_cast<std::streamsize>(head.size()));
    if (in.gcount() == static_cast<std::streamsize>(head.size()) &&
        head == mlp::kModelSignature) {
        in.close();
        return load_mlp(model_path);
    }
    in.close();
    return load_kytea(model_path);
}

namespace {

// Runs `fn` over every text, parallelized across `threads` (0 = hardware
// concurrency), returning results in input order. Workers claim contiguous
// chunks via an atomic cursor (dynamic scheduling balances the varying line
// lengths; the chunk amortizes the atomic). Each thread writes disjoint result
// slots, and the per-item functions are re-entrant (the model is immutable, its
// scratch buffers are thread_local), so no locking is needed.
template <class Result, class Fn>
std::vector<std::expected<Result, Error>> run_batch(std::span<const std::string_view> texts,
                                                    unsigned threads, Fn fn) {
    std::vector<std::expected<Result, Error>> out(texts.size());
    if (texts.empty()) {
        return out;
    }

    unsigned n = threads != 0 ? threads : std::thread::hardware_concurrency();
    if (n == 0) {
        n = 1;
    }
    n = static_cast<unsigned>(std::min<std::size_t>(n, texts.size()));

    if (n == 1) {
        for (std::size_t i = 0; i < texts.size(); ++i) {
            out[i] = fn(texts[i]);
        }
        return out;
    }

    std::atomic<std::size_t> cursor{0};
    constexpr std::size_t kChunk = 64;
    const auto worker = [&] {
        for (;;) {
            const std::size_t start = cursor.fetch_add(kChunk, std::memory_order_relaxed);
            if (start >= texts.size()) {
                break;
            }
            const std::size_t end = std::min(start + kChunk, texts.size());
            for (std::size_t i = start; i < end; ++i) {
                out[i] = fn(texts[i]);
            }
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(n - 1);
    for (unsigned t = 0; t + 1 < n; ++t) {
        pool.emplace_back(worker);
    }
    worker();  // the calling thread participates too
    for (auto& th : pool) {
        th.join();
    }
    return out;
}

}  // namespace

std::vector<std::expected<Segments, Error>> Segmenter::tokenize_all(
    std::span<const std::string_view> texts, unsigned threads) const {
    return run_batch<Segments>(texts, threads,
                               [this](std::string_view t) { return tokenize(t); });
}

}  // namespace segmentlib
