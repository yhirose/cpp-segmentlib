#pragma once

#include <expected>
#include <filesystem>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

#include "segmentlib/kytea/kytea_backend.h"
#include "segmentlib/mlp/mlp_backend.h"
#include "segmentlib/types.h"

namespace segmentlib {

// The public segmentation API. A Segmenter owns one backend and dispatches to
// it; the backend type is hidden from callers. KyTea-compatible and MLP
// backends exist today; Vaporetto will join the variant later.
//
// Thread safety: a loaded Segmenter is immutable, and the per-call scratch
// buffers are thread_local, so any number of threads may call tokenize() or
// tokenize_all() concurrently on the same const Segmenter. That is the intended
// way to share one: load once, then hand out `const Segmenter&`.
class Segmenter {
public:
    // Loads a model, auto-detecting its format: a "SegmentLibMLP " header
    // line selects the MLP backend (design.ja.md 4.7), anything else is
    // treated as a KyTea binary model.
    static std::expected<Segmenter, Error> load(const std::filesystem::path& model_path);

    // Loads a model, forcing the KyTea backend.
    static std::expected<Segmenter, Error> load_kytea(const std::filesystem::path& model_path);

    // Loads a model, forcing the MLP backend.
    static std::expected<Segmenter, Error> load_mlp(const std::filesystem::path& model_path);

    // Move-only. A Segmenter owns its model outright — 128 MB for the
    // distributed KyTea model — so an accidental copy (passing by value, or
    // `auto b = a;`) would duplicate all of it: measured at 10 ms and ~400 MB,
    // with no diagnostic and no visible symptom beyond the memory. Since the
    // loaded model is immutable and safe to use concurrently, sharing one
    // wants a `const Segmenter&`, or a shared_ptr where ownership must be
    // shared; both say so at the use site, which a silent deep copy does not.
    Segmenter(const Segmenter&) = delete;
    Segmenter& operator=(const Segmenter&) = delete;
    Segmenter(Segmenter&&) = default;
    Segmenter& operator=(Segmenter&&) = default;

    [[nodiscard]] std::expected<Segments, Error> tokenize(std::string_view text) const {
        return std::visit([&](const auto& b) { return b.tokenize(text); }, backend_);
    }

    // Tokenizes many inputs in parallel; result[i] corresponds to texts[i].
    // Inputs are independent and the model is immutable, so throughput scales
    // near-linearly with cores. `threads == 0` uses hardware_concurrency().
    [[nodiscard]] std::vector<std::expected<Segments, Error>> tokenize_all(
        std::span<const std::string_view> texts, unsigned threads = 0) const;

private:
    using AnyBackend = std::variant<kytea::KyteaBackend, mlp::MlpBackend>;

    explicit Segmenter(AnyBackend backend) noexcept : backend_(std::move(backend)) {}

    AnyBackend backend_;
};

}  // namespace segmentlib
