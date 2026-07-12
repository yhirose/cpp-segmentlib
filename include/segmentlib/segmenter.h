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
class Segmenter {
public:
    // Loads a model, auto-detecting its format: a "SegmentLibMLP " header
    // line selects the MLP backend (design.ja.md 5.7), anything else is
    // treated as a KyTea binary model.
    static std::expected<Segmenter, Error> load(const std::filesystem::path& model_path);

    // Loads a model, forcing the KyTea backend.
    static std::expected<Segmenter, Error> load_kytea(const std::filesystem::path& model_path);

    // Loads a model, forcing the MLP backend.
    static std::expected<Segmenter, Error> load_mlp(const std::filesystem::path& model_path);

    [[nodiscard]] std::expected<Segments, Error> tokenize(std::string_view text) const {
        return std::visit([&](const auto& b) { return b.tokenize(text); }, backend_);
    }

    [[nodiscard]] std::expected<Boundaries, Error> tokenize_boundaries(std::string_view text) const {
        return std::visit([&](const auto& b) { return b.tokenize_boundaries(text); }, backend_);
    }

    // Tokenizes many inputs in parallel; result[i] corresponds to texts[i].
    // Inputs are independent and the model is immutable, so throughput scales
    // near-linearly with cores. `threads == 0` uses hardware_concurrency().
    [[nodiscard]] std::vector<std::expected<Segments, Error>> tokenize_all(
        std::span<const std::string_view> texts, unsigned threads = 0) const;

    // Word-boundaries-only counterpart of tokenize_all (no tag prediction), for
    // the fast segmentation-only path. result[i] corresponds to texts[i].
    [[nodiscard]] std::vector<std::expected<Boundaries, Error>> tokenize_boundaries_all(
        std::span<const std::string_view> texts, unsigned threads = 0) const;

private:
    using AnyBackend = std::variant<kytea::KyteaBackend, mlp::MlpBackend>;

    explicit Segmenter(AnyBackend backend) noexcept : backend_(std::move(backend)) {}

    AnyBackend backend_;
};

}  // namespace segmentlib
