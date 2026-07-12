#pragma once

#include <expected>
#include <string_view>
#include <utility>

#include "segmentlib/mlp/model.h"
#include "segmentlib/types.h"

namespace segmentlib::mlp {

// The MLP segmentation backend (mlp_module_design.ja.md 2.6): same shape as
// kytea::KyteaBackend. The MLP has no tag prediction, so tokenize simply
// converts boundaries into Segments with empty tag lists.
class MlpBackend {
public:
    explicit MlpBackend(Model model) noexcept : model_(std::move(model)) {}

    [[nodiscard]] std::expected<Segments, Error>
    tokenize(std::string_view text) const;

    // Word boundaries only: byte offsets of the cut points between words.
    [[nodiscard]] std::expected<Boundaries, Error>
    tokenize_boundaries(std::string_view text) const;

    [[nodiscard]] const Model& model() const noexcept { return model_; }

private:
    Model model_;
    // Hot-path scratch lives in thread_local buffers (mlp_backend.cpp, I.5):
    // no per-call allocation, and tokenize_all-style parallel use stays safe
    // because each thread owns its scratch and the Model is immutable.
};

}  // namespace segmentlib::mlp
