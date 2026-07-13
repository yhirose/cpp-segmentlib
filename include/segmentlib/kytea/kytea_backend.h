#pragma once

#include <expected>
#include <string_view>
#include <utility>

#include "segmentlib/kytea/model.h"
#include "segmentlib/types.h"

namespace segmentlib::kytea {

// The KyTea-compatible segmentation backend: holds a loaded Model and turns
// text into words using the pointwise scorer. Satisfies the backend interface
// the Segmenter dispatches to. Word segmentation only — no tag prediction.
class KyteaBackend {
public:
    explicit KyteaBackend(Model model) noexcept : model_(std::move(model)) {}

    [[nodiscard]] std::expected<Segments, Error> tokenize(std::string_view text) const;

    [[nodiscard]] const Model& model() const noexcept { return model_; }

private:
    Model model_;
};

}  // namespace segmentlib::kytea
