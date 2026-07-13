#pragma once

#include <expected>
#include <string_view>
#include <utility>

#include "segmentlib/kytea/model.h"
#include "segmentlib/types.h"

namespace segmentlib::kytea {

// The KyTea-compatible segmentation backend: holds a loaded Model and turns
// text into words/boundaries using the pointwise scorer. Satisfies the backend
// interface the Segmenter dispatches to.
class KyteaBackend {
public:
    explicit KyteaBackend(Model model) noexcept : model_(std::move(model)) {}

    // Full segmentation. When the model carries tag data (do_tags), each
    // Segment's tag list is filled with the top candidate per level (POS,
    // reading, ...), including unknown-word readings (stage B: subword-dictionary
    // + language-model beam search, falling back to "UNK" when no candidate
    // exists — see predict_word_tags).
    [[nodiscard]] std::expected<Segments, Error> tokenize(std::string_view text) const;

    // Word boundaries only: byte offsets of the cut points between words.
    [[nodiscard]] std::expected<Boundaries, Error> tokenize_boundaries(std::string_view text) const;

    [[nodiscard]] const Model& model() const noexcept { return model_; }

private:
    Model model_;
};

}  // namespace segmentlib::kytea
