#pragma once

#include <cstdint>
#include <vector>

#include "segmentlib/kytea/char_table.h"
#include "segmentlib/kytea/model.h"

namespace segmentlib::kytea {

// Computes KyTea's per-boundary word-segmentation scores for an encoded
// sentence (the algorithm of `Kytea::calculateWS`). The result has one entry
// per character boundary (length N-1 for N characters; empty if N < 2).
//
// A boundary i sits between character i and i+1. Multiplying by the model's
// multiplier yields KyTea's confidence; a boundary is placed where that
// confidence is > 0. Since the multiplier is positive, the raw score's sign is
// what matters. Scores accumulate three feature families:
//   1. a constant bias,
//   2. character and character-type n-gram weights (Aho-Corasick matches), and
//   3. dictionary (L/I/R) weights from matched dictionary words.
[[nodiscard]] std::vector<std::int32_t> score_boundaries(const Model& model,
                                                         const EncodedText& enc);

// Same as score_boundaries(), but fills a caller-owned buffer (reused across
// calls to avoid per-call allocation on the hot path). `out` is resized to the
// number of boundaries (N-1).
void score_boundaries_into(const Model& model, const EncodedText& enc,
                           std::vector<std::int32_t>& out);

}  // namespace segmentlib::kytea
