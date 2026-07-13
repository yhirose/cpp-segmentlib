#pragma once

#include <cstdint>
#include <vector>

#include "segmentlib/mlp/dictionary.h"
#include "segmentlib/mlp/model.h"
#include "segmentlib/mlp/vocab.h"

namespace segmentlib::mlp {

// Per-call buffers for the scorer (mlp_impl_design.ja.md I.5): the
// accumulator (whichever precision the model was loaded with) and the
// dictionary-feature scratch. score_boundaries_into allocates nothing once
// these have grown to the sentence's size, matching the kytea::scorer _into
// convention.
struct Workspace {
    std::vector<std::int32_t> acc;    // H (TablePrecision::Int32)
    std::vector<std::int16_t> acc16;  // H (TablePrecision::Int16)
    DictFeatures dict;
    // Int16 fused path (kernels::fused_score_i16): the per-boundary block
    // pointers (2w window slots then active dict columns) and 2w×H scratch for
    // the window slots whose EGC needs fallback synthesis.
    std::vector<const std::int16_t*> blocks;
    std::vector<std::int16_t> fallback;
};

// Scores every EGC boundary of `enc` (M-1 scores for M clusters): the int16
// integer forward pass of design.ja.md 5.6 / mlp_impl_design.ja.md I.2,
// through the precision path the model was loaded with (model.h). A boundary
// is predicted where the score is positive. Scores are the int64 output sum
// saturated to int32 (the sign — the only thing inference uses — is
// preserved; kytea::scorer's int32 score shape is kept for uniformity).
// Int16-mode scores are in a 2^kAccShift coarser unit than Int32-mode ones;
// only their signs are comparable across the two modes.
void score_boundaries_into(const Model& model, const EncodedEgc& enc,
                           Workspace& ws, std::vector<std::int32_t>& out);

[[nodiscard]] std::vector<std::int32_t>
score_boundaries(const Model& model, const EncodedEgc& enc);

}  // namespace segmentlib::mlp
