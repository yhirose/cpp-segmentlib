#include "segmentlib/kytea/kytea_backend.h"

#include <utility>
#include <vector>

#include "segmentlib/kytea/scorer.h"

namespace segmentlib::kytea {

namespace {

// True if a boundary with the given raw score should be cut. KyTea places a
// boundary where score * multiplier > 0 (confidence threshold 0).
bool is_cut(std::int32_t score, double multiplier) noexcept {
    return static_cast<double>(score) * multiplier > 0.0;
}

}  // namespace

// Per-thread scratch reused across calls so the hot path allocates only the
// returned result, not the intermediate encoding/score buffers. thread_local
// keeps tokenize re-entrant (the Model is immutable and shared).
namespace {
thread_local EncodedText tl_enc;
thread_local std::vector<std::int32_t> tl_scores;
}  // namespace

std::expected<Segments, Error> KyteaBackend::tokenize(std::string_view text) const {
    if (auto r = model_.chars().encode_into(text, tl_enc); !r) {
        return std::unexpected(r.error());
    }
    const std::size_t n = tl_enc.length();
    Segments segments;
    if (n == 0) {
        return segments;
    }
    score_boundaries_into(model_, tl_enc, tl_scores);
    const double mult = model_.multiplier();
    std::size_t start = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const bool cut = (i + 1 < n) ? is_cut(tl_scores[i], mult) : true;
        if (cut) {
            segments.emplace_back(tl_enc.offsets[start], tl_enc.offsets[i + 1]);
            start = i + 1;
        }
    }
    return segments;
}

}  // namespace segmentlib::kytea
