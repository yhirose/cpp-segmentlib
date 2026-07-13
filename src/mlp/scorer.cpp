#include "segmentlib/mlp/scorer.h"

#include <algorithm>
#include <limits>

#include "segmentlib/mlp/kernels.h"

namespace segmentlib::mlp {

namespace {

std::int32_t saturate(std::int64_t sum) noexcept {
    return static_cast<std::int32_t>(
        std::clamp<std::int64_t>(sum, std::numeric_limits<std::int32_t>::min(),
                                 std::numeric_limits<std::int32_t>::max()));
}

// Rows of the EGC covered by window slot j of boundary i (EGC i - w + 1 + j);
// slots past either text end resolve to the PAD pseudo-EGC, itself a normal
// single-row table entry (I.2). Shared by the Int32 and Int16 scoring loops so
// this correctness-critical signed slot-index arithmetic lives in one place.
std::span<const std::uint32_t> slot_rows(const EncodedEgc& enc, std::size_t m,
                                         std::size_t i, std::size_t j,
                                         std::uint32_t w) {
    static constexpr std::uint32_t pad[] = {kPadRow};
    const auto egc = static_cast<std::int64_t>(i) - static_cast<std::int64_t>(w) +
                     1 + static_cast<std::int64_t>(j);
    return (egc >= 0 && egc < static_cast<std::int64_t>(m))
               ? enc.egc_rows(static_cast<std::size_t>(egc))
               : std::span<const std::uint32_t>(pad);
}

// The per-boundary loop of I.2, generic over the accumulator representation.
// `Acc` is int32 (exact path) or int16 (requantized path); the caller wires
// the mode-specific pieces (bias init, table add, dict-column add, relu, dot,
// output offset) so the window/boundary structure lives in one place.
template <class Acc, class InitAcc, class AddSlot, class AddDict, class Relu,
          class Dot>
void score_loop(const Model& model, const EncodedEgc& enc, Workspace& ws,
                std::vector<std::int32_t>& out, std::vector<Acc>& acc,
                InitAcc init_acc, AddSlot add_slot, AddDict add_dict, Relu relu,
                Dot dot) {
    const Config& c = model.config();
    const std::size_t m = enc.egc_count();
    const std::size_t boundaries = m - 1;
    const std::size_t h_dim = c.hidden;
    const std::size_t slots = 2u * c.char_window;

    acc.resize(h_dim);
    for (std::size_t i = 0; i < boundaries; ++i) {
        // acc = b1, then the 2w window contributions (I.2).
        init_acc(acc.data());
        for (std::size_t j = 0; j < slots; ++j) {
            add_slot(slot_rows(enc, m, i, j, c.char_window), j, acc.data());
        }

        // Active dictionary features: one column addition each (5.4).
        for (std::uint32_t o = ws.dict.offsets[i]; o < ws.dict.offsets[i + 1];
             ++o) {
            add_dict(ws.dict.indices[o], acc.data());
        }

        // ReLU, output dot product in int64, plus the b2 offset (I.1-(5)).
        relu(acc.data());
        out.push_back(saturate(dot(acc.data())));
    }
}

// The Int16 production path (design.ja.md 5.6), fused: each boundary's whole
// hidden-unit pass — b1 init, the 2w window-slot saturating adds, active dict
// columns, relu, and the output dot — runs in one register-resident sweep
// (kernels::fused_score_i16) instead of ~2w+3 separate full-width passes over
// the accumulator. Bit-identical to the unfused sequence: window slots are
// gathered in order (fallback EGCs synthesized into per-slot scratch), dict
// columns appended after, and fused_score_i16 saturating-adds them in that
// same order. Only the redundant accumulator round-trips through memory are
// removed (the profiled bottleneck), not the arithmetic.
void score_i16_fused(const Model& model, const EncodedEgc& enc, Workspace& ws,
                     std::vector<std::int32_t>& out) {
    const Config& c = model.config();
    const std::size_t m = enc.egc_count();
    const std::size_t boundaries = m - 1;
    const std::size_t h_dim = c.hidden;
    const std::size_t slots = 2u * c.char_window;
    const PrecomputeTable& precompute = model.precompute();
    const std::int16_t* b1 = model.b1_q16().data();
    const std::int16_t* w2 = model.w2().data();
    const std::int64_t b2 = model.b2_q16();

    // One scratch slot (H) per window position, not a single shared H buffer:
    // block_i16 may return a pointer into this buffer, and all `slots` block
    // pointers stay live in ws.blocks until fused_score_i16 reads them at the
    // end of the boundary — so two fallback EGCs sharing one buffer would alias
    // and corrupt each other.
    ws.fallback.resize(slots * h_dim);
    for (std::size_t i = 0; i < boundaries; ++i) {
        ws.blocks.clear();
        for (std::size_t j = 0; j < slots; ++j) {
            ws.blocks.push_back(precompute.block_i16(
                slot_rows(enc, m, i, j, c.char_window), j,
                ws.fallback.data() + j * h_dim));
        }
        for (std::uint32_t o = ws.dict.offsets[i]; o < ws.dict.offsets[i + 1];
             ++o) {
            ws.blocks.push_back(model.dict_col16(ws.dict.indices[o]).data());
        }
        const std::int64_t s = kernels::fused_score_i16(
            b1, ws.blocks.data(), ws.blocks.size(), w2, h_dim);
        out.push_back(saturate(s + b2));
    }
}

}  // namespace

void score_boundaries_into(const Model& model, const EncodedEgc& enc,
                           Workspace& ws, std::vector<std::int32_t>& out) {
    const std::size_t m = enc.egc_count();
    out.clear();
    if (m < 2) {
        return;
    }
    out.reserve(m - 1);
    model.dict().features_into(enc, ws.dict);

    const std::size_t h_dim = model.config().hidden;
    const PrecomputeTable& precompute = model.precompute();
    const std::span<const std::int16_t> w2 = model.w2();

    if (model.table_precision() == TablePrecision::Int32) {
        const std::span<const std::int32_t> b1_q = model.b1_q();
        score_loop(
            model, enc, ws, out, ws.acc,
            [&](std::int32_t* acc) { std::copy(b1_q.begin(), b1_q.end(), acc); },
            [&](std::span<const std::uint32_t> rows, std::size_t j,
                std::int32_t* acc) { precompute.add_into(rows, j, acc); },
            [&](std::uint32_t k, std::int32_t* acc) {
                kernels::add_i32(model.dict_col(k).data(), acc, h_dim);
            },
            [&](std::int32_t* acc) { kernels::relu_i32(acc, h_dim); },
            [&](const std::int32_t* acc) {
                return kernels::dot_i32(w2.data(), acc, h_dim) + model.b2_q();
            });
    } else {
        score_i16_fused(model, enc, ws, out);
    }
}

std::vector<std::int32_t> score_boundaries(const Model& model,
                                           const EncodedEgc& enc) {
    Workspace ws;
    std::vector<std::int32_t> out;
    score_boundaries_into(model, enc, ws, out);
    return out;
}

}  // namespace segmentlib::mlp
