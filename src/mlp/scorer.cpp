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
    static constexpr std::uint32_t pad[] = {kPadRow};

    acc.resize(h_dim);
    for (std::size_t i = 0; i < boundaries; ++i) {
        // acc = b1, then the 2w window contributions (I.2). Window slot j
        // covers EGC i - w + 1 + j; slots past the text ends are the PAD
        // pseudo-EGC, which is a normal single-row table entry.
        init_acc(acc.data());
        for (std::size_t j = 0; j < slots; ++j) {
            const auto egc = static_cast<std::int64_t>(i) - c.char_window + 1 +
                             static_cast<std::int64_t>(j);
            const std::span<const std::uint32_t> rows =
                (egc >= 0 && egc < static_cast<std::int64_t>(m))
                    ? enc.egc_rows(static_cast<std::size_t>(egc))
                    : std::span<const std::uint32_t>(pad);
            add_slot(rows, j, acc.data());
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
        const std::span<const std::int16_t> b1_q16 = model.b1_q16();
        score_loop(
            model, enc, ws, out, ws.acc16,
            [&](std::int16_t* acc) {
                std::copy(b1_q16.begin(), b1_q16.end(), acc);
            },
            [&](std::span<const std::uint32_t> rows, std::size_t j,
                std::int16_t* acc) { precompute.add_into_i16(rows, j, acc); },
            [&](std::uint32_t k, std::int16_t* acc) {
                kernels::add_sat_i16(model.dict_col16(k).data(), acc, h_dim);
            },
            [&](std::int16_t* acc) { kernels::relu_i16(acc, h_dim); },
            [&](const std::int16_t* acc) {
                return kernels::dot_i16(w2.data(), acc, h_dim) + model.b2_q16();
            });
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
