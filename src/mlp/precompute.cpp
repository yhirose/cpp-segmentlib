#include "segmentlib/mlp/precompute.h"

#include <cassert>
#include <cmath>

#include "segmentlib/mlp/kernels.h"

namespace segmentlib::mlp {

// The shared integer kernel of I.1-(1)/(2): given the constituent rows of
// one EGC, compute every hidden unit's contribution at window position j:
//   sum_c[k] = Σ_rows emb_q[row, k]                        (int32)
//   raw[h]   = Σ_k w1_q[h, j*d+k] · sum_c[k]               (int64)
//   emit(h, llround(raw[h] · r / n))                        (int32)
// Table construction, the Int32 fallback and the Int16 fallback all go
// through this one function, which is what makes table and fallback bit-exact
// with each other (and, in Int32 mode, with the trainer's reference forward).
template <class Emit>
void PrecomputeTable::synthesize(std::span<const std::uint32_t> rows,
                                 std::size_t j, Emit emit) const {
    std::int32_t sum_c[256];  // the loader rejects embed_dim > 256
    const std::size_t d = d_;
    assert(d <= std::size(sum_c));
    for (std::size_t k = 0; k < d; ++k) {
        std::int32_t s = 0;
        for (const std::uint32_t row : rows) {
            s += emb_q_[static_cast<std::size_t>(row) * d + k];
        }
        sum_c[k] = s;
    }
    const std::size_t in_dim = 2u * window_ * d;
    const auto n = static_cast<double>(rows.size());
    for (std::size_t h = 0; h < h_; ++h) {
        const std::int16_t* w1_row = w1_q_.data() + h * in_dim + j * d;
        std::int64_t raw = 0;
        for (std::size_t k = 0; k < d; ++k) {
            raw += static_cast<std::int64_t>(w1_row[k]) * sum_c[k];
        }
        emit(h, static_cast<std::int32_t>(
                    std::llround(static_cast<double>(raw) * r_ / n)));
    }
}

PrecomputeTable::PrecomputeTable(std::span<const std::int16_t> emb_q,
                                 std::span<const std::int16_t> w1_q,
                                 std::uint32_t vocab_size,
                                 std::uint16_t embed_dim, std::uint16_t hidden,
                                 std::uint8_t window, double r,
                                 TablePrecision precision)
    : emb_q_(emb_q.begin(), emb_q.end()),
      w1_q_(w1_q.begin(), w1_q.end()),
      d_(embed_dim),
      h_(hidden),
      window_(window),
      r_(r),
      precision_(precision) {
    const std::size_t slots = 2u * window_;
    const std::size_t entries = static_cast<std::size_t>(vocab_size) * slots * h_;
    if (precision_ == TablePrecision::Int32) {
        table32_.resize(entries);
    } else {
        table16_.resize(entries);
    }

    std::uint32_t single_row[1];
    for (std::uint32_t row = 0; row < vocab_size; ++row) {
        single_row[0] = row;  // a single-codepoint EGC (n = 1)
        for (std::size_t j = 0; j < slots; ++j) {
            const std::size_t base =
                (static_cast<std::size_t>(row) * slots + j) * h_;
            if (precision_ == TablePrecision::Int32) {
                synthesize(single_row, j, [&](std::size_t h, std::int32_t v) {
                    table32_[base + h] = v;
                });
            } else {
                synthesize(single_row, j, [&](std::size_t h, std::int32_t v) {
                    table16_[base + h] = requant_i16(v);
                });
            }
        }
    }
}

void PrecomputeTable::add_into(std::span<const std::uint32_t> rows,
                               std::size_t j, std::int32_t* acc) const {
    assert(precision_ == TablePrecision::Int32);
    if (rows.size() == 1) {
        const std::int32_t* block =
            table32_.data() +
            (static_cast<std::size_t>(rows[0]) * (2u * window_) + j) * h_;
        kernels::add_i32(block, acc, h_);
        return;
    }
    // Fallback synthesis (I.1-(2)) for multi-codepoint EGCs.
    synthesize(rows, j, [&](std::size_t h, std::int32_t v) { acc[h] += v; });
}

const std::int16_t* PrecomputeTable::block_i16(
    std::span<const std::uint32_t> rows, std::size_t j,
    std::int16_t* scratch) const {
    assert(precision_ == TablePrecision::Int16);
    if (rows.size() == 1) {
        return table16_.data() +
               (static_cast<std::size_t>(rows[0]) * (2u * window_) + j) * h_;
    }
    // Fallback (I.1-(2)) for multi-codepoint EGCs: the int32 contribution
    // requantized exactly as the table entries were, written to scratch so the
    // fused scorer can saturating-add it in window order — bit-identical to the
    // table path for the same EGC.
    synthesize(rows, j, [&](std::size_t h, std::int32_t v) {
        scratch[h] = requant_i16(v);
    });
    return scratch;
}

}  // namespace segmentlib::mlp
