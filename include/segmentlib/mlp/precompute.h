#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace segmentlib::mlp {

// Which numeric representation the first-layer table and accumulator use
// (design.ja.md 4.6):
//  - Int32: the first implementation — no clipping anywhere, bit-exact with
//    the trainer's int16 reference forward (train/quantize.cpp). The
//    verification path.
//  - Int16: the requantized table (5.6の後段最適化, NNUE's form): every
//    accumulator-scale int32 value is shifted right by kAccShift with
//    rounding and saturated to int16, and accumulation uses saturating int16
//    adds — half the memory and roughly double the SIMD throughput. Rare
//    saturations / decision flips are possible near the margin and are
//    measured against the Int32 path (I.3: 飽和発生率と判定反転をチェック).
enum class TablePrecision : std::uint8_t { Int32, Int16 };

// The int16 accumulator unit is S_acc·2^kAccShift. S_acc calibration maps
// pct99.99(|a|) to Amax = 2^22 (I.1), so the shift lands that point on
// 2^13 = 8192 — a 4× headroom inside int16 for the summation beyond the
// percentile. NOTE: this presumes a properly calibrated acc_scale; a model
// whose acc_scale is not derived from activation statistics will clip.
inline constexpr int kAccShift = 9;

// The single int32 → int16 requantization used for *every* Int16-mode
// quantity (table blocks, fallback contributions, dictionary columns, b1):
// round-to-nearest arithmetic shift, then saturate.
[[nodiscard]] constexpr std::int16_t requant_i16(std::int32_t v) noexcept {
    const std::int32_t shifted = (v + (1 << (kAccShift - 1))) >> kAccShift;
    return static_cast<std::int16_t>(std::clamp(shifted, -32768, 32767));
}

// The first-layer precompute table of design.ja.md 4.6 (NNUE style):
// table[egc][j] = W1_j · v(egc) in the accumulator integer scale S_acc
// built at load time from the quantized
// embedding and W1.
//
// The first implementation's frequent-EGC set (I.4): every single-codepoint
// EGC — i.e. every embedding row, including the PAD (row 0) and UNK (row 1)
// pseudo-entries — has a table block; multi-codepoint EGCs take the fallback
// synthesis path (I.1-(2)), which uses the identical integer computation, so
// the table is a pure cache: both paths are bit-exact by construction (in
// Int16 mode both go through the same requant_i16, preserving the property).
//
// Memory: V × 2w × H × 4 bytes in Int32 mode (~100MB at V=10k/w=5/H=256),
// half that in Int16 mode.
class PrecomputeTable {
public:
    PrecomputeTable() = default;

    // Builds the table. `emb_q` is V×d, `w1_q` is H×(2w·d), both in the 5.7
    // layout; `r` is the rescale factor R = S_e·S_w1 / S_acc. The table
    // keeps copies of `emb_q`/`w1_q` for the fallback path, so it does not
    // borrow from the caller.
    PrecomputeTable(std::span<const std::int16_t> emb_q,
                    std::span<const std::int16_t> w1_q, std::uint32_t vocab_size,
                    std::uint16_t embed_dim, std::uint16_t hidden,
                    std::uint8_t window, double r,
                    TablePrecision precision = TablePrecision::Int32);

    [[nodiscard]] TablePrecision precision() const noexcept { return precision_; }

    // acc[h] += table-or-fallback contribution of the EGC whose constituent
    // rows are `rows`, at window position j (I.2). `acc` must have H entries.
    // Single-row EGCs (and PAD, rows = {0}) hit the table; multi-row EGCs
    // synthesize. Int32 mode only — the Int16 path gathers via block_i16.
    void add_into(std::span<const std::uint32_t> rows, std::size_t j,
                  std::int32_t* acc) const;

    // Returns the H-length Int16 contribution block for the EGC `rows` at window
    // slot j, for the fused scorer (kernels::fused_score_i16). Single-row EGCs
    // (and PAD) return a pointer straight into the table; multi-row EGCs
    // synthesize into `scratch` (which must hold >= H int16) and return it. The
    // Int16 counterpart of add_into.
    [[nodiscard]] const std::int16_t* block_i16(
        std::span<const std::uint32_t> rows, std::size_t j,
        std::int16_t* scratch) const;

private:
    // Per-h fallback contribution (the shared integer kernel of I.1-(2));
    // callers requantize per mode.
    template <class Emit>
    void synthesize(std::span<const std::uint32_t> rows, std::size_t j,
                    Emit emit) const;

    std::vector<std::int32_t> table32_;  // [row][j][h], Int32 mode
    std::vector<std::int16_t> table16_;  // [row][j][h], Int16 mode
    std::vector<std::int16_t> emb_q_;    // V × d (fallback)
    std::vector<std::int16_t> w1_q_;     // H × 2w·d (fallback)
    std::uint16_t d_ = 0;
    std::uint16_t h_ = 0;
    std::uint8_t window_ = 0;
    double r_ = 0.0;
    TablePrecision precision_ = TablePrecision::Int32;
};

}  // namespace segmentlib::mlp
