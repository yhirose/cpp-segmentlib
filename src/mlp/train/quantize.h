#pragma once

#include <cstdint>
#include <vector>

#include "mlp/train/example.h"
#include "mlp/train/net.h"

// Post-training quantization (design.ja.md 5.5/5.9, mlp_impl_design.ja.md
// II.6): weights go to int16 with max-abs scales (no clipping — outlier
// weights carry the strongest evidence), the accumulator scale S_acc is
// calibrated from the first-layer activation distribution, and the result is
// validated by comparing fp32 decisions against an int16 reference forward
// pass that reproduces the inference integer path (I.1/I.2) exactly.

namespace segmentlib::mlp::train {

// Everything the exporter writes (II.7): int16 tensors in the 5.7 file
// layout, the five scales, and the biases kept as raw doubles (the loader
// requantizes them to the accumulator scale, 5.7 fields 14/16).
struct QuantizedModel {
    NetConfig config;
    std::vector<std::int16_t> embedding;  // V × d
    std::vector<std::int16_t> w1;         // H × 2w·d
    std::vector<std::int16_t> wdict;      // H × Fd
    std::vector<std::int16_t> w2;         // H
    double emb_scale = 0.0;               // S_e
    double w1_scale = 0.0;                // S_w1
    double wdict_scale = 0.0;             // S_wd
    double w2_scale = 0.0;                // S_w2
    double acc_scale = 0.0;               // S_acc (calibrated; 5.7 field 8b)
    std::vector<double> b1;
    double b2 = 0.0;
};

// Quantizes `params`. `calibration` supplies the first-layer activation
// distribution for S_acc = pct99.99(|a|) / 2^22 (I.1); the dev set (or a
// slice of it) is the intended input.
[[nodiscard]] QuantizedModel quantize(const Parameters& params,
                                      const ExampleSet& calibration,
                                      ComputeBackend& backend,
                                      std::uint32_t batch_size = 512);

// The decision-flip validation of II.6, run over every example in `set`.
struct FlipCheck {
    std::size_t examples = 0;
    std::size_t flips = 0;         // sign(y_fp32) != int16 decision
    double max_abs_y_flipped = 0;  // largest fp32 |y| among flips (should be
                                   // near zero: flips live at the margin)
    std::int64_t max_abs_acc = 0;  // observed accumulator peak; the S_acc
                                   // headroom argument expects ≲ 2^26 (I.2)
};
[[nodiscard]] FlipCheck check_sign_flips(const Parameters& params,
                                         const QuantizedModel& quantized,
                                         const ExampleSet& set,
                                         ComputeBackend& backend,
                                         std::uint32_t batch_size = 512);

// The int16 decision for one example, via the integer reference path
// (fallback synthesis I.1-(2) for every EGC — the precompute table of 5.6 is
// bit-exact with it by construction, so this is the semantics step 6 must
// reproduce). Exposed for tests; check_sign_flips uses it internally.
// `max_abs_acc`, when given, accumulates the peak |acc| observed.
[[nodiscard]] bool int16_decision(const QuantizedModel& quantized,
                                  const ExampleSet& set, std::size_t example,
                                  std::int64_t* max_abs_acc = nullptr);

}  // namespace segmentlib::mlp::train
