#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

// The three SIMD kernels of the MLP scoring hot path (mlp_impl_design.ja.md
// I.3), in int32 (first implementation) and int16 (requantized table, 5.6の
// 後段最適化) variants:
//
//   add       acc[h] += src[h]              (int32; int16 is *saturating*)
//   relu      acc[h]  = max(acc[h], 0)
//   dot       Σ_h w2[h] · acc[h] → int64    (the output layer product)
//
// plus one widening add shared with the KyTea backend's tag scorer (its
// numWeights-wide int16 weight blocks accumulate into int32 scores — the
// hottest loop of tag prediction, design.ja.md 9.2.1; the kernel is a plain
// integer op, not MLP-specific, so it lives with the others):
//
//   add_widen acc[h] += src[h]              (int16 src → int32 acc, exact)
//
// `kernels::scalar` is the always-compiled reference implementation — the
// oracle the SIMD paths are tested against (I.3) and the fallback for
// platforms with neither NEON nor AVX2. The unqualified functions dispatch at
// compile time: AArch64 always has NEON; x86 uses AVX2 when the translation
// unit is built with it (-mavx2 / /arch:AVX2), scalar otherwise. The int32
// kernels are bit-exact with scalar on every path (integer adds/products are
// associative); the int16 kernels share exact saturating semantics.

#if defined(__aarch64__) || defined(__ARM_NEON)
#define SEGMENTLIB_KERNELS_NEON 1
#include <arm_neon.h>
#elif defined(__AVX2__)
#define SEGMENTLIB_KERNELS_AVX2 1
#include <immintrin.h>
#endif

namespace segmentlib::mlp::kernels {

namespace scalar {

inline void add_i32(const std::int32_t* src, std::int32_t* acc, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        acc[i] += src[i];
    }
}

inline void relu_i32(std::int32_t* acc, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        acc[i] = std::max(acc[i], 0);
    }
}

[[nodiscard]] inline std::int64_t dot_i32(const std::int16_t* w2,
                                          const std::int32_t* acc,
                                          std::size_t n) {
    std::int64_t sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += static_cast<std::int64_t>(w2[i]) * acc[i];
    }
    return sum;
}

// Saturating int16 add (the requantized-table accumulator clips instead of
// wrapping, 5.6; saturation is rare by S_acc calibration and is measured by
// the decision-flip check).
inline void add_sat_i16(const std::int16_t* src, std::int16_t* acc,
                        std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        const std::int32_t s = static_cast<std::int32_t>(acc[i]) + src[i];
        acc[i] = static_cast<std::int16_t>(std::clamp(s, -32768, 32767));
    }
}

inline void relu_i16(std::int16_t* acc, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        acc[i] = std::max<std::int16_t>(acc[i], 0);
    }
}

[[nodiscard]] inline std::int64_t dot_i16(const std::int16_t* w2,
                                          const std::int16_t* acc,
                                          std::size_t n) {
    std::int64_t sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += static_cast<std::int32_t>(w2[i]) * acc[i];
    }
    return sum;
}

// Widening add (int16 source into an int32 accumulator, exact — no
// saturation). The KyTea tag scorer's per-match weight accumulation.
inline void add_widen_i16_i32(const std::int16_t* src, std::int32_t* acc,
                              std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        acc[i] += src[i];
    }
}

}  // namespace scalar

#if defined(SEGMENTLIB_KERNELS_NEON)

inline void add_i32(const std::int32_t* src, std::int32_t* acc, std::size_t n) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        vst1q_s32(acc + i, vaddq_s32(vld1q_s32(acc + i), vld1q_s32(src + i)));
    }
    scalar::add_i32(src + i, acc + i, n - i);
}

inline void relu_i32(std::int32_t* acc, std::size_t n) {
    const int32x4_t zero = vdupq_n_s32(0);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        vst1q_s32(acc + i, vmaxq_s32(vld1q_s32(acc + i), zero));
    }
    scalar::relu_i32(acc + i, n - i);
}

[[nodiscard]] inline std::int64_t dot_i32(const std::int16_t* w2,
                                          const std::int32_t* acc,
                                          std::size_t n) {
    int64x2_t sum = vdupq_n_s64(0);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const int32x4_t w = vmovl_s16(vld1_s16(w2 + i));  // widen int16→int32
        const int32x4_t a = vld1q_s32(acc + i);
        sum = vaddq_s64(sum, vmull_s32(vget_low_s32(w), vget_low_s32(a)));
        sum = vaddq_s64(sum, vmull_high_s32(w, a));
    }
    return vaddvq_s64(sum) + scalar::dot_i32(w2 + i, acc + i, n - i);
}

inline void add_sat_i16(const std::int16_t* src, std::int16_t* acc,
                        std::size_t n) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        vst1q_s16(acc + i, vqaddq_s16(vld1q_s16(acc + i), vld1q_s16(src + i)));
    }
    scalar::add_sat_i16(src + i, acc + i, n - i);
}

inline void relu_i16(std::int16_t* acc, std::size_t n) {
    const int16x8_t zero = vdupq_n_s16(0);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        vst1q_s16(acc + i, vmaxq_s16(vld1q_s16(acc + i), zero));
    }
    scalar::relu_i16(acc + i, n - i);
}

[[nodiscard]] inline std::int64_t dot_i16(const std::int16_t* w2,
                                          const std::int16_t* acc,
                                          std::size_t n) {
    int64x2_t sum = vdupq_n_s64(0);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const int16x8_t w = vld1q_s16(w2 + i);
        const int16x8_t a = vld1q_s16(acc + i);
        // int16×int16 → int32×4 per half, then pairwise-accumulate into int64.
        const int32x4_t lo = vmull_s16(vget_low_s16(w), vget_low_s16(a));
        const int32x4_t hi = vmull_high_s16(w, a);
        sum = vpadalq_s32(sum, lo);
        sum = vpadalq_s32(sum, hi);
    }
    return vaddvq_s64(sum) + scalar::dot_i16(w2 + i, acc + i, n - i);
}

inline void add_widen_i16_i32(const std::int16_t* src, std::int32_t* acc,
                              std::size_t n) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const int16x8_t s = vld1q_s16(src + i);
        // vaddw widens each int16 half to int32 and adds — exact.
        vst1q_s32(acc + i, vaddw_s16(vld1q_s32(acc + i), vget_low_s16(s)));
        vst1q_s32(acc + i + 4, vaddw_high_s16(vld1q_s32(acc + i + 4), s));
    }
    scalar::add_widen_i16_i32(src + i, acc + i, n - i);
}

#elif defined(SEGMENTLIB_KERNELS_AVX2)

inline void add_i32(const std::int32_t* src, std::int32_t* acc, std::size_t n) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256i a =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
        const __m256i s =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i),
                            _mm256_add_epi32(a, s));
    }
    scalar::add_i32(src + i, acc + i, n - i);
}

inline void relu_i32(std::int32_t* acc, std::size_t n) {
    const __m256i zero = _mm256_setzero_si256();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256i a =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i),
                            _mm256_max_epi32(a, zero));
    }
    scalar::relu_i32(acc + i, n - i);
}

[[nodiscard]] inline std::int64_t dot_i32(const std::int16_t* w2,
                                          const std::int32_t* acc,
                                          std::size_t n) {
    __m256i sum = _mm256_setzero_si256();  // 4 × int64 lanes
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        // Widen both operands to sign-extended 64-bit lanes; mul_epi32
        // multiplies the (sign-extended) low 32 bits of each lane — exact.
        const __m256i w32 = _mm256_cvtepi16_epi32(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(w2 + i)));
        const __m256i a32 =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
        const __m256i w_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(w32));
        const __m256i w_hi =
            _mm256_cvtepi32_epi64(_mm256_extracti128_si256(w32, 1));
        const __m256i a_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(a32));
        const __m256i a_hi =
            _mm256_cvtepi32_epi64(_mm256_extracti128_si256(a32, 1));
        sum = _mm256_add_epi64(sum, _mm256_mul_epi32(w_lo, a_lo));
        sum = _mm256_add_epi64(sum, _mm256_mul_epi32(w_hi, a_hi));
    }
    alignas(32) std::int64_t lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), sum);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3] +
           scalar::dot_i32(w2 + i, acc + i, n - i);
}

inline void add_sat_i16(const std::int16_t* src, std::int16_t* acc,
                        std::size_t n) {
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256i a =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
        const __m256i s =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i),
                            _mm256_adds_epi16(a, s));
    }
    scalar::add_sat_i16(src + i, acc + i, n - i);
}

inline void relu_i16(std::int16_t* acc, std::size_t n) {
    const __m256i zero = _mm256_setzero_si256();
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256i a =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i),
                            _mm256_max_epi16(a, zero));
    }
    scalar::relu_i16(acc + i, n - i);
}

[[nodiscard]] inline std::int64_t dot_i16(const std::int16_t* w2,
                                          const std::int16_t* acc,
                                          std::size_t n) {
    __m256i sum = _mm256_setzero_si256();  // 4 × int64 lanes
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256i w =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w2 + i));
        const __m256i a =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
        // madd: int16×int16 pair sums → 8 × int32; each ≤ 2·32767² < 2^31,
        // so the pair sum itself cannot overflow. Widen to int64 to accumulate.
        const __m256i pairs = _mm256_madd_epi16(w, a);
        sum = _mm256_add_epi64(
            sum, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(pairs)));
        sum = _mm256_add_epi64(
            sum, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(pairs, 1)));
    }
    alignas(32) std::int64_t lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), sum);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3] +
           scalar::dot_i16(w2 + i, acc + i, n - i);
}

inline void add_widen_i16_i32(const std::int16_t* src, std::int32_t* acc,
                              std::size_t n) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        // cvtepi16_epi32 sign-extends 8 int16 lanes to int32 — exact.
        const __m256i s = _mm256_cvtepi16_epi32(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i)));
        const __m256i a =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i),
                            _mm256_add_epi32(a, s));
    }
    scalar::add_widen_i16_i32(src + i, acc + i, n - i);
}

#else  // scalar dispatch

inline void add_i32(const std::int32_t* src, std::int32_t* acc, std::size_t n) {
    scalar::add_i32(src, acc, n);
}
inline void relu_i32(std::int32_t* acc, std::size_t n) {
    scalar::relu_i32(acc, n);
}
[[nodiscard]] inline std::int64_t dot_i32(const std::int16_t* w2,
                                          const std::int32_t* acc,
                                          std::size_t n) {
    return scalar::dot_i32(w2, acc, n);
}
inline void add_sat_i16(const std::int16_t* src, std::int16_t* acc,
                        std::size_t n) {
    scalar::add_sat_i16(src, acc, n);
}
inline void relu_i16(std::int16_t* acc, std::size_t n) {
    scalar::relu_i16(acc, n);
}
[[nodiscard]] inline std::int64_t dot_i16(const std::int16_t* w2,
                                          const std::int16_t* acc,
                                          std::size_t n) {
    return scalar::dot_i16(w2, acc, n);
}
inline void add_widen_i16_i32(const std::int16_t* src, std::int32_t* acc,
                              std::size_t n) {
    scalar::add_widen_i16_i32(src, acc, n);
}

#endif

}  // namespace segmentlib::mlp::kernels
