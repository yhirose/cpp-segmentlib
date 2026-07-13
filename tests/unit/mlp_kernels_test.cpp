#include <doctest/doctest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "segmentlib/mlp/kernels.h"

// The dispatched kernels (NEON on this machine; AVX2 on x86 builds with it;
// otherwise scalar pass-through) must be bit-exact with the scalar reference
// (mlp_impl_design.ja.md I.3: scalar is the oracle). Sizes include non-lane-
// multiple tails, and the int16 inputs include saturation-triggering extremes.

using namespace segmentlib::mlp;

namespace {

constexpr std::size_t kSizes[] = {1, 3, 7, 8, 15, 16, 63, 256, 300};

std::vector<std::int32_t> random_i32(std::mt19937& rng, std::size_t n,
                                     std::int32_t lo, std::int32_t hi) {
    std::uniform_int_distribution<std::int32_t> dist(lo, hi);
    std::vector<std::int32_t> v(n);
    for (auto& x : v) {
        x = dist(rng);
    }
    return v;
}

std::vector<std::int16_t> random_i16(std::mt19937& rng, std::size_t n) {
    std::uniform_int_distribution<std::int32_t> dist(-32768, 32767);
    std::vector<std::int16_t> v(n);
    for (auto& x : v) {
        x = static_cast<std::int16_t>(dist(rng));
    }
    return v;
}

}  // namespace

TEST_CASE("int32 kernels match the scalar reference") {
    std::mt19937 rng(7);
    for (const std::size_t n : kSizes) {
        CAPTURE(n);
        const auto src = random_i32(rng, n, -(1 << 26), 1 << 26);
        const auto base = random_i32(rng, n, -(1 << 26), 1 << 26);
        const auto w2 = random_i16(rng, n);

        auto expected = base;
        auto actual = base;
        kernels::scalar::add_i32(src.data(), expected.data(), n);
        kernels::add_i32(src.data(), actual.data(), n);
        CHECK(actual == expected);

        kernels::scalar::relu_i32(expected.data(), n);
        kernels::relu_i32(actual.data(), n);
        CHECK(actual == expected);

        CHECK(kernels::dot_i32(w2.data(), actual.data(), n) ==
              kernels::scalar::dot_i32(w2.data(), expected.data(), n));
    }
}

TEST_CASE("int16 kernels match the scalar reference, saturation included") {
    std::mt19937 rng(11);
    for (const std::size_t n : kSizes) {
        CAPTURE(n);
        // Full-range int16 values: adds frequently saturate in both
        // directions, exercising the clip semantics.
        const auto src = random_i16(rng, n);
        const auto base = random_i16(rng, n);
        const auto w2 = random_i16(rng, n);

        auto expected = base;
        auto actual = base;
        kernels::scalar::add_sat_i16(src.data(), expected.data(), n);
        kernels::add_sat_i16(src.data(), actual.data(), n);
        CHECK(actual == expected);

        kernels::scalar::relu_i16(expected.data(), n);
        kernels::relu_i16(actual.data(), n);
        CHECK(actual == expected);

        CHECK(kernels::dot_i16(w2.data(), actual.data(), n) ==
              kernels::scalar::dot_i16(w2.data(), expected.data(), n));
    }
}

TEST_CASE("saturating add clips at the int16 limits") {
    const std::int16_t src[] = {32767, 32767, -32768, -32768, 100};
    std::int16_t acc[] = {1, 32767, -1, -32768, -50};
    kernels::add_sat_i16(src, acc, 5);
    CHECK(acc[0] == 32767);   // 32768 clipped
    CHECK(acc[1] == 32767);   // far past the limit
    CHECK(acc[2] == -32768);  // -32769 clipped
    CHECK(acc[3] == -32768);
    CHECK(acc[4] == 50);      // no clip
}

TEST_CASE("widening add matches the scalar reference") {
    std::mt19937 rng(13);
    // Tail-exercising sizes plus the widths the KyTea tag scorer actually
    // uses: nw=21 (jp model lev0 POS candidates) and nw=4 (per-word lev1).
    constexpr std::size_t kWidenSizes[] = {1, 3, 4, 7, 8, 15, 16, 21, 63, 256};
    for (const std::size_t n : kWidenSizes) {
        CAPTURE(n);
        const auto src = random_i16(rng, n);  // full int16 range
        const auto base = random_i32(rng, n, -(1 << 26), 1 << 26);

        auto expected = base;
        auto actual = base;
        kernels::scalar::add_widen_i16_i32(src.data(), expected.data(), n);
        kernels::add_widen_i16_i32(src.data(), actual.data(), n);
        CHECK(actual == expected);
    }
}
