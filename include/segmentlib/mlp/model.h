#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <vector>

#include "segmentlib/mlp/dictionary.h"
#include "segmentlib/mlp/precompute.h"
#include "segmentlib/mlp/vocab.h"
#include "segmentlib/types.h"

namespace segmentlib::bytes {
class BinaryReader;
}

namespace segmentlib::mlp {

// The header-line signature used by backend auto-detection (design.ja.md
// 5.7/2節); the full first line is "SegmentLibMLP <version>\n".
inline constexpr std::string_view kModelSignature = "SegmentLibMLP ";

struct Config {  // 5.7 fields 1-4b
    std::uint8_t char_window = 0;      // w
    std::uint16_t embed_dim = 0;       // d
    std::uint16_t hidden = 0;          // H
    std::uint8_t num_dicts = 0;
    std::uint16_t unicode_version = 0;
};

// An immutable loaded MLP model: the raw
// quantized parameters from the file, plus the load-time derived structures —
// precompute table, dictionary matcher, dictionary columns and biases already
// converted to accumulator integer scale. All the
// scorer touches is integers; no scale multiplication survives into the hot
// path.
//
// `precision` picks the table/accumulator representation (5.6): Int16 (the
// default) is the requantized fast path — half the table memory, saturating
// int16 accumulation, decisions can differ from the file's exact semantics
// only at the |y|≈0 margin; Int32 is the exact reference path, bit-identical
// to the trainer's int16 reference forward. The file format is identical;
// this is purely a load-time choice.
class Model {
public:
    [[nodiscard]] static std::expected<Model, Error>
    load(const std::filesystem::path& path,
         TablePrecision precision = TablePrecision::Int16);
    [[nodiscard]] static std::expected<Model, Error>
    load_from_bytes(std::span<const std::byte> data,
                    TablePrecision precision = TablePrecision::Int16);

    [[nodiscard]] TablePrecision table_precision() const noexcept {
        return parts_.precompute.precision();
    }

    [[nodiscard]] const Config& config() const noexcept { return parts_.config; }
    [[nodiscard]] const Vocab& vocab() const noexcept { return parts_.vocab; }
    [[nodiscard]] const PrecomputeTable& precompute() const noexcept {
        return parts_.precompute;
    }
    [[nodiscard]] const DictMatcher& dict() const noexcept { return parts_.dict; }

    // Raw quantized layers (5.7 fields 11/12/15).
    [[nodiscard]] std::span<const std::int16_t> embedding() const noexcept {
        return parts_.embedding;
    }
    [[nodiscard]] std::span<const std::int16_t> w1() const noexcept {
        return parts_.w1;
    }
    [[nodiscard]] std::span<const std::int16_t> w2() const noexcept {
        return parts_.w2;
    }

    // Biases in accumulator integer scale (converted at load, 5.7 fields
    // 14/16; b2_q is int64 — the output dot product runs in int64, I.1-(5)).
    [[nodiscard]] std::span<const std::int32_t> b1_q() const noexcept {
        return parts_.b1_q;
    }
    [[nodiscard]] std::int64_t b2_q() const noexcept { return parts_.b2_q; }

    // Column k of W_dict in accumulator scale (I.1-(3)): H int32 values to
    // add when dictionary feature k is active.
    [[nodiscard]] std::span<const std::int32_t>
    dict_col(std::uint32_t k) const noexcept {
        return {parts_.dict_cols.data() +
                    static_cast<std::size_t>(k) * parts_.config.hidden,
                parts_.config.hidden};
    }

    // Int16-mode counterparts: everything requantized by requant_i16 (b2 by
    // the same rounded shift in int64), valid when table_precision() is Int16.
    [[nodiscard]] std::span<const std::int16_t> b1_q16() const noexcept {
        return parts_.b1_q16;
    }
    [[nodiscard]] std::int64_t b2_q16() const noexcept { return parts_.b2_q16; }
    [[nodiscard]] std::span<const std::int16_t>
    dict_col16(std::uint32_t k) const noexcept {
        return {parts_.dict_cols16.data() +
                    static_cast<std::size_t>(k) * parts_.config.hidden,
                parts_.config.hidden};
    }

    // True when the model was trained with different Unicode data than this
    // build's EGC splitter (5.7 field 4b); the loader also warns on stderr.
    [[nodiscard]] bool unicode_version_mismatch() const noexcept {
        return parts_.unicode_mismatch;
    }

private:
    struct Parts {
        Config config;
        Vocab vocab;
        std::vector<std::int16_t> embedding;  // V × d
        std::vector<std::int16_t> w1;         // H × 2w·d
        std::vector<std::int16_t> w2;         // H
        std::vector<std::int32_t> b1_q;       // H
        std::int64_t b2_q = 0;
        std::vector<std::int32_t> dict_cols;  // Fd × H, column-contiguous
        // Int16-mode requantizations of the three above (empty/0 otherwise).
        std::vector<std::int16_t> b1_q16;
        std::int64_t b2_q16 = 0;
        std::vector<std::int16_t> dict_cols16;
        PrecomputeTable precompute;
        DictMatcher dict;
        bool unicode_mismatch = false;
    };
    static Parts parse(bytes::BinaryReader& reader, TablePrecision precision,
                       unsigned format);

    explicit Model(Parts parts) noexcept : parts_(std::move(parts)) {}
    Parts parts_;
};

}  // namespace segmentlib::mlp
