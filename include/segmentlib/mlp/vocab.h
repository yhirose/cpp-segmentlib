#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "segmentlib/types.h"

namespace segmentlib::mlp {

// Reserved embedding rows (design.ja.md 5.7 field 9): row 0 is the PAD token
// (window positions past the ends of the text), row 1 is UNK (codepoints not
// in the vocabulary). Rows 2.. correspond to the model's sorted codepoint
// array in order.
inline constexpr std::uint32_t kPadRow = 0;
inline constexpr std::uint32_t kUnkRow = 1;

// An input string encoded for the MLP scorer: normalized, split into
// extended grapheme clusters, each cluster's constituent codepoints mapped to
// embedding row ids. The row-id lists are variable-length (almost always one
// codepoint per EGC in Japanese/Chinese), so they are stored CSR-style:
// cluster i owns rows[egc_starts[i] .. egc_starts[i+1]).
//
// `offsets` has size egc_count()+1 and records the byte span of each cluster
// in the *original* (un-normalized) input, so words can be sliced out of the
// source text. Normalization is codepoint-to-codepoint, so original byte
// spans are well-defined even though row ids reflect normalized codepoints.
struct EncodedEgc {
    std::vector<std::uint32_t> rows;        // constituent-codepoint row ids
    std::vector<char32_t> cps;              // normalized codepoints, ∥ rows
    std::vector<std::uint32_t> egc_starts;  // size egc_count()+1, CSR into rows
    std::vector<std::size_t> offsets;       // size egc_count()+1, original bytes

    // Number of EGCs (M). Boundary candidates are the M-1 inter-cluster gaps.
    [[nodiscard]] std::size_t egc_count() const noexcept {
        return egc_starts.empty() ? 0 : egc_starts.size() - 1;
    }

    // Row ids of cluster i's constituent codepoints.
    [[nodiscard]] std::span<const std::uint32_t> egc_rows(std::size_t i) const noexcept {
        return {rows.data() + egc_starts[i], rows.data() + egc_starts[i + 1]};
    }

    // Normalized codepoints of cluster i. Row ids alias distinct codepoints
    // under UNK, so consumers that need the identity of a cluster — the
    // dictionary matcher's EGC interning (5.4) — key on these instead.
    [[nodiscard]] std::u32string_view egc_cps(std::size_t i) const noexcept {
        return std::u32string_view(cps.data() + egc_starts[i],
                                   egc_starts[i + 1] - egc_starts[i]);
    }
};

// The MLP model's codepoint vocabulary (design.ja.md 5.3/5.7): maps a
// codepoint to its embedding row. Corresponds to kytea::CharTable, but the
// id space is embedding rows with reserved PAD/UNK rows, and there is no
// character-type channel (5.1: no heuristic features).
class Vocab {
public:
    // An empty vocabulary: every codepoint maps to UNK.
    Vocab() : Vocab(std::vector<char32_t>{}) {}

    // Takes the model's codepoint array (5.7 field 10): strictly ascending,
    // codepoint i mapping to row i+2. The loader validates the ordering when
    // parsing the model file; this constructor asserts it in debug builds.
    explicit Vocab(std::vector<char32_t> sorted_codepoints);

    // Embedding row for a codepoint; kUnkRow if it is not in the vocabulary.
    // BMP codepoints (the common case) resolve through a direct-indexed
    // table; only rarer astral codepoints fall back to binary search.
    [[nodiscard]] std::uint32_t row_of(char32_t cp) const noexcept {
        return cp < kBmpSize ? bmp_rows_[cp] : row_of_astral(cp);
    }

    // Number of embedding rows V, including the PAD and UNK rows.
    [[nodiscard]] std::uint32_t size() const noexcept {
        return static_cast<std::uint32_t>(codepoints_.size()) + 2;
    }

    // The ascending codepoint array (5.7 field 10): codepoints()[i]
    // corresponds to embedding row i+2. The model exporter writes this out
    // verbatim.
    [[nodiscard]] std::span<const char32_t> codepoints() const noexcept {
        return codepoints_;
    }

    // Encodes UTF-8 input: normalization (unicode::normalize, 方式(a)) →
    // EGC segmentation (unicode::egc) → embedding row ids. Returns
    // InvalidUtf8 on malformed input.
    [[nodiscard]] std::expected<EncodedEgc, Error> encode(std::string_view utf8) const;

    // Same as encode(), but fills a caller-owned EncodedEgc (its buffers are
    // reused, avoiding per-call allocation on the hot path). On error the
    // contents of `out` are unspecified.
    [[nodiscard]] std::expected<void, Error> encode_into(std::string_view utf8,
                                                         EncodedEgc& out) const;

private:
    static constexpr char32_t kBmpSize = 0x1'0000;

    [[nodiscard]] std::uint32_t row_of_astral(char32_t cp) const noexcept;

    std::vector<char32_t> codepoints_;      // ascending; row i+2 <-> codepoints_[i]
    std::vector<std::uint32_t> bmp_rows_;   // size kBmpSize, direct-indexed
};

}  // namespace segmentlib::mlp
