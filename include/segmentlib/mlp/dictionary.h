#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "segmentlib/mlp/vocab.h"
#include "segmentlib/text/aho_corasick.h"

namespace segmentlib::mlp {

// Dictionary binary features (design.ja.md 4.4): per dictionary, position
// relation {L: match starts right after the boundary, I: match spans it,
// R: match ends right at it} × length bucket min(EGC-length, 4). The index
// within a dictionary is `position*4 + (bucket-1)`; the global index is
// `dict*12 + that`, matching the W_dict column layout (5.7 field 13). This is
// the canonical definition — the trainer's feature generation and the
// inference matcher below both use it.
inline constexpr std::uint32_t kDictFeaturesPerDict = 12;
enum class DictPosition : std::uint32_t { Left = 0, Inside = 1, Right = 2 };

// Per-boundary active dictionary features, CSR: boundary b owns
// indices[offsets[b] .. offsets[b+1]), sorted and unique (binary clamp, 5.4).
// Also carries the matcher's per-call scratch so features_into allocates
// nothing once the buffers have grown to a sentence's size.
struct DictFeatures {
    std::vector<std::uint32_t> offsets;  // size: boundary count + 1
    std::vector<std::uint32_t> indices;

    // scratch (managed by DictMatcher::features_into)
    std::vector<std::uint32_t> keys;
    std::vector<std::vector<std::uint32_t>> per_boundary;
};

// The dictionary feature extractor: words are
// normalized, EGC-split, interned to dictionary-local EGC ids and compiled
// into one runtime-built Aho-Corasick over EGC ids. Input EGCs are keyed by
// their normalized codepoint sequence (EncodedEgc::egc_cps), never by
// embedding rows — distinct codepoints alias under UNK, and a dictionary
// match must not.
class DictMatcher {
public:
    // No dictionaries: features_into emits empty ranges everywhere.
    DictMatcher() = default;

    // Builds from raw word lists, one per dictionary channel. Entries are
    // normalized (方式(a)) before EGC splitting, matching the treatment of
    // the input side (5.7 field 17b). Invalid-UTF-8 or empty entries are
    // dropped (they can never match).
    explicit DictMatcher(std::span<const std::vector<std::string>> dictionaries);

    [[nodiscard]] std::uint32_t num_dicts() const noexcept { return num_dicts_; }
    [[nodiscard]] std::uint32_t feature_count() const noexcept {
        return num_dicts_ * kDictFeaturesPerDict;
    }

    // Computes the active features of every boundary of `enc` into `out`.
    void features_into(const EncodedEgc& enc, DictFeatures& out) const;

private:
    // Heterogeneous hashing so lookups key on u32string_view without
    // allocating a u32string per EGC on the scoring path.
    struct Hash {
        using is_transparent = void;
        [[nodiscard]] std::size_t operator()(std::u32string_view s) const noexcept {
            // FNV-1a over the codepoints; the map is loaded once and read
            // per-EGC, so simplicity beats sophistication here.
            std::size_t h = 14695981039346656037ull;
            for (const char32_t c : s) {
                h = (h ^ static_cast<std::size_t>(c)) * 1099511628211ull;
            }
            return h;
        }
        [[nodiscard]] std::size_t operator()(const std::u32string& s) const noexcept {
            return operator()(std::u32string_view(s));
        }
    };

    // Sentinel key for input EGCs absent from every dictionary word; it has
    // no transition anywhere, so the automaton falls back to the root.
    static constexpr std::uint32_t kUnknownEgc = 0xFFFFFFFFu;

    std::unordered_map<std::u32string, std::uint32_t, Hash, std::equal_to<>>
        egc_ids_;
    text::AhoCorasick<std::uint32_t, std::uint8_t> ac_;
    std::uint32_t num_dicts_ = 0;
};

}  // namespace segmentlib::mlp
