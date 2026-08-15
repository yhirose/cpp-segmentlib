#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "segmentlib/mlp/vocab.h"

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
    //
    // The matcher runs over the sentence re-encoded as normalized UTF-8, so
    // `text` is that encoding, `egc_byte_starts` its per-cluster byte offsets
    // (byte offsets, unlike EncodedEgc::egc_starts, which indexes codepoints),
    // and `egc_at_byte` the reverse map used to reject matches that would end
    // in the middle of a cluster.
    std::string text;
    std::vector<std::uint32_t> egc_byte_starts;  // size: EGC count + 1
    std::vector<std::uint32_t> egc_at_byte;      // size: text.size() + 1
    // Per-boundary bitmap of active features, `mask_words` 64-bit words per
    // boundary. The feature space is num_dicts*12, so one word covers up to
    // five dictionaries; matches set bits and the CSR output is a bit scan,
    // which emits each boundary's features ascending and deduplicated.
    std::vector<std::uint64_t> masks;
    std::size_t mask_words = 0;
};

// The dictionary feature extractor: entries are normalized and compiled into
// one FST (cpp-fstlib) keyed by their normalized UTF-8 bytes, which is queried
// with a common-prefix search from every cluster start.
//
// Matching is byte-level but the features are defined over EGCs, so a match is
// kept only when its end lands on a cluster boundary. That check is what keeps
// a dictionary entry from matching a proper prefix of a longer cluster (an
// entry "か" must not match inside the cluster "か" + U+3099). Keying on the
// normalized text rather than on embedding rows also keeps distinct codepoints
// that alias under UNK from matching each other.
class DictMatcher {
public:
    // No dictionaries: features_into emits empty ranges everywhere.
    DictMatcher() noexcept;

    // Builds from raw word lists, one per dictionary channel. Entries are
    // normalized (方式(a)) before use, matching the treatment of the input
    // side (5.7 field 17b). Invalid-UTF-8 or empty entries are dropped (they
    // can never match). An entry appearing in several channels is stored once,
    // carrying the set of channels it belongs to.
    explicit DictMatcher(std::span<const std::vector<std::string>> dictionaries);

    ~DictMatcher();
    DictMatcher(const DictMatcher& other);
    DictMatcher& operator=(const DictMatcher& other);
    DictMatcher(DictMatcher&&) noexcept;
    DictMatcher& operator=(DictMatcher&&) noexcept;

    [[nodiscard]] std::uint32_t num_dicts() const noexcept { return num_dicts_; }
    [[nodiscard]] std::uint32_t feature_count() const noexcept {
        return num_dicts_ * kDictFeaturesPerDict;
    }

    // Computes the active features of every boundary of `enc` into `out`.
    void features_into(const EncodedEgc& enc, DictFeatures& out) const;

private:
    // Holds the compiled FST and the matcher viewing it; defined in the .cpp
    // so the vendored fstlib header stays out of this public header.
    struct Impl;

    std::unique_ptr<Impl> impl_;  // null when there is nothing to match
    std::uint32_t num_dicts_ = 0;
};

}  // namespace segmentlib::mlp
