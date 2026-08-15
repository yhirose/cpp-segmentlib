#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string_view>
#include <vector>

#include "segmentlib/types.h"

namespace segmentlib::unicode {

// Version of the Unicode data the generated grapheme-break property table was
// built from, encoded as major*100 + minor (16.0 -> 1600). UAX #29 rules and
// property assignments can change between Unicode versions, so model files
// record the version they were trained with (design.ja.md 4.7 field 4b) and
// the loader warns when it differs from this segmenter's tables.
inline constexpr std::uint16_t kEgcUnicodeVersion = 1600;

// UAX #29 Grapheme_Cluster_Break property values (exactly the set the
// extended-grapheme-cluster rules consume). Values must match the packed
// table emitted by scripts/gen_egc_table.py.
enum class GraphemeBreak : std::uint8_t {
    Other = 0,
    CR,
    LF,
    Control,
    Extend,
    ZWJ,
    RegionalIndicator,
    Prepend,
    SpacingMark,
    L,    // Hangul leading consonant
    V,    // Hangul vowel
    T,    // Hangul trailing consonant
    LV,   // Hangul LV syllable
    LVT,  // Hangul LVT syllable
};

// Indic_Conjunct_Break property (UCD "InCB", introduced in Unicode 15.1),
// consumed by rule GB9c which keeps conjuncts like Devanagari क्षि together.
// Values must match the packed table emitted by scripts/gen_egc_table.py.
enum class IndicConjunctBreak : std::uint8_t {
    None = 0,
    Consonant,
    Extend,  // includes ZWJ and most InCB-relevant combining marks
    Linker,  // viramas that join two consonants into a conjunct
};

// The per-codepoint properties the EGC rules need, unpacked from the
// generated two-stage table.
struct GraphemeProps {
    GraphemeBreak gcb;
    IndicConjunctBreak incb;
    bool extended_pictographic;  // Extended_Pictographic, for GB11 (emoji ZWJ)
};

// Looks up the grapheme properties of a codepoint. Codepoints outside the
// Unicode range (> U+10FFFF) get the default properties (Other/None/false);
// callers pass codepoints that came from utf8::decode, so this never happens
// in practice.
[[nodiscard]] GraphemeProps grapheme_props(char32_t cp) noexcept;

// Incremental UAX #29 extended-grapheme-cluster boundary detector.
//
// Feed the codepoints of a text left to right; `next` reports whether an EGC
// boundary lies immediately *before* the codepoint just fed. The first call
// after construction (or reset) always reports a boundary (rule GB1, start of
// text); end of text is always a boundary too (GB2) and needs no call.
//
// The state is a few bytes (previous property + three small rule states), so
// a breaker can live on the stack of any loop that already decodes UTF-8,
// which is how egc_split_into avoids a second decoding pass.
class GraphemeBreaker {
public:
    // Feeds the next codepoint; returns true if an extended grapheme cluster
    // boundary precedes it.
    [[nodiscard]] bool next(char32_t cp) noexcept;

    // Returns to the start-of-text state, so the breaker can be reused for
    // another text without reconstruction.
    void reset() noexcept { *this = GraphemeBreaker{}; }

private:
    // GB11: have we seen Extended_Pictographic Extend* (then ZWJ)?
    enum class Gb11State : std::uint8_t { None, Pictographic, PictographicZwj };
    // GB9c: have we seen InCB=Consonant [InCB=Extend|Linker]* (with a Linker)?
    enum class Gb9cState : std::uint8_t { None, Consonant, ConsonantLinker };

    bool at_start_ = true;
    GraphemeBreak prev_ = GraphemeBreak::Other;
    Gb11State gb11_ = Gb11State::None;
    Gb9cState gb9c_ = Gb9cState::None;
    // Length of the run of Regional_Indicator codepoints ending at the
    // previous position; GB12/13 join RI codepoints pairwise, so only the
    // parity matters, but the count is kept for clarity (it cannot overflow
    // meaningfully: parity is preserved mod 2).
    std::uint32_t ri_run_ = 0;
};

// Splits UTF-8 text into extended grapheme clusters (UAX #29).
//
// Returns the byte offsets of the cluster starts plus a final entry equal to
// utf8.size(), i.e. offsets.size() == egc_count + 1 and cluster i occupies
// bytes [offsets[i], offsets[i+1]). Empty input yields {0} (zero clusters).
// The constituent codepoints of a cluster can be re-decoded from its byte
// span with utf8::decode. Returns InvalidUtf8 on malformed input.
[[nodiscard]] std::expected<std::vector<std::size_t>, Error>
egc_split(std::string_view utf8);

// Same as egc_split(), but fills a caller-owned offsets vector (its buffer is
// reused, avoiding per-call allocation on the hot path). On error the vector
// contents are unspecified.
[[nodiscard]] std::expected<void, Error>
egc_split_into(std::string_view utf8, std::vector<std::size_t>& offsets);

}  // namespace segmentlib::unicode
