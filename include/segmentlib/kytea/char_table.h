#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "segmentlib/types.h"
#include "segmentlib/unicode/normalize.h"

namespace segmentlib::kytea {

// KyTea's internal character-id type. Characters are interned into small
// integers; the automata in the model are keyed by these ids.
using CharId = std::uint16_t;

// Id reserved by KyTea for the empty string; never appears as an automaton
// transition, so unknown input characters map here and simply fail to match.
//
// This is provably equivalent to what KyTea does with unknown characters, even
// though it does something different: at inference KyTea's mapString() calls
// mapChar(str, add=true), which assigns each unknown character a *fresh* growing
// id (charTypes_.size(), i.e. > the max trained id) rather than 0. But every WS
// feature in calculateWS is an automaton match (char n-gram, type n-gram, dict),
// all keyed by the trained ids 1..K; an unknown character is absent from every
// transition whether its id is 0 (here) or K+1 (KyTea), so it can never
// participate in a char-n-gram or dictionary match either way. Its type n-gram
// contribution is unaffected because the character *type* is derived from the
// codepoint (see encode_into), not from the char id. Verified byte-for-byte
// against `kytea -notags` on unknown emoji/symbols in the golden fixtures.
inline constexpr CharId kNoChar = 0;

enum class CharType : std::uint8_t { Romaji, Hiragana, Katakana, Digit, Kanji, Other };

// Classifies a Unicode codepoint using KyTea's `findType` ranges.
//
// Note: KyTea's findType has a bug in its 4-byte UTF-8 arithmetic that
// misclassifies codepoints >= U+10000 (CJK Ext. B and beyond). We classify
// those correctly as KANJI, which can diverge from KyTea only on such rare
// characters.
[[nodiscard]] constexpr CharType classify(char32_t v) noexcept {
    if ((v >= 0x41 && v <= 0x5A) || (v >= 0x61 && v <= 0x7A) ||
        (v >= 0xFF21 && v <= 0xFF3A) || (v >= 0xFF41 && v <= 0xFF5A)) {
        return CharType::Romaji;
    }
    if (v >= 0x3040 && v <= 0x3096) {
        return CharType::Hiragana;
    }
    if ((v >= 0x30A0 && v <= 0x30FF && v != 0x30FB) || (v >= 0xFF66 && v <= 0xFF9F)) {
        return CharType::Katakana;
    }
    if ((v >= 0x30 && v <= 0x39) || (v >= 0xFF10 && v <= 0xFF19)) {
        return CharType::Digit;
    }
    if ((v >= 0x3400 && v <= 0x4DBF) || (v >= 0x4E00 && v <= 0x9FFF) ||
        (v >= 0xF900 && v <= 0xFAFF) || (v >= 0x20000 && v <= 0x2A6DF) ||
        (v >= 0x2A700 && v <= 0x2B73F) || (v >= 0x2B740 && v <= 0x2B81F) ||
        (v >= 0x2F800 && v <= 0x2FA1F)) {
        return CharType::Kanji;
    }
    return CharType::Other;
}

// The single ASCII letter KyTea uses to mark a character type (R/H/T/D/K/O),
// as a codepoint. These markers are themselves interned in the char table, so
// the type n-gram automaton is keyed by their ids.
[[nodiscard]] constexpr char32_t type_marker(CharType t) noexcept {
    switch (t) {
        case CharType::Romaji:   return U'R';
        case CharType::Hiragana: return U'H';
        case CharType::Katakana: return U'T';
        case CharType::Digit:    return U'D';
        case CharType::Kanji:    return U'K';
        case CharType::Other:    return U'O';
    }
    return U'O';
}

// KyTea's fixed half-width -> full-width (and punctuation) normalization.
// The implementation lives in unicode/normalize.h (it is shared with the MLP
// backend, design.ja.md 5.5 方式(a)); re-exported here so existing kytea code
// keeps referring to kytea::normalize.
using unicode::normalize;

// An input string encoded into this table's id space. `char_ids` and
// `type_ids` are the normalized-character and character-type sequences the
// scorer feeds to the automata. `offsets` has size N+1 and records the byte
// span of each character in the *original* (un-normalized) input, so words can
// be sliced out of the source text.
struct EncodedText {
    std::vector<CharId> char_ids;
    std::vector<CharId> type_ids;
    std::vector<std::size_t> offsets;

    [[nodiscard]] std::size_t length() const noexcept { return char_ids.size(); }
};

// Per-model character interning table, reconstructed from the model's
// serialized character-map string. Reproduces KyTea's id assignment exactly:
// each character of the map string receives ids 1, 2, 3, ... in order (id 0 is
// the reserved empty-string sentinel).
class CharTable {
public:
    // Parses the model's character-map string. Throws bytes::ParseError on
    // malformed UTF-8 (caught at the model-loading boundary).
    explicit CharTable(std::string_view char_map_utf8);

    // Interned id for a codepoint, or kNoChar if it was never seen in training.
    // BMP codepoints (the common case) resolve through a direct-indexed table;
    // only rarer astral codepoints fall back to the hash map.
    [[nodiscard]] CharId id_of(char32_t cp) const noexcept {
        return cp < kBmpSize ? bmp_ids_[cp] : id_of_astral(cp);
    }

    [[nodiscard]] std::size_t size() const noexcept { return ids_.size(); }

    // Decodes a KyteaString (a sequence of interned char ids, as tag candidate
    // strings are stored in the model) back to UTF-8, the inverse of the id
    // assignment done at construction. The reserved empty-string id 0 yields no
    // output. Used off the hot path, at load time, to materialize tag strings.
    [[nodiscard]] std::string decode(std::span<const CharId> ids) const;

    // Encodes UTF-8 input into id sequences plus byte offsets. Normalization is
    // applied before interning. Returns InvalidUtf8 on malformed input.
    [[nodiscard]] std::expected<EncodedText, Error> encode(std::string_view utf8) const;

    // Same as encode(), but fills a caller-owned EncodedText (its buffers are
    // reused, avoiding per-call allocation on the hot path).
    [[nodiscard]] std::expected<void, Error> encode_into(std::string_view utf8,
                                                         EncodedText& out) const;

    // Interned id of the type marker for each CharType (precomputed so the type
    // n-gram encoding needs no per-character lookup).
    [[nodiscard]] CharId type_id(CharType t) const noexcept {
        return type_ids_[static_cast<std::size_t>(t)];
    }

private:
    static constexpr char32_t kBmpSize = 0x1'0000;

    [[nodiscard]] CharId id_of_astral(char32_t cp) const noexcept;

    std::unordered_map<char32_t, CharId> ids_;  // authoritative; astral fallback
    std::vector<CharId> bmp_ids_;               // size kBmpSize, direct-indexed
    std::vector<char32_t> id_to_cp_;            // id -> codepoint (index 0 unused)
    std::array<CharId, 6> type_ids_{};          // by CharType
};

}  // namespace segmentlib::kytea
