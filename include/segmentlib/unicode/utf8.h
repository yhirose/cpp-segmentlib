#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace segmentlib::unicode {

// Number of bytes in the UTF-8 sequence introduced by `lead`, or 0 if `lead`
// is not a valid leading byte.
constexpr int sequence_length(unsigned char lead) noexcept {
    if (lead < 0x80) return 1;
    if ((lead >> 5) == 0b110) return 2;
    if ((lead >> 4) == 0b1110) return 3;
    if ((lead >> 3) == 0b11110) return 4;
    return 0;
}

struct Decoded {
    char32_t codepoint;
    std::size_t length;  // bytes consumed
};

// Decodes the first codepoint of `s`. Returns nullopt on malformed input:
// invalid leading byte, truncated sequence, bad continuation byte, overlong
// encoding, surrogate, or out-of-range codepoint.
std::optional<Decoded> decode(std::string_view s) noexcept;

// Appends the UTF-8 encoding of `cp` to `out`. `cp` must be a valid Unicode
// scalar value (<= U+10FFFF, not a surrogate); callers pass codepoints that came
// from decode() or a model's character table, so no validation is done here.
void encode(char32_t cp, std::string& out);

}  // namespace segmentlib::unicode
