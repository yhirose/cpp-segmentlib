#include "segmentlib/unicode/utf8.h"

namespace segmentlib::unicode {

namespace {

constexpr bool is_continuation(unsigned char b) noexcept {
    return (b >> 6) == 0b10;
}

// Smallest codepoint legally encodable in a sequence of the given length.
// Used to reject overlong encodings. Index by sequence length (1..4).
constexpr char32_t kMinCodepoint[5] = {0, 0x0, 0x80, 0x800, 0x10000};

}  // namespace

std::optional<Decoded> decode(std::string_view s) noexcept {
    if (s.empty()) {
        return std::nullopt;
    }
    const auto lead = static_cast<unsigned char>(s[0]);
    const int len = sequence_length(lead);
    if (len == 0 || static_cast<std::size_t>(len) > s.size()) {
        return std::nullopt;
    }

    char32_t cp = 0;
    switch (len) {
        case 1: cp = lead; break;
        case 2: cp = lead & 0x1F; break;
        case 3: cp = lead & 0x0F; break;
        default: cp = lead & 0x07; break;
    }
    for (int i = 1; i < len; ++i) {
        const auto b = static_cast<unsigned char>(s[i]);
        if (!is_continuation(b)) {
            return std::nullopt;
        }
        cp = (cp << 6) | (b & 0x3F);
    }

    // Reject overlong encodings, surrogates, and out-of-range codepoints.
    if (cp < kMinCodepoint[len] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
        return std::nullopt;
    }
    return Decoded{cp, static_cast<std::size_t>(len)};
}

void encode(char32_t cp, std::string& out) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x1'0000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

}  // namespace segmentlib::unicode
