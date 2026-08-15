#pragma once

namespace segmentlib::unicode {

// KyTea's fixed half-width -> full-width (and punctuation) normalization
// ("方式(a)", design.ja.md 4.5): the same table KyTea applies before feature
// extraction, promoted here so every backend (KyTea, MLP, and later
// Vaporetto) normalizes input — and dictionary entries — identically without
// depending on the kytea namespace. Returns the normalized codepoint, or `cp`
// unchanged if it is not remapped. Only BMP codepoints are ever remapped.
[[nodiscard]] char32_t normalize(char32_t cp);

}  // namespace segmentlib::unicode
