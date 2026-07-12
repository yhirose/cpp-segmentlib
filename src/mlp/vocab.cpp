#include "segmentlib/mlp/vocab.h"

#include <algorithm>
#include <cassert>

#include "segmentlib/unicode/egc.h"
#include "segmentlib/unicode/normalize.h"
#include "segmentlib/unicode/utf8.h"

namespace segmentlib::mlp {

Vocab::Vocab(std::vector<char32_t> sorted_codepoints)
    : codepoints_(std::move(sorted_codepoints)) {
    assert(std::is_sorted(codepoints_.begin(), codepoints_.end()) &&
           std::adjacent_find(codepoints_.begin(), codepoints_.end()) ==
               codepoints_.end());

    bmp_rows_.assign(kBmpSize, kUnkRow);
    for (std::size_t i = 0; i < codepoints_.size(); ++i) {
        if (codepoints_[i] < kBmpSize) {
            bmp_rows_[codepoints_[i]] = static_cast<std::uint32_t>(i) + 2;
        }
    }
}

std::uint32_t Vocab::row_of_astral(char32_t cp) const noexcept {
    const auto it = std::lower_bound(codepoints_.begin(), codepoints_.end(), cp);
    if (it != codepoints_.end() && *it == cp) {
        return static_cast<std::uint32_t>(it - codepoints_.begin()) + 2;
    }
    return kUnkRow;
}

std::expected<void, Error> Vocab::encode_into(std::string_view utf8,
                                              EncodedEgc& out) const {
    out.rows.clear();
    out.cps.clear();
    out.egc_starts.clear();
    out.offsets.clear();

    // Normalization is applied per codepoint *before* cluster segmentation
    // (design.ja.md 5.5 方式(a)), so the breaker sees normalized codepoints;
    // offsets still index the original bytes (normalization is 1:1 on
    // codepoints, so cluster spans carry over).
    unicode::GraphemeBreaker breaker;
    std::size_t pos = 0;
    while (pos < utf8.size()) {
        const auto decoded = unicode::decode(utf8.substr(pos));
        if (!decoded) {
            return std::unexpected(Error{ErrorCode::InvalidUtf8, "invalid UTF-8 in input"});
        }
        const char32_t norm = unicode::normalize(decoded->codepoint);
        if (breaker.next(norm)) {
            out.egc_starts.push_back(static_cast<std::uint32_t>(out.rows.size()));
            out.offsets.push_back(pos);
        }
        out.rows.push_back(row_of(norm));
        out.cps.push_back(norm);
        pos += decoded->length;
    }
    out.egc_starts.push_back(static_cast<std::uint32_t>(out.rows.size()));
    out.offsets.push_back(utf8.size());
    return {};
}

std::expected<EncodedEgc, Error> Vocab::encode(std::string_view utf8) const {
    EncodedEgc out;
    if (auto r = encode_into(utf8, out); !r) {
        return std::unexpected(r.error());
    }
    return out;
}

}  // namespace segmentlib::mlp
