#include "segmentlib/kytea/char_table.h"

#include "segmentlib/bytes/binary_reader.h"
#include "segmentlib/unicode/utf8.h"

namespace segmentlib::kytea {

// normalize() moved to src/unicode/normalize.cpp (shared with the MLP
// backend); kytea::normalize is a using-declaration in char_table.h.

CharTable::CharTable(std::string_view char_map_utf8) {
    CharId next_id = 1;   // id 0 is the reserved empty-string sentinel
    id_to_cp_.push_back(0);  // index 0: the empty-string sentinel
    std::size_t pos = 0;
    while (pos < char_map_utf8.size()) {
        const auto dec = unicode::decode(char_map_utf8.substr(pos));
        if (!dec) {
            throw bytes::ParseError("invalid UTF-8 in model character map");
        }
        // First occurrence wins; a repeated character does not consume an id.
        if (ids_.try_emplace(dec->codepoint, next_id).second) {
            id_to_cp_.push_back(dec->codepoint);  // index == next_id
            ++next_id;
        }
        pos += dec->length;
    }

    // Build the BMP fast-path table and precompute the type-marker ids.
    bmp_ids_.assign(kBmpSize, kNoChar);
    for (const auto& [cp, id] : ids_) {
        if (cp < kBmpSize) {
            bmp_ids_[cp] = id;
        }
    }
    for (std::size_t t = 0; t < type_ids_.size(); ++t) {
        type_ids_[t] = id_of(type_marker(static_cast<CharType>(t)));
    }
}

std::string CharTable::decode(std::span<const CharId> ids) const {
    std::string out;
    for (const CharId id : ids) {
        const char32_t cp = id < id_to_cp_.size() ? id_to_cp_[id] : 0;
        if (cp != 0) {  // id 0 (empty-string sentinel) contributes nothing
            unicode::encode(cp, out);
        }
    }
    return out;
}

CharId CharTable::id_of_astral(char32_t cp) const noexcept {
    const auto it = ids_.find(cp);
    return it == ids_.end() ? kNoChar : it->second;
}

std::expected<void, Error> CharTable::encode_into(std::string_view utf8, EncodedText& out) const {
    out.char_ids.clear();
    out.type_ids.clear();
    out.offsets.clear();
    std::size_t pos = 0;
    while (pos < utf8.size()) {
        const auto dec = unicode::decode(utf8.substr(pos));
        if (!dec) {
            return std::unexpected(Error{ErrorCode::InvalidUtf8, "invalid UTF-8 in input"});
        }
        const char32_t norm = normalize(dec->codepoint);
        out.offsets.push_back(pos);
        out.char_ids.push_back(id_of(norm));
        out.type_ids.push_back(type_id(classify(norm)));
        pos += dec->length;
    }
    out.offsets.push_back(utf8.size());
    return {};
}

std::expected<EncodedText, Error> CharTable::encode(std::string_view utf8) const {
    EncodedText out;
    if (auto r = encode_into(utf8, out); !r) {
        return std::unexpected(r.error());
    }
    return out;
}

}  // namespace segmentlib::kytea
