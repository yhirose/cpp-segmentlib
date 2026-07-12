#include "segmentlib/unicode/egc.h"

#include "segmentlib/unicode/utf8.h"

namespace segmentlib::unicode {

namespace {

#include "egc_table.inc"  // kEgcStage1, kEgcStage2, kEgcTableUnicodeVersion

// The generated table and the header constant must describe the same Unicode
// version; regenerate the table and bump kEgcUnicodeVersion together.
static_assert(kEgcTableUnicodeVersion == kEgcUnicodeVersion);

constexpr std::size_t kBlockSize = 256;

}  // namespace

GraphemeProps grapheme_props(char32_t cp) noexcept {
    std::uint8_t packed = 0;
    if (cp < 0x110000) {
        const std::size_t block = kEgcStage1[cp >> 8];
        packed = kEgcStage2[block * kBlockSize + (cp & 0xFF)];
    }
    return GraphemeProps{
        .gcb = static_cast<GraphemeBreak>(packed & 0x0F),
        .incb = static_cast<IndicConjunctBreak>((packed >> 4) & 0x03),
        .extended_pictographic = (packed & 0x40) != 0,
    };
}

bool GraphemeBreaker::next(char32_t cp) noexcept {
    const GraphemeProps props = grapheme_props(cp);
    const GraphemeBreak cur = props.gcb;

    bool is_break;
    if (at_start_) {
        is_break = true;  // GB1: break at start of text
    } else if (prev_ == GraphemeBreak::CR && cur == GraphemeBreak::LF) {
        is_break = false;  // GB3: keep CR LF together
    } else if (prev_ == GraphemeBreak::Control || prev_ == GraphemeBreak::CR ||
               prev_ == GraphemeBreak::LF) {
        is_break = true;  // GB4: break after controls
    } else if (cur == GraphemeBreak::Control || cur == GraphemeBreak::CR ||
               cur == GraphemeBreak::LF) {
        is_break = true;  // GB5: break before controls
    } else if (prev_ == GraphemeBreak::L &&
               (cur == GraphemeBreak::L || cur == GraphemeBreak::V ||
                cur == GraphemeBreak::LV || cur == GraphemeBreak::LVT)) {
        is_break = false;  // GB6: Hangul L x (L|V|LV|LVT)
    } else if ((prev_ == GraphemeBreak::LV || prev_ == GraphemeBreak::V) &&
               (cur == GraphemeBreak::V || cur == GraphemeBreak::T)) {
        is_break = false;  // GB7: Hangul (LV|V) x (V|T)
    } else if ((prev_ == GraphemeBreak::LVT || prev_ == GraphemeBreak::T) &&
               cur == GraphemeBreak::T) {
        is_break = false;  // GB8: Hangul (LVT|T) x T
    } else if (cur == GraphemeBreak::Extend || cur == GraphemeBreak::ZWJ) {
        is_break = false;  // GB9: keep Extend / ZWJ with what precedes
    } else if (cur == GraphemeBreak::SpacingMark) {
        is_break = false;  // GB9a: keep spacing marks attached
    } else if (prev_ == GraphemeBreak::Prepend) {
        is_break = false;  // GB9b: keep prepend characters attached
    } else if (gb9c_ == Gb9cState::ConsonantLinker &&
               props.incb == IndicConjunctBreak::Consonant) {
        is_break = false;  // GB9c: Indic conjunct (consonant linker x consonant)
    } else if (gb11_ == Gb11State::PictographicZwj && props.extended_pictographic) {
        is_break = false;  // GB11: emoji ZWJ sequence
    } else if (prev_ == GraphemeBreak::RegionalIndicator &&
               cur == GraphemeBreak::RegionalIndicator && (ri_run_ % 2) == 1) {
        is_break = false;  // GB12/GB13: join regional indicators pairwise
    } else {
        is_break = true;  // GB999: break everywhere else
    }

    // Update the multi-codepoint rule states with the codepoint just fed.

    // GB11 tracks ExtPict Extend* ZWJ; the Extend/ZWJ continuations only
    // count while a pictographic sequence is open.
    if (gb11_ == Gb11State::Pictographic && cur == GraphemeBreak::Extend) {
        // still Pictographic
    } else if (gb11_ == Gb11State::Pictographic && cur == GraphemeBreak::ZWJ) {
        gb11_ = Gb11State::PictographicZwj;
    } else if (props.extended_pictographic) {
        gb11_ = Gb11State::Pictographic;
    } else {
        gb11_ = Gb11State::None;
    }

    // GB9c tracks Consonant [Extend|Linker]* with at least one Linker.
    if (props.incb == IndicConjunctBreak::Consonant) {
        gb9c_ = Gb9cState::Consonant;
    } else if (gb9c_ != Gb9cState::None && props.incb == IndicConjunctBreak::Linker) {
        gb9c_ = Gb9cState::ConsonantLinker;
    } else if (gb9c_ != Gb9cState::None && props.incb == IndicConjunctBreak::Extend) {
        // sequence stays open in its current state
    } else {
        gb9c_ = Gb9cState::None;
    }

    ri_run_ = cur == GraphemeBreak::RegionalIndicator ? ri_run_ + 1 : 0;

    at_start_ = false;
    prev_ = cur;
    return is_break;
}

std::expected<std::vector<std::size_t>, Error> egc_split(std::string_view utf8) {
    std::vector<std::size_t> offsets;
    if (auto r = egc_split_into(utf8, offsets); !r) {
        return std::unexpected(r.error());
    }
    return offsets;
}

std::expected<void, Error> egc_split_into(std::string_view utf8,
                                          std::vector<std::size_t>& offsets) {
    offsets.clear();
    GraphemeBreaker breaker;
    std::size_t pos = 0;
    while (pos < utf8.size()) {
        const auto decoded = decode(utf8.substr(pos));
        if (!decoded) {
            return std::unexpected(Error{ErrorCode::InvalidUtf8,
                                         "malformed UTF-8 in EGC segmentation"});
        }
        if (breaker.next(decoded->codepoint)) {
            offsets.push_back(pos);
        }
        pos += decoded->length;
    }
    offsets.push_back(utf8.size());  // GB2: end of text
    return {};
}

}  // namespace segmentlib::unicode
