#include <doctest/doctest.h>

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "segmentlib/unicode/egc.h"
#include "segmentlib/unicode/utf8.h"

using namespace segmentlib;
using namespace segmentlib::unicode;

namespace {

// Splits `utf8` and returns the EGC substrings, for readable assertions.
std::vector<std::string> clusters(std::string_view utf8) {
    auto offsets = egc_split(utf8);
    REQUIRE(offsets.has_value());
    std::vector<std::string> out;
    for (std::size_t i = 0; i + 1 < offsets->size(); ++i) {
        out.emplace_back(utf8.substr((*offsets)[i], (*offsets)[i + 1] - (*offsets)[i]));
    }
    return out;
}

}  // namespace

TEST_CASE("grapheme_props classifies representative codepoints") {
    CHECK(grapheme_props(U'a').gcb == GraphemeBreak::Other);
    CHECK(grapheme_props(U'\r').gcb == GraphemeBreak::CR);
    CHECK(grapheme_props(U'\n').gcb == GraphemeBreak::LF);
    CHECK(grapheme_props(0x0000).gcb == GraphemeBreak::Control);
    CHECK(grapheme_props(0x3099).gcb == GraphemeBreak::Extend);   // dakuten
    CHECK(grapheme_props(0x200D).gcb == GraphemeBreak::ZWJ);
    CHECK(grapheme_props(0x1F1EF).gcb == GraphemeBreak::RegionalIndicator);
    CHECK(grapheme_props(0x0600).gcb == GraphemeBreak::Prepend);  // Arabic number sign
    CHECK(grapheme_props(0x0903).gcb == GraphemeBreak::SpacingMark);
    CHECK(grapheme_props(0x1100).gcb == GraphemeBreak::L);
    CHECK(grapheme_props(0x1161).gcb == GraphemeBreak::V);
    CHECK(grapheme_props(0x11A8).gcb == GraphemeBreak::T);
    CHECK(grapheme_props(0xAC00).gcb == GraphemeBreak::LV);   // 가
    CHECK(grapheme_props(0xAC01).gcb == GraphemeBreak::LVT);  // 각

    CHECK(grapheme_props(0x1F600).extended_pictographic);  // 😀
    CHECK_FALSE(grapheme_props(U'a').extended_pictographic);

    CHECK(grapheme_props(0x0915).incb == IndicConjunctBreak::Consonant);  // क
    CHECK(grapheme_props(0x094D).incb == IndicConjunctBreak::Linker);     // virama
    CHECK(grapheme_props(0x200D).incb == IndicConjunctBreak::Extend);     // ZWJ
    CHECK(grapheme_props(U'a').incb == IndicConjunctBreak::None);

    // Out of Unicode range: defaults, not a crash.
    CHECK(grapheme_props(0x110000).gcb == GraphemeBreak::Other);
}

TEST_CASE("egc_split on plain Japanese text is one cluster per codepoint") {
    CHECK(clusters("日本語abc") ==
          std::vector<std::string>{"日", "本", "語", "a", "b", "c"});
}

TEST_CASE("egc_split keeps combining sequences together") {
    // か + combining dakuten (U+304B U+3099), NFD が.
    CHECK(clusters("\xE3\x81\x8B\xE3\x82\x99") ==
          std::vector<std::string>{"\xE3\x81\x8B\xE3\x82\x99"});
    // CRLF is one cluster; lone controls break around everything.
    CHECK(clusters("a\r\nb") == std::vector<std::string>{"a", "\r\n", "b"});
}

TEST_CASE("egc_split keeps emoji ZWJ sequences and flags together") {
    // Family: man ZWJ woman ZWJ boy (U+1F468 U+200D U+1F469 U+200D U+1F466).
    const std::string family =
        "\xF0\x9F\x91\xA8\xE2\x80\x8D\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x91\xA6";
    CHECK(clusters(family) == std::vector<std::string>{family});

    // Two flags: RI J RI P RI J RI P -> exactly two clusters (GB12 pairing).
    const std::string flag = "\xF0\x9F\x87\xAF\xF0\x9F\x87\xB5";  // 🇯🇵
    CHECK(clusters(flag + flag) == std::vector<std::string>{flag, flag});
}

TEST_CASE("egc_split keeps Indic conjuncts together (GB9c)") {
    // क्षि = KA VIRAMA SSA VOWEL-I (U+0915 U+094D U+0937 U+093F): one cluster.
    const std::string kshi = "\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB7\xE0\xA4\xBF";
    CHECK(clusters(kshi) == std::vector<std::string>{kshi});
}

TEST_CASE("egc_split handles Hangul jamo composition") {
    // L V T (U+1100 U+1161 U+11A8) forms one syllable cluster.
    const std::string gak = "\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8";
    CHECK(clusters(gak) == std::vector<std::string>{gak});
}

TEST_CASE("egc_split of empty input yields zero clusters") {
    auto offsets = egc_split("");
    REQUIRE(offsets.has_value());
    CHECK(*offsets == std::vector<std::size_t>{0});
}

TEST_CASE("egc_split rejects malformed UTF-8") {
    auto r = egc_split("ab\x80");
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().code == ErrorCode::InvalidUtf8);
}

TEST_CASE("egc_split_into reuses the caller's buffer") {
    std::vector<std::size_t> offsets{99, 99, 99, 99};
    REQUIRE(egc_split_into("ab", offsets).has_value());
    CHECK(offsets == std::vector<std::size_t>{0, 1, 2});
    REQUIRE(egc_split_into("", offsets).has_value());
    CHECK(offsets == std::vector<std::size_t>{0});
}

TEST_CASE("GraphemeBreaker::reset returns to start-of-text state") {
    GraphemeBreaker breaker;
    CHECK(breaker.next(0x304B));         // sot
    CHECK_FALSE(breaker.next(0x3099));   // dakuten attaches
    breaker.reset();
    CHECK(breaker.next(0x3099));         // sot again: even Extend starts a cluster
}

// Full conformance run against the Unicode GraphemeBreakTest.txt for the
// version the property table was generated from. Each test line alternates
// break marks (U+00F7 break, U+00D7 no break) with hex codepoints.
TEST_CASE("UAX #29 GraphemeBreakTest conformance") {
    std::ifstream file(SEGMENTLIB_GRAPHEME_BREAK_TEST_PATH);
    REQUIRE_MESSAGE(file.is_open(),
                    "cannot open " SEGMENTLIB_GRAPHEME_BREAK_TEST_PATH);

    int cases = 0;
    std::string line;
    int line_no = 0;
    while (std::getline(file, line)) {
        ++line_no;
        if (auto hash = line.find('#'); hash != std::string::npos) {
            line.resize(hash);
        }

        // Parse the alternating mark / codepoint tokens.
        std::string text;                        // the test string, as UTF-8
        std::vector<std::size_t> expected;       // expected cluster-start offsets
        bool parse_ok = true;
        std::istringstream tokens(line);
        std::string tok;
        bool pending_break = false;              // mark seen before next codepoint
        while (tokens >> tok) {
            if (tok == "\xC3\xB7") {             // U+00F7 division sign: break
                pending_break = true;
            } else if (tok == "\xC3\x97") {      // U+00D7 multiplication sign: no break
                pending_break = false;
            } else {
                const char32_t cp =
                    static_cast<char32_t>(std::stoul(tok, nullptr, 16));
                if (cp >= 0xD800 && cp <= 0xDFFF) {
                    // Surrogates are not encodable in valid UTF-8; our decoder
                    // rejects them, so such lines (if any) are skipped.
                    parse_ok = false;
                    break;
                }
                if (pending_break) {
                    expected.push_back(text.size());
                }
                encode(cp, text);
            }
        }
        if (!parse_ok || text.empty()) {
            continue;
        }
        expected.push_back(text.size());         // trailing mark = end of text

        auto offsets = egc_split(text);
        REQUIRE_MESSAGE(offsets.has_value(), "line ", line_no, ": ", line);
        CHECK_MESSAGE(*offsets == expected, "line ", line_no, ": ", line);
        ++cases;
    }
    // Guard against silently running nothing (wrong path, format change).
    CHECK(cases > 500);
    MESSAGE("GraphemeBreakTest cases run: ", cases);
}
