#include <doctest/doctest.h>

#include "segmentlib/kytea/char_table.h"
#include "segmentlib/unicode/normalize.h"

using segmentlib::unicode::normalize;

TEST_CASE("normalize maps half-width to full-width") {
    CHECK(normalize(U'a') == U'ａ');
    CHECK(normalize(U'Z') == U'Ｚ');
    CHECK(normalize(U'0') == U'０');
    CHECK(normalize(U'(') == U'（');
    CHECK(normalize(U'?') == U'？');
}

TEST_CASE("normalize unifies punctuation variants") {
    CHECK(normalize(U'-') == U'−');          // hyphen-minus -> minus sign
    CHECK(normalize(0xFF9E + 0) == 0xFF9E);  // untouched half-width sound mark
    CHECK(normalize(U'。') == U'。');         // already normalized: unchanged
}

TEST_CASE("normalize leaves non-mapped codepoints unchanged") {
    CHECK(normalize(U'あ') == U'あ');
    CHECK(normalize(U'漢') == U'漢');
    CHECK(normalize(U' ') == U' ');
    CHECK(normalize(0x2000B) == 0x2000B);  // astral: never remapped
}

TEST_CASE("kytea::normalize remains available as a re-export") {
    CHECK(segmentlib::kytea::normalize(U'a') == U'ａ');
}
