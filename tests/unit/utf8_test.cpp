#include <doctest/doctest.h>

#include "segmentlib/unicode/utf8.h"

using namespace segmentlib::unicode;

TEST_CASE("sequence_length classifies lead bytes") {
    CHECK(sequence_length('A') == 1);
    CHECK(sequence_length(0xC3) == 2);
    CHECK(sequence_length(0xE3) == 3);
    CHECK(sequence_length(0xF0) == 4);
    CHECK(sequence_length(0x80) == 0);  // continuation byte is not a lead
    CHECK(sequence_length(0xFF) == 0);
}

TEST_CASE("decode ASCII") {
    auto d = decode("A");
    REQUIRE(d.has_value());
    CHECK(d->codepoint == U'A');
    CHECK(d->length == 1);
}

TEST_CASE("decode 3-byte character") {
    auto d = decode("\xE3\x81\x82");  // U+3042 HIRAGANA A
    REQUIRE(d.has_value());
    CHECK(d->codepoint == 0x3042);
    CHECK(d->length == 3);
}

TEST_CASE("decode 4-byte character") {
    auto d = decode("\xF0\xA0\x80\x8B");  // U+2000B
    REQUIRE(d.has_value());
    CHECK(d->codepoint == 0x2000B);
    CHECK(d->length == 4);
}

TEST_CASE("decode reads only the first codepoint") {
    auto d = decode("AB");
    REQUIRE(d.has_value());
    CHECK(d->codepoint == U'A');
    CHECK(d->length == 1);
}

TEST_CASE("decode rejects malformed input") {
    CHECK_FALSE(decode("").has_value());
    CHECK_FALSE(decode("\x80").has_value());              // lone continuation
    CHECK_FALSE(decode("\xE3\x81").has_value());          // truncated 3-byte
    CHECK_FALSE(decode("\xE3\x28\x82").has_value());      // bad continuation
    CHECK_FALSE(decode("\xC0\x80").has_value());          // overlong NUL
    CHECK_FALSE(decode("\xED\xA0\x80").has_value());      // surrogate U+D800
    CHECK_FALSE(decode("\xF4\x90\x80\x80").has_value());  // > U+10FFFF
}
