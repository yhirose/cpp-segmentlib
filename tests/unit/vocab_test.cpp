#include <doctest/doctest.h>

#include <cstdint>
#include <vector>

#include "segmentlib/mlp/vocab.h"

using namespace segmentlib;
using namespace segmentlib::mlp;

namespace {

// A small vocabulary: 日(U+65E5) 本(U+672C) 語(U+8A9E), full-width ａ(U+FF41),
// combining dakuten (U+3099), か(U+304B), and an astral kanji 𠀋(U+2000B).
// Ascending order is required by the Vocab contract.
Vocab test_vocab() {
    return Vocab{{0x304B, 0x3099, 0x65E5, 0x672C, 0x8A9E, 0xFF41, 0x2000B}};
}

}  // namespace

TEST_CASE("row_of maps vocabulary codepoints to rows 2.. in order") {
    const Vocab v = test_vocab();
    CHECK(v.size() == 9);  // 7 codepoints + PAD + UNK
    CHECK(v.row_of(0x304B) == 2);
    CHECK(v.row_of(0x3099) == 3);
    CHECK(v.row_of(0x65E5) == 4);
    CHECK(v.row_of(0x8A9E) == 6);
    CHECK(v.row_of(0x2000B) == 8);  // astral: binary-search path
}

TEST_CASE("row_of maps unknown codepoints to UNK") {
    const Vocab v = test_vocab();
    CHECK(v.row_of(U'あ') == kUnkRow);
    CHECK(v.row_of(0x2000C) == kUnkRow);  // unknown astral
    CHECK(v.row_of(0) == kUnkRow);

    const Vocab empty;
    CHECK(empty.size() == 2);  // just PAD + UNK
    CHECK(empty.row_of(U'a') == kUnkRow);
}

TEST_CASE("encode maps one row per EGC for plain text") {
    const Vocab v = test_vocab();
    const auto enc = v.encode("日本語");
    REQUIRE(enc.has_value());
    CHECK(enc->egc_count() == 3);
    CHECK(enc->rows == std::vector<std::uint32_t>{4, 5, 6});
    CHECK(enc->egc_starts == std::vector<std::uint32_t>{0, 1, 2, 3});
    CHECK(enc->offsets == std::vector<std::size_t>{0, 3, 6, 9});
}

TEST_CASE("encode normalizes before looking up rows") {
    const Vocab v = test_vocab();
    // Half-width 'a' normalizes to full-width ａ (U+FF41), which is in the
    // vocabulary — but the offsets still span the original 1-byte input.
    const auto enc = v.encode("a");
    REQUIRE(enc.has_value());
    CHECK(enc->egc_count() == 1);
    CHECK(enc->rows == std::vector<std::uint32_t>{7});
    CHECK(enc->offsets == std::vector<std::size_t>{0, 1});
}

TEST_CASE("encode groups a combining sequence into one multi-row EGC") {
    const Vocab v = test_vocab();
    // か + combining dakuten (U+304B U+3099): one EGC, two constituent rows.
    const auto enc = v.encode("\xE3\x81\x8B\xE3\x82\x99日");
    REQUIRE(enc.has_value());
    REQUIRE(enc->egc_count() == 2);
    CHECK(enc->egc_starts == std::vector<std::uint32_t>{0, 2, 3});
    CHECK(enc->rows == std::vector<std::uint32_t>{2, 3, 4});
    const auto first = enc->egc_rows(0);
    CHECK(std::vector<std::uint32_t>(first.begin(), first.end()) ==
          std::vector<std::uint32_t>{2, 3});
    CHECK(enc->offsets == std::vector<std::size_t>{0, 6, 9});
}

TEST_CASE("encode maps unknown codepoints to UNK rows") {
    const Vocab v = test_vocab();
    const auto enc = v.encode("日X語");
    REQUIRE(enc.has_value());
    // 'X' normalizes to full-width Ｘ, which is not in the vocabulary.
    CHECK(enc->rows == std::vector<std::uint32_t>{4, kUnkRow, 6});
}

TEST_CASE("encode of empty input yields zero EGCs") {
    const auto enc = test_vocab().encode("");
    REQUIRE(enc.has_value());
    CHECK(enc->egc_count() == 0);
    CHECK(enc->egc_starts == std::vector<std::uint32_t>{0});
    CHECK(enc->offsets == std::vector<std::size_t>{0});
}

TEST_CASE("encode rejects malformed UTF-8") {
    const auto enc = test_vocab().encode("\xE3\x81");
    REQUIRE_FALSE(enc.has_value());
    CHECK(enc.error().code == ErrorCode::InvalidUtf8);
}

TEST_CASE("encode_into reuses the caller's buffers") {
    const Vocab v = test_vocab();
    EncodedEgc enc;
    enc.rows.assign(16, 999);
    REQUIRE(v.encode_into("日本", enc).has_value());
    CHECK(enc.rows == std::vector<std::uint32_t>{4, 5});
    CHECK(enc.egc_count() == 2);
    REQUIRE(v.encode_into("", enc).has_value());
    CHECK(enc.egc_count() == 0);
}
