#include <doctest/doctest.h>

#include <string_view>
#include <vector>

#include "segmentlib/kytea/char_table.h"

using namespace segmentlib::kytea;
using namespace std::string_view_literals;

TEST_CASE("classify covers each character type") {
    CHECK(classify(U'a') == CharType::Romaji);
    CHECK(classify(U'Z') == CharType::Romaji);
    CHECK(classify(U'ａ') == CharType::Romaji);   // full-width
    CHECK(classify(U'あ') == CharType::Hiragana);
    CHECK(classify(U'ア') == CharType::Katakana);
    CHECK(classify(U'・') == CharType::Other);     // U+30FB is excluded from Katakana
    CHECK(classify(U'0') == CharType::Digit);
    CHECK(classify(U'９') == CharType::Digit);     // full-width
    CHECK(classify(U'水') == CharType::Kanji);
    CHECK(classify(U'、') == CharType::Other);
    CHECK(classify(U' ') == CharType::Other);
}

TEST_CASE("type_marker maps types to KyTea's letters") {
    CHECK(type_marker(CharType::Kanji) == U'K');
    CHECK(type_marker(CharType::Katakana) == U'T');
    CHECK(type_marker(CharType::Hiragana) == U'H');
    CHECK(type_marker(CharType::Romaji) == U'R');
    CHECK(type_marker(CharType::Digit) == U'D');
    CHECK(type_marker(CharType::Other) == U'O');
}

TEST_CASE("normalize folds half-width to full-width") {
    CHECK(normalize(U'a') == U'ａ');
    CHECK(normalize(U'A') == U'Ａ');
    CHECK(normalize(U'0') == U'０');
    CHECK(normalize(U'(') == U'（');
    CHECK(normalize(U'あ') == U'あ');  // unchanged: not in the table
    CHECK(normalize(U'水') == U'水');
}

TEST_CASE("CharTable reproduces KyTea id assignment") {
    // A model's char map always begins with the six type markers, then the
    // training characters in first-occurrence order.
    CharTable table("KTHRDO /あ水"sv);
    CHECK(table.id_of(U'K') == 1);
    CHECK(table.id_of(U'T') == 2);
    CHECK(table.id_of(U'H') == 3);
    CHECK(table.id_of(U'R') == 4);
    CHECK(table.id_of(U'D') == 5);
    CHECK(table.id_of(U'O') == 6);
    CHECK(table.id_of(U' ') == 7);
    CHECK(table.id_of(U'/') == 8);
    CHECK(table.id_of(U'あ') == 9);
    CHECK(table.id_of(U'水') == 10);
    CHECK(table.size() == 10);

    CHECK(table.id_of(U'火') == kNoChar);  // never seen
}

TEST_CASE("CharTable::encode produces char and type id sequences") {
    CharTable table("KTHRDO あ水"sv);  // ' ' at id 7, 'あ' 8, '水' 9
    auto enc = table.encode("あ水"sv);
    REQUIRE(enc.has_value());
    REQUIRE(enc->length() == 2);

    CHECK(enc->char_ids[0] == table.id_of(U'あ'));
    CHECK(enc->char_ids[1] == table.id_of(U'水'));
    // Type markers: あ -> H, 水 -> K.
    CHECK(enc->type_ids[0] == table.id_of(U'H'));
    CHECK(enc->type_ids[1] == table.id_of(U'K'));

    // Byte offsets span the original UTF-8 (each character is 3 bytes).
    REQUIRE(enc->offsets.size() == 3);
    CHECK(enc->offsets[0] == 0);
    CHECK(enc->offsets[1] == 3);
    CHECK(enc->offsets[2] == 6);
}

TEST_CASE("CharTable::encode normalizes before interning") {
    // Train on the full-width forms; feed half-width input.
    CharTable table("KTHRDO ａ０"sv);
    auto enc = table.encode("a0"sv);
    REQUIRE(enc.has_value());
    // 'a' normalizes to 'ａ', '0' to '０', both of which are in the table.
    CHECK(enc->char_ids[0] == table.id_of(U'ａ'));
    CHECK(enc->char_ids[1] == table.id_of(U'０'));
    CHECK(enc->type_ids[0] == table.id_of(U'R'));
    CHECK(enc->type_ids[1] == table.id_of(U'D'));
}

TEST_CASE("CharTable::decode inverts id assignment back to UTF-8") {
    CharTable table("KTHRDO あ水"sv);  // 'あ' at id 7, '水' at id 8
    const std::vector<CharId> ids = {table.id_of(U'水'), table.id_of(U'あ')};
    CHECK(table.decode(ids) == "水あ");

    // The reserved empty-string id 0 contributes nothing.
    const std::vector<CharId> with_zero = {table.id_of(U'あ'), kNoChar, table.id_of(U'水')};
    CHECK(table.decode(with_zero) == "あ水");

    CHECK(table.decode(std::vector<CharId>{}).empty());
}

TEST_CASE("CharTable::encode reports invalid UTF-8") {
    CharTable table("KTHRDO"sv);
    auto enc = table.encode("\xE3\x81"sv);  // truncated
    REQUIRE_FALSE(enc.has_value());
    CHECK(enc.error().code == segmentlib::ErrorCode::InvalidUtf8);
}
