#include <doctest/doctest.h>

#include <string_view>
#include <vector>

#include "mlp/train/corpus.h"

using namespace segmentlib;
using namespace segmentlib::mlp::train;

namespace {

constexpr BoundaryTag N = BoundaryTag::NoBound;
constexpr BoundaryTag B = BoundaryTag::Bound;
constexpr BoundaryTag U = BoundaryTag::Unknown;

}  // namespace

TEST_CASE("full corpus: words become Bound gaps, tags are stripped") {
    // The documented example line (design.ja.md 5.1).
    const auto result =
        parse_full_corpus("コーパス/ko:pasu の/no 文/buN で/de す/su 。/.");
    REQUIRE(result.has_value());
    REQUIRE(result->size() == 1);
    const AnnotatedSentence& s = (*result)[0];
    CHECK(s.text == "コーパスの文です。");
    CHECK(s.tags == std::vector<BoundaryTag>{N, N, N, B, B, B, B, B});
}

TEST_CASE("full corpus: escapes keep delimiter characters literal") {
    // "Hello World" with an escaped space, then "2024/12" with an escaped
    // slash; '&' outside a tag is literal.
    const auto result = parse_full_corpus("Hello\\ World 2024\\/12 a&b");
    REQUIRE(result.has_value());
    REQUIRE(result->size() == 1);
    const AnnotatedSentence& s = (*result)[0];
    CHECK(s.text == "Hello World2024/12a&b");
    // "Hello World" = 11 cps (10 internal gaps), then Bound, "2024/12" = 7
    // cps (6 internal), then Bound, "a&b" = 3 cps (2 internal).
    REQUIRE(s.tags.size() == 20);
    CHECK(s.tags[10] == B);
    CHECK(s.tags[17] == B);
    CHECK(s.tags[0] == N);
    CHECK(s.tags[18] == N);
}

TEST_CASE("full corpus: empty lines and repeated spaces are skipped") {
    const auto result = parse_full_corpus("\n  \nあ  い\n\n");
    REQUIRE(result.has_value());
    REQUIRE(result->size() == 1);
    CHECK((*result)[0].text == "あい");
    CHECK((*result)[0].tags == std::vector<BoundaryTag>{B});
}

TEST_CASE("partial corpus: the documented example line") {
    // design.ja.md 5.2.
    const auto result = parse_partial_corpus(
        "ヴ-ェ-ネ-ツ-ィ-ア|は|イ-タ-リ-ア|に|あ り ま す|。");
    REQUIRE(result.has_value());
    REQUIRE(result->size() == 1);
    const AnnotatedSentence& s = (*result)[0];
    CHECK(s.text == "ヴェネツィアはイタリアにあります。");
    CHECK(s.tags == std::vector<BoundaryTag>{N, N, N, N, N, B, B, N, N, N, B,
                                             B, U, U, U, B});
}

TEST_CASE("partial corpus: tags before a separator are stripped, `?` is unknown") {
    const auto result = parse_partial_corpus("み-ず/water|を?た");
    REQUIRE(result.has_value());
    REQUIRE(result->size() == 1);
    const AnnotatedSentence& s = (*result)[0];
    CHECK(s.text == "みずをた");
    CHECK(s.tags == std::vector<BoundaryTag>{N, B, U});
}

TEST_CASE("partial corpus: a single trailing separator is tolerated") {
    const auto result = parse_partial_corpus("あ|い|");
    REQUIRE(result.has_value());
    const AnnotatedSentence& s = (*result)[0];
    CHECK(s.text == "あい");
    CHECK(s.tags == std::vector<BoundaryTag>{B});
}

TEST_CASE("partial corpus: escaped separators are surface characters") {
    const auto result = parse_partial_corpus("あ|\\--い");
    REQUIRE(result.has_value());
    const AnnotatedSentence& s = (*result)[0];
    CHECK(s.text == "あ-い");
    CHECK(s.tags == std::vector<BoundaryTag>{B, N});
}

TEST_CASE("partial corpus: structural problems are MalformedCorpus") {
    CHECK(parse_partial_corpus("あい").error().code ==
          ErrorCode::MalformedCorpus);  // adjacent chars, no separator
    CHECK(parse_partial_corpus("|あ").error().code ==
          ErrorCode::MalformedCorpus);  // leading separator
    CHECK(parse_full_corpus("あ\\").error().code ==
          ErrorCode::MalformedCorpus);  // trailing escape
}

TEST_CASE("invalid UTF-8 is reported") {
    CHECK(parse_full_corpus("\xff\xfe").error().code == ErrorCode::InvalidUtf8);
    CHECK(parse_partial_corpus("\xff").error().code == ErrorCode::InvalidUtf8);
}
