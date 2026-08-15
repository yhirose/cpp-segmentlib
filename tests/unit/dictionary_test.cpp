#include <doctest/doctest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "segmentlib/mlp/dictionary.h"
#include "segmentlib/mlp/vocab.h"

using namespace segmentlib;
using namespace segmentlib::mlp;

namespace {

// The feature index as dictionary.h defines it: dict*12 + position*4 +
// (min(EGC length, 4) - 1). Spelling it once keeps the expectations below
// about which relation and length fired, not about the arithmetic.
constexpr std::uint32_t feat(std::uint32_t dict, DictPosition position,
                             std::uint32_t length) {
    return dict * kDictFeaturesPerDict +
           static_cast<std::uint32_t>(position) * 4 + std::min(length, 4u) - 1;
}

// An empty vocabulary maps every codepoint to UNK, which is exactly what the
// matcher must not be affected by: it keys on the normalized codepoints, not
// on the embedding rows those codepoints alias to.
DictFeatures features_of(const DictMatcher& matcher, std::string_view text) {
    const Vocab vocab;
    EncodedEgc enc;
    REQUIRE(vocab.encode_into(text, enc).has_value());
    DictFeatures out;
    matcher.features_into(enc, out);
    return out;
}

// The features of boundary b, as a plain vector.
std::vector<std::uint32_t> at(const DictFeatures& f, std::size_t b) {
    return {f.indices.begin() + f.offsets[b], f.indices.begin() + f.offsets[b + 1]};
}

DictMatcher matcher_of(const std::vector<std::vector<std::string>>& dictionaries) {
    return DictMatcher(dictionaries);
}

}  // namespace

TEST_CASE("a one-cluster entry marks R at the boundary after it") {
    const auto matcher = matcher_of({{"か"}});
    // か 本 — two clusters, one boundary. か covers cluster 0, so it is a
    // right-side match at that boundary.
    const auto f = features_of(matcher, "か本");
    REQUIRE(f.offsets.size() == 2);
    CHECK(at(f, 0) == std::vector<std::uint32_t>{feat(0, DictPosition::Right, 1)});
}

TEST_CASE("an entry never matches part of a grapheme cluster") {
    const auto matcher = matcher_of({{"か"}});
    // か + U+3099 (combining dakuten) is a single cluster, so the entry か is
    // a proper byte prefix of it. Matching bytes alone would fire here; the
    // cluster-boundary check is what rejects it.
    const auto f = features_of(matcher, "が本");
    REQUIRE(f.offsets.size() == 2);
    CHECK(at(f, 0).empty());
    CHECK(f.indices.empty());
}

TEST_CASE("an entry spanning clusters marks I inside and R at its end") {
    const auto matcher = matcher_of({{"日本"}});
    // 日 本 語 — boundaries 0 and 1. 日本 covers clusters 0..1, so boundary 0
    // is inside the match and boundary 1 is its right edge.
    const auto f = features_of(matcher, "日本語");
    REQUIRE(f.offsets.size() == 3);
    CHECK(at(f, 0) == std::vector<std::uint32_t>{feat(0, DictPosition::Inside, 2)});
    CHECK(at(f, 1) == std::vector<std::uint32_t>{feat(0, DictPosition::Right, 2)});
}

TEST_CASE("entries of four clusters or more share the top length bucket") {
    const auto matcher = matcher_of({{"あいうえお"}});
    // Five clusters, so the length bucket saturates at min(5, 4) - 1.
    const auto f = features_of(matcher, "あいうえおX");
    REQUIRE(f.offsets.size() == 6);
    CHECK(at(f, 3) == std::vector<std::uint32_t>{feat(0, DictPosition::Inside, 5)});
    CHECK(at(f, 4) == std::vector<std::uint32_t>{feat(0, DictPosition::Right, 5)});
}

TEST_CASE("an entry present in several dictionaries fires for each channel") {
    const auto matcher = matcher_of({{"日本"}, {"日本"}});
    CHECK(matcher.num_dicts() == 2);
    CHECK(matcher.feature_count() == 24);
    const auto f = features_of(matcher, "日本語");
    REQUIRE(f.offsets.size() == 3);
    CHECK(at(f, 0) == std::vector<std::uint32_t>{feat(0, DictPosition::Inside, 2),
                                                 feat(1, DictPosition::Inside, 2)});
    CHECK(at(f, 1) == std::vector<std::uint32_t>{feat(0, DictPosition::Right, 2),
                                                 feat(1, DictPosition::Right, 2)});
}

TEST_CASE("the feature bitmap spans several words past five dictionaries") {
    // Twelve features per dictionary, so six of them need a second 64-bit
    // mask word and the last channel's features live in it.
    std::vector<std::vector<std::string>> dictionaries(6);
    dictionaries[0] = {"日本"};
    dictionaries[5] = {"日本"};
    const auto matcher = matcher_of(dictionaries);
    CHECK(matcher.feature_count() == 72);
    const auto f = features_of(matcher, "日本語");
    REQUIRE(f.offsets.size() == 3);
    CHECK(at(f, 0) == std::vector<std::uint32_t>{feat(0, DictPosition::Inside, 2),
                                                 feat(5, DictPosition::Inside, 2)});
    CHECK(at(f, 1) == std::vector<std::uint32_t>{feat(0, DictPosition::Right, 2),
                                                 feat(5, DictPosition::Right, 2)});
}

TEST_CASE("overlapping entries all report, deduplicated per boundary") {
    const auto matcher = matcher_of({{"日", "日本", "本語"}});
    const auto f = features_of(matcher, "日本語");
    REQUIRE(f.offsets.size() == 3);
    // Boundary 0: 本語 starts right after it, 日本 spans it, 日 ends here.
    CHECK(at(f, 0) == std::vector<std::uint32_t>{feat(0, DictPosition::Left, 2),
                                                 feat(0, DictPosition::Inside, 2),
                                                 feat(0, DictPosition::Right, 1)});
    // Boundary 1: 本語 spans it and 日本 ends here.
    CHECK(at(f, 1) == std::vector<std::uint32_t>{feat(0, DictPosition::Inside, 2),
                                                 feat(0, DictPosition::Right, 2)});
}

TEST_CASE("no dictionaries yields empty ranges") {
    const DictMatcher matcher;
    CHECK(matcher.num_dicts() == 0);
    const auto f = features_of(matcher, "日本語");
    REQUIRE(f.offsets.size() == 3);
    CHECK(f.indices.empty());
}

TEST_CASE("malformed and empty entries are dropped, not fatal") {
    const auto matcher = matcher_of({{"", "\xFF\xFE", "日本"}});
    const auto f = features_of(matcher, "日本語");
    REQUIRE(f.offsets.size() == 3);
    CHECK(at(f, 0) == std::vector<std::uint32_t>{feat(0, DictPosition::Inside, 2)});
}

TEST_CASE("features_into reuses the caller's buffers") {
    // The scratch is sized per sentence, so a short sentence after a long one
    // must not inherit the long one's masks or byte index.
    const auto matcher = matcher_of({{"日本"}});
    const Vocab vocab;
    EncodedEgc enc;
    DictFeatures out;
    REQUIRE(vocab.encode_into("あいうえおX", enc).has_value());
    matcher.features_into(enc, out);
    REQUIRE(vocab.encode_into("日本語", enc).has_value());
    matcher.features_into(enc, out);
    REQUIRE(out.offsets.size() == 3);
    CHECK(at(out, 0) == std::vector<std::uint32_t>{feat(0, DictPosition::Inside, 2)});
    CHECK(at(out, 1) == std::vector<std::uint32_t>{feat(0, DictPosition::Right, 2)});
}

TEST_CASE("a copied matcher matches the same as its source") {
    const auto matcher = matcher_of({{"日本"}});
    const DictMatcher copy = matcher;  // the pimpl holds a view of its bytes
    const auto a = features_of(matcher, "日本語");
    const auto b = features_of(copy, "日本語");
    CHECK(a.indices == b.indices);
    CHECK(a.offsets == b.offsets);
    CHECK(!b.indices.empty());
}
