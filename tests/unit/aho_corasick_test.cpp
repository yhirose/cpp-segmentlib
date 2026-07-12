#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "segmentlib/text/aho_corasick.h"

using namespace segmentlib::text;

namespace {

// Builds an automaton over char keys from string patterns; the payload is the
// pattern itself, so matches are easy to assert on.
AhoCorasick<char, std::string> build_from(const std::vector<std::string>& patterns) {
    AhoCorasickBuilder<char, std::string> builder;
    for (const auto& p : patterns) {
        builder.add(std::span<const char>(p.data(), p.size()), p);
    }
    return std::move(builder).build();
}

// Collects (start, pattern) pairs for readable assertions.
std::vector<std::pair<std::size_t, std::string>> find_all(
    const AhoCorasick<char, std::string>& ac, std::string_view text) {
    std::vector<std::pair<std::size_t, std::string>> out;
    for (const auto& m : ac.match_all(std::span<const char>(text.data(), text.size()))) {
        out.emplace_back(m.end_pos - m.length + 1, *m.payload);
    }
    return out;
}

}  // namespace

TEST_CASE("classic overlapping-pattern example") {
    const auto ac = build_from({"he", "she", "his", "hers"});
    CHECK(ac.num_patterns() == 4);

    // "ushers": she@1, he@2 (suffix of she, same end), hers@2.
    const auto matches = find_all(ac, "ushers");
    CHECK(matches == std::vector<std::pair<std::size_t, std::string>>{
                         {1, "she"}, {2, "he"}, {2, "hers"}});
}

TEST_CASE("matches at equal end positions come longest first") {
    const auto ac = build_from({"a", "ba", "cba"});
    const auto matches = find_all(ac, "xcba");
    CHECK(matches == std::vector<std::pair<std::size_t, std::string>>{
                         {1, "cba"}, {2, "ba"}, {3, "a"}});
}

TEST_CASE("a pattern that is a prefix of another is found on its own") {
    const auto ac = build_from({"ab", "abcd"});
    CHECK(find_all(ac, "abcx") ==
          std::vector<std::pair<std::size_t, std::string>>{{0, "ab"}});
    CHECK(find_all(ac, "abcd") ==
          std::vector<std::pair<std::size_t, std::string>>{{0, "ab"}, {0, "abcd"}});
}

TEST_CASE("repeated and self-overlapping occurrences are all reported") {
    const auto ac = build_from({"aa"});
    CHECK(find_all(ac, "aaaa") ==
          std::vector<std::pair<std::size_t, std::string>>{{0, "aa"}, {1, "aa"}, {2, "aa"}});
}

TEST_CASE("duplicate patterns report every payload") {
    AhoCorasickBuilder<char, int> builder;
    const std::string p = "ab";
    builder.add(std::span<const char>(p.data(), p.size()), 1);
    builder.add(std::span<const char>(p.data(), p.size()), 2);
    const auto ac = std::move(builder).build();

    std::vector<int> payloads;
    const std::string text = "ab";
    ac.match(std::span<const char>(text.data(), text.size()),
             [&](std::size_t, std::size_t, const int* v) { payloads.push_back(*v); });
    CHECK(payloads == std::vector<int>{1, 2});
}

TEST_CASE("empty automaton and empty patterns") {
    AhoCorasickBuilder<char, std::string> builder;
    builder.add(std::span<const char>{}, "empty");  // ignored
    const auto ac = std::move(builder).build();
    CHECK(ac.empty());
    CHECK(ac.num_states() == 0);
    const std::string text = "anything";
    CHECK(ac.match_all(std::span<const char>(text.data(), text.size())).empty());
}

TEST_CASE("works with a wide key type (interned EGC ids)") {
    // The MLP dictionary matcher keys the automaton by uint32 EGC ids.
    using EgcId = std::uint32_t;
    AhoCorasickBuilder<EgcId, std::uint8_t> builder;
    const std::vector<EgcId> word1{100, 200};       // "日本"
    const std::vector<EgcId> word2{100, 200, 300};  // "日本語"
    builder.add(word1, 0);
    builder.add(word2, 1);
    const auto ac = std::move(builder).build();

    const std::vector<EgcId> text{999, 100, 200, 300, 999};
    const auto matches = ac.match_all(text);
    REQUIRE(matches.size() == 2);
    CHECK(matches[0].end_pos == 2);
    CHECK(matches[0].length == 2);
    CHECK(*matches[0].payload == 0);
    CHECK(matches[1].end_pos == 3);
    CHECK(matches[1].length == 3);
    CHECK(*matches[1].payload == 1);
}

TEST_CASE("builder is reusable after build") {
    AhoCorasickBuilder<char, std::string> builder;
    const std::string p1 = "ab";
    builder.add(std::span<const char>(p1.data(), p1.size()), p1);
    const auto first = std::move(builder).build();
    CHECK(first.num_patterns() == 1);

    const std::string p2 = "cd";
    builder.add(std::span<const char>(p2.data(), p2.size()), p2);
    const auto second = std::move(builder).build();
    CHECK(second.num_patterns() == 1);
    CHECK(find_all(second, "abcd") ==
          std::vector<std::pair<std::size_t, std::string>>{{2, "cd"}});
}
