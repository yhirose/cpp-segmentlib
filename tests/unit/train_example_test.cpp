#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "mlp/train/corpus.h"
#include "mlp/train/example.h"
#include "segmentlib/mlp/vocab.h"

using namespace segmentlib;
using namespace segmentlib::mlp;
using namespace segmentlib::mlp::train;

namespace {

std::vector<AnnotatedSentence> full(std::string_view content) {
    auto parsed = parse_full_corpus(content);
    REQUIRE(parsed.has_value());
    return *parsed;
}

std::vector<AnnotatedSentence> partial(std::string_view content) {
    auto parsed = parse_partial_corpus(content);
    REQUIRE(parsed.has_value());
    return *parsed;
}

}  // namespace

TEST_CASE("build_vocab keeps codepoints at or above the frequency threshold") {
    const auto sentences = full("ああ い\nあ うう");
    const Vocab vocab = build_vocab(sentences, 2);
    CHECK(vocab.row_of(U'あ') != kUnkRow);  // 3 occurrences
    CHECK(vocab.row_of(U'う') != kUnkRow);  // 2 occurrences
    CHECK(vocab.row_of(U'い') == kUnkRow);  // 1 occurrence: below threshold
    CHECK(vocab.size() == 4);               // PAD + UNK + {あ, う}
}

TEST_CASE("build_examples maps codepoint gaps onto EGC boundaries") {
    const auto sentences = full("あい うえ");
    const Vocab vocab = build_vocab(sentences, 1);
    ExampleStats stats;
    const ExampleSet set = build_examples(sentences, vocab, {}, 5, &stats);

    CHECK(set.sentence_count() == 1);
    CHECK(set.sentence_egcs(0) == 4);
    REQUIRE(set.examples.size() == 3);
    CHECK(set.examples[0].boundary == 0);
    CHECK(set.examples[0].label == 0.0f);
    CHECK(set.examples[1].boundary == 1);
    CHECK(set.examples[1].label == 1.0f);
    CHECK(set.examples[2].boundary == 2);
    CHECK(set.examples[2].label == 0.0f);
    CHECK(stats.boundaries_labeled == 3);
    CHECK(stats.sentences_skipped_conflict == 0);

    // Every EGC here is a single codepoint mapped to a real row.
    for (std::uint32_t e = 0; e < 4; ++e) {
        REQUIRE(set.egc_rows(0, e).size() == 1);
        CHECK(set.egc_rows(0, e)[0] >= 2);
    }
}

TEST_CASE("a boundary inside an EGC skips the sentence (5.5)") {
    // か + combining dakuten (U+3099) form one EGC; `|` between them
    // contradicts the representation.
    const auto sentences = partial("か|゙");
    const Vocab vocab = build_vocab(sentences, 1);
    ExampleStats stats;
    const ExampleSet set = build_examples(sentences, vocab, {}, 5, &stats);

    CHECK(set.sentence_count() == 0);
    CHECK(set.examples.empty());
    CHECK(stats.sentences_skipped_conflict == 1);
}

TEST_CASE("NoBound inside an EGC is absorbed silently (5.5)") {
    const auto sentences = partial("か-゙|た");
    const Vocab vocab = build_vocab(sentences, 1);
    ExampleStats stats;
    const ExampleSet set = build_examples(sentences, vocab, {}, 5, &stats);

    CHECK(stats.sentences_skipped_conflict == 0);
    CHECK(set.sentence_count() == 1);
    CHECK(set.sentence_egcs(0) == 2);          // が (2 cps) + た
    CHECK(set.egc_rows(0, 0).size() == 2);     // the composed cluster
    REQUIRE(set.examples.size() == 1);
    CHECK(set.examples[0].boundary == 0);
    CHECK(set.examples[0].label == 1.0f);
}

TEST_CASE("Unknown gaps produce no examples") {
    const auto sentences = partial("あ い|う");
    const Vocab vocab = build_vocab(sentences, 1);
    ExampleStats stats;
    const ExampleSet set = build_examples(sentences, vocab, {}, 5, &stats);

    REQUIRE(set.examples.size() == 1);  // only い|う is supervised
    CHECK(set.examples[0].boundary == 1);
    CHECK(stats.boundaries_unknown == 1);
    CHECK(stats.boundaries_labeled == 1);
}

TEST_CASE("dictionary features: L/I/R × length bucket per dictionary") {
    const auto sentences = full("かき を");
    const Vocab vocab = build_vocab(sentences, 1);
    const std::vector<std::vector<std::string>> dicts = {{"かき"}, {"を"}};
    ExampleStats stats;
    const ExampleSet set = build_examples(sentences, vocab, dicts, 5, &stats);

    CHECK(set.num_dicts == 2);
    CHECK(set.dict_feature_count() == 24);
    REQUIRE(set.examples.size() == 2);

    // Boundary 0 (かき|を gap inside かき... no: gap か|き): the かき match
    // [0,1] spans it → Inside, bucket len2 → index 1*4+1 = 5.
    const auto feats = [&](std::size_t ex) {
        return std::vector<std::uint32_t>(
            set.feat_indices.begin() + set.feat_offsets[ex],
            set.feat_indices.begin() + set.feat_offsets[ex + 1]);
    };
    CHECK(feats(0) == std::vector<std::uint32_t>{5});
    // Boundary 1 (き|を): かき ends there → Right bucket2 = 2*4+1 = 9; and
    // dictionary 1's を starts there → Left bucket1 = 12 + 0*4+0 = 12.
    CHECK(feats(1) == std::vector<std::uint32_t>{9, 12});
}

TEST_CASE("dictionary matching keys on normalized EGCs, so UNK aliasing is safe") {
    // き is out-of-vocabulary (appears once, threshold 2), yet the dictionary
    // match must still fire: matching keys on codepoints, not embedding rows.
    const auto sentences = full("かか かき");
    const Vocab vocab = build_vocab(sentences, 2);
    REQUIRE(vocab.row_of(U'き') == kUnkRow);
    const std::vector<std::vector<std::string>> dicts = {{"かき"}};
    const ExampleSet set = build_examples(sentences, vocab, dicts, 5, nullptr);

    REQUIRE(set.examples.size() == 3);
    // Boundary 2 (か|き of the second word) is Inside the かき match.
    const std::uint32_t lo = set.feat_offsets[2];
    const std::uint32_t hi = set.feat_offsets[3];
    REQUIRE(hi - lo == 1);
    CHECK(set.feat_indices[lo] == 5);
}
