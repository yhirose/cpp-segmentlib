#include <doctest/doctest.h>

#include <random>
#include <string>
#include <vector>

#include "mlp/train/compute_backend.h"
#include "mlp/train/corpus.h"
#include "mlp/train/example.h"
#include "mlp/train/exporter.h"
#include "mlp/train/quantize.h"
#include "mlp/train/trainer.h"
#include "segmentlib/mlp/mlp_backend.h"
#include "segmentlib/mlp/model.h"
#include "segmentlib/mlp/scorer.h"

// End-to-end train → quantize → export → load → infer
// (mlp_impl_design.ja.md III.4), plus the bit-exactness contract: the
// inference scorer must reproduce the trainer's int16 reference decisions
// exactly — the precompute table is a cache, not an approximation.

using namespace segmentlib;
using namespace segmentlib::mlp;
using namespace segmentlib::mlp::train;
using segmentlib::mlp::Vocab;

namespace {

std::string synthetic_corpus(int lines) {
    const std::vector<std::string> words = {"あい", "うえお", "かき", "くけこ",
                                            "さし", "すせそ"};
    std::mt19937 rng(9);
    std::uniform_int_distribution<std::size_t> pick(0, words.size() - 1);
    std::string corpus;
    for (int line = 0; line < lines; ++line) {
        for (int w = 0; w < 6; ++w) {
            if (w > 0) {
                corpus += ' ';
            }
            corpus += words[pick(rng)];
        }
        corpus += '\n';
    }
    return corpus;
}

}  // namespace

TEST_CASE("inference scorer is bit-exact with the trainer's int16 reference") {
    const auto sentences = parse_full_corpus(synthetic_corpus(30));
    REQUIRE(sentences.has_value());
    const Vocab vocab = build_vocab(*sentences);
    const std::vector<std::vector<std::string>> dicts = {{"かき", "うえお"}};
    const ExampleSet set = build_examples(*sentences, vocab, dicts, 2, nullptr);
    REQUIRE(set.sentence_count() == sentences->size());  // alignment below

    NetConfig config;
    config.window = 2;
    config.embed_dim = 8;
    config.hidden = 16;
    config.vocab_size = vocab.size();
    config.num_dicts = 1;

    // A briefly-trained model: weights are non-trivial but decisions still
    // sit near the margin, where any numeric mismatch would show.
    TrainOptions options;
    options.epochs = 3;
    options.batch_size = 64;
    const auto backend = make_cpu_backend();
    const TrainResult trained = mlp::train::train(config, set, nullptr, options, *backend);

    const QuantizedModel q = quantize(trained.params, set, *backend);
    const std::vector<std::byte> bytes = serialize_model(q, vocab, dicts);
    // Int32: the exact path, whose contract is bit-exactness with the
    // trainer's reference forward.
    const auto model = Model::load_from_bytes(bytes, TablePrecision::Int32);
    REQUIRE(model.has_value());

    std::size_t example = 0;
    for (const AnnotatedSentence& sentence : *sentences) {
        const auto enc = model->vocab().encode(sentence.text);
        REQUIRE(enc.has_value());
        const std::vector<std::int32_t> scores = score_boundaries(*model, *enc);
        REQUIRE(scores.size() == enc->egc_count() - 1);
        for (const std::int32_t score : scores) {
            CHECK((score > 0) == int16_decision(q, set, example));
            ++example;
        }
    }
    CHECK(example == set.examples.size());

    // The Int16 requantized path (the production default) may only diverge
    // at the |y|≈0 margin (5.6/I.3); on this deliberately margin-heavy,
    // briefly-trained model the flip rate must still be tiny.
    const auto model16 = Model::load_from_bytes(bytes);
    REQUIRE(model16.has_value());
    REQUIRE(model16->table_precision() == TablePrecision::Int16);
    std::size_t flips = 0;
    std::size_t total = 0;
    for (const AnnotatedSentence& sentence : *sentences) {
        const auto enc = model16->vocab().encode(sentence.text);
        REQUIRE(enc.has_value());
        const std::vector<std::int32_t> exact = score_boundaries(*model, *enc);
        const std::vector<std::int32_t> fast = score_boundaries(*model16, *enc);
        REQUIRE(exact.size() == fast.size());
        for (std::size_t b = 0; b < exact.size(); ++b) {
            flips += (exact[b] > 0) != (fast[b] > 0);
            ++total;
        }
    }
    CHECK(total == set.examples.size());
    CHECK(flips * 1000 <= total);  // ≤ 0.1%
}

TEST_CASE("end to end: train, export, load, tokenize") {
    const auto sentences = parse_full_corpus(synthetic_corpus(60));
    REQUIRE(sentences.has_value());
    const Vocab vocab = build_vocab(*sentences);
    const ExampleSet set = build_examples(*sentences, vocab, {}, 2, nullptr);

    NetConfig config;
    config.window = 2;
    config.embed_dim = 8;
    config.hidden = 16;
    config.vocab_size = vocab.size();

    TrainOptions options;
    options.epochs = 40;
    options.batch_size = 64;
    options.adam.lr = 3e-3f;
    options.patience = 10;

    const auto backend = make_cpu_backend();
    const TrainResult trained = mlp::train::train(config, set, &set, options, *backend);
    REQUIRE(trained.dev_metrics.f1 > 0.99);

    const QuantizedModel q = quantize(trained.params, set, *backend);
    auto model = Model::load_from_bytes(serialize_model(q, vocab, {}));
    REQUIRE(model.has_value());
    const MlpBackend mlp(std::move(*model));

    // かき|あい|うえお: word joints at bytes 6 and 12.
    const auto cuts = mlp.tokenize_boundaries("かきあいうえお");
    REQUIRE(cuts.has_value());
    CHECK(*cuts == Boundaries{6, 12});

    const auto segments = mlp.tokenize("かきあいうえお");
    REQUIRE(segments.has_value());
    REQUIRE(segments->size() == 3);
    CHECK((*segments)[1] == Segment{6, 12, {}});
}
