#include <doctest/doctest.h>

#include <random>
#include <string>
#include <vector>

#include "mlp/train/compute_backend.h"
#include "mlp/train/corpus.h"
#include "mlp/train/example.h"
#include "mlp/train/trainer.h"

using namespace segmentlib::mlp::train;
using segmentlib::mlp::Vocab;

namespace {

// A synthetic segmentable language: each character appears in exactly one
// word at a fixed position, so word boundaries are fully determined by a
// 1-character window and a small MLP must reach F1 ≈ 1.
std::string synthetic_corpus() {
    const std::vector<std::string> words = {"あい", "うえお", "かき", "くけこ",
                                            "さし", "すせそ"};
    std::mt19937 rng(9);
    std::uniform_int_distribution<std::size_t> pick(0, words.size() - 1);
    std::string corpus;
    for (int line = 0; line < 60; ++line) {
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

TEST_CASE("training converges on a small synthetic corpus") {
    const auto sentences = parse_full_corpus(synthetic_corpus());
    REQUIRE(sentences.has_value());
    const Vocab vocab = build_vocab(*sentences);
    ExampleStats stats;
    const ExampleSet set = build_examples(*sentences, vocab, {}, 2, &stats);
    REQUIRE(stats.sentences_skipped_conflict == 0);
    REQUIRE(set.examples.size() > 500);

    NetConfig config;
    config.window = 2;
    config.embed_dim = 8;
    config.hidden = 16;
    config.vocab_size = vocab.size();
    config.num_dicts = 0;

    TrainOptions options;
    options.epochs = 40;
    options.batch_size = 64;
    options.adam.lr = 3e-3f;
    options.seed = 42;
    options.patience = 10;

    const auto backend = make_cpu_backend();
    const TrainResult result = train(config, set, &set, options, *backend);

    REQUIRE(result.epochs_run > 0);
    // The loss must actually decrease...
    CHECK(result.epoch_losses.back() < result.epoch_losses.front() * 0.5);
    // ...and the boundary classification must be essentially solved.
    CHECK(result.dev_metrics.f1 > 0.99);
    CHECK(result.dev_metrics.examples == set.examples.size());

    // The returned parameters reproduce the reported metrics.
    const EvalMetrics again = evaluate(result.params, set, *backend);
    CHECK(again.f1 == doctest::Approx(result.dev_metrics.f1));
}

TEST_CASE("training without a dev set returns the final parameters") {
    const auto sentences = parse_full_corpus("あい うえ\nうえ あい\nあい あい");
    REQUIRE(sentences.has_value());
    const Vocab vocab = build_vocab(*sentences, 1);
    const ExampleSet set = build_examples(*sentences, vocab, {}, 1, nullptr);

    NetConfig config;
    config.window = 1;
    config.embed_dim = 4;
    config.hidden = 8;
    config.vocab_size = vocab.size();

    TrainOptions options;
    options.epochs = 3;
    options.batch_size = 4;

    const auto backend = make_cpu_backend();
    const TrainResult result = train(config, set, nullptr, options, *backend);
    CHECK(result.epochs_run == 3);
    CHECK(result.epoch_losses.size() == 3);
    CHECK(!result.params.embedding.empty());
}
