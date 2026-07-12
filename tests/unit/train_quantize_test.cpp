#include <doctest/doctest.h>

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "mlp/train/compute_backend.h"
#include "mlp/train/corpus.h"
#include "mlp/train/example.h"
#include "mlp/train/quantize.h"
#include "mlp/train/trainer.h"

using namespace segmentlib::mlp::train;
using segmentlib::mlp::Vocab;

namespace {

// Same tiny hand-built set as the net test: multi-codepoint EGCs, PAD slots
// and dictionary features all exercised.
ExampleSet tiny_set() {
    ExampleSet set;
    set.window = 1;
    set.num_dicts = 1;
    set.rows = {2, 3, 4, 2, 3};
    set.egc_starts = {0, 1, 3, 5};
    set.sentence_starts = {0, 3};
    set.examples = {{0, 0, 1.0f}, {0, 1, 0.0f}};
    set.feat_offsets = {0, 2, 3};
    set.feat_indices = {0, 5, 3};
    return set;
}

NetConfig tiny_config() {
    NetConfig config;
    config.window = 1;
    config.embed_dim = 3;
    config.hidden = 4;
    config.vocab_size = 5;
    config.num_dicts = 1;
    return config;
}

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

TEST_CASE("max-abs scales and bounded round-trip error") {
    const ExampleSet set = tiny_set();
    Parameters params = Parameters::init(tiny_config(), 99);
    const auto backend = make_cpu_backend();
    const QuantizedModel q = quantize(params, set, *backend);

    float emb_max = 0.0f;
    for (const float x : params.embedding) {
        emb_max = std::max(emb_max, std::abs(x));
    }
    CHECK(q.emb_scale == doctest::Approx(emb_max / 32767.0));
    CHECK(q.acc_scale > 0.0);

    // Round-trip: |S·q − w| ≤ S/2 everywhere (no clipping under max-abs).
    const auto check_roundtrip = [](std::span<const float> fp,
                                    std::span<const std::int16_t> qt,
                                    double scale) {
        REQUIRE(fp.size() == qt.size());
        for (std::size_t i = 0; i < fp.size(); ++i) {
            CHECK(std::abs(scale * qt[i] - fp[i]) <= scale * 0.5 + 1e-12);
        }
    };
    check_roundtrip(params.embedding, q.embedding, q.emb_scale);
    check_roundtrip(params.w1, q.w1, q.w1_scale);
    check_roundtrip(params.w2, q.w2, q.w2_scale);

    // W_dict is zero-initialized and untrained here: the guard scale keeps
    // divisions finite and the quantized tensor is all zeros.
    CHECK(q.wdict_scale == 1.0);
    for (const std::int16_t v : q.wdict) {
        CHECK(v == 0);
    }

    // Biases are carried as raw doubles (the loader requantizes them).
    REQUIRE(q.b1.size() == params.b1.size());
    for (std::size_t h = 0; h < q.b1.size(); ++h) {
        CHECK(q.b1[h] == doctest::Approx(params.b1[h]));
    }
}

TEST_CASE("int16 reference decisions match fp32 signs on a trained model") {
    const auto sentences = parse_full_corpus(synthetic_corpus());
    REQUIRE(sentences.has_value());
    const Vocab vocab = build_vocab(*sentences);
    const ExampleSet set = build_examples(*sentences, vocab, {}, 2, nullptr);

    NetConfig config;
    config.window = 2;
    config.embed_dim = 8;
    config.hidden = 16;
    config.vocab_size = vocab.size();

    TrainOptions options;
    options.epochs = 20;
    options.batch_size = 64;
    options.adam.lr = 3e-3f;
    options.patience = 0;

    const auto backend = make_cpu_backend();
    const TrainResult trained = train(config, set, &set, options, *backend);

    const QuantizedModel q = quantize(trained.params, set, *backend);
    const FlipCheck flips =
        check_sign_flips(trained.params, q, set, *backend);

    CHECK(flips.examples == set.examples.size());
    // int16 with a calibrated S_acc is near-lossless; on a confidently
    // trained model the decision should never flip (II.6: flips live at the
    // |y| ≈ 0 margin).
    CHECK(flips.flips == 0);
    // Headroom (I.2): the accumulator peak stays well under int32 —
    // pct99.99 maps to 2^22, and a whole window plus bias stays ≲ 2^26.
    CHECK(flips.max_abs_acc > 0);
    CHECK(flips.max_abs_acc < (std::int64_t{1} << 27));
}

TEST_CASE("int16 decision runs on an untrained model with dictionary features") {
    const ExampleSet set = tiny_set();
    Parameters params = Parameters::init(tiny_config(), 3);
    const auto backend = make_cpu_backend();
    const QuantizedModel q = quantize(params, set, *backend);
    std::int64_t max_acc = 0;
    (void)int16_decision(q, set, 0, &max_acc);
    (void)int16_decision(q, set, 1, &max_acc);
    CHECK(max_acc >= 0);  // ran through both examples without tripping
}
