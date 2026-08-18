#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "ed/train/trainer.h"
#include "mlp/train/compute_backend.h"
#include "mlp/train/corpus.h"
#include "mlp/train/example.h"
#include "mlp/train/exporter.h"
#include "mlp/train/quantize.h"
#include "segmentlib/ed/ed_backend.h"
#include "segmentlib/ed/model.h"
#include "segmentlib/mlp/scorer.h"
#include "segmentlib/segmenter.h"

// The EDLA counterpart of train_inference_test.cpp: train → quantize → export
// → load → infer, including the bit-exactness contract between the inference
// scorer and the trainer's own int16 reference.

using namespace segmentlib;
using namespace segmentlib::mlp::train;
using segmentlib::mlp::Vocab;
namespace edt = segmentlib::ed::train;

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

const std::string kEdHeader = std::string(ed::kModelSignature) + "1";

}  // namespace

TEST_CASE("EDLA inference is bit-exact with the trainer's int16 reference") {
    const auto sentences = parse_full_corpus(synthetic_corpus(30));
    REQUIRE(sentences.has_value());
    const Vocab vocab = build_vocab(*sentences);
    const std::vector<std::vector<std::string>> dicts = {{"かき", "うえお"}};
    const ExampleSet set = build_examples(*sentences, vocab, dicts, 2, nullptr);
    REQUIRE(set.sentence_count() == sentences->size());

    NetConfig config;
    config.window = 2;
    config.embed_dim = 8;
    config.hidden = 16;
    config.vocab_size = vocab.size();
    config.num_dicts = 1;

    edt::TrainOptions options;
    options.epochs = 3;
    options.batch_size = 64;
    const auto backend = make_cpu_backend();
    const edt::TrainResult trained = edt::train(config, set, nullptr, options, *backend);

    // Quantization and export are the MLP backend's, unchanged: an EDLA model
    // records the same tensors, so only the header line differs.
    const QuantizedModel q = quantize(trained.params, set, *backend);
    const std::vector<std::byte> bytes = serialize_model(q, vocab, dicts, kEdHeader);
    const auto model = ed::Model::load_from_bytes(bytes, mlp::TablePrecision::Int32);
    REQUIRE(model.has_value());

    std::size_t example = 0;
    for (const AnnotatedSentence& sentence : *sentences) {
        const auto enc = model->net().vocab().encode(sentence.text);
        REQUIRE(enc.has_value());
        const std::vector<std::int32_t> scores = score_boundaries(model->net(), *enc);
        REQUIRE(scores.size() == enc->egc_count() - 1);
        for (const std::int32_t score : scores) {
            CHECK((score > 0) == int16_decision(q, set, example));
            ++example;
        }
    }
    CHECK(example == set.examples.size());
}

TEST_CASE("end to end: train with EDLA, export, load, tokenize") {
    const auto sentences = parse_full_corpus(synthetic_corpus(60));
    REQUIRE(sentences.has_value());
    const Vocab vocab = build_vocab(*sentences);
    const ExampleSet set = build_examples(*sentences, vocab, {}, 2, nullptr);

    NetConfig config;
    config.window = 2;
    config.embed_dim = 8;
    config.hidden = 16;
    config.vocab_size = vocab.size();

    edt::TrainOptions options;
    options.epochs = 60;
    options.batch_size = 64;
    options.adam.lr = 3e-3f;
    options.patience = 15;

    const auto backend = make_cpu_backend();
    const edt::TrainResult trained = edt::train(config, set, &set, options, *backend);
    // A six-word closed vocabulary is separable; EDLA is expected to learn it,
    // which is what makes this a test of the rule rather than of the plumbing.
    CHECK(trained.dev_metrics.f1 > 0.95);

    const QuantizedModel q = quantize(trained.params, set, *backend);
    auto model = ed::Model::load_from_bytes(serialize_model(q, vocab, {}, kEdHeader));
    REQUIRE(model.has_value());
    const ed::EdBackend backend_ed(std::move(*model));

    // かき|あい|うえお: word joints at bytes 6 and 12.
    const auto segments = backend_ed.tokenize("かきあいうえお");
    REQUIRE(segments.has_value());
    REQUIRE(segments->size() == 3);
    CHECK((*segments)[1] == std::pair<std::size_t, std::size_t>{6, 12});
}

TEST_CASE("the EDLA trainer honours epochs, patience and the polarity split") {
    const auto sentences = parse_full_corpus(synthetic_corpus(20));
    REQUIRE(sentences.has_value());
    const Vocab vocab = build_vocab(*sentences);
    const ExampleSet set = build_examples(*sentences, vocab, {}, 2, nullptr);

    NetConfig config;
    config.window = 2;
    config.embed_dim = 4;
    config.hidden = 8;
    config.vocab_size = vocab.size();

    const auto backend = make_cpu_backend();

    edt::TrainOptions options;
    options.epochs = 5;
    options.batch_size = 32;
    options.patience = 0;  // disabled: run every epoch
    const edt::TrainResult full = edt::train(config, set, &set, options, *backend);
    CHECK(full.epochs_run == 5);
    CHECK(full.epoch_losses.size() == 5);
    CHECK(full.pinned_units <= config.hidden);

    // Patience 1 cannot outlast the full run, and stops as soon as dev F1
    // fails to improve.
    options.epochs = 100;
    options.patience = 1;
    const edt::TrainResult stopped = edt::train(config, set, &set, options, *backend);
    CHECK(stopped.epochs_run < 100);
    CHECK(stopped.dev_metrics.f1 > 0.0);

    // Same seed, same data, same result: the update touches a hash map keyed by
    // embedding row, so a nondeterministic iteration order would show here.
    const edt::TrainResult again = edt::train(config, set, &set, options, *backend);
    CHECK(again.epochs_run == stopped.epochs_run);
    CHECK(again.dev_metrics.f1 == stopped.dev_metrics.f1);
    CHECK(again.params.w2 == stopped.params.w2);
}

TEST_CASE("an EDLA model round-trips through Segmenter::load") {
    const auto sentences = parse_full_corpus(synthetic_corpus(20));
    REQUIRE(sentences.has_value());
    const Vocab vocab = build_vocab(*sentences);
    const ExampleSet set = build_examples(*sentences, vocab, {}, 2, nullptr);

    NetConfig config;
    config.window = 2;
    config.embed_dim = 4;
    config.hidden = 8;
    config.vocab_size = vocab.size();

    edt::TrainOptions options;
    options.epochs = 5;
    options.batch_size = 32;
    const auto backend = make_cpu_backend();
    const edt::TrainResult trained = edt::train(config, set, nullptr, options, *backend);
    const QuantizedModel q = quantize(trained.params, set, *backend);

    const std::filesystem::path path = "segmentlib_ed_roundtrip_test.mod";
    const auto written = export_model(path, q, vocab, {}, kEdHeader);
    REQUIRE(written.has_value());

    const auto segmenter = Segmenter::load(path);
    REQUIRE(segmenter.has_value());
    CHECK(segmenter->tokenize("かきあい").has_value());
    // The file says which rule trained it, and the MLP loader says so too.
    CHECK(!Segmenter::load_mlp(path).has_value());
    CHECK(Segmenter::load_ed(path).has_value());
    std::filesystem::remove(path);
}
