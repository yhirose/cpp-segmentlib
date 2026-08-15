#include <print>
#include <span>
#include <string_view>

#include "commands.h"

#ifndef SEGMENTLIB_HAVE_TRAINING

namespace segmentlib::cli {

int run_train(std::span<const std::string_view> /*args*/) {
    std::println(stderr,
                 "train: this build has no training support; reconfigure with "
                 "-DSEGMENTLIB_BUILD_TRAINING=ON");
    return 1;
}

}  // namespace segmentlib::cli

#else

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <type_traits>
#include <iterator>
#include <string>
#include <vector>

#include "mlp/train/compute_backend.h"
#include "mlp/train/corpus.h"
#include "mlp/train/example.h"
#include "mlp/train/exporter.h"
#include "mlp/train/quantize.h"
#include "mlp/train/trainer.h"

namespace segmentlib::cli {

namespace {

using namespace mlp::train;
using mlp::Vocab;

struct Options {
    std::string_view backend;
    std::vector<std::string_view> corpora;
    std::vector<std::string_view> partial_corpora;
    std::vector<std::string_view> dicts;
    std::string_view dev_corpus;  // full annotation; optional
    std::string_view model_out;

    // MLP hyperparameters (design.ja.md 4.4 defaults).
    std::uint32_t char_window = 5;
    std::uint32_t embed_dim = 64;
    std::uint32_t hidden = 256;
    std::uint32_t min_count = 2;
    std::uint32_t epochs = 100;
    std::uint32_t batch_size = 256;
    std::uint32_t patience = 15;
    float lr = 1e-3f;
    std::uint64_t seed = 42;

    bool ok = true;
};

template <class T>
bool parse_number(std::string_view v, T& out) {
    if constexpr (std::is_floating_point_v<T>) {
        // Floating-point from_chars is availability-gated on Apple's libc++
        // (macOS 13.3+ deployment targets); strtof needs a NUL-terminated
        // copy but has no such gate.
        const std::string s(v);
        char* end = nullptr;
        out = std::strtof(s.c_str(), &end);
        return end == s.c_str() + s.size() && !s.empty();
    } else {
        const auto [ptr, ec] = std::from_chars(v.data(), v.data() + v.size(), out);
        return ec == std::errc{} && ptr == v.data() + v.size();
    }
}

Options parse(std::span<const std::string_view> args) {
    Options opt;
    const auto need_value = [&](std::size_t& i) -> std::string_view {
        if (i + 1 >= args.size()) {
            std::println(stderr, "train: {} requires a value", args[i]);
            opt.ok = false;
            return {};
        }
        return args[++i];
    };
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view a = args[i];
        const auto number = [&](auto& out) {
            const std::string_view v = need_value(i);
            if (opt.ok && !parse_number(v, out)) {
                std::println(stderr, "train: invalid value '{}' for {}", v, a);
                opt.ok = false;
            }
        };
        if (a == "--backend") {
            opt.backend = need_value(i);
        } else if (a == "--corpus") {
            opt.corpora.push_back(need_value(i));
        } else if (a == "--partial-corpus") {
            opt.partial_corpora.push_back(need_value(i));
        } else if (a == "--dict") {
            opt.dicts.push_back(need_value(i));
        } else if (a == "--dev-corpus") {
            opt.dev_corpus = need_value(i);
        } else if (a == "--model-out") {
            opt.model_out = need_value(i);
        } else if (a == "--char-window") {
            number(opt.char_window);
        } else if (a == "--embed-dim") {
            number(opt.embed_dim);
        } else if (a == "--hidden") {
            number(opt.hidden);
        } else if (a == "--min-count") {
            number(opt.min_count);
        } else if (a == "--epochs") {
            number(opt.epochs);
        } else if (a == "--batch-size") {
            number(opt.batch_size);
        } else if (a == "--patience") {
            number(opt.patience);
        } else if (a == "--lr") {
            number(opt.lr);
        } else if (a == "--seed") {
            number(opt.seed);
        } else {
            std::println(stderr, "train: unexpected argument '{}'", a);
            opt.ok = false;
        }
    }
    if (opt.backend.empty()) {
        std::println(stderr, "train: --backend <kytea|vaporetto|mlp> is required");
        opt.ok = false;
    }
    if (opt.model_out.empty()) {
        std::println(stderr, "train: --model-out <path> is required");
        opt.ok = false;
    }
    if (opt.corpora.empty() && opt.partial_corpora.empty()) {
        std::println(stderr, "train: at least one --corpus or --partial-corpus is required");
        opt.ok = false;
    }
    if (opt.char_window < 1 || opt.char_window > 255 || opt.embed_dim < 1 ||
        opt.embed_dim > 256 || opt.hidden < 1 || opt.hidden > 65535 ||
        opt.batch_size < 1) {
        std::println(stderr, "train: hyperparameter out of range");
        opt.ok = false;
    }
    if (opt.dicts.size() > 255) {
        std::println(stderr, "train: at most 255 dictionaries are supported");
        opt.ok = false;
    }
    return opt;
}

int run_train_mlp(const Options& opt) {
    // 1. Corpora (full, then partial) and dictionaries.
    std::vector<AnnotatedSentence> sentences;
    for (const std::string_view path : opt.corpora) {
        auto parsed = read_full_corpus(std::filesystem::path(path));
        if (!parsed) {
            std::println(stderr, "train: {}: {}", path, parsed.error().message);
            return 1;
        }
        sentences.insert(sentences.end(),
                         std::make_move_iterator(parsed->begin()),
                         std::make_move_iterator(parsed->end()));
    }
    for (const std::string_view path : opt.partial_corpora) {
        auto parsed = read_partial_corpus(std::filesystem::path(path));
        if (!parsed) {
            std::println(stderr, "train: {}: {}", path, parsed.error().message);
            return 1;
        }
        sentences.insert(sentences.end(),
                         std::make_move_iterator(parsed->begin()),
                         std::make_move_iterator(parsed->end()));
    }
    std::vector<std::vector<std::string>> dictionaries;
    for (const std::string_view path : opt.dicts) {
        auto words = read_dictionary(std::filesystem::path(path));
        if (!words) {
            std::println(stderr, "train: {}: {}", path, words.error().message);
            return 1;
        }
        dictionaries.push_back(std::move(*words));
    }

    // 2. Vocabulary and examples.
    const Vocab vocab = build_vocab(sentences, opt.min_count);
    ExampleStats stats;
    const ExampleSet train_set =
        build_examples(sentences, vocab, dictionaries,
                       static_cast<std::uint8_t>(opt.char_window), &stats);
    std::println(stderr,
                 "train: {} sentences ({} skipped: {} EGC conflicts, {} invalid "
                 "UTF-8), {} examples, {} unsupervised gaps, vocabulary {}",
                 stats.sentences_read,
                 stats.sentences_skipped_conflict + stats.sentences_skipped_invalid,
                 stats.sentences_skipped_conflict, stats.sentences_skipped_invalid,
                 stats.boundaries_labeled, stats.boundaries_unknown, vocab.size());
    if (train_set.examples.empty()) {
        std::println(stderr, "train: no supervised boundaries in the corpora");
        return 1;
    }

    ExampleSet dev_set;
    const bool have_dev = !opt.dev_corpus.empty();
    if (have_dev) {
        auto parsed = read_full_corpus(std::filesystem::path(opt.dev_corpus));
        if (!parsed) {
            std::println(stderr, "train: {}: {}", opt.dev_corpus, parsed.error().message);
            return 1;
        }
        dev_set = build_examples(*parsed, vocab, dictionaries,
                                 static_cast<std::uint8_t>(opt.char_window), nullptr);
    }

    // 3. Train.
    NetConfig config;
    config.window = static_cast<std::uint8_t>(opt.char_window);
    config.embed_dim = static_cast<std::uint16_t>(opt.embed_dim);
    config.hidden = static_cast<std::uint16_t>(opt.hidden);
    config.vocab_size = vocab.size();
    config.num_dicts = static_cast<std::uint32_t>(dictionaries.size());

    TrainOptions options;
    options.epochs = opt.epochs;
    options.batch_size = opt.batch_size;
    options.adam.lr = opt.lr;
    options.seed = opt.seed;
    options.patience = opt.patience;
    options.log = [](std::string_view s) { std::println(stderr, "train: {}", s); };

    const auto backend = make_cpu_backend();
    const TrainResult result = mlp::train::train(
        config, train_set, have_dev ? &dev_set : nullptr, options, *backend);
    if (have_dev) {
        std::println(stderr, "train: best dev F1 {:.4f} (P {:.4f} R {:.4f})",
                     result.dev_metrics.f1, result.dev_metrics.precision,
                     result.dev_metrics.recall);
    }

    // 4. Quantize (calibrating on dev when available, 5.5/II.6) and verify.
    const ExampleSet& calibration = have_dev ? dev_set : train_set;
    const QuantizedModel quantized =
        quantize(result.params, calibration, *backend);
    const FlipCheck flips =
        check_sign_flips(result.params, quantized, calibration, *backend);
    std::println(stderr,
                 "train: quantization flipped {}/{} decisions (max |y| {:.4f})",
                 flips.flips, flips.examples, flips.max_abs_y_flipped);

    // 5. Export.
    const auto exported = export_model(std::filesystem::path(opt.model_out),
                                       quantized, vocab, dictionaries);
    if (!exported) {
        std::println(stderr, "train: {}: {}", opt.model_out, exported.error().message);
        return 1;
    }
    std::println(stderr, "train: wrote {}", opt.model_out);
    return 0;
}

}  // namespace

int run_train(std::span<const std::string_view> args) {
    const Options opt = parse(args);
    if (!opt.ok) {
        return 2;
    }
    if (opt.backend == "mlp") {
        return run_train_mlp(opt);
    }
    if (opt.backend == "kytea" || opt.backend == "vaporetto") {
        std::println(stderr, "train: backend '{}' training is not implemented", opt.backend);
        return 1;
    }
    std::println(stderr, "train: unknown backend '{}'", opt.backend);
    return 2;
}

}  // namespace segmentlib::cli

#endif  // SEGMENTLIB_HAVE_TRAINING
