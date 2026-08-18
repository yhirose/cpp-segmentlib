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

#include "ed/train/trainer.h"
#include "mlp/train/compute_backend.h"
#include "mlp/train/corpus.h"
#include "mlp/train/example.h"
#include "mlp/train/exporter.h"
#include "mlp/train/quantize.h"
#include "mlp/train/trainer.h"
#include "segmentlib/ed/model.h"

namespace segmentlib::cli {

namespace {

using namespace mlp::train;
using mlp::Vocab;

// The CLI's hyperparameter defaults are the library's, read off the structs
// they are copied into below rather than restated as literals here: the two
// have already had to be edited in lockstep once, and a one-sided edit would
// compile and silently give the CLI different behaviour from the library.
const TrainOptions kTrainDefaults;
const AdamConfig kAdamDefaults;

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
    std::uint32_t epochs = kTrainDefaults.epochs;
    std::uint32_t batch_size = kTrainDefaults.batch_size;
    std::uint32_t patience = kTrainDefaults.patience;
    float lr = kAdamDefaults.lr;
    std::uint64_t seed = kTrainDefaults.seed;
    std::string_view optimizer = "adam";  // or "sgd" (the scale-preserving control)

    // EDLA only (--backend ed). Everything above is shared with the MLP
    // backend on purpose: holding the network and the training budget fixed is
    // what makes a comparison between the two learning rules mean anything.
    std::string_view ed_embedding_update;

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
        } else if (a == "--optimizer") {
            opt.optimizer = need_value(i);
        } else if (a == "--ed-embedding-update") {
            opt.ed_embedding_update = need_value(i);
        } else {
            std::println(stderr, "train: unexpected argument '{}'", a);
            opt.ok = false;
        }
    }
    if (opt.backend.empty()) {
        std::println(stderr, "train: --backend <kytea|vaporetto|mlp|ed> is required");
        opt.ok = false;
    }
    if (opt.optimizer != "adam" && opt.optimizer != "sgd") {
        std::println(stderr, "train: --optimizer must be adam or sgd");
        opt.ok = false;
    }
    if (!opt.ed_embedding_update.empty() && opt.ed_embedding_update != "hybrid" &&
        opt.ed_embedding_update != "pure") {
        std::println(stderr, "train: --ed-embedding-update must be hybrid or pure");
        opt.ok = false;
    }
    if (!opt.ed_embedding_update.empty() && opt.backend != "ed") {
        std::println(stderr, "train: --ed-embedding-update applies to --backend ed");
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

// Everything the two trainers share: the corpora, the vocabulary and the
// examples built from them, and the network shape. Both backends see byte-identical
// inputs, which is the point -- the learning rule is meant to be the only
// difference between the models they produce.
struct Prepared {
    std::vector<std::vector<std::string>> dictionaries;
    Vocab vocab;
    ExampleSet train_set;
    ExampleSet dev_set;
    bool have_dev = false;
    NetConfig config;
};

// Steps 1-2 of the pipeline. Returns false having already reported why.
bool prepare(const Options& opt, Prepared& out) {
    // 1. Corpora (full, then partial) and dictionaries.
    std::vector<AnnotatedSentence> sentences;
    for (const std::string_view path : opt.corpora) {
        auto parsed = read_full_corpus(std::filesystem::path(path));
        if (!parsed) {
            std::println(stderr, "train: {}: {}", path, parsed.error().message);
            return false;
        }
        sentences.insert(sentences.end(),
                         std::make_move_iterator(parsed->begin()),
                         std::make_move_iterator(parsed->end()));
    }
    for (const std::string_view path : opt.partial_corpora) {
        auto parsed = read_partial_corpus(std::filesystem::path(path));
        if (!parsed) {
            std::println(stderr, "train: {}: {}", path, parsed.error().message);
            return false;
        }
        sentences.insert(sentences.end(),
                         std::make_move_iterator(parsed->begin()),
                         std::make_move_iterator(parsed->end()));
    }
    for (const std::string_view path : opt.dicts) {
        auto words = read_dictionary(std::filesystem::path(path));
        if (!words) {
            std::println(stderr, "train: {}: {}", path, words.error().message);
            return false;
        }
        out.dictionaries.push_back(std::move(*words));
    }

    // 2. Vocabulary and examples.
    out.vocab = build_vocab(sentences, opt.min_count);
    ExampleStats stats;
    out.train_set =
        build_examples(sentences, out.vocab, out.dictionaries,
                       static_cast<std::uint8_t>(opt.char_window), &stats);
    std::println(stderr,
                 "train: {} sentences ({} skipped: {} EGC conflicts, {} invalid "
                 "UTF-8), {} examples, {} unsupervised gaps, vocabulary {}",
                 stats.sentences_read,
                 stats.sentences_skipped_conflict + stats.sentences_skipped_invalid,
                 stats.sentences_skipped_conflict, stats.sentences_skipped_invalid,
                 stats.boundaries_labeled, stats.boundaries_unknown, out.vocab.size());
    if (out.train_set.examples.empty()) {
        std::println(stderr, "train: no supervised boundaries in the corpora");
        return false;
    }

    out.have_dev = !opt.dev_corpus.empty();
    if (out.have_dev) {
        auto parsed = read_full_corpus(std::filesystem::path(opt.dev_corpus));
        if (!parsed) {
            std::println(stderr, "train: {}: {}", opt.dev_corpus, parsed.error().message);
            return false;
        }
        out.dev_set = build_examples(*parsed, out.vocab, out.dictionaries,
                                     static_cast<std::uint8_t>(opt.char_window),
                                     nullptr);
    }

    out.config.window = static_cast<std::uint8_t>(opt.char_window);
    out.config.embed_dim = static_cast<std::uint16_t>(opt.embed_dim);
    out.config.hidden = static_cast<std::uint16_t>(opt.hidden);
    out.config.vocab_size = out.vocab.size();
    out.config.num_dicts = static_cast<std::uint32_t>(out.dictionaries.size());
    return true;
}

// Steps 4-5: quantize the trained parameters, report how many decisions that
// cost, and write the model behind `header`.
int finish(const Options& opt, const Prepared& p, const Parameters& params,
           ComputeBackend& backend, std::string_view header) {
    const ExampleSet& calibration = p.have_dev ? p.dev_set : p.train_set;
    const QuantizedModel quantized = quantize(params, calibration, backend);
    const FlipCheck flips = check_sign_flips(params, quantized, calibration, backend);
    std::println(stderr,
                 "train: quantization flipped {}/{} decisions (max |y| {:.4f})",
                 flips.flips, flips.examples, flips.max_abs_y_flipped);

    const auto exported = export_model(std::filesystem::path(opt.model_out), quantized,
                                       p.vocab, p.dictionaries, header);
    if (!exported) {
        std::println(stderr, "train: {}: {}", opt.model_out, exported.error().message);
        return 1;
    }
    std::println(stderr, "train: wrote {}", opt.model_out);
    return 0;
}

void report_dev(bool have_dev, const EvalMetrics& dev) {
    if (have_dev) {
        std::println(stderr, "train: best dev F1 {:.4f} (P {:.4f} R {:.4f})", dev.f1,
                     dev.precision, dev.recall);
    }
}

int run_train_mlp(const Options& opt) {
    Prepared p;
    if (!prepare(opt, p)) {
        return 1;
    }

    TrainOptions options;
    options.epochs = opt.epochs;
    options.batch_size = opt.batch_size;
    options.adam.lr = opt.lr;
    options.optimizer =
        opt.optimizer == "sgd" ? Optimizer::Sgd : Optimizer::Adam;
    options.seed = opt.seed;
    options.patience = opt.patience;
    options.log = [](std::string_view s) { std::println(stderr, "train: {}", s); };

    const auto backend = make_cpu_backend();
    const TrainResult result = mlp::train::train(
        p.config, p.train_set, p.have_dev ? &p.dev_set : nullptr, options, *backend);
    report_dev(p.have_dev, result.dev_metrics);

    return finish(opt, p, result.params, *backend, mlp::train::kMlpHeader);
}

int run_train_ed(const Options& opt) {
    Prepared p;
    if (!prepare(opt, p)) {
        return 1;
    }

    ed::train::TrainOptions options;
    options.epochs = opt.epochs;
    options.batch_size = opt.batch_size;
    options.adam.lr = opt.lr;
    options.optimizer =
        opt.optimizer == "sgd" ? Optimizer::Sgd : Optimizer::Adam;
    options.seed = opt.seed;
    options.patience = opt.patience;
    options.edla.embedding_update = opt.ed_embedding_update == "pure"
                                        ? ed::train::EmbeddingUpdate::Pure
                                        : ed::train::EmbeddingUpdate::Hybrid;
    options.log = [](std::string_view s) { std::println(stderr, "train: {}", s); };

    const auto backend = make_cpu_backend();
    const ed::train::TrainResult result = ed::train::train(
        p.config, p.train_set, p.have_dev ? &p.dev_set : nullptr, options, *backend);
    report_dev(p.have_dev, result.dev_metrics);
    std::println(stderr, "train: {}/{} hidden units pinned by the polarity split",
                 result.pinned_units, p.config.hidden);

    return finish(opt, p, result.params, *backend,
                  std::string(ed::kModelSignature) + "1");
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
    if (opt.backend == "ed") {
        return run_train_ed(opt);
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
