#pragma once

#include <cstdint>
#include <functional>
#include <string_view>
#include <vector>

#include "mlp/train/adam.h"
#include "mlp/train/example.h"
#include "mlp/train/net.h"

// The training loop: epochs of shuffled
// minibatches, per-epoch dev evaluation on boundary F1, early stopping on a
// patience counter (design.ja.md 4.9).

namespace segmentlib::mlp::train {

struct TrainOptions {
    std::uint32_t epochs = 30;
    std::uint32_t batch_size = 256;
    AdamConfig adam{};
    std::uint64_t seed = 42;
    // Stop after this many epochs without a dev-F1 improvement (0 disables
    // early stopping; ignored when no dev set is given).
    std::uint32_t patience = 5;
    // Optional per-epoch progress sink (one preformatted line per call).
    std::function<void(std::string_view)> log;
};

// Boundary-classification metrics: Bound is the positive class.
struct EvalMetrics {
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
    double accuracy = 0.0;
    std::size_t examples = 0;
};

struct TrainResult {
    // Best parameters by dev F1 when a dev set was given, else the final ones.
    Parameters params;
    std::vector<double> epoch_losses;  // mean training loss per epoch run
    EvalMetrics dev_metrics;           // of the returned parameters (dev set)
    std::uint32_t epochs_run = 0;
};

// Evaluates y>0 classification of `params` over every example in `set`.
[[nodiscard]] EvalMetrics evaluate(const Parameters& params,
                                   const ExampleSet& set,
                                   ComputeBackend& backend,
                                   std::uint32_t batch_size = 512);

// Trains from random initialization. `dev_set` may be null (no evaluation,
// no early stopping). `config.vocab_size` must match the vocabulary used to
// build the example sets.
[[nodiscard]] TrainResult train(const NetConfig& config,
                                const ExampleSet& train_set,
                                const ExampleSet* dev_set,
                                const TrainOptions& options,
                                ComputeBackend& backend);

}  // namespace segmentlib::mlp::train
