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

// Which optimizer consumes the per-batch update. Adam is the production
// choice; Sgd exists as the control that preserves gradient scale, which the
// EDLA comparison needs (sgd.h explains why).
enum class Optimizer : std::uint8_t { Adam, Sgd };

struct TrainOptions {
    std::uint32_t epochs = 100;
    std::uint32_t batch_size = 256;
    // adam.lr doubles as plain SGD's learning rate when optimizer is Sgd
    // (the CLI's --lr feeds both; only one is ever in use).
    AdamConfig adam{};
    Optimizer optimizer = Optimizer::Adam;
    std::uint64_t seed = 42;
    // Stop after this many epochs without a dev-F1 improvement (0 disables
    // early stopping; ignored when no dev set is given). Dev F1 keeps creeping
    // up across plateaus of a dozen epochs, so a small patience stops on a
    // plateau rather than at convergence: 5 cost 0.11pt of GSD test F1 against
    // 15, at no inference cost either way (design.md 4.8).
    std::uint32_t patience = 15;
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
