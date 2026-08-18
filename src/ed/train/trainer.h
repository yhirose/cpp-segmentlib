#pragma once

#include <cstdint>
#include <functional>
#include <string_view>

#include "ed/train/edla.h"
#include "mlp/train/adam.h"
#include "mlp/train/example.h"
#include "mlp/train/trainer.h"

// The EDLA training loop: epochs of shuffled minibatches, per-epoch dev
// evaluation on boundary F1, early stopping on a patience counter -- the same
// shape as mlp::train::train, with the backward pass replaced by
// Edla::local_update and Dale's law reimposed after each step.

namespace segmentlib::ed::train {

using mlp::train::AdamConfig;
using mlp::train::EvalMetrics;
using mlp::train::ExampleSet;

struct TrainOptions {
    std::uint32_t epochs = 100;
    std::uint32_t batch_size = 256;
    AdamConfig adam{};
    EdlaConfig edla{};
    std::uint64_t seed = 42;
    // Stop after this many epochs without a dev-F1 improvement (0 disables it;
    // ignored when no dev set is given). Matches the MLP default so that a
    // comparison between the two backends holds the training budget fixed.
    std::uint32_t patience = 15;
    std::function<void(std::string_view)> log;
};

struct TrainResult {
    Parameters params;                 // best by dev F1 when a dev set was given
    std::vector<double> epoch_losses;  // mean training loss per epoch run
    EvalMetrics dev_metrics;
    std::uint32_t epochs_run = 0;
    // Hidden units the Dale projection has pinned to w2 == 0 in the returned
    // parameters: capacity the polarity split cost, reported rather than
    // buried.
    std::size_t pinned_units = 0;
};

// Trains from random initialization with the EDLA rule. `dev_set` may be null.
// Evaluation reuses mlp::train::evaluate: the network being scored is the same.
[[nodiscard]] TrainResult train(const NetConfig& config, const ExampleSet& train_set,
                                const ExampleSet* dev_set,
                                const TrainOptions& options,
                                mlp::train::ComputeBackend& backend);

}  // namespace segmentlib::ed::train
