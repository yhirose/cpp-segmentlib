#include "ed/train/trainer.h"

#include <cmath>
#include <format>
#include <optional>
#include <random>
#include <string>
#include <utility>

#include "mlp/train/dataset.h"
#include "mlp/train/sgd.h"

namespace segmentlib::ed::train {

TrainResult train(const NetConfig& config, const ExampleSet& train_set,
                  const ExampleSet* dev_set, const TrainOptions& options,
                  mlp::train::ComputeBackend& backend) {
    Parameters params = init_parameters(config, options.seed);
    std::optional<mlp::train::Adam> adam;
    std::optional<mlp::train::Sgd> sgd;
    if (options.optimizer == Optimizer::Adam) {
        adam.emplace(config, options.adam);
    } else {
        sgd.emplace(options.adam.lr);
    }
    const auto step = [&](Parameters& p, const Gradients& g) {
        adam ? adam->step(p, g) : sgd->step(p, g);
    };
    mlp::train::Net net(backend);
    Edla edla(backend, options.edla, config);
    std::vector<float> w2_before;
    mlp::train::Dataset data(train_set, config.embed_dim, options.batch_size);
    std::mt19937 rng(static_cast<std::uint32_t>(options.seed));

    TrainResult result;
    Batch batch;
    Workspace ws;
    Gradients grads;

    double best_f1 = -1.0;
    std::uint32_t since_best = 0;

    const auto capture = [&](const EvalMetrics& dev) {
        result.params = params;
        result.dev_metrics = dev;
        result.pinned_units = count_pinned(params);
    };

    for (std::uint32_t epoch = 0; epoch < options.epochs; ++epoch) {
        data.shuffle(rng);
        double loss_sum = 0.0;
        for (std::size_t i = 0; i < data.num_batches(); ++i) {
            data.fill_batch(i, params.embedding, batch);
            loss_sum += net.forward(params, batch, ws) * batch.size;
            edla.local_update(params, batch, ws, grads);
            if (options.edla.learn_feedback) {
                w2_before = params.w2;
            }
            step(params, grads);
            // Reimpose the polarity the hidden-layer rule assumes: without
            // this, sign(w2[j]) drifts away from polarity(j) and the rule is
            // substituting a sign the network no longer has.
            project_dale(params);
            edla.track_feedback(params, w2_before);
        }
        const double mean_loss =
            data.num_examples() > 0
                ? loss_sum / static_cast<double>(data.num_examples())
                : 0.0;
        result.epoch_losses.push_back(mean_loss);
        result.epochs_run = epoch + 1;

        if (dev_set != nullptr) {
            const EvalMetrics dev = mlp::train::evaluate(params, *dev_set, backend);
            if (options.log) {
                // Under Kolen-Pollack, report how far the feedback has
                // converged on w2 -- that ratio is the whole claim of the
                // method, so it is measured rather than assumed.
                std::string feedback_note;
                if (options.edla.learn_feedback) {
                    double num = 0.0, den = 0.0;
                    for (std::size_t j = 0; j < params.w2.size(); ++j) {
                        const double diff = edla.feedback()[j] - params.w2[j];
                        num += diff * diff;
                        den += static_cast<double>(params.w2[j]) * params.w2[j];
                    }
                    feedback_note = std::format(
                        "  |b-w2|/|w2| {:.3f}", den > 0 ? std::sqrt(num / den) : 0.0);
                }
                options.log(std::format(
                    "epoch {:3}  loss {:.6f}  dev P {:.4f} R {:.4f} F1 {:.4f}  "
                    "pinned {}/{}{}",
                    epoch + 1, mean_loss, dev.precision, dev.recall, dev.f1,
                    count_pinned(params), config.hidden, feedback_note));
            }
            if (dev.f1 > best_f1) {
                best_f1 = dev.f1;
                since_best = 0;
                capture(dev);
            } else if (options.patience > 0 && ++since_best >= options.patience) {
                return result;  // early stop: best params already captured
            }
        } else if (options.log) {
            options.log(std::format("epoch {:3}  loss {:.6f}  pinned {}/{}", epoch + 1,
                                    mean_loss, count_pinned(params), config.hidden));
        }
    }

    if (dev_set == nullptr) {
        result.pinned_units = count_pinned(params);
        result.params = std::move(params);
    }
    return result;
}

}  // namespace segmentlib::ed::train
