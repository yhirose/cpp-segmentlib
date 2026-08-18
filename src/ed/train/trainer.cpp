#include "ed/train/trainer.h"

#include <format>
#include <random>
#include <utility>

#include "mlp/train/dataset.h"

namespace segmentlib::ed::train {

TrainResult train(const NetConfig& config, const ExampleSet& train_set,
                  const ExampleSet* dev_set, const TrainOptions& options,
                  mlp::train::ComputeBackend& backend) {
    Parameters params = init_parameters(config, options.seed);
    mlp::train::Adam adam(config, options.adam);
    mlp::train::Net net(backend);
    Edla edla(backend, options.edla);
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
            adam.step(params, grads);
            // Reimpose the polarity the hidden-layer rule assumes: without
            // this, sign(w2[j]) drifts away from polarity(j) and the rule is
            // substituting a sign the network no longer has.
            project_dale(params);
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
                options.log(std::format(
                    "epoch {:3}  loss {:.6f}  dev P {:.4f} R {:.4f} F1 {:.4f}  "
                    "pinned {}/{}",
                    epoch + 1, mean_loss, dev.precision, dev.recall, dev.f1,
                    count_pinned(params), config.hidden));
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
