#include "mlp/train/trainer.h"

#include <format>
#include <random>
#include <utility>

#include "mlp/train/dataset.h"

namespace segmentlib::mlp::train {

EvalMetrics evaluate(const Parameters& params, const ExampleSet& set,
                     ComputeBackend& backend, std::uint32_t batch_size) {
    Dataset data(set, params.config.embed_dim, batch_size);  // unshuffled
    Net net(backend);
    Batch batch;
    Workspace ws;

    std::size_t tp = 0, fp = 0, fn = 0, tn = 0;
    for (std::size_t i = 0; i < data.num_batches(); ++i) {
        data.fill_batch(i, params.embedding, batch);
        (void)net.forward(params, batch, ws);
        for (std::uint32_t b = 0; b < batch.size; ++b) {
            const bool predicted = ws.y[b] > 0.0f;  // 5.4: y > 0 ⇔ boundary
            const bool actual = batch.targets[b] > 0.5f;
            if (predicted && actual) ++tp;
            else if (predicted && !actual) ++fp;
            else if (!predicted && actual) ++fn;
            else ++tn;
        }
    }

    EvalMetrics m;
    m.examples = tp + fp + fn + tn;
    if (m.examples == 0) {
        return m;
    }
    m.accuracy = static_cast<double>(tp + tn) / static_cast<double>(m.examples);
    if (tp > 0) {
        m.precision = static_cast<double>(tp) / static_cast<double>(tp + fp);
        m.recall = static_cast<double>(tp) / static_cast<double>(tp + fn);
        m.f1 = 2.0 * m.precision * m.recall / (m.precision + m.recall);
    }
    return m;
}

TrainResult train(const NetConfig& config, const ExampleSet& train_set,
                  const ExampleSet* dev_set, const TrainOptions& options,
                  ComputeBackend& backend) {
    Parameters params = Parameters::init(config, options.seed);
    Adam adam(config, options.adam);
    Net net(backend);
    Dataset data(train_set, config.embed_dim, options.batch_size);
    std::mt19937 rng(static_cast<std::uint32_t>(options.seed));

    TrainResult result;
    Batch batch;
    Workspace ws;
    Gradients grads;

    double best_f1 = -1.0;
    std::uint32_t since_best = 0;

    for (std::uint32_t epoch = 0; epoch < options.epochs; ++epoch) {
        data.shuffle(rng);
        double loss_sum = 0.0;
        for (std::size_t i = 0; i < data.num_batches(); ++i) {
            data.fill_batch(i, params.embedding, batch);
            loss_sum += net.forward(params, batch, ws) * batch.size;
            net.backward(params, batch, ws, grads);
            adam.step(params, grads);
        }
        const double mean_loss =
            data.num_examples() > 0
                ? loss_sum / static_cast<double>(data.num_examples())
                : 0.0;
        result.epoch_losses.push_back(mean_loss);
        result.epochs_run = epoch + 1;

        if (dev_set != nullptr) {
            const EvalMetrics dev = evaluate(params, *dev_set, backend);
            if (options.log) {
                options.log(std::format(
                    "epoch {:3}  loss {:.6f}  dev P {:.4f} R {:.4f} F1 {:.4f}",
                    epoch + 1, mean_loss, dev.precision, dev.recall, dev.f1));
            }
            if (dev.f1 > best_f1) {
                best_f1 = dev.f1;
                since_best = 0;
                result.params = params;  // keep a copy of the best
                result.dev_metrics = dev;
            } else if (options.patience > 0 &&
                       ++since_best >= options.patience) {
                return result;  // early stop: best params already captured
            }
        } else if (options.log) {
            options.log(
                std::format("epoch {:3}  loss {:.6f}", epoch + 1, mean_loss));
        }
    }

    if (dev_set == nullptr) {
        result.params = std::move(params);
    }
    return result;
}

}  // namespace segmentlib::mlp::train
