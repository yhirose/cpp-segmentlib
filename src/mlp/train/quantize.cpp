#include "mlp/train/quantize.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "mlp/train/dataset.h"

namespace segmentlib::mlp::train {

namespace {

constexpr double kAmax = 4194304.0;  // 2^22 (I.1)
constexpr std::int32_t kQmax = 32767;

// max-abs scale (II.6): no clipping; a zero tensor gets scale 1.0 (the
// quantized values are all zero either way, but downstream divisions stay
// finite).
double max_abs_scale(std::span<const float> tensor) {
    float max_abs = 0.0f;
    for (const float x : tensor) {
        max_abs = std::max(max_abs, std::abs(x));
    }
    return max_abs > 0.0f ? static_cast<double>(max_abs) / kQmax : 1.0;
}

std::vector<std::int16_t> quantize_tensor(std::span<const float> tensor,
                                          double scale) {
    std::vector<std::int16_t> q(tensor.size());
    for (std::size_t i = 0; i < tensor.size(); ++i) {
        const long v = std::lround(tensor[i] / scale);
        q[i] = static_cast<std::int16_t>(std::clamp<long>(v, -kQmax, kQmax));
    }
    return q;
}

// The load-time derived values of I.1, shared by every decision.
struct Int16Reference {
    const QuantizedModel& q;
    double r;  // R = S_e·S_w1 / S_acc
    std::vector<std::int32_t> b1_q;
    std::int64_t b2_q;

    explicit Int16Reference(const QuantizedModel& quantized) : q(quantized) {
        r = q.emb_scale * q.w1_scale / q.acc_scale;
        b1_q.resize(q.b1.size());
        for (std::size_t h = 0; h < q.b1.size(); ++h) {
            b1_q[h] = static_cast<std::int32_t>(
                std::llround(q.b1[h] / q.acc_scale));  // I.1-(4)
        }
        b2_q = std::llround(q.b2 / (q.w2_scale * q.acc_scale));  // I.1-(5)
    }

    [[nodiscard]] bool decision(const ExampleSet& set, std::size_t example,
                                std::int64_t* max_abs_acc) const {
        const std::size_t d = q.config.embed_dim;
        const std::size_t h_dim = q.config.hidden;
        const std::size_t in = q.config.input_dim();
        const std::size_t slots = 2u * q.config.window;
        const ExampleSet::Example& ex = set.examples[example];
        const std::uint32_t m = set.sentence_egcs(ex.sentence);

        std::vector<std::int32_t> acc(b1_q);
        std::vector<std::int32_t> sum_c(d);

        // First layer: fallback synthesis I.1-(2) for every slot (the
        // frequent-EGC table of 5.6 is a cache of exactly this computation).
        for (std::size_t j = 0; j < slots; ++j) {
            const std::int64_t egc =
                static_cast<std::int64_t>(ex.boundary) - q.config.window + 1 +
                static_cast<std::int64_t>(j);
            static constexpr std::uint32_t pad[] = {kPadRow};
            std::span<const std::uint32_t> rows = pad;
            if (egc >= 0 && egc < m) {
                rows = set.egc_rows(ex.sentence, static_cast<std::uint32_t>(egc));
            }
            const auto n = static_cast<double>(rows.size());
            for (std::size_t k = 0; k < d; ++k) {
                std::int32_t s = 0;
                for (const std::uint32_t row : rows) {
                    s += q.embedding[static_cast<std::size_t>(row) * d + k];
                }
                sum_c[k] = s;
            }
            for (std::size_t h = 0; h < h_dim; ++h) {
                std::int64_t raw = 0;
                const std::int16_t* w1_row = q.w1.data() + h * in + j * d;
                for (std::size_t k = 0; k < d; ++k) {
                    raw += static_cast<std::int64_t>(w1_row[k]) * sum_c[k];
                }
                acc[h] += static_cast<std::int32_t>(
                    std::llround(static_cast<double>(raw) * r / n));
            }
        }

        // Dictionary columns, converted to the accumulator scale (I.1-(3)).
        const std::size_t fd = q.config.dict_features();
        const double dict_r = q.wdict_scale / q.acc_scale;
        for (std::uint32_t o = set.feat_offsets[example];
             o < set.feat_offsets[example + 1]; ++o) {
            const std::uint32_t k = set.feat_indices[o];
            for (std::size_t h = 0; h < h_dim; ++h) {
                acc[h] += static_cast<std::int32_t>(
                    std::llround(dict_r * q.wdict[h * fd + k]));
            }
        }

        if (max_abs_acc != nullptr) {
            for (const std::int32_t a : acc) {
                *max_abs_acc = std::max<std::int64_t>(
                    *max_abs_acc, std::abs(static_cast<std::int64_t>(a)));
            }
        }

        // ReLU, output dot product in int64, sign decision (I.2).
        std::int64_t sum = b2_q;
        for (std::size_t h = 0; h < h_dim; ++h) {
            const std::int32_t a = std::max(acc[h], 0);
            sum += static_cast<std::int64_t>(q.w2[h]) * a;
        }
        return sum > 0;
    }
};

}  // namespace

QuantizedModel quantize(const Parameters& params, const ExampleSet& calibration,
                        ComputeBackend& backend, std::uint32_t batch_size) {
    QuantizedModel q;
    q.config = params.config;

    q.emb_scale = max_abs_scale(params.embedding);
    q.w1_scale = max_abs_scale(params.w1);
    q.wdict_scale = max_abs_scale(params.wdict);
    q.w2_scale = max_abs_scale(params.w2);
    q.embedding = quantize_tensor(params.embedding, q.emb_scale);
    q.w1 = quantize_tensor(params.w1, q.w1_scale);
    q.wdict = quantize_tensor(params.wdict, q.wdict_scale);
    q.w2 = quantize_tensor(params.w2, q.w2_scale);

    q.b1.assign(params.b1.begin(), params.b1.end());
    q.b2 = params.b2;

    // S_acc calibration (II.6): collect the first-layer pre-activations over
    // the calibration set and take the 99.99th percentile of |a|.
    std::vector<float> abs_activations;
    {
        Dataset data(calibration, params.config.embed_dim, batch_size);
        Net net(backend);
        Batch batch;
        Workspace ws;
        for (std::size_t i = 0; i < data.num_batches(); ++i) {
            data.fill_batch(i, params.embedding, batch);
            (void)net.forward(params, batch, ws);
            const std::size_t n =
                static_cast<std::size_t>(batch.size) * params.config.hidden;
            for (std::size_t k = 0; k < n; ++k) {
                abs_activations.push_back(std::abs(ws.a[k]));
            }
        }
    }
    double pct = 0.0;
    if (!abs_activations.empty()) {
        const auto idx = static_cast<std::size_t>(
            0.9999 * static_cast<double>(abs_activations.size() - 1));
        std::nth_element(abs_activations.begin(), abs_activations.begin() + idx,
                         abs_activations.end());
        pct = abs_activations[idx];
    }
    if (pct <= 0.0) {
        pct = 1.0;  // degenerate calibration (empty set / dead layer)
    }
    q.acc_scale = pct / kAmax;
    return q;
}

bool int16_decision(const QuantizedModel& quantized, const ExampleSet& set,
                    std::size_t example, std::int64_t* max_abs_acc) {
    return Int16Reference(quantized).decision(set, example, max_abs_acc);
}

FlipCheck check_sign_flips(const Parameters& params,
                           const QuantizedModel& quantized,
                           const ExampleSet& set, ComputeBackend& backend,
                           std::uint32_t batch_size) {
    FlipCheck result;
    const Int16Reference ref(quantized);
    Dataset data(set, params.config.embed_dim, batch_size);  // unshuffled
    Net net(backend);
    Batch batch;
    Workspace ws;
    for (std::size_t i = 0; i < data.num_batches(); ++i) {
        data.fill_batch(i, params.embedding, batch);
        (void)net.forward(params, batch, ws);
        for (std::uint32_t b = 0; b < batch.size; ++b) {
            const std::size_t example =
                i * static_cast<std::size_t>(batch_size) + b;
            const bool fp32 = ws.y[b] > 0.0f;
            const bool int16 =
                ref.decision(set, example, &result.max_abs_acc);
            ++result.examples;
            if (fp32 != int16) {
                ++result.flips;
                result.max_abs_y_flipped =
                    std::max(result.max_abs_y_flipped,
                             static_cast<double>(std::abs(ws.y[b])));
            }
        }
    }
    return result;
}

}  // namespace segmentlib::mlp::train
