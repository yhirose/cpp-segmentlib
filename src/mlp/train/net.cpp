#include "mlp/train/net.h"

#include <cmath>
#include <random>

namespace segmentlib::mlp::train {

Parameters Parameters::init(const NetConfig& config, std::uint64_t seed) {
    Parameters p;
    p.config = config;
    const std::size_t v = config.vocab_size;
    const std::size_t d = config.embed_dim;
    const std::size_t h = config.hidden;
    const std::size_t in = config.input_dim();
    const std::size_t fd = config.dict_features();

    std::mt19937_64 rng(seed);
    const auto fill_normal = [&rng](std::vector<float>& t, std::size_t n,
                                    float stddev) {
        std::normal_distribution<float> dist(0.0f, stddev);
        t.resize(n);
        for (float& x : t) {
            x = dist(rng);
        }
    };

    fill_normal(p.embedding, v * d, 0.1f);
    fill_normal(p.w1, h * in, std::sqrt(2.0f / static_cast<float>(in)));
    p.wdict.assign(h * fd, 0.0f);
    p.b1.assign(h, 0.0f);
    fill_normal(p.w2, h, std::sqrt(1.0f / static_cast<float>(h)));
    p.b2 = 0.0f;
    return p;
}

double Net::forward(const Parameters& params, const Batch& batch,
                    Workspace& ws) const {
    const std::size_t b_count = batch.size;
    const std::size_t h = params.config.hidden;
    const std::size_t in = params.config.input_dim();
    const std::size_t fd = params.config.dict_features();

    ws.a.resize(b_count * h);
    ws.h.resize(b_count * h);
    ws.y.resize(b_count);
    if (b_count == 0) {
        return 0.0;
    }

    // A = X · W1ᵀ (+ F · Wdictᵀ) + b1
    backend_->gemm(false, true, b_count, h, in, 1.0f, batch.x.data(), in,
                   params.w1.data(), in, 0.0f, ws.a.data(), h);
    if (fd > 0) {
        backend_->gemm(false, true, b_count, h, fd, 1.0f, batch.f.data(), fd,
                       params.wdict.data(), fd, 1.0f, ws.a.data(), h);
    }
    for (std::size_t b = 0; b < b_count; ++b) {
        backend_->axpy(h, 1.0f, params.b1.data(), ws.a.data() + b * h);
    }

    backend_->relu(ws.a.data(), ws.h.data(), b_count * h);

    // Y = Hh · w2 + b2 (a GEMV, expressed as an n=1 GEMM)
    backend_->gemm(false, false, b_count, 1, h, 1.0f, ws.h.data(), h,
                   params.w2.data(), 1, 0.0f, ws.y.data(), 1);

    // Mean BCE with logits, in the numerically stable form
    // max(y,0) - y·t + log(1 + exp(-|y|)); accumulated in double.
    double loss = 0.0;
    for (std::size_t b = 0; b < b_count; ++b) {
        const double y = ws.y[b] + params.b2;
        ws.y[b] = static_cast<float>(y);
        const double t = batch.targets[b];
        loss += std::max(y, 0.0) - y * t + std::log1p(std::exp(-std::abs(y)));
    }
    return loss / static_cast<double>(b_count);
}

void Net::backward(const Parameters& params, const Batch& batch, Workspace& ws,
                   Gradients& grads) const {
    const std::size_t b_count = batch.size;
    const std::size_t h = params.config.hidden;
    const std::size_t in = params.config.input_dim();
    const std::size_t fd = params.config.dict_features();
    const std::size_t d = params.config.embed_dim;
    const std::size_t slots = 2u * params.config.window;

    grads.w1.assign(h * in, 0.0f);
    grads.wdict.assign(h * fd, 0.0f);
    grads.b1.assign(h, 0.0f);
    grads.w2.assign(h, 0.0f);
    grads.b2 = 0.0f;
    grads.emb_rows.clear();
    grads.emb_grads.clear();
    ws.row_index.clear();
    if (b_count == 0) {
        return;
    }

    // dY = (sigmoid(y) - t) / B  (II.3)
    ws.dy.resize(b_count);
    const float inv_b = 1.0f / static_cast<float>(b_count);
    for (std::size_t b = 0; b < b_count; ++b) {
        const float p =
            1.0f / (1.0f + std::exp(-ws.y[b]));
        ws.dy[b] = (p - batch.targets[b]) * inv_b;
        grads.b2 += ws.dy[b];
    }

    // dw2 = Hhᵀ · dY
    backend_->gemm(true, false, h, 1, b_count, 1.0f, ws.h.data(), h,
                   ws.dy.data(), 1, 0.0f, grads.w2.data(), 1);

    // dA = (dY ⊗ w2) ⊙ 1[A > 0]
    ws.da.resize(b_count * h);
    for (std::size_t b = 0; b < b_count; ++b) {
        for (std::size_t j = 0; j < h; ++j) {
            ws.da[b * h + j] = ws.dy[b] * params.w2[j];
        }
    }
    backend_->relu_backward(ws.a.data(), ws.da.data(), b_count * h);

    // dW1 = dAᵀ · X, dWdict = dAᵀ · F, db1 = Σ_b dA_b
    backend_->gemm(true, false, h, in, b_count, 1.0f, ws.da.data(), h,
                   batch.x.data(), in, 0.0f, grads.w1.data(), in);
    if (fd > 0) {
        backend_->gemm(true, false, h, fd, b_count, 1.0f, ws.da.data(), h,
                       batch.f.data(), fd, 0.0f, grads.wdict.data(), fd);
    }
    for (std::size_t b = 0; b < b_count; ++b) {
        backend_->axpy(h, 1.0f, ws.da.data() + b * h, grads.b1.data());
    }

    // dX = dA · W1, then scatter each window-slot block dv_j into the
    // constituent embedding rows with weight 1/n_j (mean-pooling backward).
    ws.dx.resize(b_count * in);
    backend_->gemm(false, false, b_count, in, h, 1.0f, ws.da.data(), h,
                   params.w1.data(), in, 0.0f, ws.dx.data(), in);
    for (std::size_t b = 0; b < b_count; ++b) {
        for (std::size_t j = 0; j < slots; ++j) {
            const std::size_t slot = b * slots + j;
            const std::uint32_t lo = batch.slot_starts[slot];
            const std::uint32_t hi = batch.slot_starts[slot + 1];
            const float inv_n = 1.0f / static_cast<float>(hi - lo);
            const float* dv = ws.dx.data() + b * in + j * d;
            for (std::uint32_t r = lo; r < hi; ++r) {
                const std::uint32_t row = batch.slot_rows[r];
                auto [it, inserted] = ws.row_index.try_emplace(
                    row, static_cast<std::uint32_t>(grads.emb_rows.size()));
                if (inserted) {
                    grads.emb_rows.push_back(row);
                    grads.emb_grads.resize(grads.emb_grads.size() + d, 0.0f);
                }
                float* g = grads.emb_grads.data() +
                           static_cast<std::size_t>(it->second) * d;
                for (std::size_t c = 0; c < d; ++c) {
                    g[c] += dv[c] * inv_n;
                }
            }
        }
    }
}

}  // namespace segmentlib::mlp::train
