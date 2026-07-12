#include "mlp/train/adam.h"

#include <cmath>

namespace segmentlib::mlp::train {

Adam::Adam(const NetConfig& config, const AdamConfig& adam) : cfg_(adam) {
    const std::size_t in = config.input_dim();
    const std::size_t h = config.hidden;
    const std::size_t fd = config.dict_features();
    const auto zero = [](Moments& mom, std::size_t n) {
        mom.m.assign(n, 0.0f);
        mom.v.assign(n, 0.0f);
    };
    zero(w1_, h * in);
    zero(wdict_, h * fd);
    zero(b1_, h);
    zero(w2_, h);
    const std::size_t emb = static_cast<std::size_t>(config.vocab_size) *
                            config.embed_dim;
    emb_m_.assign(emb, 0.0f);
    emb_v_.assign(emb, 0.0f);
    emb_t_.assign(config.vocab_size, 0u);
}

void Adam::update_dense(std::vector<float>& param,
                        const std::vector<float>& grad, Moments& mom,
                        double bias1, double bias2, bool decay) {
    const float lr = cfg_.lr;
    for (std::size_t i = 0; i < param.size(); ++i) {
        const float g = grad[i];
        mom.m[i] = cfg_.beta1 * mom.m[i] + (1.0f - cfg_.beta1) * g;
        mom.v[i] = cfg_.beta2 * mom.v[i] + (1.0f - cfg_.beta2) * g * g;
        const auto m_hat = static_cast<float>(mom.m[i] / bias1);
        const auto v_hat = static_cast<float>(mom.v[i] / bias2);
        if (decay) {
            param[i] -= lr * cfg_.weight_decay * param[i];
        }
        param[i] -= lr * m_hat / (std::sqrt(v_hat) + cfg_.eps);
    }
}

void Adam::step(Parameters& params, const Gradients& grads) {
    ++t_;
    const double bias1 = 1.0 - std::pow(cfg_.beta1, static_cast<double>(t_));
    const double bias2 = 1.0 - std::pow(cfg_.beta2, static_cast<double>(t_));
    const bool decay = cfg_.weight_decay > 0.0f;

    update_dense(params.w1, grads.w1, w1_, bias1, bias2, decay);
    update_dense(params.wdict, grads.wdict, wdict_, bias1, bias2, decay);
    update_dense(params.b1, grads.b1, b1_, bias1, bias2, false);
    update_dense(params.w2, grads.w2, w2_, bias1, bias2, decay);
    {
        const float g = grads.b2;
        b2_m_ = cfg_.beta1 * b2_m_ + (1.0f - cfg_.beta1) * g;
        b2_v_ = cfg_.beta2 * b2_v_ + (1.0f - cfg_.beta2) * g * g;
        const auto m_hat = static_cast<float>(b2_m_ / bias1);
        const auto v_hat = static_cast<float>(b2_v_ / bias2);
        params.b2 -= cfg_.lr * m_hat / (std::sqrt(v_hat) + cfg_.eps);
    }

    // Sparse embedding rows: per-row step count drives the bias correction,
    // so a rarely-seen row is corrected as if it had only ever taken its own
    // steps (lazy update, II.4).
    const std::size_t d = params.config.embed_dim;
    const bool decay_emb = decay && cfg_.decay_embedding;
    for (std::size_t i = 0; i < grads.emb_rows.size(); ++i) {
        const std::uint32_t row = grads.emb_rows[i];
        const std::uint64_t rt = ++emb_t_[row];
        const double rb1 = 1.0 - std::pow(cfg_.beta1, static_cast<double>(rt));
        const double rb2 = 1.0 - std::pow(cfg_.beta2, static_cast<double>(rt));
        const float* g = grads.emb_grads.data() + i * d;
        float* p = params.embedding.data() + static_cast<std::size_t>(row) * d;
        float* m = emb_m_.data() + static_cast<std::size_t>(row) * d;
        float* v = emb_v_.data() + static_cast<std::size_t>(row) * d;
        for (std::size_t c = 0; c < d; ++c) {
            m[c] = cfg_.beta1 * m[c] + (1.0f - cfg_.beta1) * g[c];
            v[c] = cfg_.beta2 * v[c] + (1.0f - cfg_.beta2) * g[c] * g[c];
            const auto m_hat = static_cast<float>(m[c] / rb1);
            const auto v_hat = static_cast<float>(v[c] / rb2);
            if (decay_emb) {
                p[c] -= cfg_.lr * cfg_.weight_decay * p[c];
            }
            p[c] -= cfg_.lr * m_hat / (std::sqrt(v_hat) + cfg_.eps);
        }
    }
}

}  // namespace segmentlib::mlp::train
