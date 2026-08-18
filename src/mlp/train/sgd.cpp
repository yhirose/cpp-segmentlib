#include "mlp/train/sgd.h"

#include <cstddef>
#include <vector>

namespace segmentlib::mlp::train {

namespace {

void axpy_neg(float lr, const std::vector<float>& grad, std::vector<float>& param) {
    for (std::size_t i = 0; i < param.size(); ++i) {
        param[i] -= lr * grad[i];
    }
}

}  // namespace

void Sgd::step(Parameters& params, const Gradients& grads) const {
    axpy_neg(lr_, grads.w1, params.w1);
    axpy_neg(lr_, grads.wdict, params.wdict);
    axpy_neg(lr_, grads.b1, params.b1);
    axpy_neg(lr_, grads.w2, params.w2);
    params.b2 -= lr_ * grads.b2;

    // The embedding gradient is sparse: only rows the batch touched appear.
    const std::size_t d = params.config.embed_dim;
    for (std::size_t i = 0; i < grads.emb_rows.size(); ++i) {
        float* row = params.embedding.data() +
                     static_cast<std::size_t>(grads.emb_rows[i]) * d;
        const float* g = grads.emb_grads.data() + i * d;
        for (std::size_t c = 0; c < d; ++c) {
            row[c] -= lr_ * g[c];
        }
    }
}

}  // namespace segmentlib::mlp::train
