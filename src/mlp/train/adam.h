#pragma once

#include <cstdint>
#include <vector>

#include "mlp/train/net.h"

// Adam (design.ja.md 4.9): standard moments for
// the dense tensors, lazy per-row moments for the sparse embedding gradient
// (only rows that appear in a batch update, each with its own step count for
// bias correction).

namespace segmentlib::mlp::train {

struct AdamConfig {
    float lr = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    // Decoupled (AdamW-style) weight decay on the dense weight matrices
    // (W1/W_dict/w2); never applied to biases, and to the embedding only when
    // decay_embedding is set (II.4: avoidable over-shrinking).
    float weight_decay = 0.0f;
    bool decay_embedding = false;
};

class Adam {
public:
    Adam(const NetConfig& config, const AdamConfig& adam);

    // Applies one update. `grads` must come from Net::backward with the same
    // NetConfig.
    void step(Parameters& params, const Gradients& grads);

private:
    struct Moments {
        std::vector<float> m, v;
    };

    void update_dense(std::vector<float>& param, const std::vector<float>& grad,
                      Moments& mom, double bias1, double bias2, bool decay);

    AdamConfig cfg_;
    std::uint64_t t_ = 0;  // dense step count
    Moments w1_, wdict_, b1_, w2_;
    float b2_m_ = 0.0f, b2_v_ = 0.0f;
    // Embedding moments: full-size buffers plus a per-row step count. The
    // buffers are dense in memory (V×d floats ×2 is small at this scale) but
    // updates touch only the rows present in the batch.
    std::vector<float> emb_m_, emb_v_;
    std::vector<std::uint32_t> emb_t_;
};

}  // namespace segmentlib::mlp::train
