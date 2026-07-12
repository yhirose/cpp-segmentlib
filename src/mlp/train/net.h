#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "mlp/train/compute_backend.h"
#include "mlp/train/dataset.h"
#include "mlp/train/example.h"

// The fp32 network (mlp_impl_design.ja.md 第II部): forward pass, masked BCE
// loss and backward pass over the one-hidden-layer pointwise classifier of
// design.ja.md 5.4, with the heavy lifting delegated to ComputeBackend GEMMs.

namespace segmentlib::mlp::train {

struct NetConfig {
    std::uint8_t window = 5;      // w
    std::uint16_t embed_dim = 64; // d
    std::uint16_t hidden = 256;   // H
    std::uint32_t vocab_size = 0; // V, including PAD and UNK rows
    std::uint32_t num_dicts = 0;

    [[nodiscard]] std::uint32_t input_dim() const noexcept {  // 2w·d
        return 2u * window * embed_dim;
    }
    [[nodiscard]] std::uint32_t dict_features() const noexcept {  // Fd
        return num_dicts * kDictFeaturesPerDict;
    }
};

// The trainable parameters, laid out exactly as the model file stores them
// (5.7): embedding row-major V×d, W1 row-major [h][j*d + c], W_dict
// [h][dict*12 + feat], so the exporter (step 5) quantizes without reshaping.
struct Parameters {
    NetConfig config;
    std::vector<float> embedding;  // V × d
    std::vector<float> w1;         // H × 2w·d
    std::vector<float> wdict;      // H × Fd
    std::vector<float> b1;         // H
    std::vector<float> w2;         // H
    float b2 = 0.0f;

    // He initialization for W1/w2, small normal for the embedding, zeros for
    // W_dict (binary features feed distinct columns, so zero init breaks no
    // symmetry) and biases.
    [[nodiscard]] static Parameters init(const NetConfig& config,
                                         std::uint64_t seed);
};

// Per-batch gradients. Dense tensors mirror Parameters; the embedding
// gradient is sparse (II.3): only rows touched by the batch appear, as
// emb_rows[i] with gradient block emb_grads[i*d .. i*d+d).
struct Gradients {
    std::vector<float> w1, wdict, b1, w2;
    float b2 = 0.0f;
    std::vector<std::uint32_t> emb_rows;
    std::vector<float> emb_grads;
};

// Reusable forward/backward buffers (sized on first use).
struct Workspace {
    std::vector<float> a;   // B × H pre-activation
    std::vector<float> h;   // B × H
    std::vector<float> y;   // B logits
    std::vector<float> dy;  // B
    std::vector<float> da;  // B × H
    std::vector<float> dx;  // B × 2w·d
    std::unordered_map<std::uint32_t, std::uint32_t> row_index;  // scatter map
};

class Net {
public:
    explicit Net(ComputeBackend& backend) noexcept : backend_(&backend) {}

    // Forward pass (II.1): fills ws.a/h/y and returns the mean
    // BCE-with-logits loss over the batch. (The mask of II.2 is structural:
    // unsupervised gaps never became examples, so every batch row counts.)
    [[nodiscard]] double forward(const Parameters& params, const Batch& batch,
                                 Workspace& ws) const;

    // Backward pass (II.3); requires a preceding forward on the same batch.
    // Overwrites `grads` (dense zeroed and refilled, sparse rebuilt).
    void backward(const Parameters& params, const Batch& batch, Workspace& ws,
                  Gradients& grads) const;

private:
    ComputeBackend* backend_;
};

}  // namespace segmentlib::mlp::train
