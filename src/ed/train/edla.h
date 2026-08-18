#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mlp/train/compute_backend.h"
#include "mlp/train/dataset.h"
#include "mlp/train/net.h"

// The Error Diffusion Learning Algorithm (EDLA, arXiv 2504.14814) applied to
// this library's boundary classifier: one global error signal is broadcast to
// the hidden layer, and each unit turns it into a weight change using only its
// own activity and a fixed excitatory/inhibitory polarity -- no per-layer
// gradient is propagated back through the network.
//
// The network itself is the MLP backend's, unchanged (mlp/train/net.h), and so
// are the forward pass, the optimizer and the quantizer. Holding all of that
// fixed is deliberate: the only difference between a model trained here and one
// trained by mlp::train is the credit-assignment rule, so a measured accuracy
// difference between the two backends is attributable to that rule alone.

namespace segmentlib::ed::train {

using mlp::train::Batch;
using mlp::train::ComputeBackend;
using mlp::train::Gradients;
using mlp::train::NetConfig;
using mlp::train::Parameters;
using mlp::train::Workspace;

// How the embedding table is updated. The paper's networks take fixed input
// features, so a learned input embedding is outside what it specifies; these
// are the two readings of "what EDLA does here", and the choice is a training
// flag rather than a silent decision (design.md 11).
enum class EmbeddingUpdate : std::uint8_t {
    // Reuse the exact one-hop map dX = dA · W1 (the same GEMM backpropagation
    // uses) to carry the hidden layer's EDLA signal into the embedding rows.
    // The hop crosses no nonlinearity, so it chains no activation derivatives
    // -- what EDLA actually objects to -- and it is what makes the embedding
    // depend on which characters were in the window. The default.
    Hybrid,
    // Diffuse the global signal straight into the embedding as its own layer,
    // gated by a per-dimension polarity. Faithful to "no backward map at all",
    // but the update direction is then the same for every character in the
    // batch, so the table barely differentiates. Kept for ablation.
    Pure,
};

struct EdlaConfig {
    EmbeddingUpdate embedding_update = EmbeddingUpdate::Hybrid;
    // Attenuation applied to the broadcast signal on its way to the hidden
    // layer (lambda). One hop and a per-layer learning rate make any constant
    // here redundant, so it stays 1; the term is explicit for the deeper
    // variants the paper studies, where the decay per hop is the whole point.
    float diffusion = 1.0f;

    // Kolen-Pollack (Akrout et al. 2019, arXiv:1904.05391): learn the feedback
    // magnitude instead of pinning it at the polarity's +-1. The measured cost
    // of pinning it is 0.16pt under plain SGD (design.md 11.6) -- that is the
    // discarded |w2_j|, and this recovers it without reading w2: the forward
    // and feedback weights take the same update and the same decay, each using
    // only its own value, so their difference decays geometrically.
    bool learn_feedback = false;
    // The shared decay that drives that convergence. Must be > 0 for the
    // feedback to converge at all; too large and it also shrinks w2 itself,
    // since both weights decay by construction.
    float feedback_decay = 0.01f;
    // Magnitude the feedback starts at, before it converges on w2's. The
    // default 1 is what fixed-polarity EDLA uses forever, which makes the
    // ablation exact -- but w2 is initialized near 1/sqrt(H) (~0.06 at H=256),
    // so the first updates are then an order of magnitude larger than
    // backpropagation's, and that transient is what limits the usable learning
    // rate. Setting this to that scale is not weight transport: it is the
    // initializer's public constant, not any trained weight's value.
    float feedback_init = 1.0f;
};

// The fixed excitatory (+1) / inhibitory (-1) tag of hidden unit j: the first
// half of the layer excites the output, the second half inhibits it. Derived
// from H alone, so it is never stored in the model file -- a loader recomputes
// it exactly. A contiguous split rather than an alternating one so that
// `w1[0 .. H/2)` is the excitatory pool when reading a dump.
[[nodiscard]] constexpr float polarity(std::size_t j, std::uint16_t hidden) noexcept {
    return j < static_cast<std::size_t>(hidden) / 2 ? 1.0f : -1.0f;
}

// Random initialization, plus the sign convention EDLA needs: w2[j] starts on
// the side its polarity assigns it. Everything else matches
// mlp::train::Parameters::init, seed for seed.
[[nodiscard]] Parameters init_parameters(const NetConfig& config, std::uint64_t seed);

// Clamps every w2[j] back onto its polarity's side of zero. Applied after each
// optimizer step: the hidden-unit rule below uses polarity() in place of
// sign(w2[j]), so the two have to keep agreeing for the substitution to mean
// anything.
void project_dale(Parameters& params);

// Hidden units whose w2 the projection has pinned to exactly zero. They receive
// no output-layer gradient and contribute nothing, so a growing count is the
// signal that the polarity split is costing capacity; the trainer logs it.
[[nodiscard]] std::size_t count_pinned(const Parameters& params);

class Edla {
public:
    Edla(ComputeBackend& backend, const EdlaConfig& config, const NetConfig& net)
        : backend_(&backend), config_(config), feedback_(net.hidden) {
        // The feedback starts where fixed-polarity EDLA leaves it forever, so
        // learn_feedback = false is exactly this rule with the learning turned
        // off, and the ablation compares like with like.
        for (std::size_t j = 0; j < feedback_.size(); ++j) {
            feedback_[j] = polarity(j, net.hidden) * config.feedback_init;
        }
    }

    // The EDLA counterpart of Net::backward: fills `grads` with one update
    // step, given a preceding Net::forward on the same batch. Overwrites
    // `grads` exactly as Net::backward does, so the optimizer consumes it
    // unchanged -- the values are local update directions carrying the sign
    // its subtraction expects, not partial derivatives of the loss.
    void local_update(const Parameters& params, const Batch& batch, Workspace& ws,
                      Gradients& grads) const;

    // The Kolen-Pollack step, run after the optimizer has moved w2 (and after
    // the Dale projection, so the feedback tracks the weight that actually
    // resulted). `w2_before` is w2 as it stood before that move. No-op unless
    // learn_feedback is set.
    void track_feedback(Parameters& params, const std::vector<float>& w2_before);

    // The per-unit feedback the hidden layer is currently using: +-1 under
    // fixed polarity, converging on w2 under Kolen-Pollack. The trainer reports
    // how far it has converged.
    [[nodiscard]] const std::vector<float>& feedback() const noexcept {
        return feedback_;
    }

private:
    ComputeBackend* backend_;
    EdlaConfig config_;
    std::vector<float> feedback_;
};

}  // namespace segmentlib::ed::train
