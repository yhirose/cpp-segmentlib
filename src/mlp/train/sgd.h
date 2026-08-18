#pragma once

#include "mlp/train/net.h"

// Plain stochastic gradient descent: params -= lr * grads, nothing else. It
// exists as a control, not a contender: Adam's per-parameter RMS normalization
// is invariant to scaling a parameter's gradient by a constant, which is
// exactly the shape of the difference between the EDLA update and
// backpropagation's (a per-hidden-unit factor of |w2_j|, design.md 11.6). So
// under Adam the two rules partly collapse into each other, and only an
// optimizer that keeps gradient scale -- this one -- can measure the rule
// itself. No momentum and no weight decay, so nothing else is smoothing the
// comparison either.

namespace segmentlib::mlp::train {

class Sgd {
public:
    explicit Sgd(float lr) noexcept : lr_(lr) {}

    // Applies one update. `grads` must come from Net::backward (or the EDLA
    // rule) with the same NetConfig as `params`.
    void step(Parameters& params, const Gradients& grads) const;

private:
    float lr_;
};

}  // namespace segmentlib::mlp::train
