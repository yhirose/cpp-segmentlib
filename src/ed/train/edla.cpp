#include "ed/train/edla.h"

#include <cmath>
#include <cstdlib>

namespace segmentlib::ed::train {

Parameters init_parameters(const NetConfig& config, std::uint64_t seed) {
    Parameters p = Parameters::init(config, seed);
    // Same draw as the MLP backend, then folded onto the polarity's side. Note
    // this is |w2| with a sign imposed, not a clamp of the original: clamping
    // would zero half the layer before it ever trained, and a unit pinned at
    // zero never leaves (its output-layer update is proportional to h, but its
    // contribution to y -- and so the whole reason to move it -- is zero).
    for (std::size_t j = 0; j < p.w2.size(); ++j) {
        p.w2[j] = std::abs(p.w2[j]) * polarity(j, config.hidden);
    }
    return p;
}

void project_dale(Parameters& params) {
    const std::uint16_t h = params.config.hidden;
    for (std::size_t j = 0; j < params.w2.size(); ++j) {
        const float sign = polarity(j, h);
        if (params.w2[j] * sign < 0.0f) {
            params.w2[j] = 0.0f;
        }
    }
}

std::size_t count_pinned(const Parameters& params) {
    std::size_t pinned = 0;
    for (const float w : params.w2) {
        if (w == 0.0f) {
            ++pinned;
        }
    }
    return pinned;
}

void Edla::local_update(const Parameters& params, const Batch& batch, Workspace& ws,
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

    // The global error signal, the one quantity that leaves the output layer:
    // d = t - sigmoid(y), broadcast whole to every unit below. Held here in
    // Adam's sign convention and averaged over the batch, so dy = -d/B.
    //
    // sigmoid(y) rather than the raw logit keeps the broadcast bounded to
    // [-1,1]. A single scalar reaches every layer at once with nothing
    // downstream to attenuate it, so an unbounded one would drive the whole
    // network on one wide-margin example -- the activation blow-up the paper
    // reports for deeper EDLA networks.
    ws.dy.resize(b_count);
    const float inv_b = 1.0f / static_cast<float>(b_count);
    for (std::size_t b = 0; b < b_count; ++b) {
        const float p = 1.0f / (1.0f + std::exp(-ws.y[b]));
        ws.dy[b] = (p - batch.targets[b]) * inv_b;
        grads.b2 += ws.dy[b];
    }

    // Output layer: one linear hop from the loss, so its exact gradient is
    // available locally (presynaptic activity times the broadcast signal) and
    // EDLA has nothing to approximate. Identical to backpropagation here.
    backend_->gemm(true, false, h, 1, b_count, 1.0f, ws.h.data(), h, ws.dy.data(), 1,
                   0.0f, grads.w2.data(), 1);

    // Hidden layer -- the one place EDLA departs from backpropagation.
    //
    // Backpropagation forms dA[b][j] = dy[b] · w2[j] · 1[a>0]; reading w2[j] is
    // exactly the backward pass through the output weights. EDLA replaces that
    // read with the unit's own fixed polarity, so the update needs nothing but
    // the broadcast scalar, the unit's own pre-activation, and its tag:
    //
    //     Da_j = g'(a_j) · p_j · d
    //
    // The paper's positive/negative error channels are this same expression:
    // splitting d into d+ = max(d,0) and d- = max(-d,0) and handing excitatory
    // units d+ and inhibitory units d- gives p_j·d = d+ - d- for p_j = +1 and
    // d- - d+ for p_j = -1, which is what the two channels reconstruct.
    // project_dale keeps sign(w2[j]) == p_j, so the substitution stands in for
    // the sign of the weight it replaces and only its magnitude is discarded.
    ws.da.resize(b_count * h);
    for (std::size_t b = 0; b < b_count; ++b) {
        for (std::size_t j = 0; j < h; ++j) {
            ws.da[b * h + j] = ws.dy[b] * polarity(j, params.config.hidden) *
                               config_.diffusion;
        }
    }
    backend_->relu_backward(ws.a.data(), ws.da.data(), b_count * h);

    // From here the arithmetic is the ordinary local Hebbian product of
    // presynaptic activity with the unit's own signal, so the same GEMMs apply.
    backend_->gemm(true, false, h, in, b_count, 1.0f, ws.da.data(), h, batch.x.data(),
                   in, 0.0f, grads.w1.data(), in);
    if (fd > 0) {
        backend_->gemm(true, false, h, fd, b_count, 1.0f, ws.da.data(), h,
                       batch.f.data(), fd, 0.0f, grads.wdict.data(), fd);
    }
    for (std::size_t b = 0; b < b_count; ++b) {
        backend_->axpy(h, 1.0f, ws.da.data() + b * h, grads.b1.data());
    }

    // Embedding: dX either through W1 (Hybrid) or diffused directly (Pure).
    ws.dx.resize(b_count * in);
    if (config_.embedding_update == EmbeddingUpdate::Hybrid) {
        backend_->gemm(false, false, b_count, in, h, 1.0f, ws.da.data(), h,
                       params.w1.data(), in, 0.0f, ws.dx.data(), in);
    } else {
        // One more diffusion hop instead of a backward map: every window slot
        // of every example gets the same per-dimension value, so what a row
        // learns depends only on how often it appeared with which error sign.
        for (std::size_t b = 0; b < b_count; ++b) {
            for (std::size_t j = 0; j < slots; ++j) {
                float* dv = ws.dx.data() + b * in + j * d;
                for (std::size_t c = 0; c < d; ++c) {
                    dv[c] = ws.dy[b] * polarity(c, static_cast<std::uint16_t>(d)) *
                            config_.diffusion;
                }
            }
        }
    }

    // Scatter each slot's block into its constituent rows with weight 1/n --
    // the mean-pool of the forward pass, mirrored. Sparse: only rows the batch
    // touched appear, which is what Adam's lazy per-row moments expect.
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
                float* g =
                    grads.emb_grads.data() + static_cast<std::size_t>(it->second) * d;
                for (std::size_t c = 0; c < d; ++c) {
                    g[c] += dv[c] * inv_n;
                }
            }
        }
    }
}

}  // namespace segmentlib::ed::train
