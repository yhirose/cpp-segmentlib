#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "mlp/train/compute_backend.h"
#include "mlp/train/dataset.h"
#include "mlp/train/example.h"
#include "mlp/train/net.h"

using namespace segmentlib::mlp::train;

namespace {

// A hand-built two-example ExampleSet exercising every path: a multi-
// codepoint EGC (mean pooling over n>1), PAD slots (window falls off both
// ends), and dictionary features.
ExampleSet tiny_set() {
    ExampleSet set;
    set.window = 1;  // 2 slots
    set.num_dicts = 1;
    // One sentence, 3 EGCs: {row2}, {row3,row4}, {row2,row3}.
    set.rows = {2, 3, 4, 2, 3};
    set.egc_starts = {0, 1, 3, 5};
    set.sentence_starts = {0, 3};
    set.examples = {{0, 0, 1.0f}, {0, 1, 0.0f}};
    set.feat_offsets = {0, 2, 3};
    set.feat_indices = {0, 5, 3};
    return set;
}

NetConfig tiny_config() {
    NetConfig config;
    config.window = 1;
    config.embed_dim = 3;
    config.hidden = 4;
    config.vocab_size = 5;
    config.num_dicts = 1;
    return config;
}

}  // namespace

TEST_CASE("analytic gradients match finite differences") {
    const ExampleSet set = tiny_set();
    const NetConfig config = tiny_config();
    Parameters params = Parameters::init(config, 12345);
    const auto backend = make_cpu_backend();
    Net net(*backend);
    Dataset data(set, config.embed_dim, 8);
    Batch batch;
    Workspace ws;

    // Loss as a pure function of the current parameters (the batch gather
    // reads the embedding, so it must be redone after each perturbation).
    const auto loss_at = [&]() {
        data.fill_batch(0, params.embedding, batch);
        return net.forward(params, batch, ws);
    };

    (void)loss_at();
    Gradients grads;
    net.backward(params, batch, ws, grads);

    // Scatter the sparse embedding gradient into a dense copy for lookup.
    std::vector<float> emb_grad(params.embedding.size(), 0.0f);
    for (std::size_t i = 0; i < grads.emb_rows.size(); ++i) {
        for (std::size_t c = 0; c < config.embed_dim; ++c) {
            emb_grad[static_cast<std::size_t>(grads.emb_rows[i]) *
                         config.embed_dim +
                     c] = grads.emb_grads[i * config.embed_dim + c];
        }
    }

    const float eps = 1e-2f;
    const auto check_tensor = [&](std::vector<float>& tensor,
                                  const std::vector<float>& analytic,
                                  const char* name) {
        CAPTURE(name);
        REQUIRE(tensor.size() == analytic.size());
        for (std::size_t i = 0; i < tensor.size(); ++i) {
            CAPTURE(i);
            const float saved = tensor[i];
            tensor[i] = saved + eps;
            const double plus = loss_at();
            tensor[i] = saved - eps;
            const double minus = loss_at();
            tensor[i] = saved;
            const double numeric = (plus - minus) / (2.0 * eps);
            const double tolerance =
                1e-3 + 0.02 * std::max(std::abs(numeric),
                                       static_cast<double>(
                                           std::abs(analytic[i])));
            CHECK(std::abs(numeric - analytic[i]) <= tolerance);
        }
    };

    check_tensor(params.w1, grads.w1, "w1");
    check_tensor(params.wdict, grads.wdict, "wdict");
    check_tensor(params.b1, grads.b1, "b1");
    check_tensor(params.w2, grads.w2, "w2");
    check_tensor(params.embedding, emb_grad, "embedding");

    // b2 is a scalar.
    const float saved = params.b2;
    params.b2 = saved + eps;
    const double plus = loss_at();
    params.b2 = saved - eps;
    const double minus = loss_at();
    params.b2 = saved;
    const double numeric = (plus - minus) / (2.0 * eps);
    CHECK(std::abs(numeric - grads.b2) <=
          1e-3 + 0.02 * std::abs(numeric));
}

TEST_CASE("PAD slots gather row 0 and route gradient to it") {
    const ExampleSet set = tiny_set();
    const NetConfig config = tiny_config();
    Parameters params = Parameters::init(config, 7);
    const auto backend = make_cpu_backend();
    Net net(*backend);
    Dataset data(set, config.embed_dim, 8);
    Batch batch;
    data.fill_batch(0, params.embedding, batch);

    // Example 0 (boundary 0, w=1): slots cover EGCs 0 and 1 — no PAD.
    // Example 1 (boundary 1): slots cover EGCs 1 and 2 — no PAD either; make
    // a PAD case explicit instead: boundary 0's left slot would be EGC 0, so
    // check via a single-boundary sentence below.
    ExampleSet pad_set;
    pad_set.window = 2;  // 4 slots: EGCs -1, 0, 1, 2 → PAD at both ends
    pad_set.num_dicts = 0;
    pad_set.rows = {2, 3};
    pad_set.egc_starts = {0, 1, 2};
    pad_set.sentence_starts = {0, 2};
    pad_set.examples = {{0, 0, 1.0f}};
    pad_set.feat_offsets = {0, 0};

    NetConfig pad_config = config;
    pad_config.window = 2;
    pad_config.num_dicts = 0;
    Parameters pad_params = Parameters::init(pad_config, 7);
    Dataset pad_data(pad_set, pad_config.embed_dim, 8);
    Batch pad_batch;
    pad_data.fill_batch(0, pad_params.embedding, pad_batch);

    // Slots 0 and 3 are out of range: single constituent row 0 (PAD).
    REQUIRE(pad_batch.slot_starts.size() == 5);
    CHECK(pad_batch.slot_rows[pad_batch.slot_starts[0]] == 0);
    CHECK(pad_batch.slot_rows[pad_batch.slot_starts[3]] == 0);

    Workspace ws;
    (void)net.forward(pad_params, pad_batch, ws);
    Gradients grads;
    net.backward(pad_params, pad_batch, ws, grads);
    bool pad_row_updated = false;
    for (const std::uint32_t row : grads.emb_rows) {
        if (row == 0) {
            pad_row_updated = true;
        }
    }
    CHECK(pad_row_updated);
}
