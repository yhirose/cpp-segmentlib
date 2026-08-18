#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "ed/train/edla.h"
#include "mlp/train/adam.h"
#include "mlp/train/compute_backend.h"
#include "mlp/train/dataset.h"
#include "mlp/train/example.h"
#include "mlp/train/net.h"

// EDLA's local update has no gradient to difference against, so it is pinned
// two ways instead: against values worked out by hand below, and against
// backpropagation in the one configuration where the two provably coincide.

using namespace segmentlib::mlp::train;
namespace ed = segmentlib::ed::train;

namespace {

// One sentence, two single-row EGCs (row 2 twice), one example at the gap
// between them with label 1. window=1, so the two window slots are exactly
// those two EGCs and nothing is PAD.
ExampleSet tiny_set() {
    ExampleSet set;
    set.window = 1;  // 2 slots
    set.num_dicts = 0;
    set.rows = {2, 2};
    set.egc_starts = {0, 1, 2};
    set.sentence_starts = {0, 2};
    set.examples = {{0, 0, 1.0f}};
    set.feat_offsets = {0, 0};
    return set;
}

NetConfig tiny_config() {
    NetConfig config;
    config.window = 1;
    config.embed_dim = 2;
    config.hidden = 2;  // polarity: unit 0 excitatory, unit 1 inhibitory
    config.vocab_size = 3;
    config.num_dicts = 0;
    return config;
}

// The hand-computed network. Every value below is a binary fraction, so the
// expectations in the tests are exact rather than approximate.
//
//   embedding[row 2] = [1, 1]      → x = [1, 1, 1, 1]   (2 slots × d=2)
//   w1 = [[.25 ×4], [.125 ×4]]     → a = [1, 0.5],  h = [1, 0.5]  (both > 0)
//   w2 = [2, -4],  b1 = b2 = 0     → y = 2·1 − 4·0.5 = 0
//   sigmoid(0) = 0.5,  t = 1       → dy = 0.5 − 1 = −0.5   (B = 1)
Parameters tiny_params() {
    Parameters p;
    p.config = tiny_config();
    p.embedding.assign(3 * 2, 0.0f);
    p.embedding[2 * 2 + 0] = 1.0f;  // row 2
    p.embedding[2 * 2 + 1] = 1.0f;
    p.w1 = {0.25f, 0.25f, 0.25f, 0.25f, 0.125f, 0.125f, 0.125f, 0.125f};
    p.wdict.clear();
    p.b1 = {0.0f, 0.0f};
    p.w2 = {2.0f, -4.0f};  // magnitudes differ; signs match the polarity
    p.b2 = 0.0f;
    return p;
}

}  // namespace

TEST_CASE("EDLA hidden units use their polarity where backprop uses w2") {
    const ExampleSet set = tiny_set();
    Parameters params = tiny_params();
    const auto backend = make_cpu_backend();
    Net net(*backend);
    Dataset data(set, params.config.embed_dim, 8);
    Batch batch;
    Workspace ws;

    data.fill_batch(0, params.embedding, batch);
    REQUIRE(batch.size == 1);
    const double loss = net.forward(params, batch, ws);
    CHECK(ws.y[0] == doctest::Approx(0.0));           // y = 2·1 − 4·0.5
    CHECK(loss == doctest::Approx(std::log(2.0)));    // −log sigmoid(0)

    Gradients grads;
    ed::Edla(*backend, ed::EdlaConfig{}, params.config)
        .local_update(params, batch, ws, grads);

    // dy = sigmoid(0) − 1 = −0.5, and the output layer is exact: it is one
    // linear hop from the loss, so EDLA changes nothing here.
    CHECK(grads.b2 == doctest::Approx(-0.5));
    REQUIRE(grads.w2.size() == 2);
    CHECK(grads.w2[0] == doctest::Approx(-0.5));   // h[0]·dy = 1·(−0.5)
    CHECK(grads.w2[1] == doctest::Approx(-0.25));  // h[1]·dy = 0.5·(−0.5)

    // The hidden layer is where EDLA departs: Da_j = dy · p_j (relu' = 1 for
    // both units), so [−0.5, +0.5]. Backpropagation would have used w2 itself
    // and produced [dy·2, dy·(−4)] = [−1, +2] — the magnitudes are what EDLA
    // discards, and this fixture's differ from 1 precisely to show it.
    REQUIRE(grads.b1.size() == 2);
    CHECK(grads.b1[0] == doctest::Approx(-0.5));
    CHECK(grads.b1[1] == doctest::Approx(0.5));

    // dW1[j][i] = Da_j · x_i, and every x_i is 1.
    REQUIRE(grads.w1.size() == 8);
    for (std::size_t i = 0; i < 4; ++i) {
        CHECK(grads.w1[i] == doctest::Approx(-0.5));
        CHECK(grads.w1[4 + i] == doctest::Approx(0.5));
    }

    // Hybrid embedding: dX = Da · W1 gives −0.5·0.25 + 0.5·0.125 = −0.0625 per
    // input dimension; row 2 fills both slots, so it accumulates twice.
    REQUIRE(grads.emb_rows.size() == 1);
    CHECK(grads.emb_rows[0] == 2);
    REQUIRE(grads.emb_grads.size() == 2);
    CHECK(grads.emb_grads[0] == doctest::Approx(-0.125));
    CHECK(grads.emb_grads[1] == doctest::Approx(-0.125));
}

TEST_CASE("EDLA reduces to backprop exactly when every |w2| is 1") {
    // The substitution EDLA makes is w2[j] → sign(w2[j]). With every magnitude
    // already 1 the two rules are the same expression, so the whole update
    // must agree bit for bit — a sign error anywhere would break this even
    // though the hand-computed case above cannot see magnitudes.
    const ExampleSet set = tiny_set();
    Parameters params = tiny_params();
    params.w2 = {1.0f, -1.0f};

    const auto backend = make_cpu_backend();
    Net net(*backend);
    Dataset data(set, params.config.embed_dim, 8);
    Batch batch;
    Workspace ws;
    data.fill_batch(0, params.embedding, batch);
    (void)net.forward(params, batch, ws);

    Gradients bp;
    net.backward(params, batch, ws, bp);
    (void)net.forward(params, batch, ws);  // backward consumed ws.da/ws.dx
    Gradients edla;
    ed::Edla(*backend, ed::EdlaConfig{}, params.config)
        .local_update(params, batch, ws, edla);

    CHECK(edla.b2 == bp.b2);
    CHECK(edla.w2 == bp.w2);
    CHECK(edla.b1 == bp.b1);
    CHECK(edla.w1 == bp.w1);
    CHECK(edla.emb_rows == bp.emb_rows);
    CHECK(edla.emb_grads == bp.emb_grads);
}

TEST_CASE("the pure embedding rule diffuses without consulting W1") {
    const ExampleSet set = tiny_set();
    Parameters params = tiny_params();
    const auto backend = make_cpu_backend();
    Net net(*backend);
    Dataset data(set, params.config.embed_dim, 8);
    Batch batch;
    Workspace ws;
    data.fill_batch(0, params.embedding, batch);
    (void)net.forward(params, batch, ws);

    ed::EdlaConfig cfg;
    cfg.embedding_update = ed::EmbeddingUpdate::Pure;
    Gradients grads;
    ed::Edla(*backend, cfg, params.config).local_update(params, batch, ws, grads);

    // dv_c = dy · polarity(c, d) = −0.5 · [+1, −1], the same for both slots,
    // and row 2 fills both: [−1, +1]. Note it does not depend on W1 at all,
    // which is exactly the property being tested — and the reason the table
    // barely distinguishes characters under this rule.
    REQUIRE(grads.emb_rows.size() == 1);
    REQUIRE(grads.emb_grads.size() == 2);
    CHECK(grads.emb_grads[0] == doctest::Approx(-1.0));
    CHECK(grads.emb_grads[1] == doctest::Approx(1.0));

    // The layers above are untouched by the embedding choice.
    CHECK(grads.b1[0] == doctest::Approx(-0.5));
    CHECK(grads.b1[1] == doctest::Approx(0.5));
}

TEST_CASE("repeated EDLA updates drive the loss down") {
    // The hand-computed case fixes one step; this checks the steps compose in
    // the right direction, which a sign error that happens to cancel in the
    // fixture's particular numbers would not survive.
    const ExampleSet set = tiny_set();
    Parameters params = tiny_params();
    const auto backend = make_cpu_backend();
    Net net(*backend);
    ed::Edla edla(*backend, ed::EdlaConfig{}, params.config);
    Adam adam(params.config, AdamConfig{});
    Dataset data(set, params.config.embed_dim, 8);
    Batch batch;
    Workspace ws;
    Gradients grads;

    data.fill_batch(0, params.embedding, batch);
    const double first = net.forward(params, batch, ws);
    double last = first;
    for (int step = 0; step < 300; ++step) {
        data.fill_batch(0, params.embedding, batch);
        last = net.forward(params, batch, ws);
        edla.local_update(params, batch, ws, grads);
        adam.step(params, grads);
        ed::project_dale(params);
    }
    CHECK(last < first);
    CHECK(last < 0.1);  // the single example is separable; it should be learned
}

TEST_CASE("Dale's law is imposed at init and reimposed after each step") {
    const NetConfig config = tiny_config();
    const Parameters params = ed::init_parameters(config, 7);
    REQUIRE(params.w2.size() == 2);
    CHECK(params.w2[0] > 0.0f);   // unit 0 is excitatory
    CHECK(params.w2[1] < 0.0f);   // unit 1 is inhibitory
    CHECK(ed::count_pinned(params) == 0);

    // Same magnitudes as the MLP initializer draws, only the signs are forced.
    const Parameters mlp_init = Parameters::init(config, 7);
    for (std::size_t j = 0; j < params.w2.size(); ++j) {
        CHECK(std::abs(params.w2[j]) == doctest::Approx(std::abs(mlp_init.w2[j])));
    }

    // A weight that has crossed to the wrong side is pinned, not reflected:
    // reflecting would invent a value the optimizer never asked for.
    Parameters crossed = params;
    crossed.w2[0] = -0.3f;  // excitatory unit gone negative
    crossed.w2[1] = -0.2f;  // inhibitory unit still on its own side
    ed::project_dale(crossed);
    CHECK(crossed.w2[0] == 0.0f);
    CHECK(crossed.w2[1] == doctest::Approx(-0.2f));
    CHECK(ed::count_pinned(crossed) == 1);
}

TEST_CASE("Kolen-Pollack converges the feedback onto w2 without reading it") {
    Parameters params = tiny_params();
    const auto backend = make_cpu_backend();
    ed::EdlaConfig cfg;
    cfg.learn_feedback = true;
    cfg.feedback_decay = 0.1f;
    ed::Edla edla(*backend, cfg, params.config);

    // The feedback starts at the polarity, nowhere near w2 = {2, -4}.
    CHECK(edla.feedback()[0] == doctest::Approx(1.0f));
    CHECK(edla.feedback()[1] == doctest::Approx(-1.0f));

    // Drive it with a w2 that the "optimizer" leaves alone, so the only thing
    // acting is the shared decay: the gap must then shrink by exactly
    // (1 - decay) per step, which is the property the method rests on.
    const std::vector<float> before = params.w2;
    const float gap0 = edla.feedback()[0] - params.w2[0];
    edla.track_feedback(params, before);
    const float gap1 = edla.feedback()[0] - params.w2[0];
    CHECK(gap1 == doctest::Approx(gap0 * 0.9f));

    // Iterating closes it, and w2 decays alongside — both weights shrink by
    // their own value, which is what keeps the rule transport-free.
    for (int i = 0; i < 200; ++i) {
        const std::vector<float> w2_before = params.w2;
        edla.track_feedback(params, w2_before);
    }
    CHECK(edla.feedback()[0] == doctest::Approx(params.w2[0]).epsilon(0.01));
    CHECK(edla.feedback()[1] == doctest::Approx(params.w2[1]).epsilon(0.01));
    CHECK(std::abs(params.w2[0]) < std::abs(before[0]));  // decayed, not frozen

    // With learning off the feedback never moves: the ablation is exact.
    Parameters p2 = tiny_params();
    ed::Edla fixed(*backend, ed::EdlaConfig{}, p2.config);
    const std::vector<float> w2_before = p2.w2;
    p2.w2[0] = 99.0f;
    fixed.track_feedback(p2, w2_before);
    CHECK(fixed.feedback()[0] == doctest::Approx(1.0f));
    CHECK(p2.w2[0] == doctest::Approx(99.0f));  // untouched, no decay applied
}

TEST_CASE("polarity splits the hidden layer in half, contiguously") {
    CHECK(ed::polarity(0, 4) == 1.0f);
    CHECK(ed::polarity(1, 4) == 1.0f);
    CHECK(ed::polarity(2, 4) == -1.0f);
    CHECK(ed::polarity(3, 4) == -1.0f);
    // Odd widths give the inhibitory side the extra unit; nothing depends on
    // which way it rounds, only that the loader and the trainer agree, and
    // both call this one function.
    CHECK(ed::polarity(0, 3) == 1.0f);
    CHECK(ed::polarity(1, 3) == -1.0f);
}
