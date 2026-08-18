#include <doctest/doctest.h>

#include <cstdint>

#include "mlp/train/sgd.h"

using namespace segmentlib::mlp::train;

TEST_CASE("Sgd applies params -= lr * grads, dense and sparse alike") {
    NetConfig config;
    config.window = 1;
    config.embed_dim = 2;
    config.hidden = 2;
    config.vocab_size = 3;
    config.num_dicts = 1;

    Parameters p;
    p.config = config;
    p.embedding = {0.f, 0.f, 0.f, 0.f, 1.f, 2.f};  // 3 rows × d=2
    p.w1.assign(2 * 4, 1.0f);                      // H=2 × 2w·d=4
    p.wdict.assign(2 * 12, 0.5f);                  // H=2 × Fd=12
    p.b1 = {1.0f, -1.0f};
    p.w2 = {2.0f, -2.0f};
    p.b2 = 0.25f;

    Gradients g;
    g.w1.assign(8, 2.0f);
    g.wdict.assign(24, 0.0f);
    g.wdict[5] = 4.0f;
    g.b1 = {8.0f, -8.0f};
    g.w2 = {1.0f, 1.0f};
    g.b2 = 2.0f;
    g.emb_rows = {2};  // only row 2 was touched
    g.emb_grads = {10.0f, -10.0f};

    Sgd(0.25f).step(p, g);

    CHECK(p.w1[0] == doctest::Approx(0.5f));    // 1 − 0.25·2
    CHECK(p.wdict[5] == doctest::Approx(-0.5f));
    CHECK(p.wdict[4] == doctest::Approx(0.5f));  // untouched
    CHECK(p.b1[0] == doctest::Approx(-1.0f));
    CHECK(p.b1[1] == doctest::Approx(1.0f));
    CHECK(p.w2[0] == doctest::Approx(1.75f));
    CHECK(p.b2 == doctest::Approx(-0.25f));

    // The sparse rows: row 2 moved, rows 0/1 did not.
    CHECK(p.embedding[4] == doctest::Approx(-1.5f));  // 1 − 0.25·10
    CHECK(p.embedding[5] == doctest::Approx(4.5f));   // 2 + 0.25·10
    CHECK(p.embedding[0] == 0.0f);
    CHECK(p.embedding[2] == 0.0f);
}
