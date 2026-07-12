#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <cstdio>
#include <filesystem>
#include <fstream>

#include "segmentlib/bytes/binary_writer.h"
#include "segmentlib/mlp/mlp_backend.h"
#include "segmentlib/mlp/model.h"
#include "segmentlib/mlp/scorer.h"
#include "segmentlib/segmenter.h"
#include "segmentlib/unicode/egc.h"

using namespace segmentlib;
using namespace segmentlib::mlp;

namespace {

// A fully hand-computed model: w=2 (4 slots), d=1, H=1, V=3 (PAD/UNK + あ),
// one dictionary containing ああ.
//
//   emb_q = [PAD=7, UNK=-50, あ=100],  w1_q = [3, -2, 4, -5] (slot j -> w1_q[j])
//   w2_q = [5],  b1 = 0.5,  b2 = -1.0,  wdict_q[k=5] = 10 (else 0)
//   S_e=0.01  S_w1=0.1  S_wd=0.002  S_w2=1.0  S_acc=0.001  →  R = 1.0
//
// Derived (I.1): b1_q = 500, b2_q = -1000, table[row][j] = w1_q[j]·emb_q[row],
// dict_col[5] = llround(2·10) = 20.
std::vector<std::byte> tiny_model_bytes() {
    bytes::BinaryWriter out;
    out.write_line("SegmentLibMLP 1");
    out.write<std::uint8_t>(2);    // w
    out.write<std::uint16_t>(1);   // d
    out.write<std::uint16_t>(1);   // H
    out.write<std::uint8_t>(1);    // num_dicts
    out.write<std::uint16_t>(unicode::kEgcUnicodeVersion);
    out.write<double>(0.01);   // emb
    out.write<double>(0.1);    // w1
    out.write<double>(0.002);  // wdict
    out.write<double>(1.0);    // w2
    out.write<double>(0.001);  // acc
    out.write<std::uint32_t>(3);       // V
    out.write<std::uint32_t>(0x3042);  // one real codepoint: あ
    for (const std::int16_t v : {7, -50, 100}) {  // embedding V×d
        out.write<std::int16_t>(v);
    }
    for (const std::int16_t v : {3, -2, 4, -5}) {  // W1 H×(2w·d)
        out.write<std::int16_t>(v);
    }
    for (std::size_t k = 0; k < 12; ++k) {  // W_dict H×12
        out.write<std::int16_t>(k == 5 ? 10 : 0);
    }
    out.write<double>(0.5);       // b1
    out.write<std::int16_t>(5);   // w2
    out.write<double>(-1.0);      // b2
    out.write<std::uint32_t>(1);  // dict 0: 1 entry
    out.write_cstring("ああ");
    return out.take();
}

}  // namespace

TEST_CASE("load rejects wrong signatures and malformed bodies") {
    const auto expect_code = [](std::span<const std::byte> bytes, ErrorCode code) {
        const auto model = Model::load_from_bytes(bytes);
        REQUIRE(!model.has_value());
        CHECK(model.error().code == code);
    };

    bytes::BinaryWriter kytea;
    kytea.write_line("KyTea 0.4.7");
    expect_code(kytea.data(), ErrorCode::UnsupportedModelFormat);

    bytes::BinaryWriter v2;
    v2.write_line("SegmentLibMLP 2");
    expect_code(v2.data(), ErrorCode::UnsupportedModelFormat);

    const std::vector<std::byte> valid = tiny_model_bytes();
    expect_code(std::span(valid).first(valid.size() - 10),
                ErrorCode::MalformedModel);  // truncated

    std::vector<std::byte> trailing = valid;
    trailing.push_back(std::byte{0});
    expect_code(trailing, ErrorCode::MalformedModel);
}

TEST_CASE("load parses every field and builds the derived integers") {
    const std::vector<std::byte> bytes = tiny_model_bytes();
    const auto model = Model::load_from_bytes(bytes, TablePrecision::Int32);
    REQUIRE(model.has_value());
    CHECK(model->table_precision() == TablePrecision::Int32);

    CHECK(model->config().char_window == 2);
    CHECK(model->config().embed_dim == 1);
    CHECK(model->config().hidden == 1);
    CHECK(model->config().num_dicts == 1);
    CHECK(!model->unicode_version_mismatch());

    CHECK(model->vocab().size() == 3);
    CHECK(model->vocab().row_of(U'あ') == 2);
    CHECK(model->vocab().row_of(U'い') == kUnkRow);

    REQUIRE(model->b1_q().size() == 1);
    CHECK(model->b1_q()[0] == 500);      // llround(0.5 / 0.001)
    CHECK(model->b2_q() == -1000);       // llround(-1.0 / (1.0 · 0.001))
    CHECK(model->dict_col(5)[0] == 20);  // llround((0.002/0.001) · 10)
    CHECK(model->dict_col(4)[0] == 0);
    REQUIRE(model->w2().size() == 1);
    CHECK(model->w2()[0] == 5);
}

TEST_CASE("scorer reproduces the hand computation, PAD and dict included") {
    // Int32: the exact path — this hand model's acc_scale is arbitrary (not
    // activation-calibrated), so only the exact path has predictable scores.
    const std::vector<std::byte> bytes = tiny_model_bytes();
    const auto model = Model::load_from_bytes(bytes, TablePrecision::Int32);
    REQUIRE(model.has_value());

    // ああ, boundary 0: slots cover EGCs -1,0,1,2 → PAD,あ,あ,PAD.
    //   acc = 500 + 3·7 + (-2)·100 + 4·100 + (-5)·7 = 686
    //   +20 (dict ああ spans the boundary: I, bucket 2 → feature 5) = 706
    //   y = 5·706 − 1000 = 2530
    const auto enc = model->vocab().encode("ああ");
    REQUIRE(enc.has_value());
    const std::vector<std::int32_t> scores = score_boundaries(*model, *enc);
    REQUIRE(scores.size() == 1);
    CHECK(scores[0] == 2530);

    // あい: い is UNK; no dictionary match.
    //   acc = 500 + 21 − 200 + 4·(−50) − 35 = 86;  y = 5·86 − 1000 = −570
    const auto enc2 = model->vocab().encode("あい");
    REQUIRE(enc2.has_value());
    const std::vector<std::int32_t> scores2 = score_boundaries(*model, *enc2);
    REQUIRE(scores2.size() == 1);
    CHECK(scores2[0] == -570);

    // あが (か + combining dakuten, both UNK): the second cluster has two
    // constituent rows and takes the fallback path; its mean equals the
    // single UNK row, so the score is bit-identical to あい's.
    const auto enc3 = model->vocab().encode("あが");
    REQUIRE(enc3.has_value());
    REQUIRE(enc3->egc_count() == 2);
    REQUIRE(enc3->egc_rows(1).size() == 2);
    const std::vector<std::int32_t> scores3 = score_boundaries(*model, *enc3);
    REQUIRE(scores3.size() == 1);
    CHECK(scores3[0] == scores2[0]);

    // Single cluster: no boundary candidates.
    const auto enc4 = model->vocab().encode("あ");
    REQUIRE(enc4.has_value());
    CHECK(score_boundaries(*model, *enc4).empty());
}

TEST_CASE("MlpBackend turns scores into boundaries and segments") {
    const std::vector<std::byte> bytes = tiny_model_bytes();
    auto model = Model::load_from_bytes(bytes, TablePrecision::Int32);
    REQUIRE(model.has_value());
    const MlpBackend backend(std::move(*model));

    const auto cuts = backend.tokenize_boundaries("ああ");
    REQUIRE(cuts.has_value());
    CHECK(*cuts == Boundaries{3});  // ああ splits between the two あ

    const auto segments = backend.tokenize("ああ");
    REQUIRE(segments.has_value());
    REQUIRE(segments->size() == 2);
    CHECK((*segments)[0] == Segment{0, 3, {}});
    CHECK((*segments)[1] == Segment{3, 6, {}});

    const auto none = backend.tokenize("あい");
    REQUIRE(none.has_value());
    REQUIRE(none->size() == 1);  // no cut: one segment, tags empty
    CHECK((*none)[0] == Segment{0, 6, {}});

    const auto empty = backend.tokenize("");
    REQUIRE(empty.has_value());
    CHECK(empty->empty());

    CHECK(!backend.tokenize("\xff").has_value());  // invalid UTF-8 propagates
}

TEST_CASE("Int16 mode derives the requantized quantities via requant_i16") {
    const std::vector<std::byte> bytes = tiny_model_bytes();
    const auto model = Model::load_from_bytes(bytes);  // Int16 is the default
    REQUIRE(model.has_value());
    CHECK(model->table_precision() == TablePrecision::Int16);

    // requant_i16 = round-to-nearest >>9 with saturation (kAccShift = 9):
    //   b1_q 500 → (500+256)>>9 = 1;  dict_col 20 → 0;
    //   b2_q −1000 → (−1000+256)>>9 = −2 (arithmetic shift floors).
    REQUIRE(model->b1_q16().size() == 1);
    CHECK(model->b1_q16()[0] == 1);
    CHECK(model->b2_q16() == -2);
    CHECK(model->dict_col16(5)[0] == 0);
    CHECK(requant_i16(400) == 1);
    CHECK(requant_i16(-200) == 0);
    CHECK(requant_i16(1 << 30) == 32767);    // saturates high
    CHECK(requant_i16(-(1 << 30)) == -32768);  // saturates low

    // The int16 scorer runs end to end on this model (decisions are not
    // asserted: an uncalibrated acc_scale makes the coarse unit meaningless;
    // decision agreement is tested on a really-trained model in
    // train_inference_test.cpp).
    const auto enc = model->vocab().encode("ああ");
    REQUIRE(enc.has_value());
    CHECK(score_boundaries(*model, *enc).size() == 1);
}

TEST_CASE("Segmenter::load auto-detects the MLP signature") {
    const std::vector<std::byte> bytes = tiny_model_bytes();
    const std::filesystem::path path = "segmentlib_autodetect_test.mlp";  // cwd
    {
        std::ofstream out(path, std::ios::binary);
        REQUIRE(out.good());
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
    }
    const auto segmenter = Segmenter::load(path);
    REQUIRE(segmenter.has_value());
    const auto cuts = segmenter->tokenize_boundaries("ああ");
    REQUIRE(cuts.has_value());
    CHECK(*cuts == Boundaries{3});

    // The MLP-forced loader accepts it too; the KyTea-forced loader must not.
    CHECK(Segmenter::load_mlp(path).has_value());
    CHECK(!Segmenter::load_kytea(path).has_value());
    std::filesystem::remove(path);
}

TEST_CASE("load rejects a non-ascending vocabulary") {
    // A model prefix is enough: parsing fails at the vocabulary.
    bytes::BinaryWriter out;
    out.write_line("SegmentLibMLP 1");
    out.write<std::uint8_t>(1);
    out.write<std::uint16_t>(1);
    out.write<std::uint16_t>(1);
    out.write<std::uint8_t>(0);
    out.write<std::uint16_t>(unicode::kEgcUnicodeVersion);
    for (int i = 0; i < 4; ++i) {
        out.write<double>(1.0);  // emb/w1/w2/acc scales (no dicts)
    }
    out.write<std::uint32_t>(4);       // V: two codepoints…
    out.write<std::uint32_t>(0x3044);  // い
    out.write<std::uint32_t>(0x3042);  // あ — descending: reject
    const auto model = Model::load_from_bytes(out.data());
    REQUIRE(!model.has_value());
    CHECK(model.error().code == ErrorCode::MalformedModel);
}
