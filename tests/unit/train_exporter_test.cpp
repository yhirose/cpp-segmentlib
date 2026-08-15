#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "mlp/train/exporter.h"
#include "mlp/train/quantize.h"
#include "segmentlib/bytes/binary_reader.h"
#include "segmentlib/mlp/dictionary.h"
#include "segmentlib/mlp/model.h"
#include "segmentlib/mlp/vocab.h"
#include "segmentlib/unicode/egc.h"

using namespace segmentlib;
using namespace segmentlib::mlp::train;
using segmentlib::mlp::Vocab;

namespace {

// A tiny fully hand-specified model: w=1, d=2, H=2, V=4 (PAD/UNK + 2
// codepoints), one dictionary.
QuantizedModel tiny_model() {
    QuantizedModel q;
    q.config.window = 1;
    q.config.embed_dim = 2;
    q.config.hidden = 2;
    q.config.vocab_size = 4;
    q.config.num_dicts = 1;
    q.embedding = {0, 1, 2, 3, 4, 5, 6, 7};             // 4×2
    q.w1 = {10, 11, 12, 13, 14, 15, 16, 17};            // 2×(2·1·2)
    q.wdict.assign(2 * 12, 0);
    q.wdict[0] = -5;
    q.wdict[23] = 9;
    q.w2 = {-100, 100};
    q.emb_scale = 0.5;
    q.w1_scale = 0.25;
    q.wdict_scale = 0.125;
    q.w2_scale = 2.0;
    q.acc_scale = 1e-4;
    q.b1 = {0.5, -1.5};
    q.b2 = 3.25;
    return q;
}

}  // namespace

TEST_CASE("serialize_model writes the 5.7 layout byte for byte") {
    const QuantizedModel q = tiny_model();
    const Vocab vocab{{U'あ', U'い'}};
    const std::vector<std::vector<std::string>> dicts = {{"かき", "を"}};

    const std::vector<std::byte> bytes = serialize_model(q, vocab, dicts);
    bytes::BinaryReader in(bytes);

    CHECK(in.read_line() == "SegmentLibMLP 1");

    // Config (fields 1-4b)
    CHECK(in.read<std::uint8_t>() == 1);    // w
    CHECK(in.read<std::uint16_t>() == 2);   // d
    CHECK(in.read<std::uint16_t>() == 2);   // H
    CHECK(in.read<std::uint8_t>() == 1);    // num_dicts
    CHECK(in.read<std::uint16_t>() == unicode::kEgcUnicodeVersion);

    // Scales (fields 5-8b)
    CHECK(in.read<double>() == 0.5);
    CHECK(in.read<double>() == 0.25);
    CHECK(in.read<double>() == 0.125);  // wdict_scale present (num_dicts>0)
    CHECK(in.read<double>() == 2.0);
    CHECK(in.read<double>() == 1e-4);

    // Vocabulary (fields 9-10)
    CHECK(in.read<std::uint32_t>() == 4);
    CHECK(in.read<std::uint32_t>() == 0x3042);  // あ
    CHECK(in.read<std::uint32_t>() == 0x3044);  // い

    // Embedding, W1, W_dict (fields 11-13)
    for (const std::int16_t expected : q.embedding) {
        CHECK(in.read<std::int16_t>() == expected);
    }
    for (const std::int16_t expected : q.w1) {
        CHECK(in.read<std::int16_t>() == expected);
    }
    for (const std::int16_t expected : q.wdict) {
        CHECK(in.read<std::int16_t>() == expected);
    }

    // b1 (double × H), w2, b2 (fields 14-16)
    CHECK(in.read<double>() == 0.5);
    CHECK(in.read<double>() == -1.5);
    CHECK(in.read<std::int16_t>() == -100);
    CHECK(in.read<std::int16_t>() == 100);
    CHECK(in.read<double>() == 3.25);

    // Dictionaries (field 17): the compiled FST, then the channel sets its
    // outputs index. Both entries are in channel 0, so there is one set.
    const std::uint32_t fst_size = in.read<std::uint32_t>();
    CHECK(fst_size > 0);
    const std::string fst = in.read_blob(fst_size);
    CHECK(in.read<std::uint32_t>() == 1);  // one distinct channel set
    CHECK(in.read<std::uint32_t>() == 0);  // offsets[0]
    CHECK(in.read<std::uint32_t>() == 1);  // offsets[1]
    CHECK(in.read<std::uint8_t>() == 0);   // that set holds channel 0
    CHECK(in.eof());

    // The blob is the compiler's output verbatim, which is what lets the
    // loader use it without rebuilding anything.
    CHECK(segmentlib::mlp::compile_dictionaries(dicts).fst == fst);
}

TEST_CASE("no dictionaries: wdict scale, tensor and section are absent") {
    QuantizedModel q = tiny_model();
    q.config.num_dicts = 0;
    q.wdict.clear();
    const Vocab vocab{{U'あ', U'い'}};

    const std::vector<std::byte> bytes = serialize_model(q, vocab, {});
    bytes::BinaryReader in(bytes);

    CHECK(in.read_line() == "SegmentLibMLP 1");
    in.skip(1 + 2 + 2 + 1 + 2);          // config
    CHECK(in.read<double>() == 0.5);     // emb
    CHECK(in.read<double>() == 0.25);    // w1
    CHECK(in.read<double>() == 2.0);     // w2 (no wdict scale in between)
    CHECK(in.read<double>() == 1e-4);    // acc
    in.skip(4 + 2 * 4);                  // vocab
    in.skip(q.embedding.size() * 2);
    in.skip(q.w1.size() * 2);            // no wdict tensor
    in.skip(2 * 8);                      // b1
    in.skip(2 * 2);                      // w2
    in.skip(8);                          // b2
    CHECK(in.eof());                     // and no dictionary section
}

TEST_CASE("export_model writes the same bytes to a file") {
    const QuantizedModel q = tiny_model();
    const Vocab vocab{{U'あ', U'い'}};
    const std::vector<std::vector<std::string>> dicts = {{"かき", "を"}};
    const std::vector<std::byte> expected = serialize_model(q, vocab, dicts);

    const std::filesystem::path path = "segmentlib_export_test.mlp";  // cwd
    REQUIRE(export_model(path, q, vocab, dicts).has_value());

    std::ifstream in(path, std::ios::binary);
    REQUIRE(in.good());
    std::vector<char> actual((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
    REQUIRE(actual.size() == expected.size());
    CHECK(std::memcmp(actual.data(), expected.data(), actual.size()) == 0);
    std::filesystem::remove(path);
}

TEST_CASE("a dictionary channel with no usable entries still round-trips") {
    // An empty --dict file, or one whose entries are all unusable, leaves a
    // channel with nothing in it. The compiled form still has to be a valid
    // CSR, or the trainer writes a model its own loader rejects.
    const QuantizedModel q = tiny_model();
    const Vocab vocab{{U'あ', U'い'}};
    for (const auto& dicts : {std::vector<std::vector<std::string>>{{}},
                              std::vector<std::vector<std::string>>{{""}},
                              std::vector<std::vector<std::string>>{{"\xff\xfe"}}}) {
        const auto compiled = segmentlib::mlp::compile_dictionaries(dicts);
        CHECK(compiled.offsets.size() == 1);
        CHECK(compiled.offsets.front() == 0);
        const auto model =
            segmentlib::mlp::Model::load_from_bytes(serialize_model(q, vocab, dicts));
        REQUIRE(model.has_value());
        CHECK(model->config().num_dicts == 1);
    }
}
