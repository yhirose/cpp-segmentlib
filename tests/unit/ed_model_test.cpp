#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "segmentlib/bytes/binary_writer.h"
#include "segmentlib/ed/ed_backend.h"
#include "segmentlib/ed/model.h"
#include "segmentlib/mlp/dictionary.h"
#include "segmentlib/mlp/mlp_backend.h"
#include "segmentlib/mlp/model.h"
#include "segmentlib/segmenter.h"
#include "segmentlib/unicode/egc.h"

using namespace segmentlib;

namespace {

// The same hand-computed network as mlp_model_test.cpp — w=2 (4 slots), d=1,
// H=1, V=3 (PAD/UNK + あ), one dictionary containing ああ — written behind
// whichever signature the caller asks for. That the body is identical is the
// point: an EDLA model records the same tensors, learned by a different rule.
//
//   emb_q = [PAD=7, UNK=-50, あ=100],  w1_q = [3, -2, 4, -5] (slot j -> w1_q[j])
//   w2_q = [5],  b1 = 0.5,  b2 = -1.0,  wdict_q[k=5] = 10 (else 0)
//   S_e=0.01  S_w1=0.1  S_wd=0.002  S_w2=1.0  S_acc=0.001  →  R = 1.0
std::vector<std::byte> tiny_model_bytes(const std::string& header) {
    bytes::BinaryWriter out;
    out.write_line(header);
    out.write<std::uint8_t>(2);   // w
    out.write<std::uint16_t>(1);  // d
    out.write<std::uint16_t>(1);  // H
    out.write<std::uint8_t>(1);   // num_dicts
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
    out.write<double>(0.5);      // b1
    out.write<std::int16_t>(5);  // w2
    out.write<double>(-1.0);     // b2
    const std::vector<std::vector<std::string>> dicts = {{"ああ"}};
    write_compiled_dictionaries(out, mlp::compile_dictionaries(dicts));
    return out.take();
}

std::filesystem::path write_temp(const std::vector<std::byte>& bytes,
                                 const std::filesystem::path& path) {
    std::ofstream out(path, std::ios::binary);
    REQUIRE(out.good());
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    return path;
}

}  // namespace

TEST_CASE("ED and MLP models refuse each other's signatures") {
    const std::vector<std::byte> as_ed = tiny_model_bytes("SegmentLibED 1");
    const std::vector<std::byte> as_mlp = tiny_model_bytes("SegmentLibMLP 1");

    CHECK(ed::Model::load_from_bytes(as_ed).has_value());
    CHECK(mlp::Model::load_from_bytes(as_mlp).has_value());

    // Identical bodies, so only the header line can be doing the rejecting.
    const auto ed_reads_mlp = ed::Model::load_from_bytes(as_mlp);
    REQUIRE(!ed_reads_mlp.has_value());
    CHECK(ed_reads_mlp.error().code == ErrorCode::UnsupportedModelFormat);

    const auto mlp_reads_ed = mlp::Model::load_from_bytes(as_ed);
    REQUIRE(!mlp_reads_ed.has_value());
    CHECK(mlp_reads_ed.error().code == ErrorCode::UnsupportedModelFormat);
}

TEST_CASE("ED load rejects wrong versions and malformed bodies") {
    const auto expect_code = [](std::span<const std::byte> bytes, ErrorCode code) {
        const auto model = ed::Model::load_from_bytes(bytes);
        REQUIRE(!model.has_value());
        CHECK(model.error().code == code);
    };

    bytes::BinaryWriter v2;
    v2.write_line("SegmentLibED 2");
    expect_code(v2.data(), ErrorCode::UnsupportedModelFormat);

    // Signature and version accepted, so what follows is malformed rather than
    // unsupported: the two are distinct checks.
    bytes::BinaryWriter empty;
    empty.write_line("SegmentLibED 1");
    expect_code(empty.data(), ErrorCode::MalformedModel);

    const std::vector<std::byte> valid = tiny_model_bytes("SegmentLibED 1");
    expect_code(std::span(valid).first(valid.size() - 10), ErrorCode::MalformedModel);

    std::vector<std::byte> trailing = valid;
    trailing.push_back(std::byte{0});
    expect_code(trailing, ErrorCode::MalformedModel);
}

TEST_CASE("EdBackend segments identically to MlpBackend on the same weights") {
    // The contract that lets the two backends share a scoring core: with the
    // weights held fixed, the only thing an EDLA model changes is how they were
    // learned, so any measured difference between the backends must come from
    // training rather than from inference.
    auto ed_model = ed::Model::load_from_bytes(tiny_model_bytes("SegmentLibED 1"),
                                               mlp::TablePrecision::Int32);
    auto mlp_model = mlp::Model::load_from_bytes(tiny_model_bytes("SegmentLibMLP 1"),
                                                 mlp::TablePrecision::Int32);
    REQUIRE(ed_model.has_value());
    REQUIRE(mlp_model.has_value());
    const ed::EdBackend ed_backend(std::move(*ed_model));
    const mlp::MlpBackend mlp_backend(std::move(*mlp_model));

    for (const std::string_view text : {"ああ", "あい", "あが", "あ", "", "ああああ"}) {
        const auto from_ed = ed_backend.tokenize(text);
        const auto from_mlp = mlp_backend.tokenize(text);
        REQUIRE(from_ed.has_value());
        REQUIRE(from_mlp.has_value());
        CHECK(*from_ed == *from_mlp);
    }

    // ああ cuts between the two あ (y = 5·706 − 1000 = 2530 > 0), as computed
    // by hand in mlp_model_test.cpp.
    const auto segments = ed_backend.tokenize("ああ");
    REQUIRE(segments.has_value());
    REQUIRE(segments->size() == 2);
    CHECK((*segments)[0] == std::pair<std::size_t, std::size_t>{0, 3});
    CHECK((*segments)[1] == std::pair<std::size_t, std::size_t>{3, 6});

    CHECK(!ed_backend.tokenize("\xff").has_value());  // invalid UTF-8 propagates
}

TEST_CASE("Segmenter::load auto-detects the ED signature") {
    const std::filesystem::path path =
        write_temp(tiny_model_bytes("SegmentLibED 1"), "segmentlib_autodetect_test.ed");

    const auto segmenter = Segmenter::load(path);
    REQUIRE(segmenter.has_value());
    const auto segments = segmenter->tokenize("ああ");
    REQUIRE(segments.has_value());
    REQUIRE(segments->size() == 2);
    CHECK((*segments)[0] == std::pair<std::size_t, std::size_t>{0, 3});

    // The ED-forced loader accepts it; the other two must not.
    CHECK(Segmenter::load_ed(path).has_value());
    CHECK(!Segmenter::load_mlp(path).has_value());
    CHECK(!Segmenter::load_kytea(path).has_value());
    std::filesystem::remove(path);
}

TEST_CASE("signature sniffing survives files shorter than a signature") {
    // "SegmentLibED " is one byte shorter than "SegmentLibMLP ", so load()
    // reads the longer one and may come back short. A file shorter than either
    // must simply match neither and fall through to KyTea's diagnostics, not
    // compare against bytes that were never read.
    const std::filesystem::path path = "segmentlib_short_test.mod";
    {
        std::ofstream out(path, std::ios::binary);
        REQUIRE(out.good());
        out << "Segm";
    }
    const auto segmenter = Segmenter::load(path);
    REQUIRE(!segmenter.has_value());
    CHECK(segmenter.error().code != ErrorCode::IoError);  // parsed, not unreadable
    std::filesystem::remove(path);

    // An empty file is the same story with nothing read at all.
    const std::filesystem::path empty_path = "segmentlib_empty_test.mod";
    { std::ofstream out(empty_path, std::ios::binary); }
    CHECK(!Segmenter::load(empty_path).has_value());
    std::filesystem::remove(empty_path);
}
