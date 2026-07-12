#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>
#include <vector>

#include "segmentlib/kytea/model.h"

using namespace segmentlib;
using namespace segmentlib::kytea;

namespace {

struct Writer {
    std::vector<std::byte> bytes;
    template <class T>
    void put(T v) {
        for (std::size_t i = 0; i < sizeof(T); ++i) {
            bytes.push_back(static_cast<std::byte>(
                (static_cast<std::uint64_t>(v) >> (8 * i)) & 0xFF));
        }
    }
    void put_u8(std::uint8_t v) { bytes.push_back(static_cast<std::byte>(v)); }
    void put_f64(double v) {
        std::uint64_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        put<std::uint64_t>(bits);
    }
    void put_str(std::string_view s) {  // raw bytes + NUL (KyteaString cstring form)
        for (char c : s) bytes.push_back(static_cast<std::byte>(c));
        bytes.push_back(std::byte{0});
    }
    void put_empty_dict() {  // numDicts + numStates(0)
        put_u8(0);
        put<std::uint32_t>(0);
    }
    void put_featvec(std::span<const std::int16_t> v) {
        put<std::uint32_t>(static_cast<std::uint32_t>(v.size()));
        for (auto x : v) put<std::int16_t>(x);
    }
};

// A complete but minimal binary model: doWS, numTags=0, empty char/type/self
// dictionaries, and an empty word dictionary.
std::vector<std::byte> build_min_model(double multiplier,
                                       std::span<const std::int16_t> dict_vector,
                                       std::span<const std::int16_t> biases) {
    Writer w;
    w.bytes.reserve(64);
    // header
    w.put_str("KyTea 0.4.0 B utf8");  // put_str adds a NUL; header is read as a line...
    // ...but the file uses '\n', not NUL. Fix: overwrite the trailing NUL with '\n'.
    w.bytes.back() = std::byte{'\n'};

    // config
    w.put_u8(1);              // doWS
    w.put_u8(0);              // doTags
    w.put<std::uint32_t>(0);  // numTags
    w.put_u8(3);              // charWindow
    w.put_u8(3);              // charN
    w.put_u8(3);              // typeWindow
    w.put_u8(3);              // typeN
    w.put_u8(4);              // dictN
    w.put_u8(0);              // bias flag
    w.put_f64(0.0);           // epsilon
    w.put_u8(5);              // solver
    w.put_str("KTHRDO");      // char map (NUL-terminated)

    // wsModel
    w.put<std::int32_t>(2);   // numClasses
    w.put_u8(5);              // solver
    w.put<std::int32_t>(-1);  // label 0
    w.put<std::int32_t>(1);   // label 1
    w.put_u8(1);              // bias flag
    w.put_f64(multiplier);    // multiplier

    // FeatureLookup
    w.put_u8(1);              // active
    w.put_empty_dict();       // charDict
    w.put_empty_dict();       // typeDict
    w.put_empty_dict();       // selfDict
    w.put_featvec(dict_vector);
    w.put_featvec(biases);
    w.put_featvec({});        // tagDictVector
    w.put_featvec({});        // tagUnkVector

    // numTags == 0 -> no global tag models

    // word dictionary (empty)
    w.put_empty_dict();

    // subword dictionary (empty); numTags == 0 -> no reading language models
    w.put_empty_dict();

    return std::move(w.bytes);
}

}  // namespace

TEST_CASE("Model::load_from_bytes parses a minimal model") {
    std::vector<std::int16_t> dv{10, 20, 30};
    std::vector<std::int16_t> bs{-7};
    auto bytes = build_min_model(0.25, dv, bs);

    auto model = Model::load_from_bytes(bytes);
    REQUIRE(model.has_value());

    CHECK(model->config().do_ws == true);
    CHECK(model->config().num_tags == 0);
    CHECK(model->config().char_window == 3);
    CHECK(model->config().type_window == 3);
    CHECK(model->config().dict_n == 4);
    CHECK(model->config().solver == 5);

    CHECK(model->multiplier() == doctest::Approx(0.25));
    REQUIRE(model->biases().size() == 1);
    CHECK(model->biases()[0] == -7);
    REQUIRE(model->dict_vector().size() == 3);
    CHECK(model->dict_vector()[1] == 20);

    CHECK(model->char_dict().empty());
    CHECK(model->type_dict().empty());
    CHECK(model->word_dict().empty());
    CHECK(model->num_dicts() == 0);

    // The char map "KTHRDO" was interned (markers at ids 1..6).
    CHECK(model->chars().id_of(U'K') == 1);
    CHECK(model->chars().id_of(U'O') == 6);
}

TEST_CASE("Model rejects a bad header") {
    std::vector<std::byte> junk{std::byte{'x'}, std::byte{'\n'}, std::byte{0}};
    auto model = Model::load_from_bytes(junk);
    REQUIRE_FALSE(model.has_value());
    CHECK(model.error().code == ErrorCode::MalformedModel);
}

TEST_CASE("Model rejects a truncated model") {
    auto bytes = build_min_model(0.5, {}, {});
    bytes.resize(bytes.size() / 2);  // cut it off mid-structure
    auto model = Model::load_from_bytes(bytes);
    REQUIRE_FALSE(model.has_value());
    CHECK(model.error().code == ErrorCode::MalformedModel);
}
