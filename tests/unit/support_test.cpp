#include <doctest/doctest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "segmentlib/support/expected.h"
#include "segmentlib/support/span.h"

using namespace segmentlib;

namespace {

enum class Code { Bad, Worse };

struct Err {
    Code code;
    const char* msg;
};

// Stands in for Segmenter and Model, which are move-only by design.
struct MoveOnly {
    std::unique_ptr<int> p;

    MoveOnly() : p(std::make_unique<int>(0)) {}
    explicit MoveOnly(int v) : p(std::make_unique<int>(v)) {}
    MoveOnly(const MoveOnly&) = delete;
    MoveOnly& operator=(const MoveOnly&) = delete;
    MoveOnly(MoveOnly&&) = default;
    MoveOnly& operator=(MoveOnly&&) = default;
};

Expected<MoveOnly, Err> make_move_only(bool ok) {
    if (!ok) {
        return Unexpected(Err{Code::Bad, "nope"});
    }
    return MoveOnly(42);
}

Expected<void, Err> do_void(bool ok) {
    if (!ok) {
        return Unexpected(Err{Code::Worse, "void-nope"});
    }
    return {};
}

Expected<std::vector<int>, Err> make_vec(bool ok) {
    if (!ok) {
        return Unexpected(Err{Code::Bad, "v"});
    }
    return std::vector<int>{1, 2, 3};
}

}  // namespace

TEST_CASE("Expected carries the failure branch") {
    auto bad = make_move_only(false);
    CHECK_FALSE(bad);
    CHECK_FALSE(bad.has_value());
    CHECK(bad.error().code == Code::Bad);
    CHECK(std::string(bad.error().msg) == "nope");
}

TEST_CASE("Expected holds a move-only value") {
    auto good = make_move_only(true);
    REQUIRE(good);
    CHECK(*(*good).p == 42);
    CHECK(*good->p == 42);

    MoveOnly taken = *std::move(good);
    CHECK(*taken.p == 42);
}

TEST_CASE("Expected<void, E> reports success and failure") {
    auto bad = do_void(false);
    CHECK_FALSE(bad);
    CHECK(bad.error().code == Code::Worse);

    auto ok = do_void(true);
    CHECK(ok);
    *ok;  // no value to read, but the expression has to compile
}

TEST_CASE("Expected default-constructs into the value branch") {
    // What lets the batch tokenizer size its result vector up front and have
    // the workers overwrite each slot afterwards.
    std::vector<Expected<std::vector<int>, Err>> out(3);
    REQUIRE(out[0]);
    CHECK(out[0]->empty());

    out[1] = make_vec(true);
    out[2] = make_vec(false);

    REQUIRE(out[1]);
    CHECK(out[1]->size() == 3);
    REQUIRE_FALSE(out[2]);
    CHECK(out[2].error().code == Code::Bad);
}

TEST_CASE("Span views a contiguous container") {
    std::vector<int> data{10, 20, 30, 40};

    Span<const int> s(data);
    CHECK(s.size() == 4);
    CHECK(s.data() == data.data());
    CHECK(s[2] == 30);
    CHECK_FALSE(s.empty());

    int sum = 0;
    for (int x : s) {
        sum += x;
    }
    CHECK(sum == 100);

    // Copying a Span must pick the copy constructor, not the container one.
    Span<const int> copy(s);
    CHECK(copy.size() == 4);
}

TEST_CASE("Span writes through to its element type") {
    std::vector<int> data{1, 2};
    Span<int> s(data);
    s[0] = 11;
    CHECK(data[0] == 11);
}

TEST_CASE("Span is empty by default") {
    Span<const int> s;
    CHECK(s.empty());
    CHECK(s.size() == 0);
    CHECK(s.data() == nullptr);
}

TEST_CASE("as_bytes reinterprets a span as raw bytes") {
    std::array<unsigned char, 4> arr{1, 2, 3, 4};
    auto bytes = as_bytes(Span<const unsigned char>(arr));
    REQUIRE(bytes.size() == 4);
    CHECK(static_cast<unsigned char>(bytes[3]) == 4);

    // A multi-byte element type is what pins the * sizeof(T) in size_bytes();
    // every call site in the library happens to pass a one-byte element type.
    std::array<std::uint32_t, 2> words{0, 0};
    CHECK(as_bytes(Span<const std::uint32_t>(words)).size() == 8);
}
