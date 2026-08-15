#include <doctest/doctest.h>

#include <array>
#include <cstring>
#include <span>
#include <string_view>

#include "segmentlib/bytes/binary_reader.h"

using namespace segmentlib::bytes;

namespace {
template <std::size_t N>
std::span<const std::byte> bytes_of(const std::array<unsigned char, N>& a) {
    return std::as_bytes(std::span(a));
}
}  // namespace

TEST_CASE("read u32 as little-endian") {
    std::array<unsigned char, 4> buf{0x2c, 0x01, 0x00, 0x00};  // 300
    BinaryReader r(bytes_of(buf));
    CHECK(r.read<std::uint32_t>() == 300u);
    CHECK(r.eof());
}

TEST_CASE("read i32 two's complement") {
    std::array<unsigned char, 4> buf{0xff, 0xff, 0xff, 0xff};
    BinaryReader r(bytes_of(buf));
    CHECK(r.read<std::int32_t>() == -1);
}

TEST_CASE("read u16 (KyteaChar width)") {
    std::array<unsigned char, 2> buf{0x34, 0x12};
    BinaryReader r(bytes_of(buf));
    CHECK(r.read<std::uint16_t>() == 0x1234u);
}

TEST_CASE("read bool") {
    std::array<unsigned char, 2> buf{0x00, 0x01};
    BinaryReader r(bytes_of(buf));
    CHECK(r.read_bool() == false);
    CHECK(r.read_bool() == true);
}

TEST_CASE("read NUL-terminated string") {
    std::array<unsigned char, 5> buf{'a', 'b', 'c', 0, 'x'};
    BinaryReader r(bytes_of(buf));
    CHECK(r.read_cstring() == "abc");
    CHECK(r.remaining() == 1);  // 'x' still unread
}

TEST_CASE("unterminated string throws") {
    std::array<unsigned char, 3> buf{'a', 'b', 'c'};
    BinaryReader r(bytes_of(buf));
    CHECK_THROWS_AS((void)r.read_cstring(), ParseError);
}

TEST_CASE("read_line strips the newline") {
    std::string_view hdr = "KyTea 0.4.0 B utf8\n";
    std::array<unsigned char, 32> buf{};
    std::memcpy(buf.data(), hdr.data(), hdr.size());
    BinaryReader r(std::as_bytes(std::span(buf.data(), hdr.size())));
    CHECK(r.read_line() == "KyTea 0.4.0 B utf8");
    CHECK(r.eof());
}

TEST_CASE("read past end throws") {
    std::array<unsigned char, 2> buf{0, 0};
    BinaryReader r(bytes_of(buf));
    CHECK_THROWS_AS((void)r.read<std::uint32_t>(), ParseError);
}

TEST_CASE("skip advances the cursor") {
    std::array<unsigned char, 4> buf{0, 0, 0x2a, 0};
    BinaryReader r(bytes_of(buf));
    r.skip(2);
    CHECK(r.read<std::uint8_t>() == 42u);
    CHECK_THROWS_AS(r.skip(100), ParseError);
}

TEST_CASE("require_capacity rejects counts the buffer cannot back") {
    std::array<unsigned char, 8> buf{};
    BinaryReader r(bytes_of(buf));

    CHECK_NOTHROW(r.require_capacity(4, 2));  // exactly fits
    CHECK_NOTHROW(r.require_capacity(0, 4));
    CHECK_THROWS_AS(r.require_capacity(5, 2), ParseError);

    // The point of the check: a huge count must be rejected before it is used
    // to size a container, not after the first read runs off the end.
    CHECK_THROWS_AS(r.require_capacity(0xFFFFFFFFu, 4), ParseError);
    // count * min_bytes_each must not wrap; the division form avoids it.
    CHECK_THROWS_AS(r.require_capacity(0xFFFFFFFFFFFFFFFFull, 8), ParseError);

    // A zero element size carries no information, so it cannot reject.
    CHECK_NOTHROW(r.require_capacity(0xFFFFFFFFu, 0));

    r.skip(8);  // exhausted: only a zero count is still satisfiable
    CHECK_NOTHROW(r.require_capacity(0, 1));
    CHECK_THROWS_AS(r.require_capacity(1, 1), ParseError);
}
