#include <doctest/doctest.h>

#include <cstdint>
#include <vector>

#include "segmentlib/bytes/binary_reader.h"
#include "segmentlib/bytes/binary_writer.h"

using namespace segmentlib::bytes;

TEST_CASE("integers round-trip through BinaryReader as little-endian") {
    BinaryWriter w;
    w.write<std::uint8_t>(0xAB);
    w.write<std::uint16_t>(0x1234);
    w.write<std::uint32_t>(0xDEADBEEF);
    w.write<std::int16_t>(-2);
    w.write<std::int32_t>(-100000);
    w.write<std::uint64_t>(0x0102030405060708ULL);

    // Spot-check the on-disk byte order (LE) before round-tripping.
    CHECK(w.data()[1] == std::byte{0x34});
    CHECK(w.data()[2] == std::byte{0x12});

    BinaryReader r(w.data());
    CHECK(r.read<std::uint8_t>() == 0xAB);
    CHECK(r.read<std::uint16_t>() == 0x1234);
    CHECK(r.read<std::uint32_t>() == 0xDEADBEEF);
    CHECK(r.read<std::int16_t>() == -2);
    CHECK(r.read<std::int32_t>() == -100000);
    CHECK(r.read<std::uint64_t>() == 0x0102030405060708ULL);
    CHECK(r.eof());
}

TEST_CASE("doubles round-trip as raw bytes") {
    BinaryWriter w;
    w.write(3.14159265358979);
    w.write(-0.5);
    BinaryReader r(w.data());
    CHECK(r.read<double>() == 3.14159265358979);
    CHECK(r.read<double>() == -0.5);
}

TEST_CASE("strings and header lines round-trip") {
    BinaryWriter w;
    w.write_line("SegmentLibMLP 1");
    w.write_cstring("日本語");
    w.write_cstring("");  // empty entry: just the NUL
    w.write_bool(true);
    w.write_bool(false);

    BinaryReader r(w.data());
    CHECK(r.read_line() == "SegmentLibMLP 1");
    CHECK(r.read_cstring() == "日本語");
    CHECK(r.read_cstring() == "");
    CHECK(r.read_bool());
    CHECK_FALSE(r.read_bool());
    CHECK(r.eof());
}

TEST_CASE("write_bytes appends raw bytes and take() empties the writer") {
    BinaryWriter w;
    const std::vector<std::byte> raw{std::byte{1}, std::byte{2}, std::byte{3}};
    w.write_bytes(raw);
    CHECK(w.size() == 3);

    const std::vector<std::byte> taken = w.take();
    CHECK(taken == raw);
    CHECK(w.size() == 0);
}
