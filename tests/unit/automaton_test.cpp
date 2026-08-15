#include <doctest/doctest.h>

#include <cstdint>
#include <span>
#include <vector>

#include "segmentlib/bytes/binary_reader.h"
#include "segmentlib/kytea/automaton.h"

using namespace segmentlib::kytea;
using namespace segmentlib::bytes;

namespace {

// Appends little-endian integers to a byte vector, mirroring KyTea's
// writeBinary, so a Dictionary can be assembled by hand.
struct Writer {
    std::vector<std::byte> bytes;
    template <class T>
    void put(T v) {
        for (std::size_t i = 0; i < sizeof(T); ++i) {
            bytes.push_back(static_cast<std::byte>((static_cast<std::uint64_t>(v) >> (8 * i)) & 0xFF));
        }
    }
    void put_u8(std::uint8_t v) { bytes.push_back(static_cast<std::byte>(v)); }
};

// Builds the automaton for patterns {"5", "7", "57"} over the CharId alphabet
// {5, 7}, with entries 100 ("5"), 101 ("7"), 102 ("57"). Failure links and
// propagated outputs are precomputed exactly as KyTea would serialize them.
//
//   state 0 (root): gotos 5->1, 7->2
//   state 1 ("5"):  goto 7->3, output {0}
//   state 2 ("7"):  output {1}, failure 0
//   state 3 ("57"): output {2, 1}  (also emits "7" via suffix link), failure 2
std::vector<std::byte> build_automaton() {
    Writer w;
    w.put_u8(1);           // numDicts
    w.put<std::uint32_t>(4);  // numStates

    // state 0
    w.put<std::uint32_t>(0);  // failure
    w.put<std::uint32_t>(2);  // 2 gotos
    w.put<std::uint16_t>(5); w.put<std::uint32_t>(1);
    w.put<std::uint16_t>(7); w.put<std::uint32_t>(2);
    w.put<std::uint32_t>(0);  // 0 outputs
    w.put_u8(0);              // isBranch

    // state 1 ("5")
    w.put<std::uint32_t>(0);  // failure
    w.put<std::uint32_t>(1);  // 1 goto
    w.put<std::uint16_t>(7); w.put<std::uint32_t>(3);
    w.put<std::uint32_t>(1);  // 1 output
    w.put<std::uint32_t>(0);  // -> entry 0
    w.put_u8(1);              // isBranch

    // state 2 ("7")
    w.put<std::uint32_t>(0);  // failure
    w.put<std::uint32_t>(0);  // 0 gotos
    w.put<std::uint32_t>(1);  // 1 output
    w.put<std::uint32_t>(1);  // -> entry 1
    w.put_u8(1);

    // state 3 ("57")
    w.put<std::uint32_t>(2);  // failure -> state 2
    w.put<std::uint32_t>(0);  // 0 gotos
    w.put<std::uint32_t>(2);  // 2 outputs
    w.put<std::uint32_t>(2);  // -> entry 2 ("57")
    w.put<std::uint32_t>(1);  // -> entry 1 ("7", via suffix link)
    w.put_u8(1);

    // entries: 3 payloads, each a single uint32
    w.put<std::uint32_t>(3);
    w.put<std::uint32_t>(100);
    w.put<std::uint32_t>(101);
    w.put<std::uint32_t>(102);

    return std::move(w.bytes);
}

// Builds the automaton for {"ABC" -> 200, "B" -> 201} over the alphabet
// {A=1, B=2, C=3}. The point of interest is state "AB": a non-word prefix of
// "ABC" whose failure link points at the word "B", so buildFailures propagates
// "B"'s output onto it. It therefore has a non-empty output but isBranch == 0 —
// exactly the case find_entry must reject (an exact walk of "AB" is not a word).
//
//   state 0 (root): gotos A->1, B->2
//   state 1 ("A"):  goto B->3, no output, isBranch 0
//   state 2 ("B"):  output {1}=entry 201, isBranch 1
//   state 3 ("AB"): goto C->4, output {1} (from failure 2), isBranch 0
//   state 4 ("ABC"):output {0}=entry 200, isBranch 1
std::vector<std::byte> build_branch_automaton() {
    Writer w;
    w.put_u8(1);              // numDicts
    w.put<std::uint32_t>(5);  // numStates

    // state 0 (root)
    w.put<std::uint32_t>(0);  // failure
    w.put<std::uint32_t>(2);  // 2 gotos
    w.put<std::uint16_t>(1); w.put<std::uint32_t>(1);  // A->1
    w.put<std::uint16_t>(2); w.put<std::uint32_t>(2);  // B->2
    w.put<std::uint32_t>(0);  // 0 outputs
    w.put_u8(0);              // isBranch

    // state 1 ("A")
    w.put<std::uint32_t>(0);  // failure
    w.put<std::uint32_t>(1);  // 1 goto
    w.put<std::uint16_t>(2); w.put<std::uint32_t>(3);  // B->3
    w.put<std::uint32_t>(0);  // 0 outputs
    w.put_u8(0);              // isBranch (prefix, not a word)

    // state 2 ("B")
    w.put<std::uint32_t>(0);  // failure
    w.put<std::uint32_t>(0);  // 0 gotos
    w.put<std::uint32_t>(1);  // 1 output
    w.put<std::uint32_t>(1);  // -> entry 1 (201)
    w.put_u8(1);              // isBranch

    // state 3 ("AB")
    w.put<std::uint32_t>(2);  // failure -> state 2 ("B")
    w.put<std::uint32_t>(1);  // 1 goto
    w.put<std::uint16_t>(3); w.put<std::uint32_t>(4);  // C->4
    w.put<std::uint32_t>(1);  // 1 output (propagated from "B")
    w.put<std::uint32_t>(1);  // -> entry 1 (201)
    w.put_u8(0);              // isBranch == 0 (NOT a word)

    // state 4 ("ABC")
    w.put<std::uint32_t>(0);  // failure
    w.put<std::uint32_t>(0);  // 0 gotos
    w.put<std::uint32_t>(1);  // 1 output
    w.put<std::uint32_t>(0);  // -> entry 0 (200)
    w.put_u8(1);              // isBranch

    // entries: 2 payloads
    w.put<std::uint32_t>(2);
    w.put<std::uint32_t>(200);
    w.put<std::uint32_t>(201);

    return std::move(w.bytes);
}

Automaton<int> parse(const std::vector<std::byte>& bytes) {
    std::span<const std::byte> sp(bytes);
    BinaryReader r(sp);
    return Automaton<int>::read(r, [](BinaryReader& rr) {
        return static_cast<int>(rr.read<std::uint32_t>());
    });
}

}  // namespace

TEST_CASE("parse recovers states and payloads") {
    auto a = parse(build_automaton());
    CHECK(a.num_states() == 4);
    CHECK(a.num_payloads() == 3);
    CHECK(a.num_dicts() == 1);
    CHECK(a.payload(0) == 100);
    CHECK(a.payload(2) == 102);
}

TEST_CASE("failure fallback after a longer match") {
    // {5,7,5}: "57" matches at position 1 (also emitting "7" via suffix link);
    // then at position 2 the automaton falls back 3->2->0 before matching "5".
    auto a = parse(build_automaton());
    std::vector<CharId> text{5, 7, 5};
    auto m = a.match_all(text);
    REQUIRE(m.size() == 4);
    CHECK(m[0].end_pos == 0);
    CHECK(*m[0].payload == 100);  // "5"
    CHECK(m[1].end_pos == 1);
    CHECK(*m[1].payload == 102);  // "57"
    CHECK(m[2].end_pos == 1);
    CHECK(*m[2].payload == 101);  // "7" (suffix link)
    CHECK(m[3].end_pos == 2);
    CHECK(*m[3].payload == 100);  // "5" (after failure fallback)
}

TEST_CASE("clean failure fallback with no long match") {
    // {7,5}: "7" at 0, then step(2,5)=0 forces fallback to root before "5".
    auto a = parse(build_automaton());
    std::vector<CharId> text{7, 5};
    auto m = a.match_all(text);
    REQUIRE(m.size() == 2);
    CHECK(*m[0].payload == 101);  // "7"
    CHECK(m[0].end_pos == 0);
    CHECK(*m[1].payload == 100);  // "5"
    CHECK(m[1].end_pos == 1);
}

TEST_CASE("multi-character match emits propagated outputs") {
    auto a = parse(build_automaton());
    std::vector<CharId> text{5, 7};
    auto m = a.match_all(text);
    REQUIRE(m.size() == 3);
    CHECK(m[0].end_pos == 0);
    CHECK(*m[0].payload == 100);  // "5"
    CHECK(m[1].end_pos == 1);
    CHECK(*m[1].payload == 102);  // "57"
    CHECK(m[2].end_pos == 1);
    CHECK(*m[2].payload == 101);  // "7" via suffix link, same position
}

TEST_CASE("failure link then re-match") {
    auto a = parse(build_automaton());
    std::vector<CharId> text{7, 5, 7};
    auto m = a.match_all(text);
    REQUIRE(m.size() == 4);
    CHECK(*m[0].payload == 101);  // "7"   @0
    CHECK(*m[1].payload == 100);  // "5"   @1
    CHECK(*m[2].payload == 102);  // "57"  @2
    CHECK(*m[3].payload == 101);  // "7"   @2
    CHECK(m[3].end_pos == 2);
}

TEST_CASE("unknown characters never match") {
    auto a = parse(build_automaton());
    std::vector<CharId> text{9, 9, 9};  // not in the alphabet
    CHECK(a.match_all(text).empty());
}

TEST_CASE("find_entry returns exact whole-string matches") {
    auto a = parse(build_automaton());  // {"5"->100, "7"->101, "57"->102}
    const std::vector<CharId> five{5};
    const std::vector<CharId> seven{7};
    const std::vector<CharId> fiveseven{5, 7};
    REQUIRE(a.find_entry(five) != nullptr);
    CHECK(*a.find_entry(five) == 100);
    REQUIRE(a.find_entry(seven) != nullptr);
    CHECK(*a.find_entry(seven) == 101);
    // "57" must return its own entry (output[0]), not the "7" propagated onto it.
    REQUIRE(a.find_entry(fiveseven) != nullptr);
    CHECK(*a.find_entry(fiveseven) == 102);
}

TEST_CASE("find_entry rejects non-matches and the empty string") {
    auto a = parse(build_automaton());
    const std::vector<CharId> unknown{9};       // no root goto
    const std::vector<CharId> falls_off{5, 7, 5};  // walk leaves the trie
    const std::vector<CharId> empty{};
    CHECK(a.find_entry(unknown) == nullptr);
    CHECK(a.find_entry(falls_off) == nullptr);
    CHECK(a.find_entry(empty) == nullptr);
}

TEST_CASE("find_entry rejects a prefix state carrying a propagated output") {
    auto a = parse(build_branch_automaton());  // {"ABC"->200, "B"->201}
    const std::vector<CharId> abc{1, 2, 3};
    const std::vector<CharId> b{2};
    const std::vector<CharId> ab{1, 2};  // non-word prefix with "B"'s output
    const std::vector<CharId> aonly{1};  // prefix with no output
    REQUIRE(a.find_entry(abc) != nullptr);
    CHECK(*a.find_entry(abc) == 200);
    REQUIRE(a.find_entry(b) != nullptr);
    CHECK(*a.find_entry(b) == 201);
    // "AB" has a non-empty (propagated) output but is not a word: the isBranch
    // guard must reject it.
    CHECK(a.find_entry(ab) == nullptr);
    CHECK(a.find_entry(aonly) == nullptr);
}

TEST_CASE("find_entry on an empty automaton is always null") {
    Writer w;
    w.put_u8(0);
    w.put<std::uint32_t>(0);
    auto a = parse(w.bytes);
    const std::vector<CharId> any{1, 2, 3};
    CHECK(a.find_entry(any) == nullptr);
}

TEST_CASE("absent dictionary parses to empty automaton") {
    Writer w;
    w.put_u8(0);              // numDicts
    w.put<std::uint32_t>(0);  // numStates == 0 -> no entries follow
    std::span<const std::byte> sp(w.bytes);
    BinaryReader r(sp);
    auto a = Automaton<int>::read(r, [](BinaryReader& rr) {
        return static_cast<int>(rr.read<std::uint32_t>());
    });
    CHECK(a.empty());
    CHECK(r.eof());  // nothing beyond the two header fields was consumed
    std::vector<CharId> text{1, 2, 3};
    CHECK(a.match_all(text).empty());
}

// Malformed-input rejection. Each case below is a two-state automaton that is
// well-formed except for one field; before these were validated, build_canonical
// indexed new_id/payloads_ with the bad value and wrote out of bounds at an
// offset the file chooses (confirmed under ASan: wild writes in the child
// placement and the failure-link remap, and a heap-buffer-overflow when sizing
// root_next_). They must parse to a ParseError instead.
namespace {

// root --c=2--> state 1, one payload, one output on state 1. The defaults are
// well-formed; each SUBCASE below corrupts exactly one member.
struct Fixture {
    std::uint32_t root_failure = 0;
    std::uint32_t leaf_output = 0;
    std::vector<std::pair<CharId, std::uint32_t>> root_gotos{{2, 1}};
    // States written after state 1 and counted in the header, but named by no
    // goto, i.e. unreachable from the root.
    std::uint32_t extra_states = 0;
};

std::vector<std::byte> build_fixture(const Fixture& c) {
    Writer w;
    w.put_u8(1);
    w.put<std::uint32_t>(2 + c.extra_states);

    // state 0 (root)
    w.put<std::uint32_t>(c.root_failure);
    w.put<std::uint32_t>(static_cast<std::uint32_t>(c.root_gotos.size()));
    for (const auto& [ch, next] : c.root_gotos) {
        w.put<std::uint16_t>(ch);
        w.put<std::uint32_t>(next);
    }
    w.put<std::uint32_t>(0);  // 0 outputs
    w.put_u8(0);

    // state 1 (leaf)
    w.put<std::uint32_t>(0);  // failure -> root
    w.put<std::uint32_t>(0);  // 0 gotos
    w.put<std::uint32_t>(1);  // 1 output
    w.put<std::uint32_t>(c.leaf_output);
    w.put_u8(1);

    for (std::uint32_t i = 0; i < c.extra_states; ++i) {
        w.put<std::uint32_t>(0);  // failure -> root
        w.put<std::uint32_t>(0);  // 0 gotos
        w.put<std::uint32_t>(0);  // 0 outputs
        w.put_u8(0);
    }

    w.put<std::uint32_t>(1);    // 1 payload
    w.put<std::uint32_t>(700);
    return std::move(w.bytes);
}

}  // namespace

TEST_CASE("the uncorrupted rejection fixture is itself well-formed") {
    auto a = parse(build_fixture(Fixture{}));
    const std::vector<CharId> text{2};
    REQUIRE(a.find_entry(text) != nullptr);
    CHECK(*a.find_entry(text) == 700);
}

TEST_CASE("malformed automata are rejected, not built") {
    SUBCASE("failure link beyond the state count") {
        Fixture c;
        c.root_failure = 0xDEADBEEF;
        CHECK_THROWS_AS(parse(build_fixture(c)), ParseError);
    }
    SUBCASE("goto target beyond the state count") {
        Fixture c;
        c.root_gotos = {{2, 999999}};
        CHECK_THROWS_AS(parse(build_fixture(c)), ParseError);
    }
    SUBCASE("output index beyond the payload count") {
        Fixture c;
        c.leaf_output = 42;  // only 1 payload follows
        CHECK_THROWS_AS(parse(build_fixture(c)), ParseError);
    }
    SUBCASE("root gotos not strictly ascending") {
        // build_canonical takes gotos.front() as the smallest CharId and, for
        // the root, gotos.back() as the largest, so descending keys size
        // root_next_ too small and write past it.
        Fixture c;
        c.root_gotos = {{9, 1}, {2, 1}};
        CHECK_THROWS_AS(parse(build_fixture(c)), ParseError);
    }
    SUBCASE("duplicate goto keys") {
        // Two children aliased onto one slot; caught by the same strict-ascent
        // rule.
        Fixture c;
        c.root_gotos = {{2, 1}, {2, 1}};
        CHECK_THROWS_AS(parse(build_fixture(c)), ParseError);
    }
    SUBCASE("state unreachable from the root") {
        // State 2 is declared and written but named by no goto, so it keeps
        // new_id == kNull and the remap would index fail_ at 2^32-1.
        Fixture c;
        c.extra_states = 1;
        CHECK_THROWS_AS(parse(build_fixture(c)), ParseError);
    }
}
