#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <span>
#include <utility>
#include <vector>

namespace segmentlib::text {

template <class Key, class Payload>
class AhoCorasickBuilder;

// A runtime-built Aho-Corasick matcher over sequences of `Key`, carrying a
// `Payload` per added pattern.
//
// This is the *construction* counterpart of kytea::Automaton, which is a
// read-only decoder for automata that arrive fully built inside a KyTea model
// file. Here the patterns arrive as flat lists (the MLP model's dictionary
// words, design.ja.md 4.7 field 17; Vaporetto's word lists, 4.5) and the
// automaton is built at load time. The two are deliberately separate: they
// share no on-disk format, and this one is keyed by an arbitrary Key type
// (the MLP backend uses interned EGC ids rather than codepoints).
//
// Construction goes through AhoCorasickBuilder (add patterns, then build()).
// Failure links are resolved and outputs propagated along suffix links at
// build time, so matching never walks failure links to collect outputs; each
// state's output list is complete. Transitions are binary-searched in a
// per-state sorted slice (CSR layout) — adequate for dictionary matching,
// where the automaton is a fraction of the text-scanning cost; a double-array
// layout like kytea::Automaton's is a possible later optimization.
template <class Key, class Payload>
class AhoCorasick {
public:
    // A pattern occurrence: the pattern occupies text positions
    // [end_pos - length + 1, end_pos].
    struct Match {
        std::size_t end_pos;   // index of the last key of the match
        std::size_t length;    // pattern length, in keys
        const Payload* payload;
    };

    // An empty automaton (no patterns); match() finds nothing.
    AhoCorasick() = default;

    [[nodiscard]] bool empty() const noexcept { return num_patterns() == 0; }
    [[nodiscard]] std::size_t num_patterns() const noexcept { return payloads_.size(); }
    [[nodiscard]] std::size_t num_states() const noexcept {
        return trans_offset_.empty() ? 0 : trans_offset_.size() - 1;
    }

    // Runs the automaton over `text`, invoking
    // `on_match(end_pos, length, payload)` for every pattern occurrence.
    // Matches are emitted in order of end position; at equal end positions,
    // longer matches first.
    template <class OnMatch>
    void match(std::span<const Key> text, OnMatch&& on_match) const {
        if (num_states() == 0) {
            return;
        }
        std::uint32_t state = 0;
        for (std::size_t i = 0; i < text.size(); ++i) {
            const Key k = text[i];
            std::uint32_t next = step(state, k);
            while (next == kNoState && state != 0) {
                state = fail_[state];
                next = step(state, k);
            }
            state = (next == kNoState) ? 0 : next;
            for (std::uint32_t o = out_offset_[state]; o < out_offset_[state + 1]; ++o) {
                const std::uint32_t p = out_flat_[o];
                on_match(i, pattern_lengths_[p], &payloads_[p]);
            }
        }
    }

    // Convenience wrapper that collects all matches into a vector.
    [[nodiscard]] std::vector<Match> match_all(std::span<const Key> text) const {
        std::vector<Match> result;
        match(text, [&](std::size_t end, std::size_t len, const Payload* p) {
            result.push_back(Match{end, len, p});
        });
        return result;
    }

private:
    friend class AhoCorasickBuilder<Key, Payload>;

    static constexpr std::uint32_t kNoState =
        std::numeric_limits<std::uint32_t>::max();

    // The goto transition for `k` from `state`, or kNoState if none: a binary
    // search in the state's sorted (keys, targets) slice.
    [[nodiscard]] std::uint32_t step(std::uint32_t state, Key k) const noexcept {
        const std::uint32_t lo = trans_offset_[state];
        const std::uint32_t hi = trans_offset_[state + 1];
        const auto* first = trans_keys_.data() + lo;
        const auto* last = trans_keys_.data() + hi;
        const auto* it = std::lower_bound(first, last, k);
        if (it != last && *it == k) {
            return trans_targets_[lo + static_cast<std::uint32_t>(it - first)];
        }
        return kNoState;
    }

    // Transitions in CSR form: state s owns the sorted key slice
    // trans_keys_[trans_offset_[s] .. trans_offset_[s+1]) with parallel targets.
    std::vector<std::uint32_t> trans_offset_;  // size num_states + 1
    std::vector<Key> trans_keys_;
    std::vector<std::uint32_t> trans_targets_;

    std::vector<std::uint32_t> fail_;  // per state

    // Complete (suffix-propagated) outputs in CSR form: pattern indices,
    // ordered longest pattern first within a state.
    std::vector<std::uint32_t> out_offset_;  // size num_states + 1
    std::vector<std::uint32_t> out_flat_;

    std::vector<std::size_t> pattern_lengths_;  // per pattern
    std::vector<Payload> payloads_;             // per pattern
};

// Accumulates patterns into a trie, then build() resolves failure links and
// output propagation and produces the immutable AhoCorasick matcher.
template <class Key, class Payload>
class AhoCorasickBuilder {
public:
    // Adds one pattern with its payload. Duplicate patterns are allowed; each
    // occurrence then reports every payload added for it. Empty patterns are
    // ignored (an empty dictionary word matches everywhere and means nothing).
    void add(std::span<const Key> pattern, Payload payload) {
        if (pattern.empty()) {
            return;
        }
        std::uint32_t state = 0;
        for (const Key k : pattern) {
            state = child_or_create(state, k);
        }
        nodes_[state].outputs.push_back(
            static_cast<std::uint32_t>(payloads_.size()));
        pattern_lengths_.push_back(pattern.size());
        payloads_.push_back(std::move(payload));
    }

    // Builds the matcher. The builder is consumed (left empty).
    [[nodiscard]] AhoCorasick<Key, Payload> build() && {
        AhoCorasick<Key, Payload> ac;
        ac.pattern_lengths_ = std::move(pattern_lengths_);
        ac.payloads_ = std::move(payloads_);
        if (ac.payloads_.empty()) {
            nodes_ = {Node{}};  // leave the builder reusable
            return ac;          // empty automaton: no states at all
        }

        // Sort each node's children so the matcher can binary-search them.
        for (Node& node : nodes_) {
            std::sort(node.children.begin(), node.children.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
        }

        // Standard BFS failure-link construction. Processing in BFS order
        // guarantees fail_[u] is final before any child of u is processed, so
        // outputs can be propagated along the suffix link in the same pass
        // (parent-before-child also holds for suffix links: |fail(v)| < |v|).
        const std::size_t n = nodes_.size();
        std::vector<std::uint32_t> fail(n, 0);
        std::queue<std::uint32_t> bfs;
        for (const auto& [k, child] : nodes_[0].children) {
            bfs.push(child);  // depth-1 states fail to the root
        }
        while (!bfs.empty()) {
            const std::uint32_t u = bfs.front();
            bfs.pop();
            // Propagate: u's complete output list = its own patterns (already
            // there, longest-first is trivial since a node is one pattern
            // depth) followed by everything its proper suffix matches.
            const auto& suffix_out = nodes_[fail[u]].outputs;
            nodes_[u].outputs.insert(nodes_[u].outputs.end(), suffix_out.begin(),
                                     suffix_out.end());
            for (const auto& [k, v] : nodes_[u].children) {
                // fail(v): the longest proper suffix of v that is a state —
                // follow u's failure chain until a state has a k-child.
                std::uint32_t f = fail[u];
                std::uint32_t t = child_of(f, k);
                while (t == kNull && f != 0) {
                    f = fail[f];
                    t = child_of(f, k);
                }
                fail[v] = (t == kNull) ? 0 : t;
                bfs.push(v);
            }
        }

        // Flatten into the matcher's CSR arrays.
        ac.trans_offset_.reserve(n + 1);
        ac.out_offset_.reserve(n + 1);
        ac.trans_offset_.push_back(0);
        ac.out_offset_.push_back(0);
        for (const Node& node : nodes_) {
            for (const auto& [k, child] : node.children) {
                ac.trans_keys_.push_back(k);
                ac.trans_targets_.push_back(child);
            }
            ac.trans_offset_.push_back(
                static_cast<std::uint32_t>(ac.trans_keys_.size()));
            ac.out_flat_.insert(ac.out_flat_.end(), node.outputs.begin(),
                                node.outputs.end());
            ac.out_offset_.push_back(
                static_cast<std::uint32_t>(ac.out_flat_.size()));
        }
        ac.fail_ = std::move(fail);

        nodes_ = {Node{}};  // leave the builder reusable
        return ac;
    }

private:
    static constexpr std::uint32_t kNull = std::numeric_limits<std::uint32_t>::max();

    struct Node {
        // Unsorted during add() (linear scan; dictionary words are short and
        // fan-out is modest), sorted once in build().
        std::vector<std::pair<Key, std::uint32_t>> children;
        std::vector<std::uint32_t> outputs;  // pattern indices ending here
    };

    [[nodiscard]] std::uint32_t child_of(std::uint32_t state, Key k) const noexcept {
        for (const auto& [key, child] : nodes_[state].children) {
            if (key == k) {
                return child;
            }
        }
        return kNull;
    }

    [[nodiscard]] std::uint32_t child_or_create(std::uint32_t state, Key k) {
        if (const std::uint32_t c = child_of(state, k); c != kNull) {
            return c;
        }
        const auto id = static_cast<std::uint32_t>(nodes_.size());
        nodes_.emplace_back();
        nodes_[state].children.emplace_back(k, id);
        return id;
    }

    std::vector<Node> nodes_{Node{}};  // node 0 is the root
    std::vector<std::size_t> pattern_lengths_;
    std::vector<Payload> payloads_;
};

}  // namespace segmentlib::text
