#include "segmentlib/mlp/dictionary.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <map>
#include <optional>
#include <sstream>
#include <string_view>
#include <utility>

#include "fstlib.h"
#include "segmentlib/unicode/normalize.h"
#include "segmentlib/unicode/utf8.h"

namespace segmentlib::mlp {
namespace {

// Byte offset that starts no cluster; a match ending here spans only part of
// one and is not a dictionary hit.
constexpr std::uint32_t kNoEgc = 0xFFFFFFFFu;

// Re-encodes an entry in the normalized form the scoring path sees, so the
// two sides compare byte for byte. Returns nullopt on malformed UTF-8.
std::optional<std::string> normalized_utf8(std::string_view word) {
    std::string out;
    std::size_t pos = 0;
    while (pos < word.size()) {
        const auto decoded = unicode::decode(word.substr(pos));
        if (!decoded) {
            return std::nullopt;
        }
        unicode::encode(unicode::normalize(decoded->codepoint), out);
        pos += decoded->length;
    }
    return out;
}

}  // namespace

// The compiled automaton. fst::map views the byte code rather than owning it,
// so the two live together and a copy recompiles the view over its own bytes.
struct DictMatcher::Impl {
    std::string bytes;
    fst::map<std::uint32_t> matcher;
    // Distinct channel sets in CSR form: an entry's FST output is the id of
    // its set, and set id s owns dicts[offsets[s] .. offsets[s+1]).
    std::vector<std::uint32_t> offsets;
    std::vector<std::uint8_t> dicts;

    Impl(std::string b, std::vector<std::uint32_t> o, std::vector<std::uint8_t> d)
        : bytes(std::move(b)),
          matcher(bytes.data(), bytes.size()),
          offsets(std::move(o)),
          dicts(std::move(d)) {}

    Impl(const Impl& other) : Impl(other.bytes, other.offsets, other.dicts) {}
    Impl& operator=(const Impl&) = delete;
};

DictMatcher::DictMatcher() noexcept = default;
DictMatcher::~DictMatcher() = default;
DictMatcher::DictMatcher(DictMatcher&&) noexcept = default;
DictMatcher& DictMatcher::operator=(DictMatcher&&) noexcept = default;

DictMatcher::DictMatcher(const DictMatcher& other)
    : impl_(other.impl_ ? std::make_unique<Impl>(*other.impl_) : nullptr),
      num_dicts_(other.num_dicts_) {}

DictMatcher& DictMatcher::operator=(const DictMatcher& other) {
    // Copy-and-move, so how a matcher is duplicated stays in one place above.
    DictMatcher copy(other);
    *this = std::move(copy);
    return *this;
}

DictMatcher::DictMatcher(std::span<const std::vector<std::string>> dictionaries)
    : num_dicts_(static_cast<std::uint32_t>(dictionaries.size())) {
    // (entry, channel) pairs, sorted into the byte order fst::compile wants.
    // A flat sort rather than an associative container: dictionaries run to
    // hundreds of thousands of entries and this is on the model-load path.
    std::vector<std::pair<std::string, std::uint8_t>> flat;
    for (std::size_t d = 0; d < dictionaries.size(); ++d) {
        for (const std::string& word : dictionaries[d]) {
            auto key = normalized_utf8(word);
            if (!key || key->empty()) {
                continue;
            }
            flat.emplace_back(std::move(*key), static_cast<std::uint8_t>(d));
        }
    }
    if (flat.empty()) {
        return;
    }
    std::sort(flat.begin(), flat.end());
    flat.erase(std::unique(flat.begin(), flat.end()), flat.end());

    // Intern the channel sets: an entry's output is a small set id rather than
    // the set itself, which keeps the output alphabet tiny however many
    // dictionaries (up to the format's 255) an entry belongs to.
    // The set is keyed as a byte string, not as a vector of channel ids:
    // ordering two vectors here reaches lexicographical_compare_three_way,
    // where GCC 14 cannot bound the memcmp length and rejects the build under
    // -Werror=stringop-overread. Strings compare through basic_string::compare
    // and are unaffected.
    std::map<std::string, std::uint32_t> set_ids;
    std::vector<std::uint32_t> offsets{0};
    std::vector<std::uint8_t> dicts;
    std::vector<std::pair<std::string, std::uint32_t>> input;
    std::string set_key;
    for (std::size_t i = 0; i < flat.size();) {
        // Consecutive rows sharing a key are that entry's channels, ascending.
        std::size_t j = i;
        set_key.clear();
        while (j < flat.size() && flat[j].first == flat[i].first) {
            set_key.push_back(static_cast<char>(flat[j].second));
            ++j;
        }
        const auto [it, inserted] = set_ids.try_emplace(
            set_key, static_cast<std::uint32_t>(offsets.size() - 1));
        if (inserted) {
            for (const char channel : set_key) {
                dicts.push_back(static_cast<std::uint8_t>(channel));
            }
            offsets.push_back(static_cast<std::uint32_t>(dicts.size()));
        }
        input.emplace_back(std::move(flat[i].first), it->second);
        i = j;
    }

    std::ostringstream os;
    const auto result =
        fst::compile<std::uint32_t>(input, os, /*sorted=*/true).first;
    assert(result == fst::Result::Success);
    if (result != fst::Result::Success) {
        return;
    }
    impl_ = std::make_unique<Impl>(std::move(os).str(), std::move(offsets),
                                   std::move(dicts));
}

void DictMatcher::features_into(const EncodedEgc& enc, DictFeatures& out) const {
    const std::size_t m = enc.egc_count();
    const std::size_t boundaries = m > 0 ? m - 1 : 0;

    out.offsets.clear();
    out.indices.clear();
    out.offsets.push_back(0);
    if (boundaries == 0 || num_dicts_ == 0 || !impl_) {
        out.offsets.assign(boundaries + 1, 0);
        return;
    }

    out.mask_words = (feature_count() + 63) / 64;
    out.masks.assign(boundaries * out.mask_words, 0);

    // Re-encode the sentence as normalized UTF-8, recording where each cluster
    // starts and, in the reverse direction, which byte offsets are boundaries.
    out.text.clear();
    out.egc_byte_starts.clear();
    for (std::size_t i = 0; i < m; ++i) {
        out.egc_byte_starts.push_back(static_cast<std::uint32_t>(out.text.size()));
        for (const char32_t cp : enc.egc_cps(i)) {
            unicode::encode(cp, out.text);
        }
    }
    out.egc_byte_starts.push_back(static_cast<std::uint32_t>(out.text.size()));
    out.egc_at_byte.assign(out.text.size() + 1, kNoEgc);
    for (std::size_t i = 0; i <= m; ++i) {
        out.egc_at_byte[out.egc_byte_starts[i]] = static_cast<std::uint32_t>(i);
    }

    const auto mark = [&](std::size_t b, std::uint32_t f) {
        out.masks[b * out.mask_words + f / 64] |= std::uint64_t{1} << (f % 64);
    };

    // A match over EGCs [s, e] contributes L at boundary s-1, I at
    // boundaries s..e-1, R at boundary e (5.4).
    for (std::size_t s = 0; s < m; ++s) {
        const std::uint32_t start_byte = out.egc_byte_starts[s];
        impl_->matcher.common_prefix_search(
            std::string_view(out.text).substr(start_byte),
            [&](std::size_t len, const std::uint32_t& set_id) {
                const std::uint32_t past = out.egc_at_byte[start_byte + len];
                if (past == kNoEgc) {
                    return;  // ends inside a cluster, so not a match here
                }
                const std::size_t end = past - 1;
                const auto bucket = static_cast<std::uint32_t>(
                    std::min<std::size_t>(past - s, 4) - 1);
                for (std::uint32_t k = impl_->offsets[set_id];
                     k < impl_->offsets[set_id + 1]; ++k) {
                    const std::uint32_t base =
                        impl_->dicts[k] * kDictFeaturesPerDict;
                    const auto feat = [&](DictPosition p) {
                        return base + static_cast<std::uint32_t>(p) * 4 + bucket;
                    };
                    if (s >= 1) {
                        mark(s - 1, feat(DictPosition::Left));
                    }
                    const std::uint32_t inside = feat(DictPosition::Inside);
                    for (std::size_t b = s; b < end; ++b) {
                        mark(b, inside);
                    }
                    if (end < boundaries) {
                        mark(end, feat(DictPosition::Right));
                    }
                }
            });
    }

    for (std::size_t b = 0; b < boundaries; ++b) {
        const std::uint64_t* row = out.masks.data() + b * out.mask_words;
        for (std::size_t w = 0; w < out.mask_words; ++w) {
            for (std::uint64_t bits = row[w]; bits != 0; bits &= bits - 1) {
                out.indices.push_back(static_cast<std::uint32_t>(
                    w * 64 + static_cast<std::size_t>(std::countr_zero(bits))));
            }
        }
        out.offsets.push_back(static_cast<std::uint32_t>(out.indices.size()));
    }
}

}  // namespace segmentlib::mlp
