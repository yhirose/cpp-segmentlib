#include "segmentlib/mlp/dictionary.h"

#include <algorithm>

#include "segmentlib/unicode/egc.h"
#include "segmentlib/unicode/normalize.h"
#include "segmentlib/unicode/utf8.h"

namespace segmentlib::mlp {

DictMatcher::DictMatcher(std::span<const std::vector<std::string>> dictionaries)
    : num_dicts_(static_cast<std::uint32_t>(dictionaries.size())) {
    text::AhoCorasickBuilder<std::uint32_t, std::uint8_t> builder;
    std::vector<std::uint32_t> keys;
    std::u32string egc;
    for (std::size_t d = 0; d < dictionaries.size(); ++d) {
        for (const std::string& word : dictionaries[d]) {
            // Normalize → EGC-split the entry, interning each cluster.
            keys.clear();
            egc.clear();
            unicode::GraphemeBreaker breaker;
            std::size_t pos = 0;
            bool valid = true;
            const auto flush = [&] {
                if (egc.empty()) {
                    return;
                }
                const auto [it, unused] = egc_ids_.try_emplace(
                    egc, static_cast<std::uint32_t>(egc_ids_.size()));
                keys.push_back(it->second);
                egc.clear();
            };
            while (pos < word.size()) {
                const auto decoded =
                    unicode::decode(std::string_view(word).substr(pos));
                if (!decoded) {
                    valid = false;
                    break;
                }
                const char32_t norm = unicode::normalize(decoded->codepoint);
                if (breaker.next(norm)) {
                    flush();
                }
                egc.push_back(norm);
                pos += decoded->length;
            }
            flush();
            if (valid && !keys.empty()) {
                builder.add(keys, static_cast<std::uint8_t>(d));
            }
        }
    }
    ac_ = std::move(builder).build();
}

void DictMatcher::features_into(const EncodedEgc& enc, DictFeatures& out) const {
    const std::size_t m = enc.egc_count();
    const std::size_t boundaries = m > 0 ? m - 1 : 0;

    out.offsets.clear();
    out.indices.clear();
    out.offsets.push_back(0);
    if (boundaries == 0 || num_dicts_ == 0) {
        out.offsets.assign(boundaries + 1, 0);
        return;
    }

    // Reuse the per-boundary scratch without releasing inner capacity.
    if (out.per_boundary.size() < boundaries) {
        out.per_boundary.resize(boundaries);
    }
    for (std::size_t b = 0; b < boundaries; ++b) {
        out.per_boundary[b].clear();
    }

    out.keys.clear();
    for (std::size_t i = 0; i < m; ++i) {
        const auto it = egc_ids_.find(enc.egc_cps(i));
        out.keys.push_back(it == egc_ids_.end() ? kUnknownEgc : it->second);
    }

    // A match over EGCs [s, e] contributes L at boundary s-1, I at
    // boundaries s..e-1, R at boundary e (5.4).
    ac_.match(std::span<const std::uint32_t>(out.keys),
              [&](std::size_t end, std::size_t len, const std::uint8_t* d) {
                  const std::size_t s = end - len + 1;
                  const auto bucket = static_cast<std::uint32_t>(
                      std::min<std::size_t>(len, 4) - 1);
                  const std::uint32_t base = *d * kDictFeaturesPerDict;
                  const auto feat = [&](DictPosition p) {
                      return base + static_cast<std::uint32_t>(p) * 4 + bucket;
                  };
                  if (s >= 1) {
                      out.per_boundary[s - 1].push_back(feat(DictPosition::Left));
                  }
                  for (std::size_t b = s; b < end; ++b) {
                      out.per_boundary[b].push_back(feat(DictPosition::Inside));
                  }
                  if (end < boundaries) {
                      out.per_boundary[end].push_back(feat(DictPosition::Right));
                  }
              });

    for (std::size_t b = 0; b < boundaries; ++b) {
        auto& feats = out.per_boundary[b];
        std::sort(feats.begin(), feats.end());
        feats.erase(std::unique(feats.begin(), feats.end()), feats.end());
        out.indices.insert(out.indices.end(), feats.begin(), feats.end());
        out.offsets.push_back(static_cast<std::uint32_t>(out.indices.size()));
    }
}

}  // namespace segmentlib::mlp
