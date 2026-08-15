#include "mlp/train/example.h"

#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <utility>

#include "segmentlib/unicode/normalize.h"
#include "segmentlib/unicode/utf8.h"

namespace segmentlib::mlp::train {

Vocab build_vocab(std::span<const AnnotatedSentence> sentences,
                  std::uint32_t min_count) {
    std::unordered_map<char32_t, std::uint32_t> counts;
    for (const AnnotatedSentence& sentence : sentences) {
        std::size_t pos = 0;
        while (pos < sentence.text.size()) {
            const auto decoded =
                unicode::decode(std::string_view(sentence.text).substr(pos));
            if (!decoded) {
                break;  // invalid sentences are skipped in build_examples too
            }
            ++counts[unicode::normalize(decoded->codepoint)];
            pos += decoded->length;
        }
    }
    std::vector<char32_t> kept;
    for (const auto& [cp, count] : counts) {
        if (count >= min_count) {
            kept.push_back(cp);
        }
    }
    std::sort(kept.begin(), kept.end());
    return Vocab{std::move(kept)};
}

ExampleSet build_examples(std::span<const AnnotatedSentence> sentences,
                          const Vocab& vocab,
                          std::span<const std::vector<std::string>> dictionaries,
                          std::uint8_t window, ExampleStats* stats) {
    ExampleSet set;
    set.window = window;
    set.num_dicts = static_cast<std::uint32_t>(dictionaries.size());
    set.egc_starts.push_back(0);
    set.sentence_starts.push_back(0);
    set.feat_offsets.push_back(0);

    ExampleStats local_stats;
    ExampleStats& st = stats ? *stats : local_stats;
    st = ExampleStats{};
    st.sentences_read = sentences.size();

    // The inference-side matcher generates the features, so training and
    // inference agree by construction (III.1).
    const bool with_dicts = !dictionaries.empty();
    const DictMatcher matcher(dictionaries);

    EncodedEgc enc;       // reused per sentence
    DictFeatures feats;   // reused per sentence
    struct Labeled {
        std::uint32_t boundary;
        float label;
    };
    std::vector<Labeled> labeled;

    for (const AnnotatedSentence& sentence : sentences) {
        if (!vocab.encode_into(sentence.text, enc)) {
            ++st.sentences_skipped_invalid;
            continue;
        }
        const std::size_t num_cps = enc.rows.size();
        const std::size_t num_egcs = enc.egc_count();
        assert(sentence.tags.size() + 1 == num_cps || num_cps == 0);

        // Map codepoint-gap supervision onto EGC boundaries. (A sentence
        // with a single EGC has no boundary candidates but can still carry a
        // conflicting Bound inside that EGC — the scan below catches it.) Gap g (after
        // codepoint g) coincides with EGC boundary b iff codepoint g+1 starts
        // EGC b+1. A Bound inside an EGC contradicts the representation:
        // skip the sentence (5.5). NoBound/Unknown inside an EGC agree with
        // the EGC definition and are absorbed.
        labeled.clear();
        bool conflict = false;
        std::size_t next_egc = 1;  // EGC whose start we will meet next
        for (std::size_t g = 0; g + 1 < num_cps; ++g) {
            const bool at_egc_boundary =
                next_egc < num_egcs && enc.egc_starts[next_egc] == g + 1;
            const BoundaryTag tag = sentence.tags[g];
            if (!at_egc_boundary) {
                if (tag == BoundaryTag::Bound) {
                    conflict = true;
                    break;
                }
                continue;
            }
            const auto boundary = static_cast<std::uint32_t>(next_egc - 1);
            ++next_egc;
            if (tag == BoundaryTag::Unknown) {
                ++st.boundaries_unknown;
                continue;
            }
            labeled.push_back(
                Labeled{boundary, tag == BoundaryTag::Bound ? 1.0f : 0.0f});
        }
        if (conflict) {
            ++st.sentences_skipped_conflict;
            continue;
        }
        if (labeled.empty()) {
            continue;  // nothing supervised; don't store the sentence
        }
        st.boundaries_labeled += labeled.size();

        // Dictionary features per boundary (5.4), via the shared matcher.
        if (with_dicts) {
            matcher.features_into(enc, feats);
        }

        // Store the sentence and emit its examples.
        const auto sentence_id =
            static_cast<std::uint32_t>(set.sentence_count());
        const auto row_base = static_cast<std::uint32_t>(set.rows.size());
        set.rows.insert(set.rows.end(), enc.rows.begin(), enc.rows.end());
        for (std::size_t e = 1; e < enc.egc_starts.size(); ++e) {
            set.egc_starts.push_back(row_base + enc.egc_starts[e]);
        }
        set.sentence_starts.push_back(set.sentence_starts.back() +
                                      static_cast<std::uint32_t>(num_egcs));
        for (const Labeled& l : labeled) {
            set.examples.push_back(
                ExampleSet::Example{sentence_id, l.boundary, l.label});
            if (with_dicts) {
                set.feat_indices.insert(
                    set.feat_indices.end(),
                    feats.indices.begin() + feats.offsets[l.boundary],
                    feats.indices.begin() + feats.offsets[l.boundary + 1]);
            }
            set.feat_offsets.push_back(
                static_cast<std::uint32_t>(set.feat_indices.size()));
        }
    }
    return set;
}

}  // namespace segmentlib::mlp::train
