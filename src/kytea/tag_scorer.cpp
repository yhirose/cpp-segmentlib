#include "segmentlib/kytea/tag_scorer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "segmentlib/kytea/automaton.h"
#include "segmentlib/mlp/kernels.h"

namespace segmentlib::kytea {

namespace {

using Score = std::int32_t;  // == KyTea's FeatSum for the quantized build

// KyTea's runtime defaults (KyteaConfig), not serialized in the model but needed
// to reproduce `kytea` with no options. defTag is emitted as a word's tag when
// no candidate could be produced for a level (unknown word, no subword match).
constexpr unsigned kUnkBeam = 50;
constexpr std::string_view kDefaultTag = "UNK";

// KyteaLM::scoreSingle: the log-probability contribution of the character at
// `pos` in `val` (or the sentence-end context when pos == val.size()), via
// backoff over the n-gram context. Faithful to the reference arithmetic and
// accumulation order, which the byte-exact reading depends on.
double score_single(const KyteaLM& lm, std::span<const CharId> val, int pos) {
    const int n = static_cast<int>(lm.n);
    std::u16string ngram(static_cast<std::size_t>(n), u'\0');
    int npos = n;
    if (static_cast<int>(val.size()) == pos) {
        npos--;
        pos--;
    }
    while (--npos >= 0 && pos >= 0) {
        ngram[static_cast<std::size_t>(npos)] = static_cast<char16_t>(val[static_cast<std::size_t>(pos)]);
        pos--;
    }
    double prob = 0;
    for (npos = 0; npos < n; npos++) {
        const auto it = lm.probs.find(ngram.substr(static_cast<std::size_t>(npos)));
        if (it != lm.probs.end()) {
            prob += it->second;
            return prob;
        }
        const auto fit =
            lm.fallbacks.find(ngram.substr(static_cast<std::size_t>(npos),
                                           static_cast<std::size_t>(n - npos - 1)));
        if (fit != lm.fallbacks.end()) {
            prob += fit->second;
        }
    }
    return prob + std::log(1.0 / lm.vocab_size);
}

// One beam-search hypothesis: a candidate reading (raw char ids) and its
// accumulated log-probability.
struct Hyp {
    std::vector<CharId> reading;
    double score = 0;
};

// Kytea::generateTagCandidates: a subword lattice DP over `str` (the unknown
// word's normalized char ids). Subword-dictionary matches extend hypotheses;
// each extension adds the subword's stored probability plus the language model's
// per-new-character scores; the beam is trimmed to kUnkBeam between end points.
// Returns the hypotheses reaching the end of the word (each finalized with the
// sentence-end language-model score), matching the reference summation order.
std::vector<Hyp> generate_tag_candidates(const Model& model, int lev, std::span<const CharId> str) {
    const Automaton<ProbSubwordEntry>& sub = model.subword_dict();
    const KyteaLM& lm = model.subword_models()[static_cast<std::size_t>(lev)];
    const std::size_t len = str.size();

    struct M {
        std::size_t end_pos;
        const ProbSubwordEntry* entry;
    };
    std::vector<M> matches;
    sub.match(str, [&](std::size_t end_pos, const ProbSubwordEntry& e) {
        matches.push_back({end_pos, &e});
    });

    std::vector<std::vector<Hyp>> stack(len + 1);
    stack[0].push_back(Hyp{});
    unsigned last_end = 0;
    for (const M& m : matches) {
        const ProbSubwordEntry* entry = m.entry;
        const unsigned end = static_cast<unsigned>(m.end_pos) + 1;
        const unsigned start = end - entry->word_length;
        // Trim the just-completed end point's beam before it feeds later matches.
        if (end != last_end && kUnkBeam > 0 && stack[last_end].size() > kUnkBeam) {
            std::sort(stack[last_end].begin(), stack[last_end].end(),
                      [](const Hyp& a, const Hyp& b) { return a.score > b.score; });
            stack[last_end].resize(kUnkBeam);
        }
        last_end = end;
        const std::vector<std::vector<CharId>>& etags = entry->tags[static_cast<std::size_t>(lev)];
        const std::vector<double>& eprobs = entry->probs[static_cast<std::size_t>(lev)];
        for (std::size_t j = 0; j < etags.size(); j++) {
            for (const Hyp& base : stack[start]) {
                Hyp next;
                next.reading = base.reading;
                next.reading.insert(next.reading.end(), etags[j].begin(), etags[j].end());
                next.score = base.score + eprobs[j];
                for (std::size_t pos = base.reading.size(); pos < next.reading.size(); pos++) {
                    next.score += score_single(lm, next.reading, static_cast<int>(pos));
                }
                stack[end].push_back(std::move(next));
            }
        }
    }
    std::vector<Hyp> ret = std::move(stack[len]);
    for (Hyp& h : ret) {
        h.score += score_single(lm, h.reading, static_cast<int>(h.reading.size()));
    }
    return ret;
}

// Kytea::calculateUnknownTag, reduced to producing the single winning reading
// (the output never needs the normalized probabilities). The winner is the
// highest-scoring hypothesis; ties break to the char-id-lexicographically
// smaller reading, exactly as KyTea's sort (operator< on KyteaTag) would leave
// at the front. Returns the reading's char ids, or empty when no candidate
// exists (the caller then falls back to the default tag).
std::vector<CharId> calculate_unknown_reading(const Model& model, int lev,
                                              std::span<const CharId> word) {
    if (lev >= static_cast<int>(model.subword_models().size()) ||
        model.subword_models()[static_cast<std::size_t>(lev)].empty() || word.size() > 256) {
        return {};
    }
    const std::vector<Hyp> cands = generate_tag_candidates(model, lev, word);
    const Hyp* best = nullptr;
    for (const Hyp& h : cands) {
        if (best == nullptr || h.score > best->score ||
            (h.score == best->score && h.reading < best->reading)) {
            best = &h;
        }
    }
    return best == nullptr ? std::vector<CharId>{} : best->reading;
}

// Tag n-gram features (FeatureLookup::addTagNgrams). Unlike the WS n-gram
// features, these look at the characters *outside* the word span: up to `window`
// characters before start_char and up to `window` after end_char, concatenated
// (the word itself is excluded). Each match into that combined string selects a
// candidate weight block by a reversed index — `pos = (window*2 - pos - 1) *
// numWeights` — which must be reproduced exactly. Here `window` is the n-gram
// order (config.char_n / type_n), not the WS boundary window, and scores.size()
// is numWeights (the size==1 push happens only after all features are added).
void add_tag_ngrams(const Automaton<FeatVec>& dict, std::span<const CharId> chars,
                    std::vector<Score>& scores, int window, int start_char, int end_char) {
    if (dict.empty()) {
        return;
    }
    const int len = static_cast<int>(chars.size());
    const int my_start = std::max(start_char - window, 0);
    const int my_end = std::min(end_char + window, len);

    // str = chars[my_start, start_char) + chars[end_char, my_end)
    thread_local std::vector<CharId> buf;
    buf.clear();
    for (int k = my_start; k < start_char; ++k) {
        buf.push_back(chars[static_cast<std::size_t>(k)]);
    }
    for (int k = end_char; k < my_end; ++k) {
        buf.push_back(chars[static_cast<std::size_t>(k)]);
    }

    const int offset = window - (start_char - my_start);
    const int nw = static_cast<int>(scores.size());
    dict.match(buf, [&](std::size_t end_pos, const FeatVec& vec) {
        int pos = static_cast<int>(end_pos) + offset;
        pos = (window * 2 - pos - 1) * nw;
        // The hottest loop of tag prediction (~18% of tokenize as scalar,
        // design.ja.md 9.2.1): a widening int16→int32 add the compiler does
        // not auto-vectorize, so it uses the explicit SIMD kernel. Integer
        // adds are exact — byte agreement is preserved by construction.
        mlp::kernels::add_widen_i16_i32(vec.data() + pos, scores.data(),
                                        static_cast<std::size_t>(nw));
    });
}

// Whole-word (self) weights (FeatureLookup::addSelfWeights). The word's own
// char or type string selects a weight block `[featIdx*numWeights ..]` from the
// selfDict entry (0 = char, 1 = type). Absent entry contributes nothing.
void add_self_weights(const Automaton<FeatVec>& self_dict, std::span<const CharId> word,
                      std::vector<Score>& scores, int feat_idx) {
    const FeatVec* entry = self_dict.find_entry(word);
    if (entry == nullptr) {
        return;
    }
    const int nw = static_cast<int>(scores.size());
    mlp::kernels::add_widen_i16_i32(entry->data() + feat_idx * nw,
                                    scores.data(),
                                    static_cast<std::size_t>(nw));
}

// Dictionary-match (dict,candidate) pairs for a word (Kytea::getDictionaryMatches).
// Reuses the word-dictionary entry the caller already looked up. For every
// candidate of level `lev`, emits (dict, candidate) for each sub-dictionary that
// contains it. Appends into `out` (cleared by the caller).
void get_dictionary_matches(const Model& model, const WordEntry* ent, int lev,
                            std::vector<std::pair<int, int>>& out) {
    if (ent == nullptr || ent->in_dict == 0 || static_cast<int>(ent->tag_in_dicts.size()) <= lev) {
        return;
    }
    const int num_dicts = model.num_dicts();
    const std::vector<std::uint8_t>& tid = ent->tag_in_dicts[static_cast<std::size_t>(lev)];
    for (int i = 0; i < static_cast<int>(tid.size()); ++i) {
        for (int j = 0; j < num_dicts; ++j) {
            if (((1u << j) & tid[static_cast<std::size_t>(i)]) != 0) {
                out.emplace_back(j, i);
            }
        }
    }
}

// Dictionary-match weights (FeatureLookup::addTagDictWeights). No matches ->
// tagUnkVector; otherwise, for each (dict,candidate) pair, the tagDictVector
// block at di*T*T + cand*T (T == numWeights).
void add_tag_dict_weights(const TagFeatureLookup& look,
                          const std::vector<std::pair<int, int>>& exists,
                          std::vector<Score>& scores, int nw) {
    if (exists.empty()) {
        if (!look.tag_unk_vector.empty()) {
            mlp::kernels::add_widen_i16_i32(look.tag_unk_vector.data(),
                                            scores.data(),
                                            static_cast<std::size_t>(nw));
        }
        return;
    }
    if (look.tag_dict_vector.empty()) {
        return;
    }
    for (const auto& [di, cand] : exists) {
        const int base = di * nw * nw + cand * nw;
        mlp::kernels::add_widen_i16_i32(look.tag_dict_vector.data() + base,
                                        scores.data(),
                                        static_cast<std::size_t>(nw));
    }
}

}  // namespace

void predict_word_tags(const Model& model, const EncodedText& enc,
                       std::size_t start_char, std::size_t end_char,
                       const WordEntry* ent, std::vector<std::string>& out) {
    const int num_tags = model.config().num_tags;
    const auto& gmods = model.global_mods();
    const auto& gtags = model.global_tags();
    const std::span<const CharId> char_ids{enc.char_ids};
    const std::span<const CharId> type_ids{enc.type_ids};
    const int start = static_cast<int>(start_char);
    const int fin = static_cast<int>(end_char);

    thread_local std::vector<Score> scores;
    thread_local std::vector<std::pair<int, int>> matches;

    for (int lev = 0; lev < num_tags; ++lev) {
        // Dispatch: global model (useSelf) takes precedence, else the per-word
        // model attached to the dictionary entry (which may itself be absent).
        const std::vector<std::string>* tags = nullptr;
        const TagModel* mod = nullptr;
        bool use_self = false;
        if (lev < static_cast<int>(gmods.size()) && gmods[static_cast<std::size_t>(lev)].has_value()) {
            mod = &*gmods[static_cast<std::size_t>(lev)];
            tags = &gtags[static_cast<std::size_t>(lev)];
            use_self = true;
        } else if (ent != nullptr && lev < static_cast<int>(ent->tags.size())) {
            tags = &ent->tags[static_cast<std::size_t>(lev)];
            const auto& pm = ent->tag_mods[static_cast<std::size_t>(lev)];
            mod = pm.has_value() ? &*pm : nullptr;
        }

        // Unknown tag (no dictionary candidates): estimate the reading with the
        // subword language model (stage B). No estimate (no reading model, or no
        // subword match — e.g. symbols/emoji) falls back to KyTea's default tag.
        if (tags == nullptr || tags->empty()) {
            const std::vector<CharId> reading = calculate_unknown_reading(
                model, lev, char_ids.subspan(start_char, end_char - start_char));
            if (reading.empty()) {
                out.emplace_back(kDefaultTag);
            } else {
                out.push_back(model.chars().decode(reading));
            }
            continue;
        }

        // No classifier attached: the first candidate is chosen deterministically.
        if (mod == nullptr || !mod->lookup.has_value()) {
            out.emplace_back((*tags)[0]);
            continue;
        }

        const TagFeatureLookup& look = *mod->lookup;
        const int nw = mod->num_weights;
        scores.assign(static_cast<std::size_t>(nw), 0);

        add_tag_ngrams(look.char_dict, char_ids, scores, model.config().char_n, start, fin);
        add_tag_ngrams(look.type_dict, type_ids, scores, model.config().type_n, start, fin);
        if (use_self) {
            const auto wlen = end_char - start_char;
            add_self_weights(look.self_dict, char_ids.subspan(start_char, wlen), scores, 0);
            add_self_weights(look.self_dict, type_ids.subspan(start_char, wlen), scores, 1);
            matches.clear();
            get_dictionary_matches(model, ent, 0, matches);  // KyTea hardcodes lev 0 here
            add_tag_dict_weights(look, matches, scores, nw);
        }
        mlp::kernels::add_widen_i16_i32(look.biases.data(), scores.data(),
                                        static_cast<std::size_t>(nw));
        // Binary classifier: the second class's score is the negated first (prob)
        // or zero (margin). The top candidate is invariant to that choice, so the
        // margin form suffices for producing the winning string.
        if (scores.size() == 1) {
            scores.push_back(0);
        }

        // Pick the top candidate. KyTea sorts the candidates by confidence
        // (score * multiplier) descending and takes the front; we only need that
        // front, so a single max-scan suffices. multiplier > 0, so the ordering
        // is exactly the integer-score ordering (including which pairs tie), and
        // the earliest max index is kept. Only the first scores.size() candidates
        // are considered (a per-word model can have fewer weights than strings).
        std::size_t best = 0;
        for (std::size_t i = 1; i < scores.size(); ++i) {
            if (scores[i] > scores[best]) {
                best = i;
            }
        }
        out.emplace_back((*tags)[best]);
    }
}

}  // namespace segmentlib::kytea
