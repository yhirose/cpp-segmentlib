#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "segmentlib/kytea/automaton.h"
#include "segmentlib/kytea/char_table.h"
#include "segmentlib/types.h"

namespace segmentlib::kytea {

// Quantized feature-weight type used by KyTea's default (non-"NQ") models.
using FeatVal = std::int16_t;
using FeatVec = std::vector<FeatVal>;

// The subset of KyTea's training configuration that the model file carries.
struct Config {
    std::uint8_t char_window = 0;  // characters on each side of a boundary
    std::uint8_t char_n = 0;       // max character n-gram order
    std::uint8_t type_window = 0;
    std::uint8_t type_n = 0;
    std::uint8_t dict_n = 0;  // dictionary match-length cap
    int num_tags = 0;
    bool do_ws = false;
    bool do_tags = false;
    int solver = 0;
};

// The tag-prediction subset of KyTea's FeatureLookup (the classifier attached to
// a KyteaModel). KyTea's dictVector_ is word-segmentation-only and is not kept
// here; the remaining six members are exactly what the known-word tag scorer
// (kytea.cpp:calculateTags) consumes.
struct TagFeatureLookup {
    Automaton<FeatVec> char_dict;      // charDict_: char n-gram weights
    Automaton<FeatVec> type_dict;      // typeDict_: char-type n-gram weights
    Automaton<FeatVec> self_dict;      // selfDict_: whole-word (self) weights
    FeatVec biases;                    // biases_: per-candidate bias
    FeatVec tag_dict_vector;           // tagDictVector_: dictionary-match weights
    FeatVec tag_unk_vector;            // tagUnkVector_: no-dictionary-match weights
};

// A retained KyTea classifier (KyteaModel) used for tag prediction, at one tag
// level. `lookup` is absent when the model carries no FeatureLookup, in which
// case the tag is decided deterministically as the first candidate.
struct TagModel {
    double multiplier = 1.0;  // score -> confidence scale (getMultiplier)
    // getNumWeights(): the per-candidate weight-block size. Not serialized;
    // KyTea derives it from the class count as (numClasses==2 ? 1 : numClasses),
    // because setNumClasses runs before setSolver (so solver_ is still the
    // default 1, never MCSVM_CS) during model reading.
    int num_weights = 0;
    std::optional<TagFeatureLookup> lookup;
};

// Per-dictionary-word data. The dictionary automaton carries this as its
// payload. `char_length` / `in_dict` drive the word-segmentation D features; the
// per-level tag fields (retained for tag prediction) mirror KyTea's
// ModelTagEntry: candidate strings, their dictionary bitmasks, and an optional
// per-word tag model (usually absent, since the global model is used instead).
struct WordEntry {
    std::uint32_t char_length = 0;  // length of the word in characters
    std::uint8_t in_dict = 0;       // bitmask: which sub-dictionaries contain it

    // Indexed by tag level. tags[lev][cand] is a UTF-8 candidate string;
    // tag_in_dicts[lev][cand] its per-dictionary bitmask; tag_mods[lev] the
    // per-word classifier for that level (nullopt when none was stored).
    std::vector<std::vector<std::string>> tags;
    std::vector<std::vector<std::uint8_t>> tag_in_dicts;
    std::vector<std::optional<TagModel>> tag_mods;
};

// A subword-dictionary entry (KyTea's ProbTagEntry), the payload of the
// subword automaton used for unknown-word reading estimation. Unlike the tag
// strings of a WordEntry, the candidate readings here are kept as *raw* char-id
// sequences (KyteaString), because the beam search concatenates them and scores
// them with a char-id-keyed language model; they are decoded to UTF-8 only once,
// for the winning candidate. `word_length` is the matched key's character length
// (KyTea reads it back from ProbTagEntry::word), needed to locate a match's start.
struct ProbSubwordEntry {
    std::uint32_t word_length = 0;
    // Indexed by tag level: tags[lev][cand] is a reading (raw char ids);
    // probs[lev][cand] its log-probability contribution.
    std::vector<std::vector<std::vector<CharId>>> tags;
    std::vector<std::vector<double>> probs;
};

// A retained KyTea character n-gram language model (KyteaLM), one per tag level,
// used to score candidate readings during unknown-word estimation. `n == 0`
// marks an absent model (the level has no reading LM). The two maps are keyed by
// raw char-id strings (stored as std::u16string, since CharId is 16-bit); a key
// missing from `probs` falls back through shorter contexts (see scoreSingle).
// Entries KyTea wrote as its NEG_INFINITY (-999.0) sentinel are omitted.
struct KyteaLM {
    unsigned n = 0;
    unsigned vocab_size = 0;
    std::unordered_map<std::u16string, double> probs;
    std::unordered_map<std::u16string, double> fallbacks;

    [[nodiscard]] bool empty() const noexcept { return n == 0; }
};

// A loaded KyTea model, holding exactly what word segmentation needs. Tag /
// reading / language-model sections of the file are parsed only far enough to
// advance past them. Immutable once loaded.
class Model {
public:
    static std::expected<Model, Error> load(const std::filesystem::path& path);
    static std::expected<Model, Error> load_from_bytes(std::span<const std::byte> data);

    [[nodiscard]] const Config& config() const noexcept { return parts_.config; }
    [[nodiscard]] const CharTable& chars() const noexcept { return parts_.chars; }
    [[nodiscard]] double multiplier() const noexcept { return parts_.multiplier; }

    [[nodiscard]] const Automaton<FeatVec>& char_dict() const noexcept { return parts_.char_dict; }
    [[nodiscard]] const Automaton<FeatVec>& type_dict() const noexcept { return parts_.type_dict; }
    [[nodiscard]] const FeatVec& dict_vector() const noexcept { return parts_.dict_vector; }
    [[nodiscard]] const FeatVec& biases() const noexcept { return parts_.biases; }
    [[nodiscard]] const Automaton<WordEntry>& word_dict() const noexcept { return parts_.word_dict; }
    [[nodiscard]] std::uint8_t num_dicts() const noexcept { return parts_.word_dict.num_dicts(); }

    // Global tag models, indexed by tag level (size == config().num_tags).
    // global_tags()[lev] holds that level's candidate strings (UTF-8);
    // global_mods()[lev] the classifier (nullopt when this level uses per-word
    // models instead). Together they drive the useSelf branch of tag prediction.
    [[nodiscard]] const std::vector<std::vector<std::string>>& global_tags() const noexcept {
        return parts_.global_tags;
    }
    [[nodiscard]] const std::vector<std::optional<TagModel>>& global_mods() const noexcept {
        return parts_.global_mods;
    }

    // Subword dictionary and per-level reading language models, driving
    // unknown-word reading estimation. subword_models()[lev].empty() when the
    // level has no reading model (e.g. part-of-speech).
    [[nodiscard]] const Automaton<ProbSubwordEntry>& subword_dict() const noexcept {
        return parts_.subword_dict;
    }
    [[nodiscard]] const std::vector<KyteaLM>& subword_models() const noexcept {
        return parts_.subword_models;
    }

private:
    struct Parts {
        Config config;
        CharTable chars;
        double multiplier = 1.0;
        Automaton<FeatVec> char_dict;
        Automaton<FeatVec> type_dict;
        FeatVec dict_vector;
        FeatVec biases;
        Automaton<WordEntry> word_dict;
        std::vector<std::vector<std::string>> global_tags;
        std::vector<std::optional<TagModel>> global_mods;
        Automaton<ProbSubwordEntry> subword_dict;
        std::vector<KyteaLM> subword_models;
    };

    explicit Model(Parts&& parts) : parts_(std::move(parts)) {}

    static Parts parse(bytes::BinaryReader& r);

    Parts parts_;
};

}  // namespace segmentlib::kytea
