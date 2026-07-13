#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <vector>

#include "segmentlib/kytea/automaton.h"
#include "segmentlib/kytea/char_table.h"
#include "segmentlib/types.h"

namespace segmentlib::kytea {

// Quantized feature-weight type used by KyTea's default (non-"NQ") models.
using FeatVal = std::int16_t;
using FeatVec = std::vector<FeatVal>;

// The subset of KyTea's training configuration that word segmentation needs.
// `num_tags`/`do_tags` are kept only to size and skip the model file's tag
// sections during parsing (this library does not retain or predict tags).
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

// Per-dictionary-word data needed for the word-segmentation D features. The
// dictionary automaton carries this as its payload.
struct WordEntry {
    std::uint32_t char_length = 0;  // length of the word in characters
    std::uint8_t in_dict = 0;       // bitmask: which sub-dictionaries contain it
};

// A loaded KyTea model, holding exactly what word segmentation needs. Tag /
// reading / language-model sections of the file are parsed only far enough to
// advance past them (they are not retained). Immutable once loaded.
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
    };

    explicit Model(Parts&& parts) : parts_(std::move(parts)) {}

    static Parts parse(bytes::BinaryReader& r);

    Parts parts_;
};

}  // namespace segmentlib::kytea
