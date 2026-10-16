#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "mlp/train/quantize.h"
#include "segmentlib/mlp/model.h"
#include "segmentlib/mlp/vocab.h"
#include "segmentlib/types.h"

// Model export in the 4.7 format (design.ja.md 4.7): header line, config, the five scales, vocabulary, int16 tensors,
// raw-double biases, then the raw dictionary word lists. Derived structures
// (precompute table, Aho-Corasick, requantized biases) are deliberately not
// written — the loader rebuilds them (5.7 ロード時の処理).

namespace segmentlib::mlp::train {

// Serializes the model to the 5.7 byte layout, behind `format`'s header line
// (mlp::kMlpFormat by default, ed::kEdFormat for the EDLA trainer). The body is
// identical either way -- EDLA changes how the weights were learned, not what
// the file has to record -- so only the first line tells the two apart. `vocab` must be the
// vocabulary the model was trained with (quantized.config.vocab_size ==
// vocab.size()), and `dictionaries` the raw word lists of the training
// dictionary channels, in channel order (dictionaries.size() ==
// quantized.config.num_dicts). Words are written as their raw surface forms;
// the loader normalizes and EGC-splits them (5.7 field 17b).
[[nodiscard]] std::vector<std::byte>
serialize_model(const QuantizedModel& quantized, const Vocab& vocab,
                std::span<const std::vector<std::string>> dictionaries,
                const mlp::FormatId& format = mlp::kMlpFormat);

// serialize_model + write to a file. IoError on failure.
[[nodiscard]] std::expected<void, Error>
export_model(const std::filesystem::path& path, const QuantizedModel& quantized,
             const Vocab& vocab,
             std::span<const std::vector<std::string>> dictionaries,
             const mlp::FormatId& format = mlp::kMlpFormat);

}  // namespace segmentlib::mlp::train
