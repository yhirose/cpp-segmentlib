#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "mlp/train/quantize.h"
#include "segmentlib/mlp/vocab.h"
#include "segmentlib/types.h"

// Model export in the 5.7 format (design.ja.md 5.7, mlp_impl_design.ja.md
// II.7): header line, config, the five scales, vocabulary, int16 tensors,
// raw-double biases, then the raw dictionary word lists. Derived structures
// (precompute table, Aho-Corasick, requantized biases) are deliberately not
// written — the loader rebuilds them (5.7 ロード時の処理).

namespace segmentlib::mlp::train {

// Serializes the model to the 5.7 byte layout. `vocab` must be the
// vocabulary the model was trained with (quantized.config.vocab_size ==
// vocab.size()), and `dictionaries` the raw word lists of the training
// dictionary channels, in channel order (dictionaries.size() ==
// quantized.config.num_dicts). Words are written as their raw surface forms;
// the loader normalizes and EGC-splits them (5.7 field 17b).
[[nodiscard]] std::vector<std::byte>
serialize_model(const QuantizedModel& quantized, const Vocab& vocab,
                std::span<const std::vector<std::string>> dictionaries);

// serialize_model + write to a file. IoError on failure.
[[nodiscard]] std::expected<void, Error>
export_model(const std::filesystem::path& path, const QuantizedModel& quantized,
             const Vocab& vocab,
             std::span<const std::vector<std::string>> dictionaries);

}  // namespace segmentlib::mlp::train
