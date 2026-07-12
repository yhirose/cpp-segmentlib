#pragma once

#include <string>
#include <vector>

#include "segmentlib/kytea/char_table.h"
#include "segmentlib/kytea/model.h"

namespace segmentlib::kytea {

// Predicts the top tag string for every level of one word, appending them (in
// level order) to `out`. Implements the known-word path of KyTea's
// Kytea::calculateTags: it dispatches each level to the global or per-word
// classifier, accumulates feature scores, and picks the highest-confidence
// candidate. Only the winning candidate string per level is produced — margins
// and probabilities (which don't change the argmax) are skipped.
//
// `enc` is the whole sentence; [start_char, end_char) is the word's character
// span. `ent` is the word-dictionary entry for that span (nullptr if the word
// is unknown), already looked up by the caller and reused here. Levels whose
// candidates come from a per-word model that is absent (unknown-word readings,
// deferred to stage B) contribute an empty string_view, preserving level order.
//
// Known tags are copied from the model; unknown-word readings are estimated by
// the subword language model (stage B) and decoded to UTF-8. A level with no
// candidate at all contributes KyTea's default tag ("UNK").
void predict_word_tags(const Model& model, const EncodedText& enc,
                       std::size_t start_char, std::size_t end_char,
                       const WordEntry* ent, std::vector<std::string>& out);

}  // namespace segmentlib::kytea
