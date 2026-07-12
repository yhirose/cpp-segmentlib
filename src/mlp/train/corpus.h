#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "segmentlib/types.h"

// Training-side KyTea corpus readers (design.ja.md 6節). These headers live
// under src/ (not include/): the training pipeline is a development tool and
// has no public headers (mlp_module_design.ja.md 0節).

namespace segmentlib::mlp::train {

// The supervision available for one inter-codepoint gap. Full annotation
// yields only NoBound/Bound; partial annotation (6.2) also yields Unknown
// (` ` unkBound / `?` skipBound), which is excluded from the loss (5.5).
enum class BoundaryTag : std::uint8_t { NoBound, Bound, Unknown };

// One corpus sentence: the raw (un-normalized, unescaped, tag-stripped)
// surface text plus a boundary tag per codepoint gap. tags.size() is the
// codepoint count minus one (empty for a 0/1-codepoint sentence). Tags are
// per *codepoint* gap because the corpus marks boundaries between characters;
// mapping onto EGC boundaries (and detecting conflicts with EGC interiors)
// happens later, in example generation (5.2/5.5).
struct AnnotatedSentence {
    std::string text;
    std::vector<BoundaryTag> tags;
};

// Parses a full-annotation corpus (6.1): words separated by unescaped spaces,
// `/`-introduced tags stripped, `\` escapes honored. Every gap gets a
// definite tag (Bound between words, NoBound inside a word). Empty lines are
// skipped. Errors: InvalidUtf8, or MalformedCorpus for structural problems
// (e.g. a trailing escape character).
[[nodiscard]] std::expected<std::vector<AnnotatedSentence>, Error>
parse_full_corpus(std::string_view content);

// Parses a partial-annotation corpus (6.2): codepoints alternating with a
// separator (`-` NoBound, `|` Bound, ` `/`?` Unknown), optional `/tag...`
// runs before a separator (stripped). A single trailing separator (e.g. a
// final `|` closing the last word) is tolerated and dropped.
[[nodiscard]] std::expected<std::vector<AnnotatedSentence>, Error>
parse_partial_corpus(std::string_view content);

// File-reading wrappers around the parsers above. IoError if the file cannot
// be read.
[[nodiscard]] std::expected<std::vector<AnnotatedSentence>, Error>
read_full_corpus(const std::filesystem::path& path);
[[nodiscard]] std::expected<std::vector<AnnotatedSentence>, Error>
read_partial_corpus(const std::filesystem::path& path);

// Reads a dictionary file (one word per line; an unescaped-tab/space-separated
// tag column, as in KyTea's `単語 タグ` format, is ignored). Empty lines are
// skipped. The words are raw surface forms; normalization and EGC splitting
// happen at Aho-Corasick construction (example.h), matching how the inference
// loader treats model-file dictionaries (5.7 field 17b).
[[nodiscard]] std::expected<std::vector<std::string>, Error>
read_dictionary(const std::filesystem::path& path);

}  // namespace segmentlib::mlp::train
