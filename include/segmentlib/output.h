#pragma once

#include <string>
#include <string_view>

#include "segmentlib/types.h"

namespace segmentlib {

// Appends `segments` to `out` as a KyTea full-annotation line: surface words
// separated by single spaces, with the delimiter characters (space, '/', '&',
// '\') backslash-escaped exactly as KyTea's writeSentence does. No trailing
// newline is added. `original` is the source text the segment offsets index.
void append_full_line(const Segments& segments, std::string_view original, std::string& out);

// Appends a segmentation-only line (surface words, space-separated, escaped)
// straight from a boundary result — no tag prediction. Byte-identical to
// append_full_line's output for the same segmentation, but driven by the
// tag-free `tokenize_boundaries` path.
void append_boundary_line(const Boundaries& cuts, std::string_view original, std::string& out);

// Like append_full_line, but also appends each word's predicted tags as
// "/tag" per level (surface/POS/reading), matching KyTea's default full output.
// Surfaces are escaped; tag strings are emitted verbatim (as KyTea does). Empty
// tag levels (unknown-word readings, deferred to stage B) are omitted.
void append_tagged_line(const Segments& segments, std::string_view original, std::string& out);

}  // namespace segmentlib
