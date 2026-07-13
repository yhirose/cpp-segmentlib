#pragma once

#include <string>
#include <string_view>

#include "segmentlib/types.h"

namespace segmentlib {

// Appends `segments` to `out` as a KyTea-style word-segmentation line: surface
// words separated by single spaces, with the delimiter characters (space, '/',
// '&', '\') backslash-escaped exactly as KyTea's writeSentence does. No
// trailing newline is added. `original` is the source text the segment
// offsets index.
void append_full_line(const Segments& segments, std::string_view original, std::string& out);

}  // namespace segmentlib
