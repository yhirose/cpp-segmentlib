#include "segmentlib/output.h"

namespace segmentlib {

namespace {

// Escapes the delimiter characters KyTea would otherwise treat as structure.
// All four are ASCII, so escaping byte-by-byte never touches a UTF-8
// continuation byte.
void append_escaped(std::string_view word, std::string& out) {
    for (const char c : word) {
        if (c == ' ' || c == '/' || c == '&' || c == '\\') {
            out.push_back('\\');
        }
        out.push_back(c);
    }
}

}  // namespace

void append_full_line(const Segments& segments, std::string_view original, std::string& out) {
    for (std::size_t i = 0; i < segments.size(); ++i) {
        if (i != 0) {
            out.push_back(' ');
        }
        const auto& [start, end] = segments[i];
        append_escaped(original.substr(start, end - start), out);
    }
}

}  // namespace segmentlib
