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
        const auto& s = segments[i];
        append_escaped(original.substr(s.start, s.end - s.start), out);
    }
}

void append_boundary_line(const Boundaries& cuts, std::string_view original, std::string& out) {
    // The words are the substrings delimited by the cuts, with implicit
    // endpoints at 0 and original.size(). Same surface + escaping as
    // append_full_line, so byte-identical to a tag-suppressed word line, but
    // fed straight from the (tag-free) boundary result.
    std::size_t prev = 0;
    for (std::size_t i = 0; i <= cuts.size(); ++i) {
        const std::size_t end = (i < cuts.size()) ? cuts[i] : original.size();
        if (i != 0) {
            out.push_back(' ');
        }
        append_escaped(original.substr(prev, end - prev), out);
        prev = end;
    }
}

void append_tagged_line(const Segments& segments, std::string_view original, std::string& out) {
    for (std::size_t i = 0; i < segments.size(); ++i) {
        if (i != 0) {
            out.push_back(' ');
        }
        const auto& s = segments[i];
        append_escaped(original.substr(s.start, s.end - s.start), out);
        // One "/tag" per level (POS, reading, ...). Tag strings come straight
        // from the model and are emitted unescaped, exactly as KyTea's showString
        // does. Empty levels (unknown-word readings deferred to stage B) are
        // skipped rather than emitting a bare "/".
        for (const std::string_view tag : s.tags) {
            if (tag.empty()) {
                continue;
            }
            out.push_back('/');
            out.append(tag);
        }
    }
}

}  // namespace segmentlib
