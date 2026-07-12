#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace segmentlib {

// A single segmented token. Offsets are into the original UTF-8 input.
struct Segment {
    std::size_t start;  // byte offset, inclusive
    std::size_t end;    // byte offset, exclusive
    // Predicted tags by level (POS, reading, ...); empty when tag prediction is
    // off. Owned strings, since unknown-word readings are computed at runtime
    // rather than borrowed from the model.
    std::vector<std::string> tags;

    bool operator==(const Segment&) const = default;
};

using Segments = std::vector<Segment>;

// Byte offsets of word-boundary cut points, strictly between 0 and the input
// size, in increasing order. The words are the substrings delimited by these
// cuts (with implicit endpoints at 0 and input.size()).
using Boundaries = std::vector<std::size_t>;

enum class ErrorCode {
    InvalidUtf8,
    ModelNotLoaded,
    UnsupportedModelFormat,
    MalformedModel,
    MalformedCorpus,
    IoError,
};

// A failure result. `message` points at a static string (no dynamic storage),
// so an Error can be copied and returned freely.
struct Error {
    ErrorCode code;
    std::string_view message;
};

}  // namespace segmentlib
