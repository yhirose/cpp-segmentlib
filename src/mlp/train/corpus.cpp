#include "mlp/train/corpus.h"

#include <fstream>
#include <sstream>
#include <utility>

#include "segmentlib/unicode/utf8.h"

namespace segmentlib::mlp::train {

namespace {

constexpr Error kInvalidUtf8{ErrorCode::InvalidUtf8, "invalid UTF-8 in corpus"};
constexpr Error kTrailingEscape{ErrorCode::MalformedCorpus,
                                "corpus line ends with an escape character"};
constexpr Error kMissingSeparator{
    ErrorCode::MalformedCorpus,
    "partial-annotation line has adjacent characters without a boundary mark"};
constexpr Error kLeadingSeparator{
    ErrorCode::MalformedCorpus,
    "partial-annotation line starts with a boundary mark"};

// One codepoint of a corpus line, with KyTea's `\` escaping resolved
// (design.ja.md 6節: escape applies to the delimiter characters).
struct Item {
    char32_t cp;
    bool escaped;
};

// Decodes the next item of `line` starting at `pos`, advancing `pos`.
// Returns nullopt at end of line; an Error on bad UTF-8 / trailing escape.
std::expected<std::optional<Item>, Error> next_item(std::string_view line,
                                                    std::size_t& pos) {
    if (pos >= line.size()) {
        return std::optional<Item>{};
    }
    auto decoded = unicode::decode(line.substr(pos));
    if (!decoded) {
        return std::unexpected(kInvalidUtf8);
    }
    pos += decoded->length;
    if (decoded->codepoint != U'\\') {
        return std::optional<Item>{Item{decoded->codepoint, false}};
    }
    if (pos >= line.size()) {
        return std::unexpected(kTrailingEscape);
    }
    decoded = unicode::decode(line.substr(pos));
    if (!decoded) {
        return std::unexpected(kInvalidUtf8);
    }
    pos += decoded->length;
    return std::optional<Item>{Item{decoded->codepoint, true}};
}

// Parses one full-annotation line (6.1) into a sentence, or nullopt if the
// line contains no words.
std::expected<std::optional<AnnotatedSentence>, Error>
parse_full_line(std::string_view line) {
    AnnotatedSentence sentence;
    std::size_t word_cps = 0;   // codepoints of the word being read
    std::size_t total_cps = 0;  // codepoints of the whole sentence so far
    bool in_tag = false;

    std::size_t pos = 0;
    while (true) {
        auto item = next_item(line, pos);
        if (!item) {
            return std::unexpected(item.error());
        }
        if (!*item) {
            break;
        }
        const auto [cp, escaped] = **item;
        if (!escaped && cp == U' ') {
            word_cps = 0;  // consecutive spaces / tag-only fragments are no-ops
            in_tag = false;
            continue;
        }
        if (!escaped && cp == U'/') {
            in_tag = true;
            continue;
        }
        if (in_tag) {
            continue;  // tag content (including `&` alternatives) is dropped
        }
        if (word_cps > 0) {
            sentence.tags.push_back(BoundaryTag::NoBound);
        } else if (total_cps > 0) {  // first codepoint of a later word
            sentence.tags.push_back(BoundaryTag::Bound);
        }
        unicode::encode(cp, sentence.text);
        ++word_cps;
        ++total_cps;
    }

    if (total_cps == 0) {
        return std::optional<AnnotatedSentence>{};
    }
    return std::optional<AnnotatedSentence>{std::move(sentence)};
}

// Parses one partial-annotation line (6.2), or nullopt if the line is empty.
std::expected<std::optional<AnnotatedSentence>, Error>
parse_partial_line(std::string_view line) {
    AnnotatedSentence sentence;
    std::size_t cps = 0;
    bool expect_char = true;
    bool in_tag = false;

    std::size_t pos = 0;
    while (true) {
        auto item = next_item(line, pos);
        if (!item) {
            return std::unexpected(item.error());
        }
        if (!*item) {
            break;
        }
        const auto [cp, escaped] = **item;
        const bool is_separator =
            !escaped && (cp == U' ' || cp == U'-' || cp == U'|' || cp == U'?');
        if (in_tag) {
            if (!is_separator) {
                continue;  // tag content is dropped
            }
            in_tag = false;  // a separator ends the tag run; handle it below
        }
        if (expect_char) {
            if (is_separator) {
                return std::unexpected(cps == 0 ? kLeadingSeparator
                                                : kMissingSeparator);
            }
            if (!escaped && cp == U'/') {
                return std::unexpected(kLeadingSeparator);
            }
            unicode::encode(cp, sentence.text);
            ++cps;
            expect_char = false;
            continue;
        }
        // After a character: either a tag run or a separator must follow.
        if (!escaped && cp == U'/') {
            in_tag = true;
            continue;
        }
        if (!is_separator) {
            return std::unexpected(kMissingSeparator);
        }
        switch (cp) {
            case U'-': sentence.tags.push_back(BoundaryTag::NoBound); break;
            case U'|': sentence.tags.push_back(BoundaryTag::Bound); break;
            default:   sentence.tags.push_back(BoundaryTag::Unknown); break;
        }
        expect_char = true;
    }

    if (cps == 0) {
        return std::optional<AnnotatedSentence>{};
    }
    if (expect_char) {
        // The line ended on a separator; `|` doubles as the last word's
        // terminator (6.2), so a single dangling tag is dropped.
        sentence.tags.pop_back();
    }
    return std::optional<AnnotatedSentence>{std::move(sentence)};
}

template <class ParseLine>
std::expected<std::vector<AnnotatedSentence>, Error>
parse_lines(std::string_view content, ParseLine parse_line) {
    std::vector<AnnotatedSentence> sentences;
    std::size_t pos = 0;
    while (pos <= content.size()) {
        std::size_t eol = content.find('\n', pos);
        if (eol == std::string_view::npos) {
            if (pos == content.size()) {
                break;
            }
            eol = content.size();
        }
        std::string_view line = content.substr(pos, eol - pos);
        if (line.ends_with('\r')) {
            line.remove_suffix(1);
        }
        auto parsed = parse_line(line);
        if (!parsed) {
            return std::unexpected(parsed.error());
        }
        if (*parsed) {
            sentences.push_back(std::move(**parsed));
        }
        pos = eol + 1;
    }
    return sentences;
}

std::expected<std::string, Error> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::unexpected(Error{ErrorCode::IoError, "cannot open corpus file"});
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    if (in.bad()) {
        return std::unexpected(Error{ErrorCode::IoError, "cannot read corpus file"});
    }
    return std::move(buffer).str();
}

}  // namespace

std::expected<std::vector<AnnotatedSentence>, Error>
parse_full_corpus(std::string_view content) {
    return parse_lines(content, parse_full_line);
}

std::expected<std::vector<AnnotatedSentence>, Error>
parse_partial_corpus(std::string_view content) {
    return parse_lines(content, parse_partial_line);
}

std::expected<std::vector<AnnotatedSentence>, Error>
read_full_corpus(const std::filesystem::path& path) {
    auto content = read_file(path);
    if (!content) {
        return std::unexpected(content.error());
    }
    return parse_full_corpus(*content);
}

std::expected<std::vector<AnnotatedSentence>, Error>
read_partial_corpus(const std::filesystem::path& path) {
    auto content = read_file(path);
    if (!content) {
        return std::unexpected(content.error());
    }
    return parse_partial_corpus(*content);
}

std::expected<std::vector<std::string>, Error>
read_dictionary(const std::filesystem::path& path) {
    auto content = read_file(path);
    if (!content) {
        return std::unexpected(content.error());
    }
    std::vector<std::string> words;
    std::size_t pos = 0;
    const std::string_view text = *content;
    while (pos <= text.size()) {
        std::size_t eol = text.find('\n', pos);
        if (eol == std::string_view::npos) {
            if (pos == text.size()) {
                break;
            }
            eol = text.size();
        }
        std::string_view line = text.substr(pos, eol - pos);
        if (line.ends_with('\r')) {
            line.remove_suffix(1);
        }
        // The word is the first tab/space-separated field ("単語 タグ").
        const std::size_t sep = line.find_first_of(" \t");
        if (sep != std::string_view::npos) {
            line = line.substr(0, sep);
        }
        if (!line.empty()) {
            words.emplace_back(line);
        }
        pos = eol + 1;
    }
    return words;
}

}  // namespace segmentlib::mlp::train
