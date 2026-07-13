#include "segmentlib/kytea/model.h"

#include <fstream>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "segmentlib/bytes/binary_reader.h"

namespace segmentlib::kytea {

namespace {

using bytes::BinaryReader;
using bytes::ParseError;

// KyTea's KyteaChar is a 16-bit code unit; strings are length-prefixed arrays.
constexpr std::size_t kCharBytes = sizeof(CharId);

// Reads and discards a value (used to advance past fields we don't retain).
template <class T>
void discard(BinaryReader& r) {
    static_cast<void>(r.read<T>());
}
void discard_bool(BinaryReader& r) {
    static_cast<void>(r.read_bool());
}

FeatVec read_featvec(BinaryReader& r) {
    const auto n = r.read<std::uint32_t>();
    FeatVec v(n);
    for (auto& x : v) {
        x = r.read<FeatVal>();
    }
    return v;
}

void skip_featvec(BinaryReader& r) {
    const auto n = r.read<std::uint32_t>();
    r.skip(std::size_t{n} * sizeof(FeatVal));
}

// Advances past a KyteaString (length-prefixed array of char ids) without
// materializing it. Tag candidate strings are stored this way.
void skip_kytea_string(BinaryReader& r) {
    const auto len = r.read<std::uint32_t>();
    r.skip(std::size_t{len} * kCharBytes);
}

// Advances past a Dictionary<Entry> without materializing it. Used for the
// selfDict (empty in WS models) and any dictionaries in skipped tag models.
void skip_dictionary(BinaryReader& r, const std::function<void(BinaryReader&)>& skip_entry) {
    discard<std::uint8_t>(r);  // numDicts
    const auto n_states = r.read<std::uint32_t>();
    if (n_states == 0) {
        return;  // absent dictionary: no entries follow
    }
    for (std::uint32_t s = 0; s < n_states; ++s) {
        discard<std::uint32_t>(r);  // failure
        const auto n_gotos = r.read<std::uint32_t>();
        r.skip(std::size_t{n_gotos} * (kCharBytes + sizeof(std::uint32_t)));
        const auto n_out = r.read<std::uint32_t>();
        r.skip(std::size_t{n_out} * sizeof(std::uint32_t));
        discard<std::uint8_t>(r);  // isBranch
    }
    const auto n_entries = r.read<std::uint32_t>();
    for (std::uint32_t i = 0; i < n_entries; ++i) {
        skip_entry(r);
    }
}

// Advances past a FeatureLookup (the classifier attached to a KyteaModel).
// `active == 0` means no lookup and nothing further to skip.
void skip_feature_lookup(BinaryReader& r) {
    const auto active = r.read<std::uint8_t>();
    if (active == 0) {
        return;
    }
    skip_dictionary(r, skip_featvec);  // charDict
    skip_dictionary(r, skip_featvec);  // typeDict
    skip_dictionary(r, skip_featvec);  // selfDict
    skip_featvec(r);                  // dictVector (WS only)
    skip_featvec(r);                  // biases
    skip_featvec(r);                  // tagDictVector
    skip_featvec(r);                  // tagUnkVector
}

// Advances past a KyteaModel (classifier). A class count of 0 means "no
// model" and nothing further to skip.
void skip_tag_model(BinaryReader& r) {
    const auto num_classes = r.read<std::int32_t>();
    if (num_classes == 0) {
        return;
    }
    discard<std::uint8_t>(r);  // solver
    r.skip(std::size_t{static_cast<std::uint32_t>(num_classes)} * sizeof(std::int32_t));  // labels
    discard_bool(r);           // bias flag
    discard<double>(r);        // multiplier
    skip_feature_lookup(r);
}

// Reads one dictionary word entry, keeping only the D-feature fields; the
// per-word tag information (candidate strings, dictionary bitmasks, per-word
// tag models) is skipped.
WordEntry read_word_entry(BinaryReader& r, int num_tags) {
    WordEntry e;
    e.char_length = r.read<std::uint32_t>();  // word length in characters...
    r.skip(std::size_t{e.char_length} * kCharBytes);  // ...then the characters
    for (int lev = 0; lev < num_tags; ++lev) {
        const auto n_cands = r.read<std::uint32_t>();
        for (std::uint32_t j = 0; j < n_cands; ++j) {
            skip_kytea_string(r);        // tag candidate
            discard<std::uint8_t>(r);    // tagInDicts
        }
    }
    e.in_dict = r.read<std::uint8_t>();
    for (int lev = 0; lev < num_tags; ++lev) {
        skip_tag_model(r);  // per-word tag model (usually absent)
    }
    return e;
}

// Advances past one subword-dictionary entry (KyTea's ProbTagEntry).
void skip_prob_subword_entry(BinaryReader& r, int num_tags) {
    skip_kytea_string(r);  // word (key)
    for (int lev = 0; lev < num_tags; ++lev) {
        const auto n_cand = r.read<std::uint32_t>();
        for (std::uint32_t j = 0; j < n_cand; ++j) {
            skip_kytea_string(r);  // reading (raw char ids)
            discard<double>(r);    // probability
        }
    }
}

// Advances past one KyteaLM (a per-tag-level reading language model). `n ==
// 0` marks an absent model, with nothing further to skip.
void skip_lm(BinaryReader& r) {
    const auto n = r.read<std::uint32_t>();
    if (n == 0) {
        return;
    }
    discard<std::uint32_t>(r);  // vocabSize
    auto n_entries = r.read<std::uint32_t>();
    while (n_entries-- != 0) {
        const auto key_len = r.read<std::uint32_t>();
        r.skip(std::size_t{key_len} * kCharBytes);  // key
        discard<double>(r);                          // probability
        if (key_len != n) {
            discard<double>(r);  // fallback weight
        }
    }
}

struct Header {
    std::string version;
    char format = 0;
    std::string encoding;
};

Header parse_header(BinaryReader& r) {
    const std::string line = r.read_line();
    // Expected: "KyTea <version> <T|B> <encoding>"
    std::vector<std::string_view> tok;
    std::string_view sv{line};
    std::size_t pos = 0;
    while (pos < sv.size()) {
        const auto sp = sv.find(' ', pos);
        const auto end = (sp == std::string_view::npos) ? sv.size() : sp;
        if (end > pos) {
            tok.push_back(sv.substr(pos, end - pos));
        }
        pos = (sp == std::string_view::npos) ? sv.size() : sp + 1;
    }
    if (tok.size() != 4 || tok[0] != "KyTea" || tok[2].size() != 1) {
        throw ParseError("malformed model header");
    }
    return Header{std::string(tok[1]), tok[2][0], std::string(tok[3])};
}

}  // namespace

Model::Parts Model::parse(BinaryReader& r) {
    const Header header = parse_header(r);
    if (header.format != 'B') {
        throw ParseError("only binary KyTea models are supported");
    }
    if (header.version != "0.4.0") {
        // "0.4.0NQ" (non-quantized, DISABLE_QUANTIZE builds) and other versions
        // are intentionally unsupported: distributed models are the quantized
        // "0.4.0" build, so we reject anything else rather than risk misreading.
        throw ParseError("unsupported KyTea model version");
    }
    if (header.encoding != "utf8") {
        throw ParseError("only utf8 KyTea models are supported");
    }

    Config cfg;
    cfg.do_ws = r.read_bool();
    cfg.do_tags = r.read_bool();
    cfg.num_tags = static_cast<int>(r.read<std::uint32_t>());
    cfg.char_window = r.read<std::uint8_t>();
    cfg.char_n = r.read<std::uint8_t>();
    cfg.type_window = r.read<std::uint8_t>();
    cfg.type_n = r.read<std::uint8_t>();
    cfg.dict_n = r.read<std::uint8_t>();
    discard_bool(r);     // bias flag (config-level; the model's own flag is used)
    discard<double>(r);  // epsilon
    cfg.solver = r.read<std::uint8_t>();

    CharTable chars(r.read_cstring());

    // --- wsModel (KyteaModel) ---
    const auto num_classes = r.read<std::int32_t>();
    if (num_classes < 2) {
        throw ParseError("model has no word-segmentation classifier");
    }
    discard<std::uint8_t>(r);  // solver
    r.skip(std::size_t{static_cast<std::uint32_t>(num_classes)} * sizeof(std::int32_t));  // labels
    discard_bool(r);  // bias flag
    const double multiplier = r.read<double>();

    // --- FeatureLookup (keep the WS parts, skip the tag parts) ---
    if (r.read<std::uint8_t>() == 0) {
        throw ParseError("word-segmentation model has no feature lookup");
    }
    Automaton<FeatVec> char_dict = Automaton<FeatVec>::read(r, read_featvec);
    Automaton<FeatVec> type_dict = Automaton<FeatVec>::read(r, read_featvec);
    skip_dictionary(r, skip_featvec);          // selfDict
    FeatVec dict_vector = read_featvec(r);
    FeatVec biases = read_featvec(r);
    skip_featvec(r);                           // tagDictVector
    skip_featvec(r);                           // tagUnkVector

    // --- global tag models: numTags x (word list + KyteaModel), skipped ---
    for (int i = 0; i < cfg.num_tags; ++i) {
        const auto n_words = r.read<std::uint32_t>();  // tag word list
        for (std::uint32_t j = 0; j < n_words; ++j) {
            skip_kytea_string(r);
        }
        skip_tag_model(r);
    }

    // --- word dictionary (needed for the D features) ---
    const int num_tags = cfg.num_tags;
    Automaton<WordEntry> word_dict =
        Automaton<WordEntry>::read(r, [num_tags](BinaryReader& rr) {
            return read_word_entry(rr, num_tags);
        });

    // --- subword dictionary (ProbTagEntry) + per-level reading LMs, skipped ---
    skip_dictionary(r, [num_tags](BinaryReader& rr) {
        skip_prob_subword_entry(rr, num_tags);
    });
    for (int i = 0; i < num_tags; ++i) {
        skip_lm(r);
    }

    return Model::Parts{
        .config = cfg,
        .chars = std::move(chars),
        .multiplier = multiplier,
        .char_dict = std::move(char_dict),
        .type_dict = std::move(type_dict),
        .dict_vector = std::move(dict_vector),
        .biases = std::move(biases),
        .word_dict = std::move(word_dict),
    };
}

std::expected<Model, Error> Model::load_from_bytes(std::span<const std::byte> data) {
    try {
        BinaryReader r(data);
        return Model(parse(r));
    } catch (const ParseError&) {
        return std::unexpected(Error{ErrorCode::MalformedModel, "malformed KyTea model"});
    }
}

std::expected<Model, Error> Model::load(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::unexpected(Error{ErrorCode::IoError, "could not open model file"});
    }
    std::vector<std::byte> buf;
    in.seekg(0, std::ios::end);
    const auto size = in.tellg();
    if (size < 0) {
        return std::unexpected(Error{ErrorCode::IoError, "could not read model file"});
    }
    buf.resize(static_cast<std::size_t>(size));
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    if (!in) {
        return std::unexpected(Error{ErrorCode::IoError, "could not read model file"});
    }
    return load_from_bytes(buf);
}

}  // namespace segmentlib::kytea
