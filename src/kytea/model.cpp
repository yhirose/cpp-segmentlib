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

// Reads a KyteaString (length-prefixed array of char ids) as raw ids. Tag
// candidate strings are stored this way; the caller decodes them to UTF-8.
std::vector<CharId> read_kytea_string(BinaryReader& r) {
    const auto len = r.read<std::uint32_t>();
    std::vector<CharId> s(len);
    for (auto& c : s) {
        c = r.read<CharId>();
    }
    return s;
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

// Reads a FeatureLookup, retaining the tag-prediction members. The layout
// matches KyTea's readFeatureLookup exactly; the word-segmentation-only
// dictVector is read but discarded. `active == 0` means no lookup.
std::optional<TagFeatureLookup> read_tag_feature_lookup(BinaryReader& r) {
    const auto active = r.read<std::uint8_t>();
    if (active == 0) {
        return std::nullopt;
    }
    TagFeatureLookup fl;
    fl.char_dict = Automaton<FeatVec>::read(r, read_featvec);   // charDict
    fl.type_dict = Automaton<FeatVec>::read(r, read_featvec);   // typeDict
    fl.self_dict = Automaton<FeatVec>::read(r, read_featvec);   // selfDict
    skip_featvec(r);                                            // dictVector (WS only)
    fl.biases = read_featvec(r);                                // biases
    fl.tag_dict_vector = read_featvec(r);                       // tagDictVector
    fl.tag_unk_vector = read_featvec(r);                        // tagUnkVector
    return fl;
}

// Reads a KyteaModel (classifier), retaining what tag prediction needs. A class
// count of 0 means "no model" (nullopt). numWeights is derived exactly as
// KyteaModel::setNumClasses does at read time (see TagModel::num_weights).
std::optional<TagModel> read_tag_model(BinaryReader& r) {
    const auto num_classes = r.read<std::int32_t>();
    if (num_classes == 0) {
        return std::nullopt;
    }
    TagModel m;
    discard<std::uint8_t>(r);  // solver
    r.skip(std::size_t{static_cast<std::uint32_t>(num_classes)} * sizeof(std::int32_t));  // labels
    discard_bool(r);           // bias flag
    m.multiplier = r.read<double>();
    m.num_weights = (num_classes == 2) ? 1 : num_classes;
    m.lookup = read_tag_feature_lookup(r);
    return m;
}

// Reads one dictionary word entry, keeping the D-feature fields and the per-word
// tag information (candidate strings decoded to UTF-8 via `chars`, dictionary
// bitmasks, and the usually-absent per-word tag models).
WordEntry read_word_entry(BinaryReader& r, int num_tags, const CharTable& chars) {
    WordEntry e;
    e.char_length = r.read<std::uint32_t>();  // word length in characters...
    r.skip(std::size_t{e.char_length} * kCharBytes);  // ...then the characters
    e.tags.resize(num_tags);
    e.tag_in_dicts.resize(num_tags);
    e.tag_mods.resize(num_tags);
    for (int lev = 0; lev < num_tags; ++lev) {
        const auto n_cands = r.read<std::uint32_t>();
        e.tags[lev].reserve(n_cands);
        e.tag_in_dicts[lev].reserve(n_cands);
        for (std::uint32_t j = 0; j < n_cands; ++j) {
            e.tags[lev].push_back(chars.decode(read_kytea_string(r)));  // tag candidate
            e.tag_in_dicts[lev].push_back(r.read<std::uint8_t>());      // tagInDicts
        }
    }
    e.in_dict = r.read<std::uint8_t>();
    for (int lev = 0; lev < num_tags; ++lev) {
        e.tag_mods[lev] = read_tag_model(r);  // per-word tag model (usually absent)
    }
    return e;
}

// KyTea's NEG_INFINITY sentinel (model-io.cpp): language-model entries written
// with this exact value are absent and must not be inserted into the maps.
constexpr double kNegInfinity = -999.0;

// Reads one subword-dictionary entry (KyTea's readEntry<ProbTagEntry>). The
// surface word is consumed only for its length (the automaton keys the entry);
// candidate readings are kept as raw char ids for the char-id language model.
ProbSubwordEntry read_prob_subword_entry(BinaryReader& r, int num_tags) {
    ProbSubwordEntry e;
    e.word_length = static_cast<std::uint32_t>(read_kytea_string(r).size());  // word (key)
    e.tags.resize(num_tags);
    e.probs.resize(num_tags);
    for (int lev = 0; lev < num_tags; ++lev) {
        const auto n_cand = r.read<std::uint32_t>();
        e.tags[lev].resize(n_cand);
        e.probs[lev].resize(n_cand);
        for (std::uint32_t j = 0; j < n_cand; ++j) {
            e.tags[lev][j] = read_kytea_string(r);  // reading (raw char ids)
            e.probs[lev][j] = r.read<double>();
        }
    }
    return e;
}

// Reads one KyteaLM (KyTea's readLM). n == 0 marks an absent model. Each entry
// is a key (raw char ids) plus a probability, and — unless the key is already
// the full n-gram length — a fallback weight; NEG_INFINITY values are skipped.
KyteaLM read_lm(BinaryReader& r) {
    KyteaLM lm;
    lm.n = r.read<std::uint32_t>();
    if (lm.n == 0) {
        return lm;  // absent language model
    }
    lm.vocab_size = r.read<std::uint32_t>();
    auto n_entries = r.read<std::uint32_t>();
    while (n_entries-- != 0) {
        const std::vector<CharId> key = read_kytea_string(r);
        const std::u16string k(key.begin(), key.end());
        const double prob = r.read<double>();
        if (prob != kNegInfinity) {
            lm.probs.emplace(k, prob);
        }
        if (key.size() != lm.n) {
            const double fallback = r.read<double>();
            if (fallback != kNegInfinity) {
                lm.fallbacks.emplace(k, fallback);
            }
        }
    }
    return lm;
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

    // --- global tag models: numTags x (word list + KyteaModel) ---
    std::vector<std::vector<std::string>> global_tags(cfg.num_tags);
    std::vector<std::optional<TagModel>> global_mods(cfg.num_tags);
    for (int i = 0; i < cfg.num_tags; ++i) {
        const auto n_words = r.read<std::uint32_t>();  // tag word list
        global_tags[i].reserve(n_words);
        for (std::uint32_t j = 0; j < n_words; ++j) {
            global_tags[i].push_back(chars.decode(read_kytea_string(r)));
        }
        global_mods[i] = read_tag_model(r);
    }

    // --- word dictionary (needed for the D features) ---
    const int num_tags = cfg.num_tags;
    Automaton<WordEntry> word_dict =
        Automaton<WordEntry>::read(r, [num_tags, &chars](BinaryReader& rr) {
            return read_word_entry(rr, num_tags, chars);
        });

    // --- subword dictionary (ProbTagEntry) + per-level reading LMs ---
    Automaton<ProbSubwordEntry> subword_dict =
        Automaton<ProbSubwordEntry>::read(r, [num_tags](BinaryReader& rr) {
            return read_prob_subword_entry(rr, num_tags);
        });
    std::vector<KyteaLM> subword_models;
    subword_models.reserve(num_tags);
    for (int i = 0; i < num_tags; ++i) {
        subword_models.push_back(read_lm(r));
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
        .global_tags = std::move(global_tags),
        .global_mods = std::move(global_mods),
        .subword_dict = std::move(subword_dict),
        .subword_models = std::move(subword_models),
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
