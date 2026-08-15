#include "segmentlib/mlp/model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ranges>
#include <fstream>
#include <string>
#include <utility>

#include "segmentlib/bytes/binary_reader.h"
#include "segmentlib/unicode/egc.h"

namespace segmentlib::mlp {

namespace {

using bytes::ParseError;

std::vector<std::int16_t> read_i16_tensor(bytes::BinaryReader& in,
                                          std::size_t count) {
    in.require_capacity(count, sizeof(std::int16_t));
    std::vector<std::int16_t> tensor(count);
    for (std::int16_t& v : tensor) {
        v = in.read<std::int16_t>();
    }
    return tensor;
}

double read_scale(bytes::BinaryReader& in, const char* what) {
    const double s = in.read<double>();
    if (!(s > 0.0) || !std::isfinite(s)) {
        throw ParseError(std::string("invalid scale: ") + what);
    }
    return s;
}

}  // namespace

Model::Parts Model::parse(bytes::BinaryReader& in, TablePrecision precision,
                          unsigned format) {
    Parts parts;

    // Config (fields 1-4b).
    Config& c = parts.config;
    c.char_window = in.read<std::uint8_t>();
    c.embed_dim = in.read<std::uint16_t>();
    c.hidden = in.read<std::uint16_t>();
    c.num_dicts = in.read<std::uint8_t>();
    c.unicode_version = in.read<std::uint16_t>();
    if (c.char_window < 1 || c.embed_dim < 1 || c.embed_dim > 256 ||
        c.hidden < 1) {
        throw ParseError("implausible model configuration");
    }
    if (c.unicode_version != unicode::kEgcUnicodeVersion) {
        parts.unicode_mismatch = true;
        std::fprintf(stderr,
                     "segmentlib: MLP model was trained with Unicode %u.%u EGC "
                     "rules but this build uses %u.%u; segmentation of "
                     "clusters affected by the difference may diverge\n",
                     c.unicode_version / 100, c.unicode_version % 100,
                     unicode::kEgcUnicodeVersion / 100,
                     unicode::kEgcUnicodeVersion % 100);
    }

    // Scales (fields 5-8b).
    const double emb_scale = read_scale(in, "emb_scale");
    const double w1_scale = read_scale(in, "w1_scale");
    const double wdict_scale =
        c.num_dicts > 0 ? read_scale(in, "wdict_scale") : 1.0;
    const double w2_scale = read_scale(in, "w2_scale");
    const double acc_scale = read_scale(in, "acc_scale");

    // Vocabulary (fields 9-10).
    const std::uint32_t vocab_size = in.read<std::uint32_t>();
    if (vocab_size < 2) {
        throw ParseError("vocabulary smaller than the reserved rows");
    }
    in.require_capacity(vocab_size - 2, sizeof(std::uint32_t));
    std::vector<char32_t> codepoints(vocab_size - 2);
    char32_t prev = 0;
    for (std::size_t i = 0; i < codepoints.size(); ++i) {
        const auto cp = static_cast<char32_t>(in.read<std::uint32_t>());
        if (cp > 0x10FFFF || (i > 0 && cp <= prev)) {
            throw ParseError("vocabulary codepoints not strictly ascending");
        }
        prev = cp;
        codepoints[i] = cp;
    }
    parts.vocab = Vocab{std::move(codepoints)};

    // Layers (fields 11-16).
    const std::size_t d = c.embed_dim;
    const std::size_t h = c.hidden;
    const std::size_t in_dim = 2u * c.char_window * d;
    const std::size_t fd =
        static_cast<std::size_t>(c.num_dicts) * kDictFeaturesPerDict;
    parts.embedding = read_i16_tensor(in, vocab_size * d);
    parts.w1 = read_i16_tensor(in, h * in_dim);
    const std::vector<std::int16_t> wdict =
        c.num_dicts > 0 ? read_i16_tensor(in, h * fd)
                        : std::vector<std::int16_t>{};
    in.require_capacity(h, sizeof(double));
    std::vector<double> b1(h);
    for (double& b : b1) {
        b = in.read<double>();
    }
    parts.w2 = read_i16_tensor(in, h);
    const double b2 = in.read<double>();

    // Dictionaries (field 17): format 2 carries the compiled matcher, format 1
    // the word lists it has to be compiled from.
    CompiledDictionaries compiled;
    compiled.num_dicts = c.num_dicts;
    if (format == 2) {
        if (c.num_dicts > 0) {
            compiled.fst = in.read_blob(in.read<std::uint32_t>());
            const std::uint32_t sets = in.read<std::uint32_t>();
            in.require_capacity(std::uint64_t{sets} + 1, sizeof(std::uint32_t));
            compiled.offsets.reserve(sets + 1);
            for (std::uint32_t s = 0; s <= sets; ++s) {
                compiled.offsets.push_back(in.read<std::uint32_t>());
            }
            // The offsets index `dicts`, and the channels index W_dict's
            // columns, so both are checked before anything can use them.
            if (compiled.offsets.front() != 0 ||
                !std::ranges::is_sorted(compiled.offsets)) {
                throw ParseError("dictionary channel-set offsets are not ascending");
            }
            const std::uint32_t ids = compiled.offsets.back();
            in.require_capacity(ids, 1);
            compiled.dicts.reserve(ids);
            for (std::uint32_t i = 0; i < ids; ++i) {
                compiled.dicts.push_back(in.read<std::uint8_t>());
            }
            if (std::ranges::any_of(compiled.dicts, [&](std::uint8_t d) {
                    return d >= c.num_dicts;
                })) {
                throw ParseError("dictionary channel id out of range");
            }
        }
    } else {
        std::vector<std::vector<std::string>> dictionaries(c.num_dicts);
        for (auto& dict : dictionaries) {
            const std::uint32_t entries = in.read<std::uint32_t>();
            in.require_capacity(entries, 1);  // each is at least a NUL terminator
            dict.reserve(entries);
            for (std::uint32_t e = 0; e < entries; ++e) {
                dict.push_back(in.read_cstring());
            }
        }
        compiled = compile_dictionaries(dictionaries);
    }
    if (!in.eof()) {
        throw ParseError("trailing bytes after the model");
    }

    // Load-time derived structures (I.4): everything converted to the
    // accumulator integer scale once, here.
    const double r = emb_scale * w1_scale / acc_scale;
    parts.precompute =
        PrecomputeTable(parts.embedding, parts.w1, vocab_size, c.embed_dim,
                        c.hidden, c.char_window, r, precision);
    parts.dict_cols.resize(fd * h);
    const double dict_r = wdict_scale / acc_scale;
    for (std::size_t k = 0; k < fd; ++k) {
        for (std::size_t j = 0; j < h; ++j) {
            parts.dict_cols[k * h + j] = static_cast<std::int32_t>(
                std::llround(dict_r * wdict[j * fd + k]));
        }
    }
    parts.b1_q.resize(h);
    for (std::size_t j = 0; j < h; ++j) {
        parts.b1_q[j] = static_cast<std::int32_t>(std::llround(b1[j] / acc_scale));
    }
    parts.b2_q = std::llround(b2 / (w2_scale * acc_scale));
    if (precision == TablePrecision::Int16) {
        // The int16-mode derivations: the same requant_i16 the table uses,
        // applied to the int32 quantities above; b2 gets the identical
        // rounded shift, in int64 (it is the output-sum offset, never an
        // int16 accumulator entry, so no saturation applies).
        parts.b1_q16.resize(h);
        for (std::size_t j = 0; j < h; ++j) {
            parts.b1_q16[j] = requant_i16(parts.b1_q[j]);
        }
        parts.dict_cols16.resize(fd * h);
        for (std::size_t i = 0; i < fd * h; ++i) {
            parts.dict_cols16[i] = requant_i16(parts.dict_cols[i]);
        }
        parts.b2_q16 = (parts.b2_q + (1 << (kAccShift - 1))) >> kAccShift;
    }
    parts.dict = DictMatcher(std::move(compiled));
    if (!parts.dict.valid()) {
        throw ParseError("dictionary FST is not usable");
    }
    return parts;
}

std::expected<Model, Error> Model::load_from_bytes(std::span<const std::byte> data,
                                                   TablePrecision precision) {
    try {
        bytes::BinaryReader in(data);
        const std::string header = in.read_line();
        if (!header.starts_with(kModelSignature)) {
            return std::unexpected(Error{ErrorCode::UnsupportedModelFormat,
                                         "not a SegmentLibMLP model"});
        }
        const std::string version = header.substr(kModelSignature.size());
        // Format 1 stored the dictionary as word lists and compiled them on
        // every load; format 2 stores the compiled FST. Both still load.
        if (version != "1" && version != "2") {
            return std::unexpected(Error{ErrorCode::UnsupportedModelFormat,
                                         "unsupported SegmentLibMLP version"});
        }
        return Model(parse(in, precision, version == "2" ? 2u : 1u));
    } catch (const ParseError&) {
        return std::unexpected(Error{ErrorCode::MalformedModel,
                                     "malformed SegmentLibMLP model"});
    }
}

std::expected<Model, Error> Model::load(const std::filesystem::path& path,
                                        TablePrecision precision) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::unexpected(Error{ErrorCode::IoError, "cannot open model file"});
    }
    std::vector<char> data((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
    if (in.bad()) {
        return std::unexpected(Error{ErrorCode::IoError, "cannot read model file"});
    }
    return load_from_bytes(
        std::span(reinterpret_cast<const std::byte*>(data.data()), data.size()),
        precision);
}

}  // namespace segmentlib::mlp
