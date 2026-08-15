#include "mlp/train/exporter.h"

#include <cassert>
#include <fstream>
#include <span>

#include "segmentlib/bytes/binary_writer.h"
#include "segmentlib/mlp/dictionary.h"
#include "segmentlib/unicode/egc.h"

namespace segmentlib::mlp::train {

namespace {

void write_i16_tensor(bytes::BinaryWriter& out,
                      std::span<const std::int16_t> tensor) {
    for (const std::int16_t v : tensor) {
        out.write<std::int16_t>(v);
    }
}

}  // namespace

std::vector<std::byte>
serialize_model(const QuantizedModel& quantized, const Vocab& vocab,
                std::span<const std::vector<std::string>> dictionaries) {
    const NetConfig& c = quantized.config;
    assert(c.vocab_size == vocab.size());
    assert(c.num_dicts == dictionaries.size());
    assert(quantized.embedding.size() ==
           static_cast<std::size_t>(c.vocab_size) * c.embed_dim);
    assert(quantized.w1.size() ==
           static_cast<std::size_t>(c.hidden) * c.input_dim());
    assert(quantized.wdict.size() ==
           static_cast<std::size_t>(c.hidden) * c.dict_features());
    assert(quantized.w2.size() == c.hidden);
    assert(quantized.b1.size() == c.hidden);

    bytes::BinaryWriter out;
    out.write_line("SegmentLibMLP 1");

    // Config (fields 1-4b). unicode_version records which Unicode data the
    // EGC splitter used at training time; the loader warns on mismatch.
    out.write<std::uint8_t>(c.window);
    out.write<std::uint16_t>(c.embed_dim);
    out.write<std::uint16_t>(c.hidden);
    out.write<std::uint8_t>(static_cast<std::uint8_t>(c.num_dicts));
    out.write<std::uint16_t>(unicode::kEgcUnicodeVersion);

    // Scales (fields 5-8b). wdict_scale exists only with dictionaries.
    out.write<double>(quantized.emb_scale);
    out.write<double>(quantized.w1_scale);
    if (c.num_dicts > 0) {
        out.write<double>(quantized.wdict_scale);
    }
    out.write<double>(quantized.w2_scale);
    out.write<double>(quantized.acc_scale);

    // Vocabulary (fields 9-10): V, then the ascending codepoints of rows 2..
    out.write<std::uint32_t>(vocab.size());
    for (const char32_t cp : vocab.codepoints()) {
        out.write<std::uint32_t>(cp);
    }

    // Tensors and biases (fields 11-16).
    write_i16_tensor(out, quantized.embedding);
    write_i16_tensor(out, quantized.w1);
    if (c.num_dicts > 0) {
        write_i16_tensor(out, quantized.wdict);
    }
    for (const double b : quantized.b1) {
        out.write<double>(b);
    }
    write_i16_tensor(out, quantized.w2);
    out.write<double>(quantized.b2);

    // Dictionaries (field 17): the compiled matcher, not the word lists it was
    // built from. The FST is about a fifth of their size and is what the
    // loader needs, so it is both the smaller and the cheaper thing to carry.
    if (c.num_dicts > 0) {
        write_compiled_dictionaries(out, compile_dictionaries(dictionaries));
    }
    return out.take();
}

std::expected<void, Error>
export_model(const std::filesystem::path& path, const QuantizedModel& quantized,
             const Vocab& vocab,
             std::span<const std::vector<std::string>> dictionaries) {
    const std::vector<std::byte> bytes =
        serialize_model(quantized, vocab, dictionaries);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return std::unexpected(Error{ErrorCode::IoError, "cannot open model file for writing"});
    }
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    if (!out) {
        return std::unexpected(Error{ErrorCode::IoError, "cannot write model file"});
    }
    return {};
}

}  // namespace segmentlib::mlp::train
