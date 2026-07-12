# MLP Backend Module Design

Defines the file layout, class boundaries, and dependencies for the MLP backend finalized in Section 5 of `design.ja.md`. It follows the existing KyTea backend conventions (immutable `Model` + stateless free-function `Scorer` + buffer-reusing encoder) as-is.

## 0. Overall Policy

- **Physically separate inference (the library's shipping artifact) from training (a development-time tool)**. The inference binary must not link BLAS/CUDA/Metal at all (`design.ja.md` Section 5.9).
  - Inference: `include/segmentlib/mlp/` + `src/mlp/` → goes into the existing `segmentlib` target.
  - Training: `src/mlp/train/` (no public headers) → separate target `segmentlib_train`. Only this target links BLAS/CUDA/Metal.
- Namespaces: inference uses `segmentlib::mlp`, training uses `segmentlib::mlp::train`.
- Establish 3 new **shared foundations**, reused by both MLP and a future Vaporetto backend (Section 1 below).

## 1. Shared Foundations (newly established, not MLP-specific)

| Module | Namespace | Responsibility | Motivation |
|---|---|---|---|
| `unicode/egc.h` | `segmentlib::unicode` | UAX #29 grapheme cluster segmentation. UTF-8 → sequence of EGC byte spans + constituent code points of each EGC. Also holds the grapheme-break property table (generated data) | The atomic unit of the MLP (Section 5.2). Depended on by training preprocessing, inference, and dictionary loading, all of them |
| `unicode/normalize.h` | `segmentlib::unicode` | Fixed-table normalization from half-width to full-width (promotion of the current `kytea::normalize`) | Under approach (a) (Section 5.5), MLP also uses the same normalization. Shared out to avoid an MLP → kytea namespace dependency |
| `text/aho_corasick.h` | `segmentlib::text` | **Runtime-built** Aho-Corasick (builder + matcher). Key type is a template parameter | Used for MLP's dictionary matching (at the EGC level). Vaporetto also needs the same thing, "build an AC at runtime from a flat list" (Section 4.5) |
| `bytes/binary_writer.h` | `segmentlib::bytes` | Counterpart of `BinaryReader`. Writes LE fixed-width integers, NUL-terminated strings, and raw byte sequences | Used for writing out the model in the Section 5.7 format (used by the training-side exporter) |

**Note (impact on existing code)**:
- The promotion to `unicode/normalize.h` relocates the `normalize` implementation inside `kytea/char_table.cpp`, with the `kytea` side becoming a small refactor that just re-exports it. Existing KyTea behavior is unchanged.
- The existing `kytea/automaton.h` is a "read-only AC read from a model file," whereas `text/aho_corasick.h` is "an AC built from a word list." Since their uses differ, they are not unified and remain separate (the KyTea read-only version is kept as-is).

## 2. Inference Module (`segmentlib::mlp`)

Corresponds to KyTea's three-way split of `Model` (holds state) / `scorer` (computes) / `CharTable`+`EncodedText` (encodes).

```
include/segmentlib/mlp/
  vocab.h        Vocab, EncodedEgc      … equivalent to CharTable + EncodedText
  precompute.h   PrecomputeTable        … NNUE-style precomputation table (Section 5.6)
  dictionary.h   DictMatcher            … produces dictionary binary features via EGC-level AC (Section 5.4)
  model.h        Model                  … equivalent to kytea::Model (immutable, load, holds derived structures)
  scorer.h       score_boundaries(_into)… equivalent to kytea::scorer (free function)
  mlp_backend.h  MlpBackend             … equivalent to kytea::KyteaBackend
```

### 2.1 `vocab.h` — `Vocab` / `EncodedEgc`

Corresponds to `CharTable`. Handles the code point vocabulary (Section 5.3) and encoding.

- **`Vocab`**: code point → embedding row ID. An ascending array of code points + binary search (Section 5.7, field 10). Row 0 = PAD, row 1 = UNK. Unregistered code points become UNK. `row_of(char32_t) -> uint32_t`.
- **`EncodedEgc`** (equivalent to `EncodedText`, buffer-reusable): the result of normalizing then EGC-splitting the input.
  - `egc_count`: number of EGCs, M.
  - Row IDs of the constituent code points of each EGC (variable length → CSR-style `rows` + `egc_starts`). Usually 1 per EGC for Japanese/Chinese.
  - `offsets`: size M+1, the **original text** byte span of each EGC (for word extraction).
  - (When hybrid mode is adopted) precomputation keys for frequent EGCs.
- **`encode` / `encode_into`**: `std::expected<…, Error>`. Normalize (`unicode::normalize`) → EGC-split (`unicode::egc`) → convert to row IDs. Returns `InvalidUtf8`.

### 2.2 `precompute.h` — `PrecomputeTable`

`table[egc][j] = W1_j · v(egc)` from Section 5.6. Built at load time from `(embedding, W1)` (held by Model).

- Holds int32 `[2w][H]` blocks for frequent EGCs (numeric representation from Section 5.6: int32 table + int32 accumulator).
- `add_into(acc, egc, j)`: adds the block for position j into the 256-dimensional accumulator.
- **Fallback path**: for non-frequent EGCs, take the mean of the constituent code point embeddings and multiply by `W1_j` to synthesize the value (referencing `embedding` and `W1`). This branches internally behind the same `add_into` interface.
- The threshold/capacity for "frequent" is decided by Config/threshold (roughly ~50MB, Section 5.6).

### 2.3 `dictionary.h` — `DictMatcher`

The dictionary binary features from Section 5.4. Uses `text::aho_corasick` at the EGC level.

- At load time: word lists (Section 5.7 field 17, **already normalized**) are EGC-split, EGCs are interned (as EgcId within the dictionary), and the AC is built.
- `features_into(EncodedEgc, out)`: for each boundary, sets and returns **binary** features across L/I/R × 4 length buckets × number of dictionaries (multiple matches are clamped, Section 5.4). The scorer feeds these into the `W_dict` path.
- Without a dictionary (`num_dicts==0`), returns empty.

### 2.4 `model.h` — `Model`

Corresponds to `kytea::Model`. Immutable, `load`/`load_from_bytes`, parses into `Parts`, const accessors. Holds **the raw parameters from the file plus the derived structures built at load time**.

```cpp
struct Config {                       // Section 5.7 fields 1-4b
    std::uint8_t char_window = 0;     // w
    std::uint16_t embed_dim = 0;      // d
    std::uint16_t hidden = 0;         // H
    std::uint8_t num_dicts = 0;
    std::uint16_t unicode_version = 0;
};

class Model {
public:
    static std::expected<Model, Error> load(const std::filesystem::path&);
    static std::expected<Model, Error> load_from_bytes(std::span<const std::byte>);

    const Config& config() const noexcept;
    const Vocab& vocab() const noexcept;
    const PrecomputeTable& precompute() const noexcept;  // derived (built at load time)
    const DictMatcher& dict() const noexcept;            // derived (built at load time)
    // Quantized, loaded layer parameters (used for fallback synthesis and the second layer)
    std::span<const std::int16_t> embedding() const noexcept;  // V×d
    std::span<const std::int16_t> w1() const noexcept;         // H×2w×d
    std::span<const std::int16_t> w2() const noexcept;         // H
    // b1/b2 are already converted to accumulator integer scale at load time (Section 5.7 fields 14/16)
    std::span<const std::int32_t> b1_q() const noexcept;
    std::int32_t b2_q() const noexcept;

private:
    struct Parts { Config config; Vocab vocab; PrecomputeTable precompute;
                   DictMatcher dict; /* embedding,w1,w2,b1_q,b2_q,scales */ };
    static Parts parse(bytes::BinaryReader&);   // the header line is consumed by the caller
    Parts parts_;
};
```

- Auto-detection (the Segmenter in Section 2): the leading `"SegmentLibMLP "` header line (Section 5.7).
- If `unicode_version` does not match the EGC splitter of the running environment, a warning is issued at load time (`design.ja.md` Section 5.7).

### 2.5 `scorer.h` — `score_boundaries` / `score_boundaries_into`

The same shape as `kytea::scorer`. Returns a score (int32, boundary when `y>0`) for each EGC boundary.

```cpp
[[nodiscard]] std::vector<std::int32_t>
score_boundaries(const Model&, const EncodedEgc&);

void score_boundaries_into(const Model&, const EncodedEgc&,
                           DictFeatures& scratch,           // reusable buffer for dictionary features
                           std::vector<std::int32_t>& out); // number of boundaries M-1
```

Computation for one boundary (Section 5.6): `acc=b1_q` → `precompute.add_into` called 2w times → add the `W_dict` columns for the active dictionary features → ReLU → dot product with `w2` + `b2_q` → sign. The SIMD kernel (NEON/AVX2/scalar) is contained within this.

### 2.6 `mlp_backend.h` — `MlpBackend`

The same signature as `kytea::KyteaBackend`. Since MLP has no tag estimation, `tokenize` merely converts boundaries → `Segments` (with tags empty).

```cpp
class MlpBackend {
public:
    explicit MlpBackend(Model model) noexcept;
    std::expected<Segments, Error>   tokenize(std::string_view) const;
    std::expected<Boundaries, Error> tokenize_boundaries(std::string_view) const;
private:
    Model model_;
    // Hot-path scratch is allocated per call or thread_local (be careful with tokenize_all parallelism)
};
```

### 2.7 Integration into `segmenter.h` (edit to existing code)

- Add `mlp::MlpBackend` to `AnyBackend`: `std::variant<kytea::KyteaBackend, mlp::MlpBackend>`.
- Add the `"SegmentLibMLP "` signature to `Segmenter::load`'s auto-detection.
- Add `load_mlp(path)`.
- The `std::visit` dispatch stays as-is (only the number of types increases; missing branches are caught at compile time).

## 3. Training Module (`segmentlib::mlp::train`, `src/mlp/train/`)

Not included in the inference public headers. Built as a separate target `segmentlib_train`, invoked from the CLI's `train` subcommand.

```
src/mlp/train/
  corpus.{h,cpp}          KyTea corpus loading (full/partial, Section 6) → annotated sentences
  example.{h,cpp}         sentence → training example (EGC split, normalize, vocab construction, collision skip, Section 5.5)
  dataset.{h,cpp}         minibatch assembly, embedding gather, mean pooling
  compute_backend.h       ComputeBackend abstraction (GEMM/activation/elementwise ops/gradients)
    cpu_blas.cpp            Accelerate / OpenBLAS / MKL (link-time switch)
    cuda.cpp                cuBLAS (Linux/Windows, optional)
    metal.cpp               MPSGraph (macOS, optional)
  net.{h,cpp}             fp32 forward/backward propagation (delegated to ComputeBackend)
  adam.{h,cpp}             Adam (dense layers + sparse update for embeddings, Section 5.9)
  trainer.{h,cpp}          training loop, convergence check, validation
  quantize.{h,cpp}         PTQ int16 quantization + scale synthesis + decision-flip verification (Section 5.5)
  exporter.{h,cpp}         writes out the Section 5.7 format (uses binary_writer)
```

### 3.1 `compute_backend.h` — `ComputeBackend`

An abstraction that encapsulates only matrix multiplication, activation, elementwise operations, and gradients (Section 5.9). The implementation is swapped per platform.

```cpp
class ComputeBackend {  // pure virtual, or CRTP/variant (one implementation is chosen at startup)
public:
    virtual void gemm(...) = 0;          // C = alpha*A*B + beta*C
    virtual void relu(...) = 0;
    virtual void relu_backward(...) = 0;
    virtual void axpy(...) = 0;
    // embedding gather/scatter (sparse)
    virtual ~ComputeBackend() = default;
};
```

- Training is fp32. CPU (BLAS)/CUDA/Metal are chosen here. **Unrelated to inference** (inference uses hand-written int16 SIMD, Section 5.6).
- Since only one implementation is chosen at startup, `virtual` is sufficient (indirect call cost is not an issue since this is not the inference hot path).

### 3.2 Responsibilities of `example.h` (the crux of preprocessing)

Takes on all of the following from `design.ja.md` in one place:
- Normalization (approach a) → EGC split (`unicode::egc`)
- Vocabulary construction: code point frequency tallying, below threshold → UNK (Section 5.3, guarantees training of the UNK row)
- Example generation: window row-ID sequence (PAD at edges) + dictionary features + label + mask
- **Collision handling**: sentences with a boundary inside an EGC are warned about and skipped, with statistics recorded (Sections 5.2/5.5)

## 4. Dependency Graph

```
                    ┌─────────────── Inference (segmentlib target) ───────────────┐
segmenter.h ──▶ mlp/mlp_backend ──▶ mlp/scorer ──▶ mlp/model ──▶ mlp/vocab
                                        │              │           └─▶ unicode/{egc,normalize,utf8}
                                        │              ├─▶ mlp/precompute
                                        │              └─▶ mlp/dictionary ─▶ text/aho_corasick
                                        └─▶ (SIMD kernel: NEON/AVX2/scalar)
                                                       model.load ─▶ bytes/binary_reader
                    └───────────────────────────────────────────────────────────┘

                    ┌───────── Training (segmentlib_train target, BLAS/CUDA/Metal) ─────────┐
cli train ──▶ train/trainer ──▶ train/{net, adam, dataset} ──▶ train/compute_backend
                  │                    └─▶ train/example ──▶ unicode/{egc,normalize}, text/aho_corasick
                  └─▶ train/quantize ──▶ train/exporter ──▶ bytes/binary_writer
                                           (corpus ──▶ Section 6 format)
                    └──────────────────────────────────────────────────────────────────────┘
```

- Inference depends only on `unicode/*`, `text/aho_corasick`, and `bytes/binary_reader`, and does not include BLAS/CUDA.
- The sole point of contact between training and inference is the **Section 5.7 model file** (written by train/exporter, read by mlp/model).
- The shared foundations (`unicode/*`, `text/aho_corasick`, `bytes/*`) are used by both sides.

## 5. CMake Targets

| Target | Contents | Dependencies |
|---|---|---|
| `segmentlib` (existing) | Inference. kytea + mlp + shared foundations | zstd (for Vaporetto, existing). **No BLAS/CUDA** |
| `segmentlib_train` (new, optional) | Training. `src/mlp/train/*` | Platform BLAS (Accelerate/OpenBLAS), optionally CUDA/Metal |
| CLI (existing `src/cli`) | `predict` links against `segmentlib`, `train` links against `segmentlib_train` | |

`segmentlib_train` is opted into via `option(SEGMENTLIB_BUILD_TRAINING ...)`. The BLAS implementation is selected via `SEGMENTLIB_BLAS=Accelerate|OpenBLAS|MKL`, and the GPU via `SEGMENTLIB_GPU=none|cuda|metal`.

## 6. Proposed Implementation Order

1. `unicode/egc.h` (+ break property table generation) — the starting point for everything. Unit-tested (verified against UAX #29's GraphemeBreakTest.txt)
2. Promotion of `unicode/normalize.h` + `bytes/binary_writer.h` + `text/aho_corasick.h` (shared foundations)
3. `mlp/vocab.h` (Vocab/EncodedEgc)
4. Minimal training path: `train/{corpus, example, dataset, compute_backend(cpu_blas), net, adam, trainer}` → first confirm convergence on a small corpus
5. `train/{quantize, exporter}` → Section 5.7 model export
6. `mlp/{precompute, dictionary, model, scorer, mlp_backend}` → inference. Compare accuracy/speed against KyTea on the same text (Section 5.8)
7. Integration into `segmenter.h`, CLI wiring
8. Optimization: int16 re-quantization table (Section 5.6), SIMD kernels, GPU backend (if needed)
```
