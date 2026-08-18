# Design Document

## 1. Overview

A C++ **segmentation-only** word-segmentation library using the same "pointwise prediction" approach as KyTea / Vaporetto: an independent binary classification (split / do not split) at each character boundary. Unlike lattice + Viterbi minimum-cost methods (e.g. MeCab), it needs no dictionary cost design or dynamic programming.

The public API is a single one (Section 6), backed internally by three swappable **backends**:

- **KyTea-compatible backend** (Section 3) — loads a KyTea-trained model as-is and performs inference with the same feature extraction and linear SVM classifier as KyTea. Inference only; no training support.
- **Custom MLP backend** (Section 4) — uses a custom-designed MLP (multi-layer perceptron) as the classifier instead of a linear SVM. Also has a self-implemented training engine.
- **EDLA backend** (Section 11) — the same network as the MLP backend, trained by the Error Diffusion Learning Algorithm instead of backpropagation. A study of whether a biologically motivated local learning rule can train this task, not a separate model design.

**The corpus format is always the KyTea corpus format** (Section 5). Both the KyTea-compatible and custom MLP backends take the same KyTea corpus format (full/partial annotation) as input.

## 2. Architecture: Backend Abstraction

To hide multiple backends behind the same API, `Segmenter` is a thin dispatcher unaware of backend implementation details.

The set of backends is a fixed, small, closed set ("KyTea-compatible / custom MLP / EDLA"), with no requirement for dynamic external plugins. So instead of an open extension mechanism via virtual functions (`virtual` + heap allocation), this uses **closed polymorphism via `std::variant` + `std::visit`**. This avoids vtable indirection and per-backend heap allocation, and any missing branch for an unsupported backend is caught at compile time.

```cpp
class KyteaBackend {
public:
    Expected<Segments, Error> tokenize(std::string_view text) const;
};
class MlpBackend {
public:
    Expected<Segments, Error> tokenize(std::string_view text) const;
};
class EdBackend {
public:
    Expected<Segments, Error> tokenize(std::string_view text) const;
};

using AnyBackend = std::variant<kytea::KyteaBackend, mlp::MlpBackend, ed::EdBackend>;

class Segmenter {
public:
    static Expected<Segmenter, Error> load(const std::filesystem::path& model_path);
    static Expected<Segmenter, Error> load_kytea(const std::filesystem::path& model_path);
    static Expected<Segmenter, Error> load_mlp(const std::filesystem::path& model_path);
    static Expected<Segmenter, Error> load_ed(const std::filesystem::path& model_path);

    Expected<Segments, Error> tokenize(std::string_view text) const {
        return std::visit([&](const auto& b) { return b.tokenize(text); }, backend_);
    }

    std::vector<Expected<Segments, Error>>
    tokenize_all(Span<const std::string_view> texts, unsigned threads = 0) const;

private:
    AnyBackend backend_;
};
```

**Model format auto-detection**: `Segmenter::load()` selects the backend automatically by looking at the file's leading signature (a `"SegmentLibMLP "` header line selects the MLP backend, Section 4.7, and `"SegmentLibED "` the EDLA one, Section 11; anything else falls through to the KyTea backend). The signatures differ in length, so `load()` reads the longest one it knows and compares each candidate over its own length; a file shorter than that simply matches nothing. `load_kytea(path)` / `load_mlp(path)` / `load_ed(path)` are also provided for explicitly specifying the backend; `load()` is a thin wrapper around them.

Each backend class only needs to satisfy the `tokenize` signature; the internal feature extraction, classifier, and model parser can be implemented completely independently. The only thing the backends are required to have in common is "returning the same `Segments` type."

The EDLA backend is a deliberate exception to that independence: it shares the MLP backend's model parser and scorer outright rather than reimplementing them (Section 11.2). The two differ only in how their weights were learned, and holding the inference path *identical* is what makes the comparison between them measure the learning rule instead of two separately-tuned implementations.

## 3. KyTea-Compatible Backend

- Loads a model produced by KyTea (`train-kytea`) as-is.
- Features are the same three kinds as KyTea: character n-grams, character-type n-grams, dictionary-derived word features.
- **Inference only** (no training engine).
- The model binary format is parsed directly from the KyTea model file (no conversion tool step).
- **Segmentation only** (no tag estimation).

### 3.1 Feature Extraction (KyTea-Compatible, Faithfully Reproduced)

The classifier (LIBLINEAR's linear SVM) itself is generic, but **the feature-string generation logic and character-type classification are KyTea's own implementation**, and must be reproduced byte-for-byte to preserve model compatibility.

**Character-type classification (`StringUtil::CharType`)**

Six types, determined by range checks on the Unicode codepoint of each UTF-8 character. The evaluation order matters (Romaji → Hiragana → Katakana → Digit → Kanji → Other).

| Type | Symbol | Unicode range (summary) |
|---|---|---|
| ROMAJI | `R` | `0x41-0x5A`, `0x61-0x7A` (half-width), `0xFF21-0xFF3A`, `0xFF41-0xFF5A` (full-width) |
| HIRAGANA | `H` | `0x3040-0x3096` |
| KATAKANA | `T` | `0x30A0-0x30FF` (excluding `0x30FB` middle dot), `0xFF66-0xFF9F` (half-width kana) |
| DIGIT | `D` | `0x30-0x39` (half-width), `0xFF10-0xFF19` (full-width) |
| KANJI | `K` | `0x3400-0x4DBF`, `0x4E00-0x9FFF`, `0xF900-0xFAFF`, `0x20000-0x2A6DF`, `0x2A700-0x2B73F`, `0x2B740-0x2B81F`, `0x2F800-0x2FA1F` |
| OTHER | `O` | everything else |

KyTea itself supports UTF-8/EUC/SJIS encodings; this library targets UTF-8 only.

KyTea's `findType` has a bug in its 4-byte UTF-8 (codepoint ≥ U+10000, CJK Extension B and beyond) codepoint arithmetic and misclassifies those kanji. This library classifies from the correct codepoint, so results can diverge from KyTea in this rare case (an intentional difference). They agree for ordinary Japanese text within the BMP.

**Input normalization (`normalize`)**

At inference, KyTea splits the input string into `surface` (original) and `norm` (normalized), and feature computation (character n-grams, character-type n-grams) is done entirely on `norm`. Normalization uses a fixed table (96 entries) that **folds half-width alphanumerics/symbols to full-width** (`a→ａ`, `0→０`, `(→（`, half-width kana punctuation `｢｣→「」`, etc.). Output word surfaces are cut from `surface` (the original byte string), but boundary-decision scores are computed from `norm`. This library ports the same fixed table, building the `norm`-equivalent id sequence via **UTF-8 decode → codepoint normalization → interning** (`CharTable::encode`).

**Feature string format**

At each boundary, three kinds of feature strings are generated within a window (default `charw=3`, similarly configurable for `typew`) and converted to IDs via the model's dictionary for the linear classifier.

| Feature type | Prefix format | Example |
|---|---|---|
| Character n-gram | `"X" + relative position` + the string itself | `X-2`, `X-1`, `X0`, `X1` |
| Character-type n-gram | `"T" + relative position` + character-type symbol sequence | `T-1`, `T0`, `T1` |
| Dictionary word feature | `"D" + dict index + (L\|I\|R) + match length` | `D0L1` (dict 0, left edge, length 1), `D1R3` |

`D` feature's `L`/`I`/`R` denotes the dictionary entry's positional relationship to the boundary (Left edge / Inside / Right edge).

**Feature-ID mapping is a model-embedded dictionary, not a hash**

KyTea embeds the training-time feature-string→ID dictionary in the model file. Inference does not implement its own hash function; it **reads the feature dictionary straight from the model file and looks up IDs through it**. Unknown feature strings (never seen at training time) do not exist in the model, so that feature is simply skipped.

### 3.2 Model File Format

**Header line**

```
KyTea <version> <T|B> <encoding>
```

Example: `KyTea 0.4.0 B utf8`. `version` is `"0.4.0"` for quantized builds. The format character is `T`=text, `B`=binary. This library targets only the binary `0.4.0` format. Backend auto-detection (Section 2) keys on the MLP signature rather than this line: a file that does not start with `"SegmentLibMLP "` is handed to this parser.

**Overall section order**

1. **Config**: `do_ws`, `do_tags`, `numTags`, `charWindow`, `charN`, `typeWindow`, `typeN`, `dictionaryN`, bias flag, `epsilon`, `solverType`, and the character map
2. **Word-segmentation model** (`wsModel_`): one `KyteaModel`
3. **Tag models**: `numTags` of them (this library skips these)
4. **Word dictionary**: `Dictionary<ModelTagEntry>` (this library retains only `char_length`/`in_dict`)
5. **Subword dictionary**: `Dictionary<ProbTagEntry>` (this library skips this)
6. **Language model (LM)**: `numTags` of them (this library skips these)

This library is segmentation-only. Of the six sections above, WS needs section 1 (`numTags`/`do_tags` only to know the **byte length** of section 3's tag models and the word-dictionary entries' tag info), section 2, and part of section 4 (`char_length`/`in_dict` only); sections 3, 5, 6, and the tag candidates/per-word tag models inside word dictionary entries **exist as byte sequences in the file but are skipped rather than retained** (`kytea/model.h`'s `skip_*` functions). The file format itself is KyTea's own fixed spec; this library's implementation only chooses what to retain versus skip.

**`KyteaModel` (one classifier) serialization**

- Class count (`int32_t`; 0 or below 2 means "no model," ending there)
- Solver type (a single-byte `char` enum; default is `L2R_L2LOSS_SVC_DUAL` = L2-regularized L2-loss linear SVM, dual)
- Each class's label (`int32_t` x class count)
- Bias flag (`bool`)
- `multiplier` (`double`; the scale factor for reverting quantized weights to real numbers)
- `FeatureLookup` (below)

**`FeatureLookup` = a precompiled feature→weight direct mapping for inference**

KyTea does not write out the training-time feature-string→ID dictionary as-is; it **separately builds and writes an inference-optimized `FeatureLookup` structure**. It has 7 members:

| Field | Type | Content |
|---|---|---|
| `charDict` | `Dictionary<FeatVec>` | character n-gram → per-class weight vector |
| `typeDict` | `Dictionary<FeatVec>` | character-type n-gram → per-class weight vector |
| `selfDict` | `Dictionary<FeatVec>` | dictionary-derived self-string feature → per-class weight vector |
| `dictVector` | `FeatVec` | additional fixed-length dictionary-related weights |
| `biases` | `FeatVec` | bias terms |
| `tagDictVector` / `tagUnkVector` | `FeatVec` | tag-estimation-related weights (unused by this library) |

`Dictionary<FeatVec>` is an **Aho-Corasick automaton** (an array of `DictionaryState`: `failure` links, `gotos` (character → next state, sorted for binary search), `output` (the feature indices resolved at that state)).

**`Dictionary<Entry>`'s binary layout** (a shared framework not just for `charDict`/`typeDict`/`selfDict` (`Entry=FeatVec`), but also the word dictionary (`Entry=ModelTagEntry`) and subword dictionary (`Entry=ProbTagEntry`)):

```
num dictionaries : unsigned char (1 byte; 0 means "no dictionary," no further fields written)
state count      : uint32_t
[per state] each state (DictionaryState):
    failure      : uint32_t              // failure-transition target state index
    goto count   : uint32_t
    [per goto]
        char      : KyteaChar (=unsigned short, 2 bytes)
        target    : uint32_t
    output count : uint32_t
    [per output]
        value     : uint32_t              // index of an entry resolved at this state
    isBranch     : bool (1 byte)
entry count      : uint32_t
[per entry] writeEntry<Entry>(...)     // differs by Entry type (below)
```

`gotos` is sorted by character (`KyteaChar`); the reader binary-searches via `DictionaryState::step()`.

`writeEntry<Entry>` differs by Entry type:

- `Entry = FeatVec` (for `charDict`/`typeDict`/`selfDict`): a `uint32_t` element count → `FeatVal` (default `int16_t`) x element count. **No feature-string identifier prepended at all** (the corresponding string is represented by the `DictionaryState`'s transition path itself).
- `Entry = ModelTagEntry` (for the word dictionary): `word` (`KyteaString`) → per-tag-level info (this library skips this) → `inDict` (`unsigned char`, a dictionary-membership bitmask) → a per-tag-level `KyteaModel` (this library skips this).
- `Entry = ProbTagEntry` (for the subword dictionary): this library skips this entirely.

`KyteaString`'s (a variable-length string) binary representation is a **NUL-terminated byte sequence**. There is no length prefix; the reader reads until `\0`.

**`KyteaChar`'s type is `unsigned short` (2 bytes, unsigned)**. KyTea does not keep characters as UTF-8 internally, but maps them to this 2-byte integer representation first. Since the Aho-Corasick automaton's transitions are built keyed on `KyteaChar`, one **must walk the automaton using the model's embedded character→2-byte-integer mapping, not raw UTF-8 bytes**.

**`KyteaChar` is not a fixed Unicode codepoint; it is an ID from a per-model "character intern table" numbered in training-time order of first appearance**.

- ID `0` is the reserved sentinel for the empty string. Actual characters start from **ID `1`**.
- The Config section's "character map" field's content is **a single UTF-8 string that is simply all the distinct characters the model saw at training time, concatenated in ID order (= order of first appearance)**.
- The model loader decodes this string as UTF-8 from the start, one character at a time, assigning ascending IDs starting at `1` in order of appearance, to build a `char(UTF-8) → KyteaChar` mapping. When inferring input text, the same mapping is used to convert UTF-8 characters to a `KyteaChar` sequence before walking the Aho-Corasick automaton.

**Handling of unknown characters (outside the training vocabulary)**: KyTea assigns a fresh, dynamically growing ID (`charTypes_.size()`, i.e. beyond the max trained ID) to unknown characters at inference. This library flattens all unknown characters to `kNoChar` (`0`). The two differ in ID assignment, but **segmentation output is completely equivalent** (all three WS features in `calculateWS` are automaton matches, keyed by trained IDs `1..K`; whether an unknown character's ID is `0` here or `K+1` in KyTea, it exists in no transition either way). Character-type n-gram contribution is unaffected since character type is **classified directly from the codepoint**, not the character ID. However, 4-byte UTF-8 CJK Extension B kanji (e.g. 𠮷 U+20BB7) are a separate axis — not unknown-character handling but the `findType` bug (this library correctly classifies as Kanji, KyTea misclassifies, causing type-n-gram divergence), an intentional difference.

**Inference is designed as "get the weight vector the instant an Aho-Corasick match occurs," not "string→ID→weight-array lookup"**. The model loader does not generate feature strings and hash/ID-convert them independently; it **parses this `FeatureLookup` (three Aho-Corasick automata + four weight vectors) straight from the file and holds it in the same structure**.

**Weight type (`FeatVal`) defaults to `int16_t` quantized**. Distributed trained models are normally built as a quantized build (`FeatVal = int16_t`). At load time, `int16_t` weights are multiplied by `multiplier` (`double`) to revert them to real numbers. This library supports only quantized models (`int16_t`); a non-quantized model is detected via the header's `"0.4.0NQ"` version string and rejected as an error.

### 3.3 Inference Algorithm (Word Segmentation, `Kytea::calculateWS` Reproduction)

What `KyteaBackend::tokenize` computes.

For a sentence of `N` characters, there are `N-1` boundaries (right after each character, except the last). For each boundary `i` (`0 <= i < N-1`), a score `score[i]` is accumulated in this order:

1. **Initial value**: `score[i] = biases[0]` (the first element of `FeatureLookup::biases_`, a constant common to all boundaries).
2. **Add character n-gram scores**: for every character n-gram that matches `charDict` (Aho-Corasick) against the normalized string (`sent.norm`), accumulate via `addNgramScores`'s logic. One n-gram match, centered at its occurrence position, **contributes to multiple boundaries within the window simultaneously** (`FeatVec` holds `window*2` scores as a single vector, distributed as `score[base_pos + j] += vec[j]` starting from `base_pos = pos - window`, `j` clipped to the valid range).
3. **Add character-type n-gram scores**: similarly accumulate via `typeDict` against the string converted to character-type symbols (`R`/`H`/`T`/`D`/`K`/`O`).
4. **Add dictionary-derived (D feature) scores**: `dict_->match(sent.norm)` gets Aho-Corasick matches against the word dictionary, accumulated via `addDictionaryScores`. Index computation (`len=score.size()`, `max=config.getDictionaryN()`, matched word's character length `wlen`, `lablen=min(wlen,max)-1`):
   - The matched word's **left edge** boundary (index `end-wlen`, only if `end>=wlen`) adds `dictVector[dict_index*dictLen + (end-wlen)*3*max + lablen*3 + 0]`
   - Each of the matched word's **interior** boundaries (`end-wlen+1 <= k < end`) adds `... + k*3*max + lablen*3 + 1`
   - The matched word's **right edge** boundary (index `end`, only if `end != len`) adds `... + end*3*max + lablen*3 + 2`
5. **Hard constraint (`-wsconst` equivalent) override**: if a character-type symbol specified in `config.getWsConstraint()` covers the case where two adjacent characters share the same character type, forcibly overrides that boundary's score to the "no boundary" side (the distributed jp model's `wsConstraint` is normally empty and has no effect on the default output; this library does not implement this).
6. **Final score**: `wsConfs[i] = score[i] * wsModel_->getMultiplier()`
7. **Boundary decision**: a boundary exists if `wsConfs[i] > 0`, otherwise not.

In other words, the inference logic this library implements is a simple linear sum: **"starting from the bias as the initial value, add the three kinds of Aho-Corasick match scores from charDict, typeDict, and dictVector, multiply by multiplier, and compare against 0."** The SVM's training part (LIBLINEAR) is not implemented.

## 4. Custom MLP Backend

A backend that uses a custom-designed MLP (multi-layer perceptron) as the classifier, in place of the linear SVM of Section 3. The segmentation approach itself follows the same **pointwise binary classification** as Section 3 (independently deciding "split / do not split" for each boundary candidate), and the public API satisfies the same `tokenize`. The aim is to have **the interactions among the features that KyTea/Vaporetto enumerate by hand (character n-grams, character-type n-grams, dictionary features) be automatically acquired via in-window embeddings and hidden layers**.

Training data uses the KyTea corpus format (Section 5) as-is. The training engine (forward pass, backward pass, optimization) is implemented in this library.

### 4.1 Design Philosophy, Contrasted with KyTea/Vaporetto

| | KyTea (Section 3) / Vaporetto (external comparison only, not a backend of this library) | This MLP backend (Section 4) |
|---|---|---|
| Classification approach | Pointwise binary classification | Same |
| Classifier | Linear SVM (linear sum of weights) | MLP (non-linear, multi-layer) |
| Features | Character n-grams, character-type n-grams, dictionary features, hand-designed and enumerated | In-window embeddings concatenated, interactions learned by the network |
| Character-type features | Explicit use of a heuristic 6-way classification (Section 3.1) | Not used |
| Atomic unit | Codepoint-level "character" | EGC (grapheme cluster) unit (Section 4.2) |
| Vocabulary / OOV | Model-embedded feature dictionary, unknown features skipped | Embedding is a codepoint vocabulary; EGCs are represented compositionally, with no OOV in principle (Section 4.3) |

**Not having explicit character-type features** is an intentional design decision. Room remains to introduce General Unicode Property (General Category, etc.) as an auxiliary input if generalization to unknown/low-frequency characters becomes a problem.

### 4.2 Atomic Unit, Boundary Candidates, Window Counting = EGC

The atomic unit of classification is not the codepoint but the **EGC (Extended Grapheme Cluster, UAX #29)**.

- **Boundary candidates exist only between EGCs**. Never split inside an EGC (e.g. `か` + combining dakuten `が` = U+304B U+3099, or an emoji ZWJ sequence `👨‍👩‍👦` = 6 codepoints).
- **The window is counted in "number of EGCs."** Measuring the window in codepoints would let a single `👨‍👩‍👦` consume the entire left/right window=5 budget, not even reaching the actual neighboring word. In EGC units, `👨‍👩‍👦` is treated as one token, leaving the remaining budget for the actual surrounding words.

For a sentence's EGC sequence `e[0..M-1]`, there are `M-1` boundary candidates (right after each EGC, except the last). Each boundary `i` (`0 <= i < M-1`) is classified independently.

### 4.3 EGC Representation = Compositional Embedding from Constituent Codepoints

Looking up each EGC directly as a vocabulary ID (a flat EGC-id embedding) is not used. Instead, **each EGC is decomposed into its constituent codepoint sequence, and codepoint embeddings are pooled to compose one EGC vector**.

```
EGC → decomposed into constituent codepoint sequence
  each codepoint → embedding (vocabulary = "codepoints seen at training time," small and bounded)
  → pooling → EGC vector (dimension d)
```

**A frequency threshold applies when building the vocabulary** (codepoints appearing fewer than 2 times are not put in the vocabulary and fall to UNK). Feeding low-frequency codepoints through UNK during training lets the UNK embedding learn "the average behavior of rare characters."

For Japanese and Chinese, 1 EGC ≈ 1 codepoint in almost all cases, so pooling mostly degenerates to identity, effectively behaving as a plain "codepoint embedding." Pooling becomes meaningfully active only for combining sequences, Thai, Myanmar, and emoji.

### 4.4 Network Configuration

A pointwise classifier: "concatenate each in-window EGC vector plus dictionary features, run through a 1-hidden-layer MLP, and make a binary decision." The configuration was determined by working backward from the speed requirement (Section 4.6's Layer 1 precomputation).

```
For boundary i:
  window = the EGC sequence e[i-w+1 … i+w] (w on each side, 2w total; edges are PAD tokens)
  each EGC → the compositional embedding of Section 4.3 (dimension d, pooling = mean)
  f_dict = dictionary-match binary features (below)
  h = ReLU( W1 · concat(2w × d) + W_dict · f_dict + b1 )    # 1 hidden layer
  y = w2 · h + b2                                            # scalar
  y > 0 means a boundary
```

**Fixed values**:

| Item | Value | Rationale |
|---|---|---|
| Window width `w` | 5 (each side, 10 EGCs total) | Under the Layer 1 precomputation scheme (Section 4.6), the cost of widening `w` is roughly one table-lookup addition per EGC, linear and nearly free |
| Embedding dimension `d` | 64 | ~10K codepoint vocabulary x 64 stays lightweight |
| Pooling | mean (fixed, not changeable) | mean is linear, so `W1_j·mean(e_c) = mean(W1_j·e_c)` holds, compatible with Layer 1 precomputation (Section 4.6) |
| Hidden layer | 1 layer, width H=256, ReLU | Depth has little effect for pointwise classification, and inference cost is dominated by the hidden layer onward, so it is kept shallow |
| Output | 1 unit, at inference **the sign of y** (sigmoid skipped) | `p>0.5 ⇔ y>0` |

**Dictionary-match binary features `f_dict`**: matches against a dictionary (word list) are found with a common-prefix search from every cluster start, over an FST built from the entries (Section 4.7 field 17). For matches straddling/adjacent to boundary i, 12 binary features are set: 3 positional relationships (L: match's left edge touches the boundary / I: match contains the boundary / R: match's right edge touches the boundary) x 4 match-length buckets (`min(EGC length, 4)`) (12 per dictionary channel if multiple dictionaries). If multiple dictionary words match the same (positional relationship x length bucket), the feature stays 1 (binary clamp). Works with no dictionary too (`f_dict` all zeros).

### 4.5 Training

- **Loss**: per-boundary binary cross-entropy (BCE). `sigmoid(y)` is used only for loss computation.
- **Supervision masking**: partial-annotation (Section 5.2) "unknown" positions are excluded from the loss. Intra-EGC positions are not even boundary candidates in the first place (Section 4.2), so they never appear in the loss.
- **Annotation/EGC collision handling**: a sentence where a boundary falls inside an EGC is skipped (with a warning) — the representation and label contradict each other.
- **Input normalization**: applies the same half-width→full-width fixed-table normalization as KyTea before EGC splitting and vocabulary lookup (sharing `CharTable`'s normalization table).
- **Post-training quantization**: weights and embeddings are quantized to int16 after training (PTQ).

### 4.6 Inference / C++ Implementation Approach: Layer 1 Precomputation (NNUE-Style)

Since Layer 1 is a pure linear transform, it decomposes per window position:

```
W1 · concat(v_1, …, v_2w) = Σ_j  W1_j · v_j        (W1_j is the 256×d slice for window position j)
```

So **(EGC, window position j) → W1_j·v(EGC) ∈ R^256 is precomputed into a table**. Mean pooling's linearity (Section 4.4) lets the compositional EGC embedding fold into this table too. Dictionary binary features likewise become a `W_dict` column-vector table lookup. Per-boundary inference computes:

```
acc = b1
acc += table[egc_j, j]   for 2w iterations (table lookup + 256-dim vector add)
acc += dict_col[k]       for each active dictionary feature (0 to a few)
h = ReLU(acc)
y = dot(w2, h) + b2      (one 256-dim dot product)
boundary ⇔ y > 0
```

**Table/accumulator numeric representation**: `Model::load`/`load_from_bytes` accepts `TablePrecision::{Int32,Int16}` (default `Int16`).

- **Int16**: `kAccShift=9` (2^22→2^13, 4x headroom); `requant_i16` = rounded right-shift + saturation is the sole conversion for all int16 quantities (table/dict_col/b1; b2 is an int64 with the same shift). The accumulator uses saturating add (`vqaddq_s16`).
- **Int32**: retained as a reference path. The bit-exact contract with the training side's `int16_decision` is carried by the Int32 path.

**Precomputed table**: built for frequent EGCs (≈ frequent codepoints). Rare EGCs fall back to a compositional path ("codepoint embedding → mean → multiply by W1_j").

**SIMD kernels**: `include/segmentlib/mlp/kernels.h` (header-only) has 6 add/relu/dot kernels for int32/int16, plus `add_widen_i16_i32` and the fused `fused_score_i16` that the Int16 path actually runs (Section 4.8). `kernels::scalar::*` is the always-compiled oracle; dispatch is compile-time (AArch64→NEON, x86 uses AVX2 only when `__AVX2__` is defined, otherwise scalar). Both NEON and AVX2 are bit-exactness-tested on real hardware (NEON = local ARM hardware, AVX2 = CI ubuntu-24.04 hardware + Windows MSVC hardware).

**thread_local scratch**: `Scratch{EncodedEgc,Workspace,scores}` behind the `detail::scratch()` accessor in `mlp/mlp_backend.h`, zero per-call allocation. It sits in a function rather than at namespace scope because the header is included in many translation units and the scratch must be one object program-wide.

Tokenization (UTF-8 → EGC splitting) follows UAX #29. Only the forward pass (inference) is implemented in the library proper; training is a separate component (`SEGMENTLIB_BUILD_TRAINING` opt-in), exchanged via the model file (Section 4.7).

### 4.7 Model File Format (Custom Design)

The serialization format is a custom design. **It must be readable directly with `BinaryReader`'s (`bytes/binary_reader.h`) primitives (little-endian fixed-width integers, NUL-terminated strings, `\n`-terminated header lines)**, sharing the same read infrastructure as the KyTea backend. The precomputed table (Section 4.6) is not included in the file; it is **built at load time**. The dictionary matcher is included, as a compiled FST, precisely so that it is not.

**Header line (ASCII, `\n`-terminated)**

```
SegmentLibMLP <version>\n
```

Example: `SegmentLibMLP 1\n`. This `"SegmentLibMLP "` signature is used for backend auto-detection (Section 2), mutually exclusive with KyTea's `"KyTea "` signature. An unknown version is an error.

**Binary body following the header line** (all little-endian):

| # | Field | Type | Content |
|---|---|---|---|
| **Config** | | | |
| 1 | `char_window` `w` | `uint8` | one-side window width (in EGCs). `w=5` (Section 4.4) |
| 2 | `embed_dim` `d` | `uint16` | codepoint embedding dimension. `64` (Section 4.4) |
| 3 | `hidden` `H` | `uint16` | hidden layer width. `256` (Section 4.4) |
| 4 | `num_dicts` | `uint8` | number of dictionaries (`0` allowed; if `0` the W_dict/dictionary sections are absent) |
| 4b | `unicode_version` | `uint16` | the Unicode version used for EGC splitting at training time (major x 100 + minor). The loader warns on mismatch |
| **Scales** (quantization scales) | | | |
| 5 | `emb_scale` | `double` | embedding int16→real coefficient |
| 6 | `w1_scale` | `double` | W1's int16→real coefficient |
| 7 | `wdict_scale` | `double` | W_dict's coefficient (only when `num_dicts>0`) |
| 8 | `w2_scale` | `double` | w2's coefficient |
| 8b | `acc_scale` | `double` | the accumulator's (Layer 1 activation) integer scale, chosen by calibrating the activation distribution on validation data after training |
| **Vocabulary** (codepoint vocabulary) | | | |
| 9 | `vocab_size` `V` | `uint32` | number of embedding rows; row 0=PAD, row 1=UNK (unknown codepoint) included |
| 10 | `codepoints` | `uint32 × (V-2)` | codepoints for rows 2..V-1, in **ascending** order. At inference an input codepoint is binary-searched in this array to get the row number |
| **Embedding** | | | |
| 11 | `embedding` | `int16 × (V·d)` | embedding table, row-major (row 0=PAD, row 1=UNK, row 2+ in `codepoints` order) |
| **Layer 1** | | | |
| 12 | `W1` | `int16 × (H · 2w · d)` | Layer 1 weights, row-major as `W1[h][j*d + c]` |
| 13 | `W_dict` | `int16 × (H · num_dicts · 12)` | dictionary binary-feature weights, only when `num_dicts>0` |
| 14 | `b1` | `double × H` | Layer 1 bias (unquantized) |
| **Layer 2** | | | |
| 15 | `w2` | `int16 × H` | output-layer weights |
| 16 | `b2` | `double` | output-layer bias (unquantized) |
| **Dictionaries** (only when `num_dicts>0`) | | | |
| 17a | `fst_size` | `uint32` | byte length of the compiled FST |
| 17b | `fst` | `uint8 × fst_size` | the cpp-fstlib byte code, keyed by each entry's normalized UTF-8 bytes, whose output is a channel-set id. Used as-is: the loader does not decompress or rebuild it. Carrying the compiled matcher rather than the word list is what makes a 570k-entry UniDic dictionary cost 2.1 MB and 135 ms to load, against 7.6 MB and 510 ms for the words alone |
| 17c | `set_count` | `uint32` | number of distinct channel sets |
| 17d | `set_offsets` | `uint32 × (set_count+1)` | CSR offsets into `set_dicts`, ascending, first `0`. The loader rejects anything else |
| 17e | `set_dicts` | `uint8 × set_offsets[set_count]` | dictionary channel per set member; each must be `< num_dicts` |


**Load-time processing**: (1) read vocabulary/embedding/weights, (2) build the per-position precomputed table (Section 4.6) and the expanded dictionary-feature column vectors, (3) point the dictionary matcher at the FST in the file, (4) quantize `b1`/`b2` to the accumulator's integer scale.

### 4.8 Evaluation Results (Current Measured Values)

Comparisons retrain KyTea and Vaporetto on an obtainable corpus (UD_Japanese-GSD, CC BY-SA 4.0) and compare them against the MLP backend trained on the **identical data with no dictionary**, since the distributed model's training corpus is unobtainable. All three were trained on `corpus/ud-gsd/train.kytea.txt` with no dictionary features. The KyTea backend is confirmed byte-identical to the real binary (`kytea -notags`).

**Accuracy** (`scripts/eval_segmentation.py`, boundary F-score; all three measured under one identical eval/gold):

| Test set (genre) | Boundaries | KyTea F1 | Vaporetto F1 | MLP F1 | MLP gap (vs KyTea) |
|---|---|---|---|---|---|
| GSD test (Wikipedia, in-domain) | 12,491 | 98.87% | 98.79% | 98.00% | −0.87pt |
| PUD (news/Wikipedia parallel, out-of-domain) | 27,788 | 99.24% | 99.17% | 98.57% | −0.67pt |
| GSD+PUD combined (2 genres) | 40,279 | 99.13% | 99.05% | 98.39% | −0.74pt |

No dictionary, default configuration (w=5, d=64, H=256, patience=15, seed=42). Zero quantization-induced decision flips across dev+train, 290,024 boundaries, on the real UD-GSD model. Seed-induced F1 variation is about ±0.05pt (measured over 5 seeds). **Without a dictionary the MLP trails the linear models (KyTea/Vaporetto) by ~0.7–0.9pt**; KyTea and Vaporetto are roughly tied.

Two levers were evaluated against this configuration and settled (both measured over 5 seeds). **Early-stopping patience was raised from 5 to 15 and adopted**: dev F1 keeps creeping up across plateaus a dozen epochs long, so patience 5 stopped on a plateau and cost 0.11pt of GSD test F1, while patience 15 costs only training wall-clock (about 22s → 60s) and nothing at inference. **A larger network (d=96, H=512) was rejected**: +0.23pt GSD / +0.10pt PUD for 45% of the inference speed (5.1 → 2.8 M chars/sec), a 627KB → 1511KB model and a 130ms → 467ms load. That is the same speed cost as the dictionary features rejected earlier (−48% for +0.45pt GSD), for half the accuracy and twelve times the model growth.

**The dictionary closes most of that gap, and the reference model this project builds and evaluates now uses one** (`--dict` itself stays opt-in in the trainer: the word list is a file the caller supplies). A dictionary extracted from the training corpus itself is worth +0.42pt GSD, but an external one is worth far more, because UD_Japanese's segmentation standard is UniDic's short unit and UniDic is that lexicon. All rows are 5-seed means against the same no-dictionary baseline; `scripts/fetch_unidic_dict.sh` reproduces the adopted one.

| Dictionary | Entries | GSD F1 | PUD F1 | Model | Load | Speed |
|---|---|---|---|---|---|---|
| none | 0 | 98.07% | 98.59% | 0.6 MB | 130 ms | 5.0 M chars/sec |
| training corpus, freq ≥ 2 | 8,726 | 98.49% | 98.90% | 0.7 MB | 130 ms | 2.9 |
| IPAdic | 325,869 | 98.81% | 99.09% | 1.7 MB | 137 ms | 2.4 |
| UniDic, 2–4 characters | 353,968 | 99.06% | 99.19% | 1.7 MB | 133 ms | 2.6 |
| **UniDic, 2+ characters (default)** | **565,302** | **99.12%** | **99.30%** | **2.1 MB** | **135 ms** | **2.4** |
| UniDic, all | 570,144 | 99.14% | 99.29% | 2.1 MB | 134 ms | 2.3 |

One filter matters: **single-character entries are dropped**. They are 1% of UniDic but match constantly, and the character window already sees those characters, so removing them costs nothing measurable and returns about 12% of the speed. Capping entry length at 4 was the other candidate, and is not worth it: the FST shares the prefixes and suffixes that would make long entries expensive to store, so the cap saves 0.4 MB for −0.06pt GSD / −0.11pt PUD. The same sharing is why load time barely moves with dictionary size (130–137 ms across every row above).

Two caveats. The ranking against the linear models does not change: given the same UniDic dictionary, KyTea reaches 99.43% GSD / 99.54% PUD, still ahead. And `--dict` is repeatable, so dictionaries can be stacked as separate feature channels, but that was measured and rejected for the default: UniDic + IPAdic + the corpus dictionary buys +0.10pt for 32% of the speed and a 11.4 MB model. Merging word lists into one channel instead does nothing at all, since 95% of the corpus dictionary is already in UniDic.

**Speed** (M1 Pro, `bench/bench_segment`, int16+NEON, best-of-8; no dictionary):

| Genre | MLP | Real KyTea | Ratio |
|---|---|---|---|
| GSD train (Wikipedia, 277K chars) | 5.11 M chars/sec | 1.40 M chars/sec | 3.65x |
| PUD test (news translation, 48K chars) | 5.60 M chars/sec | 1.58 M chars/sec | 3.54x |

Segmentation speed is genre-insensitive (score computation is pure integer arithmetic over the EGC window, independent of vocabulary or style).

### 4.9 Training-Side Design (Self-Implemented in C++)

The training engine is self-implemented in this library (no dependency on an external framework). Training is fp32; inference is int16 (Section 4.6); the two are exchanged via the model file (Section 4.7). **Inference (the deliverable of the library proper) is forward-pass only**, and the training component is separated at the build level (`SEGMENTLIB_BUILD_TRAINING` opt-in; the inference binary does not require BLAS/CUDA).

**Training pipeline**

```
1. Load corpus (Section 5, KyTea full/partial annotation)
2. Normalize (KyTea-compatible half-width→full-width, sharing CharTable, Section 4.5)
3. EGC split (UAX #29)
3b. Build vocabulary: tally codepoint frequencies, fall below-threshold ones to UNK
4. Generate examples: for each boundary i →
     - window [i-w+1 … i+w]'s each EGC → constituent codepoint row-id sequence (edges are PAD)
     - dictionary-match binary features f_dict (dictionary FST, Section 4.4)
     - label (boundary=1/non-boundary=0)
     - mask (partial-annotation unknown positions excluded from the loss, Section 4.5)
5. Mini-batch: embedding gather → mean pooling → concat(2w·d)
6. Forward pass → BCE (masked) → backward pass (embedding gradient is sparse)
7. Optimize: Adam
8. After convergence: PTQ int16 quantization → verify (decision-flip check, Section 4.5) → write out in the Section 4.7 format
```

**Backward-pass essentials**

- **Layer 1 and Layer 2**: standard dense-layer gradients. Matrix products are the main computation, delegated to `ComputeBackend`'s GEMM.
- **mean pooling → embedding**: since an EGC vector is the mean of its constituent codepoints, the gradient into the EGC vector is distributed to each constituent codepoint row at `1/(codepoint count)`. **The embedding table's gradient is sparse** (only rows that appeared in the batch). Adam's 1st/2nd moments are also updated only for rows that appeared.
- The `W_dict` gradient into the dictionary binary feature `f_dict` is likewise a sparse update, only for the columns of set features.

**Optimizer**: Adam.

**Quantization**: fixed on PTQ (Post-Training Quantization). QAT is not used.

**Compute backend abstraction (`ComputeBackend`)**

Confines matrix products, activations, elementwise ops, and gradients to this layer, swapped per platform. CPU implementation is written against a BLAS interface (`cblas_sgemm`, etc.); switching which BLAS is linked covers all OSes.

| Platform | First choice | CPU impl (BLAS) | GPU impl (optional) |
|---|---|---|---|
| macOS (Apple Silicon) | CPU/AMX | Accelerate | Metal/MPSGraph (usually unneeded) |
| Linux + NVIDIA | GPU | OpenBLAS/MKL | cuBLAS |
| Linux (no GPU) | CPU | OpenBLAS/BLIS | — |
| Windows + NVIDIA | GPU | OpenBLAS/MKL | cuBLAS |
| Windows (no GPU) | CPU | OpenBLAS | — |

Support policy: **macOS and Linux are first-class, Windows is best-effort**. BLAS is linked only for the training target (`segmentlib_train`).

**Inference-side portability (a separate axis from training)**: inference uses hand-written SIMD int16 NNUE style (Section 4.6), not BLAS, covering all OSes with **NEON (Apple/ARM) + AVX2 (x86 = common to Linux/Windows) + a scalar fallback**. The AVX2 path on x86 is identical between Linux and Windows.

## 5. Corpus Specification

Default delimiter characters:

| Purpose | Character | Default |
|---|---|---|
| Word boundary (full) / unknown boundary (partial) | `wordBound_` / `unkBound_` | half-width space `" "` |
| Tag boundary | `tagBound_` | `/` |
| Tag candidate separator | `elemBound_` | `&` |
| Escape | `escape_` | `\` |
| No boundary (partial) | `noBound_` | `-` |
| Boundary (partial) | `hasBound_` | `\|` |
| Skip (partial) | `skipBound_` | `?` |

Both backends' (Sections 3-4) training data is unified on this KyTea corpus format.

### 5.1 Full-Annotation Format

```
word1/tag0a&tag0b/tag1a word2/tag0 word3 ...
```

- Words are separated by a half-width space.
- Every occurrence of `/` advances the tag "level" by one.
- Multiple candidate tags within the same level can be joined with `&` (the leading candidate is used as the correct label at training time).
- Delimiter characters themselves (space, `/`, `&`, `\`) can be included in a word/tag by escaping them with `\`.
- Example:
  ```
  コーパス/ko:pasu の/no 文/buN で/de す/su 。/.
  ```
- This library's training pipeline (MLP backend) does not use tag info, only boundary info.

### 5.2 Partial-Annotation Format

```
ヴ-ェ-ネ-ツ-ィ-ア|は|イ-タ-リ-ア|に|あ り ま す|。
```

- Characters are laid out one at a time, with one of the following symbols placed between adjacent characters to represent boundary information:
  - `-` (noBound): not a boundary (within the same word). Used as a definite supervision signal.
  - `|` (hasBound): a boundary. Used as a definite supervision signal; also marks a word's end.
  - ` ` (unkBound) / `?` (skipBound): unknown. Both are treated as "no supervision signal" when reading.
- Tags can be attached at the end of each word, right before `|`, in the same `/tag0&tag1/tag2...` syntax as full annotation (this library's training uses only boundary info).
- Escape character, tag boundary, and tag-candidate separator are shared with full annotation (`\`, `/`, `&`).

## 6. C++ API

### 6.1 Basic Policy

- Rather than a mutable-object-reuse, side-effect-based API, the basic API is **functional: it takes input and returns the result as a value**.
- Input takes `std::string_view` (no ownership needed, avoiding unnecessary copies).
- Errors are represented with `Expected<T,E>` (`support/expected.h`), a minimal
  stand-in for `std::expected`, which is C++23 and out of reach for the C++17
  baseline inference targets.
- Offsets use **UTF-8 byte offsets**.
- Differences among backends (KyTea-compatible / custom MLP) are never exposed at this API layer. Callers don't need to be aware of the type of model file loaded by `Segmenter::load()`; they simply call the same `tokenize()`.

### 6.2 Types

This library is segmentation-only. `Segments` reduces to the minimal shape: a list of word-span pairs (start/end UTF-8 byte offsets) — no dedicated `Segment` struct, just a bare alias.

```cpp
using Segments = std::vector<std::pair<std::size_t, std::size_t>>;  // (start, end) pairs

enum class ErrorCode {
    InvalidUtf8,
    ModelNotLoaded,
    UnsupportedModelFormat,
    MalformedModel,
    MalformedCorpus,
    IoError,
};

struct Error {
    ErrorCode code;
    std::string_view message; // assumed to be a static string. No dynamic messages are held
};
```

### 6.3 Segmenter

```cpp
class Segmenter {
public:
    static Expected<Segmenter, Error> load(const std::filesystem::path& model_path);
    static Expected<Segmenter, Error> load_kytea(const std::filesystem::path& model_path);
    static Expected<Segmenter, Error> load_mlp(const std::filesystem::path& model_path);

    // Move-only: a Segmenter owns its model outright, so an implicit copy
    // would silently duplicate all of it (~400MB for the distributed KyTea
    // model). Share one with `const Segmenter&`.
    Segmenter(const Segmenter&) = delete;
    Segmenter& operator=(const Segmenter&) = delete;
    Segmenter(Segmenter&&) = default;
    Segmenter& operator=(Segmenter&&) = default;

    Expected<Segments, Error> tokenize(std::string_view text) const;

    std::vector<Expected<Segments, Error>> tokenize_all(
        Span<const std::string_view> texts, unsigned threads = 0) const;
};
```

- A single `tokenize()` covering segmentation only. `tokenize_all()` tokenizes many inputs in parallel (`threads == 0` uses hardware concurrency; Section 9: 44.98 M chars/sec at 8 threads on M1 Pro, 7.96x single-thread).
- An empty-string input is treated as "not an error, but an empty result" (`Segments{}`). `Error` is used only for actual abnormal cases such as invalid UTF-8 or an unloaded/unsupported model.
- A loaded `Segmenter` is immutable and its per-call scratch is `thread_local`, so any number of threads may call `tokenize()` / `tokenize_all()` concurrently on the same `const Segmenter&`. That, not copying, is how one is shared.

## 7. CLI Interface

The command name is `segmenter`. It uses a **subcommand structure** like `git`/`cargo`.

```
segmenter predict --model model.bin < input.txt > output.txt
segmenter train --backend mlp --corpus corpus.txt --model-out model.bin
```

### 7.1 `segmenter predict` (Inference)

As with KyTea / Vaporetto, this follows the **filter-style pattern of reading text from standard input and writing segmentation results to standard output**.

**Options**

| Option | Description |
|---|---|
| `--model <path>` | path to the model file (required). Auto-detects whether it is KyTea-compatible / custom MLP (Section 2) |
| `--threads <n>` | number of threads for parallel execution (`0` = `hardware_concurrency()`, default; capped at 1024) |

**Output format**

Follows KyTea's segmentation output format (a space-separated word list).

```
コーパス の 文 で す 。
```

**Escaping the surface word (`showEscapedString`)**: when a delimiter character (space, `/`, `&`, or the escape character `\` itself) is included in a word's surface form, it is escaped by prefixing it with `\` (example: input `Hello World` → `Hello \  World`, `2024/12/31` → `2024 \/ 12 \/ 31`). The implementation is consolidated in `append_full_line` (`include/segmentlib/output.h`), shared by the CLI and the golden tests.

**Input handling**: input is always fixed to UTF-8. Malformed UTF-8 bytes cause `CharTable::encode`/`Vocab::encode` to return `ErrorCode::InvalidUtf8`; the CLI flushes any output already produced for earlier lines and then aborts immediately (exit code 1, stderr message).

### 7.2 `segmenter train` (Training)

```
segmenter train --backend mlp \
  --corpus full1.txt --corpus full2.txt \
  --partial-corpus part1.txt \
  --dict dict.txt \
  --model-out model.bin \
  [--char-window 5] [--embed-dim 64] [--hidden 256] [--min-count 2] \
  [--epochs 100] [--batch-size 256] [--patience 15] [--lr 1e-3] [--seed 42]
```

`--backend mlp` and `--backend ed` are implemented (both require a `SEGMENTLIB_BUILD_TRAINING=ON` build; `train` is a stub in an OFF build). `--backend kytea` / `--backend vaporetto` return an explicit "not implemented" error (`train_command.cpp`).

Every option below means the same thing for both backends, `--lr` included: the EDLA trainer uses the same Adam optimizer over the same minibatches, and only the rule that produces the update differs (Section 11.3). That is what lets a run of each, with the flags held equal, attribute their difference to the learning rule.

| Option | Description |
|---|---|
| `--backend mlp\|ed` | the backend to train (required) |
| `--corpus <path>` | full-annotation corpus (Section 5.1). Repeatable |
| `--partial-corpus <path>` | partial-annotation corpus (Section 5.2). Repeatable |
| `--dict <path>` | dictionary file. Repeatable |
| `--dev-corpus <path>` | dev corpus (used for early stopping and quantization calibration) |
| `--model-out <path>` | trained-model output path (required) |
| `--char-window <int>` | EGC window width (default 5) |
| `--embed-dim <int>` | embedding dimension (default 64) |
| `--hidden <int>` | hidden-layer width (default 256) |
| `--min-count <int>` | codepoint vocabulary frequency threshold (default 2) |
| `--epochs <int>` | number of epochs (default 100) |
| `--batch-size <int>` | batch size (default 256) |
| `--patience <int>` | early-stopping patience (default 15) |
| `--lr <float>` | learning rate (default 1e-3) |
| `--optimizer <adam\|sgd>` | optimizer (default adam; sgd is the scale-preserving control of Section 11.6) |
| `--seed <int>` | random seed (default 42) |
| `--ed-embedding-update <hybrid\|pure>` | EDLA only: how the embedding is updated (default hybrid, Section 11.3) |

When a KyTea-compatible model needs training, the real `train-kytea` (Homebrew-distributed) is called directly as an external tool (this library's own evaluation pipeline, Section 4.8, is a working example).

## 8. Implementation Module Structure

### 8.1 Layer Structure (Lower→Upper, Upper Depends Only on Lower)

```
include/segmentlib/
├── bytes/
│   ├── binary_reader.h        # (L1) primitive binary-read cursor
│   └── binary_writer.h        # (L1) the writing counterpart (used by the training exporter)
├── unicode/
│   ├── utf8.h                 # (L1) UTF-8 decode/encode pure functions
│   ├── egc.h                  # (L1) UAX #29 EGC splitting
│   └── normalize.h            # (L1) KyTea-compatible half-width→full-width normalization
├── kytea/
│   ├── char_table.h           # (L2) character-type classification + character intern table
│   ├── automaton.h            # (L2) Aho-Corasick runtime representation (double-array)
│   ├── model.h                # (L3) Config/KyteaModel/FeatureLookup types + load()
│   ├── scorer.h                # (L4) calculateWS's score computation algorithm
│   └── kytea_backend.h        # (L5) backend interface implementation (tokenize)
├── mlp/
│   ├── vocab.h                 # (L2) codepoint vocabulary + EGC encoding
│   ├── dictionary.h            # (L2) dictionary binary-feature FST matcher (cpp-fstlib)
│   ├── kernels.h               # (L2) SIMD kernels (add/relu/dot, NEON/AVX2/scalar)
│   ├── precompute.h            # (L3) per-position precomputed table
│   ├── model.h                  # (L3) model data types + load()
│   ├── scorer.h                 # (L4) inference score computation
│   └── mlp_backend.h           # (L5) backend interface implementation (tokenize)
├── ed/                         # EDLA: the mlp network trained differently (Section 11)
│   ├── model.h                  # (L3) "SegmentLibED" signature; body parsed by mlp::Model
│   └── ed_backend.h            # (L5) backend interface implementation (scores via mlp::tokenize_with)
├── support/                    # C++17 stand-ins for later-standard facilities
│   ├── expected.h              # Expected<T,E> / Unexpected<E> (std::expected is C++23)
│   ├── span.h                  # Span<T> (std::span is C++20)
│   ├── endian.h                # byte order + byteswap (std::endian/byteswap are C++20/23)
│   └── attributes.h            # SEGMENTLIB_FLATTEN (inlining hint for the header-only hot path)
├── segmenter.h                 # (L6) public API (Section 6)
├── output.h                    # output formatting (append_full_line)
└── types.h                     # Segments/Error (Section 6.2, no dependencies)

src/                            # inference is header-only: only training and the CLI compile here
├── mlp/train/                  # training components (built only when SEGMENTLIB_BUILD_TRAINING is on)
│   ├── corpus.{h,cpp}           # KyTea corpus parser
│   ├── example.{h,cpp}          # training-example generation
│   ├── dataset.{h,cpp}          # mini-batching
│   ├── net.{h,cpp}              # forward/backward pass
│   ├── adam.{h,cpp}             # Adam optimizer
│   ├── trainer.{h,cpp}          # training loop
│   ├── quantize.{h,cpp}         # PTQ int16 quantization
│   ├── exporter.{h,cpp}         # model-file writing
│   ├── compute_backend.h        # BLAS abstraction
│   └── cpu_blas.cpp             # CPU (Accelerate/OpenBLAS) implementation
├── ed/train/                   # EDLA trainer: replaces net.cpp's backward pass, reuses the rest
│   ├── edla.{h,cpp}             # the local update rule, Dale's law, polarity
│   └── trainer.{h,cpp}          # training loop (mirrors mlp/train/trainer.cpp)
└── cli/
    ├── main.cpp                  # dispatches the `predict`/`train` subcommands
    ├── predict_command.cpp       # predict subcommand body
    └── train_command.cpp         # train subcommand body (--backend mlp and ed implemented)
```

### 8.2 Module Responsibilities

**L1: `bytes::BinaryReader`** — a thin cursor advancing over a `Span<const std::byte>`. Provides `read<T>()`, `read_cstring()`. Parse errors are thrown internally as an exception (a lightweight `ParseError`) and converted to `Expected` only at the boundary of `model.h`'s `load()`.

**L1: `unicode::utf8`** — pure functions decoding a single UTF-8 codepoint.

**L2: `kytea::CharTable`** — a value type holding the "character intern table" (ID `0` = sentinel, real characters start at `1` in order of first appearance). Provides `decode`/`encode` and character-type classification (`classify`).

**L2: `kytea::Automaton<Payload>`** — the runtime representation of `Dictionary<Entry>`. A value type holding `std::vector<State>` and `std::vector<Payload>` flat (canonical double-array). It only receives deserialized model-file data; it holds no logic to build an Aho-Corasick automaton (the file already contains a fully built automaton).

**L3: `kytea::Model`** — an immutable value type corresponding to `Config`/`KyteaModel`/`FeatureLookup`. `static auto load(std::filesystem::path) -> Expected<Model, Error>` is the sole construction path. It holds `charDict`/`typeDict`/`dictVector`/`biases`/`multiplier`/`word_dict` (only `char_length`/`in_dict`) — the minimum segmentation needs. Per-word tag information in the word dictionary, the subword dictionary, the language model, and the tag models are present in the file, so parsing skips past them to reach the correct subsequent seek position, but are not retained as values.

**L4: `kytea::scorer`** — implements the `calculateWS` algorithm as a plain function. Takes `Model`, `CharTable`, and input text, and returns a per-boundary score sequence as a pure function.

**L5: `kytea::KyteaBackend`** — a class satisfying the `tokenize` signature defined in Section 2. It holds a `Model` internally and is simply a thin adapter that calls the `scorer` functions.

**L6: `Segmenter`** — as designed in Section 2. `std::variant<KyteaBackend, MlpBackend>`.

**CLI layer** — `predict_command.cpp` is just a thin loop: "parse arguments → `Segmenter::load` → read stdin line by line, `tokenize` → write output." No business logic.

### 8.3 Repository Directory Structure

```
cpp-segmentlib/
├── CMakeLists.txt                # top level. C++23, just bundles subdirectories
├── include/segmentlib/           # public headers (per Section 8.1)
├── src/                          # implementation (per Section 8.1)
│   ├── CMakeLists.txt            # segmentlib (library) target definition
│   └── cli/
│       └── CMakeLists.txt        # segmenter (executable) target definition
├── tests/
│   ├── CMakeLists.txt
│   ├── unit/                     # per-module tests
│   ├── golden/                   # tests matched against real KyTea binary output
│   │   ├── golden_test.cpp
│   │   └── fixtures/
│   │       ├── input.txt
│   │       └── expected.txt      # `kytea -notags`'s output
│   └── consumer/                 # a downstream project; built by check_consumer.sh,
│                                 # not part of the main build
├── models/
│   ├── mlp/                      # committed MLP reference model + NOTICE (CC BY-SA 4.0)
│   └── kytea/                    # (gitignored) fetched KyTea model
├── corpus/                       # (gitignored) evaluation corpora
├── bench/                        # inference benchmarks (Section 9)
│   ├── setup.sh / run.sh / README.md
│   ├── bench_segment.cpp         # this library, in-process
│   ├── bench_kytea.cpp           # libkytea, in-process
│   └── {.vendor,corpus,results}/ # (gitignored)
├── scripts/
│   ├── fetch_kytea_model.sh
│   ├── fetch_ud_gsd_corpus.sh
│   ├── fetch_ud_pud_corpus.sh
│   ├── convert_ud_gsd_corpus.py
│   ├── eval_segmentation.py
│   ├── extract_dict.py
│   ├── check_consumer.sh
│   ├── gen_egc_table.py
│   └── strip_kytea_tags.py
├── docs/
│   └── design.ja.md / design.md
├── .github/workflows/ci.yml
└── .gitignore
```

**Design decisions**

- **The build system is CMake**.
- **One vendored dependency on the inference path**: `third_party/cpp-fstlib` (header-only, MIT), the FST behind the dictionary matcher (Section 4.4). `mlp/dictionary.h` holds the `fst::map` directly and includes it, so a consumer of the header-only inference path compiles it too and needs `third_party/` on their include path (the `segmentlib` CMake target carries it as a SYSTEM interface include). The include is written as `"cpp-fstlib/fstlib.h"` and the include path is `third_party/`, so shadowing it from a consuming project takes a matching directory name rather than just a file called `fstlib.h`. Everything else is the standard library. Only the training path (`SEGMENTLIB_BUILD_TRAINING`) links BLAS.
- **The test framework is `doctest`** (header-only, fetched via CMake's `FetchContent`).
- **`models/kytea/` and `corpus/` are not git-tracked**: added to `.gitignore`, with only download scripts like `scripts/fetch_*.sh` kept in the repository. The exception is `models/mlp/`, the trained MLP reference model (~2.1 MB, this project's own artifact): it is committed with its NOTICE so a fresh clone segments text without a download or a training run, and its output is pinned by a regression test. Retraining it is a release-time act (docs/RELEASING.md).
- **`golden/` tests use fixed data**: pairs of known input sentences and real KyTea execution results are committed as fixed data in `tests/golden/fixtures/`. Tests are skipped when the model is absent (CI-resilient).

### 8.4 Build/Toolchain Requirements

- **Inference needs only C++17.** It is header-only, and the later-standard vocabulary types it would otherwise want — `std::expected` (C++23), `std::span`, `std::endian`/`std::byteswap` (C++20) — are replaced by minimal equivalents under `include/segmentlib/support/`. This is what lets a C++17 project drop the headers in. `scripts/check_consumer.sh build-consumer 17` builds a consumer at C++17 with `-pedantic-errors`, which is the only build here that fails when a later-standard construct reaches a public header.
- **Training and the CLI still require C++23** (`SEGMENTLIB_BUILD_TRAINING`, `SEGMENTLIB_BUILD_CLI`), as does the test suite. Only the inference headers carry the C++17 promise.
- **macOS**: buildable with AppleClang (Xcode's default).
- **Linux**: GCC 14 or later.
- **Windows**: MSVC, best-effort.
- **Build steps**:
  ```
  cmake -S . -B build -G Ninja
  cmake --build build
  ctest --test-dir build --output-on-failure
  ```
  Always specify `-DCMAKE_BUILD_TYPE=Release` when benchmarking (a `Debug` build makes inference tens of times slower).
- **Using it from another project**: `add_subdirectory()` or `FetchContent`, then link the `segmentlib` target; its public include path comes with it. The developer targets — the test suite (which fetches doctest over the network), the benchmarks and the `segmenter` CLI — default to on for a standalone build and **off when segmentlib is not the top-level project**, so a consumer's `all` builds one static library. `CMAKE_BUILD_TYPE` is likewise only defaulted to `Release` when standalone, since under `FetchContent` the cache belongs to the consumer. There is no install/export set, so `find_package(segmentlib)` is not supported.
- **CI**: `.github/workflows/ci.yml`. 7 jobs: macOS arm64 (NEON), Linux x86_64 (AVX2/scalar, GCC 14 + OpenBLAS), Windows (MSVC, AVX2, best-effort), an ASan + UBSan job (the only one built with assertions on), a golden job that fetches the KyTea model (cached) and sets `SEGMENTLIB_REQUIRE_GOLDEN` so a missing model fails instead of skipping, and a consumer job (`scripts/check_consumer.sh`) that builds `tests/consumer/` against the working tree and asserts a consumer's build gets the library alone — no test/bench/CLI artifacts, no doctest fetch, no build type chosen for them. All jobs build with `-DSEGMENTLIB_WARNINGS_AS_ERRORS=ON`.

## 9. Benchmarks (Inference Speed)

`bench/`. Inference speed comparison against KyTea / Vaporetto (for the KyTea backend).

### 9.1 Design Principle

- **Same model**: all three tools run with the same weights.
- **The correctness gate comes first**: outputs are `diff`'d before timing. segmentlib is checked for byte agreement with KyTea; Vaporetto's discrepancy rate is reported.
- **Pure inference is measured in-process**: the model is loaded once, and only the tokenize loop is timed (load and I/O excluded). Uses dedicated harnesses (`bench_segment` = this library, `bench_kytea` = linked against libkytea).
- **Fixed conditions**: single-threaded, best-of-N after warmup. The metric is Unicode code points/second.
- **Corpus**: real works from Aozora Bunko (Soseki, Dazai, Akutagawa, Miyazawa; about 710,000 characters).

### 9.2 Latest Results (Apple M1 Pro, 710K Characters of Aozora Bunko, Best-of-5)

**Correctness gate**
- segmentlib vs. KyTea: **0 / 20,822 lines differ (100% agreement)**
- Vaporetto vs. KyTea: 92 / 20,822 lines differ (99.56% agreement)

**Pure inference speed (in-process, load/I-O excluded, single-threaded)**

| Tool | Load | M chars/sec | vs. KyTea |
|---|---|---|---|
| **segmentlib** | **412 ms** | **6.21** | **about 5.3x** |
| KyTea | 964 ms | 1.17 | 1.00x |
| Vaporetto | a few seconds (daachorse build) | 8.68 | about 7.4x |

**segmentlib's parallel throughput (`tokenize_all`, best-of-5)**

| Threads | M chars/sec | vs. single-thread | vs. KyTea |
|---|---|---|---|
| 1 | 5.65 | 1.00x | 4.8x |
| 2 | 12.91 | 2.28x | 11.0x |
| 4 | 25.91 | 4.59x | 22.1x |
| **8** | **44.98** | **7.96x** | **38.4x** |

**Findings**: segmentlib achieves byte-for-byte agreement with KyTea while inferring more than 5x faster single-threaded, with faster loading too. Vaporetto is still fastest single-threaded (though its output is not strictly identical, 0.44% mismatch), but segmentlib surpasses Vaporetto's single-thread speed via multithreading (about 5.2x at 8 threads).

## 10. Known Limitations / Unimplemented Features

- **Tag estimation (POS/reading)**: not performed. Segmentation only.
- **KyTea-compatible training engine**: not implemented. `segmenter train --backend kytea` returns an explicit error. Use the real `train-kytea` when a KyTea-compatible model is needed.
- **Vaporetto-compatible backend**: not built. Vaporetto is used only as an external benchmark comparison target.
- **`--encode`**: not implemented. Input is always fixed to UTF-8.
- **`--backend` (explicit override at predict time), `--scores` (per-boundary score output)**: not implemented (auto-detection has been sufficient).
- **Multi-candidate + confidence output** (KyTea's `-out conf`/`-tagmax` equivalent): not implemented.
- **Hard constraints on partial-annotation input** (`-wsconst` equivalent): not implemented. The distributed jp model's `wsConstraint` is normally empty and has no effect on the default output.
- **pmr-based API** (`std::pmr::memory_resource` injection): not implemented. To be considered if it actually becomes a bottleneck in benchmarks.
- **MLP backend's AVX2**: CI-verified on real hardware (GitHub Actions ubuntu-24.04 hardware + Windows MSVC hardware).
- **EDLA backend beyond one hidden layer**: not implemented. `PrecomputeTable` assumes the embedding→hidden map is a single linear hop, so a deeper variant needs a new inference kernel as well as a training change (Section 11.3).
- **A shipped EDLA model**: not provided. `models/` holds release-managed artifacts; the EDLA model is a research output reproduced locally with `just model-ed` (Section 11.5).

## 11. EDLA Backend

### 11.1 What It Is and Why It Is Here

The Error Diffusion Learning Algorithm (EDLA), proposed by Kaneko in 1999 and given a systematic evaluation in [Fujita, arXiv:2504.14814](https://arxiv.org/abs/2504.14814), trains a network without backpropagation. A single global error signal is computed at the output and broadcast unchanged to every layer below; each unit turns that broadcast into a weight change using only locally available quantities — its own pre-activation, the activity arriving at its synapses, and a fixed excitatory/inhibitory tag. No layer reads another layer's weights, and no chain of derivatives is propagated backwards.

This backend trains the Section 4 network with that rule instead of backpropagation. Two things made this task a fair place to try it:

- **The classifier is already binary.** EDLA natively supports one scalar output; the paper's multi-class experiments need K independent networks and about 4K times the parameters of an equivalent MLP. Pointwise boundary prediction is one output by construction, so that cost does not arise.
- **The network is already shallow.** The paper's central finding is that EDLA is close to backpropagation at one hidden layer and degrades sharply with depth. Section 4.4's network has exactly one hidden layer, so no architectural concession was needed to land in the regime where EDLA is expected to work.

That second point cuts both ways and is stated here rather than buried: **this benchmark sits in the most favourable regime the paper identifies for EDLA.** A small gap here is the expected result, not evidence that the rule generalizes to deeper networks — for which the same paper predicts, and measures, a widening gap.

### 11.2 Relationship to the MLP Backend

The EDLA backend shares the MLP backend's model parser, precompute table and scorer outright (`ed/model.h` wraps `mlp::Model`; `ed::EdBackend::tokenize` calls `mlp::tokenize_with`). It is the one place where Section 2's "backends share nothing but the `Segments` type" is deliberately set aside.

The reason is the experiment itself. The two backends are meant to differ in exactly one variable — the learning rule — so the inference path is held byte-identical rather than reimplemented. A duplicated scorer could drift, and any drift would show up as an accuracy difference indistinguishable from the thing being measured. A unit test pins the property directly: the same weights written behind both signatures segment identically (`ed_model_test.cpp`).

The training side shares just as much: corpus parsing, vocabulary and example generation, batching, the forward pass, Adam, PTQ quantization and the serializer are all the Section 4.9 code, unchanged. Only `Net::backward` is replaced.

### 11.3 The Learning Rule

Write `d = t − sigmoid(y)` for the global error signal of one example. This is the only quantity that leaves the output layer.

**Output layer** (`w2`, `b2`). One linear hop from the loss, so the exact gradient is already local — presynaptic activity times the broadcast signal — and there is nothing for EDLA to approximate. Identical to backpropagation.

**Hidden layer** (`W1`, `W_dict`, `b1`). Each unit `j` carries a fixed polarity `p_j`, `+1` for the first `H/2` units and `−1` for the rest, derived from `H` alone and never stored in the model file. The update is

```
Da_j = g'(a_j) · p_j · d
```

where backpropagation would have used `g'(a_j) · w2_j · d`. The substitution is exactly "discard the magnitude of the downstream weight, keep its sign", and it is what removes the backward read of `w2`. The paper's positive/negative error channels are the same expression in another form: splitting `d` into `d+ = max(d,0)` and `d− = max(−d,0)` and giving excitatory units `d+` and inhibitory units `d−` reconstructs `p_j·d`.

For that substitution to stand for the sign of the weight it replaces, `sign(w2_j)` must actually equal `p_j`. So Dale's law is imposed: `w2` is initialized on its polarity's side and clamped back after every optimizer step. Units clamped to exactly zero are counted and reported each epoch (`pinned N/H`), since that is capacity the polarity split has cost.

`sigmoid(y)` rather than the raw logit bounds the broadcast to `[-1,1]`. A single scalar reaches every layer at once with nothing downstream to attenuate it, so an unbounded one would let a single wide-margin example drive the whole network — the activation blow-up the paper reports for deeper EDLA networks.

**Embedding.** The paper's networks take fixed input features, so a *learned* input embedding is outside what EDLA specifies; this is an extension, and the two readings of it are a training flag rather than a silent decision:

- `--ed-embedding-update hybrid` (default): reuse `dX = dA · W1`, the same one-hop map backpropagation uses. The hop crosses no nonlinearity, so it chains no activation derivatives — what EDLA objects to — and it is what makes a row's update depend on which characters were actually in the window.
- `--ed-embedding-update pure`: diffuse the global signal into the embedding as its own layer, gated by a per-dimension polarity. Faithful to "no backward map anywhere", but the update direction is then identical for every character in the batch. Section 11.6 measures what that costs.

**Optimizer.** Adam by default, the same one the MLP backend uses, over the same minibatches with the same `--lr`; Adam's per-parameter step is computed only from that parameter's own history, so it does not violate the locality property. `--optimizer sgd` selects plain SGD instead — it exists because Adam turns out to *hide* the difference between the rules rather than merely not confound it, which Section 11.6 measures. `Gradients` is reused as the container: its values here are local update directions carrying the sign the optimizer's subtraction expects, not partial derivatives of the loss.

**Depth.** One hidden layer, fixed. Beyond the paper's stability findings, `PrecomputeTable` (Section 4.6) folds the embedding→hidden map into a per-(row, slot) table on the assumption that it is a single linear hop; a second hidden layer would need a new inference kernel, not just a training change.

### 11.4 Model Format

`"SegmentLibED 1\n"` followed by the Section 4.7 body, unchanged. An EDLA model records the same tensors — the network is the same — so only the header line distinguishes the two, and it exists so that a file states which rule produced it and the loaders refuse each other's models.

The polarity array is derived from `H` and is not stored; the trainer and any loader recompute it from the same function.

Note that the signatures differ in length (`"SegmentLibED "` is 13 bytes, `"SegmentLibMLP "` is 14), which is why `Segmenter::load` reads the longest known signature and compares each candidate over its own length (Section 2).

### 11.5 Training

```sh
segmenter train --backend ed \
    --corpus corpus/ud-gsd/train.kytea.txt \
    --dev-corpus corpus/ud-gsd/dev.kytea.txt \
    --model-out corpus/ud-gsd/ed.mod
```

Every Section 7.2 option carries over with the same meaning, `--lr` included. `just model-ed` runs the above on the reference corpus with the reference dictionary. The resulting model is not shipped: `models/` holds release-managed artifacts whose output the golden fixtures pin, so the EDLA model stays a local research artifact.

### 11.6 Evaluation

**Protocol.** Both backends were trained on `corpus/ud-gsd/train.kytea.txt` (UD_Japanese-GSD, release tag r2.18) with `corpus/ud-gsd/dev.kytea.txt` for early stopping, at the identical network size and hyperparameters (`w=5, d=64, H=256, min-count 2, batch 256, lr 1e-3, patience 15`) and no dictionary, over seeds 42-46. Accuracy is boundary F1 from `scripts/eval_segmentation.py` against the same gold files the other backends are measured on. Speeds are Apple M1 Pro; inference is the consumer-shaped build (`build-release`, no training, no tests), training is 10 fixed epochs with early stopping disabled so the two rules are compared per unit of work rather than per run-to-convergence.

**Accuracy** (5-seed mean, standard deviation in parentheses):

| Configuration | GSD test F1 | PUD test F1 | vs MLP (GSD) |
|---|---|---|---|
| MLP (backpropagation) | 98.00% (0.047) | 98.56% (0.029) | — |
| **EDLA, hybrid embedding** | **98.00%** (0.067) | **98.56%** (0.040) | **−0.00pt** |
| EDLA, pure embedding | 95.73% (0.083) | 96.67% (0.157) | −2.27pt |

**Cost** (no dictionary; the dictionary configuration is in parentheses where it differs):

| | MLP | EDLA |
|---|---|---|
| Model size | 642,348 B | 642,347 B |
| Load time | 129 ms | 129 ms |
| Inference, 1 thread | 5.72 M chars/sec | 5.79 M chars/sec |
| Training | 1.00 s/epoch (1.18 with UniDic) | 1.02 s/epoch (1.10 with UniDic) |

Model size, load time and inference speed are the same by construction: the network, the file format and the scoring code are shared, and the one-byte size difference is the signature string (`"SegmentLibED "` is a byte shorter than `"SegmentLibMLP "`). The measured inference figures differ by about 1%, which is run-to-run noise on this machine. Training costs the same because both rules issue the same GEMMs; EDLA's hidden-layer term is marginally cheaper (a sign flip where backpropagation multiplies by `w2`) but that is not where the time goes.

**On the headline number.** At one hidden layer on this task, EDLA is indistinguishable from backpropagation: the difference is smaller than the seed spread of either. That is a stronger result than the source paper's own shallow-network measurement (MNIST, one hidden layer: 97.5% EDLA against 98.2% backpropagation), so it deserves more scrutiny rather than less — and there is a specific reason to discount it.

**The Adam caveat.** Because Dale's law forces `sign(w2_j) = p_j`, EDLA's hidden-layer update and backpropagation's differ by exactly `|w2_j|`:

```
BP:    Da_j = g'(a_j) · w2_j · d  =  g'(a_j) · p_j · |w2_j| · d
EDLA:  Da_j = g'(a_j) · p_j · d
```

`|w2_j|` is constant across everything that feeds unit `j` — its whole row of `W1` and `W_dict`, and `b1[j]`. Adam divides each parameter's step by the running RMS of that parameter's own gradients, and that ratio is invariant to scaling a gradient by a constant. So for a `w2` that changed slowly, Adam would cancel the difference between the two rules almost exactly, and the layers below would receive *the same updates either way*.

The reported parity therefore says "EDLA with Adam matches backpropagation with Adam", which is weaker than "EDLA matches backpropagation". Adam was chosen to keep the optimizer from confounding the comparison (Section 11.3); it turns out to confound it in the opposite direction, by normalizing away most of what distinguishes the two rules in this architecture. What survives the cancellation is the embedding path — `dX = dA · W1` sums over `j`, so the per-unit scalings do not factor out — and the time-variation of `w2` itself. Consistent with the rules not being fully equivalent, models trained from the same seed by the two rules disagree on 532 of 543 GSD test lines: they reach equal-scoring but genuinely different solutions.

**The plain-SGD control (measured).** The control the caveat calls for was run: `--optimizer sgd` (no momentum, no decay — nothing else smoothing the comparison), learning rate swept per backend on dev (seed 42), then 5 seeds at each backend's own best.

The sweep itself is the first place the rules separate. Backpropagation keeps improving up to lr = 5.0 (dev 0.9815) and only degrades at 10. EDLA peaks at lr = 1.0 (dev 0.9791), decays past 1.5, and collapses outright at 3.0 (dev 0.7502) — a roughly five-times-narrower stable range, which is what losing both the `|w2_j|` scaling and Adam's normalization predicts, and matches the paper's observation that EDLA needs smaller learning rates under ReLU.

At each backend's own best rate (5 seeds):

| Configuration | GSD test F1 | PUD test F1 | vs BP (GSD) |
|---|---|---|---|
| MLP + SGD, lr 5.0 | 98.27% (0.111) | 98.75% (0.063) | — |
| EDLA + SGD, lr 1.0 | 98.11% (0.036) | 98.67% (0.023) | −0.16pt |

So once the optimizer stops normalizing gradient scale, EDLA trails backpropagation by 0.09–0.16pt — small, consistent across seeds (the gap exceeds either configuration's seed spread), and in the direction and rough size the shallow-network literature predicts (smaller than the paper's 0.7pt MNIST gap at the same depth). This, not the Adam tie, is the honest measure of the rule at one hidden layer: **the credit-assignment rule costs about 0.1–0.2pt here, and Adam had been hiding it.** Two secondary observations: EDLA's seed variance is about a third of backpropagation's, and it early-stops sooner (20–30 epochs against 36–76) — at its narrower best rate it reaches its plateau faster and lower.

Incidentally, both backends score higher under tuned SGD than under the Adam defaults (98.27/98.11 against 98.00 for both) — an observation about this task's optimizer landscape rather than about the rules, recorded here because Section 4.8's reference numbers are Adam-based and this suggests headroom worth a properly-protocoled sweep of its own.

**On the pure-diffusion ablation.** Updating the embedding by diffusion alone costs 2.27pt on GSD and 1.89pt on PUD — the one large effect measured here, and the reason `hybrid` is the default. The cause is structural rather than a tuning failure: under that rule `dv_c` depends only on the batch's error sign and the dimension's polarity, never on which characters were in the window, so every row present in a batch moves the same direction and the table cannot learn to tell characters apart. It is worth stating that this is the part of the design the paper does not specify (its networks take fixed inputs), so the failure is of an extension made here, not of EDLA as published.

**Where this sits relative to the paper.** The result is consistent with Fujita's finding that EDLA is close to backpropagation in shallow networks, and says nothing about the depth regime where the same paper measures the gap widening (4 hidden layers on CIFAR-10: 35.5% against 55.2%). This backend is one hidden layer deep and cannot speak to that.
