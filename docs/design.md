# Design Document

## 1. Overview

A C++ word-segmentation library adopting the same "pointwise prediction" approach as KyTea / Vaporetto.
For each character boundary, an independent binary classification (split / do not split) is performed by a classifier.
Unlike the minimum-cost method using a lattice + Viterbi (as in MeCab, etc.), this does not require dictionary cost design or dynamic programming.

**Basic architectural policy**: Keep the public API single (see Section 4), and have it internally hold multiple swappable **backends**.

- **Backend 1: KyTea-compatible** (Section 3) — Loads KyTea's trained models as-is and performs inference with the same feature extraction and linear SVM classifier as KyTea. The first goal.
- **Backend 2: Custom MLP model** (Section 4, the ultimate goal) — A backend that uses a custom-designed MLP (multi-layer perceptron) as the classifier instead of a linear SVM. Positioned as room for future accuracy improvement.

**The corpus format always uses KyTea's corpus format** (Section 5). Both the KyTea-compatible backend and the custom MLP backend assume the same KyTea corpus format (full/partial annotation) as input. Unifying the training data format is also intended to allow a fair comparison of accuracy across the two backends.

- The segmentation accuracy target is parity with KyTea (for the KyTea-compatible backend, byte-for-byte agreement with KyTea's actual output is verified).
- Speedups aim, as with Vaporetto (used here only as an external point of comparison, not as a backend of this library), at pattern matching for feature extraction (Aho-Corasick / Double-Array) and the memory layout of weight arrays (details are covered in each backend's section, and in a separate document).

## 2. Architecture: Backend Abstraction

To hide multiple backends behind the same API, `Segmenter` is a thin dispatcher that knows nothing about backend implementation details.

The set of backends — "KyTea-compatible / custom MLP" — is a **small, closed set fixed in advance**, and there is currently no requirement to add external plugins dynamically. Therefore, rather than an open extension mechanism via virtual functions (`virtual` + heap allocation), **closed polymorphism via `std::variant` + `std::visit`** is the first choice. This avoids vtable indirect calls and individual heap allocation of backend objects, and missing branches for unsupported backends can be detected at compile time.

```cpp
class KyTeaBackend { /* Section 3 */
public:
    std::expected<Segments, Error> tokenize(std::string_view text) const;
    std::expected<Boundaries, Error> tokenize_boundaries(std::string_view text) const;
};
class MlpBackend { /* Section 4 (to be detailed later) */
public:
    std::expected<Segments, Error> tokenize(std::string_view text) const;
    std::expected<Boundaries, Error> tokenize_boundaries(std::string_view text) const;
};

using AnyBackend = std::variant<KyTeaBackend, MlpBackend>;

class Segmenter {
public:
    // Automatically determine the format from the model file contents and load it
    static std::expected<Segmenter, Error> load(const std::filesystem::path& model_path);

    std::expected<Segments, Error> tokenize(std::string_view text) const {
        return std::visit([&](auto const& b) { return b.tokenize(text); }, backend_);
    }
    std::expected<Boundaries, Error> tokenize_boundaries(std::string_view text) const {
        return std::visit([&](auto const& b) { return b.tokenize_boundaries(text); }, backend_);
    }

    // Tokenize many inputs in parallel (result[i] corresponds to texts[i]; threads==0 means the
    // hardware parallelism count). Since inputs are mutually independent and the model is immutable,
    // this scales roughly proportionally to the core count (Section 9.4: 5.6x over single-thread
    // with 8 threads on an M1 Pro).
    std::vector<std::expected<Segments, Error>>
    tokenize_all(std::span<const std::string_view> texts, unsigned threads = 0) const;

private:
    AnyBackend backend_;
};
```

**Automatic model format detection**: `Segmenter::load()` looks at the signature at the start of the file and automatically chooses the backend (a KyTea model has a header line starting with `"KyTea "`; anything else is treated as the custom MLP format, Section 4.7). For cases where you want to specify the backend explicitly, `load_kytea(path)` / `load_mlp(path)` are also provided, with `load()` being a thin wrapper over them.

Each backend class only needs to satisfy the same `tokenize`/`tokenize_boundaries` signature; the internal feature extraction, classifier, and model parser can be implemented completely independently. The only thing the two backends have in common is "returning the same `Segments`/`Boundaries` types," and the design intent is that adding a future custom MLP model trained on the KyTea corpus format should require no changes at all to the other backend's code.

## 3. KyTea-Compatible Backend

- Loads the models output by KyTea (models trained with `train-kytea`) as-is.
- The features are the same three kinds as KyTea: character n-grams, character-type n-grams, and dictionary-derived word features.
- The training engine itself (LIBLINEAR equivalent) is out of scope for this library; initially only **inference** is supported.
- The model's binary format is parsed directly from KyTea's model file (no conversion-tool intermediary is used).

### 3.1 Feature Extraction (KyTea-compatible, requires faithful reproduction)

Although the classifier itself (LIBLINEAR's linear SVM) is generic, **the feature-string generation logic and character-type classification are KyTea's own implementation**, and to maintain compatibility with the model this must be reproduced faithfully, word for word (confirmed by referring to KyTea's own source `string-util.cpp` / `kytea.cpp`).

**Character-type classification (`StringUtil::CharType`)**

Six types are determined by converting one UTF-8 character to a Unicode code point and then applying the following range checks. The evaluation order matters as well (evaluated in the order Romaji → Hiragana → Katakana → Digit → Kanji → Other).

| Type | Symbol | Unicode range (summary) |
|---|---|---|
| ROMAJI | `R` | `0x41-0x5A`, `0x61-0x7A` (half-width Latin letters), `0xFF21-0xFF3A`, `0xFF41-0xFF5A` (full-width Latin letters) |
| HIRAGANA | `H` | `0x3040-0x3096` |
| KATAKANA | `T` | `0x30A0-0x30FF` (excluding `0x30FB` middle dot), `0xFF66-0xFF9F` (half-width Katakana) |
| DIGIT | `D` | `0x30-0x39` (half-width digits), `0xFF10-0xFF19` (full-width digits) |
| KANJI | `K` | `0x3400-0x4DBF`, `0x4E00-0x9FFF`, `0xF900-0xFAFF`, `0x20000-0x2A6DF`, `0x2A700-0x2B73F`, `0x2B740-0x2B81F`, `0x2F800-0x2FA1F` |
| OTHER | `O` | Everything else |

The KyTea main program supports three encodings — UTF-8/EUC/SJIS — but this library targets UTF-8 only.

Note that KyTea's `findType` has a bug (`<<18` appears in two places) in the code-point computation for 4-byte UTF-8 (code point ≥ U+10000, CJK Extension B and beyond), which misclassifies those Kanji characters. Since this library classifies from the correct code point, results may diverge from KyTea in this rare case (recorded in implementation comments and Section 8.6). They agree for ordinary Japanese text within the BMP.

**Input normalization (`normalize`, must be KyTea-compatible)**

At inference time, KyTea splits the input string into `surface` (the original) and `norm` (normalized), and all feature computation (character n-grams, character-type n-grams) is done on `norm` (`RawCorpusIO`: `norm = normalize(surface)`). Normalization uses a fixed table (about 110 entries) from `string-util-map-utf8.h`, which **folds half-width alphanumerics and symbols to full-width** (`a→ａ`, `0→０`, `(→（`, half-width Katakana punctuation `｢｣→「」`, etc.). The output word surface is cut out from `surface` (the original byte sequence), but the score used for boundary decisions is computed from `norm`. This library also ports this fixed table, building an ID sequence equivalent to `norm` in the order **UTF-8 decode → code-point normalization → interning** (`CharTable::encode`, implemented and verified).

**Feature-string format**

Based on the boundary position, the following three kinds of feature strings are generated within the range of the window width (default `charw=3`, similarly configurable for `typew`), converted to IDs via the model's dictionary, and passed to the linear classifier.

| Feature type | Prefix format | Example |
|---|---|---|
| Character n-gram | `"X" + relative position` + the string itself | `X-2`, `X-1`, `X0`, `X1` |
| Character-type n-gram | `"T" + relative position` + sequence of character-type symbols | `T-1`, `T0`, `T1` |
| Dictionary-derived word feature | `"D" + dictionary index + (L\|I\|R) + match length` | `D0L1` (dictionary 0, left-end match, length 1), `D1R3` |

The `L`/`I`/`R` in a `D` feature represents the positional relationship of the dictionary entry to the boundary (Left end / Inside middle / Right end).

**Feature-ID mapping uses a dictionary embedded in the model, not a hash**

KyTea's model builds `ids_` (feature string → ID) at training time and embeds it in the model file (`KyteaModel::mapFeat`). In other words, the inference side does not implement its own hash function; it must **load the feature dictionary from the model file as-is and look up IDs in that dictionary**. An unknown feature string (one that did not appear during training) does not exist in the model, so that feature is simply skipped (same behavior as KyTea).

If any of the above is implemented differently from KyTea (Unicode range boundaries shifted, a different feature-string format, a different default window width, etc.), loading the same model would produce different classification results, so the content of this section is strictly verified by tests (test cases are prepared to confirm byte-for-byte agreement with KyTea's actual output).

### 3.2 Model File Format (Verified from KyTea's Own Source)

By directly checking KyTea's own source (`model-io.cpp`, `model-io-binary.h`, `model-io-text.h`, `kytea.cpp`, `dictionary.h`), the following structure was confirmed.

**Header line**

```
KyTea <version> <T|B> <encoding>
```
Example: `KyTea 0.4.0 B utf8`. `version` is `MODEL_IO_VERSION` (`"0.4.0"` for quantized builds, `"0.4.0NQ"` for non-quantized builds with `DISABLE_QUANTIZE`). The format character is `T` = text, `B` = binary. If the version string does not match, KyTea itself raises an error, so this library also targets only the `0.4.0` series. The `"KyTea "` signature of this header line is also used for the backend auto-detection described in Section 2.

**Overall file section order** (`Kytea::writeModel`/`readModel`)

1. **Config**: `do_ws`, `do_tags`, `numTags`, `charWindow`, `charN`, `typeWindow`, `typeN`, `dictionaryN`, presence of bias, `epsilon`, `solverType`, and the character map (`StringUtil::serialize()`)
2. **Word-segmentation model** (`wsModel_`): one `KyteaModel`
3. **Tag models**: `numTags` pairs, each of "word list (global tag candidates) + `KyteaModel`"
4. **Word dictionary**: `Dictionary<ModelTagEntry>` (user dictionary and system dictionary; up to 8 combined into a single Aho-Corasick automaton)
5. **Subword dictionary**: `Dictionary<ProbTagEntry>` (for substring probabilities such as reading estimation)
6. **Language model (LM)**: `numTags` instances of `KyteaLM` (subword n-gram language model)

**Serialization of one `KyteaModel` (classifier)**

- Number of classes (`int32_t`. If 0 or less than 2, means "no model" and processing ends there)
- Solver type (a `char`, 1-byte enum value. 8 kinds: `L2R_LR`, `L2R_L2LOSS_SVC_DUAL`, `L2R_L2LOSS_SVC`, `L2R_L1LOSS_SVC_DUAL`, `MCSVM_CS`, `L1R_L2LOSS_SVC`, `L1R_LR`, `L2R_LR_DUAL`. The default is `L2R_L2LOSS_SVC_DUAL` = **L2-regularized L2-loss linear SVM (dual)**)
- Each class's label (`int32_t` × number of classes)
- Presence of bias (`bool`)
- `multiplier` (`double`. Scale factor to convert quantized weights back to real numbers)
- `FeatureLookup` (described below)

**`FeatureLookup` = a pre-compiled direct mapping from features to weights, built for inference**

Rather than writing the training-time feature-string → ID dictionary (`ids_`/`names_`) directly to the model file, KyTea **separately builds and writes out a `FeatureLookup` structure optimized for inference** (`buildFeatureLookups()`). It consists of the following seven elements:

| Field | Type | Content |
|---|---|---|
| `charDict` | `Dictionary<FeatVec>` | character n-gram → per-class weight vector |
| `typeDict` | `Dictionary<FeatVec>` | character-type n-gram → per-class weight vector |
| `selfDict` | `Dictionary<FeatVec>` | dictionary-derived self-string feature → per-class weight vector |
| `dictVector` | `FeatVec` | additional fixed-length weights related to the dictionary |
| `biases` | `FeatVec` | bias term |
| `tagDictVector` / `tagUnkVector` | `FeatVec` | weights related to tag estimation |

`Dictionary<FeatVec>` is an **Aho-Corasick automaton** (an array of `DictionaryState`: `failure` link, `gotos` (character → next state, sorted for binary search), `output` (the sequence of feature indices finalized at this state)); KyTea itself already uses Aho-Corasick for feature-extraction pattern matching (this confirms the positioning that Vaporetto's speedup is a Double-Array version of this).

**Binary layout of `Dictionary<Entry>` (confirmed from the `writeDictionary`/`readDictionary` templates in `model-io-binary.h`. A common framework used not only by `charDict`/`typeDict`/`selfDict` (`Entry=FeatVec`), but also the word dictionary (`Entry=ModelTagEntry`) and the subword dictionary (`Entry=ProbTagEntry`))**:

```
number of dictionaries : unsigned char (1 byte; if 0, means "no dictionary" and subsequent fields are not written)
number of states        : uint32_t
[for each state] each state (DictionaryState):
    failure      : uint32_t              // index of the failure-transition target state
    gotos count  : uint32_t
    [for each goto]
        character : KyteaChar (=unsigned short, 2 bytes)
        target    : uint32_t
    output count : uint32_t
    [for each output]
        value     : uint32_t              // index of the entry finalized at this state
    isBranch     : bool (1 byte)
number of entries : uint32_t
[for each entry] writeEntry<Entry>(...)     // differs by Entry type (see below)
```

`gotos` is sorted by character (`KyteaChar`), and the reader performs binary search via `DictionaryState::step()` (the binary-search implementation using `gotos.begin()`/`gotos.end()` from Section 3.1 can be ported directly to C++).

`writeEntry<Entry>` differs by Entry type (all confirmed via template specializations in `model-io.cpp`):

- `Entry = FeatVec` (for `charDict`/`typeDict`/`selfDict`): `uint32_t` element count → `FeatVal` (default `int16_t`) × element count. **No identifier such as the feature string is included at the head at all** (the corresponding string is represented by the `DictionaryState` transition path itself, so `Entry` purely holds only the weight vector).
- `Entry = ModelTagEntry` (word dictionary, `dict_`): `word` (`KyteaString`, below) → per tag level, (tag candidate array + each candidate's tag-dictionary membership bit `unsigned char`) → `inDict` (`unsigned char`, dictionary membership bitmask) → per tag level, `KyteaModel` (recursively in the Section 3.2 format; `0` means "no model," represented by just `int32_t 0`).
- `Entry = ProbTagEntry` (subword dictionary, `subwordDict_`): `word` → per tag level, (tag candidate array + each candidate's probability `double`).

The binary representation of `KyteaString` (a variable-length string) is `writeString` (base class `GeneralIO`), stored as a **NUL-terminated byte sequence** (content + the 1 byte `\0` are written together) (confirmed from the fact that `writeString(const std::string & str)` writes `str.length()+1` bytes). There is no length prefix; the reader reads forward until it hits `\0`.

**The type of `KyteaChar` is `unsigned short` (2 bytes, unsigned)**, and internally KyTea does not keep characters as UTF-8; it maps them once into this 2-byte integer internal representation before handling them (`StringUtil` is responsible for converting between strings and `KyteaChar` sequences). Since the Aho-Corasick automaton's transitions are built at the `KyteaChar` granularity, **the automaton must be walked using the character → 2-byte-integer mapping embedded in the model (below), not raw UTF-8 bytes**.

**`KyteaChar` is not a fixed Unicode code point but the ID of a per-model "character interning table," numbered in the order characters appeared during training** (determined by directly checking `StringUtilUtf8::mapChar`/`serialize`/`unserialize` in `string-util.cpp`).

- `mapChar(str, add)`: takes `str`, a string representing one UTF-8 character, and if it is known in `charIds_` (`std::map<string, KyteaChar>`), returns that ID. If unknown and `add=true`, assigns `charTypes_.size()` (the current registration count) as the new ID, and appends it to `charIds_`/`charTypes_` (the character type from Section 3.1)/`charNames_` (the original UTF-8 string).
- **ID `0` is a reserved sentinel for the empty string**: `unserialize()` always calls `mapChar("")` first to reserve ID `0` before reading the actual characters. Therefore actual characters start **from ID `1`**.
- **`serialize()` simply returns a single UTF-8 string that is the concatenation of all registered characters** (it just streams `charNames_[1]` through `charNames_[size()-1]` in order into an `ostringstream`). In other words, the content of the "character map" field in Section 2.2 is **a single UTF-8 string that is simply the concatenation, in ID order (= first-occurrence order), of every distinct character the model saw during training**.
- On the `unserialize(str)` side, after reserving ID `0` with `mapChar("")`, `mapString(str)` decodes `str` from the beginning one UTF-8 character at a time while calling `mapChar()`. When loading a fresh model, the table is empty, so sequential IDs `1, 2, 3, ...` are reconstructed in the order characters appear in `str` (matching the IDs assigned during training).

**Implementation point**: This library's model loader, after reading the "character map" string in Config, must decode it as UTF-8 one character at a time from the start and assign IDs starting from `1` in the order of appearance, building a `char(UTF-8) → KyteaChar` correspondence table (and a reverse lookup table if needed). When performing inference on input text as well, this correspondence table is used to convert UTF-8 characters to a `KyteaChar` sequence before walking the Aho-Corasick automaton.

**Handling of unknown characters (characters not in the training vocabulary) (confirmed in `string-util.cpp`/`kytea.cpp`, verified)**: At inference time, KyTea converts the input to a `KyteaChar` sequence via `mapString`, but since `mapString` calls `mapChar(str, add=true)` by default, **unknown characters are dynamically assigned a new ID of `charTypes_.size()` (i.e., training-time max ID + 1 and beyond)** (`string-util.cpp:113-127`). This library uniformly maps unknown characters to `kNoChar` (`0`). The two differ in ID assignment, but **the word-segmentation output is completely equivalent**. Reason: all three types of WS features in `calculateWS` (`kytea.cpp:885-919`) are Aho-Corasick matches (character n-gram, character-type n-gram, dictionary), all of which walk **an automaton keyed by the training-time IDs `1..K`**. An unknown character's ID — whether `0` (this library) or `K+1` (KyTea) — **does not exist on any transition**, so it cannot contribute at all to character n-gram or dictionary matches (equivalent). Its contribution to character-type n-grams is unaffected because character type is classified **directly from the code point** (`CharTable::encode_into`) and does not depend on the character ID (KyTea also obtains the same type via `charTypes_[id]=findType`). **Also confirmed empirically**: for unknown emoji and BMP symbols (😀😱ℵℶℷ★☆♠🍣ⅠⅡⅢ, etc.), byte-for-byte agreement with `kytea -notags` was observed (included in the golden fixtures). **However, 4-byte UTF-8 CJK Extension B Kanji (e.g., 𠮷 U+20BB7) are a separate axis** — this is not unknown-character handling but the intentional discrepancy caused by the `findType` bug from Section 3.1 (the type n-gram diverges between this library, which correctly classifies as Kanji, and KyTea, which misclassifies).

**Important: inference is designed as "get the weight vector directly at the moment of an Aho-Corasick match," not "string → ID → weight-array lookup."** This library's model loader, too, should not independently generate feature strings and hash/ID-convert them, but rather **parse this `FeatureLookup` (three Aho-Corasick automata + four weight vectors) directly from the file and hold it in the same structure** — this is the correct implementation approach.

**The weight type (`FeatVal`) is `int16_t`-quantized by default**

```cpp
#if DISABLE_QUANTIZE
    typedef double FeatVal;
    typedef double FeatSum;
#else
    typedef int16_t FeatVal;   // default
    typedef int32_t FeatSum;
#endif
```

Distributed trained models are usually built with the quantized build (`FeatVal = int16_t`). At load time, the `int16_t` weights are converted back to real-number weights by multiplying by `multiplier` (`double`). This library first prioritizes **reading quantized models (`int16_t`)**, and will support non-quantized models in the future, identified by the header version string (`"0.4.0NQ"`).

### 3.3 Inference Algorithm (Word Segmentation, Restored by Directly Examining `Kytea::calculateWS`)

By reading `calculateWS` in `kytea.cpp`, the flow from per-boundary score computation to the final segmentation decision was accurately restored. **This is exactly the computation that this library's `KyTeaBackend::tokenize`/`tokenize_boundaries` must reproduce.**

Let `N` be the number of characters in the sentence; there are `N-1` boundaries (immediately after each character, excluding the last). For each boundary `i` (`0 <= i < N-1`), `score[i]` is accumulated in the following order:

1. **Initial value**: `score[i] = biases[0]` (the first element of `FeatureLookup::biases_`. A constant common to all boundaries)
2. **Add character n-gram scores**: For all character n-grams matched by `charDict` (Aho-Corasick) against the normalized string (`sent.norm`, the string after dictionary-based surface normalization), add them via the `addNgramScores` logic. A single n-gram match, centered on its occurrence position, **contributes simultaneously to multiple boundaries within the window** (a `FeatVec` holds scores for the window width `window*2` as a single vector, and is distributed from the match position `pos`, with `base_pos = pos - window` as the origin, as `score[base_pos + j] += vec[j]` (`j` clipped to the valid range)).
3. **Add character-type n-gram scores**: Similarly added via `typeDict`, against the string converted into a sequence of character-type symbols (`R`/`H`/`T`/`D`/`K`/`O` from Section 3.1).
4. **Add dictionary-derived (D feature) scores**: An Aho-Corasick match against the word dictionary is obtained via `dict_->match(sent.norm)`, and added via `addDictionaryScores`. The index computation here is as follows (`len=score.size()`, `max=config.getDictionaryN()`, matched word's character length `wlen`, `lablen=min(wlen,max)-1`):
   - At the boundary of the match word's **left end** (position `end-wlen`, only if `end>=wlen`), add `dictVector[dictionary index*dictLen + (end-wlen)*3*max + lablen*3 + 0]` (corresponds to `D<dict index>L<lablen+1>` from Section 3.1)
   - At each boundary **inside** the match word (`end-wlen+1 <= k < end`), add `... + k*3*max + lablen*3 + 1` (corresponds to `D<dict index>I<lablen+1>`)
   - At the boundary of the match word's **right end** (position `end`, only if `end != len`), add `... + end*3*max + lablen*3 + 2` (corresponds to `D<dict index>R<lablen+1>`)
   - This formula uniquely fixes the correspondence between feature strings such as `D0L1`/`D0I2`/`D1R3` from Section 3.1 and offsets within `dictVector`.
5. **Overwrite with hard constraints (equivalent to `-wsconst`)**: If a character-type symbol specified in `config.getWsConstraint()` is included in a case where two adjacent characters have the same character type, the boundary's score is forcibly overwritten to the "no boundary" side (`-100` for non-probabilistic models, `0` for probabilistic models). A hard rule used for purposes such as not splitting runs of digits.
6. **Final score**: `wsConfs[i] = score[i] * wsModel_->getMultiplier()` (`multiplier` is the quantization scale factor from Section 3.2; if quantized, this converts the integer `score` back to a real number)
7. **Boundary decision**: if `wsConfs[i] > confidence` (default `confidence = 0`), there is a boundary; otherwise there is not. **Confirmed from the implementation of `KyteaSentence::refreshWS` that a threshold of 0 is indeed the default** (a simple inequality comparison `myConf > confidence`).
8. (Optional) If the solver is a probabilistic-model type (logistic-regression family), finally `wsConfs[i] = 1/(1+exp(-|wsConfs[i]|))` is applied as a sigmoid transform to present it as a probability, but since the boundary decision itself is determined by the sign in step 7, this transform is for display purposes only and does not affect the segmentation result itself.

In other words, the minimal inference logic to implement is a simple linear sum: **"starting from the bias as the initial value, add the three kinds of Aho-Corasick match scores from charDict, typeDict, and dictVector, multiply by multiplier, and compare against 0"** — inference (`tokenize_boundaries`) can be reproduced with just this accumulation process, without implementing the SVM's "training" part (LIBLINEAR) at all.

## 4. Custom MLP Backend (Ultimate Goal)

A backend that uses a custom-designed MLP (multi-layer perceptron) as the classifier, in place of the linear SVM of Section 3. The segmentation approach itself follows the same **pointwise binary classification** as Section 3 (independently deciding "split / do not split" for each boundary candidate), and the public API satisfies the same `tokenize`/`tokenize_boundaries`. The aim is to have **the interactions among the features that KyTea/Vaporetto enumerate by hand (character n-grams, character-type n-grams, dictionary features) be automatically acquired via in-window embeddings and hidden layers**.

The training data uses the KyTea corpus format (Section 5) as-is, allowing a fair comparison of the two backends on the same data. The training engine (forward pass, backward pass, optimization) is newly implemented in this library.

### 4.1 Design Philosophy and Comparison with KyTea/Vaporetto

| | KyTea (Section 3) / Vaporetto (external comparison only, not a backend of this library) | This MLP backend (Section 4) |
|---|---|---|
| Classification method | Pointwise binary classification | Same |
| Classifier | Linear SVM (linear sum of weights) | MLP (non-linear, multi-layer) |
| Features | Character n-grams, character-type n-grams, dictionary features, **hand-designed and enumerated** | In-window embeddings concatenated, **interactions learned by the network** |
| Character-type features | Explicitly uses heuristic 6-way classification (Section 3.1) | **Not used** (freedom from heuristic feature design is a benefit of DL; reconsider introducing Unicode General Category etc. in the future if accuracy is insufficient) |
| Atomic unit | Code-point "character" | **EGC (grapheme cluster) unit** (Section 4.2) |
| Vocabulary / OOV | Feature dictionary embedded in the model; unknown features are skipped | Embeddings use a **code-point vocabulary**; EGCs are represented compositionally, so there is in principle no OOV (Section 4.3) |

**Deliberately having no explicit character-type feature** is an intentional design decision. KyTea/Vaporetto's character-type n-grams (`H`/`K`/`T`/`R`/`D`/`O`) are entirely heuristic, and not having to hand-design such features is positioned as a benefit of the DL approach. Only if generalization to unknown/low-frequency cases becomes a problem is there room to introduce General Unicode Property (General Category, etc.) as an auxiliary input.

### 4.2 Atomic Unit, Boundary Candidates, and How Windows Are Counted = EGC

The atomic unit of classification is not the code point but the **EGC (Extended Grapheme Cluster, UAX #29)**. Reasoning and consequences:

- **Boundary candidates are only the gaps between EGCs.** Never split within an EGC (e.g., か + combining dakuten が = U+304B U+3099, or an emoji ZWJ sequence 👨‍👩‍👦 = 6 code points). EGC boundaries are a subset of code-point boundaries, and everywhere else it is always fine to say "do not split" (this is also linguistically sound). It will be empirically confirmed on training data that the KyTea corpus's gold-standard boundaries always coincide with EGC boundaries (a gold boundary never falls inside a grapheme cluster).
- **The window is counted in "number of EGCs."** If the window is measured in code-point count, a single 👨‍👩‍👦 alone would consume a left/right window=5 window and not even reach the actual adjacent word. With EGC units, 👨‍👩‍👦 can be treated as 1 token, leaving the remaining budget for the actual surrounding words. This is a design that does not waste the window budget on the number of constituent parts, and it helps with grapheme clusters, Thai orthographic syllables, Hangul ligatures, and more.

Let the sentence's EGC sequence be `e[0..M-1]`; there are `M-1` boundary candidates (immediately after each EGC, excluding the last). Each boundary `i` (`0 <= i < M-1`) is independently binary-classified.

### 4.3 Representation of EGCs = Compositional Embedding from Constituent Code Points

The approach of looking each EGC up directly as a vocabulary ID (a flat EGC-id embedding) is not adopted. The number of distinct EGCs balloons due to combining sequences, variation selectors, skin-tone modifiers, etc., and the vocabulary bloats with OOV increasing for Thai and Burmese (vocabulary explosion). Instead, **each EGC is decomposed into its constituent code-point sequence, and the code-point embeddings are pooled to compose a single EGC vector.**

```
EGC → decomposed into constituent code-point sequence
  each code point → embedding (vocabulary is "code points that appeared during training," or 256 bytes; small and bounded)
  → pooling → EGC vector (dimension d)
```

**A frequency threshold is set for vocabulary construction** (e.g., code points appearing fewer than 2 times are excluded from the vocabulary and mapped to UNK). If all occurring code points were put into the vocabulary, the UNK row (row 1 of Section 4.7) would never receive gradient, and an unknown code point at inference time would pull an untrained initial-value vector. By routing low-frequency code points through UNK during training, the UNK embedding learns "the average behavior of rare characters."

The essence of this two-layer structure is **separating the unit of prediction (EGC) from the vocabulary of the embedding (code point).**

| | Atomic unit (prediction / window) | Embedding vocabulary |
|---|---|---|
| Code-point scheme | codepoint | direct codepoint-id lookup |
| Flat-EGC scheme | EGC | direct EGC-id lookup (← vocabulary explosion) |
| **This scheme (compositional EGC)** | **EGC** | **pooled code points** |

Advantages:

- **No vocabulary explosion**: the embedding table's vocabulary is fixed at code points (a few thousand to 10,000, bounded) or bytes (256). Thai/Burmese syllables also decompose naturally into base + combining marks.
- **The OOV cliff disappears**: even an unknown EGC can have its vector composed as long as its constituent code points are known. **UNK at the EGC level is in principle unnecessary** (UNK at the code-point vocabulary level remains due to the frequency threshold above, but is hit far more rarely than UNK under the flat-EGC scheme).
- **Zero heuristic features**: there is no hand-design like a character-type table. A DL-clean solution of "learning composition from parts."

For Japanese and Chinese, 1 EGC ≈ 1 code point in the vast majority of cases, so pooling degenerates to the identity for most of them, effectively behaving as "code-point embedding." Pooling only becomes meaningfully active for combining sequences, Thai, Burmese, and emoji.

(Optional) A **hybrid** that also uses direct-lookup EGC-id embeddings for frequent EGCs, covering the rest with the compositional representation, is also an option (frequent EGCs get a direct lookup; rare ones absorb OOV via composition). The pooling method (mean / small encoder, etc.), dimensionality, and whether a hybrid is needed are worked out in the network configuration in Section 4.4.

### 4.4 Network Configuration (Finalized)

A pointwise classifier that "concatenates each in-window EGC vector plus dictionary features, passes them through a 1-hidden-layer MLP, and makes a binary decision." **The configuration was determined by working backward from the speed requirement (the layer-1 precomputation in Section 4.6).** Not placing any non-linearity before the first layer is the condition for the speedup to hold; mean pooling and a single hidden layer are consequences of this.

```
For boundary i:
  window = the EGC sequence of e[i-w+1 … i+w] (w on each side, 2w total; ends use a PAD token)
  each EGC → compositional embedding from Section 4.3 (dimension d, pooling is mean)
  f_dict = dictionary-match binary features (below)
  h = ReLU( W1 · concat(2w × d) + W_dict · f_dict + b1 )    # 1 hidden layer
  y = w2 · h + b2                                            # scalar
  y > 0 means boundary (equivalent to sigmoid(y) > 0.5; sigmoid is used only for the training-time loss computation)
```

**Finalized values**:

| Item | Value | Rationale |
|---|---|---|
| Window width `w` | **5** (5 each side, 10 EGC total) | Under the layer-1 precomputation scheme, the cost of increasing `w` is "one table-lookup-and-add per EGC," which is linear and nearly free (with a naive matrix product, the cost 2w·d×H would matter, but this constraint disappears). A wider window captures evidence for longer words and mitigates the accuracy drop when there is no dictionary |
| Embedding dimension `d` | **64** | Code-point vocabulary of ~10,000 × 64 stays lightweight |
| Pooling | **mean (finalized, not to be changed)** | Because mean is linear, `W1_j·mean(e_c) = mean(W1_j·e_c)` holds, which is compatible with the layer-1 precomputation (Section 4.6). Non-linear pooling such as attention or an encoder is not adopted because it would break this table-lookup transformation |
| Hidden layer | **1 layer, width H=256, ReLU** | For pointwise classification the benefit of depth is small, and inference cost is dominated by the hidden layer onward, so it is kept shallow. ReLU can be replaced with clipped ReLU for quantization |
| Regularization | dropout (rate to be decided experimentally) | |
| Output | 1 unit; at inference, **the sign of y** is used (sigmoid omitted) | `p>0.5 ⇔ y>0`. This is the same form as KyTea's step 7 (Section 3.3), a "score vs. 0" comparison |

**Dictionary-match binary features `f_dict`**: a **language-independent** external-knowledge channel with the same idea as KyTea's D features (Section 3.1). Dictionary (word list) matches are obtained via Aho-Corasick (sharing the implementation from Section 3, but run over the EGC sequence), and for a match that straddles/touches boundary i,

- 3 positional relationships (L: match's left end touches the boundary / I: match contains the boundary / R: match's right end touches the boundary)
- × match-length bucket (4 levels of `min(EGC length, 4)`)

giving 12 binary features in total (12 per dictionary when multiple dictionaries are supported). **If multiple dictionary words match the same (positional relationship × length bucket), the feature remains 1 (binary clamp).** KyTea's `addDictionaryScores` (Section 3.3) accumulates per match (equivalent to a count), but this backend adopts clamping to match the declaration of "binary features" and the precomputation path (an active feature = one vector addition), and this is recorded as an intentional difference from KyTea. Unlike language-specific heuristics such as character-type n-grams, a dictionary is "injection of external knowledge in the form of a word list" and is language-independent, so this is reconciled as not contradicting the "no heuristic features" policy of Section 4.1. **The layer-1 effect on a binary feature is "adding the 256-dimensional column vector corresponding to the active feature," so this rides directly on the precomputation path of Section 4.6** (one active feature = one vector addition). This also works with no dictionary (`f_dict` all zero).

### 4.5 Training

- **Loss**: per-boundary binary cross-entropy (BCE). `sigmoid(y)` is used only in the loss computation.
- **Masking the supervision signal**: "unknown" positions in partial annotation (Section 5.2) (` ` unkBound / `?` skipBound → `PROB_UNKNOWN`) are excluded from the loss computation. Full annotation provides a supervision signal for all boundaries. Intra-EGC positions are not boundary candidates to begin with, so they never appear in the loss either (Section 4.2).
- **Handling collisions between annotation and EGC**: since the corpus's boundary marks are attached at the code-point level, they can collide with the interior of an EGC. **A sentence in which a boundary (`|` or a full-annotation word boundary) falls inside an EGC is skipped in its entirety, with a warning** (the representation and the label contradict each other, so it cannot be used for training). "No boundary" (`-` or interior of a word) inside an EGC is consistent with the definition of an EGC and is silently absorbed. The skip count is reported, and also used as a statistic for the empirical check in Section 4.2 (the rate at which gold boundaries coincide with EGC boundaries).
- **Input normalization**: the same half-width→full-width fixed-table normalization as KyTea (Section 3.1 `norm`) is applied **before** EGC splitting and vocabulary lookup (scheme (a)). This aligns the input conditions with the KyTea backend to keep the comparison fair. The `CharTable` normalization table is shared.
- **Post-training quantization**: weights and embeddings are quantized to int16 after training (the same idea as KyTea itself, which uses int16 quantization + the multiplier scheme, Section 3.2). It is verified on validation data that quantization error does not flip boundary decisions.
- The optimizer, learning rate, batch composition, and dropout rate are to be decided experimentally at implementation time.

### 4.6 Inference / C++ Implementation Policy: Premised on Layer-1 Precomputation (NNUE Scheme)

**A naive forward pass is not adopted.** The matrix product `concat(2w·d=640) → 256` amounts to ~160,000 MACs per boundary, which would be several µs per boundary — **several times slower** than KyTea (~1µs/character), which relies solely on Aho-Corasick plus integer addition. The goal of "beat KyTea on CPU inference" is achieved through the following precomputation (the same structure as the NNUE evaluation function used in shogi/chess engines).

**Principle**: since layer 1 is a purely linear transform, it can be decomposed by window position:

```
W1 · concat(v_1, …, v_2w) = Σ_j  W1_j · v_j        (W1_j is the 256×d slice corresponding to window position j)
```

Therefore, **(EGC, window position j) → W1_j·v(EGC) ∈ R^256 can be precomputed as a lookup table.** Thanks to the linearity of mean pooling (Section 4.4), the compositional EGC embedding can also be folded into this table. The dictionary binary features likewise become a column-vector lookup of `W_dict`. The computation for one boundary at inference time is:

```
acc = b1
acc += table[egc_j, j]   for 2w iterations (table lookup + 256-dimensional vector addition)
acc += dict_col[k]       for each active dictionary feature (0 to a few times)
h = ReLU(acc)
y = dot(w2, h) + b2      (one 256-dimensional dot product)
boundary ⇔ y > 0
```

**≈ 2K ops per boundary.** About 1/80 of the naive implementation. With SIMD (NEON/AVX2), we expect **100–200 ns per boundary**, aiming to beat KyTea.

**Numeric representation of the table and accumulator**: `table[egc,j] = W1_j·v(egc)` is the sum of `d=64` int16×int16 products, which **can reach up to 2^36 in the worst case and does not fit in int16.** Therefore:

- **The first implementation uses an int32 table and an int32 accumulator** (a 256-dimensional × 4B = 1KB vector addition, `2w` times. SIMD throughput is halved compared to int16, but sub-µs per boundary is still achievable, keeping the speed target). There is no concern about overflow or saturation, making correctness easy to verify.
- **Re-quantization to int16 (a third scale + right shift + saturating clip) is positioned as a post-profiling optimization** (this is the form NNUE actually uses). When introduced, the saturation-occurrence rate and decision flips will be checked on validation data (the same framework as the quantization verification in Section 4.5).

- **Precomputation table**: built for frequent EGCs (≈ frequent code points). Vocabulary of 10,000 × 2w=10 positions × 256 × int16 ≈ 50MB. To reduce memory, only the most frequent entries can be table-ized, with rare EGCs falling back to the compositional path ("code-point embedding → mean → multiply by W1_j," itself a single small matrix-vector product). For Japanese/Chinese text, most cases go through the table-lookup path.
- **Sequential scanning**: since the window shifts by just 1 EGC as the boundary moves from i to i+1, reuse of the table-lookup results (differential update of the accumulator) is left as a future optimization candidate (first implement straightforwardly with `2w` additions, then profile).
- Tokenization (UTF-8 → EGC splitting) follows UAX #29. EGC-splitting logic is added on top of the existing `unicode/utf8`.
- Only the forward pass (inference) needs to be implemented. Training is done in a separate component (Python, or this library's training module), and handed off via the model file (Section 4.7).

### 4.7 Model File Format (Custom Design)

The serialization format is a custom design. **The design constraint is that it can be read as-is using `BinaryReader`'s (`bytes/binary_reader.h`) primitives (little-endian fixed-width integers, NUL-terminated strings, `\n`-terminated header line)**, sharing the same read infrastructure as the KyTea backend. The precomputation table (Section 4.6) and the Aho-Corasick automaton are not included in the file and are **built at load time** (the same approach Vaporetto takes for its own automaton).

**Header line (ASCII, `\n`-terminated)**

```
SegmentLibMLP <version>\n
```

Example: `SegmentLibMLP 1\n`. Read with `read_line()`. This `"SegmentLibMLP "` signature is used for the backend auto-detection in Section 2 (mutually exclusive with KyTea's `"KyTea "` signature). A version mismatch is treated as an error by the loader.

**Binary body following the header line** (all little-endian. Integers use `read<T>()`; `double` is stored as raw LE bytes — since `BinaryReader` only byte-swaps integers, supporting a big-endian host would separately require swapping `double`, as a future task; the same known limitation as Section 3).

| # | Field | Type | Content |
|---|---|---|---|
| **Config** | | | |
| 1 | `char_window` `w` | `uint8` | one-sided window width (in EGC count). `w=5` in Section 4.4 |
| 2 | `embed_dim` `d` | `uint16` | code-point embedding dimension. `64` in Section 4.4 |
| 3 | `hidden` `H` | `uint16` | hidden layer width. `256` in Section 4.4 |
| 4 | `num_dicts` | `uint8` | number of dictionaries (`0` allowed; if `0`, the W_dict and dictionary sections are not written) |
| 4b | `unicode_version` | `uint16` | the Unicode version used for EGC splitting at training time (major×100+minor, e.g., 15.1→`1510`). Since UAX #29 rules can change across versions, the loader warns if it mismatches the inference-side splitter |
| **Scales** (quantization scales, the same idea as KyTea's `multiplier`, Section 3.2) | | | |
| 5 | `emb_scale` | `double` | factor for embedding int16 → real number |
| 6 | `w1_scale` | `double` | factor for W1's int16 → real number |
| 7 | `wdict_scale` | `double` | factor for W_dict (only when `num_dicts>0`) |
| 8 | `w2_scale` | `double` | factor for w2 |
| 8b | `acc_scale` | `double` | **integer scale of the accumulator (layer-1 activation)**. Not derived from the weight scales; instead chosen after training by calibrating the activation distribution against validation data (`pct99.99(|a|)/Amax`, `Amax≈2^22`). The precomputation table, dictionary columns, and integerization of b1/b2 are all based on this scale (implementation details in `mlp_impl_design.ja.md` I.1). Since this is information the loader cannot reconstruct, it must be stored in the file |
| **Vocabulary** (code-point vocabulary) | | | |
| 9 | `vocab_size` `V` | `uint32` | number of embedding rows. Row 0 = PAD, row 1 = UNK (unknown code point) included |
| 10 | `codepoints` | `uint32 × (V-2)` | the code points corresponding to rows 2..V-1, stored in **ascending order**. At inference, the input code point is binary-searched in this array to obtain the row number (row 1 = UNK if not found). Rows 0/1 have no code point |
| **Embedding** | | | |
| 11 | `embedding` | `int16 × (V·d)` | embedding table. Row-major (row 0 = PAD, row 1 = UNK, row 2 onward = in the order of `codepoints`). An EGC vector is the mean of its constituent code-point rows (Section 4.3) |
| **Layer 1** | | | |
| 12 | `W1` | `int16 × (H · 2w · d)` | layer-1 weights. Row-major, `W1[h][j*d + c]` (`h` = hidden unit, `j` = window position 0..2w-1, `c` = embedding dimension). The `[·][j*d..]` slice corresponds to the per-position precomputation `W1_j` from Section 4.6 |
| 13 | `W_dict` | `int16 × (H · num_dicts · 12)` | weights for the dictionary binary features. `W_dict[h][dict*12 + feat]` (`feat` = L/I/R × 4 length buckets = 12). Only when `num_dicts>0` |
| 14 | `b1` | `double × H` | layer-1 bias (**not quantized**; kept as a real number since there are few of them and to avoid scale composition. The loader integerizes it to match the accumulator scale `emb_scale·w1_scale`) |
| **Layer 2** | | | |
| 15 | `w2` | `int16 × H` | output-layer weights |
| 16 | `b2` | `double` | output-layer bias (not quantized). Since the inference decision is `y > 0` (Section 4.4), a positive scale on `w2`'s side does not affect the sign, but `b2` must be aligned to the same scale as `w2·h`, and the loader integerizes it |
| **Dictionaries** (only when `num_dicts>0`; raw word lists for building Aho-Corasick) | | | |
| 17 | repeated for each dictionary (`num_dicts` times) | | |
| 17a | `entry_count` | `uint32` | number of words |
| 17b | `entries` | NUL-terminated UTF-8 × `entry_count` | word surface forms (read with `read_cstring()`). The loader applies scheme (a)'s normalization (Section 4.5), then splits into an EGC sequence per UAX #29 and builds an EGC-level Aho-Corasick automaton (the same normalization must be applied to dictionary words as well, since otherwise half-width-mixed dictionary words would never match if only the input side is normalized) |

**Load-time processing**: (1) read the vocabulary, embeddings, and weights; (2) build the per-position precomputation table `table[egc, j] = W1_j · v(egc)` from Section 4.6 (for frequent EGCs) and expand the dictionary feature column vectors `W_dict`; (3) build the EGC-level Aho-Corasick automaton from the word lists; (4) quantize `b1`/`b2` to the accumulator integer scale. All subsequent inference is table lookup plus integer addition only (Section 4.6).

**Future**: room is left to wrap the whole thing with a zstd-compression outer layer, as Vaporetto's own CLI tool does for its model files. In that case too, the auto-detection in Section 2 would examine the header line after zstd decompression.

### 4.8 Positioning and Evaluation Conditions

Full-scale work will begin once the Section 3/4 implementations have stabilized and an evaluation infrastructure with the common `tokenize`/`tokenize_boundaries` interface is in place.

**Comparison conditions for the "parity with KyTea" accuracy goal**: since the training corpus for the distributed model `jp-0.4.7-5.mod` (BCCWJ, etc.) is unavailable, it is not possible to create "an MLP trained on the same data as the distributed model." Comparison is therefore done by **retraining KyTea on an available corpus, and pitting it against this backend trained on the same data and the same dictionary** (the goal is "at least parity with the KyTea approach under identical conditions," not "comparison against the absolute accuracy of the distributed model"). Speed is compared against KyTea and Vaporetto on the same text, with the goal of beating KyTea on a single CPU thread.

**Known accuracy risk**: in-window embeddings alone may not be able to fully substitute for the lexical knowledge KyTea gains from its dictionary features. As mitigation, w=5 (wider than KyTea's charw=3) and the dictionary channel (Section 4.4) are provided from the start. If a gap remains even so, additional candidates include: an auxiliary General Unicode Property input (Section 4.1), widening the hidden layer, and augmenting the training data.

**Initial empirical measurement (2026-07-12, pipeline validation)**: in place of the distributed model's training data (BCCWJ, UniDic, and CSJ are all unavailable under NINJAL's permission system, and the one free corpus KyTea's official documentation cites also has a dead reference link), **UD_Japanese-GSD** (`UniversalDependencies/UD_Japanese-GSD`, CC BY-SA 4.0, short-unit = UniDic-based, the same lineage of segmentation criteria as KyTea's default model) was adopted. Fetching and conversion to KyTea format are done by `scripts/fetch_ud_gsd_corpus.sh` / `scripts/convert_ud_gsd_corpus.py` (tag 0 = UniDic POS, tag 1 = UnidicInfo's lForm reading). train: 7,050 sentences (0 collisions with EGC boundaries, 0 invalid UTF-8, 270,163 boundaries) / dev: 507 sentences / test: 543 sentences.

- `train-kytea -notags` (same train set, no dictionary) and `segmenter train --backend mlp` (defaults w=5, d=64, H=256, epochs=30, patience=5, no dictionary, early stopping and quantization calibration on dev) were trained on the same data.
- Test-set boundary F-score (`scripts/eval_segmentation.py`): **KyTea 98.65%** (P 98.73 / R 98.56) vs. **MLP 97.91%** (P 97.57 / R 98.25). Decision flips due to quantization: **0** out of 19,625 dev-set cases.
- The gap of about 0.7 points matches the expectation from Section 4.8, i.e., "without a dictionary, it does not fully reach the lexical knowledge KyTea gains from dictionary features." However, on this small corpus (this corpus has 270,000 boundaries against the ~5 million assumed in Section 4.9), this closeness is positioned as initial evidence that the default configuration (w=5, etc.) does not need to be changed. **Speed comparison has not yet been performed** (this round is a reference figure only, including CLI process-startup cost; an in-process benchmark following the Section 9 methodology will be performed together with the SIMD optimization in step 9).
- The corpus itself and the trained model (`corpus/ud-gsd/`) are gitignored, as with `models/`. Reproduction is possible with the two scripts above plus this command sequence.

**Initial speed measurement (same day, same train set of 7,050 sentences / 277,433 code points, in-process, best-of-7, `bench/bench_segment` / `bench/.vendor/bench_kytea`, same methodology as Section 9, M1 Pro)**:

| Implementation | WS inference speed | vs. KyTea |
|---|---|---|
| This library's MLP backend (`corpus/ud-gsd/mlp.mod`, **first implementation as-is, int32 table, no SIMD applied**) | 2.24 M chars/sec | **1.53x** |
| This library's KyTea backend (reference, `corpus/ud-gsd/kytea.mod`) | 9.73 M chars/sec | 6.66x |
| Real KyTea (libkytea, `corpus/ud-gsd/kytea.mod`) | 1.46 M chars/sec | 1.00x |

**The "beat KyTea on CPU inference" goal set out in Section 4.6 has already been achieved (1.53x) with just the first implementation, before applying int16 re-quantization or the SIMD kernel (the optimizations planned for the latter half of Section 4.6 / I.3).** It still, however, lags well behind this library's own KyTea backend (because the 2w additions of a 256-dimensional accumulator plus a dot product are inherently heavier than KyTea's sparse Aho-Corasick plus int32 addition). int16 re-quantization and SIMD-ization are repositioned not as a necessary condition for "beating KyTea" but as an additional optimization to close this remaining gap.

**Cross-genre generalization re-measurement (2026-07-12, applying the GSD-trained models to a different genre as-is)**: the initial measurement used only GSD's (Wikipedia-derived) 543-sentence test set, leaving it unclear whether the 0.7pt accuracy gap was noise from a small test set or a representative value. So an **out-of-domain test set in a different genre**, UD_Japanese-PUD (`UniversalDependencies/UD_Japanese-PUD`, CC BY-SA 4.0, news/Wikipedia parallel translations, 1000 sentences / 27,788 boundaries; same UniDic short-unit basis as GSD, with SUW segmentation, XPOS, and UnidicInfo readings), was added, and the `kytea.mod` / `mlp.mod` trained on GSD were evaluated on it **without retraining**. Fetching is done by `scripts/fetch_ud_pud_corpus.sh` (conversion reuses the GSD `convert_ud_gsd_corpus.py` verbatim, as a shared UD-Japanese format).

| Test set (genre) | Boundaries | KyTea F1 | MLP F1 | Gap |
|---|---|---|---|---|
| GSD test (Wikipedia, in-domain) | 12,491 | 98.65% | 97.91% | 0.74pt |
| PUD (news/Wikipedia parallel, out-of-domain) | 27,788 | **99.18%** | **98.56%** | 0.62pt |
| GSD+PUD combined (2 genres) | 40,279 | 99.02% | 98.35% | 0.67pt |

- **No accuracy degradation in the different genre** (both models score higher on PUD, since its parallel-translation sentences are more regular than GSD's Wikipedia articles), and **the MLP–KyTea gap is stable at 0.6–0.7pt**. The initial GSD-only 0.74pt is confirmed to be a representative value, not small-test-set noise.
- The gap **not widening** out-of-domain indicates that the MLP's in-window embedding approach generalizes across genres about as well as KyTea's dictionary features. This corroborates the initial judgment that "the configuration (defaults such as w=5) does not need to change," now on 3× the data across 2 genres.
- The corpus itself (`corpus/ud-pud/`) is gitignored like `models/` and `ud-gsd/`. Reproduce with the fetch script above plus `scripts/eval_segmentation.py --gold corpus/ud-pud/test.kytea.txt --command '...kytea...' --command '...segmenter...'`.

**Multi-genre speed re-measurement (same day, `bench/bench_segment` in-process, best-of-8 process runs × 15/30 iterations, M1 Pro)**: the speed measurement two sections up used only one genre — GSD train (Wikipedia-derived, 277,433 chars). PUD test (news/Wikipedia parallel translation, 48,260 chars) was added as a second genre:

| Genre | MLP (int16+NEON) | Real KyTea (libkytea) | Ratio |
|---|---|---|---|
| GSD train (Wikipedia, 277K chars) | 3.34 M chars/sec | 1.40 M chars/sec | 2.39x |
| PUD test (news translation, 48K chars) | 3.49 M chars/sec | 1.58 M chars/sec | 2.21x |

- Both genres fall within the previously recorded "2.2–2.4x" range, confirming that **segmentation speed is genre-insensitive** (expected — the score computation is pure integer arithmetic over the EGC window and does not depend on vocabulary or style). The corpus-size difference (277K vs. 48K chars) leaves some measurement noise, but no order-of-magnitude discrepancy appears.
- With this, Section 10's "re-measurement on a larger, more genre-diverse corpus" is done for both accuracy and speed. The only thing left unmeasured is a distributed-model-scale corpus (BCCWJ-equivalent), which remains unobtainable per the opening of Section 4.8.

### 4.9 Training-Side Design (Self-Implemented in C++)

The training engine is self-implemented in this library (no dependency on an external framework). Training is fp32; inference is int16 (Section 4.6); the two are exchanged via the model file (Section 4.7). **Inference (the deliverable of the library proper) is forward-pass only**, and the training component is separated at the build level (the inference binary does not require BLAS/CUDA).

**Training pipeline**

```
1. Read corpus (Section 5, KyTea full/partial annotation)
2. Normalize (scheme (a): KyTea-compatible half-width→full-width, sharing CharTable, Section 4.5)
3. Split into EGCs (UAX #29)
3b. Build vocabulary: tally the frequency of occurring code points, and route those below the threshold to UNK (Section 4.3; ensures the UNK row is trained)
4. Generate examples: for each boundary i →
     - each EGC in the window [i-w+1 … i+w] → its sequence of constituent code-point row IDs (ends use PAD)
     - dictionary-match binary features f_dict (Aho-Corasick, Section 4.4)
     - label (boundary=1/non-boundary=0)
     - mask (unknown positions in partial annotation are excluded from the loss, Section 4.5)
5. Minibatching: embedding gather → mean pooling → concat(2w·d)
6. Forward pass → BCE (masked) → backward pass (sparse gradients for embeddings)
7. Optimization: Adam
8. After convergence: PTQ int16 quantization → verification (decision-flip check, Section 4.5) → write out in the Section 4.7 format
```

**Key points of the backward pass**

- **Layers 1 and 2**: standard dense-layer gradients. The matrix product is the main computation, delegated to `ComputeBackend`'s GEMM.
- **mean pooling → embedding**: since an EGC vector is the mean of its constituent code points, the gradient with respect to the EGC vector is distributed to each constituent code-point row with weight `1/(number of code points)`. **The embedding table's gradient is sparse** (only the rows that appeared in the batch). Adam's first and second moments are likewise updated only for the rows that appeared.
- The `W_dict` gradient for the dictionary binary feature `f_dict` is also a sparse update, only for the columns of active features.

**Optimizer**: Adam (learning rate, β, and weight decay to be decided experimentally). Chosen independently for this backend because it plays well with sparse embedding updates, unrelated to KyTea/Vaporetto's linear SVM.

**Quantization**: finalized as PTQ (Post-Training Quantization, Section 4.5). Since int16 has 65,536 levels and thus minimal relative error, and the decision depends only on the sign of `y>0`, a single post-training rounding pass is nearly lossless. **QAT (Quantization-Aware Training) is not adopted this time** — a technique that inserts fake-quantization into the forward pass during training and backpropagates via STE, needed only for aggressively low bit widths such as int8/int4. It is recorded only as a **conditional fallback** for: (1) if the decision-flip check in Section 4.5 shows excessive flipping, or (2) a future move to int8 (model shrinkage, or inference on something like the Apple Neural Engine).

**Compute backend abstraction (`ComputeBackend`)**

Only matrix products, activations, elementwise operations, and gradients are confined to this layer, with platform-specific implementations swapped in. The CPU implementation is written against a BLAS interface (`cblas_sgemm`, etc.), so switching the linked BLAS is enough to support all OSes.

| Platform | First choice | CPU implementation (BLAS) | GPU implementation (optional) |
|---|---|---|---|
| **macOS (Apple Silicon)** | **CPU/AMX** | Accelerate (transparently uses the AMX matrix coprocessor) | Metal/MPSGraph (usually unnecessary) |
| **Linux + NVIDIA** | **GPU** | OpenBLAS/MKL | cuBLAS |
| Linux (no GPU) | CPU | OpenBLAS/BLIS | — |
| Windows + NVIDIA | GPU | OpenBLAS/MKL | cuBLAS |
| Windows (no GPU) | CPU | OpenBLAS | — |

- **`CpuBackend` (BLAS)**: common to all OSes. Only the internals are swapped among Accelerate / OpenBLAS / MKL.
- **`CudaBackend` (cuBLAS)**: common to Linux and Windows.
- **`MetalBackend`**: macOS only, an add-on option.
- Support policy: **macOS and Linux are first-class supported; Windows is best-effort** (C++23 features such as `std::expected` are available on MSVC 19.33+, and cuBLAS/OpenBLAS/zstd are all Windows-ready so architecturally it fits naturally, but CI/build-related work is deprioritized).

**Rationale for the design by platform**:

- **On M1, Accelerate CPU is the right answer, not GPU.** Apple Silicon has an AMX matrix unit that is transparently usable via `cblas_sgemm`, achieving effective ~1–1.5 TFLOP/s at this GEMM scale (zero kernel-launch overhead, no transfer cost thanks to unified memory). Hitting the M1 Pro GPU (~5 TFLOP) via Metal would land in the same few-minute range, not worth the effort of writing Metal.
- **On Linux, GPU is the opposite: it pays off.** Consumer x86 CPUs have no AMX equivalent, with effective BLAS at only ~200–800 GFLOP/s, whereas a discrete NVIDIA GPU is drastically faster. CUDA (cuBLAS) is the first choice.
- The ANE (Apple Neural Engine) is geared toward int8/fp16 inference and cannot be used for arbitrary backward passes; **it is not supported for training** (it could separately be a target for a future int8 inference port).

**Training-time estimate** (boundary count ~5 million, 30 epochs ≈ 150 TFLOP-equivalent; KyTea (LIBLINEAR) takes ~a few minutes at the same scale):

| Environment | Training time (rough) | vs. KyTea |
|---|---|---|
| macOS M1 Pro (Accelerate/AMX) | 3–6 minutes | ≈1–2× |
| Linux + CUDA (RTX 3060 class) | 2–5 minutes | ≈1× |
| Linux CPU (OpenBLAS, 8–16 cores) | 12–20 minutes | ≈4–6× |

Compared to the sparse linear SVM (KyTea), the dense computation per example is several hundred to 1000 times larger, but since the model is small, the absolute time stays within "minutes to tens of minutes." Where the GPU's advantage really pays off is running many hyperparameter-search iterations, expanding `d`/`H`, and increasing corpus size.

**Portability on the inference side (a separate axis from training)**: inference uses hand-written SIMD int16 NNUE style (Section 4.6), not BLAS, covering all OSes with **NEON (Apple/ARM) + AVX2 (x86 = common to Linux/Windows) + a scalar fallback**, three implementations plus a fallback. The AVX2 path on x86 is identical between Linux and Windows.

## 5. Corpus Specification (Verified from KyTea's Own Source)

`corpus-io-full.cpp` / `corpus-io-part.cpp` were directly examined. The default delimiter characters are defined in `kytea-config.cpp` as follows:

| Purpose | Character | Default |
|---|---|---|
| Word boundary (full) / Unknown boundary (partial) | `wordBound_` / `unkBound_` | half-width space `" "` |
| Tag boundary | `tagBound_` | `/` |
| Tag-candidate separator | `elemBound_` | `&` |
| Escape | `escape_` | `\` |
| No boundary (partial) | `noBound_` | `-` |
| Boundary (partial) | `hasBound_` | `\|` |
| Skip (partial) | `skipBound_` | `?` |

All backends (Sections 3–4) unify their training data on this KyTea corpus format.

### 5.1 Full Annotation Format

```
word1/tag0a&tag0b/tag1a word2/tag0 word3 ...
```

- Words are separated by a half-width space.
- Each occurrence of `/` advances the tag "level" by one (since KyTea can hold multiple tag systems simultaneously, such as reading and part-of-speech, levels express the tag type).
- To give multiple candidate tags within the same level, connect them with `&` (at training time, the first candidate is used as the correct label).
- To include a delimiter character itself (space, `/`, `&`, `\`) inside a word or tag, escape it with `\`.
- Real example (a format actually used in the training data for a distributed KyTea model):
  ```
  コーパス/ko:pasu の/no 文/buN で/de す/su 。/.
  ```

### 5.2 Partial Annotation Format

```
ヴ-ェ-ネ-ツ-ィ-ア|は|イ-タ-リ-ア|に|あ り ま す|。
```

- Characters are lined up one at a time, with one of the following symbols placed between adjacent characters to express boundary information:
  - `-` (noBound): not a boundary (within the same word). Used deterministically as a supervision signal.
  - `|` (hasBound): a boundary. Used deterministically as a supervision signal. Also serves as the end of a word.
  - ` ` (unkBound) / `?` (skipBound): unknown. When read in, both are treated as "no supervision signal in wsConfs" (`PROB_UNKNOWN`) (note that the two symbols are distinguished in the code, but the current loading logic treats them identically).
- Tags can be attached at the end of each word, immediately before the `|`, using the same syntax as full annotation: `/tag0&tag1/tag2...`.
- Escape character, tag boundary, and tag-candidate separator are shared with full annotation (`\`, `/`, `&`).

### 5.3 Notes

- Since this library is currently aimed only at **inference**, corpus loading (the training pipeline) is not directly in scope. However, the format is recorded accurately for future training-feature support (including the custom MLP backend of Section 4) and for verifying I/O compatibility with KyTea.
- The CLI's output format (Section 7.2) corresponds directly to the write logic of the above full-annotation format (`FullCorpusIO::writeSentence`).

## 6. C++ API

### 6.1 Basic Policy

- Rather than a side-effect-based API that reuses mutable objects (KyTea's `KyteaSentence&`, Vaporetto's `Sentence`),
  a **functional API that receives input and returns the result as a value** is the basic approach.
- Since the input requires no ownership, it is received as `std::string_view` (avoiding unnecessary copies).
- Errors are represented with `std::expected` (C++23). "Empty result" and "an error occurred" are distinguished by type.
- Offsets adopt **UTF-8 byte offsets** (this makes it easy to slice from the original string).
- Differences among backends (KyTea-compatible / custom MLP) are never exposed at this API layer (see Section 2). Callers don't need to be aware of the type of model file loaded by `Segmenter::load()`; they simply call the same `tokenize()`/`tokenize_boundaries()`.

### 6.2 Types

```cpp
struct Segment {
    std::size_t start;                 // UTF-8 byte offset (start, inclusive)
    std::size_t end;                   // UTF-8 byte offset (end, exclusive)
    std::vector<std::string_view> tags; // tags such as reading/POS (empty if tag estimation is not performed)
};

using Segments = std::vector<Segment>;
using Boundaries = std::vector<std::size_t>; // sequence of boundary offsets for segmentation-only use

enum class ErrorCode {
    InvalidUtf8,
    ModelNotLoaded,
    UnsupportedModelFormat,
    // ...
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
    // Automatically detect the backend type from the model file contents and load it
    static std::expected<Segmenter, Error> load(const std::filesystem::path& model_path);

    // Load with the backend explicitly specified (when not using auto-detection)
    static std::expected<Segmenter, Error> load_kytea(const std::filesystem::path& model_path);
    static std::expected<Segmenter, Error> load_mlp(const std::filesystem::path& model_path);

    // Segmentation + tag estimation
    std::expected<Segments, Error> tokenize(std::string_view text) const;

    // Segmentation only (a lightweight path that skips tag estimation)
    std::expected<Boundaries, Error> tokenize_boundaries(std::string_view text) const;
};
```

- `tokenize()` and `tokenize_boundaries()` are split at the type level in order to explicitly provide a lightweight path that can skip allocating storage for tag estimation.
- An empty-string input is treated as "not an error, but an empty result" (`Segments{}` / `Boundaries{}`). `Error` is used only for actual abnormal cases such as invalid UTF-8 or an unloaded/unsupported model.

### 6.4 Allocator (Future Consideration / Under Study)

As an extension proposal for avoiding allocation cost in high-throughput use cases (loops that continuously process large numbers of sentences), an overload that can inject a `std::pmr::memory_resource` is under consideration:

```cpp
using PmrSegments = std::pmr::vector<Segment>;

std::expected<PmrSegments, Error> tokenize(
    std::string_view text,
    std::pmr::memory_resource* mr) const;
```

By reusing a `std::pmr::monotonic_buffer_resource`, the caller can avoid reallocating internal working buffers such as boundary scores.
However, this is low priority; the plan is first to measure with a straightforward default-allocator implementation, and introduce this only if it actually turns out to be a bottleneck.

## 7. CLI Interface

The command name is `segmenter`. It uses a **subcommand structure** like `git`/`cargo`, consolidating into a single binary what the KyTea main program splits into `train-kytea` (training) and `kytea` (inference).

```
segmenter predict --model model.bin < input.txt > output.txt
segmenter train --backend kytea --corpus corpus.txt --model-out model.bin
```

### 7.1 `segmenter predict` (Inference)

As with KyTea / Vaporetto, this follows the **filter-style pattern of reading text from standard input and writing segmentation results to standard output**.

```
segmenter predict --model model.bin < input.txt > output.txt
segmenter predict --model model.bin --boundaries-only < input.txt > output.txt   # equivalent to tokenize_boundaries
segmenter predict --model model.bin --scores < input.txt > output.txt            # also output per-boundary scores
```

**Options (initial proposal)**

| Option | Description |
|---|---|
| `--model <path>` | path to the model file (required). Auto-detects whether it is KyTea-compatible / custom MLP (Section 2) |
| `--backend <kytea\|mlp>` | explicitly specify the backend (to override auto-detection) |
| `--boundaries-only` | perform only segmentation without tag estimation (uses `tokenize_boundaries`) |
| `--scores` | also output the classification score for each boundary |
| `--encode <utf8\|...>` | input encoding (only UTF-8 is supported for now; reserved as room for future extension) |

**Output format**

Follows KyTea's output format (`word/tag1/tag2 ...` space-separated) to preserve compatibility with existing toolchains (exactly the full-annotation format of Section 5.1).

```
コーパス/ko:pasu の/no 文/buN で/de す/su 。/.
```

**Escaping the surface word (`showEscapedString`)**: when a delimiter character (space, `/`, `&`, or the escape character `\` itself) is included in a word's surface form, KyTea escapes it by prefixing it with `\`. Reproducing this is essential for byte-for-byte agreement (example: input `Hello World` → `Hello \  World`, `2024/12/31` → `2024 \/ 12 \/ 31`). The implementation is consolidated in `append_full_line` (`src/output.cpp`), shared by the CLI and the golden tests. All four characters are 1-byte ASCII, so there is no conflict with UTF-8 continuation bytes.

### 7.2 `segmenter train` (Training)

**The training engine itself, as described in Section 3, is not yet started (see the Section 9 TODO), but the shape of the CLI interface is settled ahead of time.** The input corpus is unified on the KyTea corpus format (full/partial annotation) finalized in Section 5.

```
segmenter train --backend kytea \
  --corpus full1.txt --corpus full2.txt \
  --partial-corpus part1.txt \
  --dict dict.txt \
  --model-out model.bin
```

**Options (initial proposal; the `train-kytea` option names (`-charw`/`-charn` etc. from Section 3.2) are mapped as-is to their `--`-prefixed long forms)**

| Option | Description |
|---|---|
| `--backend <kytea\|mlp>` | the backend to train (required). The training engine itself is a completely separate implementation for each of the two backends (Section 2) |
| `--corpus <path>` | full-annotation corpus (Section 5.1). Can be specified multiple times |
| `--partial-corpus <path>` | partial-annotation corpus (Section 5.2). Can be specified multiple times |
| `--dict <path>` | dictionary file (`word tag` format). Can be specified multiple times |
| `--model-out <path>` | output path for the trained model (required) |
| `--char-window <int>` | window width for character n-grams (equivalent to KyTea's `-charw`, default 3) |
| `--char-n <int>` | maximum order of character n-grams (equivalent to `-charn`, default 3) |
| `--type-window <int>` | window width for character-type n-grams (equivalent to `-typew`, default 3) |
| `--type-n <int>` | maximum order of character-type n-grams (equivalent to `-typen`, default 3) |
| `--dict-n <int>` | rounding cap for dictionary match length (equivalent to `-dicn`, default 4) |
| `--cost <float>` | SVM regularization cost (equivalent to `-cost`) |
| `--eps <float>` | training convergence epsilon (equivalent to `-eps`) |
| `--boundaries-only` | train only the segmentation model, without training the tag (reading/POS) model (equivalent to `-notags`) |

How the training engine will be implemented (whether to self-implement the LIBLINEAR equivalent or interface with an external one) remains a TODO in Section 9, but **the corpus loading, feature extraction, and shape of the CLI arguments are already uniquely determined from the findings of Sections 3 and 5**, so a natural split for implementation order is "corpus parser + feature extraction (shared between training and inference)" → "linear classifier training (not yet started)."

## 8. Implementation Module Structure (`segmenter predict` MVP)

The first milestone is getting `segmenter predict --model kytea-model.bin < input.txt > output.txt` working. The data structures and algorithms of KyTea itself (Section 3) are followed, but **KyTea's implementation (manual `new`/`delete` on raw pointers, its own reference-counted string `KyteaString`, training, inference, and config parsing all living together in a single `Kytea` god class, the `THROW_ERROR` macro) is not emulated as a code structure.** A layered structure, thinly separated by responsibility, is used instead.

### 8.1 Layer Structure (Lower to Upper; Upper Depends Only on Lower)

```
include/segmentlib/
├── bytes/
│   └── binary_reader.h        # (L1) primitive binary-reading cursor
├── unicode/
│   └── utf8.h                 # (L1) pure functions for UTF-8 decode/encode
├── kytea/
│   ├── char_table.h           # (L2) character-type classification + character interning table (Sections 3.1, 3.2)
│   ├── automaton.h            # (L2) Aho-Corasick runtime representation + matching (Section 3.2 DictionaryState)
│   ├── model.h                # (L3) data types for Config/KyteaModel/FeatureLookup + load()
│   ├── scorer.h                # (L4) the calculateWS score-computation algorithm (Section 3.3)
│   └── kytea_backend.h        # (L5) Backend interface implementation (tokenize/tokenize_boundaries)
├── segmenter.h                 # (L6) public API (Section 6). Currently holds only KyTeaBackend in the variant
└── types.h                     # (L1) Segment/Segments/Boundaries/Error (Section 6.2, no dependencies)

src/
├── kytea/{char_table,automaton,model,scorer,kytea_backend}.cpp
├── segmenter.cpp
└── cli/
    ├── main.cpp                  # subcommand dispatch for `predict`/`train`
    ├── predict_command.cpp       # the body of the predict subcommand
    └── train_command.cpp         # train: currently just a stub that returns "not yet implemented"
```

### 8.2 Responsibilities and Design Intent of Each Module

**L1: `bytes::BinaryReader`** — a thin cursor that just advances over a `std::span<const std::byte>`. Provides `read<T>()` (`uint32_t`/`int32_t`/`bool`/`char`/`double`, etc.) and `read_cstring()` (the NUL-terminated `KyteaString` representation confirmed in Section 3.2). **Parse errors are internally thrown as exceptions (a lightweight `ParseError`) without being caught locally, and are converted to `std::expected` at exactly one boundary — `load()` in `model.h`.** This avoids the pitfall of KyTea's `THROW_ERROR` macro plus missed checks at call sites, while remaining consistent with the "public API is `std::expected`" policy decided in Section 6.1.

**L1: `unicode::utf8`** — a pure function that decodes one UTF-8 code point. A general-purpose subordinate module that knows nothing about KyTea-specific concepts.

**L2: `kytea::CharTable`** — a value type holding the "character interning table" confirmed in Section 3.2 (ID `0` = sentinel, real characters start at `1` in first-occurrence order). Provides `decode(std::string_view utf8) -> std::vector<std::uint16_t>` (this library's `KyteaChar` representation can just be plain `std::uint16_t`, following only the naming `KyteaChar` from KyTea), and the character-type classification of Section 3.1 (`char_type(std::uint16_t) -> CharType`, `enum class CharType : std::uint8_t { Romaji, Hiragana, Katakana, Digit, Kanji, Other }`).

**L2: `kytea::Automaton<Payload>`** — the runtime representation of `Dictionary<Entry>` from Section 3.2. Made a template so that it can support not only `Payload = FeatVec` (`charDict`/`typeDict` used by this MVP) but also, in the future, `Payload = WordDictEntry` (the word dictionary for tag estimation) with the same type. Rather than holding states as an array of raw `new`-ed pointers as KyTea does, it is a value type flatly holding `std::vector<State>` and `std::vector<Payload>`. Both an allocation-returning `match() -> std::vector<Match>` and a callback-based `match(text, callback)` for allocation-averse cases are provided (anticipating the same idea as the pmr policy of Section 3.4, first via just the callback language feature). **This module only receives the deserialization result of the model file and does not itself hold the logic to build an Aho-Corasick automaton** (`buildGoto`/`buildFailures`) — as confirmed in Section 3.2, the file already contains the pre-built automaton as-is. The custom MLP backend (Section 4), conversely, builds this `Automaton` from a flat dictionary word list at load time, so the "construction logic" will later be split out as `AutomatonBuilder`, separate from `Automaton`.

**L3: `kytea::Model`** — an immutable value type corresponding to `Config`/`KyteaModel`/`FeatureLookup` from Section 3.2. `static auto load(std::filesystem::path) -> std::expected<Model, Error>` is the sole construction path. In the MVP it is enough to hold only `charDict`/`typeDict`/`selfDict`/`dictVector`/`biases`/`multiplier`/`bias`/`wsConstraint` for `predict` to work. The word dictionary (`ModelTagEntry`), subword dictionary (`ProbTagEntry`), language model (`KyteaLM`), and tag models (`globalMods_`) **need not be retained as values, but if present in the file must be skippable correctly to the correct subsequent seek position** (since the Section 3.2 section order is fixed, an implementation that "just counts bytes and skips" is needed if not reading them). Fields will be added to `Model` when tag estimation is implemented.
  - **Regular `predict` that is not `--boundaries-only` (the command example above) also requires tag estimation**, so how the MVP's final output format is handled is supplemented in Section 8.3.

**L4: `kytea::scorer`** — implements the `calculateWS` algorithm restored in Section 3.3, directly as a function. It is a pure function that takes `Model`, `CharTable`, and the input text, and returns the per-boundary score sequence (rather than mutating a `Kytea` instance's internal state as KyTea does).
  ```cpp
  auto score_boundaries(const Model& model, const CharTable& chars, std::u16string_view normalized) -> std::vector<std::int32_t>;
  auto segment(const Model& model, const CharTable& chars, std::string_view utf8_text) -> Boundaries;
  ```

**L5: `kytea::KyteaBackend`** — a class satisfying the `tokenize`/`tokenize_boundaries` signature defined in Section 2. It holds a `Model` internally and is simply a thin adapter that calls the `scorer` functions.

**L6: `Segmenter`** — the design of Section 2 as-is. Currently `std::variant<KyteaBackend>` (MLP will be added once Section 4 is implemented).

**CLI layer** — `predict_command.cpp` is simply a thin loop: "parse arguments → `Segmenter::load` → read stdin line by line → call `tokenize` → output in the full-annotation format of Section 5.1." It holds no business logic at all.

### 8.3 MVP Scope Decision

`segmenter predict --model kytea-model.bin < input.txt > output.txt` (without `--boundaries-only`) originally implies output including tag (reading) estimation, but tag estimation requires a full-blown implementation including the word dictionary, subword language model, and tag models, which is a large scope (the `tagDictVector`/`tagUnkVector` of `FeatureLookup` from Section 3.2, and the computation formulas that, as per the Section 9 TODO, had not yet been investigated). **As the first implementation step, it is reasonable to first get `tokenize_boundaries` (segmentation only) working with `Model` able to skip the tag-related sections, and to temporarily make `predict`'s default output tag-free as well (equivalent output to `--boundaries-only`).** The layer structure of Section 8.2 is not affected by this decision (it only requires adding fields to `Model`), so adding tag estimation later does not cause significant rework.

### 8.4 Overall Repository Directory Structure

Section 8.1 is the module split within `include/`/`src/`. Here, the overall repository structure, including the build system, tests, and model assets, is decided.

```
cpp-segmentlib/
├── CMakeLists.txt                # top level. C++23, just bundles subdirectories
├── cmake/
│   └── CompilerWarnings.cmake    # common settings such as warning flags (optional)
├── include/segmentlib/           # public headers (structure from Section 8.1)
│   ├── types.h
│   ├── segmenter.h
│   ├── bytes/binary_reader.h
│   ├── unicode/utf8.h
│   └── kytea/{char_table,automaton,model,scorer,kytea_backend}.h
├── src/                          # implementation (structure from Section 8.1)
│   ├── CMakeLists.txt            # target definition for segmentlib (the library)
│   ├── segmenter.cpp
│   ├── kytea/*.cpp
│   └── cli/
│       ├── CMakeLists.txt        # target definition for segmenter (the executable)
│       ├── main.cpp
│       ├── predict_command.cpp
│       └── train_command.cpp
├── tests/
│   ├── CMakeLists.txt
│   ├── unit/                     # per-module tests (corresponding to each layer in Section 8.1)
│   │   ├── binary_reader_test.cpp
│   │   ├── utf8_test.cpp
│   │   ├── char_table_test.cpp
│   │   ├── automaton_test.cpp
│   │   └── scorer_test.cpp
│   └── golden/                   # tests that cross-check against real KyTea binary output (the policy stated at the end of Section 3.1)
│       ├── golden_test.cpp
│       └── fixtures/
│           ├── input.txt         # short test sentences
│           └── kytea_output.txt  # expected output generated by the real kytea command
├── models/                       # (gitignored) KyTea/Vaporetto models
│   └── kytea/
│       ├── jp-0.4.7-5.mod         # obtained via scripts/fetch_kytea_model.sh
│       └── jp-0.4.7-5.vaporetto.zst  # converted via bench/setup.sh
├── bench/                        # inference benchmarks (Section 9)
│   ├── setup.sh / run.sh / README.md
│   ├── bench_segment.cpp         # this library, in-process
│   ├── bench_kytea.cpp           # libkytea, in-process
│   └── {.vendor,corpus,results}/ # (gitignored)
├── scripts/
│   └── fetch_kytea_model.sh
├── docs/
│   └── design.ja.md
├── .gitignore
├── .clang-format
└── .clang-tidy
```

**Design decisions (finalized)**

- **The build system is CMake.** It is the de facto standard for the C++ ecosystem, and makes it easy to specify the compiler requirements needed to use C++23 features such as `std::expected`.
- **Zero dependency libraries within the scope of the KyTea backend MVP** (standard library only). Whether additional dependency libraries are needed once we move on to the custom MLP backend (Section 4) will be decided separately.
- **The test framework is `doctest`** (header-only). Fetched via CMake's `FetchContent`, and since it's just pulling in a single header, the build-time and dependency overhead is small.
- **`models/` is not tracked in git**: `models/` is added to `.gitignore`, and only a download script such as `scripts/fetch_kytea_model.sh` is kept in the repository (the retrieval procedure via a Wayback Machine archive is also scripted). This keeps the repository lightweight.
- **`golden/` tests use fixed data**: known pairs of input sentences and KyTea execution results are committed as fixed data under `tests/golden/fixtures/`. This means the CI environment does not need to build KyTea itself, making it self-contained. The expected values are created by using `models/kytea/jp-0.4.7-5.mod` and building/running the real KyTea locally (once created, it is reused thereafter as fixed data).

### 8.5 Build / Toolchain Requirements

- **A C++23-capable compiler is required**: uses `std::expected`, `std::byteswap`, `std::span`, and (for the CLI) `std::print`, etc.
- **macOS: builds fine with AppleClang (the stock Xcode toolchain)** (to be verified: sufficiently recent Xcode). Empirically confirmed (Xcode 26.2 / Apple Clang 17.0.0, `/usr/bin/clang++`): with a normal build that does not explicitly set a deployment target, all tests pass. Previously, Homebrew LLVM clang (`/opt/homebrew/opt/llvm/bin/clang++`) was thought to be required instead of AppleClang, but that requirement turned out to be incorrect, since it actually works without Homebrew LLVM.
  - **A point of caution is the availability gate for `std::print`/`std::format`**: the floating-point version of `std::to_chars` used internally is marked in Apple's libc++ as "available only on macOS 13.3 or later," so **explicitly setting the deployment target to macOS 13.2 or earlier results in a compile error** (`std::expected` itself is a pure template type independent of OS APIs, so it is not subject to this restriction). This has no effect under normal operation without an explicit deployment target.
  - When using Homebrew LLVM clang, the compiler can be specified explicitly: `-DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++`
- **Linux**: GCC 14 or later is assumed (not yet empirically verified).
- **Build procedure** (including AppleClang; the system's stock compiler is fine):
  ```
  cmake -S . -B build -G Ninja
  cmake --build build
  ctest --test-dir build --output-on-failure
  ```
- **Incompatibility between CMake 4.x and doctest 2.4.11**: since doctest 2.4.11's `cmake_minimum_required` falls below CMake 4's policy floor, this is worked around in `tests/CMakeLists.txt` by setting `CMAKE_POLICY_VERSION_MINIMUM=3.5` only around the `FetchContent_MakeAvailable(doctest)` call.

### 8.6 Implementation Status

- [x] **L1: `types.h`** (`Segment`/`Segments`/`Boundaries`/`ErrorCode`/`Error`)
- [x] **L1: `bytes::BinaryReader`** (LE integer read, NUL-terminated strings, line reading, bounds-checked. Errors are `ParseError` exceptions) + doctest unit tests
- [x] **L1: `unicode::utf8`** (single-code-point UTF-8 decode. Rejects overlong/surrogate/out-of-range) + doctest unit tests
- [x] CMake scaffolding (`segmentlib` static library + `segmentlib_tests`, doctest via FetchContent)
- [x] **L2: `kytea::CharTable`** (character interning table + character-type classification `classify` + normalization `normalize` + `encode`, Sections 3.1/3.2) + doctest unit tests. **Verified Config/character-map parsing against the real model `jp-0.4.7-5.mod` together with BinaryReader** (type markers are ids 1–6, 7,345 characters, confirmed `水/飲=K`, `を/む=H`, `。=O`)
  - The 4-byte UTF-8 KyTea `findType` bug is not followed; classification is done correctly (recorded in Section 3.1). Divergence from KyTea only occurs for rare CJK Extension B and beyond
- [x] **L2: `kytea::Automaton<Payload>`** (Aho-Corasick runtime representation, `Payload` + read callback separated from the entry layer. Header-only template) + doctest unit tests (gotos / failure fallback / propagated output / absent dictionary). **Verified against a real model**: `charDict` = 81,224 states / 75,377 payloads (each of size 6 = charWindow×2), `typeDict` = 233 states, `selfDict` = empty, wsModel has `numClasses=2` / `multiplier≈0.000109367`. `match` does not walk the failure chain but uses the already-propagated `output` directly (as per Section 3.2)
  - **`find_entry`** (equivalent to KyTea `Dictionary::findEntry`): walks only goto transitions, determining exact match via the terminal state's `is_branch` + `output` (retained in step A-2)
- [x] **L3: `kytea::Model`** (parses Config/wsModel FeatureLookup/word dictionary, advances the cursor to skip past tag/subword/LM-related sections. `load()` is the `std::expected` boundary) + doctest tests with a synthetic model. **Verified `Model::load` against the real model `jp-0.4.7-5.mod`**: multiplier=0.000109367, biases[0]=-193, charDict=81,224 states, word dictionary=850,724 entries / 2,080,192 states / numDicts=7. **The `dictVector` size of 84 = numDicts(7)×3×dictN(4) independently agrees**, confirming the parsing consistency of the whole file (load takes about 2.1 seconds; room for future optimization)
- [x] **Tag estimation, step A-1: retaining the model loader**(`tag_prediction_plan.ja.md` Section 2): changed the previously skipped tag model group to be retained. Holds `TagModel` (multiplier/num_weights/`TagFeatureLookup` = charDict/typeDict/selfDict/biases/tagDictVector/tagUnkVector), the global tag models (`global_tags_`/`global_mods_`, per level), and per-word tag information in the word dictionary entries (`WordEntry.tags`/`tag_in_dicts`/`tag_mods`). Added an id→UTF-8 reverse lookup `decode()` and `unicode::encode()` to `CharTable`, so tag-candidate strings are UTF-8-ized at load time. **Empirically confirmed with the real model** (`tag_prediction_plan.ja.md` Section 1.1): lev0 (POS) is a global single-choice (0 per-word entries, num_weights=21, selfDict non-empty); lev1 (reading) is a per-word single-choice (no global entries; 326,369 candidate words / 1,828 per-word models). No regression in the existing 34 WS tests
- [x] **Tag estimation, step A-2: known-word scorer + output formatting + CLI** (`tag_prediction_plan.ja.md` Sections 3–5): ported the known-word branch of `kytea.cpp:calculateTags` into `tag_scorer.cpp` (`addTagNgrams`/`addSelfWeights`/`addTagDictWeights`/`getDictionaryMatches`). Faithfully reproduced the dispatch (globalMods first → per-word model → unknown word), the loop keyed on `num_weights` (as empirically observed, this can mismatch the candidate array length), and top-candidate selection using a `std::sort` + `kyteaTagMore` replica. Added `find_entry` (equivalent to KyTea `Dictionary::findEntry`, strictly checking via retained `is_branch`) to `Automaton`. Output via `append_tagged_line` (surface escaped, tags not escaped) + `--notags`. Extended the golden tests; POS and reading for known words match byte-for-byte
- [x] **Tag estimation, step B: unknown-word reading estimation (subword dictionary + LM + beam search)** (`tag_prediction_plan.ja.md` Section 8): ported `Kytea::calculateUnknownTag`/`generateTagCandidates`/`KyteaLM::scoreSingle`. Added `subwordDict_` (`ProbSubwordEntry`: reading candidates kept as raw CharId sequences, not UTF-8-ized) and `numTags` instances of `KyteaLM` (`n`/`vocabSize` + `probs`/`fallbacks` as `unordered_map<u16string,double>`, excluding the `NEG_INFINITY=-999.0` sentinel) to the model loader (read order: wordDict → subwordDict → per-level LM, fixed by `Kytea::readModel`). `unkBeam=50`/`tagMax=3`/`defTag="UNK"` are adopted as KyTea's actual runtime defaults (not stored in the model). **Empirically confirmed: byte-for-byte agreement with KyTea's default (tagged) output across all verification — 20 golden sentences, 15 stress sentences, and 710,000 characters of Aozora Bunko text** (including unknown-word reading estimation and the `UNK` fallback). The only known discrepancy is the WS divergence caused by CJK Extension B characters (the intentional non-adoption of the `findType` bug, Section 10)
- [x] **L5: `kytea::KyteaBackend`** / **L6: `Segmenter`** (`std::variant` dispatch, `string_view` input + `std::expected`, Sections 2/7)
- [x] **CLI: `segmenter predict`** (`--model`/`--threads`, stdin→stdout filter; `segmenter train` is an unimplemented stub, Section 7). Default is tagged output (`append_tagged_line` = surface/POS/reading). `--notags`/`--boundaries-only` produce segmentation only without tags — both go through a **fast path that skips tag computation entirely** (`tokenize_boundaries_all` → `append_boundary_line`), agreeing byte-for-byte with `kytea -notags` (the golden tests also pin down that the boundary path equals the existing WS path). *Note: `--boundaries-only` was originally a dead flag that was only parsed and then ignored, until it was wired up (discovered and fixed after step B was completed)*
- [x] **Golden tests (byte-for-byte agreement with real KyTea output, Section 8.4)**: checked against `kytea -notags` using a fixed set of 15 fixture sentences. **Skipped when the model has not been fetched** (CI-resilient)
- [x] **Verified equivalence for unknown characters (outside the training vocabulary)**: confirmed, via source inspection of `string-util.cpp`/`kytea.cpp` plus byte-for-byte agreement on unknown emoji and symbols, that KyTea's dynamic ID assignment (`mapChar add=true`) and this library's flattening to `kNoChar` are completely equivalent in terms of segmentation output, since all WS features are automaton matches (keyed on training-time IDs `1..K`) (Sections 3.1, 10). Added a line of unknown characters (😀😱ℵℶℷ★☆♠🍣ⅠⅡⅢ) to the golden fixtures to pin the regression
- [x] **🎉 Achieved byte-for-byte complete agreement with KyTea**: `diff`-matched against `kytea -model jp-0.4.7-5.mod -notags` across 600 diverse lines of input (mixed half-width/full-width, alphanumerics, symbols, classical Japanese, emoticon-like runs). Even the output's symbol escaping (`showEscapedString`: prefixing space/`/`/`&`/`\` with `\`) is reproduced (e.g., `Hello \  World`, `2024 \/ 12`)

## 9. Benchmarks (Inference Speed)

`bench/`. Inference speed comparison against KyTea / Vaporetto.

### 9.1 Design Principle: "Before Speed, Guarantee the Computation Is the Same"

- **Same model**: KyTea's `jp-0.4.7-5` is converted to Vaporetto format (zstd) via `convert_kytea_model`, so all three tools run with the same weights.
- **The correctness gate comes first**: outputs are `diff`'d before timing. segmentlib is checked for byte agreement with KyTea; Vaporetto's discrepancy rate is reported.
- **Pure inference is measured in-process**: the model is loaded once, and only the tokenize loop is timed (load and I/O are excluded).
  - **Demonstrated that CLI wall-clock differences cannot measure inference**: an initial version tried a "small/large corpus, two-size differencing" approach on the CLI, but (a) I/O and output formatting dominated the marginal cost, and (b) Vaporetto's load-time variance (a few seconds) was far larger than inference (<0.1s), breaking the differencing approach. The in-process measurement (2.96M/s) and the CLI differencing result (0.60M/s) diverged by a factor of 5. → Switched to measuring each tool in-process with a dedicated harness (`bench_segment` = this library, `bench_kytea` = linked against libkytea, Vaporetto = its self-reported `Elapsed`).
- **Fixed conditions**: single-threaded, no tags, best-of-N after warmup. The metric is Unicode code points/second (matching Vaporetto's convention).
- **Corpus**: real works from Aozora Bunko (Soseki, Dazai, Akutagawa, Miyazawa; about 710,000 characters). Repeating the same line is avoided since caching would be unrealistically effective, so real text is used.

### 9.2 Latest Results (Apple M1 Pro, 710K Characters of Aozora Bunko, Best-of-5)

**Correctness gate**
- segmentlib vs. KyTea: **0 / 20,822 lines differ (100% agreement)**
- Vaporetto vs. KyTea: 92 / 20,822 lines differ (99.56% agreement. E.g. `し よう` ↔ `しよう`, "minor differences" that Vaporetto's own documentation acknowledges)

**Pure inference speed (in-process, load/I-O excluded, single-threaded)**

| Tool | Load | M chars/sec | vs. KyTea |
|---|---|---|---|
| **segmentlib** | **412 ms** | **6.21** | **≈5.3×** |
| KyTea | 964 ms | 1.17 | 1.00× |
| Vaporetto | a few seconds (daachorse construction) | 8.68 | ≈7.4× |

**segmentlib's parallel throughput (`tokenize_all`, best-of-5)**

| Thread count | M chars/sec | vs. single-thread | vs. KyTea |
|---|---|---|---|
| 1 | 5.65 | 1.00× | 4.8× |
| 2 | 12.91 | 2.28× | 11.0× |
| 4 | 25.91 | 4.59× | 22.1× |
| **8** | **44.98** | **7.96×** | **38.4×** |

(segmentlib's figures reflect optimization round 10 from Section 9.4. Before optimization: 2.84 M/s, 2.1×, load ~1450ms. Inference improved substantially through canonical double-array + direct root lookup + encode optimization; load is now, thanks to round 10's free-list-cap optimization, faster than KyTea. The 8-thread parallel figure is **about 5.2× Vaporetto's single-thread figure**. Absolute values fluctuate by ±10–20% depending on thermal state, so comparisons within the same batch are the stable metric.)

**Findings**
- segmentlib achieves **byte-for-byte agreement with KyTea while inferring more than 5x faster single-threaded, with faster loading too** (`bench_kytea`'s word-count `sink` matches exactly, confirming an identical segmentation). This is the effect of avoiding KyTea's implementation (feature-string hashing, the god class) in favor of a flat array + canonical double-array Aho-Corasick + direct root lookup.
- Vaporetto is still fastest single-threaded (a `daachorse` implementation with canonicalization, precomputed sums, and SIMD all in place), but its output is not strictly identical (0.44% mismatch). segmentlib **surpasses Vaporetto's single-thread speed via multithreading** (about 5.2× at 8 threads).

### 9.2.1 Speed Including Tag Estimation (Steps A-2/B, Reference Values)

The table above (Section 9.2) is a comparison with fixed conditions of **segmentation only** (equivalent to `-notags` / `tokenize_boundaries`). Tag estimation (POS + reading, the `tokenize` default), like KyTea's own `calculateTags`, uses an algorithm that **re-scans from the root for each word's window (the preceding/following char_n/type_n characters) every time** (structurally different from segmentation's single continuous scan over the whole sentence); the figures below are reference values for comparison against the segmentation-only path.

Inference speed (independent of load; load is a one-time cost identical regardless of inference path — see "Load time breakdown" below):

| Path | M chars/sec (single-thread) | vs. WS-only |
|---|---|---|
| WS only (`tokenize_boundaries`, same conditions as Section 9.2) | 5.6–6.8 | 1.00× |
| **With tags (`tokenize` default, POS + reading)** | **0.74–0.83** | **≈8× slower** |

(Apple M1 Pro, 710K characters of Aozora Bunko, best-of-7. The tagged figures are after the argmax-ization of Section 9.4 round 12. For reference: in external CLI measurements, KyTea itself also slows down roughly 3× going from `-notags` to its default (tagged) mode. Absolute values are environment-dependent, but our own measured tagged throughput (in-process 0.74–0.83 M/s) is faster than KyTea's tagged CLI (~0.12 M/s, including I/O). **An important measurement lesson**: right after implementing steps A-2/B, `build/` was left at `CMAKE_BUILD_TYPE=Debug` without noticing, producing a spurious "65–70x slower" figure. A `Release` reconfiguration alone recovered it from 0.10→0.66 M/s — that was a measurement mistake, not an implementation regression. **Always confirm the build configuration with `cmake -B build -DCMAKE_BUILD_TYPE=Release` before running benchmarks.**)

**Load-time breakdown (measured)**: model loading takes about 0.8–1.3 seconds (with large variance under thermal load; observations of 1.1–2.0 seconds under the same conditions also occurred). Timing each parsing stage showed that **building word_dict accounts for about 88% (~1150ms)**, while **the subword dictionary + two-level LM together account for only about 16ms (about 1%)**. The initial hypothesis, "the main cause of load-time growth is subword/LM," **was refuted by measurement** — the subword dictionary is only 3ms, and the LMs together only about 14ms. The main cost of loading is word_dict (850,000 words / 2 million states of double-array construction), which existed even in the segmentation-only era; the portion added by step A-1 (retaining per-word tag information + UTF-8-decoding tag strings for 850,000 words) is included in that figure.

**Remaining optimization opportunities not yet undertaken**:
- If aiming to shorten model load time, the target should be **word_dict parsing / double-array construction** (lazy-loading subword/LM only shaves off about 16ms = 1%, not worth it, as confirmed by measurement).
- Speeding up tag estimation itself: argmax-ization of top-candidate selection has already been done (Section 9.4 round 12, +10–20%). Payload flattening was found to have no effect via A/B measurement and was withdrawn (round 11 = a negative result). The main remaining target is `add_tag_ngrams` (about 31% in Release profiling), but a precomputation scheme that reduces the per-word window re-scan itself has not yet been examined. Currently about 8x slower than WS-only, but since it is still faster than KyTea's actual tagged measurement, priority is low.

### 9.4 Inference Optimization (Preserving Byte Agreement)

Profiling was used to measure the breakdown in detail, and optimization proceeded step by step while continuously verifying byte agreement with the `golden` test (starting at 2.84, reaching 3.37 M chars/sec, about +19%).

- **Breakdown (initial)**: `score_boundaries` is 89% (of which Aho-Corasick matching is ~120ms, weight accumulation ~90ms), `encode` is 11%.
- **Effective measure: O(1) direct table for root transitions** (`Automaton::build_root_index`). The root state is the hottest state, hit by fallback on nearly every character; replaced the binary search over its many gotos with direct `CharId` lookup → matching 120→90ms (+17%).
- **Ineffective but adopted measure**: changed `add_dictionary_scores`'s `on` array to "reuse via thread_local + clear only touched spots + add directly on match." Speed was flat (the allocator was already cheaply reusing same-size allocations, and the full scan was already vectorized), but **the code became simpler and thread safety was preserved**. Similarly applied thread_local buffer reuse in `encode_into`/`score_boundaries_into` (effective for the CLI path and multithreading).
- **Tried but not adopted: precomputed n-gram weight summation (the Vaporetto approach)**. Exploiting the fact that "all n-grams ending at the same position are added at the same offset," this precomputes the summed output weights of each Aho-Corasick state, element-wise, at load time into a single vector, so that at runtime it's a single fixed-width-8 addition per state (with margin padding to remove clipping branches). **Byte agreement held but speed dipped slightly**, so it was not adopted (`NgramScorer` was removed).
  - Reason: **KyTea's charDict has 0.93 outputs/state (81,224 states / 75,377 entries)**, i.e., states have at most about 1 output on average. Precomputed summation is "combining one into one" with no reduction, and only the overhead of fixed 8-width (only 6 needed in practice) padding allocation/copy was added.
  - **Insight**: this technique works for Vaporetto **only because it is paired with a double-array PMA (O(1) traversal)**. Once traversal becomes fast, accumulation becomes relatively heavy, and only then does precomputed summation + SIMD pay off. **Alone, and with a low outputs/state model like KyTea's, it does not help.** That is, "speeding up traversal (double-array)" must come first for precomputed summation to become meaningful afterward — an ordering dependency.
- **Effective measure, round 3: double-array-izing the Automaton** (O(1) transitions checking `check_[slot]==s` via `base_[s] + c`). Migrated all Aho-Corasick automata (charDict, typeDict, wordDict) from binary search over per-state heap-allocated `gotos` vectors, to three flat arrays `base_`/`check_`/`next_`. `failure` and `outputs` were also flattened into CSR form (`fail_`, `out_offset_`/`out_flat_`), eliminating pointer chasing into per-state vectors during traversal. **+75% over legacy under identical conditions (3.04→5.33 M chars/sec)**, maintaining byte agreement (zero diffs against golden and a 3000-line real corpus).
  - **Breakdown of the effect**: double-array-izing char/type gave only about +5%, while **double-array-izing wordDict (2.08 million states, a huge fan-out at the root) accounted for the majority (about +20% equivalent)**. Under legacy, binary search over the huge gotos list plus pointer chasing was heavy, so O(1) transitions plus cache locality gave the biggest win there.
  - **Packing density directly affects inference speed**: if the construction-time packing (total slots / total states) becomes sparse, cache efficiency drops and inference slows. Measured: pushing the cap (below) toward sparser dropped throughput to 4.2 M/s, while dense packing gave 5.3 M/s.
  - **Construction-cost (load-time) trade-off**: double-array construction is about finding a `base` that packs each state's children without collision, and a naive unbounded search causes **a tiny number of extremely-high-fan-out nodes** (like the root) to scan a free list millions of entries long, becoming super-linear (~950ms for wordDict alone). As a countermeasure, **the free-list length is capped by `kFreeListCap`** (entries beyond the cap have their oldest slot abandoned as `kClosed`, limiting the search window to near the write frontier). `262144` is **the smallest cap that reaches full packing (identical to the unbounded 2.187M slots) for KyTea's largest automaton**; larger automata degrade gracefully toward sparser packing to keep construction time bounded. As a result, load went from 150→~1150ms (about the same as KyTea's 800ms).
  - **Packing strategies tried but not adopted**: descending-fan-out placement achieves near-zero-waste full packing, but placing large nodes first maximizes the array early and blows up the free list → **construction exploded to 37 seconds**. Reserving space at the tail just for high-fan-out nodes wastes the entire CharId width for that node, **inflating slot count by more than 10x**. Both fail without a cap; file order + cap + gap-filling was best.
- **Tried but not adopted: unconditional n-gram weight addition + margin padding (round 4's initial idea)**. Since it was empirically confirmed that all payloads uniformly have width 6 (`window*2`), a version was implemented that pads `window` on both ends and unconditionally adds each match at a fixed width of 6 (removing clipping branches to enable vectorization). **Byte agreement held but speed was flat** (4.84→4.83 M/s), so it was not adopted.
  - **Reason (confirmed via profiling)**: `score` is 86% of the total, but within that, **traversal dominates and addition is essentially free**. Measured: `charDict.match` + addition (40.5ms) and `charDict.match` alone (41.4ms) were the same value = **addition cost was below the measurement threshold**. The traversal breakdown was char 22ms, type 8ms, wordDict 29ms (59ms total for traversal). As with round 2, this is the same underlying negative result — **for a KyTea model, accumulation is not the bottleneck; traversal is.**
- **Effective measure, round 4: merging the double-array's `check` and `next` into a single cell (8 bytes)**. With separate arrays, one transition touches two cache lines (`check_[slot]` and `next_[slot]`), but consolidating into a single array of `struct Cell{check,next}` (`cells_`) makes **one transition = one cache line**. This directly targets the traversal-bound reality, giving **+10% (4.84→5.33 M/s)**, **4.0× vs. KyTea**, byte agreement maintained.
- **Effective measure, round 5: canonical double-array-ization**. Renamed state IDs to slot IDs and dropped the `next` array (the slot ID is itself the transition target), consolidating `base_` and `cells_` into a single `unit_[]{base,check}`. The table shrank (wordDict: 25.8→17.5MB), and locality improved because `unit_[cur]`'s cell is shared between "the check-validation read on arrival" and "the base read for the next transition." **+17% (5.33→6.26 M/s)**, **≈4.3× vs. KyTea**, byte agreement maintained.
  - **Placement order is decisive (DFS preorder)**: canonicalization requires "the child's slot = new ID" to be settled when the parent is placed, so it must be processed in parent-to-child topological order. **BFS order is worst**: siblings scatter across the array, causing collisions to skyrocket (wordDict's `base` search scan went 96M→490M, load 5.7s). **DFS preorder** places subtrees contiguously, localizing the free-list frontier, giving 113M scans, **packing nearly 1:1 (2.08M states → 2.089M slots)**, and load converging to ~1450ms. A lesson that in double-array construction, "order" governs both cost and density.
- **Tried but not adopted: 24-bit packing of cells (8→6 bytes, round 6)**. Since state count < 2^21, a version was implemented packing `base`/`check` into 24 bits each, making `unit_` 6 bytes per cell, shrinking wordDict's table from 16.7→12.5MB (with padding on the trailing cell and 3-byte safe reads for boundary handling, assuming little-endian). **Byte agreement held but it was not adopted**:
  - **Load got worse** (under the same thermal conditions, 8B ~1450ms → 6B ~2083ms). Construction became heavier due to byte writes for `set_base`/`set_check` and the 3-byte safe reads in the `fit` search.
  - **Inference did not improve either** (in a cold comparison, 6B 5.1 < 8B 6.3). The cost of unaligned access at a 6-byte stride, plus masking, offset the gains from table shrinkage. Also, M1 Pro's **large SLC (~24MB) already held the 16.7MB 8B table**, so shrinking to 12.5MB barely reduced the miss rate (L2's 12MB isn't enough for even 12.5MB).
  - Lesson: **alignment beats table size** (on this hardware, for this model). An `{i32,i32}` landing on an 8-byte boundary is faster.
- **Effective measure, round 7: eliminating all hashing from input encoding (`encode`)**. `encode` was hitting `unordered_map` **three times per character** — for `normalize`, `id_of(character)`, and `id_of(type marker)` — accounting for about 14% overall (18ms) in profiling. Exploiting the fact that Japanese is almost entirely BMP (<U+10000), (1) `normalize` was turned into a model-independent **static BMP direct-lookup table**, (2) character-ID resolution into a **BMP direct-lookup array `bmp_ids_` (128KB)** with hash fallback only for astral characters, and (3) the 6 type-marker IDs were **precomputed** at construction time (`type_ids_[classify]`), eliminating hashing entirely. **encode 18→7.4ms (about 2.4×)**, overall speedup vs. KyTea 4.3→about 4.9×, byte agreement maintained. A reduction in pure encoding-side overhead, not the traversal (automaton) side.
- **Effective measure, round 8: direct-lookup index for root transitions**. In canonical form, the root is unit 0, and `step(0,c)` reads a **scattered slot** in the 17.5MB `unit_` array. But since most text positions do not start a dictionary word, `cur` tends to stay at root (or fall back to root), making this scattered read the most frequent access. Replaced only the root's children with a **CharId direct-lookup small array `root_next_` (~32KB, resident in L1/L2)**, equivalent to `step(0,c)` (constructed so that `root_next_[c]==step(0,c)` is guaranteed). Only when `cur==0` in the match loop does it take this fast path (other states' step is unchanged). **wordDict traversal 29→13.15ms (about 2.2×)**, overall 6.7 M/s, **4.7× vs. KyTea**, byte agreement maintained.
- **Effective measure, round 9: line-level multithreading (batch API)**. Since the inference path is already reentrant (Model is immutable, scratch is `thread_local`), `Segmenter::tokenize_all(span<string_view>, threads)` was added. **Chunks of 64 lines are dynamically assigned via an atomic cursor** (balances load against line-length variance; the atomic is amortized per chunk), and each thread writes to disjoint result slots, requiring no lock. The calling thread also participates as a worker. On 20,822 lines of Aozora Bunko on the M1 Pro: **1→7.4 / 2→14.5(1.96×) / 4→26.6(3.59×) / 8→41.4 M chars/sec (5.60×)**. **41.4 M/s is about 4× Vaporetto's single-thread figure (~10.4)**, and about 32× KyTea's single-thread figure. The parallel result agrees byte-for-byte with sequential `tokenize` (tested at 1/2/4/8 threads). The diminishing returns at higher thread counts come from mixed efficiency cores and memory bandwidth (the 17.5MB table being shared).
- **Effective measure, round 10: optimizing the free-list cap under the DFS assumption (load-time reduction)**. Initially, `daachorse`-style block-closing was planned to shrink load time, but it turned out **simply lowering the cap constant achieved the same goal**. Reason: under DFS preorder placement, *useful* free slots are always near the write frontier, and setting `kFreeListCap` too large accumulates "a long tail of old free slots from a subtree that has already been passed and will never be revisited," which `base` search wastefully sifts through. Lowering the cap from `262144→8192` brought wordDict construction from **load 2224→373ms (about 6×)**, without slowing inference (same-batch measurement 5.54→6.9 M/s). **Achieved the goal without implementing block-close.**
  - **Why inference doesn't slow down**: a small cap sparsifies wordDict (units +25%, 20.8MB), but **thanks to root direct-lookup (round 8), wordDict's `unit_` is already cold** (most accesses hit the 32KB `root_next_`). The sparsified region is deep and rarely accessed, so it isn't affected. Meanwhile char/type are small enough not to fill the cap window and **remain essentially tight** (the hot char scan retains its density). I.e., round 8's side effect made round 10 possible — an ordering dependency.
- **Tried but not adopted, round 11: flattening `Automaton<Payload>`'s payload** (a negative result). Right after completing tag-estimation steps A-2/B, seeing that tagged `tokenize` was 65–70x slower led to the hypothesis that "`addTagNgrams` (POS lev0 fires for every word) re-scans a small window per word, and `Payload=FeatVec` (`vector<vector<int16_t>>`)'s individual heap allocations cause scattered references, i.e., cache misses." An optimization was implemented that, when `Payload` is `std::vector<T>`, flattens all payloads into a contiguous buffer plus offset array and returns a `span`. **However, two problems came to light and it was withdrawn**:
  1. **The real cause was a measurement mistake**: most of the 65–70x came from `build/` having been left at `CMAKE_BUILD_TYPE=Debug`. **A `Release` reconfiguration alone recovered it from 0.10→0.42–0.66 M/s** (load also went from 5.6s→1.9s). Not an implementation regression, but an overlooked build configuration.
  2. **The flattening had no measurable effect**: running 6 alternating rounds of back-to-back, best-of-7 A/B comparison in `Release` between with/without flattening on the same machine, the distributions completely overlapped (no-flat best 0.805, median about 0.667; flat best 0.732, median about 0.678 — the variant difference was smaller than the ±15% thermal noise). The hypothesized "scattered references from individual heap allocations" was not, in practice, the bottleneck, because read-only payloads end up nearly adjacent from sequential mallocs at load time, and the vector-header array itself is already contiguous, so the extra single level of indirection was well predicted by the branch predictor.
  - **Lesson**: the moderate complexity of template duplication via `if constexpr`, branching return types (`span` vs. reference), and the latent pitfall of "empty span = not-found," was introduced before A/B measurement. **Should have measured correctly in `Release` first and confirmed the effect before adding complexity.** Withdrawn, reverting to the plain `vector<Payload>`. To truly speed up tag estimation, the fundamental fix is a precomputation scheme that reduces the per-word re-scan itself, not the storage layout (Section 9.2.1 / Section 10 TODO).
- **Effective measure, round 12: changing tag estimation's top-candidate selection from full-sort-every-word to linear argmax** (tag estimation +10–20%). `predict_word_tags` built a `(index, score×multiplier)` pair vector for each word/level's candidates (e.g. 21 POS classes) and used `std::sort` to sort descending, taking only the top. Since only the maximum element is actually needed, `std::sort` + pair-vector construction + double multiplication were removed in favor of a single simple linear maximum scan over integer scores (since `multiplier>0`, integer-score order equals confidence order, and tie structure also matches). In Release profiling, `std::__introsort` accounted for about 7% of the total. **A byte-agreement caveat**: `std::sort`'s tie-breaking order is unspecified, and prior agreement with KyTea held only because the same libc++ `std::sort` was being run. Linear argmax (the maximum with the smallest index) could theoretically branch differently on ties, but byte-for-byte agreement was confirmed on 20,822 lines of Aozora Bunko plus golden (no ties occurred in the real data). **Back-to-back A/B (best-of-7 × 4 rounds) had argmax win every round** (sort 0.63–0.71 vs. argmax 0.74–0.81 M/s, non-overlapping distributions — a real effect, in contrast to round 11). The code also got shorter (the `ranked` scratch buffer was removed).

### 9.3 Harness

- `bench/setup.sh`: fetches and cleans Aozora Bunko, builds Vaporetto, converts the model (one-time only).
- `bench/run.sh`: correctness gate → in-process inference measurement → table output. Saved to `bench/results/`.
- `bench/bench_segment.cpp` (this library) / `bench/bench_kytea.cpp` (libkytea) / Vaporetto uses `predict`'s `Elapsed`.
- `bench/.vendor/`, `bench/corpus/`, `bench/results/` are not tracked in git.

## 10. Open Issues / TODO

- [x] Model file section structure and types (finalized in Section 3.2: header → Config → wsModel → tag models → dictionary → subword dictionary → LM)
- [x] Corpus specification (full/partial annotation, finalized in Section 5)
- [x] Overall architecture binding the multiple backends together (Section 2: `std::variant`-based, `Segmenter` a thin dispatcher)
- [x] The exact byte-layout specification of the `DictionaryState` (Aho-Corasick automaton) binary layer (Section 3.2: `failure`/`gotos`/`output`/`isBranch`, and even the per-Entry-type differences of `writeEntry<Entry>` finalized)
- [x] The word-segmentation inference algorithm (Section 3.3: restored the score formula and boundary-decision threshold (`score > 0`) of `calculateWS` from `kytea.cpp`)
- [x] The correspondence between `KyteaChar` and characters (Section 3.2: not a fixed Unicode code point but a per-model character interning table; ID `0` is a reserved sentinel, real characters start at `1` in training-time first-occurrence order)
- [x] How `mapChar` handles unknown characters (not in the training vocabulary) at inference time: **resolved**. KyTea dynamically assigns a new ID (`charTypes_.size()`) to unknown characters via `mapString`→`mapChar(add=true)`, but since all WS features are automaton matches (training-time IDs `1..K`), segmentation output is **completely equivalent** between that and this library's flattening of unknown characters to `kNoChar=0` (proved in Section 3.1 plus empirical measurement; golden fixtures include a line of unknown characters, pinning byte agreement). The CJK Extension B Kanji discrepancy is a separate axis, the (intentional) `findType` bug.
- [x] Whether to primarily support the text format (`T`) or the binary format (`B`): **resolved — only binary (`B`) is supported, no support needed for text.** Distributed models are binary; the text format is never encountered unless one runs `train-kytea -modelformat text` oneself. Text-format support is not implemented.
- [x] Whether non-quantized models (`FeatVal=double`, version `"0.4.0NQ"`) need support: **resolved — not needed.** Distributed models are quantized builds (`int16_t`). Non-quantized models are only produced by a `DISABLE_QUANTIZE` build trained oneself, and are never encountered in real usage. When an `"0.4.0NQ"` header is detected, it is treated as an error (to prevent silent misreads).
- [x] Subword dictionary (`Dictionary<ProbTagEntry>`), language model (`KyteaLM`), and the tag-estimation score formulas (`addTagNgrams`/`addTagDictWeights`/`addSelfWeights`/`scoreSingle`): **implemented and resolved in steps A-2/B** (Section 8.6). Known words: `addTagNgrams` and related logic ported into `tag_scorer.cpp`; unknown-word readings: `generateTagCandidates`'s subword-lattice DP + beam search + `KyteaLM::scoreSingle`'s n-gram backoff faithfully ported. Byte-for-byte agreement with KyTea's default (tagged) output confirmed on the golden set and 710K characters of Aozora Bunko
- [x] **Decided not to build a Vaporetto-compatible backend.** Vaporetto's model binary format (zstd wrapper + `MODEL_MAGIC` + the flat-list structure of the bincode2 standard configuration) was investigated early on and the wire format confirmed empirically, but the backend itself was never implemented; the two-backend set (KyTea-compatible + custom MLP, Section 1) is final. Vaporetto remains in use only as an external benchmark comparison point (Section 9)
- [ ] Custom MLP backend's network architecture and training method (Section 4, to be designed once requirements are settled to some degree)
- [x] Speedup method for feature extraction (confirmed that KyTea itself already uses Aho-Corasick; implemented Double-Array-ization and confirmed it is highly effective — Section 9.4 rounds 3–10)
- [x] CLI subcommand structure (Section 7: `segmenter predict`/`segmenter train`. Option shape finalized; the training engine's internals remain unimplemented)
- [x] Wired up `--boundaries-only` (discovered and fixed after step B was complete that it was a dead flag, only parsed and ignored). Along with `--notags`, it now goes through the fast path that skips tag computation (`tokenize_boundaries_all`→`append_boundary_line`), agreeing byte-for-byte with `kytea -notags`. Golden tests also pin the boundary path = existing WS path agreement (Section 8.6, CLI item)
- [~] Multiple-candidate + confidence output (KyTea `-out conf`/`-tagmax`): **not implemented, as unnecessary for the normal use case (single best-scoring answer)** (both KyTea and MeCab default to single-best; N-best/marginal probabilities are niche use cases). Plan §5 originally mis-cited `-alltags`, but that option does not actually exist. Implementing this would require reproducing the margin computation, retaining all candidates, and float formatting, all of which were omitted in step A-2. Deferred until needed (`tag_prediction_plan` §5)
- [~] Partial-annotation input + hard constraints (equivalent to `-wsconst`, §238/§6.2): constrained parsing not implemented. The distributed JP model's `wsConstraint` is normally empty and has no effect on byte agreement for the default output. Deferred until needed
- [ ] Training feature (whether to self-implement the KyTea-compatible training engine, or interface with external LIBLINEAR. The `segmenter train` options themselves are already finalized in Section 7.2)
- [ ] The actual scope of `--encode` support (whether to reject non-UTF-8)
- [ ] Whether a pmr-based API is needed, to be decided after benchmarking
- [x] Inference benchmark against KyTea/Vaporetto (Section 9): correctness gate + in-process measurement. Latest figures (Section 9.2): segmentlib is 5.3× KyTea's single-thread inference speed, with faster loading too, and 38.4× at 8 threads
- [x] Inference speedup round 1 (Section 9.4): O(1) direct table for root transitions + buffer reuse, 2.84→3.37 M/s (+19%), byte agreement maintained
- [~] Inference speedup round 2: precomputed n-gram weight summation (the Vaporetto approach) was implemented, but was not adopted since the KyTea model, at 0.93 outputs/state, does not benefit (Section 9.4, negative result recorded)
- [x] Inference speedup round 3: **double-array-izing** the Automaton (O(1) transitions via `base_[s]+c`, plus CSR flattening of fail/outputs). All automata double-array-ized, **+75% over legacy (3.04→5.33 M/s)**, byte agreement maintained. wordDict (2.08 million states) contributed the most. Construction linearized via a free-list cap (`262144`); load ~1150ms (Section 9.4)
- [~] Inference speedup round 4: unconditional n-gram weight addition + SIMD (re-evaluating precomputed summation) was found via profiling to be a case where "addition is free, traversal is the bottleneck," and was not adopted (the same underlying negative result as round 2). Instead, **merging `check`/`next` into a single cell** (1 transition = 1 cache line) gave **+10% (4.84→5.33 M/s, 4.0× vs. KyTea)**, byte agreement maintained (Section 9.4)
- [x] Inference speedup round 5: **canonical double-array-ization** (state ID = slot ID, `next` dropped, consolidated into a single `unit_[]{base,check}`). **+17% (5.33→6.26 M/s, about 4.3× vs. KyTea)**, byte agreement maintained. DFS preorder placement brought packing to nearly 1:1 (BFS caused explosive collisions and was infeasible). Table also shrank from 25.8→17.5MB (Section 9.4)
- [~] Inference speedup round 6: 24-bit packing of `unit_` cells (8→6 bytes) was implemented but not adopted, due to unaligned access and increased construction cost (no inference improvement, worse load). A negative result: **alignment beats table size** (Section 9.4)
- [x] Inference speedup round 7: **eliminating all hashing from encode** (normalize/character ID/type ID moved to BMP direct-lookup/precomputation). encode 18→7.4ms (about 2.4×), about 4.9× vs. KyTea, byte agreement maintained (Section 9.4)
- [x] Inference speedup round 8: **direct-lookup index for root transitions** (`root_next_`). wordDict traversal 29→13ms (about 2.2×), 4.7× vs. KyTea, byte agreement maintained (Section 9.4)
- [x] Inference speedup round 9: **line-level multithreading** (`Segmenter::tokenize_all`, atomic dynamic assignment). **41.4 M chars/sec (5.6×, about 4× Vaporetto's single-thread figure)** with 8 threads on M1 Pro, byte agreement with sequential execution (Section 9.4)
- [x] Multithreaded the `segmenter predict` CLI via `tokenize_all` (read all lines per block → parallelize → ordered output, `--threads N` supported, bounded memory). Byte agreement maintained
- [x] Load-time reduction round 10: free-list cap `262144→8192` (optimal under the DFS assumption). wordDict construction load 2224→373ms (about 6×), inference and byte agreement maintained. Goal achieved without implementing block-close (Section 9.4)
- [x] Double-array construction speedup: block-close not implemented, but simply adjusting the free-list cap under the DFS assumption from `262144→8192` achieved a load reduction equivalent to 2224→373ms (Section 9.4, round 10). If tight packing becomes necessary, room remains to implement block-close
- [x] Multithreaded benchmark variant: added a parallel benchmark for `Segmenter::tokenize_all` (Sections 9.2/9.4: 41 M/s at 8 threads, about 5× Vaporetto's single-thread figure)
- [x] Re-measurement on a larger, more genre-diverse corpus: **both accuracy and speed done** (Section 4.8). Added UD_Japanese-PUD (out-of-domain, different genre, 1000 sentences / 27,788 boundaries) and applied the GSD-trained models without retraining. Accuracy: no degradation for either KyTea or MLP (both score higher on PUD), gap stable at 0.6–0.7pt (GSD-only 0.74pt → PUD 0.62pt → combined 0.67pt), confirming the initial 0.7pt as representative across 3× the data and 2 genres. Speed: 2.39x on GSD train / 2.21x on PUD test (vs. real KyTea), both within the previously recorded 2.2–2.4x range, confirming segmentation speed is genre-insensitive. Fetch via `scripts/fetch_ud_pud_corpus.sh`. The only thing left unmeasured is a distributed-model-scale corpus (BCCWJ), which is unobtainable (Section 4.8 opening)
- [~] `Automaton<Payload>` payload-flattening round 11: an optimization to make `FeatVec` payloads contiguous was implemented, but A/B measurement in `Release` (back-to-back, best-of-7, 6 rounds) showed no effect and it was withdrawn (negative result, Section 9.4 round 11). The real cause was an overlooked build configuration (`Debug`); a `Release` reconfiguration alone recovered it
- [x] Empirically measured the breakdown of model load time including tag estimation: **word_dict construction is about 88% (~1150ms); the subword dictionary + two-level LM together are only about 16ms (about 1%)**. The initial hypothesis that "subword/LM was the main cause" was refuted. The target for load-time reduction is word_dict, not subword/LM (lazy loading would only shave ~16ms, not worth it) (Section 9.2.1)
- [x] Changed tag estimation's top-candidate selection from full-sort-every-word to linear argmax (round 12, +10–20%, byte agreement maintained, Section 9.4)
- [ ] Speeding up `add_tag_ngrams` (about 31% in Release profiling — the largest cost in tag estimation). **Broken down via measurement**: an A/B reducing the accumulation loop to a single width consistently sped up overall `tokenize` by +9–25% (best 0.80→0.95 M/s), showing that **the 21-wide accumulation loop accounts for about 18% overall = accumulation-bound** (the initial assumption of "traversal-bound" was wrong here). Unlike WS's 6-wide case, the 21-wide int16→int32 extended-addition loop matters. However, **no elegant fix has been found**: `__restrict` pointer-ization had no effect in A/B (the compiler does not auto-vectorize the extended addition with a runtime bound of nw=21; best base 0.891 ≥ vec 0.855). Capturing that 18% would require explicit NEON intrinsics (including handling the 21-element remainder), which is inelegant, and it is not even verified that pure computational SIMD would capture the full 18% (the single-width experiment also included a reduction of 20 elements' worth of loads). Low priority (currently about 8× slower than WS-only, but still faster than KyTea's actual tagged measurement). Separately, a precomputation scheme that reduces the per-word re-scan itself (scanning the whole sentence once and referencing each word's window in O(1)) has also not yet been examined (Section 9.2.1)
