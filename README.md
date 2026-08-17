# cpp-segmentlib

A C++ Japanese word segmentation library using the pointwise prediction approach, as in KyTea / Vaporetto.
Each character boundary is classified independently as a binary decision (split / don't split). Unlike lattice + Viterbi minimum-cost approaches (e.g. MeCab), it requires no dictionary cost design or dynamic programming.

The public API is a single interface, backed internally by a set of interchangeable backends:

- **KyTea-compatible backend** — loads KyTea's trained models as-is and performs inference with the same feature extraction and linear SVM classifier. Byte-for-byte agreement with KyTea's own output is verified.
- **Custom MLP backend** — uses a quantized (int16) MLP with a SIMD-accelerated scorer instead of a linear SVM as the classifier. On UD_Japanese-GSD (no dictionary) it runs ~3.7x faster than KyTea on the same CPU thread, and trails a KyTea model trained on the same data by ~0.7-0.9pt of boundary F1.

The corpus format is always KyTea's corpus format (full/partial annotation), used consistently across both backends for training and accuracy comparison.

Segmentation speed is benchmarked against both KyTea and [Vaporetto](https://github.com/daac-tools/vaporetto) (used only as a comparison target, not as a backend); see [bench/README.md](bench/README.md).

Tests are doctest-based and run with `just test` (or `ctest --test-dir build`); `just bench` measures the reference model in-process. The `justfile` is the shortest description of how the project is built, tested and measured.

Detailed design document (English / Japanese):

- Overall design: [docs/design.md](docs/design.md) / [docs/design.ja.md](docs/design.ja.md)

## Getting started

No model is distributed with the library, so the first step is to build one.
The whole path below takes about a minute of compute plus a 140MB download, on
a checkout with CMake 3.24+, a C++23 compiler and Python 3. (C++23 is what the
trainer, the CLI and the test suite need; segmentation itself is header-only and
builds as C++17 — see "Using the library" below.)

With [just](https://github.com/casey/just) the same path is three commands,
and `just` on its own lists everything else (tests, benchmarks, evaluation):

```sh
just setup     # fetch the corpora and the UniDic dictionary
just model     # build, then train the reference model
echo '日本語の文を分割します。' | just predict
```

The rest of this section is what those recipes run, for anyone without `just`.

### Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSEGMENTLIB_BUILD_TRAINING=ON
cmake --build build --parallel
```

`SEGMENTLIB_BUILD_TRAINING=ON` is what adds the trainer, and it is the only
part that needs BLAS (Accelerate on macOS, OpenBLAS elsewhere; override with
`-DSEGMENTLIB_BLAS=`). Leave it off to build inference alone, which needs
nothing beyond the standard library. The `segmenter` CLI lands in
`build/src/cli/segmenter`.

### Train the reference model

```sh
scripts/fetch_ud_gsd_corpus.sh    # UD_Japanese-GSD, converted to KyTea format
scripts/fetch_unidic_dict.sh      # UniDic 2.1.2, converted to a word list

build/src/cli/segmenter train --backend mlp \
    --corpus corpus/ud-gsd/train.kytea.txt \
    --dev-corpus corpus/ud-gsd/dev.kytea.txt \
    --dict corpus/ud-gsd/dict_unidic.txt \
    --model-out corpus/ud-gsd/mlp.mod
```

That is 99.1% boundary F1 on GSD test and 99.3% on UD_Japanese-PUD, in a 2.1MB
model (Section 4.8 of the design document has the comparison against KyTea and
Vaporetto, and against other dictionaries). Training takes about 40 seconds and
stops itself when the dev score stops improving.

The dictionary is optional. Dropping `--dict` gives a 0.6MB model that loads
and runs faster but scores about a point lower, and carries no third-party
licence (see below). `scripts/extract_dict.py` is a middle option: it builds a
dictionary out of the training corpus itself, needing no external data.

### Segment text

`predict` reads UTF-8 text on stdin, one sentence per line, and writes
space-separated words:

```sh
$ echo '日本語の文を分割します。' | build/src/cli/segmenter predict --model corpus/ud-gsd/mlp.mod
日本 語 の 文 を 分割 し ます 。
```

The backend is chosen from the file itself, so the same command runs a KyTea
model (`scripts/fetch_kytea_model.sh` fetches one) with no other change. Words
containing a space, `/`, `&` or `\` are escaped with a backslash, as KyTea
does. `--threads N` segments the input in parallel.

In C++ the equivalent is `Segmenter::load(path)` followed by `tokenize(text)`;
see Section 6 of the design document.

## Using the library

Segmentation is **header-only and C++17**. There is no library to build or
link: put `include/` and `third_party/` on the include path (the
`segmentlib` CMake target carries both), include `<segmentlib/segmenter.h>`
and go.

```cmake
add_subdirectory(path/to/cpp-segmentlib)   # or FetchContent
target_link_libraries(your_target PRIVATE segmentlib)
```

```cpp
#include <segmentlib/segmenter.h>

auto seg = segmentlib::Segmenter::load("model.mod");
if (!seg) { /* seg.error().message */ }

auto words = seg->tokenize("日本語の文を分割します。");
if (!words) { /* words.error().message */ }
for (auto [start, end] : *words) { /* the word is text[start, end) */ }
```

The later-standard vocabulary types the code would otherwise use —
`std::expected` (C++23), `std::span` and `std::endian`/`std::byteswap`
(C++20) — have minimal stand-ins under `include/segmentlib/support/`, which is
what keeps the headers reachable from a C++17 project. `Expected<T,E>` behaves
like `std::expected` for the operations used here: `operator bool`, `operator*`
/ `operator->`, and `error()`. Consumers on a later standard are unaffected;
C++17 is a floor, not a ceiling.

The trainer, the CLI and the test suite still require C++23.

## License

MIT License. See [LICENSE](LICENSE).

**Bundled third-party code.** `third_party/cpp-fstlib` is
[cpp-fstlib](https://github.com/yhirose/cpp-fstlib) (MIT), the FST behind the
dictionary matcher; its license is at
[third_party/cpp-fstlib/LICENSE](third_party/cpp-fstlib/LICENSE). Nothing else
is vendored, and the test framework (doctest) is fetched at build time and is
not part of the library.

**Models trained with a dictionary carry that dictionary's license.** The word
list is compiled into the model file, so a model is a derivative work of the
dictionary it was trained with, and redistributing it means honouring those
terms. This matters for the reference model, which `scripts/fetch_unidic_dict.sh`
builds from **unidic-mecab 2.1.2** (UniDic Consortium), triple-licensed GPL /
LGPL / **BSD**: the BSD option permits redistribution, including commercially,
provided the copyright notice and disclaimer travel with it. The fetch script
keeps UniDic's own `COPYING` and `BSD` files next to the downloaded lexicon so
the text you have to ship is at hand. A model trained without `--dict` carries
no such obligation.
