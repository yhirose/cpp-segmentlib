# cpp-segmentlib

A C++ Japanese word segmentation library using the pointwise prediction approach, as in KyTea / Vaporetto.
Each character boundary is classified independently as a binary decision (split / don't split). Unlike lattice + Viterbi minimum-cost approaches (e.g. MeCab), it requires no dictionary cost design or dynamic programming.

The public API is a single interface, backed internally by a set of interchangeable backends:

- **KyTea-compatible backend** — loads KyTea's trained models as-is and performs inference with the same feature extraction and linear SVM classifier. Byte-for-byte agreement with KyTea's own output is verified.
- **Custom MLP backend** — uses a quantized (int16) MLP with a SIMD-accelerated scorer instead of a linear SVM as the classifier. On UD_Japanese-GSD (no dictionary) it runs ~3.7x faster than KyTea on the same CPU thread, and trails a KyTea model trained on the same data by ~0.7-0.9pt of boundary F1.

The corpus format is always KyTea's corpus format (full/partial annotation), used consistently across both backends for training and accuracy comparison.

Segmentation speed is benchmarked against both KyTea and [Vaporetto](https://github.com/daac-tools/vaporetto) (used only as a comparison target, not as a backend); see [bench/README.md](bench/README.md).

Detailed design document (English / Japanese):

- Overall design: [docs/design.md](docs/design.md) / [docs/design.ja.md](docs/design.ja.md)

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
