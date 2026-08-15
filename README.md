# cpp-segmentlib

A C++ Japanese word segmentation library using the pointwise prediction approach, as in KyTea / Vaporetto.
Each character boundary is classified independently as a binary decision (split / don't split). Unlike lattice + Viterbi minimum-cost approaches (e.g. MeCab), it requires no dictionary cost design or dynamic programming.

The public API is a single interface, backed internally by a set of interchangeable backends:

- **KyTea-compatible backend** — loads KyTea's trained models as-is and performs inference with the same feature extraction and linear SVM classifier. Byte-for-byte agreement with KyTea's own output is verified.
- **Custom MLP backend** — uses a quantized (int16) MLP with a SIMD-accelerated scorer instead of a linear SVM as the classifier. On UD_Japanese-GSD (no dictionary) it runs ~3.7x faster than KyTea on the same CPU thread, and trails a KyTea model trained on the same data by ~0.7-1.0pt of boundary F1.

The corpus format is always KyTea's corpus format (full/partial annotation), used consistently across both backends for training and accuracy comparison.

Segmentation speed is benchmarked against both KyTea and [Vaporetto](https://github.com/daac-tools/vaporetto) (used only as a comparison target, not as a backend); see [bench/README.md](bench/README.md).

Detailed design document (English / Japanese):

- Overall design: [docs/design.md](docs/design.md) / [docs/design.ja.md](docs/design.ja.md)

## License

MIT License. See [LICENSE](LICENSE).
