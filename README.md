# cpp-segmentlib

A C++ Japanese word segmentation library using the pointwise prediction approach, as in KyTea / Vaporetto.
Each character boundary is classified independently as a binary decision (split / don't split). Unlike lattice + Viterbi minimum-cost approaches (e.g. MeCab), it requires no dictionary cost design or dynamic programming.

The public API is a single interface, backed internally by a set of interchangeable backends:

- **KyTea-compatible backend** — loads KyTea's trained models as-is and performs inference with the same feature extraction and linear SVM classifier. Byte-for-byte agreement with KyTea's own output is verified.
- **Custom MLP backend** — uses a quantized (int16) MLP with a SIMD-accelerated scorer instead of a linear SVM as the classifier. On UD_Japanese-GSD (no dictionary), it reaches boundary F1 within ~0.7pt of a KyTea model trained on the same data, while running ~2.2x faster than KyTea on the same CPU thread.

The corpus format is always KyTea's corpus format (full/partial annotation), used consistently across both backends for training and accuracy comparison.

Segmentation speed is benchmarked against both KyTea and [Vaporetto](https://github.com/daac-tools/vaporetto) (used only as a comparison target, not as a backend); see [bench/README.md](bench/README.md).

Detailed design documents (English / Japanese):

- Overall design: [docs/design.md](docs/design.md) / [docs/design.ja.md](docs/design.ja.md)
- MLP backend implementation design: [docs/mlp_impl_design.md](docs/mlp_impl_design.md) / [docs/mlp_impl_design.ja.md](docs/mlp_impl_design.ja.md)
- MLP backend module design: [docs/mlp_module_design.md](docs/mlp_module_design.md) / [docs/mlp_module_design.ja.md](docs/mlp_module_design.ja.md)
- Tag prediction plan: [docs/tag_prediction_plan.md](docs/tag_prediction_plan.md) / [docs/tag_prediction_plan.ja.md](docs/tag_prediction_plan.ja.md)

## License

MIT License. See [LICENSE](LICENSE).
