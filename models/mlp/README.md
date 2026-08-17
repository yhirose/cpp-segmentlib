# Bundled MLP reference model

`ja-ud-gsd.mod` (~2.1 MB) is the reference Japanese word-segmentation model
for the MLP backend, committed so the library works out of a clone with no
download or training step. Licensing is in `NOTICE` (the model file is
CC BY-SA 4.0, separately from the repository's MIT code).

It used to be a gitignored build product at `corpus/ud-gsd/mlp.mod`; the
committed copy here replaced that arrangement.

## Provenance

| | |
|---|---|
| Trained by | segmentlib v0.1.0 (`segmenter train --backend mlp`) |
| Corpus | UD_Japanese-GSD r2.18 (commit `33e7310b`), fetched by `scripts/fetch_ud_gsd_corpus.sh` |
| Dictionary | unidic-mecab 2.1.2 word list, via `scripts/fetch_unidic_dict.sh` |
| Command | `just model` (see the `model:` recipe in the justfile for the exact flags; default seed) |
| Dev accuracy | boundary F1 0.9897 (P 0.9886, R 0.9908) on the UD-GSD dev split |

Training is seeded and was verified bit-reproducible on the machine that
produced this file (two runs, byte-identical). The GEMMs go through BLAS,
so reproduction on a different platform or BLAS implementation may differ
in the last bits; the seed and inputs above are the authoritative record.

As a segmenter it trails a KyTea model trained on the same corpus by roughly
0.7-0.9pt of boundary F1 (see the top-level README); it is the reference
model, favoring speed and zero setup over the last few tenths of a point.

Retraining policy: only when the trainer or the training inputs change, not
on every release -- docs/RELEASING.md. The regression test in
tests/golden/golden_test.cpp pins this file's output on a fixed corpus, so a
retrain also regenerates tests/golden/mlp_fixtures/expected.txt.
