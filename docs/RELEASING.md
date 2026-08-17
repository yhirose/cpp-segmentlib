# Releasing

A release is a git tag `vX.Y.Z` plus, when warranted, a retrained bundled
model. The mechanical part is automated:

```sh
just release            # dry run: shows what would happen
just release --run      # bump, commit "Release vX.Y.Z", tag, push
```

`scripts/release.sh` reads the current version from
`include/segmentlib/types.h` and requires a clean working tree and green CI
on the latest commit (the Windows job is continue-on-error and cannot
block). If the current version has never been tagged, it is released as-is
without a bump (the bootstrap case). Otherwise it picks the bump the same
way cpp-httplib/cpp-peglib pick theirs from abidiff, just with a different
detector: if `models/mlp/ja-ud-gsd.mod` differs from the last release tag
it bumps **minor** (a retrain changes segmentation output for every
downstream consumer, which is a bigger deal than most code changes),
otherwise it bumps **patch**. `--minor` / `--major` force a bigger bump
regardless of whether the model changed.

The version lives in two places, both updated by the script (there is no
other sync mechanism, so hand-edits must change both):

- `include/segmentlib/types.h`: `SEGMENTLIB_VERSION` ("X.Y.Z") and
  `SEGMENTLIB_VERSION_NUM` ("0xMMmmpp", two hex digits per component)
- `CMakeLists.txt`: `project(cpp-segmentlib VERSION X.Y.Z ...)`

## Retraining the bundled model (manual, before the release)

Retrain `models/mlp/ja-ud-gsd.mod` **only if training inputs changed** --
the trainer, the corpus pin in `scripts/fetch_ud_gsd_corpus.sh`, or the
dictionary. An inference-only or docs-only release ships the existing model
untouched; that is the normal case, and it is what keeps 3 MB binaries out
of the history for every small fix.

When retraining:

1. `just setup && just model`, then update `models/mlp/NOTICE` (corpus tag
   and commit) and `models/mlp/README.md` (training inputs, accuracy) to
   match.

2. Regenerate the MLP golden fixture and review the diff by eye before
   accepting it:

   ```sh
   build/src/cli/segmenter predict --model models/mlp/ja-ud-gsd.mod \
       < tests/golden/mlp_fixtures/input.txt \
       > tests/golden/mlp_fixtures/expected.txt
   git diff tests/golden/mlp_fixtures/expected.txt
   ```

   The fixture pins "the bundled model's output does not drift"; blindly
   overwriting it on a red test would reduce the test to a tautology. A
   diff here must be explainable by the input change that motivated the
   retrain.

3. Commit the retrain as an ordinary commit and let CI validate it; then
   `just release --run` as usual -- it detects the model changed and bumps
   minor automatically, no `--minor` flag needed.
