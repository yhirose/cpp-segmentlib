# Segmentation benchmark: segmentlib vs KyTea vs Vaporetto

Inference-only speed comparison for Japanese word segmentation.

## Design principle: compare the *same computation*

A speed comparison is only meaningful if the tools do the same work. So all
three run the **same model** — the KyTea `jp-0.4.7-5` model, converted to
Vaporetto's format with `convert_kytea_model`. Before any timing, a
**correctness gate** diffs the outputs:

- **segmentlib is byte-identical to KyTea** (0 differing lines).
- **Vaporetto diverges slightly** (~0.4% of lines) — the "minor implementation
  differences" its authors note. Its speed therefore reflects a slightly
  different (approximate) computation, which the results annotate.

## What is measured

- **Single-thread, tags off.** KyTea word segmentation only (`-notags`;
  `calculateWS` in the harness), Vaporetto without `--predict-tags`,
  segmentlib word-segmentation only.
- **Pure inference speed, in-process.** Each tool loads the model once and
  only the segmentation of the corpus is timed, so model load and process
  startup are excluded. segmentlib and KyTea use dedicated harnesses,
  `bench_segment` and `bench_kytea` (linked against libkytea), which time
  in-memory text, so I/O is excluded too, and take the best of `ITERS`
  passes. Vaporetto's `predict` reports its own tokenization time
  (`Elapsed`), which also covers reading stdin and writing the output (to
  `/dev/null` here); the best of 5 runs is taken.
- **Load time** (reading the model and building its in-memory structures) is
  reported by the two harnesses. Vaporetto builds a double-array automaton at
  load, a one-time cost of several seconds.

Throughput is in Unicode codepoints/sec, matching Vaporetto's convention.

## Corpus

Real Japanese prose from **Aozora Bunko** (public domain): a handful of works by
Sōseki, Dazai, Akutagawa, and Miyazawa (see `aozora_works.txt`), cleaned to one
sentence per line. Real varied text is used rather than a repeated seed, whose
unrealistic cache locality would inflate throughput.

## Running

Needs an installed KyTea (the `kytea` command for the correctness gate and
libkytea for `bench_kytea`, e.g. `brew install kytea`; set `KYTEA_PREFIX` for
`setup.sh` if it lives elsewhere), plus cargo, curl and python3.

```sh
scripts/fetch_kytea_model.sh      # once: get the KyTea model
bench/setup.sh                    # once: fetch corpus, build Vaporetto and bench_kytea, convert model
just bench-all                    # Release build in build-release, then bench/run.sh
```

Without `just`, configure and build `build-release` as the justfile's
`bench-build` recipe does (Release, tests and trainer off), then run
`bench/run.sh`. `just bench` uses the same build to time `bench_segment` on
the MLP reference model alone.

`run.sh` honors env overrides: `KMODEL`, `VMODEL`, `OURS`, `KYTEA`,
`VAPORETTO`, `CORPUS`, `BENCH_SEGMENT`, `BENCH_KYTEA`, `ITERS` (passes per
harness, default 7). Without `bench_kytea`, KyTea's speed shows as `n/a`.

> Benchmarks must use the **Release** build; a Debug build reports misleading
> numbers.

## Notes

- `bench/.vendor/`, `bench/corpus/`, and `bench/results/` are gitignored
  (build artifacts and fetched data).
- Numbers are machine-specific; record the CPU alongside any results.
