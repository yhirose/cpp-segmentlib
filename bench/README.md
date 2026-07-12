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

- **Single-thread, tags off.** KyTea `-notags`, Vaporetto without
  `--predict-tags`, segmentlib word-segmentation only.
- **Inference throughput (chars/sec)** with model load and process startup
  removed. Each tool is timed (via `hyperfine`) on a tiny corpus and the full
  corpus; the tiny run captures fixed cost (startup + load), so
  `throughput = full_chars / (t_full - t_tiny)`.
- **Load time** is reported separately (≈ the tiny-run time).
- **`bench_segment`** additionally measures segmentlib in-process (load once,
  loop over the corpus, time only `tokenize`) for the purest per-char number.

Throughput is in Unicode codepoints/sec, matching Vaporetto's convention.

## Corpus

Real Japanese prose from **Aozora Bunko** (public domain): a handful of works by
Sōseki, Dazai, Akutagawa, and Miyazawa (see `aozora_works.txt`), cleaned to one
sentence per line. Real varied text is used rather than a repeated seed, whose
unrealistic cache locality would inflate throughput.

## Running

```sh
scripts/fetch_kytea_model.sh      # once: get the KyTea model
bench/setup.sh                    # once: fetch corpus, build Vaporetto, convert model
cmake --build build-release       # build segmentlib + segmenter + bench_segment (Release!)
bench/run.sh                      # correctness gate + timings
```

`run.sh` honors env overrides: `KMODEL`, `VMODEL`, `OURS`, `KYTEA`,
`VAPORETTO`, `CORPUS`, `WARMUP`, `RUNS`.

> Benchmarks must use the **Release** build; a Debug build reports misleading
> numbers.

## Notes

- `bench/.vendor/`, `bench/corpus/`, and `bench/results/` are gitignored
  (build artifacts and fetched data).
- Numbers are machine-specific; record the CPU alongside any results.
