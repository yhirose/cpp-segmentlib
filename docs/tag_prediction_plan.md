# Design Plan for Implementing the Tag Prediction Backend (Stage A: POS and Reading for Known Words)

Implementation plan for **Stage A (tag prediction for words in the dictionary)**, one of the not-yet-started items in
Section 11 of `docs/design.ja.md`, "tag prediction (reading/POS) + subword dictionary and LM." Stage B (reading
prediction for unknown words: subwordDict + KyteaLM + beam search) is out of scope for this plan; only the
connection point is defined (Section 8).

This plan is based on a direct examination of the KyTea core source (`kytea.cpp:calculateTags`,
`feature-lookup.cpp`, `corpus-io-full.cpp`). The word segmentation (WS) backend is already byte-identical and
complete, and its infrastructure (`Automaton`, the `FeatureLookup` equivalent, the `Model` loader, `CharTable`) is
reused as much as possible.

## 0. Scope and Acceptance Criteria

- **In scope**: Tag prediction for words present in the dictionary. The actual model `jp-0.4.7-5` has **2 levels**
  (`lev0` = POS, `lev1` = reading). The output format is `surface/POS/reading` (e.g., `私/代名詞/わたし`), with
  `" "` as the word delimiter, `"/"` as the tag delimiter, and `"&"` as the candidate delimiter.
- **Out of scope (Stage B)**: **Reading** prediction for words not in the dictionary (unknown words). This depends
  on `calculateUnknownTag` → `generateTagCandidates` (beam search over the subwordDict + character LM), and is a
  large, independent chunk of work.
- **Acceptance criteria (important distinction)**:
  - POS (`lev0`) is predicted with a **global model** for both known and unknown words, so Stage A can aim for
    **byte-identical output for all words**.
  - Reading (`lev1`) can be predicted from the dictionary for known words, but **unknown-word readings require
    Stage B**. Therefore, Stage A's byte-identity gate should either be limited to "sentences where every word is
    in the dictionary," or evaluated **per level** (POS always matches; readings match only for known words).
  - This asymmetry is the crux of the Stage A design (Sections 6 and 8).

## 1. Structure of the Actual Model (confirmed; re-verify at implementation time)

- `numTags=2`. The dispatch in `calculateTags` is data-driven:
  - If `globalMods_[lev] != 0`, the global model is used (`useSelf=true`).
  - Otherwise, if the dictionary entry `ent->tagMods[lev]` exists, the per-word model is used (`useSelf=false`).
  - If neither exists, unknown-word handling applies (Stage B).
- The loader **retains both** (global / per-word), and dispatch is done at inference time in a data-driven manner,
  just as in KyTea.

### 1.1 Measured Results (at completion of A-1, `jp-0.4.7-5.mod`)
Measurements from scanning the actual model after switching to retention in A-1 (`num_tags=2`, `num_dicts=7`):

| lev | Global model | Per-word model | Words with candidate strings | A-2 dispatch |
|---|---|---|---|---|
| **lev0 (POS)** | **Present**: num_weights=21, candidates=21, lookup=yes (charDict 85545 payloads / typeDict / **selfDict 20217 states** / biases=21 / tagDictVector=3087=7×21×21 / tagUnkVector=21) | **0 entries** | 216,914 | **All words go through the global model** (`useSelf=true`: charN+typeN+selfWeights+tagDictWeights). Byte-identical output for all words is achievable. |
| **lev1 (reading)** | **None** (candidates=0) | **1,828 entries** (e.g.: num_weights=4, lookup=yes, **selfDict 0 states**, tagUnk=0) | 326,369 | **Per-word** (`useSelf=false`: charN+typeN only). Words with candidates but no per-word model use the deterministic `tags[0]`. Unknown-word readings require Stage B. |

- Total vocabulary: 850,724 entries. lev0 has no per-word models (global only); lev1 has no global model
  (per-word only) — the asymmetry (Section 0) is confirmed with real data.
- Note (A-2): For per-word lev1, there are cases where the number of candidate strings (e.g., 9) does not match
  the classifier's `num_weights` (e.g., 4). Since KyTea's `calculateTags` only takes `tags[i]` for
  `scores.size()` entries, **loop over `num_weights`, not the length of the candidate array**.
- selfDict / tagDictVector are non-empty only for lev0 (global) → `addSelfWeights`/`addTagDictWeights` only take
  effect on the `useSelf=true` path (consistent with the branching in `calculateTags`).

## 2. Model Loader Changes (`src/kytea/model.cpp`: skip → retain)

The current loader already **correctly scans (skips) all sections**. Stage A is mainly about "turning skips into
retention."

### 2.1 Extract a Reusable `KyteaModel` Retention Structure
Implement a **retaining version**, `TagModel`, of `KyteaModel` (i.e., the classifier), which is currently skipped
via `skip_kytea_model` / `skip_feature_lookup`, sharing structure with what is already held for WS. What must be
retained:
- `multiplier` (double), `numWeights` (= number of candidates), `labels` (can be skipped)
- Of the 7 elements of `FeatureLookup`, those needed for tags:
  `charDict` (`Automaton<FeatVec>`), `typeDict`, `selfDict` (`Automaton<FeatVec>`: currently skipped),
  `biases` (`FeatVec`), `tagDictVector` (`FeatVec`: currently skipped), `tagUnkVector` (`FeatVec`: currently
  skipped).
  (`dictVector` is WS-specific and unused for tag models.)

### 2.2 Retaining the Global Tag Model (`model.cpp:200-207`)
Retain `numTags × (word list + KyteaModel)`, which is currently skipped in the loop:
- `globalTags_[lev]`: list of candidate tag strings for each level (array of `KyteaString` → retained as UTF-8)
- `globalMods_[lev]`: the `TagModel` from Section 2.1

### 2.3 Retaining Word Dictionary Entries (`read_word_entry`, `model.cpp:102-116`)
Add retention of the per-word tag information currently skipped, to `WordEntry`:
- `tags[lev]`: candidate tag strings (`KyteaString` → UTF-8)
- `tagInDicts[lev]`: dictionary bitmask for each candidate (used in `getDictionaryMatches`)
- `tagMods[lev]`: per-word `TagModel` (absent for many words = null flag)
- `inDict` / `char_length` reuse existing retention.

### 2.4 WS Model's FeatureLookup (`model.cpp:197-198`)
The WS model's `tagDictVector`/`tagUnkVector` are normally empty. These can continue to be skipped (since tags
use the FeatureLookup on the tag-model side).

## 3. Score Computation: Known-Word Tags (porting the `else` path of `kytea.cpp:calculateTags`)

For each word `word` and each level `lev` (`startPos`/`finPos` are the character span within the sentence):

1. Select `tags` (the candidate list) and `tagMod` via the dispatch in Section 1.
2. If `tagMod` has no FeatureLookup: set `tag = tags[0]`, with margin = 100 (non-probabilistic) / 1
   (probabilistic), determined (deterministic).
3. If a FeatureLookup exists, accumulate `scores = vector<FeatSum>(numWeights, 0)` via:
   - `addTagNgrams(charStr, charDict, scores, charN, startPos, finPos)`
   - `addTagNgrams(typeStr, typeDict, scores, typeN, startPos, finPos)`
   - Only when `useSelf` (global):
     - `addSelfWeights(charStr[span], scores, 0)`
     - `addSelfWeights(typeStr[span], scores, 1)`
     - `addTagDictWeights(getDictionaryMatches(charStr[span], 0), scores)`
   - `scores[j] += biases[j]`
   - If `scores.size()==1`, push a second entry (non-probabilistic 0 / probabilistic `-scores[0]`)
4. For each candidate, build `KyteaTag(tags[i], ...)` with `scores[i] * multiplier` as the confidence, and
   **sort in descending order**.
5. Apply margin computation (non-probabilistic: `score -= secondBest`) or probability computation
   (probabilistic: `exp` → normalize).
6. The string of the **top (highest-scoring) candidate** is used for output.

### 3.1 `FeatureLookup` Methods to Port (`feature-lookup.cpp`, all small)
- `addTagNgrams` (~38 lines): Builds substrings by concatenating only the ±window characters **outside** the word
  span, and matches them against `charDict`/`typeDict`. The offset into the weight vector uses a
  **reverse-order index**, `pos = (window*2 - matchPos - offset - 1) * numWeights`, to select the candidate
  block. This formula must be **reproduced exactly, character for character** (this is the easiest place to get
  wrong on the tag side).
- `addSelfWeights` (~13 lines): Looks up `selfDict_->findEntry(word)` for a whole-word match and adds the
  weight block `[featIdx*numWeights ..]`.
- `addTagDictWeights` (~17 lines): Depending on whether there is a dictionary match, adds `tagUnkVector` (no
  match) or `tagDictVector` (match, `base = di*T*T + tagIdx*T`). `getDictionaryMatches` (~15 lines,
  `kytea.cpp:556`) builds `(dictionary index, candidate index)` pairs from `ent->tagInDicts[lev]` and
  `numDicts`.

## 4. Type / API Changes

### 4.1 `CharTable`: Add id → UTF-8 Reverse Lookup
Tag strings are held internally in the model as `KyteaString` (an id sequence). To output them, an
**id → code point** reverse lookup is needed. Add `std::vector<char32_t> id_to_cp_` (indexed by id, equivalent
to `charNames_`) to `CharTable`, and provide `decode_string(KyteaString)` to convert `KyteaString` → UTF-8. The
loader's tag strings may be converted to UTF-8 once at construction time and retained as such (keeping reverse
lookup out of the inference hot path).

### 4.2 `Segment` in `types.h`
The `tags` slot of `Segment{ begin, end, tags }` already exists (currently `{}`). Populate `tags` as
`std::vector<std::string>` (in level order). POS = tags[0], reading = tags[1].

### 4.3 `KyteaBackend::tokenize`
After boundaries are fixed in WS, if `do_tags` and a tag model exists, run the Section 3 computation for each
word and populate `Segment.tags`. `tokenize_boundaries` is unchanged. As with WS, scratch state (scores, etc.)
is reused via `thread_local` to remain reentrant.

### 4.4 CLI (`predict_command.cpp` / `output.cpp`)
- Output tags by default (following KyTea). Add `--notags` to suppress tags (current behavior = WS only).
- Extend `append_full_line`: `surface` + `"/" + tag` for each level. The surface is escaped using the equivalent
  of the existing `showEscapedString`, but **tag strings are not escaped** (matching KyTea's `showString`).
- The unknown-word marker (`unkTag_` when `w.getUnknown()`) defaults to empty. Whether it is needed should be
  confirmed empirically.

## 5. Output Formatting Details (matching `corpus-io-full.cpp:writeSentence`)

```
word1<wb>word2<wb>...      wb=" "
word = surface<tb>tag0<tb>tag1      tb="/"
(only when outputting multiple candidates, the 2nd and later tag candidates are joined with <eb>="&"; by default only the top candidate is output)
```

For Stage A, it is sufficient to support the default (top candidate only = KyTea `-out full`) = **already
implemented, byte-identical**.

Note: This section originally referred to `-alltags`, but no such KyTea option actually exists (this was an
error). Outputting multiple candidates with confidence is done via `-tagmax N` (default 3) + `-out conf` format,
which prints `surface/candidate1&candidate2&candidate3/...` followed by margin confidence values across multiple
lines. This is **not needed for the normal use case (word segmentation + POS + single best reading)**, so it will
not be implemented (both KyTea and MeCab default to single-best output). If it is implemented, it would require
reproducing the margin computation omitted in Stage A-2 (`(score - 2nd place) × multiplier`), retaining all
candidates, and float formatting. Deferred until needed.

## 6. Verification Strategy

- **Extend the golden tests**: Add `expected_tags.txt` to `tests/golden/fixtures/`, using
  `kytea -model jp-0.4.7-5.mod` (default output with tags) as the ground truth, and compare byte-for-byte against
  `predict` (with tags).
- **Realistic gating for Stage A**:
  - POS (lev0) can require an exact match across all words.
  - Reading (lev1) requires an exact match only for fixtures **limited to sentences where every word is in the
    dictionary**. For sentences containing unknown words, "allow discrepancies at the reading level (only check
    POS)" until Stage B is complete.
- **Float-matching concerns are minor for Stage A**: The output is only the **string of the top candidate** at
  each level. The top candidate is determined by the **maximum after descending sort** of `scores` (an integer
  `FeatSum`) × `multiplier`, and is fixed by the order before `exp`/normalization. That is, **determining the top
  candidate reduces to an integer score comparison**, so the risk of operation-order-dependent byte mismatches —
  the kind that arises in Stage B where a `double`-based LM is involved — is small. The only thing to watch is
  **sort stability on ties** (match KyTea's `std::sort` + `kyteaTagMore`, which uses strict `>`).
- Do not break WS regression tests (byte-identical matches on existing golden tests / benchmarks).

## 7. Expected Files to Change

- `include/segmentlib/kytea/model.h` / `src/kytea/model.cpp`: `TagModel` retention structure; switch the global /
  per-word tag models and dictionary entry tag information from skip → retain.
- `include/segmentlib/kytea/char_table.h` / `char_table.cpp`: id → UTF-8 reverse lookup.
- `src/kytea/scorer.cpp` (or a new `tag_scorer.cpp`): `addTagNgrams`/`addSelfWeights`/`addTagDictWeights`/
  `getDictionaryMatches`, and accumulation, sorting, and margin computation for known-word tags.
- `src/kytea/kytea_backend.cpp`: Wire tag computation into `tokenize`.
- `include/segmentlib/types.h`: Finalize the type of `Segment.tags` (`vector<string>`).
- `src/output.cpp` / `src/cli/predict_command.cpp`: Tag output formatting and `--notags`.
- `tests/golden/`: Golden tests with tags.

## 8. Connection Point to Stage B (Unknown-Word Readings)

- In `calculateTags`, when `tags==0 || tags->size()==0` (unknown word), `calculateUnknownTag(word, lev)` is
  called. In Stage A, treat `subwordModels_[lev]==0` as an **early return** (no reading), while still passing
  through POS (global model).
- Structures added in Stage B:
  - `subwordDict_`: `Dictionary<ProbTagEntry>` (Aho-Corasick with tags + log probabilities, a new payload type).
  - `subwordModels_[lev]`: `KyteaLM` (character n-gram language model, ~134 lines). Port the log-probability
    computation from `scoreSingle`.
  - `generateTagCandidates`: DP lattice over subword pieces + beam pruning (`unkBeam`) + `exp`/normalization/
    `tagMax` trimming.
- Stage B requires matching `double` LM scores **down to the order of operations**, which is a different quality
  of verification cost than Stage A (see the note in Section 6).

## 9. Stage Estimates (Relative)

| Stage | Main work | Scale | Difficulty of byte-matching |
|---|---|---|---|
| **A-1** | Loader retention (TagModel / global / dictionary entries) | Medium | — |
| **A-2** | Known-word scorer (Section 3) + output formatting + CLI | Medium | Low (top candidate determined by integer sort) |
| **B** | subwordDict + KyteaLM + beam search (unknown-word readings) | Large | High (float operation-order dependent) |

Stage A (A-1 + A-2) is a medium-scale effort comparable to the WS scorer + loader modifications, and is
completed as an extension of the existing infrastructure. Stage B is an independent chunk of work that can be
tackled whenever it becomes necessary.
