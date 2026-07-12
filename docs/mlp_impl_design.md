# MLP Trainer / Inference Engine Implementation Design

Building on the design in Section 5 of `design.ja.md` and the module decomposition in `mlp_module_design.ja.md`, this document defines the implementation-level formulas, data structures, and numeric representations for the inference engine (int16 forward propagation) and the trainer (fp32 forward + backward propagation, PTQ).

Notation (shared throughout):

| Symbol | Meaning | Default |
|---|---|---|
| `w` | One-sided window size (number of EGCs) | 5 |
| `d` | Codepoint embedding dimension | 64 |
| `H` | Hidden layer width | 256 |
| `2w` | Number of EGC slots in the window | 10 |
| `Fd` | Total number of dictionary binary features = `num_dicts × 12` | |
| `M` | Number of EGCs in the sentence (there are `M-1` boundaries) | |

Forward propagation (real numbers, for one boundary):

```
a   = Σ_j W1_j · v_j  +  W_dict · f  +  b1        ∈ R^H     (j=0..2w-1)
h   = ReLU(a)                                      ∈ R^H
y   = w2 · h  +  b2                                ∈ R
Decision: y > 0 means a boundary (inference). p = sigmoid(y) is used only for the training loss.
```

`v_j ∈ R^d` is the EGC vector at window position j = the mean of the embeddings of its constituent codepoints. `W1_j` is the `[·][j*d : (j+1)*d]` slice of `W1` (H×d). `f ∈ {0,1}^Fd` is the dictionary binary feature vector (Section 5.4).

---

## Part I: Inference Engine (int16, forward propagation only)

### I.1 Composing the Quantization Scales (the core of the implementation)

Training is done in fp32, inference in int16. Each tensor is converted to int16 with its **own independent scale** (weights use max-abs, no clipping; see Section II.6):

```
emb   ≈ S_e   · emb_q      (int16)
W1    ≈ S_w1  · w1_q       (int16)
Wdict ≈ S_wd  · wdict_q    (int16)
w2    ≈ S_w2  · w2_q       (int16)
```

The accumulator is unified under a **single scale `S_acc`**. By putting all three terms that get summed (layer 1, dictionary, b1) into int32 values in units of `S_acc`, the inference hot path reduces to nothing but "int32 addition." All conversions happen once at load time (no multiplication appears in the hot path).

**`S_acc` is not derived from the weight scales (important)**. If we set `S_acc = S_e·S_w1`, then with `S_e` and `S_w1` chosen to make full use of the int16 range, we get `S_acc ≈ 10^-9`, so activations that are O(1) in real terms turn into integer values of ~10^9, and **int32 is guaranteed to overflow after summing just 2w terms**. Instead, we choose `S_acc` independently via **calibration from activation statistics**:

```
S_acc = pct99.99(|a|) / Amax,   Amax = 2^22
(a: the distribution of layer-1 activations (pre-ReLU) measured on validation data after training)
```

- Headroom: a single entry (the contribution of one window position) is at most O(Amax/1), and `acc`, even as the sum of "table×2w + a few dictionary columns + b1," stays **≲2^26 — more than 5 bits of headroom against int32's 2^31**. Debug builds include a saturation-detection assert.
- Since `S_acc` cannot be reproduced on the loader side (activation statistics are only available at training time), it is **stored in the model file as `acc_scale`** (Section 5.7, field 8b).

**(1) Layer 1 (precomputed table for frequent EGCs)** — the `table` from Section 5.6. For the set of constituent codepoints C of an EGC (`n=|C|`):

```
raw[egc][j][h] = Σ_{c∈C} Σ_{k=0}^{d-1} w1_q[h, j*d+k] · emb_q[c,k]      ∈ int64 (in units of S_e·S_w1, times n)
table_q[egc][j][h] = llround( (double)raw · R / n )                      ∈ int32
    R = S_e·S_w1 / S_acc   (rescaling factor, double. Computed once at load time)
```

- The intermediate sum `raw` is at worst d·2^30 ≈ 2^36, so **int64 is mandatory**. Since `raw ≤ 2^53` always holds, it can be represented exactly as a double, and `llround((double)raw · R / n)` is **deterministic under IEEE 754** (bit-identical across platforms).
- The stored value, in units of `S_acc`, is ≲Amax and fits within int32.

**(2) Layer 1 (fallback for infrequent EGCs)** — composed at runtime using **the same formula** as (1):

```
sum_c[k] = Σ_{c∈C} emb_q[c,k]                       ∈ int32 (per k; n is at most a few dozen, so this is safe)
raw_j[h] = Σ_{k} w1_q[h, j*d+k] · sum_c[k]          ∈ int64
acc_j[h] = llround( (double)raw_j · R / n )         ∈ int32
```

Because (1) reuses this exact integer pathway, **the frequent-path and fallback-path results are bit-exact identical** (the table is merely a cache and does not alter numeric behavior). The fallback's double multiplication runs about 2w·H times per EGC, but since it is a rare path, it has no effect on speed.

**(3) Dictionary term** — converted at load time into an int32 column vector in units of `S_acc`:

```
dict_col_q[k][h] = llround( (S_wd / S_acc) · wdict_q[h,k] )   ∈ int32
```

At inference time this is just "add the column for each active feature k" (no multiplication).

**(4) b1** — a double in Section 5.7. At load time: `b1_q[h] = llround(b1[h] / S_acc)` ∈ int32.

**(5) Output layer** — `y = w2·h + b2`. `h_q` is in units of `S_acc` (ReLU commutes with a positive scale).

```
y ∝ Σ_h w2_q[h] · h_q[h]  +  b2_q          (computed in int64)
b2_q = llround( b2 / (S_w2 · S_acc) )       ∈ int64 (at load time)
Decision:  ( Σ_h w2_q[h]·h_q[h] ) + b2_q  > 0
```

Since `S_w2·S_acc > 0`, the sign is preserved (consistent with the "`y>0` decision" in Section 5.4). The multiply-accumulate is `w2_q`(int16)×`h_q`(int32), where a single term is ≤2^46, and with H=256 terms the total is ≤2^54 < 2^63 → **safe in int64**.

> This composition table is the strict formalization of design.ja.md Section 5.7's rule, "quantize `b1` into the accumulator's integer scale, and `b2` into the `w2·h` scale, at load time." `S_acc` is the file's `acc_scale`, and the `b2` scale is `= S_w2·S_acc`.

### I.2 Inference Hot Path (one boundary)

```
acc[H] = b1_q                                  // int32 copy
for j in 0..2w-1:
    egc = window[j]                            // edges use the PAD pseudo-EGC (see below)
    precompute.add_into(acc, egc, j)           // frequent: table_q lookup / infrequent: fallback composition
for k in active_dict_features:                 // output of DictMatcher
    acc += dict_col_q[k]
relu_inplace(acc)                              // acc[h] = max(0, acc[h])
sum = Σ_h w2_q[h] * (int64)acc[h]  +  b2_q     // int64
boundary ⇔ sum > 0
```

- **Handling of PAD (finalized)**: PAD is treated as a "pseudo-EGC whose only constituent is embedding row 0 (n=1)," and `table_q` permanently holds entries for it at all 2w window positions, just like any other frequent EGC (positions where the window extends past the sentence boundary always go through this table path). It never falls into the fallback path.
- **Overflow**: Thanks to the calibration of `S_acc` (Section I.1, `Amax=2^22`), `acc` stays ≲2^26 — more than 5 bits of headroom in int32 — even as the total of "table×2w + dictionary columns + b1." **Debug builds place a saturation-detection assert on every addition**; release builds have no check. int64 is required only for the intermediate sums during construction/fallback in (1)(2) and for the output multiply-accumulate in (5).
- The branch in `add_into` (table lookup for frequent EGCs / composition for infrequent ones) is hidden inside `PrecomputeTable` (the scorer calls it uniformly).

### I.3 SIMD Kernels (NEON / AVX2 / scalar)

Confined to 3 kernels inside `scorer`. All are lightweight loops of size H=256.

| Kernel | Operation | AVX2 | NEON |
|---|---|---|---|
| `add_into` | int32[256] += int32[256] | `_mm256_add_epi32` ×32 | `vaddq_s32` ×64 |
| `relu_inplace` | max(x,0) int32[256] | `_mm256_max_epi32` ×32 | `vmaxq_s32` ×64 |
| `dot_i64` | Σ int16[256]·int32[256] | 32-bit extend → multiply → int64 accumulate | `vmull` family |

- **`int16` re-quantization (later-stage optimization from Section 5.6)**: right-shift `table_q`/`acc` by `s` to convert to int16, making `add_into` an int16 SIMD op (2× throughput). This involves saturating clipping; when introduced, measure the saturation rate and decision-flip rate on validation data (same framework as I.5). The first implementation keeps int32.
- The scalar fallback is a reference implementation shared by all platforms (also used as the test oracle).

### I.4 Load-Time Construction (`Model::load`)

```
1. Read the header line, Config, scales (5 kinds, including `acc_scale`), vocabulary, embeddings, and W1/W_dict/w2 (Section 5.7)
2. Compute the rescaling factor R = S_e·S_w1 / S_acc
3. Build PrecomputeTable: for the set of frequent EGCs plus the PAD pseudo-EGC, compute table_q
   (I.1-(1), integer pathway + R rescaling, int64 intermediates)
   For infrequent EGCs, retain references to emb_q, w1_q, R (I.1-(2), bit-exact with the frequent path)
4. Convert to dict_col_q (I.1-(3)), b1_q (I.1-(4)), b2_q (I.1-(5))
5. DictMatcher: word list → normalization → EGC segmentation → build an EGC-unit AC automaton (text/aho_corasick)
6. Cross-check unicode_version against the runtime EGC segmenter, and warn on mismatch
```

The "set of frequent EGCs" can either be determined from the codepoint vocabulary (as an approximation of frequent EGCs), or the model can separately carry an explicit list of frequent EGCs. For the first implementation, it suffices to treat "single-codepoint EGCs (i.e., the codepoint vocabulary itself) as frequent, and always route multi-codepoint EGCs to the fallback path" (Japanese/Chinese text mostly takes the table path; only things like emoji get composed).

### I.5 Threading and Buffers

- A `Workspace` is prepared that bundles `EncodedEgc`, `acc[H]`, `h`, and the dictionary feature buffer. `score_boundaries_into` takes this as input and performs no allocation (following KyTea's `_into` convention).
- `tokenize_all`, like KyTea, keeps a `Workspace` per thread. `Model` is immutable, so it can be shared.

---

## Part II: Trainer (fp32, forward + backward propagation, PTQ)

### II.1 Forward Pass (fp32, mini-batch)

Batch size B. Embedding gather plus mean pooling over the window builds a dense matrix, which is then handled via GEMM.

```
X   ∈ R^{B × 2w·d}   each row = concat(v_0..v_{2w-1})     (embedding gather → mean pool)
F   ∈ R^{B × Fd}     dictionary binary features (dense; Fd is small)
A   = X · W1ᵀ + F · Wdictᵀ + b1     ∈ R^{B×H}          // GEMM ×2 + broadcast
Hh  = ReLU(A)                        ∈ R^{B×H}
Y   = Hh · w2 + b2                   ∈ R^{B}            // GEMV
P   = sigmoid(Y)
```

`X·W1ᵀ` is the dominant computation → `ComputeBackend::gemm`. `W1` is kept in the same `[h][j*d+c]` layout as inference (Section 5.7) and used directly as the right-hand operand of the GEMM.

### II.2 Loss (masked BCE)

```
L = (1/Σm_b) Σ_b  m_b · BCE(P_b, t_b)      t_b∈{0,1} label, m_b∈{0,1} mask (Section 5.5)
```

For unknown positions (partial annotation), `m_b=0` so they contribute zero to both loss and gradient.

### II.3 Backward Propagation (fp32)

The combination of sigmoid + BCE gives a simple output gradient:

```
dY_b   = (P_b - t_b) · m_b / Σm             ∈ R^{B}
dw2    = Hhᵀ · dY                            ∈ R^{H}
db2    = Σ_b dY_b
dHh    = dY ⊗ w2                             ∈ R^{B×H}
dA     = dHh ⊙ 1[A>0]                        ∈ R^{B×H}   (ReLU')
dW1    = dAᵀ · X                             ∈ R^{H×2w·d}   // GEMM
dWdict = dAᵀ · F                             ∈ R^{H×Fd}
db1    = Σ_b dA_b                            ∈ R^{H}
dX     = dA · W1                             ∈ R^{B×2w·d}   // GEMM (backprop into embeddings)
```

**Backprop from mean pooling into the embeddings (sparse)**: for window position j's block `dv_j ∈ R^d` in `dX`, scatter-add it to each constituent codepoint row of that EGC, scaled by `1/n_j`:

```
for each example b, window position j, constituent codepoint row c of the EGC:
    grad_emb[c] += dv_{b,j} / n_j
```

- **The embedding gradient is sparse — only the rows that appeared in the batch receive nonzero gradient**. The scatter is a scatter-add (`ComputeBackend`). The PAD row and the UNK row receive gradients like any ordinary row (UNK accumulates gradient from low-frequency tokens funneled in during vocabulary construction in Section II.5).

### II.4 Optimization (Adam)

- **dense** (`W1, Wdict, w2, b1, b2`): standard Adam. First and second moments are kept for every element.
- **embedding (sparse)**: only rows that appeared are updated. Each row keeps its own `(m, v, step)`, with lazy updates (moment updates and bias correction applied only to touched rows). There is also an option to exclude the embeddings from weight decay (to avoid excessive shrinkage).
- Convergence is judged via LR schedule, warmup, and early stopping (on dev-set boundary F-score).

### II.5 Data Pipeline (`train/`)

```
corpus  : KyTea full/partial annotation (Section 6) → sentences + boundary labels/mask
        (full: all boundaries t∈{0,1}, m=1 / partial: | - map to t, spaces/? map to m=0)
example :
  1. Normalization (scheme a, unicode::normalize)
  2. EGC segmentation (unicode::egc)
  3. Vocabulary construction (first pass): codepoint frequency → below-threshold entries are excluded from the vocabulary and routed to the UNK row (Section 5.3)
  4. Collision handling: sentences with a boundary inside an EGC are skipped with a warning, and statistics are recorded (Sections 5.2/5.5)
  5. Example generation: for each boundary → the window's (EGC → constituent codepoint row-ID sequence, edge PAD) + dictionary features F + t + m
dataset : shuffle → batch → gather + mean pool to build X, dense F, t, m
```

- The vocabulary and dictionary AC automaton are finalized before training starts (first pass). Examples are kept as index sequences and gathered at batch time (practical memory footprint for datasets of a few million examples).
- The dictionary AC automaton uses the same `text/aho_corasick` as inference (EGC-unit, normalized words), guaranteeing that feature generation is consistent between training and inference.

### II.6 PTQ (Post-Training Quantization) and Validation

**Weight scale selection: max-abs, no clipping**. For weights, outliers matter more (a large weight carries strong evidence in its connection), and percentile-based clipping would specifically destroy accuracy. int16 has a wide enough range that max-abs still gives sufficient resolution:

```
S_e   = max|emb|   / 32767      S_w1  = max|W1|  / 32767
S_wd  = max|Wdict| / 32767      S_w2  = max|w2|  / 32767
each *_q = round(param / S_*)     (no clipping needed since it's max-abs)
```

**Activation calibration: percentile-based** (outliers may be clipped here — rare saturation is caught by the decision-flip check):

```
Collect the distribution of layer-1 activations (pre-ReLU) a on validation data (or a subset of it)
S_acc = pct99.99(|a|) / Amax,   Amax = 2^22    (see Section I.1)
```

**Export contents**: what gets written to the file is `emb_q, w1_q, wdict_q, w2_q` (int16) + the 5 scales (`S_e, S_w1, S_wd, S_w2, S_acc`, double) + `b1, b2` (raw double values). `table_q`/`dict_col_q`/`b1_q`/`b2_q` are built at load time, so they are **not included in the file** (Section 5.7).

**Validation (decision-flip check, Section 5.5)**: on the dev set,

```
Compare the sign of fp32's y  vs  the int16 inference engine's decision, across all boundaries
Tally the flip rate, and the |y| of flipped examples (whether they're near zero)
```
- If the flip rate exceeds tolerance: adjust the `Qmax` margin, reconsider the percentile, and if that still fails, fall back to QAT (the conditional fallback in Section 5.5).
- Also record the number of saturation (clipping) events.

### II.7 Export (Section 5.7 format)

Written with `bytes/binary_writer` in the field order from Section 5.7: header line → Config(w,d,H,num_dicts,unicode_version) → the 5 scales (including `acc_scale`) → vocabulary (V, codepoints in ascending order) → `emb_q` → `w1_q` → `wdict_q` → `b1`(double) → `w2_q` → `b2`(double) → dictionary (word list). **The precomputed table, the AC automaton, and the `*_q`-converted values are not written** (they are built at load time; Section I.4).

---

## Part III: Guaranteeing Consistency Between Training and Inference (Test Design)

We verify step by step that int16 inference is consistent with fp32 training.

1. **Encoding consistency**: `example`'s window generation and `mlp/vocab`'s `encode` produce identical (row-ID sequence, dictionary features) (same normalization, EGC segmentation, and AC automaton). Use shared code and test the difference.
2. **fp32 reference forward pass**: provide a `reference_forward(y)` that runs training's forward pass in single precision.
3. **Quantization error bounds**: with random inputs, measure `|S_acc·acc_q − a_fp32|` and the sign-agreement rate of the final `y` (if the composition in Section I.1 is correct, the signs should agree almost always).
4. **End-to-end**: train on a small corpus → export → run inference with `MlpBackend`, and check that the dev F-score matches the fp32 reference (within the expected quantization degradation).
5. **Comparison with KyTea** (Section 5.8): retrain KyTea on the same corpus and compare boundary decisions and accuracy/speed on identical text.
