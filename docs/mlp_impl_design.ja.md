# MLP 学習機・推論器 実装設計

`design.ja.md` 5節の設計と `mlp_module_design.ja.md` のモジュール分割を前提に、推論器（int16 前方伝播）と学習機（fp32 前方＋逆伝播、PTQ）の実装レベルの式・データ構造・数値表現を定める。

記号（全体共通）：

| 記号 | 意味 | 既定値 |
|---|---|---|
| `w` | 片側窓幅（EGC個数） | 5 |
| `d` | コードポイント埋め込み次元 | 64 |
| `H` | 隠れ層幅 | 256 |
| `2w` | 窓のEGCスロット数 | 10 |
| `Fd` | 辞書二値素性の総数 = `num_dicts × 12` | |
| `M` | 文のEGC数（境界は `M-1` 個） | |

順伝播（実数、境界1つ分）：

```
a   = Σ_j W1_j · v_j  +  W_dict · f  +  b1        ∈ R^H     （j=0..2w-1）
h   = ReLU(a)                                      ∈ R^H
y   = w2 · h  +  b2                                ∈ R
判定: y > 0  で境界（推論）。 p = sigmoid(y) は学習の損失のみ。
```

`v_j ∈ R^d` は窓位置 j の EGC ベクトル ＝ 構成コードポイント埋め込みの mean。`W1_j` は `W1` の `[·][j*d : (j+1)*d]` スライス（H×d）。`f ∈ {0,1}^Fd` は辞書二値素性（5.4節）。

---

## 第I部：推論器（int16、前方伝播のみ）

### I.1 量子化スケールの合成（実装の核）

学習は fp32、推論は int16。各テンソルを**独立したスケール**で int16 化する（重みは max-abs、クリップなし。II.6）：

```
emb   ≈ S_e   · emb_q      (int16)
W1    ≈ S_w1  · w1_q       (int16)
Wdict ≈ S_wd  · wdict_q    (int16)
w2    ≈ S_w2  · w2_q       (int16)
```

加算器（アキュムレータ）は**単一のスケール `S_acc`** に統一する。加算される3項（第1層・辞書・b1）をすべて `S_acc` 単位の int32 に載せることで、推論ホットパスは「int32 の加算」だけになる。変換はすべてロード時に1回（ホットパスに乗算は出さない）。

**`S_acc` は重みスケールから導出しない（重要）**。`S_acc = S_e·S_w1` と置くと、int16 範囲を最大活用した `S_e`・`S_w1` では `S_acc ≈ 10^-9` となり、実数で O(1) の活性が整数値 ~10^9 に化けて **2w 個の加算で int32 が確実にオーバーフローする**。代わりに**活性統計からのキャリブレーション**で独立に選ぶ：

```
S_acc = pct99.99(|a|) / Amax,   Amax = 2^22
（a：学習後、検証データで測った第1層活性（ReLU前）の分布）
```

- ヘッドルーム：1エントリ（1窓位置の寄与）は O(Amax/1) 以下、`acc` は「table×2w ＋ 辞書列×数個 ＋ b1」の和でも **≲2^26、int32（2^31）に対し5bit以上の余裕**。デバッグビルドでは飽和検知アサートを入れる。
- `S_acc` はローダ側で再現できない（活性統計は学習時にしか得られない）ため、**モデルファイルに `acc_scale` として格納する**（5.7 フィールド8b）。

**(1) 第1層（頻出EGCの事前計算テーブル）** — 5.6節の `table`。EGC の構成コードポイント集合 C（`n=|C|`）について：

```
raw[egc][j][h] = Σ_{c∈C} Σ_{k=0}^{d-1} w1_q[h, j*d+k] · emb_q[c,k]      ∈ int64（S_e·S_w1 単位·n倍）
table_q[egc][j][h] = llround( (double)raw · R / n )                      ∈ int32
    R = S_e·S_w1 / S_acc   （リスケール係数, double。ロード時に1回計算）
```

- 中間和 `raw` は最悪 d·2^30 ≈ 2^36 なので **int64 必須**。`raw ≤ 2^53` は常に成り立つため double で正確に表現でき、`llround((double)raw · R / n)` は **IEEE754 の下で決定的**（プラットフォーム間で bit 一致）。
- 格納値は `S_acc` 単位で ≲Amax、int32 に収まる。

**(2) 第1層（非頻出EGCのフォールバック）** — 実行時に (1) と**同一の式**で合成する：

```
sum_c[k] = Σ_{c∈C} emb_q[c,k]                       ∈ int32（kごと。n は高々数十で安全）
raw_j[h] = Σ_{k} w1_q[h, j*d+k] · sum_c[k]          ∈ int64
acc_j[h] = llround( (double)raw_j · R / n )         ∈ int32
```

(1) がこの整数経路をそのまま流用するため、**頻出経路とフォールバック経路は bit-exact に一致**する（テーブルは単なるキャッシュであり、数値挙動を変えない）。フォールバックの double 乗算は 2w·H 回/EGC 程度で、稀な経路なので速度影響なし。

**(3) 辞書項** — ロード時に `S_acc` 単位の int32 列ベクトルへ変換：

```
dict_col_q[k][h] = llround( (S_wd / S_acc) · wdict_q[h,k] )   ∈ int32
```

推論時は「立っている素性 k の列を加算」だけ（乗算なし）。

**(4) b1** — 5.7 では double。ロード時に `b1_q[h] = llround(b1[h] / S_acc)` ∈ int32。

**(5) 出力層** — `y = w2·h + b2`。`h_q` は `S_acc` 単位（ReLU は正スケールと可換）。

```
y ∝ Σ_h w2_q[h] · h_q[h]  +  b2_q          （int64 で計算）
b2_q = llround( b2 / (S_w2 · S_acc) )       ∈ int64（ロード時）
判定:  ( Σ_h w2_q[h]·h_q[h] ) + b2_q  > 0
```

`S_w2·S_acc > 0` なので符号は保存される（5.4節の「`y>0` 判定」に一致）。積和は `w2_q`(int16)×`h_q`(int32) で1項 ≤2^46、H=256 項で ≤2^54 < 2^63 → **int64 で安全**。

> この合成表は design.ja.md 5.7 の「`b1` をアキュムレータ整数スケールへ、`b2` を `w2·h` スケールへ、ロード時に量子化する」を厳密化したもの。`S_acc` はファイルの `acc_scale`、`b2` スケール `= S_w2·S_acc`。

### I.2 推論ホットパス（1境界）

```
acc[H] = b1_q                                  // int32 コピー
for j in 0..2w-1:
    egc = window[j]                            // 端は PAD 擬似EGC（下記）
    precompute.add_into(acc, egc, j)           // 頻出:table_q / 非頻出:フォールバック合成
for k in active_dict_features:                 // DictMatcher の出力
    acc += dict_col_q[k]
relu_inplace(acc)                              // acc[h] = max(0, acc[h])
sum = Σ_h w2_q[h] * (int64)acc[h]  +  b2_q     // int64
境界 ⇔ sum > 0
```

- **PAD の扱い（確定）**：PAD は「埋め込み行0のみを構成要素とする擬似EGC（n=1）」として扱い、`table_q` に他の頻出EGCと同様 2w 位置分のエントリを常設する（窓が文端をはみ出す位置は常にこの table 経路）。フォールバック経路に落ちることはない。
- **オーバーフロー**：`S_acc` のキャリブレーション（I.1、`Amax=2^22`）により、`acc` は「table×2w ＋ 辞書列 ＋ b1」の総和でも ≲2^26 で int32 に5bit以上の余裕。**デバッグビルドでは加算ごとの飽和検知アサート**を置き、リリースでは検査なし。int64 が必須なのは (1)(2) の構築/フォールバック中間和と (5) の出力積和。
- `add_into` の分岐（頻出 table 引き / 非頻出合成）は `PrecomputeTable` 内に隠す（scorer は一様に呼ぶ）。

### I.3 SIMD カーネル（NEON / AVX2 / scalar）

`scorer` の中の3カーネルに閉じる。いずれも H=256 の小ループで軽い。

| カーネル | 演算 | AVX2 | NEON |
|---|---|---|---|
| `add_into` | int32[256] += int32[256] | `_mm256_add_epi32` ×32 | `vaddq_s32` ×64 |
| `relu_inplace` | max(x,0) int32[256] | `_mm256_max_epi32` ×32 | `vmaxq_s32` ×64 |
| `dot_i64` | Σ int16[256]·int32[256] | 32bit拡張→乗算→int64累積 | `vmull` 系 |

- **`int16` 再量子化（5.6節の後段最適化）**：`table_q`/`acc` を右シフト `s` で int16 化し `add_into` を int16 SIMD（スループット2倍）にする。飽和クリップを伴い、導入時は検証データで飽和率と判定反転を測る（I.5 と同じ枠組み）。第一実装は int32 のまま。
- スカラー fallback は全プラットフォーム共通の参照実装（テストの oracle にも使う）。

### I.4 ロード時構築（`Model::load`）

```
1. ヘッダ行・Config・スケール（`acc_scale` 含む5種）・語彙・埋め込み・W1/W_dict/w2 を読む（5.7）
2. リスケール係数 R = S_e·S_w1 / S_acc を計算
3. PrecomputeTable 構築: 頻出EGC集合＋PAD擬似EGCについて table_q（I.1-(1)、整数経路＋Rリスケール、中間int64）
   非頻出用に emb_q・w1_q・R への参照を保持（I.1-(2)、頻出経路と bit-exact）
4. dict_col_q（I.1-(3)）、b1_q（I.1-(4)）、b2_q（I.1-(5)）へ変換
5. DictMatcher: 単語リスト→正規化→EGC分割→EGC-unit AC 構築（text/aho_corasick）
6. unicode_version を実行環境の EGC 分割器と照合、不一致なら警告
```

「頻出EGC集合」は、コードポイント語彙（≒頻出EGCの近似）から決めるか、モデルに頻出EGCリストを別途持たせる。第一実装は「1コードポイントEGC（＝コードポイント語彙そのもの）を頻出扱い、多コードポイントEGCは常にフォールバック」で十分（日中はほぼ table 経路、絵文字等のみ合成）。

### I.5 スレッド・バッファ

- `EncodedEgc`・`acc[H]`・`h`・辞書素性バッファを束ねた `Workspace` を用意。`score_boundaries_into` はこれを受け取り割り当てをしない（KyTea の `_into` 規約）。
- `tokenize_all` は KyTea 同様スレッドごとに `Workspace` を持つ。`Model` は immutable なので共有可。

---

## 第II部：学習機（fp32、前方＋逆伝播、PTQ）

### II.1 前方（fp32、ミニバッチ）

バッチサイズ B。窓の埋め込み gather＋mean pooling で密行列を組み、GEMM 化する。

```
X   ∈ R^{B × 2w·d}   各行 = concat(v_0..v_{2w-1})     （embedding gather → mean pool）
F   ∈ R^{B × Fd}     辞書二値素性（密。Fd は小さい）
A   = X · W1ᵀ + F · Wdictᵀ + b1     ∈ R^{B×H}          // GEMM ×2 + broadcast
Hh  = ReLU(A)                        ∈ R^{B×H}
Y   = Hh · w2 + b2                   ∈ R^{B}            // GEMV
P   = sigmoid(Y)
```

`X·W1ᵀ` が主計算 → `ComputeBackend::gemm`。`W1` は推論の `[h][j*d+c]` 配置（5.7）と同一レイアウトで持ち、GEMM の右オペランドにそのまま使う。

### II.2 損失（マスク付き BCE）

```
L = (1/Σm_b) Σ_b  m_b · BCE(P_b, t_b)      t_b∈{0,1} ラベル, m_b∈{0,1} マスク（5.5節）
```

不明位置（部分アノテーション）は `m_b=0` で損失・勾配とも寄与ゼロ。

### II.3 逆伝播（fp32）

sigmoid+BCE の合成で出力勾配は簡潔：

```
dY_b   = (P_b - t_b) · m_b / Σm             ∈ R^{B}
dw2    = Hhᵀ · dY                            ∈ R^{H}
db2    = Σ_b dY_b
dHh    = dY ⊗ w2                             ∈ R^{B×H}
dA     = dHh ⊙ 1[A>0]                        ∈ R^{B×H}   (ReLU')
dW1    = dAᵀ · X                             ∈ R^{H×2w·d}   // GEMM
dWdict = dAᵀ · F                             ∈ R^{H×Fd}
db1    = Σ_b dA_b                            ∈ R^{H}
dX     = dA · W1                             ∈ R^{B×2w·d}   // GEMM（埋め込みへ逆伝播）
```

**mean pooling → 埋め込みの逆伝播（疎）**：`dX` の窓位置 j ブロック `dv_j ∈ R^d` を、その EGC の各構成コードポイント行へ `1/n_j` を掛けて散布加算：

```
for 例 b, 窓位置 j, EGC の構成コードポイント行 c:
    grad_emb[c] += dv_{b,j} / n_j
```

- **埋め込み勾配はバッチに現れた行のみ疎**。散布は scatter-add（`ComputeBackend`）。PAD 行・UNK 行も通常の行として勾配を受ける（UNK は II.5 の語彙構築で低頻度が流れ込む）。

### II.4 最適化（Adam）

- **dense**（`W1, Wdict, w2, b1, b2`）：標準 Adam。1次・2次モーメントを全要素保持。
- **embedding（疎）**：現れた行のみ更新。行ごとに `(m,v,step)` を持ち、遅延更新（触れた行だけモーメント更新＋バイアス補正）。weight decay は埋め込みには掛けない設定も選べる（過度な収縮を避ける）。
- LR スケジュール・warmup・early stop（dev の境界 F 値）で収束判定。

### II.5 データパイプライン（`train/`）

```
corpus  : KyTeaフル/部分（6節）→ 文＋境界ラベル・マスク
        （フルは全境界 t∈{0,1} m=1／部分は | - を t, 空白?を m=0）
example :
  1. 正規化(方式a, unicode::normalize)
  2. EGC分割(unicode::egc)
  3. 語彙構築（1回目パス）: コードポイント頻度→閾値未満は語彙に入れずUNK行へ（5.3節）
  4. 衝突処理: EGC内部に境界ありの文は警告スキップ、統計記録（5.2/5.5節）
  5. 例生成: 各境界→窓の(EGC→構成コードポイント行ID列, 端PAD)＋辞書素性F＋t＋m
dataset : シャッフル→バッチ→gather+mean pool で X, 密 F, t, m を作る
```

- 語彙・辞書 AC は学習開始前に確定（1回目パス）。例はインデックス列で保持し、バッチ時に gather（数M例ならメモリ実用域）。
- 辞書 AC は推論と同じ `text/aho_corasick`（EGC単位、正規化済み単語）。学習と推論で素性生成が一致することを保証。

### II.6 PTQ（学習後量子化）と検証

**重みのスケール選定：max-abs、クリップなし**。重みでは外れ値ほど重要（大きい重み＝強い証拠を運ぶ結合）であり、パーセンタイル＋クリップは精度を狙い撃ちで壊す。int16 は範囲が広く max-abs でも分解能は十分：

```
S_e   = max|emb|   / 32767      S_w1  = max|W1|  / 32767
S_wd  = max|Wdict| / 32767      S_w2  = max|w2|  / 32767
各 *_q = round(param / S_*)     （max-abs なのでクリップ不要）
```

**活性のキャリブレーション：パーセンタイル**（こちらは外れ値を切ってよい——稀な飽和は判定反転チェックで捕捉される）：

```
検証データ（またはその一部）で第1層活性（ReLU前）a の分布を収集
S_acc = pct99.99(|a|) / Amax,   Amax = 2^22    （I.1 参照）
```

**エクスポート内容**：ファイルに書くのは `emb_q, w1_q, wdict_q, w2_q`（int16）＋スケール5種（`S_e, S_w1, S_wd, S_w2, S_acc`、double）＋`b1, b2`（double 生値）。`table_q`/`dict_col_q`/`b1_q`/`b2_q` はロード時構築なので**ファイルには入れない**（5.7）。

**検証（判定反転チェック、5.5節）**：dev セットで
```
fp32 の y の符号  vs  int16 推論器の判定  を全境界で比較
反転率と、反転例の |y|（ゼロ近傍か）を集計
```
- 反転が許容超なら：`Qmax` の margin 調整、パーセンタイル見直し、それでも駄目なら QAT へ（5.5節の条件付きフォールバック）。
- 飽和（クリップ）発生数も記録。

### II.7 エクスポート（5.7形式）

`bytes/binary_writer` で 5.7 のフィールド順に書く：ヘッダ行 → Config(w,d,H,num_dicts,unicode_version) → スケール5種（`acc_scale` 含む） → 語彙(V, 昇順codepoint) → `emb_q` → `w1_q` → `wdict_q` → `b1`(double) → `w2_q` → `b2`(double) → 辞書(単語リスト)。**事前計算テーブル・AC・`*_q`変換値は書かない**（ロード時構築、I.4）。

---

## 第III部：学習と推論の一致保証（テスト設計）

int16 推論が fp32 学習と整合することを段階的に検証する。

1. **エンコード一致**：`example` の窓生成と `mlp/vocab` の `encode` が同一の（行ID列, 辞書素性）を出す（同じ正規化・EGC分割・AC）。共通コードを使い差分をテスト。
2. **fp32 リファレンス前方**：学習の前方を単精度で回す `reference_forward(y)` を用意。
3. **量子化誤差境界**：ランダム入力で `|S_acc·acc_q − a_fp32|` と最終 `y` 符号一致率を測る（I.1 の合成が正しければ符号はほぼ常に一致）。
4. **エンドツーエンド**：小コーパスで学習→エクスポート→`MlpBackend` で推論し、dev の F 値が fp32 リファレンスと一致（量子化劣化が想定内）。
5. **KyTea 比較**（5.8節）：同一コーパスで再学習した KyTea と同一テキストで境界比較、精度・速度。
```