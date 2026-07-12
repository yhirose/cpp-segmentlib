# MLPバックエンド モジュール設計

`design.ja.md` 5節で確定した MLP バックエンドの、ファイル分割・クラス境界・依存関係を定める。既存の KyTea バックエンドの規約（immutable な `Model` ＋ 状態を持たない free 関数の `Scorer` ＋ バッファ再利用エンコーダ）にそのまま倣う。

## 0. 全体方針

- **推論（ライブラリ本体の成果物）と学習（開発時ツール）を物理的に分離する**。推論バイナリは BLAS/CUDA/Metal に一切リンクしない（`design.ja.md` 5.9）。
  - 推論：`include/segmentlib/mlp/` + `src/mlp/` → 既存の `segmentlib` ターゲットに入る。
  - 学習：`src/mlp/train/`（公開ヘッダを持たない）→ 別ターゲット `segmentlib_train`。BLAS/CUDA/Metal はここだけがリンクする。
- 名前空間：推論 `segmentlib::mlp`、学習 `segmentlib::mlp::train`。
- 3つの**共有基盤**を新設し、MLP と将来の Vaporetto バックエンドで再利用する（下記1節）。

## 1. 共有基盤（新設。MLP専用ではない）

| モジュール | 名前空間 | 責務 | 動機 |
|---|---|---|---|
| `unicode/egc.h` | `segmentlib::unicode` | UAX #29 の書記素クラスタ分割。UTF-8 → EGC のバイトスパン列＋各EGCの構成コードポイント。付随して grapheme-break プロパティ表（生成データ）を持つ | MLP の原子単位（5.2節）。学習前処理・推論・辞書ロードの全てが依存 |
| `unicode/normalize.h` | `segmentlib::unicode` | 半角→全角の固定テーブル正規化（現 `kytea::normalize` を昇格） | 方式(a)（5.5節）で MLP も同じ正規化を使う。MLP→kytea 名前空間依存を避けるため共有化 |
| `text/aho_corasick.h` | `segmentlib::text` | **実行時構築**の Aho-Corasick（builder + matcher）。キー型はテンプレートパラメータ | MLP の辞書マッチ（EGC単位）。Vaporetto も「フラットリストから実行時に AC 構築」（4.5節）で同じものを要する |
| `bytes/binary_writer.h` | `segmentlib::bytes` | `BinaryReader` の対。LE固定幅整数・NUL終端文字列・生バイト列の書き出し | 5.7形式のモデル書き出し（学習側 exporter が使用） |

**注意（既存コードへの影響）**：
- `unicode/normalize.h` への昇格は、`kytea/char_table.cpp` 内の `normalize` 実装を移設し、`kytea` 側は再エクスポートするだけの小さなリファクタ。既存の KyTea 動作は不変。
- 既存 `kytea/automaton.h` は「モデルファイルから読む read-only AC」で、`text/aho_corasick.h` は「単語リストから構築する AC」。用途が異なるので統合せず別物とする（KyTea の read-only 版はそのまま残す）。

## 2. 推論モジュール（`segmentlib::mlp`）

KyTea の `Model`（保持）／`scorer`（計算）／`CharTable`+`EncodedText`（エンコード）の三分割に対応させる。

```
include/segmentlib/mlp/
  vocab.h        Vocab, EncodedEgc      … CharTable + EncodedText 相当
  precompute.h   PrecomputeTable        … NNUE方式の事前計算テーブル（5.6節）
  dictionary.h   DictMatcher            … EGC単位ACで辞書二値素性を出す（5.4節）
  model.h        Model                  … kytea::Model 相当（immutable, load, 派生構造保持）
  scorer.h       score_boundaries(_into)… kytea::scorer 相当（free関数）
  mlp_backend.h  MlpBackend             … kytea::KyteaBackend 相当
```

### 2.1 `vocab.h` — `Vocab` / `EncodedEgc`

`CharTable` に対応。コードポイント語彙（5.3節）とエンコードを担う。

- **`Vocab`**：コードポイント → 埋め込み行ID。昇順コードポイント配列＋二分探索（5.7 フィールド10）。行0=PAD、行1=UNK。未収録コードポイントは UNK。`row_of(char32_t) -> uint32_t`。
- **`EncodedEgc`**（`EncodedText` 相当、バッファ再利用可）：入力を正規化→EGC分割した結果。
  - `egc_count`：EGC 数 M。
  - 各EGCの構成コードポイント行ID（可変長 → CSR的に `rows` + `egc_starts`）。日中はほぼ1個。
  - `offsets`：サイズ M+1、各EGCの**原文**バイトスパン（単語切り出し用）。
  - （ハイブリッド採用時）頻出EGCの事前計算キー。
- **`encode` / `encode_into`**：`std::expected<…, Error>`。正規化（`unicode::normalize`）→ EGC分割（`unicode::egc`）→ 行ID化。`InvalidUtf8` を返す。

### 2.2 `precompute.h` — `PrecomputeTable`

5.6節の `table[egc][j] = W1_j · v(egc)`。ロード時に `(embedding, W1)` から構築（Model が保持）。

- 頻出EGCについて int32 の `[2w][H]` ブロックを保持（5.6の数値表現：int32テーブル＋int32アキュムレータ）。
- `add_into(acc, egc, j)`：位置 j のブロックを 256次元アキュムレータに加算。
- **フォールバック経路**：非頻出EGCは、構成コードポイント埋め込みの mean を取り `W1_j` を掛けて合成（`embedding` と `W1` を参照）。同じ `add_into` インターフェースの裏で分岐。
- 「頻出」の判定・容量は Config／閾値で決める（~50MB 目安、5.6節）。

### 2.3 `dictionary.h` — `DictMatcher`

5.4節の辞書二値素性。`text::aho_corasick` を EGC 単位で使う。

- ロード時：単語リスト（5.7 フィールド17、**正規化済み**）を EGC 分割し、EGC を interning（辞書内 EgcId）して AC 構築。
- `features_into(EncodedEgc, out)`：各境界について、L/I/R × 長さバケット4 × 辞書数 の**二値**素性（多重マッチは clamp、5.4節）を立てて返す。scorer がこれを `W_dict` 経路に流す。
- 辞書なし（`num_dicts==0`）なら空。

### 2.4 `model.h` — `Model`

`kytea::Model` に対応。immutable、`load`/`load_from_bytes`、`Parts` に parse、const アクセサ。**ファイルにある生パラメータ＋ロード時構築の派生構造**を保持する。

```cpp
struct Config {                       // 5.7 フィールド1-4b
    std::uint8_t char_window = 0;     // w
    std::uint16_t embed_dim = 0;      // d
    std::uint16_t hidden = 0;         // H
    std::uint8_t num_dicts = 0;
    std::uint16_t unicode_version = 0;
};

class Model {
public:
    static std::expected<Model, Error> load(const std::filesystem::path&);
    static std::expected<Model, Error> load_from_bytes(std::span<const std::byte>);

    const Config& config() const noexcept;
    const Vocab& vocab() const noexcept;
    const PrecomputeTable& precompute() const noexcept;  // 派生（ロード時構築）
    const DictMatcher& dict() const noexcept;            // 派生（ロード時構築）
    // 量子化ロード済みの層パラメータ（fallback合成と第2層で使う）
    std::span<const std::int16_t> embedding() const noexcept;  // V×d
    std::span<const std::int16_t> w1() const noexcept;         // H×2w×d
    std::span<const std::int16_t> w2() const noexcept;         // H
    // b1/b2 はロード時にアキュムレータ整数スケールへ変換済み（5.7 フィールド14/16）
    std::span<const std::int32_t> b1_q() const noexcept;
    std::int32_t b2_q() const noexcept;

private:
    struct Parts { Config config; Vocab vocab; PrecomputeTable precompute;
                   DictMatcher dict; /* embedding,w1,w2,b1_q,b2_q,scales */ };
    static Parts parse(bytes::BinaryReader&);   // ヘッダ行は呼び出し側で消費
    Parts parts_;
};
```

- 自動判別（2節の Segmenter）：先頭 `"SegmentLibMLP "` ヘッダ行（5.7）。
- `unicode_version` が実行環境の EGC 分割器と不一致ならロード時に警告（`design.ja.md` 5.7）。

### 2.5 `scorer.h` — `score_boundaries` / `score_boundaries_into`

`kytea::scorer` と同一の形。EGC境界ごとのスコア（int32、`y>0` で境界）を返す。

```cpp
[[nodiscard]] std::vector<std::int32_t>
score_boundaries(const Model&, const EncodedEgc&);

void score_boundaries_into(const Model&, const EncodedEgc&,
                           DictFeatures& scratch,           // 辞書素性の再利用バッファ
                           std::vector<std::int32_t>& out); // 境界数 M-1
```

1境界の計算（5.6節）：`acc=b1_q` → `precompute.add_into` を 2w 回 → アクティブ辞書素性ぶん `W_dict` 列を加算 → ReLU → `w2` と内積 + `b2_q` → 符号。SIMD カーネル（NEON/AVX2/scalar）はこの中に閉じる。

### 2.6 `mlp_backend.h` — `MlpBackend`

`kytea::KyteaBackend` と同一シグネチャ。MLP はタグ推定を持たないので `tokenize` は境界→`Segments`（tags 空）に変換するだけ。

```cpp
class MlpBackend {
public:
    explicit MlpBackend(Model model) noexcept;
    std::expected<Segments, Error>   tokenize(std::string_view) const;
    std::expected<Boundaries, Error> tokenize_boundaries(std::string_view) const;
private:
    Model model_;
    // ホットパス用スクラッチは呼び出しごとに確保 or thread_local（tokenize_all並列に注意）
};
```

### 2.7 `segmenter.h` への統合（既存編集）

- `AnyBackend` に `mlp::MlpBackend` を追加：`std::variant<kytea::KyteaBackend, mlp::MlpBackend>`。
- `Segmenter::load` の自動判別に `"SegmentLibMLP "` シグネチャを追加。
- `load_mlp(path)` を追加。
- `std::visit` ディスパッチは既存のまま（型が増えるだけ、分岐漏れはコンパイル時検出）。

## 3. 学習モジュール（`segmentlib::mlp::train`、`src/mlp/train/`）

推論の公開ヘッダには含めない。別ターゲット `segmentlib_train` としてビルドし、CLI の `train` サブコマンドから呼ぶ。

```
src/mlp/train/
  corpus.{h,cpp}          KyTeaコーパス読込（フル/部分, 6節）→ 注釈付き文
  example.{h,cpp}         文 → 学習例（EGC分割・正規化・語彙構築・衝突スキップ 5.5節）
  dataset.{h,cpp}         ミニバッチ組立、埋め込みgather、mean pooling
  compute_backend.h       ComputeBackend 抽象（GEMM/活性/要素演算/勾配）
    cpu_blas.cpp            Accelerate / OpenBLAS / MKL（リンク切替）
    cuda.cpp                cuBLAS（Linux/Windows, 任意）
    metal.cpp               MPSGraph（macOS, 任意）
  net.{h,cpp}             fp32 順伝播/逆伝播（ComputeBackend に委譲）
  adam.{h,cpp}            Adam（dense層 + 埋め込みの疎更新, 5.9節）
  trainer.{h,cpp}         学習ループ、収束判定、検証
  quantize.{h,cpp}        PTQ int16 化＋スケール合成＋判定反転検証（5.5節）
  exporter.{h,cpp}        5.7形式書き出し（binary_writer 使用）
```

### 3.1 `compute_backend.h` — `ComputeBackend`

行列積・活性化・要素演算・勾配だけを閉じ込める抽象（5.9節）。プラットフォームで実装を差し替え。

```cpp
class ComputeBackend {  // 純粋仮想 or CRTP/variant（実装は起動時に1つ選ぶ）
public:
    virtual void gemm(...) = 0;          // C = alpha*A*B + beta*C
    virtual void relu(...) = 0;
    virtual void relu_backward(...) = 0;
    virtual void axpy(...) = 0;
    // 埋め込み gather/scatter（疎）
    virtual ~ComputeBackend() = default;
};
```

- 学習は fp32。ここで CPU(BLAS)/CUDA/Metal を選ぶ。**推論とは無関係**（推論は int16 手書きSIMD、5.6節）。
- 起動時に1実装を選ぶだけなので `virtual` で十分（推論ホットパスではないため間接呼び出しコストは問題にならない）。

### 3.2 `example.h` の責務（前処理の要）

`design.ja.md` の以下を一手に引き受ける：
- 正規化（方式a）→ EGC分割（`unicode::egc`）
- 語彙構築：コードポイント頻度集計、閾値未満→UNK（5.3節、UNK行の学習を保証）
- 例生成：窓の行ID列（端PAD）＋辞書素性＋ラベル＋マスク
- **衝突処理**：EGC内部に境界ありの文は警告してスキップ、統計を記録（5.2/5.5節）

## 4. 依存関係グラフ

```
                    ┌─────────────── 推論（segmentlib ターゲット）───────────────┐
segmenter.h ──▶ mlp/mlp_backend ──▶ mlp/scorer ──▶ mlp/model ──▶ mlp/vocab
                                        │              │           └─▶ unicode/{egc,normalize,utf8}
                                        │              ├─▶ mlp/precompute
                                        │              └─▶ mlp/dictionary ─▶ text/aho_corasick
                                        └─▶ (SIMDカーネル: NEON/AVX2/scalar)
                                                       model.load ─▶ bytes/binary_reader
                    └───────────────────────────────────────────────────────────┘

                    ┌───────── 学習（segmentlib_train ターゲット, BLAS/CUDA/Metal）─────────┐
cli train ──▶ train/trainer ──▶ train/{net, adam, dataset} ──▶ train/compute_backend
                  │                    └─▶ train/example ──▶ unicode/{egc,normalize}, text/aho_corasick
                  └─▶ train/quantize ──▶ train/exporter ──▶ bytes/binary_writer
                                           （corpus ──▶ 6節フォーマット）
                    └──────────────────────────────────────────────────────────────────────┘
```

- 推論は `unicode/*`・`text/aho_corasick`・`bytes/binary_reader` のみに依存し、BLAS/CUDA を含まない。
- 学習と推論の唯一の接点は **5.7 モデルファイル**（train/exporter が書き、mlp/model が読む）。
- 共有基盤（`unicode/*`, `text/aho_corasick`, `bytes/*`）は両者から使われる。

## 5. CMake ターゲット

| ターゲット | 内容 | 依存 |
|---|---|---|
| `segmentlib`（既存） | 推論。kytea + mlp + 共有基盤 | zstd（Vaporetto用, 既存）。**BLAS/CUDA なし** |
| `segmentlib_train`（新規, optional） | 学習。`src/mlp/train/*` | プラットフォームBLAS（Accelerate/OpenBLAS）、任意で CUDA/Metal |
| CLI（既存 `src/cli`） | `predict` は `segmentlib`、`train` は `segmentlib_train` にリンク | |

`segmentlib_train` は `option(SEGMENTLIB_BUILD_TRAINING ...)` でオプトイン。BLAS 実装は `SEGMENTLIB_BLAS=Accelerate|OpenBLAS|MKL`、GPU は `SEGMENTLIB_GPU=none|cuda|metal` で選択。

## 6. 実装順序（提案）

1. `unicode/egc.h`（+ break プロパティ表生成）— 全ての起点。単体テスト（UAX #29 の GraphemeBreakTest.txt で検証）
2. `unicode/normalize.h` 昇格 ＋ `bytes/binary_writer.h` ＋ `text/aho_corasick.h`（共有基盤）
3. `mlp/vocab.h`（Vocab/EncodedEgc）
4. 学習の最小経路：`train/{corpus, example, dataset, compute_backend(cpu_blas), net, adam, trainer}` → まず小コーパスで収束を確認
5. `train/{quantize, exporter}` → 5.7 モデル書き出し
6. `mlp/{precompute, dictionary, model, scorer, mlp_backend}` → 推論。KyTea と同一テキストで精度・速度比較（5.8節）
7. `segmenter.h` 統合、CLI 配線
8. 最適化：int16 再量子化テーブル（5.6節）、SIMD カーネル、GPU バックエンド（必要なら）
```