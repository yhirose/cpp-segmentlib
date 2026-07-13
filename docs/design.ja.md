# 設計ドキュメント

## 1. 概要

KyTea / Vaporetto と同じ「点推定（pointwise prediction）」方式を採用した、C++の分かち書きライブラリ。
文字境界ごとに独立した二値分類（区切る/区切らない）を分類器で行う。
ラティス＋Viterbiによる最小コスト法（MeCab等）とは異なり、辞書コストの設計や動的計画法を必要としない。

**基本アーキテクチャ方針**：公開APIは単一（4節参照）にし、その内部に複数の**バックエンド**を差し替え可能な形で持たせる。

- **バックエンド1：KyTea互換**（3節）— KyTeaの学習済みモデルをそのまま読み込み、KyTeaと同一の特徴抽出・線形SVM分類器で推論する。最初の目標。
- **バックエンド2：独自MLPモデル**（4節、最終目標）— 線形SVMではなく、独自に設計したMLP（多層パーセンサ）を分類器として使うバックエンド。将来的な精度向上の余地として位置づける。

**コーパスフォーマットは常にKyTeaのコーパス形式を使う**（5節）。KyTea互換バックエンド・独自MLPバックエンドのいずれも、同じKyTeaコーパス形式（フル/部分アノテーション）を入力として想定する。学習データの形式を統一することで、2つのバックエンドの精度をフェアに比較できるようにする狙いもある。

- 分割精度はKyTeaと同等を目標とする（KyTea互換バックエンドについては、KyTeaの実出力とのバイト単位一致を検証する）。
- 高速化はVaporetto同様（本ライブラリのバックエンドとしてではなく、外部の比較対象としての意味で）、特徴抽出のパターンマッチ（Aho-Corasick／Double-Array）と重み配列のメモリレイアウトで狙う（詳細は各バックエンドの節、および別ドキュメントで扱う）。

## 2. アーキテクチャ：バックエンドの抽象化

複数バックエンドを同一APIの背後に隠すため、`Segmenter`はバックエンドの実装詳細を知らない薄いディスパッチャとする。

バックエンドの集合は「KyTea互換／独自MLP」という**あらかじめ決まった小さな閉じた集合**であり、外部プラグインとして動的に追加する要件は今のところない。そのため仮想関数によるオープンな拡張機構（`virtual`＋ヒープ確保）ではなく、**`std::variant` + `std::visit`によるクローズドな多態性**を第一候補とする。vtableの間接呼び出しやバックエンドオブジェクトの個別ヒープ確保を避けられ、対応していないバックエンドの分岐漏れはコンパイル時に検出できる。

```cpp
class KyTeaBackend { /* 3節 */ 
public:
    std::expected<Segments, Error> tokenize(std::string_view text) const;
    std::expected<Boundaries, Error> tokenize_boundaries(std::string_view text) const;
};
class MlpBackend { /* 4節（今後詳細化） */
public:
    std::expected<Segments, Error> tokenize(std::string_view text) const;
    std::expected<Boundaries, Error> tokenize_boundaries(std::string_view text) const;
};

using AnyBackend = std::variant<KyTeaBackend, MlpBackend>;

class Segmenter {
public:
    // モデルファイルの内容から形式を自動判別してロードする
    static std::expected<Segmenter, Error> load(const std::filesystem::path& model_path);

    std::expected<Segments, Error> tokenize(std::string_view text) const {
        return std::visit([&](auto const& b) { return b.tokenize(text); }, backend_);
    }
    std::expected<Boundaries, Error> tokenize_boundaries(std::string_view text) const {
        return std::visit([&](auto const& b) { return b.tokenize_boundaries(text); }, backend_);
    }

    // 多数の入力を並列に分かち書きする（result[i]がtexts[i]に対応、threads==0でハード
    // ウェア並列数）。入力は互いに独立でモデルはimmutableなのでコア数にほぼ比例して
    // スケールする（9.4節：M1 Pro 8スレッドで単スレッド比5.6×）。
    std::vector<std::expected<Segments, Error>>
    tokenize_all(std::span<const std::string_view> texts, unsigned threads = 0) const;

private:
    AnyBackend backend_;
};
```

**モデル形式の自動判別**：`Segmenter::load()`はファイル先頭のシグネチャを見て自動的にバックエンドを選ぶ（KyTeaモデルは`"KyTea "`で始まるヘッダ行を持つ、それ以外は独自MLP形式として扱う、4.7節）。明示的にバックエンドを指定したい場合向けに`load_kytea(path)` / `load_mlp(path)`も用意し、`load()`はその薄いラッパーとする。

各バックエンドクラスは`tokenize`/`tokenize_boundaries`という同一シグネチャさえ満たせばよく、内部の特徴抽出・分類器・モデルパーサは完全に独立して実装できる。2つのバックエンドの共通点は「同じ`Segments`/`Boundaries`型を返すこと」だけであり、KyTeaコーパス形式で学習した独自MLPモデルを将来追加する際も、もう一方のバックエンドのコードには一切手を入れずに済む設計を意図している。

## 3. KyTea互換バックエンド

- KyTeaが出力するモデル（`train-kytea`で学習されたモデル）をそのまま読み込む。
- 素性は KyTea と同じ3種：文字n-gram、文字種n-gram、辞書由来の単語素性。
- 学習エンジン自体（LIBLINEAR相当）は本ライブラリのスコープ外とし、**推論のみ**をサポートする（10節で自前実装／外部連携いずれも行わないことを最終決定）。
- モデルのバイナリフォーマットは、KyTeaのモデルファイルを直接パースする（変換ツールを挟む方式は取らない）。

### 3.1 特徴量抽出（KyTea互換・要忠実再現）

分類器（LIBLINEARの線形SVM）自体は汎用的だが、**素性文字列の生成ロジックと文字種分類はKyTea独自の実装**であり、モデルとの互換性を保つにはここを一字一句忠実に再現する必要がある（KyTea本体のソース `string-util.cpp` / `kytea.cpp` を参照して確認済み）。

**文字種分類（`StringUtil::CharType`）**

6種類のタイプを、UTF-8の1文字をUnicodeコードポイントに変換した上で以下の範囲判定で決定する。判定順序も含めて重要（ローマ字→ひらがな→カタカナ→数字→漢字→その他の順で評価する）。

| タイプ | 記号 | Unicode範囲（概要） |
|---|---|---|
| ROMAJI | `R` | `0x41-0x5A`, `0x61-0x7A`（半角英字）, `0xFF21-0xFF3A`, `0xFF41-0xFF5A`（全角英字） |
| HIRAGANA | `H` | `0x3040-0x3096` |
| KATAKANA | `T` | `0x30A0-0x30FF`（ただし`0x30FB`中点を除く）, `0xFF66-0xFF9F`（半角カナ） |
| DIGIT | `D` | `0x30-0x39`（半角数字）, `0xFF10-0xFF19`（全角数字） |
| KANJI | `K` | `0x3400-0x4DBF`, `0x4E00-0x9FFF`, `0xF900-0xFAFF`, `0x20000-0x2A6DF`, `0x2A700-0x2B73F`, `0x2B740-0x2B81F`, `0x2F800-0x2FA1F` |
| OTHER | `O` | 上記以外すべて |

KyTea本体はUTF-8/EUC/SJISの3エンコーディングに対応しているが、本ライブラリはUTF-8のみを対象とする。

なお、KyTeaの`findType`は4バイトUTF-8（コードポイント≧U+10000、CJK拡張B以降）のコードポイント計算にバグ（`<<18`が2箇所ある）があり、それらの漢字を誤分類する。本ライブラリは正しいコードポイントから分類するため、この稀なケースでKyTeaと結果が分かれうる（実装コメント・8.6節に記録）。BMP内の通常の日本語テキストでは一致する。

**入力の正規化（`normalize`、要KyTea互換）**

推論時、KyTeaは入力文字列を`surface`（原文）と`norm`（正規化）に分け、素性計算（文字n-gram・文字種n-gram）はすべて`norm`に対して行う（`RawCorpusIO`: `norm = normalize(surface)`）。正規化は`string-util-map-utf8.h`の固定テーブル（約110エントリ）で、**半角英数字・記号を全角へ畳む**（`a→ａ`, `0→０`, `(→（`, 半角カナ記号`｢｣→「」`等）ものである。出力の単語表層は`surface`（原文のバイト列）から切り出すが、境界判定に使うスコアは`norm`から計算される。本ライブラリもこの固定テーブルを移植し、**UTF-8デコード→コードポイント正規化→インターン**の順で`norm`相当のID列を作る（`CharTable::encode`、実装・検証済み）。

**素性文字列のフォーマット**

境界位置を基準に、窓幅（デフォルト`charw=3`、`typew`同様に設定可能）の範囲で以下3種の素性文字列を生成し、モデルの辞書でIDに変換して線形分類器に渡す。

| 素性種別 | プレフィックス形式 | 例 |
|---|---|---|
| 文字n-gram | `"X" + 相対位置` + 文字列そのもの | `X-2`, `X-1`, `X0`, `X1` |
| 文字種n-gram | `"T" + 相対位置` + 文字種記号の列 | `T-1`, `T0`, `T1` |
| 辞書由来の単語素性 | `"D" + 辞書インデックス + (L\|I\|R) + マッチ長` | `D0L1`（辞書0番、左端一致、長さ1）, `D1R3` |

`D`素性の`L`/`I`/`R`は、境界に対する辞書エントリの位置関係（Left端／Inside中間／Right端）を表す。

**素性IDのマッピングはハッシュではなくモデル内蔵の辞書**

KyTeaのモデルは `ids_`（特徴量文字列→ID）を学習時に構築し、モデルファイルに埋め込んでいる（`KyteaModel::mapFeat`）。つまり推論側は独自にハッシュ関数を実装するのではなく、**モデルファイルから素性辞書をそのまま読み込み、その辞書を引いてIDを求める**必要がある。未知の素性文字列（学習時に出現しなかったもの）はモデルに存在しないため、その素性は単純にスキップする（KyTeaと同じ挙動）。

上記のいずれか一つでもKyTeaと異なる実装（Unicode範囲の境界がずれる、素性文字列のフォーマットが違う、窓幅のデフォルトが違う等）をすると、同じモデルを読み込んでも別の分類結果になるため、この節の内容はテストで厳密に検証する（KyTeaの実出力とのバイト単位の一致を確認するテストケースを用意する）。

### 3.2 モデルファイルフォーマット（KyTea本体のソースで検証済み）

KyTea本体のソース（`model-io.cpp`, `model-io-binary.h`, `model-io-text.h`, `kytea.cpp`, `dictionary.h`）を直接確認した結果、以下の構成であることを確認した。

**ヘッダ行**

```
KyTea <version> <T|B> <encoding>
```
例：`KyTea 0.4.0 B utf8`。`version`は`MODEL_IO_VERSION`（量子化ビルドでは`"0.4.0"`、非量子化ビルド`DISABLE_QUANTIZE`では`"0.4.0NQ"`）。フォーマット文字は`T`=テキスト、`B`=バイナリ。バージョン文字列が一致しない場合はKyTea自身がエラーにするため、本ライブラリも`0.4.0`系のみを対象とする。このヘッダ行の`"KyTea "`シグネチャは、2節で述べたバックエンド自動判別にも使う。

**ファイル全体のセクション順序**（`Kytea::writeModel`/`readModel`）

1. **Config**：`do_ws`, `do_tags`, `numTags`, `charWindow`, `charN`, `typeWindow`, `typeN`, `dictionaryN`, バイアス有無, `epsilon`, `solverType`、および文字マップ（`StringUtil::serialize()`）
2. **分かち書きモデル**（`wsModel_`）：`KyteaModel` 1個
3. **タグモデル**：`numTags`個ぶん、各々「単語リスト（グローバルタグ候補）＋`KyteaModel`」の組
4. **単語辞書**：`Dictionary<ModelTagEntry>`（ユーザ辞書・システム辞書。最大8個までまとめて1つのAho-Corasickオートマトンに統合）
5. **サブワード辞書**：`Dictionary<ProbTagEntry>`（読み推定などの部分文字列確率用）
6. **言語モデル（LM）**：`numTags`個ぶん、`KyteaLM`（サブワードのn-gram言語モデル）

**`KyteaModel`（分類器1個）のシリアライズ**

- クラス数（`int32_t`。0または2未満なら「モデルなし」を意味しそこで終了）
- ソルバー種別（`char`1バイトの列挙値。`L2R_LR`, `L2R_L2LOSS_SVC_DUAL`, `L2R_L2LOSS_SVC`, `L2R_L1LOSS_SVC_DUAL`, `MCSVM_CS`, `L1R_L2LOSS_SVC`, `L1R_LR`, `L2R_LR_DUAL`の8種。デフォルトは`L2R_L2LOSS_SVC_DUAL`＝**L2正則化L2-lossの線形SVM（dual）**）
- 各クラスのラベル（`int32_t`×クラス数）
- バイアス有無（`bool`）
- `multiplier`（`double`。量子化された重みを実数に戻すためのスケール係数）
- `FeatureLookup`（後述）

**`FeatureLookup`＝推論用に事前コンパイルされた素性→重みの直接マッピング**

KyTeaは学習時の素性文字列→ID辞書（`ids_`/`names_`）をそのままモデルファイルに書き出すのではなく、**推論用に最適化された`FeatureLookup`構造を別途構築して書き出す**（`buildFeatureLookups()`）。中身は次の7要素：

| フィールド | 型 | 内容 |
|---|---|---|
| `charDict` | `Dictionary<FeatVec>` | 文字n-gram → クラス別重みベクトル |
| `typeDict` | `Dictionary<FeatVec>` | 文字種n-gram → クラス別重みベクトル |
| `selfDict` | `Dictionary<FeatVec>` | 辞書由来の自己文字列素性 → クラス別重みベクトル |
| `dictVector` | `FeatVec` | 辞書関連の追加固定長重み |
| `biases` | `FeatVec` | バイアス項 |
| `tagDictVector` / `tagUnkVector` | `FeatVec` | タグ推定関連の重み |

`Dictionary<FeatVec>`は**Aho-Corasickオートマトン**（`DictionaryState`の配列：`failure`リンク、`gotos`（文字→次状態、ソート済みで二分探索）、`output`（この状態で確定する素性のインデックス列））であり、KyTea自身が特徴抽出のパターンマッチに既にAho-Corasickを使っている（Vaporettoの高速化はこれをDouble-Array化したもの、という位置づけが裏付けられた）。

**`Dictionary<Entry>`のバイナリレイアウト（`model-io-binary.h`の`writeDictionary`/`readDictionary`テンプレートで確認済み。`charDict`/`typeDict`/`selfDict`（`Entry=FeatVec`）だけでなく、単語辞書（`Entry=ModelTagEntry`）・サブワード辞書（`Entry=ProbTagEntry`）にも共通の枠組み）**：

```
辞書数           : unsigned char（1バイト。0なら「辞書なし」を意味し以降のフィールドは書かれない）
状態数           : uint32_t
[状態数ぶん] 各状態（DictionaryState）:
    failure      : uint32_t              // 失敗遷移先の状態インデックス
    gotos数      : uint32_t
    [gotos数ぶん]
        文字      : KyteaChar (=unsigned short, 2バイト)
        遷移先    : uint32_t
    output数     : uint32_t
    [output数ぶん]
        値        : uint32_t              // この状態で確定するエントリのインデックス
    isBranch     : bool（1バイト）
エントリ数        : uint32_t
[エントリ数ぶん] writeEntry<Entry>(...)     // Entry型ごとに異なる（下記）
```

`gotos`は文字（`KyteaChar`）でソートされており、読み込み側は`DictionaryState::step()`で二分探索する（3.1節の`gotos.begin()`/`gotos.end()`を使った二分探索の実装がそのままC++に移植できる）。

`writeEntry<Entry>`はEntry型ごとに異なる（いずれも`model-io.cpp`のテンプレート特殊化で確認済み）：

- `Entry = FeatVec`（`charDict`/`typeDict`/`selfDict`用）：`uint32_t`の要素数 → `FeatVal`（既定`int16_t`）× 要素数。**先頭に素性文字列などの識別子は一切含まれない**（対応する文字列は`DictionaryState`の遷移パス自体が表現しているため、`Entry`は純粋に重みベクトルのみを持つ）。
- `Entry = ModelTagEntry`（単語辞書、`dict_`用）：`word`（`KyteaString`、下記）→ タグレベルごとに（タグ候補配列＋各候補のタグ辞書所属ビット `unsigned char`）→ `inDict`（`unsigned char`、辞書所属ビットマスク）→ タグレベルごとの`KyteaModel`（3.2節の形式で再帰的に、`0`ならモデルなしを意味する`int32_t 0`のみ）。
- `Entry = ProbTagEntry`（サブワード辞書、`subwordDict_`用）：`word` → タグレベルごとに（タグ候補配列＋各候補の確率`double`）。

`KyteaString`（＝可変長文字列）のバイナリ表現は`writeString`（`GeneralIO`基底クラス）で、**NUL終端のバイト列**（内容＋`\0`の1バイトを合わせて書き込む）として保存される（`writeString(const std::string & str)`が`str.length()+1`バイト書くことから確認済み）。長さプレフィックスは持たず、読み込み側は`\0`まで読み進める実装になっている。

**`KyteaChar`の型は`unsigned short`（2バイト・符号なし）**であり、KyTea内部では文字はUTF-8のまま保持せず、いったんこの2バイト整数の内部表現にマッピングしてから扱っている（`StringUtil`が文字列⇔`KyteaChar`列の相互変換を担当）。Aho-Corasickオートマトンの遷移は`KyteaChar`単位で構築されているため、**UTF-8の生バイト列ではなく、モデルに埋め込まれた文字→2バイト整数のマッピング（下記）を使ってオートマトンを辿る必要がある**。

**`KyteaChar`は固定のUnicodeコードポイントではなく、モデルごとに学習時の文字出現順で採番される「文字のインターン表」のID**（`string-util.cpp`の`StringUtilUtf8::mapChar`/`serialize`/`unserialize`を直接確認して判明）。

- `mapChar(str, add)`：UTF-8の1文字ぶんの文字列`str`を受け取り、`charIds_`（`std::map<string, KyteaChar>`）で既知ならそのIDを返す。未知かつ`add=true`なら`charTypes_.size()`（＝現在の登録数）を新しいIDとして採番し、`charIds_`/`charTypes_`（3.1節の文字種）/`charNames_`（元のUTF-8文字列）に追記する。
- **ID `0`は空文字列用の予約済みセンチネル**：`unserialize()`は必ず最初に`mapChar("")`を呼び、ID`0`を予約してから本体の文字を読み込む。したがって実際の文字は**ID `1`から**始まる。
- **`serialize()`は全登録文字を単純に連結した1つのUTF-8文字列を返す**（`charNames_[1]`から`charNames_[size()-1]`までを順に`ostringstream`に流し込むだけ）。つまり2.2節の「文字マップ」フィールドの中身は、**学習時にモデルが見た全ての異なり文字を、ID順（＝初出順）に並べて連結しただけの1本のUTF-8文字列**である。
- `unserialize(str)`側は、`mapChar("")`でID`0`を予約した後、`mapString(str)`で`str`を先頭から1文字ずつUTF-8デコードしながら`mapChar()`を呼ぶ。新規モデル読み込み時はテーブルが空なので、`str`中に現れる文字の並び順どおりに`1, 2, 3, ...`という連番IDが再構築される（学習時に割り当てられたIDと一致する）。

**実装上の要点**：本ライブラリのモデルローダは、Config内の「文字マップ」文字列を読み込んだら、それをUTF-8として先頭から1文字ずつデコードし、出現順に`1`始まりのIDを割り振って `char(UTF-8) → KyteaChar` の対応表（および必要なら逆引き表）を構築する必要がある。入力テキストを推論する際も、この対応表を引いてUTF-8文字を`KyteaChar`列に変換してからAho-Corasickオートマトンを辿る。

**未知文字（学習語彙にない文字）の扱い（`string-util.cpp`/`kytea.cpp`で確認、検証済み）**：KyTea本体は推論時、入力を`mapString`で`KyteaChar`列に変換するが、`mapString`は`mapChar(str, add=true)`を既定で呼ぶため、**未知文字には`charTypes_.size()`（＝学習時最大ID＋1以降）の新規IDを動的に採番する**（`string-util.cpp:113-127`）。本ライブラリは未知文字を一律`kNoChar`（`0`）に落とす。両者はID割り当てが異なるが、**分かち書きの出力は完全に等価**である。理由：`calculateWS`（`kytea.cpp:885-919`）のWS素性は3種すべてがAho-Corasickマッチ（文字n-gram・文字種n-gram・辞書）で、いずれも**学習時ID `1..K`でキー付けされたオートマトン**を辿る。未知文字のIDは、`0`（本ライブラリ）でも`K+1`（KyTea）でも**どの遷移にも存在しない**ため、文字n-gram・辞書マッチには一切寄与できない（等価）。文字種n-gramへの寄与は、文字種を**コードポイントから直接分類**（`CharTable::encode_into`）しており文字IDに依存しないため影響を受けない（KyTeaも`charTypes_[id]=findType`で同じ型を得る）。**実測でも裏付け済み**：未知の絵文字・BMP記号（😀😱ℵℶℷ★☆♠🍣ⅠⅡⅢ等）で`kytea -notags`とバイト単位一致（golden fixtureに収録）。**ただし4バイトUTF-8のCJK拡張B漢字（例：𠮷 U+20BB7）は別軸**——これは未知文字処理ではなく3.1節の`findType`バグ（正しくKanji分類する本ライブラリと、誤分類するKyTeaで型n-gramが分岐する）による意図的な差異。

**重要：推論は「文字列→ID→重み配列引き」ではなく「Aho-Corasickでマッチした瞬間に重みベクトルを直接得る」設計**。本ライブラリのモデルローダも、素性文字列を独自に生成してハッシュ／ID変換するのではなく、**この`FeatureLookup`（3つのAho-Corasickオートマトン＋4つの重みベクトル）をそのままファイルからパースし、同じ構造で持つ**のが正しい実装方針になる。

**重みの型（`FeatVal`）はデフォルトで`int16_t`量子化**

```cpp
#if DISABLE_QUANTIZE
    typedef double FeatVal;
    typedef double FeatSum;
#else
    typedef int16_t FeatVal;   // デフォルト
    typedef int32_t FeatSum;
#endif
```

配布されている学習済みモデルは通常量子化ビルド（`FeatVal = int16_t`）で作られている。読み込み時は`int16_t`の重みを`multiplier`（`double`）倍して実数の重みに戻す。本ライブラリはまず**量子化モデル（`int16_t`）の読解を最優先**とし、非量子化モデルはヘッダのバージョン文字列（`"0.4.0NQ"`）で判別して将来的に対応する。

### 3.3 推論アルゴリズム（単語分割、`Kytea::calculateWS`を直接確認して復元）

`kytea.cpp`の`calculateWS`を読み、境界ごとのスコア計算から最終的な分割判定までの流れを正確に復元した。**本ライブラリの`KyTeaBackend::tokenize`/`tokenize_boundaries`が再現すべき計算そのもの**。

文の文字数を`N`とすると、境界は`N-1`個ある（各文字の直後、最後の文字を除く）。各境界`i`（`0 <= i < N-1`）についてスコア`score[i]`を次の順で積み上げる：

1. **初期値**：`score[i] = biases[0]`（`FeatureLookup::biases_`の先頭要素。全境界で共通の定数）
2. **文字n-gramスコアの加算**：正規化済み文字列（`sent.norm`、辞書による表記正規化後の文字列）に対して`charDict`（Aho-Corasick）でマッチした全ての文字n-gramについて、`addNgramScores`のロジックで加算する。1つのn-gramマッチは、その出現位置を中心に**窓の中に入る複数の境界に同時に寄与**する（`FeatVec`は窓幅`window*2`分のスコアを1本のベクトルとして持ち、マッチ位置`pos`から`base_pos = pos - window`を起点に`score[base_pos + j] += vec[j]`（`j`は有効範囲にクリップ）という形で分配される）。
3. **文字種n-gramスコアの加算**：文字列を文字種記号の列（3.1節の`R`/`H`/`T`/`D`/`K`/`O`）に変換したものに対して、`typeDict`で同様に加算する。
4. **辞書由来（D素性）スコアの加算**：`dict_->match(sent.norm)`で単語辞書とのAho-Corasickマッチを取り、`addDictionaryScores`で加算する。ここでのインデックス計算は次の通り（`len=score.size()`、`max=config.getDictionaryN()`、マッチした語の文字長`wlen`、`lablen=min(wlen,max)-1`）：
   - マッチ語の**左端**の境界（`end-wlen`番目、ただし`end>=wlen`の場合のみ）に `dictVector[辞書番号*dictLen + (end-wlen)*3*max + lablen*3 + 0]` を加算（＝3.1節の`D<辞書番号>L<lablen+1>`に相当）
   - マッチ語の**内部**の各境界（`end-wlen+1 <= k < end`）に `... + k*3*max + lablen*3 + 1` を加算（＝`D<辞書番号>I<lablen+1>`）
   - マッチ語の**右端**の境界（`end`番目、ただし`end != len`の場合のみ）に `... + end*3*max + lablen*3 + 2` を加算（＝`D<辞書番号>R<lablen+1>`）
   - この式により、3.1節で説明した`D0L1`/`D0I2`/`D1R3`のような素性文字列と`dictVector`内のオフセットの対応関係が一意に確定する。
5. **ハードな制約（`-wsconst`相当）の上書き**：`config.getWsConstraint()`に指定された文字種記号が、隣接する2文字の文字種が同一のケースに含まれる場合、その境界のスコアを強制的に「境界なし」側（非確率モデルでは`-100`、確率モデルでは`0`）に上書きする。数字の連続を分割しない、といった用途のハードルール。
6. **最終スコア**：`wsConfs[i] = score[i] * wsModel_->getMultiplier()`（`multiplier`は3.2節の量子化スケール係数。量子化されていれば整数の`score`を実数に戻す）
7. **境界判定**：`wsConfs[i] > confidence`（デフォルト`confidence = 0`）なら境界あり、そうでなければ境界なし。**閾値0が既定**であることを`KyteaSentence::refreshWS`の実装で確認済み（`myConf > confidence`という単純な不等号比較）。
8. （オプション）ソルバーが確率モデル系（ロジスティック回帰系）の場合は、最後に`wsConfs[i] = 1/(1+exp(-|wsConfs[i]|))`でシグモイド変換して確率として提示するが、境界判定自体は手順7の符号で決まるため、この変換は表示用途であり分割結果そのものには影響しない。

つまり実装すべき最小限の推論ロジックは、**「biasを初期値に、charDict・typeDict・dictVectorの3種類のAho-Corasickマッチスコアを加算し、multiplierをかけて0と比較する」**という単純な線形和であり、SVMの「学習」部分（LIBLINEAR）を一切実装せずとも、この加算処理だけで推論（`tokenize_boundaries`）が再現できる。

## 4. 独自MLPバックエンド（最終目標）

3節の線形SVMに代えて、独自に設計したMLP（多層パーセプトロン）を分類器として使うバックエンド。分割方式そのものは3節と同じ**ポイントワイズ二値分類**（各境界候補について「区切る/区切らない」を独立に判定）を踏襲し、公開APIも同じ`tokenize`/`tokenize_boundaries`を満たす。KyTea/Vaporettoが手で列挙していた素性（文字n-gram・文字種n-gram・辞書素性）の**交互作用を、窓内の埋め込み表現＋隠れ層に自動獲得させる**のが狙い。

学習データはKyTeaコーパス形式（5節）をそのまま使い、2バックエンドを同一データでフェアに比較する。学習エンジン（順伝播・逆伝播・最適化）は本ライブラリで新規実装する。

### 4.1 設計思想と、KyTea/Vaporettoとの対比

| | KyTea（3節）／Vaporetto（外部比較のみ、本ライブラリのバックエンドではない） | 本MLPバックエンド（4節） |
|---|---|---|
| 分類方式 | ポイントワイズ二値分類 | 同左 |
| 分類器 | 線形SVM（重みの線形和） | MLP（非線形・多層） |
| 素性 | 文字n-gram・文字種n-gram・辞書素性を**手で設計・列挙** | 窓内の埋め込みを連結し、**交互作用をネットワークが学習** |
| 文字種特徴 | ヒューリスティックな6種分類（3.1節）を明示的に使用 | **使わない**（ヒューリスティック素性設計から解放されるのがDLの利点。精度不足なら将来 Unicode General Category 等の導入を再検討） |
| 原子単位 | コードポイント単位の「文字」 | **EGC（書記素クラスタ）単位**（4.2節） |
| 語彙・OOV | モデル内蔵の素性辞書、未知素性はスキップ | 埋め込みは**コードポイント語彙**、EGCは合成で表現し原理的にOOVなし（4.3節） |

**文字種特徴を明示的に持たない**のは意図的な設計判断である。KyTea/Vaporettoの文字種n-gram（`H`/`K`/`T`/`R`/`D`/`O`）は完全にヒューリスティックであり、こうした素性設計を人手で行わなくてよいのがDLアプローチの利点と位置づける。未知・低頻度への汎化が問題になった場合に限り、General Unicode Property（General Category 等）を補助入力として導入する余地を残す。

### 4.2 原子単位・境界候補・窓の数え方 ＝ EGC

分類の原子単位を、コードポイントではなく **EGC（Extended Grapheme Cluster、UAX #29）** とする。理由と帰結：

- **境界候補はEGCの隙間のみ**。EGCの内部（例：`か`+濁点`が` = U+304B U+3099、絵文字ZWJ連結 `👨‍👩‍👦` = 6コードポイント）では決して区切らない。EGC境界はコードポイント境界の部分集合であり、そこ以外は常に「区切らない」でよい（言語的にも妥当）。KyTeaコーパスの正解境界がすべてEGC境界と一致すること（結合列の内部に正解境界が来ないこと）は学習データで実測確認する。
- **窓は「EGCの個数」で数える**。コードポイント個数で窓を測ると、`👨‍👩‍👦` 1個だけで左右window=5の窓を食い潰し、隣の実単語にすら届かない。EGC単位なら `👨‍👩‍👦` は1トークンとして扱え、残りの枠を前後の実際の語に使える。窓の予算を構成部品の多寡に浪費しない設計であり、結合列・タイ語の正書法音節・ハングル合字などすべてに効く。

文の EGC 列を `e[0..M-1]` とすると、境界候補は `M-1` 個（各 EGC の直後、末尾を除く）。各境界 `i`（`0 <= i < M-1`）について独立に二値分類する。

### 4.3 EGCの表現 ＝ 構成コードポイントからの合成的埋め込み

各EGCをそのまま語彙IDに引く（フラットなEGC-id埋め込み）方式は採らない。異なりEGC数は結合列・異体字セレクタ・肌色修飾子などで膨らみ、タイ語・ミャンマー語で語彙が肥大しOOVが増える（語彙爆発）。代わりに、**EGCを構成コードポイント列に分解し、コードポイント埋め込みをpoolingして1本のEGCベクトルを合成する**。

```
EGC → 構成コードポイント列に分解
  各コードポイント → 埋め込み（語彙は「学習時に出現したコードポイント」or バイト256。小さく有界）
  → pooling → EGCベクトル（次元 d）
```

**語彙構築には頻度閾値を設ける**（例：出現2回未満のコードポイントは語彙に入れず UNK に落とす）。全出現コードポイントを語彙に入れると UNK 行（4.7節の行1）に勾配が一度も流れず、推論時の未知コードポイントが未学習の初期値ベクトルを引いてしまう。低頻度コードポイントを訓練中 UNK として流すことで、UNK 埋め込みが「稀な文字の平均的な振る舞い」を学習する。

この二層構造の要点は、**予測の単位（EGC）と、埋め込みの語彙（コードポイント）を分離する**ことにある。

| | 原子単位（予測・窓） | 埋め込みの語彙 |
|---|---|---|
| コードポイント方式 | codepoint | codepoint-id 直引き |
| フラットEGC方式 | EGC | EGC-id 直引き（← 語彙爆発） |
| **本方式（合成的EGC）** | **EGC** | **codepoint を pooling** |

利点：

- **語彙爆発が起きない**：埋め込みテーブルの語彙はコードポイント（数千〜1万、有界）またはバイト（256）で固定。タイ語・ミャンマー語の音節も base＋結合記号に自然分解される。
- **OOV崖が消える**：未知EGCも構成コードポイントが既知であればベクトルが合成できる。**EGC単位のUNKは原理的に不要**（コードポイント語彙のUNKは上記の頻度閾値により残るが、フラットEGC方式のUNKよりはるかに稀にしか踏まない）。
- **ヒューリスティック素性ゼロ**：文字種テーブルのような人手設計は一切なし。「部品から合成を学ぶ」というDL的にクリーンな解。

日本語・中国語は 1 EGC ≒ 1 コードポイントがほとんどなので、poolingは大半が恒等に縮退し、実質「コードポイント埋め込み」として振る舞う。結合列・タイ語・ミャンマー語・絵文字でのみpoolingが実効的に働く。

（任意）頻出EGCには直引きのEGC-id埋め込みを併用し、それ以外を合成的表現で賄う**ハイブリッド**も選べる（頻出は1引き、稀なものは合成でOOV吸収）。pooling方式（mean / 小encoder 等）・次元・ハイブリッドの要否は4.4節のネットワーク構成で詰める。

### 4.4 ネットワーク構成（確定）

「窓内の各EGCベクトル＋辞書素性を連結し、隠れ1層のMLPに通して二値判定する」ポイントワイズ分類器。**構成は速度要件（4.6節の第1層事前計算）から逆算して決定した**。第1層より前に非線形を置かないことが速度の成立条件であり、pooling=mean・隠れ1層はその帰結である。

```
境界 i について:
  窓 = e[i-w+1 … i+w] の EGC 列（左右各 w、計 2w 個。端は PAD トークン）
  各 EGC → 4.3節の合成的埋め込み（次元 d、pooling は mean）
  f_dict = 辞書マッチ二値素性（下記）
  h = ReLU( W1 · concat(2w × d) + W_dict · f_dict + b1 )    # 隠れ層 1 層
  y = w2 · h + b2                                            # スカラー
  y > 0 なら境界（sigmoid(y) > 0.5 と等価。sigmoid は学習時の損失計算にのみ使う）
```

**確定値**：

| 項目 | 値 | 根拠 |
|---|---|---|
| 窓幅 `w` | **5**（左右各5、計10 EGC） | 第1層事前計算方式では w 拡大のコストが「表引き加算1回/EGC」と線形で、ほぼタダ（素朴な行列積なら 2w·d×H が効くが、その制約が消える）。窓を広げて長い語の証拠を拾い、辞書なし時の精度低下も緩和する |
| 埋め込み次元 `d` | **64** | コードポイント語彙 ~1万 × 64 で軽量 |
| pooling | **mean（確定、変更不可）** | mean は線形なので `W1_j·mean(e_c) = mean(W1_j·e_c)` が成り立ち、第1層事前計算（4.6節）と両立する。attention・encoder 等の非線形poolingはこの表引き化を壊すため採用しない |
| 隠れ層 | **1層、幅 H=256、ReLU** | ポイントワイズ分類では深さの効果が薄く、推論コストは隠れ層以降が支配的になるため浅く保つ。ReLU は量子化時に clipped ReLU へ置換可 |
| 正則化 | dropout（率は実験で決定） | |
| 出力 | 1ユニット、推論時は **y の符号判定**（sigmoid 省略） | `p>0.5 ⇔ y>0`。KyTea の手順7（3.3節）と同型の「スコアと0の比較」になる |

**辞書マッチ二値素性 `f_dict`**：KyTea の D 素性（3.1節）と同じ発想の、**言語独立な**外部知識チャネル。辞書（単語リスト）とのマッチを Aho-Corasick（3節の実装を共用、ただしEGC列上で走らせる）で取り、境界 i を跨ぐ/接するマッチについて

- 位置関係 3種（L: マッチ左端が境界に接する / I: マッチが境界を内包 / R: マッチ右端が境界に接する）
- × マッチ長バケット（`min(EGC長, 4)` の4段階）

の計12個の二値素性を立てる（複数辞書対応時は辞書ごとに12個）。**同一（位置関係×長さバケット）に複数の辞書語がマッチしても素性は1のまま（二値clamp）**とする。KyTeaの`addDictionaryScores`（3.3節）はマッチごとに加算（カウント相当）だが、本バックエンドは「二値素性」の宣言と事前計算経路（立っている素性=ベクトル加算1回）に合わせてclampを採り、KyTeaとの意図的な差異として記録する。文字種n-gramのような言語固有ヒューリスティックとは異なり、辞書は「単語リストという外部知識の注入」であり言語非依存のため、4.1節の「ヒューリスティック素性を持たない」方針とは矛盾しないと整理する。**二値素性への第1層の作用は「立っている素性に対応する256次元列ベクトルの加算」なので、4.6節の事前計算経路にそのまま乗る**（アクティブ素性1個 = ベクトル加算1回）。辞書なしでも動作する（`f_dict` 全ゼロ）。

### 4.5 学習

- **損失**：境界ごとの二値クロスエントロピー（BCE）。`sigmoid(y)` は損失計算でのみ使用。
- **教師信号のマスク**：部分アノテーション（5.2節）の「不明」位置（` ` unkBound / `?` skipBound → `PROB_UNKNOWN`）は損失計算から除外する。フルアノテーションは全境界が教師信号。intra-EGC位置はそもそも境界候補でないため損失にも現れない（4.2節）。
- **アノテーションとEGCの衝突処理**：コーパスの境界記号はコードポイント単位で付くため、EGC内部と衝突しうる。**EGC内部に境界あり（`|` またはフルアノテーションの語境界）が来た文は、警告を出してその文ごとスキップ**する（表現とラベルが矛盾するため学習に使えない）。EGC内部の「境界なし」（`-` や語の内部）はEGCの定義と整合するので黙って吸収する。スキップ件数はレポートし、4.2節の実測確認（正解境界のEGC一致率）の統計としても使う。
- **入力の正規化**：KyTeaと同じ半角→全角固定テーブル正規化（3.1節 `norm`）を**かけてから**EGC分割・語彙化する（方式(a)）。KyTeaバックエンドと入力条件を揃え、比較をフェアにするため。`CharTable` の正規化テーブルを共用する。
- **学習後の量子化**：重み・埋め込みは学習後に int16 へ量子化する（KyTea 自身が int16 量子化＋multiplier方式（3.2節）であり、同じ発想）。量子化誤差が境界判定を反転させないことを検証データで確認する。
- 最適化器・学習率・バッチ構成・dropout率は実装時に実験で決定する。

### 4.6 推論・C++実装方針：第1層の事前計算（NNUE方式）が前提

**素朴な順伝播は採用しない**。`concat(2w·d=640) → 256` の行列積は境界1つあたり ~16万MAC となり、Aho-Corasick＋整数加算だけのKyTea（~1µs/文字）に対して数µs/境界と**数倍遅くなる**。「CPU推論でKyTea超え」の目標は、以下の事前計算で達成する（将棋・チェスの評価関数 NNUE と同じ構造）。

**原理**：第1層は純粋な線形変換なので、窓位置ごとに分解できる：

```
W1 · concat(v_1, …, v_2w) = Σ_j  W1_j · v_j        （W1_j は窓位置 j に対応する 256×d のスライス）
```

したがって **(EGC, 窓位置 j) → W1_j·v(EGC) ∈ R^256 を事前に表引き化**できる。mean pooling の線形性（4.4節）により、合成的EGC埋め込みもこの表に折り込める。辞書二値素性も同様に `W_dict` の列ベクトル表引きになる。推論時の1境界の計算は：

```
acc = b1
acc += table[egc_j, j]   を 2w 回（表引き＋256次元ベクトル加算）
acc += dict_col[k]       をアクティブ辞書素性ぶん（0〜数回）
h = ReLU(acc)
y = dot(w2, h) + b2      （256次元内積 1 回）
境界 ⇔ y > 0
```

**≒ 2K ops/境界**。素朴実装の ~1/80。SIMD（NEON/AVX2）で **1境界 100〜200ns** を見込み、KyTea 超えを狙う。

**テーブル・アキュムレータの数値表現**：`table[egc,j] = W1_j·v(egc)` は int16×int16 の積を d=64 回加算した値で**最悪 2^36 に達し int16 に収まらない**。したがって：

- **第一実装は int32 テーブル＋int32 アキュムレータ**とする（256次元×4B=1KB のベクトル加算×2w回。int16 比でSIMDスループットは半減するが、依然 1境界 サブµs で速度目標を保てる）。オーバーフローと飽和の心配がなく正しさを検証しやすい。
- **int16 への再量子化（第3のスケール＋右シフト＋飽和クリップ）は、プロファイル後の最適化**として位置づける（NNUEが行っているのはこの形）。導入時は検証データで飽和発生率と判定反転をチェックする（4.5節の量子化検証と同じ枠組み）。

- **事前計算テーブル**：頻出EGC（≒頻出コードポイント）について構築。語彙1万 × 2w=10位置 × 256 × int16 ≒ 50MB。メモリを絞る場合は頻出上位のみ表引き化し、稀なEGCは「コードポイント埋め込み → mean → W1_j を掛ける」合成経路（これも小さい行列×ベクトル1回）にフォールバックする。日中テキストでは大半が表引き経路になる。
- **逐次スキャン**：境界 i → i+1 で窓は1EGCずれるだけなので、表引き結果の再利用（アキュムレータの差分更新）も将来の最適化候補として残す（まずは素直に 2w 回加算で実装し、プロファイルしてから）。
- トークン化（UTF-8 → EGC分割）は UAX #29 に従う。既存の `unicode/utf8` を土台に EGC 分割ロジックを追加する。
- 順伝播（推論）のみ実装すればよい。学習は別コンポーネント（Pythonまたは本ライブラリの学習モジュール）で行い、モデルファイル（4.7節）で受け渡す。

### 4.7 モデルファイル形式（独自設計）

シリアライズ形式は独自設計。**`BinaryReader`（`bytes/binary_reader.h`）のプリミティブ（リトルエンディアン固定幅整数・NUL終端文字列・`\n`終端ヘッダ行）でそのまま読める**ことを設計制約とし、KyTeaバックエンドと同じ読み取り基盤を共用する。事前計算テーブル（4.6節）と Aho-Corasick オートマトンはファイルに含めず、**ロード時に構築する**（Vaporettoが自身のオートマトンを実行時に構築するのと同じアプローチ）。

**ヘッダ行（ASCII、`\n`終端）**

```
SegmentLibMLP <version>\n
```

例：`SegmentLibMLP 1\n`。`read_line()` で読む。この `"SegmentLibMLP "` シグネチャを2節のバックエンド自動判別に使う（KyTeaの`"KyTea "`シグネチャと排他）。バージョン不一致はローダがエラーにする。

**ヘッダ行に続くバイナリ本体**（すべてリトルエンディアン。整数は`read<T>()`、`double`はLE生バイトで格納——`BinaryReader`は整数のみバイトスワップするため、ビッグエンディアンホスト対応は将来課題として`double`のスワップを別途要する。3節と同じ既知の制約）。

| # | フィールド | 型 | 内容 |
|---|---|---|---|
| **Config** | | | |
| 1 | `char_window` `w` | `uint8` | 片側窓幅（EGC個数）。4.4節で `w=5` |
| 2 | `embed_dim` `d` | `uint16` | コードポイント埋め込み次元。4.4節で `64` |
| 3 | `hidden` `H` | `uint16` | 隠れ層幅。4.4節で `256` |
| 4 | `num_dicts` | `uint8` | 辞書数（`0`可。`0`ならW_dict・辞書セクションは書かれない） |
| 4b | `unicode_version` | `uint16` | 学習時のEGC分割に使ったUnicodeバージョン（メジャー×100+マイナー、例：15.1→`1510`）。UAX #29の規則はバージョンで変わりうるため、推論側の分割器と不一致ならローダが警告する |
| **Scales**（量子化スケール、KyTeaの`multiplier`（3.2節）と同発想） | | | |
| 5 | `emb_scale` | `double` | 埋め込みint16→実数の係数 |
| 6 | `w1_scale` | `double` | W1のint16→実数の係数 |
| 7 | `wdict_scale` | `double` | W_dictの係数（`num_dicts>0`のときのみ） |
| 8 | `w2_scale` | `double` | w2の係数 |
| 8b | `acc_scale` | `double` | **加算器（第1層活性）の整数スケール**。重みスケールから導出せず、学習後に検証データで活性分布をキャリブレーションして選ぶ（`pct99.99(|a|)/Amax`, `Amax≈2^22`）。事前計算テーブル・辞書列・b1/b2 の整数化はすべてこのスケール基準（実装詳細は `mlp_impl_design.ja.md` I.1）。ローダ側では再現できない情報のため必ずファイルに載せる |
| **Vocabulary**（コードポイント語彙） | | | |
| 9 | `vocab_size` `V` | `uint32` | 埋め込み行数。行0=PAD、行1=UNK（未知コードポイント）を含む |
| 10 | `codepoints` | `uint32 × (V-2)` | 行2..V-1に対応するコードポイントを**昇順**で格納。推論時は入力コードポイントをこの配列で二分探索し行番号を得る（見つからなければ行1=UNK）。行0/1はコードポイントを持たない |
| **Embedding** | | | |
| 11 | `embedding` | `int16 × (V·d)` | 埋め込みテーブル。行優先（行0=PAD、行1=UNK、行2以降=`codepoints`順）。EGCベクトルは構成コードポイント行のmean（4.3節） |
| **Layer 1** | | | |
| 12 | `W1` | `int16 × (H · 2w · d)` | 第1層重み。行優先で`W1[h][j*d + c]`（`h`=隠れユニット, `j`=窓位置0..2w-1, `c`=埋め込み次元）。この`[·][j*d..]`スライスが4.6節の位置別事前計算 `W1_j` に対応 |
| 13 | `W_dict` | `int16 × (H · num_dicts · 12)` | 辞書二値素性の重み。`W_dict[h][dict*12 + feat]`（`feat`=L/I/R×長さバケット4=12）。`num_dicts>0`のときのみ |
| 14 | `b1` | `double × H` | 第1層バイアス（**非量子化**。個数が少なくスケール合成を避けるため実数のまま。ローダがアキュムレータのスケール `emb_scale·w1_scale` に合わせて整数化する） |
| **Layer 2** | | | |
| 15 | `w2` | `int16 × H` | 出力層重み |
| 16 | `b2` | `double` | 出力層バイアス（非量子化）。推論判定は `y > 0`（4.4節）なので`w2`側の正のスケールは符号に影響しないが、`b2`は`w2·h`と同一スケールに合わせる必要があり、ローダで整数化する |
| **Dictionaries**（`num_dicts>0`のときのみ。Aho-Corasick構築用の生の単語リスト） | | | |
| 17 | 各辞書ごとに繰り返し（`num_dicts`回） | | |
| 17a | `entry_count` | `uint32` | 語数 |
| 17b | `entries` | NUL終端UTF-8 × `entry_count` | 単語表層（`read_cstring()`で読む）。ローダが**方式(a)の正規化（4.5節）をかけてから**UAX #29でEGC列に分割し、EGC単位のAho-Corasickを構築する（入力側だけ正規化すると半角混じりの辞書語が永遠にマッチしないため、辞書側にも同じ正規化を適用する） |

**ロード時の処理**：(1) 語彙・埋め込み・重みを読み、(2) 4.6節の位置別事前計算テーブル `table[egc, j] = W1_j · v(egc)`（頻出EGCぶん）と辞書素性の列ベクトル `W_dict` の展開を構築、(3) 単語リストからEGC単位Aho-Corasickを構築、(4) `b1`/`b2` をアキュムレータ整数スケールへ量子化。以降の推論はテーブル引き＋整数加算のみ（4.6節）。

**将来**：全体をzstd圧縮する外装を被せる余地を残す（Vaporetto自身のCLIツールが自らのモデルファイルに対して行っているのと同じ手法）。その場合も2節の自動判別はzstd解凍後のヘッダ行で行う。

### 4.8 位置づけ・評価条件

3節・4節の実装が固まり、共通の`tokenize`/`tokenize_boundaries`インターフェースでの評価基盤ができてから本格着手する。

**精度目標「KyTea同等」の比較条件**：配布モデル `jp-0.4.7-5.mod` の学習コーパス（BCCWJ等）は入手不可のため、「配布モデルと同一データで学習したMLP」は作れない。したがって比較は**入手可能なコーパスで KyTea を再学習し、同一データ・同一辞書で学習した本バックエンドと突き合わせる**形で行う（目標は「同一条件下でKyTea方式と同等以上」であり、「配布モデルの絶対精度との比較」ではない）。速度は同一テキストでKyTeaおよびVaporettoと比較し、CPU単スレッドでKyTea超えを目標とする。

**既知の精度リスク**：窓内埋め込みのみでは、KyTeaが辞書素性で得ている語彙知識を完全には代替できない可能性がある。緩和策として w=5（KyTeaの charw=3 より広い）と辞書チャネル（4.4節）を最初から備える。それでもギャップが残る場合の追加候補：General Unicode Property の補助入力（4.1節）、隠れ層の拡幅、学習データ増強。

**初回の実測（2026-07-12、パイプライン検証）**：配布モデルの学習データ（BCCWJ・UniDic・CSJいずれも国語研許可制で入手不可、KyTea公式が挙げる唯一の無料コーパスも参照リンク切れ）に代えて、**UD_Japanese-GSD**（`UniversalDependencies/UD_Japanese-GSD`、CC BY-SA 4.0、短単位=UniDic基準でKyTeaデフォルトモデルと同系統の分割基準）を採用。取得・KyTea形式変換は `scripts/fetch_ud_gsd_corpus.sh` / `scripts/convert_ud_gsd_corpus.py`（タグ0=UniDic品詞、タグ1=UnidicInfoのlForm読み）。train 7050文（EGC境界との衝突0件・不正UTF-8 0件、270,163境界）/ dev 507文 / test 543文。

- `train-kytea -notags`（同一train、辞書なし）と `segmenter train --backend mlp`（既定値 w=5, d=64, H=256, epochs=30, patience=5、辞書なし、dev で早期終了・量子化キャリブレーション）を同一データで学習。
- test set 境界F値（`scripts/eval_segmentation.py`）：**KyTea 98.65%**（P 98.73 / R 98.56） vs **MLP 97.91%**（P 97.57 / R 98.25）。量子化による判定反転は dev set 19,625件中 **0件**。
- 差は約0.7ポイントで、4.8節の想定どおり「辞書なしでKyTeaの辞書素性による語彙知識に完全には届かない」形。ただし小規模コーパス（design.ja.md 5.9が想定する境界数〜500万に対し本コーパスは27万）でこの近さは構成（w=5等の既定値）を変える必要がないことを示す初期エビデンスと位置づける。**速度比較は未実施**（今回はCLIプロセス起動込みの参考値のみ；9節の方法論に沿ったin-processベンチは手順9のSIMD最適化と合わせて実施）。
- コーパス実体・学習済みモデル（`corpus/ud-gsd/`）は models/ 同様 gitignore 対象。再現は上記2スクリプト＋このコマンド列で可能。

**速度の初回実測（同日、同一train文7050行・277,433コードポイント、in-process・best-of-7、`bench/bench_segment` / `bench/.vendor/bench_kytea`、9節と同方法論、M1 Pro）**：

| 実装 | WS推論速度 | KyTea比 |
|---|---|---|
| 本ライブラリ MLPバックエンド（`corpus/ud-gsd/mlp.mod`、**int32テーブル・SIMD未適用の第一実装のまま**） | 2.24 M chars/sec | **1.53x** |
| 本ライブラリ KyTeaバックエンド（参考、`corpus/ud-gsd/kytea.mod`） | 9.73 M chars/sec | 6.66x |
| 実KyTea（libkytea、`corpus/ud-gsd/kytea.mod`） | 1.46 M chars/sec | 1.00x |

**4.6節が目標に掲げた「CPU推論でKyTea超え」を、int16再量子化・SIMDカーネル（4.6節後半・I.3で計画していた最適化）を適用する前の第一実装だけで既に達成**（1.53倍）。ただし本ライブラリのKyTeaバックエンドには大きく劣る（256次元アキュムレータの2w回加算＋内積が、KyTeaの疎なAho-Corasick＋int32加算より本質的に重いため）。int16再量子化・SIMD化は「KyTea超え」の必須条件ではなく、この差を縮める追加の最適化として位置づけ直す。

**最適化後の実測（同日、I.3のSIMDカーネル＋5.6のint16再量子化テーブル＋I.5のthread_localスクラッチ適用後。同一条件・best-of-15）**：

| 実装 | WS推論速度 | KyTea比 |
|---|---|---|
| MLPバックエンド（**int16テーブル＋NEON**、既定） | **3.19 M chars/sec** | **2.2〜2.4x** |
| MLPバックエンド（int32テーブル、参照経路） | 1.64 M chars/sec | 1.1〜1.2x |
| 実KyTea（libkytea、再計測 1.34〜1.46） | 1.34〜1.46 M chars/sec | 1.00x |

- 内訳（in-processプロファイル）：encode（正規化+EGC分割）は3.0ms/7050文で無視でき、コストはスコア計算に集中。int16化＋NEONでスコア計算部は **1.9倍**（168.7ms→88.8ms）。
- **int16再量子化の検証（I.3の枠組み）**：`kAccShift=9`（キャリブレーション点 2^22 → 2^13、int16内に4倍のヘッドルーム）。UD-GSD実モデルの dev 19,641境界＋train 270,383境界で **int32経路との判定反転 0件**、test set F1 も 97.91% で不変。int16 は `Model::load` の既定（`TablePrecision::Int16`）、int32 は検証用参照経路として残置（学習側リファレンスとのbit-exact契約はint32経路が担う）。
- SIMDは NEON（AArch64、実機検証済み）／AVX2（x86、`-mavx2`時。Apple clang `-arch x86_64` でコンパイル検証のみ——x86実機での動作確認は未実施）／scalar（全プラットフォーム、テストのoracle）の3実装（`include/segmentlib/mlp/kernels.h`）。

**クロスジャンル一般化の再計測（2026-07-12、GSD学習済みモデルをそのまま別ジャンルへ適用）**：初回実測は GSD（Wikipedia由来）の test 543文のみで、精度差0.7ptが小テストセットのブレか代表値かが未確認だった。そこで**別ジャンルの out-of-domain テストセット** UD_Japanese-PUD（`UniversalDependencies/UD_Japanese-PUD`、CC BY-SA 4.0、news/Wikipedia対訳、1000文・27,788境界。GSDと同じ UniDic 短単位基準で SUW 分割・XPOS・UnidicInfo 読みを持つ）を追加し、**GSDで学習した `kytea.mod`／`mlp.mod` を再学習せずに評価**した。取得は `scripts/fetch_ud_pud_corpus.sh`（変換は GSD 用 `convert_ud_gsd_corpus.py` を UD日本語共通フォーマットとしてそのまま再利用）。

| テストセット（ジャンル） | 境界数 | KyTea F1 | MLP F1 | 差 |
|---|---|---|---|---|
| GSD test（Wikipedia、in-domain） | 12,491 | 98.65% | 97.91% | 0.74pt |
| PUD（news/Wikipedia対訳、out-of-domain） | 27,788 | **99.18%** | **98.56%** | 0.62pt |
| GSD+PUD 合算（2ジャンル） | 40,279 | 99.02% | 98.35% | 0.67pt |

- **別ジャンルでも精度は劣化せず**（両モデルとも PUD の方が高い。PUD の対訳文が GSD の Wikipedia 記事より規則的なため）、**MLP-KyTea 差も 0.6〜0.7pt で安定**。初回の GSD 単独 0.74pt は小テストセットのブレではなく代表値と確認できた。
- 差が out-of-domain で**広がらない**ことは、MLP の窓内埋め込み方式が KyTea の辞書素性と同程度にジャンルをまたいで一般化することを示す。3倍規模・2ジャンルでも「構成（w=5等の既定値）を変える必要はない」という初回の判断を追認。
- コーパス実体（`corpus/ud-pud/`）は models/・ud-gsd/ 同様 gitignore 対象。再現は上記 fetch スクリプト＋`scripts/eval_segmentation.py --gold corpus/ud-pud/test.kytea.txt --command '...kytea...' --command '...segmenter...'`。

**速度の多ジャンル再計測（同日、`bench/bench_segment` in-process・best-of-8プロセス×15/30イテレーション、M1 Pro）**：445節の速度実測は青空文庫由来の GSD train（Wikipedia系、277,433字）1ジャンルのみだった。PUD test（news/Wikipedia対訳、48,260字）を第2ジャンルとして追加計測：

| ジャンル | MLP（int16+NEON） | 実KyTea（libkytea） | 比 |
|---|---|---|---|
| GSD train（Wikipedia、277K字） | 3.34 M chars/sec | 1.40 M chars/sec | 2.39x |
| PUD test（news対訳、48K字） | 3.49 M chars/sec | 1.58 M chars/sec | 2.21x |

- 445節の「2.2〜2.4x」の範囲に両ジャンルとも収まり、**分割速度はジャンルに非感受**であることを確認（想定どおり——スコア計算は EGC 窓の整数演算のみで、語彙・文体には依存しない）。コーパスサイズの違い（277K字 vs 48K字）による測定ノイズは残るが、桁が変わるような乖離はない。
- これで design.ja.md 10節の「より大規模・多ジャンルのコーパスでの再計測」は精度・速度とも完了（未計測なのは配布モデル同等の大規模コーパス＝BCCWJ相当のみで、これは4.8節冒頭のとおり入手不可のため対象外）。

### 4.9 学習側の設計（C++自前実装）

学習エンジンは本ライブラリで自前実装する（外部フレームワーク非依存）。学習は fp32、推論は int16（4.6節）で、両者はモデルファイル（4.7節）で受け渡す。**推論（ライブラリ本体の成果物）は順伝播のみ**で、学習コンポーネントはビルド上分離する（推論バイナリに BLAS/CUDA を要求しない）。

**学習パイプライン**

```
1. コーパス読込（5節, KyTeaフル/部分アノテーション）
2. 正規化（方式(a): KyTea互換の半角→全角, CharTable共用, 4.5節）
3. EGC分割（UAX #29）
3b. 語彙構築: 出現コードポイントを頻度集計し、閾値未満は UNK に落とす（4.3節。UNK 行の学習を保証）
4. 例の生成: 各境界 i →
     - 窓 [i-w+1 … i+w] の各EGC → 構成コードポイント行ID列（端はPAD）
     - 辞書マッチ二値素性 f_dict（Aho-Corasick, 4.4節）
     - ラベル（境界=1/非境界=0）
     - マスク（部分アノテーションの不明位置は損失から除外, 4.5節）
5. ミニバッチ化: 埋め込みgather → mean pooling → concat(2w·d)
6. 順伝播 → BCE(マスク付き) → 逆伝播（埋め込みは疎勾配）
7. 最適化: Adam
8. 収束後: PTQ int16 量子化 → 検証（判定反転チェック, 4.5節）→ 5.7形式で書き出し
```

**逆伝播の要点**

- **第1層・第2層**：標準的な dense 層の勾配。行列積が主計算で、`ComputeBackend` の GEMM に委譲する。
- **mean pooling → 埋め込み**：EGCベクトルが構成コードポイントの mean なので、EGCベクトルへの勾配は各構成コードポイント行へ `1/(コードポイント数)` で分配される。**埋め込みテーブルの勾配は疎**（バッチに現れた行のみ）。Adam の 1次・2次モーメントも現れた行だけ更新する。
- 辞書二値素性 `f_dict` への `W_dict` 勾配も、立っている素性の列のみの疎更新。

**最適化器**：Adam（学習率・β・weight decay は実験で決定）。埋め込みの疎更新と相性が良く、KyTea/Vaporettoの線形SVMとは無関係に本バックエンド独自に選ぶ。

**量子化**：PTQ（Post-Training Quantization、4.5節）で確定。int16 は 65536 段階で相対誤差が極小、かつ判定は `y>0` の符号のみのため、訓練後の一括丸めでほぼ無損失。**QAT（Quantization-Aware Training）は今回採用しない**——訓練の順伝播に偽量子化（fake-quant）を挟み STE で逆伝播する手法で、int8/int4 のような攻めた低ビットで初めて要る。以下の**条件付きフォールバック**としてのみ記録する：(1) 4.5節の判定反転チェックで反転が過大な場合、(2) 将来 int8 化する場合（モデル縮小、または Apple Neural Engine 推論など）。

**計算バックエンド抽象（`ComputeBackend`）**

行列積・活性化・要素演算・勾配だけをこの層に閉じ込め、プラットフォームごとに実装を差し替える。CPU実装は BLAS インターフェース（`cblas_sgemm` 等）に対して書き、リンクする BLAS を切り替えるだけで全OSに載る。

| プラットフォーム | 第一候補 | CPU実装（BLAS） | GPU実装（任意） |
|---|---|---|---|
| **macOS (Apple Silicon)** | **CPU/AMX** | Accelerate（AMX行列コプロセッサを透過利用） | Metal/MPSGraph（通常不要） |
| **Linux + NVIDIA** | **GPU** | OpenBLAS/MKL | cuBLAS |
| Linux (GPUなし) | CPU | OpenBLAS/BLIS | — |
| Windows + NVIDIA | GPU | OpenBLAS/MKL | cuBLAS |
| Windows (GPUなし) | CPU | OpenBLAS | — |

- **`CpuBackend`（BLAS）**：全OS共通。中身のみ Accelerate / OpenBLAS / MKL を切替。
- **`CudaBackend`（cuBLAS）**：Linux + Windows 共通。
- **`MetalBackend`**：macOS のみ、後付けオプション。
- サポート方針：**macOS・Linux を一級サポート、Windows は best-effort**（C++23の`std::expected`等はMSVC 19.33+で利用可、cuBLAS/OpenBLAS/zstd も Windows 対応済みでアーキテクチャ的には自然に入るが、CI/ビルド周りの手当ては優先度を下げる）。

**プラットフォーム別の設計上の理由**：

- **M1 では GPU より Accelerate CPU が正解**。Apple Silicon には AMX 行列ユニットがあり `cblas_sgemm` から透過利用でき、この規模の GEMM で実効 ~1〜1.5 TFLOP/s（カーネル起動オーバーヘッドゼロ、ユニファイドメモリで転送コストなし）。M1 Pro GPU（~5 TFLOP）を Metal で叩いても同程度の数分で、Metal を書く工数に見合わない。
- **Linux では逆に GPU が効く**。x86 コンシューマCPUには AMX 相当がなく BLAS 実効 ~200〜800 GFLOP/s に留まる一方、離散NVIDIA GPUは桁違いに速い。CUDA(cuBLAS) が第一候補。
- ANE（Apple Neural Engine）は int8/fp16 の推論向けで任意の逆伝播に使えず、**学習には非対応**（将来の int8 推論の載せ替え先としては別途あり得る）。

**訓練時間の見積り**（境界数 ~500万・30エポック ≈ 150 TFLOP 換算。KyTea(LIBLINEAR)は同規模で ~数分）：

| 環境 | 訓練時間（目安） | KyTea比 |
|---|---|---|
| macOS M1 Pro（Accelerate/AMX） | 3〜6 分 | ≈1〜2× |
| Linux + CUDA（RTX 3060級） | 2〜5 分 | ≈1× |
| Linux CPU（OpenBLAS, 8〜16コア） | 12〜20 分 | ≈4〜6× |

疎な線形SVM（KyTea）に対し 1例あたりの密演算量は数百〜1000倍だが、モデルが小さいため絶対時間は「分〜数十分」に収まる。GPU の旨味が本当に出るのは、ハイパラ探索の多数回実行・`d`/`H` の拡大・コーパス増量時。

**推論側の可搬性（学習とは別軸）**：推論は BLAS ではなく手書き SIMD の int16 NNUE 方式（4.6節）で、**NEON（Apple/ARM）＋ AVX2（x86 = Linux/Windows 共通）＋ スカラーfallback** の3実装＋fallbackで全OSを覆う。x86 の AVX2 パスは Linux/Windows で同一。

## 5. コーパス仕様（KyTea本体のソースで検証済み）

`corpus-io-full.cpp` / `corpus-io-part.cpp` を直接確認した。デフォルトの区切り文字は `kytea-config.cpp` で以下の通り定義されている：

| 用途 | 文字 | デフォルト |
|---|---|---|
| 単語境界（フル）/ 不明境界（部分） | `wordBound_` / `unkBound_` | 半角スペース `" "` |
| タグ境界 | `tagBound_` | `/` |
| タグ候補区切り | `elemBound_` | `&` |
| エスケープ | `escape_` | `\` |
| 非境界（部分） | `noBound_` | `-` |
| 境界（部分） | `hasBound_` | `\|` |
| スキップ（部分） | `skipBound_` | `?` |

すべてのバックエンド（3〜4節）の学習データは、このKyTeaコーパス形式に統一する。

### 5.1 フルアノテーション形式

```
word1/tag0a&tag0b/tag1a word2/tag0 word3 ...
```

- 単語は半角スペースで区切る。
- `/` が出現するたびにタグの「レベル」が1つ進む（KyTeaは読み・品詞など複数タグ体系を同時に持てるため、レベルでタグ種別を表現する）。
- 同一レベル内で複数候補のタグを持たせたい場合は `&` で連結する（学習時は先頭候補が正解ラベルとして使われる）。
- 単語・タグ中に区切り文字自体（スペース、`/`、`&`、`\`）を含めたい場合は `\` でエスケープする。
- 実例（KyTea配布モデルの学習データで実際に使われている書式）：
  ```
  コーパス/ko:pasu の/no 文/buN で/de す/su 。/.
  ```

### 5.2 部分アノテーション形式

```
ヴ-ェ-ネ-ツ-ィ-ア|は|イ-タ-リ-ア|に|あ り ま す|。
```

- 文字を1文字ずつ並べ、隣接文字の間に以下いずれかの記号を置いて境界情報を表す：
  - `-`（noBound）：境界ではない（同じ単語内）。教師信号として確定的に使う。
  - `|`（hasBound）：境界である。教師信号として確定的に使う。単語の終端も兼ねる。
  - ` `（unkBound）/ `?`（skipBound）：不明。読み込み時はいずれも「wsConfsに教師信号なし（PROB_UNKNOWN）」として扱われる（コード上は2つの記号が区別されているが、現在の読み込みロジックでは同じ扱いになる点に注意）。
- タグは各語の末尾、`|` の直前に `/tag0&tag1/tag2...` の形式でフルアノテーションと同じ文法で付与できる。
- エスケープ文字・タグ境界・タグ候補区切りはフルアノテーションと共通（`\`, `/`, `&`）。

### 5.3 備考

- 現時点で本ライブラリは**推論のみ**が目的のため、コーパス読み込み（学習パイプライン）は直接のスコープ外。ただし将来の学習機能サポート（4節の独自MLPバックエンドを含む）や、KyTeaとの入出力互換性確認のため、フォーマットは正確に記録しておく。
- CLIの出力フォーマット（7.2節）は、上記フルアノテーション形式の書き込みロジック（`FullCorpusIO::writeSentence`）にそのまま対応する。

## 6. C++ API

### 6.1 基本方針

- 可変オブジェクト（KyTeaの`KyteaSentence&`、Vaporettoの`Sentence`）を使い回す副作用ベースのAPIではなく、
  **入力を受け取り結果を値として返す関数型のAPI**を基本とする。
- 入力は所有権を必要としないため `std::string_view` を受け取る（不要なコピーを避ける）。
- エラーは `std::expected`（C++23）で表現する。「結果が空」と「エラーが発生した」を型で区別する。
- オフセットは**UTF-8バイトオフセット**を採用する（元の文字列からのスライスがしやすいため）。
- バックエンド（KyTea互換／独自MLP）の違いはこのAPI層には一切露出しない（2節参照）。呼び出し側は`Segmenter::load()`で読み込んだモデルファイルの種類を意識せず、同じ`tokenize()`/`tokenize_boundaries()`を呼ぶだけでよい。

### 6.2 型

```cpp
struct Segment {
    std::size_t start;                 // UTF-8バイトオフセット（開始、含む）
    std::size_t end;                   // UTF-8バイトオフセット（終了、含まない）
    std::vector<std::string_view> tags; // 読み・品詞等のタグ（タグ推定を行わない場合は空）
};

using Segments = std::vector<Segment>;
using Boundaries = std::vector<std::size_t>; // 分かち書きのみの場合の境界オフセット列

enum class ErrorCode {
    InvalidUtf8,
    ModelNotLoaded,
    UnsupportedModelFormat,
    // ...
};

struct Error {
    ErrorCode code;
    std::string_view message; // 静的文字列を想定。動的なメッセージは持たせない
};
```

### 6.3 Segmenter

```cpp
class Segmenter {
public:
    // モデルファイルの内容からバックエンド種別を自動判別してロードする
    static std::expected<Segmenter, Error> load(const std::filesystem::path& model_path);

    // バックエンドを明示的に指定してロードする（自動判別を使わない場合）
    static std::expected<Segmenter, Error> load_kytea(const std::filesystem::path& model_path);
    static std::expected<Segmenter, Error> load_mlp(const std::filesystem::path& model_path);

    // 分かち書き + タグ推定
    std::expected<Segments, Error> tokenize(std::string_view text) const;

    // 分かち書きのみ（タグ推定を行わない高速パス）
    std::expected<Boundaries, Error> tokenize_boundaries(std::string_view text) const;
};
```

- `tokenize()` と `tokenize_boundaries()` を型レベルで分けているのは、タグ推定用のストレージ確保をスキップできる軽量パスを明示的に提供するため。
- 空文字列の入力は「エラーではなく空の結果」として扱う（`Segments{}` / `Boundaries{}`）。`Error`はUTF-8不正やモデル未読込・未対応形式などの実際の異常系のみに使う。

### 6.4 アロケータ（将来課題・検討事項）

高スループット用途（大量の文を連続処理するループ）でのアロケーションコストを避けたい場合の拡張案として、
`std::pmr::memory_resource` を注入できるオーバーロードを検討中：

```cpp
using PmrSegments = std::pmr::vector<Segment>;

std::expected<PmrSegments, Error> tokenize(
    std::string_view text,
    std::pmr::memory_resource* mr) const;
```

呼び出し側は `std::pmr::monotonic_buffer_resource` を使い回すことで、境界スコア等の内部作業バッファの再確保を避けられる。
ただし優先度は低く、まずはデフォルトアロケータの素直な実装で計測し、実際にボトルネックになった場合にのみ導入する。

## 7. CLIインターフェース

コマンド名は`segmenter`。`git`/`cargo`のような**サブコマンド構成**とし、`train-kytea`（学習）と`kytea`（推論）に分かれているKyTea本体の構成を1バイナリに統合する。

```
segmenter predict --model model.bin < input.txt > output.txt
segmenter train --backend kytea --corpus corpus.txt --model-out model.bin
```

### 7.1 `segmenter predict`（推論）

KyTea / Vaporetto と同様、**標準入力からテキストを読み、標準出力に分かち書き結果を書き出すフィルタ型**を踏襲する。

```
segmenter predict --model model.bin < input.txt > output.txt
segmenter predict --model model.bin --boundaries-only < input.txt > output.txt   # tokenize_boundaries相当
segmenter predict --model model.bin --scores < input.txt > output.txt            # 境界ごとのスコアも出力
```

**オプション（初期案）**

| オプション | 説明 |
|---|---|
| `--model <path>` | モデルファイルのパス（必須）。KyTea互換／独自MLPのいずれかを自動判別する（2節） |
| `--backend <kytea\|mlp>` | バックエンドを明示的に指定（自動判別を上書きしたい場合）。**未実装**（初期案のまま。自動判別で用が足りているため今のところ要望なし） |
| `--boundaries-only` | タグ推定を行わず分かち書きのみ実行（`tokenize_boundaries`を使用）。**実装済み** |
| `--scores` | 各境界の分類スコアを合わせて出力。**未実装** |
| `--encode` | **実装しないことを決定**（10節）。入力は常にUTF-8固定——モデル（3.2節・4.7節）もコーパス形式（5節）もUTF-8前提で、他エンコーディングを選ぶ余地がそもそもない。不正なUTF-8バイト列は`CharTable::encode`/`Vocab::encode`が`ErrorCode::InvalidUtf8`を返し、CLIは該当行以前の出力をflushした上で**即座に中断**（exit code 1・stderrにメッセージ）する。行を黙ってスキップ・切り詰める設計にはしない（バイト単位一致を旨とするこのライブラリで、壊れた入力を黙って通すのは事故の元）。実測確認済み（`printf 'valid\n\xff\xfe bad\nvalid2\n' \| segmenter predict ...` → 1行目のみ出力・exit 1）。ユニットテストは`CharTable`/`Vocab`/`corpus`層で`InvalidUtf8`を確認済み（`tests/unit/{char_table,vocab,train_corpus}_test.cpp`）

**出力フォーマット**

KyTeaの出力形式（`単語/タグ1/タグ2 ...` をスペース区切り）を踏襲し、既存ツールチェーンとの互換性を保つ（5.1節のフルアノテーション形式そのもの）。

```
コーパス/ko:pasu の/no 文/buN で/de す/su 。/.
```

**表層語のエスケープ（`showEscapedString`）**：区切り文字（スペース・`/`・`&`・エスケープ文字`\` 自身）が単語表層に含まれる場合、KyTeaは`\`で前置してエスケープする。バイト単位一致にはこの再現が必須（例：入力 `Hello World` → `Hello \  World`、`2024/12/31` → `2024 \/ 12 \/ 31`）。実装は`append_full_line`（`src/output.cpp`）に集約し、CLIとgoldenテストで共有。4文字とも1バイトASCIIなのでUTF-8継続バイトとは衝突しない。

### 7.2 `segmenter train`（学習）

**学習エンジン自体は3節で述べた通りまだ未着手（9節のTODO参照）だが、CLIのインターフェース形状は先に固めておく。** 入力コーパスは5節で確定したKyTeaコーパス形式（フル/部分アノテーション）に統一する。

```
segmenter train --backend kytea \
  --corpus full1.txt --corpus full2.txt \
  --partial-corpus part1.txt \
  --dict dict.txt \
  --model-out model.bin
```

**オプション（初期案。`train-kytea`のオプション名（3.2節の`-charw`/`-charn`等）をそのまま`--`付きの長形式に対応付ける）**

| オプション | 説明 |
|---|---|
| `--backend <kytea\|mlp>` | 学習するバックエンド（必須）。2種のバックエンドで学習エンジン自体は全く別実装になる（2節） |
| `--corpus <path>` | フルアノテーションコーパス（5.1節）。複数回指定可能 |
| `--partial-corpus <path>` | 部分アノテーションコーパス（5.2節）。複数回指定可能 |
| `--dict <path>` | 辞書ファイル（`単語 タグ`形式）。複数回指定可能 |
| `--model-out <path>` | 学習済みモデルの出力先（必須） |
| `--char-window <int>` | 文字n-gramの窓幅（KyTeaの`-charw`相当、既定3） |
| `--char-n <int>` | 文字n-gramの最大次数（`-charn`相当、既定3） |
| `--type-window <int>` | 文字種n-gramの窓幅（`-typew`相当、既定3） |
| `--type-n <int>` | 文字種n-gramの最大次数（`-typen`相当、既定3） |
| `--dict-n <int>` | 辞書マッチ長の丸め上限（`-dicn`相当、既定4） |
| `--cost <float>` | SVMの正則化コスト（`-cost`相当） |
| `--eps <float>` | 学習の収束判定epsilon（`-eps`相当） |
| `--boundaries-only` | タグ（読み・品詞）モデルを学習せず分かち書きモデルのみ学習する（`-notags`相当） |

**上記は初期案のオプション形状（`train-kytea`の`-charw`等を素直に`--`長形式へ対応付けたもの）で、実装しないことが10節で確定した。** 現在のCLI（`train_command.cpp`）は`--backend kytea`／`--backend vaporetto`を明示的に「未実装」エラーとして扱い、`--backend mlp`のみ実装されている。KyTea互換モデルが必要な場面（本ライブラリの評価・比較用途など）は、本物の`train-kytea`（Homebrew配布）を外部ツールとしてそのまま呼ぶ運用に統一する（4.8節の評価パイプラインが実例）。

## 8. 実装モジュール構成（`segmenter predict` MVP）

最初のマイルストーンは `segmenter predict --model kytea-model.bin < input.txt > output.txt` が動くこと。KyTea本体のデータ構造・アルゴリズム（3節）は踏襲するが、**KyTeaの実装（生ポインタの手動`new`/`delete`、独自参照カウント文字列`KyteaString`、学習と推論と設定パースが全部1つの`Kytea`神クラスに同居、`THROW_ERROR`マクロ）はコード構造としては見習わない**。責務ごとに薄く分離したレイヤ構成にする。

### 8.1 レイヤ構成（下位→上位、上位は下位にのみ依存）

```
include/segmentlib/
├── bytes/
│   └── binary_reader.h        # (L1) プリミティブなバイナリ読み取りカーソル
├── unicode/
│   └── utf8.h                 # (L1) UTF-8デコード/エンコードの純粋関数群
├── kytea/
│   ├── char_table.h           # (L2) 文字種分類 + 文字インターン表（3.1, 3.2節）
│   ├── automaton.h            # (L2) Aho-Corasickランタイム表現 + マッチ（3.2節 DictionaryState）
│   ├── model.h                # (L3) Config/KyteaModel/FeatureLookupのデータ型 + load()
│   ├── scorer.h                # (L4) calculateWSのスコア計算アルゴリズム（3.3節）
│   └── kytea_backend.h        # (L5) Backend I/F実装（tokenize/tokenize_boundaries）
├── segmenter.h                 # (L6) 公開API（6節）。今はKyTeaBackendのみをvariantに持つ
└── types.h                     # (L1) Segment/Segments/Boundaries/Error（6.2節、依存なし）

src/
├── kytea/{char_table,automaton,model,scorer,kytea_backend}.cpp
├── segmenter.cpp
└── cli/
    ├── main.cpp                  # `predict`/`train` のサブコマンド振り分け
    ├── predict_command.cpp       # predictサブコマンド本体
    └── train_command.cpp         # train: 現状は "not yet implemented" を返すだけの雛形
```

### 8.2 各モジュールの責務と設計意図

**L1: `bytes::BinaryReader`** — `std::span<const std::byte>`を読み進めるだけの薄いカーソル。`read<T>()`（`uint32_t`/`int32_t`/`bool`/`char`/`double`等）、`read_cstring()`（3.2節で確認したNUL終端の`KyteaString`表現）を提供する。**パースエラーは内部では例外（軽量な`ParseError`）で投げっぱなしにし、`model.h`の`load()`という一箇所の境界だけで`std::expected`に変換する**。KyTeaの`THROW_ERROR`マクロ＋呼び出し元でのチェック漏れの温床を避けつつ、6.1節で決めた「公開APIは`std::expected`」という方針とも矛盾しない。

**L1: `unicode::utf8`** — UTF-8の1コードポイントをデコードする純粋関数。KyTea固有の概念を一切知らない、汎用の下請けモジュール。

**L2: `kytea::CharTable`** — 3.2節で確認した「文字インターン表」（ID`0`=センチネル、実文字は`1`始まりの初出順）を保持する値型。`decode(std::string_view utf8) -> std::vector<std::uint16_t>`（本ライブラリでの`KyteaChar`表現は素の`std::uint16_t`でよく、KyTeaの`KyteaChar`という別名だけを踏襲する）と、3.1節の文字種分類（`char_type(std::uint16_t) -> CharType`、`enum class CharType : std::uint8_t { Romaji, Hiragana, Katakana, Digit, Kanji, Other }`）を提供する。

**L2: `kytea::Automaton<Payload>`** — 3.2節の`Dictionary<Entry>`のランタイム表現。テンプレートにして、`Payload = FeatVec`（今回のMVPで使う`charDict`/`typeDict`）だけでなく将来の`Payload = WordDictEntry`（タグ推定用の単語辞書）にも同じ型で対応できるようにしておく。KyTeaのように状態を`new`した生ポインタの配列で持つのではなく、`std::vector<State>`と`std::vector<Payload>`をフラットに持つ値型にする。マッチは戻り値をアロケートする`match() -> std::vector<Match>`と、アロケーションを避けたい場面向けのコールバック版`match(text, callback)`の両方を用意する（3.4節のpmr方針と同じ発想を、まずはコールバックという言語機能だけで先取りする）。**このモジュールはモデルファイルのデシリアライズ結果を受け取るだけで、KyTeaのようにAho-Corasickを構築するロジック（`buildGoto`/`buildFailures`）は持たない**（3.2節で確認した通り、ファイルには構築済みオートマトンがそのまま入っているため）。独自MLPバックエンド（4節）は逆に、辞書の単語リストからロード時にこの`Automaton`を構築する側になるので、「構築ロジック」は`Automaton`とは別に`AutomatonBuilder`として後日切り出す。

**L3: `kytea::Model`** — 3.2節の`Config`/`KyteaModel`/`FeatureLookup`に対応するイミュータブルな値型。`static auto load(std::filesystem::path) -> std::expected<Model, Error>`が唯一の構築経路。MVPでは`charDict`/`typeDict`/`selfDict`/`dictVector`/`biases`/`multiplier`/`bias`/`wsConstraint`のみを保持すれば`predict`は動く。単語辞書（`ModelTagEntry`）・サブワード辞書（`ProbTagEntry`）・言語モデル（`KyteaLM`）・タグモデル（`globalMods_`）は**ファイル上に存在すれば読み飛ばしてでも正しく後続位置までシークできる必要がある**（3.2節のセクション順は固定なので、読まないなら「バイト数だけ数えてスキップする」実装が要る）が、値としては保持しなくてよい。タグ推定を実装する段階で`Model`にフィールドを追加する。
  - **`--boundaries-only`ではない通常の`predict`（今回のコマンド例）はタグ推定も必要**なので、MVPの最終的な出力形式をどうするかは8.3節で補足する。

**L4: `kytea::scorer`** — 3.3節で復元した`calculateWS`のアルゴリズムをそのまま関数として実装する。`Model`・`CharTable`・入力テキストを受け取り、境界ごとのスコア列を返す純粋関数にする（KyTeaのように`Kytea`インスタンスの内部状態を書き換えるのではない）。
  ```cpp
  auto score_boundaries(const Model& model, const CharTable& chars, std::u16string_view normalized) -> std::vector<std::int32_t>;
  auto segment(const Model& model, const CharTable& chars, std::string_view utf8_text) -> Boundaries;
  ```

**L5: `kytea::KyteaBackend`** — 2節で定義した`tokenize`/`tokenize_boundaries`シグネチャを満たすクラス。中身は`Model`を保持し、`scorer`の関数を呼ぶだけの薄いアダプタ。

**L6: `Segmenter`** — 2節の設計そのまま。現時点では`std::variant<KyteaBackend>`（MLPは4節が実装され次第追加）。

**CLI層** — `predict_command.cpp`は「引数パース→`Segmenter::load`→stdin を1行ずつ読んで`tokenize`→5.1節のフルアノテーション形式で出力」というだけの薄いループ。ビジネスロジックを一切持たない。

### 8.3 MVPのスコープ判断

`segmenter predict --model kytea-model.bin < input.txt > output.txt`（`--boundaries-only`なし）は本来タグ（読み）推定込みの出力を意味するが、タグ推定は単語辞書・サブワード言語モデル・タグモデルまで含む本格実装が必要で範囲が大きい（3.2節の`FeatureLookup`の`tagDictVector`/`tagUnkVector`、9節TODOの通り計算式も未調査）。**最初の実装ステップとしては、`Model`がタグ関連セクションを読み飛ばせる状態で`tokenize_boundaries`（分かち書きのみ）を先に成立させ、`predict`のデフォルト出力もいったんタグなし（`--boundaries-only`相当の出力）にしてしまうのが妥当**。8.2節のレイヤ構成はこの判断に影響されない（`Model`にフィールドを足すだけで済む）ため、後からタグ推定を追加しても大きな手戻りにならない。

### 8.4 リポジトリ全体のディレクトリ構成

8.1節は`include/`/`src/`内部のモジュール分割。ここではビルドシステム・テスト・モデル資産を含めたリポジトリ全体の構成を決める。

```
cpp-segmentlib/
├── CMakeLists.txt                # トップレベル。C++23、サブディレクトリを束ねるだけ
├── cmake/
│   └── CompilerWarnings.cmake    # 警告フラグ等の共通設定（任意）
├── include/segmentlib/           # 公開ヘッダ（8.1節の構成）
│   ├── types.h
│   ├── segmenter.h
│   ├── bytes/binary_reader.h
│   ├── unicode/utf8.h
│   └── kytea/{char_table,automaton,model,scorer,kytea_backend}.h
├── src/                          # 実装（8.1節の構成）
│   ├── CMakeLists.txt            # segmentlib（ライブラリ）のターゲット定義
│   ├── segmenter.cpp
│   ├── kytea/*.cpp
│   └── cli/
│       ├── CMakeLists.txt        # segmenter（実行ファイル）のターゲット定義
│       ├── main.cpp
│       ├── predict_command.cpp
│       └── train_command.cpp
├── tests/
│   ├── CMakeLists.txt
│   ├── unit/                     # モジュール単位のテスト（8.1節の各層に対応）
│   │   ├── binary_reader_test.cpp
│   │   ├── utf8_test.cpp
│   │   ├── char_table_test.cpp
│   │   ├── automaton_test.cpp
│   │   └── scorer_test.cpp
│   └── golden/                   # KyTea実バイナリの実出力と突き合わせるテスト（3.1節末尾で明言した方針）
│       ├── golden_test.cpp
│       └── fixtures/
│           ├── input.txt         # 短いテスト文
│           └── kytea_output.txt  # 本物のkyteaコマンドで生成した期待値
├── models/                       # (gitignore) KyTea/Vaporettoモデル
│   └── kytea/
│       ├── jp-0.4.7-5.mod         # scripts/fetch_kytea_model.sh で取得
│       └── jp-0.4.7-5.vaporetto.zst  # bench/setup.sh で変換
├── bench/                        # 推論ベンチ（9節）
│   ├── setup.sh / run.sh / README.md
│   ├── bench_segment.cpp         # 自ライブラリ in-process
│   ├── bench_kytea.cpp           # libkytea in-process
│   └── {.vendor,corpus,results}/ # (gitignore)
├── scripts/
│   └── fetch_kytea_model.sh
├── docs/
│   └── design.ja.md
├── .gitignore
├── .clang-format
└── .clang-tidy
```

**設計判断（確定）**

- **ビルドシステムはCMake**。C++エコシステムの事実上の標準で、`std::expected`等のC++23機能を使うためのコンパイラ要件指定もしやすい。
- **依存ライブラリはKyTeaバックエンドMVPの範囲ではゼロ**（標準ライブラリのみ）。独自MLPバックエンド（4節）に進む段階で追加の依存ライブラリが必要になるかは別途判断。
- **テストフレームワークは`doctest`**（ヘッダオンリー）。CMakeの`FetchContent`で取得し、単一ヘッダを取り込むだけなのでビルド時間・依存関係のオーバーヘッドが小さい。
- **`models/`はgit管理しない**：`.gitignore`に`models/`を追加し、`scripts/fetch_kytea_model.sh`のようなダウンロードスクリプトのみをリポジトリに置く（Wayback Machineのアーカイブ経由での取得手順もスクリプト化する）。リポジトリを軽量に保つ。
- **`golden/`テストは固定データ方式**：既知の入力文とKyTea実行結果のペアを`tests/golden/fixtures/`に固定データとしてコミットする。CI環境にKyTea本体のビルドが不要になり完結しやすい。期待値は`models/kytea/jp-0.4.7-5.mod`を使い、ローカルでKyTea本体をビルド・実行して作成する（1回作れば以降は固定データとして使い回す）。

### 8.5 ビルド／ツールチェーン要件

- **C++23対応コンパイラが必須**：`std::expected`・`std::byteswap`・`std::span`・（CLIで使う）`std::print`等を使う。
- **macOS：AppleClang（Xcode標準）で問題なくビルド可能**（要検証：十分新しいXcode）。実測確認済み（Xcode 26.2 / Apple Clang 17.0.0、`/usr/bin/clang++`）：デプロイターゲットを明示しない通常のビルドでは全テストが通る。以前はAppleClangを使わずHomebrew LLVM clang（`/opt/homebrew/opt/llvm/bin/clang++`）を必須としていたが、実際にはHomebrew LLVMなしでも動くため、その要件は誤りだった。
  - **注意点は`std::print`/`std::format`のavailabilityゲート**：内部で使う浮動小数点版`std::to_chars`がAppleのlibc++で「macOS 13.3以降のみ利用可能」とマークされているため、**デプロイターゲットをmacOS 13.2以前に明示指定するとコンパイルエラーになる**（`std::expected`自体はOS API非依存の純粋なテンプレート型なので、この制約を受けない）。デプロイターゲットを指定しない通常運用なら影響しない。
  - Homebrew LLVM clangを使う場合は明示的にコンパイラを指定できる：`-DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++`
- **Linux**：GCC 14以降を想定（未実測）。
- **ビルド手順**（AppleClangを含む、システム標準コンパイラでよい）：
  ```
  cmake -S . -B build -G Ninja
  cmake --build build
  ctest --test-dir build --output-on-failure
  ```
- **CMake 4.x + doctest 2.4.11の非互換**：doctest 2.4.11の`cmake_minimum_required`がCMake 4のポリシー下限を下回るため、`tests/CMakeLists.txt`で`FetchContent_MakeAvailable(doctest)`の前後だけ`CMAKE_POLICY_VERSION_MINIMUM=3.5`を設定して回避している。

### 8.6 実装状況

- [x] **L1: `types.h`**（`Segment`/`Segments`/`Boundaries`/`ErrorCode`/`Error`）
- [x] **L1: `bytes::BinaryReader`**（LE整数read・NUL終端文字列・行読み・境界チェック付き。エラーは`ParseError`例外）＋ doctestユニットテスト
- [x] **L1: `unicode::utf8`**（1コードポイントのUTF-8デコード。overlong/サロゲート/範囲外を拒否）＋ doctestユニットテスト
- [x] CMake雛形（`segmentlib`静的ライブラリ＋`segmentlib_tests`、doctestはFetchContent）
- [x] **L2: `kytea::CharTable`**（文字インターン表＋文字種分類`classify`＋正規化`normalize`＋`encode`、3.1/3.2節）＋ doctestユニットテスト。**実モデル`jp-0.4.7-5.mod`でBinaryReaderと合わせてConfig/文字マップのパースを検証済み**（型マーカーがid 1-6、7345文字、`水/飲=K`・`を/む=H`・`。=O`を確認）
  - 4バイトUTF-8のKyTea `findType`バグは追従せず正しく分類する方針（3.1節に記録）。稀なCJK拡張B以降でのみKyTeaと分かれうる
- [x] **L2: `kytea::Automaton<Payload>`**（Aho-Corasickランタイム表現、`Payload`＋読み取りコールバックでエントリ層と分離。ヘッダオンリーテンプレート）＋ doctestユニットテスト（gotos/failure fallback/伝播済みoutput/absent辞書）。**実モデルで検証済み**：`charDict`=81224状態/75377ペイロード（各サイズ6=charWindow×2）、`typeDict`=233状態、`selfDict`=空、wsModelは`numClasses=2`/`multiplier≈0.000109367`。`match`はfailure链を辿らず伝播済み`output`をそのまま使う（3.2節通り）
  - **`find_entry`**（KyTea `Dictionary::findEntry`相当）：goto遷移のみ走査し、終端状態の`is_branch`＋`output`で完全一致を判定（段階A-2でretain）
- [x] **L3: `kytea::Model`**（Config/wsModel FeatureLookup/単語辞書をパースし、タグ・サブワード・LM関連はカーソルを進めてスキップ。`load()`が`std::expected`境界）＋ 合成モデルのdoctestテスト。**実モデル`jp-0.4.7-5.mod`で`Model::load`検証済み**：multiplier=0.000109367、biases[0]=-193、charDict=81224状態、単語辞書=850724エントリ/2080192状態/numDicts=7。**`dictVector`サイズ84 = numDicts(7)×3×dictN(4) が独立に一致**し、ファイル全体のパース整合性を裏付け（ロード約2.1秒、将来最適化余地）
- [x] **タグ推定 段階A-1: モデルローダの retain 化**（`tag_prediction_plan.ja.md` 2節）：これまでスキップしていたタグモデル群を保持に変更。`TagModel`（multiplier/num_weights/`TagFeatureLookup`＝charDict/typeDict/selfDict/biases/tagDictVector/tagUnkVector）、グローバルタグモデル（`global_tags_`/`global_mods_`、レベルごと）、単語辞書エントリの per-word タグ情報（`WordEntry.tags`/`tag_in_dicts`/`tag_mods`）を保持。`CharTable`に id→UTF-8 逆引き `decode()` と `unicode::encode()` を追加し、タグ候補文字列をロード時にUTF-8化。**実モデルで実測確認**（`tag_prediction_plan.ja.md` 1.1節）：lev0（品詞）はグローバル一択（per-word 0件・num_weights=21・selfDict非空）、lev1（読み）は per-word 一択（グローバル無し・候補326369語/per-wordモデル1828語）。WS既存34テスト回帰なし
- [x] **タグ推定 段階A-2: 既知語スコアラ＋出力整形＋CLI**（`tag_prediction_plan.ja.md` 3〜5節）：`kytea.cpp:calculateTags`のelse経路（既知語）を`tag_scorer.cpp`に移植（`addTagNgrams`/`addSelfWeights`/`addTagDictWeights`/`getDictionaryMatches`）。ディスパッチ（globalMods優先→per-wordモデル→未知語）・`num_weights`基準のループ（候補配列長と不一致な場合がある実測どおり）・`std::sort`+`kyteaTagMore`複製での先頭候補確定を忠実再現。`Automaton`に`find_entry`（KyTea `Dictionary::findEntry`相当、`is_branch`をretainして厳密化）を追加。出力は`append_tagged_line`（表層エスケープ・タグ非エスケープ）＋`--notags`。goldenを拡張し既知語の品詞・読みはバイト一致
- [x] **タグ推定 段階B: 未知語の読み推定（サブワード辞書＋LM＋ビームサーチ）**（`tag_prediction_plan.ja.md` 8節）：`Kytea::calculateUnknownTag`/`generateTagCandidates`/`KyteaLM::scoreSingle`を移植。モデルローダに`subwordDict_`（`ProbSubwordEntry`：読み候補は生CharId列で保持しUTF-8化しない）と`numTags`個の`KyteaLM`（`n`/`vocabSize`＋`probs`/`fallbacks`の`unordered_map<u16string,double>`、`NEG_INFINITY=-999.0`センチネル除外）を追加（読み順：wordDict→subwordDict→per-level LM、`Kytea::readModel`で確定）。`unkBeam=50`/`tagMax=3`/`defTag="UNK"`はKyTeaの実行時既定値（モデル非格納）を採用。**実測：ゴールデン20文＋ストレス15文＋アオゾラ71万字を含む全検証でKyTeaの既定（タグ付き）出力とバイト完全一致**（未知語の読み推定・`UNK`フォールバックまで含む）。唯一の既知の相違はCJK拡張B文字（`findType`バグの意図的不追従、10節）に由来するWS分岐のみ
- [x] **L5: `kytea::KyteaBackend`** / **L6: `Segmenter`**（`std::variant`ディスパッチ、`string_view`入力＋`std::expected`、2/6節）
- [x] **CLI: `segmenter predict`**（`--model`/`--threads`、stdin→stdoutフィルタ、`segmenter train`は未実装スタブ、7節）。既定はタグ付き出力（`append_tagged_line`＝表層/品詞/読み）。`--notags`／`--boundaries-only`はタグを付けず分かち書きのみ——両者とも**タグ計算を丸ごと省く高速パス**（`tokenize_boundaries_all`→`append_boundary_line`）を通し、`kytea -notags`とバイト一致（golden テストで境界パス＝既存WSパスの一致も固定）。※`--boundaries-only`は当初パースのみで無視される死んだフラグだったのを配線（段階B完了後に発見・修正）
- [x] **golden テスト（KyTea実出力とのバイト単位一致、8.4節）**：15文の固定fixtureで`kytea -notags`と照合。**model未取得時はスキップ**（CI耐性）
- [x] **未知文字（学習語彙外）の等価性を検証**：KyTeaの動的ID採番（`mapChar add=true`）と本ライブラリの`kNoChar`落としが、WS素性が全てオートマトンマッチ（学習時ID `1..K`）であるため分かち書き出力上は完全等価であることを、`string-util.cpp`/`kytea.cpp`のソース確認＋未知の絵文字・記号でのバイト一致で確定（3.1節・10節）。golden fixtureに未知文字行（😀😱ℵℶℷ★☆♠🍣ⅠⅡⅢ）を追加して回帰固定
- [x] **🎉 KyTeaとバイト単位で完全一致を達成**：多様な入力600行（半角/全角混在・英数字・記号・古文・顔文字的連続）で`kytea -model jp-0.4.7-5.mod -notags`と`diff`一致。出力の記号エスケープ（`showEscapedString`：スペース/`/`/`&`/`\`を`\`で前置）まで再現（例：`Hello \  World`、`2024 \/ 12`）

## 9. ベンチマーク（推論速度）

`bench/`。KyTea / Vaporetto との推論速度比較。

### 9.1 設計原則：「速度の前に、同じ計算をしていることを保証する」

- **同一モデル**：KyTeaの`jp-0.4.7-5`を`convert_kytea_model`でVaporetto形式(zstd)に変換し、3ツールとも同じ重みで動かす。
- **正確性ゲートを先に通す**：タイミング前に出力を`diff`。segmentlibはKyTeaとバイト一致、Vaporettoの相違率を報告する。
- **純粋推論はin-processで測る**：モデルを1回ロードし、tokenizeループのみを計測（ロード・I/Oを除外）。
  - **CLIのwall-clock差分では推論を測れないと実証**：初版でCLIの「小/大コーパス2サイズ差分」を試したが、(a) I/O・出力整形が限界コストを支配し、(b) Vaporettoのロード分散（数秒）が推論(<0.1s)より遥かに大きく差分が破綻。in-process実測(2.96M/s)とCLI差分(0.60M/s)が5倍乖離した。→ 各ツールを専用ハーネスでin-process計測する方式に変更（`bench_segment`＝自ライブラリ、`bench_kytea`＝libkyteaリンク、Vaporettoは自己申告の`Elapsed`）。
- **条件固定**：シングルスレッド・タグなし・ウォームアップ後best-of-N。指標はUnicodeコードポイント/秒（Vaporettoの慣習に合わせる）。
- **コーパス**：青空文庫の実在作品（漱石・太宰・芥川・宮沢、約71万文字）。同一行の繰り返しはキャッシュが非現実的に効くため実文を使う。

### 9.2 最新結果（Apple M1 Pro、青空文庫71万字、best-of-5）

**正確性ゲート**
- segmentlib vs KyTea：**0 / 20822 行相違（100%一致）**
- Vaporetto vs KyTea：92 / 20822 行相違（99.56%一致。`し よう`↔`しよう`等、Vaporetto公式が認める「些細な違い」）

**純粋推論速度（in-process、ロード・I/O除外、シングルスレッド）**

| ツール | ロード | M文字/秒 | 対KyTea |
|---|---|---|---|
| **segmentlib** | **412 ms** | **6.21** | **約5.3×** |
| KyTea | 964 ms | 1.17 | 1.00× |
| Vaporetto | 数秒（daachorse構築） | 8.68 | 約7.4× |

**segmentlibの並列スループット（`tokenize_all`、best-of-5）**

| スレッド数 | M文字/秒 | 単スレッド比 | 対KyTea |
|---|---|---|---|
| 1 | 5.65 | 1.00× | 4.8× |
| 2 | 12.91 | 2.28× | 11.0× |
| 4 | 25.91 | 4.59× | 22.1× |
| **8** | **44.98** | **7.96×** | **38.4×** |

（segmentlibの数値は9.4節の最適化 第10弾まで反映。最適化前は2.84 M/s・2.1×・ロード〜1450ms。推論は正準double-array＋root直引き＋encode最適化で大幅向上、ロードは第10弾のfree-list cap最適化でKyTeaより速い水準に。8スレッドの並列は**Vaporettoのシングルスレッドの約5.2倍**。絶対値は熱状態で±10〜20%程度ぶれるため、同一バッチ内の比較が安定した指標。）

**知見**
- segmentlibは**KyTeaとバイト完全一致しながらシングルスレッド推論5倍以上、ロードも速い**（`bench_kytea`の単語数`sink`が完全一致し、同じ分割を確認）。KyTeaの実装（素性文字列のハッシュ・神クラス）を避け、フラット配列＋正準double-array Aho-Corasick＋root直引きにした効果。
- Vaporettoはシングルスレッドでは依然最速（正準化・事前加算・SIMDが揃ったdaachorse実装）だが、出力が厳密には非一致（0.44%）。segmentlibは**マルチスレッド化でVaporettoのシングルスレッドを上回る**（8スレッドで約5.2倍）。

### 9.2.1 タグ推定込みの速度（段階A-2/B、参考値）

上表（9.2節）は**分かち書きのみ**（`-notags`相当・`tokenize_boundaries`）で条件固定した比較。タグ推定（品詞＋読み、`tokenize`既定）はKyTea本体の`calculateTags`と同じく**語ごとに窓（前後char_n/type_n文字）を毎回ルートから再走査**するアルゴリズム（分かち書きの文全体1回連続走査とは構造が異なる）で、以下は分かち書き専用経路との比較のための参考値。

推論速度（ロードとは独立。ロードは推論パスに関係なく一度きりの同一コスト＝下記「ロード時間の内訳」参照）：

| 経路 | M文字/秒（単スレッド） | 対WSのみ |
|---|---|---|
| WSのみ（`tokenize_boundaries`、9.2節と同一条件） | 5.6〜6.8 | 1.00× |
| **タグ込み（`tokenize`既定、品詞＋読み）** | **0.74〜0.83** | **約8×低速** |

（Apple M1 Pro、青空文庫71万字、best-of-7。タグ込みは9.4節 第12弾のargmax化後の値。参考：外部CLI計測ではKyTea自身も`-notags`→既定（タグ付き）で約3倍低速化。絶対値は環境依存だが、我々のタグ込み実測（in-process 0.74〜0.83 M/s）はKyTeaのタグ付きCLI（〜0.12 M/s、I/O込み）より速い。**重要な計測上の教訓**：段階A-2/B実装直後、`build/`が`CMAKE_BUILD_TYPE=Debug`のまま残っていたのに気づかず「65〜70倍低速」という数値を出してしまった。`Release`再構成のみで0.10→0.66 M/sまで回復——それは計測ミスであり実装のリグレッションではなかった。**ベンチマーク実行前は必ず`cmake -B build -DCMAKE_BUILD_TYPE=Release`でビルド構成を確認すること**）

**ロード時間の内訳（実測）**：モデルロードは約0.8〜1.3秒（熱状態でブレ大、同条件で1.1〜2.0秒の観測もあり）。パース各段を計測した結果、**word_dict の構築が約88%（〜1150ms）** を占め、**subword辞書＋2レベルLM は合計わずか約16ms（約1%）** だった。当初「ロード増加の主因は subword/LM」と推測していたが**実測で否定された**——subword辞書は3ms、LM群は14ms程度に過ぎない。ロードの主コストは、分かち書き時代から存在する word_dict（85万語・200万状態のdouble-array構築）で、段階A-1で per-word タグ情報の保持＋85万語ぶんのタグ文字列UTF-8デコードが加わった分がそこに含まれる。

**未着手の最適化余地**：
- モデルロード短縮を狙うなら対象は **word_dict のパース／double-array構築**（subword/LMの遅延ロードは約16ms＝1%しか縮まず割に合わない、実測で確認）。
- タグ推定自体の高速化：先頭候補選択のargmax化は実施済み（9.4節 第12弾、+10〜20%）。payload平坦化はA/B計測で効果なしと判明し撤回済み（第11弾＝負の結果）。`add_tag_ngrams`（Releaseプロファイルで約31%）の拡張加算ループは明示SIMDカーネル化済み（第13弾、ループ単体1.65倍・エンドツーエンドはアイドル時再計測待ち）。事前計算方式は走査でなく加算が律速のため不採用と結論（10節TODO）。現状WSのみ比で約8倍低速だがKyTeaのタグ付き実測より速い。

### 9.4 推論最適化（バイト一致を保ったまま）

プロファイルで内訳を実測し、`golden`テストでバイト一致を常時検証しながら段階的に最適化した（開始 2.84 → 3.37 M文字/秒、約 +19%）。

- **内訳（初期）**：`score_boundaries`が89%（うちAho-Corasickマッチング〜120ms、重み累積〜90ms）、`encode`が11%。
- **効いた施策：root遷移のO(1)直接テーブル化**（`Automaton::build_root_index`）。root状態は毎文字fallbackで叩かれる最ホット状態で、多数のgotosへの二分探索を`CharId`直接引きに置換 → マッチング 120→90ms（+17%）。
- **効かなかったが採用した施策**：`add_dictionary_scores`の`on`配列を「thread_localで再利用＋触れた箇所のみクリア＋マッチ時に直接加算」に変更。速度は横ばい（アロケータが同サイズ確保を安価に再利用、全走査もベクトル化されていた）だが、**コードが簡潔になりスレッドセーフも維持**。同様に`encode_into`/`score_boundaries_into`でthread_localバッファ再利用（CLIパスとマルチスレッド化に有効）。
- **試したが不採用：n-gram重みの事前加算（Vaporetto流）**。「同じ位置で終わる全n-gramは同じオフセットで加算される」ことを利用し、Aho-Corasickの各状態の出力重みをロード時に要素ごと事前加算して1本にまとめ、実行時は状態あたり固定幅8の1回加算（＋マージンパディングでクリッピング分岐除去）にする実装を行った。**バイト一致は保てたが速度は微減**したため不採用（`NgramScorer`は削除）。
  - 理由：**KyTeaのcharDictは 0.93 outputs/state（81224状態 / 75377エントリ）**で、状態あたり出力が1以下。事前加算は「1本を1本にまとめる」だけで削減がなく、固定8幅（実質6幅で足りる）＋パディング確保・コピーのオーバーヘッドだけが乗った。
  - **洞察**：この技法がVaporettoで効くのは**double-array PMA（O(1)走査）とセットだから**。走査が速くなって初めて累積が相対的に重くなり、事前加算＋SIMDが効く。**単独では、かつoutputs/stateが低いKyTeaモデルでは効かない**。＝最適化は「走査の高速化（double-array）」が先で、事前加算はその後に意味を持つ、という順序依存がある。
- **効いた施策 第3弾：Automatonのdouble-array化**（`base_[s] + c` で `check_[slot]==s` を確認するO(1)遷移）。全Aho-Corasickオートマトン（charDict・typeDict・wordDict）を、状態ごとにヒープ確保した`gotos`ベクトルへの二分探索から、`base_`/`check_`/`next_`の3本のフラット配列へ移行。`failure`と`outputs`もCSR（`fail_`・`out_offset_`/`out_flat_`）に平坦化し、走査中にstate単位ベクトルへポインタを追わないようにした。**同一条件でlegacy比 +75%（3.04→5.33 M文字/秒）**、バイト完全一致を維持（golden＋3000行の実コーパス差分ゼロ）。
  - **効果の内訳**：char/typeのdouble-array化は +5%程度だが、**wordDict（208万状態・根の巨大fan-out）のdouble-array化が大半（+約20%相当）**を占めた。legacyでは巨大gotosリストの二分探索＋ポインタ追跡が重いため、O(1)遷移＋キャッシュ局所性の恩恵が最も大きい。
  - **パッキング密度が推論速度に直結**：構築時の詰め方（総スロット数 / 総状態数）が疎になるとキャッシュ効率が落ち推論が遅くなる。実測でcap（下記）を疎側に振ると 4.2 M/s まで低下、密なら 5.3 M/s。
  - **構築コスト（load時間）のトレードオフ**：double-array構築は「各状態の子を衝突なく詰める`base`探索」で、素朴な無制限探索だと**ごく少数の超高fan-outノード**（根等）が数百万長のフリーリストを全走査し超線形になる（wordDict単体で〜950ms）。対策として**フリーリスト長を`kFreeListCap`で上限化**（超過分は最古スロットを`kClosed`で放棄→探索窓を書き込みフロンティア近傍に限定）。`262144`が**KyTeaの最大オートマトンで満充填（無制限と同一の2.187Mスロット）に到達する最小の上限**で、これより大きいオートマトンは疎化して構築時間を抑える方向に緩やかに劣化する。結果、ロードは 150→〜1150ms（KyTeaの800msと同程度）。
  - **試したが不採用の詰め方**：fan-out降順配置は満充填（ほぼ無駄ゼロ）を与えるが、大ノードを先に置くと配列が早期に最大化しフリーリストが巨大化→**構築が37秒に爆発**。高fan-outノードだけ末尾確保は、そのノードのCharId幅ぶん丸ごと無駄になり**スロットが10倍以上に膨張**。いずれもcap無しでは破綻し、file順＋cap＋隙間詰めが最良だった。
- **試したが不採用：n-gram重みの無条件加算＋マージンパディング（第4弾の当初案）**。全ペイロードが一様に幅6（`window*2`）と実測できたため、両端に`window`のパディングを張って各マッチを固定幅6で無条件加算（クリッピング分岐を除去してベクトル化可能に）する版を実装。**バイト一致は保てたが速度は横ばい**（4.84→4.83 M/s）だったため不採用。
  - **理由（プロファイルで確定）**：`score`が全体の86%だが、その内訳は**走査が支配的で加算はほぼ無償**。実測で `charDict.match`＋加算（40.5ms）と `charDict.match` のみ（41.4ms）が同値＝**加算コストは測定限界以下**。走査の内訳は char 22ms・type 8ms・wordDict 29ms（＝計59 msが走査）。第2弾（事前加算）と同じく、**KyTeaモデルでは累積ではなく走査が律速**という同根の負の結果。
- **効いた施策 第4弾：double-arrayの`check`と`next`を1セル（8バイト）に統合**。別配列だと1遷移で`check_[slot]`と`next_[slot]`の2キャッシュラインを触るが、`struct Cell{check,next}`の単一配列`cells_`にすることで**1遷移=1キャッシュライン**に。走査律速の実態に直接効き、**+10%（4.84→5.33 M/s）**、対KyTea **4.0×**、バイト一致維持。
- **効いた施策 第5弾：正準double-array化**。状態IDをスロットIDにリネームし、`next`配列を廃止（スロットIDが遷移先そのもの）、`base_`と`cells_`を単一`unit_[]{base,check}`に統合。テーブルが縮み（wordDictで25.8→17.5MB）、`unit_[cur]`のセルが「到着時のcheck検証読み」と「次遷移のbase読み」で共有され局所性が上がる。**+17%（5.33→6.26 M/s）**、対KyTea **約4.3×**、バイト一致維持。
  - **配置順が決定的（DFS前順）**：正準化は「子のスロット＝新ID」を親配置時に確定する必要があり、親→子のトポロジカル順で処理しなければならない。**BFS順は最悪**：兄弟が配列上に散らばり衝突が激増（wordDictの`base`探索走査が 96M→490M、ロード5.7秒）。**DFS前順**はサブツリーを連続配置してフリーリストのフロンティアを局所化し、走査 113M・**パッキングはほぼ1:1（208万状態→208.9万スロット）**・ロード〜1450msに収束。二重配列構築は「順序」がコスト・密度を支配するという教訓。
- **試したが不採用：セルの24bitパッキング（8→6バイト、第6弾）**。状態数<2^21なので`base`/`check`を各24bitに詰めて`unit_`を6バイト/セルにし、wordDictのテーブルを16.7→12.5MBに縮小する版を実装（末尾1セルのパディング＋3バイト安全読みで境界処理、リトルエンディアン前提）。**バイト一致は保てたが不採用**：
  - **ロードが悪化**（同一熱状態で 8B〜1450ms → 6B〜2083ms）。構築が`set_base`/`set_check`のバイト書き込み＋`fit`探索の3バイト安全読みで重くなった。
  - **推論も改善せず**（冷えた比較で 6B 5.1 < 8B 6.3）。6バイトstrideの**非アラインアクセス**＋マスク処理のコストが、テーブル縮小の利得を相殺。加えてM1 Proの**大容量SLC（〜24MB）が既に16.7MBの8Bテーブルを保持**できており、12.5MBへの縮小はミス率をほぼ下げなかった（L2 12MBには12.5MBでも収まらない）。
  - 教訓：**アラインメント＞テーブルサイズ**（このハード・このモデルでは）。8バイト境界に乗る`{i32,i32}`のほうが速い。
- **効いた施策 第7弾：入力符号化（encode）のハッシュ全廃**。`encode`は文字ごとに `normalize`・`id_of(文字)`・`id_of(型マーカー)` の**3回`unordered_map`を叩いていた**（プロファイルで全体の約14%＝18ms）。日本語はほぼBMP（<U+10000）である点を利用し、(1) `normalize`をモデル非依存の**静的BMP直引き表**に、(2) 文字ID解決を**BMP直引き配列`bmp_ids_`（128KB）**＋astralのみhashフォールバックに、(3) 6種の型マーカーIDを構築時に**事前計算**（`type_ids_[classify]`）してhashを全廃。**encode 18→7.4ms（約2.4×）**、全体で対KyTea比 4.3→約4.9×、バイト一致維持。走査（automaton）ではなく符号化側の純粋なオーバーヘッド削減。
- **効いた施策 第8弾：root遷移の直引きインデックス**。正準形では root は unit 0 で、`step(0,c)`は17.5MBの`unit_`配列の**散在スロット**を読む。だがテキストの大半の位置は辞書語を開始せず、`cur`は root（またはfailで root に落ちる）に留まるため、この散在読みが最頻。root の子だけを**CharId直引きの小配列`root_next_`（〜32KB、L1/L2常駐）**にして`step(0,c)`と等価に置換（`root_next_[c]==step(0,c)`を構築時に保証）。match ループで`cur==0`時のみこの高速路を通す（他状態のstepは不変）。**wordDict走査 29→13.15ms（約2.2×）**、全体 6.7 M/s・**対KyTea 4.7×**、バイト一致維持。
- **効いた施策 第9弾：行単位のマルチスレッド化（バッチAPI）**。推論パスは既に再入可能（Modelはimmutable・scratchは`thread_local`）なので、`Segmenter::tokenize_all(span<string_view>, threads)`を追加。**アトミックカーソルで64行チャンクを動的割当**（行長のばらつきを負荷分散、アトミックはチャンクで償却）、各スレッドは互いに素な結果スロットへ書くのでロックなし。呼び出しスレッドもワーカーとして参加。青空文庫20822行・M1 Proで **1→7.4 / 2→14.5(1.96×) / 4→26.6(3.59×) / 8→41.4 M文字/秒(5.60×)**。**41.4 M/s は Vaporetto単スレッド（〜10.4）の約4倍**、KyTea単スレッド比 約32×。並列結果は直列`tokenize`とバイト単位で一致（1/2/4/8スレッドでテスト）。高スレッド側の逓減はeffコア混在とメモリ帯域（17.5MBテーブル共有）による。
- **効いた施策 第10弾：free-list cap を DFS 前提で最適化（ロード短縮）**。当初は`daachorse`流の**ブロッククローズ**でロードを縮める計画だったが、実測すると**cap定数を下げるだけで同じ目的が達成できた**。理由：DFS前順配置では*有用な*空きスロットは常に書き込みフロンティアの至近にあり、`kFreeListCap`を大きく取ると「このサブツリーが既に通り過ぎ二度と使わない古い空きスロットの長い尾」を溜め、`base`探索がそれを無駄に舐める。cap を `262144→8192` に下げると、wordDict構築が **load 2224→373ms（約6×）**、推論は落ちない（同一バッチ実測 5.54→6.9 M/s）。**block-close 未実装で目的達成**。
  - **なぜ推論が落ちないか**：小capはwordDictを疎化する（units +25%、20.8MB）が、**root直引き（第8弾）でwordDictの`unit_`は既にcold**（大半のアクセスは32KBの`root_next_`）。疎化した部分は深い・稀アクセス領域なので効かない。一方 char/typeは小さく cap窓を埋めないので**ほぼtightのまま**（hotなchar走査は密度維持）。＝第8弾の副次効果が第10弾を可能にした順序依存。
- **試したが不採用 第11弾：`Automaton<Payload>`のpayload平坦化**（負の結果）。段階A-2/B完了直後、タグ推定込みの`tokenize`が65〜70倍低速という数値を見て「`addTagNgrams`（品詞lev0が全語で発火）が語ごとに小窓を再走査し、`Payload=FeatVec`（`vector<vector<int16_t>>`）の個別ヒープ確保が疎参照＝キャッシュミスを起こす」と推測し、`Payload`が`std::vector<T>`のとき全ペイロードを連続バッファ＋オフセット配列に平坦化して`span`を返す最適化を実装した。**しかし2つの問題が判明して撤回した**：
  1. **真因は計測ミス**：65〜70倍の大半は`build/`が`CMAKE_BUILD_TYPE=Debug`のまま残っていたため。**`Release`再構成だけで0.10→0.42〜0.66 M/sまで回復**（ロードも5.6秒→1.9秒）。実装のリグレッションではなくビルド構成の見落としだった。
  2. **平坦化に測定可能な効果なし**：`Release`で平坦化あり/なしを同一マシン・背中合わせ・best-of-7で6ラウンド交互計測したところ、両者の分布は完全に重なった（no-flat最良0.805・中央値約0.667、flat最良0.732・中央値約0.678。変種間の差＜熱ノイズ±15%）。推測した「個別ヒープ確保による疎参照」は、読み取り専用ペイロードがロード時に連番malloc→ほぼ隣接配置されること・ベクタヘッダ配列自体は連続で余分な間接参照1回は分岐予測が効くこと、から実測上のボトルネックではなかった。
  - **教訓**：`if constexpr`によるテンプレート二重化・戻り値型の分岐（`span`⇄参照）・「空span＝not-found」の潜在的落とし穴という相応の複雑性を、A/B計測前に導入してしまった。**まず`Release`で正しく計測し、効果を確認してから複雑性を入れるべきだった**。撤回して元のシンプルな`vector<Payload>`のままとした。タグ推定を本当に速くするなら、格納レイアウトではなく語ごとの再走査そのものを削減する事前計算方式が本筋（9.2.1節・10節TODO）。
- **効いた施策 第12弾：タグ推定の先頭候補選択を毎語フルソート→線形argmaxに**（タグ推定 +10〜20%）。`predict_word_tags`は各語・各レベルで候補（品詞21クラス等）の`(index, score×multiplier)`ペアベクタを作り`std::sort`で降順ソートして先頭だけ使っていた。実際に必要なのは最大要素のみなので、`std::sort`＋ペアベクタ構築＋double乗算を廃し、整数スコアの単純な線形最大スキャン1本に置換（`multiplier>0`より整数スコアの順序＝confidence順序で、同点構造も一致）。Releaseプロファイルで`std::__introsort`が全体の約7%を占めていた。**バイト一致の要注意点**：`std::sort`の同点順序は未規定で、これまでKyTeaと一致していたのは同一libc++の`std::sort`を走らせていたから。線形argmax（最小indexの最大値）は理論上は同点時に分岐しうるが、青空文庫20822行＋goldenでバイト完全一致を確認（実データで同点なし）。**背中合わせA/B（best-of-7×4ラウンド）で全ラウンドargmaxが勝利**（sort 0.63〜0.71 / argmax 0.74〜0.81 M/s、分布の重なりなし＝第11弾とは対照的に実効果あり）。コードも短くなった（`ranked`スクラッチ削除）。
- **第13弾：タグ推定の拡張加算ループを明示SIMDカーネル化**（10節TODOの`add_tag_ngrams`高速化＝タグ推定の最大コスト・全体の約18%と実測済みだった項目）。`int32 scores[j] += int16 vec[pos+j]`（nw=21）の拡張加算は、実行時境界のためコンパイラが自動ベクトル化しない（`__restrict`でも不変と実測済み）。MLP側の`kernels.h`（scalar oracle＋NEON/AVX2/scalarディスパッチ＋bit一致テストのインフラ）に拡張加算カーネル`add_widen_i16_i32`を追加し、`tag_scorer.cpp`の4箇所（`add_tag_ngrams`のマッチ加算・`add_self_weights`・`add_tag_dict_weights`・bias加算）から使用。カーネルは汎用整数演算でMLP固有ではないため配置はkernels.hのまま（KyTeaバックエンド→mlp/kernels.hのクロスモジュール依存はヘッダコメントに明記）。**整数加算は厳密なのでバイト一致は構成上保証**——golden・青空文庫20822行の新旧バイナリ出力バイト完全一致・全115テストgreenで確認。**ループ単体の実測はプロセス内交互A/B（min-time、実行時nw=21、add_tag_ngramsと同じアクセスパターン）でscalar 1.71ms vs NEON 1.04ms＝1.65倍**（2回再現・負荷に頑健な計測法）。期待されるエンドツーエンド効果は 18%×(1−1/1.65)≈**7%**。**ただしエンドツーエンドのA/Bは未確定**：計測時にマシンがビルド/nodeプロセスで高負荷（load average 31、throughputが0.24〜0.96 M/sで±50%ぶれ）だったため、tokenize全体での確認はアイドル状態での再計測待ち。1幅化実験（+9〜25%）との差分は、あちらがロード削減20要素分も含むため（10節TODOに記録済みの留意点どおり）。

- `bench/setup.sh`：青空文庫取得＋クリーン、Vaporettoビルド、モデル変換（1回だけ）。
- `bench/run.sh`：正確性ゲート → in-process推論計測 → 表出力。`bench/results/`に保存。
- `bench/bench_segment.cpp`（自ライブラリ）・`bench/bench_kytea.cpp`（libkytea）・Vaporettoは`predict`の`Elapsed`。
- `bench/.vendor/`・`bench/corpus/`・`bench/results/`はgit管理外。

## 10. 未決事項 / TODO

- [x] モデルファイルのセクション構成・型（3.2節で確定：ヘッダ→Config→wsModel→タグモデル→辞書→サブワード辞書→LM）
- [x] コーパス仕様（フル/部分アノテーション、5節で確定）
- [x] 複数バックエンドを束ねる全体アーキテクチャ（2節：`std::variant`ベース、`Segmenter`は薄いディスパッチャ）
- [x] `DictionaryState`（Aho-Corasickオートマトン）のバイナリ層の正確なバイト列仕様（3.2節：`failure`/`gotos`/`output`/`isBranch`、および`writeEntry<Entry>`のEntry型ごとの差異まで確定）
- [x] 単語分割の推論アルゴリズム（3.3節：`calculateWS`のスコア計算式と境界判定の閾値（`score > 0`）を`kytea.cpp`から復元）
- [x] `KyteaChar`と文字の対応関係（3.2節：固定Unicodeコードポイントではなく、モデルごとの文字インターン表。ID`0`は予約済みセンチネル、実文字は`1`始まりで学習時の初出順）
- [x] `mapChar`が推論時に未知文字（学習語彙にない文字）をどう扱うか：**解決**。KyTeaは`mapString`→`mapChar(add=true)`で未知文字に新規ID（`charTypes_.size()`）を動的採番するが、WS素性は全てオートマトンマッチ（学習時ID `1..K`）のため、未知文字を`kNoChar=0`に落とす本ライブラリと**分かち書き出力は完全等価**（3.1節に論証＋実測。golden fixtureに未知文字行を収録しバイト一致を固定）。CJK拡張B漢字の相違は別軸の`findType`バグ（意図的）。
- [x] テキスト形式（`T`）とバイナリ形式（`B`）のどちらを主対応にするか：**バイナリ（`B`）のみ対応で確定・対応不要**。配布モデルはバイナリで、テキスト形式は自前で`train-kytea -modelformat text`しない限り遭遇しない。テキスト形式への対応は行わない。
- [x] 非量子化モデル（`FeatVal=double`、バージョン`"0.4.0NQ"`）への対応要否：**対応不要で確定**。配布モデルは量子化ビルド（`int16_t`）。非量子化は`DISABLE_QUANTIZE`ビルドで自前学習した場合のみ生成され、実運用で遭遇しない。`"0.4.0NQ"`ヘッダを検出した場合はエラーとする方針（暗黙の誤読を防ぐ）。
- [x] サブワード辞書（`Dictionary<ProbTagEntry>`）・言語モデル（`KyteaLM`）・タグ推定のスコア計算式（`addTagNgrams`/`addTagDictWeights`/`addSelfWeights`/`scoreSingle`）：**段階A-2/Bで実装・解決**（8.6節）。既知語は`addTagNgrams`系を`tag_scorer.cpp`に移植、未知語の読みは`generateTagCandidates`のサブワード格子DP＋ビームサーチ＋`KyteaLM::scoreSingle`のn-gramバックオフを忠実移植。ゴールデン・アオゾラ71万字でKyTea既定（タグ付き）出力とバイト完全一致
- [x] **Vaporetto互換バックエンドは作らないことを決定。** Vaporettoのモデルバイナリフォーマット（zstd外装＋`MODEL_MAGIC`＋bincode2 standard設定のフラットリスト構造）は初期段階で調査しワイヤーフォーマットも実測で確認したが、バックエンド自体は実装せず、KyTea互換＋独自MLPの2バックエンド構成（1節）を最終形とする。Vaporettoは引き続き外部ベンチマーク比較対象としてのみ使用する（9節）
- [ ] 独自MLPバックエンドのネットワーク構成・学習方式（4節、要件がある程度固まってから設計）
- [x] 特徴抽出の高速化方式（KyTea自身もAho-Corasickを採用済みと判明。Double-Array化を実施し大幅に効くことを確認 — 9.4節 第3〜10弾）
- [x] CLIのサブコマンド構成（7節：`segmenter predict`/`segmenter train`。オプション形状は確定、学習エンジンの中身は未着手のまま）
- [x] `--boundaries-only` を配線（段階B完了後、パースのみで無視される死んだフラグだったのを発見・修正）。`--notags`と共にタグ計算を省く高速パス（`tokenize_boundaries_all`→`append_boundary_line`）を通し、`kytea -notags`とバイト一致。goldenで境界パス＝既存WSパスの一致も固定（8.6節CLI項）
- [~] 複数候補＋信頼度出力（KyTea `-out conf`／`-tagmax`）：**通常ユースケース（最尤1解）には不要のため実装しない**（KyTea/MeCabとも既定single-best、N-best・周辺確率はニッチ用途）。当初plan §5で`-alltags`と誤記していたがそのオプションは実在しない。実装するなら段階A-2で省いたマージン計算＋全候補保持＋float整形の再現が要る。必要が生じるまで見送り（tag_prediction_plan §5）
- [~] 部分アノテーション入力＋ハード制約（`-wsconst`相当、§238/§6.2）：制約付き解析は未実装。配布jpモデルの`wsConstraint`は通常空で既定出力のバイト一致には無影響。必要が生じるまで見送り
- [x] 学習機能（KyTea互換の学習エンジンを自前実装するか、外部LIBLINEAR連携にするか）：**どちらも行わないことに決定**。3節が当初から掲げていた「KyTeaバックエンドは推論のみ」というスコープを最終確定とする。理由：(1) 独自MLPバックエンドは自前学習エンジンが完成・実運用済み（4.9節、design5.8実績）で、学習機能自体は本ライブラリに既に存在する。(2) KyTea互換モデルの学習が必要な場面（本ライブラリの評価・比較、`corpus/ud-gsd/kytea.mod`の生成等）では、実際には常に本物の`train-kytea`（Homebrew配布、LIBLINEAR本体を内包）を外部ツールとして直接呼んでおり、この運用で用が足りている。(3) LIBLINEARの自前再実装は「素性抽出は一字一句忠実再現が必要（3.1節）だが分類器自体は汎用的」という3節の整理どおり、素性側と違い忠実再現の必然性がなく、外部の実装（LIBLINEAR本体）をそのまま使う方が正しい車輪の再発明回避になる。`segmenter train --backend kytea`は7.2節の初期案オプション形状を実装せず、現状どおり明示的な「not implemented」エラー（`train_command.cpp`）のままとする
- [x] `--encode` の実際の対応範囲（UTF-8以外を切り捨てるか）：**実装しないことに決定**（7.1節）。フラグ自体を追加せず、入力は常にUTF-8固定。不正なUTF-8は行を切り捨てず、出現行以前の出力をflushして即座にexit code 1で中断（既存の`CharTable::encode`のエラー経路で実現済み・実測確認済み）。他エンコーディングを選ぶ動機（モデル・コーパスとも常にUTF-8前提）がそもそもないため、この判断で完結
- [ ] pmr版APIの要否をベンチマーク後に判断
- [x] KyTea/Vaporettoとの推論ベンチ（9節）：正確性ゲート＋in-process計測。最新値（9.2節）：segmentlibはKyTea比 シングルスレッド推論5.3×・ロードも高速化、8スレッドで38.4×
- [x] 推論の高速化 第1弾（9.4節）：root遷移のO(1)直接テーブル化＋バッファ再利用で 2.84→3.37 M/s（+19%）、バイト一致維持
- [~] 推論の高速化 第2弾：n-gram重み事前加算（Vaporetto流）を実装したが、KyTeaモデルは0.93 outputs/stateで効かず不採用（9.4節、負の結果を記録）
- [x] 推論の高速化 第3弾：Automatonの**double-array化**（`base_[s]+c`のO(1)遷移＋fail/outputsのCSR平坦化）。全オートマトンをdouble-array化し **legacy比 +75%（3.04→5.33 M/s）**、バイト一致維持。wordDict（208万状態）の寄与が最大。構築はfree-list cap（`262144`）で線形化、ロードは〜1150ms（9.4節）
- [~] 推論の高速化 第4弾：n-gram重みの無条件加算＋SIMD（事前加算の再評価）はプロファイルで「加算は無償・走査が律速」と判明し不採用（第2弾と同根の負の結果）。代わりに**`check`/`next`を1セルに統合**（1遷移=1キャッシュライン）で **+10%（4.84→5.33 M/s、対KyTea 4.0×）**、バイト一致維持（9.4節）
- [x] 推論の高速化 第5弾：**正準double-array化**（状態ID=スロットID、`next`廃止、単一`unit_[]{base,check}`に統合）。**+17%（5.33→6.26 M/s、対KyTea 約4.3×）**、バイト一致維持。DFS前順配置がパッキングをほぼ1:1に（BFSは衝突激増で不可）。テーブルも25.8→17.5MBに縮小（9.4節）
- [~] 推論の高速化 第6弾：`unit_`セルの24bitパッキング（8→6バイト）を実装したが、非アラインアクセス＋構築コスト増で不採用（推論改善なし・ロード悪化）。**アラインメント＞テーブルサイズ**という負の結果（9.4節）
- [x] 推論の高速化 第7弾：**encode のハッシュ全廃**（normalize/文字ID/型IDをBMP直引き・事前計算に）。encode 18→7.4ms（約2.4×）、対KyTea 約4.9×、バイト一致維持（9.4節）
- [x] 推論の高速化 第8弾：**root遷移の直引きインデックス**（`root_next_`）。wordDict走査 29→13ms（約2.2×）、対KyTea 4.7×、バイト一致維持（9.4節）
- [x] 推論の高速化 第9弾：**行単位マルチスレッド化**（`Segmenter::tokenize_all`、アトミック動的割当）。M1 Pro 8スレッドで **41.4 M文字/秒（5.6×、Vaporetto単スレッドの約4倍）**、直列とバイト一致（9.4節）
- [x] `segmenter predict` CLIを`tokenize_all`でマルチスレッド化（ブロック単位で全行→並列→順序出力、`--threads N`対応、メモリ有界）。バイト一致維持
- [x] ロード短縮 第10弾：free-list cap を`262144→8192`（DFS前提で最適）。wordDict構築 load 2224→373ms（約6×）、推論・バイト一致維持。block-close未実装で目的達成（9.4節）
- [x] double-array構築の高速化：block-closeは未実装だが、free-list capをDFS前提で`262144→8192`に調整するだけでload 2224→373ms相当を達成（9.4節 第10弾）。tightパッキングが必要になった場合はblock-close実装の余地は残る
- [x] ベンチのマルチスレッド版：`Segmenter::tokenize_all`の並列ベンチを追加（9.2/9.4節：8スレッドで41 M/s、Vaporetto単スレッドの約5倍）
- [x] より大規模・多ジャンルのコーパスでの再計測：**精度・速度とも完了**（4.8節）。UD_Japanese-PUD（別ジャンル out-of-domain、1000文・27,788境界）を追加し、GSD学習済みモデルを再学習せず適用。精度：KyTea/MLP とも劣化なし（PUD の方が高い）、MLP-KyTea 差は 0.6〜0.7pt で安定（GSD単独0.74pt→PUD0.62pt→合算0.67pt）、3倍規模・2ジャンルで初回の0.7ptが代表値と確認。速度：GSD train 2.39x・PUD test 2.21x（実KyTea比）で既存の「2.2〜2.4x」レンジ内、分割速度はジャンル非感受と確認。取得は `scripts/fetch_ud_pud_corpus.sh`。残るのは配布モデル相当の大規模コーパス（BCCWJ）のみで、これは入手不可（4.8節冒頭）
- [~] `Automaton<Payload>`のpayload平坦化第11弾：`FeatVec`ペイロードを連続バッファ化する最適化を実装したが、`Release`でのA/B計測（背中合わせ・best-of-7・6ラウンド）で効果なしと判明し撤回（負の結果、9.4節 第11弾）。真因はビルド構成の見落とし（`Debug`）で、`Release`再構成のみで回復した
- [x] タグ推定込みのモデルロード時間の内訳を実測：**word_dict構築が約88%（〜1150ms）、subword辞書＋2レベルLMは合計約16ms（約1%）**。「subword/LMが主因」という当初の推測は否定された。ロード短縮の対象はword_dictであってsubword/LMではない（遅延ロードは約16msしか縮まず割に合わない）（9.2.1節）
- [x] タグ推定の先頭候補選択を毎語フルソート→線形argmaxに（第12弾、+10〜20%・バイト一致維持、9.4節）
- [~] `add_tag_ngrams`の高速化（Releaseプロファイルで約31%＝タグ推定の最大コスト）：**明示SIMDカーネル化を実装済み**（9.4節 第13弾）。21幅のint16→int32拡張加算（全体の約18%と1幅化A/Bで実測済み・`__restrict`では自動ベクトル化されず）を、`kernels.h`に追加した`add_widen_i16_i32`（scalar/NEON/AVX2、bit一致テスト付き）で置換。バイト一致は整数加算の厳密性により構成上保証（golden＋青空文庫2万行diff＋全115テストで確認済み）。ループ単体はプロセス内A/Bで**1.65倍**（期待エンドツーエンド≈7%）。**エンドツーエンドA/Bのみ未確定**（計測時マシン高負荷のためアイドル時に再計測）。なお検討済みの代替案「文全体1回走査の事前計算方式」は**不採用と結論**：走査対象は語の外側±window（char_n=3で高々6文字）と小さく、律速はマッチごとのペイロード加算（語ごとにマッチ集合が異なるため事前計算では消せない）であり、ボトルネックでない項を攻める案だった。さらにjunction越えマッチ（語の前後の連結をまたぐn-gram）は文位置ごとの事前計算では再現できずバイト一致リスクも高い
- [ ] ベンチマーク実行前は必ず`cmake -B build -DCMAKE_BUILD_TYPE=Release`でビルド構成を確認する運用の徹底（`Debug`構成のまま計測すると数十倍のミスリーディングな数値が出ることを実体験済み、9.4節 第11弾の記述参照）
