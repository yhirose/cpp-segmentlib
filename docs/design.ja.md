# 設計ドキュメント

## 1. 概要

KyTea / Vaporetto と同じ「点推定（pointwise prediction）」方式を採用した、C++の**分かち書き専用**ライブラリ。文字境界ごとに独立した二値分類（区切る/区切らない）を分類器で行う。ラティス＋Viterbiによる最小コスト法（MeCab等）とは異なり、辞書コストの設計や動的計画法を必要としない。

公開APIは単一（6節参照）で、内部に2つの**バックエンド**を差し替え可能な形で持つ：

- **KyTea互換バックエンド**（3節）— KyTeaの学習済みモデルをそのまま読み込み、KyTeaと同一の特徴抽出・線形SVM分類器で推論する。学習はサポートしない（推論のみ）。
- **独自MLPバックエンド**（4節）— 線形SVMではなく、独自に設計したMLP（多層パーセプトロン）を分類器として使うバックエンド。学習エンジンも自前実装している。

**コーパスフォーマットは常にKyTeaのコーパス形式を使う**（5節）。KyTea互換バックエンド・独自MLPバックエンドのいずれも、同じKyTeaコーパス形式（フル/部分アノテーション）を入力として想定する。

## 2. アーキテクチャ：バックエンドの抽象化

複数バックエンドを同一APIの背後に隠すため、`Segmenter`はバックエンドの実装詳細を知らない薄いディスパッチャである。

バックエンドの集合は「KyTea互換／独自MLP」という決まった小さな閉じた集合であり、外部プラグインとして動的に追加する要件はない。そのため仮想関数によるオープンな拡張機構（`virtual`＋ヒープ確保）ではなく、**`std::variant` + `std::visit`によるクローズドな多態性**を採る。vtableの間接呼び出しやバックエンドオブジェクトの個別ヒープ確保を避けられ、対応していないバックエンドの分岐漏れはコンパイル時に検出できる。

```cpp
class KyteaBackend {
public:
    std::expected<Segments, Error> tokenize(std::string_view text) const;
};
class MlpBackend {
public:
    std::expected<Segments, Error> tokenize(std::string_view text) const;
};

using AnyBackend = std::variant<kytea::KyteaBackend, mlp::MlpBackend>;

class Segmenter {
public:
    static std::expected<Segmenter, Error> load(const std::filesystem::path& model_path);
    static std::expected<Segmenter, Error> load_kytea(const std::filesystem::path& model_path);
    static std::expected<Segmenter, Error> load_mlp(const std::filesystem::path& model_path);

    std::expected<Segments, Error> tokenize(std::string_view text) const {
        return std::visit([&](const auto& b) { return b.tokenize(text); }, backend_);
    }

    std::vector<std::expected<Segments, Error>>
    tokenize_all(std::span<const std::string_view> texts, unsigned threads = 0) const;

private:
    AnyBackend backend_;
};
```

**モデル形式の自動判別**：`Segmenter::load()`はファイル先頭のシグネチャを見て自動的にバックエンドを選ぶ（KyTeaモデルは`"KyTea "`で始まるヘッダ行を持つ、それ以外は独自MLP形式として扱う、4.7節）。明示的にバックエンドを指定したい場合向けに`load_kytea(path)` / `load_mlp(path)`も用意し、`load()`はその薄いラッパーとする。

各バックエンドクラスは`tokenize`という同一シグネチャさえ満たせばよく、内部の特徴抽出・分類器・モデルパーサは完全に独立して実装できる。2つのバックエンドの共通点は「同じ`Segments`型を返すこと」だけである。

## 3. KyTea互換バックエンド

- KyTeaが出力するモデル（`train-kytea`で学習されたモデル）をそのまま読み込む。
- 素性はKyTeaと同じ3種：文字n-gram、文字種n-gram、辞書由来の単語素性。
- **推論のみをサポートする**（学習エンジンは実装しない）。
- モデルのバイナリフォーマットは、KyTeaのモデルファイルを直接パースする（変換ツールを挟まない）。
- **分かち書き専用**（タグ推定は行わない）。

### 3.1 特徴量抽出（KyTea互換・忠実再現）

分類器（LIBLINEARの線形SVM）自体は汎用的だが、**素性文字列の生成ロジックと文字種分類はKyTea独自の実装**であり、モデルとの互換性を保つにはここを一字一句忠実に再現する必要がある。

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

KyTeaの`findType`は4バイトUTF-8（コードポイント≧U+10000、CJK拡張B以降）のコードポイント計算にバグがあり、それらの漢字を誤分類する。本ライブラリは正しいコードポイントから分類するため、この稀なケースでKyTeaと結果が分かれる（意図的な差異）。BMP内の通常の日本語テキストでは一致する。

**入力の正規化（`normalize`）**

推論時、KyTeaは入力文字列を`surface`（原文）と`norm`（正規化）に分け、素性計算（文字n-gram・文字種n-gram）はすべて`norm`に対して行う。正規化は固定テーブル（約110エントリ）で、**半角英数字・記号を全角へ畳む**（`a→ａ`, `0→０`, `(→（`, 半角カナ記号`｢｣→「」`等）ものである。出力の単語表層は`surface`（原文のバイト列）から切り出すが、境界判定に使うスコアは`norm`から計算される。本ライブラリはこの固定テーブルを移植し、**UTF-8デコード→コードポイント正規化→インターン**の順で`norm`相当のID列を作る（`CharTable::encode`）。

**素性文字列のフォーマット**

境界位置を基準に、窓幅（デフォルト`charw=3`、`typew`同様に設定可能）の範囲で以下3種の素性文字列を生成し、モデルの辞書でIDに変換して線形分類器に渡す。

| 素性種別 | プレフィックス形式 | 例 |
|---|---|---|
| 文字n-gram | `"X" + 相対位置` + 文字列そのもの | `X-2`, `X-1`, `X0`, `X1` |
| 文字種n-gram | `"T" + 相対位置` + 文字種記号の列 | `T-1`, `T0`, `T1` |
| 辞書由来の単語素性 | `"D" + 辞書インデックス + (L\|I\|R) + マッチ長` | `D0L1`（辞書0番、左端一致、長さ1）, `D1R3` |

`D`素性の`L`/`I`/`R`は、境界に対する辞書エントリの位置関係（Left端／Inside中間／Right端）を表す。

**素性IDのマッピングはハッシュではなくモデル内蔵の辞書**

KyTeaのモデルは学習時の素性文字列→ID辞書をモデルファイルに埋め込んでいる。推論側は独自にハッシュ関数を実装せず、**モデルファイルから素性辞書をそのまま読み込み、その辞書を引いてIDを求める**。未知の素性文字列（学習時に出現しなかったもの）はモデルに存在しないため、その素性は単純にスキップする。

### 3.2 モデルファイルフォーマット

**ヘッダ行**

```
KyTea <version> <T|B> <encoding>
```

例：`KyTea 0.4.0 B utf8`。`version`は量子化ビルドでは`"0.4.0"`。フォーマット文字は`T`=テキスト、`B`=バイナリ。本ライブラリは`0.4.0`系のバイナリ形式のみを対象とする。このヘッダ行の`"KyTea "`シグネチャは、2節で述べたバックエンド自動判別にも使う。

**ファイル全体のセクション順序**

1. **Config**：`do_ws`, `do_tags`, `numTags`, `charWindow`, `charN`, `typeWindow`, `typeN`, `dictionaryN`, バイアス有無, `epsilon`, `solverType`、および文字マップ
2. **分かち書きモデル**（`wsModel_`）：`KyteaModel` 1個
3. **タグモデル**：`numTags`個ぶん（本ライブラリはスキップのみ）
4. **単語辞書**：`Dictionary<ModelTagEntry>`（本ライブラリは`char_length`/`in_dict`のみ保持）
5. **サブワード辞書**：`Dictionary<ProbTagEntry>`（本ライブラリはスキップのみ）
6. **言語モデル（LM）**：`numTags`個ぶん（本ライブラリはスキップのみ）

本ライブラリは分かち書き専用。上記6セクションのうちWSに必要なのは1（`numTags`/`do_tags`はセクション3・単語辞書エントリ内のタグ情報の**バイト長を知るためだけに**必要）・2・4の一部（`char_length`/`in_dict`のみ）で、3・5・6と単語辞書エントリ内のタグ候補・per-wordタグモデルは**ファイル上のバイト列としては存在するが、値として保持せず読み飛ばす**（`model.cpp`の`skip_*`系関数）。ファイル形式自体はKyTea側が定めた不変の仕様であり、本ライブラリの実装はどこを保持しどこを読み飛ばすかを選ぶだけである。

**`KyteaModel`（分類器1個）のシリアライズ**

- クラス数（`int32_t`。0または2未満なら「モデルなし」を意味しそこで終了）
- ソルバー種別（`char`1バイトの列挙値。デフォルトは`L2R_L2LOSS_SVC_DUAL`＝L2正則化L2-lossの線形SVM（dual））
- 各クラスのラベル（`int32_t`×クラス数）
- バイアス有無（`bool`）
- `multiplier`（`double`。量子化された重みを実数に戻すためのスケール係数）
- `FeatureLookup`（後述）

**`FeatureLookup`＝推論用に事前コンパイルされた素性→重みの直接マッピング**

KyTeaは学習時の素性文字列→ID辞書をそのままモデルファイルに書き出すのではなく、**推論用に最適化された`FeatureLookup`構造を別途構築して書き出す**。中身は次の7要素：

| フィールド | 型 | 内容 |
|---|---|---|
| `charDict` | `Dictionary<FeatVec>` | 文字n-gram → クラス別重みベクトル |
| `typeDict` | `Dictionary<FeatVec>` | 文字種n-gram → クラス別重みベクトル |
| `selfDict` | `Dictionary<FeatVec>` | 辞書由来の自己文字列素性 → クラス別重みベクトル |
| `dictVector` | `FeatVec` | 辞書関連の追加固定長重み |
| `biases` | `FeatVec` | バイアス項 |
| `tagDictVector` / `tagUnkVector` | `FeatVec` | タグ推定関連の重み（本ライブラリは未使用） |

`Dictionary<FeatVec>`は**Aho-Corasickオートマトン**（`DictionaryState`の配列：`failure`リンク、`gotos`（文字→次状態、ソート済みで二分探索）、`output`（この状態で確定する素性のインデックス列））である。

**`Dictionary<Entry>`のバイナリレイアウト**（`charDict`/`typeDict`/`selfDict`（`Entry=FeatVec`）だけでなく、単語辞書（`Entry=ModelTagEntry`）・サブワード辞書（`Entry=ProbTagEntry`）にも共通の枠組み）：

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

`gotos`は文字（`KyteaChar`）でソートされており、読み込み側は`DictionaryState::step()`で二分探索する。

`writeEntry<Entry>`はEntry型ごとに異なる：

- `Entry = FeatVec`（`charDict`/`typeDict`/`selfDict`用）：`uint32_t`の要素数 → `FeatVal`（既定`int16_t`）× 要素数。**先頭に素性文字列などの識別子は一切含まれない**（対応する文字列は`DictionaryState`の遷移パス自体が表現している）。
- `Entry = ModelTagEntry`（単語辞書用）：`word`（`KyteaString`）→ タグレベルごとの情報（本ライブラリはスキップ）→ `inDict`（`unsigned char`、辞書所属ビットマスク）→ タグレベルごとの`KyteaModel`（本ライブラリはスキップ）。
- `Entry = ProbTagEntry`（サブワード辞書用）：本ライブラリはスキップ。

`KyteaString`（＝可変長文字列）のバイナリ表現は**NUL終端のバイト列**として保存される。長さプレフィックスは持たず、読み込み側は`\0`まで読み進める実装になっている。

**`KyteaChar`の型は`unsigned short`（2バイト・符号なし）**であり、KyTea内部では文字はUTF-8のまま保持せず、いったんこの2バイト整数の内部表現にマッピングしてから扱っている。Aho-Corasickオートマトンの遷移は`KyteaChar`単位で構築されているため、**UTF-8の生バイト列ではなく、モデルに埋め込まれた文字→2バイト整数のマッピングを使ってオートマトンを辿る**必要がある。

**`KyteaChar`は固定のUnicodeコードポイントではなく、モデルごとに学習時の文字出現順で採番される「文字のインターン表」のID**である。

- ID `0`は空文字列用の予約済みセンチネル。実際の文字は**ID `1`から**始まる。
- Config内の「文字マップ」フィールドの中身は、**学習時にモデルが見た全ての異なり文字を、ID順（＝初出順）に並べて連結しただけの1本のUTF-8文字列**である。
- モデルローダは、この文字列を読み込んだらUTF-8として先頭から1文字ずつデコードし、出現順に`1`始まりのIDを割り振って`char(UTF-8) → KyteaChar`の対応表を構築する。入力テキストを推論する際も、この対応表を引いてUTF-8文字を`KyteaChar`列に変換してからAho-Corasickオートマトンを辿る。

**未知文字（学習語彙にない文字）の扱い**：KyTea本体は推論時、未知文字に`charTypes_.size()`（＝学習時最大ID＋1以降）の新規IDを動的に採番する。本ライブラリは未知文字を一律`kNoChar`（`0`）に落とす。両者はID割り当てが異なるが、**分かち書きの出力は完全に等価**である（`calculateWS`のWS素性は3種すべてがAho-Corasickマッチで、いずれも学習時ID `1..K`でキー付けされたオートマトンを辿るため、未知文字のIDが`0`でも`K+1`でもどの遷移にも存在しない）。文字種n-gramへの寄与は、文字種を**コードポイントから直接分類**しており文字IDに依存しないため影響を受けない。ただし4バイトUTF-8のCJK拡張B漢字（例：𠮷 U+20BB7）は別軸——これは未知文字処理ではなく`findType`バグ（正しくKanji分類する本ライブラリと、誤分類するKyTeaで型n-gramが分岐する）による意図的な差異。

**推論は「文字列→ID→重み配列引き」ではなく「Aho-Corasickでマッチした瞬間に重みベクトルを直接得る」設計**。モデルローダは、素性文字列を独自に生成してハッシュ／ID変換するのではなく、**この`FeatureLookup`（3つのAho-Corasickオートマトン＋4つの重みベクトル）をそのままファイルからパースし、同じ構造で持つ**。

**重みの型（`FeatVal`）はデフォルトで`int16_t`量子化**。配布されている学習済みモデルは通常量子化ビルド（`FeatVal = int16_t`）で作られている。読み込み時は`int16_t`の重みを`multiplier`（`double`）倍して実数の重みに戻す。本ライブラリは量子化モデル（`int16_t`）のみをサポートする（非量子化モデルはヘッダのバージョン文字列`"0.4.0NQ"`で検出しエラーとする）。

### 3.3 推論アルゴリズム（単語分割、`Kytea::calculateWS`の再現）

`KyteaBackend::tokenize`が実行する計算。

文の文字数を`N`とすると、境界は`N-1`個ある（各文字の直後、最後の文字を除く）。各境界`i`（`0 <= i < N-1`）についてスコア`score[i]`を次の順で積み上げる：

1. **初期値**：`score[i] = biases[0]`（`FeatureLookup::biases_`の先頭要素。全境界で共通の定数）
2. **文字n-gramスコアの加算**：正規化済み文字列（`sent.norm`）に対して`charDict`（Aho-Corasick）でマッチした全ての文字n-gramについて、`addNgramScores`のロジックで加算する。1つのn-gramマッチは、その出現位置を中心に**窓の中に入る複数の境界に同時に寄与**する（`FeatVec`は窓幅`window*2`分のスコアを1本のベクトルとして持ち、マッチ位置`pos`から`base_pos = pos - window`を起点に`score[base_pos + j] += vec[j]`（`j`は有効範囲にクリップ）という形で分配される）。
3. **文字種n-gramスコアの加算**：文字列を文字種記号の列（`R`/`H`/`T`/`D`/`K`/`O`）に変換したものに対して、`typeDict`で同様に加算する。
4. **辞書由来（D素性）スコアの加算**：`dict_->match(sent.norm)`で単語辞書とのAho-Corasickマッチを取り、`addDictionaryScores`で加算する。インデックス計算（`len=score.size()`、`max=config.getDictionaryN()`、マッチした語の文字長`wlen`、`lablen=min(wlen,max)-1`）：
   - マッチ語の**左端**の境界（`end-wlen`番目、ただし`end>=wlen`の場合のみ）に `dictVector[辞書番号*dictLen + (end-wlen)*3*max + lablen*3 + 0]` を加算
   - マッチ語の**内部**の各境界（`end-wlen+1 <= k < end`）に `... + k*3*max + lablen*3 + 1` を加算
   - マッチ語の**右端**の境界（`end`番目、ただし`end != len`の場合のみ）に `... + end*3*max + lablen*3 + 2` を加算
5. **ハードな制約（`-wsconst`相当）の上書き**：`config.getWsConstraint()`に指定された文字種記号が、隣接する2文字の文字種が同一のケースに含まれる場合、その境界のスコアを強制的に「境界なし」側に上書きする（配布jpモデルの`wsConstraint`は通常空で既定出力には無影響。本ライブラリは未実装）。
6. **最終スコア**：`wsConfs[i] = score[i] * wsModel_->getMultiplier()`
7. **境界判定**：`wsConfs[i] > 0`なら境界あり、そうでなければ境界なし。

つまり実装している推論ロジックは、**「biasを初期値に、charDict・typeDict・dictVectorの3種類のAho-Corasickマッチスコアを加算し、multiplierをかけて0と比較する」**という単純な線形和である。SVMの学習部分（LIBLINEAR）は実装していない。

## 4. 独自MLPバックエンド

3節の線形SVMに代えて、独自に設計したMLP（多層パーセプトロン）を分類器として使うバックエンド。分割方式そのものは3節と同じ**ポイントワイズ二値分類**（各境界候補について「区切る/区切らない」を独立に判定）で、公開APIも同じ`tokenize`を満たす。KyTea/Vaporettoが手で列挙する素性（文字n-gram・文字種n-gram・辞書素性）の**交互作用を、窓内の埋め込み表現＋隠れ層に自動獲得させる**。

学習データはKyTeaコーパス形式（5節）をそのまま使う。学習エンジン（順伝播・逆伝播・最適化）は本ライブラリで実装している。

### 4.1 設計思想と、KyTea/Vaporettoとの対比

| | KyTea（3節）／Vaporetto（外部比較のみ、本ライブラリのバックエンドではない） | 本MLPバックエンド（4節） |
|---|---|---|
| 分類方式 | ポイントワイズ二値分類 | 同左 |
| 分類器 | 線形SVM（重みの線形和） | MLP（非線形・多層） |
| 素性 | 文字n-gram・文字種n-gram・辞書素性を手で設計・列挙 | 窓内の埋め込みを連結し、交互作用をネットワークが学習 |
| 文字種特徴 | ヒューリスティックな6種分類（3.1節）を明示的に使用 | 使わない |
| 原子単位 | コードポイント単位の「文字」 | EGC（書記素クラスタ）単位（4.2節） |
| 語彙・OOV | モデル内蔵の素性辞書、未知素性はスキップ | 埋め込みはコードポイント語彙、EGCは合成で表現し原理的にOOVなし（4.3節） |

**文字種特徴を明示的に持たない**のは意図的な設計判断である。未知・低頻度への汎化が問題になった場合に限り、General Unicode Property（General Category等）を補助入力として導入する余地を残す。

### 4.2 原子単位・境界候補・窓の数え方＝EGC

分類の原子単位を、コードポイントではなく**EGC（Extended Grapheme Cluster、UAX #29）**とする。

- **境界候補はEGCの隙間のみ**。EGCの内部（例：`か`+濁点`が` = U+304B U+3099、絵文字ZWJ連結 `👨‍👩‍👦` = 6コードポイント）では決して区切らない。
- **窓は「EGCの個数」で数える**。コードポイント個数で窓を測ると、`👨‍👩‍👦` 1個だけで左右window=5の窓を食い潰してしまう。EGC単位なら`👨‍👩‍👦`は1トークンとして扱え、残りの枠を前後の実際の語に使える。

文のEGC列を`e[0..M-1]`とすると、境界候補は`M-1`個（各EGCの直後、末尾を除く）。各境界`i`（`0 <= i < M-1`）について独立に二値分類する。

### 4.3 EGCの表現＝構成コードポイントからの合成的埋め込み

各EGCをそのまま語彙IDに引く（フラットなEGC-id埋め込み）方式は採らない。代わりに、**EGCを構成コードポイント列に分解し、コードポイント埋め込みをpoolingして1本のEGCベクトルを合成する**。

```
EGC → 構成コードポイント列に分解
  各コードポイント → 埋め込み（語彙は「学習時に出現したコードポイント」。小さく有界）
  → pooling → EGCベクトル（次元 d）
```

**語彙構築には頻度閾値を設ける**（出現2回未満のコードポイントは語彙に入れずUNKに落とす）。低頻度コードポイントを訓練中UNKとして流すことで、UNK埋め込みが「稀な文字の平均的な振る舞い」を学習する。

日本語・中国語は1 EGC≒1コードポイントがほとんどなので、poolingは大半が恒等に縮退し、実質「コードポイント埋め込み」として振る舞う。結合列・タイ語・ミャンマー語・絵文字でのみpoolingが実効的に働く。

### 4.4 ネットワーク構成

「窓内の各EGCベクトル＋辞書素性を連結し、隠れ1層のMLPに通して二値判定する」ポイントワイズ分類器。構成は速度要件（4.6節の第1層事前計算）から逆算して決定した。

```
境界 i について:
  窓 = e[i-w+1 … i+w] の EGC 列（左右各 w、計 2w 個。端は PAD トークン）
  各 EGC → 4.3節の合成的埋め込み（次元 d、pooling は mean）
  f_dict = 辞書マッチ二値素性（下記）
  h = ReLU( W1 · concat(2w × d) + W_dict · f_dict + b1 )    # 隠れ層 1 層
  y = w2 · h + b2                                            # スカラー
  y > 0 なら境界
```

**確定値**：

| 項目 | 値 | 根拠 |
|---|---|---|
| 窓幅 `w` | 5（左右各5、計10 EGC） | 第1層事前計算方式ではw拡大のコストが「表引き加算1回/EGC」と線形でほぼタダ |
| 埋め込み次元 `d` | 64 | コードポイント語彙~1万×64で軽量 |
| pooling | mean（変更不可） | meanは線形なので`W1_j·mean(e_c) = mean(W1_j·e_c)`が成り立ち、第1層事前計算（4.6節）と両立する |
| 隠れ層 | 1層、幅H=256、ReLU | ポイントワイズ分類では深さの効果が薄く、推論コストは隠れ層以降が支配的になるため浅く保つ |
| 出力 | 1ユニット、推論時はyの符号判定（sigmoid省略） | `p>0.5 ⇔ y>0` |

**辞書マッチ二値素性`f_dict`**：辞書（単語リスト）とのマッチをAho-Corasick（EGC列上で走らせる）で取り、境界iを跨ぐ/接するマッチについて、位置関係3種（L/I/R）×マッチ長バケット（`min(EGC長, 4)`の4段階）の計12個の二値素性を立てる（複数辞書対応時は辞書ごとに12個）。同一（位置関係×長さバケット）に複数の辞書語がマッチしても素性は1のまま（二値clamp）。辞書なしでも動作する（`f_dict`全ゼロ）。

### 4.5 学習

- **損失**：境界ごとの二値クロスエントロピー（BCE）。`sigmoid(y)`は損失計算でのみ使用。
- **教師信号のマスク**：部分アノテーション（5.2節）の「不明」位置は損失計算から除外する。intra-EGC位置はそもそも境界候補でないため損失にも現れない。
- **アノテーションとEGCの衝突処理**：EGC内部に境界ありが来た文は、警告を出してその文ごとスキップする。
- **入力の正規化**：KyTeaと同じ半角→全角固定テーブル正規化をかけてからEGC分割・語彙化する（`CharTable`の正規化テーブルを共用）。
- **学習後の量子化**：重み・埋め込みは学習後にint16へ量子化する（PTQ）。

### 4.6 推論・C++実装方針：第1層の事前計算（NNUE方式）

第1層は純粋な線形変換なので、窓位置ごとに分解できる：

```
W1 · concat(v_1, …, v_2w) = Σ_j  W1_j · v_j        （W1_j は窓位置 j に対応する 256×d のスライス）
```

したがって**(EGC, 窓位置j) → W1_j·v(EGC) ∈ R^256を事前に表引き化**する。mean poolingの線形性（4.4節）により、合成的EGC埋め込みもこの表に折り込める。辞書二値素性も同様に`W_dict`の列ベクトル表引きになる。推論時の1境界の計算は：

```
acc = b1
acc += table[egc_j, j]   を 2w 回（表引き＋256次元ベクトル加算）
acc += dict_col[k]       をアクティブ辞書素性ぶん（0〜数回）
h = ReLU(acc)
y = dot(w2, h) + b2      （256次元内積 1 回）
境界 ⇔ y > 0
```

**テーブル・アキュムレータの数値表現**：`Model::load`/`load_from_bytes`は`TablePrecision::{Int32,Int16}`を選べる（既定は`Int16`）。

- **Int16**：`kAccShift=9`（2^22→2^13、4倍ヘッドルーム）、`requant_i16`＝丸め付き右シフト+飽和が全int16量（table/dict_col/b1、b2は同シフトのint64）の変換。アキュムレータは飽和add（`vqaddq_s16`）。
- **Int32**：検証用参照経路として残置。学習側`int16_decision`とのbit-exact契約はInt32が担う。

**事前計算テーブル**：頻出EGC（≒頻出コードポイント）について構築。稀なEGCは「コードポイント埋め込み→mean→W1_jを掛ける」合成経路にフォールバックする。

**SIMDカーネル**：`include/segmentlib/mlp/kernels.h`（ヘッダオンリー）にadd/relu/dot×int32/int16の6カーネル＋`add_widen_i16_i32`。`kernels::scalar::*`が常時コンパイルされるoracle、ディスパッチはコンパイル時（AArch64→NEON、x86は`__AVX2__`定義時のみAVX2、他はscalar）。NEON・AVX2ともbit一致テストで実機検証済み（NEON=ローカルARM実機、AVX2=CI ubuntu-24.04実機＋Windows MSVC実機）。

**thread_localスクラッチ**：`mlp_backend.cpp`内の`Scratch{EncodedEgc,Workspace,scores}`、per-call割当ゼロ。

トークン化（UTF-8 → EGC分割）はUAX #29に従う。順伝播（推論）のみをライブラリ本体が実装し、学習は別コンポーネント（`SEGMENTLIB_BUILD_TRAINING`オプトイン）で行い、モデルファイル（4.7節）で受け渡す。

### 4.7 モデルファイル形式（独自設計）

シリアライズ形式は独自設計。**`BinaryReader`（`bytes/binary_reader.h`）のプリミティブ（リトルエンディアン固定幅整数・NUL終端文字列・`\n`終端ヘッダ行）でそのまま読める**ことを設計制約とし、KyTeaバックエンドと同じ読み取り基盤を共用する。事前計算テーブル（4.6節）とAho-Corasickオートマトンはファイルに含めず、**ロード時に構築する**。

**ヘッダ行（ASCII、`\n`終端）**

```
SegmentLibMLP <version>\n
```

例：`SegmentLibMLP 1\n`。この`"SegmentLibMLP "`シグネチャを2節のバックエンド自動判別に使う（KyTeaの`"KyTea "`シグネチャと排他）。バージョン不一致はローダがエラーにする。

**ヘッダ行に続くバイナリ本体**（すべてリトルエンディアン）：

| # | フィールド | 型 | 内容 |
|---|---|---|---|
| **Config** | | | |
| 1 | `char_window` `w` | `uint8` | 片側窓幅（EGC個数）。4.4節で`w=5` |
| 2 | `embed_dim` `d` | `uint16` | コードポイント埋め込み次元。4.4節で`64` |
| 3 | `hidden` `H` | `uint16` | 隠れ層幅。4.4節で`256` |
| 4 | `num_dicts` | `uint8` | 辞書数（`0`可。`0`ならW_dict・辞書セクションは書かれない） |
| 4b | `unicode_version` | `uint16` | 学習時のEGC分割に使ったUnicodeバージョン（メジャー×100+マイナー）。不一致ならローダが警告する |
| **Scales**（量子化スケール） | | | |
| 5 | `emb_scale` | `double` | 埋め込みint16→実数の係数 |
| 6 | `w1_scale` | `double` | W1のint16→実数の係数 |
| 7 | `wdict_scale` | `double` | W_dictの係数（`num_dicts>0`のときのみ） |
| 8 | `w2_scale` | `double` | w2の係数 |
| 8b | `acc_scale` | `double` | 加算器（第1層活性）の整数スケール。学習後に検証データで活性分布をキャリブレーションして選ぶ |
| **Vocabulary**（コードポイント語彙） | | | |
| 9 | `vocab_size` `V` | `uint32` | 埋め込み行数。行0=PAD、行1=UNK（未知コードポイント）を含む |
| 10 | `codepoints` | `uint32 × (V-2)` | 行2..V-1に対応するコードポイントを**昇順**で格納。推論時は入力コードポイントをこの配列で二分探索し行番号を得る |
| **Embedding** | | | |
| 11 | `embedding` | `int16 × (V·d)` | 埋め込みテーブル。行優先（行0=PAD、行1=UNK、行2以降=`codepoints`順） |
| **Layer 1** | | | |
| 12 | `W1` | `int16 × (H · 2w · d)` | 第1層重み。行優先で`W1[h][j*d + c]` |
| 13 | `W_dict` | `int16 × (H · num_dicts · 12)` | 辞書二値素性の重み。`num_dicts>0`のときのみ |
| 14 | `b1` | `double × H` | 第1層バイアス（非量子化） |
| **Layer 2** | | | |
| 15 | `w2` | `int16 × H` | 出力層重み |
| 16 | `b2` | `double` | 出力層バイアス（非量子化） |
| **Dictionaries**（`num_dicts>0`のときのみ） | | | |
| 17a | `entry_count` | `uint32` | 語数 |
| 17b | `entries` | NUL終端UTF-8 × `entry_count` | 単語表層（正規化をかけてからUAX #29でEGC列に分割しAho-Corasickを構築） |

**ロード時の処理**：(1) 語彙・埋め込み・重みを読み、(2) 位置別事前計算テーブルと辞書素性の列ベクトルの展開を構築、(3) 単語リストからEGC単位Aho-Corasickを構築、(4) `b1`/`b2`をアキュムレータ整数スケールへ量子化。

### 4.8 評価結果（現在の実測値）

比較は入手可能なコーパス（UD_Japanese-GSD、CC BY-SA 4.0）でKyTeaとVaporettoを再学習し、**同一データ・辞書なし**で学習したMLPバックエンドと突き合わせる形で行っている（配布モデルの学習コーパスは入手不可のため）。3者とも `corpus/ud-gsd/train.kytea.txt` で学習、辞書素性なし。KyTeaは実バイナリ（`kytea -notags`）と自前バックエンドがバイト一致することを確認済み。

**精度**（`scripts/eval_segmentation.py`、境界F値。3者とも同一eval・同一goldで計測）：

| テストセット（ジャンル） | 境界数 | KyTea F1 | Vaporetto F1 | MLP F1 | MLP差(vs KyTea) |
|---|---|---|---|---|---|
| GSD test（Wikipedia、in-domain） | 12,491 | 98.87% | 98.79% | 97.91% | −0.96pt |
| PUD（news/Wikipedia対訳、out-of-domain） | 27,788 | 99.24% | 99.17% | 98.56% | −0.68pt |
| GSD+PUD合算（2ジャンル） | 40,279 | 99.13% | 99.05% | 98.35% | −0.78pt |

辞書なし・既定構成（w=5, d=64, H=256, seed=42）での結果。量子化による判定反転はUD-GSD実モデルでdev+train 290,024境界中0件。seed起因のF1変動は±0.05pt程度（5 seedで実測）。**辞書なしでは本MLPは線形モデル（KyTea/Vaporetto）に約0.7〜1.0pt負ける**。KyTeaとVaporettoはほぼ互角。

**辞書なしで線形モデルに負ける原因の分析**（境界単位の突き合わせ。差はMcNemar検定で有意：MLPのみ誤り327/KyTeaのみ誤り85（GSD, z=11.9）、542/158（PUD, z=14.5））：

- **負けの主因は「見たことはあるが局所的に曖昧なバイグラム」＝語彙的記憶の容量不足**。境界を挟む2文字が学習データで「結合でも分割でも」出現する局所曖昧バイグラムの割合は、全境界17.1%（GSD）/15.5%（PUD）に対し、MLPのみ誤りでは**27.5%/28.4%**に濃縮。こうした境界は局所では決まらず、正確な語彙パターン（「ですね」「とんでもない」等のn-gram全体）の記憶で解くしかない。KyTea/Vaporettoは疎な明示n-gram素性（GSD学習で約89万素性）が実質的な記憶テーブルとして働くのに対し、本MLPのパラメータ（埋め込み15万＋W1 16万程度）は語彙例外の記憶容量がはるかに小さい。
- **逆に、学習未出現バイグラムではMLPが相対的に強い**（合成的埋め込みの汎化が機能している証拠）。境界バイグラムが学習未出現の割合は、KyTeaのみ誤りで**42.4%/42.4%**と、MLPのみ誤りの30.3%/30.1%より高い——未出現n-gramでは素性が発火しない線形モデルの方が脆い。つまり「表現の汎化」はMLPの狙いどおり働いており、負けているのは記憶側。
- **稀な文字への偏りも実在する**（副次要因）。MLPのみ誤りでは、境界を挟む稀少側文字の学習頻度中央値が131/150と全境界（327/362）の半分以下、学習頻度2未満（≒UNK）の割合は2.8%/7.4%と全境界（0.5%/1.0%）の5〜7倍。64次元埋め込みは低頻度文字で安定しない。out-of-domain（PUD）ではカタカナ連続境界の誤りが急増し、OOV外来語・固有名詞の連結失敗が上乗せされる。
- **対策の含意**：記憶不足が主因なので、(1) **辞書素性**（学習コーパスから抽出した語リストでもよい）が最も直接的——語彙知識を全境界にショートカット注入する、(2) 埋め込み次元・隠れ幅の増強は記憶容量を足すが速度とトレードオフ、(3) 文字種の全スロット入力（4.1節の拡張余地）は稀少文字・OOV側に効きうる。なお**OOVコードポイントのUNK行をGeneral Categoryで分割する案は実験済みで効果なし**（マルチseed A/BでΔ=0.00pt）——稀少「文字」の問題は文字種の粒度では救えず、負けの本体は語彙記憶側にある。

辞書素性（学習コーパス自己抽出、`scripts/extract_dict.py`）も検証したが、精度改善（GSD +0.45pt）に対して速度低下が大きすぎる（5.11→2.67 M chars/sec、約48%減）ため不採用。

**速度**（M1 Pro、`bench/bench_segment`、int16+NEON、best-of-8）：

| ジャンル | MLP | 実KyTea | 比 |
|---|---|---|---|
| GSD train（Wikipedia、277K字） | 5.11 M chars/sec | 1.40 M chars/sec | 3.65x |
| PUD test（news対訳、48K字） | 5.60 M chars/sec | 1.58 M chars/sec | 3.54x |

分割速度はジャンルに非感受（スコア計算はEGC窓の整数演算のみで語彙・文体に依存しない）。

**Int16スコアリングの融合カーネル**（プロファイル駆動の追加最適化）：per-boundaryのスコア計算は、プロファイル（`sample`、M1 Pro）で自己時間の71.5%が「2w個のテーブルブロックをアキュムレータへ飽和加算するループ」に集中しており、その大半がアキュムレータのL1往復（load-block/load-acc/store-accがadd 1回につき3メモリ操作）という冗長なコストだと判明した（作業集合は文あたり数百KBでL2常駐、DRAM帯域律速ではない）。`kernels::fused_score_i16`は、この「b1初期化 → 2w回の窓加算 → 辞書列加算 → ReLU → 出力dot」を、アキュムレータのHチャンクをSIMDレジスタに載せたまま1パスで通す形に融合し、境界ごとに発生していた約2w+3回のフルH幅アキュムレータ往復を1回のレジスタ常駐スイープに置き換える。境界間でのインクリメンタル更新（NNUEのチェス実装が使う定石）は、同じ文字でも窓内の位置jが変わると別のW1列を参照するため構造的に成立しない——2w回の窓寄与の計算自体は変わらず、それを運ぶメモリトラフィックだけを削っている。int16飽和加算はブロックを加算する順序に結果が依存するため、融合カーネルは旧実装と同じ順序（窓位置→辞書列）で飽和加算し、既存のInt16↔Int32判定反転テストでバイト完全一致を確認している（精度・判定は不変）。単スレッド速度は約1.4〜1.5倍向上（GSD train: 3.5M→5.11M chars/sec）。

### 4.9 学習側の設計（C++自前実装）

学習エンジンは本ライブラリで自前実装する（外部フレームワーク非依存）。学習はfp32、推論はint16（4.6節）で、両者はモデルファイル（4.7節）で受け渡す。**推論（ライブラリ本体の成果物）は順伝播のみ**で、学習コンポーネントはビルド上分離する（`SEGMENTLIB_BUILD_TRAINING`オプトイン。推論バイナリにBLAS/CUDAを要求しない）。

**学習パイプライン**

```
1. コーパス読込（5節, KyTeaフル/部分アノテーション）
2. 正規化（KyTea互換の半角→全角, CharTable共用, 4.5節）
3. EGC分割（UAX #29）
3b. 語彙構築: 出現コードポイントを頻度集計し、閾値未満は UNK に落とす
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

- **第1層・第2層**：標準的なdense層の勾配。行列積が主計算で、`ComputeBackend`のGEMMに委譲する。
- **mean pooling → 埋め込み**：EGCベクトルが構成コードポイントのmeanなので、EGCベクトルへの勾配は各構成コードポイント行へ`1/(コードポイント数)`で分配される。**埋め込みテーブルの勾配は疎**（バッチに現れた行のみ）。Adamの1次・2次モーメントも現れた行だけ更新する。
- 辞書二値素性`f_dict`への`W_dict`勾配も、立っている素性の列のみの疎更新。

**最適化器**：Adam。

**量子化**：PTQ（Post-Training Quantization）で確定。QATは採用しない。

**計算バックエンド抽象（`ComputeBackend`）**

行列積・活性化・要素演算・勾配だけをこの層に閉じ込め、プラットフォームごとに実装を差し替える。CPU実装はBLASインターフェース（`cblas_sgemm`等）に対して書き、リンクするBLASを切り替えるだけで全OSに載る。

| プラットフォーム | 第一候補 | CPU実装（BLAS） | GPU実装（任意） |
|---|---|---|---|
| macOS (Apple Silicon) | CPU/AMX | Accelerate | Metal/MPSGraph（通常不要） |
| Linux + NVIDIA | GPU | OpenBLAS/MKL | cuBLAS |
| Linux (GPUなし) | CPU | OpenBLAS/BLIS | — |
| Windows + NVIDIA | GPU | OpenBLAS/MKL | cuBLAS |
| Windows (GPUなし) | CPU | OpenBLAS | — |

サポート方針：**macOS・Linuxを一級サポート、Windowsはbest-effort**。BLASは学習ターゲット（`segmentlib_train`）のみリンクする。

**推論側の可搬性（学習とは別軸）**：推論はBLASではなく手書きSIMDのint16 NNUE方式（4.6節）で、**NEON（Apple/ARM）＋ AVX2（x86 = Linux/Windows共通）＋ スカラーfallback**の3実装で全OSを覆う。x86のAVX2パスはLinux/Windowsで同一。

## 5. コーパス仕様

デフォルトの区切り文字：

| 用途 | 文字 | デフォルト |
|---|---|---|
| 単語境界（フル）/ 不明境界（部分） | `wordBound_` / `unkBound_` | 半角スペース `" "` |
| タグ境界 | `tagBound_` | `/` |
| タグ候補区切り | `elemBound_` | `&` |
| エスケープ | `escape_` | `\` |
| 非境界（部分） | `noBound_` | `-` |
| 境界（部分） | `hasBound_` | `\|` |
| スキップ（部分） | `skipBound_` | `?` |

両バックエンド（3〜4節）の学習データは、このKyTeaコーパス形式に統一する。

### 5.1 フルアノテーション形式

```
word1/tag0a&tag0b/tag1a word2/tag0 word3 ...
```

- 単語は半角スペースで区切る。
- `/`が出現するたびにタグの「レベル」が1つ進む。
- 同一レベル内で複数候補のタグを持たせたい場合は`&`で連結する（学習時は先頭候補が正解ラベルとして使われる）。
- 単語・タグ中に区切り文字自体（スペース、`/`、`&`、`\`）を含めたい場合は`\`でエスケープする。
- 実例：
  ```
  コーパス/ko:pasu の/no 文/buN で/de す/su 。/.
  ```
- 本ライブラリの学習パイプライン（MLPバックエンド）はタグ情報を使わず境界情報のみを使う。

### 5.2 部分アノテーション形式

```
ヴ-ェ-ネ-ツ-ィ-ア|は|イ-タ-リ-ア|に|あ り ま す|。
```

- 文字を1文字ずつ並べ、隣接文字の間に以下いずれかの記号を置いて境界情報を表す：
  - `-`（noBound）：境界ではない（同じ単語内）。教師信号として確定的に使う。
  - `|`（hasBound）：境界である。教師信号として確定的に使う。単語の終端も兼ねる。
  - ` `（unkBound）/ `?`（skipBound）：不明。読み込み時はいずれも「教師信号なし」として扱われる。
- タグは各語の末尾、`|`の直前に`/tag0&tag1/tag2...`の形式でフルアノテーションと同じ文法で付与できる（本ライブラリの学習は境界情報のみ使用）。
- エスケープ文字・タグ境界・タグ候補区切りはフルアノテーションと共通（`\`, `/`, `&`）。

## 6. C++ API

### 6.1 基本方針

- 可変オブジェクトを使い回す副作用ベースのAPIではなく、**入力を受け取り結果を値として返す関数型のAPI**を基本とする。
- 入力は所有権を必要としないため`std::string_view`を受け取る。
- エラーは`std::expected`（C++23）で表現する。
- オフセットは**UTF-8バイトオフセット**を採用する。
- バックエンド（KyTea互換／独自MLP）の違いはこのAPI層には一切露出しない。呼び出し側は`Segmenter::load()`で読み込んだモデルファイルの種類を意識せず、同じ`tokenize()`を呼ぶだけでよい。

### 6.2 型

本ライブラリは分かち書き専用。`Segments`は語スパン（開始・終了のUTF-8バイトオフセット）のペア列という最小の形をとる——専用の`Segment`構造体は導入せず、生の型のエイリアスとする。

```cpp
using Segments = std::vector<std::pair<std::size_t, std::size_t>>;  // (start, end) のペア列

enum class ErrorCode {
    InvalidUtf8,
    ModelNotLoaded,
    UnsupportedModelFormat,
    MalformedModel,
    MalformedCorpus,
    IoError,
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
    static std::expected<Segmenter, Error> load(const std::filesystem::path& model_path);
    static std::expected<Segmenter, Error> load_kytea(const std::filesystem::path& model_path);
    static std::expected<Segmenter, Error> load_mlp(const std::filesystem::path& model_path);

    std::expected<Segments, Error> tokenize(std::string_view text) const;

    std::vector<std::expected<Segments, Error>> tokenize_all(
        std::span<const std::string_view> texts, unsigned threads = 0) const;
};
```

- 分かち書きのみを行う`tokenize()`一本。多数の入力を並列に分かち書きする`tokenize_all()`（`threads==0`でハードウェア並列数、9節：M1 Pro 8スレッドで単スレッド比5.6×）。
- 空文字列の入力は「エラーではなく空の結果」として扱う（`Segments{}`）。`Error`はUTF-8不正やモデル未読込・未対応形式などの実際の異常系のみに使う。

## 7. CLIインターフェース

コマンド名は`segmenter`。`git`/`cargo`のような**サブコマンド構成**。

```
segmenter predict --model model.bin < input.txt > output.txt
segmenter train --backend mlp --corpus corpus.txt --model-out model.bin
```

### 7.1 `segmenter predict`（推論）

KyTea / Vaporettoと同様、**標準入力からテキストを読み、標準出力に分かち書き結果を書き出すフィルタ型**。

**オプション**

| オプション | 説明 |
|---|---|
| `--model <path>` | モデルファイルのパス（必須）。KyTea互換／独自MLPのいずれかを自動判別する（2節） |
| `--threads <n>` | 並列実行のスレッド数（`0`＝`hardware_concurrency()`、既定） |

**出力フォーマット**

KyTeaの分かち書き出力形式（スペース区切りの単語列）を踏襲する。

```
コーパス の 文 で す 。
```

**表層語のエスケープ（`showEscapedString`）**：区切り文字（スペース・`/`・`&`・エスケープ文字`\`自身）が単語表層に含まれる場合、`\`で前置してエスケープする（例：入力`Hello World` → `Hello \  World`、`2024/12/31` → `2024 \/ 12 \/ 31`）。実装は`append_full_line`（`src/output.cpp`）に集約し、CLIとgoldenテストで共有。

**入力の扱い**：入力は常にUTF-8固定。不正なUTF-8バイト列は`CharTable::encode`/`Vocab::encode`が`ErrorCode::InvalidUtf8`を返し、CLIは該当行以前の出力をflushした上で即座に中断する（exit code 1・stderrにメッセージ）。

### 7.2 `segmenter train`（学習）

```
segmenter train --backend mlp \
  --corpus full1.txt --corpus full2.txt \
  --partial-corpus part1.txt \
  --dict dict.txt \
  --model-out model.bin \
  [--char-window 5] [--embed-dim 64] [--hidden 256] [--min-count 2] \
  [--epochs 30] [--batch-size 256] [--patience 5] [--lr 1e-3] [--seed 42]
```

`--backend mlp`のみ実装されている（`SEGMENTLIB_BUILD_TRAINING=ON`ビルドが必要。OFFビルドでは`train`はスタブ）。`--backend kytea`／`--backend vaporetto`は明示的な「not implemented」エラーを返す（`train_command.cpp`）。

| オプション | 説明 |
|---|---|
| `--backend mlp` | 学習するバックエンド（必須） |
| `--corpus <path>` | フルアノテーションコーパス（5.1節）。複数回指定可能 |
| `--partial-corpus <path>` | 部分アノテーションコーパス（5.2節）。複数回指定可能 |
| `--dict <path>` | 辞書ファイル。複数回指定可能 |
| `--dev-corpus <path>` | 検証用コーパス（早期終了・量子化キャリブレーションに使用） |
| `--model-out <path>` | 学習済みモデルの出力先（必須） |
| `--char-window <int>` | EGC窓幅（既定5） |
| `--embed-dim <int>` | 埋め込み次元（既定64） |
| `--hidden <int>` | 隠れ層幅（既定256） |
| `--min-count <int>` | コードポイント語彙の頻度閾値（既定2） |
| `--epochs <int>` | エポック数（既定30） |
| `--batch-size <int>` | バッチサイズ（既定256） |
| `--patience <int>` | 早期終了の忍耐値（既定5） |
| `--lr <float>` | 学習率（既定1e-3） |
| `--seed <int>` | 乱数シード（既定42） |

KyTea互換モデルの学習が必要な場合は、本物の`train-kytea`（Homebrew配布）を外部ツールとして直接呼ぶ（本ライブラリの評価パイプライン、4.8節がその実例）。

## 8. 実装モジュール構成

### 8.1 レイヤ構成（下位→上位、上位は下位にのみ依存）

```
include/segmentlib/
├── bytes/
│   └── binary_reader.h        # (L1) プリミティブなバイナリ読み取りカーソル
├── unicode/
│   ├── utf8.h                 # (L1) UTF-8デコード/エンコードの純粋関数群
│   ├── egc.h                  # (L1) UAX #29 EGC分割
│   └── normalize.h            # (L1) KyTea互換の半角→全角正規化
├── text/
│   └── aho_corasick.h         # (L2) Aho-Corasick構築＋マッチ（ヘッダオンリー）
├── kytea/
│   ├── char_table.h           # (L2) 文字種分類 + 文字インターン表
│   ├── automaton.h            # (L2) Aho-Corasickランタイム表現（double-array）
│   ├── model.h                # (L3) Config/KyteaModel/FeatureLookupのデータ型 + load()
│   ├── scorer.h                # (L4) calculateWSのスコア計算アルゴリズム
│   └── kytea_backend.h        # (L5) Backend I/F実装（tokenize）
├── mlp/
│   ├── vocab.h                 # (L2) コードポイント語彙 + EGCエンコード
│   ├── dictionary.h            # (L2) 辞書二値素性のAho-Corasickマッチャ
│   ├── kernels.h               # (L2) SIMDカーネル（add/relu/dot、NEON/AVX2/scalar）
│   ├── precompute.h            # (L3) 位置別事前計算テーブル
│   ├── model.h                  # (L3) モデルのデータ型 + load()
│   ├── scorer.h                 # (L4) 推論スコア計算
│   └── mlp_backend.h           # (L5) Backend I/F実装（tokenize）
├── segmenter.h                 # (L6) 公開API（6節）
├── output.h                    # 出力整形（append_full_line）
└── types.h                     # Segments/Error（6.2節、依存なし）

src/
├── kytea/{char_table,automaton,model,scorer,kytea_backend}.cpp
├── mlp/{vocab,dictionary,precompute,model,scorer,mlp_backend}.cpp
├── mlp/train/                  # 学習コンポーネント（SEGMENTLIB_BUILD_TRAINING時のみビルド）
│   ├── corpus.{h,cpp}           # KyTeaコーパスパーサ
│   ├── example.{h,cpp}          # 学習例生成
│   ├── dataset.{h,cpp}          # ミニバッチ化
│   ├── net.{h,cpp}              # 順伝播・逆伝播
│   ├── adam.{h,cpp}             # Adamオプティマイザ
│   ├── trainer.{h,cpp}          # 学習ループ
│   ├── quantize.{h,cpp}         # PTQ int16量子化
│   ├── exporter.{h,cpp}         # モデルファイル書き出し
│   ├── compute_backend.h        # BLAS抽象
│   └── cpu_blas.cpp             # CPU（Accelerate/OpenBLAS）実装
├── segmenter.cpp
├── output.cpp
└── cli/
    ├── main.cpp                  # `predict`/`train` のサブコマンド振り分け
    ├── predict_command.cpp       # predictサブコマンド本体
    └── train_command.cpp         # trainサブコマンド本体（--backend mlpのみ実装）
```

### 8.2 各モジュールの責務

**L1: `bytes::BinaryReader`** — `std::span<const std::byte>`を読み進める薄いカーソル。`read<T>()`、`read_cstring()`を提供する。パースエラーは内部では例外（軽量な`ParseError`）で投げっぱなしにし、`model.h`の`load()`という境界だけで`std::expected`に変換する。

**L1: `unicode::utf8`** — UTF-8の1コードポイントをデコードする純粋関数。

**L2: `kytea::CharTable`** — 「文字インターン表」（ID`0`=センチネル、実文字は`1`始まりの初出順）を保持する値型。`decode`/`encode`と文字種分類（`classify`）を提供する。

**L2: `kytea::Automaton<Payload>`** — `Dictionary<Entry>`のランタイム表現。`std::vector<State>`と`std::vector<Payload>`をフラットに持つ値型（canonical double-array）。モデルファイルのデシリアライズ結果を受け取るだけで、Aho-Corasickを構築するロジックは持たない（ファイルには構築済みオートマトンがそのまま入っているため）。

**L3: `kytea::Model`** — `Config`/`KyteaModel`/`FeatureLookup`に対応するイミュータブルな値型。`static auto load(std::filesystem::path) -> std::expected<Model, Error>`が唯一の構築経路。保持するのは`charDict`/`typeDict`/`dictVector`/`biases`/`multiplier`/`word_dict`（`char_length`/`in_dict`のみ）——分かち書きに必要な最小限。単語辞書のper-wordタグ情報・サブワード辞書・言語モデル・タグモデルは、ファイル上に存在するので読み飛ばして正しく後続位置までシークするが、値としては保持しない。

**L4: `kytea::scorer`** — `calculateWS`のアルゴリズムをそのまま関数として実装する。`Model`・`CharTable`・入力テキストを受け取り、境界ごとのスコア列を返す純粋関数。

**L5: `kytea::KyteaBackend`** — 2節で定義した`tokenize`シグネチャを満たすクラス。中身は`Model`を保持し、`scorer`の関数を呼ぶだけの薄いアダプタ。

**L6: `Segmenter`** — 2節の設計そのまま。`std::variant<KyteaBackend, MlpBackend>`。

**CLI層** — `predict_command.cpp`は「引数パース→`Segmenter::load`→stdinを1行ずつ読んで`tokenize`→出力」というだけの薄いループ。ビジネスロジックを一切持たない。

### 8.3 ディレクトリ構成

```
cpp-segmentlib/
├── CMakeLists.txt                # トップレベル。C++23、サブディレクトリを束ねる
├── include/segmentlib/           # 公開ヘッダ（8.1節の構成）
├── src/                          # 実装（8.1節の構成）
│   ├── CMakeLists.txt            # segmentlib（ライブラリ）のターゲット定義
│   └── cli/
│       └── CMakeLists.txt        # segmenter（実行ファイル）のターゲット定義
├── tests/
│   ├── CMakeLists.txt
│   ├── unit/                     # モジュール単位のテスト
│   └── golden/                   # KyTea実バイナリの実出力と突き合わせるテスト
│       ├── golden_test.cpp
│       └── fixtures/
│           ├── input.txt
│           └── expected.txt      # `kytea -notags`の出力
├── models/                       # (gitignore) KyTea/Vaporettoモデル
├── corpus/                       # (gitignore) 評価用コーパス
├── bench/                        # 推論ベンチ（9節）
│   ├── setup.sh / run.sh / README.md
│   ├── bench_segment.cpp         # 自ライブラリ in-process
│   ├── bench_kytea.cpp           # libkytea in-process
│   └── {.vendor,corpus,results}/ # (gitignore)
├── scripts/
│   ├── fetch_kytea_model.sh
│   ├── fetch_ud_gsd_corpus.sh
│   ├── fetch_ud_pud_corpus.sh
│   ├── convert_ud_gsd_corpus.py
│   ├── eval_segmentation.py
│   └── strip_kytea_tags.py
├── docs/
│   ├── design.ja.md / design.md
│   ├── mlp_module_design.ja.md / mlp_module_design.md
│   └── mlp_impl_design.ja.md / mlp_impl_design.md
├── .github/workflows/ci.yml
├── .gitignore
├── .clang-format
└── .clang-tidy
```

**設計判断**

- **ビルドシステムはCMake**。
- **依存ライブラリは推論経路ではゼロ**（標準ライブラリのみ）。学習経路（`SEGMENTLIB_BUILD_TRAINING`）のみBLASをリンクする。
- **テストフレームワークは`doctest`**（ヘッダオンリー、CMakeの`FetchContent`で取得）。
- **`models/`・`corpus/`はgit管理しない**：`.gitignore`に追加し、`scripts/fetch_*.sh`のようなダウンロードスクリプトのみをリポジトリに置く。
- **`golden/`テストは固定データ方式**：既知の入力文とKyTea実行結果のペアを`tests/golden/fixtures/`に固定データとしてコミットする。モデル未取得時はテストがスキップされる（CI耐性）。

### 8.4 ビルド／ツールチェーン要件

- **C++23対応コンパイラが必須**：`std::expected`・`std::byteswap`・`std::span`・`std::print`等を使う。
- **macOS**：AppleClang（Xcode標準）でビルド可能。
- **Linux**：GCC 14以降。
- **Windows**：MSVC、best-effort。
- **ビルド手順**：
  ```
  cmake -S . -B build -G Ninja
  cmake --build build
  ctest --test-dir build --output-on-failure
  ```
  ベンチマーク計測時は必ず`-DCMAKE_BUILD_TYPE=Release`を指定する（`Debug`構成では推論速度が数十倍遅くなる）。
- **CI**：`.github/workflows/ci.yml`。macOS arm64（NEON）・Linux x86_64（AVX2/scalar、GCC14+OpenBLAS）・Windows（MSVC、AVX2、best-effort）の4ジョブ。goldenテストはモデル未取得時に自動スキップ。

## 9. ベンチマーク（推論速度）

`bench/`。KyTea / Vaporettoとの推論速度比較（KyTeaバックエンドについて）。

### 9.1 設計原則

- **同一モデル**：3ツールとも同じ重みで動かす。
- **正確性ゲートを先に通す**：タイミング前に出力を`diff`。segmentlibはKyTeaとバイト一致、Vaporettoの相違率を報告する。
- **純粋推論はin-processで測る**：モデルを1回ロードし、tokenizeループのみを計測（ロード・I/Oを除外）。専用ハーネス（`bench_segment`＝自ライブラリ、`bench_kytea`＝libkyteaリンク）を使う。
- **条件固定**：シングルスレッド・ウォームアップ後best-of-N。指標はUnicodeコードポイント/秒。
- **コーパス**：青空文庫の実在作品（漱石・太宰・芥川・宮沢、約71万文字）。

### 9.2 最新結果（Apple M1 Pro、青空文庫71万字、best-of-5）

**正確性ゲート**
- segmentlib vs KyTea：**0 / 20822 行相違（100%一致）**
- Vaporetto vs KyTea：92 / 20822 行相違（99.56%一致）

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

**知見**：segmentlibはKyTeaとバイト完全一致しながらシングルスレッド推論5倍以上、ロードも速い。Vaporettoはシングルスレッドでは依然最速（出力は厳密には非一致、0.44%）だが、segmentlibはマルチスレッド化でVaporettoのシングルスレッドを上回る（8スレッドで約5.2倍）。

## 10. 既知の制限・未実装機能

- **タグ推定（品詞・読み）**：行わない。分かち書き専用。
- **KyTea互換の学習エンジン**：実装しない。`segmenter train --backend kytea`は明示的エラーを返す。KyTea互換モデルが必要な場合は本物の`train-kytea`を使う。
- **Vaporetto互換バックエンド**：作らない。Vaporettoは外部ベンチマーク比較対象としてのみ使用する。
- **`--encode`**：実装しない。入力は常にUTF-8固定。
- **`--backend`（predict時の明示指定）・`--scores`（境界スコア出力）**：未実装（自動判別で用が足りている）。
- **複数候補＋信頼度出力**（KyTeaの`-out conf`／`-tagmax`相当）：実装しない。
- **部分アノテーション入力のハード制約**（`-wsconst`相当）：未実装。配布jpモデルの`wsConstraint`は通常空で既定出力には影響しない。
- **pmr版API**（`std::pmr::memory_resource`注入）：未実装。ベンチマークで実際にボトルネックになった場合に検討する。
- **MLPバックエンドのAVX2**：CIで実機検証済み（GitHub Actions ubuntu-24.04実機＋Windows MSVC実機）。
