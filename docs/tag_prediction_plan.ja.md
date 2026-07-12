# タグ推定バックエンドの実装設計プラン（段階A：既知語のPOS＋読み）

`docs/design.ja.md` 11節の未着手項目「タグ推定（読み/品詞）＋サブワード辞書・LM」のうち、
**段階A（辞書にある語のタグ推定）**の実装プラン。段階B（未知語の読み推定：subwordDict＋KyteaLM＋
ビームサーチ）は本プランのスコープ外とし、接続点のみ定義する（8節）。

本プランは KyTea 本体ソース（`kytea.cpp:calculateTags`、`feature-lookup.cpp`、`corpus-io-full.cpp`）を
直接確認した内容に基づく。分かち書き（WS）バックエンドは既にバイト一致・完成済みで、その
インフラ（`Automaton`、`FeatureLookup` 相当、`Model` ローダ、`CharTable`）を最大限流用する。

## 0. スコープと受け入れ条件

- **対象**：辞書に存在する語のタグ推定。実モデル `jp-0.4.7-5` は **2レベル**（`lev0`=品詞、`lev1`=読み）。
  出力形式は `表層/品詞/読み`（例：`私/代名詞/わたし`）、区切りは語=`" "`・タグ=`"/"`・候補=`"&"`。
- **非対象（段階B）**：辞書にない語（未知語）の**読み**推定。これは `calculateUnknownTag` →
  `generateTagCandidates`（subwordDict のビームサーチ＋文字LM）に依存し、独立した大きな塊。
- **受け入れ条件（重要な切り分け）**：
  - 品詞（`lev0`）は既知語・未知語ともに**グローバルモデル**で推定されるため、段階Aで**全語バイト一致**を狙える。
  - 読み（`lev1`）は既知語では辞書由来で推定できるが、**未知語の読みは段階B必須**。したがって段階Aの
    バイト一致ゲートは「全語が辞書内の文」に限定するか、**レベル別**（POSは常時一致／読みは既知語のみ一致）で評価する。
  - この非対称性が段階Aの設計上の肝（6節・8節）。

## 1. 実モデルの構造（確認済み・要実装時再確認）

- `numTags=2`。`calculateTags` のディスパッチはデータ駆動：
  - `globalMods_[lev] != 0` ならグローバルモデルを使う（`useSelf=true`）。
  - そうでなく辞書エントリ `ent->tagMods[lev]` があればper-wordモデルを使う（`useSelf=false`）。
  - どちらも無ければ未知語処理（段階B）へ。
- ローダは**両方（グローバル／per-word）を保持**し、ディスパッチは推論時にKyTea同様データ駆動で行う。

### 1.1 実測結果（A-1完了時、`jp-0.4.7-5.mod`）
A-1で保持に切り替えた実モデルを走査した実測（`num_tags=2`、`num_dicts=7`）：

| lev | グローバルモデル | per-word モデル | 候補文字列を持つ語 | A-2ディスパッチ |
|---|---|---|---|---|
| **lev0（品詞）** | **あり**：num_weights=21・candidates=21・lookup=yes（charDict 85545 payloads / typeDict / **selfDict 20217 states** / biases=21 / tagDictVector=3087=7×21×21 / tagUnkVector=21） | **0件** | 216,914 | **全語グローバル**（`useSelf=true`：charN+typeN+selfWeights+tagDictWeights）。全語バイト一致を狙える。 |
| **lev1（読み）** | **なし**（candidates=0） | **1,828件**（うち例：num_weights=4・lookup=yes・**selfDict 0 states**・tagUnk=0） | 326,369 | **per-word**（`useSelf=false`：charN+typeNのみ）。候補ありかつ per-word モデルなしの語は決定的 `tags[0]`。未知語の読みは段階B。 |

- 総語彙 850,724 エントリ。lev0 に per-word モデルは存在せず（グローバル一択）、lev1 にグローバルモデルは存在しない（per-word 一択）—— 非対称性（0節）を実データで確認。
- 注意点（A-2）：per-word lev1 では候補文字列数（例9）と分類器 `num_weights`（例4）が一致しないケースがある。KyTea `calculateTags` は `scores.size()` 個だけ `tags[i]` を採る挙動なので、**候補配列長ではなく `num_weights` でループ**すること。
- selfDict／tagDictVector は lev0（グローバル）でのみ非空 → `addSelfWeights`/`addTagDictWeights` は `useSelf=true` 経路でのみ効く（`calculateTags` の分岐と一致）。

## 2. モデルローダの変更（`src/kytea/model.cpp`：skip→retain）

現状のローダは全セクションを**正しく走査（スキップ）済み**。段階Aは「スキップを保持に変える」作業が中心。

### 2.1 再利用可能な `KyteaModel` 保持構造を切り出す
現在 `skip_kytea_model` / `skip_feature_lookup` で読み飛ばしている `KyteaModel`（＝分類器）を、
WS用に持っている構造と共通化して**保持する版** `TagModel` として実装する。保持すべきは：
- `multiplier`（double）、`numWeights`（=候補数）、`labels`（読み飛ばし可）
- `FeatureLookup` の7要素のうちタグに必要なもの：
  `charDict`（`Automaton<FeatVec>`）、`typeDict`、`selfDict`（`Automaton<FeatVec>`：現状スキップ）、
  `biases`（`FeatVec`）、`tagDictVector`（`FeatVec`：現状スキップ）、`tagUnkVector`（`FeatVec`：現状スキップ）。
  （`dictVector` はWS専用なのでタグモデルでは未使用。）

### 2.2 グローバルタグモデルの保持（`model.cpp:200-207`）
現在ループで読み飛ばしている `numTags × (単語リスト + KyteaModel)` を保持：
- `globalTags_[lev]`：各レベルの候補タグ文字列リスト（`KyteaString` の配列 → UTF-8 化して保持）
- `globalMods_[lev]`：2.1 の `TagModel`

### 2.3 単語辞書エントリの保持（`read_word_entry`、`model.cpp:102-116`）
現在スキップしている per-word タグ情報を `WordEntry` に追加保持：
- `tags[lev]`：候補タグ文字列（`KyteaString`→UTF-8）
- `tagInDicts[lev]`：各候補の辞書ビットマスク（`getDictionaryMatches` で使用）
- `tagMods[lev]`：per-word の `TagModel`（多くの語では非存在＝ヌルフラグ）
- `inDict` / `char_length` は既存保持を流用。

### 2.4 WSモデル側 FeatureLookup（`model.cpp:197-198`）
WSモデルの `tagDictVector`/`tagUnkVector` は通常空。ここは引き続きスキップでよい（タグはタグモデル側の
FeatureLookup を使うため）。

## 3. スコア計算：既知語タグ（`kytea.cpp:calculateTags` の else 経路を移植）

各語 `word`・各レベル `lev` について（`startPos`/`finPos` は文中の文字span）：

1. `tags`（候補リスト）と `tagMod` を1節のディスパッチで選ぶ。
2. `tagMod` に FeatureLookup が無い場合：`tag = tags[0]`、マージン=100（非確率）/1（確率）で確定（決定的）。
3. FeatureLookup がある場合、`scores = vector<FeatSum>(numWeights, 0)` を次で積む：
   - `addTagNgrams(charStr, charDict, scores, charN, startPos, finPos)`
   - `addTagNgrams(typeStr, typeDict, scores, typeN, startPos, finPos)`
   - `useSelf`（グローバル）時のみ：
     - `addSelfWeights(charStr[span], scores, 0)`
     - `addSelfWeights(typeStr[span], scores, 1)`
     - `addTagDictWeights(getDictionaryMatches(charStr[span], 0), scores)`
   - `scores[j] += biases[j]`
   - `scores.size()==1` なら2番目を push（非確率0／確率 `-scores[0]`）
4. 各候補 `scores[i] * multiplier` を confidence として `KyteaTag(tags[i], ...)` を作り、**降順ソート**。
5. マージン化（非確率：`score -= secondBest`）または確率化（確率：`exp`→正規化）。
6. 出力に使うのは**先頭（最大）候補**の文字列。

### 3.1 移植する `FeatureLookup` メソッド（`feature-lookup.cpp`、いずれも小さい）
- `addTagNgrams`（~38行）：語spanの**外側**±window文字だけを連結した部分文字列を作り、`charDict`/`typeDict`
  でマッチ。重みベクタ内オフセットは**逆順インデックス** `pos = (window*2 - matchPos - offset - 1) * numWeights`
  で候補ブロックを選ぶ。この式を**一字一句忠実に再現**する（ここがタグ側で最も間違えやすい）。
- `addSelfWeights`（~13行）：`selfDict_->findEntry(word)` で語全体一致の重みブロック `[featIdx*numWeights ..]` を加算。
- `addTagDictWeights`（~17行）：辞書マッチ有無で `tagUnkVector`（無マッチ）or `tagDictVector`（マッチ、
  `base = di*T*T + tagIdx*T`）を加算。`getDictionaryMatches`（~15行、`kytea.cpp:556`）は
  `ent->tagInDicts[lev]` と `numDicts` から `(辞書番号, 候補番号)` ペアを作る。

## 4. 型・API 変更

### 4.1 `CharTable`：id→UTF-8 逆引きの追加
タグ文字列はモデル内で `KyteaString`（id列）として持たれる。出力するには **id→コードポイント**の逆引きが必要。
`CharTable` に `std::vector<char32_t> id_to_cp_`（id順、`charNames_` 相当）を追加し、`KyteaString`→UTF-8 の
`decode_string(KyteaString)` を提供する。ローダのタグ文字列は構築時に一度だけUTF-8化して保持してもよい
（推論ホットパスに逆引きを持ち込まない）。

### 4.2 `types.h` の `Segment`
既に `Segment{ begin, end, tags }` の `tags` スロットは存在（現状 `{}`）。`tags` を
`std::vector<std::string>`（レベル順）として埋める。POS=tags[0]、読み=tags[1]。

### 4.3 `KyteaBackend::tokenize`
WS で境界確定後、`do_tags` かつタグモデルありなら各語に対し3節の計算を実行し `Segment.tags` を埋める。
`tokenize_boundaries` は変更なし。スクラッチ（scores 等）は WS と同様 `thread_local` 再利用で再入可能に。

### 4.4 CLI（`predict_command.cpp` / `output.cpp`）
- 既定でタグを出力（KyTea踏襲）。`--notags` を追加してタグ抑止（現状の挙動＝WSのみ）。
- `append_full_line` を拡張：`表層` + 各レベル `"/" + タグ`。表層は既存の `showEscapedString` 相当で
  エスケープ、**タグ文字列はエスケープしない**（KyTea `showString` に一致）。
- 未知語マーカー（`w.getUnknown()` 時の `unkTag_`）は既定で空。要否は実測で確認。

## 5. 出力整形の詳細（`corpus-io-full.cpp:writeSentence` に一致）

```
語1<wb>語2<wb>...      wb=" "
語 = 表層<tb>タグ0<tb>タグ1      tb="/"
（複数候補出力時のみ タグの第2候補以降を <eb>="&" 連結。既定は先頭候補のみ）
```

段階Aは既定（先頭候補のみ＝KyTea `-out full`）に対応すれば十分＝**実装済み・バイト一致**。

※ 当初この節で `-alltags` と書いていたが、そのようなKyTeaオプションは実在しない（誤記）。複数候補＋信頼度の出力は `-tagmax N`（既定3）＋ `-out conf` 形式で、`表層/候補1&候補2&候補3/...` の後にマージン信頼度を多行で出す。これは**通常ユースケース（分かち書き＋品詞＋読みの最尤1解）では不要**なため実装しない（KyTea/MeCabとも既定は single-best）。実装する場合は段階A-2で省略したマージン計算（`(スコア-2位)×multiplier`）＋全候補保持＋float整形の再現が要る。必要が生じるまで見送り。

## 6. 検証戦略

- **ゴールデン拡張**：`kytea -model jp-0.4.7-5.mod`（タグ付き既定出力）を正解として
  `tests/golden/fixtures/` に `expected_tags.txt` を追加、`predict`（タグ付き）とバイト比較。
- **段階Aの現実的なゲート**：
  - 品詞（lev0）は全語一致を要求できる。
  - 読み（lev1）は**全語が辞書内の文**に限定したフィクスチャで全一致を要求。未知語を含む文は
    段階B完了まで「読みレベルの相違を許容（POSのみ照合）」とする。
- **float一致の心配は段階Aでは軽い**：出力は各レベルの**先頭候補の文字列のみ**。先頭候補は
  `scores`（整数 `FeatSum`）×`multiplier` の**降順ソートの最大**で決まり、`exp`/正規化前の順序で確定する。
  すなわち**先頭候補の決定は整数スコア比較**に帰着し、`double` の LM が絡む段階B のような
  演算順序依存のバイト一致リスクは小さい。注意点は**同点時のソート安定性**（KyTea は `std::sort` +
  strict `>` の `kyteaTagMore`）を合わせることのみ。
- WS の回帰（既存ゴールデン・ベンチのバイト一致）を壊さないこと。

## 7. 想定変更ファイル

- `include/segmentlib/kytea/model.h` / `src/kytea/model.cpp`：`TagModel` 保持構造、グローバル／per-word
  タグモデルと辞書エントリのタグ情報を skip→retain。
- `include/segmentlib/kytea/char_table.h` / `char_table.cpp`：id→UTF-8 逆引き。
- `src/kytea/scorer.cpp`（or 新規 `tag_scorer.cpp`）：`addTagNgrams`/`addSelfWeights`/`addTagDictWeights`/
  `getDictionaryMatches` と既知語タグの積み上げ・ソート・マージン化。
- `src/kytea/kytea_backend.cpp`：`tokenize` にタグ計算を接続。
- `include/segmentlib/types.h`：`Segment.tags` の型確定（`vector<string>`）。
- `src/output.cpp` / `src/cli/predict_command.cpp`：タグ出力整形と `--notags`。
- `tests/golden/`：タグ付きゴールデン。

## 8. 段階B（未知語の読み）への接続点

- `calculateTags` で `tags==0 || tags->size()==0`（未知語）の場合に `calculateUnknownTag(word, lev)` を呼ぶ。
  段階Aでは `subwordModels_[lev]==0` として**早期return**（読み無し）にしておき、POS（グローバル）は通す。
- 段階Bで追加する構造：
  - `subwordDict_`：`Dictionary<ProbTagEntry>`（タグ＋log確率つき Aho-Corasick、新ペイロード型）。
  - `subwordModels_[lev]`：`KyteaLM`（文字n-gram言語モデル、~134行）。`scoreSingle` の対数確率計算を移植。
  - `generateTagCandidates`：subword片のDP格子＋ビーム刈り（`unkBeam`）＋ `exp`/正規化/`tagMax` トリム。
- 段階Bは `double` LM スコアの**演算順序までの一致**が必要で、段階Aとは検証コストの質が異なる（6節の注記参照）。

## 9. 段階見積り（相対）

| 段階 | 主作業 | 規模感 | バイト一致の難度 |
|---|---|---|---|
| **A-1** | ローダ retain（TagModel／グローバル／辞書エントリ） | 中 | — |
| **A-2** | 既知語スコアラ（3節）＋出力整形＋CLI | 中 | 低（整数ソートで先頭候補確定） |
| **B** | subwordDict＋KyteaLM＋ビームサーチ（未知語の読み） | 大 | 高（float演算順序依存） |

段階A（A-1＋A-2）は WS スコアラ＋ローダ改修と同程度の中規模で、既存インフラの延長として完結する。
段階B は独立した塊で、必要になった時点で着手すればよい。
