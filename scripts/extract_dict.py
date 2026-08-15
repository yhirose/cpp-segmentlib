#!/usr/bin/env python3
"""Extracts a segmentation dictionary from a KyTea-format training corpus.

The MLP backend (and KyTea/Vaporetto) can take a word dictionary that injects
lexical knowledge as binary L/I/R × length-bucket features at every boundary
(design.md 4.4). A dictionary of the training corpus's own words needs no
external data and measurably helps the MLP (design.md 4.8): on UD-GSD it adds
about +0.45pt (GSD) / +0.26pt (PUD) boundary F1, robust across seeds.

**Use a frequency threshold of at least 2.** Including hapax (frequency-1) words
memorizes the training segmentation and collapses precision at test time
(dict of all 20,177 words: GSD F1 97.91→96.55). freq>=2 (8,726 words) is the
sweet spot; freq>=3 and dropping single-character words are within noise of it.

Output is one surface word per line, with KyTea's escaping (space / \\ & each
prefixed by backslash) so the same file works for `segmenter train --dict`,
`train-kytea -dict`, and Vaporetto `train --dict`.

Usage:
  scripts/extract_dict.py corpus/ud-gsd/train.kytea.txt -o corpus/ud-gsd/dict.txt
  scripts/extract_dict.py train.kytea.txt --min-count 2 --min-length 1
"""
import argparse
import sys
from collections import Counter

ESCAPE_CHARS = set(" /&\\")


def unescape_words(line: str) -> list[str]:
    """Splits a KyTea word/tag line into unescaped surface forms."""
    words, word, i, in_tag = [], [], 0, False
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line):
            if not in_tag:
                word.append(line[i + 1])
            i += 2
            continue
        if c == "/":
            in_tag = True
        elif c == " ":
            in_tag = False
            if word:
                words.append("".join(word))
                word = []
        elif not in_tag:
            word.append(c)
        i += 1
    if word:
        words.append("".join(word))
    return words


def escape_word(w: str) -> str:
    return "".join("\\" + c if c in ESCAPE_CHARS else c for c in w)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", help="KyTea full-annotation corpus")
    ap.add_argument("-o", "--out", required=True, help="output dictionary path")
    ap.add_argument("--min-count", type=int, default=2,
                    help="keep words occurring at least this many times (default 2)")
    ap.add_argument("--min-length", type=int, default=1,
                    help="keep words at least this many characters (default 1)")
    args = ap.parse_args()

    freq: Counter = Counter()
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            for w in unescape_words(line.rstrip("\n")):
                if w:
                    freq[w] += 1

    kept = sorted(w for w, c in freq.items()
                  if c >= args.min_count and len(w) >= args.min_length)
    with open(args.out, "w", encoding="utf-8") as o:
        for w in kept:
            o.write(escape_word(w) + "\n")
    print(f"{len(freq)} unique words → {len(kept)} kept "
          f"(min-count={args.min_count}, min-length={args.min_length}) → {args.out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
