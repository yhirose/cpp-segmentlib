#!/usr/bin/env python3
"""Turns a UniDic lex.csv into the word list `segmenter train --dict` consumes.

UniDic's surface forms are the first CSV column. They are emitted one per line,
deduplicated, sorted, and KyTea-escaped (space / \\ & each prefixed with a
backslash) so the same file also feeds `train-kytea -dict` and Vaporetto.

**Drop single-character entries** (the `--min-length 2` default). They are about
1% of UniDic but match constantly, and the character window already sees those
characters directly: on UD-GSD their removal costs nothing measurable (GSD
99.144 -> 99.120, PUD 99.294 -> 99.300, 5 seeds) and buys back ~12% of the
inference speed the dictionary costs. `--max-length 4` is the other useful
filter: the length feature saturates at 4 clusters, so longer entries mostly add
model size (halves it, for -0.06pt GSD / -0.11pt PUD).

Usage:
  scripts/convert_unidic_dict.py lex.csv -o corpus/ud-gsd/dict_unidic.txt
"""
import argparse
import csv
import sys

ESCAPE_CHARS = set(" /&\\")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lex", nargs="+", help="UniDic lex.csv (repeatable)")
    ap.add_argument("-o", "--out", required=True, help="output dictionary path")
    ap.add_argument("--min-length", type=int, default=2,
                    help="keep entries of at least this many characters (default 2)")
    ap.add_argument("--max-length", type=int, default=10**9,
                    help="keep entries of at most this many characters")
    args = ap.parse_args()

    words, seen, dropped = [], set(), 0
    for path in args.lex:
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                w = row[0].strip()
                if not w or "�" in w:
                    dropped += 1
                    continue
                if not (args.min_length <= len(w) <= args.max_length):
                    continue
                if w not in seen:
                    seen.add(w)
                    words.append(w)

    words.sort()
    with open(args.out, "w", encoding="utf-8") as o:
        for w in words:
            o.write("".join("\\" + c if c in ESCAPE_CHARS else c for c in w) + "\n")
    print(f"{len(words)} entries written (dropped {dropped} empty/invalid, "
          f"length {args.min_length}..{args.max_length}) -> {args.out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
