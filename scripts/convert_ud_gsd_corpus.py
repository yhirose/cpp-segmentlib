#!/usr/bin/env python3
"""Converts UD_Japanese-GSD CoNLL-U files into KyTea full-annotation corpus
format (design.ja.md 6.1), so the freely-licensed UD treebank can stand in
for the (license-restricted, unobtainable) BCCWJ corpus KyTea's distributed
model was trained on (design.ja.md 5.8 / memory mlp-eval-corpus).

Tag level 0 is the UniDic XPOS (detailed POS, e.g. "名詞-普通名詞-一般");
level 1 is the reading, taken from the first comma-separated field of the
MISC column's UnidicInfo=... (lForm, the lexeme's katakana reading) — the
only field that behaves consistently across inflected and uninflected words
in samples checked by hand (水→ミズ, 学校→ガッコウ, に→ニ, が→ガ).

Usage: convert_ud_gsd_corpus.py input.conllu output.txt
"""
import re
import sys

ESCAPE_CHARS = set(" /&\\")


def escape(s: str) -> str:
    return "".join(("\\" + c if c in ESCAPE_CHARS else c) for c in s)


def reading_of(misc: str) -> str:
    for field in misc.split("|"):
        if field.startswith("UnidicInfo="):
            parts = field[len("UnidicInfo="):].split(",")
            return parts[0] if parts else ""
    return ""


def convert(in_path: str, out_path: str) -> None:
    sentences = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        words = []
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                if words:
                    fout.write(" ".join(words) + "\n")
                    sentences += 1
                    words = []
                continue
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 10:
                continue
            token_id = fields[0]
            if not re.fullmatch(r"[0-9]+", token_id):
                continue  # multi-word range or empty node (none expected, skip defensively)
            surface, _, _, xpos, misc = fields[1], fields[2], fields[3], fields[4], fields[9]
            reading = reading_of(misc)
            words.append(f"{escape(surface)}/{escape(xpos)}/{escape(reading)}")
        if words:  # file without a trailing blank line
            fout.write(" ".join(words) + "\n")
            sentences += 1
    print(f"{in_path} -> {out_path}: {sentences} sentences", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} input.conllu output.txt", file=sys.stderr)
        sys.exit(2)
    convert(sys.argv[1], sys.argv[2])
