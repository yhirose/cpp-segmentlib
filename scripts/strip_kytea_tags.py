#!/usr/bin/env python3
"""Strips a KyTea full-annotation corpus down to raw surface text (one
sentence per line, tags and word boundaries removed), for feeding to
bench/bench_segment or bench/.vendor/bench_kytea, which expect plain text.

Usage: strip_kytea_tags.py input.kytea.txt output.raw.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_segmentation import unescape_words  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} input.kytea.txt output.raw.txt", file=sys.stderr)
        return 2
    with open(sys.argv[1], encoding="utf-8") as fin, \
            open(sys.argv[2], "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            fout.write("".join(unescape_words(line)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
