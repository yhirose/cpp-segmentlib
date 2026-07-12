#!/usr/bin/env python3
"""Boundary precision/recall/F1 evaluation (design.ja.md 5.8): compares a
backend's word-segmentation output against a KyTea full-annotation gold
corpus. Used to evaluate KyTea and the MLP backend, trained on the same
corpus, against each other (memory mlp-eval-corpus).

Both backends emit their tokenized output through the same escaping
convention (space/slash/ampersand/backslash escaped with backslash;
output.cpp's append_escaped and KyTea's own showString agree, confirmed by
this repo's golden tests), so predicted and gold lines are parsed identically
here.

Usage:
  eval_segmentation.py --gold gold.kytea.txt --command 'kytea -notags -model
      m.mod -out tok' [--command '... another backend ...']
"""
import argparse
import re
import subprocess
import sys
import time

ESCAPE_CHARS = set(" /&\\")


def unescape_words(line: str) -> list[str]:
    """Splits a KyTea-escaped, space-separated word line into surface forms
    (unescaped). Ignores anything after an unescaped '/' (tags), since gold
    and predicted boundary-only lines carry no tags anyway."""
    words = []
    word = []
    i = 0
    in_tag = False
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line):
            if not in_tag:
                word.append(line[i + 1])
            i += 2
            continue
        if c == " ":
            words.append("".join(word))
            word = []
            in_tag = False
            i += 1
            continue
        if c == "/":
            in_tag = True
            i += 1
            continue
        if not in_tag:
            word.append(c)
        i += 1
    words.append("".join(word))
    return words


def boundaries_of(words: list[str]) -> tuple[str, set[int]]:
    """Concatenates words into raw text and returns the byte-offset boundary
    set (cut points strictly between 0 and len, matching segmentlib::Boundaries)."""
    text = "".join(words)
    offsets = set()
    pos = 0
    for w in words[:-1]:
        pos += len(w.encode("utf-8"))
        offsets.add(pos)
    return text, offsets


def prf1(gold: set[int], predicted: set[int]) -> tuple[float, float, float]:
    tp = len(gold & predicted)
    precision = tp / len(predicted) if predicted else 1.0
    recall = tp / len(gold) if gold else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="KyTea full-annotation gold corpus")
    ap.add_argument(
        "--command",
        action="append",
        required=True,
        help="shell command reading raw UTF-8 lines on stdin, writing "
             "space-separated (escaped) tokenized lines on stdout; "
             "repeatable, one evaluation per command",
    )
    args = ap.parse_args()

    gold_texts = []
    gold_boundaries = []
    with open(args.gold, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            words = unescape_words(line)
            text, boundaries = boundaries_of(words)
            gold_texts.append(text)
            gold_boundaries.append(boundaries)

    raw_input = "\n".join(gold_texts) + "\n"

    for command in args.command:
        start = time.perf_counter()
        result = subprocess.run(
            command, shell=True, input=raw_input, capture_output=True, text=True,
        )
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            print(f"[{command}] FAILED (exit {result.returncode}): {result.stderr}",
                  file=sys.stderr)
            continue
        pred_lines = result.stdout.split("\n")
        if pred_lines and pred_lines[-1] == "":
            pred_lines.pop()
        if len(pred_lines) != len(gold_texts):
            print(f"[{command}] line count mismatch: {len(pred_lines)} vs "
                  f"{len(gold_texts)} gold lines", file=sys.stderr)
            continue

        total_tp = total_pred = total_gold = 0
        for gold_set, pred_line in zip(gold_boundaries, pred_lines):
            pred_words = unescape_words(pred_line)
            _, pred_set = boundaries_of(pred_words)
            total_tp += len(gold_set & pred_set)
            total_pred += len(pred_set)
            total_gold += len(gold_set)
        precision = total_tp / total_pred if total_pred else 1.0
        recall = total_tp / total_gold if total_gold else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        chars = sum(len(t) for t in gold_texts)
        print(f"[{command}]")
        print(f"  P={precision:.4f} R={recall:.4f} F1={f1:.4f} "
              f"({total_gold} gold boundaries, {len(gold_texts)} sentences)")
        print(f"  {elapsed:.3f}s wall ({chars / elapsed:.0f} bytes/s, includes "
              f"process startup + model load)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
