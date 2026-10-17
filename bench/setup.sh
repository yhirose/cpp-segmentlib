#!/usr/bin/env bash
# One-time benchmark setup:
#   1. fetch + clean an Aozora Bunko corpus (real Japanese prose),
#   2. build Vaporetto (predict + convert_kytea_model),
#   3. convert the KyTea model to Vaporetto's format,
#   4. build bench_kytea, KyTea's in-process harness, against libkytea.
#
# Requirements: curl, python3, cargo, a C++ compiler, an installed KyTea (the
# `kytea` command and libkytea, e.g. `brew install kytea`; set KYTEA_PREFIX
# if it is not under Homebrew or /usr/local), and the KyTea model already
# fetched (scripts/fetch_kytea_model.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

KMODEL="models/kytea/jp-0.4.7-5.mod"
VMODEL="models/kytea/jp-0.4.7-5.vaporetto.zst"
VENDOR="bench/.vendor"
CORPUS_DIR="bench/corpus"

if [[ ! -f "$KMODEL" ]]; then
    echo "KyTea model not found; run scripts/fetch_kytea_model.sh first." >&2
    exit 1
fi

# --- 1. corpus ----------------------------------------------------------
mkdir -p "$CORPUS_DIR"
if [[ ! -f "$CORPUS_DIR/aozora.txt" ]]; then
    echo "Fetching Aozora Bunko works..."
    : > "$CORPUS_DIR/aozora.txt"
    tmp="$(mktemp)"
    while read -r path _title; do
        [[ -z "$path" || "$path" == \#* ]] && continue
        curl -sL "https://www.aozora.gr.jp/cards/$path" -o "$tmp"
        python3 bench/clean_aozora.py "$tmp" >> "$CORPUS_DIR/aozora.txt"
    done < bench/aozora_works.txt
    rm -f "$tmp"
    echo "corpus: $(wc -l < "$CORPUS_DIR/aozora.txt") lines"
fi

# --- 2. Vaporetto -------------------------------------------------------
mkdir -p "$VENDOR"
if [[ ! -x "$VENDOR/vaporetto-predict" ]]; then
    echo "Building Vaporetto..."
    if [[ ! -d "$VENDOR/vaporetto" ]]; then
        git clone --depth 1 https://github.com/daac-tools/vaporetto.git "$VENDOR/vaporetto"
    fi
    ( cd "$VENDOR/vaporetto" && cargo build --release -p predict -p convert_kytea_model )
    cp "$VENDOR/vaporetto/target/release/predict" "$VENDOR/vaporetto-predict"
    cp "$VENDOR/vaporetto/target/release/convert_kytea_model" "$VENDOR/convert_kytea_model"
fi

# --- 3. convert model ---------------------------------------------------
if [[ ! -f "$VMODEL" ]]; then
    echo "Converting KyTea model to Vaporetto format..."
    "$VENDOR/convert_kytea_model" --model-in "$KMODEL" --model-out "$VMODEL"
fi

# --- 4. bench_kytea -----------------------------------------------------
if [[ ! -x "$VENDOR/bench_kytea" ]]; then
    # --installed: plain `brew --prefix kytea` prints Homebrew's path even
    # when KyTea is not installed there, which would hide /usr/local.
    prefix="${KYTEA_PREFIX:-$(brew --prefix --installed kytea 2>/dev/null || echo /usr/local)}"
    if [[ ! -f "$prefix/include/kytea/kytea.h" ]]; then
        echo "KyTea headers not found under $prefix; install KyTea or set KYTEA_PREFIX." >&2
        exit 1
    fi
    echo "Building bench_kytea against $prefix..."
    # -w: libkytea's own headers are not warning-clean. The rpath lets run.sh
    # start it without a library search path.
    "${CXX:-c++}" -std=c++17 -O2 -w bench/bench_kytea.cpp \
        -I"$prefix/include" -L"$prefix/lib" -Wl,-rpath,"$prefix/lib" -lkytea \
        -o "$VENDOR/bench_kytea"
fi

echo "Setup complete. Run: bench/run.sh"
