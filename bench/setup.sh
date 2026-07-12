#!/usr/bin/env bash
# One-time benchmark setup:
#   1. fetch + clean an Aozora Bunko corpus (real Japanese prose),
#   2. build Vaporetto (predict + convert_kytea_model),
#   3. convert the KyTea model to Vaporetto's format.
#
# Requirements: curl, python3, cargo, and the KyTea model already fetched
# (scripts/fetch_kytea_model.sh).
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

echo "Setup complete. Run: bench/run.sh"
