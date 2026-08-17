#!/usr/bin/env bash
# Downloads UD_Japanese-GSD (CC BY-SA 4.0) and converts it to KyTea
# full-annotation format (design.ja.md 6.1) for MLP/KyTea training and
# evaluation (design.ja.md 5.8, memory mlp-eval-corpus): the corpus KyTea's
# distributed model was trained on (BCCWJ) is license-restricted and
# unobtainable, so this freely-licensed treebank — built on the same
# UniDic short-unit segmentation standard — stands in for it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/../corpus/ud-gsd"
mkdir -p "${OUT_DIR}"

# Pinned to a release tag, not master: the bundled reference model
# (models/mlp/) records exactly which corpus it was trained on, and an
# unpinned fetch would make that record meaningless. r2.18 is commit
# 33e7310b58308e85fd2b33a2fc3ef3e434f821c7.
UD_GSD_TAG="r2.18"
BASE_URL="https://raw.githubusercontent.com/UniversalDependencies/UD_Japanese-GSD/refs/tags/${UD_GSD_TAG}"

for split in train dev test; do
    echo "Downloading ${split}.conllu..."
    curl -sL "${BASE_URL}/ja_gsd-ud-${split}.conllu" -o "${OUT_DIR}/${split}.conllu"
    echo "Converting ${split}..."
    python3 "${SCRIPT_DIR}/convert_ud_gsd_corpus.py" \
        "${OUT_DIR}/${split}.conllu" "${OUT_DIR}/${split}.kytea.txt"
done

echo "Done: ${OUT_DIR}/{train,dev,test}.kytea.txt"
