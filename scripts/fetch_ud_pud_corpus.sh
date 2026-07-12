#!/usr/bin/env bash
# Downloads UD_Japanese-PUD (CC BY-SA 4.0) and converts it to KyTea
# full-annotation format, as an out-of-domain evaluation set complementing
# UD_Japanese-GSD (fetch_ud_gsd_corpus.sh). PUD is a 1000-sentence, single-file
# test corpus (news / Wikipedia, translated parallel text) built on the same
# UniDic short-unit segmentation standard as GSD, so a GSD-trained model can be
# evaluated on it to measure cross-genre generalization (design.ja.md 5.8).
#
# The CoNLL-U layout (surface column, XPOS, MISC UnidicInfo reading) is
# identical to GSD, so convert_ud_gsd_corpus.py is reused verbatim.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/../corpus/ud-pud"
mkdir -p "${OUT_DIR}"

BASE_URL="https://raw.githubusercontent.com/UniversalDependencies/UD_Japanese-PUD/master"

echo "Downloading test.conllu..."
curl -sL "${BASE_URL}/ja_pud-ud-test.conllu" -o "${OUT_DIR}/test.conllu"
echo "Converting test..."
python3 "${SCRIPT_DIR}/convert_ud_gsd_corpus.py" \
    "${OUT_DIR}/test.conllu" "${OUT_DIR}/test.kytea.txt"

echo "Done: ${OUT_DIR}/test.kytea.txt"
