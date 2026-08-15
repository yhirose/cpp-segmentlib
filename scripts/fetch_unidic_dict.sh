#!/usr/bin/env bash
# Downloads UniDic 2.1.2 (unidic-mecab) and converts its lexicon into the word
# list the MLP backend trains against (design.md 4.4/4.8).
#
# UniDic is the lexicon UD_Japanese's short-unit segmentation standard is built
# on, which is why it beats a dictionary extracted from the training corpus by
# so much here (GSD 98.07 -> 99.12 boundary F1, 5 seeds). Single-character
# entries are dropped by default; see convert_unidic_dict.py for why.
#
# License: unidic-mecab is triple-licensed GPL / LGPL / BSD by the UniDic
# Consortium (the archive's COPYING, GPL, LGPL and BSD files). Note that
# `segmenter train --dict` embeds the word list in the model file, so a model
# trained this way is a derivative of UniDic and carries its terms.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/../corpus/ud-gsd"
WORK_DIR="${OUT_DIR}/.unidic"
OUT="${OUT_DIR}/dict_unidic.txt"
URL="https://clrd.ninjal.ac.jp/unidic_archive/cwj/2.1.2/unidic-mecab-2.1.2_src.zip"

mkdir -p "${WORK_DIR}"

if [[ ! -s "${WORK_DIR}/unidic_src.zip" ]]; then
    echo "Downloading unidic-mecab 2.1.2 (about 140MB)..."
    curl -sL "${URL}" -o "${WORK_DIR}/unidic_src.zip"
fi

echo "Extracting lex.csv..."
unzip -o -q -j "${WORK_DIR}/unidic_src.zip" "*/lex.csv" "*/COPYING" "*/BSD" \
    -d "${WORK_DIR}"

echo "Converting..."
python3 "${SCRIPT_DIR}/convert_unidic_dict.py" "${WORK_DIR}/lex.csv" -o "${OUT}"

# The notice a model trained with this dictionary has to be redistributed
# with, written next to the dictionary so it is not something to reconstruct
# later from memory.
cat > "${OUT_DIR}/dict_unidic.NOTICE" <<'NOTICE'
This dictionary, and any segmentlib model trained with it (the word list is
compiled into the model file), derives from:

  unidic-mecab 2.1.2
  Copyright (c) 2011-2013, The UniDic Consortium
  https://clrd.ninjal.ac.jp/unidic/

unidic-mecab is copyrighted free software, released under any of the GPL, the
LGPL, or the BSD License, at your option. Redistribution under the BSD License
requires that the above copyright notice, this list of conditions and the
following disclaimer be retained; the full text is in the BSD file of the
unidic-mecab source archive (kept under corpus/ud-gsd/.unidic by
scripts/fetch_unidic_dict.sh).

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.
NOTICE

echo "Done: ${OUT}"
echo "Redistribution notice: ${OUT_DIR}/dict_unidic.NOTICE (ship it with any model trained on this dictionary)"
