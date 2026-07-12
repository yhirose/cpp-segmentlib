#!/usr/bin/env bash
# Downloads the KyTea Japanese model into models/kytea/.
#
# The model is no longer served directly from phontron.com (the site was
# rebuilt as a JS SPA and the old static /download/model/ path now returns
# the SPA shell instead of the file), so this pulls a Wayback Machine
# snapshot instead.
set -euo pipefail

MODEL_NAME="jp-0.4.7-5.mod"
ARCHIVE_URL="http://web.archive.org/web/20250824162500/http://www.phontron.com/kytea/download/model/${MODEL_NAME}.gz"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/../models/kytea"
mkdir -p "${OUT_DIR}"

echo "Downloading ${MODEL_NAME}.gz from Wayback Machine archive..."
curl -sL "${ARCHIVE_URL}" -o "${OUT_DIR}/${MODEL_NAME}.gz"

echo "Decompressing..."
gunzip -kf "${OUT_DIR}/${MODEL_NAME}.gz"

echo "Done: ${OUT_DIR}/${MODEL_NAME}"
