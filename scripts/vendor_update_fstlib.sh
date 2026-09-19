#!/usr/bin/env bash
# Updates the vendored copy of cpp-fstlib (third_party/cpp-fstlib) to the
# newest release tag, and records it in README.md.
#
# Upstream tags releases (vX.Y.Z) and bumps minor when the byte code format
# changes -- which a model's field 17 is -- so this tracks the newest tag
# rather than the default branch's tip.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="${SCRIPT_DIR}/../third_party/cpp-fstlib"
REPO="yhirose/cpp-fstlib"

TAG="$(git ls-remote --tags --refs "https://github.com/${REPO}.git" \
    | awk '{print $2}' | sed 's#refs/tags/##' | grep -v -- '-' | sort -V | tail -1)"
if [[ -z "${TAG}" ]]; then
    echo "error: could not resolve the newest tag for ${REPO}" >&2
    exit 1
fi
CURRENT_REV="$(grep '^Revision:' "${VENDOR_DIR}/README.md" | awk '{print $2}')"

if [[ "${TAG}" == "${CURRENT_REV}" ]]; then
    echo "Already at ${TAG}."
    exit 0
fi

echo "Updating cpp-fstlib: ${CURRENT_REV} -> ${TAG}"
curl -fsSL "https://raw.githubusercontent.com/${REPO}/${TAG}/fstlib.h" -o "${VENDOR_DIR}/fstlib.h"
curl -fsSL "https://raw.githubusercontent.com/${REPO}/${TAG}/LICENSE" -o "${VENDOR_DIR}/LICENSE"
sed -i.bak "s/^Revision: .*/Revision: ${TAG}/" "${VENDOR_DIR}/README.md"
rm -f "${VENDOR_DIR}/README.md.bak"

echo "Done. Review the diff (upstream may have moved behavior, not just this" \
     "header's contents), then \`just test\` before committing."
