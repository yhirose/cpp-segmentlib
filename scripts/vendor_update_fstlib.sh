#!/usr/bin/env bash
# Updates the vendored copy of cpp-fstlib (third_party/cpp-fstlib) to the tip
# of upstream's default branch, and records the revision in README.md.
#
# Upstream publishes no tags or releases, so "latest" always means the tip
# commit of the default branch at the time this runs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="${SCRIPT_DIR}/../third_party/cpp-fstlib"
REPO="yhirose/cpp-fstlib"

REV="$(curl -sL "https://api.github.com/repos/${REPO}/commits/HEAD" \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["sha"])')"
CURRENT_REV="$(grep '^Revision:' "${VENDOR_DIR}/README.md" | awk '{print $2}')"

if [[ "${REV}" == "${CURRENT_REV}" ]]; then
    echo "Already at upstream HEAD (${REV})."
    exit 0
fi

echo "Updating cpp-fstlib: ${CURRENT_REV} -> ${REV}"
curl -sL "https://raw.githubusercontent.com/${REPO}/${REV}/fstlib.h" -o "${VENDOR_DIR}/fstlib.h"
curl -sL "https://raw.githubusercontent.com/${REPO}/${REV}/LICENSE" -o "${VENDOR_DIR}/LICENSE"
sed -i.bak "s/^Revision: .*/Revision: ${REV}/" "${VENDOR_DIR}/README.md"
rm -f "${VENDOR_DIR}/README.md.bak"

echo "Done. Review the diff (upstream may have moved behavior, not just this" \
     "header's contents), then \`just test\` before committing."
