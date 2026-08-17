#!/usr/bin/env bash
#
# Release a new version of cpp-segmentlib.
#
# Usage: ./release.sh [--run] [--minor | --major]
#
# By default, runs in dry-run mode (no changes made).
# Pass --run to actually update files, commit, tag, and push.
# Pass --minor or --major to force a bigger bump than the automatic choice.
#
# Unlike cpp-httplib/cpp-peglib there is no abidiff CI job here, but there is
# an analogous automatic signal: whether the bundled reference model
# (models/mlp/ja-ud-gsd.mod) changed since the last release. A retrain
# changes segmentation output for every downstream consumer, which is a
# bigger deal than most code changes, so it bumps minor instead of patch --
# same reasoning as an ABI break, different detector.
#
# This script:
#   1. Reads the current version from include/segmentlib/types.h
#   2. Checks that the working directory is clean
#   3. Verifies CI status of the latest commit (the Windows job is
#      continue-on-error in the workflow, so it cannot fail the run)
#   4. Determines the version to release:
#        - if the current version has no vX.Y.Z tag yet, it is released
#          as-is (the bootstrap case: the version was bumped by hand or
#          set by a previous aborted release)
#        - otherwise: the bundled model changed since the last release tag
#          → minor bump (e.g., 0.1.0 → 0.2.0); model unchanged → patch bump
#          (e.g., 0.1.0 → 0.1.1); --minor / --major override either way
#   5. Updates include/segmentlib/types.h (both macros) and the VERSION in
#      CMakeLists.txt's project()
#   6. Commits, tags (vX.Y.Z), and pushes
#
# Retraining the bundled model is NOT part of this script: it is a
# pre-release act with its own checklist (docs/RELEASING.md step 1-2),
# done in an ordinary commit that CI then validates like any other.

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_FILE="models/mlp/ja-ud-gsd.mod"

DRY_RUN=1
FORCE_MINOR=0
FORCE_MAJOR=0
while [ $# -gt 0 ]; do
  case "$1" in
    --run)
      DRY_RUN=0
      shift
      ;;
    --minor)
      FORCE_MINOR=1
      shift
      ;;
    --major)
      FORCE_MAJOR=1
      shift
      ;;
    *)
      echo "Usage: $0 [--run] [--minor | --major]"
      exit 1
      ;;
  esac
done

# --- Step 1: Read current version from types.h ---
VERSION_HEADER="include/segmentlib/types.h"
CURRENT_VERSION=$(sed -n 's/^#define SEGMENTLIB_VERSION "\([^"]*\)"/\1/p' "$VERSION_HEADER")
IFS='.' read -r V_MAJOR V_MINOR V_PATCH <<< "$CURRENT_VERSION"

echo "==> Current version: $CURRENT_VERSION"

# --- Step 2: Check working directory is clean ---
if [ -n "$(git status --porcelain)" ]; then
  echo "Error: working directory is not clean"
  exit 1
fi

# --- Step 3: Check CI status of the latest commit ---
echo ""
echo "==> Checking CI status of the latest commit..."

HEAD_SHA=$(git rev-parse HEAD)
HEAD_SHORT=$(git rev-parse --short HEAD)
echo "    Latest commit: $HEAD_SHORT"

# Fetch all workflow runs for the HEAD commit. A commit can accumulate
# multiple runs of the same workflow (reruns), so judge only the most
# recent run of each workflow.
RUNS=$(gh run list --commit "$HEAD_SHA" \
         --json name,status,conclusion,headSha,createdAt |
       jq '[group_by(.name)[] | max_by(.createdAt)]')

NUM_RUNS=$(echo "$RUNS" | jq 'length')

if [ "$NUM_RUNS" -eq 0 ]; then
  echo "Error: No CI runs found for commit $HEAD_SHORT."
  echo "       Wait for CI to complete before releasing."
  exit 1
fi

echo "    Found $NUM_RUNS workflow run(s):"

FAILED=0
RUNNING=0
while IFS=$'\t' read -r name status conclusion; do
  # A run that hasn't completed yet has an empty conclusion; don't treat it
  # as a failure — the release should wait until CI finishes.
  if [ "$status" != "completed" ]; then
    echo "      [ .. ] $name (still running)"
    RUNNING=1
    continue
  fi

  if [ "$conclusion" = "success" ]; then
    echo "      [ OK ] $name"
  else
    echo "      [FAIL] $name ($conclusion)"
    FAILED=1
  fi
done < <(echo "$RUNS" | jq -r '.[] | [.name, .status, .conclusion] | @tsv')

if [ "$RUNNING" -eq 1 ]; then
  echo ""
  echo "Error: Some CI checks are still running. Wait for them to complete before releasing."
  exit 1
fi

if [ "$FAILED" -eq 1 ]; then
  echo ""
  echo "Error: Some CI checks failed. Fix them before releasing."
  exit 1
fi

echo "    All CI checks passed."

# --- Step 4: Determine the version to release ---
if ! git rev-parse -q --verify "refs/tags/v$CURRENT_VERSION" >/dev/null; then
  # The version in the header has never been tagged: release it as-is.
  NEW_VERSION="$CURRENT_VERSION"
  echo ""
  echo "==> v$CURRENT_VERSION is not tagged yet → releasing the current version as-is"
elif [ "$FORCE_MAJOR" -eq 1 ]; then
  NEW_MAJOR=$((V_MAJOR + 1))
  NEW_VERSION="$NEW_MAJOR.0.0"
  echo ""
  echo "==> --major specified → major bump"
elif [ "$FORCE_MINOR" -eq 1 ]; then
  NEW_MINOR=$((V_MINOR + 1))
  NEW_VERSION="$V_MAJOR.$NEW_MINOR.0"
  echo ""
  echo "==> --minor specified → minor bump"
else
  MODEL_CHANGED=0
  if ! git diff --quiet "v$CURRENT_VERSION" -- "$MODEL_FILE"; then
    MODEL_CHANGED=1
  fi

  if [ "$MODEL_CHANGED" -eq 1 ]; then
    NEW_MINOR=$((V_MINOR + 1))
    NEW_VERSION="$V_MAJOR.$NEW_MINOR.0"
    echo ""
    echo "==> $MODEL_FILE changed since v$CURRENT_VERSION → minor bump"
  else
    NEW_PATCH=$((V_PATCH + 1))
    NEW_VERSION="$V_MAJOR.$V_MINOR.$NEW_PATCH"
    echo ""
    echo "==> $MODEL_FILE unchanged since v$CURRENT_VERSION → patch bump"
  fi
fi

IFS='.' read -r N_MAJOR N_MINOR N_PATCH <<< "$NEW_VERSION"
VERSION_HEX=$(printf "0x%02x%02x%02x" "$N_MAJOR" "$N_MINOR" "$N_PATCH")

if [ "$DRY_RUN" -eq 1 ]; then
  echo "==> [DRY RUN] Version to release: $NEW_VERSION ($VERSION_HEX)"
else
  echo "==> Version to release: $NEW_VERSION ($VERSION_HEX)"
fi

# --- Step 5: Update files (no-op when releasing the current version) ---
echo ""
if [ "$DRY_RUN" -eq 1 ]; then
  if [ "$NEW_VERSION" != "$CURRENT_VERSION" ]; then
    echo "==> [DRY RUN] Would update $VERSION_HEADER and CMakeLists.txt:"
    echo "    SEGMENTLIB_VERSION     = \"$NEW_VERSION\""
    echo "    SEGMENTLIB_VERSION_NUM = \"$VERSION_HEX\""
    echo "    project(... VERSION $NEW_VERSION ...)"
    echo ""
    echo "==> [DRY RUN] Would commit, tag v$NEW_VERSION, and push."
  else
    echo "==> [DRY RUN] Version files already at $NEW_VERSION; would tag v$NEW_VERSION and push."
  fi
  echo ""
  echo "==> Dry run complete. No changes were made."
else
  if [ "$NEW_VERSION" != "$CURRENT_VERSION" ]; then
    echo "==> Updating $VERSION_HEADER and CMakeLists.txt..."
    sed -i '' "s/#define SEGMENTLIB_VERSION \"[^\"]*\"/#define SEGMENTLIB_VERSION \"$NEW_VERSION\"/" "$VERSION_HEADER"
    sed -i '' "s/#define SEGMENTLIB_VERSION_NUM \"0x[0-9a-fA-F]*\"/#define SEGMENTLIB_VERSION_NUM \"$VERSION_HEX\"/" "$VERSION_HEADER"
    sed -i '' "s/project(cpp-segmentlib VERSION [0-9.]* /project(cpp-segmentlib VERSION $NEW_VERSION /" CMakeLists.txt
    echo "    SEGMENTLIB_VERSION     = \"$NEW_VERSION\""
    echo "    SEGMENTLIB_VERSION_NUM = \"$VERSION_HEX\""

    # --- Step 6: Commit, tag, and push ---
    echo ""
    echo "==> Committing and tagging..."
    git add "$VERSION_HEADER" CMakeLists.txt
    git commit -m "Release v$NEW_VERSION"
  else
    echo "==> Version files already at $NEW_VERSION; tagging the current commit."
  fi
  git tag "v$NEW_VERSION"

  echo ""
  echo "==> Pushing..."
  git push && git push --tags

  echo ""
  echo "==> Released v$NEW_VERSION"
fi
