#!/usr/bin/env bash
# Builds tests/consumer/ against this working tree and checks that consuming
# segmentlib gets you the library and nothing else.
#
# Nothing in the test suite covers this: every other build here has segmentlib
# as the top-level project, so the consumer path is only exercised by someone
# outside the repo. Two rounds of consumer-side defects (an include path that
# resolved into the consumer's tree, and developer targets forced into the
# consumer's `all`) were both found by reading CMake files, with every job
# green. This is the regression guard for them.
#
# Usage: scripts/check_consumer.sh [build-dir]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${1:-${ROOT}/build-consumer}"

rm -rf "$BUILD"
# No SEGMENTLIB_* options and no CMAKE_BUILD_TYPE: this is the naive consumer.
cmake -S "$ROOT/tests/consumer" -B "$BUILD" -DSEGMENTLIB_ROOT="$ROOT"
# No -G, so this is the default generator (Makefiles), which is serial without
# --parallel. Every other job configures Ninja and parallelizes for free.
cmake --build "$BUILD" --parallel

fail() { echo "consumer check FAILED: $1" >&2; exit 1; }

# 1. The consumer compiles against the public headers alone, links, and runs.
"$BUILD/consumer" || fail "the consumer binary reported failures"

# 2. Our developer targets stay out of the consumer's `all`. Enumerating what
#    was actually built beats listing target names: the list would have to be
#    kept in step with our CMake, and a target added later is exactly the
#    regression this is here to catch. CMakeFiles/ holds CMake's own
#    compiler-probe binaries, which are not ours.
EXTRA="$(find "$BUILD" -type f -perm -u+x -not -path '*/CMakeFiles/*' -not -name consumer)"
[ -z "$EXTRA" ] || fail "the consumer's build produced more than the library:
$EXTRA"

# 3. The test suite's doctest dependency is not fetched over the network.
[ ! -d "$BUILD/_deps/doctest-src" ] || fail "doctest was fetched into the consumer's build"

# 4. The consumer set no build type, so segmentlib must not pick one for them.
if grep -qE '^CMAKE_BUILD_TYPE:STRING=.+$' "$BUILD/CMakeCache.txt"; then
    fail "segmentlib forced a CMAKE_BUILD_TYPE on the consumer: \
$(grep -E '^CMAKE_BUILD_TYPE:' "$BUILD/CMakeCache.txt")"
fi

echo "consumer check passed"
