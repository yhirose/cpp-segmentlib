#!/bin/bash
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

root=$(cd "$(dirname "$0")/.." && pwd)
build=${1:-$root/build-consumer}

rm -rf "$build"
# No SEGMENTLIB_* options and no CMAKE_BUILD_TYPE: this is the naive consumer.
cmake -S "$root/tests/consumer" -B "$build" -DSEGMENTLIB_ROOT="$root"
cmake --build "$build"

fail() { echo "consumer check FAILED: $1" >&2; exit 1; }

# 1. The consumer compiles against the public headers alone, links, and runs.
"$build/consumer" || fail "the consumer binary reported failures"

# 2. Our developer targets stay out of the consumer's `all`. Search by artifact
#    rather than by target name so a rename cannot silently pass this.
for artifact in segmentlib_tests segmenter bench_segment bench_kytea; do
    found=$(find "$build" -name "$artifact" -o -name "$artifact.exe" | head -1)
    [ -z "$found" ] || fail "$artifact was built into the consumer's build ($found)"
done

# 3. The test suite's doctest dependency is not fetched over the network.
[ ! -d "$build/_deps/doctest-src" ] || fail "doctest was fetched into the consumer's build"

# 4. The consumer set no build type, so segmentlib must not pick one for them.
if grep -qE '^CMAKE_BUILD_TYPE:STRING=.+$' "$build/CMakeCache.txt"; then
    fail "segmentlib forced a CMAKE_BUILD_TYPE on the consumer: \
$(grep -E '^CMAKE_BUILD_TYPE:' "$build/CMakeCache.txt")"
fi

echo "consumer check passed"
