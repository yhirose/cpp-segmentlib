# cpp-fstlib (vendored)

Upstream: https://github.com/yhirose/cpp-fstlib
Revision: v0.1.0

`fstlib.h` is copied verbatim from that tag; `LICENSE` is its own.
Update with `just vendor-update`, which tracks the newest `vX.Y.Z` tag and
replaces the header and this line together.

The include path is the *parent* directory (`third_party/`, SYSTEM INTERFACE,
see src/CMakeLists.txt), and `mlp/dictionary.h` includes
`"cpp-fstlib/fstlib.h"`. Nothing in this directory is reachable as a bare
header name, so a file added here shadows nothing.
