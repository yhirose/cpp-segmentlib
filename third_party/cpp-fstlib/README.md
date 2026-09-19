# cpp-fstlib (vendored)

Upstream: https://github.com/yhirose/cpp-fstlib
Revision: a11658e6b4c5c9b2d0a93c0891891ec13940c7ba

`fstlib.h` is copied verbatim from that revision; `LICENSE` is its own.
Upstream publishes no tags, so the revision above is what "current" means
here. Update by replacing the header and this line together.

The include path is the *parent* directory (`third_party/`, SYSTEM INTERFACE,
see src/CMakeLists.txt), and `mlp/dictionary.h` includes
`"cpp-fstlib/fstlib.h"`. Nothing in this directory is reachable as a bare
header name, so a file added here shadows nothing.
