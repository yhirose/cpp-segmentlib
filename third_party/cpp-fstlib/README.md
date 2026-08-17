# cpp-fstlib (vendored)

Upstream: https://github.com/yhirose/cpp-fstlib
Revision: 2c9af63710777ee69b4b9062aa98be4349e93d88

`fstlib.h` is copied verbatim from that revision; `LICENSE` is its own.
Upstream publishes no tags, so the revision above is what "current" means
here. Update by replacing the header and this line together.

The include path is the *parent* directory (`third_party/`, SYSTEM INTERFACE,
see src/CMakeLists.txt), and `mlp/dictionary.h` includes
`"cpp-fstlib/fstlib.h"`. Nothing in this directory is reachable as a bare
header name, so a file added here shadows nothing.
