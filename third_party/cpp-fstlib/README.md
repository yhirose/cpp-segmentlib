# cpp-fstlib (vendored)

Upstream: https://github.com/yhirose/cpp-fstlib
Revision: 2c9af63710777ee69b4b9062aa98be4349e93d88

`fstlib.h` is copied verbatim from that revision; `LICENSE` is its own.
Upstream publishes no tags, so the revision above is what "current" means
here. Update by replacing the header and this line together.

Note that this directory is on the compiler's include path (SYSTEM PRIVATE,
see src/CMakeLists.txt), so a file added here can shadow a standard header.
That is why this file is not called VERSION: on a case-insensitive
filesystem it would answer `#include <version>`.
