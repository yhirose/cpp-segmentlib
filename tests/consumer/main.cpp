// A stand-in for a project that depends on segmentlib. It sees only the
// installed-facing surface: the `segmentlib` target and the headers under
// include/segmentlib/. It never adds include/ or src/ to its own include path,
// so an internal header leaking into a public one fails here.
//
// The assertions are deliberately thin. This checks that the library can be
// consumed and called at all; what it computes is the test suite's job.
#include <cstdio>
#include <string>

#include <segmentlib/output.h>
#include <segmentlib/segmenter.h>
#include <segmentlib/types.h>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) {
        ++failures;
    }
}

}  // namespace

int main() {
    // Calls into the library proper, without needing a model file: the loader
    // reports a missing model rather than throwing or aborting.
    auto missing = segmentlib::Segmenter::load("no-such-model.bin");
    check(!missing.has_value(), "loading a missing model fails");
    check(!missing.has_value() && missing.error().code == segmentlib::ErrorCode::IoError,
          "the failure is reported as IoError");

    // The output helper over a hand-built segmentation, so the check needs no
    // trained model: "ab" + "c/d", with the KyTea escaping of the '/'.
    const std::string text = "abc/d";
    const segmentlib::Segments segments = {{0, 2}, {2, 5}};
    std::string line;
    segmentlib::append_full_line(segments, text, line);
    check(line == "ab c\\/d", "append_full_line escapes and joins");

    if (failures == 0) {
        std::printf("consumer: all checks passed\n");
        return 0;
    }
    std::printf("consumer: %d check(s) failed\n", failures);
    return 1;
}
