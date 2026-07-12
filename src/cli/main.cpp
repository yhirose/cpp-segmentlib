#include <print>
#include <span>
#include <string_view>
#include <vector>

#include "commands.h"

namespace {

int usage() {
    std::println(stderr, "usage: segmenter <command> [options]");
    std::println(stderr, "");
    std::println(stderr, "commands:");
    std::println(stderr, "  predict   segment text from stdin using a model");
    std::println(stderr, "  train     train a model (--backend mlp; needs a training-enabled build)");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv + 1, argv + argc);
    if (args.empty()) {
        return usage();
    }

    const std::string_view command = args.front();
    const std::span<const std::string_view> rest{args.data() + 1, args.size() - 1};

    if (command == "predict") {
        return segmentlib::cli::run_predict(rest);
    }
    if (command == "train") {
        return segmentlib::cli::run_train(rest);
    }
    if (command == "-h" || command == "--help") {
        usage();
        return 0;
    }
    std::println(stderr, "segmenter: unknown command '{}'", command);
    return usage();
}
