#pragma once

#include <span>
#include <string_view>

namespace segmentlib::cli {

// Each command takes the arguments following the subcommand name and returns a
// process exit code.
int run_predict(std::span<const std::string_view> args);
int run_train(std::span<const std::string_view> args);

}  // namespace segmentlib::cli
