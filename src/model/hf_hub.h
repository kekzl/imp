#pragma once
#include <string>

namespace imp {

// Resolve a model identifier to a local directory path.
// If `model_id` is already a local path, returns it as-is.
// If it looks like a HF repo ID (contains '/'), tries to download via huggingface-cli.
// Returns empty string on failure.
// Optional: revision can be "main", a branch name, or a commit hash.
std::string resolve_model_path(const std::string& model_id,
                                const std::string& revision = "");

// Check if huggingface-cli is available on the system.
bool hf_cli_available();

// Find a single .gguf file in a directory. Returns its full path.
// If multiple .gguf files exist, returns the largest one.
// Returns empty string if no .gguf files found.
std::string find_gguf_in_dir(const std::string& dir);

// Resolve a model identifier to a .gguf file path.
// Combines resolve_model_path() + find_gguf_in_dir() for convenience:
//   - If model_id points to a .gguf file directly, returns it.
//   - If model_id is a directory or HF repo ID, resolves and finds the .gguf inside.
// Returns empty string on failure.
std::string resolve_model_gguf(const std::string& model_id,
                                const std::string& revision = "");

} // namespace imp
