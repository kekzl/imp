#pragma once
#include <string>
#include "imp/types.h"

namespace imp {

// Resolve a model identifier to a local path.
// If `model_id` is already a valid local path, returns it as-is.
// If it looks like a HF repo ID (contains '/'), checks the HF cache
// (`$HUGGINGFACE_HUB_CACHE` / `$HF_HOME/hub` / `$HOME/.cache/huggingface/hub`)
// for a matching `models--<org>--<name>/snapshots/<latest>` directory.
// Returns empty string on failure.
//
// **Does not fetch.** imp's clean-host policy keeps Python tooling
// (huggingface-cli, etc.) off the host and the build container. A model
// that isn't already cached must be staged by the user (typically via
// `git clone https://huggingface.co/<org>/<name>` or by copying a cache
// directory from another machine). The `revision` argument is accepted
// for API compatibility but only the most recent cached snapshot is
// returned — pre-stage the revision you want.
std::string resolve_model_path(const std::string& model_id, const std::string& revision = "");

// Find a single .gguf file in a directory. Returns its full path.
// If multiple .gguf files exist, returns the largest one.
// Returns empty string if no .gguf files found.
std::string find_gguf_in_dir(const std::string& dir);

// Resolve a model identifier to a .gguf file path.
// Combines resolve_model_path() + find_gguf_in_dir() for convenience:
//   - If model_id points to a .gguf file directly, returns it.
//   - If model_id is a directory or HF repo ID, resolves and finds the .gguf inside.
// Returns empty string on failure.
std::string resolve_model_gguf(const std::string& model_id, const std::string& revision = "");

// Resolve a model identifier to a path and auto-detect format.
// Checks for SafeTensors first (directory with model.safetensors[.index.json]),
// then falls back to GGUF resolution.
// Sets out_format to the detected format.
// Returns empty string on failure.
std::string resolve_model_auto(const std::string& model_id, ImpModelFormat& out_format,
                               const std::string& revision = "");

// Check if a directory contains SafeTensors model files.
bool is_safetensors_dir(const std::string& dir);

}  // namespace imp
