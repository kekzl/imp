#pragma once
#include <string>
#include "imp/types.h"

namespace imp {

// Resolves a model id to a local path (as-is, or the HF cache's
// models--org--name/snapshots/<latest>). Does not fetch: clean-host policy keeps Python
// tooling off the host; stage models via git clone or a copied cache dir. `revision` is
// accepted but only the most recent cached snapshot is returned.
std::string resolve_model_path(const std::string& model_id, const std::string& revision = "");

// HF hub cache root this resolver reads (HUGGINGFACE_HUB_CACHE, HF_HOME/hub,
// ~/.cache/huggingface/hub), or "" if none. Confines an "org/repo" request-name resolution
// to the cache; a request field never names a filesystem path.
std::string hf_cache_dir();

// Find a single .gguf file in a directory. Returns its full path.
// If multiple .gguf files exist, returns the largest one.
// Returns empty string if no .gguf files found.
std::string find_gguf_in_dir(const std::string& dir);

// Resolves a model id to a .gguf path: resolve_model_path() + find_gguf_in_dir(). Returns
// the path directly if model_id already names a .gguf file.
std::string resolve_model_gguf(const std::string& model_id, const std::string& revision = "");

// Resolves a model id and auto-detects format: checks SafeTensors first (directory with
// model.safetensors[.index.json]), else falls back to GGUF. Sets out_format accordingly.
std::string resolve_model_auto(const std::string& model_id, ImpModelFormat& out_format,
                               const std::string& revision = "");

// Check if a directory contains SafeTensors model files.
bool is_safetensors_dir(const std::string& dir);

}  // namespace imp
