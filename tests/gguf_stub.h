#pragma once
#include <string>

namespace imp {
namespace test {

// Generate a minimal valid GGUF file for testing.
// Returns path to the stub file (caller must call remove_gguf_stub after use).
// Architecture: "llama" (simplest dense transformer)
// Config: 1 layer, d_model=64, n_heads=2, head_dim=32, vocab=256, d_ff=128
// Weights: random FP16, ~200 KB total
// Tokenizer: minimal BPE with 256 single-byte tokens
std::string generate_gguf_stub(const std::string& arch = "llama");

// Stub layout: /tmp/imp_stub_XXXXXX/{model.gguf,warm/}. Warm-cache files of stub loads go to
// warm/ (never the persistent imp-test-cache volume, #2192) and die with the dir.
std::string stub_cache_dir(const std::string& stub_path);

// Arms the pending RuntimeConfig so the NEXT imp_context_create warm-caches into stub_cache_dir.
// The pending config is consumed per context create: call before every create.
void arm_stub_warm_cache(const std::string& stub_path);

// Removes the stub's whole dir (model + warm cache). Ignores paths outside /tmp/imp_stub_*.
void remove_gguf_stub(const std::string& stub_path);

}  // namespace test
}  // namespace imp
