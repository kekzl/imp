#pragma once
#include <string>

namespace imp {
namespace test {

// Generate a minimal valid GGUF file for testing.
// Returns path to temp file (caller must unlink after use).
// Architecture: "llama" (simplest dense transformer)
// Config: 1 layer, d_model=64, n_heads=2, head_dim=32, vocab=256, d_ff=128
// Weights: random FP16, ~200 KB total
// Tokenizer: minimal BPE with 256 single-byte tokens
std::string generate_gguf_stub(const std::string& arch = "llama");

}  // namespace test
}  // namespace imp
