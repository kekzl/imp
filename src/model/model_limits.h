#pragma once

// Structural limits on what a checkpoint may declare about itself.
//
// Every number a loader uses to size a container comes out of the file: the
// SafeTensors header, `config.json`, GGUF metadata, or a tensor name. None of
// those are the operator's, so each one that reaches a `resize` needs a
// ceiling. `sizeof(TransformerLayer)` is 9680 bytes, so the layer count alone
// turns `"num_hidden_layers": 2147483000` into a 18.9 TiB allocation before a
// single weight has been read.
//
// The values are deliberately far above anything that exists. The largest
// public checkpoint today is ~160 layers (Llama-3.1-405B) and 256 experts
// (DeepSeek-V3); these leave two orders of magnitude of room, so a legitimate
// model cannot hit them and a hostile one cannot ask for terabytes.
//
// The treatment differs by where the number came from, and this is the part
// worth getting right:
//
//   * A count the checkpoint DECLARES (config.json, GGUF metadata) is refused.
//     The model is what it says it is, so a nonsense declaration means the file
//     is broken or hostile, and `src/model/CLAUDE.md`'s first invariant applies:
//     refuse at load rather than serve something wrong.
//   * An index parsed out of a tensor NAME is dropped with a counted warning,
//     the same treatment `safetensors_loader.cpp` gives a malformed tensor. One
//     bad name in a 900-tensor checkpoint should not cost the load.

#include "model/model_config.h"

#include <charconv>
#include <string>

namespace imp {

inline constexpr int kMaxModelLayers = 1024;
inline constexpr int kMaxModelExperts = 4096;

// Parse a non-negative decimal index out of a tensor-name component.
// Returns -1 for empty input, a non-digit, or a value that does not fit an int.
//
// `std::atoi` did this before and is unfit twice over: its overflow behaviour
// is undefined, and it stops at the first non-digit instead of rejecting it.
inline int parse_index(const std::string& s) {
    if (s.empty())
        return -1;
    int value = 0;
    const char* begin = s.data();
    const char* end = begin + s.size();
    auto [ptr, ec] = std::from_chars(begin, end, value);
    if (ec != std::errc{} || ptr != end || value < 0)
        return -1;
    return value;
}

// Check the counts a checkpoint declares about itself, before anything is
// sized from them. Returns false and fills `err` when a count is impossible;
// the caller refuses the load.
inline bool validate_declared_dimensions(const ModelConfig& cfg, std::string* err) {
    auto fail = [err](const char* what, int got, int limit) {
        if (err)
            *err = std::string(what) + " is " + std::to_string(got) + ", which exceeds the limit of " +
                   std::to_string(limit) + " (broken or hostile checkpoint)";
        return false;
    };
    if (cfg.n_layers < 0 || cfg.n_layers > kMaxModelLayers)
        return fail("declared layer count", cfg.n_layers, kMaxModelLayers);
    if (cfg.n_experts < 0 || cfg.n_experts > kMaxModelExperts)
        return fail("declared expert count", cfg.n_experts, kMaxModelExperts);
    return true;
}

}  // namespace imp
