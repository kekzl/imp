#pragma once

// MoE experts stored as one 3-D stack per projection (gpt-oss `mlp.experts.gate_up_proj`,
// Gemma-4 `experts.gate_up_proj`) become the per-expert 2-D matrices the loader reads
// (`experts.<e>.{gate,up,down}_proj.weight`): the layout every NVFP4 export of these models
// already uses (Gemma-4-26B-A4B-it-NVFP4 stores 128 x 3 matrices per layer) and the only one
// weight_map.cpp maps to expert_w_gate/up/down. The stack's layout is looked up by model_type,
// never inferred from shape: gpt-oss-20b's down stack is [32, 2880, 2880], where a shape test
// cannot tell [ne, K, N] from [ne, N, K], and both quantize into a checkpoint that loads.
// Host-only, CUDA-free, so the CPU test lane covers the split.

#include "model/safetensors_raw.h"

#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <vector>

namespace imp::quantize {

// How the fused gate_up stack pairs its rows: Gemma-4 chunks (gate rows first, then up),
// gpt-oss interleaves (g0, u0, g1, u1, ...: `gate_up[..., ::2]` / `[..., 1::2]`).
enum class GateUpOrder { Concatenated, Interleaved };

struct StackedExpertLayout {
    // gpt-oss stores [ne, K, N] (`hidden @ gate_up_proj[e]`); Gemma-4 stores [ne, N, K]
    // (`F.linear(hidden, gate_up_proj[e])`), the orientation a Linear weight has.
    bool transposed = false;
    GateUpOrder gate_up = GateUpOrder::Concatenated;
};

// The layout `model_type` stores its expert stacks in, or nullopt when this tool has no
// descriptor for it (the caller refuses the checkpoint).
std::optional<StackedExpertLayout> stacked_expert_layout(const std::string& model_type);

// `model_type` from config.json: top level first, then text_config (Gemma-4 wraps its text
// model). "" when the file is unreadable or neither is a string.
std::string model_type_from_config(const std::string& config_json_path);

enum class ExpertPart { Gate, Up, Down };

struct DestackedMatrix {
    std::string name;  // `<prefix>.experts.<e>.<proj>.weight`
    int64_t expert = 0;
    ExpertPart part = ExpertPart::Down;
    int64_t N = 0, K = 0;  // [N, K], the orientation a Linear weight has
};

// Every 2-D matrix `stack` becomes, expert-major (e0 gate, e0 up, e1 gate, ...): gate and up of
// one expert are adjacent so the caller can give them one tensor scale. Error when the tensor is
// not a float 3-D `...experts.gate_up_proj` / `...experts.down_proj`, when the fused row count is
// odd, or when K is not a multiple of the 16-value micro-block.
std::expected<std::vector<DestackedMatrix>, std::string> destack_plan(const RawTensor& stack,
                                                                      const StackedExpertLayout& layout);

// The [N, K] FP16 rows of one planned matrix, widened from BF16/F16 like every other source
// weight. `m` must come from destack_plan(stack, layout).
std::vector<uint16_t> destack_read(const RawTensor& stack, const StackedExpertLayout& layout,
                                   const DestackedMatrix& m);

}  // namespace imp::quantize
