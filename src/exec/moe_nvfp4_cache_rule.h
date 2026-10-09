#pragma once

// GGUF MoE NVFP4 decode cache: a layer is cached whole or not at all. Every consumer reads
// up + down (+ gate when gated); a lone projection is dead VRAM (Qwen3.6-35B-A3B UD-Q4_K_M:
// Q5_K/Q6_K down only, 5760 MiB never read, KV pool starved to 16 blocks).
#include <cstddef>
#include <cstdint>

namespace imp {

struct MoeProjCacheState {
    bool present = false;    // the projection exists in this layer
    bool cached = false;     // already in the cache
    bool cacheable = false;  // the re-quant path accepts it
};

// NVFP4 bytes of one packed [ne, rows, cols] expert tensor: e2m1 data + FP8 scale per 16 + FP32 per expert.
[[nodiscard]] constexpr size_t moe_nvfp4_packed_bytes(int64_t ne, int64_t rows, int64_t cols) {
    const size_t elems = static_cast<size_t>(ne) * static_cast<size_t>(rows) * static_cast<size_t>(cols);
    return elems / 2 + elems / 16 + static_cast<size_t>(ne) * sizeof(float);
}

[[nodiscard]] constexpr bool moe_layer_cache_whole(MoeProjCacheState gate, MoeProjCacheState up,
                                                   MoeProjCacheState down) {
    auto ok = [](MoeProjCacheState p) { return p.present && (p.cached || p.cacheable); };
    return ok(up) && ok(down) && (!gate.present || ok(gate));
}

}  // namespace imp
