#pragma once

// GGUF MoE NVFP4 decode cache: a layer is cached whole or not at all. Every consumer reads
// up + down (+ gate when gated); a lone projection is dead VRAM (Qwen3.6-35B-A3B UD-Q4_K_M:
// Q5_K/Q6_K down only, 5760 MiB never read, KV pool starved to 16 blocks).
namespace imp {

struct MoeProjCacheState {
    bool present = false;    // the projection exists in this layer
    bool cached = false;     // already in the cache
    bool cacheable = false;  // the re-quant path accepts it
};

[[nodiscard]] constexpr bool moe_layer_cache_whole(MoeProjCacheState gate, MoeProjCacheState up,
                                                   MoeProjCacheState down) {
    auto ok = [](MoeProjCacheState p) { return p.present && (p.cached || p.cacheable); };
    return ok(up) && ok(down) && (!gate.present || ok(gate));
}

}  // namespace imp
