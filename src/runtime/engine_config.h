#pragma once

// EngineConfig alone: TUs that size or plan an engine (vram_budget.cpp, tests) need the
// struct, not Engine, and stop rebuilding on every engine.h edit.

#include "core/qtype.h"

#include <cstddef>
#include <string>

namespace imp {

struct EngineConfig {
    int max_batch_size = 0;       // 0 = auto (engine detects from model size vs VRAM)
    int max_seq_len = 0;          // 0 = auto (engine detects from model metadata + VRAM)
    int kv_cache_max_blocks = 0;  // 0 = auto
    int kv_ceiling_seqs = 0;      // KV pool ceiling in max_seq_len sequences, 0 = max_batch_size
    int swa_shared_tokens = 0;    // SWA group tokens charged once per pool (prefill chunk past the burst)
    bool use_cuda_graphs = true;
    bool use_pdl = true;
    QType compute_dtype = QType::F16;

    // Default sampling parameters
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = 0;
    int seed = -1;

    // KV cache dtype: FP16 (default) or FP8_E4M3 for ~50% KV VRAM savings
    QType kv_cache_dtype = QType::F16;

    // SSM state dtype: FP32 (default) or FP16 for ~50% VRAM savings on h_state
    QType ssm_state_dtype = QType::F32;

    // VRAM budget: max GPU memory to use (MiB), 0 = use all available
    size_t vram_budget_mb = 0;

    // Budget-planner tuning (imp.conf [vram], see RuntimeConfig::Vram).
    // kv_fraction: KV pool share of post-reserve VRAM, clamped [0.05, 1.0].
    // vram_reserve_floor_pct: reserve floor % of total VRAM, clamped [0, 50]; the 4 % allocator headroom
    // always applies.
    float kv_fraction = 1.0f;
    int vram_reserve_floor_pct = 0;
    // library_reserve_mb: cuBLAS/CUTLASS claim on the first forward pass
    // (MEMORY.md A1.5). -1 = measured default. Fixed charge: floors the
    // reserve above regardless of --vram-budget.
    int library_reserve_mb = -1;

    // Layer offloading: number of layers to keep on GPU (-1 = all on GPU, 0 = all offloaded)
    int gpu_layers = -1;

    // KV cache block size (tokens per block). 0 = auto = 16, or imp.conf
    // kv_cache.block_size. Explicit values are checked by kv_block_size_error()
    // at init and refused, not rounded.
    int kv_block_size = 0;

    // Chunked prefill
    int prefill_chunk_size = -1;  // -1 = per-arch default (2048 if supported, 0 otherwise)

    // FP8 prefill weight cache: uses FP8 E4M3 instead of FP16 for ~2x prefill throughput
    bool use_fp8_prefill = false;

    // NVFP4 decode weight cache: -1=auto, 0=off, 1=additive (FP16+NVFP4), 2=NVFP4 only
    int use_nvfp4_decode = -1;
    // nvfp4_decode_all: extend the NVFP4 decode cache to Q4_K/Q3_K/Q2_K.
    // fp8_lm_head: gemm.nvfp4_lm_head=fp8, or auto with a 16-bit head (#2224): the budget reserves the
    // FP8 head instead of NVFP4 (larger than NVFP4, so an auto fallback to NVFP4 stays inside it).
    // lm_head_source_only: auto with an 8-bit quantized head, neither FP8 nor NVFP4 is built (#2224).
    bool nvfp4_decode_all = false, fp8_lm_head = false, lm_head_source_only = false;
    // Planner credits the Q8_0 sources Phase 4c frees for their IMMA planes (engine init sets it).
    bool q8_drop_source = false;
    // Planner prices dense Q4_K weights at 0, no FP16 copy (engine init sets it).
    bool q4k_in_place = false;

    // Minimum KV cache tokens. Budget planner guarantees at least this many
    // tokens of KV capacity before allocating weight caches. 0 = auto.
    int min_kv_tokens = 0;

    // Dual-path quantization: attention weights (WQ/WK/WV/WO) stay at FP8 for quality,
    // FFN weights (gate/up/down) use NVFP4 for 2x bandwidth reduction during decode.
    bool dual_path_quant = false;

    // Prefix caching: reuse KV cache blocks for shared token prefixes
    bool use_prefix_caching = false;
    std::string prefix_cache_path;  // path to persist prefix cache (empty = disabled)
    // Cap on cache_control/cache_prompt-pinned blocks, percent of the KV
    // pool (0 = pin requests are ignored).
    int prefix_pin_budget_pct = 25;

    // Vision (multimodal)
    std::string mmproj_path;  // path to mmproj GGUF, empty = text-only

    // StreamingLLM (Xiao et al., 2023): keep first N sink tokens + last W
    // tokens, drop the rest. FP16 GQA decode path only; quantized KV dtypes
    // ignore these settings.
    bool streaming_kv_enabled = false;
    bool streaming_kv_auto = true;   // auto-enable StreamingLLM when KV cache >90% full
    int streaming_kv_n_sinks = 4;    // # initial tokens to always attend
    int streaming_kv_window = 0;     // 0 = derive from ModelConfig::sliding_window
    int streaming_kv_threshold = 0;  // 0 = auto: n_sinks + window + 2 * the resolved KV block size
};

}  // namespace imp
