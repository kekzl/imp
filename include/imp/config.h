#pragma once

#include "imp/types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    // Device
    int device_id;

    // Memory
    size_t gpu_memory_pool_size;   // Pre-allocated GPU pool (bytes), 0 = auto
    size_t kv_cache_max_blocks;    // Max KV cache blocks, 0 = auto

    // Inference
    int max_batch_size;
    int max_seq_len;
    ImpDType compute_dtype;        // FP16, BF16, or FP8_E4M3

    // Sampling defaults
    float temperature;
    float top_p;
    int top_k;
    int max_tokens;

    // CUDA 13.1 features
    int enable_green_contexts;     // 0 = off, 1 = on
    float green_ctx_prefill_ratio; // SM ratio for prefill (default 0.8)
    int enable_pdl;                // Programmatic Dependent Launch
    int enable_cuda_graphs;        // CUDA Graph capture for decode

    // Speculative decoding
    int enable_speculative;        // 0 = off, 1 = on
    const char* draft_model_path;  // path to draft model (GGUF/SafeTensors)
    ImpModelFormat draft_model_format; // format of the draft model
    int spec_k;                    // number of draft tokens (default 4)

    // Self-speculative decoding (layer-skip draft from same model)
    int enable_self_speculative;   // 0 = off, 1 = on
    int self_spec_k;               // draft tokens per step (default 2)
    int self_spec_exit_layer;      // layers to run in draft (-1 = auto)
    int self_spec_skip_n;          // layers to skip in draft (-1 = auto)

    // Layer offloading
    int gpu_layers;                // Layers to keep on GPU (-1 = all, 0 = all offloaded)

    // KV cache precision
    ImpDType kv_cache_dtype;       // FP16 (default), FP8_E4M3, or INT8 for half-size KV cache

    // SSM state precision
    ImpDType ssm_state_dtype;      // FP32 (default) or FP16 for SSM h_state

    // VRAM budget
    size_t vram_budget_mb;         // Max GPU memory to use (MiB), 0 = use all available

    // Chunked prefill
    int prefill_chunk_size;        // Max tokens per prefill chunk (0 = no chunking)

    // Prefill weight cache precision
    int use_fp8_prefill;           // 0 = FP16 weight cache (default), 1 = FP8 E4M3 prefill cache

    // NVFP4 decode weight cache
    int use_nvfp4_decode;          // -1 = auto (sm_120→mode2, sm_90→mode1), 0 = off, 1 = additive, 2 = NVFP4 only

    // MXFP4 prefill: use CUTLASS MXFP4 GEMM for prefill (converts NVFP4 cache to MXFP4 format)
    int use_mxfp4_prefill;         // 0 = off (default), 1 = on (requires sm_120 + NVFP4 cache)

    // Prefix caching
    int use_prefix_caching;        // 0 = off (default), 1 = on — reuse KV blocks for shared prefixes
    char prefix_cache_path[512];   // path to save/load prefix cache (empty = disabled)

    // Threading
    int num_cpu_threads;           // 0 = auto (hardware_concurrency)

    // Vision (multimodal)
    const char* mmproj_path;       // Path to mmproj GGUF file (NULL = text-only)
} ImpConfig;

// Returns a config with sensible defaults
ImpConfig imp_config_default(void);

#ifdef __cplusplus
}
#endif
