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

    // Layer offloading
    int gpu_layers;                // Layers to keep on GPU (-1 = all, 0 = all offloaded)

    // SSM state precision
    ImpDType ssm_state_dtype;      // FP32 (default) or FP16 for SSM h_state

    // VRAM budget
    size_t vram_budget_mb;         // Max GPU memory to use (MiB), 0 = use all available

    // Threading
    int num_cpu_threads;           // 0 = auto (hardware_concurrency)
} ImpConfig;

// Returns a config with sensible defaults
ImpConfig imp_config_default(void);

#ifdef __cplusplus
}
#endif
