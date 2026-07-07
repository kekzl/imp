#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    IMP_DTYPE_FP32 = 0,
    IMP_DTYPE_FP16 = 1,
    IMP_DTYPE_BF16 = 2,
    IMP_DTYPE_FP8_E4M3 = 3,
    IMP_DTYPE_FP8_E5M2 = 4,
    IMP_DTYPE_INT8 = 5,
    IMP_DTYPE_INT4 = 6,
    IMP_DTYPE_INT32 = 7,
    IMP_DTYPE_FP4_E2M1 = 8,
    // Values 9 and 10 (TURBOQUANT / TURBOQUANT_LITE aliases) removed 2026-07-07
    // (TurboQuant retired Phase 5, 2026-05-17; the aliases had no callers).
    IMP_DTYPE_NVFP4 = 11,            // NVFP4 KV cache: packed FP4 + UE4M3 per-group_of_16 scales
    IMP_DTYPE_MXFP4_KV = 12,        // MXFP4-KV cache: packed FP4 + UE8M0 per-group_of_16 scales (Path A retirement target)
} ImpDType;

typedef enum {
    IMP_ARCH_LLAMA = 0,
    IMP_ARCH_MISTRAL = 1,
    IMP_ARCH_MIXTRAL = 2,
    IMP_ARCH_DEEPSEEK = 3,
    IMP_ARCH_NEMOTRON_H_MOE = 4,
    IMP_ARCH_QWEN3 = 5,
    IMP_ARCH_QWEN3_MOE = 6,
    IMP_ARCH_GEMMA3 = 7,
    IMP_ARCH_LLAMA4 = 8,
    IMP_ARCH_GENERIC = 9,
    IMP_ARCH_QWEN35 = 10,
    IMP_ARCH_QWEN35_MOE = 11,
    IMP_ARCH_GEMMA4 = 12,
    IMP_ARCH_QWEN36_MOE = 13,
    IMP_ARCH_GPT_OSS = 14,
    IMP_ARCH_NOMIC_BERT = 15,
} ImpModelArch;

typedef enum {
    IMP_QUANT_NONE = 0,
    IMP_QUANT_Q4_0 = 1,
    IMP_QUANT_Q4_K_M = 2,
    IMP_QUANT_Q8_0 = 3,
    IMP_QUANT_FP8 = 4,
    IMP_QUANT_FP8_E4M3 = 5,
    IMP_QUANT_NVFP4 = 6,
} ImpQuantType;

typedef enum {
    IMP_FORMAT_GGUF = 0,
    IMP_FORMAT_SAFETENSORS = 1,
} ImpModelFormat;

#ifdef __cplusplus
}
#endif
