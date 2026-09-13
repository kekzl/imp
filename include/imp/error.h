#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    IMP_SUCCESS = 0,
    IMP_ERROR_INVALID_ARG = -1,
    IMP_ERROR_OUT_OF_MEMORY = -2,
    IMP_ERROR_CUDA = -3,
    IMP_ERROR_FILE_NOT_FOUND = -4,
    IMP_ERROR_INVALID_MODEL = -5,
    IMP_ERROR_UNSUPPORTED = -6,
    IMP_ERROR_INTERNAL = -7,
    IMP_ERROR_CANCELLED = -8,
    // Engine refused: KV pool is smaller than the prompt needs, no eviction can fix it.
    // Distinct from CANCELLED (client went away / internal abort): the one cancellation a caller
    // can act on (shorten prompt, more VRAM). Invariant I6: OOM is typed and recoverable.
    IMP_ERROR_CAPACITY = -9,
} ImpError;

const char* imp_error_string(ImpError err);

#ifdef __cplusplus
}
#endif
