#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    IMP_SUCCESS              =  0,
    IMP_ERROR_INVALID_ARG    = -1,
    IMP_ERROR_OUT_OF_MEMORY  = -2,
    IMP_ERROR_CUDA           = -3,
    IMP_ERROR_FILE_NOT_FOUND = -4,
    IMP_ERROR_INVALID_MODEL  = -5,
    IMP_ERROR_UNSUPPORTED    = -6,
    IMP_ERROR_INTERNAL       = -7,
    IMP_ERROR_CANCELLED      = -8,
} ImpError;

const char* imp_error_string(ImpError err);

#ifdef __cplusplus
}
#endif
