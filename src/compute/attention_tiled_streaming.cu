#include "compute/attention_tiled_streaming.h"
#include "core/logging.h"
#include <cuda_runtime.h>

namespace imp {

bool attention_tiled_streaming_prefill(const Tensor& Q, const Tensor& K,
                                       const Tensor& V, Tensor& O, float scale,
                                       bool causal, int sliding_window,
                                       float softcap, int q_offset,
                                       cudaStream_t stream) {
    // Stub: returns false so the dispatcher falls back to cuBLAS.
    // Real implementation lands in subsequent tasks.
    (void)Q; (void)K; (void)V; (void)O; (void)scale; (void)causal;
    (void)sliding_window; (void)softcap; (void)q_offset; (void)stream;
    return false;
}

}  // namespace imp
