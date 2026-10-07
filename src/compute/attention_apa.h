#pragma once

#include <cuda_runtime.h>

#include "core/tensor.h"

namespace imp {

// APA prefill (attention.apa_eps > 0, opt-in): pass 1 all-FP4, KV tiles whose share of the running row sum
// exceeds eps go to an exact FP16 pass 2 (log-sum-exp merge); one launch. q [n, nh*hd], k/v [kv_len, nkv*hd]
// FP16, o [n, nh*hd]; causal, Q row 0 at q_offset, kv_len = q_offset + n. hd 128, nh % nkv == 0.
// false = declined (shape or scratch).
[[nodiscard]] bool attention_apa_prefill(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& o, int n,
                                         int kv_len, int nh, int nkv, int hd, float scale, int q_offset,
                                         float eps, cudaStream_t stream);

// Planned T2 demand (ExecT2Demand::apa_scratch), set at engine init: the first call takes this many bytes
// once, larger needs decline. 0 = unplanned (take what the call needs, grow on demand).
void attention_apa_set_workspace_bound(size_t bytes);

// Pre-cudaDeviceReset hook: drops the arena scratch pointer.
void attention_apa_reset_static_cuda_state();

}  // namespace imp
