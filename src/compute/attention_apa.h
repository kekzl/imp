#pragma once

#include <cuda_fp16.h>
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

// Same, reading the paged FP16 cache directly (no gathered K/V): keys [0, tail) through block_table
// (block_size slots of [nkv][hd]), keys [tail, kv_len) from k_tail / v_tail [kv_len - tail, nkv*hd]; tiles of
// earlier chunks come from layer kv_layer's cache when owner (request id, >= 0) quantized them; owner -1
// runs uncached, another owner restarts the cache.
[[nodiscard]] bool attention_apa_prefill_paged(const Tensor& q, const half* k_pool, const half* v_pool,
                                               const int* block_table, int block_size, const Tensor& k_tail,
                                               const Tensor& v_tail, int tail, Tensor& o, int n, int kv_len,
                                               int nh, int nkv, int hd, float scale, int q_offset, float eps,
                                               int kv_layer, int owner, cudaStream_t stream);

// Planned T2 demand (ExecT2Demand::apa_scratch), set at engine init: the first call takes this many bytes
// once, larger needs decline. 0 = unplanned (take what the call needs, grow on demand).
void attention_apa_set_workspace_bound(size_t bytes);

// Per-layer tile caches for chunked prefill (attention_apa_prefill_paged): n_layers x
// apa::kv_state_bytes(1, nkv, 128, cap_tokens) from the T2 arena on first use; cap_tokens 0 = off. Engine
// init.
void attention_apa_set_kv_states(int n_layers, int nkv, int cap_tokens);

// One tiny flat and paged call (and incremental with the tile cache) at engine warmup: the first launch
// loads the kernels and takes the planned workspace, 55 ms on Qwen3-8B, which warmup prompts below
// attention.apa_min_kv would leave to the first long request. No-op unless hd 128 (hd 0 = d_model / nh) and
// eps > 0; buffers come from the end of the planned workspace slab.
void attention_apa_warmup(int nh, int nkv, int hd, int d_model, float eps, cudaStream_t stream);

// Pre-cudaDeviceReset hook: drops the arena scratch pointer.
void attention_apa_reset_static_cuda_state();

}  // namespace imp
