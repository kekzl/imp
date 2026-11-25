// Inkling2-Loop attention of a layer application (inkling_forward.h): the full-attention applications
// read their K/V through a block table (B-111 S2a: an identity table over ctx_), the sliding ones a ring.

#include "exec/inkling_forward.h"

#include "compute/inkling_ops.h"
#include "core/logging.h"
#include "memory/kv_cache.h"

namespace imp {

using namespace inkling;

namespace {

__global__ void iota_k(int* d, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n)
        d[i] = i;
}

}  // namespace

bool InklingForward::init_ctx_blocks_(int max_ctx) {
    const int n = (max_ctx + kKVBlockSize - 1) / kKVBlockSize;
    ctx_blocks_ = static_cast<int*>(alloc_(static_cast<size_t>(n) * sizeof(int)));
    if (!ctx_blocks_)
        return false;
    iota_k<<<(n + 255) / 256, 256>>>(ctx_blocks_, n);
    IMP_CUDA_CHECK_LAUNCH();
    return cudaPeekAtLastError() == cudaSuccess;
}

void InklingForward::attend_(const InklingLayer& ly, size_t app, bool global, int T, int q_stride,
                             cudaStream_t s) {
    const InklingConfig& c = w_->cfg;
    const int QD = c.n_heads * c.head_dim, RD = c.n_heads * c.d_rel;
    AttnArgs a;
    a.q = proj_h_, a.rel = proj_h_ + QD, a.sink = proj_h_ + QD + RD, a.gate = proj_h_ + QD + RD + c.n_heads;
    a.q_scale = ly.q_scale, a.rel_scale = ly.rel_scale, a.proj = ly.rel_proj;
    a.k = k_[app], a.v = v_[app], a.out = ao_h_, a.d_pos = d_pos_;
    a.part = part_, a.max_ctx = max_ctx_;
    a.work = attn_gemm_ ? attn_work_ : nullptr, a.work_bytes = attn_work_bytes_;
    a.pos0 = T > kGraphRows ? pos_ : -1;
    a.T = T, a.stride = q_stride, a.n_heads = c.n_heads, a.n_kv = c.n_kv_heads, a.hd = c.head_dim;
    a.d_rel = c.d_rel, a.rel_extent = global ? c.rel_extent : c.window;
    a.window = global ? 0 : c.window, a.ring = global ? 0 : ring_;
    a.blocks = global ? ctx_blocks_ : nullptr, a.kv_block = kKVBlockSize;
    a.scale = 1.0f / sqrtf(static_cast<float>(c.head_dim)), a.q_thr = c.qk_clip, a.rel_thr = c.rel_clip;
    a.log_alpha = global ? c.log_alpha : 0.0f, a.log_floor = c.log_floor;
    attention(a, s);
}

}  // namespace imp
