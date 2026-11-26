// Inkling2-Loop attention of a layer application (inkling_forward.h): the full-attention applications
// read and write their K/V through a block table (B-111: the engine's paged pool, or an identity table
// over K/V of the forward's own), the sliding ones a ring; the LoRA deltas of its q / k / v / o.

#include "exec/inkling_forward.h"

#include "compute/inkling_ops.h"
#include "compute/lora_rows.h"
#include "core/logging.h"
#include "memory/kv_cache.h"

#include <algorithm>

namespace imp {

using namespace inkling;

namespace {

__global__ void iota_k(int* d, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n)
        d[i] = i;
}

}  // namespace

bool InklingForward::init_kv_(bool own) {
    const InklingConfig& c = w_->cfg;
    n_blocks_ = (max_ctx_ + kInklingKvAhead + kKVBlockSize - 1) / kKVBlockSize;
    blocks_ = static_cast<int*>(alloc_(static_cast<size_t>(n_blocks_) * sizeof(int)));
    if (!blocks_)
        return false;
    kv_bound_ = own;
    table_rows_ = own ? n_blocks_ * kKVBlockSize : 0;
    if (!own)  // a run() before set_table() reads block 0, never past the pool
        return cudaMemsetAsync(blocks_, 0, static_cast<size_t>(n_blocks_) * sizeof(int), nullptr) ==
               cudaSuccess;
    iota_k<<<(n_blocks_ + 255) / 256, 256>>>(blocks_, n_blocks_);
    IMP_CUDA_CHECK_LAUNCH();
    const size_t rows = static_cast<size_t>(n_blocks_) * kKVBlockSize,
                 KD = static_cast<size_t>(c.n_kv_heads) * c.head_dim;
    for (int a = 0; a < c.n_apps() && ok_; ++a)
        if (c.global[static_cast<size_t>(c.app_layer(a))]) {
            k_[static_cast<size_t>(a)] = Hf_(rows * KD);
            v_[static_cast<size_t>(a)] = Hf_(rows * KD);
        }
    return ok_ && cudaPeekAtLastError() == cudaSuccess;
}

bool InklingForward::bind_kv(std::span<half* const> k, std::span<half* const> v) {
    const InklingKvLayout l = inkling_kv_apps(w_->cfg, false);
    if (kv_bound_ || static_cast<int>(k.size()) < l.n_global || static_cast<int>(v.size()) < l.n_global) {
        IMP_LOG_ERROR("inkling: K/V of %d full-attention applications, %zu pool layers given%s", l.n_global,
                      k.size(), kv_bound_ ? " (already bound)" : "");
        return false;
    }
    for (size_t a = 0; a < l.app_layer.size(); ++a)
        if (const int p = l.app_layer[a]; p >= 0) {
            k_[a] = k[static_cast<size_t>(p)];
            v_[a] = v[static_cast<size_t>(p)];
            if (!k_[a] || !v_[a])
                return false;
        }
    kv_bound_ = true;
    return true;
}

void InklingForward::set_table(const int* d_table, int n, cudaStream_t s) {
    const int m = std::min(n, n_blocks_);
    table_rows_ = d_table && m > 0 ? m * kKVBlockSize : 0;
    if (d_table && m > 0)
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(blocks_, d_table, static_cast<size_t>(m) * sizeof(int),
                                           cudaMemcpyDeviceToDevice, s));
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
    a.blocks = global ? blocks_ : nullptr, a.kv_block = kKVBlockSize;
    a.scale = 1.0f / sqrtf(static_cast<float>(c.head_dim)), a.q_thr = c.qk_clip, a.rel_thr = c.rel_clip;
    a.log_alpha = global ? c.log_alpha : 0.0f, a.log_floor = c.log_floor;
    attention(a, s);
}

void InklingForward::delta_(int layer, LoraProj p, int plane, const half* x, int ldx, half* y, int ldy, int T,
                            cudaStream_t s) {
    const LoraWeights* lw = lora_ && lora_->a ? lora_->a->get(layer, p) : nullptr;
    if (!lw)
        return;
    LoraRowsArgs a;
    a.A = static_cast<const half*>(lw->A), a.B = static_cast<const half*>(lw->B);
    a.r = lw->r, a.K = lw->K, a.N = lw->N, a.scale = lora_->a->scale();
    a.x = x, a.ldx = ldx, a.y = y, a.ldy = ldy, a.rows = T;
    a.t = lora_->t[plane];
    lora_delta_rows(a, s);
}

}  // namespace imp
