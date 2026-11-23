// Inkling2-Loop expert projections of a layer application (inkling_forward.h): which kernel serves
// which call size.

#include "exec/inkling_forward.h"

#include "compute/gemm_grouped_nvfp4_f16_tile.h"
#include "compute/inkling_ops.h"
#include "core/logging.h"
#include "quant/nvfp4_gemm.h"

namespace imp {

using namespace inkling;

bool InklingForward::moe_rows_(const InklingLayer& ly, int T, cudaStream_t s) {
    const InklingConfig& c = w_->cfg;
    const InklingExperts& ex = ly.ex;
    const int slots = T * c.top_k, H = c.hidden;
    if (H % 8 != 0 || (ex.fp8 && (!ex.w13_bias || !ex.w2_bias)) ||
        !moe_sort(idx_, slots, c.n_experts, perm_, off_, s))
        return false;
    moe_rows_copy(xm_, perm_, xs_, slots, H, c.top_k, true, s);
    const size_t n13 = static_cast<size_t>(2) * c.inter * H, n2 = static_cast<size_t>(H) * c.inter;
    GroupedNvfp4F16Args g;
    g.a = xs_, g.rows = slots, g.N = 2 * c.inter, g.K = H, g.ne = c.n_experts, g.d_offsets = off_;
    g.d = gu_;
    if (ex.fp8) {
        g.w = ex.w13_q, g.w_stride = n13, g.sc = ex.w13_s, g.sc_stride = n13 / 32;
        g.d_tensor_scales = ex.w13_ts;
    } else {
        g.w = ex.w13.packed_data, g.w_stride = ex.w13.expert_stride_packed;
        g.sc = ex.w13.micro_scales, g.sc_stride = ex.w13.expert_stride_ms;
        g.d_tensor_scales = ex.w13.tensor_scales;
    }
    if (!(ex.fp8 ? gemm_grouped_mxfp8_f16_tile(g, ex.w13_bias, s) : gemm_grouped_nvfp4_f16_tile(g, s)))
        return false;
    expert_act(gu_, act_, slots, c.inter, c.gate_cap, c.up_cap, s);
    g.a = act_, g.N = H, g.K = c.inter, g.d = xs_;
    if (ex.fp8) {
        g.w = ex.w2_q, g.w_stride = n2, g.sc = ex.w2_s, g.sc_stride = n2 / 32;
        g.d_tensor_scales = ex.w2_ts;
    } else {
        g.w = ex.w2.packed_data, g.w_stride = ex.w2.expert_stride_packed;
        g.sc = ex.w2.micro_scales, g.sc_stride = ex.w2.expert_stride_ms;
        g.d_tensor_scales = ex.w2.tensor_scales;
    }
    if (!(ex.fp8 ? gemm_grouped_mxfp8_f16_tile(g, ex.w2_bias, s) : gemm_grouped_nvfp4_f16_tile(g, s))) {
        IMP_LOG_ERROR("inkling: grouped expert GEMM refused the down projection (%d x %d)", H, c.inter);
        ok_ = false;
    }
    moe_rows_copy(xs_, perm_, ey_, slots, H, 1, false, s);
    return true;
}

bool InklingForward::moe_small_(const InklingLayer& ly, int T, cudaStream_t s) {
    const InklingConfig& c = w_->cfg;
    const InklingExperts& ex = ly.ex;
    const int slots = T * c.top_k, H = c.hidden;
    if (!ex.fp8 || !ex.w13_bias || !ex.w2_bias || !moe_group(idx_, slots, grp_, s))
        return false;
    const size_t n13 = static_cast<size_t>(2) * c.inter * H, n2 = static_cast<size_t>(H) * c.inter;
    MoeSmallArgs m;
    m.groups = grp_, m.slots = slots, m.fp8 = true;
    m.x = xm_, m.x_group = c.top_k, m.y = gu_, m.N = 2 * c.inter, m.K = H;
    m.w = ex.w13_q, m.w_stride = n13, m.sc = ex.w13_s, m.sc_stride = n13 / 32, m.ts = ex.w13_ts,
    m.bias = ex.w13_bias;
    if (!moe_small(m, s))
        return false;
    expert_act(gu_, act_, slots, c.inter, c.gate_cap, c.up_cap, s);
    m.x = act_, m.x_group = 1, m.y = ey_, m.N = H, m.K = c.inter;
    m.w = ex.w2_q, m.w_stride = n2, m.sc = ex.w2_s, m.sc_stride = n2 / 32, m.ts = ex.w2_ts,
    m.bias = ex.w2_bias;
    if (!moe_small(m, s)) {
        IMP_LOG_ERROR("inkling: small-tile expert GEMM refused the down projection (%d x %d)", H, c.inter);
        ok_ = false;
    }
    return true;
}

// xm_ -> ey_ for the slots idx_.
void InklingForward::experts_(const InklingLayer& ly, int T, cudaStream_t s) {
    const InklingConfig& c = w_->cfg;
    const int slots = T * c.top_k, H = c.hidden;
    // 2..8 rows, MXFP8 layers: one tensor-core tile per chosen expert (64 slots of 29 experts: 155 us
    // against 350 for the slot GEMV; on NVFP4 the slot GEMV wins, 87 against 147). Above: grouped GEMMs.
    if (T > 1 && T <= kGraphRows && ly.ex.fp8 && grouped_ && moe_small_(ly, T, s))
        return;
    if (T > kGraphRows && grouped_ && moe_rows_(ly, T, s))
        return;
    if (ly.ex.fp8) {
        gemv_mxfp8_slots(ly.ex.w13_q, ly.ex.w13_s, idx_, xm_, gu_, slots, 2 * c.inter, H, c.top_k, s);
        expert_act(gu_, act_, slots, c.inter, c.gate_cap, c.up_cap, s);
        gemv_mxfp8_slots(ly.ex.w2_q, ly.ex.w2_s, idx_, act_, ey_, slots, H, c.inter, 1, s);
        return;
    }
    gemv_nvfp4_moe_decode(ly.ex.w13, idx_, xm_, gu_, 2 * c.inter, H, H, slots, s, c.top_k);
    expert_act(gu_, act_, slots, c.inter, c.gate_cap, c.up_cap, s);
    gemv_nvfp4_moe_decode(ly.ex.w2, idx_, act_, ey_, H, c.inter, c.inter, slots, s, 1);
}

}  // namespace imp
