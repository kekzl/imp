#include "exec/moe_workspace.h"
#include "compute/gemm.h"         // moe_imma_prefill_supported
#include "compute/mmq_q8_imma.h"  // mmq_imma_moe_shape_ok
#include "core/logging.h"
#include "model/model.h"
#include <cuda_runtime.h>
#include <utility>

namespace imp {

MoeExpertServe classify_moe_experts(const Model& model, bool moe_imma_prefill, QType compute_dtype) {
    MoeExpertServe s{true, moe_imma_prefill && compute_dtype == QType::F16, false};
    bool any_f16 = false, all_f16 = true;
    for (int i = 0; i < model.config().n_layers; i++) {
        const auto& L = model.layer(i);
        for (const Tensor* t : {&L.expert_gate_packed, &L.expert_up_packed, &L.expert_down_packed}) {
            if (!t->data)
                continue;
            const bool f16 = t->qtype == QType::F16 && t->on_device;
            any_f16 = any_f16 || f16;
            all_f16 = all_f16 && f16;
        }
    }
    s.f16_all = any_f16 && all_f16;
    auto fp4 = [](const Tensor& t) { return !t.data || t.qtype == QType::NVFP4 || t.qtype == QType::MXFP4; };
    // Packed experts are [ne][N][K].
    auto imma = [](const Tensor& t) {
        if (!t.data)
            return true;
        if (!t.on_device || !moe_imma_prefill_supported(t.qtype))
            return false;
        const int qkind = moe_imma_prefill_qkind(t.qtype);
        return qkind != 0 &&
               mmq_imma_moe_shape_ok(qkind, static_cast<int>(t.shape[1]), static_cast<int>(t.shape[2]));
    };
    for (int i = 0; i < model.config().n_layers && (s.cutlass_fp4 || s.imma_all); i++) {
        const auto& L = model.layer(i);
        s.cutlass_fp4 = s.cutlass_fp4 && fp4(L.expert_up_packed) && fp4(L.expert_down_packed);
        s.imma_all = s.imma_all && imma(L.expert_gate_packed) && imma(L.expert_up_packed) &&
                     imma(L.expert_down_packed);
    }
    s.imma_all = s.imma_all && !s.cutlass_fp4;
    return s;
}

// Whole-struct reset: owners release through their own family, T2 arena tenants (pointer
// and scale arrays, closed by ~Engine) and views are nulled. No buffer's release depends on
// another's. A member added later is released here unless it is carried over below.
void MoEWorkspace::free() {
    MoEWorkspace next;
    next.expert_hist = std::move(expert_hist);
    next.hist_layers = hist_layers;
    next.hist_experts = hist_experts;
    next.hist_top_k = hist_top_k;
    next.imb_acc = std::move(imb_acc);
    next.imb_layers = imb_layers;
    next.imb_experts = imb_experts;
    next.expert_trace = std::move(expert_trace);
    next.trace_weights = trace_weights;
    next.trace_cursor = std::move(trace_cursor);
    next.trace_capacity = trace_capacity;
    next.trace_top_k = trace_top_k;
    next.h_routing_readback = std::move(h_routing_readback);
    *this = std::move(next);
}

}  // namespace imp
