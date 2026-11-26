// Inkling2-Loop's dense projections as borrowed FP16 WeightHandles (exec/inkling_registry.h).

#include "exec/inkling_registry.h"

#include "model/inkling_weights.h"

namespace imp {

namespace {

TensorID borrow_fp16(WeightRegistry& reg, TensorKind kind, const half* w, int64_t rows, int64_t cols) {
    if (!w)
        return kInvalidTensorID;
    const TensorID id = reg.reserve(kind, rows, cols);
    WeightHandle& h = reg.handle(id);
    h.primary_tier = h.prefill_tier = h.decode_tier = StorageTier::FP16;
    h.payload.fp16.data = const_cast<half*>(w);
    h.source_data = w;
    h.source_qtype = QType::F16;
    return id;
}

}  // namespace

std::vector<InklingProjIds> register_inkling_weights(const InklingWeights& w, WeightRegistry& reg) {
    const InklingConfig& c = w.cfg;
    const int64_t H = c.hidden, QD = static_cast<int64_t>(c.n_heads) * c.head_dim;
    const int64_t KD = static_cast<int64_t>(c.n_kv_heads) * c.head_dim;
    std::vector<InklingProjIds> ids(w.layers.size());
    for (size_t i = 0; i < w.layers.size(); ++i) {
        const InklingLayer& ly = w.layers[i];
        const int64_t TD = c.global[i] ? static_cast<int64_t>(c.n_kv_heads) * 4 : 0;
        InklingProjIds& p = ids[i];
        p.qkv = borrow_fp16(reg, TensorKind::QKV_FUSED, ly.wq, ly.proj_rows, H);
        p.q = borrow_fp16(reg, TensorKind::WQ, ly.wq, inkling_q_rows(c), H);
        p.k = borrow_fp16(reg, TensorKind::WK, ly.wk, KD + TD, H);
        p.v = borrow_fp16(reg, TensorKind::WV, ly.wv, KD + TD, H);
        p.o = borrow_fp16(reg, TensorKind::WO, ly.wo, H, QD);
    }
    return ids;
}

}  // namespace imp
