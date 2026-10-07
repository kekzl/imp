#include "compute/attention_ra2.h"

#include "compute/ra2/ra2.cuh"
#include "core/cuda_static_reset.h"
#include "core/logging.h"
#include "memory/engine_arena.h"

namespace imp {

// Grow-only T2 arena scratch (AUDIT B13 family): a bump arena never frees, so a grow hands a new
// slice and the old one stays valid inside a captured graph. Arena exhausted = decline (FA2 serves).
static uint8_t* s_ra2_ws = nullptr;
static size_t s_ra2_ws_cap = 0;
static uint64_t s_ra2_ws_gen = 0;

static uint8_t* ensure_ra2_workspace(size_t need) {
    const uint64_t gen = engine_arena().generation();
    if (gen != s_ra2_ws_gen) {  // model swap: re-take instead of trusting a stale capacity
        s_ra2_ws_gen = gen;
        s_ra2_ws = nullptr;
        s_ra2_ws_cap = 0;
    }
    if (s_ra2_ws != nullptr && need <= s_ra2_ws_cap)
        return s_ra2_ws;
    auto slab = engine_arena().take_bytes(need);
    if (slab.empty()) {
        IMP_LOG_WARN(
            "ra2 prefill: workspace %.2f MiB unavailable from the T2 arena (%.1f MiB free), declining",
            need / (1024.0 * 1024.0), engine_arena().remaining() / (1024.0 * 1024.0));
        return nullptr;
    }
    s_ra2_ws = reinterpret_cast<uint8_t*>(slab.data());
    s_ra2_ws_cap = need;
    return s_ra2_ws;
}

bool attention_ra2_prefill(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& o, int n, int kv_len,
                           int nh, int nkv, int hd, float scale, int q_offset, cudaStream_t stream) {
    if (q.qtype != QType::F16 || k.qtype != QType::F16 || v.qtype != QType::F16 || o.qtype != QType::F16)
        return false;
    const ra2::Problem p{1, n, kv_len, nh, nkv, hd, q_offset, /*causal=*/true, scale};
    ra2::KvSource kv{};
    kv.kind = ra2::KvKind::Flat;
    kv.k = k.data;
    kv.v = v.data;
    if (!ra2::supported(p, kv))
        return false;
    const size_t need = ra2::workspace_bytes(p);
    uint8_t* ws = ensure_ra2_workspace(need);
    if (ws == nullptr)
        return false;
    const bool ok = ra2::prefill(static_cast<const half*>(q.data), kv, static_cast<half*>(o.data), p, ws,
                                 need, stream);
    if (!ok)
        IMP_LOG_WARN("ra2 prefill: launch failed (%s), declining", cudaGetErrorString(cudaPeekAtLastError()));
    return ok;
}

void attention_ra2_reset_static_cuda_state() {
    s_ra2_ws = nullptr;
    s_ra2_ws_cap = 0;
}

namespace {
IMP_REGISTER_CUDA_STATIC_RESET(attention_ra2_reset_static_cuda_state);
}  // namespace

}  // namespace imp
