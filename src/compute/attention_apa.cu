#include "compute/attention_apa.h"

#include "compute/ra2/apa/apa.cuh"
#include "core/cuda_static_reset.h"
#include "core/logging.h"
#include "memory/engine_arena.h"

#include <algorithm>

namespace imp {

// T2 arena scratch (AUDIT B13 family), taken once at the planned bound (exec_t2_demand apa_scratch);
// unplanned it grows, and a bump arena keeps each old slice valid inside a captured graph. Arena
// exhausted or need over the plan = decline (ra2 / FA2 serve).
static uint8_t* s_apa_ws = nullptr;
static size_t s_apa_ws_cap = 0;
static uint64_t s_apa_ws_gen = 0;
static size_t s_apa_ws_bound = 0;  // planned demand, attention_apa_set_workspace_bound

void attention_apa_set_workspace_bound(size_t bytes) { s_apa_ws_bound = bytes; }

static uint8_t* ensure_apa_workspace(size_t need) {
    const uint64_t gen = engine_arena().generation();
    if (gen != s_apa_ws_gen) {  // model swap: re-take instead of trusting a stale capacity
        s_apa_ws_gen = gen;
        s_apa_ws = nullptr;
        s_apa_ws_cap = 0;
    }
    if (s_apa_ws != nullptr && need <= s_apa_ws_cap)
        return s_apa_ws;
    if (s_apa_ws != nullptr && s_apa_ws_bound > 0)  // planned slab taken: no second take outside the plan
        return nullptr;
    const size_t take = std::max(need, s_apa_ws_bound);
    auto slab = engine_arena().take_bytes(take);
    if (slab.empty()) {
        IMP_LOG_WARN(
            "apa prefill: workspace %.2f MiB unavailable from the T2 arena (%.1f MiB free), declining",
            take / (1024.0 * 1024.0), engine_arena().remaining() / (1024.0 * 1024.0));
        return nullptr;
    }
    s_apa_ws = reinterpret_cast<uint8_t*>(slab.data());
    s_apa_ws_cap = take;
    return s_apa_ws;
}

bool attention_apa_prefill(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& o, int n, int kv_len,
                           int nh, int nkv, int hd, float scale, int q_offset, float eps,
                           cudaStream_t stream) {
    if (q.qtype != QType::F16 || k.qtype != QType::F16 || v.qtype != QType::F16 || o.qtype != QType::F16 ||
        hd != 128 || eps <= 0.0f)
        return false;
    const apa::Problem p{1, n, kv_len, nh, nkv, hd, q_offset, /*causal=*/true, scale};
    apa::KvSource kv{};
    kv.k = k.data;
    kv.v = v.data;
    if (!apa::supported(p, kv))
        return false;
    const size_t need = apa::workspace_bytes(p);
    uint8_t* ws = ensure_apa_workspace(need);
    if (ws == nullptr)
        return false;
    const bool ok = apa::prefill(static_cast<const half*>(q.data), static_cast<const half*>(k.data),
                                 static_cast<const half*>(v.data), static_cast<half*>(o.data), p, eps, ws,
                                 need, stream);
    if (!ok)
        IMP_LOG_WARN("apa prefill: launch failed (%s), declining", cudaGetErrorString(cudaPeekAtLastError()));
    return ok;
}

void attention_apa_reset_static_cuda_state() {
    s_apa_ws = nullptr;
    s_apa_ws_cap = 0;
}

namespace {
IMP_REGISTER_CUDA_STATIC_RESET(attention_apa_reset_static_cuda_state);
}  // namespace

}  // namespace imp
