#include "compute/attention_apa.h"

#include "compute/apa/apa.cuh"
#include "core/cuda_static_reset.h"
#include "core/logging.h"
#include "memory/engine_arena.h"

#include <algorithm>
#include <vector>

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

// Per-layer KvStates (tiles cached across prefill chunks), one T2 arena slab of n_layers x stride taken
// on first use; set by attention_apa_set_kv_states at engine init. Unset or slab unavailable: no cache.
static int s_kvs_layers = 0, s_kvs_nkv = 0, s_kvs_cap = 0;
static std::vector<apa::KvState> s_kvs;
static uint64_t s_kvs_gen = 0;

void attention_apa_set_kv_states(int n_layers, int nkv, int cap_tokens) {
    s_kvs_layers = cap_tokens > 0 ? n_layers : 0;  // cap 0: tile cache not planned
    s_kvs_nkv = nkv;
    s_kvs_cap = cap_tokens;
    s_kvs.clear();
}

static apa::KvState* kv_state(int layer, int nkv, int kv_len) {
    const uint64_t gen = engine_arena().generation();
    if (gen != s_kvs_gen) {
        s_kvs_gen = gen;
        s_kvs.clear();
    }
    if (layer < 0 || layer >= s_kvs_layers || nkv != s_kvs_nkv || kv_len > s_kvs_cap)
        return nullptr;
    if (s_kvs.empty()) {
        const size_t stride = (apa::kv_state_bytes(1, nkv, 128, s_kvs_cap) + 255) & ~size_t(255);
        auto slab = engine_arena().take_bytes(stride * s_kvs_layers);
        if (slab.empty()) {
            IMP_LOG_WARN("apa prefill: tile cache %.1f MiB unavailable from the T2 arena, running without",
                         stride * s_kvs_layers / (1024.0 * 1024.0));
            s_kvs_layers = 0;
            return nullptr;
        }
        auto* base = reinterpret_cast<uint8_t*>(slab.data());
        for (int l = 0; l < s_kvs_layers; ++l)
            s_kvs.push_back(apa::kv_state_carve(base + l * stride, 1, nkv, 128, s_kvs_cap));
    }
    return &s_kvs[layer];
}

bool attention_apa_prefill_paged(const Tensor& q, const half* k_pool, const half* v_pool,
                                 const int* block_table, int block_size, const Tensor& k_tail,
                                 const Tensor& v_tail, int tail, Tensor& o, int n, int kv_len, int nh,
                                 int nkv, int hd, float scale, int q_offset, float eps, int kv_layer,
                                 cudaStream_t stream) {
    if (q.qtype != QType::F16 || k_tail.qtype != QType::F16 || v_tail.qtype != QType::F16 ||
        o.qtype != QType::F16 || hd != 128 || eps <= 0.0f || block_table == nullptr)
        return false;
    const apa::Problem p{1, n, kv_len, nh, nkv, hd, q_offset, /*causal=*/true, scale};
    const auto* qh = static_cast<const half*>(q.data);
    auto* oh = static_cast<half*>(o.data);
    const auto* kt = static_cast<const half*>(k_tail.data);
    const auto* vt = static_cast<const half*>(v_tail.data);
    apa::KvState* st = kv_state(kv_layer, nkv, kv_len);
    const size_t need = apa::workspace_bytes(p, /*own_kv=*/st == nullptr);
    uint8_t* ws = ensure_apa_workspace(need);
    if (ws == nullptr)
        return false;
    const bool ok = st != nullptr ? apa::prefill_incremental(qh,
                                                             apa::paged_reader(k_pool, v_pool, block_table,
                                                                               block_size, kt, vt, tail),
                                                             oh, p, eps, *st, ws, need, stream)
                                  : apa::prefill_paged(qh, k_pool, v_pool, block_table, block_size, kt, vt,
                                                       tail, oh, p, eps, ws, need, stream);
    if (!ok)
        IMP_LOG_WARN("apa prefill (paged): launch failed (%s), declining",
                     cudaGetErrorString(cudaPeekAtLastError()));
    return ok;
}

void attention_apa_reset_static_cuda_state() {
    s_apa_ws = nullptr;
    s_apa_ws_cap = 0;
    s_kvs.clear();
}

namespace {
IMP_REGISTER_CUDA_STATIC_RESET(attention_apa_reset_static_cuda_state);
}  // namespace

}  // namespace imp
