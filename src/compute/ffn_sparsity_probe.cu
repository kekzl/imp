#include "compute/ffn_sparsity_probe.h"
#include "runtime/process_diag.h"
#include "core/logging.h"
#include "memory/engine_arena.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <vector>

namespace imp {
namespace {

constexpr int kMaxLayers = 256;
constexpr int kNumThresholds = 5;
constexpr int kSlotsPerLayer = kNumThresholds + 1;  // [under_t0..under_t4, total]

// Held in __constant__ memory so the device side sees identical values.
__device__ __constant__ float d_thresholds[kNumThresholds] = {0.005f, 0.01f, 0.02f, 0.05f, 0.1f};
const float h_thresholds[kNumThresholds] = {0.005f, 0.01f, 0.02f, 0.05f, 0.1f};

struct ProbeState {
    std::mutex mu;
    bool initialized = false;
    bool enabled = false;
    unsigned long long* d_counters = nullptr;  // [kMaxLayers * kSlotsPerLayer]
    int max_layer_seen = -1;
};

ProbeState g_state;

void ensure_init_locked() {
    if (g_state.initialized) return;
    g_state.enabled = process_diag_ffn_sparsity_probe();
    if (g_state.enabled) {
        const size_t bytes = sizeof(unsigned long long) * kMaxLayers * kSlotsPerLayer;
        // T2 (engine-persistent): fixed size, allocated once, never freed. The
        // arena is the tier for that, and it takes this file off the I1
        // allowlist. Direct allocation only when the arena is closed.
        // T2 (engine-persistent) and NO direct-allocation fallback on purpose.
        // Fixed size, allocated once, never freed — the arena is exactly that
        // tier. Keeping a cudaMalloc fallback would leave this file on the I1
        // allowlist for a path that only runs when the arena is closed, which
        // for a diagnostic probe means "not in an engine, so nothing to probe"
        // (AUDIT B34: a fallback keeps the site even when the site never runs).
        auto slab = engine_arena().take_bytes(bytes);
        if (slab.empty()) {
            IMP_LOG_WARN("ffn-sparsity-probe: T2 arena unavailable — probe disabled");
            g_state.d_counters = nullptr;
            g_state.enabled = false;
        } else {
            g_state.d_counters = reinterpret_cast<unsigned long long*>(slab.data());
            cudaMemset(g_state.d_counters, 0, bytes);
        }
    }
    g_state.initialized = true;
}

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + __expf(-x));
}

__global__ void probe_silu_kernel(int layer, const __half* __restrict__ gate,
                                  const __half* __restrict__ up, int K,
                                  unsigned long long* __restrict__ counters) {
    unsigned long long* base = counters + static_cast<size_t>(layer) * kSlotsPerLayer;

    unsigned int local[kNumThresholds];
#pragma unroll
    for (int t = 0; t < kNumThresholds; ++t) local[t] = 0u;
    unsigned int total = 0u;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    for (int i = tid; i < K; i += nthreads) {
        const float g = __half2float(gate[i]);
        const float u = __half2float(up[i]);
        const float s = fabsf(silu_f(g) * u);
        total += 1u;
#pragma unroll
        for (int t = 0; t < kNumThresholds; ++t) {
            if (s < d_thresholds[t]) local[t] += 1u;
        }
    }

    const unsigned int lane = tid & 31;
#pragma unroll
    for (int t = 0; t < kNumThresholds; ++t) {
        unsigned int v = local[t];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
        if (lane == 0) atomicAdd(&base[t], static_cast<unsigned long long>(v));
    }
    unsigned int vt = total;
    for (int off = 16; off > 0; off >>= 1) vt += __shfl_xor_sync(0xFFFFFFFFu, vt, off);
    if (lane == 0) atomicAdd(&base[kNumThresholds], static_cast<unsigned long long>(vt));
}

}  // namespace

void probe_ffn_silu_sparsity(int layer, const __half* gate, const __half* up, int K,
                             cudaStream_t stream) {
    {
        std::lock_guard<std::mutex> lk(g_state.mu);
        ensure_init_locked();
        if (!g_state.enabled || g_state.d_counters == nullptr) return;
        if (layer < 0 || layer >= kMaxLayers) return;
        if (layer > g_state.max_layer_seen) g_state.max_layer_seen = layer;
    }
    if (gate == nullptr || up == nullptr || K <= 0) return;

    constexpr int kThreads = 256;
    probe_silu_kernel<<<1, kThreads, 0, stream>>>(layer, gate, up, K, g_state.d_counters);
    IMP_CUDA_CHECK_LAUNCH();
}

void flush_ffn_sparsity_probe_log() {
    std::lock_guard<std::mutex> lk(g_state.mu);
    if (!g_state.enabled || g_state.d_counters == nullptr || g_state.max_layer_seen < 0) return;

    const int n_layers = g_state.max_layer_seen + 1;
    const size_t bytes = sizeof(unsigned long long) * static_cast<size_t>(n_layers) * kSlotsPerLayer;
    std::vector<unsigned long long> host(static_cast<size_t>(n_layers) * kSlotsPerLayer, 0ull);
    cudaError_t e = cudaMemcpy(host.data(), g_state.d_counters, bytes, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        IMP_LOG_ERROR("ffn-sparsity-probe: D2H copy failed: %s", cudaGetErrorString(e));
        return;
    }

    unsigned long long model_under[kNumThresholds] = {0ull, 0ull, 0ull, 0ull, 0ull};
    unsigned long long model_total = 0ull;

    for (int L = 0; L < n_layers; ++L) {
        const unsigned long long* row = &host[static_cast<size_t>(L) * kSlotsPerLayer];
        const unsigned long long total = row[kNumThresholds];
        if (total == 0ull) continue;
        for (int t = 0; t < kNumThresholds; ++t) {
            const unsigned long long under = row[t];
            const double frac = static_cast<double>(under) / static_cast<double>(total);
            std::fprintf(stderr,
                         "[ffn-sparsity] layer=%3d thr=%.4f skip_frac=%.4f under=%llu total=%llu\n",
                         L, h_thresholds[t], frac, under, total);
            model_under[t] += under;
        }
        model_total += total;
    }

    if (model_total > 0ull) {
        std::fprintf(stderr, "[ffn-sparsity] MODEL n_layers=%d total_rows_seen=%llu\n",
                     n_layers, model_total);
        for (int t = 0; t < kNumThresholds; ++t) {
            const double frac = static_cast<double>(model_under[t]) / static_cast<double>(model_total);
            std::fprintf(stderr,
                         "[ffn-sparsity] MODEL thr=%.4f skip_frac=%.4f under=%llu total=%llu\n",
                         h_thresholds[t], frac, model_under[t], model_total);
        }
    }

    const size_t reset_bytes = sizeof(unsigned long long) * kMaxLayers * kSlotsPerLayer;
    cudaMemset(g_state.d_counters, 0, reset_bytes);
    g_state.max_layer_seen = -1;
}

}  // namespace imp
