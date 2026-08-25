// Pre-dequant Phase 3e: Marlin W4A16 batched-decode sidecar (gemm.marlin).
// Repacks dense NVFP4 weights (the same set Phase 0b registered for the
// decode cache) into the vendored Marlin tile layout, largest first, until
// the VRAM budget runs out. The sidecar serves the M 2..32 decode GEMMs in
// executor_gemm_dispatch.cu; prefill / M=1 GEMV / spec-verify keep their
// paths, so this is a strict addition per covered weight.

#include "exec/executor.h"
#include "exec/quant_pipeline.h"
#include "exec/pre_dequant_internal.h"
#include "memory/vram_query.h"
#include "quant/marlin/marlin_w4a16.h"
#include "runtime/config.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

namespace imp {

void QuantPipeline::nvfp4_build_marlin_sidecar_(const ModelConfig& cfg, const VRAMBudget& budget,
                                                cudaStream_t stream) {
    const auto& rc = runtime_config();
    if (!rc.gemm.marlin)
        return;

    struct Candidate {
        const Tensor* w;
        size_t bytes;  // marlin data + scales for this weight
    };
    std::vector<Candidate> cands;
    auto consider = [&](const Tensor& w) {
        if (w.qtype != QType::NVFP4 || !w.data || !w.scales || !w.on_device)
            return;
        const int N = static_cast<int>(w.shape[0]);
        const int K = static_cast<int>(w.shape[1]) * 2;  // packed K/2 → logical K
        if (!marlin_w4a16::shape_supported(N, K))
            return;
        if (wcache_->marlin.count(w.data))
            return;
        cands.push_back({&w, (size_t)N * K / 2 + (size_t)N * K / 16});
    };
    Model* mut_model = const_cast<Model*>(model_);
    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& L = mut_model->layer(i);
        consider(L.wq);
        consider(L.wk);
        consider(L.wv);
        consider(L.wo);
        consider(L.w_gate);
        consider(L.w_up);
        consider(L.w_down);
        consider(L.w_gate_shared);
        consider(L.w_up_shared);
        consider(L.w_down_shared);
        consider(L.ssm_in);
        consider(L.ssm_out);
        consider(L.gdn_gate);
        // MoE expert weights are served by the grouped GEMM, not the dense
        // M<=32 dispatch — no Marlin entry for them.
    }
    if (cands.empty())
        return;

    // Budget = free VRAM minus everything still to be allocated AFTER this
    // phase — the batch-shaped SSM/GDN state, the KV pool, and the reserve
    // floor (library reserve + safety live inside it). Two rejected designs:
    // subtracting only the floor starved the SSM allocation outright
    // (rejected 4848 MiB on the 27B at batch 32), and letting an explicit
    // marlin_budget_mb OVERRIDE this bound pushed the SSM state into the raw
    // cudaMalloc fallback — which on WSL2/WDDM "succeeds" into host memory
    // and lands on the 6.5x bandwidth cliff. marlin_budget_mb is therefore a
    // CAP, never an override; the operator makes room by lowering
    // max_batch_size or max_seq_len (both shrink the downstream charge).
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    size_t free_mem = 0, total_mem = 0;
    vram_budget_mem_get_info(&free_mem, &total_mem);
    // Floor: the allocator's own headroom (5% of total; vram_allocator.cu
    // enforces the same figure) plus safety. NOT budget.reserve_bytes — that
    // carries the library reserve, which by phase 3e is already inside
    // "used" (cuBLAS/CUTLASS initialize during weight upload), and charging
    // it again left 3.4 GiB idle on the mbs=24 measurement.
    const size_t floor = vram_reserve_floor(total_mem, 5) + (256ULL << 20);
    // Operator KV pin shrinks the pool below the plan's figure; charge the
    // smaller of the two (per-block cost from the plan's own numbers).
    size_t kv_charge = budget.kv_cache_bytes;
    if (rc.kv_cache.max_blocks > 0 && budget.kv_max_blocks > 0) {
        const size_t per_block = budget.kv_cache_bytes / budget.kv_max_blocks;
        kv_charge = std::min(kv_charge, (size_t)rc.kv_cache.max_blocks * per_block);
    }
    const size_t downstream = budget.ssm_footprint_bytes + kv_charge + floor;
    size_t marlin_budget = free_mem > downstream ? free_mem - downstream : 0;
    if (rc.gemm.marlin_budget_mb > 0)
        marlin_budget = std::min(marlin_budget, (size_t)rc.gemm.marlin_budget_mb * 1024 * 1024);

    // Largest first: decode GEMM time is proportional to bytes streamed, so
    // budget buys the most win on the biggest weights.
    std::stable_sort(cands.begin(), cands.end(),
                     [](const Candidate& a, const Candidate& b) { return a.bytes > b.bytes; });

    int built = 0, skipped = 0;
    for (const auto& c : cands) {
        if (c.bytes + (4 << 20) > marlin_budget) {  // keep 4 MiB slack per entry
            skipped++;
            continue;
        }
        const Tensor& w = *c.w;
        const int N = static_cast<int>(w.shape[0]);
        const int K = static_cast<int>(w.shape[1]) * 2;
        marlin_w4a16::MarlinWeight mw;
        if (!marlin_w4a16::prepare(w.data, w.scales, w.tensor_scale, N, K, mw, stream)) {
            skipped++;
            continue;
        }
        wcache_->marlin[w.data] = mw;
        wcache_->marlin_bytes += c.bytes;
        marlin_budget -= c.bytes;
        built++;
    }
    if (built || skipped)
        IMP_LOG_INFO(
            "Marlin W4A16 sidecar: %d weights repacked (%.1f MiB), %d skipped "
            "(budget/shape), %.1f MiB budget left",
            built, wcache_->marlin_bytes / (1024.0 * 1024.0), skipped, marlin_budget / (1024.0 * 1024.0));
}

}  // namespace imp
