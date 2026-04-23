#include "graph/executor.h"
#include "graph/executor_kernels.h"
#include "graph/executor_helpers.h"
#include "compute/gemm.h"
#include "compute/gemm_cutlass_sm120.h"
#include "quant/dequant_gpu.h"
#include "quant/fp8_quant.h"
#include "quant/nvfp4_gemm.h"
#include "core/logging.h"
#include "memory/vram_allocator.h"
#include "runtime/storage_planner.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <unordered_set>

namespace imp {

// ---------------------------------------------------------------------------
// Helpers to infer StorageTier from wcache_ maps and populate handles.
// ---------------------------------------------------------------------------
namespace {

// Infer StorageTier from which wcache_ map the source pointer landed in.
StorageTier infer_tier_from_wcache(const WeightCaches& wc, const void* src_ptr) {
    if (wc.cutlass_nvfp4.count(src_ptr)) return StorageTier::CUTLASS_NVFP4;
    if (wc.cutlass_mxfp4.count(src_ptr)) return StorageTier::MXFP4;
    if (wc.nvfp4.count(src_ptr))         return StorageTier::NVFP4;
    if (wc.fp8.count(src_ptr))           return StorageTier::FP8;
    if (wc.fp16.count(src_ptr))          return StorageTier::FP16;
    return StorageTier::Undefined;
}

// Fill a handle's payload by borrowing pointers from wcache_ entries.
void borrow_payload_from_wcache(WeightHandle& h, const WeightCaches& wc,
                                const void* src_ptr) {
    switch (h.primary_tier) {
        case StorageTier::FP16: {
            auto it = wc.fp16.find(src_ptr);
            if (it != wc.fp16.end()) {
                h.payload.fp16.data = static_cast<half*>(it->second.data);
            }
            break;
        }
        case StorageTier::FP8: {
            auto it = wc.fp8.find(src_ptr);
            if (it != wc.fp8.end()) {
                h.payload.fp8.data    = static_cast<__nv_fp8_e4m3*>(it->second.weight.data);
                h.payload.fp8.d_scale = it->second.d_scale;
            }
            break;
        }
        case StorageTier::NVFP4: {
            auto it = wc.nvfp4.find(src_ptr);
            if (it != wc.nvfp4.end()) {
                h.payload.nvfp4.data         = static_cast<uint8_t*>(it->second.packed_data);
                h.payload.nvfp4.block_scales = static_cast<uint8_t*>(it->second.micro_scales);
                h.payload.nvfp4.tensor_scale  = nullptr;  // host float only, no device ptr
                h.payload.nvfp4.tensor_scale_2 = nullptr;
            }
            break;
        }
        case StorageTier::CUTLASS_NVFP4: {
            auto it = wc.cutlass_nvfp4.find(src_ptr);
            if (it != wc.cutlass_nvfp4.end()) {
                h.payload.cutlass_nvfp4.weight       = const_cast<void*>(it->second.data);
                h.payload.cutlass_nvfp4.sf           = it->second.scale_factors;
                h.payload.cutlass_nvfp4.global_scale = const_cast<float*>(&it->second.tensor_scale);
            }
            break;
        }
        case StorageTier::MXFP4: {
            auto it = wc.cutlass_mxfp4.find(src_ptr);
            if (it != wc.cutlass_mxfp4.end()) {
                h.payload.mxfp4.weight        = const_cast<void*>(it->second.data);
                h.payload.mxfp4.scales        = it->second.scale_factors;
                h.payload.mxfp4.linear_scales = it->second.linear_scales;
                h.payload.mxfp4.hadamard_bs   = it->second.hadamard_bs;
            }
            break;
        }
        default: break;
    }
}

} // anonymous namespace

// Shared helpers from executor_helpers.h (vram_alloc, vram_free)

static inline void deduct_budget(size_t& budget, size_t amount) {
    budget = (budget > amount) ? (budget - amount) : 0;
}

static bool create_fused_weight_pair(
    const Tensor& w_a, const Tensor& w_b,
    const std::unordered_map<const void*, Tensor>& fp16_cache,
    VRAMAllocator* allocator,
    size_t& total_cache_bytes, size_t remaining_budget,
    cudaStream_t stream,
    std::unordered_map<int, Tensor>& out_map, int layer_idx,
    bool& should_stop)
{
    should_stop = false;
    if (!w_a.data || !w_b.data) return false;
    auto it_a = fp16_cache.find(w_a.data);
    auto it_b = fp16_cache.find(w_b.data);
    if (it_a == fp16_cache.end() || it_b == fp16_cache.end()) return false;

    int a_rows = static_cast<int>(w_a.shape[0]);
    int K = static_cast<int>(w_a.shape[1]);
    size_t one_sz = static_cast<size_t>(a_rows) * K * sizeof(half);

    if (total_cache_bytes + 2 * one_sz > remaining_budget) {
        should_stop = true;
        return false;
    }

    void* fused_buf = vram_alloc(allocator, 2 * one_sz, "fp16_weight_cache");
    if (!fused_buf) {
        should_stop = true;
        return false;
    }

    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(fused_buf, it_a->second.data, one_sz,
                     cudaMemcpyDeviceToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(static_cast<char*>(fused_buf) + one_sz,
                     it_b->second.data, one_sz,
                     cudaMemcpyDeviceToDevice, stream));

    int64_t shape[2] = {2 * a_rows, static_cast<int64_t>(K)};
    out_map[layer_idx] = Tensor(fused_buf, DType::FP16, 2, shape, true);
    total_cache_bytes += 2 * one_sz;
    return true;
}

template <typename Fn>
static void for_each_dense_weight(const Model& model, const ModelConfig& cfg, Fn&& fn) {
    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& L = model.layer(i);
        fn(L.wq, L.wq_qtype);
        fn(L.wk, L.wk_qtype);
        fn(L.wv, L.wv_qtype);
        fn(L.wo, L.wo_qtype);
    }
    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& L = model.layer(i);
        fn(L.ssm_in, L.ssm_in_qtype);
        fn(L.ssm_out, L.ssm_out_qtype);
        fn(L.w_gate_shared, L.w_gate_shared_qtype);
        fn(L.w_up_shared, L.w_up_shared_qtype);
        fn(L.w_down_shared, L.w_down_shared_qtype);
        fn(L.w_gate, L.w_gate_qtype);
        fn(L.w_up, L.w_up_qtype);
        fn(L.w_down, L.w_down_qtype);
    }
}

// ---------------------------------------------------------------------------
// Pre-dequantize quantized weights to FP16 on GPU
// ---------------------------------------------------------------------------

void GraphExecutor::pre_dequant_weights(cudaStream_t stream, const VRAMBudget& budget) {
    if (!initialized_ || !model_) return;
    // Skip all weight caching for debugging numerical precision issues

    const auto& cfg = model_->config();
    size_t total_cache_bytes = 0;
    int cached_count = 0;
    bool budget_exhausted = false;

    // Compute effective cache budget from free VRAM minus reserve.
    // This preserves the existing per-phase budget tracking while the VRAMBudget
    // struct controls strategy-level decisions (which phases to skip).
    size_t free_vram = 0, total_vram = 0;
    IMP_CUDA_CHECK_LOG(cudaMemGetInfo(&free_vram, &total_vram));
    // Reserve at least 10% of total VRAM as headroom to avoid shared/system
    // memory fallback on WSL2 (not visible via nvidia-smi).
    size_t min_reserve = std::max(budget.reserve_bytes, total_vram / 10);
    size_t remaining_budget = (free_vram > min_reserve)
                              ? (free_vram - min_reserve) : 0;

    // Phase 4.2: run StoragePlanner for diagnostic purposes.
    // The plan output is NOT used to drive allocation yet — the existing legacy code
    // path still decides what to allocate. Log discrepancies between the plan and
    // the legacy decisions so we can catch bugs before Phase 4.4+ flips to
    // plan-driven allocation. Actual storage ownership flip happens in Phase 5.
    {
        hints_.vram_budget_bytes = remaining_budget;
        StoragePlan diag_plan = plan_storage(*model_, cfg, hints_);
        if (diag_plan.failed) {
            IMP_LOG_WARN("StoragePlanner (diagnostic): plan failed — %s",
                         diag_plan.failure_reason.c_str());
        } else {
            IMP_LOG_INFO("StoragePlanner (diagnostic): %zu entries, projected VRAM %.2f MiB",
                         diag_plan.entries.size(),
                         diag_plan.projected_vram_bytes / (1024.0 * 1024.0));
        }
    }

    // --- Phase 0: Register NVFP4 pre-quantized weights directly (no quantization needed) ---
    if (cfg.is_nvfp4_prequant) {
        int prequant_count = 0;
        auto register_prequant = [&](const TransformerLayer::NvFP4PreQuantWeight& nw,
                                      const Tensor& weight) {
            if (!nw.valid() || !weight.data) return;
            NvFP4QuantResult result;
            result.packed_data = weight.data;       // FP4 packed [N, K/2]
            result.micro_scales = nw.weight_scale.data;  // FP8 E4M3 [N, K/group_size]
            result.N = weight.shape[0];
            // K is packed: shape[1] stores K/2 for FP4
            result.K = weight.shape[1] * 2;
            // tensor_scale from weight_scale_2 (scalar FP32, may be on host or device)
            if (nw.weight_scale_2.data) {
                float h_scale = 1.0f;
                if (nw.weight_scale_2.on_device) {
                    cudaMemcpy(&h_scale, nw.weight_scale_2.data, sizeof(float), cudaMemcpyDeviceToHost);
                } else {
                    memcpy(&h_scale, nw.weight_scale_2.data, sizeof(float));
                }
                result.tensor_scale = h_scale;
            }
            // Only register if both data and scales are on device
            if (weight.on_device && nw.weight_scale.on_device) {
                wcache_.nvfp4[weight.data] = result;
                prequant_count++;
            } else {
                IMP_LOG_DEBUG("NVFP4 prequant: skipping %p (data_dev=%d, scale_dev=%d)",
                              weight.data, weight.on_device, nw.weight_scale.on_device);
            }
        };

        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            // Dense weights
            register_prequant(L.nvfp4_q, L.wq);
            register_prequant(L.nvfp4_k, L.wk);
            register_prequant(L.nvfp4_v, L.wv);
            register_prequant(L.nvfp4_o, L.wo);
            register_prequant(L.nvfp4_gate, L.w_gate);
            register_prequant(L.nvfp4_up, L.w_up);
            register_prequant(L.nvfp4_down, L.w_down);
            // Expert weights
            for (size_t e = 0; e < L.expert_nvfp4_gate.size(); e++) {
                if (e < L.expert_w_gate.size()) register_prequant(L.expert_nvfp4_gate[e], L.expert_w_gate[e]);
                if (e < L.expert_w_up.size())   register_prequant(L.expert_nvfp4_up[e],   L.expert_w_up[e]);
                if (e < L.expert_w_down.size()) register_prequant(L.expert_nvfp4_down[e], L.expert_w_down[e]);
            }
        }
        // LM head (output projection)
        register_prequant(model_->nvfp4_out_proj(), model_->output_proj());

        if (prequant_count > 0) {
            IMP_LOG_INFO("NVFP4 pre-quantized: registered %d weights directly (no quantization)", prequant_count);
        }
    }

    // Helper: does this qtype benefit from NVFP4 conversion? (> 4.5 bits/elem)
    auto nvfp4_beneficial = [](GGMLQuantType qt) -> bool {
        switch (qt) {
            case GGMLQuantType::Q8_0: case GGMLQuantType::Q8_K:
            case GGMLQuantType::Q6_K: case GGMLQuantType::Q5_K:
                return true;
            default: return false;
        }
    };

    if (wcache_.use_fp8) {
        // Skip Phase 1 entirely: FP8 cache (Phase 2) is the primary path.
        // FP8 is 50% smaller than FP16 and uses FP8×FP8 cuBLASLt (2x throughput
        // on sm_120 tensor cores).  Fused KV/gate+up (saving 1 launch each) are
        // replaced by individual FP8 GEMMs with 2x throughput — net win.
        IMP_LOG_INFO("FP8 prefill: skipping FP16 cache (Phase 1), "
                     "all dense weights → FP8 cache (Phase 2)");
    } else if (budget.strategy == VRAMBudget::NVFP4_DECODE_ONLY) {
        // Skip Phase 1: sub-8-bit weights don't benefit from FP16 expansion.
        // NVFP4 decode cache is the priority — all VRAM goes to Phase 3.
        // Prefill uses CUTLASS NVFP4 GEMM (for eligible weights) or on-the-fly dequant.
        IMP_LOG_INFO("NVFP4 decode only: skipping FP16 cache (Phase 1), "
                     "VRAM reserved for NVFP4 decode cache");
    } else {
        // --- Phase 1: FP16 weight cache + fused KV + fused gate+up ---
        auto cache_weight = [&](const Tensor& w, GGMLQuantType qtype) {
            if (!w.data || !dequant_gpu_supported(qtype)) return;
            if (wcache_.fp16.count(w.data)) return;  // already cached
            if (budget_exhausted) return;

            int rows = static_cast<int>(w.shape[0]);
            int cols = static_cast<int>(w.shape[1]);
            size_t fp16_bytes = static_cast<size_t>(rows) * cols * sizeof(half);

            if (total_cache_bytes + fp16_bytes > remaining_budget) {
                budget_exhausted = true;
                IMP_LOG_INFO("FP16 cache: VRAM budget reached after %d tensors (%.1f / %.1f MiB), "
                             "remaining weights will use on-the-fly dequant",
                             cached_count, total_cache_bytes / (1024.0 * 1024.0),
                             remaining_budget / (1024.0 * 1024.0));
                return;
            }

            void* fp16_buf = vram_alloc(vram_alloc_, fp16_bytes, "fp16_weight_cache");
            if (!fp16_buf) {
                budget_exhausted = true;
                IMP_LOG_WARN("FP16 cache: allocation failed after %d tensors (%.1f MiB)",
                             cached_count, total_cache_bytes / (1024.0 * 1024.0));
                return;
            }

            dequant_gpu(w.data, fp16_buf, qtype, rows, cols, stream);

            Tensor fp16_tensor(fp16_buf, DType::FP16, w.ndim, w.shape, true);
            wcache_.fp16[w.data] = fp16_tensor;
            total_cache_bytes += fp16_bytes;
            cached_count++;
        };

        // Priority order: attention weights first (critical for cuBLAS prefill),
        // then SSM, shared experts, and dense FFN.  This ensures hybrid models
        // like Nemotron (23 SSM + 6 attention layers) cache all attention weights
        // before SSM weights exhaust the VRAM budget.
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            cache_weight(L.wq, L.wq_qtype);
            cache_weight(L.wk, L.wk_qtype);
            cache_weight(L.wv, L.wv_qtype);
            cache_weight(L.wo, L.wo_qtype);
        }
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            cache_weight(L.ssm_in, L.ssm_in_qtype);
            cache_weight(L.ssm_out, L.ssm_out_qtype);
            cache_weight(L.w_gate_shared, L.w_gate_shared_qtype);
            cache_weight(L.w_up_shared, L.w_up_shared_qtype);
            cache_weight(L.w_down_shared, L.w_down_shared_qtype);
            // When NVFP4 decode is active, skip dense FFN FP16 cache for eligible
            // weights.  Decode benefits more from NVFP4 (~47% BW reduction) than
            // prefill loses from on-the-fly dequant.  NVFP4 is also ~3.5x smaller
            // per tensor, so skipping FFN FP16 frees massive VRAM for full NVFP4.
            if (wcache_.nvfp4_decode_mode == 0 || !nvfp4_beneficial(L.w_gate_qtype))
                cache_weight(L.w_gate, L.w_gate_qtype);
            if (wcache_.nvfp4_decode_mode == 0 || !nvfp4_beneficial(L.w_up_qtype))
                cache_weight(L.w_up, L.w_up_qtype);
            if (wcache_.nvfp4_decode_mode == 0 || !nvfp4_beneficial(L.w_down_qtype))
                cache_weight(L.w_down, L.w_down_qtype);
        }

        // Create fused KV weights for strided batched prefill GEMM.
        // Each entry concatenates [wk; wv] as [2*nkv*hd, d_model] FP16 for one layer.
        int fused_kv_count = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            bool stop = false;
            if (create_fused_weight_pair(L.wk, L.wv, wcache_.fp16, vram_alloc_,
                                         total_cache_bytes, remaining_budget,
                                         stream, wcache_.fused_kv, i, stop))
                fused_kv_count++;
            else if (stop) break;
        }

        // Create fused gate+up weights for strided batched prefill GEMM.
        // Each entry concatenates [w_gate; w_up] as [2*d_ff, d_model] FP16 for one layer.
        int fused_gu_count = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            // Both must be the same shape (d_ff x d_model)
            if (L.w_gate.data && L.w_up.data &&
                (L.w_gate.shape[0] != L.w_up.shape[0] ||
                 L.w_gate.shape[1] != L.w_up.shape[1])) continue;
            bool stop = false;
            if (create_fused_weight_pair(L.w_gate, L.w_up, wcache_.fp16, vram_alloc_,
                                         total_cache_bytes, remaining_budget,
                                         stream, wcache_.fused_gate_up, i, stop))
                fused_gu_count++;
            else if (stop) break;
        }

        if (cached_count > 0) {
            IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
            wcache_.fp16_bytes = total_cache_bytes;
            IMP_LOG_INFO("FP16 weight cache: %d tensors, %.2f MiB (incl. %d fused KV, %d fused gate+up)",
                         cached_count, total_cache_bytes / (1024.0 * 1024.0),
                         fused_kv_count, fused_gu_count);
        }
    } // end Phase 1

    // Deduct Phase 1 allocation from shared budget
    deduct_budget(remaining_budget, total_cache_bytes);

    // --- Phase 2: FP8 cache for uncached weights (primary when wcache_.use_fp8) ---
    // When wcache_.use_fp8 is true and Phase 1 was skipped, this is the primary path
    // for ALL dense projection weights.  FP8 is 50% smaller than FP16 and uses
    // FP8×FP8 cuBLASLt with 2x tensor core throughput on sm_120.
    // Uses qscratch_.dequant as FP16 staging buffer (stream ordering ensures safety).
    //
    // Budget cap: respect budget.fp8_cache_bytes to leave room for NVFP4 decode cache.
    // Without this cap, large models (Gemma-3-12B) exhaust VRAM on FP8 prefill,
    // leaving too little for NVFP4 decode → runtime dequant fallback → 10x slowdown.
    size_t fp8_budget = std::min(remaining_budget, budget.fp8_cache_bytes);
    if (wcache_.use_fp8) {
        size_t fp8_total = 0;
        int fp8_count = 0;
        bool fp8_exhausted = false;

        // Collect weights to convert
        struct FP8OverflowEntry {
            const void* orig_ptr;
            Tensor weight;
            GGMLQuantType qtype;
            size_t n_elems;
        };
        std::vector<FP8OverflowEntry> fp8_entries;

        auto collect_weight_fp8 = [&](const Tensor& w, GGMLQuantType qtype) {
            if (!w.data || !dequant_gpu_supported(qtype)) return;
            if (wcache_.fp16.count(w.data)) return;
            if (wcache_.fp8.count(w.data)) return;
            if (fp8_exhausted) return;

            size_t n_elems = static_cast<size_t>(w.shape[0]) * w.shape[1];
            size_t fp8_bytes = n_elems;

            if (fp8_total + fp8_bytes + sizeof(float) > fp8_budget) {
                fp8_exhausted = true;
                IMP_LOG_INFO("FP8 cache: budget reached after %d tensors (%.1f / %.1f MiB, "
                             "saving %.1f MiB for NVFP4 decode)",
                             fp8_count, fp8_total / (1024.0 * 1024.0),
                             fp8_budget / (1024.0 * 1024.0),
                             (remaining_budget - fp8_budget) / (1024.0 * 1024.0));
                return;
            }

            fp8_entries.push_back({w.data, w, qtype, n_elems});
            fp8_total += fp8_bytes + sizeof(float);
            fp8_count++;
        };

        // Same priority order — attention first, then SSM/FFN
        for_each_dense_weight(*model_, cfg, collect_weight_fp8);

        if (!fp8_entries.empty() && qscratch_.dequant) {
            // Pre-allocate reusable calibration temp buffers
            int max_grid = 0;
            size_t total_fp8_bytes = 0;
            for (auto& e : fp8_entries) {
                int threads_needed = (static_cast<int>(e.n_elems) + 3) / 4;
                int grid = (threads_needed + 255) / 256;
                if (grid > max_grid) max_grid = grid;
                total_fp8_bytes += e.n_elems;
            }

            float* d_block_maxes = nullptr;
            float* d_absmax = nullptr;
            float* d_scales_all = nullptr;
            IMP_CUDA_CHECK_LOG(cudaMalloc(&d_block_maxes, (size_t)max_grid * sizeof(float)));
            IMP_CUDA_CHECK_LOG(cudaMalloc(&d_absmax, sizeof(float)));
            IMP_CUDA_CHECK_LOG(cudaMalloc(&d_scales_all, fp8_entries.size() * sizeof(float)));

            // Bulk-allocate all FP8 data
            uint8_t* d_fp8_bulk = static_cast<uint8_t*>(
                vram_alloc(vram_alloc_, total_fp8_bytes, "fp8_weight_cache"));
            if (!d_fp8_bulk) {
                cudaError_t e = cudaGetLastError();
                IMP_LOG_WARN("FP8 weight cache bulk alloc failed (%.1f MiB): %s",
                             total_fp8_bytes / (1024.0 * 1024.0), cudaGetErrorString(e));
            }

            int actual_count = 0;
            size_t fp8_offset = 0;
            for (size_t i = 0; i < fp8_entries.size() && d_fp8_bulk; i++) {
                auto& e = fp8_entries[i];
                int rows = static_cast<int>(e.weight.shape[0]);
                int cols = static_cast<int>(e.weight.shape[1]);

                // Dequant to qscratch_.dequant (reused each iteration, stream-ordered)
                dequant_gpu(e.weight.data, qscratch_.dequant, e.qtype, rows, cols, stream);

                void* fp8_buf = d_fp8_bulk + fp8_offset;
                fp8_offset += e.n_elems;

                // Async calibrate + quantize (no host sync)
                calibrate_and_quantize_fp8_async(
                    qscratch_.dequant, fp8_buf, static_cast<int>(e.n_elems),
                    d_block_maxes, max_grid,
                    d_absmax, d_scales_all + static_cast<ptrdiff_t>(i), stream);

                Tensor fp8_t(fp8_buf, DType::FP8_E4M3, e.weight.ndim, e.weight.shape, true);
                wcache_.fp8[e.orig_ptr] = {fp8_t, 0.0f, d_scales_all + static_cast<ptrdiff_t>(i)};
                actual_count++;
            }

            if (actual_count > 0) {
                IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
                // Read back scales
                std::vector<float> h_scales(actual_count);
                IMP_CUDA_CHECK_LOG(cudaMemcpy(h_scales.data(), d_scales_all, actual_count * sizeof(float),
                           cudaMemcpyDeviceToHost));
                for (int i = 0; i < actual_count; i++) {
                    auto it = wcache_.fp8.find(fp8_entries[i].orig_ptr);
                    if (it != wcache_.fp8.end()) {
                        it->second.host_scale = h_scales[i];
                    }
                }
            }

            IMP_CUDA_CHECK_LOG(cudaFree(d_block_maxes));
            IMP_CUDA_CHECK_LOG(cudaFree(d_absmax));
            // Track bulk buffers for cleanup
            wcache_.fp8_overflow_scales = d_scales_all;
            wcache_.fp8_overflow_count = actual_count;
            wcache_.fp8_overflow_data = d_fp8_bulk;
            wcache_.fp8_overflow_data_size = total_fp8_bytes;
            fp8_count = actual_count;
        }

        if (fp8_count > 0) {
            wcache_.fp8_bytes = fp8_total;
            size_t fp16_equivalent = 0;
            for (auto& [ptr, entry] : wcache_.fp8) {
                fp16_equivalent += entry.weight.numel() * sizeof(half);
            }
            IMP_LOG_INFO("FP8 weight cache: %d tensors, %.2f MiB (%.2f MiB saved vs FP16)",
                         fp8_count, fp8_total / (1024.0 * 1024.0),
                         (fp16_equivalent - fp8_total) / (1024.0 * 1024.0));
        } else {
            IMP_LOG_INFO("FP8 prefill: no weights cached (budget=0 or no eligible weights)");
        }
    }

    // Deduct Phase 2 allocation from shared budget
    deduct_budget(remaining_budget, wcache_.fp8_bytes);

    // --- Phase 3: NVFP4 decode weight cache ---
    // Converts eligible weights (> 4.5 bits/elem) to NVFP4 format for faster
    // decode GEMV.  Mode 2 ("only") uses incremental processing: quantize from
    // FP16 cache and free each entry immediately (NVFP4 ≈ 28% of FP16 size, so
    // each conversion is net VRAM-negative, bootstrapping space for more tensors).
    // Mode 1 ("additive") uses standard batch processing with FP16 cache intact.
    if (wcache_.nvfp4_decode_mode > 0) {
        const char* mode_str = (wcache_.nvfp4_decode_mode == 1) ? "additive" : "only";

        // Build exclusion sets for weights that should NOT get NVFP4 quantization.
        std::unordered_set<const void*> nvfp4_exclude_ptrs;

        // Dual-path mode: attention weights stay at FP8 for quality.
        if (wcache_.dual_path_quant) {
            for (int i = 0; i < cfg.n_layers; i++) {
                const auto& L = model_->layer(i);
                if (L.wq.data) nvfp4_exclude_ptrs.insert(L.wq.data);
                if (L.wk.data) nvfp4_exclude_ptrs.insert(L.wk.data);
                if (L.wv.data) nvfp4_exclude_ptrs.insert(L.wv.data);
                if (L.wo.data) nvfp4_exclude_ptrs.insert(L.wo.data);
            }
            IMP_LOG_INFO("Dual-path quant: excluding %zu attention weights from NVFP4 cache",
                         nvfp4_exclude_ptrs.size());
        }

        // GDN/SSM models: exclude ssm_in/ssm_out projections from NVFP4.
        // These feed the recurrent scan which accumulates quantization error
        // in state H across tokens. 4-bit degrades quality on 9B+ models.
        {
            int n_ssm_excluded = 0;
            for (int i = 0; i < cfg.n_layers; i++) {
                const auto& L = model_->layer(i);
                if (L.ssm_in.data)  { nvfp4_exclude_ptrs.insert(L.ssm_in.data); n_ssm_excluded++; }
                if (L.ssm_out.data) { nvfp4_exclude_ptrs.insert(L.ssm_out.data); n_ssm_excluded++; }
            }
            if (n_ssm_excluded > 0)
                IMP_LOG_INFO("GDN/SSM: excluding %d recurrent projections from NVFP4 cache", n_ssm_excluded);
        }

        // Collect eligible weights first, then process.
        struct NvFP4Entry {
            const void* orig_ptr;
            Tensor weight;
            GGMLQuantType qtype;
            bool from_scratch;
        };
        std::vector<NvFP4Entry> nvfp4_entries;

        auto collect_weight_nvfp4 = [&](const Tensor& w, GGMLQuantType qtype) {
            if (!w.data) return;
            if (!nvfp4_beneficial(qtype)) return;
            if (wcache_.nvfp4.count(w.data)) return;
            // Skip excluded weights (dual-path attention, GDN/SSM recurrent projections)
            if (nvfp4_exclude_ptrs.count(w.data)) return;

            int cols = static_cast<int>(w.shape[1]);
            if (cols % 16 != 0) return;

            bool from_scratch = (wcache_.fp16.find(w.data) == wcache_.fp16.end());
            if (from_scratch && (!dequant_gpu_supported(qtype) || !qscratch_.dequant)) return;
            nvfp4_entries.push_back({w.data, w, qtype, from_scratch});
        };

        // LM head first: largest single weight (vocab × d_model), biggest bandwidth win.
        collect_weight_nvfp4(model_->output_proj(), model_->out_proj_qtype_);

        // Dense attention + FFN: every tensor benefits every decode step.
        for_each_dense_weight(*model_, cfg, collect_weight_nvfp4);

        if (wcache_.nvfp4_decode_mode == 2 && !nvfp4_entries.empty()) {
            // Mode 2 incremental: process FP16-cached entries first (each conversion
            // frees net VRAM since NVFP4 ≈ 28% of FP16), then from-scratch entries.
            // Sort: FP16-cached first (smallest first to bootstrap), then from-scratch.
            std::stable_sort(nvfp4_entries.begin(), nvfp4_entries.end(),
                [](const NvFP4Entry& a, const NvFP4Entry& b) {
                    if (a.from_scratch != b.from_scratch) return !a.from_scratch;
                    size_t a_sz = static_cast<size_t>(a.weight.shape[0]) * a.weight.shape[1];
                    size_t b_sz = static_cast<size_t>(b.weight.shape[0]) * b.weight.shape[1];
                    return a_sz < b_sz;
                });

            float* d_absmax_buf = nullptr;
            float* d_tscale_buf = nullptr;
            IMP_CUDA_CHECK_LOG(cudaMalloc(&d_absmax_buf, sizeof(float)));
            IMP_CUDA_CHECK_LOG(cudaMalloc(&d_tscale_buf, sizeof(float)));

            int actual_count = 0;
            size_t actual_bytes = 0;
            int actual_from_fp16 = 0;
            int actual_from_scratch = 0;

            for (auto& e : nvfp4_entries) {
                int rows = static_cast<int>(e.weight.shape[0]);
                int cols = static_cast<int>(e.weight.shape[1]);
                size_t nvfp4_bytes = static_cast<size_t>(rows) * cols / 2 +
                                     static_cast<size_t>(rows) * cols / 16 + 4;

                // Check actual free VRAM (10% of total as safety margin)
                size_t free_mem = 0, total_mem = 0;
                IMP_CUDA_CHECK_LOG(cudaMemGetInfo(&free_mem, &total_mem));
                size_t nvfp4_safety = std::max(total_mem / 10, static_cast<size_t>(1024 * 1024));
                if (free_mem < nvfp4_bytes + nvfp4_safety) {
                    IMP_LOG_INFO("NVFP4 incremental: VRAM exhausted after %d tensors "
                                 "(%.1f MiB, %.1f MiB free)", actual_count,
                                 actual_bytes / (1024.0 * 1024.0), free_mem / (1024.0 * 1024.0));
                    break;
                }

                const half* fp16_ptr = nullptr;
                void* tmp_buf = nullptr;

                if (e.from_scratch) {
                    size_t need = static_cast<size_t>(rows) * cols * sizeof(half);
                    void* dq_buf = qscratch_.dequant;
                    if (need > qscratch_.dequant_size) {
                        if (cudaMalloc(&tmp_buf, need) != cudaSuccess || !tmp_buf) continue;
                        dq_buf = tmp_buf;
                    }
                    dequant_gpu(e.weight.data, dq_buf, e.qtype, rows, cols, stream);
                    fp16_ptr = reinterpret_cast<const half*>(dq_buf);
                } else {
                    auto it = wcache_.fp16.find(e.orig_ptr);
                    fp16_ptr = reinterpret_cast<const half*>(it->second.data);
                }

                Tensor fp16_view(const_cast<half*>(fp16_ptr), DType::FP16, 2,
                                 e.weight.shape, true);

                NvFP4QuantResult result;
                quantize_fp16_to_nvfp4_async(fp16_view, result,
                                              d_absmax_buf, d_tscale_buf, stream);

                // Sync immediately so we can read tensor_scale and free FP16
                IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));

                float h_tscale;
                IMP_CUDA_CHECK_LOG(cudaMemcpy(&h_tscale, d_tscale_buf, sizeof(float), cudaMemcpyDeviceToHost));
                result.tensor_scale = h_tscale;
                wcache_.nvfp4[e.orig_ptr] = result;
                actual_bytes += nvfp4_bytes;
                actual_count++;

                if (tmp_buf) IMP_CUDA_CHECK_LOG(cudaFree(tmp_buf));

                // Free FP16 cache entry to reclaim VRAM for next weight
                if (!e.from_scratch) {
                    auto it = wcache_.fp16.find(e.orig_ptr);
                    if (it != wcache_.fp16.end()) {
                        size_t freed = it->second.nbytes();
                        vram_free(vram_alloc_, it->second.data);
                        wcache_.fp16.erase(it);
                        wcache_.fp16_bytes -= freed;
                        actual_from_fp16++;
                    }
                } else {
                    actual_from_scratch++;
                }
            }

            IMP_CUDA_CHECK_LOG(cudaFree(d_absmax_buf));
            IMP_CUDA_CHECK_LOG(cudaFree(d_tscale_buf));

            wcache_.nvfp4_bytes = actual_bytes;
            IMP_LOG_INFO("NVFP4 decode cache: %d tensors, %.2f MiB "
                         "(%d from FP16, %d from scratch, mode: %s)",
                         actual_count, actual_bytes / (1024.0 * 1024.0),
                         actual_from_fp16, actual_from_scratch, mode_str);
        } else if (!nvfp4_entries.empty()) {
            // Mode 1 standard batch: quantize entries that fit in budget, single sync.
            size_t budget_used = 0;
            int nvfp4_count = 0;
            int nvfp4_from_scratch = 0;
            bool budget_exhausted = false;

            std::vector<NvFP4Entry> budgeted;
            for (auto& e : nvfp4_entries) {
                size_t rows = e.weight.shape[0], cols = e.weight.shape[1];
                size_t nvfp4_bytes = rows * cols / 2 + rows * cols / 16 + 4;
                if (budget_used + nvfp4_bytes > remaining_budget) {
                    if (!budget_exhausted) {
                        budget_exhausted = true;
                        IMP_LOG_INFO("NVFP4 cache: VRAM budget reached after %d/%zu tensors "
                                     "(%.1f / %.1f MiB)",
                                     nvfp4_count, nvfp4_entries.size(),
                                     budget_used / (1024.0 * 1024.0),
                                     remaining_budget / (1024.0 * 1024.0));
                    }
                    continue;
                }
                budget_used += nvfp4_bytes;
                nvfp4_count++;
                if (e.from_scratch) nvfp4_from_scratch++;
                budgeted.push_back(e);
            }

            float* d_absmax_buf = nullptr;
            IMP_CUDA_CHECK_LOG(cudaMalloc(&d_absmax_buf, sizeof(float)));

            float* d_tscales_all = nullptr;
            IMP_CUDA_CHECK_LOG(cudaMalloc(&d_tscales_all, budgeted.size() * sizeof(float)));

            std::vector<void*> tmp_bufs;
            for (size_t i = 0; i < budgeted.size(); i++) {
                auto& e = budgeted[i];
                const half* fp16_ptr = nullptr;
                int rows = static_cast<int>(e.weight.shape[0]);
                int cols = static_cast<int>(e.weight.shape[1]);

                if (e.from_scratch) {
                    size_t need = static_cast<size_t>(rows) * cols * sizeof(half);
                    void* dq_buf = qscratch_.dequant;
                    if (need > qscratch_.dequant_size) {
                        void* tmp = nullptr;
                        if (cudaMalloc(&tmp, need) != cudaSuccess || !tmp) continue;
                        dq_buf = tmp;
                        tmp_bufs.push_back(tmp);
                    }
                    dequant_gpu(e.weight.data, dq_buf, e.qtype, rows, cols, stream);
                    fp16_ptr = reinterpret_cast<const half*>(dq_buf);
                } else {
                    auto it = wcache_.fp16.find(e.orig_ptr);
                    fp16_ptr = reinterpret_cast<const half*>(it->second.data);
                }

                Tensor fp16_view(const_cast<half*>(fp16_ptr), DType::FP16, 2,
                                 e.weight.shape, true);

                NvFP4QuantResult result;
                quantize_fp16_to_nvfp4_async(fp16_view, result,
                                              d_absmax_buf,
                                              d_tscales_all + i,
                                              stream);
                wcache_.nvfp4[e.orig_ptr] = result;
            }

            IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
            for (void* p : tmp_bufs) IMP_CUDA_CHECK_LOG(cudaFree(p));

            std::vector<float> h_tscales(budgeted.size());
            IMP_CUDA_CHECK_LOG(cudaMemcpy(h_tscales.data(), d_tscales_all,
                       budgeted.size() * sizeof(float),
                       cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < budgeted.size(); i++) {
                auto it = wcache_.nvfp4.find(budgeted[i].orig_ptr);
                if (it != wcache_.nvfp4.end()) {
                    it->second.tensor_scale = h_tscales[i];
                }
            }

            IMP_CUDA_CHECK_LOG(cudaFree(d_absmax_buf));
            IMP_CUDA_CHECK_LOG(cudaFree(d_tscales_all));

            wcache_.nvfp4_bytes = budget_used;
            if (nvfp4_from_scratch > 0) {
                IMP_LOG_INFO("NVFP4 decode cache: %d tensors, %.2f MiB (%d from FP16 cache, %d via dequant scratch, mode: %s)",
                             nvfp4_count, budget_used / (1024.0 * 1024.0),
                             nvfp4_count - nvfp4_from_scratch, nvfp4_from_scratch, mode_str);
            } else {
                IMP_LOG_INFO("NVFP4 decode cache: %d tensors, %.2f MiB (mode: %s)",
                             nvfp4_count, budget_used / (1024.0 * 1024.0), mode_str);
            }
        }

        // In "only" mode (2), release remaining FP16 cache.
        // Before freeing, migrate FP16 weights to FP8 cache so prefill retains
        // fast FP8 GEMM.  FP8 = half the size of FP16, net 50% VRAM savings.
        if (wcache_.nvfp4_decode_mode == 2 && !wcache_.fp16.empty()) {
            int migrated = 0;
            size_t migrated_bytes = 0;
            if (wcache_.use_fp8) {
                struct MigrateEntry {
                    const void* orig_ptr;
                    Tensor fp16_tensor;
                    size_t n_elems;
                };
                std::vector<MigrateEntry> to_migrate;
                for (auto& [orig_ptr, fp16_tensor] : wcache_.fp16) {
                    if (wcache_.fp8.count(orig_ptr)) continue;
                    size_t n = static_cast<size_t>(fp16_tensor.shape[0]) * fp16_tensor.shape[1];
                    to_migrate.push_back({orig_ptr, fp16_tensor, n});
                }

                if (!to_migrate.empty()) {
                    int max_grid = 0;
                    size_t total_fp8_bytes = 0;
                    for (auto& e : to_migrate) {
                        int threads_needed = (static_cast<int>(e.n_elems) + 3) / 4;
                        int grid = (threads_needed + 255) / 256;
                        if (grid > max_grid) max_grid = grid;
                        total_fp8_bytes += e.n_elems;
                    }

                    float* d_block_maxes = nullptr;
                    float* d_absmax = nullptr;
                    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_block_maxes, (size_t)max_grid * sizeof(float)));
                    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_absmax, sizeof(float)));

                    float* d_scales_all = nullptr;
                    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_scales_all, to_migrate.size() * sizeof(float)));

                    uint8_t* d_fp8_bulk = nullptr;
                    d_fp8_bulk = static_cast<uint8_t*>(
                        vram_alloc(vram_alloc_, total_fp8_bytes, "fp8_migration_cache"));
                    if (!d_fp8_bulk) {
                        cudaError_t e = cudaGetLastError();
                        IMP_LOG_WARN("FP8 migration cache alloc failed (%.1f MiB): %s",
                                     total_fp8_bytes / (1024.0 * 1024.0), cudaGetErrorString(e));
                    }

                    size_t fp8_offset = 0;
                    for (size_t i = 0; i < to_migrate.size() && d_fp8_bulk; i++) {
                        auto& e = to_migrate[i];
                        void* fp8_buf = d_fp8_bulk + fp8_offset;
                        fp8_offset += e.n_elems;

                        calibrate_and_quantize_fp8_async(
                            e.fp16_tensor.data, fp8_buf, static_cast<int>(e.n_elems),
                            d_block_maxes, max_grid,
                            d_absmax, d_scales_all + i, stream);

                        Tensor fp8_t(fp8_buf, DType::FP8_E4M3, e.fp16_tensor.ndim,
                                     e.fp16_tensor.shape, true);
                        wcache_.fp8[e.orig_ptr] = {fp8_t, 0.0f, d_scales_all + static_cast<ptrdiff_t>(i)};
                        migrated++;
                        migrated_bytes += e.n_elems + sizeof(float);
                    }

                    wcache_.fp8_migrated_data = d_fp8_bulk;
                    wcache_.fp8_migrated_data_size = total_fp8_bytes;

                    if (migrated > 0) {
                        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
                        std::vector<float> h_scales(migrated);
                        IMP_CUDA_CHECK_LOG(cudaMemcpy(h_scales.data(), d_scales_all, migrated * sizeof(float),
                                   cudaMemcpyDeviceToHost));
                        int idx = 0;
                        for (size_t i = 0; i < to_migrate.size() && idx < migrated; i++, idx++) {
                            auto it = wcache_.fp8.find(to_migrate[i].orig_ptr);
                            if (it != wcache_.fp8.end()) {
                                it->second.host_scale = h_scales[idx];
                            }
                        }
                    }

                    IMP_CUDA_CHECK_LOG(cudaFree(d_block_maxes));
                    IMP_CUDA_CHECK_LOG(cudaFree(d_absmax));
                    wcache_.fp8_migrated_scales = d_scales_all;
                    wcache_.fp8_migrated_count = migrated;
                }
            }

            // Free remaining FP16 cache — but KEEP entries that have no NVFP4
            // or FP8 alternative (e.g. GDN `ssm_in`/`ssm_out` on hybrid models
            // like Qwen 3.5/3.6). Without this, run_gdn falls back to on-the-fly
            // dequant which produces ~5% per-element drift at L0 and cascades
            // to sign-flips at the shared MLP → garbage output.
            size_t freed = 0;
            size_t kept_bytes = 0;
            int kept_count = 0;
            std::vector<const void*> to_erase;
            for (auto& [ptr, tensor] : wcache_.fp16) {
                const bool has_nvfp4 = (wcache_.nvfp4.find(ptr) != wcache_.nvfp4.end());
                const bool has_fp8   = (wcache_.fp8.find(ptr)   != wcache_.fp8.end());
                if (has_nvfp4 || has_fp8) {
                    vram_free(vram_alloc_, tensor.data);
                    freed += static_cast<size_t>(tensor.shape[0]) * tensor.shape[1] * sizeof(half);
                    to_erase.push_back(ptr);
                } else {
                    kept_bytes += static_cast<size_t>(tensor.shape[0]) * tensor.shape[1] * sizeof(half);
                    kept_count++;
                }
            }
            for (auto p : to_erase) wcache_.fp16.erase(p);
            wcache_.fp16_bytes = kept_bytes;
            if (kept_count > 0) {
                IMP_LOG_INFO("NVFP4 only mode: preserved %d FP16 entries (%.2f MiB) "
                             "with no NVFP4/FP8 alternative (GDN/hybrid weights)",
                             kept_count, kept_bytes / (1024.0 * 1024.0));
            }

            // Free fused caches (prefill uses individual FP8 weights)
            for (auto& [idx, tensor] : wcache_.fused_kv) {
                if (tensor.data) vram_free(vram_alloc_, tensor.data);
            }
            wcache_.fused_kv.clear();
            for (auto& [idx, tensor] : wcache_.fused_gate_up) {
                if (tensor.data) vram_free(vram_alloc_, tensor.data);
            }
            wcache_.fused_gate_up.clear();

            remaining_budget += freed;
            wcache_.fp8_bytes += migrated_bytes;
            IMP_LOG_INFO("NVFP4 only mode: freed FP16 cache (%.2f MiB), migrated %d weights to FP8 (%.2f MiB)",
                         freed / (1024.0 * 1024.0), migrated, migrated_bytes / (1024.0 * 1024.0));
        }

        // --- NVFP4 second pass: cache remaining tensors with freed VRAM ---
        // After FP16-Free and FP8 migration, VRAM that was locked by FP16 cache is
        // now available. Re-run NVFP4 for entries that were skipped due to VRAM pressure.
        if (budget.nvfp4_second_pass && !nvfp4_entries.empty()) {
            float* d_absmax_buf2 = nullptr;
            float* d_tscale_buf2 = nullptr;
            IMP_CUDA_CHECK_LOG(cudaMalloc(&d_absmax_buf2, sizeof(float)));
            IMP_CUDA_CHECK_LOG(cudaMalloc(&d_tscale_buf2, sizeof(float)));

            int second_count = 0;
            size_t second_bytes = 0;

            for (auto& e : nvfp4_entries) {
                if (wcache_.nvfp4.count(e.orig_ptr)) continue;  // already cached
                int rows = static_cast<int>(e.weight.shape[0]);
                int cols = static_cast<int>(e.weight.shape[1]);
                size_t nvfp4_bytes = static_cast<size_t>(rows) * cols / 2 +
                                     static_cast<size_t>(rows) * cols / 16 + 4;

                size_t free_mem2 = 0, total_mem2 = 0;
                IMP_CUDA_CHECK_LOG(cudaMemGetInfo(&free_mem2, &total_mem2));
                size_t nvfp4_safety2 = std::max(total_mem2 / 10, static_cast<size_t>(1024 * 1024));
                if (free_mem2 < nvfp4_bytes + nvfp4_safety2) break;

                // Dequant from quantized weights via scratch buffer
                size_t need = static_cast<size_t>(rows) * cols * sizeof(half);
                void* dq_buf = qscratch_.dequant;
                void* tmp_buf = nullptr;
                if (!dequant_gpu_supported(e.qtype) || !qscratch_.dequant) continue;
                if (need > qscratch_.dequant_size) {
                    if (cudaMalloc(&tmp_buf, need) != cudaSuccess || !tmp_buf) continue;
                    dq_buf = tmp_buf;
                }
                dequant_gpu(e.weight.data, dq_buf, e.qtype, rows, cols, stream);

                Tensor fp16_view(reinterpret_cast<half*>(dq_buf), DType::FP16, 2,
                                 e.weight.shape, true);
                NvFP4QuantResult result;
                quantize_fp16_to_nvfp4_async(fp16_view, result,
                                              d_absmax_buf2, d_tscale_buf2, stream);
                IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));

                float h_tscale;
                IMP_CUDA_CHECK_LOG(cudaMemcpy(&h_tscale, d_tscale_buf2, sizeof(float), cudaMemcpyDeviceToHost));
                result.tensor_scale = h_tscale;
                wcache_.nvfp4[e.orig_ptr] = result;
                second_bytes += nvfp4_bytes;
                second_count++;

                if (tmp_buf) IMP_CUDA_CHECK_LOG(cudaFree(tmp_buf));
            }

            IMP_CUDA_CHECK_LOG(cudaFree(d_absmax_buf2));
            IMP_CUDA_CHECK_LOG(cudaFree(d_tscale_buf2));

            if (second_count > 0) {
                wcache_.nvfp4_bytes += second_bytes;
                IMP_LOG_INFO("NVFP4 second pass: %d additional tensors, %.2f MiB",
                             second_count, second_bytes / (1024.0 * 1024.0));
            }
        }

        // --- Phase 3b: Convert NVFP4 weights to CUTLASS sm_120 block-scaled format ---
        // Must be AFTER FP16 free to avoid peak VRAM exceeding physical memory.
        // The CUTLASS cache is a full copy (repacked data + SfAtom scales), so it
        // approximately doubles the NVFP4 cache VRAM.  Budget-aware: stop if VRAM runs out.
        if (!wcache_.nvfp4.empty() && cutlass_sm120_nvfp4_available()) {
            // After incremental mode, remaining_budget is stale.  Use actual free VRAM.
            size_t ct_budget;
            if (wcache_.nvfp4_decode_mode == 2) {
                IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
                size_t free_mem = 0, total_mem = 0;
                IMP_CUDA_CHECK_LOG(cudaMemGetInfo(&free_mem, &total_mem));
                size_t kCtReserve = std::max(total_mem / 10, static_cast<size_t>(256ULL * 1024 * 1024));
                ct_budget = (free_mem > kCtReserve) ? (free_mem - kCtReserve) : 0;
            } else {
                ct_budget = (remaining_budget > wcache_.nvfp4_bytes)
                            ? (remaining_budget - wcache_.nvfp4_bytes) : 0;
            }
            int ct_count = 0;
            size_t ct_total = 0;
            bool ct_exhausted = false;
            for (auto& [ptr, nvfp4] : wcache_.nvfp4) {
                if (ct_exhausted) break;
                // Estimate CUTLASS allocation (only scale factors — data is borrowed)
                size_t est = cutlass_nvfp4_sf_size(static_cast<int>(nvfp4.N),
                                                    static_cast<int>(nvfp4.K));
                if (ct_total + est > ct_budget) {
                    ct_exhausted = true;
                    IMP_LOG_INFO("CUTLASS NVFP4 cache: VRAM budget reached after %d tensors "
                                 "(%.1f / %.1f MiB)",
                                 ct_count, ct_total / (1024.0 * 1024.0),
                                 ct_budget / (1024.0 * 1024.0));
                    break;
                }
                CutlassNvFP4Weight cw;
                convert_nvfp4_to_cutlass(nvfp4, cw, stream);
                if (cw.data) {
                    wcache_.cutlass_nvfp4[ptr] = cw;
                    ct_total += cw.sf_bytes;
                    ct_count++;
                }
            }
            if (ct_count > 0) {
                IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
                wcache_.cutlass_nvfp4_bytes = ct_total;
                deduct_budget(remaining_budget, ct_total + wcache_.nvfp4_bytes);
                IMP_LOG_INFO("CUTLASS sm_120 NVFP4 weight cache: %d tensors, %.2f MiB",
                             ct_count, ct_total / (1024.0 * 1024.0));
            }
        }

        // Phase 3c-native: register MXFP4 GGUF weights directly in CUTLASS cache.
        {
            // These bypass NVFP4 entirely — the GGUF data is unpacked into
            // separate E2M1 data + SfAtom UE8M0 scales on GPU.
            // For native MXFP4, allocate activation buffers if not already done.
            if (cutlass_sm120_mxfp4_available()) {
                // Check if any layer has MXFP4 weights
                bool has_mxfp4 = false;
                auto check_mxfp4 = [&](const Tensor&, GGMLQuantType qt) {
                    if (qt == GGMLQuantType::MXFP4) has_mxfp4 = true;
                };
                for (int i = 0; i < cfg.n_layers && !has_mxfp4; i++) {
                    const auto& L = model_->layer(i);
                    check_mxfp4(L.wq, L.wq_qtype);
                    check_mxfp4(L.wk, L.wk_qtype);
                    check_mxfp4(L.w_gate, L.w_gate_qtype);
                    check_mxfp4(L.ssm_in, L.ssm_in_qtype);
                    check_mxfp4(L.ssm_out, L.ssm_out_qtype);
                }

                // Allocate MXFP4 scratch if needed and not already allocated
                if (has_mxfp4 && !qscratch_.mxfp4_act_sf) {
                    int max_k = 0, max_n = 0;
                    for (int i = 0; i < cfg.n_layers; i++) {
                        const auto& L = model_->layer(i);
                        if (L.wq.data && L.wq.ndim >= 2) {
                            max_n = std::max(max_n, (int)L.wq.shape[0]);
                            max_k = std::max(max_k, (int)L.wq.shape[1]);
                        }
                        if (L.w_gate.data && L.w_gate.ndim >= 2) {
                            max_n = std::max(max_n, (int)L.w_gate.shape[0]);
                            max_k = std::max(max_k, (int)L.w_gate.shape[1]);
                        }
                        if (L.w_down.data && L.w_down.ndim >= 2) {
                            max_n = std::max(max_n, (int)L.w_down.shape[0]);
                            max_k = std::max(max_k, (int)L.w_down.shape[1]);
                        }
                        if (L.ssm_in.data && L.ssm_in.ndim >= 2) {
                            max_n = std::max(max_n, (int)L.ssm_in.shape[0]);
                            max_k = std::max(max_k, (int)L.ssm_in.shape[1]);
                        }
                        if (L.ssm_out.data && L.ssm_out.ndim >= 2) {
                            max_n = std::max(max_n, (int)L.ssm_out.shape[0]);
                            max_k = std::max(max_k, (int)L.ssm_out.shape[1]);
                        }
                    }
                    if (max_k > 0) {
                        qscratch_.mxfp4_act_sf_size = cutlass_mxfp4_sf_size(max_tokens_, max_k);
                        qscratch_.mxfp4_workspace_size = gemm_mxfp4_cutlass_sm120_workspace(max_tokens_, max_n, max_k);
                        qscratch_.mxfp4_act_sf = vram_alloc(vram_alloc_, qscratch_.mxfp4_act_sf_size, "mxfp4_act_sf");
                        qscratch_.mxfp4_workspace = (qscratch_.mxfp4_workspace_size > 0)
                            ? vram_alloc(vram_alloc_, qscratch_.mxfp4_workspace_size, "mxfp4_workspace")
                            : nullptr;
                        // Also need CUTLASS activation data buffer
                        if (!qscratch_.cutlass_act_data) {
                            qscratch_.cutlass_act_data_size = static_cast<size_t>(max_tokens_) * (max_k / 2);
                            qscratch_.cutlass_act_data = vram_alloc(vram_alloc_, qscratch_.cutlass_act_data_size, "cutlass_act_data");
                        }
                        IMP_LOG_INFO("Native MXFP4: allocated activation scratch (sf=%.2f MiB)",
                                     qscratch_.mxfp4_act_sf_size / (1024.0 * 1024.0));
                    }
                }
            }

            // Convert NVFP4 weights to MXFP4 (UE8M0 scales) if MXFP4 prefill is enabled.
            // Same packed FP4 data (borrowed), only allocates new scale factor buffers.
            // Note: Hadamard rotation requires MR-GPTQ pre-rotated weights (SafeTensors).
            // For GGUF models, we use direct scale conversion (no rotation).
            if (wcache_.use_mxfp4 && qscratch_.mxfp4_act_sf != nullptr && cutlass_sm120_mxfp4_available()) {
                int mx_count = 0;
                size_t mx_total = 0;
                for (auto& [ptr, nvfp4] : wcache_.nvfp4) {
                    // Only convert weights where K is multiple of 32 (MXFP4 requirement)
                    if (nvfp4.K % 32 != 0) continue;
                    CutlassMxFP4Weight mw;
                    convert_nvfp4_to_mxfp4_cutlass(nvfp4, mw, stream);
                    if (mw.data) {
                        wcache_.cutlass_mxfp4[ptr] = mw;
                        mx_total += mw.sf_bytes;
                        mx_count++;
                    }
                }
                if (mx_count > 0) {
                    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
                    wcache_.cutlass_mxfp4_bytes = mx_total;
                    IMP_LOG_INFO("CUTLASS sm_120 MXFP4 weight cache: %d tensors, %.2f MiB",
                                 mx_count, mx_total / (1024.0 * 1024.0));
                }
            }
        }

        // Native MXFP4 GGUF: unpack and register directly in CUTLASS cache.
        // TEMPORARILY DISABLED — debugging illegal memory access
        if (qscratch_.mxfp4_act_sf != nullptr && cutlass_sm120_mxfp4_available()) {
            int mx_native = 0;
            size_t mx_native_bytes = 0;
            auto register_if_mxfp4 = [&](const Tensor& w, GGMLQuantType qt, bool is_attn = true) {
                if (qt != GGMLQuantType::MXFP4 || !w.data || !w.on_device) return;
                if (w.ndim < 2 || w.shape[1] % 32 != 0) return;
                if (wcache_.cutlass_mxfp4.count(w.data)) return;  // already registered
                CutlassMxFP4Weight mw;
                if (unpack_mxfp4_gguf(w.data, w.shape[0], w.shape[1], mw, stream)) {
                    mw.hadamard_bs = is_attn ? cfg.mxfp4_hadamard_attn : cfg.mxfp4_hadamard_ffn;
                    wcache_.cutlass_mxfp4[w.data] = mw;
                    mx_native_bytes += mw.sf_bytes + static_cast<size_t>(w.shape[0]) * (w.shape[1] / 2);
                    mx_native++;
                }
            };
            for (int i = 0; i < cfg.n_layers; i++) {
                const auto& L = model_->layer(i);
                register_if_mxfp4(L.wq, L.wq_qtype, true);
                register_if_mxfp4(L.wk, L.wk_qtype, true);
                register_if_mxfp4(L.wv, L.wv_qtype, true);
                register_if_mxfp4(L.wo, L.wo_qtype, true);
                register_if_mxfp4(L.w_up, L.w_up_qtype, false);
                register_if_mxfp4(L.w_gate, L.w_gate_qtype, false);
                register_if_mxfp4(L.w_down, L.w_down_qtype, false);
                // GDN-specific weights (Qwen3.5)
                register_if_mxfp4(L.ssm_in, L.ssm_in_qtype, true);
                register_if_mxfp4(L.ssm_out, L.ssm_out_qtype, true);
                register_if_mxfp4(L.gdn_gate, L.gdn_gate_qtype, true);
                register_if_mxfp4(L.gdn_alpha, L.gdn_alpha_qtype, true);
                register_if_mxfp4(L.gdn_beta, L.gdn_beta_qtype, true);
            }
            register_if_mxfp4(model_->output_proj(), model_->out_proj_qtype_);
            if (mx_native > 0) {
                IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
                wcache_.cutlass_mxfp4_bytes += mx_native_bytes;
                wcache_.use_mxfp4 = true;
                IMP_LOG_INFO("Native MXFP4 GGUF: %d tensors, %.2f MiB (direct → CUTLASS)",
                             mx_native, mx_native_bytes / (1024.0 * 1024.0));

                // Sync and check for errors from unpack kernels
                IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
                { cudaError_t e = cudaGetLastError();
                  if (e != cudaSuccess) IMP_LOG_ERROR("MXFP4 unpack error: %s", cudaGetErrorString(e)); }

                // Check if MXFP4 GEMV is available (linear_scales populated).
                // GDN models need FP16 fallback because GDN forward reads weights
                // directly (ssm_in, ssm_out, gdn_gate, etc.) — not through gemm_dispatch.
                bool force_fallback = (std::getenv("IMP_MXFP4_FP16_FALLBACK") != nullptr);
                bool has_gdn = (cfg.ssm_inner_size > 0);
                bool mxfp4_gemv_available = !force_fallback && !has_gdn;
                for (auto& [p, m] : wcache_.cutlass_mxfp4)
                    if (!m.linear_scales) { mxfp4_gemv_available = false; break; }

                if (mxfp4_gemv_available) {
                    IMP_LOG_INFO("MXFP4 GEMV: all %d weights have linear_scales, skipping FP16 fallback",
                                 mx_native);
                }

                // Dequant MXFP4 → FP16 for decode (only when MXFP4 GEMV not available).
                // Single bulk allocation to avoid CUDA heap fragmentation.
                size_t fp16_total = 0;
                if (!mxfp4_gemv_available) {
                for (auto& [p, m] : wcache_.cutlass_mxfp4)
                    if (!wcache_.fp16.count(p))
                        fp16_total += static_cast<size_t>(m.N) * m.K * sizeof(half);
                }

                void* d_fp16_bulk = nullptr;
                if (fp16_total > 0) {
                    cudaError_t ae = cudaMalloc(&d_fp16_bulk, fp16_total);
                    if (ae != cudaSuccess) {
                        IMP_LOG_ERROR("MXFP4 FP16 bulk alloc failed: %s (%.1f MiB)",
                                     cudaGetErrorString(ae), fp16_total / (1024.0*1024.0));
                        d_fp16_bulk = nullptr;
                    }
                }

                if (d_fp16_bulk) {
                    size_t offset = 0;
                    for (auto& [ptr, mw] : wcache_.cutlass_mxfp4) {
                        if (wcache_.fp16.count(ptr)) continue;
                        size_t fp16_bytes = static_cast<size_t>(mw.N) * mw.K * sizeof(half);
                        void* d_fp16 = static_cast<char*>(d_fp16_bulk) + offset;
                        offset += fp16_bytes;

                    // CPU-side dequant: download raw, dequant on CPU, upload FP16.
                    // GPU kernel has mysterious illegal memory access — using CPU as workaround.
                    {
                        int64_t N = mw.N, K = mw.K;
                        int bpr = static_cast<int>(K / 32);
                        size_t raw_bytes = static_cast<size_t>(N) * bpr * 17;
                        std::vector<uint8_t> h_raw(raw_bytes);
                        IMP_CUDA_CHECK_LOG(cudaMemcpy(h_raw.data(), ptr, raw_bytes, cudaMemcpyDeviceToHost));

                        std::vector<uint16_t> h_fp16(static_cast<size_t>(N) * K);  // raw FP16 bits
                        static const float e2m1[16] = {
                            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

                        for (int64_t r = 0; r < N; r++) {
                            for (int b = 0; b < bpr; b++) {
                                const uint8_t* blk = h_raw.data() + (r * bpr + b) * 17;
                                uint8_t ue8m0 = blk[16];
                                // UE8M0 → float
                                uint32_t fbits = static_cast<uint32_t>(ue8m0) << 23;
                                float scale;
                                memcpy(&scale, &fbits, sizeof(float));

                                int64_t base = r * K + b * 32;
                                for (int i = 0; i < 16; i++) {
                                    int lo = blk[i] & 0xF;
                                    int hi = (blk[i] >> 4) & 0xF;
                                    float v0 = e2m1[lo] * scale;
                                    float v1 = e2m1[hi] * scale;
                                    // Float→FP16 bit conversion
                                    __half h0 = __float2half(v0);
                                    __half h1 = __float2half(v1);
                                    memcpy(&h_fp16[base + i*2], &h0, 2);
                                    memcpy(&h_fp16[base + i*2+1], &h1, 2);
                                }
                            }
                        }
                        IMP_CUDA_CHECK_LOG(cudaMemcpy(d_fp16, h_fp16.data(), fp16_bytes, cudaMemcpyHostToDevice));
                    }
                    int64_t shape[2] = {mw.N, mw.K};
                    wcache_.fp16[ptr] = Tensor(d_fp16, DType::FP16, 2, shape, true);
                    }
                }  // end if (d_fp16_bulk)

                if (fp16_total > 0) {
                    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
                    { cudaError_t e = cudaGetLastError();
                      if (e != cudaSuccess) IMP_LOG_ERROR("MXFP4 dequant kernel error: %s", cudaGetErrorString(e)); }
                    IMP_LOG_INFO("MXFP4 decode fallback: dequant → FP16 cache %.2f MiB",
                                 fp16_total / (1024.0 * 1024.0));

                    // Replace model weight tensor pointers with FP16 data.
                    // This ensures ALL code paths (GEMV, direct gemm, etc.) see
                    // valid FP16 data instead of raw MXFP4 blocks.
                    auto replace_weight = [&](Tensor& w, GGMLQuantType& qt) {
                        auto it = wcache_.fp16.find(w.data);
                        if (it != wcache_.fp16.end() && qt == GGMLQuantType::MXFP4) {
                            w = it->second;
                            qt = GGMLQuantType::F16;
                        }
                    };
                    for (int i = 0; i < cfg.n_layers; i++) {
                        TransformerLayer& L = const_cast<Model*>(model_)->layer(i);
                        replace_weight(L.wq, L.wq_qtype);
                        replace_weight(L.wk, L.wk_qtype);
                        replace_weight(L.wv, L.wv_qtype);
                        replace_weight(L.wo, L.wo_qtype);
                        replace_weight(L.w_up, L.w_up_qtype);
                        replace_weight(L.w_gate, L.w_gate_qtype);
                        replace_weight(L.w_down, L.w_down_qtype);
                        // GDN-specific weights (Qwen3.5)
                        replace_weight(L.ssm_in, L.ssm_in_qtype);
                        replace_weight(L.ssm_out, L.ssm_out_qtype);
                        replace_weight(L.gdn_gate, L.gdn_gate_qtype);
                        replace_weight(L.gdn_alpha, L.gdn_alpha_qtype);
                        replace_weight(L.gdn_beta, L.gdn_beta_qtype);
                    }
                    replace_weight(const_cast<Model*>(model_)->out_proj_,
                                   const_cast<Model*>(model_)->out_proj_qtype_);
                    IMP_LOG_INFO("MXFP4 → FP16: replaced %d weight tensor pointers", (int)wcache_.fp16.size());
                }
            }
        }

        // Cache MoE expert weights — done after FP16 free so mode 2 has full budget
        int nvfp4_moe_count = 0;
        size_t nvfp4_moe_total = 0;
        size_t moe_budget;
        if (wcache_.nvfp4_decode_mode == 2) {
            size_t free_mem = 0, total_mem = 0;
            IMP_CUDA_CHECK_LOG(cudaMemGetInfo(&free_mem, &total_mem));
            constexpr size_t kMoeReserve = 128ULL * 1024 * 1024;
            moe_budget = (free_mem > kMoeReserve) ? (free_mem - kMoeReserve) : 0;
        } else {
            moe_budget = (remaining_budget > wcache_.nvfp4_bytes)
                         ? (remaining_budget - wcache_.nvfp4_bytes) : 0;
        }
        bool moe_budget_exhausted = false;

        auto cache_moe_expert_nvfp4 = [&](const Tensor& packed, GGMLQuantType qtype) {
            if (!packed.data) return;
            if (!nvfp4_beneficial(qtype)) return;
            if (wcache_.nvfp4_moe.count(packed.data)) return;
            if (moe_budget_exhausted) return;
            if (!packed.on_device) return;
            if (packed.ndim < 3) return;

            int ne = static_cast<int>(packed.shape[0]);
            int rows = static_cast<int>(packed.shape[1]);
            int cols = static_cast<int>(packed.shape[2]);
            if (cols % 16 != 0) return;
            if (!dequant_gpu_supported(qtype) || !qscratch_.dequant) return;

            size_t nvfp4_bytes = static_cast<size_t>(ne) * rows * cols / 2 +
                                 static_cast<size_t>(ne) * rows * cols / 16 +
                                 static_cast<size_t>(ne) * sizeof(float);

            if (nvfp4_moe_total + nvfp4_bytes > moe_budget) {
                moe_budget_exhausted = true;
                IMP_LOG_INFO("NVFP4 MoE cache: VRAM budget reached after %d MoE tensors "
                             "(%.1f / %.1f MiB)", nvfp4_moe_count,
                             nvfp4_moe_total / (1024.0 * 1024.0),
                             moe_budget / (1024.0 * 1024.0));
                return;
            }

            NvFP4MoEQuantResult result;
            quantize_packed_experts_to_nvfp4(
                packed.data, qtype, ne, rows, cols,
                qscratch_.dequant, result, stream);

            wcache_.nvfp4_moe[packed.data] = result;
            nvfp4_moe_total += nvfp4_bytes;
            nvfp4_moe_count++;
        };

        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            cache_moe_expert_nvfp4(L.expert_gate_packed, L.expert_gate_qtype);
            cache_moe_expert_nvfp4(L.expert_up_packed,   L.expert_up_qtype);
            cache_moe_expert_nvfp4(L.expert_down_packed,  L.expert_down_qtype);
        }

        if (nvfp4_moe_count > 0) {
            wcache_.nvfp4_moe_bytes = nvfp4_moe_total;
            IMP_LOG_INFO("NVFP4 MoE cache: %d tensors, %.2f MiB",
                         nvfp4_moe_count, nvfp4_moe_total / (1024.0 * 1024.0));
        } else if (wcache_.nvfp4.empty()) {
            IMP_LOG_INFO("NVFP4 decode: no eligible weights found (all ≤ 4.5 bits/elem)");
        }
    }

    // --- Phase 3c (standalone): Native MXFP4 GGUF when NVFP4 decode is disabled ---
    // This runs for GDN models where NVFP4 is auto-disabled but weights are MXFP4.
    if (wcache_.nvfp4_decode_mode == 0 && wcache_.cutlass_mxfp4.empty() &&
        cutlass_sm120_mxfp4_available()) {
        // Check if any layer has MXFP4 weights
        bool has_mxfp4 = false;
        for (int i = 0; i < cfg.n_layers && !has_mxfp4; i++) {
            const auto& L = model_->layer(i);
            if (L.wq_qtype == GGMLQuantType::MXFP4 || L.w_gate_qtype == GGMLQuantType::MXFP4 ||
                L.ssm_in_qtype == GGMLQuantType::MXFP4 || L.ssm_out_qtype == GGMLQuantType::MXFP4)
                has_mxfp4 = true;
        }
        if (has_mxfp4) {
            // Allocate MXFP4 scratch
            int max_k = 0, max_n = 0;
            for (int i = 0; i < cfg.n_layers; i++) {
                const auto& L = model_->layer(i);
                auto check = [&](const Tensor& w) {
                    if (w.data && w.ndim >= 2) {
                        max_n = std::max(max_n, (int)w.shape[0]);
                        max_k = std::max(max_k, (int)w.shape[1]);
                    }
                };
                check(L.wq); check(L.wk); check(L.w_gate); check(L.w_down);
                check(L.ssm_in); check(L.ssm_out); check(L.gdn_gate);
            }
            if (max_k > 0 && !qscratch_.mxfp4_act_sf) {
                qscratch_.mxfp4_act_sf_size = cutlass_mxfp4_sf_size(max_tokens_, max_k);
                qscratch_.mxfp4_act_sf = vram_alloc(vram_alloc_, qscratch_.mxfp4_act_sf_size, "mxfp4_act_sf");
                if (!qscratch_.cutlass_act_data) {
                    qscratch_.cutlass_act_data_size = static_cast<size_t>(max_tokens_) * (max_k / 2);
                    qscratch_.cutlass_act_data = vram_alloc(vram_alloc_, qscratch_.cutlass_act_data_size, "cutlass_act_data");
                }
            }
            // FIRST: dequant alpha/beta to FP16 BEFORE in-place unpack
            // (dequant_mxfp4_to_fp16 reads raw 17-byte blocks which get compacted by unpack)
            {
                size_t fp16_total = 0;
                struct SmallWeight { const void* ptr; int64_t N, K; };
                std::vector<SmallWeight> small_weights;
                for (int i = 0; i < cfg.n_layers; i++) {
                    const auto& L = model_->layer(i);
                    auto collect = [&](const Tensor& w, GGMLQuantType qt) {
                        if (qt != GGMLQuantType::MXFP4 || !w.data) return;
                        small_weights.push_back({w.data, w.shape[0], w.shape[1]});
                        fp16_total += static_cast<size_t>(w.shape[0]) * w.shape[1] * sizeof(half);
                    };
                    collect(L.gdn_alpha, L.gdn_alpha_qtype);
                    collect(L.gdn_beta, L.gdn_beta_qtype);
                }
                if (fp16_total > 0) {
                    void* d_fp16_bulk = nullptr;
                    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_fp16_bulk, fp16_total));
                    if (d_fp16_bulk) {
                        size_t offset = 0;
                        for (auto& sw : small_weights) {
                            size_t bytes = static_cast<size_t>(sw.N) * sw.K * sizeof(half);
                            void* d_fp16 = static_cast<char*>(d_fp16_bulk) + offset;
                            offset += bytes;
                            dequant_mxfp4_to_fp16(sw.ptr, sw.N, sw.K, d_fp16, stream);
                            int64_t shape[2] = {sw.N, sw.K};
                            wcache_.fp16[sw.ptr] = Tensor(d_fp16, DType::FP16, 2, shape, true);
                        }
                        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
                        IMP_LOG_INFO("MXFP4 → FP16 (alpha/beta): %.2f MiB (%d tensors)",
                                     fp16_total / (1024.0 * 1024.0), (int)small_weights.size());
                        for (int i = 0; i < cfg.n_layers; i++) {
                            TransformerLayer& L = const_cast<Model*>(model_)->layer(i);
                            auto replace = [&](Tensor& w, GGMLQuantType& qt) {
                                auto it = wcache_.fp16.find(w.data);
                                if (it != wcache_.fp16.end() && qt == GGMLQuantType::MXFP4) {
                                    w = it->second; qt = GGMLQuantType::F16;
                                }
                            };
                            replace(L.gdn_alpha, L.gdn_alpha_qtype);
                            replace(L.gdn_beta, L.gdn_beta_qtype);
                        }
                    }
                }
            }

            // THEN: register + unpack MXFP4 weights (in-place compaction)
            int mx_count = 0;
            auto register_mx = [&](const Tensor& w, GGMLQuantType qt, bool is_attn) {
                if (qt != GGMLQuantType::MXFP4 || !w.data || !w.on_device) return;
                if (w.ndim < 2 || w.shape[1] % 32 != 0) return;
                if (wcache_.cutlass_mxfp4.count(w.data)) return;
                CutlassMxFP4Weight mw;
                if (unpack_mxfp4_gguf(w.data, w.shape[0], w.shape[1], mw, stream)) {
                    mw.hadamard_bs = is_attn ? cfg.mxfp4_hadamard_attn : cfg.mxfp4_hadamard_ffn;
                    wcache_.cutlass_mxfp4[w.data] = mw;
                    mx_count++;
                }
            };
            for (int i = 0; i < cfg.n_layers; i++) {
                const auto& L = model_->layer(i);
                register_mx(L.wq, L.wq_qtype, true);
                register_mx(L.wk, L.wk_qtype, true);
                register_mx(L.wv, L.wv_qtype, true);
                register_mx(L.wo, L.wo_qtype, true);
                register_mx(L.w_up, L.w_up_qtype, false);
                register_mx(L.w_gate, L.w_gate_qtype, false);
                register_mx(L.w_down, L.w_down_qtype, false);
                register_mx(L.ssm_in, L.ssm_in_qtype, true);
                register_mx(L.ssm_out, L.ssm_out_qtype, true);
                register_mx(L.gdn_gate, L.gdn_gate_qtype, true);
                register_mx(L.gdn_alpha, L.gdn_alpha_qtype, true);
                register_mx(L.gdn_beta, L.gdn_beta_qtype, true);
            }
            register_mx(model_->output_proj(), model_->out_proj_qtype_, true);
            if (mx_count > 0) {
                IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
                wcache_.use_mxfp4 = true;

                // In-place unpack: raw blocks are compacted to [N, K/2] within the
                // SAME buffer. No separate data allocation, no free needed.
                // The raw buffer tail (scale bytes) is wasted (~6% overhead) but
                // avoids the 50% peak VRAM spike of out-of-place unpack.
                IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
                { cudaError_t e = cudaGetLastError();
                  if (e != cudaSuccess) IMP_LOG_ERROR("MXFP4 registration CUDA error: %s", cudaGetErrorString(e)); }
                IMP_LOG_INFO("Native MXFP4 GGUF (standalone): %d tensors registered (in-place)", mx_count);

                // Alpha/beta FP16 dequant was done BEFORE in-place unpack (above).
            }
        }
    }

    // Build WeightRegistry from wcache_ contents (phase-2 shim).
    registry_.clear();
    // Explicit kind overrides t.kind which is UNKNOWN after weight_upload.cu
    // creates fresh Tensor descriptors (TensorKind is not preserved through
    // the upload code paths). Phase 5 plan-driven allocation requires kind to
    // be correct, so we pass it explicitly from the field position.
    auto register_tensor = [&](const Tensor& t, TensorKind kind) -> TensorID {
        if (!t.data) return kInvalidTensorID;
        StorageTier tier = infer_tier_from_wcache(wcache_, t.data);
        if (tier == StorageTier::Undefined) return kInvalidTensorID;
        TensorID id = registry_.reserve(kind, t.shape[0], t.ndim > 1 ? t.shape[1] : 1);
        auto& h = registry_.handle(id);
        h.primary_tier = tier;
        borrow_payload_from_wcache(h, wcache_, t.data);
        return id;
    };

    for (int i = 0; i < cfg.n_layers; ++i) {
        // const_cast: model_ is const Model* but the *_id fields are metadata
        // stamped exactly once here during load — safe to mutate.
        auto& L = const_cast<Model*>(model_)->layer(i);
        L.wq_id       = register_tensor(L.wq,       TensorKind::WQ);
        L.wk_id       = register_tensor(L.wk,       TensorKind::WK);
        L.wv_id       = register_tensor(L.wv,       TensorKind::WV);
        L.wo_id       = register_tensor(L.wo,       TensorKind::WO);
        L.w_gate_id   = register_tensor(L.w_gate,   TensorKind::W_GATE);
        L.w_up_id     = register_tensor(L.w_up,     TensorKind::W_UP);
        L.w_down_id   = register_tensor(L.w_down,   TensorKind::W_DOWN);
        // Shared-expert FFN — matches StoragePlanner enumeration from PR #38.
        L.w_gate_shared_id = register_tensor(L.w_gate_shared, TensorKind::W_GATE);
        L.w_up_shared_id   = register_tensor(L.w_up_shared,   TensorKind::W_UP);
        L.w_down_shared_id = register_tensor(L.w_down_shared, TensorKind::W_DOWN);
        L.ssm_in_id   = register_tensor(L.ssm_in,   TensorKind::SSM_IN);
        L.ssm_out_id  = register_tensor(L.ssm_out,  TensorKind::SSM_OUT);
        L.gdn_gate_id = register_tensor(L.gdn_gate, TensorKind::GDN_GATE);

        // Per-expert TensorIDs (Task 3.4)
        const int ne_layer = static_cast<int>(L.expert_w_gate.size());
        const int ne_up    = static_cast<int>(L.expert_w_up.size());
        const int ne_down  = static_cast<int>(L.expert_w_down.size());
        L.expert_gate_ids.assign(ne_layer, kInvalidTensorID);
        L.expert_up_ids.assign(ne_up,    kInvalidTensorID);
        L.expert_down_ids.assign(ne_down, kInvalidTensorID);
        for (int e = 0; e < ne_layer; ++e) L.expert_gate_ids[e] = register_tensor(L.expert_w_gate[e], TensorKind::EXPERT_GATE);
        for (int e = 0; e < ne_up;    ++e) L.expert_up_ids[e]   = register_tensor(L.expert_w_up[e],   TensorKind::EXPERT_UP);
        for (int e = 0; e < ne_down;  ++e) L.expert_down_ids[e] = register_tensor(L.expert_w_down[e], TensorKind::EXPERT_DOWN);
        L.moe_gate_id           = register_tensor(L.moe_gate,             TensorKind::ROUTER);
        L.shared_expert_gate_id = register_tensor(L.shared_expert_gate_inp, TensorKind::SHARED_EXPERT_GATE);

        // Borrow nvfp4_moe pointers for packed 3D expert NVFP4 cache (Task 3.4)
        {
            auto it = wcache_.nvfp4_moe.find(L.expert_gate_packed.data);
            L.nvfp4_moe_gate_ptr = (it != wcache_.nvfp4_moe.end()) ? &it->second : nullptr;
        }
        {
            auto it = wcache_.nvfp4_moe.find(L.expert_up_packed.data);
            L.nvfp4_moe_up_ptr = (it != wcache_.nvfp4_moe.end()) ? &it->second : nullptr;
        }
        {
            auto it = wcache_.nvfp4_moe.find(L.expert_down_packed.data);
            L.nvfp4_moe_down_ptr = (it != wcache_.nvfp4_moe.end()) ? &it->second : nullptr;
        }
        // Borrow fp16 pointers for packed expert tensors (Task 3.4)
        {
            auto it = wcache_.fp16.find(L.expert_gate_packed.data);
            L.fp16_packed_gate_cache = (it != wcache_.fp16.end()) ? &it->second : nullptr;
        }
        {
            auto it = wcache_.fp16.find(L.expert_up_packed.data);
            L.fp16_packed_up_cache = (it != wcache_.fp16.end()) ? &it->second : nullptr;
        }
        {
            auto it = wcache_.fp16.find(L.expert_down_packed.data);
            L.fp16_packed_down_cache = (it != wcache_.fp16.end()) ? &it->second : nullptr;
        }
    }
    // Register model-level (non-layer) tensors.
    const_cast<Model*>(model_)->out_proj_id = register_tensor(model_->output_proj(), TensorKind::LM_HEAD);
    const_cast<Model*>(model_)->tok_emb_id  = register_tensor(model_->token_embedding(), TensorKind::TOK_EMBED);

    // Register fused KV / gate+up overlays. Layer-keyed (not pointer-keyed)
    // because a fused tensor is built fresh — the source pointers (wk, wv) are
    // the *unfused* weights and don't appear in any per-tensor wcache_ map.
    auto register_fused = [&](TensorKind kind, const Tensor& t) -> TensorID {
        if (!t.data) return kInvalidTensorID;
        TensorID id = registry_.reserve(kind, t.shape[0], t.ndim > 1 ? t.shape[1] : 1);
        auto& h = registry_.handle(id);
        h.primary_tier = StorageTier::FP16;
        h.payload.fp16.data = static_cast<half*>(t.data);
        return id;
    };
    for (int i = 0; i < cfg.n_layers; ++i) {
        auto& L = const_cast<Model*>(model_)->layer(i);
        if (auto it = wcache_.fused_kv.find(i); it != wcache_.fused_kv.end()) {
            L.fused_kv_id = register_fused(TensorKind::FUSED_KV, it->second);
        }
        if (auto it = wcache_.fused_gate_up.find(i); it != wcache_.fused_gate_up.end()) {
            L.fused_gate_up_id = register_fused(TensorKind::FUSED_GATE_UP, it->second);
        }
    }

    IMP_LOG_INFO("WeightRegistry populated with %zu handles (phase-2 shim)",
                 registry_.size());

    // Phase 4 (Option C) overlay diagnostic: report ideal vs actual overlay
    // population. The plan enumerates every quantize-able tensor at its
    // preferred tier ("ideal overlay"). The registry tracks tensors actually
    // cached by the runtime ("actual overlay"). Native GGUF blocks (Q4_K_M,
    // Q5_K_M, Q6_K, Q8_0, MXFP4) stay as mmap'd `Model::gpu_allocations_`
    // and are dequantized per kernel call — they bypass the overlay layer
    // entirely, so the diff between plan and registry is informational, not
    // an error.
    {
        StoragePlan ideal_plan = plan_storage(*model_, cfg, hints_);
        size_t plan_overlay = 0;
        size_t plan_fp16 = 0, plan_fp8 = 0, plan_nvfp4 = 0;
        size_t plan_cutlass_nvfp4 = 0, plan_mxfp4 = 0, plan_fp32 = 0;
        for (const auto& e : ideal_plan.entries) {
            switch (e.tier) {
                case StorageTier::FP16:          ++plan_fp16; ++plan_overlay; break;
                case StorageTier::FP8:           ++plan_fp8; ++plan_overlay; break;
                case StorageTier::NVFP4:         ++plan_nvfp4; ++plan_overlay; break;
                case StorageTier::CUTLASS_NVFP4: ++plan_cutlass_nvfp4; ++plan_overlay; break;
                case StorageTier::MXFP4:         ++plan_mxfp4; ++plan_overlay; break;
                case StorageTier::FP32:          ++plan_fp32; break;
                case StorageTier::Undefined:     break;
            }
        }
        size_t registry_count = registry_.size();
        IMP_LOG_INFO("Phase-4 overlay: registry=%zu cached / plan-ideal=%zu "
                     "(uncached %zu remain as native GGUF blocks)",
                     registry_count, plan_overlay,
                     plan_overlay > registry_count ? plan_overlay - registry_count : 0);
        IMP_LOG_INFO("Phase-4 plan-ideal tiers: fp16=%zu fp8=%zu nvfp4=%zu "
                     "cutlass_nvfp4=%zu mxfp4=%zu fp32=%zu",
                     plan_fp16, plan_fp8, plan_nvfp4, plan_cutlass_nvfp4,
                     plan_mxfp4, plan_fp32);
        IMP_LOG_INFO("Phase-4 wcache actual: fp16=%zu fp8=%zu nvfp4=%zu "
                     "cutlass_nvfp4=%zu cutlass_mxfp4=%zu nvfp4_moe=%zu "
                     "fused_kv=%zu fused_gate_up=%zu",
                     wcache_.fp16.size(), wcache_.fp8.size(), wcache_.nvfp4.size(),
                     wcache_.cutlass_nvfp4.size(), wcache_.cutlass_mxfp4.size(),
                     wcache_.nvfp4_moe.size(), wcache_.fused_kv.size(),
                     wcache_.fused_gate_up.size());
    }
}

} // namespace imp
