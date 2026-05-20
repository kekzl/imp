#include "exec/executor.h"
#include "exec/executor_kernels.h"
#include "exec/executor_helpers.h"
#include "exec/pre_dequant_internal.h"
#include "compute/gemm.h"
#include "compute/gemm_cutlass_sm120.h"
#include "quant/dequant_gpu.h"
#include "quant/fp8_quant.h"
#include "quant/nvfp4_gemm.h"
#include "core/logging.h"
#include "memory/vram_allocator.h"
#include "runtime/storage_planner.h"
#include "runtime/config.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <unordered_set>

namespace imp {

using imp::pre_dequant_internal::borrow_payload_from_wcache;
using imp::pre_dequant_internal::create_fused_weight_pair;
using imp::pre_dequant_internal::deduct_budget;
using imp::pre_dequant_internal::for_each_dense_weight;
using imp::pre_dequant_internal::infer_tier_from_wcache;
using imp::pre_dequant_internal::nvfp4_beneficial;

// ---------------------------------------------------------------------------
// Pre-dequantize quantized weights to FP16 on GPU
// ---------------------------------------------------------------------------

void GraphExecutor::pre_dequant_weights(cudaStream_t stream, const VRAMBudget& budget) {
    if (!initialized_ || !model_)
        return;
    // Skip all weight caching for debugging numerical precision issues

    const auto& cfg = model_->config();

    // Compute effective cache budget from free VRAM minus reserve.
    // This preserves the existing per-phase budget tracking while the VRAMBudget
    // struct controls strategy-level decisions (which phases to skip).
    size_t free_vram = 0, total_vram = 0;
    IMP_CUDA_CHECK_LOG(cudaMemGetInfo(&free_vram, &total_vram));
    // Reserve at least 10% of total VRAM as headroom to avoid shared/system
    // memory fallback on WSL2 (not visible via nvidia-smi).
    size_t min_reserve = std::max(budget.reserve_bytes, total_vram / 10);
    size_t remaining_budget = (free_vram > min_reserve) ? (free_vram - min_reserve) : 0;

    // Phase 4.2: run StoragePlanner for diagnostic purposes.
    // The plan output is NOT used to drive allocation yet — the existing legacy code
    // path still decides what to allocate. Log discrepancies between the plan and
    // the legacy decisions so we can catch bugs before Phase 4.4+ flips to
    // plan-driven allocation. Actual storage ownership flip happens in Phase 5.
    {
        hints_.vram_budget_bytes = remaining_budget;
        StoragePlan diag_plan = plan_storage(*model_, cfg, hints_);
        if (diag_plan.failed) {
            IMP_LOG_WARN("StoragePlanner (diagnostic): plan failed — %s", diag_plan.failure_reason.c_str());
        } else {
            IMP_LOG_INFO("StoragePlanner (diagnostic): %zu entries, projected VRAM %.2f MiB",
                         diag_plan.entries.size(), diag_plan.projected_vram_bytes / (1024.0 * 1024.0));
        }
    }

    // --- Phase 0: Promote NVFP4 pre-quantized weights to Tensor sidecars ---
    // (body extracted to pre_dequant_phase0_promote_nvfp4_sidecars_)
    pre_dequant_phase0_promote_nvfp4_sidecars_(cfg, stream);

    // --- Phase 0b: register prequant-promoted NVFP4 weights in CUTLASS cache ---
    // (body extracted to pre_dequant_phase0b_register_cutlass_nvfp4_)
    pre_dequant_phase0b_register_cutlass_nvfp4_(cfg, stream);

    // --- Phase 1: FP16 weight cache + fused KV + fused gate+up (extracted) ---
    pre_dequant_phase1_fp16_cache_(cfg, budget, remaining_budget, stream);

    // --- Phase 2: FP8 cache for uncached weights (extracted) ---
    pre_dequant_phase2_fp8_cache_(cfg, budget, remaining_budget, stream);

    // --- Phase 3: NVFP4 decode weight cache + 3b CUTLASS + 3c-native (extracted) ---
    pre_dequant_phase3_nvfp4_decode_(cfg, budget, remaining_budget, stream);


    // --- Phase 3c (standalone): Native MXFP4 GGUF when NVFP4 decode is disabled (extracted) ---
    pre_dequant_phase3c_standalone_mxfp4_(cfg, stream);

    // --- Phase 4: tensor registry + overlay diagnostic + NVFP4 device-args (extracted) ---
    pre_dequant_phase4_tensor_registry_(cfg, stream);
}


void GraphExecutor::pre_dequant_phase3c_standalone_mxfp4_(
    const ModelConfig& cfg, cudaStream_t stream) {
    if (!(wcache_.nvfp4_decode_mode == 0 && wcache_.cutlass_mxfp4.empty() &&
          cutlass_sm120_mxfp4_available()))
        return;
    // Check if any layer has MXFP4 weights
    bool has_mxfp4 = false;
    for (int i = 0; i < cfg.n_layers && !has_mxfp4; i++) {
        const auto& L = model_->layer(i);
        if (L.wq.qtype == QType::MXFP4 || L.w_gate.qtype == QType::MXFP4 ||
            L.ssm_in.qtype == QType::MXFP4 || L.ssm_out.qtype == QType::MXFP4)
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
            check(L.wq);
            check(L.wk);
            check(L.w_gate);
            check(L.w_down);
            check(L.ssm_in);
            check(L.ssm_out);
            check(L.gdn_gate);
        }
        if (max_k > 0 && !qscratch_.mxfp4_act_sf) {
            qscratch_.mxfp4_act_sf_size = cutlass_mxfp4_sf_size(max_tokens_, max_k);
            qscratch_.mxfp4_act_sf = vram_alloc(vram_alloc_, qscratch_.mxfp4_act_sf_size, "mxfp4_act_sf");
            if (!qscratch_.cutlass_act_data) {
                qscratch_.cutlass_act_data_size = static_cast<size_t>(max_tokens_) * (max_k / 2);
                qscratch_.cutlass_act_data = vram_alloc(vram_alloc_, qscratch_.cutlass_act_data_size,
                                                        "cutlass_act_data");
            }
        }
        // FIRST: dequant alpha/beta to FP16 BEFORE in-place unpack
        // (dequant_mxfp4_to_fp16 reads raw 17-byte blocks which get compacted by unpack)
        {
            size_t fp16_total = 0;
            struct SmallWeight {
                const void* ptr;
                int64_t N, K;
            };
            std::vector<SmallWeight> small_weights;
            for (int i = 0; i < cfg.n_layers; i++) {
                const auto& L = model_->layer(i);
                auto collect = [&](const Tensor& w, QType qt) {
                    if (qt != QType::MXFP4 || !w.data)
                        return;
                    small_weights.push_back({w.data, w.shape[0], w.shape[1]});
                    fp16_total += static_cast<size_t>(w.shape[0]) * w.shape[1] * sizeof(half);
                };
                collect(L.gdn_alpha, L.gdn_alpha.qtype);
                collect(L.gdn_beta, L.gdn_beta.qtype);
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
                        wcache_.fp16[sw.ptr] = Tensor(d_fp16, QType::F16, 2, shape, true);
                    }
                    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
                    IMP_LOG_INFO("MXFP4 → FP16 (alpha/beta): %.2f MiB (%d tensors)",
                                 fp16_total / (1024.0 * 1024.0), (int)small_weights.size());
                    for (int i = 0; i < cfg.n_layers; i++) {
                        TransformerLayer& L = const_cast<Model*>(model_)->layer(i);
                        auto replace = [&](Tensor& w, QType& qt) {
                            auto it = wcache_.fp16.find(w.data);
                            if (it != wcache_.fp16.end() && qt == QType::MXFP4) {
                                w = it->second;
                                qt = QType::F16;
                            }
                        };
                        replace(L.gdn_alpha, L.gdn_alpha.qtype);
                        replace(L.gdn_beta, L.gdn_beta.qtype);
                    }
                }
            }
        }

        // THEN: register + unpack MXFP4 weights (in-place compaction)
        int mx_count = 0;
        auto register_mx = [&](const Tensor& w, QType qt, bool is_attn) {
            if (qt != QType::MXFP4 || !w.data || !w.on_device)
                return;
            if (w.ndim < 2 || w.shape[1] % 32 != 0)
                return;
            if (wcache_.cutlass_mxfp4.count(w.data))
                return;
            CutlassMxFP4Weight mw;
            if (unpack_mxfp4_gguf(w.data, w.shape[0], w.shape[1], mw, stream)) {
                mw.hadamard_bs = is_attn ? cfg.mxfp4_hadamard_attn : cfg.mxfp4_hadamard_ffn;
                wcache_.cutlass_mxfp4[w.data] = mw;
                mx_count++;
            }
        };
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            register_mx(L.wq, L.wq.qtype, true);
            register_mx(L.wk, L.wk.qtype, true);
            register_mx(L.wv, L.wv.qtype, true);
            register_mx(L.wo, L.wo.qtype, true);
            register_mx(L.w_up, L.w_up.qtype, false);
            register_mx(L.w_gate, L.w_gate.qtype, false);
            register_mx(L.w_down, L.w_down.qtype, false);
            register_mx(L.ssm_in, L.ssm_in.qtype, true);
            register_mx(L.ssm_out, L.ssm_out.qtype, true);
            register_mx(L.gdn_gate, L.gdn_gate.qtype, true);
            register_mx(L.gdn_alpha, L.gdn_alpha.qtype, true);
            register_mx(L.gdn_beta, L.gdn_beta.qtype, true);
        }
        register_mx(model_->output_proj(), model_->out_proj_.qtype, true);
        if (mx_count > 0) {
            IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
            wcache_.use_mxfp4 = true;

            // In-place unpack: raw blocks are compacted to [N, K/2] within the
            // SAME buffer. No separate data allocation, no free needed.
            // The raw buffer tail (scale bytes) is wasted (~6% overhead) but
            // avoids the 50% peak VRAM spike of out-of-place unpack.
            IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
            {
                cudaError_t e = cudaGetLastError();
                if (e != cudaSuccess)
                    IMP_LOG_ERROR("MXFP4 registration CUDA error: %s", cudaGetErrorString(e));
            }
            IMP_LOG_INFO("Native MXFP4 GGUF (standalone): %d tensors registered (in-place)", mx_count);

            // Alpha/beta FP16 dequant was done BEFORE in-place unpack (above).
        }
    }
}

}  // namespace imp
