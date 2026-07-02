// Pre-dequant Phase 3c: standalone MXFP4.
// Handles MXFP4-source GGUF models when the NVFP4 decode pipeline
// (Phase 3) is disabled. Dequantizes small alpha/beta tensors to FP16
// (must happen BEFORE in-place unpack, which compacts raw blocks),
// then registers + in-place unpacks the bulk MXFP4 weights into the
// CUTLASS sm_120 MXFP4 cache.
//
// Extracted from executor_pre_dequant.cu in Phase 3 of the architecture
// refactor roadmap. This is the final extraction — after this PR,
// executor_pre_dequant.cu is the pure orchestrator.
//
// LEGACY / MAINTENANCE MODE (2026-05-24): MXFP4 is supported but not the
// dev priority. NVFP4 + SafeTensors is where the hero models live. Ship
// cleanup fixes here (load errors, missing pointer replaces, resource
// leaks) and move on — don't chase residual output-quality bugs on
// community MXFP4 quants without an external reference engine
// (llama.cpp / HF Transformers) to compare against. See memory note
// `feedback_gguf_mxfp4_legacy_2026_05_24` for the full rule.

#include "exec/executor.h"
#include "exec/quant_pipeline.h"
#include "exec/pre_dequant_internal.h"
#include "compute/gemm_cutlass_mxfp4_sm120.h"
#include "core/logging.h"
#include "memory/vram_allocator.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <vector>

namespace imp {

void QuantPipeline::pre_dequant_phase3c_standalone_mxfp4_(
    const ModelConfig& cfg, cudaStream_t stream) {
    if (!(wcache_->nvfp4_decode_mode == 0 && wcache_->cutlass_mxfp4.empty() &&
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
        if (max_k > 0 && !qscratch_->mxfp4_act_sf) {
            qscratch_->mxfp4_act_sf_size = cutlass_mxfp4_sf_size(max_tokens_, max_k);
            qscratch_->mxfp4_act_sf = vram_alloc(vram_alloc_, qscratch_->mxfp4_act_sf_size, "mxfp4_act_sf");
            if (!qscratch_->cutlass_act_data) {
                qscratch_->cutlass_act_data_size = static_cast<size_t>(max_tokens_) * (max_k / 2);
                qscratch_->cutlass_act_data = vram_alloc(vram_alloc_, qscratch_->cutlass_act_data_size,
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
                    // Track bulk for shutdown cleanup (same pattern as Phase 3).
                    // Phase 3 and Phase 3c are mutually exclusive in practice
                    // (gated on `nvfp4_decode_mode`), so only one writes here.
                    wcache_->fp16_bulk_data = d_fp16_bulk;
                    wcache_->fp16_bulk_data_size = fp16_total;
                    size_t offset = 0;
                    for (auto& sw : small_weights) {
                        size_t bytes = static_cast<size_t>(sw.N) * sw.K * sizeof(half);
                        void* d_fp16 = static_cast<char*>(d_fp16_bulk) + offset;
                        offset += bytes;
                        dequant_mxfp4_to_fp16(sw.ptr, sw.N, sw.K, d_fp16, stream);
                        int64_t shape[2] = {sw.N, sw.K};
                        wcache_->fp16[sw.ptr] = Tensor(d_fp16, QType::F16, 2, shape, true);
                    }
                    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
                    IMP_LOG_INFO("MXFP4 → FP16 (alpha/beta): %.2f MiB (%d tensors)",
                                 fp16_total / (1024.0 * 1024.0), (int)small_weights.size());
                    for (int i = 0; i < cfg.n_layers; i++) {
                        TransformerLayer& L = const_cast<Model*>(model_)->layer(i);
                        auto replace = [&](Tensor& w, QType& qt) {
                            auto it = wcache_->fp16.find(w.data);
                            if (it != wcache_->fp16.end() && qt == QType::MXFP4) {
                                w = it->second;
                                qt = QType::F16;
                                // The model tensor now points at EXECUTOR-owned
                                // cache memory (fp16_bulk_data) — a second
                                // engine on this handle would read it dangling
                                // after this executor's teardown (#830).
                                const_cast<Model*>(model_)->mark_sources_consumed();
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
            if (wcache_->cutlass_mxfp4.count(w.data))
                return;
            CutlassMxFP4Weight mw;
            if (unpack_mxfp4_gguf(w.data, w.shape[0], w.shape[1], mw, stream)) {
                mw.hadamard_bs = is_attn ? cfg.mxfp4_hadamard_attn : cfg.mxfp4_hadamard_ffn;
                wcache_->cutlass_mxfp4[w.data] = mw;
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
            wcache_->use_mxfp4 = true;

            // In-place unpack: raw blocks are compacted to [N, K/2] within the
            // SAME buffer. No separate data allocation, no free needed.
            // The raw buffer tail (scale bytes) is wasted (~6% overhead) but
            // avoids the 50% peak VRAM spike of out-of-place unpack.
            //
            // The compaction is DESTRUCTIVE: the model's source buffers no
            // longer hold GGUF raw MXFP4 blocks, so a second engine on this
            // model handle cannot re-run this unpack (it would read the
            // already-compacted layout as raw blocks → illegal access, #830).
            // Mark the model so Engine::init rejects a second engine cleanly.
            const_cast<Model*>(model_)->mark_sources_consumed();
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
