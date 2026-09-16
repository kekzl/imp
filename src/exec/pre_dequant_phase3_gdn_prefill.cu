// Phase 3d (gemm.nvfp4_gdn_proj_prefill): NVFP4 prefill copies of the full-precision GDN
// projections. Native-NVFP4 hybrids whose recipe keeps in_proj/gate/out_proj in BF16
// (Qwen3.6-35B) run those GEMMs through cuBLAS FP16 at prefill, ~320 TFLOPS on sm_120 for
// M=512..4096 (35-40% of the FP16-accumulate peak, tile choice and merged N measured flat).
// The CUTLASS block-scaled NVFP4 GEMM runs the same rows at 65-80% of the FP4 peak. The copy
// is registered in its own map (wcache_.cutlass_nvfp4_prefill): the F16 source stays the
// primary tier for M=1 (FP8 sidecar / FP16 GEMV) and the batched decode rows (M<=32), so
// only true prefill changes numerics (W4A4 on the state-feeding projections, as the vllm
// recipe of Qwen3.8-27B ships natively).

#include "compute/gemm_cutlass_sm120.h"
#include "core/dispatch_policy.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "exec/quant_pipeline.h"
#include "quant/nvfp4_quant.h"
#include <cuda_runtime.h>
#include <string>

namespace imp {

void QuantPipeline::nvfp4_prefill_cache_gdn_projections_(const ModelConfig& cfg, cudaStream_t stream) {
    const std::string& sel = dispatch_policy().gemm.nvfp4_gdn_proj_prefill;
    if (sel.empty() || sel == "false" || sel == "off" || sel == "0")
        return;
    const bool all = (sel == "true" || sel == "all" || sel == "1");
    auto picked = [&](const char* name) {
        if (all)
            return true;
        const std::string list = "," + sel + ",";
        return list.find("," + std::string(name) + ",") != std::string::npos;
    };
    const bool want_in = picked("in"), want_gate = picked("gate"), want_out = picked("out");
    if (!want_in && !want_gate && !want_out) {
        IMP_LOG_WARN(
            "gemm.nvfp4_gdn_proj_prefill=\"%s\": expected false|all|<in,gate,out list>, flag ignored",
            sel.c_str());
        return;
    }
    if (!cutlass_sm120_nvfp4_available()) {
        IMP_LOG_WARN("gemm.nvfp4_gdn_proj_prefill: CUTLASS sm_120 NVFP4 unavailable, flag ignored");
        return;
    }

    int n_weights = 0;
    size_t bytes = 0;
    auto quantize_one = [&](const Tensor& w) {
        if (!w.data || !w.on_device || w.ndim != 2)
            return;
        if (w.qtype != QType::F16 && w.qtype != QType::BF16)
            return;  // native NVFP4 / FP8 projections already have their own prefill route
        const int rows = static_cast<int>(w.shape[0]);
        const int cols = static_cast<int>(w.shape[1]);
        if (cols % 64 != 0 || rows % 16 != 0)
            return;
        if (wcache_->cutlass_nvfp4_prefill.count(w.data))
            return;

        Tensor fp16_view(w.data, QType::F16, 2, w.shape, /*on_device=*/true);
        const float tscale = calibrate_nvfp4_scales(fp16_view, stream);
        NvFP4QuantResult result;
        quantize_fp16_to_nvfp4_with_scale(fp16_view, tscale, result, stream);
        result.tensor_scale = tscale;
        result.N = rows;
        result.K = cols;

        CutlassNvFP4Weight cw;
        convert_nvfp4_to_cutlass(result, cw, stream);  // owns the SfAtom buffer, borrows packed data
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
        // The linear micro-scales only feed the SfAtom conversion; no GEMV reads this copy.
        NvFP4QuantResult micro;
        micro.micro_scales = result.micro_scales;
        free_nvfp4_result(micro);
        result.micro_scales = nullptr;
        wcache_->cutlass_nvfp4_prefill_src.push_back(result);
        wcache_->cutlass_nvfp4_prefill[w.data] = cw;
        n_weights++;
        bytes += static_cast<size_t>(rows) * cols / 2 + cw.sf_bytes;
    };

    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& L = model_->layer(i);
        if (!L.ssm_in.data || !L.gdn_gate.data)
            continue;  // GDN layers only (Mamba2 pure-SSM layers keep the FP16 path)
        if (want_in)
            quantize_one(L.ssm_in);
        if (want_gate)
            quantize_one(L.gdn_gate);
        if (want_out)
            quantize_one(L.ssm_out);
    }

    wcache_->cutlass_nvfp4_prefill_bytes = bytes;

    if (n_weights > 0)
        IMP_LOG_INFO(
            "GDN NVFP4 prefill: %d F16 in/gate/out projections -> NVFP4 (%.1f MiB), M>32 rows on "
            "the CUTLASS GEMM",
            n_weights, bytes / (1024.0 * 1024.0));
    else
        IMP_LOG_INFO("GDN NVFP4 prefill: no eligible F16 GDN projection (flag on, model has none)");
}

}  // namespace imp
