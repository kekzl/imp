// Phase 3d (gemm.nvfp4_gdn_proj_prefill, gemm.mxfp8_gdn_proj_prefill): tensor-core prefill
// copies of the full-precision GDN projections. Native-NVFP4 hybrids whose recipe keeps
// in_proj/gate/out_proj in BF16 (Qwen3.6-35B) run those GEMMs through cuBLAS FP16 at prefill,
// ~320 TFLOPS on sm_120 for M=512..4096 (35-40% of the FP16-accumulate peak, tile choice and
// merged N measured flat). The CUTLASS block-scaled GEMMs run the same rows at 65-80% of the
// FP4 peak (NVFP4, W4A4) or on the FP8 pipe (MXFP8, W8A8, E4M3 with UE8M0 per 32). The copies
// sit in their own maps (wcache_.cutlass_nvfp4_prefill / cutlass_mxfp8_prefill): the F16
// source stays the primary tier for M=1 (FP8 sidecar / FP16 GEMV) and the batched decode rows
// (M<=32), so only true prefill changes numerics. A projection named in both flags takes
// MXFP8 (checked first at dispatch).

#include "compute/gemm_cutlass_mxfp8_sm120.h"
#include "compute/gemm_cutlass_sm120.h"
#include "core/dispatch_policy.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "exec/executor_helpers.h"
#include "exec/quant_pipeline.h"
#include "quant/nvfp4_quant.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <string>

namespace imp {

namespace {

struct ProjSelection {
    bool in = false, gate = false, out = false;
    bool any() const { return in || gate || out; }
};

// "false"/"off"/"0" none, "true"/"all"/"1" every role, else a comma list of in|gate|out.
// A malformed list is reported once and treated as none.
ProjSelection parse_projection_selection(const std::string& sel, const char* flag) {
    ProjSelection s;
    if (sel.empty() || sel == "false" || sel == "off" || sel == "0")
        return s;
    if (sel == "true" || sel == "all" || sel == "1") {
        s.in = s.gate = s.out = true;
        return s;
    }
    const std::string list = "," + sel + ",";
    s.in = list.find(",in,") != std::string::npos;
    s.gate = list.find(",gate,") != std::string::npos;
    s.out = list.find(",out,") != std::string::npos;
    if (!s.any())
        IMP_LOG_WARN("%s=\"%s\": expected false|all|<in,gate,out list>, flag ignored", flag, sel.c_str());
    return s;
}

bool eligible_f16_projection(const Tensor& w) {
    return w.data && w.on_device && w.ndim == 2 && (w.qtype == QType::F16 || w.qtype == QType::BF16) &&
           w.shape[1] % 64 == 0 && w.shape[0] % 16 == 0;
}

}  // namespace

void QuantPipeline::nvfp4_prefill_cache_gdn_projections_(const ModelConfig& cfg, cudaStream_t stream) {
    const auto& gemm_cfg = dispatch_policy().gemm;
    ProjSelection nv = parse_projection_selection(gemm_cfg.nvfp4_gdn_proj_prefill,
                                                  "gemm.nvfp4_gdn_proj_prefill");
    ProjSelection mx = parse_projection_selection(gemm_cfg.mxfp8_gdn_proj_prefill,
                                                  "gemm.mxfp8_gdn_proj_prefill");
    if (!nv.any() && !mx.any())
        return;
    if (nv.any() && !cutlass_sm120_nvfp4_available()) {
        IMP_LOG_WARN("gemm.nvfp4_gdn_proj_prefill: CUTLASS sm_120 NVFP4 unavailable, flag ignored");
        nv = ProjSelection{};
    }
    if (mx.any() && !cutlass_sm120_mxfp8_available()) {
        IMP_LOG_WARN("gemm.mxfp8_gdn_proj_prefill: CUTLASS sm_120 MXFP8 unavailable, flag ignored");
        mx = ProjSelection{};
    }

    int n_nv = 0, n_mx = 0;
    size_t nv_bytes = 0, mx_bytes = 0;
    int64_t mx_max_k = 0, mx_max_n = 0;

    auto quantize_nvfp4 = [&](const Tensor& w) {
        if (wcache_->cutlass_nvfp4_prefill.count(w.data))
            return;
        Tensor fp16_view(w.data, QType::F16, 2, w.shape, /*on_device=*/true);
        const float tscale = calibrate_nvfp4_scales(fp16_view, stream);
        NvFP4QuantResult result;
        quantize_fp16_to_nvfp4_with_scale(fp16_view, tscale, result, stream);
        result.tensor_scale = tscale;
        result.N = static_cast<int>(w.shape[0]);
        result.K = static_cast<int>(w.shape[1]);

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
        n_nv++;
        nv_bytes += static_cast<size_t>(w.shape[0]) * w.shape[1] / 2 + cw.sf_bytes;
    };

    auto quantize_mxfp8 = [&](const Tensor& w) {
        if (wcache_->cutlass_mxfp8_prefill.count(w.data))
            return;
        CutlassMxFP8Weight cw;
        cw.N = w.shape[0];
        cw.K = w.shape[1];
        cw.data_bytes = static_cast<size_t>(cw.N) * cw.K;
        cw.sf_bytes = cutlass_mxfp8_sf_size(static_cast<int>(cw.N), static_cast<int>(cw.K));
        cw.data = vram_alloc(vram_alloc_, cw.data_bytes, "mxfp8_prefill_copy");
        cw.scale_factors = cw.data ? vram_alloc(vram_alloc_, cw.sf_bytes, "mxfp8_prefill_sf") : nullptr;
        if (!cw.data || !cw.scale_factors) {
            IMP_LOG_WARN("gemm.mxfp8_gdn_proj_prefill: copy of [%lld x %lld] refused (VRAM), kept FP16",
                         (long long)w.shape[0], (long long)w.shape[1]);
            if (cw.data)
                vram_free(vram_alloc_, cw.data);
            return;
        }
        quantize_fp16_to_mxfp8_cutlass(w.data, cw.data, cw.scale_factors, static_cast<int>(cw.N),
                                       static_cast<int>(cw.K), stream);
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
        wcache_->cutlass_mxfp8_prefill[w.data] = cw;
        n_mx++;
        mx_bytes += cw.data_bytes + cw.sf_bytes;
        mx_max_k = std::max(mx_max_k, w.shape[1]);
        mx_max_n = std::max(mx_max_n, w.shape[0]);
    };

    // A projection named in both flags gets the MXFP8 copy only.
    auto place = [&](const Tensor& w, bool want_mx, bool want_nv) {
        if (!eligible_f16_projection(w))
            return;  // native NVFP4 / FP8 projections already have their own prefill route
        if (want_mx)
            quantize_mxfp8(w);
        else if (want_nv)
            quantize_nvfp4(w);
    };

    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& L = model_->layer(i);
        if (!L.ssm_in.data || !L.gdn_gate.data)
            continue;  // GDN layers only (Mamba2 pure-SSM layers keep the FP16 path)
        place(L.ssm_in, mx.in, nv.in);
        place(L.gdn_gate, mx.gate, nv.gate);
        place(L.ssm_out, mx.out, nv.out);
    }

    wcache_->cutlass_nvfp4_prefill_bytes = nv_bytes;
    wcache_->cutlass_mxfp8_prefill_bytes = mx_bytes;

    if (n_nv > 0)
        IMP_LOG_INFO(
            "GDN NVFP4 prefill: %d F16 in/gate/out projections -> NVFP4 (%.1f MiB), M>32 rows on "
            "the CUTLASS GEMM",
            n_nv, nv_bytes / (1024.0 * 1024.0));
    else if (nv.any())
        IMP_LOG_INFO("GDN NVFP4 prefill: no eligible F16 GDN projection (flag on, model has none)");

    if (n_mx > 0) {
        // Activation scratch for the MXFP8 route: E4M3 [max_tokens, max_K] plus SfAtom scales and
        // the CUTLASS workspace, sized once at the largest copied projection.
        auto& qs = *qscratch_;
        qs.mxfp8_act_data_size = static_cast<size_t>(max_tokens_) * mx_max_k;
        qs.mxfp8_act_sf_size = cutlass_mxfp8_sf_size(max_tokens_, static_cast<int>(mx_max_k));
        qs.mxfp8_workspace_size = gemm_mxfp8_cutlass_sm120_workspace(max_tokens_, static_cast<int>(mx_max_n),
                                                                     static_cast<int>(mx_max_k));
        qs.mxfp8_act_data = vram_alloc(vram_alloc_, qs.mxfp8_act_data_size, "mxfp8_act_data");
        qs.mxfp8_act_sf = vram_alloc(vram_alloc_, qs.mxfp8_act_sf_size, "mxfp8_act_sf");
        qs.mxfp8_workspace = qs.mxfp8_workspace_size > 0
                                 ? vram_alloc(vram_alloc_, qs.mxfp8_workspace_size, "mxfp8_workspace")
                                 : nullptr;
        if (!qs.mxfp8_act_data || !qs.mxfp8_act_sf || (qs.mxfp8_workspace_size > 0 && !qs.mxfp8_workspace)) {
            IMP_LOG_WARN("GDN MXFP8 prefill: activation scratch (%.1f MiB) refused, the copies stay unused",
                         (qs.mxfp8_act_data_size + qs.mxfp8_act_sf_size + qs.mxfp8_workspace_size) /
                             (1024.0 * 1024.0));
            if (qs.mxfp8_act_data)
                vram_free(vram_alloc_, qs.mxfp8_act_data);
            if (qs.mxfp8_act_sf)
                vram_free(vram_alloc_, qs.mxfp8_act_sf);
            if (qs.mxfp8_workspace)
                vram_free(vram_alloc_, qs.mxfp8_workspace);
            qs.mxfp8_act_data = qs.mxfp8_act_sf = qs.mxfp8_workspace = nullptr;
            qs.mxfp8_act_data_size = qs.mxfp8_act_sf_size = qs.mxfp8_workspace_size = 0;
        }
        IMP_LOG_INFO(
            "GDN MXFP8 prefill: %d F16 in/gate/out projections -> MXFP8 (%.1f MiB), M>32 rows on the "
            "CUTLASS GEMM; activation scratch %.1f MiB",
            n_mx, mx_bytes / (1024.0 * 1024.0),
            (qs.mxfp8_act_data_size + qs.mxfp8_act_sf_size + qs.mxfp8_workspace_size) / (1024.0 * 1024.0));
    } else if (mx.any()) {
        IMP_LOG_INFO("GDN MXFP8 prefill: no eligible F16 GDN projection (flag on, model has none)");
    }
}

}  // namespace imp
