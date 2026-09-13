// Pre-dequant Phase 0 + 0b: NVFP4 loader-side setup. Phase 0 promotes NVFP4 SafeTensors
// sidecars (scales, codebooks) to device tensors; Phase 0b registers CUTLASS-NVFP4 weight
// metadata for the prefill GEMM path. Colocated as both are loader-side concerns.

#include "core/dispatch_policy.h"
#include "exec/executor.h"
#include "exec/quant_pipeline.h"
#include "exec/pre_dequant_internal.h"
#include "exec/nvfp4_expert_offload.h"
#include "exec/nvfp4_merged_scale_guard.h"
#include "compute/gemm_cutlass_sm120.h"
#include "core/logging.h"
#include "quant/nvfp4_quant.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace imp {
namespace {

// Rows of the weight_scale tensor a slot was promoted against, 0 when the slot
// has no scratch entry of its own. Only valid before the scratch is cleared.
int64_t promoted_plane_rows(const Model& model, const std::string& key) {
    auto it = model.nvfp4_scratch_.find(key);
    if (it == model.nvfp4_scratch_.end() || it->second.weight_scale.ndim != 2)
        return 0;
    return it->second.weight_scale.shape[0];
}

// Load-time assertion over projection groups that CAN share a scale plane, run once after
// the split arms above; a violation is refused rather than served (every failure mode
// here decodes one projection against another's micro-scales). arm_qkv/arm_gate_up mark
// layers where the fix-up arm actually wrote sibling pointers into the base's plane,
// distinct from the weight merely being fused (Phi-4-reasoning-plus-NVFP4 ships fused
// tensors whose siblings still promote from independent scale allocations).
void assert_merged_scale_provenance(const Model& model, const ModelConfig& cfg,
                                    const std::vector<char>& arm_qkv,
                                    const std::vector<char>& arm_gate_up) {
    int n_groups = 0, n_fused = 0, n_one_plane = 0;
    std::string err;
    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& L = model.layer(i);
        const std::string lk = "L" + std::to_string(i) + ".";
        MergedScaleGroup groups[2] = {};
        groups[0].layer = i;
        groups[0].what = "q|k|v";
        groups[0].fused = L.qkv_split_from_fused;
        groups[0].spans_one_plane = arm_qkv[i] != 0;
        groups[0].scale_row_bytes = L.wq.shape[1] / 8;
        if (L.wq.qtype == QType::NVFP4 && L.wk.qtype == QType::NVFP4 && L.wv.qtype == QType::NVFP4) {
            groups[0].count = 3;
            groups[0].m[0] = {L.wq.scales, L.wq.tensor_scale, L.wq.shape[0],
                              promoted_plane_rows(model, lk + "wq")};
            groups[0].m[1] = {L.wk.scales, L.wk.tensor_scale, L.wk.shape[0],
                              promoted_plane_rows(model, lk + "wk")};
            groups[0].m[2] = {L.wv.scales, L.wv.tensor_scale, L.wv.shape[0],
                              promoted_plane_rows(model, lk + "wv")};
        }
        groups[1].layer = i;
        groups[1].what = "gate|up";
        groups[1].fused = L.gate_up_split_from_fused;
        groups[1].spans_one_plane = arm_gate_up[i] != 0;
        groups[1].scale_row_bytes = L.w_gate.shape[1] / 8;
        if (L.w_gate.qtype == QType::NVFP4 && L.w_up.qtype == QType::NVFP4) {
            groups[1].count = 2;
            groups[1].m[0] = {L.w_gate.scales, L.w_gate.tensor_scale, L.w_gate.shape[0],
                              promoted_plane_rows(model, lk + "w_gate")};
            groups[1].m[1] = {L.w_up.scales, L.w_up.tensor_scale, L.w_up.shape[0],
                              promoted_plane_rows(model, lk + "w_up")};
        }
        for (const MergedScaleGroup& g : groups) {
            if (g.count == 0)
                continue;
            n_groups++;
            if (g.fused)
                n_fused++;
            if (g.spans_one_plane)
                n_one_plane++;
            if (!merged_scale_group_ok(g, &err))
                throw std::runtime_error(
                    "NVFP4 merged-scale provenance violated: " + err +
                    ". Serving this would decode one projection against another's micro-scales at "
                    "exit code 0.");
        }
    }
    if (n_groups > 0)
        IMP_LOG_INFO("merged-scale provenance: %d groups checked, %d fused, %d sharing one scale plane",
                     n_groups, n_fused, n_one_plane);
}

}  // namespace

void QuantPipeline::pre_dequant_phase0_promote_nvfp4_sidecars_(
    const ModelConfig& cfg, cudaStream_t stream) {
    if (!cfg.is_nvfp4_prequant)
        return;

    // SafeTensors loader + weight_map.cpp deposit each scale tensor into
    // model_->nvfp4_scratch_, keyed by slot name ("L5.wq", "L5.expert_w_gate.7", "out_proj");
    // weight_upload.cu uploads weight_scale/weight_scale_2/input_scale first.
    // Resolves each key to its main weight tensor and copies device pointers + the FP32
    // tensor scalar (reciprocal pre-applied) onto its qtype/scales/tensor_scale sidecar, so
    // the hot path reads NVFP4 metadata off the weight tensor with no cache lookup. Scratch
    // map is cleared after.

    // model_ is held as const Model* in the executor; the prequant
    // promote step is the one place we deliberately mutate it after
    // load. const_cast is local and bounded to this block.
    Model* mut_model = const_cast<Model*>(model_);

    // Track diagnostic stats across all promoted weights (helps surface
    // pathological scales that would otherwise silently corrupt output).
    int n_zero_scale = 0;
    int n_nonfinite_after_flip = 0;
    int n_with_input_scale = 0;
    int n_wrong_scale_dtype = 0;
    int n_native_fp8 = 0;
    auto promote = [&](const NvFP4PreQuantWeight& sc, Tensor& w, const char* key) {
        if (!sc.valid() || !w.data)
            return false;

        // Native FP8 weights are not NVFP4 and must not be promoted as such. A Modelopt
        // MIXED_PRECISION export (Nemotron-3.5) stores Mamba in/out projections as F8_E4M3 with a
        // single FP32 weight_scale and no weight_scale_2; treating that as NVFP4's two-level
        // layout would mislabel the scale as per-16 micro-scales. Record the scalar and stop:
        // Phase 1 expands it to FP16 (sm_120 has no FP8 prefill GEMM).
        if (w.qtype == QType::FP8_E4M3) {
            float s = 1.0f;
            if (sc.weight_scale.data) {
                // Unlike NVFP4 micro-scales this one is a scalar, so it is
                // readable wherever it happens to live — it is not uploaded.
                if (sc.weight_scale.on_device)
                    cudaMemcpy(&s, sc.weight_scale.data, sizeof(float), cudaMemcpyDeviceToHost);
                else
                    memcpy(&s, sc.weight_scale.data, sizeof(float));
            }
            if (!(s > 0.0f) || !std::isfinite(s)) {
                IMP_LOG_WARN(
                    "FP8 weight %s has a non-finite or non-positive weight_scale (%.6g) — "
                    "using 1.0; check this checkpoint",
                    key, s);
                s = 1.0f;
            }
            w.tensor_scale = s;
            n_native_fp8++;
            return false;  // not an NVFP4 promotion — nothing else to do here
        }

        // Promotion is host bookkeeping only; it must never mix address spaces: w.scales follows
        // the same pointer discipline as w.data, so a device weight with a host scale (or the
        // reverse) would hand one to the wrong consumer. Both-on-host is legitimate for MoE
        // experts (the expert cache stages them at decode time); dense weights have no host path
        // so they keep the device requirement. Skipping this made #1403's placement unservable.
        const bool both_device = (w.on_device && sc.weight_scale.on_device);
        const bool both_host = (!w.on_device && !sc.weight_scale.on_device);
        if (!both_device && !(both_host && is_expert_key(key))) {
            IMP_LOG_DEBUG("NVFP4 prequant: skipping %s (data_dev=%d, scale_dev=%d)", key, w.on_device,
                          sc.weight_scale.on_device);
            return false;
        }
        // F8: enforce compressed-tensors NVFP4 spec, weight_scale must be float8_e4m3fn. Guards
        // against NVFP4<->MXFP4 cross-misrouting (weight_scale would be U8/UE8M0 power-of-two)
        // and other dtype mismatches that would silently corrupt output via the FP8 E4M3 decoder.
        std::string scale_dtype_err;
        if (!nvfp4_validate_weight_scale_dtype(sc.weight_scale.qtype, &scale_dtype_err)) {
            n_wrong_scale_dtype++;
            IMP_LOG_WARN("NVFP4 prequant: %s — %s. Skipping promotion (weight stays in "
                         "dequant->cuBLAS fallback path).", key, scale_dtype_err.c_str());
            return false;
        }

        // F6: enforce group_size=16 between weight_packed [N, K/2] and weight_scale [N, K/16].
        // The kernel hard-codes kMicroBlockSize=16 (nvfp4_gemm.cu:31); a mismatch would silently
        // add ~12.5% per-element step quant noise (group_size != 16) or misalign scales onto
        // wrong rows (transposed weight_scale). Both routes are 2D at promote time.
        if (w.ndim == 2 && sc.weight_scale.ndim == 2) {
            // Fused projection splits (qkv_proj -> wq/wk/wv): the scale tensor covers the full fused
            // weight (more rows) while the sub-projection has fewer. Accept when the scale is a
            // superset covering the sub-projection's row range.
            bool is_fused_sub = (sc.weight_scale.shape[0] > w.shape[0]);
            if (!is_fused_sub) {
                std::string shape_err;
                if (!nvfp4_validate_packed_scale_shapes(w.shape[0], w.shape[1],
                                                       sc.weight_scale.shape[0], sc.weight_scale.shape[1],
                                                       &shape_err)) {
                    n_wrong_scale_dtype++;
                    IMP_LOG_WARN("NVFP4 prequant: %s — %s. Skipping promotion.", key, shape_err.c_str());
                    return false;
                }
            }
        }
        float h_scale = 1.0f;
        if (sc.weight_scale_2.data) {
            if (sc.weight_scale_2.on_device) {
                cudaMemcpy(&h_scale, sc.weight_scale_2.data, sizeof(float), cudaMemcpyDeviceToHost);
            } else {
                memcpy(&h_scale, sc.weight_scale_2.data, sizeof(float));
            }
        }
        // Defensive promotion: zero/NaN/Inf weight_scale_2 -> 0.0f. Both Modelopt (multiply) and
        // llm-compressor (divide -> 1/x) paths are guarded (nvfp4_promote_weight_scale_2,
        // quant/nvfp4_quant.h); unguarded, a non-finite scale propagates through the GEMM and
        // contaminates the entire layer's hidden state.
        bool was_zeroed = false;
        float promoted_scale = nvfp4_promote_weight_scale_2(h_scale, cfg.is_llm_compressor_nvfp4,
                                                            &was_zeroed);
        if (was_zeroed) {
            if (cfg.is_llm_compressor_nvfp4) {
                if (h_scale == 0.0f) {
                    n_zero_scale++;
                    IMP_LOG_WARN("NVFP4 prequant: %s has weight_scale_2=0 — zeroing tensor_scale "
                                 "(would otherwise produce Inf via reciprocal flip)", key);
                } else {
                    n_nonfinite_after_flip++;
                    IMP_LOG_WARN("NVFP4 prequant: %s reciprocal flip produced non-finite scale "
                                 "from h_scale=%.6g — zeroing", key, h_scale);
                }
            } else {
                if (h_scale == 0.0f) {
                    n_zero_scale++;
                    // Modelopt with h_scale=0 is intentional: a calibrated
                    // null layer. Log at INFO, no WARN.
                    IMP_LOG_DEBUG("NVFP4 prequant: %s has weight_scale_2=0 (Modelopt null layer)", key);
                } else {
                    n_nonfinite_after_flip++;
                    IMP_LOG_WARN("NVFP4 prequant: %s has non-finite weight_scale_2=%.6g — zeroing "
                                 "tensor_scale", key, h_scale);
                }
            }
        }
        if (sc.input_scale.data) {
            n_with_input_scale++;
        }
        w.qtype = QType::NVFP4;
        w.scales = sc.weight_scale.data;
        w.tensor_scale = promoted_scale;
        return true;
    };

    // Resolve "L{idx}.{slot}" / "out_proj" / "L{idx}.expert_w_*.{e}"
    // back to the corresponding main weight tensor. Returns nullptr for
    // unknown / out-of-range keys.
    auto resolve = [&](const std::string& key) -> Tensor* {
        if (key == "out_proj")
            return &mut_model->out_proj_;
        if (key.size() < 3 || key[0] != 'L')
            return nullptr;
        size_t dot = key.find('.', 1);
        if (dot == std::string::npos)
            return nullptr;
        int idx = std::atoi(key.substr(1, dot - 1).c_str());
        if (idx < 0 || idx >= cfg.n_layers)
            return nullptr;
        auto& L = mut_model->layers_[idx];
        std::string slot = key.substr(dot + 1);

        // Per-expert: "expert_w_{kind}.{e}"
        if (slot.rfind("expert_w_", 0) == 0) {
            size_t dot2 = slot.find('.');
            if (dot2 == std::string::npos)
                return nullptr;
            std::string kind = slot.substr(0, dot2);
            int e = std::atoi(slot.substr(dot2 + 1).c_str());
            std::vector<Tensor>* vec = nullptr;
            if (kind == "expert_w_gate")
                vec = &L.expert_w_gate;
            else if (kind == "expert_w_up")
                vec = &L.expert_w_up;
            else if (kind == "expert_w_down")
                vec = &L.expert_w_down;
            if (!vec || e < 0 || e >= static_cast<int>(vec->size()))
                return nullptr;
            return &(*vec)[e];
        }

        // Per-layer dense / shared.
        if (slot == "wq")
            return &L.wq;
        if (slot == "wk")
            return &L.wk;
        if (slot == "wv")
            return &L.wv;
        if (slot == "wo")
            return &L.wo;
        // Gemma-4: mlp.{gate,up,down}_proj weights land in w_*_shared
        // (weight_map.cpp routes them there). Fall back to shared when
        // primary is null.
        if (slot == "w_gate")
            return L.w_gate.data ? &L.w_gate : &L.w_gate_shared;
        if (slot == "w_up")
            return L.w_up.data ? &L.w_up : &L.w_up_shared;
        if (slot == "w_down")
            return L.w_down.data ? &L.w_down : &L.w_down_shared;
        if (slot == "w_gate_shared")
            return &L.w_gate_shared;
        if (slot == "w_up_shared")
            return &L.w_up_shared;
        if (slot == "w_down_shared")
            return &L.w_down_shared;
        // Mamba2 SSM (Nemotron-H): in_proj/out_proj are NVFP4-quantized
        // for layers not feeding into attention; their scales must promote
        // to the tensor sidecars so the runtime dequant path finds them.
        if (slot == "ssm_in")
            return &L.ssm_in;
        if (slot == "ssm_out")
            return &L.ssm_out;
        // GDN (Qwen3.5 linear_attn) NVFP4 projections: gdn_gate runs native
        // NVFP4 (registered in Phase 0b); gdn_alpha/gdn_beta are FP16_ONLY and
        // get dequant→FP16 below after promotion.
        if (slot == "gdn_gate")
            return &L.gdn_gate;
        if (slot == "gdn_alpha")
            return &L.gdn_alpha;
        if (slot == "gdn_beta")
            return &L.gdn_beta;
        return nullptr;
    };

    int prequant_count = 0;
    for (auto& [key, sc] : mut_model->nvfp4_scratch_) {
        Tensor* w = resolve(key);
        if (!w) {
            IMP_LOG_WARN("NVFP4 prequant: unresolved scratch key '%s'", key.c_str());
            continue;
        }
        if (promote(sc, *w, key.c_str()))
            prequant_count++;
    }
    // Fused projection scale split: weight_map split qkv_proj/gate_up_proj data pointers, but
    // scales routed as fused tensors to ALL sub-projections; fix sub-projection scales to the
    // correct row offsets. Provenance-gated since #1960: the shape predicate alone is also
    // satisfied by a separate-tensor checkpoint whose sibling merely failed to promote, which
    // would aim the repair at the base's plane instead (nvfp4_merged_scale_guard.h).
    {
        const auto& mc = cfg;
        int hd = mc.head_dim > 0 ? mc.head_dim : (mc.d_model / mc.n_heads);
        int q_rows = mc.n_heads * hd;
        int kv_rows = mc.n_kv_heads * hd;
        std::vector<char> arm_qkv(mc.n_layers, 0), arm_gate_up(mc.n_layers, 0);
        int n_qkv_split = 0, n_gateup_split = 0, n_declined = 0;
        std::string first_decline;
        auto decline = [&](int layer, const char* what, const std::string& why) {
            n_declined++;
            if (first_decline.empty())
                first_decline = std::string(what) + " on layer " + std::to_string(layer) + ": " + why;
        };
        for (int i = 0; i < mc.n_layers; i++) {
            auto& L = mut_model->layers_[i];
            const std::string lk = "L" + std::to_string(i) + ".";
            // Fused QKV: wq got scales from promote, wk/wv need split from wq's scales
            if (L.wq.qtype == QType::NVFP4 && L.wk.data && L.wv.data &&
                L.wk.qtype != QType::NVFP4 && L.wq.scales &&
                L.wq.shape[0] == q_rows && L.wk.shape[0] == kv_rows) {
                FusedSplitRequest r;
                r.provenance = L.qkv_split_from_fused;
                r.base_rows = L.wq.shape[0];
                r.sib_rows = L.wk.shape[0];
                r.n_sibs = 2;
                r.base_k_packed = L.wq.shape[1];
                r.sib_k_packed = L.wk.shape[1];
                r.plane_rows = promoted_plane_rows(*mut_model, lk + "wq");
                std::string why;
                if (L.wv.shape[0] != kv_rows) {
                    decline(i, "qkv", "wv has " + std::to_string(L.wv.shape[0]) + " rows, not " +
                                          std::to_string(kv_rows));
                } else if (!fused_split_eligible(r, &why)) {
                    decline(i, "qkv", why);
                } else {
                    size_t scale_row_bytes = static_cast<size_t>(r.base_k_packed / 8);
                    L.wk.qtype = QType::NVFP4;
                    L.wk.tensor_scale = L.wq.tensor_scale;
                    L.wk.scales = static_cast<char*>(L.wq.scales) +
                                  static_cast<size_t>(q_rows) * scale_row_bytes;
                    L.wv.qtype = QType::NVFP4;
                    L.wv.tensor_scale = L.wq.tensor_scale;
                    L.wv.scales = static_cast<char*>(L.wq.scales) +
                                  static_cast<size_t>(q_rows + kv_rows) * scale_row_bytes;
                    arm_qkv[i] = 1;
                    n_qkv_split++;
                }
            }
            // gate_up split: w_gate.scales is the fused base; w_up has no scales of its own
            // (weight_scale routed to w_gate only). Copy qtype + tensor_scale from w_gate and offset
            // the scales pointer.
            if (L.w_gate.qtype == QType::NVFP4 && L.w_up.data &&
                L.w_up.qtype != QType::NVFP4 && L.w_gate.scales) {
                FusedSplitRequest r;
                r.provenance = L.gate_up_split_from_fused;
                r.base_rows = L.w_gate.shape[0];
                r.sib_rows = L.w_up.shape[0];
                r.n_sibs = 1;
                r.base_k_packed = L.w_gate.shape[1];
                r.sib_k_packed = L.w_up.shape[1];
                r.plane_rows = promoted_plane_rows(*mut_model, lk + "w_gate");
                std::string why;
                if (r.base_rows != r.sib_rows) {
                    decline(i, "gate_up", "gate has " + std::to_string(r.base_rows) +
                                              " rows, up has " + std::to_string(r.sib_rows));
                } else if (!fused_split_eligible(r, &why)) {
                    decline(i, "gate_up", why);
                } else {
                    size_t scale_row_bytes = static_cast<size_t>(r.base_k_packed / 8);
                    L.w_up.qtype = QType::NVFP4;
                    L.w_up.tensor_scale = L.w_gate.tensor_scale;
                    L.w_up.scales = static_cast<char*>(L.w_gate.scales) +
                                    static_cast<size_t>(r.base_rows) * scale_row_bytes;
                    arm_gate_up[i] = 1;
                    n_gateup_split++;
                }
            }
        }
        if (n_qkv_split > 0 || n_gateup_split > 0)
            IMP_LOG_INFO("Fused projection scale split: %d QKV + %d gate_up layers",
                         n_qkv_split, n_gateup_split);
        if (n_declined > 0)
            IMP_LOG_WARN("Fused projection scale split declined %d time(s) (first: %s). Those "
                         "siblings keep their own scales or stay unpromoted; a split without "
                         "provenance would have aimed them into another weight's scale plane.",
                         n_declined, first_decline.c_str());
        assert_merged_scale_provenance(*mut_model, cfg, arm_qkv, arm_gate_up);
    }

    // GDN alpha/beta (Qwen3.5 linear_attn.in_proj_a/in_proj_b) are FP16_ONLY: the delta-rule
    // decay/learning-rate projections are precision-sensitive and the GDN GEMM expects FP16.
    // If promoted to NVFP4, dequant back to FP16 here before plan_storage runs. ssm_in/
    // ssm_out/gdn_gate stay native NVFP4 (Phase 0b, gemv_nvfp4, same as Nemotron-H).
    {
        int n_gdn_dequant = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            auto& L = mut_model->layers_[i];
            for (Tensor* w : {&L.gdn_alpha, &L.gdn_beta}) {
                if (w->qtype != QType::NVFP4 || !w->data || !w->scales)
                    continue;
                int64_t N = w->shape[0];
                int64_t K = w->shape[1] * 2;  // packed K/2 → logical K
                NvFP4QuantResult q;
                q.packed_data = w->data;
                q.micro_scales = w->scales;
                q.tensor_scale = w->tensor_scale;
                q.N = N;
                q.K = K;
                q.owned = false;
                void* fp16buf = nullptr;
                if (cudaMallocAsync(&fp16buf, static_cast<size_t>(N) * K * 2, stream) != cudaSuccess) {
                    IMP_LOG_WARN("GDN NVFP4→FP16 dequant: alloc failed (layer %d) — leaving NVFP4", i);
                    continue;
                }
                dequantize_nvfp4_to_fp16(q, fp16buf, stream);
                // Old FP4 data + borrowed micro_scales are tiny (N≈n_heads); leave
                // them resident rather than risk a use-after-free on the async stream.
                w->data = fp16buf;
                w->qtype = QType::F16;
                w->scales = nullptr;
                w->tensor_scale = 1.0f;
                w->shape[1] = K;
                w->compute_strides();
                n_gdn_dequant++;
            }
        }
        if (n_gdn_dequant > 0)
            IMP_LOG_INFO("GDN NVFP4 alpha/beta: dequantized %d FP16_ONLY projections to FP16",
                         n_gdn_dequant);
    }

    // Drop the scratch — its data pointers (weight_scale, weight_scale_2,
    // input_scale) are now device pointers borrowed by the main tensors,
    // and the host-side metadata isn't needed anymore.
    mut_model->nvfp4_scratch_.clear();

    if (prequant_count > 0) {
        IMP_LOG_INFO("NVFP4 pre-quantized: promoted %d weights to Tensor sidecars", prequant_count);
        if (n_zero_scale > 0 || n_nonfinite_after_flip > 0) {
            IMP_LOG_WARN("NVFP4 prequant: %d zero weight_scale_2 / %d non-finite weight_scale_2 "
                         "(weights zeroed defensively, applies to both Modelopt and llm-compressor)",
                         n_zero_scale, n_nonfinite_after_flip);
        }
        if (n_wrong_scale_dtype > 0) {
            IMP_LOG_WARN("NVFP4 prequant: %d weights had non-FP8_E4M3 weight_scale dtype "
                         "(skipped — possible NVFP4/MXFP4 cross-misroute or corrupt checkpoint)",
                         n_wrong_scale_dtype);
        }
        if (cfg.is_llm_compressor_nvfp4 && n_with_input_scale > 0) {
            // input_scale is a SmoothQuant-style activation-rescaling vector some llm-compressor
            // exports ship. Imp does NOT apply it at inference (absorbing it did not fix
            // Mistral-3.2-NVFP4 drift). Kept only as a diagnostic path via
            // diagnostics.audit_nvfp4_scales; without it, scratch tensors are skipped on GPU upload.
            IMP_LOG_INFO(
                "NVFP4 prequant: %d Linears carry input_scale (intentionally NOT applied; "
                "set diagnostics.audit_nvfp4_scales=true for stats).",
                n_with_input_scale);
        }
    }
    if (n_native_fp8 > 0) {
        IMP_LOG_INFO(
            "Mixed precision: %d native FP8 weight(s) carry a per-tensor scale — "
            "expanded to FP16 in Phase 1 (sm_120 has no FP8 prefill GEMM)",
            n_native_fp8);
    }
}

void QuantPipeline::pre_dequant_phase0b_register_cutlass_nvfp4_(
    const ModelConfig& cfg, cudaStream_t stream) {
    if (!cfg.is_nvfp4_prequant)
        return;
    Model* mut_model = const_cast<Model*>(model_);

    // Phase 0b: register prequant-promoted NVFP4 weights in wcache_->nvfp4 (Phase 0 set
    // qtype=NVFP4 on the Tensor sidecars but didn't build the CUTLASS cache Phase 3b needs).
    // Without this, prefill falls to gemm_nvfp4 dequant->cuBLAS instead of native CUTLASS
    // NVFP4xNVFP4, amplifying per-block quant noise on long context.
    // Applies uniformly to all NVFP4-prequant models: the CUTLASS non-determinism that once
    // required a format-specific skip-guard here is fixed upstream (PR #165); the guard is gone.
    if (cutlass_sm120_nvfp4_available()) {
        int ct_count = 0;
        // Native-NVFP4 checkpoints: in_proj/out_proj (ssm_in/ssm_out) exist ONLY as NVFP4 bytes
        // (no FP16 copy, dequant_gpu has no NVFP4 path), so they MUST be registered here or
        // decode reads raw NVFP4 through cuBLAS and produces garbage. GGUF-quant hybrids use the
        // gemm.fp8_ssm_proj sidecar instead (phase 2b). Always register.
        auto register_prequant = [&](const Tensor& w) {
            if (w.qtype != QType::NVFP4 || !w.data || !w.scales)
                return;
            // These caches feed device kernels: every pointer registered here is dereferenced on the
            // GPU. A host-resident weight here is an illegal access, not a slow path. Host-resident
            // MoE experts are served by the expert cache instead (exec/nvfp4_expert_offload.h).
            if (!w.on_device)
                return;
            if (wcache_->nvfp4.count(w.data))
                return;
            NvFP4QuantResult tmp;
            tmp.packed_data = w.data;
            tmp.micro_scales = w.scales;
            tmp.owned = false;  // borrows resident model weight storage — don't cudaFree on teardown
            tmp.tensor_scale = w.tensor_scale;
            tmp.N = w.shape[0];
            tmp.K = w.shape[1] * 2;  // packed K/2 → logical K
            // Register in NVFP4 cache for decode GEMV (gemv_nvfp4_kpar needs per-16 FP8
            // micro_scales, not SfAtom). Do NOT build the CUTLASS SfAtom buffer here: Phase 3b
            // (nvfp4_decode_convert_cutlass_) iterates this same map and unconditionally rebuilds
            // cutlass_nvfp4[ptr] for every entry, so a SfAtom built here is immediately orphaned
            // (leaked, never freed) rather than reused. Phase 3b is the authoritative, budget-aware
            // builder; seeding nvfp4 here is all Phase 0b needs to do.
            wcache_->nvfp4[w.data] = tmp;
            ct_count++;
        };
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = mut_model->layer(i);
            register_prequant(L.wq);
            register_prequant(L.wk);
            register_prequant(L.wv);
            register_prequant(L.wo);
            register_prequant(L.w_gate);
            register_prequant(L.w_up);
            register_prequant(L.w_down);
            register_prequant(L.w_gate_shared);
            register_prequant(L.w_up_shared);
            register_prequant(L.w_down_shared);
            register_prequant(L.ssm_in);
            register_prequant(L.ssm_out);
            register_prequant(L.gdn_gate);  // Qwen3.5 GDN output gate — native NVFP4
            for (const auto& ew : L.expert_w_gate) register_prequant(ew);
            for (const auto& ew : L.expert_w_up) register_prequant(ew);
            for (const auto& ew : L.expert_w_down) register_prequant(ew);
        }
        register_prequant(mut_model->out_proj_);

        if (ct_count > 0) {
            // Decode-cache registration only; Phase 3b builds the CUTLASS SfAtom
            // for these entries (see register_prequant note above).
            IMP_LOG_INFO("NVFP4 prequant: registered %d weights in decode cache "
                         "(CUTLASS SfAtom built once in Phase 3b)",
                         ct_count);
        }
        {
            // Report what this loop actually registered, not what the recurrent projections are
            // assumed to be: on a mixed-precision Modelopt hybrid the projections are FP8 in the
            // checkpoint, so register_prequant never sees them, phase 1 gives them an FP16 prefill
            // companion, and phase 3 excludes them from the NVFP4 cache outright.
            int n_ssm = 0, n_ssm_cached = 0;
            for (int i = 0; i < cfg.n_layers; i++) {
                const auto& L = mut_model->layer(i);
                if (L.ssm_in.data) {
                    n_ssm++;
                    n_ssm_cached += wcache_->nvfp4.count(L.ssm_in.data) ? 1 : 0;
                }
                if (L.ssm_out.data) {
                    n_ssm++;
                    n_ssm_cached += wcache_->nvfp4.count(L.ssm_out.data) ? 1 : 0;
                }
            }
            if (n_ssm > 0)
                IMP_LOG_INFO(
                    "GDN/SSM: %d of %d recurrent projections registered in the NVFP4 "
                    "decode cache here; the rest keep their source precision and take "
                    "whatever prefill/decode tier phase 4 assigns",
                    n_ssm_cached, n_ssm);
        }
    }
}

}  // namespace imp
