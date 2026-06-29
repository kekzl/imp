// Pre-dequant Phase 0 + 0b: NVFP4 loader-side setup.
// Phase 0: promote NVFP4 SafeTensors sidecars (scales, codebooks) to
// device tensors. Phase 0b: register CUTLASS-NVFP4 weight metadata for
// the prefill GEMM path.
//
// Both phases run consecutively as NVFP4 loader-side concerns and are
// colocated here to keep related logic in one file.
//
// Extracted from executor_pre_dequant.cu in Phase 3 of the architecture
// refactor roadmap. See pre_dequant_internal.h for shared helpers.

#include "exec/executor.h"
#include "exec/quant_pipeline.h"
#include "exec/pre_dequant_internal.h"
#include "compute/gemm_cutlass_sm120.h"
#include "core/logging.h"
#include "quant/nvfp4_quant.h"
#include "runtime/config.h"

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace imp {

void QuantPipeline::pre_dequant_phase0_promote_nvfp4_sidecars_(
    const ModelConfig& cfg, cudaStream_t stream) {
    if (!cfg.is_nvfp4_prequant)
        return;

    // The SafeTensors loader + weight_map.cpp deposit each scale tensor into
    // model_->nvfp4_scratch_, keyed by canonical slot name (e.g. "L5.wq",
    // "L5.expert_w_gate.7", "out_proj"). Every entry's weight_scale /
    // weight_scale_2 / input_scale tensors are uploaded to the GPU by
    // weight_upload.cu before this phase runs.
    //
    // For each entry: resolve the key back to the corresponding main weight
    // tensor (layer.wq, layer.expert_w_gate[e], etc.) and copy the device
    // pointers + the FP32 tensor scalar (with the llm-compressor reciprocal
    // trick pre-applied) onto its .qtype / .scales / .tensor_scale sidecar
    // fields. After the loop runs, the runtime hot path reads NVFP4 metadata
    // straight off the weight tensor — no cache lookup, no per-call
    // reconstruction. The scratch map is then cleared.

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
    auto promote = [&](const NvFP4PreQuantWeight& sc, Tensor& w, const char* key) {
        if (!sc.valid() || !w.data)
            return false;
        if (!w.on_device || !sc.weight_scale.on_device) {
            IMP_LOG_DEBUG("NVFP4 prequant: skipping %s (data_dev=%d, scale_dev=%d)", key, w.on_device,
                          sc.weight_scale.on_device);
            return false;
        }
        // F8: enforce compressed-tensors NVFP4 spec: weight_scale must be
        // float8_e4m3fn. Defending against NVFP4↔MXFP4 cross-misrouting
        // (where weight_scale would be U8 / UE8M0 power-of-two) and other
        // accidental dtype mismatches that would silently corrupt output
        // through the FP8 E4M3 decoder.
        std::string scale_dtype_err;
        if (!nvfp4_validate_weight_scale_dtype(sc.weight_scale.qtype, &scale_dtype_err)) {
            n_wrong_scale_dtype++;
            IMP_LOG_WARN("NVFP4 prequant: %s — %s. Skipping promotion (weight stays in "
                         "dequant->cuBLAS fallback path).", key, scale_dtype_err.c_str());
            return false;
        }

        // F6: enforce group_size=16 between weight_packed [N, K/2] and
        // weight_scale [N, K/16]. The kernel hard-codes kMicroBlockSize=16
        // (nvfp4_gemm.cu:31); a shape mismatch would silently produce
        // 12.5% per-element step quant noise on roughly half the elements
        // (group_size != 16) or align scales onto wrong rows
        // (transposed weight_scale). Both routes are 2D at promote time
        // (per-expert weights have been split by weight_upload.cu).
        if (w.ndim == 2 && sc.weight_scale.ndim == 2) {
            // For fused projection splits (qkv_proj → wq/wk/wv), the scale
            // tensor covers the full fused weight (e.g., 7680 rows) while the
            // sub-projection has fewer rows (e.g., 1280). Accept if the scale
            // is a superset that covers the sub-projection's row range.
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
        // Defensive promotion: zero, NaN, ±Inf weight_scale_2 → 0.0f. Both
        // Modelopt (multiply) and llm-compressor (divide → 1/x) paths are
        // guarded; see nvfp4_promote_weight_scale_2 in quant/nvfp4_quant.h.
        // Without this guard a non-finite scale would propagate through the
        // GEMM and contaminate the entire layer's hidden state.
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
    // Fused projection scale split: weight_map split qkv_proj/gate_up_proj
    // data pointers, but scales were routed as fused tensors to ALL sub-
    // projections. Now that promote() set .scales = fused GPU pointer,
    // fix the sub-projection scales to point at the correct row offsets.
    {
        const auto& mc = cfg;
        int hd = mc.head_dim > 0 ? mc.head_dim : (mc.d_model / mc.n_heads);
        int q_rows = mc.n_heads * hd;
        int kv_rows = mc.n_kv_heads * hd;
        int n_qkv_split = 0, n_gateup_split = 0;
        for (int i = 0; i < mc.n_layers; i++) {
            auto& L = mut_model->layers_[i];
            // Fused QKV: wq got scales from promote, wk/wv need split from wq's scales
            if (L.wq.qtype == QType::NVFP4 && L.wk.data && L.wv.data &&
                L.wk.qtype != QType::NVFP4 && L.wq.scales &&
                L.wq.shape[0] == q_rows && L.wk.shape[0] == kv_rows) {
                int64_t K_packed = L.wq.shape[1];
                size_t scale_row_bytes = static_cast<size_t>(K_packed / 8);
                L.wk.qtype = QType::NVFP4;
                L.wk.tensor_scale = L.wq.tensor_scale;
                L.wk.scales = static_cast<char*>(L.wq.scales) +
                              static_cast<size_t>(q_rows) * scale_row_bytes;
                L.wv.qtype = QType::NVFP4;
                L.wv.tensor_scale = L.wq.tensor_scale;
                L.wv.scales = static_cast<char*>(L.wq.scales) +
                              static_cast<size_t>(q_rows + kv_rows) * scale_row_bytes;
                n_qkv_split++;
            }
            // gate_up split: w_gate.scales is fused base, w_up needs offset
            // Fused gate_up: w_gate got scales from promote, w_up has no scales
            // (weight_scale routed to w_gate only). Copy qtype + tensor_scale
            // from w_gate and offset the scales pointer.
            if (L.w_gate.qtype == QType::NVFP4 && L.w_up.data &&
                L.w_up.qtype != QType::NVFP4 && L.w_gate.scales) {
                int64_t K_packed = L.w_gate.shape[1];
                int64_t half_rows = L.w_gate.shape[0];
                size_t scale_row_bytes = static_cast<size_t>(K_packed / 8);
                L.w_up.qtype = QType::NVFP4;
                L.w_up.tensor_scale = L.w_gate.tensor_scale;
                L.w_up.scales = static_cast<char*>(L.w_gate.scales) +
                                static_cast<size_t>(half_rows) * scale_row_bytes;
                n_gateup_split++;
            }
        }
        if (n_qkv_split > 0 || n_gateup_split > 0)
            IMP_LOG_INFO("Fused projection scale split: %d QKV + %d gate_up layers",
                         n_qkv_split, n_gateup_split);
    }

    // GDN alpha/beta (Qwen3.5 linear_attn.in_proj_a/in_proj_b) are FP16_ONLY:
    // the delta-rule decay / learning-rate projections are precision-sensitive
    // and the GDN GEMM expects FP16. If the checkpoint stored them NVFP4,
    // promote() above set qtype=NVFP4; dequant them back to FP16 here (before
    // plan_storage runs, so it tiers them FP16). ssm_in/ssm_out/gdn_gate stay
    // native NVFP4 (registered in Phase 0b, gemv_nvfp4 — same as Nemotron-H).
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
            // input_scale is a SmoothQuant-style activation-rescaling vector
            // shipped by some llm-compressor exports. Imp does NOT apply it
            // at inference: the long-context-bug investigation refuted the
            // hypothesis that absorbing it would fix Mistral-3.2-NVFP4
            // drift (see memory `llm_compressor_input_scale_dead_end_2026_05_07.md`).
            // Loading is kept as a diagnostic path: pass IMP_AUDIT_NVFP4_SCALES=1
            // for per-Linear stats. Without that env var, scratch tensors
            // are skipped during GPU upload (no VRAM cost).
            IMP_LOG_INFO(
                "NVFP4 prequant: %d Linears carry input_scale (intentionally NOT applied; "
                "set IMP_AUDIT_NVFP4_SCALES=1 for stats).",
                n_with_input_scale);
        }
    }
}

void QuantPipeline::pre_dequant_phase0b_register_cutlass_nvfp4_(
    const ModelConfig& cfg, cudaStream_t stream) {
    if (!cfg.is_nvfp4_prequant)
        return;
    Model* mut_model = const_cast<Model*>(model_);

    // --- Phase 0b: register prequant-promoted NVFP4 weights in CUTLASS cache ---
    //
    // Phase 0 set qtype=NVFP4 directly on the main weight Tensor sidecars
    // but did NOT populate wcache_->nvfp4 (the legacy decode-cache map that
    // Phase 3b iterates to build the CUTLASS cache). Without this loop,
    // prefill (M>1) for prequant SafeTensors models falls through to
    // gemm_nvfp4 dequant→cuBLAS in executor_kernels.cu — slower AND
    // noisier than native CUTLASS NVFP4×NVFP4. The FP16-act ×
    // NVFP4-deq-FP16 round-trip amplifies SmoothQuant-induced per-block
    // quant noise on long context (Mistral-3.2-NVFP4 breaks above ~90
    // tokens; see memory/nvfp4_long_context_regression_2026_04_28.md).
    //
    // Build the CUTLASS cache directly from the Tensor sidecars (data
    // pointer, scales, tensor_scale, shape) so the prefill dispatch
    // at executor_kernels.cu:1975 fires on the native FP4×FP4 path.
    //
    // Historical EXCLUSION (added 2026-05-02, REMOVED 2026-05-14):
    // The llm-compressor NVFP4 format previously triggered a Skip-Guard
    // that fell back to dequant→cuBLAS because the CUTLASS NVFP4×NVFP4
    // path produced (a) numerical regressions on borderline tokens
    // (3/4 phase4 vs 4/4 guard-on baseline) and (b) cross-capture
    // non-determinism (2/4 graph_replay vs 4/4 guard-on baseline).
    // Memo `cutlass_nvfp4_sm120_nondeterministic_2026_05_05` extended
    // this to "universal sm_120 CUTLASS NVFP4 non-determinism" — even
    // Modelopt format showed 1/4 graph_replay on CUTLASS v4.4.2.
    //
    // CUTLASS v4.5.0 (PR #165, 2026-05-13) silently fixed this.
    // Re-evaluation 2026-05-14 via `scripts/validate_safetensors.py`:
    //   - Gemma-4-NVFP4 (llm-compressor): graph_replay 4/4, phase4
    //     parity with guard-on, det3=True, prefill 10.2k → 22.1k tok/s
    //     (2.18× win at pp=512).
    //   - Qwen3-30B-A3B-NVFP4-Modelopt: graph_replay 4/4 (was 1/4).
    // Guard is no longer load-bearing; removing it restores the
    // CUTLASS fast path for all NVFP4-prequant models uniformly.
    if (cutlass_sm120_nvfp4_available()) {
        int ct_count = 0;
        // NOTE on the GDN/SSM quality lock and gemm.nvfp4_ssm_proj:
        // For NATIVE-NVFP4 checkpoints the in_proj/out_proj (ssm_in/ssm_out)
        // weights only EXIST as NVFP4 bytes — there is no FP16 copy and
        // dequant_gpu has no NVFP4 path, so they MUST be registered here or
        // decode produces garbage (the uncached fallback would run cuBLAS on
        // raw NVFP4 bytes). They are therefore ALREADY in the NVFP4 decode
        // cache and already use the fast gemv_nvfp4_kpar path — the "FP16 SSM
        // tax" does not apply to native-NVFP4 hybrids. The gemm.nvfp4_ssm_proj
        // opt-in is meaningful only for GGUF-quant hybrids (Qwen3.6-35B Q4_K_M),
        // gated in pre_dequant_phase3's collect_candidates. Always register.
        auto register_prequant = [&](const Tensor& w) {
            if (w.qtype != QType::NVFP4 || !w.data || !w.scales)
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
            // Register in NVFP4 cache for decode GEMV (gemv_nvfp4_kpar needs
            // per-16 FP8 micro_scales, not SfAtom). Without this, decode falls
            // through to the CUTLASS_NVFP4 case which uses source_scales —
            // but that path produced zero output on some models.
            //
            // VRAM-AUDIT (2026-06-12): do NOT build the CUTLASS SfAtom buffer
            // here. Phase 3b (nvfp4_decode_convert_cutlass_) iterates exactly
            // this wcache_->nvfp4 map and rebuilds cutlass_nvfp4[ptr] for every
            // entry, unconditionally overwriting whatever this loop inserted —
            // so a SfAtom built here is immediately orphaned (the map drops the
            // pointer without freeing it) and stays resident, dead, until exit.
            // On Qwen3-Coder-30B-A3B NVFP4 that orphan was ~1782 MiB. Phase 3b
            // is the authoritative, budget-aware builder whose entries Phase 4
            // snapshots into the weight handles the GEMM kernels actually read;
            // seeding nvfp4 here is all Phase 0b needs to do.
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
            int n_ssm = 0;
            for (int i = 0; i < cfg.n_layers; i++) {
                const auto& L = mut_model->layer(i);
                if (L.ssm_in.data) n_ssm++;
                if (L.ssm_out.data) n_ssm++;
            }
            if (n_ssm > 0)
                IMP_LOG_INFO("GDN/SSM (native NVFP4): %d recurrent projections IN NVFP4 decode cache "
                             "(no FP16 source exists; fast gemv_nvfp4_kpar at decode)",
                             n_ssm);
        }
    }
}

}  // namespace imp
