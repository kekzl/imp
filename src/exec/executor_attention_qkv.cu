// QKV-projection dispatch block for GraphExecutor::run_attention.
//
// This is NOT a standalone translation unit — it is textually #include'd inside
// the body of GraphExecutor::run_attention (executor_attention.cu), between the
// braces of the QKV-projection scope. It is therefore omitted from the CMake
// source list and must not be compiled on its own. The contents are byte-for-
// byte the original inline block; see executor_attention.cu for surrounding
// context and local variables in scope.
        auto* q8 = static_cast<block_q8_1*>(qscratch_.q8_1_buf);
        // MXFP4 decode path: native MXFP4 GEMV with UE8M0 scales
        const WeightHandle* mxfp4_hwq = (ly.wq_id != kInvalidTensorID) ? &registry_.handle(ly.wq_id)
                                                                       : nullptr;
        const WeightHandle* mxfp4_hwk = (ly.wk_id != kInvalidTensorID) ? &registry_.handle(ly.wk_id)
                                                                       : nullptr;
        const WeightHandle* mxfp4_hwv = (ly.wv_id != kInvalidTensorID) ? &registry_.handle(ly.wv_id)
                                                                       : nullptr;
        bool mxfp4_qkv = (!has_attn_output_gate && n == 1 && mxfp4_hwq &&
                          mxfp4_hwq->primary_tier == StorageTier::MXFP4 &&
                          mxfp4_hwq->payload.mxfp4.linear_scales && mxfp4_hwk &&
                          mxfp4_hwk->primary_tier == StorageTier::MXFP4 &&
                          mxfp4_hwk->payload.mxfp4.linear_scales && mxfp4_hwv &&
                          mxfp4_hwv->primary_tier == StorageTier::MXFP4 &&
                          mxfp4_hwv->payload.mxfp4.linear_scales);
        // NVFP4 decode path: uses FP16 input (no Q8_1 quantization needed).
        // Two sources: (1) primary NVFP4 storage tier (native NVFP4 models),
        // (2) secondary NVFP4 decode cache populated by Phase 3 for
        // Q8_0/Q6_K/Q5_K models — c8763ad refactor lost this fallback;
        // restore by also probing `wcache_.nvfp4`.
        auto nv_q_it = wcache_.nvfp4.find(ly.wq.data);
        auto nv_k_it = wcache_.nvfp4.find(ly.wk.data);
        auto nv_v_it = wcache_.nvfp4.find(ly.wv.data);
        const bool nv_q_secondary = (nv_q_it != wcache_.nvfp4.end());
        const bool nv_k_secondary = (nv_k_it != wcache_.nvfp4.end());
        const bool nv_v_secondary = (nv_v_it != wcache_.nvfp4.end());
        const bool wq_is_nvfp4 = (mxfp4_hwq && mxfp4_hwq->primary_tier == StorageTier::NVFP4) ||
                                 nv_q_secondary;
        const bool wk_is_nvfp4 = (mxfp4_hwk && mxfp4_hwk->primary_tier == StorageTier::NVFP4) ||
                                 nv_k_secondary;
        const bool wv_is_nvfp4 = (mxfp4_hwv && mxfp4_hwv->primary_tier == StorageTier::NVFP4) ||
                                 nv_v_secondary;
        bool nvfp4_qkv = (!has_attn_output_gate && n == 1 && wq_is_nvfp4 && wk_is_nvfp4 && wv_is_nvfp4);
        // Gemma-4: disable fused QKV when FP32 accum is active — the fused kernel
        // reads FP16 h instead of fp32_hidden_, losing precision through 128-expert routing.
        bool fused_qkv = (!has_attn_output_gate && n == 1 && q8 != nullptr && qscratch_.d8_buf != nullptr &&
                          no.qtype == QType::F16 && ly.wq.qtype == ly.wk.qtype &&
                          ly.wk.qtype == ly.wv.qtype && is_dp4a_qtype(ly.wq.qtype) &&
                          !(using_fp32_accum && prof.is_gemma4));
        if (mxfp4_qkv) {
            // MXFP4 fused QKV: RMSNorm, optional Hadamard, then MXFP4 GEMV
            rmsnorm(h, ly.attn_norm, no, eps, stream, norm_w_off_);
            int q_rows = static_cast<int>(mxfp4_hwq->shape[0]);
            int k_rows = static_cast<int>(mxfp4_hwk->shape[0]);
            int v_rows = static_cast<int>(mxfp4_hwv->shape[0]);
            int K = static_cast<int>(mxfp4_hwq->shape[1]);
            int hbs = mxfp4_hwq->payload.mxfp4.hadamard_bs;
            if (hbs > 0 && hadamard_block_size_valid(hbs))
                hadamard_transform_fp16(static_cast<const half*>(no.data), static_cast<half*>(no.data), 1, K,
                                        hbs, stream);
            // Reconstruct CutlassMxFP4Weight structs from handle payloads.
            auto make_mxfp4 = [](const WeightHandle* h) {
                CutlassMxFP4Weight mw;
                mw.data = h->payload.mxfp4.weight;
                mw.scale_factors = h->payload.mxfp4.scales;
                mw.linear_scales = h->payload.mxfp4.linear_scales;
                mw.hadamard_bs = h->payload.mxfp4.hadamard_bs;
                mw.tensor_scale = 1.0f;
                mw.N = static_cast<int>(h->shape[0]);
                mw.K = static_cast<int>(h->shape[1]);
                mw.sf_bytes = cutlass_mxfp4_sf_size(mw.N, mw.K);
                mw.owns_data = false;
                return mw;
            };
            auto mw_q = make_mxfp4(mxfp4_hwq);
            auto mw_k = make_mxfp4(mxfp4_hwk);
            auto mw_v = make_mxfp4(mxfp4_hwv);
            gemv_mxfp4_qkv_fused(mw_q, mw_k, mw_v, static_cast<const half*>(no.data),
                                 static_cast<half*>(qv.data), static_cast<half*>(kk.data),
                                 static_cast<half*>(vv.data), q_rows, k_rows, v_rows, K, stream);
        } else if (nvfp4_qkv) {
            // NVFP4 fused QKV: RMSNorm to FP16, then NVFP4 GEMV (no Q8_1 needed)
            rmsnorm(h, ly.attn_norm, no, eps, stream, norm_w_off_);
            // Pick the NvFP4QuantResult: secondary cache (Q8_0+NVFP4-decode) is
            // already a populated struct; primary-tier handle needs reconstruction.
            auto from_handle = [](const WeightHandle* hw) {
                NvFP4QuantResult tmp;
                tmp.packed_data = hw->payload.nvfp4.data;
                tmp.micro_scales = hw->payload.nvfp4.block_scales;
                tmp.tensor_scale = (hw->payload.nvfp4.tensor_scale != nullptr)
                                       ? *hw->payload.nvfp4.tensor_scale
                                       : 1.0f;
                tmp.N = static_cast<int>(hw->shape[0]);
                tmp.K = static_cast<int>(hw->shape[1]) * 2;
                return tmp;
            };
            NvFP4QuantResult nv_q = nv_q_secondary ? nv_q_it->second : from_handle(mxfp4_hwq);
            NvFP4QuantResult nv_k = nv_k_secondary ? nv_k_it->second : from_handle(mxfp4_hwk);
            NvFP4QuantResult nv_v = nv_v_secondary ? nv_v_it->second : from_handle(mxfp4_hwv);
            int q_rows = nv_q.N;
            int k_rows = nv_k.N;
            int v_rows = nv_v.N;
            int K = nv_q.K;
            gemv_nvfp4_qkv_fused(nv_q, nv_k, nv_v, static_cast<const half*>(no.data),
                                 static_cast<half*>(qv.data), static_cast<half*>(kk.data),
                                 static_cast<half*>(vv.data), q_rows, k_rows, v_rows, K, stream);
        } else if (fused_qkv) {
            // Fused: RMSNorm + Q8_1 quantization in one kernel (no norm_out write)
            int K = static_cast<int>(ly.wq.shape[1]);
            // LoRA needs the normed projection input materialized — side-write
            // norm_out (the kernel supports it; FFN's variant always does).
            half* lora_no = (lora_ && lora_->has_qkv(layer)) ? static_cast<half*>(no.data) : nullptr;
            rmsnorm_quantize_q8_1(static_cast<const half*>(h.data),
                                  static_cast<const half*>(ly.attn_norm.data), q8, qscratch_.d8_buf,
                                  lora_no, K, eps, stream, norm_w_off_);
            int q_rows = static_cast<int>(ly.wq.shape[0]);
            int k_rows = static_cast<int>(ly.wk.shape[0]);
            int v_rows = static_cast<int>(ly.wv.shape[0]);
            dispatch_gemv_qkv_fused(ly.wq.qtype, ly.wq.data, ly.wk.data, ly.wv.data, q8, qscratch_.d8_buf,
                                    static_cast<half*>(qv.data), static_cast<half*>(kk.data),
                                    static_cast<half*>(vv.data), q_rows, k_rows, v_rows, K, stream);
        } else {
            // Separate RMSNorm + dispatch.
            // Gemma-4 FP32 accum path: read FP32 residual directly to avoid the
            // FP16 round-trip that drops ~1-2% precision per layer and causes
            // the last-token hidden state to drift by sign-flip at L29.
            if (using_fp32_accum && prof.is_gemma4) {
                Tensor fp32_h = view_tokens(fp32_hidden_, n);
                rmsnorm_fp32_to_fp16(fp32_h, ly.attn_norm, no, eps, stream, norm_w_off_);
            } else {
                rmsnorm(h, ly.attn_norm, no, eps, stream, norm_w_off_);
            }

            // FP8 prefill path: quantize norm_out→FP8 once, 3 separate FP8 GEMMs
            // Use handle tier checks instead of wcache_ probes.
            const WeightHandle* fp8_hwq = (ly.wq_id != kInvalidTensorID) ? &registry_.handle(ly.wq_id)
                                                                         : nullptr;
            const WeightHandle* fp8_hwk = (ly.wk_id != kInvalidTensorID) ? &registry_.handle(ly.wk_id)
                                                                         : nullptr;
            const WeightHandle* fp8_hwv = (ly.wv_id != kInvalidTensorID) ? &registry_.handle(ly.wv_id)
                                                                         : nullptr;
            bool fp8_qkv_available = (fp8_hwq && fp8_hwq->primary_tier == StorageTier::FP8 && fp8_hwk &&
                                      fp8_hwk->primary_tier == StorageTier::FP8 && fp8_hwv &&
                                      fp8_hwv->primary_tier == StorageTier::FP8);
            if (n > 1 && !state.force_fp16_gemm && fp8_qkv_available && qscratch_.fp8_act != nullptr &&
                qscratch_.d_act_scale != nullptr) {
                Tensor fp8_no(qscratch_.fp8_act, QType::FP8_E4M3, no.ndim, no.shape, true);
                quantize_fp16_to_fp8_e4m3(no, fp8_no, qscratch_.d_act_scale, stream,
                                          qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax,
                                          qscratch_.fp8_max_grid);
                // Reconstruct FP8 weight tensors from handle payloads.
                auto make_fp8_tensor = [](const WeightHandle* hw) {
                    int64_t wshape[2] = {hw->shape[0], hw->shape[1]};
                    return Tensor(hw->payload.fp8.data, QType::FP8_E4M3, 2, wshape, true);
                };
                Tensor fp8_tq = make_fp8_tensor(fp8_hwq);
                Tensor fp8_tk = make_fp8_tensor(fp8_hwk);
                Tensor fp8_tv = make_fp8_tensor(fp8_hwv);
                gemm_cublaslt(fp8_no, fp8_tq, q_target, 1.0f, 0.0f, qscratch_.d_act_scale,
                              fp8_hwq->payload.fp8.d_scale, stream);
                gemm_cublaslt(fp8_no, fp8_tk, kk, 1.0f, 0.0f, qscratch_.d_act_scale,
                              fp8_hwk->payload.fp8.d_scale, stream);
                gemm_cublaslt(fp8_no, fp8_tv, vv, 1.0f, 0.0f, qscratch_.d_act_scale,
                              fp8_hwv->payload.fp8.d_scale, stream);
            } else {
                // Try fused K+V path: single strided batched GEMM for both
                // projections. Read via WeightRegistry handle — the wcache_
                // map is no longer the lookup mechanism (it remains the
                // storage owner; cleanup happens via wcache_.clear()).
                // Gemma 4 per-layer shapes break strided-batched K+V layout assumptions.
                const Tensor* fused_kv = nullptr;
                Tensor fused_from_handle;
                if (ly.fused_kv_id != kInvalidTensorID) {
                    const auto& h = registry_.handle(ly.fused_kv_id);
                    if (h.payload.fp16.data) {
                        fused_from_handle = Tensor(h.payload.fp16.data, QType::F16, 2, h.shape, true);
                        fused_kv = &fused_from_handle;
                    }
                }
                if (n > 1 && fused_kv && !per_layer_shapes) {
                    // Q: still separate (different output dim with GQA)
                    gemm_via_handle_(ly.wq_id, no, q_target, ctx);
                    // K+V: one batched cuBLAS call
                    gemm_kv_batched(no, *fused_kv, kk, vv, stream);
                } else {
                    // NVFP4-CUTLASS prefill QKV reads the SAME normed input
                    // three times — quantize it into the activation scratch
                    // once (Q's dispatch) and let K/V skip the re-quantize
                    // via the act-quant hint. Bit-identical reuse.
                    GemmContext kv_ctx = ctx;
                    if (n > 1 && prefill_routes_cutlass_nvfp4_(ly.wq_id) &&
                        prefill_routes_cutlass_nvfp4_(ly.wk_id) &&
                        (ly.wv.data == nullptr || prefill_routes_cutlass_nvfp4_(ly.wv_id))) {
                        kv_ctx = ctx.with_act_quant_hint(no.data, n,
                                                         static_cast<int>(no.shape[1]));
                    }
                    gemm_via_handle_(ly.wq_id, no, q_target, ctx);
                    gemm_via_handle_(ly.wk_id, no, kk, kv_ctx);
                    if (ly.wv.data != nullptr) {
                        gemm_via_handle_(ly.wv_id, no, vv, kv_ctx);
                    }
                    // else: K=V sharing path — vv populated below from kk.
                }
            }
        }

        // Apply Q/K/V biases if present (Qwen2) — fused 3-way for 1 launch
        add_bias_3way(qv, ly.q_bias, kk, ly.k_bias, vv, ly.v_bias, stream);

        // LoRA deltas on the raw projection outputs (pre QK-norm / pre RoPE —
        // matches HF PEFT semantics of wrapping the Linear). Every QKV arm
        // above materializes the normed input in `no` (the fused dp4a arm
        // side-writes it when an adapter is active).
        if (lora_) {
            if (const LoraWeights* w = lora_->get(layer, LoraProj::Q))
                lora_delta_(*w, no.data, qv.data, n, stream);
            if (const LoraWeights* w = lora_->get(layer, LoraProj::K))
                lora_delta_(*w, no.data, kk.data, n, stream);
            if (const LoraWeights* w = lora_->get(layer, LoraProj::V))
                lora_delta_(*w, no.data, vv.data, n, stream);
        }
