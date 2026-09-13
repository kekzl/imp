// NOT a standalone TU: textually #include'd inside GraphExecutor::run_attention,
// between the braces of the QKV-projection scope. Omitted from CMake; must
// not be compiled alone. Byte-for-byte the original inline block; see
// executor_attention.cu for surrounding context and local variables in scope.
        if (prof.attn_variant == AttnVariant::MLA) {
            // MLA (DeepSeek-V2/V3) materialized two-step KV projection.
            // kv_a = norm_out @ kv_a_proj^T [n, kv_lora_rank+rope_dim]; latent =
            // rmsnorm(kv_a[:,:kv_lora_rank]); k_rope = kv_a[:,kv_lora_rank:] (MQA-style
            // shared); kv_b = latent @ kv_b_proj^T [n, n_heads*(nope_dim+v_head_dim)];
            // K[h] = [pe(rope_dim)|nope(nope_dim)], V[h] = kv_b's last v_head_dim dims.
            // RoPE layout: pe FIRST so rope_forward's leading-dims rotation applies
            // unchanged; Q is reordered from HF [nope|pe] to [pe|nope] to match K.
            // LIMITATION: LoRA deltas are NOT applied on the MLA path (Q-proj delta and
            // the latent kv_a/kv_b projections have no LoRA wiring); a LoRA adapter on
            // a DeepSeek MLA model is silently ignored except for a one-time warning.
            if (lora_ && lora_->has_any()) {
                static bool mla_lora_warned = false;
                if (!mla_lora_warned) {
                    mla_lora_warned = true;
                    IMP_LOG_WARN("LoRA: adapter is loaded but the MLA attention path "
                                 "(DeepSeek-V2/V3) does not apply Q/kv_a/kv_b LoRA deltas "
                                 "— adapter ignored for attention projections");
                }
            }

            // 1. Attention RMSNorm: norm_out = rmsnorm(hidden, attn_norm).
            // Helper (with no small-M consumer named) keeps the producer-
            // quantize tag honest when this arm rewrites a tagged buffer.
            rmsnorm_for_smallm_(h, ly.attn_norm, no, kInvalidTensorID, n, eps, stream, norm_w_off_);

            const int kv_lora_rank   = cfg.kv_lora_rank;
            const int rope_dim       = cfg.qk_rope_head_dim;
            const int nope_dim       = cfg.qk_nope_head_dim;
            const int v_head_dim_mla = cfg.v_head_dim;
            const int kva_out        = kv_lora_rank + rope_dim;
            const int kvb_out        = nh * (nope_dim + v_head_dim_mla);

            // 2. Q projection + reorder [nope|pe] -> [pe|nope] in-place.
            gemm_via_handle_(ly.wq_id, no, qv, ctx);
            mla_reorder_q(static_cast<half*>(qv.data), n, nh, nope_dim, rope_dim, stream);

            // kv_a scratch is pre-allocated (mla_*_buf_), NOT cudaMallocAsync'd here:
            // stream-ordered alloc/free is rejected inside CUDA-graph capture (decode loop), silently falling
            // back to eager and degenerating.
            void* kv_a_buf = mla_kv_a_buf_;
            {
                int64_t kva_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(kva_out)};
                Tensor kv_a(kv_a_buf, QType::F16, 2, kva_shape, true);
                gemm_via_handle_(ly.kv_a_proj_id, no, kv_a, ctx);

                // 4. Extract latent [n, kv_lora_rank] — first kv_lora_rank cols of kv_a.
                void* latent_buf = mla_latent_buf_;
                IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(
                    latent_buf,
                    static_cast<size_t>(kv_lora_rank) * sizeof(half),
                    kv_a_buf,
                    static_cast<size_t>(kva_out) * sizeof(half),
                    static_cast<size_t>(kv_lora_rank) * sizeof(half),
                    static_cast<size_t>(n),
                    cudaMemcpyDeviceToDevice, stream));

                // 5. Extract k_rope [n, rope_dim] — last rope_dim cols of kv_a.
                void* k_rope_buf = mla_k_rope_buf_;
                IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(
                    k_rope_buf,
                    static_cast<size_t>(rope_dim) * sizeof(half),
                    static_cast<const char*>(kv_a_buf) + static_cast<size_t>(kv_lora_rank) * sizeof(half),
                    static_cast<size_t>(kva_out) * sizeof(half),
                    static_cast<size_t>(rope_dim) * sizeof(half),
                    static_cast<size_t>(n),
                    cudaMemcpyDeviceToDevice, stream));

                // 6. latent = rmsnorm(latent, kv_a_layernorm)
                {
                    int64_t lat_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(kv_lora_rank)};
                    Tensor lat(latent_buf, QType::F16, 2, lat_shape, true);
                    rmsnorm(lat, ly.kv_a_layernorm, lat, eps, stream);
                }

                // 7. kv_b = latent @ kv_b_proj^T   [n, kvb_out]
                void* kv_b_buf = mla_kv_b_buf_;
                {
                    int64_t lat_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(kv_lora_rank)};
                    int64_t kvb_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(kvb_out)};
                    Tensor lat(latent_buf, QType::F16, 2, lat_shape, true);
                    Tensor kvb(kv_b_buf,   QType::F16, 2, kvb_shape, true);
                    gemm_via_handle_(ly.kv_b_proj_id, lat, kvb, ctx);
                }

                // Scatter into K [pe|nope] and V. V is [n,n_heads,hd], over-allocated to
                // K's head_dim (hd=rope_dim+nope_dim): real v_head_dim values first, tail
                // zeroed. This lets prefill kernels (which assume V shares K's head_dim)
                // and the paged-decode KV write see a uniform hd-wide V layout; the zero tail contributes
                // nothing to P.V.
                const int v_dst_hd = rope_dim + nope_dim;  // == hd
                mla_assemble_kv(
                    static_cast<const half*>(kv_b_buf),
                    static_cast<const half*>(k_rope_buf),
                    static_cast<half*>(kk.data),
                    static_cast<half*>(vv.data),
                    n, nh, nope_dim, v_head_dim_mla, rope_dim, stream, v_dst_hd);
            }
            // No cudaFree: mla_*_buf_ are persistent workspace (graph-safe).
        } else {
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
        // NVFP4 decode path uses FP16 input (no Q8_1 quantization). Two sources:
        // primary NVFP4 storage tier (native models), and the secondary NVFP4
        // decode cache for Q8_0/Q6_K/Q5_K models (c8763ad regressed this fallback; restored by also probing
        // wcache_.nvfp4).
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
            // Gemma-4 FP32 accum path: read the FP32 residual directly to avoid the
            // FP16 round-trip that drops ~1-2% precision per layer and drifts the
            // last-token hidden state (sign-flip at L29).
            if (using_fp32_accum && prof.is_gemma4) {
                Tensor fp32_h = view_tokens(fp32_hidden_, n);
                rmsnorm_fp32_to_fp16(fp32_h, ly.attn_norm, no, eps, stream, norm_w_off_);
            } else {
                // Producer fusion: quantize into the small-M scratch inside
                // the norm kernel when Q will take that route (batched
                // decode, CUTLASS_NVFP4 tier); falls back to plain rmsnorm.
                rmsnorm_for_smallm_(h, ly.attn_norm, no, ly.wq_id, n, eps, stream, norm_w_off_);
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
                // Fused K+V path: one strided batched GEMM for both projections, looked
                // up via WeightRegistry handle (wcache_ is the storage owner only, not the
                // lookup mechanism now). Gemma 4 per-layer shapes break the strided-batched K+V layout
                // assumption.
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
                    // NVFP4-CUTLASS prefill QKV reads the SAME normed input three times:
                    // quantize once (Q's dispatch) into the activation scratch, K/V skip the
                    // re-quantize via the act-quant hint. Bit-identical reuse.
                    GemmContext kv_ctx = ctx;
                    if (n > 1 && prefill_routes_cutlass_nvfp4_(ly.wq_id, n) &&
                        prefill_routes_cutlass_nvfp4_(ly.wk_id, n) &&
                        (ly.wv.data == nullptr || prefill_routes_cutlass_nvfp4_(ly.wv_id, n))) {
                        kv_ctx = ctx.with_act_quant_hint(no.data, n,
                                                         static_cast<int>(no.shape[1]));
                    }
                    // Batched decode on the small-M NVFP4 path: q|k|v as one
                    // launch (the 16-tile k/v shapes ride q's wave instead of
                    // a striped kernel + reduce each). Declines fall through.
                    bool qkv_multi = false;
                    if (n > 1 && ly.wv.data != nullptr) {
                        const TensorID qkv_ids[3] = {ly.wq_id, ly.wk_id, ly.wv_id};
                        Tensor* qkv_outs[3] = {&q_target, &kk, &vv};
                        qkv_multi = try_smallm_multi_dispatch_(qkv_ids, qkv_outs, 3, no, ctx);
                    }
                    // Single-stream decode on gated attention (the ungated
                    // nvfp4_qkv path above already fuses): q|k|v in one
                    // NVFP4 GEMV launch instead of three.
                    if (!qkv_multi && n == 1 && ly.wv.data != nullptr)
                        qkv_multi = try_attn_qkv_fused_m1_(ly, no, q_target, kk, vv, stream);
                    if (!qkv_multi) {
                        gemm_via_handle_(ly.wq_id, no, q_target, ctx);
                        gemm_via_handle_(ly.wk_id, no, kk, kv_ctx);
                        if (ly.wv.data != nullptr) {
                            gemm_via_handle_(ly.wv_id, no, vv, kv_ctx);
                        }
                    }
                    // else: K=V sharing path — vv populated below from kk.
                }
            }
        }

        // Apply Q/K/V biases if present (Qwen2) — fused 3-way for 1 launch
        add_bias_3way(qv, ly.q_bias, kk, ly.k_bias, vv, ly.v_bias, stream);

        // LoRA deltas on the raw projection outputs (pre QK-norm/pre RoPE, matches
        // HF PEFT's wrapped-Linear semantics). Every QKV arm materializes the
        // normed input in `no` (the fused dp4a arm side-writes it when an adapter is active).
        if (lora_) {
            if (const LoraWeights* w = lora_->get(layer, LoraProj::Q))
                lora_delta_(*w, no.data, qv.data, n, stream);
            if (const LoraWeights* w = lora_->get(layer, LoraProj::K))
                lora_delta_(*w, no.data, kk.data, n, stream);
            if (const LoraWeights* w = lora_->get(layer, LoraProj::V))
                lora_delta_(*w, no.data, vv.data, n, stream);
        }
        } // end else (non-MLA path)
