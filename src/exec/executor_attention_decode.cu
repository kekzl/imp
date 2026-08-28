// Decode attention dispatch block for GraphExecutor::run_attention.
#include <utility>
//
// This is NOT a standalone translation unit — it is textually #include'd inside
// the body of GraphExecutor::run_attention (executor_attention.cu), inside the
// `else { ... }` (decode) branch. It is therefore omitted from the CMake source
// list and must not be compiled on its own. The contents are byte-for-byte the
// original inline block; see executor_attention.cu for surrounding context and
// local variables in scope (including the `after_attention` label).
        // MLA absorbed decode (Phase 3, opt-in attention.mla_absorb): the
        // current token's latent + decoupled key were just written into the
        // per-layer latent cache (above, before the prefill/decode branch). Run
        // the mathematically-equivalent absorbed attention directly from the
        // latent cache — no full per-head K/V materialization, no paged cache
        // read. Single-token decode (n==1) only; multi-token (spec) decode falls
        // through to the materialized paged path.
        if (mla_absorb_active && n == 1) {
            half* cache_layer = static_cast<half*>(mla_absorb_cache_) +
                                static_cast<size_t>(layer) * mla_absorb_layer_stride_;
            // ao is [n, nh*hd]; the absorbed kernel writes nh*v_head_dim compact
            // (same layout the paged decode kernel produces), narrowed downstream.
            mla_absorbed_decode(static_cast<const half*>(qv.data),
                                static_cast<const half*>(ly.kv_b_proj.data), cache_layer,
                                static_cast<half*>(ao.data), mla_absorb_scores_, state.context_lens,
                                nh, hd, cfg.qk_rope_head_dim, cfg.qk_nope_head_dim, cfg.kv_lora_rank,
                                cfg.v_head_dim, mla_absorb_max_seq_, scale, stream);
            goto after_attention;
        }

        // Decode: write new token's K/V to cache first
        if (rope_k_deferred) {
            // Fused: apply RoPE to K during KV cache write (saves 1 kernel launch)
            int kv_layer = get_kv_layer(kv_layer_map_, layer);
            KVCache* cache = state.kv_cache;
            const int kv_block_size_d = cache->block_size();
            int row_elems = nkv * hd;
            int block_stride = kv_block_size_d * row_elems;
            int threads = std::min(row_elems, 256);
            // Per-layer rope_dim (same as prefill rope path above): Gemma 4
            // uses full hd; longrope_freqs encodes the partial-rotary schedule.
            int effective_rope_dim;
            if (prof.is_gemma4) {
                effective_rope_dim = hd;
            } else {
                effective_rope_dim = (cfg.rope_dim > 0) ? cfg.rope_dim : hd;
                if (effective_rope_dim > hd)
                    effective_rope_dim = hd;
            }
            const int pairs = effective_rope_dim / 2;
            const float inv_scaling = 1.0f / layer_rope_freq_scale;
            Tensor kv_view = view_tokens(k_, n);
            Tensor vv_view = view_tokens(v_, n);
            dim3 fused_grid(n, 2);
            write_kv_cache_rope_fused_kernel<<<fused_grid, threads, 0, stream>>>(
                static_cast<const half*>(kv_view.data), static_cast<const half*>(vv_view.data),
                state.positions, layer_block_tables, static_cast<half*>(cache->k_ptr(kv_layer, 0)),
                static_cast<half*>(cache->v_ptr(kv_layer, 0)), block_stride, row_elems, kv_block_size_d, n,
                state.max_blocks_per_seq, state.n_sequences, nkv, hd, layer_rope_theta, inv_scaling, pairs,
                cfg.rope_neox, longrope_freqs);
            IMP_CUDA_CHECK_LAUNCH();
            // Sparse decode attention: maintain per-block key min/max (reads
            // the post-RoPE K rows back from the cache). Gate on the layer's
            // STATIC window so a runtime StreamingLLM override cannot leave
            // holes in the metadata of a full-attention layer.
            if (cache->key_minmax_enabled() && layer_swa_window(cfg, prof, layer) <= 0) {
                sparse_update_key_minmax(cache->qtype(), cache->k_ptr(kv_layer, 0),
                                         cache->key_minmax_ptr(kv_layer, 0), state.positions,
                                         layer_block_tables, nkv, hd, kv_block_size_d, n,
                                         state.max_blocks_per_seq, state.n_sequences, stream);
            }
        } else {
            write_kv_cache(layer, state, stream);
        }

        // DEBUG: force cuBLAS attention for decode to isolate paged attention bugs.
        // When enabled, uses the same materialized QK^T path as prefill.
        const bool force_cublas_decode = runtime_config().attention.force_cublas_decode;
        if (force_cublas_decode && n == 1 && attn_scores_buf_) {
            // Reconstruct K/V from cache for this position
            KVCache* cache_dbg = state.kv_cache;
            int kv_layer_dbg = get_kv_layer(kv_layer_map_, layer);
            int ctx_len = 0;
            cudaMemcpy(&ctx_len, state.context_lens, sizeof(int), cudaMemcpyDeviceToHost);
            // Allocate temp K/V for all context tokens
            int kv_elems = ctx_len * nkv * hd;
            half *k_flat = nullptr, *v_flat = nullptr;
            cudaMalloc(&k_flat, kv_elems * sizeof(half));
            cudaMalloc(&v_flat, kv_elems * sizeof(half));
            // Copy from paged KV cache to contiguous buffer
            int kv_bs = cache_dbg->block_size();
            int n_blocks = (ctx_len + kv_bs - 1) / kv_bs;
            // Heap-sized to n_blocks — a fixed [1024] stack array overran 4x at
            // the 64K context cap (n_blocks=4096) and silently smashed the frame.
            std::vector<int32_t> h_block_table(n_blocks > 0 ? n_blocks : 1);
            cudaMemcpy(h_block_table.data(), layer_block_tables, n_blocks * sizeof(int32_t),
                       cudaMemcpyDeviceToHost);
            // SWA trailing-free holes (-1) leave their rows zeroed — they sit
            // outside the window mask, so the reference output is unaffected.
            cudaMemset(k_flat, 0, kv_elems * sizeof(half));
            cudaMemset(v_flat, 0, kv_elems * sizeof(half));
            for (int b = 0; b < n_blocks; b++) {
                int block_id = h_block_table[b];
                if (block_id < 0)
                    continue;
                int toks_in_block = std::min(kv_bs, ctx_len - b * kv_bs);
                size_t row_bytes = nkv * hd * sizeof(half);
                half* k_src = static_cast<half*>(cache_dbg->k_ptr(kv_layer_dbg, block_id));
                half* v_src = static_cast<half*>(cache_dbg->v_ptr(kv_layer_dbg, block_id));
                cudaMemcpy(k_flat + b * kv_bs * nkv * hd, k_src, toks_in_block * row_bytes,
                           cudaMemcpyDeviceToDevice);
                cudaMemcpy(v_flat + b * kv_bs * nkv * hd, v_src, toks_in_block * row_bytes,
                           cudaMemcpyDeviceToDevice);
            }
            // Reshape for cuBLAS attention: Q[1,nh,hd], K[ctx_len,nkv,hd], V[ctx_len,nkv,hd]
            int64_t k_shape[2] = {ctx_len, nkv * hd};
            int64_t v_shape[2] = {ctx_len, nkv * hd};
            Tensor k_cont(k_flat, QType::F16, 2, k_shape, true);
            Tensor v_cont(v_flat, QType::F16, 2, v_shape, true);
            // n=1 cuBLAS attention. causal=true + q_offset=ctx_len-1 makes the
            // single query row see the full context AND keeps the sliding-window
            // mask correct (the old causal=false/q_offset=0 form broke SWA
            // layers: abs_row=0 never triggered the window). Sinks pass through
            // so this debug arm stays a faithful reference for gpt-oss (#547).
            {
                int64_t s_shape[3] = {(int64_t)nh, 1, (int64_t)ctx_len};
                half* s_buf = nullptr;
                cudaMalloc(&s_buf, (size_t)nh * ctx_len * sizeof(half));
                Tensor s_view(s_buf, QType::F16, 3, s_shape, true);
                attention_cublas_prefill(qv, k_cont, v_cont, ao, s_view, nh, nkv, hd, scale, /*causal=*/true,
                                         cfg.attn_logit_softcap, /*q_offset=*/ctx_len - 1, stream,
                                         layer_sliding_window, attn_sinks);
                cudaFree(s_buf);
            }
            cudaFree(k_flat);
            cudaFree(v_flat);
            goto after_attention;
        }

        // Paged attention: Q shape depends on batch size
        int n_seq = state.n_sequences;
        // For MLA: V head dim may be narrower than Q/K head dim.
        // vhd == hd for all non-MLA models (v_head_dim == 0 or == head_dim).
        const int vhd = (cfg.is_mla() && cfg.v_head_dim > 0 && cfg.v_head_dim != hd)
                            ? cfg.v_head_dim : hd;
        // For decode, n_tokens == n_sequences (one token per seq)
        int64_t qd[4] = {n_seq, 1, nh, hd};
        // Output is [batch, 1, n_heads, vhd] — narrower for MLA
        int64_t od[4] = {n_seq, 1, nh, vhd};
        Tensor q4 = qv.reshape(4, qd);
        // attn_out_ is allocated for nh*hd; for MLA the live output is only
        // nh*vhd (the paged kernel writes nh*vhd contiguously per token).
        // reshape() enforces equal numel, so build the narrower view from the
        // pointer directly when vhd != hd; otherwise reshape (identical layout).
        Tensor o4 = (vhd != hd) ? Tensor(ao.data, ao.qtype, 4, od, /*on_device=*/true)
                                : ao.reshape(4, od);

        KVCache* cache = state.kv_cache;
        const int kv_bs = cache->block_size();
        int total_blk = cache->total_blocks();
        QType cache_dtype = cache->qtype();
        int64_t cs[4] = {static_cast<int64_t>(total_blk), static_cast<int64_t>(kv_bs),
                         static_cast<int64_t>(nkv), static_cast<int64_t>(hd)};
        // Use mapped KV layer index for hybrid models (attention layers only)
        int kv_layer = get_kv_layer(kv_layer_map_, layer);
        Tensor k_c(cache->k_ptr(kv_layer, 0), cache_dtype, 4, cs, true);
        Tensor v_c(cache->v_ptr(kv_layer, 0), cache_dtype, 4, cs, true);

        // L2 persistence hint: keep this layer's KV cache in L2 during attention.
        // RTX 5090 has 96 MB L2 — enough for ~3K tokens of KV at FP8.
        set_l2_persist_kv(stream, k_c.data, k_c.nbytes() + v_c.nbytes());

        // Sparse decode attention (attention.sparse_topk_tokens): score all
        // context blocks against the current queries and swap in a compacted
        // block table + context lens. The paged kernels below are unmodified;
        // contexts at or below the budget pass through bit-identically. Only
        // plain decode on full-attention layers (spec verify chunks and
        // SWA/streaming layers keep full attention; the init gate limits the
        // metadata pool to F16/FP8 caches, non-MLA, uniform geometry).
        const int* attn_bt = layer_block_tables;
        const int* attn_ctx_lens = state.context_lens;
        int attn_max_blocks = state.max_blocks_per_seq;
        if (qscratch_.sparse_budget_blocks > 0 && cache->key_minmax_enabled() &&
            layer_sliding_window <= 0 && layer_n_sinks <= 0 && !state.chunk_decode_attn &&
            n == n_seq && attn_max_blocks > 0 &&
            attn_max_blocks <= qscratch_.sparse_max_ctx_blocks && nh / nkv <= 16) {
            // Proof-of-activity for A/B arms: the selection itself is decided
            // device-side, but max_context_len is a host value - once it
            // exceeds the budget, at least one sequence is actually sparse.
            static bool logged_sparse_active = false;
            if (!logged_sparse_active &&
                state.max_context_len > qscratch_.sparse_budget_blocks * kv_bs) {
                logged_sparse_active = true;
                IMP_LOG_INFO("sparse decode attention ACTIVE: ctx %d > budget %d tokens",
                             state.max_context_len, qscratch_.sparse_budget_blocks * kv_bs);
            }
            sparse_select_blocks(static_cast<const half*>(q4.data),
                                 cache->key_minmax_ptr(kv_layer, 0), layer_block_tables,
                                 state.context_lens, n_seq, nh, nkv, hd, kv_bs,
                                 state.max_blocks_per_seq, qscratch_.sparse_budget_blocks,
                                 qscratch_.sparse_sink_blocks, qscratch_.sparse_recent_blocks,
                                 qscratch_.sparse_scores, qscratch_.sparse_block_tables,
                                 qscratch_.sparse_context_lens, stream);
            attn_bt = qscratch_.sparse_block_tables;
            attn_ctx_lens = qscratch_.sparse_context_lens;
            attn_max_blocks = qscratch_.sparse_budget_blocks;
        }

        if (cache_dtype == QType::INT4) {
            dispatch_record::set_attn_decode(AttnDecodePath::INT4);
            // INT4 paged attention with per-head scales and INT4 unpack (Split-K enabled)
            paged_attention_set_splitk_scratch(qscratch_.splitk, qscratch_.splitk_size);
            paged_attention_decode_int4(q4, k_c, v_c, o4,
                                        static_cast<const half*>(cache->k_scale_ptr(kv_layer, 0)),
                                        static_cast<const half*>(cache->v_scale_ptr(kv_layer, 0)),
                                        layer_block_tables, state.context_lens, kv_bs, scale,
                                        state.max_context_len, layer_sliding_window, cfg.attn_logit_softcap,
                                        stream, state.max_blocks_per_seq, layer_n_sinks, attn_sinks);
        } else if (cache_dtype == QType::NVFP4) {
            // TC vs non-TC is decided per shape below.
            dispatch_record::set_attn_decode(AttnDecodePath::NVFP4);
            // NVFP4 paged attention: packed FP4 + UE4M3 per-group_of_16 scales (Split-K enabled)
            paged_attention_set_splitk_scratch(qscratch_.splitk, qscratch_.splitk_size);
            // BitDecoding TC dispatch opt-in: kv_cache.bitdecoding_qk routes to the WMMA-Q.K variant; default
            // keeps the scalar-FFMA path unchanged.
            // The BitDecoding TC variant has no sink pointer, and its residual
            // reduce is a second reduction that would need its own (#1345). A
            // sink model therefore stays on the scalar NVFP4 kernel, which
            // applies them — silently dropping the sink column is what this
            // whole issue is about.
            bool use_bitdecoding_tc = runtime_config().kv_cache.bitdecoding_qk;
            if (use_bitdecoding_tc && attn_sinks) {
                static bool warned_tc_sinks = false;
                if (!warned_tc_sinks) {
                    warned_tc_sinks = true;
                    IMP_LOG_WARN(
                        "kv_cache.bitdecoding_qk: the NVFP4 tensor-core decode kernel does not "
                        "apply learned attention sinks — using the scalar NVFP4 kernel for this "
                        "model instead (#1345).");
                }
                use_bitdecoding_tc = false;
            }
            if (use_bitdecoding_tc) {
                // QW6 from the phase-5 review §2.1 (archived in #604): gate the entire
                // residual-arg marshalling (8+ trailing args) behind a single
                // null-check. Default path (residual_on=false) uses the
                // function's default arguments — no explicit nullptr/0 marshalling
                // at the call site, no dead per-layer state lookups.
                const bool residual_on = state.kv_manager != nullptr && state.kv_manager->residual_enabled();
                const uint8_t* k_scales = static_cast<const uint8_t*>(cache->k_scale_ptr(kv_layer, 0));
                const uint8_t* v_scales = static_cast<const uint8_t*>(cache->v_scale_ptr(kv_layer, 0));
                if (!residual_on) {
                    dispatch_record::set_attn_decode(AttnDecodePath::NVFP4_TC);
                    paged_attention_decode_nvfp4_tc(q4, k_c, v_c, o4, k_scales, v_scales, layer_block_tables,
                                                    state.context_lens, kv_bs, scale, state.max_context_len,
                                                    layer_sliding_window, cfg.attn_logit_softcap, stream,
                                                    state.max_blocks_per_seq);
                } else {
                    // Phase 3b residual read. Two activation paths:
                    //   (multi-seq) state.d_residual_seq_slots != nullptr: kernel
                    //     reads per-batch metadata from the device arrays. Used by
                    //     batched decode.
                    //   (single-seq legacy) state.kv_seq_id >= 0: kernel uses the
                    //     scalar form. Used by single-seq decode that hasn't been
                    //     migrated to the array form (e.g. early-init smoke).
                    const half* k_res = nullptr;
                    const half* v_res = nullptr;
                    int res_count = 0;
                    int res_n = state.kv_manager->residual_n_tokens();
                    int res_widx = 0;
                    const half* k_res_base = nullptr;
                    const half* v_res_base = nullptr;
                    int res_seq_stride_elems = 0;
                    if (state.d_residual_seq_slots != nullptr) {
                        // Multi-seq array form
                        k_res_base = static_cast<const half*>(
                            state.kv_manager->residual_k_layer_base(kv_layer));
                        v_res_base = static_cast<const half*>(
                            state.kv_manager->residual_v_layer_base(kv_layer));
                        res_seq_stride_elems = static_cast<int>(
                            state.kv_manager->residual_seq_stride_bytes() / sizeof(__half));
                    } else if (state.n_sequences == 1 && state.kv_seq_id >= 0) {
                        // Single-seq scalar form
                        k_res = static_cast<const half*>(
                            state.kv_manager->residual_k_ptr(state.kv_seq_id, kv_layer));
                        v_res = static_cast<const half*>(
                            state.kv_manager->residual_v_ptr(state.kv_seq_id, kv_layer));
                        auto rs = state.kv_manager->residual_state(state.kv_seq_id);
                        res_count = rs.fill_count;
                        res_widx = rs.write_idx;
                    }
                    dispatch_record::set_attn_decode(AttnDecodePath::NVFP4_TC);
                    paged_attention_decode_nvfp4_tc(q4, k_c, v_c, o4, k_scales, v_scales, layer_block_tables,
                                                    state.context_lens, kv_bs, scale, state.max_context_len,
                                                    layer_sliding_window, cfg.attn_logit_softcap, stream,
                                                    state.max_blocks_per_seq, 0, k_res, v_res, res_count,
                                                    res_n, res_widx, k_res_base, v_res_base,
                                                    res_seq_stride_elems, state.d_residual_seq_slots,
                                                    state.d_residual_counts, state.d_residual_write_idxes,
                                                    state.kv_manager->d_residual_fc_ptr(),
                                                    state.kv_manager->d_residual_widx_ptr());
                }
            } else {
                paged_attention_decode_nvfp4(q4, k_c, v_c, o4,
                                             static_cast<const uint8_t*>(cache->k_scale_ptr(kv_layer, 0)),
                                             static_cast<const uint8_t*>(cache->v_scale_ptr(kv_layer, 0)),
                                             layer_block_tables, state.context_lens, kv_bs, scale,
                                             state.max_context_len, layer_sliding_window,
                                             cfg.attn_logit_softcap, stream, state.max_blocks_per_seq,
                                             layer_n_sinks, attn_sinks);
            }
        } else if (cache_dtype == QType::MXFP4_KV) {
            dispatch_record::set_attn_decode(AttnDecodePath::MXFP4_KV);
            // MXFP4-KV paged attention: same kernel as NVFP4 but with UE8M0 scale decode.
            // BitDecoding TC path not supported for MXFP4_KV in Slice 2 — always scalar.
            paged_attention_set_splitk_scratch(qscratch_.splitk, qscratch_.splitk_size);
            paged_attention_decode_mxfp4_kv(q4, k_c, v_c, o4,
                                            static_cast<const uint8_t*>(cache->k_scale_ptr(kv_layer, 0)),
                                            static_cast<const uint8_t*>(cache->v_scale_ptr(kv_layer, 0)),
                                            layer_block_tables, state.context_lens, kv_bs, scale,
                                            state.max_context_len, layer_sliding_window,
                                            cfg.attn_logit_softcap, stream, state.max_blocks_per_seq,
                                            layer_n_sinks, attn_sinks);
        } else if (cache_dtype == QType::INT8) {
            dispatch_record::set_attn_decode(AttnDecodePath::INT8);
            // INT8 dp4a paged attention with per-head scales (Split-K enabled)
            paged_attention_set_splitk_scratch(qscratch_.splitk, qscratch_.splitk_size);
            paged_attention_decode_int8(q4, k_c, v_c, o4,
                                        static_cast<const half*>(cache->k_scale_ptr(kv_layer, 0)),
                                        static_cast<const half*>(cache->v_scale_ptr(kv_layer, 0)),
                                        layer_block_tables, state.context_lens, kv_bs, scale,
                                        state.max_context_len, layer_sliding_window, cfg.attn_logit_softcap,
                                        stream, state.max_blocks_per_seq, layer_n_sinks, attn_sinks);
        } else if (cache_dtype == QType::FP8_E4M3) {
            dispatch_record::set_attn_decode(AttnDecodePath::FP8);
            // FP8 paged attention with on-the-fly dequant (Split-K enabled)
            float kv_scale = (!kv_scales_.empty() && kv_layer < static_cast<int>(kv_scales_.size()))
                                 ? kv_scales_[kv_layer]
                                 : 1.0f;
            paged_attention_set_splitk_scratch(qscratch_.splitk, qscratch_.splitk_size);
            paged_attention_decode_fp8(q4, k_c, v_c, o4, attn_bt, attn_ctx_lens, kv_bs, scale,
                                       kv_scale, state.max_context_len, layer_sliding_window,
                                       cfg.attn_logit_softcap, stream, attn_max_blocks,
                                       layer_n_sinks, attn_sinks);
        } else {
            dispatch_record::set_attn_decode(AttnDecodePath::FP16);
            paged_attention_set_splitk_scratch(qscratch_.splitk, qscratch_.splitk_size);
            paged_attention_decode(q4, k_c, v_c, o4, attn_bt, attn_ctx_lens, kv_bs, scale,
                                   state.max_context_len, layer_sliding_window, cfg.attn_logit_softcap,
                                   stream, attn_max_blocks, layer_n_sinks, attn_sinks, vhd);
        }
        if (attn_sinks && !paged_attention_applies_sinks(cache_dtype)) {
            static bool warned_sinks_kv = false;
            if (!warned_sinks_kv) {
                warned_sinks_kv = true;
                IMP_LOG_WARN("attention sinks: only the FP16 paged decode kernels apply sinks — "
                             "kv_cache.dtype=%d ignores them; use fp16 KV for this model.",
                             std::to_underlying(cache_dtype));
            }
        }

        // Clear L2 persistence hint (weights loaded next need L2 space)
        clear_l2_persist(stream);
