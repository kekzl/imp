// Decode attention dispatch block for GraphExecutor::run_attention.
#include <utility>
// NOT a standalone TU: textually #include'd inside GraphExecutor::run_attention
// as the body of the `decode_attend` lambda (state/n/qv/kk/vv/ao/
// layer_block_tables are its parameters; everything else is captured).
// Omitted from CMake; must not be compiled alone. `return` replaces the old `goto after_attention`.
        // MLA absorbed decode (opt-in attention.mla_absorb): runs the
        // mathematically-equivalent absorbed attention directly from the latent
        // cache (written just before this branch) - no full K/V materialization,
        // no paged cache read. Single-token decode (n==1) only; multi-token (spec) falls through to the
        // materialized paged path.
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
            return;  // leaves the decode_attend lambda (executor_attention.cu)
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
            if (prof.rope_full_head_dim) {
                effective_rope_dim = hd;
            } else {
                effective_rope_dim = (cfg.rope_dim > 0) ? cfg.rope_dim : hd;
                if (effective_rope_dim > hd)
                    effective_rope_dim = hd;
            }
            const int pairs = effective_rope_dim / 2;
            const float inv_scaling = 1.0f / layer_rope_freq_scale;
            // The lambda's K/V views: rows [row_begin, row_begin + n) of the
            // shared workspaces (the whole batch, or a mixed step's riders).
            Tensor kv_view = kk;
            Tensor vv_view = vv;
            dim3 fused_grid(n, 2);
            write_kv_cache_rope_fused(
                fused_grid, threads, stream, static_cast<const half*>(kv_view.data),
                static_cast<const half*>(vv_view.data),
                state.positions, layer_block_tables, static_cast<half*>(cache->k_ptr(kv_layer, 0)),
                static_cast<half*>(cache->v_ptr(kv_layer, 0)), block_stride, row_elems, kv_block_size_d, n,
                state.max_blocks_per_seq, state.n_sequences, nkv, hd, layer_rope_theta, inv_scaling, pairs,
                cfg.rope_neox, longrope_freqs);
            // Sparse decode attention metadata is updated for ALL layers in one
            // batched launch at the end of the forward: selection force-includes the
            // recent blocks, so the one-step lag is harmless.
        } else {
            write_kv_cache(layer, state, stream, row_begin, n);
        }

        // DEBUG: force cuBLAS attention for decode to isolate paged attention bugs.
        // When enabled, uses the same materialized QK^T path as prefill.
        const bool force_cublas_decode = dispatch_policy().attention.force_cublas_decode;
        if (force_cublas_decode && n == 1 && attn_scores_buf_ &&
            cublas_decode_reference(state, layer, get_kv_layer(kv_layer_map_, layer), layer_block_tables, qv,
                                    ao, nh, nkv, hd, scale, cfg.attn_logit_softcap, layer_sliding_window,
                                    attn_sinks, stream))
            return;  // leaves the decode_attend lambda (executor_attention.cu)

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
        // nh*vhd. reshape() enforces equal numel, so build the narrower view from
        // the pointer directly when vhd!=hd; otherwise reshape (identical layout).
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

        // Sparse decode attention (attention.sparse_topk_tokens): scores all
        // context blocks against current queries, swaps in a compacted block
        // table + context lens; contexts at/below budget pass through bit-
        // identically. Covers plain decode AND spec verify chunks (already
        // per-row "sequences"). SWA/streaming layers keep full attention; the init
        // gate limits it to F16/FP8/NVFP4 caches, no absorbed MLA, uniform geometry.
        const int* attn_bt = layer_block_tables;
        const int* attn_ctx_lens = state.context_lens;
        int attn_max_blocks = state.max_blocks_per_seq;
        // max_context_len bounds every row (a captured graph holds its pow2 high-water mark): at or
        // below the engage length the selection is an identity copy, so skip both kernels (6.7 us
        // per layer at 32 streams, Qwen3-14B-NVFP4).
        if (qscratch_.sparse_budget_blocks > 0 && cache->key_minmax_enabled() &&
            state.max_context_len > qscratch_.sparse_engage_blocks * kv_bs && layer_sliding_window <= 0 &&
            layer_n_sinks <= 0 && n == n_seq && n_seq <= qscratch_.sparse_max_rows && attn_max_blocks > 0 &&
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
                                 qscratch_.sparse_engage_blocks, qscratch_.sparse_table_blocks,
                                 qscratch_.sparse_scores, qscratch_.sparse_block_tables,
                                 qscratch_.sparse_context_lens, qscratch_.sparse_score_meanstd,
                                 qscratch_.sparse_score_std_coef, stream);
            attn_bt = qscratch_.sparse_block_tables;
            attn_ctx_lens = qscratch_.sparse_context_lens;
            attn_max_blocks = qscratch_.sparse_table_blocks;
        }
        // NVFP4 split-K sizes on the context the kernel walks (#2440): past the engage length a sparse row
        // holds the budget, below it the identity copy of the whole context.
        const bool sparse_selects = attn_bt != layer_block_tables &&
                                    state.max_context_len > qscratch_.sparse_engage_blocks* kv_bs;
        const int attn_max_ctx = sparse_selects
                                     ? std::min(state.max_context_len, qscratch_.sparse_budget_blocks* kv_bs)
                                     : state.max_context_len;

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
            // BitDecoding TC dispatch opt-in (kv_cache.bitdecoding_qk): routes to the
            // WMMA-Q.K variant; default keeps scalar-FFMA. The TC variant has no sink
            // pointer and its residual reduce would need its own (#1345), so a sink
            // model stays on the scalar NVFP4 kernel, which applies sinks correctly.
            bool use_bitdecoding_tc = dispatch_policy().kv_cache.bitdecoding_qk;
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
                // QW6 (#604): gate the entire residual-arg marshalling (8+ trailing args)
                // behind one null-check. Default path (residual_on=false) uses the
                // function's default arguments, no explicit marshalling at the call site.
                const bool residual_on = state.kv_manager != nullptr && state.kv_manager->residual_enabled();
                const uint8_t* k_scales = static_cast<const uint8_t*>(cache->k_scale_ptr(kv_layer, 0));
                const uint8_t* v_scales = static_cast<const uint8_t*>(cache->v_scale_ptr(kv_layer, 0));
                if (!residual_on) {
                    dispatch_record::set_attn_decode(AttnDecodePath::NVFP4_TC);
                    paged_attention_decode_nvfp4_tc(q4, k_c, v_c, o4, k_scales, v_scales, attn_bt,
                                                    attn_ctx_lens, kv_bs, scale, attn_max_ctx,
                                                    layer_sliding_window, cfg.attn_logit_softcap, stream,
                                                    attn_max_blocks);
                } else {
                    // Phase 3b residual read, two activation paths: multi-seq
                    // (state.d_residual_seq_slots != nullptr) reads per-batch device metadata
                    // (batched decode); single-seq legacy (state.kv_seq_id >= 0) uses the
                    // scalar form (unmigrated single-seq decode).
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
                    paged_attention_decode_nvfp4_tc(q4, k_c, v_c, o4, k_scales, v_scales, attn_bt,
                                                    attn_ctx_lens, kv_bs, scale, attn_max_ctx,
                                                    layer_sliding_window, cfg.attn_logit_softcap, stream,
                                                    attn_max_blocks, 0, k_res, v_res, res_count, res_n,
                                                    res_widx, k_res_base, v_res_base, res_seq_stride_elems,
                                                    state.d_residual_seq_slots, state.d_residual_counts,
                                                    state.d_residual_write_idxes,
                                                    state.kv_manager->d_residual_fc_ptr(),
                                                    state.kv_manager->d_residual_widx_ptr());
                }
            } else {
                paged_attention_decode_nvfp4(q4, k_c, v_c, o4,
                                             static_cast<const uint8_t*>(cache->k_scale_ptr(kv_layer, 0)),
                                             static_cast<const uint8_t*>(cache->v_scale_ptr(kv_layer, 0)),
                                             attn_bt, attn_ctx_lens, kv_bs, scale, attn_max_ctx,
                                             layer_sliding_window, cfg.attn_logit_softcap, stream,
                                             attn_max_blocks, layer_n_sinks, attn_sinks);
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
            // MLA: V slots are hd wide, zero past vhd (mla_assemble_kv v_dst_head_dim=hd): symmetric decode
            // (split-K, templated HD) into kk, dead after the KV write above, then compaction (#2374).
            paged_attention_decode_mla_padded(q4, k_c, v_c, o4, static_cast<half*>(kk.data), kk.numel(),
                                              attn_bt, attn_ctx_lens, kv_bs, scale, state.max_context_len,
                                              layer_sliding_window, cfg.attn_logit_softcap, stream,
                                              attn_max_blocks, layer_n_sinks, attn_sinks, vhd);
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
