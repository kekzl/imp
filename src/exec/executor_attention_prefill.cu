// Prefill attention dispatch block for GraphExecutor::run_attention.
//
// This is NOT a standalone translation unit — it is textually #include'd inside
// the body of GraphExecutor::run_attention (executor_attention.cu), inside the
// `if (state.is_prefill) { ... }` branch. It is therefore omitted from the CMake
// source list and must not be compiled on its own. The contents are byte-for-
// byte the original inline block; see executor_attention.cu for surrounding
// context and local variables in scope (including the `after_attention` label).
        // Chunked prefill: when prefill_offset > 0, queries from this chunk must
        // attend to past chunks K/V already in the paged cache. Gather past
        // [0, prefill_offset) KV → contiguous, append current chunk, then run
        // rectangular attention_cublas_prefill with q_offset.
        //
        // Note: cudaMallocAsync per layer here violates CLAUDE.md "No cudaMalloc in hot
        // loops". Acknowledged exception — chunked prefill is excluded from CUDA-graph
        // capture (graphs only capture decode), so the alloc is amortised in the
        // memory pool and runs once per chunk per layer, not per token.
        const int q_offset = state.prefill_offset;
        if (q_offset > 0) {
            KVCache* cache = state.kv_cache;
            QType kvt = cache->qtype();
            // Defense-in-depth: engine resolves out-of-scope models to chunk_size=0,
            // so this code only runs for FP16 / FP8 / NVFP4 / MXFP4_KV KV.
            const bool kvt_ok = (kvt == QType::F16 || kvt == QType::FP8_E4M3 || kvt == QType::NVFP4 ||
                                 kvt == QType::MXFP4_KV || kvt == QType::INT4);
            // sliding_active and attn_shapes_vary are now both supported:
            //   - cuBLAS softmax accepts a sliding_window argument (PR feat(attn): sw),
            //   - the rectangular cuBLAS path dispatches per-layer with nh/nkv/hd.
            // KV dtype remains the only hard gate (the gather kernels are
            // wired for FP16 / FP8 / NVFP4 only).
            if (!kvt_ok) {
                IMP_LOG_ERROR(
                    "chunked_prefill: unsupported KV dtype %d at L%d — engine should have prevented this",
                    (int)kvt, layer);
                std::abort();
            }

            int kv_layer = get_kv_layer(kv_layer_map_, layer);
            int kv_bs = cache->block_size();
            int ctx_len = q_offset + n;
            // Of the chunked branches below, ONLY cuBLAS consumes attn_scores_.
            // FP16-QK FA2 (hd=128) and the tiled FMHA dispatch are O(n) and need no
            // S-matrix. Uniform per-layer shapes (GDN/Mamba2 hybrids: zeros on
            // non-attention layers, one distinct nonzero value) are served by the
            // O(n) family too — only truly heterogeneous shapes (Gemma-4 dual
            // head_dim 256/512) and learned sinks (gpt-oss) require cuBLAS.
            const bool shapes_uniform = !per_layer_shapes || attn_shapes_uniform();
            const bool chunk_fa2_serves = shapes_uniform && !attn_sinks && hd == 128 &&
                                          runtime_config().attention.fa2_fp16qk != "never";
            // Tiled FMHA correctness domain: uniform shapes, no learned sinks.
            const bool chunk_fmha_ok = shapes_uniform && !attn_sinks;
            // attn_scores_ is sized [nh, s_cap, s_cap] (square). Chunked cuBLAS stores
            // an [nh, n, ctx_len] FP16 matrix (or FP32 = 2× when use_fp32_s). The
            // capacity constraint is `n * ctx_len <= s_cap²`, NOT `ctx_len <= s_cap`
            // (which was the previous guard — overly strict, wrongly aborted at any
            // chunked step where the cumulative ctx_len crossed s_cap even though
            // the actual matrix size still fit). FP32 takes 2× — same gate as
            // attention_cublas_prefill::use_fp32_s; if FP32 would overflow, the
            // attention call falls back to FP16-S automatically. So we only need to
            // guard the FP16 footprint here.
            int s_cap = attn_scores_buf_ ? static_cast<int>(attn_scores_.shape[1]) : 0;
            int64_t fp16_elems_needed = static_cast<int64_t>(n) * ctx_len;
            int64_t fp16_elems_avail = static_cast<int64_t>(s_cap) * s_cap;
            const bool smatrix_fits =
                s_cap > 0 && n <= s_cap && fp16_elems_needed <= fp16_elems_avail;
            // Spec-verify chunks (small n, ctx_len grows EVERY verify step) on
            // hd!=128 configs land on cuBLAS below the FMHA threshold — and
            // cuBLAS re-runs its per-new-shape algo selection on each call
            // (100 MiB workspace memset + candidate benchmark + blocking
            // event sync, per layer per verify: ~93 such trios/verify measured
            // on Qwen3.6-27B MTP-only, 12-15 ms/verify of pure churn; FMHA
            // reads ms/verify 78 → 60, +31% e2e, #847). The tiled FMHA keeps
            // no shape-keyed state — prefer it for small chunks inside its
            // correctness domain. hd==128 never reaches this (FA2 serves it);
            // learned sinks / heterogeneous shapes are excluded by
            // chunk_fmha_ok and keep cuBLAS.
            const bool small_growing_chunk = n <= 32 && hd != 128;
            const bool prefer_fmha =
                chunk_fmha_ok &&
                ((runtime_config().attention.fmha_prefill_threshold > 0 &&
                  ctx_len >= runtime_config().attention.fmha_prefill_threshold) ||
                 small_growing_chunk);
            if (!chunk_fa2_serves && !chunk_fmha_ok && !smatrix_fits) {
                // Sinks/heterogeneous shapes can ONLY be served by cuBLAS — the
                // engine clamps chunk sizes (max_safe_prefill_chunk) and rejects
                // unservable prompts upfront, so this is defense-in-depth.
                IMP_LOG_ERROR(
                    "chunked_prefill: attn_scores_ capacity %d×%d=%lld too small for "
                    "n=%d × ctx_len=%d = %lld at L%d — engine should have prevented this",
                    s_cap, s_cap, (long long)fp16_elems_avail, n, ctx_len, (long long)fp16_elems_needed,
                    layer);
                std::abort();
            }
            // Graph-captured verify replay mode (#847): kernels must not bake
            // the growing q_offset/ctx_len — grids and the persistent KV
            // scratch are sized for ctx_capacity and the real lengths are read
            // from device (d_past_len for the gather bound and append offset,
            // context_lens[0] for the attention KV length).
            const bool cap_replay = state.ctx_capacity > 0 && state.d_past_len != nullptr;
            if (cap_replay &&
                (!chunk_fa2_serves || ctx_len > state.ctx_capacity || state.n_sequences != 1 ||
                 chunk_capture_k_ == nullptr || chunk_capture_ctx_ < state.ctx_capacity)) {
                // The engine gates capture on chunk_capture_supported() and
                // pre-allocates the scratch — reaching here is a wiring bug.
                // Throwing fails the capture (or the eager warmup) cleanly and
                // the engine dooms capture for this model.
                throw std::runtime_error("chunked_prefill: capture-replay preconditions violated");
            }
            const int gather_cap = cap_replay ? state.ctx_capacity : q_offset;
            const int* d_past = cap_replay ? state.d_past_len : nullptr;
            size_t full_bytes = (size_t)ctx_len * nkv * hd * sizeof(half);

            half* k_full = nullptr;
            half* v_full = nullptr;
            bool used_eager_scratch = false;
            if (cap_replay) {
                k_full = chunk_capture_k_;
                v_full = chunk_capture_v_;
            } else {
                // Persistent gather scratch (grow-only, 64 MiB steps so a
                // growing ctx doesn't re-allocate every chunk). Falls back to
                // the per-call alloc when the grow fails.
                if (chunk_eager_bytes_ < full_bytes) {
                    constexpr size_t kGrowStep = 64u << 20;
                    const size_t cap = ((full_bytes + kGrowStep - 1) / kGrowStep) * kGrowStep;
                    if (chunk_eager_k_) { cudaFreeAsync(chunk_eager_k_, stream); chunk_eager_k_ = nullptr; }
                    if (chunk_eager_v_) { cudaFreeAsync(chunk_eager_v_, stream); chunk_eager_v_ = nullptr; }
                    chunk_eager_bytes_ = 0;
                    if (cudaMallocAsync(&chunk_eager_k_, cap, stream) == cudaSuccess &&
                        cudaMallocAsync(&chunk_eager_v_, cap, stream) == cudaSuccess) {
                        chunk_eager_bytes_ = cap;
                    } else {
                        if (chunk_eager_k_) { cudaFreeAsync(chunk_eager_k_, stream); chunk_eager_k_ = nullptr; }
                        if (chunk_eager_v_) { cudaFreeAsync(chunk_eager_v_, stream); chunk_eager_v_ = nullptr; }
                    }
                }
                if (chunk_eager_bytes_ >= full_bytes) {
                    k_full = chunk_eager_k_;
                    v_full = chunk_eager_v_;
                    used_eager_scratch = true;
                } else {
                    cudaMallocAsync(&k_full, full_bytes, stream);
                    cudaMallocAsync(&v_full, full_bytes, stream);
                }
            }

            // Gather past KV [0, q_offset) directly into k_full[0..q_offset], v_full[0..q_offset].
            if (kvt == QType::F16) {
                paged_kv_gather_fp16(k_full, static_cast<const half*>(cache->k_ptr(kv_layer, 0)),
                                     state.block_tables, gather_cap, kv_bs, nkv, hd, stream, d_past);
                paged_kv_gather_fp16(v_full, static_cast<const half*>(cache->v_ptr(kv_layer, 0)),
                                     state.block_tables, gather_cap, kv_bs, nkv, hd, stream, d_past);
            } else if (kvt == QType::FP8_E4M3) {
                float kv_scale = (!kv_scales_.empty() && kv_layer < (int)kv_scales_.size())
                                     ? kv_scales_[kv_layer]
                                     : 1.0f;
                paged_kv_gather_fp8_to_fp16(k_full,
                                            static_cast<const __nv_fp8_e4m3*>(cache->k_ptr(kv_layer, 0)),
                                            state.block_tables, kv_scale, gather_cap, kv_bs, nkv, hd,
                                            stream, d_past);
                paged_kv_gather_fp8_to_fp16(v_full,
                                            static_cast<const __nv_fp8_e4m3*>(cache->v_ptr(kv_layer, 0)),
                                            state.block_tables, kv_scale, gather_cap, kv_bs, nkv, hd,
                                            stream, d_past);
            } else if (kvt == QType::NVFP4) {
                paged_kv_gather_nvfp4_to_fp16(k_full, static_cast<const uint8_t*>(cache->k_ptr(kv_layer, 0)),
                                              static_cast<const uint8_t*>(cache->k_scale_ptr(kv_layer, 0)),
                                              state.block_tables, gather_cap, kv_bs, nkv, hd, stream,
                                              d_past);
                paged_kv_gather_nvfp4_to_fp16(v_full, static_cast<const uint8_t*>(cache->v_ptr(kv_layer, 0)),
                                              static_cast<const uint8_t*>(cache->v_scale_ptr(kv_layer, 0)),
                                              state.block_tables, gather_cap, kv_bs, nkv, hd, stream,
                                              d_past);
            } else if (kvt == QType::MXFP4_KV) {
                paged_kv_gather_mxfp4_kv_to_fp16(k_full,
                                                 static_cast<const uint8_t*>(cache->k_ptr(kv_layer, 0)),
                                                 static_cast<const uint8_t*>(cache->k_scale_ptr(kv_layer, 0)),
                                                 state.block_tables, gather_cap, kv_bs, nkv, hd, stream,
                                                 d_past);
                paged_kv_gather_mxfp4_kv_to_fp16(v_full,
                                                 static_cast<const uint8_t*>(cache->v_ptr(kv_layer, 0)),
                                                 static_cast<const uint8_t*>(cache->v_scale_ptr(kv_layer, 0)),
                                                 state.block_tables, gather_cap, kv_bs, nkv, hd, stream,
                                                 d_past);
            } else {  // INT4 — symmetric 4-bit with per-head FP16 scale
                paged_kv_gather_int4_to_fp16(k_full, static_cast<const uint8_t*>(cache->k_ptr(kv_layer, 0)),
                                             static_cast<const half*>(cache->k_scale_ptr(kv_layer, 0)),
                                             state.block_tables, gather_cap, kv_bs, nkv, hd, stream,
                                             d_past);
                paged_kv_gather_int4_to_fp16(v_full, static_cast<const uint8_t*>(cache->v_ptr(kv_layer, 0)),
                                             static_cast<const half*>(cache->v_scale_ptr(kv_layer, 0)),
                                             state.block_tables, gather_cap, kv_bs, nkv, hd, stream,
                                             d_past);
            }

            // Append current chunk's K/V at offset q_offset.
            if (cap_replay) {
                kv_chunk_append_fp16(k_full, static_cast<const half*>(kk.data), state.d_past_len, n,
                                     nkv * hd, stream);
                kv_chunk_append_fp16(v_full, static_cast<const half*>(vv.data), state.d_past_len, n,
                                     nkv * hd, stream);
            } else {
                cudaMemcpyAsync(k_full + (size_t)q_offset * nkv * hd, kk.data,
                                (size_t)n * nkv * hd * sizeof(half), cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(v_full + (size_t)q_offset * nkv * hd, vv.data,
                                (size_t)n * nkv * hd * sizeof(half), cudaMemcpyDeviceToDevice, stream);
            }

            int64_t kv_full_shape[2] = {(int64_t)(cap_replay ? state.ctx_capacity : ctx_len),
                                        (int64_t)(nkv * hd)};
            Tensor k_full_t(k_full, QType::F16, 2, kv_full_shape, /*on_device=*/true);
            Tensor v_full_t(v_full, QType::F16, 2, kv_full_shape, /*on_device=*/true);

            // Chunked prefill: choose FMHA or cuBLAS.
            //
            // REVERTED (#493 routed ALL hd=128 prefill into FA2 "regardless of
            // the threshold" for +7-25% pp512): both fp8-quantizing FMHA
            // kernels (FA2 and the fp8 FMHA) carry per-layer e4m3 score noise
            // that compounds across layers into prompt-blind/degenerate output
            // — every hd=128 model failed the degen suite at short prompts
            // (Qwen3-8B answered "What is 17 + 25?" with "2 5 2 5 2 5...").
            // The FP16 cuBLAS path is the accuracy reference; FMHA stays
            // gated behind the S-matrix-capacity threshold where it has
            // history in production. fp8-attention quality above the
            // threshold is tracked separately (see issue #511).
            // (chunk_fa2_serves / prefer_fmha / smatrix_fits computed above,
            // with the S-matrix capacity guard.)

            // Chunked attention routing (#548): prefer the FP16-QK FA2 kernel
            // at EVERY ctx_len, not only below the threshold. It is O(n)
            // memory (no S-matrix) like the fp8 family but carries no e4m3
            // score noise — the long-context chunked path through the
            // e4m3 FMHA family drifted teacher-forced NLL by up to ~25%
            // (ChunkedPrefillTest.LongContext_Chunk_Invariance is the gate).
            // Configs f16-QK declines (hd != 128, e.g. gemma-3 hd=256) fall
            // through to the tiled dispatch chain, which serves them with the
            // FP16 WMMA kernel — the fp8-QK family is opt-in only (#511:
            // raw e4m3 Q/K conversion read teacher-forced PPL 549 vs 16.6 on
            // gemma-3-12b when it served these chunks). cuBLAS below the
            // threshold.
            if (cap_replay) {
                // Replay mode is FA2-only: the S-matrix/FMHA fallbacks size and
                // bound work from the baked ctx_len. A decline here means the
                // engine-side eligibility check and the kernel disagree — fail
                // the capture rather than record a stale-length fallback.
                if (!try_fa2_fp16qk_prefill(runtime_config(), qv, k_full_t, v_full_t, ao, n,
                                            state.ctx_capacity, nh, nkv, hd, scale,
                                            layer_sliding_window, cfg.attn_logit_softcap, q_offset,
                                            stream, state.context_lens)) {
                    throw std::runtime_error("chunked_prefill: FA2 declined a capture-replay chunk");
                }
            } else if (chunk_fa2_serves &&
                try_fa2_fp16qk_prefill(runtime_config(), qv, k_full_t, v_full_t, ao, n, ctx_len, nh,
                                       nkv, hd, scale, layer_sliding_window, cfg.attn_logit_softcap,
                                       q_offset, stream)) {
                // chunked prefill: FP16-QK FA2 (no S-matrix, no e4m3 noise)
            } else if (smatrix_fits && !prefer_fmha) {
                // cuBLAS: reference path below the FMHA threshold; also the only
                // chunked path that honors learned sinks (attn_sinks → gpt-oss)
                // and heterogeneous per-layer shapes (Gemma-4).
                attention_cublas_prefill(qv, k_full_t, v_full_t, ao, attn_scores_, nh, nkv, hd, scale,
                                         /*causal=*/true, cfg.attn_logit_softcap, q_offset, stream,
                                         layer_sliding_window, attn_sinks);
            } else {
                // Tiled FMHA dispatch: no S-matrix needed, O(n) memory. Serves
                // uniform-shape chunks above the FMHA threshold, any chunk the
                // S-matrix cannot hold, and fa2-declined chunks on models whose
                // S-matrix was deliberately skipped (guarded unservable above:
                // sinks/heterogeneous shapes never reach this branch).
                int64_t q4s[4] = {1, (int64_t)n, (int64_t)nh, (int64_t)hd};
                int64_t kv4s[4] = {1, (int64_t)ctx_len, (int64_t)nkv, (int64_t)hd};
                int64_t o4s[4] = {1, (int64_t)n, (int64_t)nh, (int64_t)hd};
                Tensor q4 = qv.reshape(4, q4s);
                Tensor k4 = k_full_t.reshape(4, kv4s);
                Tensor v4 = v_full_t.reshape(4, kv4s);
                Tensor o4 = ao.reshape(4, o4s);
                attention_prefill_dispatch(q4, k4, v4, o4, scale, /*causal=*/true, layer_sliding_window,
                                           cfg.attn_logit_softcap, stream, runtime_config(), q_offset);
            }

            if (!cap_replay && !used_eager_scratch) {
                cudaFreeAsync(k_full, stream);
                cudaFreeAsync(v_full, stream);
            }

            // Persist current chunk's K/V (same as non-chunked path)
            write_kv_cache(layer, state, stream);
            goto after_attention;
        }

        // Prefill dispatch (post-Phase-2 + Phase-5 Track D):
        // attn_sinks: only the cuBLAS softmax understands learned sinks.
        const bool force_cublas_attn = per_layer_shapes || attn_sinks != nullptr;
        const bool s_matrix_fits = attn_scores_buf_ != nullptr &&
                                   n <= static_cast<int>(attn_scores_.shape[1]);
        // FMHA only above the S-matrix threshold — see the chunked path above
        // for why the #493 prefer-FA2-at-every-length override was reverted.
        const bool prefer_fmha = !force_cublas_attn &&
                                 (n >= runtime_config().attention.fmha_prefill_threshold);

        // NOTE (#566→#511): sliding-window prefill used to be routed AWAY
        // from the cuBLAS path here (`non_gemma4_sliding`) into the tiled
        // fallback chain — a relic from before attention_cublas_prefill grew
        // its sliding_window softmax mask. The historic "catastrophically
        // wrong hd=256+window attention" (gemma-3-12b PPL 42 vs llama.cpp
        // 1.0) was root-caused to the fp8-QK kernel that used to lead that
        // chain: raw e4m3 Q/K conversion compounds per-layer score error on
        // real activations (#511) — measured PPL 549 vs 16.6 when it served
        // gemma-3 chunks. fp8-QK is opt-in now; the FP16 WMMA kernel that
        // serves hd=256 instead is PPL-identical to cuBLAS (15.53 both,
        // n=3441 incl. window). cuBLAS stays preferred below the threshold
        // for throughput and as the materialized reference.
        // FP16-QK FA2 is the primary hd=128 prefill kernel at EVERY length — it is
        // O(n) memory (no S-matrix) AND at-or-above the materialized cuBLAS path
        // across the whole range (Qwen3-Coder-30B NVFP4: ~parity pp512 within the
        // prefill restart noise, +24% pp1024, +52% pp2048; measured 2026-06-12).
        // Try it BEFORE the S-matrix gate so the
        // fast path does not depend on the attn_scores buffer being allocated or
        // large enough for n: a too-small buffer used to make s_matrix_fits false
        // and drop n≈512 chunks into the slow tiled dispatch (−93% pp512). cuBLAS
        // stays the fallback for the configs f16-QK declines (hd != 128) and for
        // force_cublas_attn (learned sinks / per-layer shapes).
        if (!force_cublas_attn &&
            try_fa2_fp16qk_prefill(runtime_config(), qv, kk, vv, ao, n, n, nh, nkv, hd, scale,
                                   layer_sliding_window, cfg.attn_logit_softcap, /*q_offset=*/0,
                                   stream)) {
            // handled by FA2 f16 — no S-matrix needed
        } else if (s_matrix_fits && !prefer_fmha) {
            attention_cublas_prefill(qv, kk, vv, ao, attn_scores_, nh, nkv, hd, scale,
                                     /*causal=*/true, cfg.attn_logit_softcap,
                                     /*q_offset=*/0, stream, layer_sliding_window, attn_sinks);
        } else {
            if (attn_sinks) {
                static bool warned_sinks_fmha = false;
                if (!warned_sinks_fmha) {
                    warned_sinks_fmha = true;
                    IMP_LOG_WARN("attention sinks: S-matrix too small (n=%d) — FMHA fallback IGNORES "
                                 "sinks; output will be wrong. Lower the prefill chunk size.",
                                 n);
                }
            }
            // FMHA fallback: tiled O(n) memory chain.
            int64_t q4s[4] = {1, (int64_t)n, (int64_t)nh, (int64_t)hd};
            int64_t kv4s[4] = {1, (int64_t)n, (int64_t)nkv, (int64_t)hd};
            int64_t o4s[4] = {1, (int64_t)n, (int64_t)nh, (int64_t)hd};
            Tensor q4 = qv.reshape(4, q4s);
            Tensor k4 = kk.reshape(4, kv4s);
            Tensor v4 = vv.reshape(4, kv4s);
            Tensor o4 = ao.reshape(4, o4s);
            attention_prefill_dispatch(q4, k4, v4, o4, scale, /*causal=*/true, layer_sliding_window,
                                       cfg.attn_logit_softcap, stream, runtime_config(), /*q_offset=*/0);
        }

        // Persist K, V into cache for later decode steps
        write_kv_cache(layer, state, stream);
