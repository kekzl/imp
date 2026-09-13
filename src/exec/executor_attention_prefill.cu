// NOT a standalone TU: textually #include'd as the body of the
// `prefill_attend_seq` lambda inside GraphExecutor::run_attention.
// Omitted from CMake; must not be compiled alone. Per-sequence geometry
// arrives as lambda parameters; everything else is captured by reference. `return` falls through to
// `after_attention`.
        // Chunked prefill (prefill_offset>0): gathers past [0,prefill_offset) KV
        // to contiguous, appends the current chunk, runs rectangular
        // attention_cublas_prefill with q_offset. cudaMallocAsync per layer here
        // is an acknowledged exception to "no cudaMalloc in hot loops": chunked
        // prefill is excluded from CUDA-graph capture, so the alloc is amortised per chunk per layer, not per
        // token.
        if (q_offset > 0) {
            KVCache* cache = state.kv_cache;
            QType kvt = cache->qtype();
            // Defense-in-depth: engine resolves out-of-scope models to chunk_size=0,
            // so this code only runs for FP16 / FP8 / NVFP4 / MXFP4_KV KV.
            const bool kvt_ok = (kvt == QType::F16 || kvt == QType::FP8_E4M3 || kvt == QType::NVFP4 ||
                                 kvt == QType::MXFP4_KV || kvt == QType::INT4 || kvt == QType::INT8);
            // sliding_active and attn_shapes_vary are both supported: cuBLAS softmax
            // accepts a sliding_window arg, the rectangular cuBLAS path dispatches
            // per-layer with nh/nkv/hd. KV dtype remains the only hard gate (gather kernels wired for
            // FP16/FP8/NVFP4 only).
            if (!kvt_ok) {
                IMP_LOG_ERROR(
                    "chunked_prefill: unsupported KV dtype %d at L%d — engine should have prevented this",
                    (int)kvt, layer);
                std::abort();
            }

            int kv_layer = get_kv_layer(kv_layer_map_, layer);
            int kv_bs = cache->block_size();
            int ctx_len = q_offset + n;
            // Of the chunked branches, ONLY cuBLAS consumes attn_scores_. FP16-QK FA2
            // (hd=128) and tiled FMHA are O(n), no S-matrix. Uniform per-layer shapes
            // (GDN/Mamba2 hybrids) are served by the O(n) family too; only truly
            // heterogeneous shapes (Gemma-4 dual hd 256/512) and learned sinks (gpt-oss) require cuBLAS.
            const bool shapes_uniform = !per_layer_shapes || attn_shapes_uniform();
            // hd=256 rides the stage-1 FA2 port (attention.fa2_hd256, default
            // on since #932): same fp16-qk kernel family, Bq=64/TWOSLOT instance.
            const bool fa2_hd_ok = fa2_serves_head_dim(hd, dispatch_policy().attention.fa2_hd256);
            // FA2 is chosen per-layer: heterogeneous Gemma-4 SWA layers (hd=256)
            // qualify even though the model isn't uniform, since the gather and
            // attention are per-layer. Previously gated on shapes_uniform, forcing all
            // Gemma-4 layers onto cuBLAS.
            const bool chunk_fa2_serves = !attn_sinks && fa2_hd_ok &&
                                          dispatch_policy().attention.fa2_fp16qk != "never";
            // The tiled WMMA FMHA serves any fused head_dim (incl. hd=512), per-layer,
            // so Gemma-4 global layers qualify too. Only learned sinks below the threshold still require
            // cuBLAS (folded into WMMA FMHA above it, #992).
            const bool fmha_hd_ok = fmha_serves_head_dim(hd);
            const bool chunk_fmha_ok = fmha_hd_ok;
            // attn_scores_ is [nh, s_cap, s_cap]; chunked cuBLAS stores [nh, n,
            // ctx_len] FP16 (or FP32 = 2x when use_fp32_s). Capacity constraint is
            // n*ctx_len <= s_cap^2, NOT ctx_len <= s_cap (the old guard wrongly
            // aborted once cumulative ctx_len crossed s_cap even when the matrix still
            // fit). FP32 overflow auto-falls back to FP16-S, so only the FP16 footprint needs guarding here.
            int s_cap = attn_scores_buf_ ? static_cast<int>(attn_scores_.shape[1]) : 0;
            int64_t fp16_elems_needed = static_cast<int64_t>(n) * ctx_len;
            int64_t fp16_elems_avail = static_cast<int64_t>(s_cap) * s_cap;
            const bool smatrix_fits =
                s_cap > 0 && n <= s_cap && fp16_elems_needed <= fp16_elems_avail;
            // Spec-verify chunks (small n, growing ctx_len) on hd!=128 land on cuBLAS
            // below the FMHA threshold, and cuBLAS re-runs per-new-shape algo
            // selection on every call (workspace memset + benchmark + blocking sync),
            // pure churn per verify. The tiled FMHA keeps no shape-keyed state, so it
            // is preferred for small chunks inside its correctness domain (#847).
            // hd==128 never reaches this (FA2 serves it); sinks/heterogeneous shapes stay on cuBLAS.
            const bool small_growing_chunk = n <= 32 && hd != 128;
            // hd=512 never prefers the fused path: the SMEM-capped WMMA FMHA is
            // several times slower than cuBLAS there. Stays on cuBLAS while the
            // S-matrix fits; the fused hd=512 kernel serves only the S-matrix-overflow (O(n)) fallback.
            const bool prefer_fmha =
                chunk_fmha_ok && hd != 512 &&
                ((dispatch_policy().attention.fmha_prefill_threshold > 0 &&
                  ctx_len >= dispatch_policy().attention.fmha_prefill_threshold) ||
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
            // Graph-captured verify replay mode (#847): kernels must not bake the
            // growing q_offset/ctx_len; grids and the persistent KV scratch are sized
            // for ctx_capacity, real lengths read from device (d_past_len for the
            // gather/append offset, context_lens[0] for the attention KV length).
            const bool cap_replay = state.ctx_capacity > 0 && state.d_past_len != nullptr;
            if (cap_replay &&
                (!chunk_fa2_serves || ctx_len > state.ctx_capacity || state.n_sequences != 1 ||
                 chunk_capture_k_ == nullptr || chunk_capture_ctx_ < state.ctx_capacity)) {
                // Reaching here means the engine's chunk_capture_supported() gate and its
                // scratch pre-allocation disagree with this dispatch: a wiring bug.
                // Throwing fails the capture (or eager warmup) cleanly instead of dooming it silently.
                throw std::runtime_error("chunked_prefill: capture-replay preconditions violated");
            }
            // #846 KV-append-quant (opt-in): serves the whole chunk straight from the
            // NVFP4 paged cache by appending the current chunk FIRST, then reading
            // [0,ctx_len) as FP4 (no gather, no FP16 materialization, no in-kernel
            // quantize). Falls through to the gather path if declined; tail write skipped then (KV already
            // persisted).
            bool paged_kv_written = false;
            if (!cap_replay && kvt == QType::NVFP4 && hd == 128 && shapes_uniform && !attn_sinks &&
                state.n_sequences == 1 && dispatch_policy().attention.mxfp4_paged_kv) {
                write_kv_cache(layer, state, stream, row_begin, n, bt_flat, bt_swa_flat, seq_positions);
                paged_kv_written = true;
                int64_t q4s[4] = {1, (int64_t)n, (int64_t)nh, (int64_t)hd};
                int64_t o4s[4] = {1, (int64_t)n, (int64_t)nh, (int64_t)hd};
                Tensor q4 = qv.reshape(4, q4s);
                Tensor o4 = ao.reshape(4, o4s);
                if (fmha_sm120_mxfp4_prefill_paged(
                        q4, o4, static_cast<const half*>(kk.data), static_cast<const half*>(vv.data),
                        static_cast<const uint8_t*>(cache->k_ptr(kv_layer, 0)),
                        static_cast<const uint8_t*>(cache->k_scale_ptr(kv_layer, 0)),
                        static_cast<const uint8_t*>(cache->v_ptr(kv_layer, 0)),
                        static_cast<const uint8_t*>(cache->v_scale_ptr(kv_layer, 0)), layer_block_tables,
                        kv_bs, ctx_len, nkv, scale, /*causal=*/true, layer_sliding_window,
                        cfg.attn_logit_softcap, stream, q_offset,
                        dispatch_policy().attention.mxfp4_promote_budget)) {
                    return;
                }
                // Declined: the gather below covers only [0, q_offset) and the
                // fresh-FP16 append fills the current chunk — correct even
                // though the cache now already holds the current chunk too.
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
                                     layer_block_tables, gather_cap, kv_bs, nkv, hd, stream, d_past);
                paged_kv_gather_fp16(v_full, static_cast<const half*>(cache->v_ptr(kv_layer, 0)),
                                     layer_block_tables, gather_cap, kv_bs, nkv, hd, stream, d_past);
            } else if (kvt == QType::FP8_E4M3) {
                float kv_scale = (!kv_scales_.empty() && kv_layer < (int)kv_scales_.size())
                                     ? kv_scales_[kv_layer]
                                     : 1.0f;
                paged_kv_gather_fp8_to_fp16(k_full,
                                            static_cast<const __nv_fp8_e4m3*>(cache->k_ptr(kv_layer, 0)),
                                            layer_block_tables, kv_scale, gather_cap, kv_bs, nkv, hd,
                                            stream, d_past);
                paged_kv_gather_fp8_to_fp16(v_full,
                                            static_cast<const __nv_fp8_e4m3*>(cache->v_ptr(kv_layer, 0)),
                                            layer_block_tables, kv_scale, gather_cap, kv_bs, nkv, hd,
                                            stream, d_past);
            } else if (kvt == QType::NVFP4) {
                paged_kv_gather_nvfp4_to_fp16(k_full, static_cast<const uint8_t*>(cache->k_ptr(kv_layer, 0)),
                                              static_cast<const uint8_t*>(cache->k_scale_ptr(kv_layer, 0)),
                                              layer_block_tables, gather_cap, kv_bs, nkv, hd, stream,
                                              d_past);
                paged_kv_gather_nvfp4_to_fp16(v_full, static_cast<const uint8_t*>(cache->v_ptr(kv_layer, 0)),
                                              static_cast<const uint8_t*>(cache->v_scale_ptr(kv_layer, 0)),
                                              layer_block_tables, gather_cap, kv_bs, nkv, hd, stream,
                                              d_past);
            } else if (kvt == QType::MXFP4_KV) {
                paged_kv_gather_mxfp4_kv_to_fp16(k_full,
                                                 static_cast<const uint8_t*>(cache->k_ptr(kv_layer, 0)),
                                                 static_cast<const uint8_t*>(cache->k_scale_ptr(kv_layer, 0)),
                                                 layer_block_tables, gather_cap, kv_bs, nkv, hd, stream,
                                                 d_past);
                paged_kv_gather_mxfp4_kv_to_fp16(v_full,
                                                 static_cast<const uint8_t*>(cache->v_ptr(kv_layer, 0)),
                                                 static_cast<const uint8_t*>(cache->v_scale_ptr(kv_layer, 0)),
                                                 layer_block_tables, gather_cap, kv_bs, nkv, hd, stream,
                                                 d_past);
            } else if (kvt == QType::INT8) {  // symmetric 8-bit, per-head FP16 scale
                paged_kv_gather_int8_to_fp16(k_full, static_cast<const int8_t*>(cache->k_ptr(kv_layer, 0)),
                                             static_cast<const half*>(cache->k_scale_ptr(kv_layer, 0)),
                                             layer_block_tables, gather_cap, kv_bs, nkv, hd, stream, d_past);
                paged_kv_gather_int8_to_fp16(v_full, static_cast<const int8_t*>(cache->v_ptr(kv_layer, 0)),
                                             static_cast<const half*>(cache->v_scale_ptr(kv_layer, 0)),
                                             layer_block_tables, gather_cap, kv_bs, nkv, hd, stream, d_past);
            } else {  // INT4 — symmetric 4-bit with per-head FP16 scale
                paged_kv_gather_int4_to_fp16(k_full, static_cast<const uint8_t*>(cache->k_ptr(kv_layer, 0)),
                                             static_cast<const half*>(cache->k_scale_ptr(kv_layer, 0)),
                                             layer_block_tables, gather_cap, kv_bs, nkv, hd, stream,
                                             d_past);
                paged_kv_gather_int4_to_fp16(v_full, static_cast<const uint8_t*>(cache->v_ptr(kv_layer, 0)),
                                             static_cast<const half*>(cache->v_scale_ptr(kv_layer, 0)),
                                             layer_block_tables, gather_cap, kv_bs, nkv, hd, stream,
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

            // #493's "route ALL hd=128 prefill into FA2 regardless of threshold" was
            // REVERTED: both fp8-quantizing FMHA kernels carry per-layer e4m3 score
            // noise that compounds into prompt-blind/degenerate output at short
            // prompts. FP16 cuBLAS is the accuracy reference; FMHA stays gated behind
            // the S-matrix-capacity threshold. fp8-attention quality above the threshold: #511.

            // #548: prefer the FP16-QK FA2 kernel at EVERY ctx_len for chunked
            // attention, not only below the threshold - O(n) memory like the fp8
            // family but no e4m3 score noise (the e4m3 FMHA family drifted
            // teacher-forced NLL badly on long-context chunks, gated by
            // ChunkedPrefillTest.LongContext_Chunk_Invariance). Declines (hd!=128)
            // fall through to the tiled FP16 WMMA dispatch; fp8-QK stays opt-in only (#511). cuBLAS below the
            // threshold.
            if (cap_replay) {
                // Replay mode is FA2-only: the S-matrix/FMHA fallbacks size and bound
                // work from the baked ctx_len. A decline here means the engine-side
                // eligibility check and the kernel disagree; fail the capture rather than record a
                // stale-length fallback.
                if (!try_fa2_fp16qk_prefill(dispatch_policy(), qv, k_full_t, v_full_t, ao, n,
                                            state.ctx_capacity, nh, nkv, hd, scale,
                                            layer_sliding_window, cfg.attn_logit_softcap, q_offset,
                                            stream, state.context_lens)) {
                    throw std::runtime_error("chunked_prefill: FA2 declined a capture-replay chunk");
                }
                dispatch_record::set_attn_prefill_outer(AttnPrefillOuter::FA2_FP16QK);
            } else if (chunk_fa2_serves &&
                try_fa2_fp16qk_prefill(dispatch_policy(), qv, k_full_t, v_full_t, ao, n, ctx_len, nh,
                                       nkv, hd, scale, layer_sliding_window, cfg.attn_logit_softcap,
                                       q_offset, stream)) {
                // chunked prefill: FP16-QK FA2 (no S-matrix, no e4m3 noise)
                dispatch_record::set_attn_prefill_outer(AttnPrefillOuter::FA2_FP16QK);
            } else if (smatrix_fits && !prefer_fmha) {
                // cuBLAS: below-threshold reference for uniform models, learned sinks, and
                // hd=512 Gemma-4 global layers whenever their S-matrix fits (faster than
                // the SMEM-capped fused hd=512 kernel). hd=256 SWA layers took FA2 above; only hd=512 layers
                // land here.
                dispatch_record::set_attn_prefill_outer(AttnPrefillOuter::CUBLAS);
                attention_cublas_prefill(qv, k_full_t, v_full_t, ao, attn_scores_, nh, nkv, hd, scale,
                                         /*causal=*/true, cfg.attn_logit_softcap, q_offset, stream,
                                         layer_sliding_window, attn_sinks);
            } else if (hd == 512 && !prefer_fmha && s_cap > 0 &&
                       attention_cublas_prefill_sliced(qv, k_full_t, v_full_t, ao, attn_scores_, nh, nkv,
                                                       hd, scale, /*causal=*/true, cfg.attn_logit_softcap,
                                                       q_offset, stream, layer_sliding_window,
                                                       attn_sinks)) {
                dispatch_record::set_attn_prefill_outer(AttnPrefillOuter::CUBLAS_SLICED);
                // hd=512 S-matrix overflow (long ctx): cuBLAS in workspace-sized q-row
                // slices, faster than the whole-chunk FMHA hd=512 fallback (the FMHA's
                // Bq=16 tile re-reads K/V ~Sq/16 times, KV-bandwidth-amplified). Returns
                // false only when even a 16-row slice overflows the workspace; FMHA then serves.
            } else {
                // Tiled FMHA dispatch: no S-matrix, O(n) memory. Serves uniform-shape
                // chunks above the FMHA threshold, any chunk the S-matrix cannot hold, and
                // hd=512 only when the workspace is too degraded even for 16-row cuBLAS
                // slices. Learned sinks ride along (#992).
                int64_t q4s[4] = {1, (int64_t)n, (int64_t)nh, (int64_t)hd};
                int64_t kv4s[4] = {1, (int64_t)ctx_len, (int64_t)nkv, (int64_t)hd};
                int64_t o4s[4] = {1, (int64_t)n, (int64_t)nh, (int64_t)hd};
                Tensor q4 = qv.reshape(4, q4s);
                Tensor k4 = k_full_t.reshape(4, kv4s);
                Tensor v4 = v_full_t.reshape(4, kv4s);
                Tensor o4 = ao.reshape(4, o4s);
                dispatch_record::set_attn_prefill_outer(AttnPrefillOuter::FMHA_CHAIN);
                attention_prefill_dispatch(q4, k4, v4, o4, scale, /*causal=*/true, layer_sliding_window,
                                           cfg.attn_logit_softcap, stream, dispatch_policy(), q_offset,
                                           static_cast<const half*>(attn_sinks));
            }

            if (!cap_replay && !used_eager_scratch) {
                cudaFreeAsync(k_full, stream);
                cudaFreeAsync(v_full, stream);
            }

            // Persist current chunk's K/V (same as non-chunked path).
            // Skipped when the paged-FP4 branch already appended it above.
            if (!paged_kv_written)
                write_kv_cache(layer, state, stream, row_begin, n, bt_flat, bt_swa_flat, seq_positions);
            return;
        }

        // Prefill dispatch: attn_sinks understood only by the cuBLAS softmax.
        // Uniform per-layer shapes (GDN/Mamba2 hybrids) are FA2-servable, same as
        // the chunked path (#924); only truly heterogeneous shapes (Gemma-4 dual
        // head_dim) and learned sinks (gpt-oss) require cuBLAS. Without this,
        // hybrids never took single-shot FA2, breaking the "FA2 serves all
        // attention" FP8-KV deterministic-skip promise for short prompts.
        // Gemma-4: hd=256 SWA layers (5/6 of layers) take FA2 f16-QK per-layer;
        // hd=512 global layers stay on cuBLAS (faster than the SMEM-capped fused
        // hd=512 kernel, Bq=16 forced by its 99 KB opt-in). The fused hd=512
        // kernel exists only as the O(n)-memory overflow fallback for long context.
        const bool s_matrix_fits = attn_scores_buf_ != nullptr &&
                                   n <= static_cast<int>(attn_scores_.shape[1]);
        // FMHA only above the S-matrix threshold (see the chunked path for why the
        // #493 prefer-FA2-everywhere override was reverted). hd=512 never prefers
        // the fused path (slower than cuBLAS); stays on cuBLAS until the S-matrix no longer fits.
        const bool prefer_fmha = (n >= dispatch_policy().attention.fmha_prefill_threshold) && hd != 512;

        // FP16-QK FA2 is the primary hd=128 prefill kernel at EVERY length: O(n)
        // memory (no S-matrix), at or above cuBLAS throughput across the whole
        // range. Tried BEFORE the S-matrix gate so the fast path doesn't depend on
        // attn_scores_ being allocated large enough for n (a too-small buffer used
        // to drop n~512 chunks into the slow tiled dispatch). cuBLAS is the
        // below-threshold fallback for configs FA2 declines (hd not in {128,256},
        // or hd=256 with fa2_hd256 off) and for learned sinks. Sliding-window
        // prefill (#566) uses the same cuBLAS path (it grew a sliding_window
        // softmax mask); the historic hd=256+window failure was root-caused to
        // the fp8-QK kernel's raw e4m3 conversion (#511), now opt-in only - the
        // FP16 WMMA kernel serving hd=256 is PPL-identical to cuBLAS.
        if (attn_sinks == nullptr &&
            try_fa2_fp16qk_prefill(dispatch_policy(), qv, kk, vv, ao, n, n, nh, nkv, hd, scale,
                                   layer_sliding_window, cfg.attn_logit_softcap, /*q_offset=*/0,
                                   stream)) {
            // handled by FA2 f16 — no S-matrix needed (hd 128/256, incl. Gemma-4 SWA)
            dispatch_record::set_attn_prefill_outer(AttnPrefillOuter::FA2_FP16QK);
        } else if (s_matrix_fits && !prefer_fmha) {
            // Materialized cuBLAS: below-threshold reference for uniform models and
            // learned sinks, and the hd=512 Gemma-4 global layers whenever the
            // S-matrix fits (faster than the SMEM-capped fused hd=512 kernel). hd=256 SWA layers took FA2
            // above.
            dispatch_record::set_attn_prefill_outer(AttnPrefillOuter::CUBLAS);
            attention_cublas_prefill(qv, kk, vv, ao, attn_scores_, nh, nkv, hd, scale,
                                     /*causal=*/true, cfg.attn_logit_softcap,
                                     /*q_offset=*/0, stream, layer_sliding_window, attn_sinks);
        } else if (hd == 512 && !prefer_fmha && attn_scores_buf_ != nullptr &&
                   attention_cublas_prefill_sliced(qv, kk, vv, ao, attn_scores_, nh, nkv, hd, scale,
                                                   /*causal=*/true, cfg.attn_logit_softcap,
                                                   /*q_offset=*/0, stream, layer_sliding_window,
                                                   attn_sinks)) {
            dispatch_record::set_attn_prefill_outer(AttnPrefillOuter::CUBLAS_SLICED);
            // hd=512 S-matrix overflow: cuBLAS in workspace-sized q-row slices, faster
            // than the whole-prompt FMHA hd=512 fallback. False only when even a
            // 16-row slice overflows; FMHA below then serves.
        } else {
            // FMHA fallback: tiled O(n) memory chain, sink-capable since #992 (routes
            // sinks to WMMA FMHA, throws instead of falling through to a sink-blind
            // kernel). Reached by uniform models above the FMHA threshold, and by
            // hd=512 layers only when the workspace is too degraded for 16-row cuBLAS slices.
            int64_t q4s[4] = {1, (int64_t)n, (int64_t)nh, (int64_t)hd};
            int64_t kv4s[4] = {1, (int64_t)n, (int64_t)nkv, (int64_t)hd};
            int64_t o4s[4] = {1, (int64_t)n, (int64_t)nh, (int64_t)hd};
            Tensor q4 = qv.reshape(4, q4s);
            Tensor k4 = kk.reshape(4, kv4s);
            Tensor v4 = vv.reshape(4, kv4s);
            Tensor o4 = ao.reshape(4, o4s);
            dispatch_record::set_attn_prefill_outer(AttnPrefillOuter::FMHA_CHAIN);
            attention_prefill_dispatch(q4, k4, v4, o4, scale, /*causal=*/true, layer_sliding_window,
                                       cfg.attn_logit_softcap, stream, dispatch_policy(), /*q_offset=*/0,
                                       static_cast<const half*>(attn_sinks));
        }

        // Persist K, V into cache for later decode steps
        write_kv_cache(layer, state, stream, row_begin, n, bt_flat, bt_swa_flat, seq_positions);
