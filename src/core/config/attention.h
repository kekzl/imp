#pragma once

// One of nine RuntimeConfig sections split from core/dispatch_policy.h:
// isolates a TU that touches only this section from the other eight's churn.
// Pure move, byte-identical; dispatch_policy.h still includes all nine.

#include <cstdint>
#include <string>
#include <vector>

namespace imp::cfg {

struct Attention {
    std::string fp8_prefill = "auto";
    int fmha_prefill_threshold = -1;  // -1 = auto (derived from S-matrix capacity)
    // Register-resident FA2 prefill kernel (f16 QK^T, hd 64/128/256). "on" (default) serves F16
    // in the tiled prefill chain; declines to the FP16 WMMA FMHA otherwise.
    std::string fmha_fa2 = "on";
    // hd=512 prefill (Gemma-4 global layers), single-shot and chunk continuation: "fa2" = split-D FA2
    // (attention_fa2_hd512.cu, the "auto" default when no window / sinks), "fmha" = WMMA
    // FMHA with a forward KV scan, a row identical in every chunk (#2167); "cublas" = S-matrix
    // cuBLAS (algorithm picked by n); "auto" = fa2, else fmha under runtime.deterministic, else cublas.
    std::string hd512_prefill = "auto";
    // RA2 prefill (compute/ra2/): INT8 QK^T, NVFP4 P x two-term NVFP4 V, quantized per call; hd 64/128/256,
    // causal, no window/softcap/sinks, declines under runtime.deterministic. Lossy, so opt-in.
    bool ra2_prefill = false;
    // APA prefill (compute/apa/): FP4 pass 1, hot KV tiles (share of the running row sum > apa_eps) to
    // exact FP16 pass 2; hd 128, causal, declines as ra2; 0 = off. Serves kv_len >= apa_min_kv (FA2, or RA2
    // when on, below) and chunks of >= 256 rows. With ra2_prefill, apa_min_kv 40960 is the measured split.
    float apa_eps = 0.01f;
    int apa_min_kv = 8192;
    // Per-layer FP4 tile cache across prefill chunks: a chunk quantizes only its new keys. T2 arena
    // n_layers x 72*hd B per 64 tokens per kv head at max_seq_len (Llama-3.2-3B 108k: 3.7 GiB). Opt-in.
    bool apa_tile_cache = false;
    // mxfp4_fp16_fallback: dequant native MXFP4 GGUF weights to an FP16 cache for decode.
    bool mxfp4_fp16_fallback = false;
    // MXFP4->FP16 cache pruning: "legacy" caches FP16 for every MXFP4 tensor.
    // "pruned" skips MoE expert_*_packed (bypassed by the batch-dequant path)
    // and LM head (routed through generic-dequant); needed for Qwen3.5-27B MXFP4 on 32 GiB.
    std::string mxfp4_fp16_cache_policy = "legacy";
    bool force_cublas_decode = false;
    // MLA absorbed-decode latent KV (DeepSeek-V2/V3). Off = materialized Stage A
    // (full per-head K/V reconstructed at projection, standard paged attention).
    // On: stores only the compressed latent + decoupled RoPE key, single-sequence only.
    bool mla_absorb = false;
    bool no_qknorm_fused = false;
    bool gate_concat = false;
    // Max VRAM (MiB) for the materialized cuBLAS-attention S-matrix; caps
    // prefill context on the fast path before falling back to FMHA (auto
    // fmha_prefill_threshold = cap+1). Auto-shrinks if the alloc fails. Default 384.
    int attn_scores_mib = 384;
    // Sparse decode attention (Quest-class top-k page selection). -1=auto (per-arch
    // budget, sparse_decode_default_tokens), 0=off, >0 = budget in tokens; bit-identical
    // to dense at or below budget.
    int sparse_topk_tokens = -1;
    // Below this context length decode stays dense even when the budget is
    // exceeded: the selection's win only outgrows its overhead past ~12k on
    // the measured dense model (8k measured -7%, 16k +6%, 32k +25%).
    int sparse_min_ctx = 12288;
    // Blocks covering the first sparse_sink_tokens positions are always kept.
    int sparse_sink_tokens = 16;
    // Blocks covering the last sparse_recent_tokens positions are always kept
    // (includes the partially filled tail block).
    int sparse_recent_tokens = 256;
    // Page score: true = mean of the page's keys with their stddev as offset
    // (arXiv 2605.27740); false = the Quest min/max corner bound. Same formula
    // sum_d(q*center+|q|*offset); corner ranks worse at small budgets (driven by one extreme token).
    bool sparse_score_meanstd = true;
    // Offset weight for sparse_score_meanstd. Larger keeps more of the
    // spread, 0 ranks on the mean alone. Ignored by the corner bound.
    float sparse_score_std_coef = 1.0f;
    // Sparse prefill: a continuation chunk attends to its own chunk densely plus the top
    // sparse_prefill_topk_tokens past tokens, one page selection per chunk from
    // sparse_prefill_rows sampled query rows (same page score). -1=auto (per-arch budget,
    // sparse_prefill_default_tokens), 0=off. Shares the key metadata
    // pool and sink tokens with sparse decode; the last sparse_prefill_recent_tokens past
    // positions are always kept.
    int sparse_prefill_topk_tokens = -1;
    int sparse_prefill_rows = 16;
    int sparse_prefill_recent_tokens = 1024;
    // Qwen4Exp QSA indexer (learned block top-k over the attention context, exact below 2051
    // tokens). qsa=false: dense attention on every row (not the model above 2050 tokens).
    // qsa_force: run the selected path on every prefill row (correctness A/B vs dense).
    // Default off: at 4503 tokens it loses on every axis measured (decode 50.6 vs 54.1 tok/s,
    // pp 650 vs 912 tok/s, PPL 4.7558 vs 4.6978) and the paged-vs-FA2 re-rounding flips MoE
    // routing on short prompts (degen_suite 48/50 on, 50/50 off). "Turn on above ~8k" withdrawn
    // (0.44.0: 11283 ctx 18.14 -> 15.88 tok/s). 0.45.0, 13863-token prompt: tg512 62.28-63.84 on
    // vs 55.99-57.10 tok/s dense (CHANGELOG 0.44.0 Added, 0.45.0 Changed).
    bool qsa = false;
    bool qsa_force = false;
    int qsa_rows = 16;  // query rows per selection pass (scratch K/V = rows x 2051 tokens x KV row)
    bool qsa_debug = false;  // log FA2 vs paged-on-cache vs selected max diffs (layer 3, prefill)
};

// Sparse attention and QSA tuning constants, former keys never moved from their default.
// Sparse decode stays dense below this context: its win outgrows the selection overhead past ~12k
// (dense model: 8k -7 %, 16k +6 %, 32k +25 %).
inline constexpr int kSparseMinCtx = 12288;
// Blocks covering the first kSparseSinkTokens / last kSparseRecentTokens positions are always kept.
inline constexpr int kSparseSinkTokens = 16;
inline constexpr int kSparseRecentTokens = 256;
// Mean+std page score: weight of the page's key stddev against its mean.
inline constexpr float kSparseScoreStdCoef = 1.0f;
// Sparse prefill: query rows sampled per chunk (kernel max 64); past positions always kept.
inline constexpr int kSparsePrefillRows = 16;
inline constexpr int kSparsePrefillRecentTokens = 1024;
// QSA prefill: query rows per selection pass (scratch K/V = rows x 2051 tokens x KV row).
inline constexpr int kQsaRows = 16;
}  // namespace imp::cfg
