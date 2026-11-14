#pragma once

// One of nine RuntimeConfig sections split from core/dispatch_policy.h:
// isolates a TU that touches only this section from the other eight's churn.
// Pure move, byte-identical; dispatch_policy.h still includes all nine.

#include <cstdint>
#include <string>
#include <vector>

namespace imp::cfg {

struct Speculative {
    // Prompt-lookup speculation (batch-1, dense), default on. Switches
    // the HISTORY MATCHER only; entering the verify step is decided by
    // spec_any_drafter (runtime/spec_gates.h), which asks all three sources.
    bool ngram = true;
    // Sampled requests (temperature > 0, top_k != 1) verify too: each chunk row draws with the eager
    // sampler's seed, a draft token is accepted while the draw equals it (exact, compute/sampling.h).
    // Off = greedy requests only.
    bool sampled = true;
    // Speculation on MoE models with NATIVE-NVFP4 experts (gate also requires
    // profile().moe_experts_nvfp4). GGUF-MoE verify re-dequants every activated
    // expert per step and never engages regardless of this flag. imp-cli
    // --bench pins this false so the perf-baseline decode signal stays raw.
    bool moe = true;
    // Token-Recycling adjacency drafting (ACL 2025, arXiv 2408.08696): engine-
    // scoped cross-request token->top-M successors table fed from emitted
    // bigrams + verify-chunk top-K logits. Runs as a fallback AFTER suffix/
    // n-gram and MTP; lossless via greedy argmax verify. Default off; on with recycle_width 8 under
    // --gpu-layers unless set (offload_spec_defaults.h, a verify costs one streamed pass there).
    bool token_recycling = false;
    // Multi-candidate verify: verify `recycle_width` adjacency candidates in
    // one chunk, each with its own t0 row and private KV-block copies (no token
    // mask needed, argmax accept stays lossless). Rows capped at bucket 17, 64 under --gpu-layers
    // (width*(1+depth)<=17). Default 1 (linear): wider buckets run CUTLASS at
    // low effective bandwidth, not worth the accept lift.
    int recycle_width = 1;
    // SuffixDecoding-style indexed drafting (arXiv 2411.04975): hash-indexed
    // suffix matching (O(1) amortized vs legacy O(n) backward scan) with
    // frequency-voted continuations and adaptive draft length past k up to
    // suffix_k_max. false = legacy single-most-recent scan.
    bool suffix = true;
    // Corpus drafting (#2421): text files under this path (file or directory, <= 64 MiB) form an
    // engine-wide suffix index, asked when the request's own history has no match (speculative.suffix).
    // "" = off.
    std::string ngram_corpus;
    // DSpark / DFlash block drafter checkpoint directory (deepseek-ai/dflash_qwen3_8b_block7: 7 drafts,
    // z-lab/Qwen3-8B-DFlash-b16: 15 drafts, embed / lm_head read from the target): drafts from the target's
    // hidden states; forces batch 1. "" = off.
    std::string dspark_model;
    // Longer suffix matches trade draft frequency for precision, and precision
    // wins: a higher min_match cuts false n-gram matches (e.g. in number
    // tables) that never verify, at the cost of fewer drafts overall.
    int min_match = 6;   // shortest accepted suffix n-gram match
    int max_match = 12;  // longest suffix extension searched
    // After this many consecutive draft misses, the request gives up on
    // speculation and re-enters the async conditional graph loop (eager
    // per-token path costs ~2x). 0 = never give up.
    int give_up_after = 64;
    // Speculation on hybrid (GDN/SSM) models: the verify chunk advances
    // recurrent state through rejected draft positions, so the committed state
    // slab is snapshotted before the chunk (restored on partial acceptance).
    // imp-cli --bench pins this false (baseline-semantics rule, like moe/suffix).
    bool hybrid = true;
    // MTP-head chain-draft length. 0=off, -1=auto (default), >0=fixed depth.
    // VRAM cost is per-checkpoint (read the load line, not this comment).
    // Independent of `ngram`: with ngram=false a fixed mtp_k drafts even where
    // the matcher finds nothing. AUTO engages for single-stream
    // (max_batch_size==1) runs on a checkpoint with a loadable head, and for
    // concurrent GDN-hybrid servers via mtp_concurrent; it leaves an explicit
    // `ngram` alone. Auto stays OFF for other concurrent servers and under
    // runtime.deterministic (MTP greedy trajectories are not eager-equal,
    // docs/LIMITATIONS.md).
    // Resolution lives in tools/common/mtp_auto.* (both tools decide before an engine exists).
    int mtp_k = -1;
    // MTP head KV slots when the batched verify is off: one request binds, the others keep the head
    // cache of other chats, so a chat UI's title / follow-up request between two turns no longer wipes
    // it and the next turn drafts over its prefix-cache hit (Qwen3.8-27B-NVFP4: 64 MiB per slot).
    // 1 = the pre-v0.63 single slot.
    int mtp_history_slots = 4;
    // MTP head KV rows per slot, a ring: past this many positions the head attends the last ones only and
    // keeps drafting. Clamped to the model context; with many slots (batched verify) a 1 GiB budget caps it,
    // floor 2048. Qwen3.8-27B: 4 KiB per row per slot; 40k prompt 222 / 276 drafts accepted (full: 222 /
    // 277).
    int mtp_kv_window = 16384;
    // With mtp_carry: while at most this many requests decode (capped at mtp_history_slots), one
    // forward verifies every bound request's MTP draft (2 rows per request, one spare recurrent slot
    // per row, 79.5 MiB each on Qwen3.8-27B); a bigger batch decodes plain. 0 = off.
    // Qwen3.8-27B-NVFP4, 4 overlapping chats: 83.72 -> 117.45 tok/s per turn, c32 unchanged.
    int mtp_batch_verify = 4;
    // Drafts per bound request in that step (N + 1 rows, N head passes): 4 at most, 1 when the factor
    // rows do not fit. Unset, it follows mtp_k down (a row past the drafted depth is a pad);
    // mtp_k=auto on a concurrent GDN-hybrid server follows it up (tools/common/mtp_auto.cpp).
    int mtp_batch_verify_k = 4;
    // Drafts per request once 4 requests share the step (2..mtp_batch_verify_k, factor rows only): each
    // draft is one more head pass over every request. Qwen3.8-27B-NVFP4 server, 4 chats x 20k, all
    // streams: 530.61 tok/s at 3 drafts, 517.18 at 4; 1 chat x 20k: 194.25 at 3, 211.32 at 4.
    int mtp_batch_verify_k_wide = 3;
    // Deepest step a run of full accepts reaches (mtp_batch_verify_k..8, factor rows only, below 4 requests,
    // inside the 8-row GEMM tile): all N drafts accepted -> N + 2 next. 0 = off. Qwen3.8-27B-NVFP4 agent
    // replay, 1 chat, same text: +1.99 % tok/ms at 6, +2.84 % at 7, +2.12 % at 8 (9 rows).
    int mtp_batch_verify_k_streak = 7;
    // External MTP head for a checkpoint without one: a .safetensors file with the Qwen mtp.* tensors of
    // the same base model (Qwen3.8-27B: the NVFP4 export's mtp.safetensors serves the ggml-org Q4_K_M).
    // Read only when MTP is requested and the model brings no head; empty = off.
    std::string mtp_head;
    // Serve the MTP chain's full-vocab logits GEMV from the NVFP4 LM-head
    // decode cache when one exists (#847 lever 3): the chain re-reads the LM
    // head once per drafted token, more traffic than the main forward at k=4.
    // Draft-only precision; verification stays lossless. Off = keep the FP16 chain GEMV. A 16-bit head
    // served as FP8 gets an NVFP4 draft copy of rows [0, mtp_draft_vocab) while MTP is on (Qwen3.8-27B-NVFP4:
    // 682 MiB at all rows, 270 MiB at 98304). The batched draft passes (mtp_batch_verify) read it too, 4 rows
    // per weight pass. A concurrent GDN-hybrid server with MTP on keeps it only over a row prefix: with
    // mtp_draft_vocab=0 set it defaults to false unless set (the full copy costs 1990 KV blocks).
    bool mtp_nvfp4_head = true;
    // Batched MTP drafts score only LM-head rows [0, mtp_draft_vocab) of the FP8 head (0 = all); the NVFP4
    // draft copy holds only those rows, for the lone chain too. 99.94 % of familiar dev50h output (1.54M
    // tokens) sits below 98304. Verify stays full vocab. Unset, a concurrent GDN-hybrid server takes 98304.
    int mtp_draft_vocab = 0;
    // MTP economics guard: after an 8-verify sample, average emitted
    // tokens/verify below this dooms MTP for the request. 0 = disabled (raw
    // measurement). Negative (default) = k-aware threshold 1 + f*k with f=0.40,
    // since break-even is chunk_cost(k+1 rows)/decode_cost and that ratio grows
    // with k (a fixed absolute floor is wrong for every k: a chain of k emits
    // at most k+1/verify). Positive value = absolute floor. Separate from the
    // hard floor in engine_spec_ngram.cpp (accepted*100 < drafted*15 dooms ALL
    // speculation, n-gram included); this one only unbinds MTP.
    float mtp_econ_min_emit = -1.0f;
    // Graph-captured verify chunk (#847): cache one CUDA graph per draft-length
    // bucket, replay each verify step (chunk metadata/KV lengths read from
    // device buffers, so the graph survives context growth). Drafts pad up to
    // the bucket length (padding rows causally invisible, KV rolled back with
    // rejected drafts). Engages only where FP16-QK FA2 serves the chunk
    // (uniform hd=128, no sinks/MLA/LongRoPE) on non-hybrid models. Falls back to eager on capture failure.
    bool capture = true;
};

// Speculation tuning constants, former keys measured once and never moved.
// Draft tokens per n-gram / recycling verify step (verify cost is ~flat in k).
inline constexpr int kSpecDraftK = 16;
// Suffix drafter: frequency-voted continuations extend a draft past kSpecDraftK up to this.
inline constexpr int kSpecSuffixKMax = 64;
// #964 stage 2: past this context a 1-token draft is dropped (depth >= 2 still verifies);
// MoE-NVFP4 and hybrids are exempt.
inline constexpr int kSpecShallowDraftCtx = 12288;
// A request that gave up on drafts runs async-loop bursts of this many tokens between re-probes.
inline constexpr int kSpecBurst = 128;
// A draft miss falls back to this many async-loop tokens (no recapture) instead of ~2x eager steps.
inline constexpr int kSpecMissBurst = 8;
// Token recycling: successors kept per token, linear depth 3 = chunk 4 = one batched-GEMV
// weight sweep (#1055), and a hop drafts only after 1 re-confirmation (a verify costs ~1.4x a decode).
inline constexpr int kRecycleSlots = 8;
inline constexpr int kRecycleDepth = 3;
inline constexpr int kRecycleMinStreak = 1;
// Block drafter (DSpark): below 1.8 emitted tokens per verify after 8 verifies a request stops
// drafting until the next burst re-arm. Qwen3-8B-NVFP4 prose: 1.8 348, 2.0 300, nospec 282 tok/s.
inline constexpr float kDsparkMinEmit = 1.8f;
// Captured verify: gather grids and K/V scratch sized for this context (2 * cap * nkv * hd * 2 B,
// clamped to max_seq_len); longer verify steps run eager.
inline constexpr int kSpecCaptureCtxCap = 32768;
}  // namespace imp::cfg
