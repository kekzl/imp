#pragma once

// One of nine RuntimeConfig sections split from core/dispatch_policy.h:
// isolates a TU that touches only this section from the other eight's churn.
// Pure move, byte-identical; dispatch_policy.h still includes all nine.

#include <cstdint>
#include <string>
#include <vector>

namespace imp::cfg {

struct Speculative {
    // Prompt-lookup speculation (batch-1, greedy, dense), default on. Switches
    // the HISTORY MATCHER only; entering the verify step is decided by
    // spec_any_drafter (runtime/spec_gates.h), which asks all three sources.
    bool ngram = true;
    // Context cap for dense chunk-verify drafting (#964): past this context,
    // drafting is gated per step. MoE-NVFP4 and GDN-hybrid requests are exempt
    // (deep drafts pay for themselves). Superseded in practice by the
    // depth-aware gate below (shallow_draft_ctx); default off (0 = no cap).
    int draft_ctx_cap = 0;
    // #964 stage 2: past this request context, 1-token n-gram/suffix drafts are
    // discarded instead of verified (miss-burst path serves at plain decode
    // speed); depth>=2 drafts always verify. MoE-NVFP4 and hybrid exempt. 0 = never gate.
    int shallow_draft_ctx = 12288;
    // #964: run the dense verify chunk's attention through the batched-decode
    // split-K paged kernels (rows as same-KV "sequences" with per-row context
    // len; causal by construction) instead of the small-M prefill FA2 tile +
    // full-context FP16 KV gather. Requires the 3/5 capture buckets. Dense
    // non-MLA/non-SWA/non-MoE/non-hybrid only; MoE/hybrid keep the FA2 chunk path.
    bool verify_decode_attn = true;
    // #998: verify-chunk GEMMs (M<=largest capture bucket) read the NVFP4
    // decode overlay in one weight pass per MR tile instead of the M>1 prefill
    // dequant path. Also aligns verify argmax with the decode path. Kill switch for A/B.
    bool verify_nvfp4_gemm = true;
    // Route the verify chunk's native-NVFP4 GEMMs through the small-M mxf4nvf4
    // pipeline (gemm.nvfp4_smallm) instead of the batched multi-row GEMV, which
    // reads weights slower at small M. Argmax parity is not a concern: a
    // speculative arm never reproduces non-speculative greedy output anyway. Default off.
    bool verify_smallm = false;
    // Make the verify chunk's NVFP4 GEMV reduce K exactly like the M=1 decode
    // GEMV (32 partial sums, one per warp lane, vs the batched kernel's 128).
    // The rounding difference reached the STOP decision and truncated answers
    // at speculative.mtp_k=1 (docs/LIMITATIONS.md). Off by default, pending its own measurement.
    bool verify_row_parity = false;
    // Speculation on MoE models with NATIVE-NVFP4 experts (gate also requires
    // profile().moe_experts_nvfp4). GGUF-MoE verify re-dequants every activated
    // expert per step and never engages regardless of this flag. imp-cli
    // --bench pins this false so the perf-baseline decode signal stays raw.
    bool moe = true;
    // #1003 stage 1: at decode batch>1, one request per step may run its spec
    // verify (round-robin) while the rest decode batched. Three guards: a
    // draft-depth floor (min_draft = 2x batch), a pipeline yield, and an
    // adaptive yield cadence (8->64 steps exponential backoff on empty turns).
    // Batch-1 behavior unchanged. Kill switch for A/B.
    bool batch_rr = true;
    // Batched verify on the GDN hybrid: every decoding request forwards 2 rows
    // (last token + draft) per step; needs a spare recurrent slot per batch
    // slot (halves the auto batch). Default off (docs/plans/2026-09-11-batched-mtp-verify.md).
    bool batch_verify = false;
    // Carry the drafted row of a batched verify as (g, k, delta) + one conv tap
    // instead of a second full recurrent slot. The spare slot duplicates the
    // full state pool (79.5 MiB/slot on Qwen3.8-27B, clamping max_batch 32->18);
    // the factored form is ~3.3 MiB. Opt-in (docs/plans/2026-09-12-factored-verify-spare.md).
    bool factored_spare = false;
    int k = 16;  // draft tokens per verify step (verify cost is ~flat in k)
    // Token-Recycling adjacency drafting (ACL 2025, arXiv 2408.08696): engine-
    // scoped cross-request token->top-M successors table fed from emitted
    // bigrams + verify-chunk top-K logits. Runs as a fallback AFTER suffix/
    // n-gram and MTP; lossless via greedy argmax verify. Default off.
    bool token_recycling = false;
    int recycle_slots = 8;  // successors kept per token (MRU/rank order)
    // Linear draft length. Default 3 -> chunk 4 -> capture bucket 4 =
    // exactly one batched-GEMV weight sweep (M=4, #1055); deeper chains
    // pad into bucket 5+ and pay a second sweep.
    int recycle_depth = 3;
    // Precision gate (#1055): only draft hops whose front slot was
    // re-confirmed at least this many times (bigram repeat / model top-1
    // stable). A verify costs ~1.4x a decode step; precision beats recall. 0 = draft on any known successor.
    int recycle_min_streak = 1;
    // Multi-candidate verify: verify `recycle_width` adjacency candidates in
    // one chunk, each with its own t0 row and private KV-block copies (no token
    // mask needed, argmax accept stays lossless). Rows capped at bucket 17
    // (width*(1+depth)<=17). Default 1 (linear): wider buckets run CUTLASS at
    // low effective bandwidth, not worth the accept lift.
    int recycle_width = 1;
    // MTP multi-candidate width: W>1 drafts W chains per verify step (chain 0
    // continues the head's top-1, chains 1..W-1 branch at the first position on
    // top-W ids). Chains verify as one multi-candidate chunk (private KV blocks,
    // longest matching prefix wins). Device-side, no per-step host sync; width
    // capped by kMtpMaxChainK and kMtpMaxTopW. Default 1 = linear path.
    int mtp_tree_width = 1;
    // Branch only when the head is unsure: extra chains (mtp_tree_width>1) are
    // always drafted but verified only when the head's top-1/top-2 logit margin
    // is below this. 0 = always branch. See docs/plans/2026-08-31-mtp-multicandidate-hybrid.md.
    float mtp_tree_margin = 2.0f;
    // SuffixDecoding-style indexed drafting (arXiv 2411.04975): hash-indexed
    // suffix matching (O(1) amortized vs legacy O(n) backward scan) with
    // frequency-voted continuations and adaptive draft length past k up to
    // suffix_k_max. false = legacy single-most-recent scan.
    bool suffix = true;
    int suffix_k_max = 64;
    // Longer suffix matches trade draft frequency for precision, and precision
    // wins: a higher min_match cuts false n-gram matches (e.g. in number
    // tables) that never verify, at the cost of fewer drafts overall.
    int min_match = 6;   // shortest accepted suffix n-gram match
    int max_match = 12;  // longest suffix extension searched
    // After this many consecutive draft misses, the request gives up on
    // speculation and re-enters the async conditional graph loop (eager
    // per-token path costs ~2x). 0 = never give up.
    int give_up_after = 64;
    // Prompt-lookup drafting is cold by construction (matches against earlier
    // generated text). The economics guard (mtp_econ_min_emit) can't catch the
    // cold phase: it arms at spec_verifies>=8 per request, and a short request
    // never reaches that. This gates on tokens generated SO FAR instead. 0 = disabled.
    int min_history = 0;
    // Burst-hybrid: while given up, the async loop runs in bursts of this many
    // tokens; after each burst the request re-probes drafts for a couple of
    // steps. 0 = give-up is final.
    int burst = 128;
    // On a draft miss, fall back to the async loop for this many tokens (cheap
    // rearm, no graph recapture) instead of paying the ~2x eager tax until the
    // next draft. 0 = stay eager between drafts (legacy).
    int miss_burst = 8;
    // Reuse the parked captured graph across bursts (rearm instead of
    // recapture). The #683 wrong-token bug was the fresh-capture loop
    // initializing position/context one too high, not the rearm itself; both paths now share first-forward
    // semantics.
    bool burst_rearm = true;
    // Speculation on hybrid (GDN/SSM) models: the verify chunk advances
    // recurrent state through rejected draft positions, so the committed state
    // slab is snapshotted before the chunk (restored on partial acceptance).
    // imp-cli --bench pins this false (baseline-semantics rule, like moe/suffix).
    bool hybrid = true;
    // MTP-head chain-draft length. 0=off, -1=auto (default), >0=fixed depth.
    // VRAM cost is per-checkpoint (read the load line, not this comment).
    // Independent of `ngram`: with ngram=false a fixed mtp_k drafts even where
    // the matcher finds nothing. AUTO engages only for single-stream
    // (max_batch_size==1) runs on a checkpoint with a loadable head, and leaves
    // an explicit `ngram` alone. Auto stays OFF for concurrent serving (head
    // VRAM comes out of the batch slot budget) and under runtime.deterministic
    // (MTP greedy trajectories are not eager-equal, docs/LIMITATIONS.md).
    // Resolution lives in tools/common/mtp_auto.* (both tools decide before an engine exists).
    int mtp_k = -1;
    // Adaptive MTP chain depth (AIMD): a fully accepted chain grows the next
    // draft by one row (up to mtp_k), any rejection sheds one row (floor 1).
    // The economics guard prices the average depth that actually ran, not the
    // configured ceiling. Off = fixed depth mtp_k (A/B kill switch).
    bool mtp_adaptive_k = true;
    // Serve the MTP chain's full-vocab logits GEMV from the NVFP4 LM-head
    // decode cache when one exists (#847 lever 3): the chain re-reads the LM
    // head once per drafted token, more traffic than the main forward at k=4.
    // Draft-only precision; verification stays lossless. Off = keep the FP16 chain GEMV.
    bool mtp_nvfp4_head = true;
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
    // Context capacity the captured gather grids and persistent K/V scratch are
    // sized for: 2 * ctx_cap * nkv * hd * 2B VRAM. Verify steps beyond this run
    // eager. Clamped to the model's max_seq_len.
    int capture_ctx_cap = 32768;
};
}  // namespace imp::cfg
