#pragma once

// Speculative configuration, one of the nine sections split out of
// core/dispatch_policy.h on 2026-08-21.
//
// WHY. dispatch_policy.h aggregates all nine and is included by 23 translation
// units, of which 21 touch two sections or fewer. Adding one field to it costs
// 137.1 s of incremental rebuild, against 9.1 s for a small .cpp and 14.6 s for
// the largest .cu the file-size gate polices. A TU that needs only this section
// can include only this header and stop rebuilding when the others change.
//
// This is F-10 one level down, and dispatch_policy.h's own preamble records the
// original: config.h was included by 22 files, 85 TUs transitively, and changed
// 130 times in six months - "the highest build cost in the repo". Lifting nine
// sections into an aggregate fixed that, and gave the aggregate the same
// property for the same reason.
//
// Pure move: the contents below are byte-identical to their previous form, and
// dispatch_policy.h includes every one of these, so no existing include breaks.

#include <cstdint>
#include <string>
#include <vector>

namespace imp::cfg {

struct Speculative {
    // Prompt-lookup speculation, default-on (batch-1, greedy, dense). This
    // switches the HISTORY MATCHER only. It used to switch the whole verify
    // step, which meant setting it false silently disabled MTP and token
    // recycling too, with no diagnostic: mtp_k=2 drafted zero tokens. Entering
    // the verify is now decided by spec_any_drafter (runtime/spec_gates.h),
    // which asks all three sources.
    bool ngram = true;
    // Context cap for DENSE chunk-verify drafting (#964). With the
    // verify_decode_attn route below (2026-07-12), dense n-gram WINS up
    // to at least 12k context on the captured-verify (server) path:
    // Qwen3-8B Q8_0 route+capture vs spec-off +44% @512, +43% @2k,
    // +25% @8k, +29% @12288 — but still −23% at 15872 (the near-msl
    // band: the capture-ready ceiling ends there and the remaining
    // verify cost is the weight-bound small-M GEMMs + per-step host
    // work, not attention). Drafting is gated once a request's context
    // crosses the cap (checked per step). MoE-NVFP4 and GDN-hybrid
    // requests are exempt (deep drafts: Coder-30B code-edit 15.9
    // tok/verify, MTP chains — the verify pays for itself there).
    // History: pre-route the FA2-tile verify was net-negative past ~2k
    // (−62% @16k, 2026-07-11) and the cap defaulted 2048; the route
    // (verify_decode_attn) then raised it to 12288. Superseded 2026-07-12
    // by the DEPTH-aware gate below (shallow_draft_ctx): the residual
    // long-context loss was never a context cost — it was draft depth
    // (a verify step costs ~1.4x a decode step @512 rising to ~2.6x
    // @16k, so 1-token drafts stop paying past ~14k while 3-token
    // drafts win +24% at 13k). Hard cap now defaults OFF. 0 = no cap.
    int draft_ctx_cap = 0;
    // #964 stage 2: past this request context, 1-token n-gram/suffix
    // drafts are discarded instead of verified (the miss-burst path
    // serves the step at plain decode speed); drafts of depth >= 2 are
    // always verified. Bench evidence (Qwen3-8B Q8_0, route+capture):
    // depth-3 drafts +24% @13312, depth-1 drafts −23% @15872 — the
    // break-even for depth 1 sits near 14k where verify/decode cost
    // reaches ~2x. MoE-NVFP4 and hybrid requests are exempt (deep
    // drafts). 0 = never gate.
    int shallow_draft_ctx = 12288;
    // #964 structural fix: run the dense verify chunk's ATTENTION through
    // the batched-decode split-K paged kernels (rows become same-KV
    // "sequences" with per-row context lens p0+1+i — causality holds by
    // construction; pad rows attend 1 token) instead of the small-M
    // prefill FA2 tile + full-context FP16 KV gather. nsys @16k (Qwen3-8B
    // Q8_0): 557 us/layer FA2 vs ~65 us/layer split-K — ~20 ms of the
    // ~30 ms verify step, and the reason FP8 KV never moved the verify
    // cost (the old path gathered to FP16 first). Composes with the
    // captured verify; requires the 3/5 capture buckets (the split count
    // bakes from the PADDED row count). Measured end-to-end (route +
    // capture vs spec-off, 3 trials): +44% @512, +43% @2k, +25% @8k,
    // +29% @12288. Dense non-MLA/non-SWA/non-MoE/non-hybrid only;
    // MoE/hybrid keep the FA2 chunk path. Kill switch for A/B.
    bool verify_decode_attn = true;
    // #998: verify-chunk GEMMs (M <= largest capture bucket) read the
    // NVFP4 decode overlay in one weight pass per MR tile instead of the
    // M>1 prefill dequant path. On GGUF K-quants without a direct
    // small-M kernel (Q6_K) the per-chunk source dequant cost ~7x a
    // decode step (dequant = 52% of the tg window at ctx 2048 on
    // Qwen3-14B Q6_K, tg -39% vs spec-off). Also aligns verify argmax
    // with the decode path (same weights). Real prefills are never
    // affected. Kill switch for A/B.
    bool verify_nvfp4_gemm = true;
    // Route the verify chunk's native-NVFP4 GEMMs through the small-M
    // mxf4nvf4 pipeline kernel (gemm.nvfp4_smallm) instead of the batched
    // multi-row GEMV. The GEMV reads the weight sweep at ~1300 GB/s at M=3
    // (70% of verify kernel time, profiled 2026-08-27); the smallm kernel
    // reads the same weights at the decode-class ~1600. Argmax parity is not
    // the objection: the batched-GEMV path already documents that a
    // speculative arm does not reproduce the non-speculative greedy output.
    //
    // Measured 2026-08-27 (Qwen3.8-27B-NVFP4, mtp_k=1 + ngram=false,
    // 1024-token thinking chats): +3-6% in an isolated smoke (107.7/106.6 vs
    // 100.6-104.6), but only +1-2% with mixed pairs in a 2-round alternating
    // A/B (BASE 104.5-107.7 vs COMBO 104.9-109.6, accept unchanged ~73%) -
    // inside the greedy-trajectory variance of this measurement. Default off
    // until a cleaner harness can resolve it; the mechanism is real, the
    // e2e effect is smaller than the kernel share suggests.
    bool verify_smallm = false;
    // Make the speculative verify chunk's NVFP4 GEMV reduce K exactly the way
    // the M=1 decode GEMV does, so the two paths agree bit for bit. Both
    // compute the same products; decode groups them into 32 partial sums (one
    // per warp lane) and the batched verify kernel into 128 (one per block
    // thread). That rounding difference reached the STOP decision: at
    // speculative.mtp_k=1 on Qwen3.8-27B-NVFP4 it truncated 2 of 6 answers
    // after ~40 tokens (docs/LIMITATIONS.md). Off by default in the PR that
    // introduced it, pending its own measurement.
    bool verify_row_parity = false;
    // Speculation on MoE models with NATIVE-NVFP4 experts (the gate
    // additionally requires profile().moe_experts_nvfp4). Measured on
    // Qwen3-Coder-30B-FP4 (2026-07-02): code-edit +49-81% (93% accept,
    // 15.9 tok/verify), draft-poor code-gen -3-7% (miss_burst hybrid
    // bounds the downside). GGUF-MoE verify re-dequants every activated
    // expert per step (-22% measured) and never engages regardless of
    // this flag. imp-cli --bench pins this false so the canonical
    // perf-baseline decode signal stays raw (verify inherits grouped-GEMM
    // restart variance).
    bool moe = true;
    // #1003 stage 1: at decode batch > 1, ONE request per step may run
    // its spec verify (round-robin, cyclic id order) while the remaining
    // rows decode batched. Three guards make it pay: a draft-depth floor
    // (min_draft = 2x batch — the whole batch stalls for the verify, so
    // shallow drafts measured -10% unguarded), a pipeline yield (the
    // chained batched decode otherwise never gives a turn), and an
    // ADAPTIVE yield cadence (8 -> 64 steps exponential backoff on empty
    // turns; fruitless chain breaks alone cost -1.5..-2.7% before it).
    // Default-ON matrix (Coder-30B NVFP4, 2026-07-15, 2 trials/cell,
    // stream-free host): code-edit +7.9/+11.0% at batch 8 (twice,
    // non-overlapping trials), +2.6/+4.4% at 16; diverse chat neutral
    // (+2.2/+3.5% at 4/8, remaining negatives inside the +-6% intra-arm
    // trial spread). Batch-1 behavior unchanged. Kill switch for A/B.
    bool batch_rr = true;
    int k = 16;  // draft tokens per verify step (verify cost is ~flat in k)
    // Token-Recycling adjacency drafting (ACL 2025, arXiv 2408.08696;
    // plan docs/plans/2026-07-22-token-recycling-spec-tree.md):
    // engine-scoped cross-request `token -> top-M successors` table fed
    // from emitted bigrams and the model's own verify-chunk top-K
    // logits. Fires on unigram context, so it drafts on fresh
    // reasoning/agentic prose where suffix/n-gram matching finds
    // nothing (measured 0 drafts on reasoning, 2026-07-22/23). Runs as
    // a fallback AFTER suffix/n-gram and MTP; lossless via the greedy
    // argmax verify. Default off until the reasoning A/B proves it.
    bool token_recycling = false;
    int recycle_slots = 8;  // successors kept per token (MRU/rank order)
    // Linear draft length. Default 3 -> chunk 4 -> capture bucket 4 =
    // exactly one batched-GEMV weight sweep (M=4, #1055); deeper chains
    // pad into bucket 5+ and pay a second sweep.
    int recycle_depth = 3;
    // Precision gate (#1055): only draft hops whose front slot was
    // re-confirmed at least this many times (bigram repeated / model
    // top-1 stable). A verify step costs ~1.4x a decode step while a
    // miss just rides the async-loop burst — precision beats recall.
    // 0 = draft on any known successor.
    int recycle_min_streak = 1;
    // Multi-candidate verify (route (a) of the spec-tree plan): when the
    // decode-attn route is available (dense, non-MLA/SWA/MoE/hybrid, no
    // penalties/MTP), verify `recycle_width` adjacency candidates in one
    // chunk — each candidate gets its own t0 row and private copies of
    // the KV blocks its rows write (per-row block tables), so no token
    // mask is needed and the argmax accept stays lossless. Rows are
    // capped at the 17 bucket (width * (1+depth) <= 17), i.e. width 4
    // -> depth 3. Default 1 (linear): at bucket 17 the chunk GEMMs run
    // CUTLASS at ~51% effective bandwidth (measured 2026-07-23) — the
    // multi-candidate accept lift (2.0 vs 1.44 emitted/verify) does not
    // pay for the 2x-costlier chunk; linear bucket-4 chunks ride the
    // single-sweep batched GEMV instead.
    int recycle_width = 1;
    // SuffixDecoding-style indexed drafting (arXiv 2411.04975):
    // hash-indexed suffix matching (O(1) amortized vs the legacy O(n)
    // backward scan per verify step) with frequency-voted continuations
    // across all occurrences, and adaptive draft length — a draft backed
    // by multiple agreeing occurrences or a maximal-length (max_match)
    // context match extends past `k` up to `suffix_k_max`. false =
    // legacy single-most-recent scan.
    bool suffix = true;
    int suffix_k_max = 64;
    // Longer suffix matches trade draft frequency for precision — and
    // precision wins decisively: min_match 6 vs 3 measured +16% on
    // code-edit (50% acceptance) while cutting the structured-content
    // worst case from -13% to -2% (false 3-gram matches in number
    // tables produce drafts that never verify).
    int min_match = 6;   // shortest accepted suffix n-gram match
    int max_match = 12;  // longest suffix extension searched
    // After this many consecutive draft misses the request gives up on
    // speculation and re-enters the async conditional graph loop (the
    // eager per-token path costs ~2x vs the loop — a draft-poor context
    // must not pay that for the whole generation). 0 = never give up.
    int give_up_after = 64;
    // Prompt-lookup drafting is COLD by construction: it matches a suffix of
    // the generated text against earlier occurrences, and until the generation
    // is long enough there is nothing to match. Measured on Qwen3-14B Q6_K
    // (the north-star model), one request, same prompt, 2026-08-21:
    //
    //   generated   accepted   tok/verify
    //     128         0.0 %       0.00     drafter never fires
    //     256         6.2 %       2.00     fires and LOSES
    //     512        39.6 %       7.33     strongly profitable
    //    1024        36.1 %       6.78
    //
    // A verify costs ~50 ms against a ~6 ms decode step on this checkpoint, so
    // 2.0 tokens per verify is four times under water while 6.78 pays. The
    // economics guard below cannot catch the cold phase: it arms on
    // spec_verifies >= 8 PER REQUEST and a short request produces about one, so
    // the verdict is never made rather than made badly.
    //
    // Conditioning on tokens generated SO FAR, which is known at the decision
    // point, rather than on the request's eventual length, which is not.
    // 0 disables the gate.
    int min_history = 0;
    // Burst-hybrid: while given up, the async loop runs in bursts of
    // this many tokens; after each burst the request re-probes drafts
    // for a couple of steps (think models produce their draft-rich
    // region only after the reasoning prose). 0 = give-up is final.
    int burst = 128;
    // On a draft miss the request falls back to the async loop for this
    // many tokens (cheap rearm, no graph recapture) instead of paying
    // the ~2x eager per-token tax until the next draft shows up.
    // 0 = stay eager between drafts (legacy behavior).
    int miss_burst = 8;
    // Reuse the parked captured graph across bursts (rearm instead of
    // recapture, ~10-20 ms saved per burst). The #683 wrong-token
    // artifact was NOT the rearm itself but the fresh-captured loop
    // initializing position/context one too high (fixed in
    // CudaGraphConditionalRunner::setup) — rearm and fresh capture now
    // share the same first-forward semantics.
    bool burst_rearm = true;
    // Speculation on hybrid (GDN/SSM) models. The verify chunk advances
    // recurrent state through rejected draft positions, so the committed
    // per-sequence state slab is snapshotted before the chunk; a fully
    // accepted draft pays only that copy (~60 MiB D2D on Qwen3.6-35B),
    // a partial acceptance restores the slab and re-forwards the
    // accepted prefix (one extra chunk forward, amortized over the
    // accepted tokens). imp-cli --bench pins this false (same
    // baseline-semantics rule as the moe/suffix pins).
    bool hybrid = true;
    // MTP-head chain-draft length for the verify loop (models shipping a
    // trained MTP head, e.g. Qwen3.6 model_mtp.safetensors). 0 = off.
    // When >0 the server loads the head and enables MTP. Its VRAM cost is
    // per checkpoint, not a constant: Qwen3.8-27B's dense-MLP head measures
    // 0.79 GiB (15 tensors, BF16 to FP16 on upload), where the ~1.6 GiB this
    // line used to quote came from a MoE head. Read the load line rather
    // than this comment;
    // drafts fill verify steps where the suffix/ngram matcher has no
    // match (draft-poor prose — 78-94% depth-1 accept on Qwen3.6-35B-A3B,
    // PR #804). imp-cli equivalent: --mtp-spec-decode <k>.
    // Independent of `ngram` above: with ngram=false and mtp_k=2 the head
    // drafts and the matcher does not (measured 100 drafts over 50 verifies
    // on Qwen3.8-27B, against 0 before the gate split).
    int mtp_k = 0;
    // Adaptive MTP chain depth (AIMD): a fully accepted chain grows the next
    // draft by one row (up to mtp_k), any rejection sheds one row (floor 1).
    // Draft-poor prompts converge to k=1 verifies instead of paying the
    // deep-chunk verify cost at low accept (k=2 fixed measured 84.9-145.8
    // tok/s against 101.6-106.8 at k=1 on the same prompts, 2026-08-27);
    // draft-rich prompts climb back to the configured mtp_k. The economics
    // guard prices the average chain depth that actually ran, not the
    // configured ceiling. Off = fixed chain depth mtp_k (A/B kill switch).
    bool mtp_adaptive_k = true;
    // Serve the MTP chain's full-vocab logits GEMV from the NVFP4 LM-head
    // decode cache when one exists (#847 lever 3). The chain re-reads the
    // LM head once per drafted token (~2.5 GB FP16 on Qwen3.6-27B's 248k
    // vocab — more traffic than the main forward at k=4); NVFP4 reads ~4x
    // less. Draft-only precision, verification stays lossless. Off = keep
    // the FP16 chain GEMV (A/B kill switch).
    bool mtp_nvfp4_head = true;
    // MTP economics guard: after an 8-verify sample, average emitted
    // tokens per MTP-filled verify below this dooms MTP drafting for the
    // request (the verify chunk + chain cost can't amortize). 0 disables
    // the guard (raw-economics measurement). Re-derive when chain/verify
    // costs change — note a chain of k can emit at most k+1 per verify, so
    // values >= k+1 doom that k unconditionally.
    //
    // NEGATIVE (the default) selects a k-aware threshold, 0 disables the
    // guard, and a positive value is taken as an absolute floor.
    //
    // It used to be an absolute 4.0, which doomed MTP at every chain length
    // this engine can run: a chain of k emits at most k+1 per verify, so 4.0
    // is unreachable for k=1 (max 2.0), k=2 (max 3.0) and k=3 (max 4.0). The
    // guard therefore unbound MTP after its 8-verify sample by arithmetic,
    // whatever the speed, and the 21 % that mtp_k=2 now delivers could not be
    // received by anyone who did not also override this key. 4.0 came from
    // #852, when the verify ran eagerly and a partial acceptance re-forwarded
    // the accepted prefix through the whole model; both are long gone.
    //
    // An absolute value cannot be right for every k, because break-even is
    // chunk_cost(k+1 rows) / decode_cost and that ratio grows with the chain.
    // Measured on Qwen3.8-27B-NVFP4 after the 2026-08-18 launch fixes, cost
    // per verify against an 11.21 ms decode step:
    //
    //   k=1  15.59 ms -> break-even 1.39, emits 1.721
    //   k=2  19.65 ms -> break-even 1.75, emits 2.195
    //   k=3  27.27 ms -> break-even 2.43, emits 2.629
    //
    // IT IS AN ACCEPTANCE RULE, and writing it in emitted tokens hid that.
    // A verify emits exactly 1 + accepted (the base token plus every accepted
    // draft; confirmed in the data: 1.721 = 1 + 0.721, 2.195 = 1 + 1.195,
    // 2.612 = 1 + 1.612). So `1 + f k` is precisely "unbind below f draft
    // acceptance", and the break-evens above become 39.0 %, 37.5 % and 47.7 %,
    // which vary far less across k than 1.39 / 1.75 / 2.43 do. Acceptance is
    // the natural unit here; emitted tokens are not.
    //
    // f = 0.40 gives 1.4 / 1.8 / 2.2 against break-evens of 1.39 / 1.75 / 2.43
    // and measured emissions of 1.721 / 2.195 / 2.612. Essentially exact at
    // k=1, 2.9 % strict at k=2, and 9.5 % PERMISSIVE at k=3, where a marginally
    // losing chain now survives. That direction is deliberate: a permissive
    // guard costs a few percent on a bad workload, a strict one costs the whole
    // feature on a good one, and the 4.0 this replaced was the strict failure
    // taken to its limit. An earlier attempt at f = 0.5 was 11 to 12.5
    // percentage points stricter than break-even at k=1 and k=2 for the same
    // reason it looked reasonable: nobody had converted it to acceptance.
    //
    // Separate from the hard floor three lines below in engine_spec_ngram.cpp:
    // `accepted*100 < drafted*15` dooms ALL speculation for a request, n-gram
    // included, and re-enables the async decode loop. This one only unbinds
    // MTP. Same quantity, different scope, and they are not to be merged.
    float mtp_econ_min_emit = -1.0f;
    // Graph-captured verify chunk (#847): cache one CUDA graph per
    // draft-length bucket and replay it each verify step — the chunk
    // metadata and KV lengths are read from device buffers, so the graph
    // survives context growth. Removes the eager launch-pacing tax
    // (~1800 launches/verify cycle). Drafts are padded up to the bucket
    // length (extra rows are causally invisible to the real rows and
    // their KV is rolled back with the rejected drafts). Engages only
    // where the FP16-QK FA2 kernel serves the chunk (uniform hd=128, no
    // sinks/MLA/LongRoPE) on non-hybrid models; anything else stays
    // eager. Any capture failure falls back to the eager verify and
    // disables capture for the process after repeated failures.
    bool capture = true;
    // Context capacity the captured gather grids and the persistent K/V
    // scratch are sized for (2 x ctx_cap x nkv x hd x 2B VRAM, e.g. 2x
    // 32 MiB at 32k for nkv=4/hd=128). Verify steps whose context
    // exceeds this run eager. Clamped to the model's max_seq_len.
    int capture_ctx_cap = 32768;
};
}  // namespace imp::cfg
