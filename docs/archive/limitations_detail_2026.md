<!--
layer: L1
audience: records
verified: 2026-09-22
commit: 9cbb8004
-->

# Limitations detail: speculative decoding investigations

Full investigation records behind the condensed rows in [`LIMITATIONS.md`](../LIMITATIONS.md).
Append-only. Numbers keep their original provenance and are not re-verified against current
code; a PROV block or a dated measurement is a record of that afternoon, not a claim about HEAD.

## Speculative decoding economics

Thinking-traffic ledger (2026-08-27), Qwen3.8-27B-NVFP4:

| finding | verdict | numbers |
|---|---|---|
| MTP effectively dead on thinking traffic: think-interior loop burst desynced the MTP cache | fixed | drafted_total 1 over a 768-token essay |
| three fixes (burst-site exclusions for MTP-bound requests, verify inside the think block, in-think emit trim) + verify-argmax banned-token mask | shipped | `mtp_k=1` + `speculative.ngram=false`: 102.1-105.9 tok/s vs 87.2-87.6 default on a 1024-token thinking chat (+17-21 %, all six MTP runs above all six default runs); degen suite 50/0 twice |
| `ngram=false` must stay in the pair | keep | matcher ON reproducibly derails one trivial prompt into an empty-content completion (model stops inside think; greedy under chunk-shaped forwards is not the eager trajectory); k=2 shows the same class |

**Default since 2026-08-29: `mtp_k=-1` (auto).** Auto engages the pair (`mtp_k=2`, `ngram=false`)
on a single-stream run whose checkpoint ships a head: 95.8 -> 141.6 tok/s on the thinking prompt
above, degen 50/0. Declines for concurrent serving (the head binds one request and costs batch
slots) and under `runtime.deterministic` (chunk-greedy != eager-greedy, so the reproducibility
promise outranks the speedup). `speculative.mtp_k=0` opts out; an explicitly set `ngram` is never
overruled.

```
[PROV: commit=3ce0c326+mtp-fixes date=2026-08-27 hw=RTX5090
       model=Qwen3.8-27B-NVFP4 quant=NVFP4 cuda=13.3 path=imp-server
       n=3 runs/arm, /v1/chat/completions, 1024-token thinking answer, temperature=0
       cmd=`imp-server --set speculative.mtp_k=1 --set speculative.ngram=false` vs defaults
       degen: `tools/analysis/degen_suite.py` 2x 50 checks]
```

Adaptive chain depth (2026-08-27): `speculative.mtp_adaptive_k` (default on) walks the chain
between 1 and `mtp_k` on acceptance (full accept +1 row, any rejection -1); the economics guard
prices the depth that ran, so a deep `mtp_k` no longer dooms draft-poor requests. Opt-in pair is
now `mtp_k=2` + `ngram=false`: thinking chats 111.1-113.3 tok/s vs 106.3-108.0 at k=1,
94.9-110.2 at fixed k=2, 86.9-87.7 spec-off (+27-30 %), 3/3 alternating rounds each; draft-rich
prompts 156.6-158.2 (parity with fixed k=2, +31 % over k=1); cost: no-think prose -1.5 % median
vs k=1. Degen suite 50 checks, one bistable think-budget FAIL that re-reads 2x 10/0 in category
re-runs.

```
[PROV: commit=perf/mtp-adaptive-k date=2026-08-27 hw=RTX5090
       model=Qwen3.8-27B-NVFP4 quant=NVFP4 cuda=13.3 path=imp-server
       harness=tools/analysis/mtp_adaptive_ab.sh (3 rounds alternating,
       THINK=1 arm for the thinking numbers, ARMS=k0 for spec-off)]
```

## Nemotron MTP defect

Two accept rates on the released Qwen3.8-27B-NVFP4 row, different measurements: **82.7 %** is
offline top-1 accept, teacher-forced, verify loop pinned off (`scripts/mtp_accuracy_bench.sh`:
89.0 / 75.6 / 81.9 / 84.3 % on factual / verbose-think / code / instruction, 127 scored positions
each). **67.0 %** is what the verify chunk accepted on the serving path over 30 prompts (4299 of
6415 drafts, `/metrics`). The gap is the cost of drafting into a chunk.

Third-row defect (`nemotron_h`, MoE + Mamba2 hybrid): a cached verify-chunk graph replayed against
a state it was not captured for does not reproduce an eager forward of that state: logit deltas to
23.8, roughly 1 generation in 8 with a visibly duplicated word. Not the drafter, not the attention
route; `moe.skip` drops the rate 11.2 % -> 0.8 %, so the MoE pass carries it, but MoE alone is not
enough (the MoE+GDN row is MoE and clean); the stale baked value is not identified.
`tests/test_spec_capture_fidelity.cpp` gates the two clean rows and fails on this one. The two
clean rows are not bit-exact either: capture picks its cuBLASLt algorithm once, so roughly
0.1-0.2 % of replays differ from eager on a healthy model; the gate's 2 % threshold sits above
that floor.

## MTP truncation

**MTP speculation truncates answers: 2 of 6 prompts end after ~40 tokens with a re-statement of
the question (measured 2026-08-19).** Truncated answer: 164 B, `finish_reason: "stop"`, its last
clause the tail of the prompt. Qwen3.8-27B-NVFP4, `speculative.mtp_k=1`,
`speculative.ngram=false`, `server.prefix_cache=false`, six prompts, `max_tokens: 400`:

| arm | degenerate | lengths |
|---|---|---|
| `mtp_k=1` | **2 / 6** | 164 B, 286 B, and four of 1198-1793 B |
| `mtp_k=0` (control) | **0 / 6** | 1146-1784 B |

Deterministic, not noise: with `runtime.deterministic_gemm=true` four fresh processes produce the
same truncated answer byte for byte (4 / 4, identical sha); without it 3 / 4. The pin stabilises
the state, it does not remove it. Mechanism (`diagnostics.spec_trace`): the final verify reads
`p0=114 t0=12482 draft=[13] argmax=[13,248046]`; 12482 is `" matters"`, 13 is `"."`, 248046 is
`<|im_end|>`. The bonus token comes from the last chunk row, which predicts end-of-turn where
single-token decode keeps writing.

`speculative.verify_nvfp4_gemm=false` (chunk off the batched NVFP4 GEMV): truncation 0 / 6 on the
same prompts, acceptance unchanged (72-80 %), throughput gone:

| arm | tok/s (r1, r2) | vs k=0 |
|---|---|---:|
| k=0 | 88.97, 88.93 | |
| k=1, `verify_nvfp4_gemm=false` | 80.50, 77.43 | **-11.2 %** |
| k=1, default (truncates) | see `roadmap.md` | +21.3 % |

So the +21.3 % comes from precisely the kernel that causes the truncation. Both accumulate in
`float`; the difference is summation order.

**Pass 1 (2026-08-21): re-measured, four kernel-divergence sites.** 2 / 6 reproduces
byte-for-byte. Verify chunk and M=1 decode take different kernels at four sites, all gated on
`n == 1`:

| site | difference | status |
|---|---|---|
| q/k/v, gate/up projections | K reduced in 32 partial sums (decode, `gemv_nvfp4_multirow_kernel<8>`, one warp per output row) vs 128 (verify, `gemv_nvfp4_kpar_mb_fp16_kernel`, one block per output row) | **fixed**, `speculative.verify_row_parity` |
| down projection GEMM | none: `n_mb = 1088 > 512`, both take the 128-wide path (measured) | already equal |
| SwiGLU into down | decode keeps `silu(gate)*up` in float registers, verify rounds to FP16 in a separate kernel | open |
| RoPE + KV write, QK-norm | `can_fuse_rope_kv` and the fused QK-norm are `n == 1` only (`executor_attention.cu:365,405`) | open |

Inner loops otherwise instruction-identical (same `cvt.rn.f16x2.e2m1x2` dequant, same 16 fma
pairs, same scale); site 1 is a pure grouping difference, closed free
(`gemv_nvfp4_multirow_mb_kernel` keeps the 32-lane warp partition and reads each weight
micro-block once for all rows).

| arm | tok/s (r1, r2) | vs k=0 | degenerate |
|---|---|---|---:|
| k=0 | 88.56, 88.47 | | 0 / 6 |
| k=1, default | 104.46, 104.02 | +17.8 % | 2 / 6 |
| k=1, `verify_row_parity=true` | 105.52, 105.07 | **+19.0 %** | **1 / 6** |

Refuted as stated: "MTP is either fast and wrong, or correct and slower than no speculation at
all"; the parity arm is the fastest of the three. A fused multi-row down projection measured
**-27.7 %** against the batched path and did not change the count; removed. Correction: the
-27.7 % was the implementation, not the fix (single-row helper called once per activation row, MR
full sweeps of the 5120x17408 weight; its comment's "cannot be hoisted" claim is wrong,
micro-block outer / rows inner works); the rewritten hoisting version still does not help and is
not shipped.

**Pass 2 (2026-08-21): closing the remaining sites does NOT reduce the count.** Sites 3 and 4
built and proven live (one-shot log line each, n=3), one build, six prompts, two fresh processes
per arm:

| arm | degenerate | which prompts |
|---|---|---|
| parity off | 2 / 6 | 1, 4 |
| site 1 (K-reduction width) | 1 / 6 | 1 |
| sites 1 + 4 (QK-norm/RoPE fusion) | 1 / 6 | 1 |
| sites 1 + 3 + 4 (+ SwiGLU fusion) | **2 / 6** | 1, **3** |

Stable within an arm, different prompts between arms: each perturbation reshuffles which prompts
fall over. Sites 3 and 4 not shipped (no measured benefit; site 3 makes it worse); only site 1
ships, because it is free and faster, not because 1 / 6 is proven better than 2 / 6 on six
prompts. Caveat: `verify_nvfp4_gemm=false` reaches 0 / 6 including prompt 1, which no parity arm
does; it routes the chunk to CUTLASS, a third kernel, so 0 / 6 is a property of these six prompts,
not an established mechanism. The `n == 1` path is pervasively fused (QKV, gate/up, o, down,
QK-norm+RoPE), so a verify chunk is a different execution graph, not the same graph at a different
width; real parity is that architecture, not four kernels.

**Pass 3 (2026-08-21): the stop decision is NOT a knife edge; refutes pass 2's explanation.**
`diagnostics.spec_trace` top-2 logit gap per chunk row, 892 verify steps / 1784 chunk rows:

| rows | n | p05 | median | p95 |
|---|---|---|---|---|
| all chunk rows | 1784 | 0.13 | **1.74** | 9.01 |
| bonus rows (last row of each chunk) | 892 | 0.12 | 1.71 | 8.89 |
| rows whose argmax is EOS (`248046`) | 2 | | **1.79** | |

```
top1=248046 (<|im_end|>)  top2=271  gap=1.9852
top1=248046 (<|im_end|>)  top2=271  gap=1.5986
```

Both EOS decisions sit above the median gap of an ordinary position; 33.2 % of all rows (593 of
1784) are tighter than 1.0. The disagreement with the single-token path is substantial, not
marginal; the suspect moves off the LM-head numerics onto the hidden state feeding that row. Rate
pricing any fix: 2 of 892 verify steps propose EOS (0.22 %); a guard re-running those through
single-token decode would fire ~1 in 450 verify steps and cost one decode step.

```
[PROV: commit=86479ce4 date=2026-08-21 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=imp-server n=1 process, 6 prompts,
       max_tokens=400, 892 verify steps
       cmd=`imp-server --think-budget 0 --set speculative.mtp_k=1
       --set speculative.ngram=false --set server.prefix_cache=false
       --set runtime.deterministic_gemm=true --set diagnostics.spec_trace=true`
       gaps from the `gap=[id1>id2:x]` field this pass added to the verify trace,
       top-2 over the full vocab of each chunk row's logits. The arm reproduced
       2/6 degenerate, so the traced run is the failing one. NOT measured: the
       same positions under no speculation - there is no verify chunk at
       mtp_k=0 and therefore no trace, so the comparison here is EOS rows
       against ordinary rows of the same run.]
```

**Pass 3, localisation: the truncation is not localised, and the requested hidden-state diff is
not obtainable.**

- The position does not exist in the non-speculative arm: on prompt 1 the k=1 and k=0 answers
  share only their first 49 bytes of 165 and 2358.
- The instrument cannot see the decode loop: `diagnostics.dump_hidden_dir` is host-side, decode is
  CUDA-graph replayed. 40 generated tokens -> 5 distinct dump steps; 200 tokens -> 5.
  `speculative.capture=false` makes every step visible but changes the arm: 2 / 6 then 1 / 6
  across two fresh processes, vs capture-on stable at 2 / 6 both times. (Correction to a pass-2
  claim: `dump_hidden_dir` does not destroy the phenomenon; a run with it set still truncates at
  exactly 164 bytes with `finish_reason=stop` and still captures graphs; it cannot see the steps
  that matter.)
- The divergence point carries no signal (k=1 vs k=0, first differing byte):

| prompt | k=1 bytes | k=0 bytes | first divergence | |
|---|---|---|---|---|
| 1 | **165** | 2358 | 49 | truncated |
| 2 | 2293 | 2223 | 271 | ok |
| 3 | 2157 | 2583 | 103 | ok |
| 4 | 2572 | 2574 | 135 | ok |
| 5 | 2687 | 2666 | 48 | ok |
| 6 | 2634 | 2701 | 49 | ok |

All six diverge between byte 48 and 271; the two earliest (48, 49) produce full clean answers.
Truncation is one outcome among six of the ordinary divergence documented in "MTP vs greedy
divergence" below, not a separate upstream event with a location; that is why the parity sites
reshuffle instead of reducing.

```
[PROV: commit=2a049185 date=2026-08-21 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       cmd=`--set speculative.mtp_k={0,1} --set speculative.ngram=false
       --set server.prefix_cache=false --set runtime.deterministic_gemm=true --think-budget 0`
       note=six prompts, max_tokens 600, temperature 0, top_k 1; dump-step counts
       from `diagnostics.dump_hidden_dir`]
```

**Pass 4 (2026-08-21): a stop-decision guard was built, measured and removed; the ordinary decode
path agrees the turn is over.** Guard: when a chunk row's argmax is a stop token, drop the row and
hand the position to the single-row decode path. One build, six prompts, two fresh processes per
arm:

| arm | degenerate | guard fires |
|---|---|---|
| guard on, parity off | 2 / 6, 2 / 6 | 8 in 898 verify steps (0.89 %) |
| guard on, parity on | 1 / 6, 1 / 6 | 4 in 1126 verify steps (0.36 %) |

Identical to the same arms without it. On the truncating prompt the guard fired twice in that
request's 24 verify steps and the answer still ended at exactly 164 bytes: the confident stop is a
property of the state, not of the projection reading it. Cost (paired alternating, two rounds):
k=0 88.36 / 88.51, parity 106.74 / 105.65, parity + guard 105.50 / 104.34; the guard is +18.6 %
over k=0 and 1.2 % below parity: affordable, removed anyway because it does not change the count.
Its fire rate disagrees with the 0.22 % EOS-proposal prediction by 1.6 to 4x (the trace counted
proposals it could see; the guard counts every stop-argmax chunk row outside a think block; the
larger number is the real one).

**This closes the line.** Four approaches measured: per-kernel parity at four sites, routing the
chunk off the overlay, a fused multi-row SwiGLU, the guard. None reaches 0 / 6; pass 3 says why
none could (divergence is complete by byte 271 on every prompt).

```
[PROV: commit=fa21f28e date=2026-08-21 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       cmd=`tools/analysis/mtp_truncation_check.sh 1` with `MTP_EXTRA_SET` per arm; guard
       built locally, not in the tree
       note=six prompts, max_tokens 400, temperature 0, top_k 1; throughput from three
       400-token requests per arm, arms alternated]
```

**Consequence: `speculative.mtp_k` stays 0 by default**, despite the +18.6 to +21.3 % on this
model (`roadmap.md`): a throughput win that truncates one answer in six is not a win. Refuted:
"closing this needs the chunk path to agree with decode numerically, which is a kernel change,
not a flag" (three of four kernels agree and the count did not fall; handing the stop decision to
decode did not either). What stands: speculation puts this model on a different trajectory from
byte 48 onward on every prompt, and one trajectory in six ends early. Caveat on the +21.3 %: the
arms generate different text and the speculative arm sometimes stops early; token counts ran 2100
at k=1 against 1847 at k=0, so it is not "shorter answers decode faster", but not like-for-like
either.

## MTP vs greedy divergence

**Speculative decoding does not reproduce non-speculative greedy output on a GDN hybrid, and
cannot by construction.** Contract: change speed, not tokens; here it changes tokens.
Qwen3.8-27B-NVFP4, `runtime.deterministic_gemm=true`, `speculative.ngram=false`,
`server.prefix_cache=false` both arms, three prompts x 256 greedy tokens, stable control (two
no-speculation processes byte-identical; held across eleven processes): at `mtp_k=2` all three
prompts diverge, first at bytes 79 / 332 / 243 (0-indexed). Early token flips, both answers
coherent, different generations. Predates the 2026-08-17 verify work (same prompts diverge with
the older eager replay).

Structural cause, not a kernel defect: in a speculative arm every emitted token comes out of the
multi-row verify chunk, none out of the single-token decode step (chunk built at
`src/runtime/engine_spec_ngram.cpp:717,751`, only emit site `:988`). Proven by an image whose
`n == 1` decode path emitted pure garbage while its speculative arm stayed byte-identical to
stock. Decode and chunk also dispatch different kernels per shape: 10240x5120 and 12288x5120 take
`gemv_nvfp4_kpar` (32-lane `warp_k_loop`) at decode and `gemm_nvfp4_batched` in the chunk; FFN
shapes 17408x5120 and 5120x17408 never appear on the decode side (the `n == 1`-gated fused NVFP4
kernels serve them, `src/exec/executor_ffn.cu:98,140`). "Speculation reproduces non-speculative
decode" could only hold by kernel coincidence, and it does not. Withdrawn mechanism: "the verify
advances the recurrent state through the chunk kernels while decode uses the single-token path";
`--set gdn.chunkwise_scan=false` is byte-inert on both arms and the divergence survives at the
same offsets.

Five decode-side hypotheses dead, each killed by a switch or a proven-live patch; do not re-run:

| # | hypothesis | instrument | result |
|---|---|---|---|
| 1 | GDN chunkwise scan | `--set gdn.chunkwise_scan=false` | byte-inert both arms |
| 2 | fused QK-norm + RoPE | `--set attention.no_qknorm_fused=true` | byte-inert both arms |
| 3 | NVFP4 `use_multirow` K-partition split | patch at `src/quant/nvfp4_gemv_dense.cu:380`; dispatch log shows 10240x5120 and 12288x5120 moved `multirow=1` -> `multirow=0` at decode | byte-inert both arms |
| 4 | fused NVFP4 FFN at decode | two-site patch at `src/exec/executor_ffn.cu:98,140`; log shows both FFN shapes on `gemv_nvfp4_kpar`, `gemv_nvfp4_gate_up_fused` never fired | decode byte-identical to stock across 2281 greedy tokens |
| 5 | attention kernel family | mirror flip at `src/exec/executor_attention.cu:473` (decode on the chunk's FA2 path) | moves prompts 2 and 3 decode-side, prompt 1 byte-identical; with `--set speculative.capture=false` (both paths eager `FA2_FP16QK`) divergence still at byte 79 |

All five are decode-side; direction matters. A chunk-side substitution does move the text:
`--set speculative.verify_nvfp4_gemm=false` moves the first difference to bytes 58 / 130 / 150 and
still never reaches the non-speculative reference; the chunk-side standing fact is the weaker one,
no kernel choice tried closes the gap. The eliminations hold at text level, not offset level: the
non-speculative prompt-1 answer is one 1282-byte text in 11 processes (md5
`35e7d9a93d14fe18a604a58fb6456388`), the `mtp_k=2` prompt-1 answer one 470-byte text in 8 processes
(md5 `c8a63ca87dc1ad95bc185b8c4e97d7b8`), differing from byte 79 on. Prompt 2 behaves the same;
prompt 3 does not (243 in five pairs, 253 in two, inseparable from the cross-process instability
below) and is only evidence that all three diverge.

```
[PROV: commit=e3c48aa2 date=2026-08-18 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=imp-server n=3 prompts x 256 greedy tokens
       per arm, fresh process per arm, one arm per process
       cmd=`--set server.prefix_cache=false --set runtime.deterministic_gemm=true
       --set speculative.ngram=false --set speculative.mtp_k=0|2
       --set speculative.mtp_econ_min_emit=0`, request `temperature 0, think_budget 0`
       divergence offsets from `cmp` over the saved answer bytes, kernel identity from
       the dispatch log, card idle]
```

## MTP cross-process stability

**A speculative arm is not byte-stable across processes at temperature 0 with identical flags;
usually byte-identical, sometimes not.** `mtp_k=2`: eight of nine processes agree byte for byte on
all three prompts; the ninth differs on prompt 3 (737 vs 733 bytes, first difference at byte 243)
with aggregate speculation counters identical at 476/371/238. `mtp_k=1`, coarser: two processes
produced 471 vs 1213 bytes on prompt 1, 326 drafts / 278 accepts vs 420 / 345. The no-speculation
arm is byte-stable across all eleven of its processes (prompt 1 the same 1282-byte text), so this
is the speculative path, not the host, and not localized to one prompt or chain length. Settles a
contradiction: the "two fresh processes byte-identical" observation in "MTP vs greedy divergence"
is the eight-of-nine case; the commit body of `ea547a53` (#1467) asserted non-reproducibility with
no number; both describe the same effect at different sample sizes. Callers: use
`speculative.mtp_k=0` with `speculative.ngram=false` for an arm that must reproduce; no setting
makes a speculative arm reproducible (in-process history moves it too, see "The same request can
produce different output on its second pass" in `LIMITATIONS.md`).

```
[PROV: commit=e3c48aa2 date=2026-08-18 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=imp-server n=3 prompts x 256 greedy tokens
       per process, 9 processes at mtp_k=2, 2 at mtp_k=1, 11 at mtp_k=0
       cmd=`--set server.prefix_cache=false --set runtime.deterministic_gemm=true
       --set speculative.ngram=false --set speculative.mtp_k=0|1|2
       --set speculative.mtp_econ_min_emit=0`, request `temperature 0, think_budget 0`
       answer bytes compared with `cmp`, drafts and accepts from /metrics deltas, card idle]
```

## MTP acceptance rate

**The MTP head accepts 75.0 % of its first draft; six explanations for the gap to published
figures are dead.** Qwen3.8-27B-NVFP4, 10-prompt mixed corpus (exposition, code, arithmetic,
enumeration), n-gram disabled, `mtp_k=1` (acceptance = first-position acceptance), two rounds,
fresh process each: **74.8 % and 75.2 %** (sigma 1.2, >1200 drafts per cell), 84.7 and 85.8 tok/s
against ~88 without speculation. At `mtp_k=2` the aggregate is 58.0-64.1 %.

```
[PROV: commit=37cd1543 date=2026-08-17 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=imp-server n=10 prompts x 256 greedy tokens
       per arm, 2 rounds, fresh process per arm, alternating
       cmd=`--set speculative.ngram=false --set speculative.mtp_k=1|2
       --set speculative.mtp_econ_min_emit=0 --set server.prefix_cache=false`
       request `temperature 0, think_budget 0`; acceptance from /metrics deltas
       (imp_spec_accepted_total / imp_spec_drafted_total), card idle]
```

Ruled out by measurement, do not re-run:

1. **Draft lm_head precision.** `speculative.mtp_nvfp4_head` true vs false: 55.9/55.5 % against
   51.6/56.6 %, fully overlapping. NVFP4 head ~8 % faster at no acceptance cost; default right.
2. **Quantised head weights.** MTP tensors carry no scale companions: BF16.
3. **Missing `gamma = 1 + W` offset.** All seven MTP norms upload through the offset-applying path
   (`up_norm` in `weight_upload.cu`).
4. **Hidden-state convention.** `diagnostics.mtp_prenorm_h` moves nothing (62.8 % against 62.8 %);
   `pre_fc_norm_hidden` normalises its input anyway.
5. **RoPE defect.** Disabling rotation looked like +7.6 points on one prompt set and reverses
   across lengths: 72.7/55.6 at 112 tokens, 66.5/68.3 at 607, 77.7/67.9 at 2767 (on/off).
6. **Uninitialised MTP KV cache.** Zeroing left the per-process spread intact (12.6 points before,
   10.1 after); first-position acceptance moved 73.2 -> 75.0, inseparable from noise at two
   samples per condition.

Measurement guidance: (a) greedy decoding is deterministic; two identical runs agreeing to a tenth
of a point measures determinism, not the effect; vary the workload. (b) Acceptance depends on the
text; spread is only a signal once output is identical.

## MTP GDN hybrid launch fix (resolved)

**RESOLVED (2026-08-18). MTP does not lose on a GDN hybrid; imp's MTP path lost, and no longer
does.** Two defects in how work was launched, not what it computed (`ea547a53`):

1. `ssm_conv1d_prefill_f32_silu_kernel` had a grid over tokens: a two-row verify chunk ran on two
   blocks of a 170-SM card, each block walking every channel serially.
2. `executor_ssm_gdn.cu` built its `GemmContext` without `cur_spec_verify_` (FFN and attention
   both pass it), so the `M<=4` batched-GEMV branch from #998/#1055 was unreachable for every GDN
   projection: 48 of 64 layers on this model.

| | before | after |
|---|---:|---:|
| no speculation | 84.47 tok/s | 86.21 tok/s |
| MTP k=2 | 75.26 tok/s | **104.06 tok/s** |
| kernel time per emitted token, k=2 | 11.35 ms | **8.93 ms** |
| CUTLASS launches per verify | 296.3 | 8.9 |

From 10.9 % slower than not speculating to 20.7 % faster. The kernel figure reproduces across two
independent runs on different corpora (8.93 and 8.93; no-speculation arm 11.21 and 11.29).

```
[PROV: commit=ea547a53 date=2026-08-18 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=imp-server n=3 prompts x 256 greedy tokens,
       2 alternating rounds, fresh process per arm, nsys per arm
       cmd=`--set speculative.ngram=false --set speculative.mtp_k=0|2
       --set speculative.mtp_econ_min_emit=0 --set server.prefix_cache=false`
       wall from usage.completion_tokens over request wall time, kernel from
       cuda_gpu_kern_sum, both from the SAME arm]
```

Not a cross-engine comparison: vLLM 0.27.1 was measured here only for MTP acceptance, 59.7 %
against imp's 58-64 % (parity; retired the drafter as a suspect). The cross-engine throughput
comparison lives in [`BENCHMARKS.md`](../BENCHMARKS.md) ("re-measured on one client", 2026-09-02).
Cost of learning this: six dead drafter-accuracy hypotheses (see "MTP acceptance rate" above) and
days chasing a published 87 % target that described an unpinned regime; acceptance was never the
problem.

**MTP stays off by default; a decision, not an omission.** `speculative.mtp_k` remains 0; enable
with `--set speculative.mtp_k=2`. Measured on Qwen3.8-27B-NVFP4 after the launch fixes:

- **+15.2 % decode** on the shipped economics guard, two rounds: 109.48 and 96.81 tok/s at k=2
  against 89.54 and 89.50 without. The rounds differ by 13 % and ran 342 against 93 verifies on
  identical settings: read it as roughly +8 % to +22 %. The 3.7x swing in draft opportunity is not
  understood.

```
[PROV: commit=6c2c9445 date=2026-08-18 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=imp-server n=3 prompts x 256 greedy tokens,
       2 alternating rounds, fresh process per arm
       cmd=`--set speculative.ngram=false --set speculative.mtp_k=0|2
       --set server.prefix_cache=false`, NO mtp_econ_min_emit override so the
       shipped guard applies; tok/s from usage.completion_tokens over request
       wall time, verify counts from /metrics]
```

- **0.79 GiB VRAM** for the draft head (15 tensors, BF16 to FP16 on upload), paid whether or not it
  drafts. The `~1.6 GiB` in `dispatch_policy.h` is not this checkpoint (dense MLP head; a MoE head
  costs more).
- **Output stops being reproducible across processes**; golden tests must pin `speculative.mtp_k`.
- Measured on one checkpoint; other MTP-carrying models untested.

The engine says this at load: a checkpoint shipping an MTP head with `mtp_k=0` gets one INFO line
naming the flag, the measured gain and the two prices. Name-only probe across all three checkpoint
layouts (sidecar file, shard index, single-file header); keys on the fusion projection the
loader's dispatch uses, not any `mtp.*` name, so an MTP tensor group imp could not load is not
advertised. Without it the option is invisible (head only loads at `mtp_k > 0`).

Not finished: a marginal chunk row still costs 4.22 ms, 38 % of a full decode step (down from
8.09 ms, 71 %), on a batch-1 memory-bound decode where it should be near-free. Levers, measured,
size order:

1. per-row FMA work in `gemv_nvfp4_kpar_mb_fp16`, which scales by construction; an HMMA tile
   kernel is the honest answer if it is the floor.
2. capture-bucket floor of 3: a two-row chunk is padded to three, ~17 % of its GEMV time for a row
   that does not exist.
3. argmax + D2H + rollback host round-trip: grows the non-kernel gap 0.31 -> 0.68 ms/token, eats
   16 % of the kernel win. Smallest; named fix in `roadmap.md`.

### Pre-snapshot build (superseded)

Superseded by the fix above; kept as the record of the broken build. Qwen3.8-27B-NVFP4, greedy,
256 tokens, thinking off, economics guard disabled, two alternating rounds, fresh process per arm:

| `mtp_k` | tok/s (round 1 / 2) | emitted per verify | draft acceptance |
|---|---|---|---|
| 0 | 89.2 / 87.2 | | |
| 2 | 83.3 / 80.1 | 2.22 / 2.25 | 58.8 / 54.3 % |
| 4 | 64.0 / 61.9 | 2.56 / 2.46 | 36.7 / 34.5 % |
| 6 | 53.8 / 52.9 | 2.55 / 2.44 | 24.6 / 23.9 % |

Emitted-per-verify saturates near 2.5 from `mtp_k=4` while the chunk grows linearly in k;
acceptance falls 58.8 % -> 23.9 %. Break-even needs the verify below 2.25 emitted tokens times the
decode step; no chain length on prose reaches it. Chain-length form of the `docs/roadmap.md`
result: acceptance is the lever, chain length is not.

```
[PROV: commit=3cf2af24 date=2026-08-17 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=imp-server n=3 runs of 256 greedy tokens per
       arm, 2 alternating rounds
       cmd=`--set speculative.mtp_k=0|2|4|6 --set speculative.mtp_econ_min_emit=0
       --set runtime.max_seq_len=8192`, request `temperature 0, think_budget 0`
       rates and acceptance from /metrics deltas, card idle at ~0.9 GiB]
```

Pre-snapshot build (commit `a82dec8f`, before the mid-chunk recurrent snapshot of #1459):
`mtp_k=2` at 65.5 tok/s, and enabling MTP cost ~17 % even when the drafter never fired. The
snapshot work lifted `mtp_k=2` to the numbers above; the ~17 % figure has not been re-measured
since.
