<!--
layer: L1
audience: operators
verified: 2026-08-13
commit: 81ffa573
-->

# Limitations

What imp does not do, does badly, or does without a test behind it. This file
exists so a reader can decide against imp quickly. The top five are repeated in
the README on purpose.

Things that are absent *by decision* rather than by omission are in
[`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md), with the measurement that made the
decision.

## The five a new reader should weigh first

1. **One GPU, one chip.** `sm_120a` only: no multi-GPU, no tensor parallelism,
   and no other GPU vendor or generation. What consumer Blackwell has and lacks
   is in [`internals/ARCHITECTURE.md`](internals/ARCHITECTURE.md).
2. **One model resident at a time.** 32 GB fits one; serving a second means a
   swap, and the requesting call pays the load.
3. **No GPU in CI.** Every kernel-level correctness and performance check runs on
   the maintainer's machine before a push. If that matters to your risk model,
   it should count against imp.
4. **Decode measurements move several percent between sessions on this host.**
   Any single number you read anywhere, including ours, is one sample of a
   distribution. See [`PERF.md`](PERF.md).
5. **Single-author project.** No support rotation, no SLO, no security response
   process.

## Untested code paths (every 🟡 from `FEATURES.md`)

These have a code path and no gate. They may work; nothing proves it.

- **Llama-4** architecture: loads, no dedicated gate.
- **FP8 E5M2**: the type exists, nothing exercises it.

## Known-bad and known-limited behaviour

- **Calibrated KV-cache scales shipped in a checkpoint are not read.** Six local
  checkpoints carry `*.self_attn.{k_proj.k_scale,v_proj.v_scale}` (96 tensors on
  Qwen3-Coder-30B-A3B-FP4, 12 on NVIDIA-Nemotron-3.5-Lightning-30B, which has 6
  attention layers of 52); no consumer for them exists in the tree, and the FP8
  KV path derives its own. Whether adopting them changes output quality is
  unmeasured. They were invisible until #1497, counted among 270 false
  "unrecognised weight name" warnings for the MTP sidecar.

- **INT4 KV cache produces empty output on gpt-oss.** Its sink term is correct
  and is unit-tested against a sink-aware reference; 4 bits per value on a
  64-wide head is simply not enough. It falls back to FP16 rather than pretending.
- **Host-offloaded NVFP4 MoE experts are slow, and nothing gates them.** They run
  correctly (Qwen3-30B-A3B-NVFP4, all 48 MoE layers on host: 23.3 tok/s against
  384.0 resident), but the price is steep and the only checks that exercise the
  path are manual. The CPU lane covers the slot arithmetic, not the kernels.
  The remaining cost is that prefill goes through the serial per-expert
  fallback, one expert per GEMM. A placement the expert cache cannot hold at all
  is still refused at load rather than served wrong (#1403).

  [PROV: commit=8a7bd8c date=2026-08-13 hw=RTX5090 model=Qwen3-30B-A3B-NVFP4-Modelopt
         quant=NVFP4 cuda=13.3 path=nvfp4-moe-host-offload
         cmd=`imp-cli --max-tokens 220 --temperature 0 --set moe.force_host_experts=48`
         n=1 note=single greedy run per arm; resident arm is the same command
         without the --set. Cold cache, short prompt — not a benchmark figure]
- **Batched and solo decode are not bit-identical.** Joining a batch costs
  rounding, measured at 0.22 % of the logit range, with identical greedy argmax.
  A neighbour's *content* provably cannot reach another row. No flag makes the
  two bit-equal; pin batch composition if you need that.
- **MoE routing uses atomics**, so identical seeds can diverge.
- **Speculative decoding is not universally profitable.** On Nemotron-3.5 the MTP
  head accepts **39 %** of its drafts on the serving path, which agrees with the
  **41 %** the offline harness scores on the same three prompts — the two numbers
  are the same quantity and they now match. The **0-9 %** this entry used to quote
  was a defect, not a property of the head: on a Mamba2 hybrid a fully rejected
  verify committed an unwritten recurrent snapshot, so the model's own predictions
  became garbage and nothing could be accepted afterwards (fixed 2026-08-20;
  `executor_ssm_gdn.cu` now wires both halves of the slab, as the GDN path already
  did). **It still does not pay here**, for the honest reason: a verify chunk emits
  only ~1.41 tokens and costs more than that, so k=1 loses roughly half the decode
  rate with the economics guard disabled, and the shipped guard's verdict now sits
  on the break-even and flips between runs. Leave `speculative.mtp_k` at 0 on this
  model; the measured table is in [`roadmap.md`](roadmap.md). On Qwen3.8-27B-NVFP4 it does pay
  since `ea547a53` — `speculative.mtp_k=1` measured +21.3 % — but **only at k=1**:
  an extra chunk row still costs half a decode step, so k=3 buys 2 %. Numbers and
  the profile that localises that cost: [`roadmap.md`](roadmap.md).
- **MTP is released for one model class, and the class that is left out has a
  measured defect, not a missing feature.** `speculative.mtp_k` stays **0
  everywhere** — nothing below is on by default; the table says what a user opts
  into and what they get.

  | class | example | cached verify graph vs an eager forward of the same state | MTP |
  |---|---|---|---|
  | dense GDN hybrid | Qwen3.8-27B-NVFP4 | 1 of 1033 replays disagree (0.10 %) | **released**, `mtp_k=1` measured **+21.3 %** decode |
  | MoE + GDN hybrid | Qwen3.6-35B-A3B-NVFP4 | 2 of 1013 (0.20 %) | released, unmeasured for throughput |
  | MoE + Mamba2 hybrid (`nemotron_h`) | NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | **176 of 1318 (13.4 %)** | **not released** |

  The released row carries two accept rates and they are not the same
  measurement. **82.7 %** is the head's offline top-1 accept, teacher-forced over
  four prompt classes with the verify loop pinned off
  (`scripts/mtp_accuracy_bench.sh`: 89.0 / 75.6 / 81.9 / 84.3 % on factual /
  verbose-think / code / instruction, 127 scored positions each). **67.0 %** is
  what the verify chunk actually accepted on the serving path over 30 prompts
  (4299 of 6415 drafts, `/metrics`). The offline number asks whether the head
  would have been right; the serving number asks how often the chunk took it.
  The gap is the cost of drafting into a chunk rather than one step at a time.

  The defect that keeps the third row out: a cached verify-chunk graph replayed
  against a state it was not captured for does not reproduce what an eager
  forward of that state computes — logit deltas to 23.8, and roughly one
  generation in eight carries a visibly duplicated word. It is not the drafter
  and it is not the attention route; removing the MoE pass (`moe.skip`) drops the
  rate from 11.2 % to 0.8 %, so the MoE pass carries it — but MoE alone is not
  enough, since the second row is MoE and clean. The value the graph bakes stale
  is not yet identified. `tests/test_spec_capture_fidelity.cpp` gates the first
  two rows and fails on the third.

  The first two rows are not bit-exact either: capture picks its cuBLASLt
  algorithm once, so ~0.1-0.2 % of replays differ from eager on a healthy model.
  That is the floor the gate's 2 % threshold sits above.

- **Speculation is off for most real requests, by rules that are easy to
  miss.** It requires greedy sampling (`temperature: 0` or `top_k: 1`), so any
  request with a temperature gets none; and a think budget disables it inside
  the think block, which on a reasoning model is most of the answer. The server
  sets `think_budget` to 0.5 by default, so on such a model speculation never
  runs out of the box. Penalties are **not** a blocker at the default
  `repeat_last_n: 0`: the verify replicates them for the unbounded window.

  **Six further request features disable it outright**, in one condition
  (`src/runtime/engine_spec_ngram.cpp:295-297`): `logprobs`, `json_mode`,
  `json_schema`, `regex_pattern`, `grammar` and `tool_constraint_tools`. The
  verify chunk replicates no FSM masks, so a constrained request stays eager.
  In practice that means **every tool call and every structured output runs
  without speculation** — the agentic workload this engine targets, and the
  constrained-decoding surface it ships. A `logprobs` request is eager too,
  which also makes logprobs unusable as an instrument for observing what
  speculation does.
  Other engines have neither rule, because rejection sampling makes a verify
  distribution-preserving at any temperature and none of them force `</think>`.
- **The same request can produce different output on its second pass through
  one process.** Two things speculation keeps across requests carry it: the
  n-gram drafter's corpus, and the MTP head's own KV cache, which resumes over
  the longest common prefix of the previous history. Draft acceptance therefore
  depends on what the process has served before, and accepted drafts change
  which tokens are emitted. Observed here as three prompts run twice in one
  server, where the first differed on the second pass while two fresh processes
  running the same three prompts in the same order were byte-identical.
  A peer's conformance tier had been carrying this as an unexplained sporadic
  divergence. Not a defect on its own, but a golden-output test has to pin
  `speculative.ngram` and `speculative.mtp_k`, or it is testing the history too.

  **Only on the greedy path.** Speculation requires `temperature: 0` or
  `top_k: 1`, so a sampled request never drafts and none of this reaches it —
  measured as `imp_spec_drafted_total` not moving at all on a
  `temperature: 0.8` request with the drafter enabled. A sampled request that
  diverges between identical runs is a different mechanism: without
  `runtime.deterministic_gemm` the forward is not bit-reproducible, and at a
  temperature a last-bit difference in the logits moves the sampled token
  whenever the top candidates are close.
- **On a recurrent model, a sampled request can differ depending on what the
  server answered before it, and hybrid prefix caching is the carrier.** A GDN
  model's recurrent state is cumulative, so reusing KV blocks alone cannot skip
  prefill; imp therefore saves the recurrent state at block boundaries in a
  `RecurrentSnapshotStore` and restores it on a prefix hit, in which case the
  state slab is overwritten from the snapshot instead of being reset
  (`src/runtime/engine_sampling_stop.cpp:297`). The lookup keys on the KV prefix
  hash, so it only fires on genuinely identical tokens, but **every chat request
  begins with the same chat-template header**, so the first block matches across
  unrelated prompts. The restored state is then the one another prompt's prefill
  produced under different chunk boundaries, which is not what recomputing would
  give on a forward that is not bit-reproducible.

  Measured on Qwen3.8-27B-NVFP4, temperature 0.8, fixed seed, sequential
  requests, keyed on `reasoning_content` plus `content`: 12 identical requests
  give one distinct answer in a fresh process, and one distinct answer even with
  prefix caching on, but **three distinct answers when 12 requests of an
  unrelated prompt precede them in the same process**. Setting
  `server.recurrent_snapshot_mb=0` returns it to 1 of 12, twice, against 3 of 12
  three times at the 256 MiB default.

  **Characterised with `speculative.ngram=false`.** Every divergent cell above
  had speculation disabled; the same 12-then-12 structure with speculation at
  its default was stable across 24 requests. That is a weak negative against a
  3-in-12 effect, so it does not establish that the default is immune, only
  that the effect has not been observed there. It matters because the flag does
  not merely switch n-gram drafting: it also selects the burst bound of the
  on-device decode loop (`speculative.miss_burst` against
  `runtime.decode_burst`), so the two settings are different decode
  configurations even on a sampled request that never drafts.

  This is a mode to select, not a defect: the tokens really do match, so the
  reused state is legitimate for that prefix, and disabling the store also
  disables hybrid prefix caching. It is the recurrent path paying the
  reproducibility price prefix caching charges everywhere. Consequence for
  callers: a harness that pins a seed to make two runs comparable does not get
  that guarantee from the seed alone if both runs share a long-lived server with
  other traffic between them. Pin `server.recurrent_snapshot_mb=0`, or use a
  fresh process per arm.
- **MTP speculation truncates answers: 2 of 6 prompts end after ~40 tokens with
  a re-statement of the question (measured 2026-08-19).** This is the same
  divergence as the entry below, but it is not the harmless half of it — the
  answer is not "different but coherent", it is unusable:

  ```
  # Paged KV Cache in LLM Inference
  ## The Problem: Traditional KV Cache Wastes Memory
  In a large language model (LLM) inference engine, and why block size matters.
  ```

  (164 bytes, `finish_reason: "stop"`. The last clause is the tail of the prompt.)

  Qwen3.8-27B-NVFP4, `speculative.mtp_k=1`, `speculative.ngram=false`,
  `server.prefix_cache=false`, six prompts, `max_tokens: 400`:

  | arm | degenerate | lengths |
  |---|---|---|
  | `mtp_k=1` | **2 / 6** | 164 B, 286 B, and four of 1198-1793 B |
  | `mtp_k=0` (control) | **0 / 6** | 1146-1784 B |

  **It is deterministic, not noise.** With `runtime.deterministic_gemm=true`
  four fresh processes produce the same truncated answer byte for byte
  (**4 / 4**, identical sha). Without it — the default — three of four do
  (**3 / 4**); the pin does not remove the state, it stabilises it, so the one
  process in four that escapes is the accident, not the rule.

  **The mechanism, from `diagnostics.spec_trace`:** the final verify reads
  `p0=114 t0=12482 draft=[13] argmax=[13,248046]` — token 12482 is `" matters"`,
  13 is `"."`, and **248046 is `<|im_end|>`**. The bonus token that a verify
  emits after an accepted draft comes from the last chunk row, and that row
  predicts end-of-turn where the single-token decode path keeps writing. So the
  structural divergence below is not confined to *which* coherent answer you
  get; it reaches the stop decision.

  **A switch fixes it, and it costs the entire win.**
  `speculative.verify_nvfp4_gemm=false` routes the verify chunk off the batched
  NVFP4 GEMV: truncation drops to **0 of 6** on the same prompts, both offenders
  included, with acceptance unchanged (72-80 %). Throughput goes with it:

  | arm | tok/s (r1, r2) | vs k=0 |
  |---|---|---:|
  | k=0 | 88.97, 88.93 | — |
  | k=1, `verify_nvfp4_gemm=false` | 80.50, 77.43 | **-11.2 %** |
  | k=1, default (truncates) | see roadmap | +21.3 % |

  So **the +21.3 % comes from precisely the kernel that causes the truncation.**
  Both accumulate in `float` — the difference is summation order, not precision,
  and it is enough to flip the stop decision on a third of prompts. No setting
  of this pair is both fast and correct.

  **Re-measured and part-diagnosed 2026-08-21.** The count reproduces exactly
  on the current build (2 / 6, the 164-byte answer byte-for-byte), so the
  finding had not expired. What has changed is the explanation, and it was
  narrower than "summation order" in one place and wider in another.

  The verify chunk and the M=1 decode step take DIFFERENT kernels at four
  independent sites, all gated on the same `n == 1` condition:

  | site | how they differ | status |
  |---|---|---|
  | q/k/v, gate/up projections | K reduced in 32 partial sums (decode, `gemv_nvfp4_multirow_kernel<8>`, one warp per output row) against 128 (verify, `gemv_nvfp4_kpar_mb_fp16_kernel`, one block per output row) | **fixed**, `speculative.verify_row_parity` |
  | down projection GEMM | nothing: `n_mb = 1088 > 512` so both take the 128-wide path. Measured, not assumed | already equal |
  | SwiGLU into down | decode keeps `silu(gate)*up` in float registers, verify rounds it to FP16 in a separate kernel first | open |
  | RoPE + KV write, QK-norm | `can_fuse_rope_kv` and the fused QK-norm are both `n == 1` only (`executor_attention.cu:390,405`) | open |

  The inner loops are otherwise instruction-for-instruction identical - same
  `cvt.rn.f16x2.e2m1x2` dequant, same 16 fma pairs, same scale. So the first
  row of that table is a pure grouping difference, and closing it costs
  nothing: `gemv_nvfp4_multirow_mb_kernel` keeps the 32-lane warp partition AND
  still reads each weight micro-block once for all rows.

  | arm | tok/s (r1, r2) | vs k=0 | degenerate |
  |---|---|---|---:|
  | k=0 | 88.56, 88.47 | | 0 / 6 |
  | k=1, default | 104.46, 104.02 | +17.8 % | 2 / 6 |
  | k=1, `verify_row_parity=true` | 105.52, 105.07 | **+19.0 %** | **1 / 6** |

  So the sentence this entry used to end on - *"MTP is either fast and wrong,
  or correct and slower than no speculation at all"* - is **wrong as stated**.
  One of the two divergences is closable at no cost, and the parity arm is the
  fastest of the three. What is true is narrower: the remaining truncation
  needs the other two sites, and the SwiGLU one is expensive. A fused
  multi-row down projection was built and measured at **-27.7 %** against the
  batched path (it re-reads the 5120x17408 weight per row) **and it did not
  change the count** - it was removed rather than shipped.

  **2026-08-21, second pass: closing the remaining sites does NOT reduce the
  count, and that refutes the framing this entry rests on.** Sites 3 and 4 were
  both built and both proven live (a one-shot log line each, `n=3`), on one
  build, six prompts, two fresh processes per arm:

  | arm | degenerate | which prompts |
  |---|---|---|
  | parity off | 2 / 6 | 1, 4 |
  | site 1 (K-reduction width) | 1 / 6 | 1 |
  | sites 1 + 4 (QK-norm/RoPE fusion) | 1 / 6 | 1 |
  | sites 1 + 3 + 4 (+ SwiGLU fusion) | **2 / 6** | 1, **3** |

  Stable within an arm across fresh processes, different prompts between arms.
  So the divergences are not accumulating toward agreement: each perturbation
  reshuffles which prompts fall over. That is the signature of a **marginal stop
  decision** - the bonus token off the last chunk row is near-tied between
  continuing and `<|im_end|>` at those positions, and any numerical change moves
  a different subset across the line.

  Consequence for this entry: "closing this needs the chunk path to agree with
  decode numerically" is not established. Three of the four sites now agree and
  the count did not fall. Sites 3 and 4 are **not shipped** - they add kernels
  and complexity for no measured benefit, and site 3 makes it worse. Only site 1
  ships, and it ships because it is free and faster, not because 1 / 6 is a
  proven improvement over 2 / 6 on six prompts.

  One caveat that survives and points somewhere: routing the chunk off the
  overlay entirely (`verify_nvfp4_gemm=false`) reaches 0 / 6 including prompt 1,
  which no parity arm does. So something about the overlay path is not captured
  by per-kernel agreement with decode. The engine's `n == 1` path is
  pervasively fused - QKV, gate/up, o, down, QK-norm+RoPE each have their own
  `n == 1` branch - so a verify chunk is a different execution graph rather than
  the same graph at a different width. Reaching real parity is that
  architecture, not four kernels.

  **2026-08-21, third pass: the stop decision is NOT a knife edge. Measured,
  and it refutes the explanation the second pass offered.**

  The second pass concluded from the reshuffling that the bonus token must be
  near-tied between continuing and `<|im_end|>`. That was an explanation of a
  pattern, not a measurement, and it is wrong. `diagnostics.spec_trace` now
  reports the top-2 logit gap per chunk row; over 892 verify steps / 1784 chunk
  rows on the six prompts:

  | rows | n | p05 | median | p95 |
  |---|---|---|---|---|
  | all chunk rows | 1784 | 0.13 | **1.74** | 9.01 |
  | bonus rows (the last row of each chunk) | 892 | 0.12 | 1.71 | 8.89 |
  | rows whose argmax is `<|im_end|>` | 2 | | **1.79** | |

  ```
  top1=248046 (<|im_end|>)  top2=271  gap=1.9852
  top1=248046 (<|im_end|>)  top2=271  gap=1.5986
  ```

  Both EOS decisions sit **above** the median gap of an ordinary position, and
  **33.2 % of all rows (593 of 1784) are tighter than 1.0**. The verify chunk is
  not hesitating when it ends the turn; it is about as confident as usual.

  So the disagreement with the single-token path is **substantial, not
  marginal**: at that position the chunk's last row believes the turn is over by
  a normal margin while decode keeps writing. That moves the suspect off the
  LM-head numerics - three of the four divergence sites are closed and it did
  not help - and onto the hidden state feeding that row.

  **The rate, which prices any fix: 2 of 892 verify steps propose EOS at all,
  0.22 %.** A guard that re-runs just those positions through the ordinary
  single-token decode path and takes that verdict would fire on roughly one
  verify step in 450 and cost one decode step when it does.

  ```
  [PROV: commit=86479ce4 date=2026-08-21 hw=RTX5090 model=Qwen3.8-27B-NVFP4
         quant=NVFP4 cuda=13.3 path=imp-server n=1 process, 6 prompts,
         max_tokens=400, 892 verify steps
         cmd=`imp-server --think-budget 0 --set speculative.mtp_k=1
         --set speculative.ngram=false --set server.prefix_cache=false
         --set runtime.deterministic_gemm=true --set diagnostics.spec_trace=true`;
         gaps from the `gap=[id1>id2:x]` field this pass added to the verify
         trace, top-2 over the full vocab of each chunk row's logits. The arm
         reproduced 2/6 degenerate, so the traced run is the failing one.
         NOT measured: the same positions under no speculation - there is no
         verify chunk at mtp_k=0 and therefore no trace, so the comparison here
         is EOS rows against ordinary rows of the same run.]
  ```

  **Correction to the cost figure below.** The -27.7 % measured for a fused
  multi-row SwiGLU was a property of the implementation, not of the fix: the
  first version called the single-row helper once per activation row, i.e. MR
  full sweeps of the 5120x17408 down weight, and its own comment claimed the
  weight "cannot be hoisted without changing the accumulation". That is wrong -
  micro-block outer, rows inner gives each row the same fma sequence. The
  rewritten version hoists correctly. It still is not shipped, for the reason
  above: it does not help.

  Also corrected: `verify_nvfp4_gemm=false` does NOT make the verify chunk
  agree with decode. It routes the chunk to CUTLASS, a third kernel. It
  reaching 0 / 6 is a property of these six prompts, not of an established
  mechanism.

  **Consequence: `speculative.mtp_k` stays 0 by default**, despite the +21.3 %
  it measures on this model ([`roadmap.md`](roadmap.md)). A throughput win that
  truncates a third of answers is not a win, and the cheap escape does not
  exist: MTP is either fast and wrong, or correct and slower than no speculation
  at all. Closing this needs the chunk path to agree with decode numerically,
  which is a kernel change, not a flag.

  *Caveat this places on that +21.3 %:* the two arms do not generate the same
  text, and the speculative arm sometimes stops early, so the comparison is
  between workloads that differ. The token counts ran the other way (2100 at
  k=1 against 1847 at k=0), so it is not simply "shorter answers decode
  faster", but the number is not a like-for-like one either.

- **Speculative decoding does not reproduce the non-speculative greedy output
  on a GDN hybrid, and it cannot by construction.** Its contract is that it
  changes speed and not tokens; here it changes tokens. Measured on
  Qwen3.8-27B-NVFP4 with `runtime.deterministic_gemm=true`,
  `speculative.ngram=false` and `server.prefix_cache=false` on both arms, three
  prompts at 256 greedy tokens each, against a stable control (two
  no-speculation processes byte-identical, and that control has held across
  eleven processes): with `mtp_k=2`, **all three prompts** diverge from the
  no-speculation answer, first at bytes 79 / 332 / 243 (0-indexed). That is an
  early token flip, not a rounding tail. Both answers are coherent; they are
  different generations. Predates the 2026-08-17 verify work: the same prompts
  diverge with the older eager replay.

  **The cause is structural, not a kernel defect.** In a speculative arm every
  emitted token comes out of the multi-row verify chunk and none out of the
  single-token decode step: the chunk is built at
  `src/runtime/engine_spec_ngram.cpp:740,751` and the only emit site in that
  file is `:988`. Proven by an image whose `n == 1` decode path was broken badly
  enough to emit pure garbage, and whose speculative arm still produced output
  byte-identical to the stock speculative arm. Decode and the verify chunk also
  dispatch different kernels for the same weights, per shape: 10240x5120 and
  12288x5120 take `gemv_nvfp4_kpar` with its 32-lane `warp_k_loop` partition at
  decode and `gemm_nvfp4_batched` in the chunk, and the FFN shapes 17408x5120
  and 5120x17408 never appear on the decode side at all, because the
  `n == 1`-gated fused NVFP4 kernels serve them there
  (`src/exec/executor_ffn.cu:98,140`). So "speculation reproduces
  non-speculative decode" is not a property this design can deliver. It could
  only hold by kernel coincidence, and it does not.

  **The mechanism this entry used to state is withdrawn.** It read: the verify
  advances the recurrent state through the chunk kernels while plain decode
  advances it through the single-token path. `--set gdn.chunkwise_scan=false` is
  byte-inert on both arms and the divergence survives at the same offsets, so
  that is not it.

  **Five decode-side hypotheses are dead, each killed by a switch or by a patch
  that instrumentation proved live. Do not re-run them:**

  1. **GDN chunkwise scan** (`--set gdn.chunkwise_scan=false`): byte-inert on
     both arms.
  2. **Fused QK-norm + RoPE** (`--set attention.no_qknorm_fused=true`):
     byte-inert on both arms.
  3. **The NVFP4 `use_multirow` K-partition split**
     (`src/quant/nvfp4_gemv_dense.cu:387`, patched out): byte-inert on both
     arms. The dispatch log confirms shapes 10240x5120 and 12288x5120 moved from
     `multirow=1` to `multirow=0` at decode, so the instrument was live.
  4. **The fused NVFP4 FFN at decode** (two-site patch at
     `src/exec/executor_ffn.cu:98,140`): decode output byte-identical to stock
     across 2281 greedy tokens. The log confirms both FFN shapes moved onto
     `gemv_nvfp4_kpar`, and `gemv_nvfp4_gate_up_fused` never fired.
  5. **The attention kernel family** (mirror flip at
     `src/exec/executor_attention.cu:516`, putting the `n == 1` decode step on
     the chunk's FA2 path): it moves prompts 2 and 3 on the decode side and
     leaves prompt 1 byte-identical, and prompt 1 is exactly where the
     speculative divergence sits, at byte 79. With `--set
     speculative.capture=false` putting both paths on the same eager
     `FA2_FP16QK` branch, the divergence is still at byte 79.

  **All five are decode-side substitutions, and the direction matters.** This is
  not "kernel identity is irrelevant". A chunk-side substitution does move the
  generated text: `--set speculative.verify_nvfp4_gemm=false` moves the first
  difference to bytes 58 / 130 / 150 on the three prompts, and it still never
  reaches the non-speculative reference. The standing fact for the chunk side is
  the weaker one, that no chunk-side kernel choice tried so far closes the gap.

  **What makes the decode-side eliminations hard to argue with is that the
  invariance is at text level, not at offset level.** Across every decode-side
  substitution the non-speculative prompt-1 answer is one and the same 1282-byte
  text in 11 processes (md5 `35e7d9a93d14fe18a604a58fb6456388`), and the
  `mtp_k=2` prompt-1 answer is one and the same 470-byte text in 8 processes
  (md5 `c8a63ca87dc1ad95bc185b8c4e97d7b8`). The two differ from byte 79 on.
  Prompt 2 behaves the same way, so prompts 1 and 2 carry this record.
  **Prompt 3 does not**: it reads 243 in five pairs and 253 in two, and the 253
  cannot be separated from the cross-process instability in the entry below.
  Prompt 3 is evidence that all three prompts diverge, nothing more.

```
[PROV: commit=e3c48aa2 date=2026-08-18 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=imp-server n=3 prompts x 256 greedy tokens
       per arm, fresh process per arm, one arm per process
       cmd=`--set server.prefix_cache=false --set runtime.deterministic_gemm=true
       --set speculative.ngram=false --set speculative.mtp_k=0|2
       --set speculative.mtp_econ_min_emit=0`, request `temperature 0,
       think_budget 0`; divergence offsets from `cmp` over the saved answer
       bytes, kernel identity from the dispatch log, card idle]
```

- **A speculative arm is not byte-stable across processes at temperature 0,
  with identical flags. Fresh processes are usually byte-identical and
  sometimes not.** At `mtp_k=2`, eight of nine processes on identical flags
  agree byte for byte on all three prompts, and the ninth differs on prompt 3
  (737 against 733 bytes, first difference at byte 243) with its aggregate
  speculation counters identical at 476/371/238. At `mtp_k=1` it is coarser:
  two processes on identical flags produced 471 against 1213 bytes on prompt 1,
  with 326 drafts / 278 accepts against 420 / 345. The no-speculation arm is
  byte-stable across all eleven of its processes, prompt 1 the same 1282-byte
  text every time, so this is a property of the speculative path and not of the
  host. It is not localized to one prompt or one chain length.

  **It also settles a contradiction between this file and the commit history.**
  The second-pass entry above reports that "two fresh processes running the same
  three prompts in the same order were byte-identical" and scopes its finding to
  a second pass inside one process. That observation is true and it is not
  general: it is the eight-of-nine case, and two processes cannot see the ninth.
  The commit body of `ea547a53` (#1467) states the opposite in passing, as the
  reason a test had to pin a kernel instead of comparing generations, that
  speculative arms are not reproducible across processes. It carried no number
  and no mechanism, so it was an assumption. It is measured now, and the two
  statements are both about the same effect seen from different sample sizes.

  Consequence for callers: a harness that pins temperature 0 to make two runs
  comparable does not get that from the seed alone while speculation is on. Use
  `speculative.mtp_k=0` with `speculative.ngram=false` for an arm that has to
  reproduce. There is no setting that makes a speculative arm reproducible here,
  because the history inside a process moves it as well, see the entry above.

```
[PROV: commit=e3c48aa2 date=2026-08-18 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=imp-server n=3 prompts x 256 greedy tokens
       per process, 9 processes at mtp_k=2, 2 at mtp_k=1, 11 at mtp_k=0
       cmd=`--set server.prefix_cache=false --set runtime.deterministic_gemm=true
       --set speculative.ngram=false --set speculative.mtp_k=0|1|2
       --set speculative.mtp_econ_min_emit=0`, request `temperature 0,
       think_budget 0`; answer bytes compared with `cmp`, drafts and accepts
       from /metrics deltas, card idle]
```

- **The MTP head accepts 75.0 % of its first draft, and six explanations for
  the gap to published figures are dead.** Measured on Qwen3.8-27B-NVFP4 over a
  10-prompt mixed corpus (exposition, code, arithmetic, enumeration), n-gram
  disabled so the head is the only drafter, `mtp_k=1` so acceptance *is*
  first-position acceptance, two rounds with a fresh process each:
  **74.8 % and 75.2 %** (σ 1.2, >1200 drafts per cell), 84.7 and 85.8 tok/s
  against ~88 without speculation. At `mtp_k=2` the aggregate is 58.0-64.1 %.

```
[PROV: commit=37cd1543 date=2026-08-17 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=imp-server n=10 prompts x 256 greedy tokens
       per arm, 2 rounds, fresh process per arm, alternating
       cmd=`--set speculative.ngram=false --set speculative.mtp_k=1|2
       --set speculative.mtp_econ_min_emit=0 --set server.prefix_cache=false`,
       request `temperature 0, think_budget 0`; acceptance from /metrics
       deltas (imp_spec_accepted_total / imp_spec_drafted_total), card idle]
```

  Ruled out as causes of the gap, each by measurement, so they do not need
  running again:

  1. **Draft lm_head precision.** `speculative.mtp_nvfp4_head` true vs false:
     55.9/55.5 % against 51.6/56.6 %, fully overlapping. The NVFP4 head is ~8 %
     faster and costs no acceptance, so the default is right.
  2. **Quantised head weights.** The MTP tensors carry no scale companions in
     the checkpoint: they are BF16.
  3. **A missing `gamma = 1 + W` offset.** All seven MTP norms upload through
     the offset-applying path (`up_norm` in `weight_upload.cu`).
  4. **The hidden-state convention.** `diagnostics.mtp_prenorm_h` moves nothing
     (62.8 % against 62.8 %), which follows from `pre_fc_norm_hidden`
     normalising its input anyway.
  5. **A RoPE defect.** Disabling rotation looked like a +7.6-point win on one
     prompt set and reverses across prompt lengths: 72.7/55.6 at 112 tokens,
     66.5/68.3 at 607, 77.7/67.9 at 2767 (on/off). Rotation is fine.
  6. **An uninitialised MTP KV cache.** Zeroing it left the per-process spread
     intact (12.6 points before, 10.1 after) and moved first-position
     acceptance 73.2 → 75.0, which two samples per condition cannot separate
     from noise.

  **Two measurement errors of ours are in that list and are the reusable part.**
  (a) Findings 5 and 6 both survived a first pass because a run was *repeated*
  rather than *varied*: greedy decoding makes a run deterministic, so two
  identical runs agreeing to the tenth of a point is a statement about
  determinism, not about the effect. Vary the workload, not the repetition.
  (b) The per-process spread in (6) was read as a defect signal when the
  processes had generated **different text** — imp's forward is not
  reproducible across processes, and acceptance depends on what was generated.
  Spread across processes is only a defect signal once the output is identical.

  **The 87 % reference is not yet comparable.** It is a published figure for
  this architecture class from another engine, and the chain depth, batch size
  and acceptance definition (per-token or per-chain) behind it are unpinned.
  Pin the regime before treating the 12-point gap as a target.

```
[PROV: commit=37cd1543 date=2026-08-17 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=imp-server n=10 prompts x 256 greedy tokens
       per arm, 2 rounds, fresh process per arm, alternating
       cmd=`--set speculative.ngram=false --set speculative.mtp_k=1|2
       --set speculative.mtp_econ_min_emit=0 --set server.prefix_cache=false`,
       request `temperature 0, think_budget 0`; acceptance from /metrics
       deltas (imp_spec_accepted_total / imp_spec_drafted_total), card idle]
```
- **RESOLVED (2026-08-18). MTP does not lose on a GDN hybrid. imp's MTP path
  lost, and it no longer does.** The entry below stated a conclusion about the
  technique that its evidence never supported, and the dispute is settled by
  fixing the engine rather than by argument. Two defects, both in how work was
  launched rather than what it computed (`ea547a53`):

  1. `ssm_conv1d_prefill_f32_silu_kernel` had a grid over tokens, so a two-row
     verify chunk ran on two blocks of a 170-SM card while each block walked
     every channel serially.
  2. `executor_ssm_gdn.cu` built its `GemmContext` without `cur_spec_verify_`,
     which FFN and attention both pass. The `M<=4` batched-GEMV branch from
     #998/#1055 was therefore unreachable for **every GDN projection**, i.e. for
     48 of 64 layers on this model.

  | | before | after |
  |---|---:|---:|
  | no speculation | 84.47 tok/s | 86.21 tok/s |
  | MTP k=2 | 75.26 tok/s | **104.06 tok/s** |
  | kernel time per emitted token, k=2 | 11.35 ms | **8.93 ms** |
  | CUTLASS launches per verify | 296.3 | 8.9 |

  Speculation went from 10.9 % slower than not speculating to 20.7 % faster.
  The kernel figure reproduces across two independent runs on different corpora
  (8.93 and 8.93; the no-speculation arm reads 11.21 and 11.29).

```
[PROV: commit=ea547a53 date=2026-08-18 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=imp-server n=3 prompts x 256 greedy tokens,
       2 alternating rounds, fresh process per arm, nsys per arm
       cmd=`--set speculative.ngram=false --set speculative.mtp_k=0|2
       --set speculative.mtp_econ_min_emit=0 --set server.prefix_cache=false`;
       wall from usage.completion_tokens over request wall time, kernel from
       cuda_gpu_kern_sum, both from the SAME arm]
```

  **This is not a cross-engine comparison and must not be read as one.** vLLM
  0.27.1 was measured on this box only for MTP *acceptance*, where it reads
  59.7 % against imp's 58-64 %: parity, which is what retired the drafter as a
  suspect. Its throughput was not measured in a form worth trusting, so no
  claim about imp against vLLM speed exists in either direction.

  **What it cost to learn this**: six hypotheses about drafter accuracy, all
  dead, and a published 87 % acceptance target chased for days that turned out
  to describe an unpinned regime. Acceptance was never the problem. Detail in
  the entry above.

  **MTP stays off by default, and that is a decision rather than an omission.**
  `speculative.mtp_k` remains 0. Enable it with `--set speculative.mtp_k=2`.
  What you get and what you pay, all measured on Qwen3.8-27B-NVFP4 after the
  launch fixes:

  - **+15.2 % decode** on the shipped economics guard, two rounds: 109.48 and
    96.81 tok/s at k=2 against 89.54 and 89.50 without speculation. The two
    rounds differ by 13 % and ran 342 against 93 verifies on identical
    settings, so read this as a range of roughly +8 % to +22 %, not a number.
    The cause of that 3.7x swing in draft opportunity is not understood.

```
[PROV: commit=6c2c9445 date=2026-08-18 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=imp-server n=3 prompts x 256 greedy tokens,
       2 alternating rounds, fresh process per arm
       cmd=`--set speculative.ngram=false --set speculative.mtp_k=0|2
       --set server.prefix_cache=false`, NO mtp_econ_min_emit override so the
       shipped guard applies; tok/s from usage.completion_tokens over request
       wall time, verify counts from /metrics]
```

  - **0.79 GiB of VRAM** for the draft head on this model, paid whether or not
    it drafts (15 tensors, BF16 to FP16 on upload). The `~1.6 GiB` that
    `dispatch_policy.h` quotes is not this checkpoint: its head is a dense
    MLP, and a MoE head costs more.
  - **Output stops being reproducible across processes.** A golden-output test
    has to pin `speculative.mtp_k` or it is testing the drafter's luck; see the
    history-dependence entry above.
  - Measured on one checkpoint. Other MTP-carrying models are untested.

  **The engine says this at load.** A checkpoint that ships an MTP head while
  `speculative.mtp_k` is 0 gets one INFO line naming the flag, the measured
  gain and the two prices. It is a name-only probe across all three
  checkpoint layouts (sidecar file, shard index, single-file header), so it
  reads no weight and costs nothing on a load that does not want the head. It
  asks for the fusion projection that the loader's own dispatch keys on, not
  for any `mtp.*` name, so a group of MTP tensors imp could not actually load
  does not get advertised. Without it the option is invisible: the head is
  only loaded when `mtp_k > 0`, so an engine that has not loaded it also cannot
  mention it.

  **Not finished.** A marginal chunk row still costs 4.22 ms, 38 % of a full
  decode step (down from 8.09 ms, 71 %), on a batch-1 memory-bound decode where
  the weights are already streaming and it should be near-free. Remaining
  levers, measured, in size order:

  1. the per-row FMA work in `gemv_nvfp4_kpar_mb_fp16`, which scales by
     construction; an HMMA tile kernel is the honest answer if it is the floor.
  2. the capture-bucket floor of 3: a two-row chunk is padded to three and pays
     ~17 % of its GEMV time for a row that does not exist.
  3. the argmax + D2H + rollback host round-trip, which grows the non-kernel gap
     from 0.31 to 0.68 ms/token and eats 16 % of the kernel win. Smallest of the
     three, and the one with a named fix already in `roadmap.md`.

- ~~**MTP loses on a GDN hybrid at every chain length, and the guard is right to
  disable it.**~~ Superseded by the entry above; the measurements stand as a
  record of the broken build. Measured on Qwen3.8-27B-NVFP4, greedy, 256 tokens, thinking off,
  economics guard disabled so it runs throughout, two alternating rounds with a
  fresh process per arm:

  | `mtp_k` | tok/s (round 1 / 2) | emitted per verify | draft acceptance |
  |---|---|---|---|
  | 0 | 89.2 / 87.2 | | |
  | 2 | 83.3 / 80.1 | 2.22 / 2.25 | 58.8 / 54.3 % |
  | 4 | 64.0 / 61.9 | 2.56 / 2.46 | 36.7 / 34.5 % |
  | 6 | 53.8 / 52.9 | 2.55 / 2.44 | 24.6 / 23.9 % |

  A longer chain does not buy its way out. Emitted-per-verify saturates near
  2.5 from `mtp_k=4` on while the verify chunk grows linearly in k, so the
  fourth and later draft tokens are paid for on every verify and almost never
  accepted: acceptance falls from 58.8 % to 23.9 % across the sweep. Break-even
  needs the verify below 2.25 emitted tokens times the decode step, and no
  chain length on prose reaches it. This is the chain-length form of the same
  result recorded in `docs/roadmap.md`: acceptance is the lever, chain length
  is not.

```
[PROV: commit=3cf2af24 date=2026-08-17 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=imp-server n=3 runs of 256 greedy tokens per
       arm, 2 alternating rounds
       cmd=`--set speculative.mtp_k=0|2|4|6 --set speculative.mtp_econ_min_emit=0
       --set runtime.max_seq_len=8192`, request `temperature 0, think_budget 0`;
       rates and acceptance from /metrics deltas, card idle at ~0.9 GiB]
```

  An earlier measurement on the pre-snapshot build (commit `a82dec8f`, before
  the mid-chunk recurrent snapshot of #1459) put `mtp_k=2` at 65.5 tok/s and
  found that enabling MTP cost ~17 % even when the drafter never fired. The
  snapshot work lifted `mtp_k=2` to the numbers above; the ~17 % figure has not
  been re-measured since.
- **`imp-cli --prompt` prints only about 10 tokens.** Byte-level comparisons have
  to go through the server's JSON.
- **Prefill graph capture is disabled per model** when one NVFP4 weight exceeds
  the dequant-workspace cap. Decode graphs are unaffected.
- **No video input.** Images work; video needs a decoder this tree does not vendor.
- **No vision architecture registry.** The tower allowlist covers Qwen3-VL-shaped
  encoders and Gemma. InternVL and Pixtral each need a port.

## Model-specific blockers

- **Qwen3.5-27B MXFP4**: blocked on a checkpoint imp can decode, not on a bug.
  No MXFP4 SafeTensors decode path exists outside gpt-oss.
- **Gemma-4 on the FP8-KV quality gate**: its baseline perplexity on the gate
  corpus is broken, so it stays FP16 KV unless you opt in.
- **Qwen3.6-35B / Qwen3.5**: declare no FP8 KV hint, so they stay FP16 KV by
  default.

## Operational sharp edges

- **A successful `cudaMalloc` proves nothing about free VRAM on WSL2.** The
  driver oversubscribes into host memory and returns success; 28 GiB allocates
  with 22.6 GiB reported free. The tell is bandwidth, ~1530 vs ~237 GB/s, and the
  symptom is a 6.5x throughput cliff.
- **Free VRAM only ever decreases within a process** on WSL2/WDDM, however
  cleanly CUDA released it. Anything sized from `cudaMemGetInfo` reads a moving
  floor.
- **No `/health` field separates a server that started beside another process
  from a healthy one.** It does not fail: it gets a smaller KV pool and reports
  that pool as sitting at its own ceiling — 23 339 blocks on an idle card
  against 17 406 beside a neighbour holding 23.4 GiB, both answering `ok`. The
  plan and the install-time baseline both read `cudaMemGetInfo`, the same source
  as the defect, so neither can point at it. The one signal is the weight-upload
  delta against the checkpoint on disk, which warns at load; measurement and the
  two refuted fields are in [`MEMORY.md`](internals/MEMORY.md) B8.
- **A prefix-cache hit is not token-identical to a fresh prefill.** The hit skips
  the cached prefix, so it computes over a different chunk split and accumulates
  in a different order. Measured on Qwen3-4B-Q8_0 through the C API, at the first
  position where the two differ: fresh picks token 55486 (logit 38.242889) over
  279 (38.081467), a gap of 0.161; the cached run returns the same two in the
  other order (38.253368 against 38.188511). The shift the two paths put on one
  token is 0.172, larger than the gap between the candidates, so a near-tie
  flips. Answers stay coherent and agree for many tokens before they part, but
  greedy plus prefix caching is not bit-reproducible. `tests/test_prefix_cache_e2e.cpp`
  asserts a common prefix rather than equality for that reason.

- **`kv_cache.swa_snapshot_mb` set below one snapshot silently disables prefix
  caching** — worse than setting it to zero. It warns since #1092.
