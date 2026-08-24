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
- **Phi-4**: an alias onto the LLaMA path, no checkpoint of its own in a gate.
- **Qwen3.6-35B-A3B vision**: shares the Qwen3-VL tower; `make test-vision`
  runs gemma-3-4b-vl and Qwen3-VL-4B-Instruct, never a 35B-A3B checkpoint.
- **Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q8_K**: dequant paths with no gate reading a
  checkpoint in those formats. Q4_0, Q8_0, Q4_K, Q5_K and Q6_K do have one.
- **`/v1/rerank` against llama.cpp**: the cross-check is opt-in behind
  `COMPARE_URL=`, so the default `make test-rerank` does not run it.
- **`/admin/suspend`, `/admin/resume`** and **`server.model_swap`**: implemented,
  no gate exercises either.
- **The generation half of the HTTP contract**: SSE frame structure, usage
  accounting, `finish_reason` and tool-call streaming. CI's `Real API contract
  (model-less)` job deselects every test that produces a token, because
  generation needs a GPU and there is no GPU runner (#1600, #1559). They run in
  `make test-server` on a machine with a card, and in no CI job. The job prints
  the collected-here vs collected-total counts so the gap is visible in its own
  log rather than inferred from a job name.
- **The per-token cost of the server streaming path**: the perf gate benches
  `imp-cli`, which never enters the SSE writer, the tool-argument filter or the
  per-chunk JSON serialisation (#1685). A change to `tools/imp-server/` cannot
  regress the pinned numbers because the pinned numbers do not measure it.
  Adding `tools/` to the pre-push path filter would not help: it would run a
  benchmark of a different binary. Closing this needs a server-side benchmark
  harness, which does not exist.

All seven were green in `FEATURES.md` without a gate until #1680, which makes
them invisible here - the legend's whole point.

## Gates that do not exist

These are absent instruments, not untested features: nothing in the tree
produces the number, so no threshold can be set on it.

- **No correctness gate against a reference implementation** (#1571). There is
  no KL divergence against an fp16/bf16 forward, no perplexity-drift baseline
  and no tool-schema conformance rate. `scripts/validate_safetensors.py:11-14`
  lists the phases it cannot run, and why: no BF16 checkpoint is on disk, and
  imp consumes pre-quantised weights. `make test-niah` exists
  (`Makefile:315`) and no workflow invokes it, because like every target with
  `check-gpu` among its prerequisites it needs a card. Quantisation quality is
  therefore judged by the degeneration smoke prompts and by hand, not against a
  reference with a threshold.

- **No soak or endurance test** (#1642). The largest request count any test
  drives is 10 concurrent requests (`tests/api/test_concurrency.py:37`). Three
  shipped comments describe what a soak would assert and no soak exists to
  assert it: `tools/imp-server/metrics_memory.cpp:56`,
  `tests/test_memory_backend.cpp:223` and `src/memory/alloc_interpose.cpp:129`,
  the last of which describes an instrument meant "to be run once under a soak
  and read afterwards". A leak, a fragmenting KV pool or a slow handle
  exhaustion surfaces in production rather than in a gate.

Both need a GPU runner or a long-running machine with a card, and CI has
neither.

## Known-bad and known-limited behaviour

- **`server.green_contexts=true` does not give you green contexts on this
  chip.** `cudaDevResourceGenerateDesc` fails for the decode partition
  (`one or more resources passed in are not valid resource types for the
  operation`), the manager falls back to ordinary priority streams with
  distinct memSyncDomains, and `has_green_contexts()` is false from then on.
  Measured 2026-08-22 on the RTX 5090 at 170 SMs, both at the 80/20 split and
  at the 99/1 retry. Consequence worth knowing: the dynamic SM reconfiguration
  in `step_schedule()` is gated on that flag, so it never runs here - the
  reconfigure race #1656 describes is real code but unreachable on sm_120.

- **Remote `image_url` fetching is off by default, and when on it is still
  vulnerable to DNS rebinding.** `--allow-remote-images` classifies the
  destination before connecting, but the check and the connection are two
  separate resolutions, so a name whose records change in between still reaches
  a private address. Closing that needs a connect-time callback, which httplib
  does not expose. The off-by-default is what carries this; treat the flag as
  "the network this server sits on is trusted".

- **The VRAM planner's weight-cache reserve is an estimate with a floor, not a
  measurement, and there is no retry if it is wrong.** A start that overcommits
  still ends in `imp_context_create` aborting rather than degrading to a smaller
  KV pool. #1631 fixed the case that made `imp-server` unstartable at defaults
  on `Qwen3-8B-Q8_0` by raising the reserve to the planner's own projection plus
  the reserve floor, and the margin matters: the projection alone plans 9977 KV
  blocks and still OOMs, while the arm that works plans 7079. That is a 500 MiB
  edge, so a model whose demand sits inside it can still fail to start. The
  robust form is a retry at a smaller pool; it is not implemented.

- **The measured library reserve is only remembered if the cache path outlives
  the process.** `vram.library_reserve_cache` defaults inside the container, so
  a `docker run --rm` server re-measures every start and plans with the 3900 MiB
  constant, which is wrong in both directions (measured: 0 MiB on Qwen3-4B
  IQ4_NL, 7460 MiB on Qwen3-8B-Q8_0). Mount that path to keep it.

- **JSON Schema: assertion keywords imp cannot enforce are a `400`, not a
  weaker grammar.** `minimum`, `maximum`, `exclusiveMinimum`,
  `exclusiveMaximum`, `multipleOf`, `allOf`, `not`, `uniqueItems`,
  `patternProperties`, `propertyNames`, `prefixItems`, `contains`,
  `minContains`, `maxContains`, `minProperties`, `maxProperties`,
  `dependentRequired`, `dependentSchemas`, `if`/`then`/`else` are rejected at
  admission (#1567). They used to be accepted and dropped, so a request that
  bounded its output was answered by an unbounded grammar at HTTP 200. Pure
  annotations (`format`, `title`, `description`, `examples`, `default`,
  `$schema`) are still ignored, which is what Draft 2020-12 says they do.
  `const` is enforced, as a one-member enum.

- **JSON Schema: `enum` and `const` members must be strings.** The FSM emits an
  enum as quoted string content (`schema_constrain.cu:790`), so `{"enum":[1,2]}`
  has no representation and is a `400`. Before #1564 it constrained the model to
  the empty string instead.

- **JSON Schema: `additionalProperties` as a schema object is not enforced, it
  reads as `true`.** The boolean form is enforced. The object form (Pydantic
  emits it for `Dict[str, T]`) is parsed and its constraint on extra keys is
  dropped, which is weaker than asked for but not wrong. Before #1564 it
  truncated the schema at that key: everything after it, `properties` included,
  was silently discarded and the request downgraded to `json_object`.

- **A `pattern` the regex engine cannot compile is not enforced, and the request
  still returns 200.** `compile_patterns()` warns and leaves the node
  unconstrained (`json_schema.cpp:558`), unlike a top-level `regex` constraint,
  which is refused at admission. Same class as the keyword case above, on the
  path that has no admission screen in front of it.

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
  into and what they get. On a checkpoint that ships a head, `GET /health`
  reports `mtp_head_available` with the trade, so an operator can see the
  switched-off gain without reading the startup log (#1537). The default itself
  is unchanged: the head costs VRAM (0.79 GiB on Qwen3.8-27B-NVFP4) and turning
  it on for everyone is a decision, not a fix.

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

  **2026-08-21, third pass: the truncation is not localised at the truncating
  position, and the requested hidden-state diff is not obtainable.** The
  question this pass set out to answer was where the chunk row's hidden state
  parts company with the decode step's at the position that truncates. Two
  independent measurements say that question has no answer on this build, and a
  third says it would not have been the right question anyway.

  *The position does not exist in the non-speculative arm.* On prompt 1 the k=1
  and k=0 answers share only their first 49 bytes of 165 and 2358. The
  non-speculative path never visits the truncating position, so "the decode
  step's hidden state at the same position" is not a thing that was computed.

  *The instrument cannot see the decode loop.* `diagnostics.dump_hidden_dir` is
  host-side, and decode is CUDA-graph replayed, so the dump only covers passes
  that re-enter host code. Measured on one build, same prompt, two token
  budgets:

  | max_tokens | tokens generated | distinct dump steps |
  |---|---|---|
  | 40 | 40 | 5 |
  | 200 | 200 | **5** |

  Constant in the generation length. Everything past the capture passes is
  invisible to it. Turning capture off makes every step visible but changes the
  arm being measured: with `speculative.capture=false` the same six prompts gave
  2 / 6 and then 1 / 6 across two fresh processes, where capture-on is stable at
  2 / 6 both times. The instrument that could see the position destabilises the
  phenomenon it would measure.

  *And the divergence point carries no signal.* Same build, six prompts, k=1
  against k=0, first differing byte:

  | prompt | k=1 bytes | k=0 bytes | first divergence | |
  |---|---|---|---|---|
  | 1 | **165** | 2358 | 49 | truncated |
  | 2 | 2293 | 2223 | 271 | ok |
  | 3 | 2157 | 2583 | 103 | ok |
  | 4 | 2572 | 2574 | 135 | ok |
  | 5 | 2687 | 2666 | 48 | ok |
  | 6 | 2634 | 2701 | 49 | ok |

  All six diverge early, between byte 48 and byte 271, and the two that diverge
  earliest after the truncating one (48 and 49) produce full, clean answers.
  Early divergence is the norm here, not the symptom. Speculation not
  reproducing greedy output is already documented above; what this adds is that
  truncation is **one outcome among six of that ordinary divergence**, not a
  separate upstream event with a location to find. That is why the four
  per-kernel parity sites reshuffle which prompts fall over instead of reducing
  the count: they perturb a trajectory that has already left the greedy path on
  every prompt.

  *Correction to a claim made during the second pass:* `dump_hidden_dir` was
  said to destroy the phenomenon. It does not. A run with it set still truncates
  at exactly 164 bytes with `finish_reason=stop`, and still captures graphs. The
  accurate statement is the one above: it cannot see the steps that matter.

  [PROV: commit=2a049185 date=2026-08-21 hw=RTX5090 model=Qwen3.8-27B-NVFP4
   cmd=`--set speculative.mtp_k={0,1} --set speculative.ngram=false --set server.prefix_cache=false --set runtime.deterministic_gemm=true --think-budget 0`
   note=six prompts, max_tokens 600, temperature 0, top_k 1; dump-step counts from `diagnostics.dump_hidden_dir`]
  **2026-08-21, fourth pass: a stop-decision guard was built, measured and
  removed. The ordinary decode path agrees that the turn is over.** The one
  approach the three earlier passes had not tried was to stop trusting the chunk
  row's stop decision at all: when a verify chunk row's argmax is a stop token,
  drop that row and hand the position to the ordinary single-row decode path
  instead. This does not ask what a non-speculative run would have done, since
  that trajectory no longer exists; it asks whether the non-chunk path, from
  this same state, also ends the turn.

  It does. One build, six prompts, two fresh processes per arm:

  | arm | degenerate | guard fires |
  |---|---|---|
  | guard on, parity off | 2 / 6, 2 / 6 | 8 in 898 verify steps (0.89 %) |
  | guard on, parity on | 1 / 6, 1 / 6 | 4 in 1126 verify steps (0.36 %) |

  Identical to the same arms without it. The decisive line is in the per-request
  counters rather than the totals: on the truncating prompt the guard fired
  **twice in that request's 24 verify steps**, so the position really was handed
  to the decode path, and the answer still ended at exactly 164 bytes. The
  confident stop is a property of the state, not of the projection that reads it.

  Cost, paired alternating arms, two rounds: k=0 88.36 / 88.51, parity 106.74 /
  105.65, parity + guard 105.50 / 104.34. The guard is **+18.6 % over k=0** and
  1.2 % below parity alone, so it was affordable. It was removed anyway, on the
  same standard that removed sites 3 and 4 above: it does not change the count.

  Its fire rate also disagrees with the 0.22 % predicted from the EOS-proposal
  trace in the second pass, by a factor of 1.6 to 4. The trace counted proposals
  it could see; the guard counts every chunk row whose argmax is a stop token
  outside a think block. The larger number is the real one.

  **This closes the line.** Four independent approaches have now been measured
  against this truncation: per-kernel parity at four sites, routing the chunk off
  the overlay, a fused multi-row SwiGLU, and this guard. None reaches 0 / 6, and
  the third pass above explains why none of them could: the truncation is one
  outcome of a divergence that has already happened on every prompt by byte 271.

  [PROV: commit=fa21f28e date=2026-08-21 hw=RTX5090 model=Qwen3.8-27B-NVFP4
   cmd=`tools/analysis/mtp_truncation_check.sh 1` with `MTP_EXTRA_SET` per arm; guard built locally, not in the tree
   note=six prompts, max_tokens 400, temperature 0, top_k 1; throughput from three 400-token requests per arm, arms alternated]
  **Consequence: `speculative.mtp_k` stays 0 by default**, despite the +18.6 to
  +21.3 % it measures on this model ([`roadmap.md`](roadmap.md)). A throughput
  win that truncates one answer in six is not a win.

  The sentence this paragraph used to end on - *"closing this needs the chunk
  path to agree with decode numerically, which is a kernel change, not a flag"* -
  is **refuted**. Three of the four kernels now agree and the count did not fall;
  the fourth pass handed the stop decision to the decode path itself and the
  count did not fall either. Numerical agreement is not what stands between MTP
  and a shippable default. What stands there is that speculation puts this model
  on a different trajectory from byte 48 onward on every prompt, and one
  trajectory in six ends early.

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
- **Quantised KV is a default only for QWEN35, and it is a trade, not a
  freebie.** `kv_cache.dtype=auto` now resolves to NVFP4 for that family
  (Qwen3.8-27B and its Qwen3.5 siblings): measured **+0.29..0.35 %** perplexity
  on Qwen3.8-27B-NVFP4 and **+0.15..0.18 %** on Qwen3.5-4B mxfp4, alternating
  arms, in exchange for `max_model_len` going 48 512 -> **131 072** tokens on a
  32 GB card. Every other family keeps its previous default, deliberately: the
  MoE GDN siblings (QWEN36_MOE, QWEN35_MOE) are excluded because FP8 KV already
  costs them +1.47 % PPL — NVFP4 attention weights compound with a quantised KV
  — and NVFP4 KV is the more aggressive quantiser, unmeasured there. Opt out
  with `kv_cache.dtype=fp16`; opt in elsewhere with `=nvfp4`.
  [PROV: commit=982cd43 date=2026-08-24 hw=RTX5090 model=Qwen3.8-27B-NVFP4,Qwen3.5-4B-mxfp4 quant=NVFP4
   cuda=13.3 path=server-prefill
   cmd="imp-cli --perplexity ppl_corpus_45k.txt --set kv_cache.dtype={fp16,nvfp4}, alternating arms"
   n=3]
- **GDN-hybrid concurrency was fixed on 2026-08-24; the numbers below are the
  OLD behaviour, kept because the fix is one release old.** Until then a GDN
  decode step served ONE sequence and concurrent ones were time-multiplexed, so
  32 streams delivered what one delivered (81.5 tok/s aggregate on
  Qwen3.8-27B-NVFP4 against 1427 for a dense model). `runtime.gdn_batched_decode`
  (default on) now batches them: **474.9 tok/s aggregate at 32-way, 5.8x**, with
  no cross-sequence contamination and single-stream decode unchanged. Set it
  false to get the old rotation back.
  [PROV: commit=ce77a94 date=2026-08-24 hw=RTX5090 model=Qwen3.8-27B-NVFP4 quant=NVFP4
   cuda=13.3 path=server-batched-decode
   cmd="imp-server --set runtime.max_batch_size=32 --set runtime.max_seq_len=4096; 32 concurrent POST /v1/completions max_tokens=200"
   n=1]
- **The Qwen3.8-27B FP8 release is a quantization SOURCE, not a servable
  checkpoint.** `imp-quantize` reads its per-layer `layers-N.safetensors` layout,
  but nothing serves it: its weights are 25.87 GiB against roughly 27 GiB of
  usable VRAM before the KV pool and the recurrent state, and `sm_120` has no FP8
  GEMM, so even if it fit there would be no fast path. Quantize it to NVFP4 and
  serve that. Dequantizing it to FP16 at load is not an option either — that is
  the 51.75 GiB BF16 footprint.

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
