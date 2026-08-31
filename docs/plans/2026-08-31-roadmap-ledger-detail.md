# Roadmap ledger detail (moved out of docs/roadmap.md on 2026-08-31)

Record. The sections below were the investigation narrative behind roadmap
verdicts; `docs/roadmap.md` keeps the verdict, the number and a link here.
Text is verbatim as of `docs/roadmap.md` at commit 6afcf35f; `path:line`
citations inside are as they stood on that day and are not gated in this
file.

### The MTP verify on a GDN hybrid, decomposed (2026-08-17 .. 08-19)

Trigger: an RTX 5060 Ti report of +85% from MTP-2 on Qwen3.8-27B while imp
measured a loss. Not the comparison it looks like: his 47.4 tok/s is WITH MTP
vs baseline 25.7; both engines sit near their rooflines (84% and 93%) and the
4x bandwidth difference means the same absolute draft overhead amortises
against a 39 ms step, not an 11.8 ms one.

Per-verify cost, k=2, guard off, 300 verifies:

| phase | before #1455 | after | note |
|---|---:|---:|---|
| chunk forward (captured, 3 rows) | 17.8 ms | 17.8 ms | vs an 11.8 ms decode step |
| replay of the accepted prefix | 25.1 ms | 17.2 ms | fires in ~37-48% of verifies |
| argmax + D2H + rollback | 16.4 ms | 10.7 ms | includes the replay |
| decode | 64.3 tok/s | 84.4 tok/s | 83.7 without MTP |

Workload sensitivity (same server, defaults, only predictability differs):

| workload | tok/s | draft acceptance | emitted/verify |
|---|---:|---:|---:|
| ordinary prose, no speculation | 89.1 | - | - |
| ordinary prose, n-gram default | 89.4 | 11.2% | - |
| ordinary prose, MTP k=2 | 87.9 | 58% | 2.3 |
| **repeat a text verbatim** | **876.5** | **98.3%** | **36.6** |

**RESOLVED 2026-08-18 by fixing the engine:** two launch defects (a grid over
tokens; a `GemmContext` built without `cur_spec_verify_`) kept every GDN
projection off the small-M batched GEMV. Fixed in `ea547a53`: MTP k=2 104.06
vs 86.21 spec-off (was 75.26 vs 84.47); kernel ms/emitted-token 11.35 ->
8.93. Every "speculation does not pay" statement below described the broken
build. Not a cross-engine claim: vLLM was measured for acceptance only
(59.7% vs imp 58-64% - parity, so the drafter and the published 87% figure
describe a regime nobody pinned).

Retractions, kept so they are not re-derived:

- ~~"Spend on drafters, not the repair path"~~ retracted 2026-08-18: at k=1
  (75.0% accept, 1.748 emitted/verify) MTP still lost on the old build;
  pricing the two levers: acceptance 75 -> 87% buys +3.2%, verify 20.6 ->
  13.6 ms buys +45.7%. Verify price was the lever, backwards from the k=2
  reading.
- ~~"92% more launches per emitted token"~~ withdrawn 2026-08-18: the
  normalisation divided by the REQUESTED 16384 tokens off a `0.00 ms` bench
  line; real emission was ~1e3 tokens. Absolute kernel times stand.
- ~~"0.72 forwards/token so speculation moves fewer bytes"~~ and the
  launch-count framing generally (incl. a 22.3% CUTLASS share from a broken
  CSV parse - nsys kernel names contain commas): launches fell, GPU time did
  not; counting launches never measured cost.
- Accounting error under all of it: **a verify replaces a decode step only
  when accepted; on rejection it is additional** - a full weight sweep per
  verify regardless of emission. That is why chain-length saturation is a
  consequence, not a separate result.

**k-sweep on the fixed build (2026-08-19):**

| k | chunk rows | tok/s (r1, r2) | mean | vs k=0 | accept | emitted/verify | ms/verify |
|---|---:|---|---:|---:|---:|---:|---:|
| 0 | 1 | 86.04, 86.02 | 86.03 | - | - | - | - |
| **1** | 2 | 104.44, 104.19 | **104.31** | **+21.3%** | 76.0% | 1.76 | 16.89 |
| 2 | 3 | 103.86, 97.78 | 100.82 | +17.2% | 59.9% | 2.20 | 21.82 |
| 3 | 4 | 88.08, 87.34 | 87.71 | +2.0% | 49.9% | 2.50 | 28.52 |

k=0 reproduces to 0.02% across rounds. Blocker at the time was correctness
(2/6 prompts ended after ~40 tokens re-stating the question,
deterministically; detail in [`LIMITATIONS.md`](../LIMITATIONS.md)) - and the
control that isolated the head (`ngram=false`) is also what HID that defect;
`deterministic_gemm` pins a degenerate answer stable, not absent.

Slope attribution (nsys k=1 vs k=3, `--cuda-graph-trace=node` mandatory):

| kernel | share of k=3-k=1 growth | per launch k=1 -> k=3 |
|---|---:|---|
| `gemv_nvfp4_kpar_mb_fp16` (batched verify GEMM) | **65.1%** | 26.95 -> 32.87 us, **+22.0%** |
| `gemv_fp16` | 11.5% | 64.02 -> 66.23 us, +3.4% |
| `gemv_nvfp4_multirow_fp32` | 8.4% | 440.10 -> 442.90 us, +0.6% |
| `gdn_scan_fused` (recurrent state) | **2.3%** | 8.72 -> 10.03 us, +15.0% |

The slope lives in the forward (the batched GEMV's imperfect amortisation),
not the recurrent state. Fit: `4.96 ms + 5.82 ms x rows` (was 5.36 + 6.53) -
an extra row still costs 50% of an 11.62 ms decode step, so chain length is
not a lever (k=3 buys 2%). Acceptance did not move (76.0 vs 75.0%). Per
launch reported on purpose: cross-process greedy is not deterministic (700
vs 92 tokens on the same prompt), per-token normalisation across profiled
processes would report that.

```
[PROV: commit=3c3e9ac9 date=2026-08-19 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=imp-server n=3 prompts x 2 alternating rounds
       cmd=`imp-server --think-budget 0 --set speculative.ngram=false
       --set speculative.mtp_k=0|1|2|3 --set speculative.mtp_econ_min_emit=0
       --set server.prefix_cache=false`; throughput on the `make build` image,
       tokens from usage.completion_tokens, verifies from /metrics. nsys arms on
       the dev build (nsys ships only in imp:toolchain) — sound because both arms
       are the same binary and the claim is relative, not an absolute rate]
```

**Four-arm table (2026-08-18, tokens counted from API responses):**

| arm | rows | tokens | verifies | kernel ms/token | emitted/verify | cost/verify |
|---|---:|---:|---:|---:|---:|---:|
| k=0 | 1 | 723 | 0 | 11.39 | n/a | 11.39 ms |
| k=1 | 2 | 729 | 424 | 11.33 | 1.715 | 19.43 ms |
| k=2 | 3 | 721 | 334 | 11.35 | 2.153 | 24.44 ms |
| k=3 | 4 | 768 | 291 | 11.98 | 2.629 | 31.50 ms |

Speculation buys no GPU time (kernel ms/token flat k=0..2); cost/verify is
linear in rows (`5.36 + 6.53 x rows`, residuals -0.50/+1.01/-0.52/+0.01) -
an extra row costs 57% of a decode step where bandwidth-bound it should be
near-free. At k=1 the repair path is structurally unreachable, so the second
row alone costs 8.04 ms.

```
[PROV: commit=196a3384 date=2026-08-18 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=imp-server n=3 prompts x 256 greedy tokens per
       arm cmd=`--set speculative.ngram=false --set speculative.mtp_k=0|1|2|3
       --set speculative.mtp_econ_min_emit=0 --set server.prefix_cache=false`,
       nsys -t cuda --cuda-graph-trace=node; kernel time = cuda_gpu_kern_sum
       total, tokens from usage.completion_tokens, verifies from /metrics]
```

Buried, so they are not re-run:

- Six drafter-accuracy hypotheses (draft lm_head precision, quantised head,
  gamma=1+W offset, hidden-state convention, RoPE, uninitialised MTP KV):
  all measured dead ([`LIMITATIONS.md`](../LIMITATIONS.md)).
- MoE draft head: this checkpoint's head has no experts/router.
- Unfused verify chunk: per-launch flat at 32.5 us across 2-3 rows.
- Repair forward as main cost: unreachable at k=1, ~21% of verifies at k=2.
- Async conditional-graph loop (MTP switches it off): worth 1.0% measured,
  not the assumed 27-45%.
- Recurrent-state mechanism for the greedy divergence: disproven
  (`gdn.chunkwise_scan=false` byte-inert, divergence survives).
- Five decode-side kernel hypotheses (chunkwise scan, fused QK-norm+RoPE,
  multirow split, fused FFN, attention family): all null, each instrument
  proven live first; structurally impossible anyway - every emitted token in
  a spec arm comes from the verify chunk, no decode kernel is on the path.
- Chunk-side kernel choice: `verify_nvfp4_gemm=false` moves the divergence
  offsets (58/130/150 vs 79/332/243), never closes it.
- Correction: ALL three prompts diverge with mtp_k=2 (not two of three);
  control holds across eleven spec-off processes.
- Cross-process spec reproducibility at temp 0: does not hold (8/9 processes
  agree at k=2, the two k=1 processes disagree with each other) - why #1457
  and #1467 both "saw" contradictory things. Documented in
  [`LIMITATIONS.md`](../LIMITATIONS.md).
- Economics guard constant: right guard, coarse constant - 4.0 was derived
  on the eager verify; measured break-even 2.42 (now the k-aware default,
  see `speculative.mtp_econ_min_emit`).

### MoE host offload - from "CPU cold experts" to a shipped LRU path

Origin: compute cold experts on the CPU (ktransformers shape) to reach
80B-120B on 32 GB. The campaign measured its way OUT of that design - the
LRU expert cache + streaming won, no AVX kernels, no `GOAL.md` amendment.
All numbers Q4_K_M GGUF unless marked NVFP4.

Budget measurements (2026-08-10 .. 08-11):

| question | answer | numbers |
|---|---|---|
| host streaming bandwidth | 62.5 GB/s | 16 threads, 24 GiB, NT loads; saturates at 4 threads; ~65% of DDR5-6000 dual-channel |
| routing skew | real, grows with expert count | coverage at 40% resident: Qwen3-30B-A3B 84.7% (2.12x flat), gpt-oss-20b 71.6% (1.79x) |
| does a static hot set transfer? | NO - prompt-dependent | cross-validated: -15.2 / -29.5 points vs each prompt's oracle (78.8/58.7 vs 94.0/88.1%) |
| 8 KiB round-trip cost | 86.2 us D2H+H2D, size-independent (latency not bandwidth) | single D2H+sync 45.4, +kernel between 126.5, bare launch 34.6; the old 165 us figure was a host-CALL cost, 3.8x overstated. Trap: pageable H2D returns before arrival - only synchronized numbers compare (pinned times "slower", 45 vs 22 us, for that reason) |
| calibration curve (9 prompts, 300 splits) | first prompt buys almost all | held-out coverage: flat 40.0% (38 tok/s ceiling), 1 prompt 67.6% (63), 3 -> 70.3%, 8 -> 72.6% (71), oracle 92.1% (142) |
| temporal locality | strong - LRU needs no calibration | median reuse distance 2 tokens; 45% repeat next token, 80% within 8; steady-state hit 94.7/90.0/95.3% at 40% residency = 0.38-0.80 new experts/layer/token; warm in ~64 tokens |
| host-compute vs stream-into-VRAM (120B-A5B shape) | streaming wins ~3x | static split + host compute: 620 MB @62.5 GB/s + round trips = 14.0 ms/token; LRU streaming: 120-228 MB @25.6 GB/s = 4.7-8.9 ms; PCIe reaches 50.6 GB/s at 64 MiB batches |
| can a promotion be hidden? | yes IF prefetched a layer ahead - which is unreachable | overlap bench: prefetched +0.3-0.7 ms vs +10.5-10.7 ms in-front; routing for layer N known only at layer N; recency prediction carries 42-47% (4-token window 63-68%) and fails exactly on misses |
| 120B ceiling band | 52 (held-out) .. 103 (matched) tok/s; plan against 63-71 | 0.35-1.37 GB cold/token + 4.1 ms transfers; idle-GPU numbers |

Ledger of the shipped path (Qwen3-30B-A3B, all 48 MoE layers host-resident
unless noted):

| step | verdict | numbers |
|---|---|---|
| `ExpertCache` at full offload | holds its hit rate | 88.7% hit, ~0.9 new experts/layer/token (traces predicted 0.38-0.80); 24.98 tok/s vs 6.63 staging-only vs 311.24 resident |
| offload penalty decomposition | 19x = 2.47x graphs x 7.8x rest | resident graphs-ON 286.94 / graphs-OFF 116.13 / offload ~14.9 tok/s; launches 18 052 (graph-hidden) / 52 024 / 197 809 |
| ~19 ms/token host issue cost | the bound half | 3091 `cudaLaunchKernel`/token (4.58 us median) + 632 memcpy/token; dequant_q4k 47.1% of GPU time |
| fused MMVQ instead of dequant->GEMV | REFUTED cleanly | expert-path kernel time -43% (504.8 -> 288.7 ms), e2e 0% - launch count unchanged; the path is host-bound, only launch count is currency |
| slot-indexed fused MoE kernels (#1370) | SHIPPED, 2.1x | the LRU pool already IS the contiguous tensor the resident kernels index (`expert_cache.h` slot_ptr; `d_lookup_` holds slot indices); decode 22.9 -> 48.3 tok/s, launches 197 809 -> 61 585 (within 18% of resident) |
| correctness evidence for #1370 | perplexity could not see it (E6: teacher-forced = prefill, gated path is n==1) | supported instead by: same kernels/bytes as resident, addressing failure is loud, decode-only A/B token-equal ~25 then tie-diverges, two families coherent multi-turn |
| remaining D2H+sync per layer | why 2.1x not 16x | residency needs host routing; prefetch already refuted above |
| graphs under offload | BLOCKED, flag kept as escape hatch | `moe.allow_graphs_under_offload` capture aborts every attempt (`moe_host_args_capture_guard` throws - host-read routing); was never deliverable, descriptions corrected #1373 |
| NVFP4 "mandatory on-device" | was unenforced - silent wrong answers | force_host_experts=8: fluent WRONG answer at 88.77 tok/s, =48: repeats `ftp`, all exit 0 (three guards each missed it); refusal now in `model/expert_placement.h` |
| NVFP4 offload decode built (2026-08-13) | correct now | resident 361.97 -> 384.03; 8-layer arm 88.77 wrong -> 44.54 correct (it was fast because it skipped its GEMMs); full offload 23.03 correct. Estimate missed the prefill half: per-expert NVFP4 M>1 fallback handed the 593 MiB matrix to `gemm_nvfp4` (IMA) - staged via the same slot pool |
| the promotion's blast radius | 4 more consumers read qtype as location | Phase 0b host pointer into CUTLASS, Phase 3 D2D from host source, micro-scales still uploaded, `can_decode_fast` unstamped. **Rule: `qtype` says HOW to decode bytes, never WHERE they live** |
| prefill thrash gate (NVFP4) | shipped | misses 106 108 -> 30 451 (3.5x), pp512 flat, tg +4.6% (inside spread - the miss count is the result) |
| `moe.pin_host_experts` (default off) | +14.8% pp512 (6/6), 4.4x load time | WSL2 cannot page-lock mmap; pageable staging costs 76.2 us host vs 2.8 pinned (9.6 vs 32.4 GB/s). One slab per (layer, projection) - a per-expert pinned buffer costs 24.7 s of cudaHostAlloc. Method note: 3 paired rounds read decode -33% and that was noise (off arm spans 34.7-66.1); 6 pairs decide |
| per-layer device staging | 2.5x prefill, ONLY with pinning | 317.6 -> 790.8 tok/s (324 MiB, one layer live); pinning off: 252-286 = staging buys nothing (driver stages pageable anyway; pinning is also what makes a projection contiguous). Re-prices pin_host_experts: break-even ~32k -> ~7.5k prompt tokens; gap to resident 61x -> 22x |
| staged CUTLASS grouped prefill | +136% prefill, opt-in on an UNEXPLAINED decode cost | `moe.staged_cutlass_prefill`: pp512 663.2 -> 1563.9 (6/6, <1% spread), tg 59.4 -> 37.7 (-36%, 6/6) after long prefill but FASTER (25.5 -> 30.6) after pp8 - inherited cache-state difference, unexplained, so not default. Campaign total: prefill 285 -> 1564, gap to resident 11x. smallM branch is unreachable (device-args path selected at `executor_forward_moe_cutlass.cu:159` first); remaining kernel lever is `dequantize_nvfp4_kernel` (52% of kernel time) |
| final regime | transfer-bound, as the budget modelled | H2D 150 GB at ~51 GB/s = 41% of step, kernels ~26%, launches ~24%; overlap does not pay (3 GEMVs x 6.4 us << 100 us threshold) |
| cache budget sweep (`moe.expert_cache_budget_pct`, #1374) | 2.47x from a config value | 5% = 24 slots = 36.6% hit = 10.51 tok/s; 15% default = 73/74.1%/20.99; 30% = 146/89.4%/30.51; 50% = 244/96.2%/51.86. Default deliberately NOT raised (on a real non-fitting model that VRAM is KV + weight cache) |
| the floor | exact, not conservative | below `3*top_k` slots/layer the cache retains NOTHING (0.0-0.4% hit at 14/19 slots vs 24-cell working set, slower than bypass). Sizing rule: offload needs `n_layers x 3 x top_k x slot_size` VRAM first - ~3.5 GB (120B-A5B), ~8.6 GB (Qwen3-Next-80B top_k 10) |
| two unpredicted keeps | - | budget runs the "wrong" way and it helps (pool = % of FREE VRAM, so offload grows the cache 32 -> 73 slots, hit 37.3 -> 89.4%); prefill was thrashing (384 active cells vs 73 slots = 19% structural ceiling) - working-set gate worth +5.6% pp512 (5/5) and lifts decode hit 88.7 -> 95.7% |
| capture-able offload, what it would take | routing AND residency device-side | routing exists (`expert_indices`, `d_lookup_`); a miss needs a host-issued H2D with no device-side ask - either zero misses (contradicts premise) or a defined miss result. 30B at 48 tok/s puts the 52-103 band in reach of a 120B |

Measurement traps this path enforces: prefill varies ~15% between runs of
the SAME arm - only paired alternating rounds decide; decode moves with
prefill length (cache warmth: pp8 -> 72.7% -> 13.82 tok/s, pp512 -> 88.7% ->
24.98); cold vs warm differ 2.4x (20.99 vs 49.60) - state which one a number
is. Reproduce: `tools/analysis/expert_cache_offload_sweep.sh` (MODE=ab),
`tools/analysis/moe_routing_skew.sh`, `tools/analysis/host_transfer_latency.cu`,
`tools/analysis/expert_promotion_overlap.cu`; nsys with
`--set runtime.cuda_graphs=never` on any resident comparison arm.

### Mission gaps 3-10, the closed entries in full

- **(3) Model swap** - `server.model_swap` (default on): in-flight
  generations drain (never cancelled), a failed load restores the previous
  model; `/v1/models` lists the directory. Serial by nature: 32 GB holds one
  model; warm cache (#956) makes repeats cheap.
- **(4) Regex constraint** - `response_format: {"type":"regex"}` +
  `guided_regex`, built on the in-tree `RegexNfa` (a second engine was
  written, measured identical, discarded). The work was the decode-time
  wrapper: mask cache, EOS gating, and closing every mask bypass (spec-ngram
  + graph-loop routers, two `apply_mask` sites, thinking suppression, pooled
  state).
- **(5) Speculation tree** - built and measured 2026-08-31, not a win.
  `speculative.mtp_tree_width=W` verifies W MTP chains as one multi-candidate
  chunk (dense: private KV blocks; GDN hybrid: W recurrent slots, winner
  replayed through the captured graph). Qwen3.8-27B-NVFP4: tree ceiling
  top-2 covers +6..+10 points over top-1 at depth 1 (E[accept] +0.15..0.33);
  same-state trace +12.3% emitted/verify; think traffic W=2 vs linear
  adaptive-k -6.4/-6.9% ungated, -0.8/-5.8% with the margin gate
  (`speculative.mtp_tree_margin`). Rows are the cost (M<=4 batched GEMV vs
  CUTLASS tile, LM head per MR=4 rows, serial alternate-chain drafting).
  Default W=1; levers and tables in
  [`plans/2026-08-31-mtp-multicandidate-hybrid.md`](2026-08-31-mtp-multicandidate-hybrid.md).
  "No trained draft head" retired 2026-08-19: the MTP head pays +21.3% at
  k=1. `token_recycling` re-measured 2026-08-19 on the fixed build: **-0.27%,
  neutral** (was -7.0% on 2026-07-27; same cause as the MTP flip -
  `ea547a53` made the verify chunk cheaper). Level moved too: 156 tok/s where
  the old measurement read 99.37, because #1102 sits between (NVFP4 decode
  cache was double-charged for KV, leaving 100/280 tensors decoding from
  Q6_K; fixed, sweep flat 162-163 across every capacity).

  | `token_recycling` | tok/s (r1, r2) | mean | drafted | accepted | verifies |
  |---|---:|---:|---:|---:|---:|
  | off | 155.95, 155.07 | 155.51 | 48 | 4 (8.3%) | 3 |
  | on | 154.76, 155.41 | 155.08 | 77 | 9 (11.7%) | 27 |

  ```
  [PROV: commit=d374df1b date=2026-08-19 hw=RTX5090 model=Qwen3-14B-Q6_K
         quant=Q6_K cuda=13.3 path=imp-server n=3 reasoning prompts x 2
         alternating rounds cmd=`imp-server --think-budget 0 --set
         speculative.token_recycling=true|false --set server.prefix_cache=false`,
         tokens from usage.completion_tokens, counters from /metrics]
  ```

- **(6) Context spill below VRAM - do not build it** (scoped 2026-08-01).
  The "silent context loss" half was stale: a prompt past the window is a
  typed refusal; generation hitting the window ends `finish_reason:
  "length"`; StreamingLLM eviction is double-gated (<10% free blocks AND
  FP16 KV) and since 2026-07-31 client-visible
  (`usage.prompt_tokens_details.evicted_tokens`, verified on all three
  dialects). The capacity half is shelved on three measurements: no
  reproducible trigger (AUDIT B84: 4k/32k/128k all granted; holding 128k at
  81 MiB free costs 1% decode), the spill lands on a 6.5x cliff (AUDIT B36:
  1531 vs 237 GB/s resident/spilled - the #1103 mechanism), and each
  transfer blocks the host ~165 us (`executor_elementwise.cu:409`) = 4.7% of
  a 3.5 ms step regardless of bytes. Structural: spill cannot be expressed
  in the block table (negative ids already mean "skipped"; quantized kernels
  don't check the sign), so restore must happen before `batch.cpp` builds
  the table - prefetch has one good hook (`prefill_allocate_kv_blocks_`) and
  decode knows only the next block ~3.5 ms ahead. Revisit on: a pool clamped
  below request, a shallower H2D cliff, or workloads past the 128K ceiling.
- **(7) Agentic quality vs competitors** - `tools/analysis/agentic_compare.py`
  published in [`BENCHMARKS.md`](../BENCHMARKS.md) (3 families, 4 budgets,
  8-turn sessions). Headline is a defaults difference: at 200-token budget
  imp keeps every contract, llama.cpp needs ~800. Found a real imp bug on
  first run (Llama-3.2 bare-JSON tool calls dropped, #1088). vLLM/SGLang
  deliberately out of scope (weight format + VRAM).
- **(8) GBNF** - a nondeterministic pushdown simulator
  (`src/compute/gbnf_grammar.cpp`, parser in `gbnf_parser.cpp`) behind the
  same `apply_mask` contract. Two dominating costs: mask build 333 ms -> 12
  ms (interned parse stacks + memoised successor sets), and refusal
  discipline (left recursion incl. nullable-prefix and star-over-nullable,
  undefined rules, missing root, repetition bombs = compile-time 400). UTF-8
  assembled across token boundaries, overlong encodings rejected. Mutation
  run found a real hole: recompile cleared the stack arena but not memoised
  transitions - second grammar on a pooled manager decoded with the first
  one's.
- **(9) /v1/rerank** - causal-LM cross-encoder (Qwen3-Reranker class), query
  + document scored JOINTLY in one prefill-only forward reading two logits;
  the entry's premise (needs BERT-style seq-classification arch) was wrong
  for the current generation. Validated vs llama.cpp on the same GGUF: top-1
  agreement 3/3 queries, median score delta 0.0014. Gate: `make
  test-rerank`. Known: first call after load scores ~1e-3 off later ones
  (cold vs cached-prefix arithmetic); ordering unaffected.
- **(10) External agent harnesses** - stage 2 runs real binaries: aider
  (OpenAI), Claude Code (`ANTHROPIC_BASE_URL`), OpenAI Agents SDK
  (`/v1/responses`), each landing a real edit in a throwaway repo. Paid for
  itself on run 1: with tools present the STREAMING path handed reasoning to
  the user as the answer (non-streaming was correct); pinned at three
  levels. The Responses leg measured a model property that looks like a bug:
  Qwen3-8B at 400-token budget makes the call (232 used), at 1400 it
  reasons past the tool (511) - the leg pins temp 0 / 400 tokens.
  OpenHands stays out (docker-in-docker, disproportionate).

Explicitly NOT gaps: continuous batching, prefix caching, per-request LoRA,
embeddings, the three APIs, `/metrics`, suspend/resume, sampler surface (DRY,
mirostat, typical_p, logit_bias).

### Known limitations, the roadmap copy

- **MTP on Nemotron-3.5: head drafts (41% offline), verify chunk uneconomic.**
  History in three steps, each superseding the last:

  1. 2026-08-12 build: k=1 -41%, k=2 -54%, k=3 -65% (216/166/129 vs 364) -
     draft path uncaptured, three syncs per draft token. Device-side draft
     MoE (`gemv_f16_moe_decode`) bought +14-23% on the spec path and did not
     change the verdict (bound said even a FREE draft caps at +10.7%).
  2. 2026-08-19 re-measure after `ea547a53`: reads -2.0% at k=1 - but only
     because the acceptance-poor floor unbinds after 8 verifies at 0/8
     accepted; serving accept 0-9% vs 43.9% offline did not add up.

     | k | tok/s (r1, r2) | mean | vs k=0 | drafted | accepted | verifies |
     |---|---|---:|---:|---:|---:|---:|
     | 0 | 366.11, 363.45 | 364.78 | - | - | - | - |
     | 1 | 357.45, 357.63 | 357.54 | **-2.0%** | 24 | 2 (8%) | 24 |
     | 2 | 353.59, 349.88 | 351.74 | -3.6% | 48-80 | 1-7 (2-9%) | 24-40 |

     ```
     [PROV: commit=02872bdf date=2026-08-19 hw=RTX5090
            model=NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 quant=NVFP4 cuda=13.3
            path=imp-server n=3 prompts x 2 alternating rounds
            cmd=`tools/analysis/mtp_k_sweep.sh` with MTP_MODEL set, counters from
            /metrics, give-up line from the server log]
     ```

  3. **2026-08-20, resolved: the 0-9% was a defect.** A fully rejected verify
     adopts `spec_snap_slab` (`engine_spec_ngram.cpp:944-950`); `run_gdn` has
     written that slab since #847, `run_ssm` NEVER did (`ssm_scan_prefill`
     had no snapshot parameter) - every fully rejected verify on a Mamba2
     hybrid committed uninitialised VRAM as recurrent state. Poison-fill
     proof: 0 of 26 378 240 bytes written without the wiring, 99.71% with it.

     | | offline top-1 accept | serving accept | k=1 decode |
     |---|---:|---:|---:|
     | before | 851/2097 = 40.6% | 0/24 = 0.0% | 354.80, 356.54 tok/s |
     | after | 861/2097 = 41.1% | 590/1507, 587/1510 = 39.2/38.9% | 177.17, 175.12 tok/s |

     The two counters are the same quantity at k=1 and now agree. Economics
     verdict unchanged, reason corrected: guard disabled, k=1 costs -51%
     (176 vs 363); shipped guard lands 258-341 because 1.41 emitted/verify
     sits ON the 1+0.40k break-even and flips between runs. The drafter was
     never the problem; the verify chunk is. Not MTP-specific: the same
     branch serves any drafter (pre-fix n-gram derailed into unrelated prose
     after the first fully-rejected verify). The 270 `mtp.*` unrecognised-
     weight warnings on this checkpoint closed as #1497
     (`divert_to_mtp` gated on `llm_compressor_format`, Modelopt exports
     reach the generic mapper; head uploads anyway).

     ```
     [PROV: commit=8a7f2763 date=2026-08-20 hw=RTX5090
            model=NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 quant=NVFP4 cuda=13.3
            path=spec-verify/mtp-draft
            cmd=`imp-server --think-budget 0 --set speculative.ngram=false --set
                 speculative.mtp_k=1 --set speculative.mtp_econ_min_emit=0 --set
                 server.prefix_cache=false`, 3 prompts x 700 max_tokens; offline arm
                 `imp-cli --mtp-spec-decode 1 --set speculative.hybrid=false` on the
                 same 3 prompts
            n=2 per arm, arms alternated, fresh process per arm]
     ```

  Context: NVIDIA's own DSpark drafter measured -42% in vLLM on this card;
  the model card recommends no speculation at H100-class bandwidth.

- **Blackwell only.** No Hopper, Ada, Ampere, AMD, Intel, Apple, CPU.
- **Qwen3.5-27B MXFP4 untested** - old wording stale in each part: the
  alpha/beta NaN is moot (GDN alpha/beta pinned FP16_ONLY,
  `tensor_kind_table.cu`); the SafeTensors loader warns rather than refuses;
  the real blocker is no MXFP4 SafeTensors decode path outside gpt-oss.
  Blocked on a decodable checkpoint, not on a bug.
- **Gemma-4 Q4_K_M code-gen drift** - no longer reproduces (2026-06-13 and
  2026-08-11); original file gone, cannot A/B. If another Q4_K_M quant
  degenerates: Q5_K_M or Q8_0.
- ~~Native-FP8 weights decode through their FP16 companion~~ - **closed
  2026-08-12, measured +7.5% median decode** (27 pairs, t=3.33; order
  balanced: ON-first reads +8.2%, OFF-first +3.8%; bytes/bandwidth predicts
  +6.9%; do not quote the +11.1% mean). Mechanism existed
  (`fp8_decode_sidecar` rule) and did not recognise a native-FP8 source;
  `FP8CacheEntry::native_source` now drives it. Two traps: registering
  without the phase-4 rule moved PREFILL to FP8 (status 15); phase 3 then
  freed the FP16 copy prefill needs - visible only under `--bench` decode
  mode 2. One lookup answering two questions, twice.
- ~~No dequant path for native FP8 weights~~ - **closed**: FP16 companion at
  load (Nemotron-3.5 MIXED_PRECISION: 5935 NVFP4 expert tensors + 46 FP8
  Mamba projections; sm_120 has no FP8 prefill GEMM, raw bytes used to reach
  cuBLAS as `CUDA_R_8F_E4M3` = status 15). Costs 1698 MiB FP16 cache; init
  24.4/32.6 GB. See [`MODELS.md`](../MODELS.md).
