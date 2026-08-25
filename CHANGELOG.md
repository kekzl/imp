# Changelog

All notable changes since v0.6. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

**This is a changelog, not a journal.** One to three lines per entry: what
changed, from the reader's side, plus the number or issue that makes it
checkable. The investigation behind a change (hypotheses, what was ruled out,
how it was measured) belongs in `docs/` (`quantization.md`, `roadmap.md`,
`AUDIT.md`, `docs/plans/`) or in `docs/MISSION_JOURNAL.md`, and the entry links
there instead of retelling it.

## [Unreleased]

### Fixed

- **A 32-stream burst no longer starves its own tail.** Three defects, one
  per layer, found by tracing per-request arrival and first-prefill times:
  the HTTP worker pool was sized to cores (cpp-httplib default max(8,
  cores-1) = 15 here) while a streamed completion holds its worker for the
  whole generation, so late requests ARRIVED 4-7 s late; the prefill
  scheduler ran one forward per step while anyone decoded
  (`prefill_batch_decode_cap` 1 -> 0, replaced by a token-charged budget:
  one full `prefill_chunk_decode_cap` chunk per step as before, several
  small prompts together, each charged a 256-token launch-cost floor); and
  the prefill round-robin rotor drifted over the shrinking batch (now
  rotates by last-served request id). At 32 streams: TTFT max 6.8-8.0 s ->
  2.2-3.7 s, stragglers 1-10 -> 0, and the 4-wave bench reads 1047-1073
  tok/s on EVERY wave against 629/954-991 before - the wave-1 "ramp" was
  these defects. Details: `docs/plans/2026-08-24-qwen38-port.md`.

- **The decode loop's host time is now instrumented end to end**
  (`diagnostics.step_timing`: engine phases + step_impl blocks;
  `IMP_WORKER_TIMING=1`: server worker phases). It settled the last idle
  question with numbers: the steady-state loop is clean (resume/constrained/
  schedule at 0/0/1 us) and the residual "outside" time is the WAVE RAMP -
  prefills plus one CUDA-graph capture per never-seen batch size (75 captures
  across a 4-wave run; wave 1 reads 704 tok/s against 953-976 for waves 2-4).
  Batch-size bucketing with padded rows (the vLLM answer) is the priced
  follow-up in `docs/roadmap.md`.

- **Token delivery no longer preempts the GPU driver loop.** Each step's
  `push_token` woke its SSE handler immediately, and at 32 streams ~6.4 ms of
  handler work (detokenise + socket write) ran per step BEFORE the worker
  could start the next GPU step - 19% of the step period, measured by the new
  `diagnostics.step_timing` phase attribution. The worker now stages the
  step's events and hands them to a notifier thread in one batch; wakeups
  happen while the GPU is busy. 32-stream aggregate 933-963 -> 967-990 tok/s
  (median 987, three runs), per-request event order preserved, SSE and
  finish_reason verified, degen_suite 50/50.

- **The small-M GEMM's A4 variant exists and is measured** (`gemm_nvfp4_smallm_a4`
  + graph-safe `quantize_fp16_to_nvfp4_into`): both sides packed NVFP4. E2e it
  reads 742-747 tok/s aggregate at 32 streams against 955-963 with the CUTLASS
  path - worse than the FP16 variant - and its microbench stays bimodal at a
  92 KiB activation working set, retiring L2 eviction as the main driver. The
  route to Marlin-class small-M economics is a genuine Marlin port (cp.async
  pipeline, stripe partitioning); the config comment carries the numbers.
  Default stays OFF.

- **A Marlin-class small-M NVFP4 GEMM exists now** (`gemm_nvfp4_smallm`,
  `gemm.nvfp4_smallm`, default OFF): W4A16 dequant-to-FP16 in smem + HMMA +
  split-K on the plain weight layout. Isolated on the batched-decode shape
  (M=32 N=5120 K=5120) it reads 23.9 us median against the shipping CUTLASS
  tile's 41.4 - the "no-split ceiling" from the five-approach survey falls.
  In the real step the GDN scan's 9.7 GB/step of L2 traffic evicts the
  activation tile and it runs at 45.8 us, costing ~11% aggregate, hence the
  default. The two routes that flip the sign are priced in the config
  comment. Kernel-level and host-reference tests ship with it; the eight
  measurement iterations (536.5 -> 23.9 us, including a 32-way smem bank
  conflict, a warmup-ramp trap and an L2 set-conflict bimodality) are in
  `docs/plans/2026-08-24-qwen38-port.md`.

- **The n>1 decode loop no longer re-uploads each row's whole output history
  from pageable memory every step.** With the server's repetition-penalty
  default every batched row paid one pageable H2D per step (8.5k per
  32-stream wave, each a synchronous host stall); histories now live in
  per-request device slots and ONE kernel appends each step's sampled tokens
  straight from the sample slots. Measured throughput-neutral (the stalls
  were not the idle driver); the host-side stall class is gone and the old
  shared-buffer path remains as fallback.

- **The seven decode-pipeline gates now say which one closed** (same logging
  blind spot the prefill-graph gates had, #1646): one INFO line per process
  from the first batch that asked, plus an ENTERED/DECLINED line. Found
  immediately what it exists to find: the pipeline was assumed inactive on a
  workload where it in fact entered.

- **The auto batch resolver priced a hybrid's KV 4x too high and its recurrent
  state at zero: Qwen3.8-27B auto-sized `max_batch_size` 5 where 28 fits.**
  It counted all 64 layers as KV-carrying (16 are) at FP16 (the model defaults
  NVFP4 KV) and ignored the 151.5 MiB/sequence DeltaNet state, which unlike KV
  does not clamp downstream. Measured at 32 concurrent requests, out of the
  box: 224.7 -> 630.2 tok/s aggregate (auto 5 -> 28; a hand-pinned
  `runtime.max_batch_size=32` with `max_seq_len=4096` reads 937.1).

- **`max_seq_len: auto` ignored VRAM entirely on packed-4-bit KV.**
  `qtype_elem_bytes(NVFP4)` is 0 (half-bytes are inexpressible), so
  `kv=0 B/tok` and the resolver fell through to the 131072 cap regardless of
  the card; the downstream KV clamp was the only thing keeping it honest.
  NVFP4/MXFP4_KV now size as 4-bit like INT4.

- **Concurrent GDN decode now batches: 81.5 -> 474.9 tok/s aggregate at 32-way,
  5.8x.** A GDN decode step used to serve ONE sequence, with concurrent ones
  time-multiplexed round-robin, because the recurrent scan kernels were
  single-sequence. That forced the whole step — including the FFN and attention
  projections, which are ordinary GEMMs — onto the M=1 path: profiled at 32-way
  load, 82 % of GPU time sat in GEMV kernels and 1 % in CUTLASS GEMM, where a
  dense model is the other way round. The scan that genuinely cannot batch was
  3.8 % of that profile. Sequences are independent (each owns a state slot), so
  they now parallelise across `blockIdx.y` while tokens within a sequence stay
  sequential. Wall time for 32x200 tokens: 78.5 s -> 13.5 s, no cross-sequence
  contamination, single-stream decode unchanged (82.29 against a 79.78 baseline
  on the same build, i.e. noise). `runtime.gdn_batched_decode=false` restores the
  rotation. The scan kernel then became the largest non-GEMM item, and it was
  spilling: at the 128/128 shape one thread owned a whole state column, which
  costs 128 registers for the state alone — ptxas reported 255 registers, an
  88-byte stack frame and 128 bytes of spill stores, 2 blocks/SM and 8.3 %
  occupancy. Splitting the column across two adjacent lanes (the two dot
  products reduce with one `__shfl_xor_sync`) removes the spill entirely: **180
  registers, 0 spill**, and the kernel is **18 % faster at n_seq=32** (116.3 ->
  98.3 us). End to end, warm: 794 -> **886 tok/s aggregate**. Single-sequence
  decode keeps the unsplit kernel, where that shape is faster. Re-profiled after
  the batching fix: CUTLASS GEMM 1.0 % -> **66.2 %** of GPU
  time (a dense model reads 71.8 %), M=1 GEMV ~82 % -> ~1.5 %, and the scan runs
  21 024 times where it ran 230 400 — once per batch step per layer instead of
  once per token. The scan is now 19.3 % of the profile and the next lever; it
  scales sublinearly because each thread holds 128 state floats in registers.

- **The MTP prefill feed no longer costs -83% prefill.** Enabling MTP fed the
  draft head's KV cache one prompt token at a time, a full 425M-param M=1
  forward per token: Qwen3.8-27B pp512 7426 -> 1252 tok/s. Dense heads now
  feed in one M=rows pass per prefill chunk (`mtp_feed_batch`); measured
  pp512 6385-6510 tok/s across 5 runs against a 7070-7123 baseline (-8.8%),
  accept rate unchanged at 90.5-93.3%. MoE heads keep the per-pair loop.

- **`--set speculative.mtp_k=K` now enables MTP under `imp-cli`.** The startup
  hint recommended exactly that spelling, but the CLI only read its own
  `--mtp-spec-decode` flag and silently ignored the `--set`: a bench with the
  recommended knob measured "MTP off" while claiming it on.

- **A latent race in the GDN scan kernel, and the batched scan that exposed it.**
  Between reading `s_reduce[0]` for the K normalisation and overwriting
  `s_reduce` for the Q reduction there was no barrier, so a fast thread could
  store `q_sq` into `s_reduce[0]` while a slower warp was still loading `k_inv`
  from it — that warp then normalised K by Q's norm. Invisible at the shipped
  grid: it launches `(n_heads)` blocks, at most 48 here against 170 SMs, so
  every block had an SM to itself and its four warps ran in lockstep. Measured
  with a wider grid: stable through 128 blocks, non-deterministic at 256 (16384
  of 4194304 state floats differing between two identical runs). Fixed with one
  `__syncthreads()`; decode cost 79.82 against 79.78 tok/s without it, i.e.
  nothing.

- **`reasoning_effort` was accepted on the wire and never reached the chat
  template.** Every request rendered the template's own default. On
  Qwen3.8-27B-NVFP4 the same message now renders 11 / 41 / 53 prompt tokens for
  `medium` / `low` / `xhigh`, and 53 when the field is absent (its default);
  before, `low` and `xhigh` both produced 67 on an identical prompt.

- **jinja: `x is undefined` was true for any falsy value.** `undefined` was not
  in the parser's known-test list, so the expression fell through to
  `x == undefined` and `false == none` compares equal. Any HF template asking
  "did the caller set this?" got "no" while the caller was explicitly saying
  false — on Qwen3.8 a suppressed-thinking render still emitted the
  reasoning-effort preamble. `is not undefined` added alongside.

### Added

- **Decode-graph prewarm at init** (`runtime.graph_prewarm`, default on,
  no-op at `max_batch_size` 1): one staggered dummy batch walks every batch
  size once before the engine goes ready, so all per-size decode graphs
  (32/32 in 2.3 s on Qwen3.8-27B) capture at startup instead of during the
  first wave of real traffic. Wave-1 median request latency -3-12% at 32
  streams; aggregate throughput unchanged - which retires the "captures
  cost wave-1 throughput" attribution, see
  `docs/plans/2026-08-24-qwen38-port.md`.
- **NVFP4 KV is now the default for QWEN35 (Qwen3.8-27B and its Qwen3.5
  siblings), taking `max_model_len` from 48 512 to 131 072 tokens.** The KV cache
  holds K and V for the 16 attention layers only — 64 KiB/token at FP16 — and on
  a GDN hybrid it is what bounds context, so its dtype decides how much context
  fits. Cost, measured with alternating arms over `ppl_corpus_45k.txt`: +0.29 to
  +0.35 % perplexity on Qwen3.8-27B-NVFP4 (FP16 bit-stable at 4.6124 across 3
  runs), +0.15 to +0.18 % on Qwen3.5-4B mxfp4. `degen_suite.py` against a server
  on NVFP4 KV: 50 checks, 0 FAIL. This is a deliberate capacity trade, not the
  ~neutral bar the two FP8 gates use — `kv_cache.dtype=fp16` opts out. The MoE
  GDN siblings stay on their old default on purpose (FP8 KV already costs
  QWEN36_MOE +1.47 % PPL; NVFP4 KV is more aggressive and unmeasured there).
  New gate `kv_nvfp4_default_safe()` with a CPU-lane allowlist test.
- **The 128k context point is reachable on this card — the default KV dtype was
  the limit, not the hardware.** Qwen3.8-27B declares no FP8-KV hint, so the
  default stays FP16 and `max_model_len` resolves to 48 512 tokens. Measured,
  same card, nothing else changed: `kv_cache.dtype=fp8` → **96 960** tokens for
  +0.02 % perplexity; `nvfp4` → **131 072** (172 032 with `runtime.max_seq_len`
  raised) for +0.53 %. The KV cache holds K and V for the 16 attention layers
  only — 64 KiB/token at FP16, and that dtype is a config key. TTFT at nvfp4:
  852 ms / 7.7 s / 34.4 s at 5225 / 41 680 / 123 822 prompt tokens, output
  coherent. This corrects the phase-5 note that called 128k unreachable.
- `gdn_scan_fused_f32_batched()`: the GDN scan over N independent sequences in
  one launch, sequences on `blockIdx.y`. Tokens within a sequence stay
  sequential — that part genuinely cannot batch — but separate sequences share
  nothing except weights. Bit-identical to N single-sequence launches across 7
  cases incl. sparse unordered state slots. This is the kernel half of making
  concurrent GDN decode batch; the executor and scheduler still run one sequence
  per step, so it is not yet on the serving path.
- **GDN hybrids gain nothing from concurrency, measured.** 32 concurrent
  200-token requests on one host: Qwen3-14B-NVFP4 (dense, no GDN) 1427 tok/s
  aggregate; Qwen3.8-27B-NVFP4 **81.5**, the same as its single-stream rate;
  Qwen3.6-35B-A3B-NVFP4 132 against 320 single-stream, i.e. slower under load.
  All 32 are admitted and in flight at once (start spread 0.02 s, peak in-flight
  32, verified per request) and their outputs stay byte-exact and isolated — the
  work simply does not batch. **Profiled**: CUTLASS GEMM is 71.8 % of GPU time on
  the dense control and 1.0 % here, while M=1 GEMV kernels are ~82 % here and ~7 %
  there — 305 152 GEMV launches for 4800 tokens across 64 layers is one per token
  per layer, against 48 000 CUTLASS launches per batch step on the dense model.
  Cause is `engine_scheduler.cpp:1503`: the recurrent scan kernels are
  single-sequence, so a GDN decode step serves one sequence and concurrent ones
  round-robin. The scan itself is 3.8 % of the profile and forces the other 82 %
  onto the M=1 path. Recorded in `LIMITATIONS.md` so a deployment is
  planned for latency rather than aggregate throughput. TTFT on the same build:
  767 ms at 5225 prompt tokens, 7760 ms at 41680; 128k is unreachable
  (`max_model_len` 48512). llama.cpp cannot serve this checkpoint at all — the
  reference build b10524 has no `Qwen3_5` converter.
- **Qwen3.8-27B vision measured against the HF reference.** Preprocessing and
  splicing are exact — 124 prompt tokens and 64 image placeholder tokens (8×8
  from a 16×16 grid) on both sides, both fixtures. Top-1 and top-2 tokens agree;
  softmax total variation is 0.104-0.318 with an image against 0.0042 for a
  text-only control through the same template path, so the vision path does add
  divergence and the decision survives it. Text-only unchanged: perplexity 4.6124
  against 4.6202, −0.17 %. Video stays absent by decision and is rejected at
  admission rather than silently dropped.
- **Qwen3.8-27B serves 32 concurrent sequences with byte-identical output**, on
  this card, with no code change — `runtime.max_batch_size=32`,
  `kv_cache.dtype=fp8`, `runtime.max_seq_len=4096`, `runtime.deterministic=true`,
  `server.prefix_cache=false`. Resident: weights 17920 MiB, GDN state 4848 MiB
  (48 layers × 32 sequences), KV 42976 tokens. 32 interleaved requests across two
  unrelated prompt families returned 16/16 and 16/16 byte-identical to their
  batch=1 references with zero cross-contamination, and a 32-way load leaves the
  next sequential request identical to the cold one (3/3). What blocked the load
  before is the pre-upload reserve clamping KV headroom to `total_vram/5`
  regardless of dtype (`engine_weight_upload.cpp:180-195`); bounding
  `max_seq_len` puts the estimate under the clamp. Byte-identity needs both
  switches: prefix cache off gets 31/32, determinism closes the last one.
  Trade-offs (short context, FP8 KV, re-prefill on multi-turn) in
  [`docs/plans/2026-08-24-qwen38-port.md`](docs/plans/2026-08-24-qwen38-port.md).
- **Two diagnostics that make reference parity measurable.**
  `diagnostics.dump_final_logits_dir` writes the post-soft-cap LM logits of each
  forward pass as FP32 `.npy`; `diagnostics.dump_gdn_state_dir` writes the
  Gated-DeltaNet recurrent state plus a per-pass RMS / non-finite line. Both skip
  passes captured into a CUDA graph, where the device-to-host copy would fail the
  capture. First result on Qwen3.8-27B-NVFP4 against transformers 5.15.1 BF16:
  perplexity 4.6202 vs 4.4194 (+4.54 %, the 4-bit price), top-1 token agreeing on
  14 of 17 prompts, and **0 non-finite state values across a 46579-token
  prefill** with a flat RMS. At that length the prediction still matches exactly
  (softmax total variation 0.0000) while the logit values drift — pearson
  0.986 → 0.747 from short to 46k context. See
  [`docs/plans/2026-08-24-qwen38-port.md`](docs/plans/2026-08-24-qwen38-port.md).
- `write_npy_fp32` now has tests (NPY v1.0 magic, 64-byte header padding,
  row-major payload, `(1, N)` rank for a single row). Every dump above is read
  back through numpy, so a silently transposed or unreadable header would have
  looked like an engine defect.
- Qwen3.8-27B tokenizer parity against HuggingFace: 32 cases, encode **and**
  decode compared byte-for-byte (`make test-gpu`, needs no GPU — it reads
  `tokenizer.json` only). Both halves mutation-validated.
- `docs/plans/2026-08-24-qwen38-port.md`: measured inventory of what Qwen3.8-27B
  support already is, with the four gaps that remain and the memory arithmetic
  behind them (KV 65536 B/token, recurrent state 151.5 MiB/sequence).

## [0.30.1] - 2026-08-24

### Fixed

- **The build-from-source docker recipe did not mount the cache volume, and the
  quickstart called it optional.** Without it the memory plan charges a 3900 MiB
  library-reserve constant on every start and that comes out of the KV pool, so
  a copied recipe costs KV capacity rather than startup time. AUDIT B77 measured
  639 MiB per restart on Qwen3-14B-Q6_K; on a model whose first forward claims
  almost nothing the gap is the whole constant. Measured on Qwen3.8-27B-NVFP4,
  mounted against not, KV dtype held at FP8: 3030 to 3639 blocks, 96960 to
  116448 tokens, +20 %, because the second start plans the measured 3291 MiB
  instead of the 3900 MiB constant. Both recipes mount it now and the
  quickstart says what the second half of the volume is for.

- **The library-reserve warning recommended a value the server does not charge**
  (#1746). It named the forward window while the plan, the cache and the next
  start all use `max(forward_window, whole-init)`, which on the NVFP4 path is
  three orders of magnitude larger: an operator pinning the recommended 4 MiB on
  a model that charges 3060 got exactly the under-reserve the same warning
  describes in its other branch. The report now runs after the charge is decided
  and names that number, prints both figures so the gap is visible, and cites
  AUDIT B79 rather than the B41 stability claim B79 superseded. Verified on
  Qwen3.8-27B-NVFP4: the warning recommends 3060 MiB, which is what the line
  above it records.

- **`live pass would have said N` printed a rescue floor, not a reading**
  (#1747). It logged `kv_max_blocks` after `min_kv_tokens` had raised it, so a
  start that hit the floor reported a lower bound of the configuration as if it
  were the live pass's own figure, and a reader nearly concluded a pool held 512
  blocks where it held 3639. Both numbers are named when they differ: `KV
  blocks: plan 3628 (live pass sized 197, raised to 512 by min_kv_tokens)`.

## [0.30.0] - 2026-08-24

### Added

- **Per-launch expert imbalance is recorded and readable while serving**
  (#1548). `max(M_e)` is what decides grouped-GEMM cost, because the kernel pads
  every expert to one M tile and a single hot expert sets it for all of them. It
  was computed on the host at three sites, used to pick the tile, and dropped;
  the only record was a whole-process activation histogram written at shutdown,
  which averages that skew away. Four device counters per layer now carry peak
  and mean `max(M_e)`, `/metrics` serves them as `imp_moe_expert_imbalance` and
  `imp_moe_expert_peak_rows` from a running server, the histogram JSON gains a
  `per_layer_imbalance` block, and `moe_routing_skew.py` prints the ratio.
  Measured on Qwen3-30B-A3B-NVFP4 at 40 tokens: layer 0 at 13.1x, so 92 % of
  the tile rows are padding. Recorded only at n > 1, where the number decides
  something: a decode step would pay a kernel launch per MoE layer for nothing.


- **Chat-template goldens for nine families** (#1572), rendered from the upstream
  templates and compared byte-for-byte: ChatML (three checkpoints), Gemma, Llama 3,
  Llama 2, Mistral V3, Nemotron, DeepSeek R1, Phi. Previously only Harmony had a
  golden and the rest were structural smoke tests. `make chat-goldens` re-pins them.
  The goldens compare the rendered prompt, not token IDs, so the whole set runs in
  the CPU lane with no skips.

- **Fuzz targets for the six parsers that take untrusted bytes** (#1620), in
  `fuzz/`: JSON Schema, regex, GBNF, the tool-call stream filter, the
  SafeTensors loader and `tokenizer.json`. Standard `LLVMFuzzerTestOneInput`
  entry points for libFuzzer (`-DIMP_FUZZERS=ON`, clang), and the same functions
  driven over a committed corpus plus a deterministic mutator in the CPU lane,
  0.7 s, on every PR. The corpus is the inputs that actually broke something.
  `docs/audit/SETTLED.md` S-28 and `AUDIT_ARCH_2026_07_29.md` both claimed these
  surfaces were "fuzzed, in CI" while no fuzz target existed; both are corrected.

- **A `Sanitizers` CI job** (#1621). No lane ran any sanitizer in any category
  before: ASan/UBSan existed behind a manual `make asan`, and the only job that
  mentions compute-sanitizer is gated on a GPU runner that does not exist. It
  builds test-core and test-text with ASan+UBSan and runs them, including the
  fuzz corpus. Measured worth: against four reverted parser fixes the corpus
  catches 3 of 5 targets without ASan and 4 of 5 with it - an out-of-bounds read
  is invisible otherwise.

- **`VramOwned<T>`: an owning handle for a `VRAMAllocator` allocation.** The
  allocator says of itself, in its own destructor, that it is "a tracker, not an
  owner", so ownership lived with the caller as a raw pointer plus a free somewhere
  else. That spelling is where this week's allocation defects were: four pointers
  freed through an allocator that had not produced them, and 128 MiB released with
  the wrong API. The handle carries the allocator that produced it and releases
  through that one; move-only, and no `release()`, because a raw-pointer escape
  would make the class writable again. Two callers converted, and the direct-site
  allowlist shrinks 466 to 464. (`AUDIT.md` R7 records that earlier audits referred
  to this type before it existed.)

- **The release blocker is now enforced where it is defined.** `GOAL.md` makes a
  hero regressing against a competitor a release blocker, over seven heroes, of
  which gates observed two: Gemma-4 sat 5.3 % down for six weeks and no gate
  could have said so because none looked. `scripts/check-release.sh` gains
  `make bench-competitive` as a fourth model-backed stage, failing on any hero
  under a 5 % decode lead and naming which and by how much. The two heroes this
  host cannot contest (Qwen3-Coder-30B-A3B and Nemotron-H are NVFP4-only, and
  llama.cpp has no NVFP4 path on sm_120) are **printed with the reason**, because
  a hero nobody measured must not read like a hero that passed.

- **Invariant I2 has a gate: `make check-alloc-interpose`.** "Nothing allocates device
  memory while serving" was stated, counted, and unobservable: the counter only sees what
  routes through `Backend`, and the `--wrap` interposer that would see the rest compiled in
  no make target and no CI job. The gate builds it, drives 20 requests at batch 4 with
  NVFP4 residual KV and an MTP chain, and pins the count. First run: **46 allocations while
  serving**. 27 were the MTP workspace allocated after the phase flip and are now labelled
  as the context-setup step they are; the remaining 19 are named by site in
  [`DEBT_LEDGER`](docs/audit/DEBT_LEDGER_2026_08_21.md) and the pin only ever goes down.

- **`diagnostics.spec_trace` reports the top-2 logit gap per verify chunk row.**
  The trace said which token a row picked and never by how much, which is the
  question that decides whether a disagreement with the decode path is a coin flip
  or a real difference of opinion. Off by default; the logit buffer is only
  allocated when the flag is on.

- **`make bench-competitive` re-runs the llama.cpp comparison with the competitor
  image pinned by digest.** The previous sweep was six weeks and 548 upstream
  builds old and compared against a build the repo did not record. Re-measured
  against b10524: imp leads on all four heroes (+148 %, +44 %, +29 %, +16 %) and
  llama.cpp is flat to slightly slower than b9976 on every shared model. It also
  reports imp with and without n-gram speculation, which is what caught that the
  drafter never engages on Qwen3-14B Q6_K. ([`BENCHMARKS.md`](docs/BENCHMARKS.md))

- **`speculative.verify_row_parity` makes the verify chunk reduce K the way the
  decode step does.** The two paths grouped the same products into 32 partial sums
  (decode) and 128 (verify); the rounding difference reached the stop decision and
  truncated answers under `speculative.mtp_k=1`. Off by default. On Qwen3.8-27B-NVFP4
  it halves the truncation rate (2/6 to 1/6) and measures faster, not slower:
  105.30 tok/s against 104.24 with it off and 88.52 at `mtp_k=0`.

- **Three gates that could not fail now can.** The file-size allowlist gave its 29
  entries no size limit at all, so `engine_scheduler.cpp` grew 1074 to 1962 code LOC
  (+83 %) with CI green throughout; each entry now pins a measured `code_loc` and
  drift either way fails (`--update` re-pins). The number of GTest cases that run in
  no CI lane is pinned at 968 macros and named as a macro count in its own failure
  message. And 28 functions defined inline in a header with no caller anywhere were
  removed, with a gate to keep the count at zero.


- **`make check-alloc-pairs` fails when a pointer is freed by the wrong
  allocator**, in CI as well. Two passes: within a file, and across files for
  member variables, because the 128 MiB pair above allocates and frees in
  different translation units and no per-file check can see it.

### Changed

- **`pdl.h` said the launch attribute overlaps a kernel's tail with the next
  kernel's head; it cannot** (#1655). Programmatic dependent launch needs a
  device half, and no kernel in `src/` calls
  `cudaTriggerProgrammaticLaunchCompletion()` or
  `cudaGridDependencySynchronize()`, so a converted edge releases where the
  default edge did. Measured before deciding: `runtime.no_pdl` on against off
  is 12508 vs 12455 tok/s prefill and 385.8 vs 382.3 tok/s decode, both inside
  the arms' spread. The wiring stays, the claim is corrected, and this audit's
  summary rows stop counting PDL as working idiom while their own evidence file
  records `griddepcontrol: 0`.


- **The "speculation is off" diagnosis names the gate that refused** (#1538,
  #1539). It printed eighteen request fields and marked none of them, so
  answering "why is speculation off for my request" meant re-deriving
  `spec_verify_gates_ok_` by hand against a log line. Both issues were filed
  against that line. It now reads `refused by 'sampling_not_greedy'`, from the
  same function that makes the decision, and the field dump stays because the
  neighbouring values are usually wanted too.

- **Both gates that keep speculation off a default server request were measured
  and both stay** (#1538, #1539), recorded in `docs/DESIGN_DECISIONS.md`.
  Qwen3-14B-Q6_K on one RTX 5090, arms alternated: enabling speculation by
  dropping `temperature` to 0 is not faster (157.6 against 157.8 tok/s), and
  relaxing the think-block gate doubles verify steps for 3.0 % less throughput
  (157.9 against 162.7). A verify there costs eight to ten decode steps and
  returns 5.83 tokens.


- **`docs/LIMITATIONS.md` gains a "Gates that do not exist" section** (#1571,
  #1642), for absent instruments rather than untested features: no KL / PPL
  drift gate against a reference forward, and no soak or endurance test. Both
  need a card, and CI has no GPU runner, so a reader deciding against imp sees
  the gap without reading the issue tracker. The bullet on the generation half
  of the HTTP contract now also cites #1559, which is the same wall.


- **The tree is C++23 where the build said it already was.** `bool f(...,
  std::string& err)` is gone from `src/`, `tools/` and `include/` (36 sites, 15
  of them header declarations), replaced by `std::expected`; host pointer+length
  pairs became `std::span` (12 uses to 87). Device pointers deliberately stay
  raw: a span over one is a silent host segfault at the first `s[0]`. What nvcc
  13.3 accepts in **device** code was measured on the card rather than assumed,
  and the audit's "nvcc constrains what is usable in `.cu`" is refuted:
  [`CPP23.md`](docs/internals/CPP23.md).


- **A red gate now blocks the merge.** Ruleset `Require CI` requires exactly one
  context, `Build`, so every other check was advisory and two PRs merged over a red
  `File size` in forty minutes. `scripts/ci_static_gates.sh` is one list run from two
  places: the first step of `Build`, where a failure makes the required context red,
  and the named jobs with a filter, so which gate failed still has a check name. Ahead
  of the compile, so a drifted allowlist fails in ~15 s rather than after three
  minutes. `Lint`, `Mock API contract`, `clang-tidy` and `Real API contract` stay
  advisory for stated reasons; repo settings are untouched.

- **Why a verify chunk costs 8.4x a decode step on Q6_K is recorded as open, with
  three refuted explanations.** It is what makes n-gram speculation net-negative on
  short requests on the north-star checkpoint. The chunk being a different, unfused
  execution graph explains the shape; the 3.79x on NVFP4 against 8.4x on Q6_K for the
  same model has no mechanism. Refuted and not to be re-run: the per-chunk source
  dequant (a `ngram=false` control gives the identical 2800 launches, so it is all
  prefill), #998's overlay not firing (its kernel is in the spec-on trace and absent
  from spec-off), and a slow verify kernel (20.7 us per call against 68.1 us for
  decode `gate_up`). ([`DEBT_LEDGER`](docs/audit/DEBT_LEDGER_2026_08_21.md))

- **The north-star model's default decode has two values, and which one you get
  is a coin flip.** Qwen3-14B Q6_K measures 162 tok/s when the n-gram drafter
  stays quiet and ~154 when it engages, over four otherwise identical isolated
  runs. Where it engages it accepts 6.2 % at ~50 ms per verify against a ~6.2 ms
  decode step. It is a cold start, not the model: on a 1024-token request the
  same checkpoint accepts 36.1 % at 6.78 tokens per verify. The economics guard
  meant to catch this cannot arm, because `spec_verifies >= 8` is per request and
  a 128-token request produces about one verify.
  ([`BENCHMARKS.md`](docs/BENCHMARKS.md))

- **Every hero that reaches `gemm.nvfp4_lm_head`'s MoE arm is now priced.**
  gpt-oss-20b MXFP4 was the only one besides Gemma-4 that hits `is_dense=false`
  without being in the rule's calibration set: `on` buys **+10.2 % decode for
  +18.2 % PPL** (413.02 to 455.24, 105.86 to 125.12), losing more clearly than
  Gemma-4's +7.4 / +9.0. The categorical call is correct on both.
  ([`GOAL.md`](docs/GOAL.md))

- **Gemma-4's 5.3 % decode drop since July is a priced trade, not a regression.**
  Bisected to `63df2d30` (#982's `gemm.nvfp4_lm_head` auto rule), then named inside
  that two-change commit by flag rather than by splitting it: `on` buys **+7.4 %
  decode for +9.0 % PPL** on Gemma-4-26B-A4B UD-Q4_K_M. Losing by the rule's own
  standard, so the categorical MoE arm made the right call for a model that was
  never in its calibration set. Priced in `GOAL.md`'s trades list.

- **The `--bench` prompt is not self-repetitive, and six files said it was.**
  `imp-cli --bench` builds it as `tokens[i] = i % vocab_size`, so at
  `--bench-pp 512` it is 512 distinct ids and every 6-gram is unique; with
  `speculative.min_match = 6` the prompt cannot supply one draft. The ~99.9 %
  acceptance is real but comes from the **generation** looping under
  `ignore_eos`, and it turns on the *quantisation*: Qwen3-14B NVFP4 accepts 504
  of 504 where the same model at Q6_K accepts 6 of 96. Corrected in `GOAL.md`, `BENCHMARKS.md` and the
  `benchmark-cuda` and `server-api` skills. The operational guidance is
  unchanged: pass `--set speculative.ngram=false` to both arms of a decode A/B.

- **A stop-decision guard for MTP was built, measured and removed, and it closes
  the line.** Handing a verify chunk's stop token to the ordinary decode path
  instead of trusting the chunk row leaves the truncation count unchanged (2/6
  and 1/6, same as without it) while the guard demonstrably fires: twice within
  the truncating prompt's own 24 verify steps. The confident stop is a property
  of the state, not of the projection. This also refutes the entry's standing
  claim that closing it needs numerical agreement between the chunk and decode
  paths. ([`LIMITATIONS.md`](docs/LIMITATIONS.md))

- **The MTP truncation is documented as one outcome of ordinary speculative
  divergence, not a defect with a location.** All six probe prompts diverge from
  the non-speculative answer between byte 48 and byte 271, and the two that
  diverge earliest after the truncating one produce full, clean answers, so the
  divergence point carries no signal. The hidden-state diff that would localise it
  is not obtainable: `diagnostics.dump_hidden_dir` is host-side and decode is
  graph-replayed, giving 5 dump steps for both 40 and 200 generated tokens.
  ([`LIMITATIONS.md`](docs/LIMITATIONS.md))

### Fixed

- **The degeneration suite asserted the thinking contract #1560 replaced.** Its
  `anthropic-thinking` category, written 2026-06-06, required a `thinking` block
  on a `/v1/messages` request that never asked for one. Extended thinking is
  opt-in upstream and imp matches that since #1560, so the check failed the
  release gate against correct behaviour. Both sides are pinned now, and the
  opt-in path (`thinking.type=enabled` on either transport, plus
  `display: "omitted"`) gained the four checks that nothing covered while the
  default was asserted to think.


- **The CUTLASS NVFP4 GEMM handler accepted a `beta` and dropped it** (#1547).
  `GemmKernelArgs::beta` reached it and the epilogue is built with a literal 0,
  so a `beta = 1` call would have written the product over the residual instead
  of adding to it, and answered with quiet garbage. It refuses now, which hands
  the caller to the dequant path that does honour beta. Unreachable today
  because both callers that can set beta exclude the tier, which is one edit
  away from not being true.


- **The deterministic MoE combine scanned every expert row per output column,
  not per token** (#1546). The row search sat inside the column loop, so the
  O(total_rows) scan ran once per column chunk rather than once per token: 8
  times over 4096 rows for a 2048-wide model at 512 tokens, to find the same 8
  rows each time. Gathered once into shared memory now, accumulation order
  unchanged (ascending row), so the numbers are identical. Measured at
  n_tokens=512, top_k=8, d_model=2048: **3.159 ms to 0.0127 ms per call, 249x**.

- **The deterministic MoE permutation assigned every slot on one thread**
  (#1546). Phase 4 walked all `n_tokens * top_k` entries serially, a chain of
  4096 dependent shared-memory read-modify-writes, and became the dominant cost
  once the combine above was fixed. It processes one block-sized chunk at a
  time now, which produces the identical layout because chunks run in index
  order and a thread counts only lower thread ids. **0.177 ms to 0.059 ms, 3.0x**,
  which puts it inside the run-to-run spread of the non-deterministic path it
  mirrors. Both are opt-in (`runtime.deterministic_gemm`); the default atomic
  paths are untouched.

- **A tool-call test asserted the contract #1729 replaced.** `{"type":"object"}`
  as tool parameters is enforceable since #1729, and
  `ToolCallBuilderRejectsUnenforceable` still required it to be declined. It
  lives in `test-moe-gdn`, a GPU module, so no CI job could report it and #1735
  merged with it red. Pulled onto the new contract, with the XML dialect kept as
  the negative control: that one still declines, because it renders parameter
  keys as tags and a schema declaring none has no tag to render.


- **`additionalProperties` was parsed and never read, so a free-form object
  could only ever be `{}`** (#1729). The FSM behaved as if every object were
  `additionalProperties: false`: a caller who wrote `true` silently got only the
  declared keys, and `{"type":"object"}` accepted no document but the empty one.
  Undeclared keys are now legal where the schema allows them, and their values
  are parsed by an embedded `JsonGrammar` (nesting, escapes and the number
  sub-state included) rather than by the schema FSM. A property-less object
  parses as free-form, matching JSON Schema's default; an object that declares
  properties stays closed unless it says otherwise.

- **`strict: true` was all-or-nothing, so one loose tool disabled argument
  enforcement for every tool in the request** (#1597). OpenAI defines `strict`
  per function and imp now enforces it per function: a tool that asked for it
  keeps its schema, one that did not enters the name enum with a free-form
  parameter schema instead of dropping the whole request to the prompt hint. The
  realistic case is an agent set mixing a schema-bound `write_file` with a
  free-text `bash`, where the caller got post-hoc validation for the one tool
  whose arguments must parse. Needed #1729: before it there was nothing to put
  in the enum for a free-form tool.


- **`sim_token_valid` restored the JSON grammar state by hand, and that list had
  already gone stale once.** #1104 added the RFC 8259 number sub-state to the
  FSM and not to the eleven-field save/restore, so a simulated token that walked
  into a number left `num_seen_frac` / `num_need_digit` mutated on the real
  state. The grammar is a `JsonGrammar` struct now, vocabulary-free, and the
  snapshot is one copy — a field added to the grammar round-trips because it is
  in the grammar. Pure move: `advance_char` and `compute_allowed_mask`
  referenced no vocabulary member, and the 86 JSON/schema property tests pass
  unchanged. Groundwork for #1729.

- **A clipped NVFP4 activation scale was silent** (#1544). The dynamic quantiser
  encodes the per-16-block scale as `absmax/6` into UE4M3, which saturates at
  448 - so a block with `absmax > 2688` quantises against a scale that is too
  small, and `float_to_fp8_e4m3` clamps and returns without a word. Measured,
  largest per-16-block absmax over a 4096-token prefill:

  | model | largest block absmax | of the 2688 ceiling |
  |---|---|---|
  | Gemma-4-12B-NVFP4 | 2468 | **92%** |
  | Nemotron-3-Nano-30B-A3B-NVFP4 | < 1500 | < 56% |

  It does not fire on these models; the headroom on Gemma-4 is 8%. A device flag
  records a crossing and `gemm_cleanup()` reports it at shutdown, so it stops
  being silent. The per-tensor global scale that would remove the ceiling is not
  in here - that changes the quantiser's numerics and needs its own equivalence
  test.

- **Three API-contract tests could have run in CI all along, and two passed
  vacuously** (#1600). The `Real API contract` lane selects `-m nomodel`;
  measured against the same model-less server it runs, 63 of 126 tests pass but
  only 58 are marked, so five were coverable and deselected. Two of the five
  pass on *nothing*: `test_sequence_numbers_monotonic` collected an empty SSE
  list and asserted `[] == sorted([])`, and `test_probes_agree_on_context_length`
  returned early when `/v1/models` was empty. Those two are fixed rather than
  marked - the first now fails on an empty stream, which it would also have done
  with a model - and the other three are marked. **CI lane 58 -> 61 tests.** The
  generation contract (57 tests) still needs a GPU runner.

- **The pre-push gate ran verify-fast for Python-only pushes.** It already
  stripped `.md` before deciding; `.py` now goes with it, `scripts/verify.sh`
  invoking no Python at all (checked). Same class as #1723 in the other hook.
  `guard_precommit_filter` covers both hooks now, and fails if either starts
  skipping buildable source.

- **Two generators had drifted out of the `tests/refs/` index** — the table that
  says which golden each one writes and which test consumes it, and the only way
  rule 1 of that README ("every golden value traces to a committed generator")
  can be checked at all. `gen_chat_goldens.py` and `gen_tokenizer_golden.py` are
  listed now, and `scripts/docs_lint.py` fails when a `tests/refs/gen_*.py` has
  no row, so the index cannot go stale again.

- **A cuBLASLt matmul failure discarded the benchmarked algo without saying so**
  (#1545). `reselect_algo_for_entry` replaced the timed probe's pick with
  heuristic `results[0]` for the rest of the process and logged nothing; only a
  *second* failure logged anything, so a shape could run on a different algo
  than the one the probe chose, differently on each process start. The cache
  entry now records whether its algo came from the probe or the heuristic, and
  replacing a benchmarked one logs it once with the shape. The reselection
  itself is unchanged - the retry is what keeps the matmul alive.

- **A `top_k` one candidate over the batching limit cost 14.5% of throughput,
  and changed what the model sampled from** (#1654). `sample_topk_topp_async`
  refused the CUB regime (`top_k > 128`), so the whole batch fell back to
  per-sequence synchronous sampling - one host round trip per sequence per step
  instead of one pinned gather for the batch. The CUB path was already all-async
  internally; only its trailing readback forced the sync, so it enqueues now.
  Qwen3-8B Q8, six concurrent sequences, `top_k=128` against `top_k=129` (one
  candidate of 151k apart, so the path is the only variable), median of three
  rounds:

  | | top_k=128 | top_k=129 | delta |
  |---|---|---|---|
  | before | 555.4 tok/s | 475.0 tok/s | **-14.5%** |
  | after | 546.5 tok/s | 544.6 tok/s | -0.3% |

  `sample_topk_topp_device` also **clamped** `top_k` to 128 with a warning, so a
  request with `top_k=200` sampled from 128 candidates alone in the batch and
  from 200 sharing it - the same request, two distributions, decided by its
  neighbours. Both honour the request now.

- **Multi-sequence decode on the residual path faulted with an illegal memory
  access when CUDA graphs were on** (#1708). Silent wrong output, then a sticky
  context: the first fault took every later request in the process with it, and
  the client saw `finish_reason: length` with a degenerate string, not an error.
  Three things on that path were resolved on the host at capture time - a
  device pointer array `cudaMallocAsync`'d per call and per layer inside the
  captured region and freed right after it, the ring index baked into those
  pointers, and a host-side ring advance that a replay never runs (the
  device-side advance was gated on `kv_seq_id`, which only the single-sequence
  path sets). All three are computed on the device now. Qwen3-8B-Q8_0,
  `kv_cache.bitdecoding_residual_tokens=64`, six concurrent requests:

  | | before | after |
  |---|---|---|
  | answers | 6/6 empty | 6/6 correct |
  | `illegal memory access` in the log | 15 | 0 |

  One request was clean before and stays clean; graphs off was clean before and
  stays clean. `ResidualKvWriteMulti.ReplayFollowsTheRingInsteadOfTheCapturedIndex`
  captures a graph, advances the ring on the device only, and asserts the replay
  followed it - it fails on a frozen index, which is what the old path had.

- **The prefill latency guard capped chunk size, never chunk count** (#1643).
  One engine step ran a chunk for *every* prefilling request, so k concurrent
  ingests inserted k chunk forwards between two decode steps of every decoder -
  and the pinned 1024 measurement behind the size cap was taken with a single
  ingest, so nothing in it depends on k. New `runtime.prefill_batch_decode_cap`
  (default 1) bounds the count while anyone is decoding, with a rotating start
  index. Qwen3-8B Q8, one streaming decoder against three ~5.2k-token ingests:

  | | before | after |
  |---|---|---|
  | worst inter-token gap | 259 / 254 ms | 112 / 88 ms |
  | gaps over 100 ms | 6 / 6 | 1 / 0 |
  | ingest wall time | 2017.7 ms | 2381.4 ms (+18.0%) |

  The stall is spread, not removed: gaps over 50 ms go 6 -> 17. Harness:
  `scripts/bench_prefill_latency.py`.

- **A promoted request that missed its first chunk was never scheduled again.**
  `Scheduler::schedule()` re-queued in-flight prefills on `prefill_offset > 0`,
  so a request promoted but not served in that tick stayed PREFILLING, admitted
  and holding KV, in no batch ever again. Nothing reached it while every
  promotion was served immediately; with the cap above, two of three concurrent
  ingests hung until the 300 s request timeout.

- **No engine log line could be attributed to an HTTP request** (#1582, second
  half). The server's `imp-N` and the engine's own request counter are disjoint
  counters, and under concurrency they do not even run in step - three requests
  sent at once mapped `imp-3 -> req 3`, `imp-2 -> req 4`, `imp-1 -> req 5`, so
  the mapping cannot be inferred by arithmetic. `add_request()` now publishes it
  once per request (`request imp-0 -> engine req 2`), which is what the eleven
  engine sites printing `req %d` need: several of them hold only the integer and
  have no `Request` to reach a string through. The two id spaces are joined, not
  merged. Embeddings and rerank carry no client-facing id and are unaffected.

- **The pre-commit GPU gate ran the full suite for Markdown and Python edits.**
  Its filter selected on the path prefix alone, and `tools/` and `tests/` also
  hold the `CLAUDE.md` tree and the gate/generator scripts - so editing
  `tests/CLAUDE.md` paid an image build plus the whole GTest suite for a result
  that cannot move. `.html` and `.txt` still gate (`cmake/embed_webui.cmake`
  compiles the web UI into `imp-server`; tests read corpora from `tests/refs` at
  runtime). New CPU-lane guard `guard_precommit_filter` pins 19 cases on both
  sides - over-excluding fails it too, because CI has no GPU runner and a C++
  change that skips this hook is gated nowhere.

- **gpt-oss tool calls were dropped entirely** (#1716). Harmony's envelope is a
  channel with a recipient, not a tag:

  ```
  <|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"Berlin"}<|call|>
  ```

  `parse_tool_calls()` dispatched on family and had no `HARMONY` branch, so the
  call fell through to the ChatML `<tool_call>` scanner and disappeared: the
  response carried an **empty `content` with `finish_reason: "stop"`** while the
  model's own `reasoning_content` read *"We should call get_weather with city:
  Berlin"*. Measured on `gpt-oss-20b-mxfp4`, `tool_choice: "auto"`, 10 requests
  per row:

  | path | before | after |
  |---|---|---|
  | `/v1/chat/completions` | 0 / 10 | **10 / 10** |
  | the same, streaming | 0 / 10 | **10 / 10** |

  Two separate places had to change, and the second is why a green unit test of
  the parser was not enough: `split_harmony_channels()` consumes the channels
  ~60 lines before the tool parse runs, so by then there was nothing left to
  find. The raw text is kept for it. The streaming router had the same shape -
  it routed by channel name alone, and `commentary to=functions.get_weather`
  matched neither the reasoning nor the content branch.

  Controls, all unchanged: no tools (streaming and not) still answers `4` with
  `finish_reason: "stop"`; tools present but no call wanted still answers
  `banana` with no `tool_calls` key.

- **Every second or third decode burst recaptured the conditional graph** (#1647).
  `rearm()` refuses when `context_len + step_limit` passes the captured ceiling
  `initial_context_len + max_steps`, and since #1636 `max_steps` was the KV
  reservation for the *current* burst - so the ceiling was two or three bursts
  wide by construction. The block-table buffer had the same shape: sized to the
  table as it stood, outgrown by the next burst. Both had to move; each one alone
  leaves the other gate refusing. Measured, `imp-cli --bench --bench-pp 16
  --bench-reps 3 --max-tokens 128` on Qwen3-8B-Q8_0, three alternating rounds:

  | | graph captures | tg128 tok/s (median) |
  |---|---|---|
  | before | 36 | 280.56 |
  | after | 8 | 288.66 (+2.89%) |

- **Jinja: `trim_blocks`, `lstrip_blocks` and the `safe` filter** (#1572). transformers
  renders every chat template with both whitespace flags on; imp had neither, so a
  template written against HF leaked one newline per block tag and every tag line's
  indentation into the prompt. Nemotron-3-Nano rendered 20 stray spaces in front of
  `<|im_start|>assistant`. `safe` was an unknown filter and dropped the value it
  marked. All nine golden families now match the reference exactly.

- **No committed artifact carried a per-kernel register or spill number**
  (#1549). The build never asked for resource usage, while 82 hand-set
  `__launch_bounds__` in `src/` steer register allocation by hand and
  `src/compute/CLAUDE.md` says never to add one blind.

  `make kernel-resources` reads the **built library** with `cuobjdump
  -res-usage`, so it needs no GPU and no special build flags - which is what
  makes a register-pressure gate possible in CI at all, where every throughput
  gate is impossible. Measured on the current build:

  | | |
  |---|---|
  | kernels | 823 |
  | at risk (REG >= 240 or a local frame) | **71** |
  | sitting exactly at the 255 ceiling | **6** |
  | with a non-zero local frame | 70 |

  The six at the ceiling are `gdn_scan_chunkwise_kernel`,
  `gdn_scan_fused_kernel` and `fmha_sm120_fa2_kernel` - the GDN scan and the FA2
  prefill, both hot. **No kernel spills today** (`LOCAL` is 0 everywhere), so the
  pin is a keep-it-that-way ratchet rather than a list of debt.

  `tools/kernel_resource_baseline.txt` is a **two-way ratchet**, like
  `tools/alloc_allowlist.txt`: a kernel that starts spilling fails, and so does
  a pinned kernel that improved, so the list cannot go stale in either
  direction. Verified against all three drifts - a new entry, a stale entry and
  a moved number - each failing with the kernel named.

  Runs in the CI `Build` job, the one required context, right after the build.
  `verify-fast` compares throughput at 8 %; one kernel dropping over the
  register cliff inside a 48-layer forward is far below that.

- **Pool growth allocated past every I2 instrument, and the staging ring cost
  more than it saved** (#1649, #1653). Two memory costs nothing was measuring.

  `Backend::commit()` and `commit_range()` acquire physical memory on a growable
  region exactly as `acquire()` does, and neither consulted the phase guard - so
  a growable KV pool committing pages under load was counted by none of the
  three I2 instruments. Both are non-virtual wrappers around
  `do_commit()` / `do_commit_range()` now, for the reason `acquire()` wraps
  `do_acquire()`: a backend cannot forget the guard. The guard runs on the
  **delta actually committed**, not on the request, which overstates; shrinking
  is not counted.

  The pinned staging ring for the weight upload was `4 x 128 MiB`, two constants
  that had never been varied, and pinning 512 MiB of host memory cost 503 ms to
  acquire and 115 ms to release against the 208 ms of H2D the ring exists to
  overlap. Swept on `Qwen3-8B-Q8_0`, load time only, 3 starts per point:

  | ring x chunk | 4x128 | 4x32 | 2x64 | 2x32 | 4x16 | 4x8 | **4x4** | 4x2 | 8x8 |
  |---|---|---|---|---|---|---|---|---|---|
  | median load | 4.55 s | 4.16 | 4.12 | 4.00 | 4.00 | 3.93 | **3.84** | 3.96 | 3.93 |

  Monotone down to 4 MiB and back up at 2: the pinning cost dominates the
  overlap the whole way, and below 4 MiB the per-chunk event pairs cost more
  than the pinning saves. Confirmed against the old default over 5 alternating
  starts each - **4.55 s against 3.87 s, ranges not overlapping** - so the
  default is `4 x 4 MiB`, **-0.68 s (-14.9 %) of every process start** and
  512 MiB of pinned host memory down to 16.

  A pair of keys (`vram.upload_ring_depth`, `vram.upload_ring_chunk_mib`) rather
  than constants, because the optimum is a property of the host's pinning cost.

- **`tool_choice` degraded to a prompt hint on every family but ChatML, and
  `/v1/messages` returned thinking nobody asked for** (#1592, #1541). Two ways
  the API answered something other than what the request said.

  `tool_choice: "required"` and a named function are enforced by the decode FSM
  only where the family's tool envelope has a grammar - `chatml` for
  `"required"`, `chatml` and `llama3` for a named function. Everywhere else the
  request was accepted with 200 and prose. Measured before the fix, 10 requests
  each at `temperature 0.7`:

  | model | family | `tool_choice` | tool calls |
  |---|---|---|---|
  | gemma-3-12b Q4_K_M | `gemma` | `required` / named | **0 / 10** each |
  | gemma-4-26B Q4_K_M | `gemma` | `required` | **0 / 10** |
  | gpt-oss-20b MXFP4 | `harmony` | `required` / named | **0 / 10** each |
  | Qwen3-4B Q8_0 | `chatml` | `required` / named | 10 / 10 each |

  0 of 40 on the families without a grammar, so it is a **400** now
  (`code: "tool_choice_unenforceable"`) on all three dialects. `"auto"` is
  untouched - Gemma-4 produced 1 of 10 there, and a best-effort call is what
  `auto` asks for.

  On `/v1/messages`, extended thinking is **opt-in**, as the dialect specifies.
  imp's server default (`think_budget = 0.5`) made a reasoning model reason on
  every request, so `content[0]` was a thinking block and `content[0].text` was
  empty for a client that never asked. Measured on `Qwen3.6-27B-Text-NVFP4-MTP`:

  | request `thinking` | `content` blocks | `content[0].text` |
  |---|---|---|
  | absent | `[text]` | `"Hi"` |
  | `{"type":"adaptive"}` | `[thinking, text]` | `""` |
  | `{"type":"adaptive","display":"omitted"}` | `[text]` | `"Hi"` |

  The block order was never the bug - Anthropic puts thinking first too. Only
  the default moved, and only on this dialect: `/v1/chat/completions` keeps
  `reasoning_content` as a separate field where no index shifts.

  Found while measuring: **gpt-oss tool calls are dropped entirely** - the model
  says in its own reasoning that it means to call the tool, and the response
  carries an empty `content` with `finish_reason: "stop"`. Filed as #1716; the
  400 above is what a caller sees for it now instead of nothing.

- **No shipped binary had a machine-readable output mode** (#1583). `--json`
  puts **exactly one JSON document on stdout** and every human line on stderr,
  on `imp-cli --bench` / `--perplexity` / `--prompt` and on `imp-bench`;
  `--interactive` refuses it, because a token stream is not one document.

  ```
  $ imp-cli --model "$MODEL" --bench --bench-pp 128 --bench-reps 1 --max-tokens 16 --json 2>/dev/null
  {"mode":"bench","model":"...","prefill_tps":5502.57,"decode_tps":438.59,"pp_tokens":128,
   "pp_ms":23.26,"tg_tokens":16,"tg_ms":36.48,"reps":1,"peak_vram_mib":11188}
  ```

  The promise is structural rather than audited: stdout is pointed at stderr
  for the whole run and the real stdout kept on a private fd, so a print site
  added later cannot break it.

  `scripts/gen_perf_baseline.sh`, `scripts/verify.sh` and
  `scripts/bench_gate.sh` read the JSON instead of regexing the table, which
  made the column layout a contract nobody had written down - and one whose
  spacing inside the parens varies with the magnitude (`(13310.12 tok/s)`
  against `( 148.58 tok/s)`). A missing key now aborts the run; an empty
  capture used to produce a median over fewer samples than the header printed.

  `--prompt --json` reports what stdout would have shown, not
  `decode(output_ids)`: the hidden stop and think markers stay hidden, so the
  document and the terminal agree.

- **Admission reserved for the prompt and the graph loop reserved for the
  whole generation** (#1635, #1636, #1662). Three ways the KV pool promised
  what it could not keep.

  `Scheduler::schedule` tested `can_allocate(prompt)` and ignored `max_tokens`,
  so a batch whose prompts all fit could run the pool dry mid-generation and
  the loser was cancelled after the client had already received part of the
  answer. It admits on prompt + `max_tokens` now, and the promise is **held**
  (`imp_kv_blocks_reserved`, new gauge) until the blocks are written - a test
  that is not held admits the next request against the same memory one round
  later. On a pool too small to ever hold prompt + `max_tokens` the reserve is
  clamped to the pool, which is the old behaviour: no admission rule can
  promise memory that does not exist.

  The trade, measured on `Qwen3-8B-Q8_0.gguf` with 8 concurrent
  `max_tokens: 1024` requests: at the default pool it costs nothing (551.3
  against 519.2 tok/s, inside the run spread). On a 150-block pool `main`
  truncates one to two of the eight answers per run and reports them as
  `finish_reason: "length"`, indistinguishable from a spent budget; this PR
  finishes all eight at **228.5 against 526.3 tok/s**. The reserve is sized by
  `max_tokens`, so the lever is `max_tokens`, and the server default of 8192 is
  its worst case.

  `can_allocate` also stopped counting live sequences as reclaimable. Its slow
  path summed the LRU list on the assumption that `evict_lru()` could hand
  those blocks back, while `evict_lru()` has had no production caller for
  exactly the reason that freeing a live sequence's KV corrupts it.

  `prepare_graph_loop` booked KV for the entire remaining generation before a
  burst that is bounded to `speculative.miss_burst`, `runtime.decode_burst` or
  16. Measured on `Qwen3-8B-Q8_0.gguf`, `max_tokens: 8192`, same 355-token
  answer both ways:

  | reservation | peak `imp_kv_blocks_live` |
  |---|---|
  | whole generation (before) | 514 |
  | the burst (after) | **26** |

  It was paid for out of the prefix cache, because `append_block` reclaims
  cached blocks when the free pool is empty. On a 600-block pool with a warm
  prefix, same 344-token answer: `imp_kv_blocks_cached` **176 -> 108** before,
  **176 -> 198** after.

  Third: a KV pool that does not fit halves and retries down to the 16-block
  floor instead of failing the load, with a WARN per attempt naming planned,
  retried and the shortfall in MiB. Everything that sizes that pool runs before
  the allocation, so all of it is a projection - #1631 fixed one wrong one, and
  #1662 shows that even a correct projection is not sufficient on its own.

- **Two binaries returned exit 0 after doing no work** (#1584). `imp-bench`
  counted invocations rather than measurements, so a host with no CUDA device
  printed "Benchmarks run: 4" and exited 0 with nothing measured; every bench
  entry point was `void` and could not report otherwise. They return `bool`
  now, the summary reads "run: N of M requested", and a shortfall exits 7.
  Measured: `imp-bench gemm` without `--gpus` is **exit 7** and "0 of 1", with
  a GPU **exit 0** and "1 of 1". `imp-server` likewise fell through a failed
  `listen_after_bind` to `return 0`, so a supervisor that restarts on non-zero
  did not restart a server that never listened.

- **Per-request log lines bypassed `diagnostics.log_level`** (#1582, first
  half). Eleven `fprintf(stderr, ...)` sites fire once or more per HTTP
  request, which is exactly when volume matters, and none of them reached the
  level check every `IMP_LOG_*` site in `src/` passes through. They are on the
  facility now, so they carry a timestamp and an origin and obey the level.
  Measured: three requests produce **3 lines at the default level and 0 at
  `--set diagnostics.log_level=warn`**. Startup and pre-logging failures stay
  on stderr deliberately. The second half of that issue - two disjoint
  request-id spaces - is a design decision and is left open.

- **The degeneration battery had zero call sites, and the gate that stood in
  for it saw one shape** (#1573). `tools/analysis/degen_suite.py` is 41 checks
  that exit non-zero correctly and that nothing ever ran; it is wired into
  `make test-server` now, where the running server it needs already exists.
  First run: **35 checks, 0 fail, 5 s**. Against `--kv-nvfp4` plus the residual
  knob it found a real one on its first attempt (stream and non-stream
  disagreeing at greedy) - added to #1708.

  The pre-push smoke gate also gained the two criteria the skill it stands in
  for already specifies: no token more than 4 times in a row, no 3-gram more
  than 3 times. The old distinct-token count could not see a loop: a stream
  with **15** distinct tokens in its last 32 passes it while repeating a 3-gram
  four times. The token floor is per-prompt, because this gate's own prompt is
  the "single-word factual" case the skill excepts - it answers in 11 tokens,
  one above a flat limit of 10.

  `verify.sh` also prints a UTC wall clock at the start and in the summary. Two
  GPU jobs from different sessions overlapped twice in one day, and "did that
  land inside my gate?" could not be answered from a log that carried no time;
  the repeat run that settled it (283.35 against 283.04 tok/s, harmless) cost
  more than the two lines. UTC because the script re-execs into a container
  whose clock is UTC while the host runs local time.

- **`json_schema`: a `\uXXXX` escape compiled to a literal `?`** (#1563). The
  schema string parser skipped the four hex digits and appended `?`. That is
  not an edge case: `json.dumps` defaults to `ensure_ascii=True`, so a schema
  round-tripped through any Python client arrives with every non-ASCII
  character escaped - and the same parser reads enum values, property names,
  `required` entries and `pattern`, so the compiled grammar then *forced* the
  model to emit `?` where the caller asked for a character. Escapes decode to
  UTF-8 now, surrogate pairs included; a lone surrogate becomes U+FFFD and a
  truncated escape ends the string rather than inventing one. Four CPU tests,
  all four red without the fix.

- **The regex constraint re-classified the whole vocabulary on every request**
  (#1568). `prepare_grammar` has skipped that work for an unchanged grammar
  since it was written; `prepare_regex` never had the check, so ~151K tokens
  were re-classified on the scheduler thread for every request - and a client
  that pins one pattern sends it on every request. Measured on
  `Qwen3-4B-Instruct-2507-Q8_0.gguf`, same pattern eight times, median of
  requests 2-8: **73.0 ms before, 37.0 ms after (-49.3%)**.

- **A growable KV pool zeroed new blocks on stream 0 and published them to a
  non-blocking stream** (#1652). `cudaMemset` runs on the legacy default
  stream; the engine decodes on a `cudaStreamNonBlocking` stream, which by
  construction has no ordering relationship with it - so a memset could retire
  *after* the first KV write into the same blocks and zero live KV. On a
  36-layer model that is 72 unordered memsets over exactly the blocks the next
  decode step fills. Stream 0 is synchronised before the new capacity is
  published; growth is rare, and the path already commits driver pages.

- **`max_batch_size` above the decode-graph pool ran eager, silently** (#1646).
  Every graph path is gated on `n_sequences <= 64`, so a larger configured
  batch fell to an eager forward with no clamp, no warning and no log line.
  Measured on this box, graphs on against off: **454 vs 190 tok/s**, i.e. the
  configured value costs 2.4x decode the moment it exceeds the pool. Not
  clamped - the value also bounds admission and KV sizing - but said out loud,
  once, where the number is resolved.

- **The multi-sequence residual metadata buffer is persistent** (#1648). It was
  `cudaMallocAsync`'d every decode step and its address baked into a captured
  `forward_logits` graph that is then replayed, with no invalidation watching
  it. That only ever worked because the default pool's release threshold is
  pinned to `UINT64_MAX` so the same address came back - a pool setting, not an
  invariant the graph path asserts. Allocated once beside `d_kv_slot_buf_`,
  strided by capacity so a graph captured at one batch width stays correct at
  another. It does **not** fix that path: measured on
  `Qwen3-8B-Q8_0.gguf` with `--kv-nvfp4` and residual on, six concurrent
  requests fault identically before and after (106998 / 111543 `illegal memory
  access` lines), while `runtime.cuda_graphs=never` gives 0 and six correct
  answers. Filed as #1708.

- **`json_schema`: an unconstrained `integer` had no digit bound** (#1540). At
  the server's default temperature the sampler could sit in the digit state and
  emit `1020000000000000000000000000000000000000` for a population field - a
  value no int64 consumer can read back; at temperature 0 the same request
  answered `13528079`. The FSM stops the digit run at 19, int64's width, and
  masks further digits so the model closes the value instead. `number` is
  unchanged: there the digits carry precision, not magnitude.

- **A checkpoint's unloaded MTP head is visible in `/health`** (#1537).
  `speculative.mtp_k` defaults to 0, so a checkpoint that ships a head runs
  without a documented +8 to +22% decode, and the only notice was one INFO line
  an operator running a container never sees. `GET /health` reports
  `mtp_head_available` with the trade. Measured on `Qwen3.8-27B-NVFP4`: present
  by default, absent with `--set speculative.mtp_k=2`, where the head loads
  (15 tensors, 0.79 GiB). The default is unchanged - turning it on for everyone
  costs VRAM and is a decision, not a fix.

- **What `runtime.deterministic` covers is now written down, and gated**
  (#1574). It reaches four kernel sites through
  `process_diag_deterministic_gemm()`; `gemm_cutlass_grouped_3x.cu` - the
  primary GEMM for NVFP4 weights and every GGUF quant - reads none of them,
  while the doc said "GEMM" without scoping it. Measured on
  `Qwen3.8-27B-NVFP4`, three fresh processes per arm, teacher-forced NLL: **on
  1.3113 / 1.3113 / 1.3113, off 1.3113 / 1.2889 / 1.2889** - so the mode does
  make that checkpoint reproducible, through the sites it does cover, and what
  is missing is the guarantee rather than the effect. Greedy bytes were
  identical in all six runs and could not see any of it. Known limit 5 in
  `docs/determinism.md` names the uncovered path, and
  `tools/check_determinism_sites.py` fails when the code's sites and the doc's
  list drift apart. Second hole closed: `reselect_algo_for_entry` replaced the
  warmup-validated algo with a heuristic pick on a runtime matmul failure, with
  no deterministic check - it now refuses in deterministic mode instead.

- **The only end-to-end determinism gate never ran** (#1575).
  `*DetEvalE2ETest*` takes its `GTEST_SKIP` branch unless a model env var is
  set, and the pre-commit hook's "full suite" stage (`make test-gpu`) set none -
  so the gate existed and executed nowhere except a target run by hand.
  `test-gpu` runs it explicitly now, with only the two variables it needs.
  Measured: 6 tests, 74 s.

- **`--use_fast_math` is named as part of the determinism envelope** (#1576).
  Every CUDA TU is compiled with it in both shipped configurations, which is
  fine and deliberate - but it means the guarantees are about one binary, not
  one commit. `docs/determinism.md` says so, and the eval recipe now says to
  pin the image.

- **The ITL histogram measured a per-request mean on the wrong ladder**
  (#1577). `imp_inter_token_seconds` observed one value per request - the mean
  - on the request-duration bucket ladder, whose first bound is 5 ms. imp
  decodes at 300-450 tok/s, so every observation landed in that first bucket
  and `histogram_quantile` returned a function of the bounds rather than of the
  data. It is one observation per token now, on a millisecond ladder. Measured
  on a live server, four requests: 4 observations in one bucket before, **39
  observations spread over three buckets** after (5 under 2 ms, 34 under 3 ms,
  39 under 5 ms).

- **TTFT was recorded on the streaming path only** (#1578), while
  `imp_requests_total` counted both - so the histogram described half the
  traffic and did not say which half. The non-streaming path records it at its
  first token. Measured: 3 non-streaming plus 1 streaming request produce 4
  observations, not 1.

- **`imp_requests_failed_total` counted 5xx only** (#1579). Every refusal this
  server is designed to make is a 4xx (`tools/imp-server/CLAUDE.md`), so the
  error counter was blind to the entire designed error surface.
  `imp_requests_rejected_total` is its own series, because "the server broke"
  and "the server refused" want different alerts.

- **Nothing measured queueing or batching** (#1580).
  `imp_queue_time_seconds` is the admission wait with prefill excluded, so a
  busy server can be told from a slow one; `imp_decode_batch_{steps,rows}_total`
  and `imp_decode_batch_max` say how many sequences actually decoded together.
  Measured with six concurrent requests: `decode_batch_max 6`, 775 rows over
  289 steps (2.68 mean), and 5 of 10 queue-time observations above 5 ms.

- **The shipped Grafana dashboard plotted only last-value gauges** (#1581). Six
  panels added, all percentile timeseries off the histograms that already
  existed: request duration, TTFT, ITL, queue time, decode batch size, and
  refusals against failures.

- **The YaRN RoPE branch computed its angle in float and never reduced it**
  (#1630). #1316 fixed exactly this in `rope_forward`'s other two branches; the
  YaRN one kept calling the fast intrinsics on an unreduced argument, and the
  long-context regression test could not see it because it runs at the default
  `ext_factor = 0.0f` and so takes the linear branch. Both halves are fixed -
  the angle is formed in double at all four call sites and reduced before the
  intrinsic - and a second test drives the YaRN branch against double truth to
  position 131071. Against the unfixed kernel it fails at that position.

- **Quantised paged-decode kernels dereferenced the `-1` block-table sentinel**
  (#1678). StreamingLLM eviction writes it, the FP16 kernel has skipped it as
  defense-in-depth since #963, and the FP8, FP8-tile, INT8, INT4, NVFP4 and
  NVFP4-TC twins read it straight into a pointer. 12 guards plus the tiled
  kernel, which prefetches through a `cp_async` ring and so clamps the address
  and drops the tokens with its validity mask instead of skipping the block.
  Measured cost over 10 alternating runs on `Qwen3-8B-Q8_0.gguf` with
  `--kv-fp8`: **not separable from the noise** - median 384.88 against 396.41
  tok/s while the arms' own spread is 4.1% and 6.4%, and the guarded arm is
  faster in 4 of the 10 paired rounds.

- **The FMHA tile table named three `Bq` the selector never picks** (#1679).
  Its first three branches compare against `max_smem / 2`, so at hd=64 it takes
  Bq=64 (the comment said 128) and at hd=96 and hd=128 it takes Bq=32 (the
  comment said 64). Corrected in the kernel comment and in
  `docs/internals/KERNELS.md`, computed from `compute_smem_sm120` against the
  measured `cudaDevAttrMaxSharedMemoryPerBlockOptin` of 101376.

- **`/v1/messages` held the first SSE byte behind a 100 ms poll** (#1558). The
  wait existed so `message_start` could carry cache accounting, on the claim
  that it cost no measurable TTFT - and it inverted against its own
  justification: it exits on the first iteration when the queue is empty and
  runs the full 100 ms when the request is queued, which is when TTFT matters.
  Measured on `Qwen3-4B-Instruct-2507-Q8_0.gguf`, 8 concurrent streams, time to
  `message_start`: **median 118.5 ms before (max 121.0), 11.4 ms after (max
  12.8)**. The final `message_delta` already re-reports the accounting.

- **`thinking: {"type": "adaptive"}` was a no-op** (#1560). It is the on-mode
  current SDKs send, matched neither branch, and set nothing - the request ran
  at the server's default `think_budget` while the client believed it had
  configured thinking. `display: "omitted"` is honoured on both transports
  (the model still reasons, the block is not returned), and `budget_tokens: 0`
  with `type: enabled` now disables thinking instead of leaving the default.

- **`thinking` blocks carry a `signature`, and the stream emits
  `signature_delta`** (#1555). The field did not exist anywhere in the server
  while Anthropic's SDKs round-trip it. It is a deterministic digest of the
  block text, not an attestation: it proves the block came back unedited and
  nothing more, and the code that emits it says so.

- **`/v1/models` advertised a context the KV pool cannot serve** (#1542). The
  resolver's `max_seq_len` is a plan and the pool is clamped after it, so a
  prompt between the two was accepted as servable and was not. Measured on
  `Qwen3.8-27B-NVFP4`: the log plans 131072, the pool holds 96960, and all four
  probes (`/v1/models`, `/props`, `/info`, `/health`) now answer 96960.

- **`anthropic-version` and `anthropic-beta` are read** (#1562). Both were
  ignored, so a beta-gated request got a 200 and a response that does not
  implement it. They are echoed back, and an unknown beta warns once per value.
  Neither is enforced - refusing a request that omits a header imp does not
  need would break more than it fixes - and `docs/API.md` states the asymmetry.


- **`/v1/messages` never reported which stop sequence ended a turn** (#1550).
  A match came back as `stop_reason: "end_turn"` with `stop_sequence: null` on
  both transports; the Anthropic value `"stop_sequence"` was produced by no
  code path. The matched text now rides out of the holdback matcher and the
  non-streaming path alike. Measured on `Qwen3-4B-Instruct-2507-Q8_0.gguf` with
  `stop_sequences: ["4"]`: `stop_reason: "stop_sequence"`, `stop_sequence: "4"`,
  streaming and not. While making the match reportable, the matcher started
  cutting at the **earliest** occurrence rather than at the first list entry
  that occurs anywhere - list order shipped the text between two stops.

- **A 429 on `/v1/messages` came back in the OpenAI envelope** (#1551). Both
  pre-routing guards wrote `{"error":{...}}` with no top-level `"type":"error"`,
  so an Anthropic SDK could not classify it, twenty lines above an auth path
  that did branch. One helper now picks the envelope from the path, and the six
  sites that spelled the test out use it.

- **Anthropic error types are Anthropic's** (#1556). `server_error` and
  `capacity_error` are this server's inventions and were emitted at seven sites
  plus forwarded verbatim through the non-streaming shim. They map to
  `api_error` and `overloaded_error`; anything unrecognised falls back on the
  status. `content_filter` -> `refusal` for the same reason.

- **A mid-stream fault is an `error` SSE event, not a completed turn** (#1552,
  #1553). The event did not exist: a request timeout arrived as `stop_reason:
  "max_tokens"` (indistinguishable from the model reaching its budget) and an
  admission refusal as `"capacity"`, which is not an Anthropic stop_reason at
  all, while the non-streaming path answered 503 for the same condition.
  Measured with `--request-timeout 1`: `event: error` with
  `{"type":"timeout_error"}`.

- **`tool_result.is_error` was read by nothing** (#1557). A failed tool became
  an ordinary successful `role: "tool"` turn, so the model was told the call
  worked. The failure is labelled in the content, which is the only channel the
  OpenAI tool turn has.

- **Every `/v1/messages` response carries a `request-id`** (#1561), and error
  bodies repeat it as `request_id`. Neither existed anywhere in the server.

- **Jinja: `{% set x %}...{% endset %}` printed its body and left the variable
  empty, and macro default parameters never parsed** (#1565, #1566). The block
  form of `set` is what Gemma-4's shipped `chat_template.jinja` builds
  `captured_content` with: `endset` was one of the tags the parser skipped
  silently, so the body rendered inline and `captured_content | trim | length`
  was always 0. Macro defaults tested for token `OP "="`, which the lexer never
  emits for a bare `=` (it emits `ASSIGN`), so `is_nullable=false` became two
  extra positional parameters. Measured on a minimal template: before
  `... of France?] LEN[0]`, after `OUT[... of France?] LEN[30]`.

- **Jinja: an unsupported tag is a parse error naming the tag, not a silent
  skip** (#1565). `parse()` returned true unconditionally, so
  `ChatTemplate`'s "fall back to the hardcoded template" branch could never
  run and an unimplemented construct produced a wrong prompt with no log line.
  `{% raw %}`, `{% include %}`, `{% filter %}`, `{% block %}` and an unbalanced
  end tag now fail with `unsupported or unbalanced tag: {% <name> %}`;
  `{% generation %}` stays a no-op because it does not affect rendered text.
  All 15 chat templates in this project's model directory still parse.

- **The HF tokenizer-parity test ran for the first time** (#1569, #1570). It
  needed `IMP_TEST_GOLDEN`, which nothing in the repo set, against a golden
  file that was never committed - so it skipped on every run there has ever
  been. The golden is now a committed header
  (`tests/refs/tokenizer_golden_qwen3.h`, generated by
  `tests/refs/gen_tokenizer_golden.py`), which also removes the hand-written
  JSON scan that cut each case at the first `}` and therefore never checked
  the one case containing `{"key": "value"}`. The bar is every case rather
  than 80% of them, `decoded` is asserted where it used to be generated and
  ignored, and the corpus grew 20 -> 32 with more whitespace runs (the #657
  class) and chat-control literals. Result on
  `Qwen3-4B-Instruct-2507-Q8_0.gguf`: 32/32 encode and 32/32 decode.


- **Growing a KV pool with sliding-window layers wrote past the layer's own
  region** (#1699). `commit_blocks_` zeroes newly committed blocks with one
  memset per layer sized `(blocks - first_new) * layer_block_bytes_[l]`, but a
  windowed layer's region is `swa_max_blocks_` blocks, not `max_blocks_` (24
  against 256 on the failing configuration). The commit loop directly above has
  that clamp and says so; the memset loop had the same arithmetic and none. For
  the last windowed layer the write leaves the reservation:
  `cudaErrorIllegalAddress`, which is sticky, so one fault took 36 suites down
  with it. For a windowed layer that is not last, the overrun lands in the next
  layer's live KV and zeroes it silently, which is the worse half. Measured on
  the same invocation: 73 failures and 21 illegal accesses before, 0 and 0
  after.

- **Admission could starve a long prompt, and two knobs named `max_batch_size`
  parked admitted rows** (#1634, #1637). The pending queue is re-sorted
  shortest-first on every arrival, which is deliberate against head-of-line
  blocking and unbounded on its own: under sustained short traffic a long
  prompt is overtaken every round forever. Aging bounds it - a request waiting
  `Scheduler::kAgingRounds` rounds sorts ahead of everything younger, ties by
  length - so the property survives and the starvation does not.
  `docs/roadmap.md` said "scheduling is arrival order", which it never was;
  corrected in place with the date. Separately, `EngineConfig::max_batch_size`
  caps admission while `runtime.max_batch_size` truncates the decode batch, and
  when the second was smaller the rows admitted beyond it were prefilled, held
  their KV and never decoded until a head row finished. Admission is clamped to
  the smaller of the two and logs that it did.

- **The `compute_120f` PTX fallback is assembled in CI** (#1650). It ships as
  `code=compute_120f`, the PTX-only form, so `ptxas` never ran over it and the
  first thing that would was the driver's JIT on a GB203 - a card nobody in
  this project owns. `scripts/check_ptx_fallback.sh` extracts every PTX image
  from a built artefact and assembles it; no GPU needed, since ptxas is a
  compiler. Measured on `libimp.a`: all 155 images assemble for `sm_120`. It
  runs as the separate `PTX fallback` job, which builds the `imp` library with
  the fallback on, because the required `Build` job configures
  `IMP_DISABLE_120F_FALLBACK=ON` and its binary carries no PTX at all. The
  second gencode costs +53.1% device-compile time over the three heaviest TUs
  (47278 ms against 30886 ms), which is why `Build` keeps its opt-out.

- **`"speculative": false` left two of three drafters running, and three
  server decisions had no counter** (#1639, #1640, #1641). The documented
  per-request switch fed only the n-gram matcher: the MTP head and token
  recycling kept drafting and the verify step kept running for a caller who had
  turned speculation off. It now covers all three (`false` disables; `true`
  still cannot conjure an MTP head the checkpoint lacks), and the request field
  is named `spec_override` rather than `spec_ngram_override`. `/metrics` gained
  `imp_requests_timed_out_total` (a `--request-timeout` kill is
  `finish_reason: "length"` on the wire, indistinguishable from a spent budget),
  `imp_kv_pressure_rejections_total` and `imp_kv_pool_growths_total`. The
  pressure counter fires at four of the six cancellation sites: a failed
  metadata allocation and a snapshot mismatch are different faults, and both
  now carry a comment saying they are excluded on purpose.

- **The perf gate measured a different quantity than the pin it compares
  against, and two CI jobs claimed coverage they do not have** (#1600, #1624,
  #1625, #1685). `scripts/bench_gate.sh` benched with n-gram speculation ON
  while `tests/perf_baseline.json` states `speculative.ngram=false` in its own
  `methodology` field; both scripts pass the flag now, and
  `docs/internals/BENCHMARKING.md` carries a table of the remaining differences
  instead of calling them one gate. The gate prints the pin's date, age and
  model, and warns above 30 days: the measurement contract is single-session,
  and host drift over a month (4.01 % measured on this box between runs hours
  apart) is larger than the gate can tell from a code change. The CI job that
  ran `pytest -m nomodel` is called `Real API contract (model-less)` now and
  prints how many tests it deselected; the generation half, and the absence of
  any server-side perf gate, are in `LIMITATIONS.md` rather than implied by a
  green check.

- **Eight places where the OpenAI surface answered instead of refusing, or said
  nothing about itself** (#1590, #1591, #1593, #1595, #1596, #1598, #1599,
  #1602). `response_format` with an unknown `type`, or a known type whose
  payload is missing, was dropped silently and the request answered as free
  text with 200; it is a 400 now, because a constraint that did not apply and
  one that did look identical to the caller otherwise. `best_of > 1` was
  accepted and ignored: also a 400, since imp generates no candidate set.
  `finish_reason` shipped `cancelled` and `capacity`, neither in the OpenAI
  enum, sending clients through their default branch; both map to `length`.
  The streaming path wrote a server-authored English sentence into
  `delta.content` where the non-streaming path wrote nothing, so the two
  transports disagreed about the same request; it goes to the log now.
  `error.param` and `error.code` exist on the shared envelope, so a context
  overflow is `context_length_exceeded` rather than an English sentence.
  `GET /v1/models/{id}` is registered, so `client.models.retrieve()` no longer
  404s on the served model. `system_fingerprint` is emitted on all four
  response shapes. And `docs/API.md` states the four sampling defaults that are
  not OpenAI's, including one the issue did not name: **`top_k: 0` is not off,
  it is 50**, a tighter truncation than the 40 default.

- **Streamed logprobs were absent whenever a `stop` sequence was set, and
  `/v1/completions` returned the wrong shape** (#1588, #1589, #1601). The
  streaming driver attached per-token logprobs only on the branch taken when a
  request carried no `stop`; with any stop present every chunk went out through
  the logprob-free writer. Measured against a real server, Qwen3-4B-Q8_0,
  `stop` set: 0 of 8 chunks carried logprobs before, 5 of 10 after.
  `/v1/completions` returned the **Chat** object (`{"content":[...]}`) on a
  `text_completion` response, so an SDK reading `.logprobs.tokens` found
  nothing; it returns `{tokens, token_logprobs, top_logprobs, text_offset}` now,
  and streams one chunk per token with its own offset (verified: every streamed
  offset equals the length of the text reassembled so far). Both shapes come
  from `utils.cpp` so they cannot drift apart again, and the token attribution
  behind them is a pure component in `stream_pipeline.h` with 14 CPU-lane tests
  covering it plus `safe_token_json` / `token_bytes_json`, which had no test in
  any lane.

- **Four tools that described themselves wrongly** (#1585, #1586, #1587,
  #1663). The pre-push gate's gtest filter carried `AttentionTest.*`, a suite
  renamed away before the pattern was added on 2026-04-27: gtest reports
  success for a filter that matches nothing, so the only gate that runs CUDA
  kernels against correctness ran **zero attention tests for four months**. The
  corrected pattern adds 67 tests and 2 seconds (3 s / 268 to 5 s / 335), and a
  new `guard_verify_filter` fails when any pattern matches nothing. `CLAUDE.md`
  priced `verify-fast` at 90 s while the target's own prerequisite is a full
  image build; measured, the script half is 37 s and the build is what costs
  minutes. `docs_lint.py` walked gitignored paths, so a local scratch directory
  produced 160 errors on every run and the working answer became a `grep -v`.
  And the ten-value `ImpError` taxonomy never reached a process exit code:
  every binary collapsed onto 1, so a caller had to parse English to tell "no
  such file" from "out of VRAM". Exit codes are the taxonomy, 1 to 9, and
  `imp-quantize`'s undocumented 2 for usage errors is now 1 like everywhere
  else. The teardown guard added with #1632 was itself a pure negative test
  (a grep that passes when it finds nothing, which is also what it does when
  pointed at the wrong file); it now asserts the six replacement calls are
  present too.

- **Three things a request left behind when it did not end normally** (#1632,
  #1633, #1644). Six cancellation sites in the scheduler freed the sequence's
  KV and never released its recurrent-state slot; the pool is fixed-size, and
  an empty one puts every later sequence on `id % cap` aliasing, which is two
  live sequences sharing one SSM state. They go through one teardown helper
  now, and a CPU-lane guard fails if a seventh site frees KV directly. A
  request cancelled while still queued was promoted anyway, because only the
  active list was filtered by status, and the promotion overwrote `CANCELLED`
  with `PREFILLING`: a full generation ran, holding KV and a batch slot, for a
  client that had already disconnected. And on the non-pool prefill path a
  sliding-window model leaked one SWA block table per chunk, because
  `free_prefill_buffers` had no parameter for it and only the allocation-failure
  path freed it.

- **One request could cost the server an unbounded amount of work, and two of
  the limits keyed on what the client wrote** (#1614, #1615, #1616, #1617,
  #1618, #1619, #1622). The per-IP rate limit preferred `X-Forwarded-For`
  whenever present, so varying one header both bypassed the limit and added a
  permanent tracker entry; the header is now believed only from a peer named by
  `--trusted-proxy`, and buckets that go quiet are evicted. Rate limiting
  covered seven exact paths, so `/tokenize`, `/v1/messages/count_tokens` and
  `/admin/*` were reachable at any rate; it now covers everything except
  `/health` and `/metrics`. `n`, rerank `documents`, embeddings `input` and
  `logit_bias` each multiplied one request's work with no ceiling and are
  capped by `--max-n` (8), `--max-batch-items` (512) and `--max-logit-bias`
  (1024). `logit_bias` also cost one **blocking** device-to-host copy per entry
  per decode step in three copied loops; it is one upload and one kernel now.
  The 404 envelope echoed the raw request path through `json::dump()`, which
  throws on ill-formed UTF-8, turning a 404 into a 500 with an empty body. And
  the shipped compose file published on every interface with no way to switch
  authentication on: the host binding is `127.0.0.1` by default and
  `IMP_API_KEY` reaches `--api-key`. Read, write and keep-alive limits are
  configured rather than inherited from whatever cpp-httplib defaults to.

- **A checkpoint could size imp's allocations, its parser stack, and the path it
  opens** (#1611, #1612, #1613). A layer index parsed out of a tensor name went
  straight into `resize`, and `sizeof(TransformerLayer)` is 9680 bytes, so
  `model.layers.2147483000.…` asks for 18.9 TiB; `num_hidden_layers` in
  `config.json` and `block_count` in GGUF reach the same `resize` without a
  tensor name at all. Declared counts are now refused above 1024 layers / 4096
  experts, an index out of a name is dropped with a counted warning, and
  `std::atoi` is gone: it returned 0 for `"4294967296"`, so that name silently
  overwrote layer 0. Shard filenames from `model.safetensors.index.json` are
  concatenated onto the model directory and were opened and mmapped as given,
  `../` included; they must now be bare filenames. Both recursive-descent
  parsers are depth-capped: measured, `JsonParser` takes SIGSEGV before 40 000
  nesting levels on an 8 MiB stack while a SafeTensors header may declare
  128 MiB, and the Jinja template parser, whose input is the `chat_template`
  inside the model file, does the same.

- **A paged decode launcher answered an unserved `head_dim` by leaving the
  output unwritten** (#1674). Seventeen sites logged an error and returned, so
  `O` kept the previous layer's `attn_out_` and the answer was silently wrong -
  the failure mode `SETTLED` S-22 exists to forbid, while the prefill chain
  throws for the same class of miss. They throw now, and
  `paged_attention_serves_head_dim()` refuses the dtype at init with a fallback
  to FP16 KV, the way the sink guard beside it already did. NVFP4 serves no
  head_dim 96; nothing said so anywhere.

- **Sink models lost the FMHA chunk carve-out** (#1675). `max_safe_prefill_chunk`
  still treated learned sinks as cuBLAS-only and skipped all three no-clamp
  returns for them, although #992 made the FP16 WMMA FMHA tier sink-capable and
  the dispatch routes sinks straight there. gpt-oss-20b prefill on a quiet card,
  median of 3 runs each: **24289 -> 35579 tok/s at pp4096 (+46.5 %)**, chunk
  sizes 432/832/1072/1760 -> a flat 2048. Decode unchanged (157.9 -> 157.4, both
  arms spread ~3 %).

- **`attention.fa2_fp16qk="never"` was not an off switch above the threshold**
  (#1676). It is documented as restoring the materialized path and did so only
  below `fmha_prefill_threshold`; above it the FMHA chain re-entered the same
  FA2 kernel with `fp16_qk=true`. The explicit fp8-QK opt-in (`never` together
  with `fp8_fmha=on`) still takes FA2, which is what that pair is for.

- **`ATTENTION_DISPATCH.md`'s decode table named one kernel per dtype** (#1677)
  where each launcher fans out over up to five, and its FP16 row named a symbol
  that does not exist in the tree. Rewritten as launcher plus the kernels it can
  pick, with the head_dim coverage that #1674 made explicit.


- **A second `ImpContext` in one process was accepted and then broke both**
  (#1629). `imp.h` told callers to create one context per thread; the engine
  arena and the graph-slot pool are process-global, the second
  `engine_arena_open()` returned `InvalidArgument` into a discarded value, and
  the first `imp_context_free()` released both out from under the other. A
  second LIVE context is refused now; sequential create/free/create is
  unaffected. The arena's INFO line also printed "N MiB reserved" before the
  open and regardless of its result.

- **Green-context reconfiguration destroyed both streams with work in flight**
  (#1656): `reconfigure()` runs from `step_schedule()` when the prefill/decode
  mix changes, `cudaStreamDestroy` does not wait, and the replacement streams
  carry no ordering against the old work. They are drained first now. Not
  reachable on sm_120 - see the new `LIMITATIONS.md` entry on why green
  contexts fall back to ordinary streams here.


- **`/v1/messages` and `/v1/responses` built and dumped a JSON object for every
  emitted token** (#1657), which is what the shared writer's own header forbids
  on the hot path and what `/v1/chat/completions` stopped doing when it got
  `SSEChunkWriter`. Both now build the constant part of the frame once per block
  and only escape the token between the halves. Measured in isolation: 0.568 us
  per delta against 0.006, a factor of 87. Wire output is byte-compatible -
  same event count, same key sets, same reassembled text including non-ASCII,
  checked against a recording from the pre-fix binary.


- **`--set` accepted any value for 157 of the 185 bound keys** (#1627). The key
  half was rejected, the value half was not: `parse_bool`/`parse_int`/
  `parse_float` returned the current value for input they could not read, with
  no warning, so `--set server.prefix_cache=disabled` kept the default and said
  nothing. `stoi` also stopped at the first non-digit, making `16k` parse as 16.
  Both are a rejection now, in `--set` and in `imp.conf`.

- **`speculative.batch_rr` could not be set** (#1638). A default-on scheduling
  switch read on the decode path (`engine_scheduler.cpp:1422,2882`) whose own
  comment calls it a kill switch for A/B, bound to no config key.

- **`runtime.debug_raw` did four of its seven effects through dead env vars**
  (#1628). `IMP_NO_WARMUP`, `IMP_DETERMINISTIC_GEMM`, `IMP_NO_EXPERT_CACHE` and
  `IMP_GDN_REF` are read by nothing in the tree, so warmup, deterministic
  cuBLAS, the MoE expert cache and the GDN scan all stayed at their normal
  settings while the log line and `imp.conf.example:97` said otherwise. All
  four are config assignments now; the switch each stood for existed.

- **The `clang-tidy` CI job had never linted a file** (#1626). Its first `git`
  call died with exit 128 (`dubious ownership` - the container UID is not the
  checkout's owner), the file list came back empty, and `continue-on-error`
  reported green. It trusts the checkout now, and a failing `git diff` is an
  error rather than an empty list.

- **`prefill_chunk_size` had no `imp.conf` key** (#1645) while the knob that
  merely caps it did. It sets the TTFT/ITL trade for every prefill and was
  reachable only from the CLI and a per-request override. `runtime.prefill_chunk_size`
  now, CLI wins over the file. Two comments documenting the retired default 512
  are corrected to 2048.

- **`runtime.debug_raw` printed `graphs=0(none)`** (#1658): "graphs are off,
  reason: graphs still enabled". It wrote `use_cuda_graphs = 0` directly instead
  of going through `demote_graphs_`, so the reason stayed `None`. It has its own
  reason now.

- **`make gen-perf-baseline` had no GPU guard** (#1623) while every other bench
  target does - the one target that re-pins what the perf gate compares against.


- **Twelve stale or self-contradicting documentation claims** (#1543, #1594,
  #1651, #1668-#1673, #1680-#1682, #1684). The load-bearing ones: `CLAUDE.md` and
  `AGENTS.md` told every agent that sm_120a has no TMA-WS grouped GEMM, which
  the shipped cubin contains; `SM120.md`'s decode roofline was off by 1023x and
  the "~28:1 memory:compute ratio" was derived from that quotient (it is
  ~28,000:1); `MODELS.md` told operators to bound KV with `--max-seq-len`, which
  exits 1 on `imp-server` (the two engine log lines that gave the same advice are fixed below, #1681); the quickstart's first command needs an image nothing
  in the quickstart builds; seven `FEATURES.md` rows were green without a gate.
  Two numbers were **withdrawn rather than corrected**: the MoE host-offload LRU
  row had four values for one measurement and the checkpoint is not on this host
  to re-run it (#1669), and the CI-lane case count is a command now rather than
  a literal, having gone 248 stale in nine days (#1673).

- **`sync_docs.py` published a provenance block it made up** (#1684):
  `cuda=13.3` and `commit=1e4fad60` were string literals overwriting a baseline
  that records `"cuda": "unknown"` and no commit at all. It reads the file now.
  `gen_perf_baseline.sh` captures both going forward - its CUDA probe was
  `a | b | c || fallback`, and sed exits 0 on empty input so the fallback never
  ran. Verified in the build container: 13.3 instead of empty.

- **`docs_lint.py` promised checks it did not run** (#1683). The header claimed
  all seven checks fail the build while staleness only warned, and the
  frontmatter error named four fields while one was validated: `audience:` and
  `commit:` were read by no line. Both are checked now, and a document edited
  since the commit it claims to be verified against is reported - 41 of them
  are. That check counts edits to the file, not commits to the repo: a
  threshold on repo commits was the first attempt and did not fire on the case
  that motivated it.

- **The engine told server operators to use a flag that kills the server**
  (#1681). `engine_kv_cache_init.cpp:432,459` say "lower --max-seq-len"; both
  messages are shared by imp-cli and imp-server, and `imp-server --max-seq-len N`
  exits 1. They name `--set runtime.max_seq_len=N` now.


- **A nested `tools[].function.parameters` crashed the whole server** (#1607).
  Every parser on the request path is recursive and none bounded depth,
  nlohmann included and it runs first: measured here, 50 000 nested arrays parse
  and `dump()` fine and 100 000 segfault the process, i.e. ~100 KB of body
  against a 100 MiB cap, unauthenticated, taking every in-flight stream with it.
  Bodies deeper than 100 levels are now a `400` at all nine request-parse sites,
  and `json_string_to_value` and the `tojson` walker have their own caps behind
  that. The check does **not** live in the pre-routing handler: httplib calls
  that before the body is read, where `req.body` is empty, and a 10 000-level
  body still returned 200 from there.

- **Streamed tool arguments arrived as U+FFFD wherever a multi-byte character
  crossed a delta boundary** (#1554). `StreamToolCallFilter` holds back
  `close_tag_.size() - 1` **bytes** so a partially arrived close tag cannot leak
  into the arguments, and that cut lands mid-character; each half is
  JSON-encoded into its own SSE delta, where `dump_safe` substitutes. Measured
  on Qwen3-8B-Q8_0 with a forced `tool_choice`: 10 replacement characters in one
  argument, 0 after the fix, with the non-streaming control clean throughout.
  The buffered 48-byte chunker was hardened at the same time, but it is not the
  path a shipped model takes here.

- **`image_url` was an SSRF primitive: any host, any port, redirects followed,
  no size cap, no read timeout** (#1610). An unauthenticated request body chose
  where the server connected, which on a container host means loopback, the
  compose network and the cloud metadata address. Measured, not read: with a
  listener in the server's own network namespace, the pre-fix binary fetched
  `http://127.0.0.1:9999/` on request; the fixed one does not, with the flag off
  or on. Remote fetching is now behind `--allow-remote-images` (default off);
  with it on the destination is resolved and refused if loopback, link-local,
  RFC1918, CGNAT or ULA, redirects are not followed, the body is capped at
  32 MiB and reads time out at 10 s. The error string is uniform and no longer
  echoes the URL, so it cannot report which ports are open.

- **`imp-server` did not start at shipped defaults on the repo's own
  perf-baseline model** (#1631). `imp-server --model Qwen3-8B-Q8_0.gguf` on an
  idle 32 GB card ended in 537 CUDA out-of-memory lines and exit 1: the planner
  cross-checks its NVFP4 heuristic against the storage planner's projection but
  only acted on a 2x divergence, and this model diverges 1.35x (6100 vs 4511
  MiB), so the KV pool took the difference and the first cuBLASLt call had
  nothing left. Any divergence now raises the reserve. The pool is smaller
  (11390 to 7079 blocks, 105136 tokens of capacity) and the server answers in
  3 s. `scripts/test_server_default_start.sh` gates it, wired into
  `make test-server`, because every other server battery boots with capacity
  flags and the default configuration was covered by none of them.

- **`json_schema`: two shapes desynced the parser and truncated the rest of the
  schema** (#1564). `additionalProperties` as a schema object and a non-string
  `enum` member each hit a parse helper that returns a default without consuming
  input, so every key after them was dropped. With `properties` gone the request
  silently became `json_object`; `{"enum":[1,2,3]}` constrained the model to the
  empty string. The object form of `additionalProperties` now parses (its
  constraint on extra keys is not enforced), a non-string `enum` is a `400`, and
  trailing or unclosed input fails the parse instead of returning a partial tree.

- **`json_schema`: assertion keywords are refused instead of dropped** (#1567).
  `minimum`, `maximum`, `multipleOf`, `allOf`, `not`, `uniqueItems` and thirteen
  more were accepted and ignored, so a caller who bounded a field got an
  unbounded grammar at HTTP 200. They are a `400` now, per the contract in
  `docs/API.md`. `const` is implemented. Annotations (`format`, `title`,
  `description`, `default`) stay ignored. **Behaviour change for clients that
  send these keywords today.**

- **Three request-driven parsers had no cost bound** (#1608, #1609). Nesting
  depth mapped 1:1 onto stack frames in the schema, regex and GBNF parsers
  (`((((` is one frame per byte), and the regex `{n,m}` quantifier cloned its
  atom `n` times with no cap: `a{2000000000}` ran a two-billion iteration
  allocating loop on an HTTP worker thread, at admission, before the engine
  lock. Depth is capped at 64, repeats at 1024 (matching the GBNF parser's
  existing bound) and one pattern at 100k NFA states.

- **Four out-of-bounds accesses in the model-file parsers, all reachable from a
  checkpoint directory before any inference runs** (#1603, #1604, #1605, #1606).
  A SafeTensors tensor was validated with one element width and read with
  another (I16: 2 on disk, 4 in the QType it was mapped to), an unknown dtype
  skipped the only validator that looks at `offset_start`, the shape product had
  no overflow or sign guard, and a negative `tokenizer.json` id indexed
  `vocab_` at `size_t(-1)`. A dtype with no equal-width engine type is now
  refused instead of re-typed, and token ids are bounded at both ends.

- **`make asan` could not build its own binaries** (#1659). Two test files that
  use nlohmann sat in the unconditional `test-core` list while nlohmann is only
  fetched under `IMP_BUILD_SERVER`, which the sanitizer target turns off, so
  test-core did not compile in any `-DIMP_BUILD_SERVER=OFF` configuration.
  Now clean: test-core 734, test-text 200, no ASan or UBSan report.

- **imp-quantize read every subnormal value in an F16 `scale_inv` grid up to
  1025x too large** (`0x0001` as 6.1e-05 where the value is 5.96e-08): the
  hand-written widening pasted the subnormal mantissa under a normal exponent
  instead of renormalising, and each such scale multiplies a whole weight block.
  All 2046 subnormal patterns were affected. That dtype is documented as not
  seen in a released checkpoint. Found by merging that conversion out of ten
  files into `src/core/fp_bits.h` and checking the copies against each other
  over all 2^16 half and all 2^32 float patterns first.

- **`imp-quantize --help` was undefined behaviour.** `costs 1-4% of` inside a
  printf format string makes `% o` a conversion specifier. GCC warned about it
  and the warning was never read.


- **The GPU guard asked one question and answered two with it.**
  `require_free_gpu.sh` refused on `memory.used > 2000 MiB` alone, which misses a
  tenant that is compute-heavy and VRAM-light (a full Unreal Engine render on this
  box peaks at 2385 MiB and **71 %** utilisation, so ~700 MiB of its own) and
  misfires on one that is not there at all: the card idles at **1675 MiB**, seen as
  low as 1435, so 2000 left ~325 MiB of margin. It refused a commit during a
  container teardown on 2026-08-21. It now samples five times and splits the two
  questions: sustained utilisation for "is someone computing", minimum-across-samples
  memory for "is there room", thresholds derived from this box rather than rounded,
  and the samples printed on refusal so a flicker is distinguishable from a tenant.
  An empty `docker ps` now says a Windows-side process would never appear there,
  instead of printing nothing.

- **`main`'s `File size` gate was red and a PR merged through it.** #1523 added 6
  code lines to `engine_scheduler.cpp` without re-pinning its allowlist ceiling, the
  check failed, and the merge went ahead because ruleset `14716423` requires exactly
  one context (`Build`). Re-pinned 1962 to 1968. The enumeration of which checks are
  required and which are advisory, and what that means for the five gates shipped this
  week, is [`DEBT_LEDGER`](docs/audit/DEBT_LEDGER_2026_08_21.md) section (j).

- **A failed perplexity run reported `PPL=1.0000`.** "perplexity failed:
  insufficient KV capacity" and `mean_nll=0.0000 PPL=1.0000` on consecutive
  lines, so anyone reading the log takes the failure for a perfect score.
  Exactly zero summed NLL over a non-empty span cannot come from a real forward,
  so it now says the buffer was never filled instead of printing a number. The
  CLI's own result line was already unreachable on failure and no published
  figure came from such a run; this closes the log-reader's path.

- **A kernel whose only caller was its own test.** `fp32_accum_add_fp16_kernel` had
  a declaration, a definition and a green test, and no production launch site. Every
  existing dead-code check read it as covered: the decl-only sweeps filter on two
  mentions, the header-inline gate filters on header definitions, and a caller query
  finds the test. Removed with its test.

- **Ten sites logged at FATAL and then carried on.** `IMP_LOG_FATAL` only writes a
  log line; `IMP_CHECK` is the only thing that aborts. One reported
  `WeightRegistry::handle: id out of range` and indexed out of range on the next
  line; two returned from a dispatch leaving the output tensor unwritten; three MoE
  staging sites said in their own comments that continuing hands a host pointer to a
  device kernel, and then did. Also: the expert-cache parity checker returned `false`
  into callers that discarded it, so the debug facility detected a host/device
  divergence and continued. `make check-log-fatal` keeps the count at zero.

- **`[calibration] out_path` now writes a file.** The key was parsed, documented in
  `config.h` and offered in `imp.conf.example`, and read by nothing: setting it
  produced no calibration file and no warning, because `imp_calibration_write()`
  takes the path as an argument and `--calibrate` was the only way in. `--calibrate`
  still wins when both are given.

- **A decode dispatch branch that logged an error and returned without writing its
  output** now throws. `gemv_dispatch`'s `CUTLASS_NVFP4` case left the output tensor
  holding whatever the workspace held before, behind one ERROR line. Unreachable on
  today's tier assignment, which is exactly where such a path survives unnoticed.

- **A dead FP16 kernel removed from `gdn.cu`.** `vhead_tiled_to_grouped` and its
  kernel were declared, defined and called nowhere; the one consumer of
  `gdn.vhead_reorder` has only ever used the FP32 variant.

- **Four device pointers were freed through an allocator that had not produced
  them**, which is invalid CUDA and silent. `mtp_forward.cu` allocated a 4-byte
  token id with `cudaMalloc` and freed it with `cudaFreeAsync` once per MTP draft
  step (it is now a persistent workspace slot, so the allocation is gone as well);
  `chunk_eager_k_`/`_v_` were 128 MiB of `cudaMallocAsync` released with `cudaFree`
  at executor teardown, which returns success without returning the block to the
  async pool; the shared-workspace grow branch swapped a `vram_alloc()` buffer for
  a `cudaMallocAsync` one. See `AUDIT.md` B10.

## [0.29.0] - 2026-08-21

### Added

- **MTP speculative decoding pays on Qwen3.8-27B-NVFP4: `speculative.mtp_k=1`
  measures +21.3 % decode** (104.31 against 86.03 tok/s). The default stays 0 for
  a measured reason, see [`LIMITATIONS.md`](docs/LIMITATIONS.md).

- **A fully rejected draft no longer re-forwards a row to recover the recurrent
  state.** The scan writes a second copy on its way past instead: a whole model
  forward skipped in 29 % of verifies on Qwen3.8-27B-NVFP4 at k=2 (#1459).

- **`/health` says whether the KV pool can still grow (`kv_pool_growable`).** A
  fixed pool and a growable one at its ceiling used to look identical, and the
  two want opposite reactions.

- **A checkpoint that ships an MTP head now says so when nothing is using it.**
  One log line naming `speculative.mtp_k`, the measured gain and its two prices.
  The default stays 0.

- **`diagnostics.spec_capture_fidelity` checks a cached speculative verify graph
  against an eager forward of the same state.** Off by default (one bool test per
  verify step). `make test-spec-fidelity` runs it as a gate, and
  `scripts/check-release.sh` now has it as a third model-backed stage.

### Fixed

- **`diagnostics.no_nvfp4_decode_cache` no longer changes prefill.** The knob is
  a decode-side bisection tool, but its early return also skipped the CUTLASS
  NVFP4 *prefill* conversion, so all 5935 cached tensors lost their prefill
  payload and the dense FFN dropped from W4A4 to W4A16: 9.165 against 9.051
  perplexity on NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4. Default behaviour
  is unchanged; only the knob was over-broad.

- **Speculative decoding no longer corrupts Mamba2 hybrids.** A fully rejected
  draft chunk adopts the recurrent snapshot the chunk forward writes as of its
  first row (#1459); the GDN path wrote that slab, the Mamba2 path never did, so
  it committed uninitialised VRAM instead: 0 of 26 378 240 bytes written,
  measured device-side. Output degenerated from the first fully rejected verify
  and MTP acceptance on NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 read 0 %
  where the head actually drafts at 39 %. Affects any drafter, not just MTP.
  Details and the before/after table: [`roadmap.md`](docs/roadmap.md).

- **`runtime.cuda_graphs=never` now works on dense models.** The check sat inside
  the MoE branch of weight upload, so on any model without experts the value was
  read and never acted on while the dispatch line still printed `graphs=1`.
  Measured on Qwen3.8-27B-NVFP4: `never` goes from 2 graph captures to 0.
  Closes `AUDIT.md` G1.

- **An unknown `runtime.cuda_graphs` value warns instead of being ignored.**
  `=off` used to parse fine and change nothing, which is how it produced a
  byte-identical A/B that read as a refutation.

- **The perf gate now compiles the change it is gating.** The four `verify*`
  targets had no `build` prerequisite, so on a host without cmake they measured
  whatever `imp:test` already held, and said `SKIP build` while doing it.

- **`check-gpu` refuses on occupied VRAM, not on an empty process list.** On
  WSL2 that list stays empty while a container holds the card: a co-tenant run
  reported 30.39 tok/s against a 287.19 baseline, all of it host.

- **`make test-e2e` runs the model battery instead of skipping all of it.** The
  `models/` it mounted holds absolute symlinks that dangle in the container, so
  every path missed: 25 tests over 6 suites now run where none ran before.

- **A model env var that names a path that is not there fails instead of
  skipping.** Unset still skips, that is a missing prerequisite; set-but-absent
  is a misconfiguration (`imp_test::require_readable`, 23 call sites).

- **A `see FOO.md` in code now has to resolve, whatever the prefix.** The gate
  only checked `docs/`-prefixed paths, so 25 pointers to the deleted
  `TEST_AUDIT.md` survived it; 15 dead names over 25 sites are resolved.

- **`response_format: json_schema` returns valid JSON for a free string value
  again.** A token carrying string content *and* the closing quote got neither
  category, so the pre-filter dropped it before the FSM. Costs ~5.6 % decode.

- **`response_format: regex` and `grammar` answer the request instead of
  `!!!!...` until `max_tokens`.** The allow list was uploaded at the width of the
  logits row into a tokenizer-sized buffer. Broken since #1091/#1095.

- **`speculative.ngram=false` no longer silently disables MTP and token
  recycling too.** The entry gate to the shared verify step asked whether n-gram
  drafting was on, so `mtp_k=2` drafted exactly zero tokens with it off (#1464).

- **The MTP head is refused up front when it does not fit.** 272 allocations
  decided one at a time left a partial upload resident for the life of the
  process. The load line now reports device free consumed, 6168 MiB on Nemotron-3.5.

- **The speculation economics guard no longer rules MTP out by arithmetic at
  every chain length.** It was an acceptance rule stated as a cost rule, and it
  fired before any measurement could contradict it (#1470, #1473).

- **`scripts/check-release.sh` fails when a server battery is red.** `make
  test-server` is the only place `handlers.cpp` and `batching_engine` run end to
  end; the `json_schema` defect above passed the old gate every time.

- **`scripts/verify.sh` no longer reports OK when no model-backed gate ran.** In
  a fresh worktree the perf, peak-VRAM, graphs and smoke gates all skipped and
  the gate still exited 0. A missing `models/` now fails with the path.

- **The weight-upload log line called a neighbour on the card "weights".** The
  same 3263 MiB checkpoint consumes 3264 MiB of device free on an idle card and
  8446 MiB beside a 23.4 GiB process. See [`MEMORY.md`](docs/internals/MEMORY.md) B8.

- **`PrefixCacheE2ETest` asserted bit-equality that the design cannot give.** A
  cache hit chunks differently and flips a near-tie: gap 0.161 between the top
  two candidates, 0.172 shift between the paths.

- **The GDN layers reach their fast path during the verify chunk** (#1467).

- **`scripts/mtp_accuracy_bench.sh` no longer measures nothing quietly.** It
  printed `WARN: no mtp line` whenever the economics guard unbound the head, and
  averaged a partial result over four classes regardless. It now names which of
  five causes fired, exits 1 with nothing measured, and defaults to the released
  checkpoint: 82.7 % offline top-1 on Qwen3.8-27B-NVFP4.

## [0.28.0] - 2026-08-17

### Added

- **The KV pool can grow into what it asked for (`kv_cache.growable`, off by
  default).** A server started while another process still holds the card sizes
  its pool against VRAM that has not been released yet and lands on the rescue
  floor, where it stays for its whole life: capacity is planned once. With this,
  address space is reserved for the pre-clamp plan and physical memory is
  committed for the clamped number, so the pool grows when a request needs more
  than it has and the card can spare it. Measured on this box: base address
  invariant across growth, a captured CUDA graph still correct after 1.5 GiB was
  committed underneath it, 1.18 ms per 256 MiB commit. `/health` reports
  `kv_ceiling_blocks` beside the total, and a floored pool that can still grow
  no longer answers 503, because restarting a server that is healing is wrong.
  `kv_cache.growable_initial_pct` starts the pool deliberately below what the
  card appears to allow, which is the answer to the other half of the same
  unreliable reading: a second server started against a card already holding
  31.4 GiB took its full 10.2 GiB of KV anyway, i.e. spilled into host memory
  with nothing reporting an error. End to end on Qwen3-14B-NVFP4: started at
  810 of 8192 blocks, grew to 1582 to serve a 25 222-token prompt, answered
  coherently, and decode measured 358.6 against 352.4 tok/s for a fixed pool,
  i.e. no cost inside the host's own variance. See
  [`MEMORY.md`](docs/internals/MEMORY.md) A7 step 7.

- **`/health` reports the KV pool, and refuses to call a clamped server
  healthy.** A server started while another process still held the card comes
  up with a KV pool at its rescue floor: it loads, reports `ok`, keeps
  advertising the model's full context, and cancels every prompt past a few
  hundred tokens with a message about the prompt. `/health` now answers 503 with
  `code: "kv_pool_floored"` for that state, and carries `kv_blocks_total`,
  `kv_block_size` and `kv_capacity_tokens` whenever a model is loaded. The code
  is stable so a client can tell a permanent 503 from a retryable one. Reported
  from production, where the quiet version cost an afternoon of looking at the
  wrong component. See [`API.md`](docs/API.md).

- **Speculative decoding stops paying for an eager forward it does not need.**
  On a hybrid, a partially accepted draft re-forwards the accepted prefix to
  rebuild the recurrent state, and that re-forward ran outside the CUDA graph:
  25.1 ms for one or two rows against 17.8 ms for the graph-captured three-row
  chunk it was repairing. It replays the captured graph now. On Qwen3.8-27B
  that is 64.3 to **84.4 tok/s** on the MTP path, which takes MTP from costing
  22 % to costing nothing against 83.7 tok/s without it. A tree-ceiling research
  probe that scanned the whole vocabulary once per drafted token (713 us, 12 %
  of GPU time) is now `diagnostics.mtp_tree_probe`, default off.

- **The server now says when an answer was lost to thinking.** A reply with an
  empty `content` beside a full `reasoning_content` is not a defect, it is a
  token budget consumed before the answer started, and it reads exactly like a
  broken engine. The log now names it, with the amount of thinking and the
  finish reason. See [`TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).

### Changed

- **Quantization quality improved on both measured sizes, calibrated and not.**
  Sharing a tensor scale across fused layers helps the AWQ path as much as
  round-to-nearest: Qwen3-0.6B perplexity 30.10 → **29.42** uncalibrated and
  28.48 → **27.60** with `--calib`, Qwen3-1.7B 20.43 → **20.39** and 19.21 →
  **18.71**, against BF16 references of 24.08 and 17.22. `--calib` with
  `--format vllm` is measured for the first time here.

### Fixed

- **Converting an FP8 checkpoint no longer writes weights whose scales were
  thrown away.** `imp-quantize` pairs each E4M3 weight with its block-scale grid
  and consumes the grid, but a weight the tool keeps at full precision was then
  copied through as raw E4M3 bytes, which are still valid E4M3 and simply mean
  something else. On Qwen3.8-27B-FP8 that is the whole MTP draft head, the one
  part whose corruption costs only draft acceptance and so has no loud symptom.
  Such weights are widened to full precision now and declared unquantized;
  the forecast for that checkpoint goes from 18.80 to 19.15 GiB.

- **A model whose `head_dim` no fused kernel serves no longer aborts on a long
  prompt.** The prefill dispatch knows the tiled FMHA covers head dims 64/96/
  128/256/512; the chunk clamp did not ask, and returned the chunk unclamped
  whenever the context crossed `attention.fmha_prefill_threshold`. On MLA
  models (`head_dim` 192) nothing then bounded the chunk and the cuBLAS
  fallback hit its own S-matrix limit, killing the process with
  `engine should have prevented this`. Reproduced on DeepSeek-V2-Lite, a
  perplexity run over 45k tokens: `std::abort()` before, 13.2787 after. Both
  sides now read the rule from one header.
## [0.27.0] - 2026-08-17

### Fixed

- **A sharded compressed-tensors checkpoint no longer loses its MTP draft
  head.** The shard-drop asked "is this name skipped by the translator", but
  `mtp.*` is skipped *and* then diverted into the MTP map, so the shard
  carrying the draft head was discarded before any tensor was read, and
  spec-decode silently never engaged. On Qwen3.8-27B the head is back with a
  **67-89 % draft accept rate** where it previously reported "model has no MTP
  head loaded".

- **compressed-tensors checkpoints without a `recipe.yaml` are no longer read as
  Modelopt.** imp detected the format from that file alone, but the checkpoint's
  declaration is `quantization_config` in `config.json`, and the two formats
  store the tensor scale as reciprocals of each other. Such a checkpoint loaded
  and generated with every weight scaled by `absmax²/36`: perplexity **1.2e47**
  against 31.05.

### Added

- **`imp-quantize --format vllm` writes a checkpoint vLLM can serve.** The new
  layout is compressed-tensors `nvfp4-pack-quantized` (`.weight_packed` /
  `.weight_global_scale`, declared in `config.json`); `--format modelopt` stays
  the default. Verified end to end: vLLM 0.27.1 loads the output as
  compressed-tensors NVFP4A16 and generates. See
  [`quantization.md`](docs/quantization.md).

### Changed

- **`imp-quantize` says up front when a flag combination will not load where you
  are aiming.** `--lm-head` with `--format vllm` produces a checkpoint vLLM
  refuses (its `ParallelLMHead` takes no scales). On imp the flag costs nothing
  *extra* (the default already converts a native head to NVFP4 at load), but it
  makes that trade irreversible, so the doc now states what the default itself
  costs: Qwen3.8-27B perplexity **4.5707 → 4.6158 (+0.99 %) for +10.4 % decode**,
  and the win survives concurrency (+25 % ITL at 4 requests, +8 % at 16). Also
  refuses a source with no `config.json` before converting rather than after.
  See [`quantization.md`](docs/quantization.md).

- **Fused layers now share one tensor scale.** Engines merge q/k/v and gate/up
  into a single linear that carries one scale, so three independently calibrated
  scales left two matrices dequantized against the third's. The amax spread
  inside those groups reaches 3.7×. Also the better quantization: Qwen3-0.6B
  perplexity **30.40 → 29.42** over `ppl_corpus_45k.txt`.

## [0.26.0] - 2026-08-15

### Fixed

- **`json_schema` requests no longer come back with empty `content` on reasoning
  models.** Structured output disables thinking for `json_mode`, tools, regex and
  grammar; `json_schema` was missing from that list, so the constrainer's gate
  held its mask open for a reasoning block. On a model whose `</think>` is a
  multi-token BPE sequence (Qwen3.8-27B) the block never closed in the text the
  splitter reads, and the answer stayed in `reasoning_content`. Measured on
  Qwen3.8-27B: **0 of 8 valid at temperature 0, now 8 of 8**, and the quality
  suite goes 42/45 to **45/45**. Qwen3.6-27B, whose `</think>` is one token, was
  unaffected and stays at 45/45.

- **`imp-quantize` no longer destroys the MTP draft head.** Its loader takes
  `mtp.*.weight` by name and knows nothing about the `weight_scale` companions,
  so a quantized head arrived as packed nibbles read as BF16. Nothing errored:
  the head loaded, drafted, and every draft was rejected. On Qwen3.8-27B that
  read **0 of 24 drafts accepted against 81% for a checkpoint whose head is
  BF16**, and turning speculation on cost 17% decode. Excluding the head (810
  MiB) restores acceptance to 48% and makes `--mtp-spec-decode` neutral instead
  of harmful.

### Added

- **FP8 sources verified end to end: an FP8-only release now runs where it could
  not before.** `Qwen/Qwen3.8-27B-FP8` does not load directly (weights 26 952
  MiB, upload aborts at layer 60); quantized to NVFP4 it serves at 16 466 MiB.
  The double quantization costs **0.24%** perplexity against the BF16 route of
  the same model (4.6262 vs 4.6151 on `ppl_corpus_45k.txt`), and both routes
  produce the same 19 024 MiB checkpoint from the same 504 tensors.

- **`imp-quantize` reports which bytes did not shrink, and why.** The ratio
  answers "did it work", not "why is it still this big", and those have
  different fixes. On Qwen3.8-27B the breakdown reads 5.60 GiB kept at source
  precision, 30% of the output: 2425 MiB embedding, 2425 MiB lm_head, 875 MiB
  vision tower. Every line is a role deliberately left in source precision, so
  it doubles as the list of what could be traded when a checkpoint misses the
  card.

- **`imp-quantize` reads block-scaled FP8 checkpoints as a source.** Releases
  published only in FP8 (DeepSeek-V3, Qwen3.8's FP8 line) were refused tensor by
  tensor, because the accepted set was BF16/F16. Both conventions store the same
  layout: an E4M3 weight beside a `weight_scale_inv` grid of 128x128 tiles,
  differing only in the scale dtype. The tile size is derived from the two
  shapes, so a grid no single block explains is refused rather than read with a
  wrong stride. The write path is not yet exercised on a real FP8 checkpoint.

- **`imp-quantize --dry-run` now forecasts the output size and whether it fits.**
  It prints the same `size: A -> B` line the real run does, plus the card's
  budget once the CUDA context and library reserve are subtracted. Verified on
  Qwen3.8-27B: the forecast reads `51.75 GiB -> 18.58 GiB (2.79x)`, the figure
  the write then produced, in seconds instead of 25 minutes. The writing path
  now fails loudly if its actual buffers ever disagree with that arithmetic.

- **Qwen3.8-27B runs, text and vision.** Its vision tower is the one imp already
  had, declared under a third `vision_config.model_type` (`qwen3_5`), so enabling
  it was one allowlist entry: 333 tensors, 878.8 MiB. Decode `tg256` 89.85 tok/s,
  prefill `pp512` 7524 tok/s, teacher-forced PPL 4.62 on `ppl_corpus_45k.txt`.
  No published NVFP4 export runs on `sm_120` (they carry FP8 attention, which the
  card has no GEMM for): quantize the BF16 release with `imp-quantize`.
  ([`docs/MODELS.md`](docs/MODELS.md))

- **`imp-quantize` no longer quantizes a vision tower.** `model.visual.*` and
  `vision_tower.*` are copied at source precision, because the tower upload path
  accepts F16/BF16/F32 only, so an NVFP4 tower could not be read back.

- **NVFP4 MoE experts can now live on host**, so a checkpoint whose experts
  exceed VRAM runs instead of being refused. Qwen3-30B-A3B-NVFP4 with all 48 MoE
  layers off-GPU answers correctly at 23.0 tok/s, against 384.0 fully resident.
  ([`docs/roadmap.md`](docs/roadmap.md))

- **`moe.pin_host_experts`** (default off) makes prefill **2.9x** faster on the
  NVFP4 host-offload path: pp512 goes from 276.6 to 790.8 tok/s on
  Qwen3-30B-A3B-NVFP4 with every MoE layer host-resident. It pins the experts at
  load and stages a whole layer per transfer instead of two per expert; the two
  only work together. Decode is unaffected. Off by default because it costs
  4.4x model-load time. ([`docs/roadmap.md`](docs/roadmap.md))

- **`moe.staged_cutlass_prefill`** (default off) runs host-resident NVFP4 experts
  through the CUTLASS grouped prefill instead of a per-expert dequant, lifting
  pp512 from 663 to **1564 tok/s** on Qwen3-30B-A3B-NVFP4 with every MoE layer
  host-resident. Needs `moe.pin_host_experts`. Opt-in: decode measures 36 % lower
  after a long prefill in the same test, which is reproducible but unexplained
  and reverses on a short one. ([`docs/roadmap.md`](docs/roadmap.md))

### Fixed

- **Builds from a cold Docker cache were broken.** The CUTLASS pin read `4.7.0`
  while every tag in that repository carries a `v` prefix, so
  `git clone --branch 4.7.0` failed; cached builds kept working, which is why CI
  stayed green. Three of the four Dockerfile `ARG` defaults had also drifted
  behind the pins they mirror (GoogleTest, cpp-httplib, CUTLASS), so building
  with plain `docker build` used different dependencies than the gate.
  `scripts/check_dep_pins.sh` now enforces both: `make build` runs the offline
  drift half, CI's Lint job additionally resolves every tag upstream.

- **A failed first graph replay skipped that step's forward pass entirely**, so the
  sampler read the previous step's logits and greedy decoding repeated a token for
  as long as the failure lasted. The step now runs eagerly, as every other capture
  failure path in the same function already did.

- **Prefill graph capture now says why it did not happen.** `runtime.prefill_graph`
  has defaulted to on since 2026-05-17, but seven conditions gate the capture and
  none of them logged anything, so a model that never captures looked exactly like
  one that does. Measured: on Qwen3-8B-Q8_0 and Qwen3-Coder-30B-A3B-NVFP4 the LM
  head (vocab x d_model x 2 B = 1187 and 594 MiB) exceeds the 512 MiB dequant
  workspace cap, and FP8 KV closes a second gate.

### Changed

- **Split-K attention merge is 21.9 % faster** (`paged_attention_reduce_kernel`,
  12 992 -> 10 144 ns at base clock). The per-split `(m, l)` pairs are staged into
  shared memory with one parallel load instead of being walked serially by a
  single thread, and each split weight is computed once instead of once per
  thread. Worth **+1.39 %** decode at 8k context on Qwen3-Coder-30B-A3B (10 of 10
  paired runs, sign test p = 0.002); nothing at short context, where the kernel is
  small. Output is bit-identical, covered by a new test.

- **The "2.6x prefill variance" figure is retracted across the docs.** It was a
  citation carried forward; cuBLAS algo re-timing measures 3.50 % over nine
  process starts, and the spread is a property of the model rather than of
  cuBLAS: 0.6-1.2 % on Qwen3-8B Q8_0 against 37.6 % on a fully resident NVFP4 MoE
  model. `docs/PERF.md` now owns the figures; `BENCHMARKS.md` keeps its method
  note as written (it is a record) with a dated correction beneath it.

- **`diagnostics.prefill_graph_ignore_dequant_cap`** (default off) lets a probe run
  keep prefill capture enabled when only the dequant-workspace cap blocks it, so
  the guarded path's reachability can be measured from one binary.
  that release was published after v0.25.0 and the image workflow moved `latest`
  (and `0`) on every publish regardless of version. The moving tags now only
  advance when the release really is the newest.
- **MoE prefill no longer evicts the decode working set from the expert cache**
  on the NVFP4 host path: it now honours the same working-set rule the GGUF path
  has had since #1365. Cache misses drop 3.5x (106k to 30k over a pp512+tg256
  run on Qwen3-30B-A3B-NVFP4 with every MoE layer host-resident); throughput
  moves within noise. ([`docs/roadmap.md`](docs/roadmap.md))
- **An NVFP4 MoE checkpoint whose experts do not fit is refused at load**
  instead of skipping their GEMMs and answering from the rest at exit code 0.
  The refusal now fires only when the expert cache cannot hold a layer's working
  set; otherwise the host path above serves it. GGUF experts are unaffected.
  ([`docs/roadmap.md`](docs/roadmap.md))

## [0.25.0] - 2026-08-13

The Nemotron-H family decodes ~3x faster, Qwen3.6-35B sees images, and native-FP8
checkpoints load. 40 PRs since 0.24.0; the reasoning behind each entry is in the
linked PR.

### Added

- **Native FP8 weights load: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` runs**,
  the first checkpoint here that ships them (45/45 on `degen_suite.py`; **362
  tok/s** decode with the graph fix below, against vLLM 0.27.1's 351 on the same
  file). Modelopt `MIXED_PRECISION` puts 46 Mamba projections in FP8 and 5935 MoE
  tensors in NVFP4; sm_120 has no FP8 prefill GEMM, so the FP8 tensors get an
  FP16 companion at load (1698 MiB). (#1385, #1386)
- **Qwen3.6-35B-A3B sees images**: no new encoder and nothing to download. The
  checkpoint has always shipped a complete Qwen3-VL tower (333 `model.visual.*`,
  851.8 MiB), and two gates dropped it. Text-only checkpoints are unaffected.
  (#1379, #1384)
- **`moe.expert_cache_budget_pct`**: the share of free VRAM the MoE expert LRU
  cache may claim, previously hardcoded at 15. **Default unchanged at 15**; at 50
  a fully host-resident Qwen3-30B-A3B-Q4_K_M decodes 10.51 → 51.86 tok/s (hit
  rate 36.6 % → 96.2 %). (#1374)
- **Nemotron-3.5's MTP head loads and drafts, and measurably should not be
  used.** 43.9 % top-1 accept, still -32 % decode, because the draft step runs
  outside CUDA graphs while the main decode no longer does. `--mtp-spec-decode`
  stays opt-in and off by default; the bottleneck is the verify chunk, not the
  draft. (#1390, #1391)

### Changed

- **The entire Nemotron-H family decodes ~3x faster**: CUDA graphs were demoted
  for pure-SSM layers by a `not yet` nobody had retested; nothing in the Mamba2
  scan is capture-hostile. tg256, spec off: Nemotron-3-Nano 148 → **386**,
  Nemotron-Labs-3-Elastic 70 → **381**, Nemotron-3.5-Lightning 126 → **362**
  tok/s (vLLM 0.27.1 reads 351 on the same card and checkpoint).
  `docs/supported-models.md` and `AUDIT_ARCH` had both blamed the architecture;
  corrected. (#1389)
- **MoE host-offload decode ~2.1x: the fused decode kernels reach host-resident
  experts.** Feeding them slot indices instead of expert ids needs no new kernel
  and no staging copy. Qwen3-30B-A3B-Q4_K_M, all 48 MoE layers on host: 22.9 →
  48.3 tok/s, CUDA launches -69 %. (#1370)
- **MoE host-offload decode ~2x again: the expert cache stopped maintaining a
  device mirror nothing reads.** `cudaMemcpyAsync` -59 %, decode ~20 → ~41 tok/s,
  output byte-identical. The mirror is now built only under
  `moe.expert_cache_debug_parity`, its sole reader. (#1376)
- **Native-FP8 weights decode from their own bytes: +7.5 % median** on
  Nemotron-3.5-Lightning (27 order-balanced pairs, t=3.33; +6.9 % predicted from
  bytes and bandwidth). Decode had been reading the FP16 prefill companion, 2
  B/elem where the checkpoint ships 1, on the bandwidth-bound path. No extra
  VRAM, lossless. (#1388)
- **The MoE expert LRU cache is skipped for a dispatch it cannot hold**: prefill
  asked for 384 slots against the 73 a Qwen3-30B-A3B gets and retained nothing.
  Median +5.6 % pp512 at no decode cost, decode hit rate 88.7 % → 95.7 %, output
  byte-identical. (#1365)
- **Dependencies: CUTLASS v4.6.2 → 4.7.0, googletest v1.17.0 → v1.18.0,
  cpp-httplib v0.50.1 → v0.53.0.** CUTLASS is the primary GEMM path on sm_120, so
  it was A/B'd rather than assumed: decode neutral to within 0.05 % on two NVFP4
  models. Note for the next bump, NVIDIA dropped the `v` prefix, the tag is
  `4.7.0`. (#1392)

### Fixed

- **`check-release.sh` aborted silently right after a release cut.** An empty
  `[Unreleased]`, exactly what cutting a release leaves behind, made its
  changelog-hygiene `grep` exit 1, and `set -euo pipefail` killed the run with no
  FAIL line, before `make verify-fast` ever executed. (#1394)
- **A load guard cried wolf on every working up/down-only MoE.** It asked only
  about `expert_w_gate`, which no Nemotron-H has, and went unnoticed because the
  caller discards the bool. Now checks all three projections. (#1385)
- **`imp-quantize` refuses `--keep-attn-gate` together with `--calib`** instead of
  writing a checkpoint that loads, generates, and is silently wrong: the fused
  Q+gate `q_proj` skips the group's column scale that the calibration planner
  folds into `input_layernorm` regardless. (#1382)

## [0.24.0] - 2026-08-10

### Added

- **`imp-server` starts without `--model`**: model-less it answers `/health`,
  `/v1/models`, `/metrics` and the parameter-validation surface; the first
  request naming a model under `--models-dir` loads it, and a request that
  resolves to no model gets 503.
- **CI tests the shipping server, not only its stand-in**: new `Real API
  contract` job runs the 42 `nomodel` API tests against the built `imp-server`.
  Until now all 82 assertions described `tests/api/mock_server.py`. (#1302)

### Changed

- **Batch invariance is a stated boundary, not an open question**:
  `docs/determinism.md` records what holds: a batch neighbour's content cannot
  reach another row (asserted bit-exactly), joining a batch costs rounding only
  (0.22 % of the logit range). No flag makes batched and solo bit-equal. (#1314)
- **The deterministic-mode E2E suite runs again**: it was gated on an env var
  nothing set and had skipped since #542. `DetEvalE2ETest` is value-parameterised,
  so filters need `*DetEvalE2ETest*`; the old `DetEvalE2ETest.*` matches nothing
  and gtest calls that PASSED. (#1299)

### Fixed

- **Learned attention sinks reach the INT8 / INT4 / NVFP4 / MXFP4 KV decode
  kernels.** Only FP16 and FP8 applied them, so gpt-oss on any other quantised KV
  served a softmax denominator short one column and answered nothing at all.
  INT4 keeps the FP16 fallback: its sink term is correct, the 4-bit grid is not.
  (#1345)
- **`kv_cache.dtype=int8` survives a prefix-cache hit.** It was the one quantised
  KV dtype without a `paged_kv_gather` kernel, so the partial prefill after a hit
  aborted the process with no HTTP response: three identical requests scored
  200/000/000, now 200/200/200. (#1348)
- **Raising `max_tokens` buys answer room on a reasoning model.** The think budget
  reserved a flat 256 tokens for the answer, so above `max_tokens=512` the answer
  stayed pinned at 256 whatever the caller asked, 600/1500/3000/4096 all returned
  ~1000 characters and never `finish_reason: "stop"`. The reserve now scales
  (`max(256, max_tokens/4)`); at or below 1024 nothing changes. (#1297)
- **An unknown `kv_cache.dtype` says so instead of silently staying FP16.**
- **An unmatched route answers with a JSON error envelope.** `POST /v1/nope`
  returned 404 with a zero-length body, so a client reading `error.message` got a
  parse error instead of a reason; `/v1/messages*` paths get the Anthropic shape.
- **`n` on `/v1/chat/completions` is documented and tested as `[1,4]`**: the
  suite asserted `n=2` is a 400, true of the mock and never of the server.

## [0.23.0] - 2026-08-07

### Added

- **`tools/analysis/layer_ab_diff.py`**: per-layer divergence between two runs of
  the same architecture, so a bad checkpoint can be traced to a *block* rather
  than to "it is worse". It settled #1273: attention blocks add divergence
  (median +0.0156), GDN blocks are slightly corrective (-0.0017).

### Changed

- **A constraint imp cannot compile is a `400`, not an unconstrained answer.**
  `response_format: regex`/`grammar` and the `guided_regex` / `guided_grammar` /
  `grammar` aliases are validated at admission with the engine's own parsers.
  Previously the rejection was logged server-side and the request answered with
  free text at HTTP 200. **Breaking:** a client sending a pattern imp refuses now
  gets an error instead of an answer. (#1256)
- **The container keeps its caches across recreation.** `docker-compose.yml` gives
  imp-server a named `imp-cache` volume and the image creates
  `/home/imp/.cache/imp`, so a fresh volume is not root-owned and unwritable,
  which had disabled both the warm weight cache and the library-reserve
  measurement. Cold init on Qwen3-14B-NVFP4 7949 → **2099 ms** once populated.

### Fixed

- **The final RMSNorm missed Qwen3.5/3.6's `gamma = 1 + W` offset. This is what
  #1273 was.** Every other norm took the offset; the output norm did not, so a
  SafeTensors checkpoint scaled the last hidden state by `W` instead of `1 + W`.
  PPL on `ppl_corpus_45k`: Qwen3.6-27B-Text-NVFP4 **65.13 → 7.53**,
  Ornith-1.0-35B-NVFP4 **16.16 → 7.07**, Qwen3.6-35B-A3B-NVFP4 **13.65 → 6.82**.
  GGUF and dense checkpoints byte-identical. (#1287,
  [`docs/quantization.md`](docs/quantization.md))
- **A `json_object` reply that hits `max_tokens` parses.** #1096 forbids a closer
  after a comma and #1104 demands one once the budget is spent; where they met
  nothing was legal and the constraint was released. Qwen3.6-35B-A3B-NVFP4 at
  `max_tokens=40`: 0/12 valid → 6/6. (#1291)
- **A KV pool too small for the requested context says so at load.** A pool that
  is a real size and still cannot hold one `max_seq_len` sequence was silent,
  and every full-length request was cancelled at admission while the load
  reported success. Operator-set `max_seq_len` only. (#1251)
- **A MoE GGUF could load fine and then cancel every generation.** The NVFP4 MoE
  cache reserved KV room without counting allocator headroom, so the pool fell to
  the 16-block floor: Qwen3.6-35B-A3B-UD-Q4_K_M went from **512 tokens** and
  `finish_reason: "cancelled"` to **40224 tokens** and a full reply. (#1251)
- **A `tool_choice` that contradicts the request is a `400`.** Naming a function
  absent from `tools` had the model invent a call to something the caller never
  described; `"required"` with no tools was answered as an ordinary turn.
- **A content part this server cannot read is a `400`, not an answer.** A
  `video_url` part produced a 200 replying to a prompt the model never saw. Only
  `text` and `image_url` are read; anything else is named in the error.
- **Constrained decoding works the same on `/v1/messages` as on
  `/v1/chat/completions`**: `guided_regex`, `guided_grammar`, `grammar` and
  `response_format` were dropped by the Anthropic shim (measured: `'ZZZ6'` vs
  free-form prose), and a malformed one was not rejected there either.
- **A `"system"` message on `/v1/messages` acts as a system prompt** instead of
  being rendered as a *user* turn. The text reached the model with the wrong
  semantics and nothing said so.
- **`(?:…)` is honoured instead of mis-compiled.** The support check passed
  non-capturing groups but the engine had no `?:` form, so `(?:a|b)c` compiled to
  `(:a|b)c`, matched `bc`, rejected `ac`, reported success. Affects
  `response_format: regex` and JSON-Schema `pattern`.
- **`response_format: regex` honours `^…$`.** Edge anchors were refused outright,
  so the most natural way to write a pattern got *no* constraint and HTTP 200:
  `^[0-9]{3}$` now returns `221` instead of `How can I assist you today?`.
  Interior anchors (`a^b`) stay refused.
- **`diagnostics.dump_hidden_dir` works on models other than Gemma-4.** The dump
  sat inside the Gemma-4 `layer_out_scale` branch and behind
  `debug_forward_enabled()`, so everywhere else it wrote no files and said
  nothing. It now keys off its own switch and honours the configured directory.

### Reverted

- **The GQA tile split-count boost (#1270) is reverted (#1271).** +1.3/+4.8/+10.0 %
  at 8k/16k/32k on Qwen3-8B-Q8_0, but **-7.30 % at 32k on Qwen3-30B-A3B-NVFP4**, a
  pinned hero. One model is not a heuristic; the condition separating the two
  cases is not established, so it does not ship.

The 2026-08-06 error-path campaign (#1252-#1265) is written up in
[`docs/MISSION_JOURNAL.md`](docs/MISSION_JOURNAL.md).

## [0.22.0] - 2026-08-05

### Added

- **`tools/analysis/vision_sight_check.py`**: answers "is this tower blind or
  just weak?" in ~2 minutes by scoring a counting battery against the **best
  constant answer**: #1246 reported 4/6 correct, which was exactly the score of
  always replying "1". Gemma-4-26B + mmproj reads 8/8 against a 2/8 baseline;
  with the image withheld it reports BLIND, which validates the check itself.
- **`imp-quantize --calib-groups ABCD`**: runs any subset of the AWQ planner's
  four scale groups, which is what showed the `--calib` failure is the *attention*
  half only. `--calib-groups BD` scores **9.7922 on Qwen3-14B** against
  round-to-nearest's 9.9252, the best measured configuration; the default `ABCD`
  costs +2.68. Use `BD` on wide-GQA models.
  ([`docs/quantization.md`](docs/quantization.md))

### Changed

- **The obvious fix for the AWQ attention failure is REFUTED.** Splitting group
  C's tied statistic makes Qwen3-0.6B worse by 0.71 PPL (28.89 → 29.59) and does
  not rescue the 14B (12.48, still +2.55 over round-to-nearest). The `max` tie is
  a real coupling, not a bug. No code change: the measurement is the deliverable.
- **The 2026-07-29 architecture audit is closed at 25/25** and `SETTLED.md` says
  so: §G had kept six resolved findings under "Open", and F-10/F-12/F-24 carried
  no status line at all. Three of the six closed by refuting their own fix.
- **`check-release.sh` section 1d cross-checks finding status** between
  `AUDIT_ARCH_2026_07_29.md` and `SETTLED.md`. The ledger had gone stale the same
  way three times (F-6, F-15, F-10), each time pointing a later pass at work
  already done.

### Fixed

- **A system message no longer costs an image its tokens.** The Gemma vision
  prompt keyed its image block on "message index 0 is the user message", so any
  request opening with a system prompt rendered text-only and the model answered
  fluently that it could not see an image. gemma-3-4b + mmproj: **37 prompt
  tokens and a refusal → 296 tokens and a correct description.** (#1246)
- **An mmproj GGUF this loader cannot read is refused, not half-loaded.**
  Qwen3-VL's mmproj assigned 247 of 316 tensors and reported success; the 69 it
  dropped were exactly the fused `attn_qkv`, DeepStack mergers, second projector
  layer and temporal patch conv, so the encoder passed null slots to
  `vision_gemm`. Gemma-3 (439/439) and Gemma-4v (356/356) are unaffected.

## [0.21.0] - 2026-08-05

### Added

- **`diagnostics.log_level`** (`debug|info|warn|error|fatal`), debug logging
  could not be switched on at all: nothing called `log_set_level()`, so all 76
  `IMP_LOG_DEBUG` sites were unreachable. An unrecognised value warns and keeps
  the current level rather than silently falling back. Measured: 0 debug lines by
  default, 359 with `--set diagnostics.log_level=debug`.
- **The MoE prefill dispatch checks itself against its routing model.**
  `select_moe_prefill_path` had zero production callers, ten test callers and a
  comment asking for the two predicates to be kept in sync by hand, so a reorder
  left the routing test green while describing a dispatch that no longer existed.
  Each tier now replays the model against what the chain observed.
- **`docs/audit/SETTLED.md`**: the ledger an audit reads *before* forming
  hypotheses, gated by 49 anchors in `check-release.sh` (CI job `Release
  hygiene`). Eight of the 2026-07-29 audit's thirteen hypotheses were refuted
  because they described duplication earlier campaigns had already collapsed.
  (#1215)
- **CI gate: every CUDA kernel launch must carry a post-launch error check**
  (`tools/check_launch_guards.py`, job `Launch guards`). The convention was ~99 %
  adopted and 0 % enforced: the whole Qwen3-VL vision tower sat at 9 launches / 0
  checks, where a launch failure would have produced silently wrong image
  embeddings. Now 407/407 in-scope launches are guarded. (#1206)
- **`Resolved dispatch:` log line**: which attention and MoE kernels a model
  actually ran, e.g. `attn_prefill=fa2_fp16qk attn_decode=paged_fp8
  moe_prefill=cutlass3x`. The tiers all decline by returning `false` with no log,
  so a model dropping to a slower path used to leave no trace. Recorded from
  inside the real dispatch, so it cannot disagree with what ran. (#1205)
- **`--metrics-require-auth`**: fold `/metrics` back under the `--api-key`
  check. Off by default, but the endpoint discloses model name, `d_model` and
  cumulative token counts to anyone who can reach the port. (#1207)

### Changed

- **The nine dispatch config sections moved to `core/`.** `runtime/config.h` was
  included by 85 translation units and changed 130 times in six months, the
  highest build cost in the repo. `exec/` and `compute/` no longer include it at
  all; `config.h` 1143 → 403 lines, TUs pulling it 85 → 18. Resolved-dispatch
  lines are bit-identical across the GPU suite. (audit F-10)
- **`runtime/engine.h` reaches 23 translation units instead of 33**: it was the
  #2 build-cost header, and `imp_internal.h` pulled it in front of 17 includers
  while storing nothing but a `unique_ptr<Engine>`. Cuts the rebuild set 30 %.
  (audit F-24)
- **GEMM algo selection repeats across process restarts.** Each candidate was
  timed once, at M=16 a ~30-120 µs window that is mostly launch overhead, so the
  pick was noise: 4 of 8 shapes chose 3-4 different kernels over 5 fresh
  processes. Now 7 of 8 pick identically, at no load-time or throughput cost.
  (audit F-9, [`docs/audit/SETTLED.md`](docs/audit/SETTLED.md))
- **The Qwen3-VL vision tower, the Gemma mmproj path and the decode batch pool
  are T2 arena tenants** (audit F-12). 792.2 MiB on Qwen3-VL-4B plus 224.1 MiB of
  pipeline/encoder scratch sat outside the tier model; `src/vision/` no longer
  uses `VRAMAllocator` at all. Direct allocation sites **482 → 471**. A test
  asserts the reservation equals what is taken, which caught a 192 MiB undercount
  on its first run.
- **`imp-cli --max-tokens` defaults to 8192, not 256**: 256 predates reasoning
  models, where the think block alone overruns it and the answer comes back empty
  with `finish_reason=length`. `imp-server` was already at 8192. (#1209)
- **The 26 flags `imp-cli` and `imp-server` both accept are parsed once**
  (`tools/common/args_common.{h,cpp}`). Two hand-written else-if chains that had
  to agree by review alone; every handler was verified byte-identical first, and
  every default matched except `max_tokens`. (#1209)
- **The engine logs through one mechanism again**: all 90 raw `fprintf(stderr)`
  sites in `src/` go through `IMP_LOG_*`, so `log_level=error` no longer still
  prints `[gemm-algo]` and `[DEBUG_FWD]`. `log_message()` is now
  `format(printf)`-checked, which found five real `%lld`-vs-`int64_t` mismatches
  in `weight_upload.cu`.
- **Pre-`cudaDeviceReset` hooks register themselves**: the eleven module hooks
  were listed by hand, so a twelfth lazy static added without an entry dangled
  behind an armed guard. Side effect: the one `core → compute` backward edge is
  gone. (#1207)
- **`IMP_SPEC_TRACE`, `IMP_JUMP_TRACE` and `IMP_PPL_DUMP` are config keys**
  (`diagnostics.spec_trace` / `.jump_trace` / `.ppl_dump`). All three had crept
  back as raw `getenv()` calls; the env names still work. (#1207)
- **Two layering cycles closed by moving a type down, not by adding an include**:
  `WeightHandle` to `core/`, `CutlassMxFP4Weight` to `src/quant/`. `compute →
  model` goes 9 → 7.

### Fixed

- **A second engine in one process ran on freed memory.** The module statics that
  take a slice from the engine's T2 arena were re-armed only by a reset wired to
  `imp_api_suspend.cpp` and never to `~Engine`, so cuBLASLt matmul'd into a stale
  pointer: status 14, illegal memory access, and every later test in the process
  died on a context it did not break. Full GPU suite **57 failures → 1**, 111 IMA
  lines → none.
- **An unrecognised architecture string loaded silently as GENERIC.** The
  Llama-shaped fallback is deliberate, but it said nothing, so a genuinely
  unsupported checkpoint looked supported and produced plausible wrong output.
  (#1206)
- **A failed chat-template init was discarded**: on failure the template stays
  inert and every `/v1/chat/completions` request falls back to raw prompt
  concatenation with no role markers, which reads as a model-quality problem.
  Found by marking the 14 two-phase `init()`/`setup()` methods `[[nodiscard]]`.
  (#1206)
- **Nothing bound the public C enums to their internal counterparts**, so a wrong
  id made `imp_model_architecture()` report the wrong architecture to every C-API
  consumer with a green build. `tests/test_c_api_enum_binding.cpp` closes the
  loop. (#1206)
- **A C-API embedding got different kernels than `imp-cli` from the same config.**
  `process_diag_install()` ran only in the tool mains, so a library consumer's
  `attention.*`, `moe.*`, `gdn.*` and `runtime.*` settings were honoured by
  `exec/` and ignored by `compute/`. (#1205)
- **`EngineRelaunchTest` asserted on a process-wide counter**: its 1024 MiB bound
  on `cudaMemPoolAttrUsedMemCurrent` only held when the test ran first. Now takes
  a baseline and asserts on its own delta; still fires at 4076 MiB if #834's
  `cudaFreeAsync` is reverted.
- **`StubModelTest.CreateContextAndInfer` no longer hides a poisoned context**:
  its failure path skipped with "expected without GPU", indistinguishable from a
  real skip.

### Removed

- **Two dead modules and sixteen uncalled functions**: `compute/gemv_ggml_compat`
  (174 lines, its kernel launched only by a dead wrapper) and `core/threading`
  (88 lines, a `ThreadPool` included by its own `.cpp` only). Found by BFS over
  the call graph from live roots; the one-level "does it have a launcher" check
  counted the dead kernel as live.
- **Ten more declared-defined-never-called functions**, among them four CuTe TMA
  descriptor builders and four **empty-bodied** `gdn_*` legacy stubs where a
  caller would have got a silent no-op.
- **Three KV/bias kernels built and linked but never launched**:
  `write_kv_cache_kernel`, `write_kv_cache_fp8_kernel`,
  `add_fp16_bias_to_fp32_kernel`. The two tests covering the non-fused FP16 write
  moved onto the kernel the engine actually launches. (#1216)

## [0.20.2] - 2026-08-02

### Fixed

- **Constrained decoding dropped every non-ASCII character.** With
  `response_format: json_schema` or `json_object`, "Die Bären hören" came back as
  "Die Baren horen", German and every other non-English language was unusable.
  `char` is signed, so every byte of a multi-byte UTF-8 sequence read as negative
  and lost `CAT_STRING_CHAR`; the category mask then dropped those tokens before
  the FSM was consulted. GBNF and regex were never affected. (#1197)
- **An image sent to a model that cannot see it is refused, not ignored.** A
  checkpoint whose tower imp does not understand loads text-only and says so, but
  `image_url` parts were accepted and answered from the text alone, so the caller
  got a confident description of a picture the model never received. Now `400`
  with code `vision_unavailable`, in every dialect. (#1198)
- **Constrained generation could stop with the JSON document still open**,
  returning 200 with a reply that does not parse: `<|im_end|>` decodes to
  printable ASCII, so the free-string shortcut allowed it without simulating.
  Measured on Qwen3-VL-4B, where it sat at rank 2 on the last step. (#1199)
- **The build is warning-free again**: five had accumulated, including `tmpnam`
  in a fixture that then fed `system("rm -rf " + dir)` (now `mkdtemp` +
  `std::filesystem::remove_all`, so no shell parses the path).

### Changed

- **CI can be run against a ref on demand** (`gh workflow run CI --ref main`). A
  squash merge by auto-merge starts no workflow run: it is attributed to
  `GITHUB_TOKEN`, so `main`'s reported CI state can be many commits old. It was
  ten commits old on 2026-08-01.

## [0.20.1] - 2026-08-01

### Changed

- **CUDA 13.3.0 → 13.3.1** (nvcc V13.3.33 → V13.3.73), the newest toolkit there
  is. Same release string, so CI's nvcc check and the ccache keys are untouched.
  Perf-neutral: decode tg128 287.95 vs 288.38 tok/s, both against the 287.19
  baseline.
- **`imp-quantize --calib` says where it is validated and where it is not.** A
  model too big to run in BF16 can be calibrated off any quantization of itself,
  which produced the first `--calib` result on Qwen3-14B, and it is negative: PPL
  **9.9252 uncalibrated vs 12.6016 / 12.2853** calibrated. It still helps on
  Qwen3-0.6B/1.7B. (`docs/quantization.md`)
- **A dead `docs/…` pointer in a code comment is a gate failure.** Sixteen design
  memos were deleted in earlier consolidation PRs and the 38 comments citing them
  stayed behind.

### Fixed

- **`--set` with a key that does not exist is an error, not a warning.** A typo
  silently measured the default instead: the published AWQ harness passed
  `--set gemm.deterministic=true` (the key is `runtime.deterministic_gemm`), so
  its three scoring runs never got the determinism they asked for. Re-run with
  the key fixed, the published numbers reproduce unchanged. An unknown key in
  `imp.conf` stays a warning, a config file may outlive the build.

## [0.20.0] - 2026-08-01

### Added

- **Vision: Qwen3-VL**: imp describes images end to end, from `imp-cli --image`
  and from `/v1/chat/completions`. Dynamic resolution (a 1795x2397 photo becomes
  972 image tokens), DeepStack taps, three-axis M-RoPE. Text-only models and
  prompts are bit-identical to before. (#1163-#1180)
- **Several images in one request** (Qwen3-VL): each `image_url` part is encoded
  in prompt order; previously the last image silently won. An unreadable
  `image_url` is now a 400, because dropping one would slide every later picture
  onto the wrong placeholder. C API: `imp_add_image{,_from_memory}`.
- **`POST /v1/rerank`**: Cohere/Jina/vLLM-compatible, scoring query and document
  jointly in one forward. Validated against llama.cpp on the same GGUF (top-1
  agreement 3/3, median score delta 0.0014). Gate: `make test-rerank`.
- **GBNF grammar-constrained decoding**: `response_format: {"type":"grammar"}`,
  llama.cpp's `grammar`, vLLM's `guided_grammar`. Grammars the simulator cannot
  honour (left recursion, undefined rules, missing `root`) are refused rather than
  mis-enforced.
- **Regex-constrained decoding**: `response_format: {"type":"regex"}` and
  `guided_regex`. Lookaround, anchors, `\b` and backreferences are refused.
- **`imp-quantize` (EXPERIMENTAL)**: first-party BF16/FP16 → NVFP4 conversion
  writing the layout the loader already reads, sharded sources included.
  Qwen3-0.6B PPL 28.48 with `--calib` vs BF16 24.06, 30.10 without.
- **AWQ-class activation calibration** (`imp-cli --calibrate`, `imp-quantize
  --calib`) over a 13.5k-token corpus: Qwen3-0.6B 30.10 → **28.48**, Qwen3-1.7B
  20.43 → **19.21**, `degen_suite.py` 45/45. Forces `deterministic_gemm`:
  without it two identical runs differ on 94 % of recorded floats.
- **Model swapping on request** (`server.model_swap`, default on), a request
  naming another model in the models directory swaps to it instead of 404ing.
  In-flight generations drain first; a failed load restores the previous model.
- **Web UI at `GET /`**: single-page chat client embedded in the binary,
  streaming over the existing SSE endpoint, with per-token latency bars and a
  separate thinking channel.
- **`evicted_tokens`**: the caller is told when StreamingLLM dropped context, in
  all three dialects. The key is absent unless eviction fired, so its presence is
  the signal.
- **SWA window snapshots** (`kv_cache.swa_snapshot_mb`, default 0), prefix
  caching and SWA-aware KV sizing now combine. Opt-in: ~50-100 ms warm TTFT at
  1-2K contexts for +8 % prefill at 13K plus the KV savings.
- **Qwen-Coder XML tool-call grammar** for the `<function=NAME>` dialect
  (Qwen3-Coder, Qwen3.6), 0/3 → 3/3 compiling `write_file` contents on Coder-30B.
- **External agent gates** (`make test-agents-external`): aider, Claude Code and
  the OpenAI Agents SDK must each land a real edit in a throwaway repo.
- **Generative property batteries for both constrained-decode FSMs** in the CPU
  lane, checked against nlohmann/json as an independent oracle.
- **Measurement tools**: `tools/analysis/agentic_compare.py` (cross-engine
  agentic reliability) and `ctx_capacity_decode_sweep.sh` (the regime the CI perf
  gate cannot see).

### Changed

- **SWA-aware KV sizing is tri-state `auto|on|off`** (default `auto`): the savings
  (gpt-oss ~2x, gemma-3/4 ~5-6x KV tokens) are taken only when prefix caching is
  off, so warm-prefix TTFT is untouched.
- **Auto `max_seq_len` ceiling raised 64K → 128K.** (#1004)
- **Engine-persistent (T2) arena for `compute/` scratches**: the CUTLASS grouped
  workspace reservation drops 512 MiB → 1 MiB (`get_workspace_size()` returns
  152 320 B across every geometry tried). Qwen3-30B-A3B-NVFP4 own peak VRAM
  20932 → 20454 MiB.
- **"Prefer a published Modelopt checkpoint" no longer stands unmeasured**:
  against a bit-identical Modelopt export of Qwen3-14B: **Modelopt 10.0301,
  `imp-quantize` without `--calib` 9.9252**. Enough to retire the blanket advice,
  not to reverse it.
- **`scripts/check-release.sh` runs in CI** as `Release hygiene` and pins the
  version across `CMakeLists.txt`, `CHANGELOG.md` and `docs/BENCHMARKS.md`. It
  was wired into nothing before; cutting this release found two maintainer paths
  that had been on `main` for days.
- **One `needs_constrained` flag** replaces `needs_json_mode` /
  `needs_schema_mode`; regex and GBNF take the same pipelined path. Measured
  neutral, shipped for consistency.
- **`--mem-report` names every `VRAMAllocator` charge** instead of estimating the
  executor's. Diagnostics only.
- `tools/imp-server/tool_call.cpp` split (Gemma-4 dialect parser to
  `tool_call_gemma.cpp`), clearing the repo's only hard-review file-size
  violation.

### Fixed

- **Decode paid for context capacity it never used, up to -38 % on the served
  path.** The NVFP4 decode cache's reservation was subtracted from the budget it
  spends from. Same 280-token request at `max_seq_len` 1024 → 40960: **160.10 →
  99.29 tok/s before, 162.77 → 163.24 after.** (#1100)
- **The library reserve was measured through a window that missed most of it**,
  costing one model 4x its KV capacity. Qwen3.6-35B-A3B-NVFP4's second start gets
  **16 384 tokens instead of 4096**; attribution 98.3/96.6/89.7 % → ~100 %.
- **A MoE checkpoint whose experts imp cannot read now fails to load** instead of
  routing through null experts and generating garbage, measured on `gpt-oss-20b`
  in BF16, which logged "unrecognised layer weight" per layer, loaded, and
  produced `:!!!!!!!!!!`.
- **The persisted prefix cache dropped the KV scales**, so a restored block
  decoded against whatever scales were in the pool, wrong attention, no error.
  Only on KV dtypes with a separate scale pool (NVFP4, INT8, INT4, MXFP4_KV).
  Format version 3.
- **The prefix cache could serve one request's image to another**: it is
  addressed by token ids and every image token carries the same id. Block hashes
  are now salted with the image content.
- **An image that spanned a prefill chunk boundary got the wrong half of itself.**
  Both vision kernels find "the k-th image token" by scanning the span they are
  handed, which under chunked prefill is one chunk. Reachable on defaults.
- **`top_k` above 128 sampled from the previous decode step's candidates**:
  `cub::DeviceTopK::MaxPairs` writes nothing from its second call while returning
  `cudaSuccess`, and nothing checked the code. Replaced with a full-vocabulary
  sort. (#1142)
- **The sampler drew the same quantile on every token.** Seeds are `base_seed +
  step` and an LCG's first output is affine in its seed, so the draw was
  effectively constant. Fixed with a splitmix32 finalizer; same seed still gives
  the same token.
- **A CUDA graph could replay a kernel whose scratch pointer had been freed**:
  six `compute/` statics grew with `cudaFree` + `cudaMalloc` and now come from the
  T2 arena, which never frees. (AUDIT B13)
- **The first `response_format` request after a model load could come back
  unconstrained**: `JsonConstrainer` allocated its allow list lazily mid-decode,
  and under VRAM pressure `apply_mask` returned without masking and without
  logging. (#1104)
- **Streaming leaked the chain of thought as the answer whenever tools were
  present**: a tool request renders a pre-closed think block, so the model emits
  only the closer and the splitter waited for an opener that could never arrive.
  Found by pointing the real Claude Code binary at imp-server.
- **Streamed non-ASCII text was corrupted** (`"größer"` → `gr<?><?>ßer`): a BPE
  token can end mid-character, and the stop-sequence holdback cut at a byte
  offset. Now stitched at detokenization and cut at a codepoint boundary.
- **`imp-quantize` wrote a "quantized" checkpoint whose experts were untouched**:
  the 3-D refusal sat behind a `.weight` name test that no real stacked checkpoint
  matches, so experts were copied through as BF16 while `hf_quant_config.json`
  announced NVFP4. Such a checkpoint is now refused before anything is written.
- **`imp-quantize` silently produced a broken MoE checkpoint**: the HF-standard
  per-expert 2-D layout was quantized and loaded and then emitted garbage. The
  cause is the **MLA latent projections** and the **MoE router**, both now
  refused. DeepSeek-V2-Lite 29.26 → 8.91 GiB (3.28x).
- **`json_object` accepted a trailing comma (#1096) and the schema-less FSM
  emitted structurally invalid JSON (#1067)**: both container-stack bugs; the
  schema FSM was never affected.
- **Llama 3.2 tool calls were dropped**: 3.2 emits a bare JSON object where 3.1
  used the `<function=F>` envelope. Accepted now, but only for a name the request
  offered: a fabricated call is worse than a missed one. Llama-3.2-3B 4/6 → 6/6.
- **An undersized `kv_cache.swa_snapshot_mb` silently disabled prefix caching**:
  strictly worse than `0`. It now warns with the required size and both ways out.
- **`/image` in the interactive CLI loaded a picture the prompt never
  referenced**: the multi-turn path branched on the mmproj tower alone, so on
  Qwen3-VL it rendered a prompt with no image tokens.
- Tool-call enforcement derives from the post-load template; multi-turn tool
  replay re-renders in the XML shape the model emits; Qwen XML close tags are
  matched newline-anchored. `--mem-report` counted the FP8 SSM sidecar twice.

### Removed

- **`speculative.recycle_loop`** (verify-in-loop) and ~1.5k LOC of support. A
  nine-class prompt sweep found no class where the loop beat the same
  configuration with it off; against the eager drafter it cost a consistent
  5.6-8.3 %. The eager `speculative.token_recycling` stays (measured neutral).

## [0.19.2] - 2026-07-17

Hardening release: a latent KV prefix-cache corruption fix plus a
diagnostics/robustness sweep. Decode measured neutral at every step
(Qwen3-Coder-30B NVFP4 spec-OFF tg256 402.6 ± 0.3 tok/s).

### Added

- **Post-launch CUDA error checks at 399 kernel-launch sites**: new
  `IMP_CUDA_CHECK_LAUNCH()` (cudaPeekAtLastError-based, so downstream propagation
  is unchanged). Launch-config failures surface where they happen instead of at
  the next synchronizing call; coverage ~1 % → 100 % of `<<<>>>` sites in `src/`.
  (PR #1044)
- **RAII owners for CUDA graph handles** (`core/cuda_raii.h`): every manual
  destroy+null pair replaced, error paths structurally leak-safe, semantics
  preserved 1:1. (PR #1045)
- **`cache_control` per-breakpoint pin boundary**: the LAST marked block now
  bounds the prompt-KV pin instead of always pinning the whole prompt, which
  reduces pin-budget pressure in multi-turn agent loops. Additive; 7 new contract
  tests. (#1046, PR #1049)
- **`make asan`**: reproducible host-code ASan+UBSan over the CPU test binaries,
  WSL2-capable (unlike compute-sanitizer). Baseline: 0 imp-code findings. (#1047)

### Fixed

- **KV prefix-cache block double-ownership after rollback.** Rollback freed
  hash-registered blocks to ref 0 without erasing the hash entries, so a later
  same-prefix allocation `inc_ref`'d a block sitting in the free list and the next
  free pushed it in twice, silent cross-request KV corruption. Trigger: KV-pool
  pressure during prefix-cache allocation plus a same-prefix retry, with prefix
  caching default-on. Found by the new `LeakUnderSustainedChurn` test. (PR #1044)
- **Sticky CUDA error after the expected graph exec-update fallback**: a handled
  reinstantiate path left a stale per-thread error until teardown cleared it with
  a WARN. (PR #1048)
- Last two compiler warnings cleared, a full rebuild is 0 warnings under
  `-Wall -Wextra -Wpedantic`. (PR #1044)

## [0.19.1] - 2026-07-17

### Added

- **Per-layer attention routing for heterogeneous models** (Gemma-4 dual head_dim
  256/512): the hd=256 SWA majority rides FA2 f16-QK while hd=512 global layers
  stay on the faster materialized cuBLAS path, with a new fused WMMA FMHA hd=512
  instantiation as the O(n)-memory fallback. Gemma-4 keeps full 2048-row chunks at
  any context (was ~190 rows at 64k); Gemma-4-12B pp16384 **+5.3 %**. (PR #1042)
- **`gemm.fp8_attn_proj`**: per-row-scale FP8 E4M3 decode sidecar for
  full-precision attention projections. Default `auto` = full q/k/v/o on gpt-oss,
  whose BF16 projections had no decode cache and ran as 2 B/elem FP16 GEMVs
  (33.5 % of the decode window): gpt-oss-20b decode **349.7 → 391.2 tok/s**,
  turning the llama.cpp b9976 tie into a +13-19 % lead. PPL unaffected by
  construction. (#984, PR #990)

### Changed

- **Dependency bumps**: CUTLASS v4.5.3 → v4.6.1 (measured perf-neutral on sm_120
  decode) and cpp-httplib v0.48.0 → v0.50.1, which picks up three security fixes
  including a TLS use-after-free in `SSLClient`, imp-server uses it for image-URL
  fetches. Dockerfile ARG defaults re-synced with `cmake/imp-deps.cmake`.
- **`gemm.nvfp4_lm_head` is now `"auto"` with a per-model net rule**: ON for
  native BF16/F16 heads (+8-16 % decode, +2.2 % PPL) and small dense GGUF heads,
  OFF for larger or MoE GGUF heads, which measured net-negative. Legacy
  `true`/`false` still parse. (#982, PR #990)

### Fixed

- **n-gram spec decode was -39 % at moderate context on GGUF K-quants.** The
  verify chunk took the M>1 prefill dispatch, which dequantizes the full
  quantized source per GEMM, so on Qwen3-14B Q6_K a verify step cost ~7x a decode
  step and speculation lost even at 100 % accept. Verify GEMMs now read the NVFP4
  decode overlay: 14B Q6_K at ctx 2048 **91.9 → 153.2 tok/s**, 8B Q8_0
  **209.8 → 312.6**. Greedy output byte-identical to spec-off. (#998)
- **Prefill CUDA graph replayed stale chunk geometry on continuation chunks**:
  the graph was only invalidated on (chunk_len, block_count) changes, so chunk 2+
  attended with chunk-1 geometry and silently truncated long context
  (teacher-forced PPL 8.30 → 15.35 past the second chunk on Qwen3-4B Q8). Only
  offset-0 chunks are captured now. (#981)
- **Quantized-KV prefill chunks no longer attempt graph capture**: the
  dynamic-scale append does a D2H absmax sync per chunk, which is illegal under
  capture, so every chunk aborted its capture and wasted a full forward. Every
  >2048-token prompt on a Qwen3 GGUF had been paying this.
- **KV-pool exhaustion at decode is diagnosable**: the reject-newest cancel was
  completely silent and surfaced as a bare "internal error". The scheduler now
  logs the exhaustion with remedies, warns at admission when a prompt leaves less
  than one KV block of headroom, and `imp_decode_step` returns
  `IMP_ERROR_CANCELLED` instead of `IMP_ERROR_INTERNAL`. (PR #1042)

## [0.19.0] - 2026-07-12

First cross-engine PPL-parity measurement, with two real tokenizer/quant fixes
out of it; dense n-gram speculation now wins long context; batched serving
861 → 1173 tok/s at 16 streams.

### Added

- **Suspend to RAM** (`POST /admin/suspend` / `/admin/resume`), park the loaded
  weights in host RAM and free the GPU completely, then resume in seconds. Only
  weights stay warm; sessions and KV do not. Models whose device buffers are
  transformed in place answer a clean 501, and capture is gated on host
  `MemAvailable` (507) rather than driving the host into swap.
- **On-disk warm weight cache** (`[warm_cache] enabled`, default on): the first
  cold load persists its *transformed* uploads to `~/.cache/imp/warm`; later boots
  mmap them. Near-zero for GGUF and NVFP4, ~model-size for BF16-dense where it
  saves the most. Version- and fingerprint-guarded.
- **`diagnostics.ppl_first` / `.ppl_last`**: NLL counting window for
  `--perplexity`, matching llama-perplexity's `first = n_ctx/2` for exact
  cross-engine alignment.

### Changed

- **Dense n-gram speculation now WINS on long context.** The verify chunk moved
  off the small-M prefill FA2 tile onto the batched-decode split-K paged kernels
  (557 → ~65 µs/layer at 16k), plus a depth-aware gate that discards 1-token
  drafts past the point where a verify stops paying. Versus spec-off on
  Qwen3-8B-Q8_0: **+45 % at 512 ctx, +27 % at 13312, -0.6 % at 15872**, where the
  same three points read -8 % / -23 % / -62 % before. Output token-identical to
  greedy. (#964)
- **FP8 KV cache auto-enables on GGUF Qwen3 dense/MoE.** GGUF exports never
  declare the FP8 hint the auto policy required, so long-context GGUF decode was
  leaving ~40 % on the table (Qwen3-8B Q8_0 at 16k: **150 → 211 tok/s**). Only
  families measured PPL-neutral at 16k are admitted (≤0.15 %).
- **Concurrent decode at 16 streams: 861 → 1173 tok/s (+36 %)** on
  Qwen3-Coder-30B-A3B-FP4, above the published vLLM reference for this model class
  on a 5090 at per-stream TPOT parity, single-stream unchanged. Four
  nsys-attributed fixes, the largest being one pinned strided sampling readback
  per step instead of a pageable one per row (which blocked the host ~850 µs per
  sequence per step).
- **Pipelined batched decode** (`runtime.decode_pipeline`, default on), at n≥2
  with CUDA graphs one decode step stays in flight, so host bookkeeping overlaps
  GPU compute. Coder-30B-FP4 at 16 streams 915 → 970 tok/s, TPOT 17.0 → 16.1 ms;
  n=1 never pipelines. Uniform-composition runs are bit-identical to the per-step
  path.
- **`gemm.nvfp4_lm_head_cutlass` default ON**: the batched-decode LM head runs as
  one CUTLASS NVFP4 GEMM, one head weight read per batch instead of ceil(n/4).
  This was the opt-in behind the 1173 tok/s headline. PPL cost +1.9-2.1 % on
  MoE/hybrid, +0.2-0.5 % on dense; batch=1 output bit-identical either way.
- **FP8 SSM-projection decode sidecar extended to GGUF hybrids**
  (`gemm.fp8_ssm_proj`): the Q8_0-kept GDN projections of UD quants were in no
  decode cache at all. Qwen3.6-35B-A3B UD-Q4_K_M decode **224.4 → 272.0 tok/s**,
  ahead of llama.cpp's ~229; PPL +1.8 %, a documented trade. Sub-8-bit GDN sources
  are excluded on purpose.
- **Roofline baseline re-pinned** (config_version 4): the old pin predated
  FA2-hd256 and the FP8 sidecar. Adds an `nvfp4-hybrid` cell and reaches 0
  unclassified kernels, from 51-63 % of the q4k-moe prefill window.

### Fixed

- **Qwen3.5/3.6 GGUF tokenization was non-canonical**: `tokenizer.ggml.pre =
  "qwen35"` fell through to the gpt2 per-char-punct fallback, over-splitting
  symbol runs by +13 % tokens on a 95 KB corpus. Now routed to the qwen2 scanner,
  token streams verified identical to `llama-tokenize`. Found by the first
  PPL-parity sweep.
- **The NVFP4-LM-head opt-outs were dead on GGUF checkpoints**: the collector
  added the head unconditionally, so `gemm.nvfp4_lm_head=false` silently did
  nothing. That head's quantization is the entire cross-engine PPL gap vs
  llama.cpp (+1.5…+4.8 %).
- **Qwen3.6-35B illegal memory access / silent garbage past 16k context**: when
  StreamingLLM auto-enabled, eviction retained a ceil-aligned window while the
  paged decode kernels start reading floor-aligned, so they read a freed -1
  sentinel block. (#963)
- **gemma-3-12b GGUF decode illegal-memory-access**, the last hard crash in the
  known-issues list: VRAM-budget starvation silently dropped 35 of 49 tensors to
  the from-scratch NVFP4 build that corrupts decode. Those weights now stay on the
  dequant-at-decode path, a bandwidth loss, never garbage.
- **KV floor now covers the full advertised context on cheap-KV models**: the old
  min(16384, 4×max_seq_len) floor gave a hybrid a 16.4k pool that a 16k prompt
  fills to 94 %, tripping StreamingLLM on a request that fits outright.
- **Greedy request-order independence in default mode**: the documented "30B
  NVFP4 MoE nondeterministic at temp=0" flipper: the first request of a process
  ran one decode step on a different kernel mix than every later one. The decode
  graph pool is pre-armed in warmup and `runtime.warmup` defaults true; verified
  3 processes × 12 requests = 36/36 byte-identical.

### Removed

- **`gemm.nvfp4_ssm_proj`**: bit-rotted in the tier refactors (71 tok/s against
  its original 248 on Qwen3.6-35B Q4_K_M) and superseded by the GGUF branch of
  `gemm.fp8_ssm_proj`, which is both faster and quality-safer than 4-bit into the
  recurrent scan.

## [0.18.1] - 2026-07-10

### Changed

- **The per-token SSE streaming loop is one shared driver**
  (`tools/imp-server/stream_driver.cpp`): the outer token loop was hand-copied
  per dialect (~600 LOC each) and kept drifting (#892, #941). The three handlers
  are now thin wire-format adapters. Net -732 LOC. (#951)
- **Directory sweep**: removed four March-era one-off tools with zero references
  and the dead `CUDA_ARCHITECTURES` build-arg from `docker-compose.yml` (silently
  ignored since the sm_120a-only build).
- **Docs consolidated under `docs/`; repo root holds only README / CHANGELOG /
  CONTRIBUTING / AGENTS.md / CLAUDE.md.** Superseded point-in-time reports removed
  (full text in git history, indexed in `docs/archive/README.md`).
- **Structural audit #6** (`docs/archive/structural_debt_2026_07_10.md`), swept
  the ~40 PRs since audit #5; confirmed findings filed as #941, #942, #943.
  Cleanup landed alongside.
- Internal single-sourcing: the SSM conv-channel count is a derived
  `ModelConfig::ssm_conv_channels()` (was hand-derived at 9 sites), and the
  native-NVFP4 cache-demand scan runs once per engine init instead of four times.
  (#952)

### Added

- **`gemm.fp8_ssm_proj` (default ON)**: FP8 E4M3 decode sidecar for the
  native-precision GDN/Mamba projections on NVFP4 hybrids: Qwen3.6-35B decode
  **268.6 → 320.3 tok/s** spec-off. These projections were the single largest
  decode slice (34.6 % of GPU time as FP16 GEMVs) because producer recipes exclude
  them from NVFP4 and 4-bit GEMV on their wide shapes *regresses*. Per-row scales
  keep PPL flat (one per-tensor scale over the fused pack cost +4 %).

### Fixed

- **The decode CUDA graph re-derives its launch topology when the context
  high-water mark grows**: a long-prompt request after short ones no longer
  wedges the engine. The intended re-capture trigger never fired because the
  decode batch pool pads `max_blocks_per_seq`, so a graph captured at ctx≈35
  replayed a stale topology at ctx≈2400 and faulted, after which every request
  returned 0 tokens after 300 s. The full degeneration suite now passes against
  the Qwen3.6-35B server for the first time. (#948)
- **Drift bugs surfaced by the streaming-loop unification** (#951):
  `/v1/responses` emitted an empty `function_call_arguments.delta` for buffered
  tool calls and reported `reasoning_tokens: 0` regardless; `/v1/messages` streams
  were missing `imp_inter_token_seconds`, dropped mid-args tool-call arguments and
  left the engine request running on a failed keepalive write; neither route set
  the think-budget state, so enforcement never engaged there.
- **Streaming `/v1/responses` requests update server metrics and send SSE
  keepalives**: the loop never touched `requests_total`, the token counters or
  the histograms, and emitted nothing during long prefills, so reverse proxies
  could kill the idle stream. (#941)
- **The pre-upload KV reserve computed 0 bytes for NVFP4/MXFP4_KV cache dtypes**:
  it multiplied by raw `dtype_size()`, which has no case for packed 4-bit KV. The
  packing- and scale-aware calculation is now shared between the VRAM planner, the
  expert-offload reserve and the KV init log. (#942)
- **`workspace_estimate()` no longer charges the 256 MiB cuBLAS S-matrix on
  FA2-served configs**: the allocator has skipped that buffer since #932, but the
  estimate still held phantom headroom out of the cache/KV planners. (#943)
- **An explicit `enable_thinking: true` is honored on templates that default the
  switch to a closed block** (e.g. Qwen3.5-4B). Only `false` was ever stamped, so
  an explicit *true* was dropped and the model answered directly instead of
  reasoning. Without an explicit request the variable stays undefined, so the
  template author's default wins.

## [0.18.0] - 2026-07-09

### Added

- **Stage-1 HD=256 FA2 port** (`attention.fa2_hd256`, default off): the
  register-resident FA2 prefill kernel gains head_dim=256 instances (the
  double-buffer would need 135 KB smem against the 99 KB sm_120 opt-in; the pv-f16
  variant fits in 228 registers with zero spills). On Qwen3.6-35B-A3B-NVFP4:
  kernel **4.3x** the SMEM-tiled WMMA FMHA, e2e prefill +10.6 % pp4096 / +24.8 %
  pp8192, teacher-forced PPL 10.44 vs 10.58 baseline.

### Changed

- **HD=256 FA2 is default-on, and the FP8-KV deterministic forcing is lifted for
  it.** head_dim=256 models (Qwen3.6 hybrids, gemma-3-class) route prefill through
  the f16-QK FA2 kernel; single-shot prefill gains the chunked path's uniform-shape
  refinement, so GDN/Mamba2 hybrids take FA2 at any head_dim instead of falling to
  cuBLAS. Learned sinks (gpt-oss) and heterogeneous per-layer shapes (gemma-4) keep
  cuBLAS.

### Fixed

- **Qwen3.5-4B-mxfp4 returns its answer in `content`, not `reasoning_content`.**
  The server defaulted thinking ON whenever the template merely *mentioned*
  `enable_thinking`, but this template defaults it to a pre-*closed* empty block,
  so the splitter trapped the whole answer in `reasoning_content` with empty
  `content` on every endpoint. The flag is now reconciled against what the
  template actually rendered into the prompt tail. Genuine reasoning models are
  unaffected. (#934 follow-up)
- **MXFP4-GDN hybrids no longer serve `!!!…` garbage when VRAM is tight.** On GDN
  hybrids the native MXFP4 GEMV is unavailable, so decode needs the FP16 dequant
  cache (~4x the raw MXFP4 bytes) resident, and the VRAM budget never charged for
  it: the KV pool ate the headroom, the fallback failed to allocate, and decode ran
  against weights with no usable kernel. The budget now reserves it up front, and
  the fallback throws a legible out-of-VRAM error instead of silently skipping the
  alloc. (#934)
- **Qwen3.5-4B-mxfp4 decode no longer aborts CUDA-graph capture with cuBLAS
  status-14.** The GDN projection GEMM (FP16, N=32) was rejected by the
  capture-safe WMMA kernel's `N < BN` guard and fell through to cuBLASLt, which
  fails under stream capture on sm_120. The kernel already masks partial tiles in
  both load and store, so the guard was needlessly conservative. Pinned by a new
  N=32 test.
- **Deterministic cuBLAS GEMM validates its algo choice.**
  `runtime.deterministic_gemm` picked cuBLASLt's top heuristic candidate blindly,
  skipping the per-candidate warmup the timing path uses precisely because the
  heuristic can return algos that fail at runtime on sm_120: a bad `results[0]`
  then failed with status 14 and the `void` wrapper continued with a garbage
  buffer. It now probes candidates in stable heuristic order and takes the first
  that survives. Determinism preserved.
- **A totally-failed GEMM is fatal instead of silent garbage.** When both
  `cublasLtMatmul` and the `cublasGemmEx` fallback fail, `gemm` throws rather than
  leaving an uninitialised output buffer for the forward pass to turn into
  gibberish.

## [0.17.3] - 2026-07-09

Native-NVFP4 serving fix release: the mandatory decode caches are reserved before
the elastic VRAM consumers, so large NVFP4-prequant MoE models reach full decode
cache coverage and captured decode graphs under pure default config.

### Fixed

- **VRAM ordering: mandatory NVFP4 decode caches are reserved before workspaces
  and the KV pool.** They were built last from already-starved free VRAM, and a
  partial cache aborts decode graph capture, which pinned Qwen3.6-35B-A3B-NVFP4 at
  **26-40 tok/s** under default config. A balloon allocation holds the exact demand
  until the cache build: default config now reaches full caches and a captured
  decode graph, **247-249 tok/s** with a 138k-token KV pool. GGUF/FP16 budget
  arithmetic is byte-identical (pinned by test); escape hatch
  `[vram] native_cache_reserve`. (#926)
- **Loud WARN when the KV pool collapses below its token floor.** With an oversized
  `max_batch_size` the batch-scaled workspaces can shrink the pool to the 16-block
  minimum; longer requests were silently cancelled at admission while `/v1/models`
  kept advertising the full context. Log-only. (#927)

## [0.17.2] - 2026-07-08

### Added

- **Context-window auto-detection across the three live conventions**, so
  OpenAI-compatible clients can stop keeping a hard-coded table: `GET /v1/models`
  carries vLLM's `max_model_len` and llama.cpp's `meta.n_ctx_train`, plus new
  `GET /props` (llama.cpp shape) and `GET /info` (TGI shape). All three report the
  same engine-detected `max_seq_len`. (#921)

## [0.17.1] - 2026-07-08

### Changed

- **Adopted C++23 idioms across the tree, no functional change** (#919):
  `std::to_underlying` at ~67 enum-cast sites (a hard compile error on non-enums,
  so the build itself confirms every site), `deducing this` collapsing four
  duplicated const/non-const accessor pairs, and `static operator()` on six
  stateless functors. `std::expected`, `std::mdspan`, `std::print` and
  `[[assume]]` were deliberately left out.

### Tests

- Cover `format_tool_response` and `reconstruct_tool_call_output`. (#914)

## [0.17.0] - 2026-07-08

Toolchain-modernization release: the engine builds as **C++23** on an **Ubuntu
26.04 / GCC 15.2 / CUDA 13.3** base (was C++20 / Ubuntu 24.04 / GCC 13). Decode
verified neutral.

### Added

- **FP8 tile decode-attention kernels**: token-tiled FP8 split-K decode with K and
  V staged in one cp.async group, long-context decode **+51 %** (#899); a
  GQA-batched variant reads each KV head once across the warp group for a further
  **+14 %** (#900).

### Changed

- **C++ standard raised to C++23** (host + CUDA). CMake's NVIDIA-CUDA module has no
  CUDA23 dialect flag, so the build teaches it `-std=c++23` explicitly. No source
  changes were required. (#916)
- **Build toolchain to Ubuntu 26.04 / GCC 15.2**, which catches the GCC-15
  missing-include class in CI. Note: nvcc silently drops `-std=c++23` on a host
  compiler older than GCC 14, so dev/profiling images must be on this base; the
  `impdev:ncu` recipe is committed at `tools/Dockerfile.ncu`. (#907)
- Retired the legacy config surface: env-var seeding down to `IMP_DETERMINISTIC` +
  `IMP_FMHA_FA2`, turboquant aliases and dead flags (#879); `imp.conf.example`,
  `--help` and config comments synced to parser reality (#878).
- VRAM-layer audit: dead modules removed, one reserve floor, honest budget logs
  (#877). Tokenizer dropped its duplicated JSON parser for shared
  `model/json_util` (#887). Analysis/roofline tooling tracks the latest toolkit
  (#908, #904).

### Fixed

- **MLA (DeepSeek-V2/V3) YaRN rope-mscale**: the RoPE cos/sin were scaled by
  `yarn_get_mscale(factor, mscale_all_dim)` instead of the HF ratio, inflating the
  rotary embedding by 1.261x on V2-Lite, and the error compounds with position.
  Same-corpus PPL against HF bf16 on DeepSeek-V2-Lite: 534-token **+24.4 % to
  +2.75 %** (imp 7.78 to 6.43, HF 6.25). Generalizes correctly to V3, where the
  two mscales differ. (#880)
- **MTP draft-head mrope applies YaRN / rope-scaling** (was plain NeoX RoPE), so
  the drafter no longer drifts from the verifier on rope-scaled models. (#913)
- **Async mempool is trimmed on `Model` teardown**, not only at the C-API
  boundary, releasing device memory between in-process model swaps. (#915)
- **A failed CUDA-graph capture no longer wedges the engine**, plus planner-driven
  KV-pool sizing. (#874, #875)
- **GCC 15 build**: added the `<algorithm>` / `<numeric>` includes libstdc++15 no
  longer pulls in transitively. (#903, #906)
- No-GPU audit sweep #888-#894: server admission control, observability, `/health`
  locking, embeddings, API strictness and tool-call suppression, plus dead code and
  doc drift (#901).

## [0.16.2] - 2026-07-04

FP4-attention research batch: the #846 program (SageAttention3, ThriftAttention,
KV-append-quant) is closed end to end with measurements on every branch. All new
knobs are research scaffolds and ship **default-off**.

### Added

- **`attention.mxfp4_promote_budget`** (default 0): ThriftAttention-style outlier
  block promotion in the MXFP4 FMHA, where the top-scoring fraction of visible KV
  tiles computes exactly instead of FP4. Takes the FP4 attention quality gate from
  +9.9 % / +4.4 % NLL to **-0.6 % / -0.2 %** at a 5 % budget. (#870)
- **`attention.mxfp4_paged_kv`** (default off): chunked-prefill continuation reads
  K/V directly from the paged NVFP4 KV cache, with the current chunk staying fresh
  FP16 via force-promoted tiles. Quality gate passes (+0.34 % NLL at 9.3k);
  kernel-level perf refuted, so it ships as a quality-validated scaffold. (#872)

### Changed

- Docs: `MISSION_JOURNAL` records the full measurement chain. FP4-MMA delivers as
  advertised (tensor pipe 40.8 % to 2.2 %) but in-kernel K quantization costs
  3.34x FA2's instruction budget; quantizing the RECENCY window is the entire
  quality cost of FP4 KV storage (stored-FP4 current chunk +3.7-5.4 % NLL even
  with exact compute, stored-FP4 past is roughly free). (#871, #872)

## [0.16.1] - 2026-07-04

Spec-verify economics batch (#847 ladder): chunk-path overhaul, default
suffix-speculation on Qwen3.6-27B prompt-echo **81 to 131 tok/s** (+61 % vs
v0.16.0), 35B-A3B +10-15 %.

### Added

- **Encoder/embedding-model support**: `nomic-bert` GGUF checkpoints load into a
  dedicated encoder path (bidirectional no-KV forward, BERT WordPiece tokenizer,
  true LayerNorm-with-bias kernel, mean pooling and L2 on device).
  `/v1/embeddings` serves it; HF-oracle-verified at cos ≥ 0.999 on Q8_0. Classic
  BERT/bge/e5 stay rejected. (#836)
- **SuffixDecoding-style suffix drafter**: a per-request suffix index over prompt
  plus generated output with frequency-voted continuations and adaptive draft
  length replaces plain n-gram prompt-lookup as the default draft source
  (`speculative.suffix`). (#848)
- **Speculative decoding on hybrid (GDN/SSM) models** (`speculative.hybrid`,
  default on): the verify chunk snapshots the committed recurrent-state slab and
  restores it on partial acceptance. Measured greedy: Nemotron-3-Nano-30B code-edit
  **+60 %**, Qwen3.6-35B code-edit +18 %, Qwen3.6-27B prompt-echo **+156 %**.
  Token-lossless verified. (#847)
- **MTP verify activation** (`--mtp-spec-decode <k>`, default off): the trained MTP
  head feeds the verify loop when the suffix matcher has no match. Loads both
  sidecar MoE heads and embedded dense heads, with an economics guard that dooms
  MTP drafting per request when average emitted/verify cannot beat the async loop
  (measured: accept 44-91 % but net-negative on current verify economics). (#847)
- **MTP chain lm_head via the NVFP4 decode cache**: the per-draft full-vocab GEMV
  reads the NVFP4 LM-head cache when one exists, ~3.5x less HBM traffic (2.5 GB to
  0.7 GB per drafted token on Qwen3.6-27B's 248k vocab). Draft-only precision,
  verification stays lossless. Note a chain of k emits at most k+1 tokens per
  verify, so k=2 cannot pass the default economics threshold. (#847)
- **Graph-captured verify chunk**: per-(bucket x KV-tier) CUDA graphs replace the
  eager verify forward, reading the real KV length from device so a captured graph
  replays correctly as context grows. Coder-30B echo **+65 %**, Q8 +7 %;
  Mamba2/GDN hybrids capture too but measure ±0, being scan-dominated rather than
  launch-bound. (#856, #855, #859, #861)
- **Opt-in schema jump-ahead** (`constrained.jump_ahead`, default OFF): a
  char-level FSM probe drafts structurally-forced spans as teacher-forced chunks,
  every emitted token still masked-sampled from its true logits row. Measured
  net-negative (-11 % on 14B-NVFP4) because the model picks context-dependent
  tokenization splits the canonical draft misses. #844 closed; kept as scaffold.
  (#849)
- **NVFP4-attention research knobs** (#868, idea #846, refuted):
  `attention.mxfp4_blockscale`, `mxfp4_ksmooth`, `mxfp4_pv_fp4`. Per-16 blockscale
  rescues FP4-QK from the catastrophic per-row failure mode (PPL 31546 to 5.90 on
  Qwen3-14B-NVFP4 at 199 tokens) but the residual noise compounds with context
  (+10 % NLL at 9k). All three default OFF.

### Changed

- **Small hd≠128 verify/boundary chunks prefer the tiled FMHA over cuBLAS.**
  cuBLAS re-runs its per-new-shape algo selection on every call (workspace memset,
  candidate benchmark, blocking event sync), and spec-verify chunks grow `ctx_len`
  every step, so hd=256 models paid ~93 such trios per verify. Measured on
  Qwen3.6-27B: verify **78 to 59 ms**, 34 to 44-46 tok/s (+31 %).
- **Small-M NVFP4 GEMM: batched GEMV replaces the dequant fallback** for M ≤ 16.
  The weight is read once per MR=4 tile at 0.25x FP16 bytes instead of
  dequantizing the whole weight every chunk: `dequantize_nvfp4_kernel` drops from
  48 % to 4 % of GPU time on a Qwen3.6-27B MTP-only verify.
- **Batched verify/eval LM heads**: the spec-verify logits GEMV over drafts+1 rows
  reads each weight tile once per 4-row batch instead of once per row (Coder echo
  **+50 %**), and the dp4a LM-head path gets the same treatment. (#854, #857)
- **Device-side MTP draft chain** (dense MTP heads): chain step i's argmax lands in
  a device slot and feeds step i+1's embedding lookup on-device, so one D2H drains
  the whole chain. E2e-neutral today, but required groundwork for capturing the
  chain into a graph. MoE MTP heads keep the host loop.
- **Persistent K/V gather scratch for the eager chunked path** replaces a
  ~140-alloc-per-verify `cudaMallocAsync`/`FreeAsync` pair on hybrids, removing the
  acknowledged hot-loop-malloc exception.

### Fixed

- **Non-gated NVFP4 MoE da_cache never built**: `!empty()` id checks against the
  loader's pre-sized all-invalid id vectors meant RELU² models took the per-call
  H2D fallback, whose `cudaMemcpyAsync` from stack vectors replays from dead stack
  addresses once recorded into a verify graph. With the cache built, hybrid verify
  graphs record and replay cleanly (Nemotron-3-Nano 23/23, deterministic). (#860,
  #861)
- **MoE host-args launches and NVFP4 capture-refusal now fail loud under stream
  capture**: the M>1 dequant fallback's silent return and an unchecked D2H both
  recorded graphs with missing work (`misaligned address` at replay, `<unk>`
  output). Also a genuine eager bug: the hybrid conv tail wrote zeros instead of
  shifting the previous state on chunks shorter than `conv_kernel`. (#858, #859)
- **Chunked-prefill q_offset and fully-masked-row guard in the opt-in MXFP4 FMHA**:
  continuation chunks masked with local row indices, and fully-masked rows could
  poison the online-softmax denominator. (#868)
- **Schema keys reject backslash escapes**: `OBJECT_KEY` accepted `\` and swallowed
  it, so the emitted text carried the escape while the FSM matched the raw key.
  (#850, #851)
- **tools + json_schema preamble slack raised**, so the schema mask no longer fires
  mid-deliberation on tool-call requests. (#840, #842)

## [0.16.0] - 2026-07-02

### Added

- **Hard per-process VRAM budget**: `--vram-budget <mb>`, `[runtime]
  vram_budget_mb`, and the previously-inert C-API field. All 19 sizing sites see a
  virtual GPU of the given size, so multiple imp-server processes can share one
  card; a co-tenant's pre-existing usage never counts against this process's
  budget. Verified with two simultaneously-started servers (9000 + 8000 MiB) at
  15.9 GiB device total. Best-effort cap, so leave ~1 GiB real headroom. (#838)

### Fixed

- **Model unload leaked weights-sized VRAM** (~8.3 GiB per Qwen3-8B-Q8_0 cycle):
  weights are `cudaMallocAsync`-allocated but were freed with plain `cudaFree`,
  which returns success WITHOUT returning the blocks to the async mempool on this
  stack. Freed with `cudaFreeAsync` everywhere. The reload test now probes actual
  re-allocatability, since WSL2/WDDM under-reports reclaimed pages. (#834, #837)
- **Encoder-only models are rejected on the SafeTensors/HF path too**:
  `is_encoder_only_arch` was case-sensitive, so HF class names slipped past the
  GGUF-only reject and ran a BERT encoder through the causal-LM prefill and
  sampler, giving an illegal memory access on the first `/v1/embeddings` request.
  (#818, #835)
- **A second engine on the same loaded model handle no longer IMAs**: for GGUF
  MXFP4 GDN models the first engine consumes the model sources destructively, so a
  create/free/create cycle rebuilt caches from dangling memory. `Engine::init` now
  rejects a second engine on a consumed model with a clear "reload the model"
  error. (#830, #835)

## [0.15.0] - 2026-07-02

### Added

- **Prefix caching for hybrid (SSM/GDN) models via recurrent-state snapshots.**
  Reused KV blocks alone cannot skip prefill on a recurrent model, so the engine
  snapshots the per-sequence state slab once per prefill at the largest
  block-aligned prompt position and restores it on a hit. Per-turn TTFT on
  Qwen3.6-35B-A3B-NVFP4 goes from 1.6-6.7 s (linear in history) to a flat
  **1.4-1.9 s**, a 3.5x win at ~10k tokens of history. New
  `server.recurrent_snapshot_mb` budget (default 256 MiB). (#831)

### Changed

- **Hybrid concurrent decode is fair (round-robin) instead of head-of-line.** The
  recurrent scan kernels are single-sequence, so sessions time-slice the decode;
  previously the batch-1 clamp kept the oldest request every step and a second
  session produced its first token only after the first finished (measured: 6.6 s
  of starvation). The slice now rotates every
  `runtime.hybrid_decode_quantum` tokens (default 128). (#831)

### Fixed

- **Prefix-cache reuse no longer counts a chain hole as a reused prefix.** LRU
  eviction can drop an early block while later blocks survive; reuse now stops at
  the first non-cached block, so the caller never skips prefill over a hole with
  uncomputed KV. (#831)
- **The decode graph pool invalidates when the recurrent state slot changes.** The
  captured graph bakes one slot's state pointers, so overlapping request lifetimes
  could replay a graph against a different sequence's state. (#831)

## [0.14.0] - 2026-07-02

### Added

- **OpenAI Predicted Outputs (`prediction`)**: client-supplied predicted
  completion text is tokenized into the n-gram draft corpus, never forwarded
  through the model, so output stays a faithful greedy decode.
  `usage.completion_tokens_details.{accepted,rejected}_prediction_tokens` is
  reported. (#825)
- **Streaming through the conditional-graph decode loop**: streaming requests poll
  the mapped ring buffer per token instead of taking a per-step host round-trip, so
  SSE delivers each token as the device burst runs. Max inter-token gap Q8 197 to
  17 ms, MoE 28 to 7.6 ms. (#822)
- **Agentic API-compliance batch**: streaming tool-call dialects (Gemma-4,
  Qwen3.6 XML) via a shared stream filter, `/v1/messages/count_tokens`, OpenAI SSE
  keepalives, `max_completion_tokens`, stop-cap 4 to 16. (#818, #820)
- **`tools/multiturn_bench.py`**: agentic multi-turn replay benchmark with
  per-turn TTFT/decode via SSE, OpenAI-compatible so it runs unchanged against
  vLLM/llama.cpp. (#826, #827)

### Performance

- **n-gram speculation on native-NVFP4 MoE**: the `is_moe` gate is relaxed for
  native-NVFP4 experts (`speculative.moe`, default on), GGUF-MoE stays on the async
  loop where verify re-dequantizes experts. Qwen3-Coder-30B-A3B-FP4 code-edit
  **+49-81 %** at 93 % draft acceptance, Modelopt-30B +29 %. (#824)
- **Serving latency**: decode-aware prefill chunk cap while a decoder is active,
  admission-aware decode bursts so a waiter's prefill is not starved, and
  cross-turn output KV reuse. All no-ops for single-stream; greedy
  byte-identical. (#823)

### Fixed

- **Native-NVFP4 VRAM budget starved the weight caches**: the budget estimated 0
  bytes for native-NVFP4 checkpoints, so the KV hard-clamp took all post-weight
  VRAM and the CUTLASS SF cache never built. **Qwen3-Coder-30B-FP4 server 31.8 to
  300 tok/s**, dense Qwen3-14B-NVFP4 106 to 209. (#826)
- **Qwen3-Coder-30B multi-turn empty output**: imp's Jinja engine did not strip the
  template file's single trailing newline, so the generation prompt rendered with
  an extra blank line and the model emitted an immediate EOS on borderline
  multi-turn contexts. Templates without a trailing newline were unaffected. (#828)
- **json_schema not enforced under concurrency**: the engine-global
  `ConstraintManager` had per-request state holes. Constraint state is now
  per-request with an engine pool. (#821)
- **Long-context chunked-prefill abort / 32-token chunks**: the engine clamped
  every prefill chunk to `cap²/total` and aborted gpt-oss when the clamp hit 0.
  Offset-aware `max_safe_prefill_chunk()` plus `attn_shapes_uniform()` lets
  uniform-per-layer hybrids use the O(n) chunked paths: **Qwen3.6-35B pp10k +80 %,
  Nemotron +115 %, gpt-oss pp40k +69 %, Gemma-4 +35 %**, PPL parity ≤1 %. (#819)
- **Paged-decode split-K / cluster launch failure** falls back to single-split
  GQA/MHA instead of erroring. (#817)

## [0.13.0] - 2026-06-30

### Added

- **DeepSeek-V2 Multi-head Latent Attention (MLA)**, the first MLA architecture in
  imp. Stage A reconstructs the full K/V from the latent at projection time so
  every existing attention/paged-KV/RoPE kernel is reused unchanged (#802); an
  opt-in absorbed latent-KV decode stores only the 512-dim latent plus 64-dim
  decoupled-RoPE key for the long-context VRAM win (#803). Validated on
  DeepSeek-V2-Lite.
- **`gemma4_unified` multimodal Gemma-4 checkpoints** are mapped to the Gemma-4
  arch instead of falling back to the llama path and crashing. The text tower is a
  standard dense Gemma-4. (#814)
- **NVFP4-quantized GDN (`linear_attn`) projections** load for NVFP4 hybrid
  checkpoints, so the GDN path is coherent instead of garbage. (#812)
- **SafeTensors/NVFP4 prompt-test battery coverage**: `validate_safetensors.py`
  sweeps one model per (architecture, NVFP4 source-format) cell instead of a
  handful of MoE checkpoints. (#815)

### Fixed

- **gpt-oss GGUF 2^-4 residual rescale**: the official Q8_0-dense
  `gpt-oss-20b-mxfp4.gguf` produced garbage (PPL 2739) while the bf16 GGUF was
  fine. The rescale subtracted 4 from the biased exponent, which leaves denormal
  scales unscaled (16x too large) and flushes exponents 1..4 to zero; on the
  official file 91922/368640 blocks of the attention `wo` had exp==0, cascading
  into a ~20x layer-0 MoE blowup and wrong expert routing. Now scaled in the float
  domain: **PPL 2739 to 4.65**, matching the HF reference. A CPU test sweeps all
  65536 fp16 bit patterns. (#808, #809)
- **Gemma-4 garbled after a GDN/SSM model in the same process.** Gemma-4's global
  layers use head_dim=512, whose paged-decode GQA kernel needs a ~64 KiB shared
  memory opt-in, and that opt-in sat behind a one-shot `static bool` on a kernel
  shared by every model in the process. Re-armed as a high-water-mark opt-in. One
  model per process was never affected. (#815)
- **Cross-model CUDA-error-leak hardening**: a failed best-effort L2
  access-policy-window recorded a sticky per-context error that could outlive the
  engine and trip the next model's guarded kernels. (#815)
- **MTP draft head used silu where the head expects a sigmoid attn-output gate**,
  which crippled draft quality. Correcting it lifts K=1 acceptance from ~10 % to
  85 %+ on Qwen3.6. (#804)
- **NVFP4 MoE expert-scale `cudaFree` guard against offset pointers.** (#813)

## [0.12.6] - 2026-06-26

Patch release: a focused fix chain for the post-`</think>` answer-headroom logic
across all three reasoning formats. Short answers stop cleanly instead of
padding, and reasoning models no longer return empty content when `max_tokens` is
tight.

### Fixed

- **Post-`</think>` grace is content-aware**: the grace that suppresses a too-eager
  stop now releases the instant a real answer token appears, so a complete short
  answer stops on its own `<|im_end|>` instead of being padded to the raw-distance
  budget. (#798)
- **Whitespace tokens no longer release the grace**: the `\n` a model emits right
  after `</think>` must not count as answer content, otherwise a stop following it
  produced a 0-content completion (reproducible ~75 % on Qwen3.6 for terse
  prompts). (#799)
- **gpt-oss Harmony answer-headroom budget**: gpt-oss reasons in the Harmony
  `analysis` channel and has no `<think>` token, so the budget never armed and a
  tight `max_tokens` was consumed entirely inside reasoning. It now force-emits the
  whole final-channel opener when reasoning reaches the reserve limit, because
  forcing `<|end|>` alone lets the model re-open analysis. Corpus empty-content
  count **18 to 2**. (#800)

## [0.12.5] - 2026-06-26

### Added

- **Adversarial degeneration prompt corpus**: 250 prompts across 8 categories
  (repetition, think-leak, special tokens, adherence, long context, multi-turn,
  multilingual, format), driven by `degen_suite.py --corpus` instead of the
  previous five-prompt battery. (#795)
- **Task to skill routing table** at the top of `CLAUDE.md`, so a fresh session
  maps a task straight to the skill that carries its playbook. (#794)

### Changed

- **`DegenerationTest` loads SafeTensors/NVFP4 models**, not just GGUF, closing a
  coherence-coverage gap on the priority quant. (#792)

### Fixed

- **Streaming reasoning leaked into the `content` channel** while
  `reasoning_content` was also populated. The two streaming handlers shared a
  private demux that flipped to content at the *first* `</think>` and could not
  re-enter on a multi-token marker (Qwen3.6 ships them as multi-BPE added tokens).
  Now one shared `StreamReasoningSplitter` that re-enters via text scan and holds
  back only a partial marker. Also caps the think-budget answer reserve, which had
  force-cut models that finish thinking within the window. (#793)

### Removed

- **graphify knowledge-graph integration** (the skill, `.graphifyignore`, the
  ignore rules and the `CLAUDE.md` section): it bloated the repo without enough
  payoff for this codebase. (#794)

## [0.12.4] - 2026-06-25

### Fixed

- **Native-NVFP4 MoE server crash on the first request** (`gemm_nvfp4:
  B.shape[1]=… must equal weight K=…`). The weight-dispatch shim derived the M>1
  prefill GEMM's K from the weight handle's `shape[1]`, which is the *packed* K/2
  for prequant-loaded NVFP4 weights. K now comes from the activation, per the GEMM
  contract. The fallback was only reached because the v0.12.3 KV-budget defaults
  can starve the CUTLASS NVFP4 prefill workspace. (#790)

## [0.12.3] - 2026-06-25

### Added

- **Agentic server hardening**: per-request speculative-decode toggle,
  inter-token-latency and cancellation metrics, prefix-cache safety under
  concurrency, an Anthropic-style keep-alive ping, and an agent benchmark harness.
  (#770)
- **Long-context KV-budget defaults for agentic workloads**: the auto
  `max_seq_len` cap and KV-cache fraction are tuned so NVFP4 models no longer
  starve the KV cache while VRAM sits free. (#771)
- **Per-request vision binding**: images travel on the request and the worker
  encodes on admission, so vision requests batch with text instead of pausing the
  engine. (#774)
- **n-gram speculative decode is on by default** for dense models, gated off for
  MoE. (#781)
- **`parallel_tool_calls` is honored** on `/v1/chat/completions`: `false` emits at
  most one tool call, streaming and non-streaming. (#782)

### Changed

- **NVFP4 MoE decode ~2.6 % faster** (Qwen3-30B-A3B-NVFP4, tg256): the SwiGLU
  down-projection precomputes `silu(gate)*up` once per element instead of once per
  output row. Greedy output byte-identical. (#787, #788)
- **CI is much faster on PRs**: clang-tidy moved to its own non-required job scoped
  to changed files, so the required `Build` check dropped from ~26 min to **~4
  min**; docs-only PRs skip the CUDA build. (#785, #780, #783)
- **File-size gate** (`tools/check_filesize.py`) flags oversized translation units
  by recompile blast radius; the 12 largest god-files were split with no functional
  change. (#784)

### Fixed

- **Soundness and hardening**: four HIGH-severity fixes, being a KV-cache
  use-after-free on eviction, fail-fast on a poisoned context, Anthropic
  `x-api-key` auth, and a bounded decode-burst. (#772)
- **Vision global image bind restored** for the C-API / imp-cli path, a #774
  regression. (#776)
- **NUL byte removed** from a `handlers.cpp` comment that made `grep` treat the
  most-edited server file as binary. (#782)

## [0.12.2] - 2026-06-23

### Fixed

- **gpt-oss Harmony channels are parsed** instead of leaking into the response:
  analysis and commentary map to `reasoning_content` (Anthropic: `thinking`) and
  the final channel to `content`, with all control markup stripped. The MXFP4
  decode itself was already correct. (#765)
- **`/v1/completions` streaming is per-token again** for think-capable models; it
  was buffering the whole response into a single SSE frame. (#766)
- **Port-in-use fails fast**: the listen socket is bound before the model load, so
  a conflict errors in under a second instead of after a full load. (#766)
- **gpt-oss MXFP4 SafeTensors load** no longer prints the stale "no SafeTensors
  MXFP4 decode path" warning; the path exists. (#765)

### Changed

- **cpp-httplib v0.46.1 to v0.48.0** (security hardening plus a fix that ignores
  Range headers on unknown-length streaming responses).

## [0.12.1] - 2026-06-23

A bug-fix release closing the issues found in a black-box acceptance test of
v0.12.0. No kernel or perf changes.

### Changed

- **Prefix/prompt caching is ON by default for the server.** It shipped
  default-off, so `cache_read_input_tokens` always reported 0 and warm prompts got
  no TTFT win unless an `imp.conf` opted in, contradicting the documented
  behaviour. Still auto-disabled for SSM/GDN models. (#758)
- **The released Docker image ships `imp-bench`** (it is documented; CI keeps it
  off for build speed). (#760)

### Fixed

- **SSE streaming is real per-token again** on `/v1/chat/completions` and
  `/v1/messages`. A single-stream request buffered every token and flushed at
  generation end (TTFT ≈ full latency), because the GPU-autonomous decode loop only
  surfaces tokens once the whole burst completes. Streaming requests now stay on
  per-step decode (~2-5 % decode cost on 8B-Q8); non-streaming keeps the faster
  loop. (#754)
- **Streaming no longer hangs the client**: `/v1/messages` with a thinking model
  and a small `max_tokens`, and `/v1/completions` with `stream:true`, could spin
  forever without a terminal event when the final token was swallowed by the
  think-strip path. (#755, #757)
- **`response_format: json_schema` can no longer emit invalid JSON**: an unbounded
  `integer`/`number` field let a degeneration-prone model run a digit loop to
  `max_tokens`, leaving the document unterminated. (#751)
- **`think_budget = 0` disables thinking** as documented, instead of removing the
  budget cap, which made a think-capable model reason until `max_tokens` and return
  empty `content`. (#752)
- **A SafeTensors directory with a trailing slash is addressable again**: the model
  id is the path basename, so a trailing slash made it empty and every request was
  rejected. (#756)
- **Clearer model-load errors**: a present-but-corrupt file reports "invalid or
  corrupt model file" instead of "file not found", and a missing local path is no
  longer misrouted to the HuggingFace resolver. (#759)
- **Version string no longer drifts**: `imp_version()` was hardcoded `0.11.2` while
  the project was at `0.12.0`; it is single-sourced from CMake now. (#760)
- **Docs**: dropped the stale "no continuous batching" claim, added the required
  `model` field to the README quickstart curl, documented the C API as
  source-build-only. (#760)

## [0.12.0] - 2026-06-21

### Added

- **VRAM-aware auto `max_batch_size`, up to ~2.4x server throughput on MoE.** The
  old heuristic sized the concurrency cap by weight footprint alone, so a >20 GB
  model was pinned to batch=1 and served concurrent requests one at a time with
  ~10 GB of VRAM idle. The cap now derives from real post-load headroom. Measured
  on Qwen3-Coder-30B-A3B-FP4: auto cap **1 to 15**, aggregate decode **258 to 609
  tok/s at 16 concurrent**; Qwen3-14B-NVFP4 4 to 17. Serving-only: `imp-cli` and
  `--bench` still force batch=1, so the perf baseline is unchanged.
- **`imp-bench nvfp4`**, an isolated NVFP4 dense GEMM bench mode used to refute the
  cp.async-occupancy hypothesis on sm_120a.

### Changed

- **MoE NVFP4 models load materially faster**: the CUTLASS NVFP4 SF cache is
  slab-allocated in a single `malloc` instead of 18.6k (**-785 ms** on a 30B MoE),
  and the per-expert SfAtom conversion is batched (18.6k to 337 convert launches).
- **Zero-warning build and single-arch CI**: every remaining compiler warning
  silenced, and CI compiles `sm_120a` only, halving device-compile time. The
  shipped fatbin keeps the `compute_120f` PTX fallback for 5080/5070.
- **Internal clang-tidy cleanup**: ~51 host-side findings, mostly
  int-multiplication widening before 64-bit use in size math. No behaviour change.
- CI: `setup-python` v5 to v6 (Node 24).

### Fixed

- **Dense models no longer OOM-crash at startup under auto `max_batch_size`.** The
  auto sizing uses a 4096-token reference context but the KV pool is provisioned at
  `max_batch_size × max_seq_len`, so dense Q8 on a 32 GB card asked for 57.6 GB and
  aborted. The pool is now clamped to the VRAM that physically remains; it is paged
  with admission control, so a smaller pool only bounds concurrency under load.
- **`docker run imp:latest --help` prints imp-server's flags.** A leading flag was
  taken as the command name and fell through to `exec --help`, printing the bash
  builtin's help. The entrypoint now follows the standard official-image pattern.

## [0.11.3] - 2026-06-17

### Added

- **Stage-3 server test gate** (`make test-server`) boots a real `imp-server` and
  gates on the OpenAI and Anthropic wire batteries, plus a gcov coverage harness
  (`make coverage`).
- **Developer tooling and build hygiene**: `CMakePresets.json`, `AGENTS.md`, an
  in-repo `CLAUDE.md`, `.clang-tidy` + `make tidy`, a CI `lint` job, single-sourced
  dependency pins (`cmake/imp-deps.cmake`), `BENCHMARKING.md`, and
  `scripts/bench_gate.sh`.
- **Anthropic `/v1/messages` honours the `thinking` field**; it was dropped in the
  Anthropic to OpenAI conversion, so the request could not influence thinking at
  all. `budget_tokens` maps to imp's fractional `think_budget`.

### Changed

- **CUTLASS v4.5.1 to v4.5.2**, verified by a full build and 187/187 quant tests
  including the grouped-GEMM fp64 oracles: NVFP4 GEMM bit-exact under 4.5.2.
- **Internal hygiene**: a structural audit, 84 mechanical clang-tidy fixes, and a
  canonical sm_120a kernel-spec doc with standalone reference kernels.

### Fixed

- **Q4_0 GGUF decode was silently wrong.** The Q4_0 dp4a GEMV read packed nibbles
  INTERLEAVED while ggml Q4_0 is SPLIT, and trusted a mis-scaled zero-point, so
  every Q4_0 dense and MoE decode produced garbage. It was never caught because no
  Q4_0 model is in the test suite; a new fp64 oracle surfaced it.
- **Disabling thinking now actually suppresses reasoning**, from two root causes.
  Tokenizer: Qwen3 ships `<think>`/`</think>` as added tokens with
  `special=false, normalized=false`, and imp only atomic-matched `special` tokens,
  so they were BPE-split and the template's closed no-think block was just text the
  model re-opened. imp now follows HF semantics for any `normalized=false` added
  token. Server: the heuristic that re-enables thinking on a `<think>` in the
  prompt tail fired on the *closed* block too, and now requires an unclosed prefix.
- **Over-long prompts are rejected with a 400 instead of crashing the server.** The
  gate used the model's *declared* max context while the engine VRAM-auto-sizes the
  actual allocated context, so a prompt between the two overran the per-sequence KV
  buffers. New `imp_context_max_seq_len()` is the gate.
- **Embeddings reject inputs longer than the single-pass hidden buffer** (was a
  server abort): `/v1/embeddings` mean-pools every token's hidden state, which only
  fits when the whole input is prefilled in one pass.
- **`scripts/bench_gate.sh` was silently broken**: a stray `2>&1` left the parsed
  stderr empty, so `set -e` exited before the gate could report, and the gate had
  in fact never run.
- **`[runtime] max_batch_size` from imp.conf is honored for engine sizing.** The
  server built `ImpConfig.max_batch_size` from the CLI flag only, so the config
  value reached the engine as the decode-batch cap and never as scheduler/KV
  sizing. The default changed 4 to 0 to match the documented "0 = auto".

## [0.11.2] - 2026-06-14

### Added

- `tests/test_server_robustness.py`: a server-level battery asserting that
  malformed JSON, invalid UTF-8, non-object, wrong-type and missing-field input on
  every endpoint returns 4xx with an error envelope, never 5xx.

### Fixed

- **Bad request input returns 400 with a JSON error envelope, never a bare 500.**
  Invalid UTF-8 in a body made `json::parse` throw, the error envelope echoed the
  offending bytes, and `err.dump()` then threw because dump rejects ill-formed
  UTF-8, which escaped the parse-error catch and surfaced as an opaque 500 with no
  body. Added a global exception handler plus a `dump_safe()` used on every
  response, SSE, error and request-log body.

## [0.11.1] - 2026-06-14

### Added

- `tests/test_server_0token_battery.py` and
  `tests/test_server_embed_chat_interleave.sh`: server-level regression coverage
  for the wedge below, gating on the empty-completion rate.

### Fixed

- **Embeddings no longer cancels in-flight generations.** The `/v1/embeddings`
  handler took exclusive C-API access via `BatchingEngine::stop()`, which cancels
  every in-flight request, so under interleaved embed and chat load any running
  generation came back empty; `stop()` also left the cancelled sequences' KV blocks
  allocated. Replaced by a graceful `pause()`/`resume()` handshake that lets the
  worker finish in-flight requests before parking. (#710)

## [0.11.0] - 2026-06-13

47 commits since v0.10.0. The entire measured cross-engine perplexity gap turned
out to be tokenization, not numerics; four families are now byte-identical to
llama.cpp/HF. Plus NVFP4 long-context prefill at-or-ahead of vLLM, FP8-KV honoring
the model hint, opt-in n-gram speculation, and full gpt-oss-20b GGUF MoE support.

### Added

- **N-gram prompt-lookup speculative decoding** (opt-in): draft tokens are matched
  from the prompt/context suffix and verified in a burst-hybrid loop, output stays
  token-identical to plain greedy. CLI ~+6 % on long generations; opt-in because
  draft-poor workloads regress. (#668-#670)
- **gpt-oss-20b GGUF MoE**: full GGUF path with MXFP4 to NVFP4 expert conversion,
  expert biases, attention sinks, sliding-window attention and residual rescale.
  The GGUF checkpoint previously NaN'd in MoE prefill. (#690)
- **`IMP_PPL_DUMP=full`** dumps per-position NLL for cross-engine perplexity
  forensics. (#655)

### Changed

- **`kv_cache.dtype` defaults to `auto`, honoring the model author's
  `kv_cache_quant_algo=FP8` hint** for arch families that pass a long-context
  quality gate. Allowlisted today: Qwen3 dense and Qwen3 MoE, measured at
  **+1.07 % PPL on Qwen3-14B**, neutral on 30B-A3B, ~768 MiB KV VRAM saved. Other
  hint-declaring families stay FP16 until verified.

### Performance

- **Prefill chunk size default 512 to 2048**: MoE pp2048 **+127 %**
  (Qwen3-30B-A3B 15.7k to 35.7k tok/s), pp4096 +77 %. Also fixed a grouped
  device-args GEMM silent corruption at n≈900. (#672)
- **FP16-QK FA2 is the primary hd=128 prefill**, at or above cuBLAS at every pp
  (pp1024 +24 %, pp2048 +52 %), so the S-matrix buffer is skipped for hd=128:
  **-380 MiB** device memory. MoE pp4096 now 4 % ahead of vLLM. (#687)
- **FA2 full-rate accumulate default-on**: -18 % pp4096 kernel time, MoE e2e
  +9.7 %, PPL unchanged. (#673, #674)
- **Conditional-graph decode loop for NVFP4 think models**: **+45 %** think-decode
  by keeping the reasoning loop inside one captured graph. (#649)
- **Pipelined constrained decoding**: `json_schema` decode **102 to 235 tok/s**.
  (#650, #651)
- **FP8-KV deterministic-cuBLAS forcing scoped to non-FA2 configs**, removing a
  -35 % pp4096 MoE tax on `--kv-fp8`. (#682)
- **VRAM reclamation on NVFP4**: fallback-only workspaces skipped on SafeTensors
  (+827 MiB), duplicated per-expert micro-scales freed (-1728 MiB on 30B-A3B),
  CUTLASS scale-factor dedup (-1810 MiB on NVFP4-prequant). (#678, #679, #685,
  #686, #689)

### Fixed

- **Qwen2/Qwen3 tokenization was non-canonical on symbol/digit sequences.** The
  gpt2 pre-tokenizer fallback split every punctuation character individually and
  grouped digits in threes, so canonical BPE merges were impossible: on a 10 KB
  code corpus imp produced 3690 tokens against llama.cpp's 3084 (+20 %), inflating
  matched-band NLL by +70 %. With the faithful `qwen2_pre_tokenize`, token streams
  are id-identical to llama.cpp and the NLL gap is **+1.3 %** (Qwen3-8B-Q8_0 corpus
  PPL 40.5 to 10.98). (#657)
- **SPM tokenization: USER_DEFINED pieces are literal-matched, so gemma
  multi-space runs were never canonical.** gemma-3 stores indentation tokens as
  user-defined symbols with literal-space pieces, which imp's ▁-substituting BPE
  could never reproduce. gemma-3-12b: token ids identical to llama.cpp, matched-band
  NLL **+37.5 % to -0.4 %**, corpus PPL 15.53 to 10.57. (#657)
- **fp8-QK FMHA demoted to opt-in: gemma-3 long-context prefill was catastrophically
  degraded.** The raw unscaled Q/K to e4m3 conversion compounds per-layer score
  error on real activations: teacher-forced PPL gemma-3-12b **16.6 to 549** once
  chunked prefill crossed the S-matrix cap and the fp8 kernel started serving. The
  original long-context validation never exercised the kernel. Now `"never"` by
  default; gemma-3-12b at 8.3k tokens goes 549 to **11.1**. (#511)
- **Prefill dispatch chain exhaustion throws instead of silently emitting garbage**:
  `flash_attention_blackwell` declined hd=256 by falling back to a kernel whose
  launch also fails at hd=256, unchecked, leaving the output buffer as garbage
  (teacher-forced PPL ~1e10 when forced). (#654)
- **The speculative conditional-graph loop wrote KV one slot too high**:
  `setup()` double-incremented the first-forward position, so every fresh-captured
  verify loop duplicated the last burst KV entry. Byte-perfect across mb=8/4/0 and
  Q8 / NVFP4-dense / NVFP4-MoE after the fix. (#683, #692)
- **`attn_logit_softcap` was silently dropped on the cuBLAS FP32-S prefill path**,
  so Gemma-2-class hd=256 prefill skipped the cap. (#688)
- **`gemm_kv_batched` output stride** is derived from the actual K/V pointer
  distance, fixing a Q4_0 determinism mismatch plus a cross-block WAR hazard in the
  fused FP32 to FP16 softmax downcast. (#677, #691)
- **Constrained decoding SIGBUS** on SafeTensors `json_mode` with raw control
  characters in constrained strings. (#650)

## [0.10.0] - 2026-06-09

151 commits since v0.9.1: gpt-oss-20b support, LoRA hot-swap, the INT8-IMMA
prefill family that took GGUF prefill from "always behind" to ahead of llama.cpp
on the MoE/Q6_K heroes, a GeForce-Blackwell tensor-core-rate recalibration, and a
roofline audit that mapped the remaining decode/prefill ceilings.

### Added

- **gpt-oss-20b**: MXFP4 experts converted to NVFP4 at load, attention sinks,
  Harmony channel split, YaRN/split-K/FP16-range fixes. CUTLASS grouped-GEMM
  prefill registration took pp512 from ~1.9k to **16-19k tok/s** (~10x); decode
  310-345 tok/s. (#547, #572, #574)
- **LoRA / PEFT adapter hot-swap**: runtime low-rank deltas, no weight patching.
  (#522, #571)
- **IQ4_NL / IQ4_XS i-quant GGUF support.** (#556, #561)
- **Gemma-4 vision** (gemma4v) plus Gemma-3 mmproj projector load. (#490, #489)
- **Teacher-forced perplexity tool** `imp-cli --perplexity`, chunk-aware since
  #553. (#481)
- **Anthropic `cache_control` prompt caching**: prefix-cache pinning and usage
  accounting; `prefix_cache` default on. (#522, #541)
- **`attention.fa2_f16acc`** opt-in: +3-4 % pp2048/pp4096 NVFP4 prefill for
  +0.37 % PPL. (#597)

### Performance

- **INT8-IMMA prefill GEMM family** (default on since #617): fused dequant on INT8
  tensor cores for Q8_0/Q4_K/Q6_K/Q5_1 including MoE grouped variants.
  Qwen3-30B-A3B and Qwen3-14B-Q6_K prefill now **ahead of llama.cpp**;
  gemma-4-26B MoE +111 % cumulative; Q8 dense 1.13x. (#612-#619)
- **GeForce tensor-core-rate calibration**: sm_120 silicon runs FP4 block-scale at
  ½ datasheet and FP16/FP8 f32-accumulate at ¼ rate, so the roofline peaks were
  corrected. Unlocked `gemm.cublas_fp16_acc` and CUTLASS small-N pingpong (kv_proj
  2.1x). (#606, #611)
- **FA2 prefill**: ldmatrix operand fetch, register-resident Q and an 8-warp fp16qk
  variant (-28 % late-chunk kernel); FP16-QK short-prefill path (+25-35 % pp512).
  (#609, #525, #493)
- **dp4a GEMV**: 16-B-aligned `block_q8_1` removes the activation-load ceiling;
  LDG.128 Q4_K/Q5_K weight loads. (#619, #607)
- **NVFP4 decode**: NVFP4 lm_head default-on for GDN/hybrid models (+11.4 % on
  Qwen3.6-35B); opt-in NVFP4 for recipe-excluded hybrid projections. (#483, #486)
- **Warp-per-row FP16 RMSNorm for batch prefill.** (#620)

### Fixed

- **SafeTensors Llama-family RoPE**: NeoX layout for LLAMA/MISTRAL/MIXTRAL/LLAMA4
  (GGUF pre-permutes Q/K, HF does not), which fixed Phi-4 prompt-blind output.
  (#503)
- **Gemma-3 garbage output**: `apply_arch_defaults` double-counted the
  `norm_weight_offset`, since llama.cpp already bakes the `+1` into
  `*norm.weight`.
- **Nemotron-H NoPE attention** (#518); **MXFP4 GGML type-39 nibble order is
  split, not linear** (#567); **sliding-window prefill** routed through the cuBLAS
  masked softmax as the correctness reference for hd=256 with a window (#566,
  #569).
- **Pinned-staging reuse race** corrupted chunked continuations, the root cause of
  the "fa2_fp16qk Llama bug"; **in-place float to half S/P-tile compaction race** in
  the WMMA prefill kernels. (#548, #568, #528, #539)
- **Recurrent-state slot leak/concurrency for SSM/GDN** (#500, #501);
  **model-reload SIGSEGV and VRAM retention** with strict OpenAI model semantics
  (#507).
- **Determinism**: `[runtime] deterministic` works via the C API with bit-stable
  perplexity, proven on a GDN hybrid; prefix-cache stale-table hit fixed via
  content-compare. (#542, #538)
- **Constrained decoding**: per-token FSM simulation for schema JSON, `$ref`/
  `$defs` including recursive schemas, regex enforcement, whole-token validation;
  **thinking** default now requires template evidence, not just a vocab `<think>`
  token. (#497-#499, #517, #562, #513, #563)
- **Gemma-3 chunked prefill enabled** (byte-identical greedy vs single-shot across
  SWA boundaries).

### Changed

- **`ModelProfile`**: one source of truth for architecture classification, with all
  hot-path `cfg.arch == X` checks routed through it. (#622, #623, #625)
- **VRAM cache rebuild**: RAII ownership of all 8 caches, so a double-free is now a
  compile error; one authoritative storage tier. (#621)
- **Roofline audit** (`docs/archive/roofline_2026_06_07.md` plus the
  `tools/roofline/` ncu+nsys pipeline): shipped the `attn_fa2` f16-acc lever and
  documented the structural ceilings of MoE-decode `gemv_nvfp4`, MoE-prefill
  `gemm_grouped_nvfp4` and hd=256 prefill coverage. (#597, #600, #601, #603)
- **CUDA 13.3 native images**; dependency pins CUTLASS v4.5.1, GTest v1.17.0,
  nlohmann/json v3.12.0, httplib v0.46.1. (#520)
- **Server-level degeneration suite** `tools/analysis/degen_suite.py`. (#508)

## [0.9.1] - 2026-05-27

### Fixed

- **FP8 prefill degeneration on sm_120**: cuBLAS 13.4 `cublasLtMatmul` returns
  `CUBLAS_STATUS_NOT_SUPPORTED` for FP8 E4M3 GEMMs at non-aligned M on consumer
  Blackwell, and the `cublasGemmEx` fallback silently produced garbage (no
  per-tensor scales), corrupting the KV cache and degenerating decode on **all
  GGUF models**. FP8 prefill is auto-disabled on sm_120; cuBLAS algo benchmarking
  now validates return status during warmup. (#446)
- **Server hallucination at turn boundaries**: thinking models at high temperature
  could hallucinate `Human`/`<think>` turn markers, leaking internal reasoning.
  Fixed with stop-sequence detection. (#442)
- **CUDA graph crash on Nemotron-H**: Mamba2 SSM layers auto-detected and excluded
  from graph capture. (#443)
- **Gemma-4 dense (31B) weight mapping**: `mlp.{gate,up,down}_proj` was
  unconditionally routed to shared expert slots, breaking dense Gemma-4 models.
  (#444)
- FP16 cache VRAM overcommit on dense Q4_K_M (#435); test segfaults on the weight
  registry (#437); CUDA teardown errors in the `Model` destructor (#439); a Q5_K
  forward-pass NaN from cross-test cuBLAS state contamination (#445).

### Added

- **Phi-4-reasoning-plus NVFP4**: fused `qkv_proj`/`gate_up_proj` support. (#429)
- **Nemotron-Labs-3-Elastic-30B-A3B NVFP4**, a newer QAD quant, ~70 tok/s decode.
- **Gemma-4 NVFP4 decode cache for Q*_K source weights**, dropping the "per-layer
  head_dim not yet supported" carve-out: Gemma-4-26B-A4B-it-Q4_K_M pp512 1713 to
  **2394 tok/s** (+40 %), tg256 176 to 197 (+12 %).
- **Chunked prefill for INT4 KV** via a new `paged_kv_gather_int4_to_fp16` kernel
  mirroring the FP16/FP8/NVFP4 gather variants.

### Performance

- **dp4a dense prefill for Q4_K/Q5_K**: computes directly from quantized blocks
  (0.55 B/elem) instead of the FP16 weight cache (2.0 B/elem) at small M. (#436)
- **Q4_K_M GGUF support**: dequant fallback, FP8 D2H fix and fused MoE dp4a.
  Qwen3-30B Q4_K_M pp512=3616, tg256=271. (#431, #432, #414)
- **CUTLASS NVFP4 dispatch fix**: zero-copy MoE expert registration eliminated a
  15 GiB D2H copy on Qwen3.6-35B. (#428)
- **Gemma-4 FP8 prefill carve-out removed**: re-measured on Q4_K_M as neutral with
  a long-context advantage (pp2048 **+7.3 %**), and FP8 also halves the activation
  cache.

### Changed

- CUTLASS v4.5.0 to v4.5.1 (#447); cpp-httplib v0.45.0 to v0.46.0 (#448).

## [0.9.0] - 2026-05-10

Sixty-plus PRs since v0.8.0. NVFP4 prefill goes from 1.2k to 13k tok/s on
Qwen3-Coder-30B-A3B-NVFP4, the NVFP4 KV cache lands (16k to 40k tokens at the
same VRAM), and chunked-prefill correctness closes the long-context cliff. The
build target moves to `sm_120a`.

### Added

- **NVFP4 KV cache** (opt-in `--kv-nvfp4`): 4 bits per element plus a per-block
  scale takes 16k to **40k tokens at the same VRAM**, 3.9x compression against
  FP16. A vectorized PTX `cvt.rn.f16x2.e2m1x2` decode path closed the dequant gap
  (+25.6 %, parity with the FP16 baseline). (#108, #125)
- **NemotronH hybrid Mamba2 + MoE + Attention NVFP4** loads end to end, and a
  KV-cache-sizing fix clears the multi-chunk hang on long context: tg128 **42 to
  319 tok/s** after dynamic NVFP4 MoE reserve sizing. (#104, #109)
- **BitDecoding TC paged decode (phases 0-3)**: WMMA Q.K dot dispatch,
  block-softmax, FP16 residual buffer, multi-seq, splitk and graph-safe. Reaches
  parity with the FP16 baseline (193 vs 193 tok/s on Qwen3-4B Q8 NVFP4-KV, from
  50). Opt-in, because NVFP4-MoE and dual-head_dim regressions do not justify a
  flip yet. (#142-#149)
- **Multimodal Qwen3.6-VL NVFP4 loader**: all HF Qwen3.6-NVFP4 repos ship a VL/Omni
  base, and the loader strips the multimodal prefix on text-only weights. (#152)
- **Native SentencePiece (`.model`) parser**, dropping the Python fallback for
  Mistral-family tokenizers. (#128)
- **Zero-config SafeTensors auto-detect**, so no `--arch` is needed for supported
  repos. (#116)
- **Server**: tools and JSON-schema coordination (preamble pass-through for
  reasoning models, schema preamble close), plus opt-in `--log-requests` JSONL.
  (#103, #112, #119, #155)
- **CUDA 13.2 modernization**: `cudaMemcpyWithAttributesAsync` replaces the
  stream-attribute dance for L2 streaming hints, plus the `add.f32x2` intrinsic.
  (#131)
- **GHCR release pipeline** publishing the Docker image on tagged release (#101),
  CI ccache with path-aware keys (#122, #127), and auto-merge on owner PRs (#123).
- New unit tests (`test_kv_gather`, `test_attention_chunked`,
  `test_chunked_prefill`), `tests/perf_baseline_chunked.json` and
  `make verify-chunked`.

### Changed

- **Build target is `sm_120a`** (was `sm_120f`), unlocking the full RTX 5090
  feature set. The historical `ptxas` C7600 workaround is obsolete on CUDA
  13.2.1+. (#105)
- **`prefill_chunk_size` default sentinel `-1`** = per-arch default. Single-chunk
  is the default for SWA / Gemma-4 long-context recall; multi-chunk is gated on
  attention-shape uniformity for hybrids. (#130, #114, #117)
- **Auto `max_seq_len` for hybrid models** corrected, soft-cap default lifted to
  16K. (#157)
- **Public-release readiness pass**: documentation rewrite, the hygiene gate
  `scripts/check-release.sh`, and removal of dev-internal scratch files.

### Performance

- **NVFP4 MoE prefill fast-path**: Qwen3-Coder-30B-A3B-NVFP4 pp512 **1241 to 13046
  tok/s** (10.5x). Direct-from-NVFP4 grouped GEMM with cached problem shapes;
  previously it fell through to dequant plus cuBLAS per chunk. Qwen3.6-NVFP4,
  Gemma-4-NVFP4 and Qwen3-30B-A3B-Modelopt prefill all double or better. (#160)
- **The 1024 to 4096 prefill cliff is closed**: the n≤1024 cap is removed and the
  S-matrix buffer grew 256 to 1024 MiB. Qwen3-4B Q8_0 pp4096 +28 %, Qwen3-8B +18 %.
  (#110)
- **GDN α+β fused GEMV decode** and 4-way input fusion (`ssm_in` + `gdn_gate` + α +
  β) speed up Qwen3.5/3.6 GDN decode end to end. (#153, #154)
- **NVFP4 MoE GrpGemm cache**: +4-7 % on Qwen3-Coder-30B-A3B-NVFP4 prefill. The
  gate+up fusion alongside it is opt-in and does not yet beat baseline. (#161)
- **Default mem-pool retain and `cudaGraphExecUpdate` re-capture** make chunked
  prefill on the NVFP4 KV cache graph-safe. (#149)

### Fixed

- **Chunked prefill correctness**: chunks ≥2 now read past chunks' K/V from the
  paged cache via new `paged_kv_gather_*` kernels and a rectangular
  `attention_cublas_prefill(q_offset)`. Previously `prefill_chunk_size > 0`
  produced silently wrong logits for full-attention models and full degeneration
  for SWA models like Gemma-4. (#130)
- **Chunked prefill on hybrid GDN+MoE / Mamba2+MoE archs**: prompts where
  `total_input > effective_chunk` were rejected outright. The Mamba2 conv1d kernel
  now reads trailing context from `conv_state` at the chunk boundary, and the
  carve-out is gated by attention-shape uniformity. (#156)
- **Chunked prefill for Gemma-4** (SWA plus dual head_dim 256/512): the cuBLAS
  softmax kernels take a `sliding_window` parameter, so SWA layers route through
  cuBLAS instead of the naive FP32 workaround. Validated at 1508 tok/s with decode
  bit-exact to single-chunk.
- **Chunked prefill for INT4 KV** via `paged_kv_gather_int4_to_fp16`. INT4's
  pre-existing long-context quality regression is independent of chunked prefill.
- **`llm-compressor` zero / non-finite `tensor_scale` and `input_scale`**: a
  defensive guard prevents NVFP4 zero-norm collapse on those exports. (#113)
- **Graph-safe `gemm_nvfp4` dequant fallback** via
  `set_nvfp4_dequant_workspace()` and a capture guard. (#121)
- **`d_pf_block_tables_` undersize**, sized from `max_blocks` rather than
  `blocks_per_seq` (#134); **MoE NVFP4 `expert_gemm` uses the cached buffer** on
  the non-gated arch path (#115); **NVFP4 chunked prefill `attn_scores_` capacity**
  sized correctly (#149).
- **Gemma-4 FP8-prefill carve-out reason corrected**: perf, not correctness. The
  cuBLASLt FP8 algo is slower at Gemma-4 shapes; output is bit-identical. (#137)
- **SafeTensors + NVFP4 audit F1-F8**: model header guard, missing-tensor
  messaging, NVFP4 `tensor_scale` finite-check, `input_scale` visibility,
  multimodal prefix, `arch_norm_offset`, RMSNorm `1+W`, and the
  `A_log → -exp(A_log)` SafeTensors path. (#126)

### Known issues (carry-over)

- NVFP4 MoE prefill ceiling at ~16k tok/s warm against vLLM single-seq 18.5k, a
  1.42x gap (#161). Spec-decode / MTP still off on NVFP4 decode-cache models.
  CUTLASS NVFP4 sm_120 non-determinism under graph replay (user-facing output is
  fine; only the determinism test trips). Prefill throughput varies up to 2.6x
  between container restarts due to cuBLAS autotuning, so compare decode-only for
  reliable A/B.

## [0.8.0] - 2026-05-03

Forty-plus PRs since v0.7.0. NVFP4-prequant SafeTensors hits production:
Mistral-3.2, Gemma-4, Qwen3.6 and Qwen3-Coder are all coherent on single-turn,
sampling, multi-turn and short long-context. CUDA Graphs lit up for prequant
SafeTensors.

### Added

- **Native function calling for Gemma-4 and Qwen3.6.** The root cause was a
  tokenizer bug, not missing parsers: multi-character markers like `<|tool_call>`
  were BPE'd as raw UTF-8 bytes, so the model never saw the trained marker and
  answered with markdown JSON code blocks. The encoders now run a longest-match
  pre-split against CONTROL-flagged added tokens before BPE, and
  `parse_tool_calls_gemma()` covers Gemma's non-JSON syntax. (#97)
- **NVFP4 SafeTensors loader from llm-compressor** (phases 1 and 2), plus
  Qwen3.6-NVFP4 plumbing. (#63, #64, #65, #71)
- **Anthropic `/v1/messages` endpoint**, non-streaming then streaming. (#35, #36)
- **`imp.conf` is the configuration interface**: ~50 `IMP_*` env vars retired for
  sectioned TOML keys, with `--set section.key=value` for per-run overrides.
  Loading precedence: `--config`, `$IMP_CONFIG`, `./imp.conf`,
  `~/.config/imp/imp.conf`, embedded defaults. (#72)
- **KV-cache safety default flip**: the default KV dtype is FP16 and FP8 is opt-in,
  which fixes Mistral, DeepSeek and Qwen3.5-GDN out of the box on first decode.
  (#51)
- **CUDA Graph coverage expansion**: speculative-verify graphs, the SigLIP vision
  graph, default mem-pool retain and `cudaGraphExecUpdate` re-capture. (#53)
- **SM120 FMHA optimisation pass**: float4 tile loads plus hardware FP4 conversion,
  **+11-13 % prefill** on Qwen3-4B Q8_0 at pp=8192. (#55, #56)
- **Faster cold start, 24 s to 18 s on Qwen3.6-NVFP4**: skip MTP/vision-only shards
  when neither is wired up, `MAP_POPULATE` + `MADV_WILLNEED` on weight mmaps, a
  larger pinned staging ring, and concurrent SafeTensors shard parse. (#97)
- **JSON config plumbing** (`generation_config.json` sampling defaults,
  `special_tokens_map.json`, Mistral V3 tokenizer-config flags) and a type-system
  refactor unifying `QType` and `Tensor` sidecars. (#74, #77, #72)
- **Split `imp-tests` into 8 per-module binaries** to speed up filtered runs
  (#57), plus `tools/analysis/` PTX survey scripts (#67).

### Fixed

- **FP8 KV warmup-calibration bug**: `Engine::warmup()` ran a forward pass with
  synthetic BOS tokens, the online calibration treated it as the first prefill and
  locked `kv_scales_` to a too-small absmax, and never recalibrated. Real
  generation then overflowed FP8 dynamic range and degenerated within ~30 tokens.
  Warmup now drops the calibration flags and the write path promotes the scale
  monotonically. (#89)
- **NVFP4 prequant CUTLASS prefill cache**: Phase 0 set `qtype = NVFP4` on the main
  weight tensors but the cache build only iterated the legacy map, so prequant
  prefill fell through to dequant plus cuBLAS, allocating ~40 MiB of FP16 scratch
  per layer per prefill. Post-fix decode: Mistral-3.2-NVFP4 81 to 101,
  Qwen3.6-NVFP4 117-142 to 217, **Qwen3-Coder-30B-A3B-NVFP4 51 to 272 tok/s**, and
  `--no-cuda-graphs` is no longer needed. (#88)
- **NVFP4 prequant MoE decode fast-path**: Qwen3.6-NVFP4 went **8.34 to 117-142
  tok/s**, Gemma-4-NVFP4 ~42 to 157-180. Three bugs: the `can_decode_fast`
  whitelist excluded NVFP4-prequant models, the contiguous per-expert NVFP4 buffer
  was never built for SafeTensors per-expert layouts, and per-expert allocations
  were freed per layer. (#85)
- **Six Qwen3.5/3.6-NVFP4 SafeTensors loader bugs** blocking coherent decode: the
  RMSNorm `1+W` convention, HF-grouped vs GGUF-tiled GDN head layout,
  `partial_rotary_factor` and `rope_theta` from nested `rope_parameters`, the
  `A_log → -exp(A_log)` transform, and an `fp32_scan` buffer populated outside
  `debug_forward`. Per-layer correlation against GGUF Q4_K_M is now ≥0.997 across
  all 40 layers. (#81)
- **Qwen3.5 GDN Q8_0 α/β qtype mismatch**: `upload_weight` pre-dequanted Q8 to FP16
  without updating `qtype`, so the dispatcher mis-interpreted the bytes and the
  state collapsed. (#59)
- **MoE expert-offload auto-pick** tries 10 % overhead before falling back to 30 %:
  Qwen3-Coder-30B Q6_K **77 to 234 tok/s**. (#54)
- **Server fixes for Open WebUI on Qwen3.6-NVFP4**: a UTF-8 boundary walk in the
  reasoning stream (German umlauts came out as `f??r`), leaked stop tokens dropped
  before the `is_last` gate, and `repetition_penalty` default 1.0 to 1.05 to break
  multi-turn loop degeneration. (#97)
- **MXFP4 GDN-fallback dequant** replaced a buggy CPU path with a GPU kernel (#58);
  **MXFP4 FP16-fallback VRAM oversubscription** now gives a clear error instead of
  failing silently (#60); **Mistral-3.2-NVFP4 `use_default_system_prompt`** is
  honoured, skipping a 600-token default system prompt (#78).

### Known issues (carry-over)

- FP8 KV still breaks Llama-3.2 / Mistral-Small-3.1 / DeepSeek-R1-Distill out of
  the box; the default is FP16 and opt-in is per model after testing. Residual
  NVFP4 long-context model-behaviour issue on long English prose. Prefill
  throughput varies up to 2.6x between container restarts.

## [0.7.0] - 2026-04-23

Correctness and platform release: the long-context dispatch cliff is gone, Gemma-4
and the Qwen3.5/3.6 GDN family produce clean output on Blackwell, and CUDA 13.2.1
with stream priorities and mem-sync domains is live.

### Added

- **StreamingLLM smart KV cache**: attention sinks plus a sliding window, which
  keeps long-conversation coherence without unbounded VRAM growth. (#26)
- **Weight-storage refactor**: `TensorKind` + `StoragePlanner` + `gemm_dispatch`,
  collapsing a 21-parameter dispatch to 5. No functional change. (#27)
- **CUTLASS 3.x NVFP4 grouped-GEMM scaffold** with fused MoE quantize, default ON
  for all batch sizes after the gate+up shared-quantize optimisation (decode 51 vs
  legacy 37 on Qwen3-Coder-30B-A3B NVFP4). (#22)
- **CUDA 13.2.1 base images**, stream priorities, mem-sync domains and cluster
  spread. (#16, #17)
- **Qwen3.6 `ModelArch::QWEN36_MOE` scaffold** (GDN + MoE hybrid) and GDN reference
  infrastructure with multi-turn state preservation. (#23, #25)
- **`tools/analysis/layer_diff.py`**, an .npy-based per-layer tensor diff between
  imp and llama.cpp for drift analysis (#20), plus CUDA graph diagnostics
  (#11-#14).

### Fixed

- **FP8 FMHA long-context cliff at n>1024**: `fmha_sm120_fp8_kernel` placed
  `S_tile` too close to `KV_fp8`, so V row `Bkv/2+` overwrote the P values the PV
  MMA was about to read, giving NaN on every attention layer above n=1024. It was
  invisible to prior benchmarks because pp=512/1024 always stayed on cuBLAS and
  decode uses paged attention. All tested models are now coherent at n≥1025. (#33)
- **Qwen3.5/3.6 GDN fused-kernel launch_bounds**: `__launch_bounds__(HD, 2)`
  miscompiled at HD=128, and dropping to `(HD, 1)` fixes Qwen3.5-4B/9B Q8_0
  coherence and takes Qwen3.6 tg256 from **36 to 57 tok/s**. The partial-RoPE pair
  offset was wrong in the same place; both fixes are needed. (#30)
- **Qwen3.6 h_state FP32 preservation**: the engine auto-downgraded
  `ssm_state_dtype` to FP16 for all SSM models while the GDN scan writes FP32, so
  each layer's scan overflowed 1 MB into the next layer's state region and produced
  NaN at L38. (#28)
- **Gemma-4 Q4_K_M CUDA-graph decode**: the split-K pipeline kernel issued one 16 B
  `cp.async` per load, missing half the data for HEAD_DIM=512 on global layers.
  tg256 Q4_K_M **55 to 183 tok/s**.
- **Gemma-4 SWA long-context degeneration** above 1024 tokens on global layers
  (#21), **per-layer `rope_freqs` ignored on global layers**, which cut L13/L14
  drift from 11-15 % to under 2 % (#20), **host-resident MoE silent corruption**
  when experts are CPU-offloaded, and an **FP32 router plus half rope_dim on global
  layers** that caused expert mis-pick at L29.
- **Qwen3.5 GDN L2-window CUDA errors**: an access-policy window larger than
  `cudaDevAttrMaxAccessPolicyWindowSize` (128 MiB on the RTX 5090) silently
  poisoned the stream. Now clamped.
- **Gemma-4's ≥3120-token limit was VRAM, not architecture**: the default ceiling
  lifted to ~7881 tokens, and `--min-kv-tokens 14000` reaches 11242.

### Changed

- Gemma-4 CUDA graphs enabled by default for the decode fast-path. Benchmark docs
  refreshed with the quality caveat for Q4_K_M on complex code-gen prompts.
  cpp-httplib 0.40.0 to 0.42.0.

### Known issues

- Qwen3-Coder-30B-A3B NVFP4 still needs `--no-cuda-graphs` for coherence on the MoE
  routing path. Prefill throughput varies up to 2.6x between container restarts.
  The FP8 FMHA path is ~30 % slower than cuBLAS at the dispatch boundary on small
  dense models; output is correct. MXFP4 GGUFs use the imp-proprietary tensor-type
  31, which llama.cpp reads as the removed `Q4_0_4_4`, so cross-tool perplexity
  comparison needs a standard-format MXFP4 export.

## [0.6] - 2026-04

- NVIDIA Model Optimizer NVFP4 prequant SafeTensors loading (Qwen3-Coder-30B-A3B-FP4
  verified), imp-server SafeTensors support with `resolve_model_auto()` format
  detection, the HuggingFace chat-template array format, and Jinja2 macro support,
  which fixed the Qwen3.5 "ignores prompt" symptom.

## [0.5.1], [0.4.1], [0.4], [0.2]

Pre-0.6 tags retained for reference. See `git log` for details.
