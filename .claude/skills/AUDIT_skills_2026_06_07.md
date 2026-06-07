# Skill Audit — 2026-06-07

Audit, fix, and extension of the agent skills in `.claude/skills/`. Goal: an agent
working on imp triggers the right skill for each recurring workflow and the skill
content matches the current repo state (no stale facts, no dead references).

## Phase 1 — Audit of existing skills

| Skill | Purpose | Trigger quality | Problems found | Action |
|---|---|---|---|---|
| benchmark-cuda | Kernel/engine benchmarking, ncu/nsys, roofline | Good keywords; no negative cases | ① "thermal throttling skews A/B" red flag — this GPU is water-cooled and never throttles; the real artifacts (idle-downclock ~1 s ramp reading LOW, cross-day host drift 8–15% #526, back-to-back sweeps 6–10% low) were absent. ② Warmup "≥3 iterations" doesn't cover the 1 s clock ramp. ③ Bench methodology (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, 10 reps, 3+ trials, one model/process) missing. ④ No Docker/WSL2 context although the host has no CUDA toolkit; nsys WSL2 flags and ncu mount recipe missing. ⑤ Claimed baseline metric `tg256`; `tests/perf_baseline.json` gates on `tg128`. ⑥ No publish-numbers guidance (BENCHMARKS.md/README/baseline-refresh rules). | **fix** (done) |
| check-degeneration | Output-coherence battery after hot-path changes | Very good (symptoms + "after enabling X") | Current (2026-06-04); all paths/test names/flags verified correct. Missing: re-downloaded quant caveat (06-06 Qwen3-4B/Llama-3.2-3B are different files; greedy tie-prompts diverge → NLL not byte-equality), and the docker-run pattern for the server behind `degen_suite.py`. | **fix** (minor, done) |
| sm120-cuda-expert | Writing/optimizing sm_120a kernels | Good | ① "FP8 prefill cache +40–60%" listed as a working technique — FP8 prefill is default-DISABLED on sm_120 (`engine_init_resolver.cpp:156`, cuBLAS `NOT_SUPPORTED` at non-aligned M). ② `attention_mxf4nvf4_probe.cu` referenced under `src/compute/` — actual location `tests/bench/`. ③ known-issues referenced `executor_moe.cu` — doesn't exist (`executor_forward_moe*.cu`/`expert_cache.cu`). ④ "AsyncGraphLoop … max_steps=255" — class is `CudaGraphConditionalRunner`, max_steps is sized per request. ⑤ Dead-end "CUTLASS TC GEMM at M=1, retry on 4.5+" — pin is v4.5.1 since PR #546; retry condition met, not re-probed. ⑥ Missing shipped levers: FA2 stack (default-on, fp16qk, bank-conflict padding, cp.async dbuf), NVFP4 lm_head, `nvfp4_ssm_proj`; missing 2026-05-30 roofline finding (occupancy raise = refuted lever on decode GEMV) nuancing "Law 2: Occupancy is king". | **fix** (done) |

No merge/split/deprecate cases — the three skills are correctly delineated (measure / write / validate-output) and cross-reference each other.

Side finding (docs, not skills): `GOAL.md:29` still lists H100/H200 (sm_90a) as maintained target hardware and `GOAL.md:76` a "CUTLASS Hopper FMHA path", which contradicts CLAUDE.md/README ("sm_120a only, no Hopper"). Not touched here — flagging for a docs pass.

## Phase 2 — Gap analysis

| Missing skill | Justification | Priority | Outcome |
|---|---|---|---|
| building-and-testing | Every session builds/tests; high-cost tribal knowledge: no host toolkit, root-owned `build/`, no `--mount=type=cache`, models symlinks, dep pins in two places, CI has no GPU runner (local-only GPU validation), ruleset check named `Build`, determinism/PPL caveats | P1 | **created** |
| server-api | ~15% of commits are server work; endpoints (OpenAI+Anthropic), real SSE streaming (older audit docs say "synthetic" — obsolete since `main.cpp:192`), json_schema/`sim_advance`, cache_control prefix pinning, strict #507 model semantics, validation tooling | P1 | **created** |
| add-model-arch | Recurring (gpt-oss #572, Nemotron, Gemma-4, vision): integration checklist + wrong-output diagnostic fingerprints (RoPE NeoX, NoPE, YaRN inversion, swa_layers, h_state FP32) | P2 | **created** |
| quant-formats | Reference for loaders/dequant work: GGUF vs NVFP4 worlds, StorageTier dispatch contract, decode cache, KV dtypes; thin pointer to `docs/quantization.md`/`quant-pipeline.md` | P3 | **created** |
| release/bench docs | Real staleness risk but would collide with benchmark-cuda triggers | — | **folded into benchmark-cuda** ("Publishing numbers") |
| profiling/roofline | Already covered by benchmark-cuda | — | fix only |

## Phase 3 — Changes made

- `benchmark-cuda/SKILL.md`: rewritten (see Phase 1 row). New sections: STOP (5 measurement artifacts of this box), Methodology, ncu/nsys WSL2+Docker recipes, compute-sanitizer-not-on-WSL2, Publishing numbers. Description gains negative cases.
- `sm120-cuda-expert/SKILL.md`: FP8-prefill row replaced by explicit disabled-note; FA2/NVFP4-lever rows added; Law 1/2 corrected (graph-runner class name; occupancy ceiling caveat). `references/known-issues.md`: CUTLASS-4.5 retry row updated, `executor_moe.cu` ref fixed, 3 negative results added (FP8 prefill, GDN in/out NVFP4, occupancy raise/KPAR→MR). `references/ptx-patterns.md`: 13.3 re-verification note, probe path fixed.
- `check-degeneration/SKILL.md`: server docker-run example; quant-file caveat block (re-downloads, NLL-vs-byte-equality, Qwen3.6 non-determinism).
- New: `building-and-testing/`, `server-api/`, `add-model-arch/`, `quant-formats/` (each a single SKILL.md).
- New: `README.md` (index + boundary map), this report.

All factual claims in new/changed content were verified against the working tree
(Makefile, CMakeLists, `src/runtime/config.h`, `engine_init_resolver.cpp`,
`storage_tier.h`, `model_arch.h`, `tools/imp-server/args.{h,cpp}`,
`tools/imp-server/main.cpp:192`, `tests/perf_baseline*.json`, scripts/, tests/api/).
One error caught in self-review: `sim_advance` lives in `schema_constrain.{h,cu}`,
not `json_schema.h` — fixed before commit.

## Phase 4 — Trigger-test matrix

Assessment: does the description fire on the positive prompts and stay silent on the negative ones? (✓ = description covers it verbatim or by clear synonym.)

| Skill | Should fire (examples) | Must NOT fire | Verdict |
|---|---|---|---|
| benchmark-cuda | "is this decode regression real?" · "profile the GEMV with ncu" · "refresh the perf baseline" | "make the kernel faster" (→ sm120) · "does the model still output sane text" (→ check-degeneration) | ✓ — both negatives named in description |
| sm120-cuda-expert | "write an NVFP4 GEMV variant" · "kernel emits HMMA instead of mxf4nvf4" · "smem layout for hd=128" | "how fast is it" (→ benchmark-cuda) · "which quant format is this" (→ quant-formats) | ✓ — pairing note covers measurement; quant-formats negative covered from the other side |
| check-degeneration | "after the KV change, does Qwen still answer coherently?" · "output is ' own own own'" | "server returns 404" (→ server-api) · "tests fail to build" (→ building-and-testing) | ✓ |
| building-and-testing | "run the GPU tests" · "CI is blocked" · "bump GTest" · "build dir is root-owned, can't delete" | "bench the model" (→ benchmark-cuda) · "think-leak in responses" (→ server-api/check-degeneration) | ✓ — both negatives named |
| server-api | "add a field to /v1/messages" · "tool calling returns malformed JSON" · "cache_control doesn't report cache_read tokens" | "imp-cli prints garbage" (→ add-model-arch/check-degeneration) · "kernel slow" | ✓ — CLI-only negative named |
| add-model-arch | "add support for Qwen4" · "model loads but is prompt-blind" · "digits come out scrambled" | "Q8_0 block layout question" (→ quant-formats) · "decode tok/s low" (→ benchmark/sm120) | ✓ — both negatives named |
| quant-formats | "how does NVFP4 two-level scaling work here?" · "weight lands on slow gemm_nvfp4 path" · "which KV dtype saves most VRAM" | "write a faster dequant kernel" (→ sm120) · "is Q6_K decode at roofline" (→ benchmark-cuda) | ✓ — both negatives named |

Cross-check: no two descriptions claim the same trigger words for different intents;
overlapping nouns ("perf baseline" in benchmark-cuda vs "verify-fast" in
building-and-testing) are disambiguated by intent verbs (measure/refresh vs run/gate).
No dead references (all cited paths smoke-checked, see Phase 3).

Smoke tests: command surfaces verified against source rather than executed end-to-end
(GPU jobs are long-running): Makefile targets exist as cited; `imp-server` flags match
`args.cpp`; `degen_suite.py` flags/categories match source; ncu host install present
at `/opt/nvidia/nsight-compute/2026.2.0/`; nsys on host PATH.

## Residual trigger risks

1. **"slow" without context** can plausibly route to benchmark-cuda or sm120-cuda-expert — acceptable: the skills cross-reference each other in line 1, so either entry point converges.
2. **degen_suite.py is server-level** but owned by check-degeneration; server-api links to it. A prompt like "run the API battery" may fire server-api first — its Validation section routes to the same tool, so no wrong outcome.
3. **quant-formats vs add-model-arch** on "NVFP4 model produces garbage": both could fire; add-model-arch's fingerprint table routes quant-layout cases onward. Watch in practice.
4. GOAL.md's stale Hopper lines (see Phase 1 side finding) could mislead an agent reading GOAL.md directly — skills now consistently say sm_120a-only, but the doc itself should be fixed in a separate docs PR.
