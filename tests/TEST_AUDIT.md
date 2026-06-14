# Test Suite Audit & Refactor Plan — Re-Audit 2026-06-06

Commit base: `7a94d81f` (post-#574) · Predecessor: [`docs/TEST_AUDIT.md`](../docs/TEST_AUDIT.md)
(2026-06-04, Phase-1 gap analysis; its top-10 program was implemented via PRs
#527–#539). This document is the **delta re-audit** after PRs #538–#574
(gpt-oss, LoRA, IQ4, chunked-NLL rework, CUTLASS grouped GEMM, issue sweep)
plus the dimensions not covered by the 06-04 audit (perf variance,
stale/dead, runtime/split).

Phase 0 note: `AGENTS.md` and `AUDIT.md` do not exist in the repo
(audit sources were CLAUDE.md, docs/archive/tile-fa2-dispatch-shelved.md,
docs/TEST_AUDIT.md, CMakeLists, Makefile, scripts/verify.sh, tests/refs/README.md).
That dispatch log describes the shelved cuTile-FA2 track and does not concern the suite.

Classification (as 06-04): **A** = independent reference (fp64/format spec,
committed generator, documented tolerance) · **A−** = real reference, but
benign data / unjustified or unasserted tolerance · **B** =
tautological (imp-vs-imp, round-trip) · **C** = smoke/structural.

---

## 1. Coverage Matrix (Module × Test Type)

State after #574. "CI" = runs in GitHub CI (no GPU runner → unit only);
"local" = needs GPU and/or model.

| Module | unit | integration | e2e | numeric-correctness | perf | untested (most valuable gaps) |
|---|---|---|---|---|---|---|
| core (tensor/config) | ✓ (structural) | — | — | — | — | reshape/slice numerics (unchanged since 06-04) |
| compute/attention | ✓ | ✓ | ✓ greedy-locks | **A**: crosspath fp64-Golden (6 prefill paths pairwise, `test_attention_crosspath.cu`); **A**: paged-Oracle as TYPED_TEST over 6 KV-Dtypes incl. INT4/INT8 (`test_attention_paged_oracle.cu`, R8/#582) | bench-only | `attention_blackwell.cu` [routing table: **closed #577** `test_routing_decision.cpp`; attention sinks (gpt-oss #572): **A closed, #584** `test_gpt_oss_sinks_ref.cu`] |
| compute/other kernels | ✓ | — | — | A (RoPE/RMSNorm/softmax/reduce vs CPU; **YaRN long-seq fp64 #584** `test_gpt_oss_yarn_ref.cu`) | — | GPT_OSS_GLU activation, sampling numerics (only "token in vocab") |
| quant | ✓ | ✓ | ✓ | **A**: `test_gguf_dequant_ref.cu` (Q8_0/Q6_K/Q4_K/IQ4_NL/IQ4_XS as TYPED_TEST vs fp64 format spec, byte-LCG, no-NaN-Guard); **A**: `test_nvfp4_outlier_ref.cu` (adversarial fixtures); **A**: `test_gpt_oss_mxfp4_convert_ref.cu` (R1.1/#576, bit-exact); **A**: `test_cutlass_grouped_ref.cu` (R1.2/#576, fp64-CPU-Ref on identical quant bits) | bench-only | — |
| memory/KV | ✓ | ✓ | ✓ prefix-cache E2E (#538 ship-gate, 4/4 active) | A (FP8-KV calibration) | — | eviction+refill output stability, INT8/INT4/NVFP4-KV accuracy bands, vram allocator budget |
| model/loader/tokenizer | ✓ incl. fault-injection (#535 family) | ✓ | ✓ | A (merges, Jinja; **Harmony-render-Golden vs HF #584** `test_gpt_oss_harmony_golden.cpp`) | — | hf_hub |
| exec | ✓ partial | ✓ | ✓ | **A** (grouped GEMM via `test_cutlass_grouped_ref.cu`, see §2) | — | `executor_ffn.cu` isolated, `executor_lora.cu` kernel (only E2E via test_lora) [grouped-vs-fallback routing: **closed #577** `test_routing_decision.cpp`] |
| lora (new #571) | — | — | ✓ `test_lora.cpp` (A−: zero-B/nonzero-B identity) | — | — | kernel isolation, multi-adapter, rank edge cases |
| runtime | ✓ (think/stop, scheduler, json-FSM incl. $ref/$defs #562) | ✓ | ✓ determinism-E2E (#542) | — | graphs-gate in verify.sh | ConditionalRunner, request-lifecycle/abort, warmup-token type |
| vision | ✓ CPU preprocessing (#564) | ✓ GPU encoder+projector frozen-golden (R9/#583: SigLIP+gemma4v, committed 64² PNG → projector spots, f16-class ≤1e-2 rel + NaN/Inf-Guard, mmproj standalone without LM) | manual (gemma-3/4 VL) | — | — | vision RoPE/norm single kernels isolated (golden only locks the encoder output) |
| api/server | ✓ SSE-utils, anthropic-transform, stream_pipeline (real source files compiled in, not mock!) | ✓ relaunch | ✓ | — | — | `handlers.cpp` itself (only its utils tested), abort path |
| api/HTTP (Python) | ✓ mock (contract/errors/lifecycle) | ✓ real (model-bound) | ✓ | — | TTFT/decode gates (real-only) | recursive json_schema at API level; /v1/messages streaming E2E |

Compared to 06-04: ~45% direct → noticeably better in attention/quant/runtime/api
(6 of 10 risks closed, see §6); vision-GPU now covered via frozen-golden
(R9/#583), exec isolation unchanged blind.

## 2. Hot-Path Kernels — Depth of Correctness Tests

| Kernel | Tests | Class | Finding |
|---|---|---|---|
| NVFP4 quantize→dequant→GEMV | `test_nvfp4_outlier_ref.cu` + `tests/refs/gen_nvfp4_outlier_golden.py` | **A** | fp64 reference from format definition (E2M1+UE4M3+1/512-floor), 4 adversarial distributions (Gemma collapse class #514/#516), hard no-NaN/Inf-Guard. Exemplary. |
| NVFP4 block-scaled MMA (mxf4nvf4) | `test_mxf4nvf4_qkt_validate.cu` | A− | E2M1-exact inputs → reference lossless, but benign data regime; precision edge never exercised. `_probe` = smoke. |
| NVFP4-GEMV-Loop | `test_nvfp4_gemv_kpar_loop.cu` | B | Deliberately documented negative result (repro mathematically equivalent); regression latch, not a correctness proof. OK as is. |
| **CUTLASS grouped GEMM (pp512-10×-path #574)** | `test_cutlass_grouped_3x_nvfp4.cu` + `test_cutlass_grouped_ref.cu` (R1.2/#576) | **A** | grouped-vs-per-expert (B, staging) PLUS independent fp64-CPU reference on the identical quant bits the GPU consumes (quant error cancels → f16-accumulation class ≤1e-2, measured ~4.9e-4). Boundary distributions M=0/1/200, single-active-expert. |
| FP8-E4M3 encoder | `test_cutlass_nvfp4_alpha.cu` | A− | canonical bit limits pinned exactly (448/overflow/240-cliff); prefill part smoke. |
| fp8 FMHA | `test_fmha_fp8.cu` | A− | CPU reference correct, but tolerance deliberately NOT asserted ("characterized", README §2 — e4m3 on short rows 0.58–0.71 rel, #512 mechanism). Quality gate is by policy at the E2E locks. Fill partly still `%13` pattern. |
| FA2 prefill (all 6 paths incl. cuBLAS-legacy) | `test_attention_crosspath.cu` + fp64-Golden | **A** | Pairwise f16-class agreement 1e-2 (measured ~4e-4 post-#528), realistic heavy-tail data regime, bit-identical LCG fixtures Python↔C++. The "killer assert" from risk #1 exists. |
| FA2 chunked continuation (#553/#568) | `test_attention_chunked.cu` + `test_chunked_prefill.cu` | A−/E2E | Kernel test + teacher-forced-NLL gates (since #553 NLL instead of byte-equality — correct due to re-download quant drift). hd=256 parity locks via #570. |
| Paged decode F16 | `test_attention_paged_oracle.cu` | **A** | fp64-Ref from original f16 K/V; kv_len=333 deliberately not block-aligned. |
| Paged decode FP8/INT8/INT4 | ditto (envelopes; since R8/#582 as TYPED_TEST over all 6 KV-Dtypes) | A− | Quant paths "characterized" with ASSERTED frozen envelopes (correction 06-07: INT4/INT8 envelopes already existed since PR #534 — the "none" initial finding was stale). Risk #6 from 06-04 closed. |
| Paged NVFP4-TC | `test_attention_paged_nvfp4_tc*.cu` | C local | Launch+SASS-Guard; numerics via offline microbench + E2E (synthetic data NaNs both paths — documented). |
| GGUF dequant incl. IQ4 (#561) | `test_gguf_dequant_ref.cu` | **A** | see §1. Edge cases (d=0, NaN-d, max-scale) included. |
| MMVQ/dp4a | `test_mmvq.cu`, `test_gemm_dp4a.cu` | B (diagnostic) | dp4a-vs-MMVQ without hard threshold; the real reference runs via `test_gguf_dequant_ref.cu` (GEMV ≤2.5e-2 with measured envelope). |
| MoE routing | `test_moe.cu` | A− | CPU-top-k reference, but hardcoded unambiguous logits — no tie cases. Executor test = not-NaN. |
| **gpt-oss MXFP4 experts + sinks (#572)** | — | **C→A for sinks/Harmony/YaRN (#584)** | Sinks: fp64-softmax-Ref + eviction geometry (`test_gpt_oss_sinks_ref.cu`); Harmony: HF-render-golden exact (`test_gpt_oss_harmony_golden.cpp`); YaRN long-seq fp64 up to 131071 + inversion sensitivity (`test_gpt_oss_yarn_ref.cu`). MXFP4 converter: **A closed #576** (`test_gpt_oss_mxfp4_convert_ref.cu`, bit-exact). |
| GDN/SSM | `test_gdn.cu`, `test_ssm.cu` | A− | CPU delta-rule-scan reference present; tolerance only implicit (EXPECT_NEAR), data synthetic-benign. |

Note on the audit prompt: "TMA warp-spec grouped GEMM" does not exist on sm_120
(no TMA-WS/tcgen05); the grouped-GEMM path is CUTLASS `mma.sync` —
evaluated accordingly above.

## 3. Quant Correctness: Goldens & Tolerances

- **`tests/refs/` is the working schema** (Phase 2 legacy): committed
  numpy generators, bit-exact regeneration, tolerance policy with justification
  in `tests/refs/README.md` (f16-class ≤1e-2 rel measured ~4e-4; fp8
  characterized-not-blessed; NVFP4 ≤1e-1 + E2E-locks; generator crosscheck
  ≤1e-9). Three consumers: crosspath, nvfp4_outlier, e2e_greedy_locks.
- **Gap (as of 06-07 largely closed via #576/#584):** grouped GEMM
  and gpt_oss_mxfp4_convert now have class-A references per the policy;
  YaRN/Harmony goldens committed. Remaining: mxf4nvf4-validate (benign
  data); FP8-FMHA fixtures still partly use the `%13` fill pattern that made
  #525 vacuous.
- GGUF dequant tolerances are documented AND justified (dequant ≤1e-3 =
  pure f32 rounding; GEMV ≤1e-2; dp4a/MMVQ 2.5e-2 with activation-quant
  noise, measured envelope) — target state.

## 4. Determinism

- **DISABLED_ inventory (3):** 2× `test_determinism_e2e.cpp`
  (cross-context on GDN-Hybrid — documented, justified limit from #542)
  · 1× `test_attention_fmha_mxfp4.cu:141 DISABLED_BasicHD256` ("requires
  large shared memory; disabled pending smem optimization" — justified, but
  without issue ref).
- **Exact-equal asserts are deliberately and correctly used:** greedy-locks
  (`test_e2e_greedy_lock.cpp:170`) run 2× fresh-context, so that an
  atomics flip is itself the finding; PPL bit-identity is the deliverable
  of deterministic mode (#542); prefix-cache fresh-vs-hit token-equality is
  the ship-gate (#538). Chunked prefill uses NLL tolerance since #553 instead of
  byte-equality (correct response to the logit-tie/quant-drift class).
- **MoE atomics:** Qwen3.6 nondeterminism is documented and covered via
  `[runtime] deterministic` + DetEvalE2ETest; a quantifying
  "atomics spread probe" (N=20, spread-bound) is still missing (honorable
  mention 06-04).
- ~51 GTEST_SKIP paths, all legitimately patterned (model/HW/build-feature).
  No seed variation in any test (all seed=42) — acceptable with pure
  greedy focus, a hole for sampling numerics (see §1).

## 5. Perf Tests vs. Measurement Variance

- `verify.sh` gate: median of trials with 15 s cooldown, decode 3% hard;
  **prefill is WARN-only** — deliberate and correct response to cuBLAS's
  2.6× container-restart variance. Decode daily drift (8–15%,
  #526) is NOT detected: the gate compares against the baseline without
  clock/power plausibility check (`nvidia-smi` sampling DURING the bench,
  healthy = ~2850 MHz SM / 13801 MHz mem / ~500 W) → on depressed days
  false-positive regression possible.
- **Three baseline files, three schemas** (legacy implicit / v1 / north_star
  without version field); `perf_baseline_chunked.json` 29 days old;
  `north_star` is read by no target except `verify-north-star`.
- `tests/api/test_perf_regression.py` gates TTFT-p95/decode-p50 hard against
  keys `["ttft"]`/`["throughput"]` in `perf_baseline.json` — **which do not
  exist there at all** (verify.sh schema). Real-only-marked, drops out in the
  mock run, but would run against missing keys in a real run →
  check/repair.
- All 11 `*Bench*` GTests (ctest -L perf) are print-only diagnostics without
  asserts — fine, as long as that is documented; they should not create the
  impression of a gate.

## 6. Gaps vs. Known Priorities

| Priority (prompt) | State 06-06 |
|---|---|
| FA2-prefill coverage incl. legacy cuBLAS path | **CLOSED** — crosspath tests all 6 paths pairwise (A-class); chunked continuation via #553/#570 NLL/parity locks |
| /v1/messages streaming real | **OPEN at E2E level.** Since #564, envelope/reasoning-split/anthropic-transform are unit-tested (against the real server source files, not mock — good). Real streaming behavior E2E still only OpenAI-side; Anthropic-SSE-E2E missing. |
| Tool-arg schema validation | Partial: pass-through in `test_tools.py`; FSM level incl. $ref/$defs in `test_json_constrain.cu` (#562). **API-level recursive schema untested.** |
| Constrained output | Well covered (whole-token-FSM #517/#519 chain + #562). |
| **New since 06-04, untested:** | executor_lora kernel. [gpt_oss_mxfp4_convert: **closed #576** (bit-exact fp64-Ref); grouped-vs-fallback routing: **closed #577**; attention sinks, Harmony parity vs HF, YaRN long sequence: **closed #584**] |
| Open 06-04 risks: | #2 partial (mode-2-from-scratch still without external oracle), #6 **closed** (INT4/INT8-paged envelopes existed since PR #534 — initial finding stale; since R8/#582 TYPED_TEST over all KV-Dtypes), #9 deferred (justified — MTP dead end), #10 partial (fault injection ✓, Unicode roundtrip ✓ via fixtures+robustness; NUL class via SSE tests ✓). |

## 7. Stale / Dead

| Finding | Evidence | Assessment |
|---|---|---|
| `tests/golden/*.txt` (5 files) | **zero consumers** (grep over tests/scripts/tools/CI); 2× Mistral-Small = deleted model; Makefile `test-golden` itself points to pytest | **dead → delete** (low-risk) |
| `tests/api/test_outputs/` (21 committed .txt) | run artifacts of `test_repetition_compare.sh` (writes there), committed once months ago | **artifacts → delete + .gitignore** (low-risk) |
| `tests/refs/gen_reference.py` | zero usage; infrastructure for future HF-layer comparisons | dormant — keep, mark as dormant in README |
| `tests/fixtures/` | **NOT dead** (agent's initial finding wrong): `scripts/validate_safetensors.py:628,892` consumes both files | keep |
| Stale comment "FMHA sm120 requires sm_90+ (WMMA fallback)" | `test_attention_fmha_sm120.cu:84` | misleading on sm_120a-only engine → fix (low-risk) |
| Comment "same fate as the SM100-only tcgen05 family" (`tests/bench/mmq_q4k_imma_bench.cu:30`) | agent's initial finding "dead commentary" — checked: correct in context (tcgen05 IS sm_100-only, which is exactly why it is missing on sm_120) | keep |
| `test_engine_chat.sh`, `test_server_concurrent.sh`, `test_server_embed_chat_interleave.sh`, `test_server_0token_battery.py`, `test_server_robustness.py`, `tests/api/elchtest.sh` | called by no target/CI (need a running server + GPU) | manual tools → documented as manual in each script header, do not delete |
| TurboQuant/BitDecoding/FP4-PV remnants | cleanly removed; only comment reference in CMakeLists (documentary) | ✓ no action needed |
| Apparent duplicates (mmq_q4k×5, mxf4nvf4×3, attention_mxfp4×2) | checked: pyramid unit→layout→bench resp. paged-vs-FMHA | **no duplicates** |
| `tests/bench/*.cu` in `IMP_COMPUTE_SOURCES` (CMakeLists ~196–207) | only under `IMP_BUILD_TESTS OR IMP_BUILD_BENCH`, commented (P3/P5 decision) | accepted; no prod leak in the default release without tests |

## 8. Runtime & Parallelizability, GPU/Host Separation

- **Split is clean at the binary level:** 8 binaries, ctest labels
  unit/gpu/perf; unit binaries are pure .cpp without device code (cuda_fp16.h
  only for host conversion). CI (no GPU runner) builds + skips test job;
  ctest runs without `-j` — correct for GPU tests (VRAM/context contention),
  for the unit-label subset `-j` would be safe.
- **Fragility (CLOSED #580, R5):** the unit/gpu split of `test-e2e`
  hung on a hardcoded gtest_filter string in CMakeLists
  (`_unit_e2e_filter`) — test renaming silently moved tests into the wrong
  label. CPU and GPU tests are interleaved in `test_e2e.cpp`/`test_continuous_batching.cpp`
  (StubModelTest shares a fixture across CPU and GPU subtests),
  so no clean separate binary without fixture duplication → instead
  keep filter + guard test `guard_e2e_lane_split`
  (`scripts/check_e2e_lane_split.sh`), which checks via `--gtest_list_tests` that
  the filter resolves EXACTLY to the frozen CPU set (37 tests) — a
  rename now fails loudly instead of silently moving. In addition:
  `gtest_discover_tests()` registered ALL tests again individually next to the
  label aggregates → `ctest` without label ran everything twice (measured:
  1215 ctest entries); removed → now 14 entries (3 unit + 1 guard + 6 gpu
  + 4 perf), each test runs exactly once.
- **No per-test timeout in CMake** (only CI-global 120 s); long E2E tests
  (model load + 128 tokens) block the sequential run.
- **Measured (2026-06-06, RTX 5090, without models):** total suite 1,154 tests;
  7 of 8 binaries together < 11 s, but **test-attention alone 241 s** —
  the Makefile assumption "GPU tests < 30s" is stale. Drivers are the
  paged/crosspath oracle sweeps; a `-L unit` run stays < 1 s. (Makefile
  comment `test-gpu` corrected, #580; as of 06-07 it is ~1,202 tests.)
- **Python-API suite hangs on nothing:** `run_mock_tests.sh` (CPU-capable,
  mock) is called by neither CI nor verify.sh — the only
  CPU-CI-capable contract suite runs only manually.
- Model gating via `IMP_TEST_MODEL*` env vars, copied decentrally across the test files.
  **CLOSED #581 (R6):** central registry `tests/test_models.h`
  (header-only) holds the env-var names (`imp_test::kEnv*`) + accessors
  (`env_path`/`env_path_or`) in ONE place; the ~14 consumers now pull the
  name from the registry instead of from copied string literals (the
  GTEST_SKIP calls remain at the call site, because GTEST_SKIP returns from the
  *calling* function). The hardcoded
  `/models/...` fallback paths (degeneration/api_generate/relaunch/lora/chunked)
  are NOT a defect: they match the container mount `-v $(PWD)/models:/models`
  from the Makefile and skip cleanly when the file is missing — hence kept as
  call-site-visible literals. `test_mtp_forward.cpp` encoded a
  host path (`/home/kekz/models/...`) that never existed in the container → normalized to
  `/models/...` container style (#581), now skips consistently.

---

## Prioritization (Impact × Effort)

**P1 — catches real bugs, moderate effort**
1. `gpt_oss_mxfp4_convert.cu` reference test (format-spec-fp64 like
   `test_gguf_dequant_ref.cu`; MXFP4 nibble order was JUST a real bug
   in the issue-sweep #560 family!). Effort: S–M.
2. CUTLASS grouped GEMM against an independent reference (per-expert-fp32-CPU or
   dense-cuBLAS-fp16 path instead of the same adapter). The new 10×-prefill
   path deserves more than B-class. Effort: M.
3. Paged-INT4/INT8 oracle following the pattern of `test_attention_paged_oracle.cu`
   (methodology exists, just add dtypes). Effort: S–M.
4. `attention_dispatch` routing-table test ((hd,seq,dtype)→path, pure
   host logic; #493 was exactly this class). Effort: S.
5. Clarify `test_perf_regression.py` baseline-key mismatch (hits nothing
   or breaks on the first real run). Effort: S.

**P2 — Velocity/Robustness**
6. Wire the Python mock suite into CI (`pytest -m "not perf and not tools"`,
   CPU-only — the only CI-capable contract check). Effort: S.
7. ~~Attention-sinks unit (sink-logit shift vs CPU-softmax reference) +
   Harmony-template golden vs HF-tokenizer output.~~ **DONE (P2.7, #584):**
   `test_gpt_oss_sinks_ref.cu` (gpt-oss sink-logit vs fp64-softmax reference,
   incl. StreamingLLM slot-eviction geometry), `test_gpt_oss_harmony_golden.cpp`
   (imp-Jinja-render vs HF `apply_chat_template` golden, exact), plus
   `test_gpt_oss_yarn_ref.cu` (YaRN long-sequence parity up to pos 131071 vs fp64,
   sensitive to the #547 rope_freq_scale inversion). Generators+goldens in
   `tests/refs/` (gen_harmony_golden.py, gen_yarn_rope_golden.py).
8. Extend the decode gate with clock/power plausibility check (#526 class:
   on mem-clock < 13801 MHz WARN instead of FAIL). Effort: S.
9. ~~Decouple unit_e2e_filter from test-name coupling + clean up duplicate
   ctest registration.~~ **DONE (R5, #580):** keep filter +
   guard test `guard_e2e_lane_split` (rename-proof via `--gtest_list_tests`
   comparison against the frozen CPU set); `gtest_discover_tests` removed
   (1215→14 ctest entries, no more double run).

**P3 — Hygiene (low-risk, immediately executable)**
10. Delete `tests/golden/` (dead), delete `tests/api/test_outputs/` +
    .gitignore, fix the misleading skip message in `test_attention_fmha_sm120.cu`,
    mark manual scripts as manual,
    mark `tests/refs/gen_reference.py` as dormant.

**Do not do:** Spec-decode exactness (#9 from 06-04) stays deferred
(MTP = proven dead end at current precision); do not turn bench tests into
gates (variance); no seed sweeps for greedy paths.

---

## Phase 2 — Refactor Plan (proposal, risk/effort per item)

| # | Item | Risk | Effort |
|---|---|---|---|
| R1 | **Reference-first for new hot paths:** P1.1–P1.3 as one package "class-A anchors for #572/#574 paths" — generators per `tests/refs/` schema (committed, bit-exact, tolerance justified) | low (only new tests) | M |
| R2 | **Routing/host-logic units:** P1.4 attention_dispatch + grouped-vs-fallback decision in `executor_forward` as CPU tests (no GPU needed → CI lane) | low | S |
| R3 | **CI-lane Python mock** (P2.6): new CI step after build, `pytest tests/api -m "not perf and not tools"` against mock_server; runs without GPU | low (CI-only) | S |
| R4 | **Harden perf gate** (P1.5 + P2.8): repair test_perf_regression keys or switch suite to verify.sh schema; verify.sh samples clocks.mem/power during the bench and degrades FAIL→WARN on depressed-host signature; baseline files get a uniform `schema_version` field | medium (gate semantics change — coordination, because CI behavior is affected) | M |
| R5 | **Decouple test-e2e split** (P2.9): stub unit tests in own binary; keep label aggregates; reduce gtest_discover_tests double registration to label aggregates — **DONE (#580):** own binary discarded (CPU/GPU tests share fixtures, not separable without duplication); instead keep filter + guard `guard_e2e_lane_split` (`scripts/check_e2e_lane_split.sh`, rename-proof), `gtest_discover_tests` removed (1215→14 ctest entries, no double run), Makefile `<30s` comment fixed | medium (runner rework, touch Makefile/CI/verify.sh paths) | M |
| R6 | **Model env registry:** one `tests/test_models.h` (header-only) with the env vars + accessors instead of a copied pattern; mechanical migration — **DONE (#581):** `tests/test_models.h` (`imp_test::kEnv*` + `env_path`/`env_path_or`), ~14 files migrated (env names from registry, SKIP stays at call site, /models fallbacks preserved), `test_mtp_forward` host path → container path normalized | low (mechanical, semantics-preserving) | M (grind work) |
| R7 | **Hygiene batch** (P3.10) | minimal | S |
| R8 | **Parametrization Quant×Arch** (prompt wish): NOT recommended as a large matrix — the greedy-lock/NLL-E2E suite already parametrizes over the actually present models, and a synthetic arch matrix (LLaMA/Mistral/DeepSeek/…) without weights only tests loader paths that `test_e2e_models`/loader tests already cover. Instead: TYPED_TEST over KV-Dtypes in the paged-oracle (R1) and over quant formats in the dequant-ref — there parametrization is cheap and real. | low | S–M |
| R9 | **Vision-GPU golden** (the only completely blind GPU area): a frozen SigLIP-encoder golden (small committed image → projector-output spots, tolerance f16-class) — **DONE (#583)**: `tests/test_vision_golden.cu` + `tests/refs/vision_encoder_golden.h`, covers SigLIP **and** gemma4v (committed 64² PNG, mmproj standalone without LM), ≤1e-2 rel + 5e-3 abs + NaN/Inf-Guard, clean SKIP without model, `make test-vision` (dump mode regenerates) | low | M |

Recommended order: R7 (immediately) → R2+R3 (cheap, CI impact) → R1 (core) →
R4 → R6 → R5 → R9. Integrate R8 into R1.

— End of Phase 1+2. Phase 3 only after approval; marked as low-risk and thus
executable without further approval: **R7 (hygiene batch)**.
