# AUDIT.md — imp Test-Coverage Audit (Step 1 of TEST_AUDIT.md)

**Date:** 2026-06-15 · **Scope:** all of `tests/` + inline test usage vs `src/` subsystems.
**Method:** CMakeLists test-registration analysis + 6 parallel read-only subsystem audits
(compute, attention, quant, moe/gdn/kv, model/tokenizer/vision, api/runtime/e2e), each
mapping production code → test → oracle → tolerance, then cross-checked against the source.

**This is the Step-1 deliverable. No tests have been written or changed yet.** It ends with
a prioritized plan (Step 3 tiers) awaiting go/no-go.

---

## 0. TL;DR — where the safety net is thin

The suite is large (~110 test files, 8 binaries, ~574 GTest cases) and in several places
**exemplary**. The weak spots are not "missing tests" broadly — they are **specific kernels
and contracts with no *independent* oracle**, where a silent corruption passes today:

1. **Q4_0 dequant/GEMV has zero correctness oracle** (only e2e no-NaN smoke). Common GGUF
   format; was a real bug site (PR #691 `Q4_0Deterministic`).
2. **FP8 E4M3 dequant values are never checked against an independent reference** — only loose
   round-trip (1.5 abs on a ±2 range). FP8 KV is now *auto-default* for Qwen3 (PR #704).
3. **MoE run-to-run nondeterminism is asserted as *exact-equal*, not epsilon-bounded** — the
   known Qwen3-30B-A3B-NVFP4 greedy A/B flip has no characterization test. *(The grouped-GEMM
   expert compute itself IS oracled — see §7; the original "no oracle" finding was a cross-agent
   false positive.)*
4. **mxfp4 attention uses the wrong oracle methodology** (absolute-error vs imp's own FP16 kernel,
   `EXPECT_LT(max_err,0.5)` — not per-block relative-error vs a dequantized reference).
5. **No tokenizer HF-parity and no byte-exact chat-template golden run in CI** — the one true
   parity test is env-gated/skipped and only requires ≥80% match. The entire prior cross-engine
   PPL gap (#657) was a pretokenizer bug — exactly this blind spot.
6. **The real server `handlers.cpp` (~4600 LOC) is ~0% exercised in CI** — the Python `tests/api/`
   suite runs against a hand-written `mock_server.py`. Tool-call arg parsing and API-key auth
   have **no test at all**.

**Important caveat (§7):** this audit was assembled from 6 parallel per-area sub-agents, each of
which saw only its own slice. A post-audit reconciliation found **several false-positive "no
oracle" gaps** — coverage that exists in a *different* test binary than the agent searched. The
verified-genuine gap list is in §7; the tables in §5 are annotated accordingly. These map onto the
dispatch's Tier-0/Tier-1/Tier-2 priorities (§5).

---

## 1. Test infrastructure & the CI reality

- **8 GTest binaries** (`CMakeLists.txt:449-606`): `test-core, test-text, test-compute,
  test-attention, test-quant, test-kv, test-moe-gdn, test-e2e`. Split so a kernel change relinks
  one binary, not all 60 files.
- **`ctest` registration is via 3 label aggregates** (`unit`/`gpu`/`perf`, lines 627-652), NOT
  `gtest_discover_tests` (that double-ran the suite, R5/#580). The `unit`/`gpu` split inside
  `test-e2e` is a gtest filter guarded by `guard_e2e_lane_split` so a rename can't silently move
  a test into the wrong lane.
- **CI has no GPU runner** (per `building-and-testing`): the `gpu` and `perf` lanes are
  **skipped in CI**; only the `unit` lane + the mock API suite run. GPU correctness/perf is a
  **local-only** gate (`make verify-fast`). This is the single biggest structural fact of this
  audit: **most of the strong GPU oracles below never execute in CI** — they protect against
  local regressions only.
- **`make coverage`** (per memory, PR #716) measured ~51% server line coverage; the real
  handler path is ~0% in CI. This audit confirms the structural cause (§4G, §api).

---

## 2. Subsystem coverage summary (what's strong vs thin)

### compute (test-compute) — mostly strong, a few no-oracle GEMVs
- **Strong, independent oracle:** `test_rope.cu` (CPU fp32 ref), `test_gpt_oss_yarn_ref.cu`
  (fp64 + committed numpy golden + inversion sensitivity guard — best-in-class), `test_layernorm`,
  `test_activation`, `test_reduce`, `test_embedding`, `test_gemm` (CPU GEMM + cuBLAS),
  `test_gemm_q4k_fused_prefill` (full in-test Q4_K dequant ref), `test_ffn_sparsity`,
  `test_executor_kernels`.
- **No independent oracle (HIGH):** **Q6_K dp4a GEMV** (`test_gemm_dp4a.cu:130` finite-only),
  **MMVQ** (differential vs dp4a only — shared bug passes), **FP8 GEMV** (`test_fp8_gemm.cu:63`
  tested only with all-zero input → output zero).
- **Property-only (MED):** stochastic sampling (no chi-square vs softmax dist), general softmax
  rows (sum=1 property, not abs-err vs CPU).
- **Edge gaps:** norm/activation/GEMM lack {1,31,32,33,511,512,513}-boundary + NaN/Inf + K-not-
  mult-of-MMA-tile; embedding has no out-of-range-token-ID test.

### attention (test-attention) — best-covered subsystem; two flagship suites
- **Exemplary:** `test_attention_crosspath.cu` answers the Tier-0 question — **FA2 ↔ legacy
  cuBLAS parity is asserted** (f32-score chain agrees pairwise ≤1e-2, both track an fp64 ref
  pinned to a numpy golden at 1e-9, on realistic heavy-tailed magnitudes). `test_attention_
  paged_oracle.cu` is a typed-test over 6 KV dtypes vs fp64-from-original-f16 with per-dtype
  *characterized* envelopes — the correct quant methodology.
- **Parity shape holes:** HD=128 only; **seqlen ∈ {1,2,max} absent everywhere**; HD∈{64,96,256}
  never cross-path parity-tested (hd=256 = historic #566 hot-spot); no non-causal or B>1 in the
  parity assert; tile-boundary ±1 (32±1/512±1) not hit.
- **Wrong oracle (HIGH):** mxfp4 attention (`test_attention_mxfp4.cu:187`,
  `test_attention_fmha_mxfp4.cu:117`) compares FP4 to an FP16 kernel with *absolute*-error tol on
  uniform fills — a biased dequant within 0.1 abs passes. INT4 paged decode
  (`test_paged_attention.cu:1148`) builds its CPU ref from the *same* INT4-dequantized K/V →
  tautology (and INT4 is not in the paged_oracle typed list).
- All `GTEST_SKIP`s here are sm-gated → they **run** on the sm_120a target; only
  `DISABLED_BasicHD256` (`test_attention_fmha_mxfp4.cu:139`) never runs.

### quant (test-quant) — strong GGUF/NVFP4 anchors, FP8/Q4_0/MXFP4-GEMV thin
`test_gguf_dequant_ref.cu` is the class-A anchor (fp64 host re-derivation from the format def,
explicitly anti-tautology). Per-format matrix:

| Format | Round-trip | Dequant-vs-ref | Oracle | Verdict |
|---|---|---|---|---|
| Q8_0 | n/a | **yes** | fp64 host re-deriv + IMMA exact-model | strong |
| Q4_K_M | n/a | **yes** | fp64 host re-deriv; IMMA grid-derived | strong (gemm uses *mean*-rel — masks local error) |
| Q5_K | n/a | **partial** | fp64 ref exists but only dp4a GEMV path tested | no dequant-kernel assert |
| Q6_K | n/a | **yes** | fp64 host re-deriv | strong |
| **Q4_0** | n/a | **NO** | none — e2e no-NaN smoke only | **no oracle at all** |
| **FP8 E4M3** | yes (loose) | **NO** | imp round-trip + imp-FP16 consistency | **no independent value oracle** |
| NVFP4 | yes | **yes** | spec host decode + fp64 numpy golden | strong on tested paths |
| INT8 | n/a | yes | CPU-naive reimpl | ok (same one-line formula) |
| MXFP4 | n/a | converter **yes** / GEMV **NO** | fp64 spec (converter); imp-vs-imp (GEMV) | GEMV arithmetic unchecked |

CI compounding: the genuine NVFP4 GEMM oracles (`test_cutlass_grouped_ref.cu`, `_3x`, `_alpha`,
`_smallM`) are all `GTEST_SKIP` on non-sm120 → **CI runs ~0% of the real NVFP4 GEMM math**, only
imp-vs-imp dispatch/wiring. `test_weight_dispatch.cu` tests wiring, not format→tier *selection*
(tier hand-set, scales forced to 1.0).

### moe / gdn / kv (test-moe-gdn, test-kv) — routing/GDN/KV solid, expert-GEMM blind
- **Strong oracle:** MoE *routing* (`test_moe.cu` `cpu_topk_gating` — selected experts, softmax
  weights, prefix-sum, gather/scatter), GDN scan (full delta-rule CPU ref), SSM conv1d (causal
  no-future-leak), KV write (paged addressing, INT8 round-trip), KV gather (bit-exact + FP8
  dequant-vs-ref), FP8/INT8 KV decode (split-K parity), prefix cache (byte-equivalence on a
  zero-init pool — non-tautological, + ref-count/eviction/stale-block guards).
- **No oracle (HIGH):** **grouped-GEMM expert compute** (only NaN/shape/self-consistency);
  **MoE nondeterminism bound** (`MoEExecutorTest.Deterministic` does a same-process double-forward
  asserting *exact* equality — neither detects nor bounds the documented NVFP4 atomic-scatter drift).
- **No oracle (MED):** FP8 KV *write*/quantize path (only read/decode side); end-to-end
  prefix-cache *logit* equivalence (only KV-byte identity).

### model / tokenizer / chat-template / vision (test-core, test-text, vision)
- **Strong:** loader fault-injection (`test_gguf_fault_injection.cpp` 19 cases — bad magic, trunc,
  2^60 counts, OOB offsets; non-tautological), SafeTensors/SPM/llm-compressor parsers,
  `test_tokenizer_robustness.cpp` (byte-level GPT2 encode∘decode identity over all 256 bytes,
  surrogates, OOB-id), dequant-on-load (shares the class-A `test_gguf_dequant_ref.cu`).
- **HIGH gaps:** **no tokenizer HF-parity in CI** — `test_tokenizer_compat.cpp` is the only true
  HF-golden test, but `GTEST_SKIP` unless `IMP_TEST_MODEL`/`IMP_TEST_GOLDEN` are set (CI sets
  neither), bar is only ≥80% (`:172`), golden not committed. Pretokenizer chunk truths are
  hand-typed constants, no committed generator. **Chat templates are token-COUNT/contains
  asserts, not byte-exact**, for all 9 families *except* gpt-oss (the only HF-golden:
  `test_gpt_oss_harmony_golden.cpp`). **Tool-call / thinking-channel rendering is detected but
  never rendered/asserted.**
- **MED:** vision goldens (`test_vision_golden.cu`) are committed stability locks (no fp64 oracle)
  but GPU-gated → never run in CI; HF-config arch detection covers 7 arches (Phi-4/Nemotron/
  Qwen3-dense/Mistral/gpt-oss not unit-tested); SafeTensors has no full-model load test.

### api / runtime / e2e (test-e2e, test-core server bits, tests/api/*.py)
- **Strong:** `test_e2e_greedy_lock.cpp` (frozen token sequences through the CUDA-graph path),
  `test_determinism_e2e.cpp` (same-context bit-identical greedy + PPL), `test_json_constrain.cu`
  (deep schema-FSM: pattern masking, premature-close/trailing-comma rejection, $ref/recursion,
  ~92 asserts), `test_sse_stream_utils.cpp` (byte-level NUL-leak #510 + think-leak regression —
  **note: this file embeds literal NUL bytes, so `grep` reports it "binary"; it is in fact
  assertion-heavy, not assert-free**), `test_anthropic_transform.cpp`, `test_config.cpp`,
  `test_routing_decision.cpp`.
- **HIGH gaps:** **tool-call arg parsing** (`tool_call.cpp` 754 LOC, `parse_tool_calls_*` /
  `validate_tool_call`) has **zero unit tests**; **API-key auth** (`main.cpp:152-168`,
  constant-time Bearer compare, /health bypass) is **untested**; **real-server logprobs**
  sum + top-k ordering stability is unasserted (Tier-0 contract).
- **Structural:** the Python `tests/api/` suite runs against `mock_server.py`; `run_mock_tests.sh`
  even excludes the only real-server marks (`tools`, `perf`). `/v1/messages` is absent from the
  mock; its synthetic streaming event sequence is asserted **nowhere**. The #712 robustness
  battery (`test_server_robustness.py`, 72 cases) and the #710 0-token battery are **manual, not
  CI-wired**.

---

## 3. Reference oracles available for new tests (Step 2)

Good news: most oracles needed for the gaps already exist in-repo and can be reused:
- **GGUF dequant fp64 re-derivation** → `test_gguf_dequant_ref.cu` (extend to Q4_0, Q5_K-dequant).
- **NVFP4 spec decode** → `test_nvfp4_quant_ref.cu` / `src/compute/nvfp4_quant_ref.cu` (reuse for
  NVFP4/MXFP4 GEMV value oracles).
- **MXFP4 E2M1×UE8M0 spec decode** → `test_gpt_oss_mxfp4_convert_ref.cu` LUT (reuse for MXFP4 GEMV).
- **fp64 attention ref + numpy golden** → `test_attention_crosspath.cu` /
  `gen_attention_crosspath_golden.py` (extend to seqlen{1,2,max}, HD≠128).
- **CPU top-k gating** → `test_moe.cu` (pairs with a CPU/cuBLAS per-expert matmul for grouped-GEMM).
- **HF `apply_chat_template` / `AutoTokenizer`** → `generate_tokenizer_golden.py` +
  `gen_harmony_golden.py` (extend to per-family tokenizer + chat-template + tool-call goldens).
- FP8 E4M3: no in-repo oracle → add a host E4M3 decode LUT (or PyTorch `to(float8_e4m3)` golden).

---

## 4. Hygiene findings

- **A. Dead test:** `tests/test_gdn_kernel.cu` is built standalone (`add_executable(test-gdn …)`,
  `CMakeLists.txt:718`) but **never `add_test`-registered**, is printf/`main`-style with **0
  EXPECT/ASSERT**, and its CPU ref was superseded by the identical `gdn_scan_cpu` in `test_gdn.cu`.
  → **Delete or convert to GTest.**
- **B. Intentional bench (no-assert, expected):** `test_mxf4nvf4_mma_variants_bench.cu` and the
  other `*_bench` files — print-only, correctly outside the correctness lanes.
- **C. False-positive note:** `test_sse_stream_utils.cpp` is NOT assert-free (see §api) — `grep`
  mis-classifies it as binary due to embedded NUL test vectors.
- **D. `DISABLED_`:** `test_determinism_e2e.cpp` (×2, cross-context GDN limit — documented,
  correct), `test_attention_fmha_mxfp4.cu` (`DISABLED_BasicHD256` — mxfp4 hd=256 uncovered),
  `test_chunked_prefill.cu`. None should be naively re-enabled (known boundaries).
- **E. `GTEST_SKIP` that hides coverage in CI:** the NVFP4 GEMM ref tests + all GPU/model-gated
  tests (vision golden, tokenizer HF-compat, e2e model tests) skip in the GPU-less CI. This is
  expected given no GPU runner, but means the *strong* oracles are local-only.
- **F. Loose/arbitrary tolerances to revisit:** FP8 scaled round-trip (1.5 abs / 0.5 rmse on a ±2
  range), GEMM mean-rel 3%/1% (mean masks localized error), smallM 3e-2/5e-2 (self-described
  "generous, tighten later"), `nvfp4_quant_hw` 0.15 RMS.
- **G. CI executes ~0% of the real `handlers.cpp` and ~0% of the real NVFP4 GEMM math.**

---

## 5. Prioritized gap table & plan (Step 3 tiers)

Ranked by blast radius (wrong dequant/mask/routing silently corrupts every token = high;
`/health` = low). Maps to the dispatch's tier structure.

### Tier 0 — highest blast radius (do first)

| subsystem | risk | current | target | reference source |
|---|---|---|---|---|
| Q4_0 dequant + GEMV | high | e2e no-NaN only | dequant-vs-ref + dp4a GEMV err bound | extend `test_gguf_dequant_ref.cu` (fp64 `d·(nibble−8)`) |
| FP8 E4M3 dequant values | high (auto-default #704) | loose round-trip only | dequant-vs-ref, grid-derived rel bound | host E4M3 LUT / PyTorch `float8_e4m3` golden |
| MMVQ Q4_K/Q5_K/Q8_0 | high | mmvq↔dp4a differential | abs err vs CPU dequant GEMV | in-test GGUF dequant CPU ref |
| Q6_K dp4a GEMV | high | finite/no-NaN only | abs/rel err vs dequant | port `test_gemm_q4k_fused_prefill.cu` ref |
| FP8 GEMV (`gemv_fp8`) | high | zero-input only | nonzero abs err | host fp8 decode + GEMV |
| mxfp4 attention oracle | high | abs-err vs FP16 kernel | per-block rel-err vs host dequant | fp64 from dequantized mxfp4 grid (mirror paged_oracle) |
| FA2 parity seqlen {1,2,max} | high | min tested Sq=24 | crosspath parity at Sq=1,2,max | existing fp64 crosspath ref |
| Sampling determinism (greedy/logprobs) | high (Tier-0 contract) | greedy locked; logprobs unasserted | greedy bit-exact + logprob sum/top-k order stable | self-consistency + softmax dist |

### Tier 1

| subsystem | risk | current | target | reference source |
|---|---|---|---|---|
| MoE grouped-GEMM expert compute | high | NaN/shape/self-consistency | per-element output vs reference | CPU-naive per-expert matmul or cuBLAS dense-per-expert |
| MoE nondeterminism bound | high | exact-equal double-forward | drift ≤ ε logits / ≤ N flips over K cold runs | fresh-executor reruns, NVFP4 atomic-scatter path |
| MXFP4 GEMV arithmetic | high | imp-vs-imp dispatch | dequant-vs-ref, per-block rel budget | reuse `test_gpt_oss_mxfp4_convert_ref.cu` LUT |
| Q5_K dequant kernel | med | dp4a GEMV only | add dequant-kernel case | existing `ref_dequant_q5_k` |
| FA2 parity HD∈{64,96,256} | med-high | WMMA/fp8-vs-CPU only | crosspath assert per HD | crosspath fp64 + numpy golden |
| INT4 paged decode | med | self-referential dequant | add INT4 to paged_oracle typed-test | fp64 from original-f16 K/V |
| FP8 KV write/quantize path | med-high | none (read side only) | write→read round-trip bound | host FP16→FP8 quantize ref |
| Tile-boundary seqlen (32±1, 512±1) | med | odd lengths only | explicit ±1-of-tile cases | existing fp64 refs |
| Dispatch format→tier selection | med | wiring-only, scale=1.0 | assert selection + real scales | format-detection vs expected tier |

### Tier 2

| subsystem | risk | current | target | reference source |
|---|---|---|---|---|
| Tokenizer HF parity (per family) | high | opt-in, CI-dark, ≥80% bar, no golden | committed byte-exact golden, runs in CI | HF `AutoTokenizer` via `generate_tokenizer_golden.py` |
| Pretokenizer regex chunking | high | hand-typed constants | committed generator → chunk goldens | HF `tokenizers.pre_tokenize_str` |
| Chat-template render (8 families) | high | token-count/contains only | byte-exact rendered-string golden | HF `apply_chat_template` (extend `gen_harmony_golden.py`) |
| Tool-call render + arg parsing | high | detection only / 0 unit tests | render+parse per family + schema validation | `tool_call.cpp`; HF templates with tools |
| API-key auth | high | untested | 401 bad / 200 good / timing-safe / health-bypass | `main.cpp:152-168` |
| Real-server logprobs | high | 5xx smoke only | descending top-k, stable order, sum sanity | `utils.cpp:355` |
| `/v1/messages` real + synthetic streaming | high/med | transform unit-tested; e2e absent | assert synthetic SSE sequence + round-trip | `anthropic.cpp` |
| json_schema returned-output validity (e2e) | med | FSM masking only | real-server completion validates vs schema | `test_json_constrain.cu` |
| Wire #712 robustness + #710 0-token batteries into CI | med | manual scripts | gated CI job | existing `test_server_robustness.py` etc. |
| Vision encoder golden in CI | med | committed, GPU-gated | gated GPU lane or numeric anchor | (no fp64 oracle — document as lock) |
| GGUF MoE/GDN/vision structural parse + HF-config arches | med | llama-only / 7 arches | per-arch parse smoke + config fixtures | format spec / committed config.json |
| Prefix-cache logit equivalence (e2e) | med | KV-byte identity only | hit-vs-cold logits identical | cold full-prefill forward |

### Deliberately OUT of scope (low blast radius / correct-as-is)
- `/health`, `/v1/models`, `/metrics` contract (trivial handlers, mock-tested).
- Bench-only files (`*_bench`, `test_mma_peak_saturated`, `test_mxf4nvf4_probe`) — timing, not
  correctness, by design.
- `cluster_launch` host helpers (fully covered).
- Re-enabling the 3 `DISABLED_` determinism-boundary tests (documented hardware limits).
- `compute-sanitizer` lane: **cannot run on this WSL2 host** (WDDM, no debugger interface — per
  `building-and-testing`); a native-Linux CI lane is the only place Step-4 racecheck/initcheck
  can live. Flagged, not implementable here.

---

## 6. Bugs surfaced during the audit

One dead artifact (`test_gdn_kernel.cu`, §4A) and one layout inconsistency surfaced while
building the Tier-0 oracles:

**F1 — `gemv_q4_0_q8_1` nibble-layout inconsistency (Q4_0 dp4a GEMV).**
Building a Q4_0 dp4a-GEMV oracle (feeding a standard ggml Q4_0 block, the exact bytes that
`dequant_q4_0_kernel` decodes bit-exactly — see the now-green `GgufDequant/Q4_0` oracle) produced
a ~6× error vs the independent fp64 reference (`max_rel=6.27`, gpu=-14.6 vs ref=3.37).
Root cause: `Q4_0_Traits::dp4a_block` (`src/compute/gemv_dp4a_traits.cuh:251`, via
`unpack_nibbles_2`) consumes nibbles **interleaved** — block element `2k`=`qs[k]`&0xF,
`2k+1`=`qs[k]`>>4 — whereas standard ggml Q4_0 and imp's own `dequant_q4_0_kernel`
(`src/quant/dequant_gpu.cu:299`) are **split** (element `e`=`qs[e]`&0xF, `e+16`=`qs[e]`>>4).
No repack to interleaved was found, and Q4_0 decode is registered as `StorageTier::FP16`
(`src/exec/gemm_kernel_gguf.cu:287`, dequant→fp16 GEMV), so `gemv_q4_0_q8_1` appears latent /
not on the production Q4_0 decode path. **Repro:** `run_dp4a_gemv("Q4_0", QType::Q4_0, 256, 1024,
gemv_q4_0_q8_1)` against `ref_dequant_q4_0`. **Status:** quarantined (not asserted) pending a
decision — fix the kernel's unpack to split layout, or confirm it is dead and remove it. The
Q4_0 *dequant* arithmetic is now independently oracled regardless.

Remaining genuinely no-/weak-oracle paths (FP8 dequant/GEMV now oracled; mxfp4 attention bias
now guarded) are tracked in §7. Any further real bug found while building tests will be filed
here with a minimal repro, per the dispatch rule (tests adapt to the engine, not the reverse).

---

## 7. Reconciliation — false positives & verified-genuine gap list

The 6 parallel sub-agents were each scoped to one area and could not see oracles living in another
test binary. Cross-checking against the source found these **false-positive "no oracle" findings**
(coverage that already exists — do NOT duplicate):

| Claimed gap (origin) | Reality — already covered | Evidence |
|---|---|---|
| "Q6_K dp4a GEMV has no oracle" (compute agent, from `test_gemm_dp4a.cu`) | Independent fp64 oracle exists in the **quant** binary | `test_gguf_dequant_ref.cu:783` `Q6_K_GemvDp4a` vs fp64 `ref_dequant_q6_k` |
| "MMVQ has no independent oracle" (compute agent, from `test_mmvq.cu`) | fp64 oracle exists for MMVQ Q8_0/Q4_K | `test_gguf_dequant_ref.cu:787-788` `*_GemvMmvq` |
| "MoE grouped-GEMM expert compute has no reference oracle" (moe/gdn agent) | Independent **per-expert fp64 CPU matmul** of the read-back NVFP4 bits exists | `test_cutlass_grouped_ref.cu` (header §11, `dequant_to_fp64`, body `:212`) — sm120-gated, **runs on the RTX 5090 target** |
| "greedy not bit-exact run-to-run" (implied Tier-0) | Greedy argmax determinism + e2e token locks already present | `test_sampling.cu:40-96`, `test_e2e_greedy_lock.cpp`, `test_determinism_e2e.cpp` (same-context) |

**Verified-genuine gaps actually worth implementing** (each re-confirmed against current source):

*Tier 0*
- **Q4_0 dequant + GEMV** — no fp64 oracle anywhere; only constant-nibble e2e smoke
  (`test_quant_integration.cu:73`). `gemv_q4_0_q8_1` kernel exists and is untested for correctness.
- **Q5_K dequant kernel** — `ref_dequant_q5_k`/`build_q5_k` already exist but Q5_K is absent from
  the dequant TYPED_TEST (`test_gguf_dequant_ref.cu:586`) — only its dp4a GEMV is tested.
- **FP8 E4M3 independent value oracle** — only imp↔imp round-trip (`test_quant.cu:322`) +
  saturation; no independent E4M3-decode LUT check of `cast_fp8_to_fp16`.
- **FP8 GEMV nonzero** — `gemv_fp8` tested only with all-zero input (`test_fp8_gemm.cu:75`).
- **mxfp4 attention proper oracle** — replace abs-err-vs-FP16 (`test_attention_mxfp4.cu:197`,
  `test_attention_fmha_mxfp4.cu:117`) with per-block rel-err vs host-dequant reference.
- **FA2↔cuBLAS crosspath parity at seqlen {1,2,max}** — crosspath min tested Sq=24.

*Tier 1*
- **MoE nondeterminism ε-bound** — `MoEExecutorTest.Deterministic` asserts exact-equal
  same-process; no cold-rerun drift characterization on the NVFP4 atomic-scatter path.
- **FP8 KV write/quantize path** — only read/decode side tested.
- **FA2 crosspath parity at HD∈{64,96,256}**; **INT4 in paged_oracle typed-test**;
  **dispatch format→tier selection** (currently wiring-only, scales forced to 1.0).

*Tier 2*
- **Tokenizer HF-parity golden** committed + CI-runnable (currently env-gated, ≥80% bar);
  **pretokenizer chunk goldens** from a committed generator.
- **Byte-exact chat-template goldens** per family (currently count/contains only, except gpt-oss).
- **Tool-call render + arg parsing** unit tests (`tool_call.cpp`, 0 tests today).
- **API-key auth** (401/200/constant-time/health-bypass) — untested.
- **Real-server logprobs** sum + top-k descending-order stability — untested (handlers.cpp).
- **`/v1/messages` synthetic-streaming event sequence** assertion.
- **CI-wire** the #712 robustness battery + #710 0-token battery.
- **`tests/README.md`** (Step-4 deliverable).

Net effect: the original §5 tables remain accurate for *risk*, but the Tier-0 quant/compute work
is ~half what it first looked like (Q6_K GEMV, MMVQ, and grouped-GEMM were already oracled), and
greedy bit-exactness is already locked. Implementation proceeds against this reconciled list only.
