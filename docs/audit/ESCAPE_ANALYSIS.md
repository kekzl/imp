# Escape analysis

Date: 2026-08-08 · Commit `ad067d76` · Companion to `docs/audit/TEST_INVENTORY.md`
and `docs/audit/MUTATION_BASELINE.md`

For every confirmed bug and every surviving mutant: *why did the suite not catch
this?*

| Class | Meaning |
|---|---|
| E1 | No test exists for this path or behaviour |
| E2 | Test exists, assertion too weak |
| E3 | Test exists but never runs |
| E4 | Test exists, tolerance too loose |
| E5 | Test mocks the thing under test |
| E6 | Test exists but its inputs cannot trigger the fault |
| E7 | Non-determinism hides it |

---

## Distribution

| Class | Count | Where |
|---|---:|---|
| E6 | 4 | M20, M21, M23, M31 |
| E1 | 2 | M29, M30 |
| E3 | 2 | BUG-1; the 54 CUDA-gated `test-core` cases |
| E7 | 2 | M10, M22 (verdict indeterminate — the only oracle is BUG-1's own test) |
| E5 | 1 | the entire `tools/imp-server` HTTP surface |

**E6 dominates.** That is a specific, cheap-to-fix diagnosis: the tests exist,
they are pointed at the right functions, and they are written with inputs under
which the fault is invisible. Four of the eight real survivors are a bad
*fixture*, not a missing test — and three of those four were closed in
iteration 2 with new inputs and no new machinery.

A ninth candidate, **M25, turned out to be an equivalent mutant** and is
excluded: see `MUTATION_BASELINE.md`. It is recorded here because "surviving
mutant ⇒ test gap" is only true for non-equivalent mutants, and this campaign
filed one issue on the wrong side of that distinction before measuring it.

The structural findings are the E3 and E5 entries, and they are the severe ones:
a gate that never runs, and an API tested against a reimplementation of itself.

---

## E3 — test exists but never runs

### BUG-1: the deterministic-mode E2E gate has never executed, and it fails

`DetEvalE2ETest` (`tests/test_determinism_e2e.cpp`) was added in #542 to prove
that `[runtime] deterministic` actually makes greedy decoding reproducible. It
is gated on `IMP_TEST_MOE_MODEL`.

Nothing sets that variable. It appears in exactly three places: the registry
header that declares it (`tests/test_models.h:64`), the test itself, and
`tests/.env.test` — a file **no script sources**. `make test-gpu` does not set
it. `scripts/verify.sh` does not set it. The documented full-suite recipe does
not set it. So the fixture takes its `GTEST_SKIP` branch on every run the repo
knows how to launch, and the lane has been green-by-skipping since #542.

Point it at a real model and it fails (`loop/repro/BUG-1.sh`, reproduced 2/2
from fresh processes; `loop/repro/BUG-1-models.sh` for the matrix, 3 fresh
processes per model):

| Model | Arch | greedy, graphs ON | graphs OFF | PPL bit-identical |
|---|---|---|---|---|
| `gpt-oss-20b-mxfp4` | MoE | fail 3/3 | fail 3/3 | fail |
| `Qwen3-4B-Instruct-Q8_0` | dense | **fail 3/3** | pass 3/3 | pass 3/3 |
| `Qwen3.5-4B-mxfp4` | GDN hybrid | pass 3/3 | pass 3/3 | pass 3/3 |

The dense row disproves the fixture's own header claim that "dense models pass
trivially since they skip the routed-expert kernels"
(`tests/test_determinism_e2e.cpp:28`), and it fails **only** with CUDA graphs on,
which narrows that half of the fault to capture/replay. gpt-oss fails both
variants, so there is a second, graph-independent contribution. The GDN hybrid
— the architecture #542 validated against — is the one that still passes.

Filed as #1299.

**Why this class is the worst one:** the test is correct, the assertion is
strong (byte equality of the generated string), and it names the failure
precisely in its own message. None of that helped, because the gate condition
silently removed it.

**The same shape, one step less severe:** 54 of the 60 `test-core` cases that
skip without a GPU are the KV-cache manager suite. They are *conditional* — a
local `make verify-fast` does run them — but they are invisible to the only gate
that runs on every PR.

## E5 — test mocks the thing under test

The `Mock API contract` CI job runs `pytest tests/api` with `IMP_USE_MOCK=1`,
which starts `tests/api/mock_server.py`: 555 lines of Python reimplementing the
endpoints. `tools/imp-server/` is never executed by CI.

The 82 tests that run assert status codes and field presence — for instance that
`temperature=2.5` returns 400 with `"temperature"` in the message
(`tests/api/test_errors.py:36`). The mock implements exactly that
(`mock_server.py:217-220`). Whether **imp-server** does is untested. Every
mutant in the dispatch's "API" category would survive by construction, which is
why none were written: the experiment's outcome is already known from the job
definition.

**Closed in iteration 12 (#1302).** `--model` became optional, so the shipping
binary starts model-less and answers its whole request-validation surface
without a GPU. The `Real API contract` job runs the 42 tests marked `nomodel`
against `build/imp-server` itself. Pointing the suite at the real server
immediately falsified two of its assertions and found one server defect — see
`TEST_HARDENING_LOG.md`, iteration 12. What remains mock-only is generation:
anything that needs tokens back still needs a GPU, and the split is now visible
as a marker rather than implicit in the job definition.

## E6 — the test exists, the inputs cannot trigger the fault

### M20 / M21 — `top_p` is never given a value that truncates anything

`SamplingTest.TopPFiltering` (`tests/test_sampling.cu:164`) is the designated
nucleus-sampling test. Its comment says `top_p=0.5`; the code passes `0.99`
against logits `{5.0 @2, 5.0 @7, -100.0 × 8}`. The tail mass is ~e⁻¹⁰⁵, so the
assertion `token == 2 || token == 7` holds whether or not the nucleus filter
runs at all.

Across the entire C++ suite, `top_p` only ever takes the values **1.0** (a
no-op), **0.95**, and **0.99**. The CUB-path test
(`SamplerCubPath.EachCallSamplesFromItsOwnLogitsNotThePreviousCalls:416`) uses
`top_p=0.95` with one token at logit 30.0 and 151 935 at −20.0 — again ~all mass
on the winner.

Fix: one test where the top-k candidates carry a genuinely spread distribution
and `top_p` must cut it, asserting the *set* of reachable tokens over many
seeds. Cheap, and it kills both mutants.

### M23 — the block hash is only ever probed by changing the first token

`KVCacheManagerTest.BlockHashDeterministic` (`tests/test_kv_cache.cpp:512`)
proves "different tokens → different hash" by mutating `tokens[0]`. A hash that
ignores the *last* token of a block passes it unchanged. Nothing else varies a
single token: `ContentAddressedPrefixCaching` uses `std::iota` sequences that
differ everywhere.

That is the highest-consequence gap in the list. A prefix cache that collides on
a block differing only in its final token serves another request's KV — the
silent-wrong-output failure mode, cross-request.

### M31 — the sink term lives in a helper no test reaches

There are three independent sink implementations, not two:

| Reduction | Sink handling | Covered by |
|---|---|---|
| `paged_attention_gqa_kernel` (FP16, non-split) | inline, `attention_paged.cu:266-274` | `GQA_NoSplitK_HD64_Sinks` (added in iteration 2) |
| `paged_attention_reduce_kernel` (split-K merge) | inline, `attention_paged.cu:1087-1099` | `GQA_SplitK_HD64_Sinks` |
| `crosswarp_reduce_and_write` (shared helper) | `attention_paged_common.cuh:384-391` | **nothing** |

M31 mutates the third. Its callers are the SM120 thread-block-**cluster** kernel
(`attention_paged.cu:1342`) and every **quantised-KV** decode kernel — int4, int8,
nvfp4, nvfp4_tc, fp8. Every test that touches those passes `n_sinks=0` and a null
sink pointer, so the term is reached by zero tests.

The first version of this section said the gap was "the non-split reduction".
That was wrong, and the test written from it did not kill the mutant — which is
how the real answer surfaced. #1303 carries the correction and the recipe: extend
`tests/test_fp8_kv_cache.cu`'s decode harness with a sink variant, lifting the
CPU reference from `run_gptoss_shape_splitk_case`.

The comment at `test_paged_attention.cu:674` is worth quoting, because it is the
same lesson one iteration earlier: *"The split-K branch only activates when
scratch is set — none of the older tests set it, so the hd=64 split-K path was
never covered."*

**Closed in iteration 13.** The recipe above (an FP8-KV decode with sinks) would
not have worked either: `paged_attention_decode_fp8` has no sink parameter at
all, and neither do the int4/int8/nvfp4 launchers — all six quantised callers
leave `attn_sinks` at its `nullptr` default. The branch is reachable through
exactly one launch configuration, the cluster kernel's: no split-K scratch
(`num_splits` stays 1), `n_q_per_kv` in {2,4,8}, head_dim in {64,96,128,256} and
**at least 8 context blocks**. `GQA_Cluster_HD64_Sinks` (256 tokens, scratch
withheld) hits it; with M31 injected it is the only one of the 16
`PagedAttentionTest` cases that fails.

A second defect had to be fixed for that test to be able to fail at all: the
sink values were numerically invisible. See `TEST_HARDENING_LOG.md`,
iteration 13.

## E1 — no test exists

### M29 / M30 — nothing in the suite can observe performance

M29 disables split-K entirely; M30 removes its scratch-capacity guard. Both are
correctness-neutral in the shapes the suite exercises, and the `perf` ctest label
resolves to the filter `*Perf*:*Bench*:*Throughput*`, which matches two
microbenchmarks — neither of which asserts a threshold on the paged-attention
decode path.

**Stated limitation:** the repo's real performance gate is not in the test suite.
It is `tests/perf_baseline.json`, enforced end-to-end by `make verify-fast`
against the `imp:test` image. That gate was **not** run against these mutants —
doing so needs a full `make build` per mutant. So the honest claim is "no *test*
catches M29/M30", not "nothing catches them".

## E7 — non-determinism hides it

### M10 / M22 — the only oracle that reacts is itself red

Both mutants produced failures in `DetEvalE2ETest` when re-run in isolation, and
in the main run both were discounted because that suite was already failing in
the baseline. Since BUG-1 shows the suite is red on the clean tree, a red run
under the mutant carries no information. They are scored as survivors, which is
the conservative direction.

This is the compounding cost of BUG-1: it does not just hide its own bug, it
removes the only end-to-end determinism oracle the suite has, and with it the
ability to judge two unrelated mutants.

---

## What this implies for the next iteration

1. **Fix the gate conditions before writing any test.** An E3 escape is free to
   fix and neutralises a correct, strong, existing test.
2. **Five of nine survivors are fixture problems.** New assertions are not
   needed; new *inputs* are. That is the cheapest ratio of the campaign.
3. **The KV-cache hash lifecycle is the one place where a survivor maps to a
   cross-request silent-wrong-output failure.** M23 first.
