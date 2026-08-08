# Test-hardening log

Per-iteration scorecard for the adversarial test-suite campaign. Reports:
[`TEST_INVENTORY.md`](TEST_INVENTORY.md) · [`MUTATION_BASELINE.md`](MUTATION_BASELINE.md) ·
[`ESCAPE_ANALYSIS.md`](ESCAPE_ANALYSIS.md). Working evidence lives in `loop/`
(local-only, not committed).

---

## Iteration 1 — 2026-08-08 — focus: baseline (Phase 0) — commit: `ad067d76`

**Mutation score: 73.5 %** (25/34 killed, full local suite) — no previous
baseline. **CI-lane score: 2.9 %** (1/34).

| Category | Score | | Category | Score |
|---|---|---|---|---|
| rope | 4/4 = 100 % | | masking | 5/6 = 83 % |
| scaling | 4/4 = 100 % | | indexing | 4/5 = 80 % |
| quantization | 3/3 = 100 % | | kvcache | 1/3 = 33 % |
| memory | 2/2 = 100 % | | sampling | 0/3 = 0 % |
| numerics | 2/2 = 100 % | | controlflow | 0/2 = 0 % |

**Bugs found:** S1: 1 · S2: 0 · S3: 0 · S4: 0 · S5: 1 (#1302, structural)

**Escape distribution:** E1: 2 · E2: 0 · E3: 2 · E4: 0 · E5: 1 · E6: 5 · E7: 2

**Tests added:** 0 — this iteration was Phase 0 only, as the dispatch requires
(baseline before any hardening). Each gap is filed with the concrete test that
would close it. Iteration 2 writes them, and each will be watched failing
against its mutant before it is watched passing.

**Dead tests remaining:** 4 (`DISABLED_`) — 2 benchmarks, 2 documented
determinism known-limits (#554). None newly dead. The larger number is
*conditional*: **934 of 2 125** executable cases (44 %) need a GPU that no
automated gate has; 61 of them skip silently inside binaries CI does run, 54 of
those the KV-cache manager suite. (These two figures were first published as
927/2 060 — measured against a stale `build-dev`; corrected in iteration 2.)

**Issues filed:** #1299 (S1, determinism) · #1300 (`top_p`) · #1301 (KV prefix
hash) · #1302 (imp-server has no coverage) · #1303 (non-split sink) ·
#1304 (meta/scorecard)

**Attacked but clean — do not re-mine in iteration 2:**
attention masking (causal, sliding-window, sink slot eviction), attention
scaling (QK scale drop/double, split-K partial merge), RoPE (theta, neox pair
layout, Q-vs-K, position offset), Q4_K/Q5_K dequant (nibble order, zero-point,
scale swap), `__syncthreads()` removal in both paged-decode pipelines, online
softmax running-max, softcap tanh, page-table batch offset and KV block stride,
block-hash parent chaining. All killed, most by real oracle tests.

**Blocked on:** nothing hard. Two stated limitations:
1. The end-to-end perf gate (`tests/perf_baseline.json` via `make verify-fast`)
   was not run against M29/M30 — it needs a full `make build` per mutant. So
   "no test catches them" is proven; "nothing catches them" is not.
2. M10 and M22 are indeterminate: their only reacting oracle is the test that
   #1299 shows is red on `main`. They resolve for free once #1299 is fixed.

**Tree clean:** `git status --porcelain` → only new untracked files under
`docs/audit/`, `loop/`, `tools/mutation/`. Zero production files modified; all
34 mutants byte-restored and verified after each run.

### Self-check

1. **Did I modify, skip or loosen any existing test?** No. Zero edits under
   `tests/`.
2. **Did every mutant get reverted?** Yes — byte-restore in a `finally`, plus a
   `git status --porcelain` assertion after each of the 34, plus a final check.
3. **Did I watch every new test fail before it passed?** N/A — no tests added
   this iteration.
4. **Is every bug claim backed by a script I ran twice?** Yes.
   `loop/repro/BUG-1.sh` reproduced 2/2 from fresh processes;
   `loop/repro/BUG-1-models.sh` ran 3 fresh processes per model across 3 models.
5. **Did I fix any production code?** No.
6. **Are all new tests wired into CI?** N/A — none added.

### What the run cost in wrong answers, and what fixed them

Recorded because each was a claim that would have shipped as fact:

* The first assertion classifier reported 628 A0 tests (29.8 %). Two bugs: it
  did not follow one-line delegating test bodies, and its brace matcher counted
  `'}'` inside char literals, truncating bodies to nothing. Corrected figure:
  402 (19.1 %), and only 5 tests with no assertion at all.
* 20 red tests in the first GPU run were a typo in my own model path — and
  revealed that `test_determinism_e2e.cpp:54` fails instead of skipping on a
  bad path, contradicting `tests/README.md:5`.
* 38 red `test-e2e` tests were WSL2 VRAM exhaustion, not defects
  (`ChunkedPrefillTest.*` is 7/7 in isolation). Reported as an environment
  property, not a bug.
* M22 and M10 first read as survivors, then as kills, and are now recorded as
  indeterminate — because the oracle that reacted is itself red (#1299).
* M29 "killed" one test on its first re-check and passed 7/7 on the second.
  Chasing that flake is what found #1299.

### Next iteration

Stopping criteria are not met: score < 85 %, two categories at 0 %, one new S1.

Focus `I6` (sampling & stop conditions) — the 0 % category with a filed issue
and a written test plan — then `I2` (paging / KV / prefix cache), the 33 %
category where the survivors map to cross-request silent-wrong-output.

---

## Iteration 2 — 2026-08-08 — focus: `I6` sampling & stop conditions — commit: `ad067d76`

**Mutation score: 84.8 %** (28/33 killed) — prev **75.8 %** (25/33). Denominator
is 33, not 34: M25 was measured to be an **equivalent mutant** and excluded.

| Category | Score | prev |
|---|---|---|
| **sampling** | **2/3 = 67 %** | 0/3 = 0 % |
| **kvcache** | **2/2 = 100 %** | 1/2 = 50 % |
| masking | 5/6 = 83 % | unchanged |
| indexing | 4/5 = 80 % | unchanged |
| controlflow | 0/2 = 0 % | unchanged |
| rope / scaling / quantization / memory / numerics | 100 % | unchanged |

**Bugs found:** S1: 1 (#1305) · S2: 0 · S3: 0 · S4: 0 · S5: 0

**Escape distribution (new):** E6: 1 (#1305 — the CUB-path test's only fixture
has a positive maximum)

**Tests added: 6, each verified against the fault it targets: yes**

| Test | Binary | Kills | Runs in CI |
|---|---|---|---|
| `KVCacheManagerTest.BlockHashDiscriminatesEveryTokenPosition` | test-core | M23 | **yes** |
| `PrefixEquivTest.RollbackOfPartialAllocationDropsItsHashes` | test-kv | — (path was undriven) | no (GPU) |
| `SamplingTest.TopPTruncatesMultiblockPath` / `…Shifted` | test-compute | M20 | no (GPU) |
| `SamplingTest.TopPTruncatesCubPath` / `…Shifted` | test-compute | M21 | no (GPU) |
| `PagedAttentionTest.GQA_NoSplitK_HD64_Sinks` / `GQA_NoSplitK_HD64` | test-attention | — (see #1303) | no (GPU) |

Only one of the six lands in the lane that gates a merge; the rest are GPU
tests, which is the standing structural finding from iteration 1, not a choice
made here.

**`SamplingTest.TopPTruncatesCubPath` is RED on `main` and stays red.** It is
red because the product is wrong (#1305), and the dispatch forbids weakening an
oracle to make a suite green. CI does not run `test-compute`, so no gate turns
red; `make test-gpu` and `verify-fast` will show it.

**Dead tests remaining:** 4 (unchanged).

**Issues filed:** #1305 (S1). **Corrections posted:** #1301 (M25 half withdrawn
— equivalent mutant), #1303 (the uncovered sink path is the shared reduction
used by the cluster and quantised-KV kernels, not the FP16 non-split one).

**Attacked but clean:** the multiblock sampler under both all-negative and
shifted logits; the block-hash chain at every token position; the rollback path
of `allocate_blocks_with_prefix`; the FP16 non-split sink reduction.

**Blocked on:** #1299 still blocks a verdict for M10 and M22 — the only
end-to-end determinism oracle is the test that bug makes red.

**Tree clean:** production code untouched; `git diff` limited to `tests/`,
`tools/mutation/`, `docs/audit/`.

### How #1305 was found

Not by hunting — by writing the test for #1300 and taking its control assertion
seriously. The new `top_p` test asserts, as a guard against its own fixture
degenerating, that the same seeds *do* reach outside the nucleus when `top_p=1.0`.
On the CUB path that guard tripped: all 200 seeds returned token 0.

The discriminator was cheap and decisive. Softmax is shift-invariant, so the
same distribution expressed as `ln(p)` and as `ln(p)+5` must sample identically.
It does not: the all-negative form returns token 0 every time, the shifted form
works. That isolates the fault to `softmax_max_kernel`'s signed `atomicMax` on a
float bit pattern (`sampling_topk_topp.cu:397`), whose ordering is inverted for
negative floats, so the `-FLT_MAX` sentinel is never replaced. The same pattern
exists at `mtp_forward.cu:260`.

### Self-check

1. **Did I modify, skip or loosen any existing test?** One signature changed:
   `run_gptoss_shape_splitk_case(bool)` → `(bool, int seq_len = 256)`. The two
   existing callers are unchanged in behaviour; nothing was skipped, disabled,
   renamed or loosened.
2. **Did every mutant get reverted?** Yes — `recheck.sh` restores from a saved
   copy and prints `git status` after each run.
3. **Did I watch every new test fail before it passed?** For the four with a
   target mutant, yes (M20, M21, M23 killed; logged in `loop/evidence/recheck/`).
   The other two are honestly labelled above as covering a previously-undriven
   path rather than killing a mutant.
4. **Is every bug claim backed by a script I ran twice?** Yes —
   `loop/repro/BUG-2.sh`, 2/2 from fresh processes.
5. **Did I fix any production code?** No.
6. **Are all new tests wired into CI?** One is (`test-core`). Five are GPU tests
   and run under `make test-gpu` / `verify-fast`; there is no CI lane that can
   execute them, which is #1304's subject.

### Corrections this iteration forced

* **The iteration-1 inventory was measured against a stale `build-dev`** compiled
  on the branch `fix/answer-reserve-scales`, so `test-core` was reported as 958
  instead of `main`'s 955 (three branch-only `ForceThinkEnd.*` cases). Caught by
  diffing test-name lists when a count moved the wrong way. All counts
  re-measured; `TEST_INVENTORY.md` carries the note.
* **M25 was filed as a test gap and is not one.** Measured, not argued: the
  lookup site rejects the stale entry on its own.
* **M31's escape was mis-stated**, and the test written from the wrong diagnosis
  is what proved it wrong.

### Next iteration

Stopping criteria still not met: score 84.8 % (< 85 %), `controlflow` at 0 %,
one new S1 this round. Focus `I2` (paging / KV / prefix cache) and the two
indeterminate mutants, which unblock as soon as #1299 lands.
