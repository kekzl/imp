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

---

## Iteration 3 — 2026-08-08 — focus: `I2` paging / KV / prefix cache — commit: `d1906167`

**Mutation score: 87.8 %** (36/41) — prev 84.8 % (28/33). Eight new mutants
(M35–M42) aimed at the KV manager, the prefix hash chain and StreamingLLM
eviction; the denominator grows with them.

| Category | now | was |
|---|---|---|
| **kvcache** | **8/8 = 100 %** | 2/2 |
| masking | 6/7 = 86 % | 5/6 = 83 % |
| indexing | 5/6 = 83 % | 4/5 = 80 % |
| sampling | 2/3 = 67 % | unchanged |
| controlflow | 0/2 = 0 % | unchanged |
| rope / scaling / quantization / memory / numerics | 100 % | unchanged |

**Bugs found:** none. S1: 0 · S2: 0 · S3: 0 · S4: 0 · S5: 0

**Escape distribution (new):** E1: 1 (M35 — no test ever passed a `content_salt`)

**Tests added: 2, both verified against the fault they target: yes**

| Test | Binary | Kills |
|---|---|---|
| `PrefixEquivTest.ContentSaltSeparatesIdenticalTokenPrefixes` | test-kv | M35 |
| `PrefixEquivTest.RandomisedWorkloadKeepsProbeAllocationAndPoolConsistent` | test-kv | M38 |

The randomised one is a property harness rather than a scenario: a seeded
workload of overlapping prefixes, frees and evictions, with sequence lengths
sitting on and either side of the block boundary, asserting three invariants in
every state — probe equals actual reuse, no block appears twice in one
sequence's table, and the pool comes back whole once everything is freed and the
cache drained (the shape of #1115). It earned its place immediately by killing
M38 alongside the existing `LongestCachedPrefixProbe`.

**The one survivor was a genuine hole and is now closed.** M35 drops
`content_salt`, the seed that keeps two multimodal prompts with identical token
ids but different images from sharing KV. Both production call sites pass
`req->vision_content_hash` (`engine_scheduler.cpp:598`, `scheduler.cpp:88`), and
**no test in the suite passed a non-zero salt** — so the parameter could be
deleted outright and everything stayed green, while the second request answered
about the first request's picture.

**Attacked and clean — do not re-mine:**

- Prefix reuse contiguity after a miss (`ChainHoleStopsReuse` kills it).
- Partial trailing blocks registered as cacheable, and block-count rounding
  (`PrefixCachingWithPartialLastBlock` kills both).
- Stale hash entries on cached-block reclaim (`EvictionThenRefillIsNewNotStaleHit`).
- StreamingLLM: the extra boundary block from #963, and sink pinning.
  `EvictMiddleBlocksRetainsKernelWindowStart` is the #963 regression test and it
  still bites; sink pinning is covered by `EvictMiddleBlocksKeepsSinksAndWindow`.
- The SWA snapshot suite writes real byte patterns to device memory and asserts
  exact block indices — A4-grade, left alone.

**A source-level inconsistency that is not a bug.** `longest_cached_prefix_blocks`
is called without a `content_salt` at `engine_sampling_stop.cpp:337` and `:405`
while the allocator is called with one, so the two hash chains would diverge for
a multimodal request. It cannot fire: `hybrid_prefix_reuse_limit_` returns 0 for
any vision request twelve lines earlier (`:329`). Recorded so the next reader
does not re-derive it — and so that if that guard is ever relaxed, the salt has
to be threaded through with it.

**Blocked on:** unchanged — #1299 still blocks M10 and M22.

### Resolved: the perf gate does see M29

Iteration 1 left this open as a stated limitation — "no *test* catches M29" was
proven, "nothing catches it" was not, because the real gate
(`tests/perf_baseline.json` via `make verify-fast`) needs a full `make build`
per mutant. Measured instead as an alternating A/B on the same tree
(`tools/mutation/perf_ab.sh`, Qwen3-4B-Q8_0, the gate's own bench invocation):

```
round 1: clean=450.88  M29=288.12
round 2: clean=451.12  M29=287.51
round 3: clean=450.69  M29=287.45
```

**−36 % decode**, twelve times the gate's 3 % threshold, with the arms
alternating so drift cancels. So M29 is caught — by the perf gate, not by a
test. `controlflow` at 0 % measures the test suite, and for M29 that is the
whole truth: it is a gate-layer responsibility, not a missing test. **M30**
(the split-K scratch-capacity guard) is still unmeasured; it is a safety check
whose removal is neither correctness- nor throughput-visible in the shapes the
suite exercises.

One false alarm worth recording so nobody re-derives it: the M29 arm lands at
~287.5 tok/s and `tests/perf_baseline.json` pins `tg128: 287.19`, which briefly
looked like "the baseline is pinned 36 % low and would accept split-K being
disabled". It is not. The pin is for **Qwen3-8B-Q8_0**; the A/B ran Qwen3-**4B**.
The same 8B bench on the current tree gives **288.47 tok/s vs the 287.19 pin
(+0.45 %)** — the baseline is accurate. Two different models, one coincidence.

**Tree clean:** production code untouched.

### Self-check

1. **Did I modify, skip or loosen any existing test?** No.
2. **Did every mutant get reverted?** Yes — 8 mutants, tree checked after each.
3. **Did I watch every new test fail before it passed?** Yes: M35 against
   `ContentSaltSeparatesIdenticalTokenPrefixes`, M38 against the randomised
   harness.
4. **Is every bug claim backed by a script I ran twice?** No bugs claimed this
   iteration.
5. **Did I fix any production code?** No.
6. **Are all new tests wired into CI?** No — both are `test-kv` (GPU). The
   manager's own bookkeeping is CUDA-gated because the fixture allocates a real
   pool, which is the standing finding in #1304.

### Next iteration

87.8 % is above the 85 % bar, but `controlflow` sits at 0 % (< 70 %), so the
stopping criteria are still not met. That category is M29/M30 — both
perf-only, and the suite has no perf oracle at all; closing it means either a
kernel-level timing assertion or accepting that `tests/perf_baseline.json` is
the gate and saying so explicitly.

Remaining survivors overall: M10, M22 (indeterminate, blocked on #1299), M29,
M30 (perf-only), M31 (#1303, needs an FP8-KV decode with sinks).

---

## Iteration 4 — 2026-08-08 — focus: `I1` API surface & streaming — commit: `ce64a1cb`

**Mutation score: unchanged at 87.8 %** — no mutants were run. This iteration
was a hunt, and it was pointed at the one surface where a finding is guaranteed
novel: CI never executes `tools/imp-server/` (#1302), so nothing there has ever
been under test.

**Bugs found:** S1: 0 · S2: 0 · S3: 0 · S4: 0 · **S5→S1: 1 (#1310)**

**Escape distribution (new):** E5: 1 — the test exists, asserts exactly the
broken invariant, and runs against a reimplementation that cannot exhibit it.

### Method: differential probe, real server vs the mock CI tests

`tools/mutation/api_diff.py` sends one table of edge-case requests to both a
real `imp-server` and `tests/api/mock_server.py`, and reports where they
disagree. 46 cases, five status divergences:

| case | real | mock | |
|---|---:|---:|---|
| `n=2` | 200 | 400 | mock rejects `n>1`; the server returns two choices |
| unknown `model` | 404 | 200 | mock ignores the model field on chat completions |
| `max_tokens: null` | 200 | crash | `TypeError: '<' not supported between 'int' and 'NoneType'` |
| JSON array body | 400 | crash | `AttributeError: 'list' object has no attribute 'get'` |
| JSON scalar body | 400 | crash | `TypeError: argument of type 'int' is not iterable` |

Three are the mock 500-ing where the server answers correctly — the stand-in is
*weaker* than the thing it stands for. The `n=2` row is the sharper one:
`tests/api/test_errors.py` asserts a 400 with `"n" must be 1`, so CI encodes a
constraint the shipping server does not have.

Everything else matched case for case, which is worth stating: the server's
parameter validation is not the problem.

### #1310 — transport changes content

Comparing *content* rather than status found what the mock structurally cannot.
The model's third token opens `😊` (`f0 9f 98 8a`); with `max_tokens=3`:

```
non-streaming bytes: 48656c6c6f2120efbfbd    'Hello! <U+FFFD>'
streaming     bytes: 48656c6c6f2120          'Hello! '
```

`Utf8Stitch::feed` (`tools/imp-server/utils.cpp:12`) holds an incomplete tail
back and is wired into the streaming driver only. On the other path the dangling
bytes reach `dump_safe` (`utils.cpp:6`), whose `json::error_handler_t::replace`
substitutes U+FFFD. `stream_pipeline.h:91` already documents that mechanism in a
comment — the guard was added for one path and not the other. `/v1/messages` is
affected identically. Reproduced 2/2 (`loop/repro/BUG-3.sh`).

**Tests added: 1, verified to fail against the fault: yes**

`test_stream_nonstream_agree_across_truncation_points` sweeps `max_tokens` 1–8
and asserts both that the transports agree and that non-streaming introduces no
U+FFFD. It is the campaign's cleanest single artefact for #1302:

```
$ IMP_USE_MOCK=1 pytest test_streaming.py -q          # the CI lane
5 passed in 0.05s
$ IMP_TEST_URL=<real server> pytest -k truncation -q
E   AssertionError: max_tokens=3: transports disagree
```

Green against the mock, red against the server, same test, same commit. It
therefore does **not** turn CI red; it turns `make test-server` red, which is
correct until #1310 is fixed.

`tests/api/test_streaming.py:36` already asserted the same invariant and passed,
for two independent reasons: the mock has no tokenizer and emits whole ASCII
words, and it pins `max_tokens=16`, which does not truncate mid-character even
against a real server.

**Attacked and clean:** `max_tokens` at 0 / −1 / 10⁹ / null / missing / string;
`temperature` and `top_p` bounds and type errors; `top_k` 0 and −1; missing,
non-array and empty `messages`; empty, whitespace and null content; missing,
unknown and out-of-order roles; system-only prompts; unknown fields; stop
sequences empty, empty-list, ten-deep and matching text in the prompt; emoji,
combining marks, RTL, escaped lone surrogate; a 200 000-character message;
structured content blocks; malformed, empty, array and scalar request bodies.
The server answered all of them the way the mock does or better.

**Blocked on:** unchanged — #1299 still blocks M10 and M22.

### Self-check

1. **Did I modify, skip or loosen any existing test?** No — one test added.
2. **Did every mutant get reverted?** No mutants run this iteration.
3. **Did I watch every new test fail before it passed?** Yes, against a real
   server at `max_tokens=3`; and confirmed green against the mock, which is the
   finding, not an accident.
4. **Is every bug claim backed by a script I ran twice?** Yes —
   `loop/repro/BUG-3.sh`, 2/2.
5. **Did I fix any production code?** No.
6. **Are all new tests wired into CI?** The new test runs in CI and passes
   there — against the mock. Making it meaningful in CI is #1302, not something
   a test can fix.

### Next iteration

`I5` (model ingestion & error paths) is untouched: truncated weight files, a
missing tensor, wrong shapes, metadata that disagrees with the tensors — all of
which must produce a clear diagnostic rather than a segfault.
`tests/test_gguf_fault_injection.cpp` exists (18 cases) and is CPU-only, so
mutants there would land in the lane that gates a merge.

---

## Iteration 5 — 2026-08-08 — focus: `I5` model ingestion & error paths — commit: `cda991a6`

**Mutation score: 88.9 %** (40/45) — prev 87.8 % (36/41). Four new loader
mutants (M43–M46), all killed.

| Category | score |
|---|---|
| **ingestion** (new) | **4/4 = 100 %** |
| everything else | unchanged |

**Bugs found:** S1: 0 · S2: 0 · **S3: 1 (#1312)** · S4: 0 · S5: 0

**Escape distribution (new):** E1: 1 (#1312 — no test covers the semantic layer)

### The GGUF container layer is genuinely well defended

`tests/test_gguf_fault_injection.cpp` has 18 cases and they hold:

| Mutant | Killed by |
|---|---|
| M43 per-tensor bounds guard removed | `TensorOffsetPastEof`, `TensorOffsetMaxU64`, `TensorDimOverflow`, `TensorDimNegative`, `NonexistentTensorType` — five at once |
| M44 magic unchecked | `BadMagic` |
| M45 version gate widened | `GgufLoaderTest.UnsupportedVersion`, `BadVersion` |
| M46 zero-alignment guard removed | `ZeroAlignmentMetadata` — by SIGFPE, see below |

All four land in `test-core`, i.e. **the lane that gates a merge**. This is the
one area of the codebase where CI genuinely protects against the fault class.

### #1312 — the semantic layer has no tests at all

Every one of those 18 cases attacks the binary container. None checks whether
the tensors present match what the metadata claims. So:

```
GGUF declaring llama.block_count = 2, shipping no layer tensor
  -> load_gguf returns a Model, n_layers=2
  -> layer 0: wq=(nil) wk=(nil) ffn_down=(nil)
  -> layer 1: wq=(nil) wk=(nil) ffn_down=(nil)
  -> the only log line is an unrelated "No tokenizer data found"
```

`n_attn` is counted (`gguf_loader.cpp:730`) and printed next to `cfg.n_layers`
in one format string (`:769`) but never compared. All three consistency checks
in the file are `WARN` (`:855`, `:862`, `:866`) and none covers this case.
`engine_weight_upload.cpp:145-149` then papers over it: `if (n_attn == 0)
n_attn = mcfg.n_layers;`.

**Scope discipline:** verified at the loader boundary only. No forward pass was
run on such a model, so no IMA or wrong-output claim is made — S3, not S1. The
same class has shipped here before in a different loader (Qwen3-VL: 247/316
tensors, `nullptr` into `vision_gemm`).

The committed test is a **characterisation** test, not the invariant. This file
runs in the CI lane, and a red required check blocks every merge in the repo —
that call belongs to the owner. #1312 carries the strict version, which is the
same test with its assertions inverted.

### A defect in this harness, found by a mutant that should have died

M46 first scored SURVIVED. It does not survive: removing the zero-alignment
guard makes `align()` execute `pos_ % 0`, and the binary dies with **rc=136
(SIGFPE)**. A crashed gtest binary prints no `[  FAILED  ]` line, and `run.py`
only looked for those — so a kill-by-crash read as "no failures" and scored as a
survival. That is precisely the shape of a harness that flatters the suite it
measures.

Fixed: `crashed(rc, output)` now treats "neither exited 0 nor named a failing
test" as a kill, on every lane including the CI ones, and a crash is never
discounted as pre-existing (the baseline ran clean). Re-scored: **M46 KILLED,
`test-core: crashed (rc=136)`**.

**Audited every earlier run for the same mistake** — `rc != 0` with an empty
failure list across all `loop/evidence/mutation-results*.json`: M46 is the only
occurrence. All previously reported scores stand.

**Attacked and clean:** magic, version, three truncation points, KV and tensor
count overflows, string length past EOF and overflowing, unknown array element
type with a huge count, tensor offset past EOF and at U64 max, dim overflow and
negative dims, nonexistent tensor type, zero alignment.

### Self-check

1. **Did I modify, skip or loosen any existing test?** No — one added.
2. **Did every mutant get reverted?** Yes; and the one manual apply/restore
   outside the harness was byte-restored from a saved copy and verified with
   `git status`.
3. **Did I watch every new test fail before it passed?** The committed test is a
   characterisation test, so this does not apply; it is labelled as such in the
   source and in #1312.
4. **Is every bug claim backed by a script I ran twice?** Yes — the probe ran
   2/2 from fresh processes before anything was written down.
5. **Did I fix any production code?** No. The only fix was to my own harness.
6. **Are all new tests wired into CI?** Yes — `test-core`.

### Next iteration

Untouched foci: `I4` (concurrency, cancellation, VRAM pressure) and `I7` (long
context & window boundaries). `I4` is the one with a documented history on this
box — #1044/#1045 was cross-request KV corruption under load — and the dispatch's
cancel-storm and eviction-under-pressure cases have no equivalent anywhere in
the suite.

---

## Iteration 6 — 2026-08-08 — focus: `I4` concurrency, cancellation, VRAM pressure — commit: `863df810`

**Mutation score: unchanged at 88.9 %** — a hunt, no mutants. Chosen because
this is the focus with history on this box (#1044/#1045 was cross-request KV
corruption under load) and because the dispatch's cancel-storm and
eviction-under-pressure cases have no equivalent anywhere in the suite.

**Bugs found:** **S1: 1 (#1314)** · S2: 0 · S3: 0 · S4: 0 · S5: 0

**Escape distribution (new):** E1: 1 — no test asserts batch invariance in any
form.

### #1314 — greedy output depends on who else is in the batch

`tools/mutation/cancel_storm.py` runs five survivor prompts alone, then
alongside 45 unrelated 512-token requests, and compares byte for byte.

```
solo, 5 runs each:                 distinct outputs
  List the first five prime numbers.        1
  Name three primary colours.               1
  What is the capital of France?            1
  Count from one to seven.                  1

under 45 concurrent requests:
  DIVERGED 'Name three primary colours.'
    alone: '... primary colors are red, blue, and yellow.'
    storm: '... primary colours are red, blue, and yellow.'
  DIVERGED 'List the first five prime numbers.'
    alone: '**2, 3, 5, 7, 11**\n\nA prime number is a natural number greater ...'
    storm: '**2, 3, 5, 7, 11**. ✅'
```

One greedy argmax flip (`colors` → `colours`) is enough to redirect the whole
continuation, which is what the second case shows.

### Four hypotheses, separated by experiment

| | result |
|---|---|
| the prompts are simply unstable | **no** — 5/5 identical solo runs each |
| cancellation corrupts survivors (#1044/#1045 class) | **no** — victims run to completion, same divergence |
| cancel → resubmit serves a half-evicted prefix | **no** — clean in every round |
| `IMP_DETERMINISTIC=1` prevents it | **no** — same two prompts, both rounds |

So cancellation is exonerated, the prefix cache is exonerated, and the remaining
variable is batch composition — which survives the strongest reproducibility
switch the engine has. Reproduced 2/2 with `loop/repro/BUG-5.sh`, 4/4 counting
the deterministic-mode rounds.

Distinct from #1299: these prompts are byte-stable back-to-back on one context
here, which is precisely what #1299 shows failing elsewhere. Same symptom class,
different trigger, different fix surface.

`docs/determinism.md` and #554 list the documented boundaries — dense greedy
logit ties, CUB top-k > 128, `typical_p` atomicAdd, GDN cross-context. Batch
invariance appears in neither the guarantees nor the exclusions.

**Tests added: 0.** The right oracle for this is logit-level, not text-level:
run a sequence alone, run it batched with unrelated padding sequences of
different lengths, assert the logits match within a stated tolerance and the
argmax is identical at every step. That belongs in `test-e2e` next to the
executor, needs no server, and localises the fault to a kernel instead of a text
diff. Writing it against a fault I have localised only to "batch composition"
would be guessing at the seam; #1314 carries the recipe.

**Attacked and clean:** cancel storm with 45 abortive clients (survivors
unaffected); cancel → immediate resubmit of the same prompt (prefix cache
serves correctly); 50 concurrent requests against a server whose max batch is
smaller (queueing does not corrupt state — the three non-diverging survivors are
byte-identical in every round).

**Not attacked:** deliberate VRAM exhaustion and the allocation-failure path.
`compute-sanitizer` is unusable on WSL2 (recorded previously), so racecheck /
initcheck /synccheck remain out of reach on this box; that is a standing
limitation of the environment, not a decision.

### Self-check

1. **Did I modify, skip or loosen any existing test?** No.
2. **Did every mutant get reverted?** No mutants this iteration.
3. **Did I watch every new test fail before it passed?** No tests added — and
   the reason is stated above rather than papered over.
4. **Is every bug claim backed by a script I ran twice?** Yes —
   `loop/repro/BUG-5.sh`, 2/2, plus two deterministic-mode rounds.
5. **Did I fix any production code?** No.
6. **Are all new tests wired into CI?** N/A.

### Next iteration

`I7` (long context & window boundaries) is the last untouched focus: behaviour
at and beyond the trained context length, RoPE scaling at the extremes, and the
sliding-window boundaries that `evict_middle_blocks` already showed are
delicate.

---

## Iteration 7 — 2026-08-08 — focus: `I7` long context & window boundaries — commit: `2d1d0500`

**Mutation score: unchanged at 88.9 %** — a hunt, no mutants.

**Bugs found:** S1: 0 · S2: 0 · S3: 0 · **S4: 1 (#1316)** · S5: 0

**Escape distribution (new):** E6: 1 — the RoPE tests use a proper CPU-reference
oracle and are only ever run at positions where the fault is invisible.

### #1316 — the whole long-context RoPE regime was unverified

The hypothesis came from reading, not from a sweep: `rope.cu:101` uses the fast
intrinsics `__cosf`/`__sinf`, whose argument reduction NVIDIA specifies as
accurate only for |x| < 48039 — and at `pair_idx == 0` the frequency is exactly
`1.0f`, so the angle *is* the position. This model family trains to 131072.

Measured against the same CPU reference the existing tests use (which computes
the angle identically in float, so the comparison isolates the transcendental
and not the angle's representation):

```
pos      40   3.0e-6        pos    8000   8.5e-4
pos     500   5.7e-5        pos   16000   1.6e-3
pos    1000   9.3e-5        pos   32768   8.9e-4
pos    2000   2.3e-4        pos  131071   1.0e-2
```

Every RoPE test in the file stops at position 40:

```
121:  pos_host = {0, 5}      247:  pos_host(batch*seq_len, 0)
192:  pos_host = {0, 5}      301:  pos_host = {3, 7}
373:  pos_host = {0, 1, 2, 3, 10, 20, 30, 40}
```

The sharp part is not the coverage gap but what it concealed: **the existing
tolerances are already incompatible with the existing kernel at ordinary context
lengths.** `RopeBasicFP32` holds itself to `1e-4`, exceeded from about position
2000; `RopePositionInvariance` holds itself to `1e-5`, exceeded from about
position 300. Neither has ever had to reconcile its tolerance with the kernel it
tests, because neither runs there.

**Filed S4, not S1, on purpose.** What is established is numerical drift, not a
wrong answer — no long-context perplexity measurement was made, and the issue
says so. Two facts argue against dismissing it: 1e-2 absolute on unit-scale
activations is well above FP16 noise, and it lands on the lowest-frequency
rotary pair, which is the one carrying long-range position information.

**Tests added: 1** — `RoPETest.LongContextPositionsMatchCpuReference`, sweeping
to 131071 with `kTol = 2e-2`. That bound is the **measured envelope with
headroom, not a specification**: it puts the range under test for the first time
and catches a regression, without asserting that 1e-2 is acceptable. The test
and its comment say this explicitly so nobody later reads the tolerance as a
blessing.

**Not measured:** the YaRN and LongRoPE branches. YaRN's `inv_scaling < 1`
shrinks the angle so it is probably less exposed — an expectation, not a
measurement, and labelled as such in #1316.

### Self-check

1. **Did I modify, skip or loosen any existing test?** No. In particular the two
   tolerances this iteration calls into question (`1e-4`, `1e-5`) were left
   exactly as they are — they are correct at the positions they run at, and
   changing them to accommodate a finding would be the precise inversion of the
   job.
2. **Did every mutant get reverted?** No mutants this iteration.
3. **Did I watch every new test fail before it passed?** No — and it would be
   dishonest to claim otherwise. This test bounds a measured envelope rather
   than targeting an injected fault; it was written from the sweep, and the
   sweep is quoted in the source so the numbers can be rechecked.
4. **Is every bug claim backed by a script I ran twice?** The sweep is a gtest,
   run repeatedly across two position sets while narrowing the crossing point.
5. **Did I fix any production code?** No.
6. **Are all new tests wired into CI?** No — `test-compute` is GPU. Standing
   finding, #1304.

### Rotation complete

All seven foci have now had a pass: `I1` API surface (it. 4), `I2` paging/KV
(it. 3), `I3` numerics & quantisation (covered by the iteration-1 mutant
categories, 100 % across rope/scaling/quantization/numerics), `I4` concurrency
(it. 6), `I5` ingestion (it. 5), `I6` sampling (it. 2), `I7` long context (this
one).

---

## Iteration 8 — 2026-08-08 — focus: `I1` (second pass) — commit: `a960aa7a`

**Mutation score: 90.0 %** (45/50) — prev 88.9 % (40/45). Five new mutants
(M47–M51), all killed after one new test.

| Category | score |
|---|---|
| **api** (new) | **5/5 = 100 %** |
| everything else | unchanged |

**Bugs found:** none.

**Escape distribution (new):** E6: 1 — the test targets the right property with
an input that cannot reach it.

### Attacking the API from the other side

Iteration 4 attacked the running server and found #1310. This pass attacked the
half of the server that CI *does* compile: `anthropic.cpp`, `tool_call.cpp`,
`responses.cpp`, `utils.cpp` and `constraint_validation.cpp` are all linked into
`test-core`, so a mutant there is measured against the merge gate rather than
against a local GPU run. That half had never been mutated.

Four of five died to tests that already existed:

| Mutant | Killed by |
|---|---|
| `Utf8Stitch` releases an incomplete tail instead of carrying it | `RejoinsCharacterSplitAcrossTwoPieces`, `ReassemblesFourByteCharacterOneByteAtATime` |
| Anthropic usage stops subtracting prefix-cache hits | `CacheReadAndCreationMapped`, `UsageSplitsCacheReadFromInput`, `UsageCachedClampedToPrompt` |
| `finish_reason=length` maps to `end_turn` | `FinishReasonMapping` |
| the `cached > prompt` clamp is dropped | `UsageCachedClampedToPrompt` |

So the Anthropic transform and the UTF-8 stitcher are genuinely defended, in the
lane that gates a merge. Worth stating plainly after seven iterations of
cataloguing what CI cannot see.

### The one gap: a guard whose test cannot reach it

M48 widens `Utf8Stitch::feed`'s `<= 3` bound, which exists so that genuinely
invalid input is passed through instead of parked forever. There *is* a test for
that property — `Utf8Stitch.DoesNotStallOrLoseBytesOnInvalidInput` — and it
survives the mutant.

The reason is in `utf8_complete_len`: it walks back from the end **over
continuation bytes** (0x80–0xBF) to the last lead byte, so the tail it parks is
`1 + trailing continuations`. The existing test feeds `"\xFF\xFE\xFD\xFC\xFB"`,
in which no byte is a continuation — the walk-back stops at the final byte, the
tail is **1**, and the `<= 3` bound never binds. The guard is never exercised.

Reaching it needs an invalid lead followed by at least three continuation bytes:
`"\xFF\x80\x80\x80"` parks at index 0, tail 4. Added as
`Utf8Stitch.InvalidLeadFollowedByContinuationsIsNotHeldBackForever`; verified to
fail against M48 and to pass on `main`. It runs in CI.

This is the same shape as #1300 and #1316: a correct test aimed at the right
property, written with an input under which the fault is invisible. Third
occurrence of that pattern in this campaign, and the cheapest class to fix.

**Tests added: 1, verified against the fault it targets: yes.**

### Self-check

1. **Did I modify, skip or loosen any existing test?** No.
2. **Did every mutant get reverted?** Yes — five mutants, tree checked after
   each, and again after the `recheck.sh` verification.
3. **Did I watch every new test fail before it passed?** Yes — M48.
4. **Is every bug claim backed by a script I ran twice?** No bugs claimed.
5. **Did I fix any production code?** No.
6. **Are all new tests wired into CI?** Yes — `test-core`.

### Next

Second pass continues with `I2`. The seams worth re-mining are the ones where
the first pass found the tests good but the *inputs* narrow — that pattern has
now produced three findings and is cheaper per finding than anything else this
campaign has done.

---

## Iteration 9 — 2026-08-08 — focus: constrained decoding — commit: `5a0e3027`

**Mutation score: 90.4 %** (47/52) — prev 90.0 % (45/50). Five mutants written
(M52–M56); two scored, both killed after one new test, two ruled equivalent,
one refused by the harness.

| Category | score |
|---|---|
| **constrain** (new) | **2/2 = 100 %** |
| everything else | unchanged |

**Bugs found:** none.

### The point of this iteration was to falsify my own claim

The iteration-1 assertion audit measured the constrained-decoding suites as
overwhelmingly A0 — `test_json_constrain.cu` 40/44, `test_gbnf_grammar.cpp`
22/22 — and I softened that in the report: *"for an acceptor, `EXPECT_TRUE
(accepts(s))` is the value under test, not a smoke check"*. That was an
argument, not a measurement. Mutating the FSM settles it.

**The claim holds, and for a sharper reason than I gave.** Two mutants re-open
#1096 in `compute_allowed_mask()` — `ARRAY_NEED_VALUE` admitting `]` again,
`OBJECT_NEED_KEY` admitting `}` — and nothing fails. That looks damning for
about ten minutes. It is not: `apply_mask()` uses the mask only as a
**pre-filter** and then runs `sim_token_valid()` on every candidate that passes
it (`json_constrain.cu:645-655`), and `advance_char()` enforces the
trailing-comma rule independently. So M52 and M53 are **equivalent mutants** —
the mask half of #1096 is defence in depth over the half that decides.

That makes three equivalent mutants found this campaign (M25, M52, M53), all
three the same shape: a redundant guard layered over a load-bearing one. Worth
naming as a pattern, because "surviving mutant ⇒ test gap" would have produced
three wrong issues.

### What was actually missing

| Mutant | Verdict |
|---|---|
| M54 — a number ending at whitespace no longer needs a digit (`"1. 1"`, #1104) | KILLED by `JsonConstrainFsm.NumberGrammarMatchesRfc8259` |
| M55 — GBNF `kMaxStackDepth` cap removed | **SURVIVED** — genuine gap |
| M56 — schema depth cap | refused: the anchor matched two sites, so the harness declined to mutate an arbitrary one |

`expand()` drops a continuation once it is 128 rule references deep, because a
self-referential grammar otherwise grows the work list without bound. **No test
in `test_gbnf_grammar.cpp` nested anything at all**, so removing the cap left
all 22 green while turning a recursive grammar into a hang.

**Tests added: 2, both verified against the fault they target.**

- `GbnfGrammarTest.DeepSelfRecursionStaysBounded` — `root ::= "[" root "]" | "x"`
  driven 512 deep. Kills M55.
- `JsonConstrainPropertyTest.RejectsTrailingCommas` — `[1,]`, `{"a":1,}` and
  four relatives, cross-checked against `nlohmann::json::accept`. It does **not**
  kill M52/M53, and its comment says so outright, with the measurement: the
  generator never produces a trailing comma, so the shape had no coverage, but
  the path it covers is the one that decides.

Both run in CI.

### Self-check

1. **Did I modify, skip or loosen any existing test?** No.
2. **Did every mutant get reverted?** Yes — and M56 never applied, because the
   harness refuses an ambiguous anchor rather than guessing.
3. **Did I watch every new test fail before it passed?** For M55, yes. For the
   trailing-comma test, no — and rather than dress that up, the test's own
   comment records that the mask mutants do not move it and why.
4. **Is every bug claim backed by a script I ran twice?** No bugs claimed.
5. **Did I fix any production code?** No.
6. **Are all new tests wired into CI?** Yes — both `test-core`.

---

## Iteration 10 — 2026-08-08 — closing out `controlflow` — commit: `d584c530`

**Mutation score: unchanged at 90.4 %** (47/52). No tests committed, and that is
the result rather than a shortfall.

### M30 is reachable, but not from where I attacked it

`controlflow` has been the one category below the 70 % floor for eight
iterations. M29 was resolved in #1309 by measurement — the perf gate sees it at
−36 %, twelve times its threshold. M30, the split-K scratch-capacity guard, was
the last open item in the campaign.

Built the test: an FP16 decode with a deliberately undersized *advertised*
scratch and a canary filling the slack past it, so that an overrun becomes an
assertion rather than a silent write into memory nothing reads. It passes with
the mutant applied.

The reason is that the guard has no caller on that path:

```
$ rg -n 'compute_splitk_splits' src/compute/*.cu
attention_paged_nvfp4.cu:377      attention_paged_int4.cu:535
attention_paged_nvfp4.cu:473      attention_paged_int8.cu:481
attention_paged_nvfp4_tc.cu:1080  attention_paged_fp8.cu:576
```

`paged_attention_decode` (FP16) makes its own split decision inline. The tests
were **reverted rather than committed**: a test whose comment claims to exercise
a guard it cannot reach is the same defect this campaign has filed four times
under E6, and shipping one would have been the worst possible way to close a
category.

### Two mutants, one seam

M31 (#1303, the sink term) and M30 both live in `attention_paged_common.cuh` and
are both reachable only from the cluster kernel and the five quantised-KV decode
launchers — and every test touching those passes `n_sinks=0` and a default-sized
scratch. Recorded as an addendum on #1303 with a recipe that closes both from
`tests/test_fp8_kv_cache.cu`, which already builds the FP8 KV cache, the CPU
reference, and split-K scratch.

The canary is the load-bearing part of that recipe and is worth keeping in mind:
`cudaMalloc` rounds up generously, so an overrun lands in slack nothing reads and
the output stays correct. Comparing outputs alone cannot see it — measured here,
not assumed.

### Where the campaign stands against its own stopping rule

> Two consecutive iterations find zero new S1/S2 **and** mutation score ≥ 85 %
> with no category below 70 %.

| condition | status |
|---|---|
| two consecutive iterations without a new S1/S2 | **met** — iterations 8, 9 and 10 |
| mutation score ≥ 85 % | **met** — 90.4 % (47/52) |
| no category below 70 % | **not met** — `controlflow` 0/2 |

`controlflow` cannot clear the bar without changing what the suite is for. Its
two mutants are a throughput regression the perf gate already catches by design
(#1309) and a memory-safety guard on a launcher family the suite does not
exercise (#1303). Reaching 70 % would mean either importing a perf assertion the
repo deliberately keeps outside the test suite, or writing the FP8 test — which
is worth doing and is now specified, but belongs to whoever owns that area
rather than to an audit that must not touch production code.

The substantive stopping condition is met. The letter of it is not, and the gap
is one named, specified test.

---

## Iteration 11 — 2026-08-09 — focus: speculative decoding — commit: `69411feb`

**Mutation score: unchanged at 90.4 %** — a hunt.

**Bugs found:** none. **S4/observability: 1 (#1321).**

### The invariant held; the first test that said so was worthless

The dispatch names it: *"With a fixed seed, spec-on vs spec-off must match."*
Nothing asserted it — `test_ngram_draft.cpp`, `test_suffix_draft.cpp` and
`test_token_recycle_draft.cpp` cover the draft **sources** in isolation, never
the end-to-end equality. The per-request toggle from #522 (`"speculative": bool`,
`handlers_chat_params.cpp:246`) makes the A/B a same-process comparison.

First attempt, five ordinary prompts, greedy, seed pinned:

```
List the first five prime numbers.   ok
Explain why the sky is blue.         ok
Write the word banana five times.    ok
Count from one to twelve.            ok
Name three primary colours.          ok
0 divergence(s)
```

The server log for those same requests:

```
[spec-ngram] verify_steps=0 miss_steps=155 drafted=0 accepted=0 (0.0%)
[spec-ngram] verify_steps=0 miss_steps=167 drafted=0 accepted=0 (0.0%)
```

**`drafted=0` on every one.** The n-gram matcher needs repetition in the
context; on those prompts it never fires, so the test compared the
non-speculative path against itself. Five green ticks proving nothing — the
exact shape this campaign has filed five times as E6, produced here by my own
hand.

With repetition-inducing prompts the drafter engages (`drafted=250
accepted=218`, 87.2 %) and the invariant **holds byte for byte**. Clean negative
result, and this time an earned one.

### #1321 — the guard that should exist cannot be written

`Engine::spec_stats_` is private with no accessor; `/metrics` has no spec
series. So a test's only vacuity guard is a proxy on the *fixture* — "is the
output repetitive enough that drafting was possible" — which is conservative and
has false negatives. A `count from 1 to 60` prompt drafts well (its token
pattern repeats) but has no repeated word n-gram at all, so a word-level guard
rejects a fixture that works. That prompt was dropped from the committed test
for exactly that reason, and the reason is written next to it: **a guard with
false negatives is worse than none.**

**Tests added: 1** — `tests/api/test_chat.py::TestSpeculativeDecoding`, two
repetitive prompts, asserting byte equality plus the fixture guard. It skips
under `IMP_USE_MOCK=1` with a reason naming #1302: the mock has no tokenizer and
no drafter, so there is nothing there for it to be right or wrong about.
Skipping visibly beats passing vacuously — and that skip is itself an argument
for #1302 rather than a way around it.

Verified in both lanes: mock 8 passed / 3 skipped (CI unchanged), real server
2 passed.

### Self-check

1. **Did I modify, skip or loosen any existing test?** No. The one skip added is
   a gate condition on a stand-in that cannot implement the feature, stated at
   the call site, not a silenced failure.
2. **Did every mutant get reverted?** No mutants this iteration.
3. **Did I watch every new test fail before it passed?** Yes, twice over — first
   against the mock (the fixture guard fired), then the counting-prompt case
   that exposed the guard's false negative.
4. **Is every bug claim backed by a script I ran twice?** No product bug
   claimed; the vacuity finding is backed by the server's own counters, quoted.
5. **Did I fix any production code?** No.
6. **Are all new tests wired into CI?** The test runs in CI and skips there, for
   a stated reason. Making it meaningful in CI is #1302; making it
   self-validating anywhere is #1321.
