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
