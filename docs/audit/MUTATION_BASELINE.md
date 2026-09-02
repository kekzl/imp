# Mutation-testing baseline

Date: 2026-08-08 · Commit `ad067d76` (v0.23.0, `main`) · RTX 5090 / WSL2 / CUDA 13.3.1
Harness: `tools/mutation/run.py` · Catalogue: `tools/mutation/catalogue.json` ·
Patches: `loop/mutants/M*.patch` · Raw results: `loop/evidence/mutation-results.json`

42 mutants across 10 categories (34 in iteration 1, eight paging/KV additions
in iteration 3), each a semantically meaningful fault from a real bug class in
LLM inference — not arithmetic noise. Every mutant: inject →
incremental rebuild → run the suite → revert → verify `git status --porcelain`
is empty. The tree was clean after all 34.

---

## Headline

| Metric | Value |
|---|---:|
| **Baseline mutation score, full local suite** | **25 / 33 = 75.8 %** |
| **After the iteration-2 tests** | **28 / 33 = 84.8 %** |
| **After iteration 3 (8 paging/KV mutants added)** | **36 / 41 = 87.8 %** |
| **Mutation score, GitHub CI (`ctest -L unit`)** | **1 / 33 = 3.0 %** (baseline) |
| Equivalent mutants (excluded from the denominator) | 1 — M25 |
| Build failures (broken mutants) | 0 |
| Timeouts | 0 |

The two numbers answer different questions. The first is "does a test for this
exist anywhere in the repo". The second is "would the merge gate have stopped
it". Only `M24` — the prefix-cache block hash no longer chaining its parent —
was caught by CI, because `KVCacheManagerTest.BlockHashChaining` happens not to
need a CUDA device.

**33 of 34 injected faults reach `main` green.** Among them: a causal mask off
by one, a dropped `1/sqrt(d)`, RoPE applied to Q but not K, a swapped Q4_K
nibble order, and a removed `__syncthreads()`.

This is a property of the gate topology (no GPU runner, by owner decision on
2026-08-03), not of the tests: 24 of those 33 *are* caught, seconds later, by a
binary that only a human on a 5090 ever runs.

## Per category

| Category | Killed | Score | |
|---|---|---:|---|
| rope | 4/4 | 100 % | CPU-reference oracle at 1e-4, both Q and K |
| scaling | 4/4 | 100 % | |
| quantization | 3/3 | 100 % | fp64-reference dequant tests |
| memory | 2/2 | 100 % | dropped `__syncthreads()` both caught |
| numerics | 2/2 | 100 % | |
| masking | 5/6 | 83 % | |
| indexing | 4/5 | 80 % | |
| **kvcache** | **1/2** | **50 %** | M25 is equivalent, see below |
| **sampling** | **0/3** | **0 %** | |
| **controlflow** | **0/2** | **0 %** | |

After iterations 2–3 (`docs/audit/TEST_HARDENING_LOG.md`): sampling 2/3 = 67 %
(M20, M21 killed by the new `TopPTruncates*` tests), **kvcache 8/8 = 100 %**
(M23 by `BlockHashDiscriminatesEveryTokenPosition`, which runs in CI; M35 by
`ContentSaltSeparatesIdenticalTokenPrefixes`; M36–M42 by tests that already
existed), masking 6/7, indexing 5/6.

The eight iteration-3 mutants (M35–M42) targeted the KV manager, the prefix hash
chain and StreamingLLM eviction. Seven were killed by tests that were already
there — including both halves of the #963 boundary-block fix — which is the
strongest single piece of evidence in this campaign that the KV subsystem's
*tests* are good and its problem is only that CI cannot run them.

Attention numerics, RoPE and dequantisation are genuinely well covered — the
kills come from real oracle tests (a CPU reference, an FP16 reference, an fp64
reference), not from smoke checks. The holes are the sampler, the KV-cache
hash lifecycle, and anything whose only symptom is performance.

## Survivors

Each is a confirmed test gap. Escape classes and the named test that should
have caught each are in `docs/audit/ESCAPE_ANALYSIS.md`.

| ID | Category | Fault injected | Why nothing saw it |
|---|---|---|---|
| M20 | sampling | Multiblock sampler ignores `top_p` | No test passes a `top_p` that truncates a non-negligible tail |
| M21 | sampling | CUB sampler path ignores `top_p` | Same; the CUB test puts ~all mass on one token |
| M22 | sampling | `temperature<=0` no longer routes to greedy in the batched decode path | Only oracle is `DetEvalE2ETest`, which is red on `main` (BUG-1) |
| M23 | kvcache | Block hash skips the **last** token of a block | `BlockHashDeterministic` varies `tokens[0]`, never the last |
| M10 | indexing | `block_token_range` reads one slot past `ctx_len` | Same oracle problem as M22 |
| M29 | controlflow | Split-K never used (slow path always) | Correctness-neutral; no perf test in the suite sees it |
| M30 | controlflow | Split-K scratch-capacity guard removed | No test builds a shape that overflows the partial buffer |
| M31 | masking | Attention sink dropped from the softmax denominator | The only sink tests are shaped so **split-K** fires; the mutated reduction is the non-split one |

### M25 is an equivalent mutant, not a survivor

Reported in the first pass as a test gap. It is not. Stubbing
`drop_stale_hash_if_last` out does not change observable behaviour, because the
lookup site rejects the stale entry independently
(`src/memory/kv_cache_manager.cpp:509-517`) and treats it as a miss. Measured:
with the function stubbed, that path's `IMP_LOG_WARN` fires exactly once across
`PrefixEquivTest.*` and all 11 tests still pass; on the clean tree it never
fires. The eager cleanup is defence in depth.

Equivalent mutants belong out of the denominator, which is why the headline is
25/33 rather than 25/34. What *was* true is narrower and is now covered:
nothing drove the rollback path at all, so
`PrefixEquivTest.RollbackOfPartialAllocationDropsItsHashes` was added.

### M10 and M22 are indeterminate, not clean survivals

Both showed failures when re-run against `DetEvalE2ETest`. That oracle is
**itself red on the clean tree** (`loop/repro/BUG-1.sh`, reproduced 2/2 from
fresh processes, and 5/5 in a repeat loop), so a red run under the mutant proves
nothing. They are scored as survivors here because that is the conservative
direction: counting an unreliable red as a kill would inflate the score.

This is also why the harness runs the CI lane in full before short-circuiting,
and why `run.py` discounts every failure that was already present in the
baseline — without that, all 38 capacity-induced `test-e2e` failures would have
"killed" every mutant.

## 2026-09-02 recheck: four anchors had gone dead, and the CI lane measured

Two questions, run against `339ce7c7` with the harness's two new modes.

**`--verify-anchors` (no injection, checks every `find` still matches once):
4 of 56 had drifted**, so those mutants had silently stopped testing anything
and were nevertheless in the denominator of the 87.8% above.

| Mutant | Why it died |
|---|---|
| M22 | its site moved with the `executor_sampling.cu` split |
| M50 | the `stop_reason` mapping became a function returning a literal |
| M54 | the number-completeness check moved into `schema_constrain.cu` |
| M56 | a third depth-cap site appeared, so the anchor matched twice |

All four repaired. M50, injectable again, is killed immediately by
`AnthropicResponse.FinishReasonMapping` - a test that existed the whole time
and had nothing to catch.

**`--ci-only` (the three lanes `ctest -L unit` runs, no GPU) over the 21
host-side mutants: 15 killed, 5 survived = 75.0%.** Every survivor is in
`src/memory/kv_cache_manager.cpp`:

| Mutant | What passes the merge gate |
|---|---|
| M25 | stale `block->hash` entries never dropped (equivalent mutant, see above) |
| M35 | `content_salt` ignored: two prompts with identical token ids but different images share KV |
| M36 | prefix reuse no longer contiguous: uncomputed KV inside a "skipped" range |
| M38 | the probe reports a cached prefix one block longer than the chain reaches |
| M40 | reclaiming a cached block leaves its prefix-hash entry pointing at a free block |

They are not missing tests: `test-kv` catches all five. They are unreachable
from CI for one mechanical reason - every one needs `KVCacheManager` state, and
`KVCache`'s constructor allocates VRAM and throws without a device, so each of
those tests opens with `SKIP_IF_NO_CUDA()`. The hash function itself is static
and CPU-testable, which is exactly why M23 and M24 ARE caught in CI.

**Closed the same day.** `KVCache::for_accounting()` builds the id space, free
list, ref counts and geometry with no pool (`BlockPool::open_slots` was already
the mode where "the caller owns the memory"), and every data pointer aborts on
such a cache. `tests/test_kv_accounting.cpp` adds five tests to `test-core`,
one per fault:

| Mutant | Killed by |
|---|---|
| M35 | `KVAccounting.SaltSeedsTheChainOnLookupToo` |
| M36 | `KVAccounting.ReuseStopsAtTheFirstMiss` |
| M38 | `KVAccounting.ProbeCountsTheCachedChainExactly` |
| M40 | `KVAccounting.ReclaimDropsTheHashEntryWithTheBlock` |

**CI-lane score on the 21 host-side mutants: 15/20 -> 20/21 = 95.2%.** The only
survivor left is M25, the equivalent mutant. `ctest -L unit` went 1624 -> 1629
macros; the unlaned pin (1054) is unchanged.

The same day the rest of the class followed: `MakeManager()` in
`test_kv_cache.cpp` builds an accounting cache, so 43 more tests execute in the
CI lane instead of skipping. **test-core without a GPU: 62 skipped -> 19**, and
the surviving skips are exactly the tests that read or write KV bytes, drive the
SWA groups or persist the cache. Four kvcache mutants are now caught by the
tests written for them rather than by a neighbour: M37 and M39 by
`PrefixCachingWithPartialLastBlock`, M41 and M42 by
`EvictMiddleBlocksKeepsSinksAndWindow`.

Note for the next reader: `check_test_lanes.py` counts MACROS per binary, not
executions, so a test that skips at runtime still reads as "in a CI lane". The
skip count above is the honest figure for what the merge gate exercises, and
it is gated now: the three lane commands write GTest JSON reports and
`guard_unit_skips` (inside `ctest -L unit`) compares the skipped set with
`tools/unit_lane_skips.txt` by name. Five scheduler admission tests (accounting
cache) and the four `MLAConfig` config tests (embedded DeepSeek-V2-Lite
`config.json`) left the set the same day, then the per-layer/SWA accounting
form took the SWA id-space, the SWA trailing-free table and the per-layer NVFP4
scale geometry out of it (2 pointer-only GPU tests remain): **without a GPU
the lane skips test-core 14 of 1327, test-text 5 of 242, the e2e subset 1 of
62**, every one a line with its reason.

## Method notes that changed a result

* **Cheap-first, full-before-survival.** A mutant is KILLED as soon as any lane
  fails, but SURVIVED only after all eight binaries have run. A kill from a
  cheap lane is sound; a survival from one is not.
* **Timeouts are not kills.** None occurred, but the harness records them
  separately: a hung suite is not an assertion.
* **Revert is never `git checkout`.** Original bytes are captured before the
  edit and written back in a `finally`; `git status --porcelain` is checked
  after every mutant and the run aborts if it is dirty.
* **Anchors must match exactly once.** Three catalogue entries initially matched
  2–4 sites (`const int* bt = block_tables + ...` appears six times in
  `attention_paged.cu`); the harness refuses those rather than mutating an
  arbitrary one.
* **M23 first reported ERROR**, not a survival: a smoke-test artefact left
  `loop/mutants/M23.patch` root-owned and the harness could not overwrite it.
  Re-run after fixing ownership → SURVIVED.

## Reproducing

```
tools/mutation/run.py --list
tools/mutation/run.py --catalogue tools/mutation/catalogue.json
tools/mutation/run.py --only M20,M21                      # one or a few
tools/mutation/run.py --verify-anchors                    # do all 56 still match their source
tools/mutation/run.py --ci-only --only M23,M35            # what `ctest -L unit` would catch
tools/mutation/recheck.sh M25 test-e2e 'PrefixCacheE2ETest.*' 3   # isolated oracle, N runs
```

`run.py` refuses to start if the production tree is dirty.
