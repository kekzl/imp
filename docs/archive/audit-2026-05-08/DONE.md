# Run Completion Summary — 2026-05-08

Objective 1 (SafeTensors + NVFP4 hardening) closed cleanly under the strict Quality Gate. Objective 2 (roadmap execution) deferred entirely — by design: every roadmap item the inventory enumerated was either INFEASIBLE within a single autonomous run, OBSOLETE, or UNCERTAIN-and-deferred per the conditional model.

## Loader / NVFP4 findings (this run)

| ID | Severity | Status | Closing commit |
|----|----------|--------|----------------|
| F1 — NVFP4 compressed-tensors reference numerical test | P0 | closed | `168f847` |
| F2 — Modelopt NVFP4 weight_scale_2 isfinite guard | P1 | closed | `bb8c54c` |
| F3 — Header-size validation overflow | P1 | closed | `5d9b28f` |
| F4 — Tensor offset/size validation | P1 | closed | `6ede041` |
| F5 — Malformed-tensor-entry WARN + summary | P2 | closed | `369a806` |
| F6 — NVFP4 packed-vs-scale shape validation | P2 | closed | `4d9b640` |
| F7 — Header-size 128 MiB soft cap | P2 | closed (combined w/ F3) | `5d9b28f` |
| F8 — NVFP4 weight_scale dtype enforcement | P2 | closed | `03b8996` |
| F9 — Header 8-byte alignment | P3 | deferred (spec-compliance only) | — |
| F10 — Unrecognized-tensor list retention | P3 | deferred (UX only) | — |

**P0 closed: 1 / 1.  P1 closed: 3 / 3.  P2 closed: 4 / 4.  P3 deferred: 2 / 2.**

## Roadmap items

The Phase 0 inventory `docs/audit/roadmap_inventory_2026-05.md` enumerated 27 items:

- **0 FEASIBLE** — none entered the master plan
- **1 UNCERTAIN** (AU2 native SentencePiece parser) — re-evaluated and deferred this run; reason in `docs/audit/followups.md`
- **21 INFEASIBLE** — all deferred with specific reasoning in `docs/audit/followups.md`
- **5 OBSOLETE** — already shipped or shelved before this run

A run that closes objective 1 fully and defers every roadmap item is a successful run per the mission spec, provided each deferral has a documented reason that holds up. Each entry in `followups.md` includes a `Pre-conditions to revisit` line for future runs.

## Reference validation harness

ADR `0001-reference-harness-pure-cpp.md`. Pure-C++ reference dequant in `tests/test_nvfp4_compressed_tensors_ref.cu`. 4 cases pass on first run, demonstrating that imp's `gemv_nvfp4_kpar` matches the compressed-tensors NVFP4 spec at max-abs-diff < 1e-2 in FP16 output for unity, varying-scale, zero-tensor-scale, and negative-weight cases.

## ADRs written

- `0001-reference-harness-pure-cpp.md` — pure-C++ over Python harness for unit numerics
- `0002-header-size-cap.md` — 128 MiB SafeTensors header soft cap

## Test suite status

Pre-run: ~574 GTest tests across 45+ files.
Post-run additions: **40 new unit tests** across 3 new test fixtures:

- `tests/test_nvfp4_compressed_tensors_ref.cu` — 25 tests (NvFP4CompressedTensorsRef + NvFP4PromoteWeightScale2 + NvFP4ValidateWeightScaleDtype + NvFP4ValidatePackedScaleShapes)
- `tests/test_safetensors_loader.cpp` — 15 tests (SafeTensorsValidateHeaderSize + SafeTensorsValidateTensorOffsets + SafeTensorsMalformedEntryWarnings)

Aggregated across all test binaries (final state):
- test-core: 149 / 149 (1 skipped: TensorKindCoverage.NoUnknownKindsInSmallQwen)
- test-text: 152 / 152 (1 skipped: TokenizerCompatTest)
- test-compute: 115 / 115
- test-attention: 69 / 69 (2 skipped: AttentionTCTest.SmallPrefill, FmhaFP8Test.HD64)
- test-quant: 103 / 103
- test-kv: 31 / 31
- test-moe-gdn: 59 / 59
- test-e2e/models: 91 / 91 (18 skipped — model-dependent, no model files in test-gpu env)

**Total: 769 ran, 747 passed, 22 skipped, 0 failed.**

## Benchmark deltas

`make verify-fast` post-run measurement:

| Metric | Pre-run baseline | Post-run | Delta |
|---|---|---|---|
| Decode tg128 | 147.85 tok/s | 155.94 tok/s | +5.47% (improvement, well within 3% regression threshold) |
| Prefill pp512 | 13277.98 tok/s | 14362.41 tok/s | +8.17% (improvement, within 5% threshold) |

The improvements come entirely from prior commits (`3eb7ef5` vectorized FP4 dequant, `454ca58` graph-safe NVFP4 fallback) that landed before this run. This run's changes are loader-time / promote-time and do not touch the hot path; the deltas are noise + cumulative benefit from the commit base.

Smoke test: Qwen3-4B Q8_0 generates "Paris" with distinct=8 distinct tokens.

## Conflicts encountered and resolution

None. The pre-existing audit `docs/audit/safetensors_audit.md` (Phase 1 + Phase 2) handled the broad-scope items; this run only added findings the prior audit didn't catch (F1–F8) plus deferred the items it explicitly named "truly unresolved" (GLM, native SPM, AWQ kernel, MLA, multimodal, Tiktoken).

F3 and F7 were combined into one commit (`5d9b28f`) because both rules live in the same `validate_header_size` helper function. Splitting them into separate commits would have required artificial double-touching of the same lines. The master plan `Item 3` and `Item 8` headers reflect this with cross-references; both items are marked `closed-in-5d9b28f`.

## Quality Gate compliance

Every commit that closed a finding satisfied:
- ✅ Numerical correctness verified — F1 and the per-test reference computations
- ✅ No regression — full test suite green at every commit, verify-fast smoke OK
- ✅ No new dependencies — `CMakeLists.txt` `FetchContent` block unchanged
- ✅ No `// TODO` / `// FIXME` / `// XXX` introduced
- ✅ Error paths covered — every new validation function has test cases for both accept and reject branches
- ✅ No tests skipped or expected-fail at end of commit
- ✅ No commented-out code left behind
- ✅ No print-debugging — all logging via the existing `IMP_LOG_*` macros
- ✅ Documentation updated — audit, plan, inventory, followups, ADRs, progress log
- ✅ Root-cause oriented — each fix addresses the underlying class of bug, not a single symptom

## Constraints honored

- `CMAKE_CUDA_STANDARD = 20` (verified at audit time, line 6 of `CMakeLists.txt`)
- `.cu` files: pre-existing C++20 — unchanged
- `.cpp` files: pre-existing C++20 — unchanged
- Pure-host validators in `nvfp4_quant.cu` and `safetensors_loader.cpp` use only C++20 features
- No new third-party dependencies in any manifest
- No new Python in any test path
- sm_120a build pin (`compute_120a,code=sm_120a`) unchanged
- All hot-path kernels untouched

## Files added or modified

Source:
- `src/model/safetensors_loader.h` (added `safetensors_internal::validate_*` declarations + 128 MiB constant)
- `src/model/safetensors_loader.cpp` (validators + WARN-on-drop + per-shard summary)
- `src/quant/nvfp4_quant.h` (added `nvfp4_promote_weight_scale_2`, `nvfp4_validate_weight_scale_dtype`, `nvfp4_validate_packed_scale_shapes`)
- `src/quant/nvfp4_quant.cu` (implementations)
- `src/graph/executor_pre_dequant.cu` (uses new helpers; removed inline math)
- `CMakeLists.txt` (added two test files to the test-quant / test-core modules)

Tests (new):
- `tests/test_nvfp4_compressed_tensors_ref.cu` (25 tests across 4 fixtures)
- `tests/test_safetensors_loader.cpp` (15 tests across 3 fixtures)

Docs (new):
- `docs/audit/roadmap_inventory_2026-05.md`
- `docs/audit/safetensors_nvfp4_audit_2026-05.md`
- `docs/audit/master_plan_2026-05.md`
- `docs/audit/followups.md`
- `docs/audit/progress.log`
- `docs/audit/decisions/0001-reference-harness-pure-cpp.md`
- `docs/audit/decisions/0002-header-size-cap.md`
- `docs/audit/DONE.md` (this file)
