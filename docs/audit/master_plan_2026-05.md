# Master Plan — 2026-05-08

Combines the Phase 1 audit findings (`docs/audit/safetensors_nvfp4_audit_2026-05.md`) with the Phase 0 roadmap inventory (`docs/audit/roadmap_inventory_2026-05.md`).

The roadmap inventory yielded 0 FEASIBLE items and 1 UNCERTAIN (deferred). The master plan therefore consists entirely of loader/NVFP4 hardening findings F1–F8.

## Reference-validation harness decision

**Choice: Option 3 (pure C++ reference).**

ADR `0001-reference-harness-pure-cpp.md` records the rationale. Briefly:
- Option 1 (existing imp Python infra): no `transformers` / `vllm` venv is wired into imp's tests today; only `tests/api/` uses Python and that's HTTP-level testing, not numerics.
- Option 2 (subprocess into the user's `~/.cache/huggingface` venv): would require detection logic, dependency-availability handling at test time, and a cross-process fixture. The mission says "use what imp has"; imp doesn't have a Python harness for unit-level numerics.
- Option 3 (pure C++ reference): mechanical, dependency-free, deterministic, exact for the dequant formula. Trade-off is "catches dequant bugs but not full end-to-end logit drift" — but logit drift is what the existing scripts/validate_safetensors.py harness already handles at integration level (Phase-4 battery, etc.). Unit-level numerics is the gap, and a pure-C++ reference fills it precisely.

Tolerance: max-abs-diff < 1e-2 in FP16 (FMA-order divergence between sequential reference and imp's parallel-warp accumulator dominates). 1e-5 (the option-3 default in the mission spec) would only apply to a per-element bit-exact check, which is feasible only for representable values; the broader GEMV correctness needs the looser tolerance.

## Plan items

Ordered: P0 first, then P1, then P2. Within each tier, dependency order: harness/reference comes first because every subsequent test re-uses it.

### Item 1 — F1: compressed-tensors NVFP4 reference test (P0) — `closed-in-168f847`

**Change:** add `tests/test_nvfp4_compressed_tensors_ref.cu`. Builds a synthetic compressed-tensors weight in memory exactly per the spec (`weight_packed` uint8 nibble-packed FP4, `weight_scale` FP8 E4M3 group_size=16, `weight_scale_2` FP32 scalar), constructs a Tensor view with `qtype=NVFP4`, and routes it through `gemv_nvfp4_kpar`. A pure-host reference dequant computes the expected output `Y = X · W^T` from the spec formula `val = fp4 * fp8_e4m3_to_fp32(weight_scale) * weight_scale_2`. Verifies max-abs-diff < 1e-2.

**Files touched:** `tests/test_nvfp4_compressed_tensors_ref.cu` (new), `tests/CMakeLists.txt` (add to test list).

**Risk/blast radius:** test-only, no source changes. Worst case: test fails on first run → I have evidence that imp's NVFP4 dequant disagrees with the spec. Best case: test passes → spec compliance is now CI-enforced and any future regression breaks the test.

**Quality Gate:** numerical correctness ✓, no regression (test-only) ✓, no new deps ✓, no TODO ✓, error path tested via separate sub-cases (NaN scale, zero scale) ✓, no skip ✓, doc unchanged (test, not API) ✓, root-cause oriented (the test IS the root-cause-prevention mechanism) ✓.

### Item 2 — F2: Modelopt NVFP4 weight_scale_2 isfinite guard (P1) — `closed-in-bb8c54c`

**Change:** in `executor_pre_dequant.cu:262-279`, extend the existing zero/non-finite guard to also cover the Modelopt branch (`promoted_scale = h_scale;` at line 277-278 today). New behavior: if `!std::isfinite(h_scale)` for either format, set `promoted_scale = 0.0f` and bump a counter; if `h_scale == 0.0f` for Modelopt (currently silent), log INFO that the layer's weights are intentionally null. Update the end-of-loop summary to surface both Modelopt and llm-compressor counts.

**Files touched:** `src/graph/executor_pre_dequant.cu`, `tests/test_nvfp4_compressed_tensors_ref.cu` (extend with NaN/Inf cases — depends on Item 1's fixture infrastructure).

**Risk/blast radius:** isolated to the promote function. Existing llm-compressor path is unchanged. New guard is a strict superset of the old one — same behavior for finite inputs, defensive zero for non-finite.

**Quality Gate:** unit test exercising both NaN and +Inf weight_scale_2 in both Modelopt and llm-compressor paths; the test asserts the layer produces zero output (not NaN/Inf) and the WARN counter increments. No new deps. No TODO.

### Item 3 — F3 + F7: header-size overflow-safe validation + 128 MiB cap (P1+P2) — `closed-in-5d9b28f`

**Change:** `src/model/safetensors_loader.cpp:519-524` — replace `8 + header_size > file_size` with `header_size > file_size - 8`. Uses pre-existing invariant `file_size >= 8` from line 495.

**Files touched:** `src/model/safetensors_loader.cpp`, `tests/test_safetensors_loader.cpp` (new).

**Risk/blast radius:** affects only the malformed-file rejection path. Real-world SafeTensors files have header_size < a few MB and file_size > 1 GB, so the overflow case is unreachable in practice — but a hostile/corrupted file could trip it.

**Quality Gate:** new unit test writes a 16-byte file with `header_size = UINT64_MAX` and asserts `load_safetensors` returns false without crashing. Companion test with `header_size = file_size - 7` (off-by-one) asserts rejection.

### Item 4 — F4: tensor offset/size validation (P1) — `closed-in-6ede041`

**Change:** `src/model/safetensors_loader.cpp:572-580` — add three checks:
1. `offset_start <= offset_end` (else WARN + skip tensor)
2. `tensor_data_offset + offset_start <= file_size` (else WARN + skip)
3. `offset_end - offset_start == expected_nbytes(shape, dtype)` where `expected_nbytes` multiplies the shape with `dtype_bytes(QType)` (else WARN + skip)

Add `dtype_bytes` helper (private to the .cpp, mapping `QType::F16 → 2`, `QType::F32 → 4`, `QType::FP8_E4M3 → 1`, `QType::INT8 → 1`, `QType::INT32 → 4`, `QType::BF16 → 2`).

**Files touched:** `src/model/safetensors_loader.cpp`, `tests/test_safetensors_loader.cpp`.

**Risk/blast radius:** adds rejections that previously slipped through silently. Existing valid checkpoints emit `offset_end - offset_start == nelem * sizeof(dtype)` per spec — they will continue to load.

**Quality Gate:** unit tests for each of the three sub-cases (start>end, oob start, size mismatch). All produce a WARN and skip the tensor without crashing.

### Item 5 — F8: NVFP4 weight_scale dtype enforcement (P2) — `closed-in-03b8996`

**Change:** `src/graph/executor_pre_dequant.cu:237-287` — in `promote()`, before applying the formula, verify `sc.weight_scale.qtype == QType::FP8_E4M3`. If not, WARN naming the key and the actual qtype, and return false (skip promotion — weight stays in its loaded state for the dequant→cuBLAS fallback path to handle).

**Files touched:** `src/graph/executor_pre_dequant.cu`, `tests/test_nvfp4_compressed_tensors_ref.cu` (extend).

**Risk/blast radius:** isolated check. Real Modelopt + llm-compressor checkpoints use FP8 E4M3 for `weight_scale`; the only way this triggers is on a NVFP4↔MXFP4 cross-misrouted file.

**Quality Gate:** unit test that supplies a `weight_scale` with `qtype = QType::INT8` (UE8M0 bytes) and verifies `promote` rejects it.

### Item 6 — F6: NVFP4 packed shape vs scale shape validation (P2) — `closed-in-4d9b640`

**Change:** in the same `promote()` function, add `sc.weight_scale.shape[1] * 16 == w.shape[1] * 2` (where `w.shape[1]` is the packed-half K, so logical K = `w.shape[1] * 2`). On mismatch, WARN + skip promotion.

**Files touched:** `src/graph/executor_pre_dequant.cu`, `tests/test_nvfp4_compressed_tensors_ref.cu`.

**Risk/blast radius:** adds a rejection. Real checkpoints always satisfy this; the check is defense against group_size != 16 or a transposed `weight_scale`.

**Quality Gate:** unit test feeding mismatched shapes; verifies skip + WARN.

### Item 7 — F5: malformed tensor entry warnings (P2) — `closed-in-369a806`

**Change:** `src/model/safetensors_loader.cpp:554-580` — replace each silent `continue` with `IMP_LOG_WARN("safetensors: dropping tensor '%s' — <reason>")` and an end-of-shard summary count. Reasons: missing dtype, missing shape, ndim>kMaxDims, missing data_offsets.

**Files touched:** `src/model/safetensors_loader.cpp`, `tests/test_safetensors_loader.cpp`.

**Risk/blast radius:** behavior-equivalent (still skips). Adds log lines; no functional change for valid files.

**Quality Gate:** unit test crafting a SafeTensors blob where one tensor has malformed shape and asserting (a) the load returns success with the other tensors present, (b) the malformed tensor name appears in the WARN log via gtest's stderr capture.

### Item 8 — F7: header-size upper bound (P2) — `closed-in-5d9b28f` (combined with F3, see Item 3)

**Change:** `src/model/safetensors_loader.cpp:519` — soft-cap header_size at 128 MiB (a 128 MiB header would represent a >50 M-tensor checkpoint; real models are O(1000) tensors). Above the cap: ERROR + return false.

**Files touched:** `src/model/safetensors_loader.cpp`, `tests/test_safetensors_loader.cpp`.

**Risk/blast radius:** rejects pathological files. Real models have headers <100 KiB.

**Quality Gate:** unit test constructing a file with `header_size = 200 MiB` and verifying rejection. Recorded in ADR `0002-header-size-cap.md`.

## Global ordering

1. **Item 1** (F1, P0) — reference test; depends on nothing
2. **Item 2** (F2, P1) — Modelopt isfinite; reuses Item 1 fixture
3. **Item 3** (F3, P1) — header overflow; new test file
4. **Item 4** (F4, P1) — offset validation; same test file as Item 3
5. **Item 5** (F8, P2) — weight_scale dtype check; reuses Item 1 fixture
6. **Item 6** (F6, P2) — packed shape check; reuses Item 1 fixture
7. **Item 7** (F5, P2) — malformed-tensor warnings; reuses Item 3 test infrastructure
8. **Item 8** (F7, P2) — header upper bound; reuses Item 3 test infrastructure

This ordering means:
- the synthetic NVFP4 fixture in `tests/test_nvfp4_compressed_tensors_ref.cu` is built once (Item 1) and reused by Items 2/5/6
- the synthetic SafeTensors blob fixture in `tests/test_safetensors_loader.cpp` is built once (Item 3) and reused by Items 4/7/8

No conflicts with existing roadmap items (none are FEASIBLE this run).

## Per-item Quality-Gate feasibility statement

For each item: every Quality Gate bullet can be met because the change is small, isolated, has a deterministic synthetic test fixture, doesn't require GPU model files at runtime (the existing 322-test suite on the audit branch is the regression base, and these new tests slot into the same `IMP_BUILD_TESTS=ON` GTest harness), needs no new dependencies, and has clear root-cause-vs-symptom delineation (each item fixes a specific silent-correctness path).

## Benchmark baseline

`tests/perf_baseline.json` is unchanged through Phase 2 (PR #116). All eight items are loader-time / promote-time changes that do not touch the hot path. Decode tok/s is observed only via `make verify-fast` smoke (Qwen3-4B Q8_0 capital-of-France) per the existing pre-push hook. No compute kernel modifications planned.
