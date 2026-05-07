# SafeTensors + NVFP4 Hardening Audit (2026-05-08)

**Scope:** read-only audit on top of the existing `docs/audit/safetensors_audit.md` (Phase 1 + Phase 2, PR #116). Focus: gaps that Phase 2 left open or that surfaced after the cv-narrow shipping in PR #125. Each finding cites `file:line`, with severity (P0–P3), one-paragraph explanation, and a regression-test target.

This audit complements rather than replaces the prior one. The prior audit's "Items that remain truly unresolved" section enumerates P2-class deferred work (GLM, native SentencePiece, AWQ kernel, DeepSeek MLA, multimodal); those are tracked in `docs/audit/followups.md`. The findings below are smaller hardening items that *can* land at full Quality Gate inside this run.

## Audit method

- Re-read every file Phase 1 named, plus the post-Phase-2 commits (`454ca58`, `3eb7ef5`, `b6c2b9c`).
- Re-read every NVFP4 test fixture (`tests/test_nvfp4_*.cu`).
- Inspected real on-disk checkpoints (`/home/kekz/models/Gemma-4-26B-A4B-it-NVFP4/`, `/home/kekz/models/Qwen3-30B-A3B-NVFP4-Modelopt/`) for `weight_packed` / `weight_scale` / `weight_scale_2` shape and dtype.
- Walked the dequant code path end-to-end: `safetensors_loader.cpp` → `weight_map.cpp` → `weight_upload.cu` → `executor_pre_dequant.cu` (Phase 0 promote) → `nvfp4_gemm.cu` (`gemv_nvfp4_row` formula).

## Findings

### F1 — No reference-numerical test for compressed-tensors NVFP4 dequant — **P0**

**Where:** `tests/test_nvfp4_*.cu` — every existing test either uses imp's own `quantize_fp16_to_nvfp4` and round-trips through the same kernel pair (`test_nvfp4_quant_ref.cu`, `test_nvfp4_quant_hw.cu`, `test_nvfp4_gemv_kpar_loop.cu`), or is a no-op crash-check (`test_nvfp4_quant.cu`).

**Why this is P0:** The mission's guarantee is "no silent correctness bugs". The compressed-tensors NVFP4 spec is explicit about three-component dequant `val = fp4 * fp8_e4m3_to_fp32(weight_scale_block) * weight_scale_2_fp32`. Roundtrip-only tests cannot detect a bug where imp's quantizer and dequantizer have a paired sign-flip or nibble-swap convention that disagrees with the on-disk format produced by Modelopt / llm-compressor. They can also not detect a missing factor — e.g. dropping `weight_scale_2` would still pass the existing roundtrip, but produce 1×–448× wrong output on real checkpoints.

**Regression target:** unit test in `tests/test_nvfp4_compressed_tensors_ref.cu` that:
1. Constructs a synthetic `weight_packed` (uint8 nibble-packed FP4 values), `weight_scale` (FP8 E4M3, group_size=16), `weight_scale_2` (FP32 scalar) on the host.
2. Computes the expected output `Y = X · W^T` using a pure-host reference dequant strictly from the spec.
3. Routes the same buffers through imp's `gemv_nvfp4_kpar` GEMV path.
4. Verifies max-abs-diff < 1e-2 (HW-cvt FP4 path is bit-exact for representable values; FMA-order is the only divergence source).

Tolerance from the mission's harness rules ("Option 3: max abs diff < 1e-5") is too tight for FP16 GEMM accumulation order; 1e-2 is realistic for K=128 FP16 dot-product.

### F2 — Modelopt NVFP4 `weight_scale_2` not isfinite-checked — **P1**

**Where:** `src/graph/executor_pre_dequant.cu:262-279`.

PR #113 added a defensive guard for the **llm-compressor** path: if `h_scale == 0` or `1.0f / h_scale` is non-finite, the promoted scale is set to 0.0 (zeroing the layer's contribution rather than producing NaN/Inf). The **Modelopt** branch at the `else` clause (`promoted_scale = h_scale;`) has no such guard. A malformed Modelopt checkpoint with `weight_scale_2 = NaN` or `±Inf` would propagate non-finite values into the GEMM output, contaminating the entire layer's hidden state and likely the KV cache.

**Regression target:** unit test that calls the promote helper (or a thin integration test through `executor_pre_dequant.cu` Phase 0) with both NaN and +Inf weight_scale_2 values, and verifies promoted_scale is zeroed defensively for both.

### F3 — Header-size validation has integer overflow — **P1**

**Where:** `src/model/safetensors_loader.cpp:519-524`.

```cpp
uint64_t header_size = 0;
std::memcpy(&header_size, data, sizeof(uint64_t));
if (8 + header_size > file_size) {       // <-- overflow
    munmap(mmap_base, file_size);
    return false;
}
```

A malicious/corrupt SafeTensors file that declares `header_size = UINT64_MAX - 4` makes `8 + header_size` wrap to `3`. The check `3 > file_size` is false, so the loader proceeds and constructs `JsonParser(json_data, static_cast<size_t>(header_size))` with a length far past the mmap region. The parser does bounded reads, but they read past the end of the mapped region → SIGSEGV.

**Regression target:** unit test that writes a synthetic 16-byte file containing `header_size = UINT64_MAX` and verifies `load_safetensors` returns false without crashing.

**Fix:** change to `header_size > file_size - 8` (with explicit ordering to avoid underflow when `file_size < 8`, but `file_size >= 8` is already enforced at line 495).

### F4 — Tensor offsets not validated against shape × dtype — **P1**

**Where:** `src/model/safetensors_loader.cpp:572-584`.

```cpp
uint64_t offset_start = static_cast<uint64_t>(offsets_val->arr[0].as_int());
uint64_t offset_end = static_cast<uint64_t>(offsets_val->arr[1].as_int());
if (tensor_data_offset + offset_end > file_size)
    continue;
```

Three real bugs:
1. `offset_start` is read but never used in validation. A malformed file with `offset_start > offset_end` would compute a negative "size" and the consuming kernel would read *backwards* into adjacent tensor data.
2. `offset_end - offset_start` is not compared to `nelem(shape) * sizeof(dtype)`. A file declaring an FP16 tensor of shape [1024, 1024] but with `offset_end - offset_start = 1024` (1KB instead of 2MB) would silently load 0.05% of the actual weight data, then return zeroes (or unrelated bytes) for the rest.
3. `tensor_data_offset + offset_start > file_size` is not checked — the start of the buffer might be past EOF even if the (truncated) `offset_end` is in-bounds (which can happen with a corrupted-but-self-consistent file).

**Regression target:** unit tests for each of the three malformed cases.

**Fix:** validate `offset_start <= offset_end`, `offset_end - offset_start == expected_nbytes(shape, dtype)`, and `tensor_data_offset + offset_end <= file_size`. Compute expected_nbytes via a small `dtype_bytes(QType)` helper.

### F5 — Silent drop of malformed tensor entries — **P2**

**Where:** `src/model/safetensors_loader.cpp:554-580`.

Five `continue` statements silently skip tensor entries when:
- `dtype` is missing (`!dtype_val`)
- `dtype` value is not a string
- `shape` is missing
- `shape` is not an array
- `ndim > kMaxDims`
- `data_offsets` is missing/wrong arity
- offset bounds fail (post-fix from F4 above)

A user with a corrupt checkpoint sees `tensor_map` come back partially populated; the model loads as if some tensors are absent; downstream null-checks make load look "successful". No log line is emitted indicating which tensors were dropped or why.

**Regression target:** unit tests that craft a SafeTensors blob where one tensor's `shape` is malformed, and verify a WARN line is emitted naming the tensor and the reason.

**Fix:** replace each silent `continue` with a counter-bumped `IMP_LOG_WARN` summarized at end of `load_shard`.

### F6 — NVFP4 `weight_packed` shape vs `weight_scale` shape not validated — **P2**

**Where:** `src/graph/executor_pre_dequant.cu:237-287` (promote) and `src/quant/nvfp4_gemm.cu:96-108` (kernel).

The kernel hard-assumes group_size=16 (`kMicroBlockSize = 16` at `nvfp4_gemm.cu:31`) and reads `n_mb = K / 16` micro-scales per row. The loader/promote step never verifies that the loaded `weight_scale` has shape `[N, K/16]`. A pathological checkpoint with `weight_scale.shape == [N, K/8]` (group_size=8) would silently load: every other micro-scale would be misinterpreted as the "next" group's scale, producing 12.5% step quant noise on roughly half the elements. Output would not crash — it would just be subtly wrong.

**Regression target:** unit test that constructs `weight_scale` with the wrong group dimension and verifies promote rejects it (warns + skips promotion, leaving the weight in its non-NVFP4 state where the dequant→cuBLAS fallback will run).

**Fix:** add a shape sanity check in `promote()` that compares `sc.weight_scale.shape[1]` against `w.shape[1] / 16` (or equivalently `w.shape[1] * 2 / 16` for the packed-K-half storage). Skip promotion + WARN on mismatch.

### F7 — Header-size upper bound missing — **P2**

**Where:** `src/model/safetensors_loader.cpp:519-524`.

`header_size` is bounded above only by `file_size - 8`. A pathological file of size 1 GiB with `header_size = 1 GiB - 8` (i.e. claiming the entire file is JSON header) would have `JsonParser` allocate / scan a 1 GiB JSON. The parser doesn't impose a separate ceiling. For real-world checkpoints, the header is typically <100 KiB; a multi-megabyte header is suspicious.

**Regression target:** unit test that constructs a 64 KiB file with `header_size = 65528` (just under file_size) and verifies the loader either accepts or rejects it deterministically.

**Fix:** soft-cap `header_size` at, say, 128 MiB; emit ERROR + return false above the cap. Documented in an ADR.

### F8 — `weight_scale` dtype not enforced — **P2**

**Where:** loader silently accepts whatever dtype `weight_scale` was written in. The compressed-tensors spec mandates `float8_e4m3fn` for NVFP4 and `uint8` (UE8M0) for MXFP4. A model marked NVFP4 in `recipe.yaml` but with `weight_scale.dtype == "U8"` would still load through the NVFP4 promote path, then `gemv_nvfp4_row` would interpret the UE8M0 bytes as E4M3 and produce ~2× wrong scales (powers of two interpreted as E4M3 normals).

This is a NVFP4↔MXFP4 cross-misrouting risk. Phase 1 of the prior audit listed it as a P0 cross-cutting concern but Phase 2 didn't add an explicit guard.

**Regression target:** unit test that constructs a synthetic prequant scratch entry where `weight_scale.qtype = QType::INT8` (UE8M0 bytes) and verifies promote rejects it with WARN + skips promotion.

**Fix:** validate `sc.weight_scale.qtype == QType::FP8_E4M3` in `promote()` before applying the formula. Skip + WARN on mismatch.

### F9 — `header_size` not aligned check — **P3** (spec-compliance only)

**Where:** `src/model/safetensors_loader.cpp:519-534`.

The SafeTensors spec recommends 8-byte alignment of the JSON header end (so tensor data begins at an 8-byte boundary). imp doesn't enforce or warn. Files produced by reference SafeTensors libraries always satisfy this, so it's spec-compliance, not correctness. Skip in this run unless a regression model surfaces.

### F10 — `unrecognized layer weights` counter not summarized for SafeTensors — **P3**

**Where:** `src/model/weight_map.cpp:1053`.

The counter exists but only fires per-tensor at log-level WARN. End-of-load summary line gives counts but the PER-TENSOR list is not retained. For diagnosing why a model loads but produces wrong output, a top-N list of unrecognized tensor names would speed user debugging. Not a correctness bug.

## Findings summary

| ID | Title | Severity | Action |
|----|---|---|---|
| F1 | Reference numerical test for compressed-tensors NVFP4 dequant absent | P0 | Add new unit test |
| F2 | Modelopt NVFP4 weight_scale_2 not isfinite-checked | P1 | Extend guard + unit test |
| F3 | Header-size validation has integer overflow | P1 | Reorder check + unit test |
| F4 | Tensor offsets not validated against shape × dtype | P1 | Validate + unit test |
| F5 | Silent drop of malformed tensor entries | P2 | WARN + summary + unit test |
| F6 | NVFP4 weight_packed vs weight_scale shape not validated | P2 | Promote-time check + unit test |
| F7 | Header-size upper bound missing | P2 | Soft-cap + ERROR + unit test |
| F8 | NVFP4 weight_scale dtype not enforced | P2 | Promote-time check + unit test |
| F9 | Header-size 8-byte alignment | P3 | Skip (spec-compliance only) |
| F10 | Unrecognized-tensor list not retained | P3 | Skip (UX, not correctness) |

P0: 1 — ships first, mandatory under Quality Gate.
P1: 3 — all silent correctness bugs.
P2: 4 — all silent correctness bugs that need contrived inputs to trigger.
P3: 2 — deferred; documented here for future reference.

## Cross-cutting items not in scope this run

These appear in the prior audit and remain unresolved; they are not actionable at full Quality Gate within a single autonomous run. Deferred to `docs/audit/followups.md`.

- GLM architecture mapping
- Native SentencePiece (`.model`) parser
- AWQ INT4 dequant kernel
- DeepSeek MLA attention path
- Multimodal SafeTensors loaders
- Tiktoken parser
