# ADR 0001 — Reference validation harness uses pure C++

**Date:** 2026-05-08
**Status:** Accepted
**Context:** Mission requires a numerical reference for compressed-tensors NVFP4 dequant.

## Decision

Pure C++ reference dequant in `tests/test_nvfp4_compressed_tensors_ref.cu`, computing `val = fp4_to_fp32(nibble) * fp8_e4m3_to_fp32(weight_scale_block) * weight_scale_2_fp32` element-wise on the host, compared against imp's `gemv_nvfp4_kpar` GEMV at max-abs-diff < 1e-2. Choice over (1) imp Python infra (none wired for unit-numerics) and (2) external venv subprocess (would add cross-process fixture complexity and runtime dependency on the user's HF cache). Pure C++ is dependency-free, deterministic, and bit-exact for the dequant formula.

## Consequences

Catches dequant bugs (sign-flip, nibble-order, missing factor) but not full end-to-end logit drift; that is covered separately by `scripts/validate_safetensors.py`'s phase-4 battery at integration level.
