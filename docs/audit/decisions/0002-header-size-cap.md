# ADR 0002 — SafeTensors header-size soft cap at 128 MiB

**Date:** 2026-05-08
**Status:** Accepted
**Context:** Phase 1 finding F7 — `header_size` was bounded only by `file_size - 8`, allowing pathological files that claim multi-GB JSON headers.

## Decision

Soft-cap `header_size` at 128 MiB (`134217728` bytes). Above the cap, return false with `IMP_LOG_ERROR` naming the file path and the declared header size. 128 MiB is ~1000× larger than any real model header (Llama-3.1-405B's index is ~100 KiB), so the cap is far above legitimate use and far below pathological inputs that would force the JSON parser to scan multi-GB.

## Consequences

Rejects malicious / corrupt files faster (before the JSON parser allocates). No effect on real checkpoints. Hard-coded constant (not configurable) — if someone releases a model with a >128 MiB header in the future, this becomes a friction point and we'll re-evaluate. Recorded in `safetensors_loader.cpp` as a named constant.
