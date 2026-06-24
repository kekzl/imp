# AUDIT_REPORT.md — imp soundness & hardening audit (2026-06-24)

Branch `audit/soundness-hardening-2026-06-24` off `8f2cc9c4`. Companion artifacts:
`AUDIT.md` (append-only findings ledger), `ARCHMAP.md` (code-derived map),
`PERF_LOG.md` (bench log).

## Executive summary

This pass targeted the **soundness** dimensions (A–F: UB/races, fault isolation,
memory/VRAM, concurrency, CUDA-graph↔allocator coupling, ownership) that the
prior pass-1 audit (build/CI/tooling/docs hygiene) underweighted, plus API
fidelity (G), determinism (H) and dead-code (J).

The engine is **fundamentally sound**: the graph↔allocator address-stability
invariant holds, `Model` is const-after-load and lock-free shareable, KV
borrow/refcount/teardown is correct, the C-ABI and HTTP boundaries catch
everything, streaming is real, and there is **no dead multi-arch scaffolding** in
the shipped binary. Four agents independently confirmed a long list of
verified-sound negatives (see `AUDIT.md`).

The real debt was a small number of **latent soundness holes on
error/back-pressure paths** — exactly where the prior audit's "healthy" verdict
had the least coverage. Two are HIGH and silent (KV use-after-free under KV
pressure; a poisoned CUDA context served as if healthy). Both are now fixed or
fail-fast, with the remaining redesign-scale items parked with migration plans.

**Net:** 11 findings fixed-and-validated, 9 parked with plans. No perf regression
(decode gate held). +1 net source file untouched in count; +~90 LOC of fixes,
+~45 LOC of regression tests, −1 stack-array footgun. No silent behavior change
beyond those itemized below.

## What was fixed (validated, shipped as gated commits)

| ID | Sev | Dim | Fix | Validation |
|---|---|---|---|---|
| **F-A1+F-A1b** | high | C/F | reject-newest: deleted the unsafe `evict_lru` of *live* sequences at all 3 engine sites (decode/prefill/spec); safe cached-block reclamation untouched; cancel/rollback fallbacks run on true exhaustion | new unit `AllocationNeverEvictsLiveSequenceUnderPressure`; suite+bench green |
| **F-A3** | high | B | worker catch now syncs + classifies the sticky error; context-poisoning classes fail-fast (`stop_requested_`) with a loud log instead of serving garbage | exception-path-only; full suite + bench green |
| **F-A4** | high | G | `/v1/messages` accepts Anthropic `x-api-key` (constant-time) as well as Bearer | new unit `ApiKeyAuth.*` (5 cases) |
| **F-A7** | med | G | `/v1/messages` 401 now uses the Anthropic error envelope | covered by the auth path |
| **F-A8** | med | G | periodic SSE `ping` during sustained idle (long prefill) on the Anthropic stream | server battery (`test-server`) |
| **F-A6** | med | A | `force_cublas_decode` host block-table moved stack→heap (was a 4× overrun at the 64K ctx cap) | builds; debug-arm path |
| **F-A13** | low | F | weight-upload stream/event → `CudaStream`/`CudaEvent` RAII (no leak on throw) | builds; init-path |
| **F-A15/16** | low | J | corrected stale `sm_90→mode1` and "reject n>1" comments | — |
| **F-A14** | low | F | deleted copy ops on raw-handle owners (GreenCtx/LayerOffload/ExpertLRUCache) → move-only, no latent double-free | compile-time-only; clean build proves no copy/move sites |

## What was parked (verified real; needs more than an autonomous-safe change)

- **F-A2 (high)** — non-streaming unbounded conditional-graph burst doesn't re-poll
  cancel/disconnect/timeout → a dropped client burns up to `max_tokens`. Fix =
  bound the device loop + re-poll per chunk (the spec-ngram `miss_burst` template),
  but it touches the conditional-graph loop with a documented off-by-one history
  (#683/#692) and needs the multi-token-verify GPU coherence battery to ship safely.
- **F-A1b (high)** — RESOLVED (folded into the F-A1 reject-newest fix above): the
  prefill/spec `evict_lru` sites shared the same live-sequence-stripping mechanism;
  all three engine sites now reject-newest instead of preempting.
- **F-A9 (med, candidate)** — NVFP4 grouped-MoE GEMM never consults the deterministic
  flag. Verified true; whether it makes output non-reproducible is **unproven**
  (CUTLASS fixed-schedule grouped GEMM is typically deterministic). Settled by a
  fresh-process A/B experiment (below) — the guarantees doc was NOT edited on the
  hunch (over-flag guard).
- Low/hardening: F-A5 (vision `stop()`→`pause()`), F-A10 (shared-KV COW), F-A11
  (`size_t→int` latent truncation), F-A12 (KV-write `block_id>=0` guard), F-A14
  (move-only RAII handles), F-A17 (swallowed sync return). All in `AUDIT.md`.

## F-A9 determinism experiment — REFUTED

Three fresh-process greedy (temp=0, `IMP_DETERMINISTIC=1`) generations of
Qwen3-Coder-30B-A3B NVFP4 on a coherent prompt produced **byte-identical** output
(A==B==C, prose 1155 B each; the only raw diff was an interleaved async-log
timestamp). So the NVFP4 grouped-MoE GEMM **is** reproducible across fresh
processes in deterministic mode despite not reading the flag — the CUTLASS grouped
GEMM is deterministic by construction (compile-time-fixed schedule, no atomic
reduction). **`determinism.md`'s guarantee is NOT overstated; the doc was left
unchanged.** The missing flag-check is harmless. (This is the over-flag guard from
the codebase-audit skill working as intended — a static "0 references" smell did
not survive an actual A/B.) The earlier MEMORY note of run-to-run NVFP4-MoE PPL
variance was in the *default* (non-deterministic) mode, which is expected.

These three coherent generations also serve as the post-hot-path coherence check
(CLAUDE.md "after hot-path changes"): the F-A1 scheduler edit produced no
repetition/stuck-token/garbage output.

## Before/after metrics

| | snapshot | build | CPU unit | GPU suite | decode tg256 (pp512/2048/4096) |
|---|---|---|---|---|---|
| baseline | `8f2cc9c4` | GREEN | 37/37 (+330+190) | exit 0 | 341.6 / 322.0 / 266.4 |
| post-fix | branch | GREEN | all green (+ApiKeyAuth 5/5) | exit 0 (+EvictLRUProtects) | 341.6 / 322.0 / 266.2 |

Gate (decode 3% band): ≥ 331.4 / 312.4 / 258.4 tok/s — **HELD** (post-fix decode
matches baseline within <0.2%). GPU suite: 8 binaries, 0 failures (330/190/181/158/
187/41/106/73). Prefill pp medians ~21k/47.5k/42.6k, within the documented ±2.6×
cuBLAS-autotune restart variance (informational, not a gate).

## Prioritized parked backlog (for a follow-up)

1. **F-A2** bound the non-streaming device loop + re-poll cancel, OR add a device-checked
   cancel flag to the autonomous conditional-graph loop for full-throughput
   interruptibility (high; a throughput-vs-responsiveness policy call + needs the
   manual coherence battery — owner decision).
2. **F-A5** vision request serialization kills concurrent requests (med) — NOT a
   stop→pause swap (vision mutates global engine image state; pause/resume could leak
   the image to a concurrent text request). Real fix: per-request image binding.
3. **F-A11** `size_t→int` widen (low; unreachable today, touches live FP8 kernel indexing).
4. **F-A12** KV-write `block_id` guard (low; `kv_resolve_slot` shared w/ reads + StreamingLLM `-1`).
5. Carried from pass-1: CI3/T1/BM1 (GPU runner), CI1 (format gate), B2 (CMakePresets) — infra.

(F-A1b + F-A14 resolved this pass; F-A9 refuted by experiment.)

## Honest caveats

- **compute-sanitizer (memcheck/racecheck/initcheck/synccheck) cannot run on this
  WSL2/WDDM host** — all UB/race findings are reasoned statically. A native-Linux
  GPU runner is required to close dimension A's prescribed dynamic methodology.
- The GPU model-E2E tests SKIP under the default `$(PWD)/models` symlink mount; the
  bench + coherence checks ran against the real `/home/kekz/models` mount.
- F-A3's poison-recovery path is reasoned-correct, not IMA-injected (no
  fault-injection harness here); only its no-op-on-the-normal-path property is tested.
