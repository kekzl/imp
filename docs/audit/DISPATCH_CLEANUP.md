# DISPATCH_cleanup: technical-debt cleanup log

Append-only. Earlier entries are never rewritten; corrections are new dated entries.

Rules of the dispatch: behaviour-neutral (bit-exact or the existing documented tolerance), perf-neutral (decode +-2 % of the baseline median, prefill inside the baseline spread), one logical change per commit (`cleanup(<area>): <what> [DISPATCH_cleanup Phase N]`), no test removed or loosened, no `#if 0`, no `-Wno-*`, no raised `filesize_thresholds.toml` entry. Out of scope: MoE atomics nondeterminism, the `/v1/messages` streaming path, FA2 coverage gaps.

## 2026-10-15 Phase 1: inventory (scout)

Inventory: [`DEBT_INVENTORY_2026_10_15.md`](DEBT_INVENTORY_2026_10_15.md). Per-kernel resource table (cuobjdump `-res-usage`, release binary): [`PTXAS_BASELINE_2026_10_15.md`](PTXAS_BASELINE_2026_10_15.md).

| scout | rows | risk L / M / H | not evaluable |
|---|---|---|---|
| host + build (H-001..H-035) | 35 | 30 / 5 / 0 | per-TU compile time (LOC proxy only), include removals need a build, `-Wconversion` under g++/nvcc |
| CUDA (C-001..C-035) | 35 (30 actionable, 5 negative or BLOCKED) | 20 / 10 / 0 | compute-sanitizer BLOCKED on WSL2 (`src/compute/CLAUDE.md`); spill vs local array needs `-Xptxas -v`; per-instance launch proof needs a GPU trace |

| fact | value | source |
|---|---|---|
| kernel instances in the release binary | 1018 (466 names) | cuobjdump, v0.53.0 |
| kernel names with no reference in `src/` | 0 | C-001 |
| emitted targets | `sm_120a` SASS (131 ELF) + `compute_120f` PTX (130) | H-035 / C-012 |
| `-rdc` TUs missing from the 120f PTX | 26 instances (dense NVFP4 decode GEMVs, mmq_imma_raw) | C-012 |
| clang-tidy over 426 TUs incl. `modernize-*`, `-Wshadow`, `-Wconversion` | 16012 unique (6155 in .cpp/.h); bugprone and analyzer pins at 0 | H-027, H-028 |
| TODO / FIXME / HACK / XXX in code | 0 | H-035 |

Out of scope seen (logged, not analysed): MoE atomics nondeterminism, `/v1/messages` streaming, FA2 coverage gaps. The legacy causal_softmax + cuBLAS attention path stays: SETTLED S-8 keeps it as the Gemma-4 head_dim 512 tier; no FA2 call-site table was built.

## 2026-10-15 Phase 2: triage (orchestrator)

Rank = (risk reduction + maintainability gain) / effort. Each batch <= ~400 changed lines, one area. No H-risk row exists; M rows below need an explicit go entry here before they start.

### BATCH_PLAN

| batch | area | rows | risk | gate notes |
|---|---|---|---|---|
| B1 | ownership, ODR | H-018 (rule of five on 7 RAII types: an implicit copy of `MoeRoutingBuffers` double-frees), H-019, H-023, H-024, H-026 | L | compile-only for H-018 (a compile error exposes a real copy); SASS unchanged |
| B2 | stream ordering | C-020 (logit_bias fallback reads logits via legacy `cudaMemcpy`, race), C-021 (legacy `cudaMemset` of `d_lookup_` at runtime), C-022, C-025, C-026, C-027 | L | bit-exact outputs; decode A/B for C-020 (per-step path with logit_bias) |
| B3 | launch checks, gates | C-014 (`pdl_launch.cuh:37` fallback unchecked: 68 of 106 `pdl::launch` sites), C-015 (`.cuh` launches outside the guard gate), H-008, H-016 (82 discarded cuBLAS statuses + gate extension), H-017 | L | success path unchanged; gate selftests extended |
| B4 | dead code | H-001, H-003..H-007, H-010, C-003, C-008..C-011 (dead `sm_ver >= 90` and arch guards), C-013 + H-034 (`IMP_USE_CUTLASS` cannot be OFF), H-033 | L | evidence column per row; SASS diff for guard removals (expected identical) |
| B5 | magic numbers | H-029 (`101376` smem opt-in, 5 sites), H-031, H-032, C-032..C-034 | L | constexpr rename, SASS identical |
| B6 | headers, build | H-020 (two constrain headers in `executor.h`, ~125 TUs each), H-021, H-009 (10 test/bench TUs inside `libimp`) | L | compile-only; build time before/after |
| B7 | duplicated helpers | H-012 (7 `stoi`/`atoi` layer-index parses -> `parse_index`), H-014 | L | new malformed-index unit test per loader |
| B8 | warnings | H-027 (11 host shadow/shorten sites, then `-Wshadow` on), H-025 | L | zero new warnings |

### Deferred, needs an explicit go

| rows | why |
|---|---|
| C-004..C-007 (kernel variant merges) | acceptable only with byte-identical `cuobjdump -sass` per instance |
| C-012 | owner decision: `IMP_DISABLE_120F_FALLBACK=ON` + SM120.md correction, or drop `-rdc` |
| C-017, C-018 (PDL registration of two production verify kernels) | perf path; SETTLED A2-2 anchor to correct first |
| C-023, C-024, C-029, C-031 | perf-affecting; measure first |
| H-002, H-011, H-013, H-015, H-022, H-028, H-030 | M risk or behaviour change (H-013 turns silent 0 into an error; H-030 may be a KV-sizing bug, needs a GPU repro) |
| C-002 | not new: SETTLED H A1-9 |

Phase 3 (baseline) runs before any edit and needs a GPU slot (~2 h: reference dumps for five models, Qwen3-Coder-30B-A3B NVFP4 decode/prefill >= 5 iterations x 2 containers, roofline snapshot).
