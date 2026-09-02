---
name: code-graph
description: Use when a question is about *structure* rather than text - who calls or launches a symbol, where it is defined, what a change would reach, whether something is dead, how a request gets from the API to a kernel. Triggers on "who calls", "who launches this kernel", "where is X defined", "what breaks if I change", "is this still used", "is this dead", "trace the path from X to Y", "blast radius", "what depends on this header". Do NOT use for free-text search (`rg` is better and cheaper) or to open a file whose path you already know.
---

# code-graph - ask the index, then verify

imp has a symbol and call graph in `.codegraph/codegraph.db` (CodeGraph v1.5.0, `codegraph` on PATH). It answers reverse and scope-resolving questions a text search cannot: which function calls a symbol, who reaches it transitively, what a header change touches.

## Hard rules

| # | Rule | Evidence |
|---|---|---|
| 0 | `codegraph sync` before you trust it; it does NOT sync itself and `codegraph status` prints "up to date" on a stale DB | 2026-08-19: 16 days / 236 commits behind, empty answer for every symbol added in that window; sync cost 2.8 s (279 files). 2026-09-02: DB dated 2026-08-21, `codegraph query gdn_scan_chunkpar` = "No results" |
| 1 | Control symbol before believing a negative | run the same query on a live symbol (`codegraph callers write_kv_cache` -> 2); if the control is empty too, the DB or the edge kind is missing (`docs/audit/SETTLED.md`, "control symbol") |
| 2 | Reverse question -> graph; text question -> `rg` | "who calls / what breaks" ~30 tokens via the graph vs 500-2500 via `rg` + reading; "which files mention X" is grep territory |
| 3 | The name in the answer must be the name you asked | trap 1 below |
| 4 | Kernel-launch and destructor questions are `rg` questions right now | the DB has edge kinds `calls contains extends imports instantiates references` and NO `launches` edges (checked 2026-09-02); `ccg enrich`, which materialises them, aborts with `sqlite3.IntegrityError: UNIQUE constraint failed: idx_edges_identity` (`docs/audit/DEBT_LEDGER_2026_08_21.md`). Every `__global__` kernel reads as uncalled |

## Commands

```bash
codegraph sync                    # first
codegraph query   <name>          # where is it
codegraph callers <symbol>        # who calls it
codegraph callees <symbol>        # what it calls
codegraph node    <symbol>        # source + immediate callers/callees (on a FILE arg: prints the file, NOT dependents)
codegraph impact  <symbol>        # what a change reaches
codegraph explore "<topic>"       # symbols + call paths in one shot
```

`ccg` = `~/github.com/kekzl/cplusplus-cuda-graph/ccg`: `ccg enrich` (CUDA launch edges + implicit destructor calls + macro-generated symbols; BROKEN, rule 4), `ccg coverage` (one-level: a kernel with a dead launcher still counts as covered), `ccg kernels`, `ccg revert`.

## Header dependents (SQL only)

```bash
docker run --rm -v "$PWD/.codegraph:/db:ro" python:3.12-slim python -c "
import sqlite3; c = sqlite3.connect('file:/db/codegraph.db?immutable=1', uri=True)
for (p,) in c.execute(\"select n.file_path from edges e join nodes n on n.id=e.source \"
    \"where e.target='file:src/runtime/config.h' and e.kind='imports'\"): print(p)"
```

Re-derive the count every time (`config.h` importers: 48 -> 23 -> 32 -> 28 across four measurements as `src/core/dispatch_policy.h` and `src/core/config/*.h` split the surface).

## Three traps, all confirmed on this repo

1. **A symbol the graph lacks is answered as if it had it.** `gemv_q6k_q8_1` is token-pasted (`IMP_DP4A_QUANT_TYPES(IMP_DEFINE_GEMV_DP4A)`, `src/compute/gemm_dp4a.cu`); the query returned `dispatch_gemv_fp32` (caller of the DIFFERENT `gemv_q6k_q8_1_fp32`) with no nearest-match warning. Enrich materialised those 21 symbols once (`gemv_q6k_q8_1` -> `dispatch_dp4a_gemv`, `src/exec/executor_kernels.cu`); any other macro level or generated header misleads the same way. Name mismatch = stop and `rg`.
2. **"No callers found" != dead.** With enrich, 99% of 423 kernels (468 by 2026-08-27) had a launcher; 74% without; the rest are reached through function-pointer struct fields. Real reachability = BFS over `calls`/`references`/`instantiates` from roots in `tools/`, `tests/`, `src/api/`. Never re-flag: `src/exec/gemm_kernel_*.cu` (registry table binding), `src/core/logging.cpp` (`IMP_LOG_*`), `src/memory/alloc_interpose.cpp` (default-OFF flag), `src/quant/turboquant_fp4.cuh` (device inline), destructors. Confirm with `rg -n '\bname\b' src/ tools/ tests/ include/` (the residue after that grep found the two dead kernels removed in #1220).
3. **`codegraph node <file>` prints the file, not its dependents.** Use the SQL above.

## Destructors

Implicit destructor calls exist only through `ccg enrich` (85% of ~65-71 destructors had a caller with it, 0% without). With enrich broken, `codegraph callers "~PinnedBuffer"` returns nothing; `rg` for holders instead. Destruction through a base pointer is never covered (`~HostPinnedAllocator`, `~HostRegistrar` show no caller and are live).

## What it cannot know

No preprocessor (macro-generated functions absent, template instantiations collapse to the primary), no runtime (which `try_*` dispatch candidate fires: `src/compute/dispatch_paths.h` lists candidates, `src/compute/dispatch_record.h` records what ran).

Pairs with **codebase-audit** (verify before acting) and **sm120-cuda-expert** (once you know the kernel).
