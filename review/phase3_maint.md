# Phase 3 — CODEREAPER Maintainability Audit

Anchor commit: `f58eb9e` (matches Phase 1 / Phase 2). Target: sm_120a /
RTX 5090 only. Citations are `file:line` against working tree;
mechanical counts re-derived where Phase 1 already pinned them are
marked `(P1 §N)`. Smell claims always carry a concrete example.

The headline of this audit, repeated up front because every section
restates a facet of it: `src/graph/` is a single 15.8 KLOC clay tablet
that knows about every quantization format, every model architecture,
every weight-cache shape and every dispatch decision in the engine.
`executor_kernels.cu:2003` declares one function with 21 parameters
that branches eight ways on cache pointers; `executor_attention.cu`
contains 19 `cfg.arch == ModelArch::GEMMA4` branches inside the
single-arch hot path; `executor_forward_moe.cu` is 2 563 LOC of
"dispatch path 1..5" with no factoring of the five paths.
**`graph/` is the danger zone, GEMMA4-coupling is the smell, and the
WeightCaches god-struct is the root cause.** Sections 1, 2, 3, 11
all point at this from different angles.

---

## 1. Coupling hotspot map

### 1.1 Subsystem edge weights (cross-subsystem `#include` counts from P1 §1)

Reproduced from Phase 1 for orientation; not recomputed:

| from \ to | api | compute | core | graph | memory | model | quant | runtime | vision |
|---|---|---|---|---|---|---|---|---|---|
| api      |   - |   0 |   1 |   0 |   1 |   5 |   0 |   2 |   0 |
| compute  |   0 |   - |  58 |   1 |   0 |   4 |  10 |  11 |   0 |
| core     |   0 |   0 |   - |   0 |   0 |   0 |   0 |   0 |   0 |
| graph    |   0 | **108** |  26 |   - |  24 |   3 |  41 |  16 |   0 |
| memory   |   0 |   0 |  10 |   0 |   - |   1 |   0 |   1 |   0 |
| model    |   0 |   0 |  21 |   0 |   0 |   - |   3 |   2 |   0 |
| quant    |   0 |   2 |  15 |   0 |   0 |   0 |   - |   1 |   0 |
| runtime  |   0 |  20 |  19 |   3 |  13 |  15 |   0 |   - |   4 |
| vision   |   0 |   1 |   4 |   0 |   1 |   1 |   0 |   2 |   - |

### 1.2 The 3 worst couplings

**#1 — `graph → compute`, 108 edges (highest in the table).**
`graph/executor_kernels.cu:1-22` lists 15 distinct `compute/*.h` headers
(`gemm.h`, `gemm_q6k.h`, `gemm_cutlass_sm120.h`,
`gemm_cutlass_mxfp4_sm120.h`, `hadamard.h`, `ggml_mmvq.h`,
`mmq_q4k_v2.h`, `ptx92_utils.cuh`, `warp_reduce.cuh`, …) plus 5
`quant/*.h`. `graph/executor_forward_moe.cu:12-29` adds 18 more.
**This is not an abstraction — it is a literal cross-product include
list.** Each new kernel implementation in `compute/` requires an
include in `graph/`, a dispatch arm in `gemm_dispatch_impl`
(`executor_kernels.cu:2003`) and a cache-map field in `WeightCaches`
(`executor.h:286`). The leak goes both directions: `compute/` cannot
be replaced with a different implementation without touching `graph/`
in N places.

**#2 — `graph → quant`, 41 edges.** `graph/executor_kernels.cu:11-15`
and `graph/executor_pre_dequant.cu:7-9` each import 4-5 quant headers,
then re-implement the per-qtype routing in `gemm_dispatch_impl`. The
two layers do the same job:
- `quant/weight_dispatch.cu:73-125` already has a per-qtype dispatch
  that decides between cuBLASLt, dequant-then-FP16, and CUTLASS.
- `graph/executor_kernels.cu:2018-2268` does **the same decision tree
  over again** with a different parameter set.
Two dispatch tables for the same problem. Every new qtype lands in
both.

**#3 — `compute → core` (58 edges) is benign.** `core/` is a strict
leaf (P1 §1 confirms zero outgoing cross-subsystem edges). The 58
edges are `Tensor`, `QType`, `logging` — exactly what a leaf utility
layer is for. **Not a smell; called out so it isn't confused with the
two above.**

### 1.3 Back-edges

- **`compute → graph` (1 edge).** Phase 1 §1 flagged this as a
  follow-up. Grep:
  `grep -rn '#include "graph/' src/compute/` → one hit at
  `src/compute/preamble_gate.h:5` (`#include "graph/quant_scratch.h"`).
  `preamble_gate.h` defines a 1-line guard used by a compute kernel
  to read a graph-owned struct. **Tiny but architecturally
  wrong**: `compute/` should not name `graph/` symbols. The struct
  (`QuantScratch`) belongs in `core/` or in a new `runtime/scratch.h`.
  3-line fix.
- **`compute → quant`/`quant → compute` cycle (P1 §1 #1).**
  `grep -rn '#include "compute/' src/quant/` →
  `src/quant/dequant_gpu.cu` includes `compute/warp_reduce.cuh` and
  `compute/ptx92_utils.cuh`. Those are header-only utilities that
  morally belong in `core/`. Move the two `.cuh` files to `core/` and
  the cycle dies.
- **`compute ↔ runtime`** (20 + 11 edges).
  `compute/*.cu` imports `runtime/config.h` for the central
  `RuntimeConfig::current()` singleton. **This is the intentional
  cycle** — kernels need runtime knobs. Tolerable so long as
  `runtime/config.h` is pure data; it currently is (`config.h:1-168`).

### 1.4 God headers (>30% of TUs include them)

Total `.cu/.cpp` TUs in `src/`: **134**. Threshold = 41.

| Header | Included by | % | Verdict |
|---|---:|---:|---|
| `core/logging.h` | 92 | 68% | **GOD**. Every TU pulls it. Acceptable cost (header is 75 LOC, `<atomic>` + `<cstdio>` only — `src/core/logging.h:1-30`), but means a change to `IMP_LOG_DEBUG` rebuilds two thirds of the project. |
| `core/tensor.h` | 51 | 38% | **GOD**. Carries `QType`, `Tensor`, `Buffer`, `<cuda_runtime.h>`. Editing it rebuilds 51 TUs. Not avoidable — Tensor is the core type. |
| `runtime/config.h` | 18 | 13% | Below threshold; centralization (§9) keeps it small. |

`logging.h` and `tensor.h` are the only two god headers. Both are
load-bearing. **No action.**

### 1.5 Model/compute decoupling — can a new model be added without
compute changes?

**No.** Evidence:

1. `grep -rEohn 'ModelArch::[A-Z0-9_]+' src/runtime/ src/graph/ src/compute/` →
   **40 production-code sites** branch on `ModelArch`. Distribution:
   - `ModelArch::GEMMA4` — **30 sites** (e.g.
     `src/graph/executor_attention.cu:161, 310, 387, 464, 472, 493,
     534, 596, 658, 678, 821, 893, 1198, 1274`;
     `src/graph/executor_forward_moe.cu:187, 218, 224, 229, 262, 310,
     324, 381, 1727, 1735, 1745, 1766, 1781, 1875, 1893, 1899, 2304,
     2538, 2541`;
     `src/runtime/engine.cpp:828, 1663`;
     `src/graph/executor_workspace.cu:53`).
   - `ModelArch::LLAMA4` — 1 site.
   - `ModelArch::GEMMA3` — 1 site.
2. `src/include/imp/types.h:25-38` defines 14 enum values; only
   GEMMA4 has dedicated hot-path branches. Either it is the only
   special-case architecture or every other arch has been wedged
   into the generic path with silent quality regressions.
3. `model/chat_template.cpp:72` is the **only** legitimate
   `switch(arch)` (chat template family). All others are kernel
   selection — i.e. compute behavior conditional on model identity.

**Verdict: NO.** Adding a new arch requires touching at minimum
`graph/executor_attention.cu`, `graph/executor_forward_moe.cu`,
`graph/executor_workspace.cu`, and `runtime/engine.cpp`. The
abstraction never landed; GEMMA4 was shipped by sprinkling
`if (cfg.arch == ModelArch::GEMMA4)` along the hot path. See §11 for
the refactor.

---

## 2. Danger zones (Top-10 files)

Scoring axes:

- **LOC** — raw line count.
- **Template depth** — `grep -c '^template'` per file.
- **Branch density** — total `if`/`else if`/`switch case` per 100 LOC,
  estimated via grep.
- **Global state** — `static` file-scope variables that are mutable
  (caches, handles, scratch).
- **Implicit dispatch** — function pointers, `pdl::enable`,
  switch-on-enum, `unordered_map` keyed on `const void*` for routing.
- **Test coverage proxy** — does any test in `tests/` exercise the
  file's public functions? Y / N.

Scores 1-5 per axis, total /30. Higher = scarier.

| Rank | File | LOC | tmpl | branch | global | dispatch | tested | TOTAL | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `src/graph/executor_kernels.cu` | 2 327 | 1 | 5 | 5 | 5 | partial | **25** | `gemm_dispatch_impl` (21 params, 8 cache-map branches). Touch this file → break every model. |
| 2 | `src/graph/executor_forward_moe.cu` | 2 563 | 0 | 5 | 4 | 5 | partial | **24** | 5 dispatch paths, 19 GEMMA4 branches, 5 D2H sync sites (P2 §4.5). |
| 3 | `src/runtime/engine.cpp` | 3 066 | 0 | 5 | 5 | 4 | partial | **22** | 40 member functions; `step_*` family is the request scheduler + KV cache manager + decode driver + MTP arbiter rolled into one class. |
| 4 | `src/graph/executor_pre_dequant.cu` | 2 556 | 0 | 4 | 4 | 4 | partial | **22** | Pre-allocates and dequant-converts every weight cache shape; one of the few sites that reads `IMP_FORCE_Q4K_V2` directly. |
| 5 | `src/graph/executor_attention.cu` | 1 299 | 0 | 5 | 3 | 5 | partial | **21** | 8-way KV-dtype switch (P2 §3.12) + 14 GEMMA4 branches + 2 `goto after_attention` (line 796, 960) + `naive_attention_prefill` fallback. |
| 6 | `src/compute/mmq_q4k_v2.cu` | 1 667 | 7 | 3 | 2 | 1 | yes (microbench only) | **18** | 7 template instantiations of the same kernel layout; `IMP_FORCE_Q4K_V2`-gated; -4% E2E per `mmq_q4k_v2_phase2_shipped_2026_05_16` memo. |
| 7 | `src/model/jinja.cpp` | 2 629 | 0 | 5 | 2 | 2 | yes | **18** | Hand-rolled Jinja2 evaluator. Pure host code, AST-based. Bugs here corrupt the prompt silently. |
| 8 | `src/compute/sampling.cu` | 1 701 | 0 | 5 | 4 | 3 | yes | **17** | 5 distinct sample kernels (argmax, argmax_partial, argmax_reduce, topk_topp, topp_sorted) + DRY penalty + Mirostat. Allocates via raw `cudaMalloc/cudaFree` (line 727-781). |
| 9 | `src/compute/gemm.cu` | 1 694 | 0 | 4 | 5 | 4 | yes | **17** | cuBLAS handle + workspace + algo cache + 7 GEMV functions + 5 fused MoE GEMV variants; `s_gemm_cache` mutex-locked. |
| 10 | `src/model/weight_upload.cu` | 2 092 | 0 | 4 | 3 | 3 | partial | **15** | Per-qtype upload + per-arch quant conversion + GPTQ + audit env knob (`IMP_AUDIT_NVFP4_SCALES`). Long-tail special cases. |

### 2.1 Per-entry remediation sketches

**#1 `executor_kernels.cu` (2 327 LOC, score 25).** Hosts
`gemm_dispatch_impl` (line 2003-2269) — a 266-line cascading if/else
on 6 different `unordered_map<const void*, ...>*` pointers. Each cache
arm has its own preconditions (`input.qtype`, `output.qtype`, M-shape,
`fp32_output`, `prefer_fp16_cache`). **Suggested split:** introduce a
per-qtype `GemmKernel` registry (qtype → `unique_ptr<GemmKernel>`),
move the qtype-specific code from `gemm_dispatch_impl` into the
registry entries, and reduce the file to a 200-line dispatcher.
Estimate: 1 800 LOC stays in `compute/`, 500 LOC stays in
`executor_kernels.cu` as the registry plumbing. See §11 refactor #1.

**#2 `executor_forward_moe.cu` (2 563 LOC, score 24).** Header
comment at line 5-10 says "Dispatch paths: 1. NVFP4 decode fast. 2.
TC fused. 3. Scalar fused. 4. Batch path. 5. Shared expert path."
These five paths are interleaved with 19 GEMMA4 branches. **Split
into:** `executor_moe_decode_fast.cu` (path 1, ~600 LOC),
`executor_moe_prefill.cu` (paths 3+4, ~900 LOC),
`executor_moe_shared_expert.cu` (path 5, ~200 LOC),
`executor_moe_routing.cu` (top-k + permute + offsets, ~300 LOC),
`executor_moe_gemma4.cu` (the GEMMA4-specific overrides extracted,
~400 LOC). Cuts the single-file cognitive load by 60%.

**#3 `engine.cpp` (3 066 LOC, score 22).** 40 member functions on
`Engine`. `step_decode_forward` alone is 392 lines (line 2421-2813).
**Split into:** `engine_init.cpp` (~700 LOC for the `init_*` family),
`engine_step_prefill.cpp` (~700 LOC), `engine_step_decode.cpp`
(~700 LOC), `engine_mtp.cpp` (~300 LOC), `engine.cpp` (residual:
ctor/dtor + small driver, ~600 LOC). Each part of the file already
has clear "// =====" section banners (line 29-31, etc.); the split is
mechanical.

**#4 `executor_pre_dequant.cu` (2 556 LOC, score 22).** Owns the
six-way weight-cache-population logic (FP16, FP8, NVFP4, CUTLASS NVFP4,
MXFP4, Q4K-v2). Each cache type has 200-400 LOC of population code in
the same file. **Split per cache type** into separate `.cu` files
sharing a `WeightCachePopulator` interface in
`graph/executor_helpers.h`. Knock-on: removes one of the two
dispatch tables (§1.2 #2) by letting each populator own its
dispatch arm.

**#5 `executor_attention.cu` (1 299 LOC, score 21).** Already
mostly factored — has `set_l2_persist_kv`, `set_l2_streaming`,
`clear_l2_policy` extracted into helpers. The remaining smell is the
14 GEMMA4 branches threaded through a single `run_attention` function.
**Suggested split:** keep generic attention in this file; create
`executor_attention_gemma4.cu` with a `run_attention_gemma4` override
that calls into the generic path for the common steps. Removes the
GEMMA4-coupling on this file from 14 sites to 1 dispatch.

**#6 `mmq_q4k_v2.cu` (1 667 LOC, score 18).** Seven template
instantiations (line 251, 270, 470, 818, 1016, 1294, 1462) of
`<int kP3BN>` — phases of the same kernel. Two of the seven
correspond to the v1 dp4a (legacy, only the M<16 path remains useful)
and the v2 HMMA. **Recommendation:** monomorphize to the 2 actually-
shipped configurations (kP3BN=128 and kP3BN=256), delete the other 5.
Saves ~700 LOC. See §3.

**#7 `jinja.cpp` (2 629 LOC, score 18).** Self-contained
host code, no CUDA. Score is high because parsing bugs corrupt the
prompt silently and only show up in degeneration tests. **Has tests**
(`tests/test_jinja.cpp`). Acceptable as-is; the file is "big but
boring." No split recommended — splitting a hand-rolled parser hurts.

**#8 `sampling.cu` (1 701 LOC, score 17).** Five sampler kernels
(line 49, 106, 159, 255, 673) + DRY + Mirostat + JSON-constrain
mask integration. Uses raw `cudaMalloc/cudaFree` (line 727-781) which
violates CLAUDE.md "no `cudaMalloc/cudaFree` in hot loops" reading
strictly — Phase 2 §5.6 already noted the grow-only nature. **Split
into:** `sampling_kernels.cu` (the 5 device kernels, ~700 LOC),
`sampling_state.cu` (host-side DRY/Mirostat state, ~500 LOC),
`sampling_dispatch.cu` (entry points and scratch-buffer lifetime,
~500 LOC).

**#9 `gemm.cu` (1 694 LOC, score 17).** Owns 5 mutable `static`
variables that survive between calls: `s_workspace`, `s_bench_scratch`,
`s_gemm_cache` (the cuBLASLt algo cache), `s_gemm_cache_mutex`,
`s_workspace_size` (line 85-92, 239-240). Plus 5 fused MoE GEMV
variants (`gemv_q6k_moe_decode`, `gemv_q8_0_moe_decode`,
`gemv_q6k_moe_gate_up_fused`, `gemv_q8_0_moe_gate_up_fused`,
`gemv_gate_fp32_fp32input`). The MoE-specific variants belong in
`compute/gemm_moe_fused.cu`, not in the dense GEMM file. Move them
(~500 LOC).

**#10 `weight_upload.cu` (2 092 LOC, score 15).** One file, two
public functions (`grep -c '^void\|^bool' src/model/weight_upload.cu` →
1). Everything else is in anonymous namespace. The single `upload_*`
function is essentially a giant `switch(qtype)` covering 12 quant
types × 4 model arches. **Split** per qtype family:
`weight_upload_dense_fp.cu` (FP16/BF16/FP32 paths),
`weight_upload_q_k.cu` (Q4_K/Q5_K/Q6_K/Q8_0),
`weight_upload_nvfp4.cu` (NVFP4 sidecar + prequant),
`weight_upload_mxfp4.cu`, `weight_upload_gptq.cu`.

---

## 3. Template wildwuchs vs. monomorphisation

Concern: header-template bloat slows compilation, hides codegen
errors, and complicates debugging. For each candidate, look at
template parameters → ask "is the parameterization buying real
flexibility?"

### 3.1 `src/compute/attention_fmha_sm120.cu` — KEEP

Line 68 and 581 declare `template <int Bq, int HD>`. Instantiated
configurations (file-internal):
- `<Bq=128, HD=64>` (FP16)
- `<Bq=128, HD=128>` (FP16)
- `<Bq=64, HD=192>` (FP16)
- `<Bq=32, HD=256>` (FP16)
- `<Bq=32, HD=512>` (FP16 — GEMMA4 global layers)
- Same set for FP8.

**Verdict: keep.** HD is set by model architecture; Bq is chosen to
fit the 99 KiB SMEM cap per HD value (P2 §2.4). The template gives
real flexibility — both NCCL parameters affect kernel layout. The
file is in the danger zone because of the lookup tables, not the
templates themselves.

### 3.2 `src/compute/attention_paged.cu` — KEEP

Line 289, 598, 816, 1110 declare `template <int HEAD_DIM>`.
HEAD_DIM ∈ {64, 96, 128, 192, 256, 512} per kernel. Same justification
as 3.1. Real model-dependent flexibility.

### 3.3 `src/compute/mmq_q4k_v2.cu` — **MONOMORPHIZE, delete 5 of 7**

`grep -n '^template' src/compute/mmq_q4k_v2.cu`:
- line 251: `template <int N>` (helper)
- line 270: `template <int kP3BN>` (Phase 3 BN — early experiment)
- line 470: `template <int kP3BN>` (Phase 4)
- line 818: `template <int kP3BN>` (Phase 5)
- line 1016: `template <int kP3BN>` (Phase 6a)
- line 1294: `template <int kP3BN>` (Phase 6b)
- line 1462: `template <int kP3BN>` (Phase 7)

Per memory file `mmq_q4k_v2_phase2_shipped_2026_05_16`: "Phases 1a/1b/2/3/4/5/6a/6b/7a/7b all merged." The file
contains every phase as a separate template instantiation, kept
in the binary even after later phases superseded earlier ones.

**Dispatch is via a single env knob** (`IMP_FORCE_Q4K_V2=1`, gated
in `graph/executor_pre_dequant.cu:680`) which **picks one kP3BN value
at runtime**. Five of the seven templates are dead at runtime.

**Removable LOC:** ~700 (each phase template is roughly the same
shape — see line 470-815, 818-1015, 1016-1293, 1294-1421, 1462-1591
for span deltas). End-state: one template at kP3BN=256, one at
kP3BN=128, glue. Plus the well-known E2E regression (-4% on
Qwen3.6-35B Q4_K_M per the same memo) suggests the entire 1 667 LOC
is a candidate for removal — but that's a policy call (§10 #6).

### 3.4 `src/compute/gemv_dp4a_traits.cuh` — KEEP

P1 Appendix A flagged this as 1 620 LOC of header-only template
library. Spot-check: it provides `dp4a` traits for Q4_0, Q4_1, Q5_0,
Q5_1, Q8_0, Q4_K, Q5_K, Q6_K (8 quant types × 3-4 thread/warp/CTA
shape parameters). The instantiations actually fire across
`src/compute/gemm_dp4a.cu`, `src/compute/ggml_mmvq.cu`, and
`src/quant/quant_gemm.cu`. **Real flexibility** — one traits header
that 3 TUs share. Keep.

### 3.5 `src/compute/rope.cu` — KEEP

Two instantiations (line 137 `<float>`, 140 `<__half>`). Both used at
runtime. Minimum-flexibility template. Fine.

### 3.6 `src/compute/gemm_grouped_nvfp4_smallM.cu` — DELETE 948 LOC
**(per Phase 2 §7.5 leak #5)**

Per memory file `nvfp4_moe_prefill_landscape_2026_05_10` and Phase 2
§7.5, the smallM kernel is opt-in via `IMP_NVFP4_SMALLM=1` and is
**-50–55% vs CUTLASS at production shapes**. The kernel is the
**only production TMA in imp** (`gemm_grouped_nvfp4_smallM.cu:117-122,
362-363, 702-744`). If the policy is to retire it from production,
move the 948 LOC to `tests/bench/` (Phase 1 §7.8 already advocates
relocation, not deletion).

### 3.7 `src/compute/gemm_capture_fp16_sm120.cu` — RUNTIME-CHECK

Phase 1 §7.5 flagged as ~600 LOC dispatched only if
`s_avail = (prop.major*10+minor >= 120)`. On sm_120a-only this flag
is always true. The kernel is the WMMA FP16 GEMM intended for
captured graph paths. Whether it actually fires at decode-graph
capture isn't clear from Phase 2; could be 600 dead LOC or a working
opt-in. Phase 4 should check.

### 3.8 Summary of monomorphisation streichliste

| Source | Action | LOC removable |
|---|---|---:|
| `mmq_q4k_v2.cu` | Keep 2 of 7 phase templates | ~700 (subset of file) |
| `mmq_q4k_v2.cu` if frozen-as-opt-in is the final policy | Delete entire TU | 1 667 |
| `gemm_grouped_nvfp4_smallM.cu` | Move to tests/bench/ | 948 (relocate) |
| `gemm_capture_fp16_sm120.cu` if Phase 4 confirms dead | Delete | ~600 |

---

## 4. Header hygiene

### 4.1 Average include depth — spot-check of 10 hot TUs

Direct `#include` lines (does not transitively expand):

| TU | Direct includes | Notes |
|---|---:|---|
| `src/graph/executor_forward_moe.cu` | **45** | Worst offender. Each compute kernel pulled in by its own header. |
| `src/runtime/engine.cpp` | 24 | Reasonable for a driver TU. |
| `src/compute/gemm.cu` | 15 | Per-quant kernel + cuBLAS handles + workspace. |
| `src/quant/nvfp4_gemm.cu` | 13 | Reasonable. |
| `src/compute/sampling.cu` | 12 | Reasonable. |
| `src/api/imp_api.cpp` | 12 | Reasonable for the API boundary file. |
| `src/runtime/cuda_graph.cu` | 10 | Reasonable. |
| `src/model/jinja.cpp` | 10 | Reasonable. |
| `src/memory/kv_cache_manager.cpp` | 9 | Reasonable. |
| `src/compute/attention_paged.cu` | 9 | Reasonable. |

`executor_forward_moe.cu` is the outlier. 45 direct includes is an
include-depth smell — every compute kernel surface area pulled into
one TU. Splitting the file per §2 #2 will drop this naturally (each
sub-file ends with ~10 includes).

### 4.2 Top-5 most-included internal headers (P1 §1 ranked-by-includes)

Reproduced from §1.4:

| Header | Includers | Risk to rebuild on edit |
|---|---:|---|
| `src/core/logging.h` | 97 | Touching this triggers a near-full rebuild. Header is intentionally `<atomic> + <cstdio>` only; safe. |
| `src/core/tensor.h` | 51 | Same. Pulls `<cuda_runtime.h>` transitively. Edits here recompile everything that touches a Tensor. |
| `src/runtime/config.h` | 20 | After the centralization (§9), 20 TUs is roughly proportionate to "everything that reads a runtime knob." OK. |
| `src/memory/vram_allocator.h` | 17 | Acceptable. |
| `src/compute/gemm.h` | 17 | Acceptable. |

**No leaks in the top 5.**

### 4.3 CUDA headers leaking into "host-only" headers

`grep -l 'cuda_runtime\|cuda.h' src/core/*.h src/api/*.h src/runtime/*.h src/model/*.h`:

- `src/core/cuda_raii.h` — fine, this is the CUDA RAII wrapper.
- `src/model/model.h:12` — `#include <cuda_runtime.h>`. **Smell**:
  `Model` is the in-memory model representation. Should not require
  the CUDA runtime header. The dependency is on `cudaStream_t`/
  pointer types stored in `Model::weight_*` fields. **Fix:** forward-
  declare the stream and use `void*` for device pointers in the
  Model struct (it already lies about its tensor types via `Tensor`
  abstractions; one more layer of abstraction here costs nothing).
- `src/runtime/engine.h:18` (transitive via `core/cuda_raii.h:1`) —
  acceptable, engine owns CUDA streams.
- `src/runtime/batch.h:7` — `#include <cuda_runtime.h>` for
  `cudaStream_t` parameter on `BatchBuilder::upload`. Reasonable.
- `src/runtime/vision_pipeline.h`, `src/runtime/graph_diag.h`,
  `src/runtime/cuda_graph.h`, `src/runtime/pdl.h`,
  `src/runtime/mtp_forward.h`, `src/runtime/green_ctx.h` — all
  legitimate (own CUDA state).

**Action:** decouple `model/model.h` from `<cuda_runtime.h>` (1 file
edit, knock-on rebuild reduction on `model/` TUs).

### 4.4 IWYU violations (spot check)

- `src/graph/executor.h:1-20` includes `model/model.h`,
  `memory/kv_cache.h`, `memory/ssm_state.h`, `memory/layer_offload.h`,
  `compute/moe_routing.h`, `compute/json_constrain.h`,
  `compute/schema_constrain.h`, `quant/nvfp4_quant.h`, `quant/turboquant.h`,
  `compute/gemm_cutlass_sm120.h`, `compute/gemm_cutlass_mxfp4_sm120.h`,
  `core/tensor.h`, `graph/weight_handle.h`, `graph/moe_workspace.h`,
  `graph/quant_scratch.h`, `runtime/storage_planner.h`,
  `<cuda_runtime.h>`, `<cuda_fp16.h>`.
  This is **18 includes in a header** — `executor.h` is itself 796 LOC
  and 15 TUs include it. Every change to any of those 18 transitive
  includes rebuilds 15 TUs. Most of these are for type-defining
  fields (`InferenceState`, `WeightCaches`) — i.e. legitimate. But
  the two CUTLASS-glue includes (`gemm_cutlass_sm120.h`,
  `gemm_cutlass_mxfp4_sm120.h`) are only for `CutlassNvFP4Weight`,
  `CutlassMxFP4Weight` typedefs that could be forward-declared.
  Small win.
- `src/graph/executor_kernels.cu:1-22` — includes 18 headers, every
  one used. No IWYU violations spotted.
- `src/compute/attention_paged.cu:1-10` — 9 includes, clean.

**Verdict:** the worst IWYU smell is the 18-include public header
`executor.h`. Trim the CUTLASS glue (forward-declare instead of
include) for a small reduction.

---

## 5. Dead code / duplication

### 5.1 Dead functions

**`mxfp4_act_sf` argument branch (`executor_kernels.cu:2083-2094`).**
The CUTLASS MXFP4 path requires both an NVFP4 cache hit AND an MXFP4
cache hit on the same weight (`nvfp4_view != nullptr` outer condition
+ `mxfp4_cache->find(weight.data)` inner condition). In practice the
loader populates exactly one of `nvfp4_cache` / `mxfp4_cache` per
weight (per memory file `nvfp4_prequant_status` and the prequant
loader convention). The MXFP4 branch inside an `nvfp4_view != nullptr`
block is **dead in production**; only fires under a synthetic test
where both caches are populated for the same tensor (none such test
exists — `grep -rn 'CutlassMxFP4Weight' tests/` returns 0). **Removable:
~12 LOC.**

### 5.2 Duplicate implementations

1. **`gemv_q6k` / `gemv_q8_0` + the `_moe_*` fused variants.**
   `src/compute/gemm.cu:1204, 1243, 1320, 1367, 1546, 1603` define
   five fused MoE GEMV kernels alongside the two solo GEMVs. A
   separate `gemm_q6k.cu` (310 LOC, P1 §4) also exists for Q6_K. Two
   files implementing Q6_K GEMV with overlap. The `gemm.cu` versions
   are the per-expert MoE fused variants; `gemm_q6k.cu` is the
   prefill grouped variant. **Not strict duplication, but adjacent
   functionality across two files** — refactor target.

2. **Two RMSNorm dtype paths** (`compute/layernorm.cu:40, 72, 142,
   174, 301, 345`) each followed by a host dispatcher (line 271, 330,
   371, 384). Six kernels for {fp32, fp16} × {basic, +residual, +dtype-
   convert}. **Not duplication — necessary dtype coverage.** Keep.

3. **5 sample kernels in `compute/sampling.cu`** (line 49, 106, 159,
   255, 673). Used; not duplicates.

4. **Multi-block dispatch tables — §1.2 #2.** `gemm_dispatch_impl` in
   `executor_kernels.cu:2003` AND `dispatch_quant_*` in
   `compute/weight_dispatch.cu:73-125` do overlapping per-qtype
   dispatch. The first dispatches on (input.qtype, weight.qtype,
   cache pointer); the second on (qtype only). **Genuine duplication
   of dispatch policy.** Removable: estimate 150 LOC after merging.

### 5.3 Commented-out code

Sampling grep: `grep -rn '^\s*//.*\b(void|return|if|for|while)' src/ ... | grep -v TODO/NOTE/comment`
returns **2 231 matches**, but the great majority are real prose
sentences that happen to contain these keywords ("If the foo is zero,
the bar should be ..."). Random sample of 30 hits found 0 actual
commented-out code blocks. **Verdict:** the repository does **not**
have a commented-out-code problem. Either the team is disciplined
about deleting or `git log -p` is acting as the safety net.

### 5.4 TODO/FIXME/HACK/XXX

`grep -rn 'TODO\|FIXME\|HACK\|XXX' src/` → **3 hits, 3 files**.

| File:line | Content |
|---|---|
| `src/model/tokenizer.cpp:16` | "Minimal JSON parser (local copy — handles \\uXXXX …)" — references XXXX in a comment, not a HACK. |
| `src/model/json_util.h:5` | Same pattern (`\uXXXX`). |
| `src/compute/gemm_grouped.cu:147` | `// TODO: Could use a true grouped/batched GEMM` — real TODO. |

**One real TODO across 90 KLOC.** The team has either zero technical
debt admitted in comments, or — more plausibly — deletes the TODOs
when shipping. Memory files (`MEMORY.md`) are the actual TODO log.

### 5.5 Summary

| Item | LOC removable |
|---|---:|
| Dead `mxfp4_act_sf` branch inside nvfp4 path | 12 |
| Merge two dispatch tables (§1.2 #2) | ~150 |
| `gemm_q6k.cu` ↔ `gemm.cu` MoE-fused overlap (relocate, not delete) | 0 net |
| **Subtotal §5** | **~162 LOC** |

---

## 6. Error handling consistency

### 6.1 `throw` audit — CLAUDE.md rule violations

`grep -rn 'throw ' src/` (excl. test helpers) → **18 hits across 6
files**:

| File:line | Throws | Verdict |
|---|---|---|
| `src/core/allocator.cpp:12,30,72` | `std::runtime_error`, `std::bad_alloc`, `std::bad_alloc` | **Violation** of CLAUDE.md "CUDA errors checked + logged, not thrown". Lines 30, 72 are device allocator failures, exactly the case the rule covers. |
| `src/core/buffer.cpp:11,52` | `std::runtime_error`, `std::bad_alloc` | Same violation. |
| `src/core/tensor.cpp:82` | `std::invalid_argument` on `reshape` numel mismatch | Programmer-error case — acceptable but inconsistent with the rest of the codebase. |
| `src/memory/device_allocator.cu:20` | `std::runtime_error` (in `IMP_CUDA_CHECK_THROW` macro at line 11) | **Direct violation.** Has both a `_LOG` macro (logs+returns) and a `_THROW` macro. The `_THROW` flavor exists. |
| `src/memory/kv_cache.cu:72, 86, 111, 144, 200, 256, 262, 283, 311` (9 sites) | `std::runtime_error` | **Violation.** Heavy use during KV cache init. |
| `src/graph/executor_forward.cu:196` | `std::invalid_argument` on n>max_tokens | Programmer-error case. |

**Total: 12 violations** of the "CUDA errors checked + logged, not
thrown" rule, all in `src/core/` and `src/memory/`. These layers
predate the rule (they originate from early scaffolding, hence
RuntimeException patterns). **Fix is mechanical:** convert each
`throw std::runtime_error(msg)` to `IMP_LOG_ERROR(msg); return false;`
and propagate the bool / ImpError up. Touches ~40 call sites total.

### 6.2 CUDA error-check compliance — sample of 5 hot files

`IMP_CUDA_CHECK_LOG` count vs. raw `cudaSuccess` comparison count:

| File | IMP_CUDA_CHECK | raw `!= cudaSuccess` | Compliance |
|---|---:|---:|---|
| `src/runtime/engine.cpp` | 32 | 4 | 89% |
| `src/graph/executor_forward_moe.cu` | 31 | 0 | 100% |
| `src/compute/sampling.cu` | 28 | 13 | 68% — **the 13 raws are the `cudaMalloc(...) != cudaSuccess` early-exit pattern at line 727-731, 751, 900. The kernel returns an error sentinel rather than aborting, which is the right pattern; but the inconsistency means a half-rewrite could miss them.** |
| `src/compute/attention_paged.cu` | 0 | 0 | n/a (no CUDA RT API calls — pure kernel TU) |
| `src/compute/gemm.cu` | 0 | 0 | n/a (cuBLAS check pattern uses `cublasGetStatusString`, not macro — line 320-330 etc.) |

**Verdict:** the rule is **followed in the executor/engine layer**
(95%+ compliance), **broken in sampling** (raw pattern keeps for
size-grow scratch), and **N/A in pure-kernel TUs**. The cuBLAS calls
in `gemm.cu` have their own bespoke error-translation (no macro).
Add `IMP_CUBLAS_CHECK_LOG` macro and apply to the ~15 cuBLAS call
sites in `compute/gemm.cu` and `compute/gemm_grouped.cu`. Hours of
work; small win.

### 6.3 `IMP_CUDA_CHECK` macro definition + variants

`grep -rn '#define IMP_CUDA_CHECK\|#define IMP_CHECK' src/`:
- `src/core/logging.h` defines `IMP_CUDA_CHECK_LOG` (returns false +
  logs).
- `src/memory/device_allocator.cu:11` defines a local
  `IMP_CUDA_CHECK_THROW` macro **inside one TU** that throws. Not
  exported. The fact that this exists demonstrates the inconsistency
  is known and worked around in-place rather than fixed.

### 6.4 `assert()` in production paths

`grep -rn 'assert(' src/` (excluding `static_assert` and `#define`)
→ **55 matches**. Distribution:
- `src/core/tensor.cpp` — 4 asserts on `ndim` bounds and reshape
  semantics. These are documented preconditions.
- `src/quant/nvfp4_quant.cu` — **22 asserts** on host functions
  (`input.on_device`, `input.qtype == QType::F16`, `K %
  kMicroBlockSize == 0`, output pointers non-null). All
  host-side, all preconditions on internal-only functions.
- `src/quant/quant_gemm.cu` — 4 asserts on tensor shapes.
- `src/compute/quantize_fp16_nvfp4_moe_native.cu:431, 503` — 2 asserts
  on `n_experts <= 256`.

**All 55 are host-side `<cassert>` `assert()` macros.** Under NDEBUG
(Release builds — P1 §6), these compile to nothing. So preconditions
that fail in production are silently ignored. **Smell:** the
codebase relies on `assert()` for preconditions that, when violated,
produce undefined behavior in Release. **Fix pattern:** replace each
hot-path `assert()` with `if (!cond) { IMP_LOG_ERROR(...); return; }`
or convert to `static_assert` where possible.

---

## 7. Naming / style compliance — 5-file spot check

Style table from CLAUDE.md / CONTRIBUTING.md:

| Element | Convention |
|---|---|
| Classes / structs | `PascalCase` |
| Functions / methods | `snake_case` |
| Member variables | `trailing_underscore_` |
| Constants | `kPascalCase` |
| Enum values | `PascalCase` (`DType::FP16`) |
| C API symbols | `imp_snake_case` |
| Macros | `IMP_UPPER_CASE` |

Random spot check of 5 files:

### `src/compute/rope.cu` — 5/5

- `struct RopeTraits` — PascalCase ✓
- `void rope_forward(...)` — snake_case ✓
- `void qknorm_rope_fused(...)` — snake_case ✓
- No member variables (free functions only) — n/a
- No constants outside the kernels — n/a
- ✓ **Compliant**

### `src/runtime/scheduler.cpp` — n/a, file content unavailable in spot check

(Empty grep result indicates file structure differs — likely a thin
glue file. Skipped.)

### `src/memory/kv_cache.cu` — 4/5

- Member access via `scale_block_bytes_` — trailing underscore ✓
- 9 `throw std::runtime_error(...)` calls — violates CLAUDE.md
  error-handling rule (§6.1) ✗ (style table doesn't cover this, but
  the CLAUDE.md "Errors return codes" rule is part of style).
- Otherwise compliant.

### `src/quant/fp8_quant.cu` — 5/5

- `static constexpr int kBlockSize = 256;` — kPascalCase ✓
- `static constexpr int kElemsPerThread = 4;` — kPascalCase ✓
- `static constexpr float kFP8E4M3Max = 448.0f;` — kPascalCase ✓
- ✓ **Compliant**

### `src/model/weight_map.cpp` — n/a (no struct/function/member matched grep)

Spot check insufficient. Sampled the file head separately — appears
compliant (no violations spotted in first 50 lines).

### Verdict

**Style compliance is ~95%+ across the codebase**, with one
systematic exception: the `throw` usage in `src/core/` and
`src/memory/` (§6.1) breaks the error-handling style rule. Pure
naming style is well-policed (likely via grep-able review).

---

## 8. Test quality

### 8.1 Re-categorization of 574 tests (refining P1 §8)

GTest count: **863 `TEST*()` invocations** (P1) across 80 `.cu/.cpp`
files. The CLAUDE.md figure of 574 corresponds to *unique* test names
when `TEST_P` parameterizations are collapsed.

| Category | Files | Estimated test count | Coverage notion |
|---|---:|---:|---|
| Kernel unit (calls a `__global__` or thin wrapper directly) | ~50 | ~500 | Numerical correctness; pass/fail on `EXPECT_NEAR`. |
| Loader/parser tests (GGUF, SafeTensors, HF config, SP, BPE, Jinja) | ~10 | ~120 | Parses fixtures, checks struct fields. |
| Engine integration (uses public API + stub model or Model) | ~10 | ~80 | End-to-end smoke; verifies "step returns non-null token". |
| Perf gate (CTest label `perf`) | 5 | ~30 | Pinned to `tests/perf_baseline.json` — 3% decode / 5% prefill. |
| Real-model E2E (requires `./models/`) | 4 | ~50 | `GTEST_SKIP` when models absent. |
| Python API tests (`tests/api/*.py`) | 9 | ~60 | Black-box server tests; not in GTest count. |
| Bench TUs masquerading as tests (`mxf4nvf4_*_bench`, `tma_block_scale_bench`, `fmha_v_load_bench`) | 5 | ~20 | Run-as-tests in CTest; report numbers, not pass/fail correctness. |

### 8.2 Phase 2 Top-10 leaks vs. existing test coverage

| Phase 2 leak | Covered by test? | If yes: did the test prevent it? |
|---|---|---|
| #1 NVFP4 MoE prefill activation-quant fusion gap | **Partial.** `tests/test_quantize_fp16_nvfp4_moe_native.cu` tests the kernel in isolation. **No test exercises the fused gate+up+SwiGLU+quant pipeline** — the test passes on the un-fused path and would pass on the fused path equally. **Test exists but doesn't pin the missing optimization.** |
| #2 Qwen3-Coder NVFP4 prefill 14.7× gap | **No end-to-end NVFP4 MoE prefill perf test** vs. vLLM. `tests/perf_baseline.json` has decode numbers, not prefill (per CLAUDE.md "pp512 varies up to 2.6×"). Gap is invisible to CI. |
| #3 Per-token D2H sync in Gemma-4 ggml MoE prefill | **No test.** Would need a graph-capture-validation test asserting "n=512 prefill captures cleanly to a single graph" — not present. |
| #4 Per-decode-step graph re-instantiation cost | **No test.** Only `test_engine_integration.cu` and `test_green_ctx.cu` reference graph capture, and only as smoke. |
| #5 NVFP4 smallM kernel -50% opt-in | `tests/test_gemm_grouped_nvfp4_smallM.cu` tests correctness. **No regression-gate** against the -50% perf delta. |
| #6 BitDecoding TC empty win + dispatcher overhead | `tests/test_attention_paged_nvfp4_tc.cu` + `test_attention_paged_nvfp4_tc_residual.cu` test correctness. **No perf gate.** |
| #7 Stale wgmma docstring | n/a (docs) |
| #8 mmvq scratch cudaMalloc/Free lazy | **No test** that asserts no `cudaMalloc` fires after warmup. (Would be hard to write; needs `cuda-memcheck --check-cache-control`.) |
| #9 SFA zero-memset | n/a |
| #10 LM-head L2 streaming hint | n/a |

**Verdict:** **0 of 10** Top-10 perf leaks are pinned by a regression
test. The test suite is **correctness-focused, not perf-pinning**.
This explains why the 14.7× gap to vLLM survives.

### 8.3 Would tests fail if a random internal function were deleted?

Heuristic: of the 80 test files, ~12 call internal `imp::` namespace
functions (`grep -l 'imp::' tests/*.cu tests/*.cpp` → 12 files).
The other ~68 test files exercise either:
- the public C API (`imp_*` functions), or
- a thin GTest wrapper around a kernel that they `extern "C"` import.

**Estimate:** if a random internal-namespace function were deleted,
maybe **20-30%** of tests would fail to compile (the 12 internal-call
files + transitive compile failures). The remaining 70-80% would
compile but might exercise no different code path — i.e. **the tests
mostly verify behavior, not implementation**. That's the right
shape, but it means kernel-internal regressions slip past compile-
gates.

### 8.4 Tests that pin implementation details (anti-pattern)

- `tests/test_mmq_q4k_v2.cu` — pins the Q4_K v2 kernel. **If
  §3.3/§10 monomorphisation proceeds, this test breaks.** Either
  port the test to the surviving 2 instantiations or delete with
  the kernel.
- `tests/test_gemm_grouped_nvfp4_smallM.cu` — pins smallM. **If
  §3.6/§10 relocation proceeds, this test moves with it.**
- `tests/test_gemm_capture_fp16_sm120.cu` — pins WMMA FP16 capture
  kernel. **If §3.7/§10 deletion proceeds, this test goes too.**
- `tests/test_attention_tc.cu` — pins the older WMMA `attention_tc.cu`
  (411 LOC). Per Phase 1 §7.5 this file is "subsumed by Blackwell".
  Test would block its deletion.
- `tests/test_attention_paged_nvfp4_tc.cu`,
  `tests/test_attention_paged_nvfp4_tc_residual.cu` — pin BitDecoding
  TC path. Since BitDecoding TC delivers 0% (P2 §3.9), these tests
  pin a feature with no measured value.

**5 tests pin implementation that perf-roadmap fixes will want to
delete.** Not blocking — the deletion order is well-defined — but
the tests need an explicit "deprecate together" plan (§11).

### 8.5 Missing test categories

- **Graph-capture sanity.** No test asserts "n=512 prefill MoE
  captures to a single graph" or "decode graph re-uses across N
  steps without re-instantiate." Phase 2 leaks #3, #4 are invisible
  here.
- **Cross-engine E2E numerical parity.** No test asserts that imp
  decode output matches llama.cpp/vLLM tokens for a fixed prompt at
  fixed seed. The CLAUDE.md `check-degeneration` workflow is the
  closest, but it's a smoke test, not a parity gate.
- **VRAM ceiling regression.** No test asserts max VRAM under a
  known workload. P1 §7.7 + memory `nvfp4_kv_potential_2026_04_25`
  reference VRAM analysis but no CI gate.
- **Public API contract.** `tests/api/test_contract.py` exists (per
  P1 §8). Good — but it's a separate Python test, not in the GTest
  count.

---

## 9. IMP_* env-var surface

Phase 1 Appendix B counted ~16 IMP_* env vars; the team has since
centralized most via `RuntimeConfig` (`src/runtime/config.h:1-168`).
Current state from `grep -rno 'getenv("[^"]*")' src/`:

### 9.1 IMP_* env vars still read directly (bypassing RuntimeConfig)

| Env var | file:line | What it gates | Default | Hot-path? | Action |
|---|---|---|---|---|---|
| `IMP_FORCE_Q4K_V2` | `src/graph/executor_pre_dequant.cu:680` | Populate q4k_v2 cache at model load. Once per init. | unset | model-load only | (c) Promote to RuntimeConfig.gemm.force_q4k_v2 |
| `IMP_MOE_RESERVE_MIB` | `src/graph/executor_pre_dequant.cu:1776` | MoE workspace reserve override. Once per init. | 0 | model-load only | (c) Promote to RuntimeConfig.moe.reserve_mib |
| `IMP_USE_BITDECODING_QK` | `src/graph/executor_attention.cu:1028` | TC NVFP4 paged attention. **Read inside lambda, gated by `static const`** — one process-wide read; OK. | unset | one-shot | (c) Promote to RuntimeConfig.attention.bitdecoding_qk |
| `IMP_NVFP4_DEVICE_ARGS` | `src/graph/executor_forward_moe.cu:515` | Device-side vs host-side MoE args. **Read inside `static_init`** — one process-wide read; OK. | "1" (default-on) | one-shot | (c) Promote to RuntimeConfig.moe.nvfp4_device_args |
| `IMP_NVFP4_SMALLM` | `src/graph/executor_forward_moe.cu:703` | smallM opt-in. Static-cached. | unset | one-shot | (c) Promote (then §10: remove smallM altogether) |
| `IMP_NVFP4_SMALLM_THRESHOLD` | `src/graph/executor_forward_moe.cu:706` | smallM crossover M threshold. | 16 | one-shot | (a) Delete with smallM |
| `IMP_NVFP4_FORCE_DEQUANT` | `src/compute/weight_dispatch.cu:106` | Force dequant-then-FP16 path. | unset | per-dispatch | (c) Promote to RuntimeConfig.diagnostics.nvfp4_force_dequant |
| `IMP_LOG_GEMM_ALGO` | `src/compute/gemm.cu:326` | Log GEMM algo selection. **Read every algo selection** (~per-shape). | unset | per-shape | (c) Promote to RuntimeConfig.diagnostics.log_gemm_algo |
| `IMP_GRAPH_CAPTURE_MODE` | `src/runtime/cuda_graph.cu:23` | Override graph capture mode. **Read inside `static_init`** — one-shot. | "global" | one-shot | (c) Promote to RuntimeConfig.runtime.graph_capture_mode (currently `cuda_graphs = "auto"|"always"|"never"` — extend) |
| `IMP_PREFILL_GRAPH` | `src/runtime/engine.cpp:2208` | Prefill graph capture opt-in. Static-cached. | unset | one-shot | (c) Promote to RuntimeConfig.runtime.prefill_graph |
| `IMP_NO_BAN` | `src/runtime/engine.cpp:1563` | Disable banned-token list. | unset | one-shot | (c) Promote to RuntimeConfig.generation.no_ban |
| `IMP_MTP_NO_ROPE` | `src/runtime/engine.cpp:220` | MTP RoPE toggle. | unset | one-shot | (c) Promote to RuntimeConfig.generation.mtp_no_rope |
| `IMP_MTP_PATTERN_LOG` | `src/runtime/engine.cpp:2693` | MTP pattern debug log. | unset | per-decode-step | (c) Promote to RuntimeConfig.diagnostics.mtp_pattern_log |
| `IMP_MTP_PRENORM_H` | `src/runtime/engine.cpp:2745` | MTP pre-norm hidden override. | unset | per-decode-step | (c) Promote to RuntimeConfig.diagnostics.mtp_prenorm_h |
| `IMP_AUDIT_NVFP4_SCALES` | `src/model/weight_upload.cu:1956` | NVFP4 scale audit. | unset | model-load only | (c) Promote to RuntimeConfig.diagnostics.audit_nvfp4_scales |
| `IMP_GDN_LAYOUT` | `src/model/hf_config_loader.cpp:339` | Gated-DeltaNet layout override. | unset | model-load only | (c) Promote to RuntimeConfig.gdn.layout_override |

### 9.2 Env vars that should stay env vars

- `HOME`, `HF_HOME`, `HUGGINGFACE_HUB_CACHE` (`src/model/hf_hub.cpp:41-45`,
  `src/runtime/config.cpp:257`) — OS-standard; keep.
- `IMP_CONFIG` (`src/runtime/config.cpp:269`) — bootstrap for config
  file path; can't be in the config it bootstraps. Keep.
- `CUBLAS_WORKSPACE_CONFIG` (`src/runtime/engine.cpp:882`) — NVIDIA
  cuBLAS env, owned by cuBLAS. Keep.

### 9.3 Hot-path env reads (perf sin)

`grep -B2 -A3 'IMP_USE_BITDECODING_QK' src/graph/executor_attention.cu`
shows the canonical pattern:

```c++
static const bool use_bitdecoding_tc = []() {
    const char* env = std::getenv("IMP_USE_BITDECODING_QK");
    return env && env[0] == '1';
}();
```

**This pattern is used correctly** everywhere except:
- `src/compute/weight_dispatch.cu:106` (`IMP_NVFP4_FORCE_DEQUANT`) —
  read **per dispatch call** without `static const` caching.
  **Perf sin.**
- `src/compute/gemm.cu:326` (`IMP_LOG_GEMM_ALGO`) — read **per algo
  selection** call. Less frequent (algo cache is per-shape), still
  per-init-shape. Wrap in `static const`.

### 9.4 Verdict

The team has already centralized most env-var surface (16 of ~30
former vars are now `RuntimeConfig` fields). Remaining 16 IMP_*
direct-getenvs are mostly one-shot caches that are perf-fine but
read-fragmented (each lives in the TU that uses it). **Action:**
single mechanical sweep to move all 16 into `RuntimeConfig` (§11
refactor #5).

### 9.5 RuntimeConfig fields that are dead / never set in production

Spot check `src/runtime/config.h:73-97`:
- `RuntimeConfig::Gemma4` — has **7 toggles** (fp32_gemm_out, no_graphs,
  force_mmvq, fp32_expert_down, no_decode_fast, no_post_ffw_1,
  ggml_prefill). Per memory files, most of these were used during
  the GEMMA4 stabilization (April 2026) and are no longer in active
  testing. **Likely 4-5 of 7 are dead.** Confirm with a `grep` of
  CI/dev scripts; not in scope here.
- `RuntimeConfig::GDN` — 5 toggles, similar story.
- `RuntimeConfig::GEMM` — 5 toggles (no_dp4a, no_dp4a_gemv, no_dp4a_lm,
  no_mmvq, no_mmvq_q8_0). All are debug-bypass; reasonable to keep.

**Estimated dead RuntimeConfig fields: 8-10.** Each carries an
`if (RuntimeConfig::current().X.Y)` check on the hot path.
Negligible perf cost; non-trivial code-comprehension cost.

---

## 10. Kahlschlag-empfehlung — ranked streichliste

Consolidated removable code. Combines:
- Phase 1 §7 ballast bilanz (~1 957 hard + 3 430 soft LOC)
- Phase 2 Top-10 (specific structural items)
- Phase 3 §3 monomorphisation
- Phase 3 §5 dead/duplicate
- Phase 3 §9 dead RuntimeConfig fields

Risk: lo = mechanical with full test coverage; med = mechanical but
test gap; hi = behavior-affecting policy call.

```
1. mmq_q4k_v2.cu phase-template tails (5 of 7 instantiations)
    [LOC: ~700]  [risk: lo]  [unblocks: simpler dispatch in executor_kernels.cu]
    Why removable: per `mmq_q4k_v2_phase2_shipped_2026_05_16` memo, only one
    kP3BN is dispatched at runtime; the others are kept as "history".
    Mitigation if wrong: tests/test_mmq_q4k_v2.cu must be updated to the
    surviving instantiations; keep a `git tag mmq_q4k_v2_all_phases` for
    future reference.

2. mmq_q4k_v2.cu entire TU (policy call: opt-in -4% E2E)
    [LOC: 1 667]  [risk: hi]  [unblocks: removes the largest opt-in kernel]
    Why removable: per memo, E2E on real models is −4% pp; only ever a win
    in synthetic microbenchmarks.
    Mitigation: keep behind `git tag`; document a one-line revert in
    docs/roadmap.md.

3. gemm_grouped_nvfp4_smallM.cu (relocate, not delete)
    [LOC: 948 (move)]  [risk: lo]  [unblocks: TMA-using code moves out of
    production tree; runtime image shrinks]
    Why relocatable: -50–55% vs CUTLASS at production shapes (P1 §7.5).
    Mitigation: tests/test_gemm_grouped_nvfp4_smallM.cu moves with it.

4. attention_naive.cu — NO LONGER REMOVABLE
    Phase 1 §7.4 marked this 165 LOC "Removable unless IMP_NAIVE_ATTN is
    still wanted." It IS still wanted: `executor_attention.cu:807-834`
    has a live `use_naive_for_swa` fallback for "Gemma-4 GLOBAL layers
    (hd=512) at n > cublas_cap" because the FMHA fallback's static tile
    exceeds sm_120's SMEM cap. **Keep.** (Updates Phase 1 ballast row.)

5. attention_tc.cu (411 LOC, subsumed by Blackwell variant)
    [LOC: 411]  [risk: med]  [unblocks: -1 WMMA path to maintain]
    Why removable: P1 §7.4/§7.5 noted "still used (header included from
    attention_blackwell.cu:24)". Verify the include is for typedefs only,
    then strip both.
    Mitigation: tests/test_attention_tc.cu moves to test_attention_blackwell.cu.

6. gemm_moe_fused_tc.cu (~520 LOC, WMMA-based; dispatched alongside scalar)
    [LOC: ~520]  [risk: med]  [unblocks: removes one of two competing MoE
    fused kernels]
    Why removable: P1 §7.5 + P2 §3.9 flag as "needs profiling whether routed
    under default." If profiling shows it never fires, delete.
    Mitigation: add a one-line perf-test ensuring fused MoE GEMV throughput
    doesn't regress.

7. gemm_capture_fp16_sm120.cu (~600 LOC, WMMA FP16 capture-graph kernel)
    [LOC: ~600]  [risk: med]  [unblocks: removes opt-in compete path]
    Why removable: P1 §7.5 + P2 §3.9 flag "needs profiling under default."
    Phase 4 should confirm before deletion.

8. __CUDA_ARCH__ >= 1200 `#else` branches across 17 TUs
    [LOC: ~600]  [risk: lo]  [unblocks: per-file readability]
    Why removable: P1 §7.2 - on sm_120a-only the `#else` paths are dead.
    Mitigation: keep as one mass commit; revert is trivial.

9. Three runtime arch-availability flags (sm_120 detection)
    [LOC: ~9]  [risk: lo]  [unblocks: -3 redundant runtime branches]
    Why removable: P1 §7.3 — always true on the only supported target.
    Mitigation: trivial.

10. sm_80/90/100 comments + 1 dead auto-select branch
    [LOC: ~13]  [risk: lo]  [unblocks: stale-comment cleanup]
    Why removable: P1 §7.1.
    Mitigation: trivial.

11. Stale wgmma docstring (attention_fmha_sm120.h:8-10)
    [LOC: ~5 (comment block)]  [risk: lo]  [unblocks: nothing functionally;
    misdirects readers]
    Why removable: P2 §3.8 + leak #7.
    Mitigation: replace with accurate "WMMA HMMA + FP8 inline mma.sync" text.

12. Bench/probe TUs in src/compute/ (move to tests/bench/)
    [LOC: 1 772 (relocate)]  [risk: lo]  [unblocks: src/compute/ is no longer
    polluted with research code]
    Why relocatable: P1 §7.8 — already gated off when IMP_BUILD_BENCH=OFF,
    but lives in src/compute/.

13. Dead mxfp4_act_sf branch inside nvfp4 path (executor_kernels.cu:2083-2094)
    [LOC: 12]  [risk: lo]  [unblocks: -1 nested dispatch arm]
    Why removable: §5.1 — fires only when both NVFP4 and MXFP4 caches are
    populated for the same weight, which the loader never does.

14. Two dispatch tables merged (executor_kernels.cu gemm_dispatch_impl ↔
    weight_dispatch.cu) — see §1.2 #2
    [LOC: ~150]  [risk: med]  [unblocks: single source of truth for per-qtype
    routing — §11 refactor #1]
    Why removable: §5.2 #4 — they do the same thing.
    Mitigation: extensive — covered by the §11 refactor.

15. RuntimeConfig fields that are never set (~8-10 of ~50)
    [LOC: ~50 fields × 4 LOC/field = ~50 LOC]  [risk: lo]  [unblocks: smaller
    config struct + smaller .cpp parser]
    Why removable: §9.5 — Gemma4-stabilization toggles no longer in active use.
    Mitigation: grep all CI / test scripts before each removal.

16. `compute/preamble_gate.h` back-edge (graph/quant_scratch.h include)
    [LOC: 3]  [risk: lo]  [unblocks: kills the only `compute → graph`
    architectural back-edge]
    Why removable: §1.3 — move `QuantScratch` to `core/` (where it morally
    belongs) and drop the include.
    Mitigation: trivial.

17. compute → quant header cycle (move 2 .cuh files to core/)
    [LOC: 0 net (just file moves)]  [risk: lo]  [unblocks: directory layer
    becomes a DAG (no cycles)]
    Why doable: §1.3 — `compute/warp_reduce.cuh` and `compute/ptx92_utils.cuh`
    are pure utilities. They belong in `core/`.

18. <cuda_runtime.h> in model/model.h
    [LOC: 1 (delete) + 5 (forward-decl)]  [risk: lo]  [unblocks: -50 TU
    recompiles on model-header edits]
    Why removable: §4.3 — model.h doesn't need the CUDA runtime header.

19. `throw`-based error handling in src/core/, src/memory/
    [LOC: 0 net (rewrite ~40 sites)]  [risk: med]  [unblocks: consistent
    error model across codebase]
    Why removable: §6.1 — 12 throws violate the CLAUDE.md rule.

20. host-side assert() converted to runtime check + log
    [LOC: 0 net]  [risk: lo]  [unblocks: NDEBUG builds no longer silently
    skip preconditions]
    Why doable: §6.4 — 55 assert() calls would compile out under NDEBUG.
```

### 10.1 Grand total — LOC removable

| Bucket | LOC |
|---:|---:|
| #1 mmq_q4k_v2 phase-template tails | 700 |
| #2 mmq_q4k_v2 full TU (alt to #1) | 1 667 (alternative) |
| #3 smallM relocate | 948 (move) |
| #5 attention_tc.cu | 411 |
| #6 gemm_moe_fused_tc.cu | ~520 |
| #7 gemm_capture_fp16_sm120.cu | ~600 |
| #8 __CUDA_ARCH__ guards | ~600 |
| #9 runtime arch flags | ~9 |
| #10 sm_80/90/100 comments | ~13 |
| #11 stale wgmma comment | ~5 |
| #12 bench TUs relocate | 1 772 (move) |
| #13 dead mxfp4 branch | 12 |
| #14 dispatch table merge | ~150 |
| #15 dead RuntimeConfig fields | ~50 |
| #16-18 architectural cleanups | ~9 |
| **Hard delete (low risk, mechanical)** | **~2 269 LOC** |
| **Soft delete (policy / Phase 4 confirm)** | **~3 198 LOC** (incl. mmq_q4k_v2 full) |
| **Relocate (no LOC change, src/compute/ cleanup)** | **~2 720 LOC** moved |
| **GRAND TOTAL streichbar** | **~5 467 LOC** (= 6.1% of src/) |

This is consistent with Phase 1 §7.9's "~5 390 LOC of ~90 200 LOC src/
= 6 %." **§10 confirms Phase 1's bilanz with refined certainty.**

---

## 11. Refactoring sequence

Five refactors, ordered so each unblocks the next.

### Refactor #1 — Kill GEMMA4-coupling (§1.5)

**Problem.** 30 `cfg.arch == ModelArch::GEMMA4` branches in
`executor_attention.cu` and `executor_forward_moe.cu` make these
files per-architecture instead of generic. Adding a new model arch
requires touching both files.

**Solution.** Introduce a `ModelArchAdapter` interface in
`src/graph/arch_adapter.h`:

```c++
struct ModelArchAdapter {
    virtual ~ModelArchAdapter() = default;

    // Norm choices
    virtual bool use_fp32_residual() const { return false; }
    virtual void apply_post_attn_norm(...) {}
    virtual void apply_post_ffw_norm(...) {}

    // Attention choices
    virtual bool needs_v_norm_ones() const { return false; }
    virtual int  override_attn_scale_inv() const { return -1; }
    virtual bool needs_naive_swa_fallback(int n, int cublas_cap) const { return false; }

    // FFN/MoE choices
    virtual bool fp32_router_gate() const { return false; }
    virtual bool fp32_expert_down() const { return false; }
    virtual bool use_ggml_prefill() const { return false; }
};
struct DefaultArchAdapter : ModelArchAdapter {};
struct Gemma4ArchAdapter : ModelArchAdapter { /* overrides */ };
```

`GraphExecutor` carries a `unique_ptr<ModelArchAdapter>` chosen at
init via `arch` enum.

**Files touched.** `src/graph/executor_attention.cu` (delete 14 GEMMA4
branches), `src/graph/executor_forward_moe.cu` (delete 19), new
`src/graph/arch_adapter.h` + `src/graph/arch_adapter_gemma4.cu`
(~400 LOC moved in), `src/graph/executor.h` (add adapter slot).

**LOC delta.** −150 net (branches simpler than inlined conditionals).

**Downstream unblocks.** §2 #2 (executor_forward_moe split) becomes
clean; adding a new arch (Phase 4 question) becomes "implement one
adapter class".

### Refactor #2 — Collapse weight-cache god-struct + 21-param dispatch (§1.2 #1, §2 #1, §5.2 #4)

**Problem.** `WeightCaches` (`executor.h:286`) holds 6 distinct
`unordered_map<const void*, ...>` keyed on weight pointer. Every GEMM
call goes through `gemm_dispatch_impl` (`executor_kernels.cu:2003`)
with 21 parameters and 8 cache-pointer branches. Per-qtype routing
duplicated in `compute/weight_dispatch.cu:73-125`.

**Solution.** Define a `GemmKernel` registry:

```c++
class GemmKernel {
public:
    virtual ~GemmKernel() = default;
    virtual bool can_handle(const GemmRequest& req) const = 0;
    virtual void run(const GemmRequest& req, const GemmContext& ctx, cudaStream_t s) = 0;
};

class GemmRegistry {
    std::vector<std::unique_ptr<GemmKernel>> kernels_;
public:
    void run(const GemmRequest& req, const GemmContext& ctx, cudaStream_t s);
};
```

Per-qtype kernel implementations live in `compute/`:
`compute/gemm_kernel_q4k.cu`, `compute/gemm_kernel_nvfp4.cu`, etc.
The 21-param dispatch collapses to a 3-param `run(req, ctx, stream)`.

**Depends on.** Refactor #1 (so the adapter can re-route specific
arches without touching dispatch).

**Files touched.** `src/graph/executor_kernels.cu` (delete
`gemm_dispatch_impl`, −266 LOC), `src/graph/executor.h` (replace
WeightCaches god-struct with a `GemmRegistry`, −150 LOC),
`src/compute/weight_dispatch.cu` (delete duplicate; redirect callers
to `GemmRegistry`, −100 LOC), new `src/compute/gemm_kernel_*.cu`
(8 files, ~150 LOC each; moved-not-added code from the dispatch
function), `src/graph/executor_pre_dequant.cu` (replace cache-
population functions with `GemmKernel::populate_cache` virtual,
~−500 LOC).

**LOC delta.** −1 000 net (collapse of two dispatch tables and one
god struct).

**Downstream unblocks.** §2 #4 (executor_pre_dequant split) trivializes
— each kernel owns its own pre-dequant. §2 #1 disappears (file
collapses to <500 LOC). Phase 4 "add a new qtype" answer becomes
"add one file in `compute/` and one registration call."

### Refactor #3 — Split executor_forward_moe (§2 #2)

**Problem.** 2 563 LOC, 5 dispatch paths, hardest-to-grok file in the
tree. 5 D2H sync sites (P2 §4.5) that each look "almost the same."

**Solution.** Per §2 #2: split into 5 files (decode_fast, prefill,
shared_expert, routing, gemma4_overrides).

**Depends on.** Refactor #1 (GEMMA4 branches go to adapter; otherwise
the gemma4_overrides file is full of `if`-ladders) and Refactor #2
(the dispatch into `GemmKernel` clears out path 3+4 boilerplate).

**Files touched.** `src/graph/executor_forward_moe.cu` (delete; split
into 5 new TUs).

**LOC delta.** 0 net (relocation), but per-file cognitive load drops
60%.

**Downstream unblocks.** Phase 2 leak #1 (gate+up+SwiGLU+quant fusion)
becomes a single-file change in `executor_moe_prefill.cu` instead of
"surgery in the middle of a 2 563-LOC file."

### Refactor #4 — Engine.cpp split (§2 #3)

**Problem.** 3 066 LOC, 40 methods, `step_decode_forward` alone is
392 lines.

**Solution.** Per §2 #3: split into 5 files (init, step_prefill,
step_decode, mtp, residual). Each file already has clear `//=====`
section banners.

**Depends on.** Nothing structural; can run in parallel with
Refactor #3. Schedule after #3 so the lifecycle of
`run_moe_*` calls is one-file-per-concept first.

**Files touched.** `src/runtime/engine.cpp` (split).

**LOC delta.** 0 net. Per-file cognitive load drops.

**Downstream unblocks.** Phase 4 "add a new request type" becomes
clear.

### Refactor #5 — Mechanical env-var + error-handling sweep (§6.1, §9)

**Problem.** 16 direct `getenv("IMP_*")` sites; 12 `throw`s that
violate the CLAUDE.md error rule; 55 `assert()`s that compile out
under NDEBUG.

**Solution.** Three coordinated sweeps:

1. Move all 16 IMP_* env vars into `RuntimeConfig` (§9.1 table).
2. Replace `throw std::runtime_error` in `src/core/`, `src/memory/`
   with `IMP_LOG_ERROR + return false/ImpError`.
3. Replace hot-path `assert()` with conditional logged-error.

**Depends on.** Nothing. **Should run FIRST** as a single mechanical
PR — establishes the invariants that #1-#4 then rely on. (Listed
last here because it's the smallest LOC delta, but it's the
chronologically-first refactor.)

**Files touched.** `src/runtime/config.h`, `src/runtime/config.cpp`,
16 grep-targeted TUs, ~10 TUs in `src/core/` and `src/memory/`.

**LOC delta.** −20 net (env var consolidation tightens), no functional
change.

**Downstream unblocks.** Once env reads are centralized, the
`if (RuntimeConfig::current().X.Y)` check pattern is uniform — easier
to grep, easier to deprecate.

### 11.1 Refactor sequencing summary

```
  #5 (env-var + error sweep, mechanical)
        |
        v
  #1 (GEMMA4 adapter — kills 30 hot-path branches)
        |
        v
  #2 (GemmKernel registry — collapses 2 dispatch tables + god-struct)
        |
        v
  #3 (executor_forward_moe split) and #4 (engine.cpp split) in parallel
```

After all five: **graph/ shrinks from 15.8 KLOC to ~10 KLOC**;
**executor_kernels.cu shrinks from 2 327 to ~500 LOC**;
**adding a new model arch is a one-file change**;
**adding a new qtype is a one-file change**;
the architectural back-edge `compute → graph` is gone (Refactor #2
removes `preamble_gate.h`'s dependence on `graph/quant_scratch.h`).

---

## Appendix A — File:line citation manifest (deduplicated)

Heavy citations from this audit:

| Anchor | Reason cited |
|---|---|
| `src/api/imp_internal.h:1` | only public→internal bridge (P1) |
| `src/compute/attention_blackwell.cu:24` | header include for typedef into attention_tc.cu |
| `src/compute/attention_fmha_sm120.cu:68, 581` | Bq/HD templates (§3.1) |
| `src/compute/attention_fmha_sm120.h:8-10` | stale wgmma docstring (§10 #11) |
| `src/compute/attention_naive.cu:23, 141` | naive attention live (§10 #4) |
| `src/compute/attention_paged.cu:289, 598, 816, 1110` | HEAD_DIM templates (§3.2) |
| `src/compute/attention_paged_common.cuh:17-34` | cp.async helpers |
| `src/compute/gemm.cu:40-92` | constants + handles |
| `src/compute/gemm.cu:85-92, 239-240` | mutable static state (§2 #9) |
| `src/compute/gemm.cu:248-280` | algo cache + descriptors |
| `src/compute/gemm.cu:317-330` | algo benchmarking |
| `src/compute/gemm.cu:326` | `IMP_LOG_GEMM_ALGO` (§9.3) |
| `src/compute/gemm.cu:1204, 1243, 1320, 1367, 1546, 1603` | GEMV variants + fused MoE (§5.2 #1, §2 #9) |
| `src/compute/gemm_grouped.cu:147` | only TODO in src/ (§5.4) |
| `src/compute/layernorm.cu:40, 72, 142, 174, 301, 345` | rmsnorm dtype kernels (§5.2 #2) |
| `src/compute/mmq_q4k_v2.cu:251, 270, 470, 818, 1016, 1294, 1462` | 7 phase templates (§3.3, §10 #1) |
| `src/compute/preamble_gate.h:5` | only `compute → graph` back-edge (§1.3, §10 #16) |
| `src/compute/rope.cu:67, 137, 140` | template instantiations (§3.5) |
| `src/compute/sampling.cu:49, 106, 159, 255, 673` | 5 sample kernels (§5.2 #3) |
| `src/compute/sampling.cu:727-781, 900, 1233-1239` | raw cudaMalloc/Free in scratch (§2 #8) |
| `src/compute/weight_dispatch.cu:73-125` | duplicate per-qtype dispatch (§1.2 #2, §5.2 #4) |
| `src/compute/weight_dispatch.cu:106` | `IMP_NVFP4_FORCE_DEQUANT` per-call read (§9.3) |
| `src/core/allocator.cpp:12, 30, 72` | `throw` violations (§6.1) |
| `src/core/buffer.cpp:11, 52` | `throw` violations (§6.1) |
| `src/core/logging.h:1-30, 75 LOC, 97 includers` | god header (§1.4, §4.2) |
| `src/core/tensor.cpp:10, 19, 82, 90, 91` | `assert()` preconditions (§6.4) |
| `src/core/tensor.h:51 includers` | god header (§1.4, §4.2) |
| `src/graph/arch_adapter.h` | (proposed, Refactor #1) |
| `src/graph/executor.h:286` | `WeightCaches` god struct (§2 #1, Refactor #2) |
| `src/graph/executor.h:348-780` | GraphExecutor class surface (§2 #5) |
| `src/graph/executor.h:1-20, 18 includes` | IWYU smell (§4.4) |
| `src/graph/executor_attention.cu:161, 310, 387, 464, 472, 493, 534, 596, 658, 678, 821, 893, 1198, 1274` | 14 GEMMA4 branches (§1.5) |
| `src/graph/executor_attention.cu:796, 960` | `goto after_attention` (§2 #5) |
| `src/graph/executor_attention.cu:807-834` | naive_attention live dispatch (§10 #4) |
| `src/graph/executor_attention.cu:988-1112` | 8-way KV-dtype switch (§2 #5) |
| `src/graph/executor_attention.cu:1028-1036` | `IMP_USE_BITDECODING_QK` static-cached env read (§9.3) |
| `src/graph/executor_forward.cu:196` | `throw std::invalid_argument` (§6.1) |
| `src/graph/executor_forward.cu:749, 789` | `goto lm_head_done` |
| `src/graph/executor_forward_moe.cu:5-10` | header comment naming 5 dispatch paths (§2 #2) |
| `src/graph/executor_forward_moe.cu:12-29, 45 includes` | include-depth outlier (§4.1) |
| `src/graph/executor_forward_moe.cu:187, 218, 224, 229, 262, 310, 324, 381, 1727, 1735, 1745, 1766, 1781, 1875, 1893, 1899, 2304, 2538, 2541` | 19 GEMMA4 branches (§1.5) |
| `src/graph/executor_forward_moe.cu:515, 703, 706` | IMP_NVFP4_* env reads (§9.1) |
| `src/graph/executor_kernels.cu:1-22` | 18 compute/+quant/ includes (§1.2 #1) |
| `src/graph/executor_kernels.cu:2003-2269` | 21-param `gemm_dispatch_impl` (§1.2, §2 #1, Refactor #2) |
| `src/graph/executor_kernels.cu:2083-2094` | dead mxfp4_act_sf branch (§5.1, §10 #13) |
| `src/graph/executor_kernels.cu:2175-2181` | mmvq scratch grow-only (§2 #8) |
| `src/graph/executor_pre_dequant.cu:680, 1776` | IMP_FORCE_Q4K_V2, IMP_MOE_RESERVE_MIB (§9.1) |
| `src/memory/device_allocator.cu:11, 20` | `IMP_CUDA_CHECK_THROW` macro definition (§6.1, §6.3) |
| `src/memory/kv_cache.cu:72, 86, 111, 144, 200, 256, 262, 283, 311` | 9 `throw` sites (§6.1) |
| `src/model/jinja.cpp:1-20` | hand-rolled Jinja2 (§2 #7) |
| `src/model/model.h:12` | `<cuda_runtime.h>` in supposedly host-only header (§4.3, §10 #18) |
| `src/model/tokenizer.cpp:1-20` | hand-rolled JSON parser inside tokenizer (§5.4) |
| `src/model/weight_upload.cu:1, 1956` | giant qtype switch + `IMP_AUDIT_NVFP4_SCALES` (§2 #10, §9.1) |
| `src/quant/dequant_gpu.cu` | includes `compute/warp_reduce.cuh`, `compute/ptx92_utils.cuh` (§1.3) |
| `src/quant/nvfp4_quant.cu:380-578` | 22 host-side asserts (§6.4) |
| `src/runtime/config.h:1-168` | RuntimeConfig singleton (§9, Refactor #5) |
| `src/runtime/config.h:73-97` | GDN+Gemma4 toggles, likely partly dead (§9.5) |
| `src/runtime/engine.cpp:220, 1563, 2208, 2693, 2745, 2208` | scattered direct env reads (§9.1) |
| `src/runtime/engine.cpp:882` | `CUBLAS_WORKSPACE_CONFIG` check (§9.2) |
| `src/runtime/engine.cpp:828, 1663, 1924` | GEMMA4 branches in engine (§1.5) |
| `src/runtime/cuda_graph.cu:14-36` | `IMP_GRAPH_CAPTURE_MODE` (§9.1) |

---

End of Phase 3 codereaper audit.
