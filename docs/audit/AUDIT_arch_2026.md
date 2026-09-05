<!--
layer: L3
audience: agents
verified: 2026-09-05
commit: ef664dd8
-->

# AUDIT_arch_2026 - Architecture audit, 2026 baseline

Record. Append-only, true of the day it names (`docs/audit/README.md` convention). Read-only pass at `ef664dd8` (branch `perf/engine-h-fanin-cut-and-attention-split-verdict`, clean tree, one commit ahead of `main` at #1907). No GPU job was run; every perf statement cites an existing artifact or is marked HYPOTHESIS. No source file was edited. No PR.

Roles: `scout` (P0 inventory + one report per axis, 12 reports), `profiler` (existing artifacts only: `tools/roofline/history/`, `docs/PERF.md`, `docs/audit/PERF_LOG.md`, `docs/roadmap.md` lever ledger), `validator` (P2: every S0/S1 re-read against the code with the opposite hypothesis, by the coordinating session). Dispatch brief: the "DISPATCH: Architecture Audit - imp (2026 baseline)" text of 2026-09-05.

## How to read this file

| section | what it holds |
|---|---|
| P0 | module map, dependency direction, cycles, file-size ridership, churn, CI shape |
| P1 | twelve axis reports, verbatim from the scouts (A split in A1 dispatch coverage / A2 hardware features; F split in F1 parsers / F2 HTTP) |
| P2 | falsification pass: verdict per S0/S1, the kill list (downgraded or removed) and the merges |
| P3 | ranked dispatch queue, dependency-ordered, one dispatch per item, the three best value-to-blast-radius items marked |
| P4 | known-and-accepted, restated |

Finding IDs are `<axis>-<n>` and stable across sections. Severities in P1 are the scout's claim; the surviving severity is the one in P2/P3.

## Verify the brief before you generate from it (SETTLED F, applied to this dispatch)

The dispatch brief carried stale or wrong premises. Each was checked against the tree before any hypothesis was generated from it.

| brief said | repo | evidence |
|---|---|---|
| "C++20" (fixed constraint) | C++23 | `CMakeLists.txt:4-7` |
| "`/v1/messages` streaming is synthetic (known)" | real per-token stream over the shared driver | `tools/imp-server/handlers_messages.cpp:56-62,287`, `stream_driver.cpp:241` (axis E) |
| "no p50/p99 histograms"-class expectation | `imp_ttft_seconds`, `imp_inter_token_seconds`, `imp_request_duration_seconds` histograms exist; `/v1/completions` got them in #1896 | `handlers_chat.cpp:275-590`, `tests/test_server_metrics.py:67-85` (axis E) |
| "cuBLAS autotuning nondeterminism, up to 2.6x prefill spread; is there a persist/pin mechanism" | magnitude never measured; instability was in near-ties; estimator fixed #1228, 7/8 shapes stable; persisting (R-16) REJECTED with two refuted designs | SETTLED G F-9; `src/compute/gemm.cu` matches verbatim (`kBenchmarkRounds=3`, `kTargetWindowMs=0.5`, `kAlgoMargin=0.10`) (axis A2) |
| "legacy `causal_softmax` + cuBLAS fallback is the known MoE-prefill lever" | 0.0 % on hd 128/256; the tier is Gemma-4 hd=512 by design (S-8) | axis A1 |
| "spec decode per-request toggle (known gap)" | exists: `"speculative": true/false`, tri-state `Request::spec_override` | `src/runtime/request.h:96-101` (axis C) |
| "`graph_max_*` keys" | no such keys; decode graph pool `kMaxGraphPoolSize = 64` keyed by `n_sequences-1` | `src/runtime/engine.h:471-472` (axis C) |
| "paged KV block size 16, `kv_cache.block_size`" | no config key, no CLI flag; compile-time 16 since the initial commit, resolver picks 32 for `n_kv_heads <= 4` | axis B, J-1 |
| "MoE atomics nondeterminism (known)" | default F16 MoE scatter is atomic-free; CUTLASS grouped GEMM has no split-K/atomics; the measured default-mode drift was cuBLASLt algo timing on a dense hybrid | axis D |
| "per-key rate limit" (also `docs/roadmap.md:98`) | per peer IP, one shared API key | `tools/imp-server/rate_limit.cpp:13-29` (axis E) |
| "`n`/`best_of`/`logprobs` unbounded, JSON depth, constant-time key compare, key/prompt logging, `max_tokens` unbounded" | all already closed: `--max-n`, `best_of>1` refused, depth cap 100 non-recursive (#1607), constant-time both headers, no key/prompt log lines, `max_tokens` clamped to remaining context | axis F2 |
| "LoRA body field is a path" | name only, resolved in a map filled from `--lora NAME=PATH` | `handlers_chat_params.cpp:516` (axis F2) |
| memory notes "cudaGraphInstantiate 10-44 ms/req", "prefix-cache blocks counted as used", "cuda_graphs=never dead on dense", "#1897 19 MiB alloc under capture" | all CLOSED (#1895, #1879, `engine_weight_upload.cpp:291-323`, `engine.cpp:851-853`) | axis B, C |
| "93.6 % floats differ" (default mode not deterministic) | measured on `--calibrate` statistics files, PPL identical both runs; cause was cuBLASLt algo timing, fixed | axis D |
| "compute-sanitizer / ASan / UBSan run anywhere in CI" | ASan/UBSan: yes, `Sanitizers` job, path-gated, not required. compute-sanitizer: three routes, all dead (GPU lane dormant; `make sanitize` fails on WSL2; `verify.sh` never calls it) | axis F2, D |
| "TEST_INVENTORY.md numbers" (2026-08-08) | month-old baseline; 4 headline claims now false (2701 macros / 1617 laned / 1084 unlaned today) | axis I |

## Coverage statement

Per axis, from each report's own Coverage section (full lists in P1). "Full" = every line of the named files opened; "sampled" = ranges opened and the cited lines verified; "swept" = grep across the tree with every hit inspected.

| axis | full | sampled / swept | skipped |
|---|---|---|---|
| A1 kernels: dispatch | SETTLED, `src/compute/CLAUDE.md`, `SM120.md`, `dispatch_paths.{h,cpp}`, `dispatch_record.h`, `attention_dispatch.cu`, `attention_dispatch_decision.h`, `executor_gemm_dispatch.cu` (700), `executor_attention_prefill.cu` (469), `moe_prefill_decision.h`, `gemm_kernel_registry.{h,cu}` + all nine `gemm_kernel_*.cu` registration blocks, `test_attention_dispatch_rules.cpp` | `executor_forward_moe*.cu`, FA2 / blackwell / cublas / q4k / grouped / cutlass / dp4a / mmq / mxfp4 kernels by range; `executor_workspace*.cu`, `engine_scheduler.cpp`, `engine_init_resolver.cpp` by range; config defaults; roofline run `1d5b9230_20260831_180644` (aggregated) | kernel numerics, `tests/` beyond the rules test |
| A2 kernels: hardware features | `src/compute/CLAUDE.md`, `SM120.md`, SETTLED, `kernel_resource_baseline.txt`, `pdl.h`, `pdl_device.cuh`, `KERNELS.md` 5-7, smallM kernel | `gemm.cu`, `gdn.cu`, FA2, paged fp8/nvfp4 by range; grep sweeps for TMA/PDL/`__launch_bounds__`/`mbarrier`/`cp.async`/SKU names, ~200 PDL hits, 120 launch-bounds hits | `third_party/`, CUTLASS-generated, vision kernels, constrainer kernels, `tools/standalone/` |
| B memory & KV | `AUDIT.md`, `MEMORY.md`, SETTLED D-G, `kv_cache.h`, `kv_cache_manager.h`, `vram_owned.h`, `cuda_static_reset.h`, alloc + log-fatal allowlists | `kv_cache_manager.cpp`, `engine_kv_cache_init.cpp`, `engine_prefill*.cpp`, `engine_scheduler.cpp`, `engine_spec_mtp.cpp`, sampling statics; gates run: alloc sites/pairs, log-fatal | `vmm_backend`, `block_pool`, `arena`, `plan`, `scratch_stack` internals, kernel bodies, `src/quant/`, `weight_upload.cu` |
| C scheduler | `scheduler.{h,cpp}`, `graph_eligibility.{h,cpp}`, `batching_engine.h`, both CLAUDE.md, ARCHITECTURE phases 3-4 | `engine_scheduler.cpp`, `engine_prefill.cpp`, `engine_decode_pipeline.cpp`, `engine_spec_*.cpp`, `cuda_graph.cu`, handlers | kernel bodies |
| D correctness | `determinism.md`, `check_determinism_sites.py`, `check_launch_guards.py`, `cuda_raii.h`, `logging.h:55-140`, `test_capture_abort.cu`, `test_determinism_e2e.cpp`, `batching_engine.cpp` fault path | MoE/sampling/GEMM kernels by range, `executor_*`, `engine_*`, `imp_api.cpp`, ~45 test files by name listing | |
| E serving | `handlers_messages.cpp`, `stream_driver.*`, `stream_pipeline.h`, `batching_engine.*`, `main.cpp`, `rate_limit.cpp`, `metrics_memory.cpp`, `tool_stream_filter.h`, `fuzz_tool_stream.cpp`, logprobs + metrics tests | `handlers*.cpp`, `anthropic.cpp`, `reasoning_split.h` by range | `handlers_responses.cpp`, `handlers_rerank.cpp`, `image_fetch.cpp`, `tracing.cpp`, `webui/`, constrainer internals |
| F1 parsers | `gguf_parse.cpp`, `gguf_loader_internal.h`, `gguf_loader.h`, `model_limits.h`, `json_util.*`, `sentencepiece_loader.cpp`, `lora_adapter.*`, `executor_lora.cu`, `weight_cache_file.*`, `tensor.{h,cpp}`, `hf_hub.cpp` | `gguf_loader.cpp`, `safetensors_loader.cpp`, `hf_config_loader.cpp`, `tokenizer.cpp`, `jinja.cpp`, `vision_loader.cpp` by range | |
| F2 HTTP | `main.cpp`, `args.*`, `rate_limit.*`, `handlers_admin.cpp`, `image_fetch.cpp`, `fuzz/`, sanitizer supps, ci.yml sanitizer/test/tidy jobs | `utils.cpp` (nesting guard, auth, base64), `handlers_chat_params.cpp`, `handlers_chat_core.cpp` by range | |
| G architecture | ARCHITECTURE, ARCHMAP, SETTLED, 07-29 audit 11.1-11.3, DEBT_LEDGER 1-2, CONTRIBUTING "Other rules", CPP23.md, `check_function_size.py` parser, `qtype.{h,cpp}`, `imp_api_suspend.cpp` | `imp_api.cpp`, `executor_attention_{prefill,decode}.cu`, `engine_init_resolver.cpp`, `engine_scheduler.cpp`, `vram_budget.cpp`, `engine.cpp`, `model_config.h` by range; machine sweeps over 842 files / 551 TUs (include graph, reverse BFS, SCC), `rg` for virtual/throw/expected/optional/FATAL/nodiscard/getenv; control: `engine.h` reverse-BFS = 39 TUs, matching S-33 | `tests/` internals, `webui/`, `fuzz/` bodies, kernel numerics, anything with a perf claim |
| H build/CI | all 4 workflows, `dependabot.yml`, `Dockerfile`, `CMakePresets.json`, `cmake/*.cmake`, `ci_static_gates.sh`, `check_dep_pins.sh`, `bench_gate.sh`, roofline README/regress/config | `CMakeLists.txt`, `Makefile`, `verify.sh`, `check-release.sh` by range; attribution grep over `src/` | `src/**`, `tests/**`, other roofline modules, `webui` beyond bundled assets |
| I tests | `tests/CLAUDE.md`, `tests/README.md`, TEST_INVENTORY, MUTATION_BASELINE, lane/skip gates, Makefile test targets, CMake 640-1090 | test files by `rg -c '^TEST'`, `verify.sh` filters, hooks (md5 compared) | |
| J docs | docs-layers skill, `docs_lint.py`, `sync_docs.py`, `check_doc_citations.py`, `PERF.md`, `MODELS.md`, plans README, roadmap tables, ARCHITECTURE phase table, MEMORY invariant table | README, BENCHMARKS, FEATURES (10 rows), LIMITATIONS (8 items), KERNELS, `performance.md`, AGENTS, the CLAUDE.md tree; all three doc gates run | `docs/archive/`, `docs/audit/*` except README + anchors, MISSION_JOURNAL, `vram_audit.md`, `.github/` templates |
| P0 (coordinator) | `SETTLED.md`, `docs/audit/README.md`, `roadmap.md` 1-140, `LIMITATIONS.md` 1-120, `ci_static_gates.sh`, ruleset via `gh api`, allowlists, include matrix, churn, file-size and function-size gate output | ARCHITECTURE 36-80, ARCHMAP 1-60, backward-edge files opened | |

Depth caveat: this is a one-day read-only pass with no GPU job. Where a scout writes HYPOTHESIS the number does not exist yet; the P3 queue names the instrument that would produce it.

## P0 - Inventory (scout, verified 2026-09-05 at ef664dd8)

### Layers and LOC (raw `wc -l` over .cu/.cuh/.cpp/.h; code LOC per file via `tools/check_filesize.py`)

| layer | raw LOC | role (from `docs/audit/ARCHMAP.md` "Layer DAG") |
|---|---:|---|
| `src/compute` | 55112 | kernels: GEMM (cuBLASLt, CUTLASS, dp4a, IMMA), FA2, paged decode, GDN/SSM, sampling, constrainers |
| `src/exec` | 29427 | `GraphExecutor` forward pass, pre-dequant phases, GEMM registry, workspace |
| `src/runtime` | 22061 | `Engine`, scheduler, CUDA-graph decode, spec decode, `RuntimeConfig` |
| `src/model` | 21422 | loaders (GGUF, SafeTensors), tokenizer, chat template, weight map, weight upload |
| `src/memory` | 10231 | backend, arena, block pool, KV cache + manager, prefix cache, plan |
| `src/quant` | 7447 | dequant/quant kernels, NVFP4 GEMV/GEMM small-M |
| `src/vision` | 4851 | SigLIP + Qwen3-VL towers, pipeline |
| `src/core` | 3007 | Tensor, Buffer, RAII, logging, dispatch policy, config sections |
| `src/api` | 1282 | C ABI |
| `src/lora` | 328 | adapter loader |
| `tools/imp-server` | 15021 | httplib server, 3 dialects, batching engine |
| `tools/imp-quantize` | 2839 | quantizer |
| `tools/imp-cli` | 1655 | CLI |
| `tools/imp-bench` | 1282 | bench |
| `tests` | 87283 | 224 C++ test files, 2701 GTest macros (1617 in a CI lane, 1084 GPU-only) |

### Dependency direction (count of `#include "<col>/..."` lines in `src/<row>/`)

Intended DAG (`docs/audit/ARCHMAP.md:10-16`): `api -> runtime -> exec -> compute -> quant`, `exec -> memory -> core`, `runtime -> model -> core`, `vision -> compute/model`, `lora -> runtime/model`.

| includer \ includee | api | compute | core | exec | lora | memory | model | quant | runtime | vision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| api | 3 | 0 | 4 | 1 | 0 | 4 | 5 | 0 | 3 | 0 |
| compute | 0 | 201 | 124 | 0 | 0 | 16 | 7 | 16 | **31** | 0 |
| core | 0 | 0 | 26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| exec | 0 | 233 | 113 | 181 | 3 | 42 | 10 | 94 | **20** | 1 |
| lora | 0 | 0 | 2 | 0 | 1 | 0 | 1 | 0 | 0 | 0 |
| memory | 0 | 0 | 20 | 0 | 0 | 51 | **3** | 0 | **1** | 0 |
| model | 0 | 0 | 26 | **1** | 0 | 7 | 72 | 3 | **4** | 4 |
| quant | 0 | **5** | 24 | 0 | 0 | 1 | 0 | 40 | **9** | 0 |
| runtime | 0 | 46 | 46 | 12 | 2 | 46 | 21 | 0 | 105 | 5 |
| vision | 0 | 1 | 16 | 0 | 0 | 8 | 3 | 0 | **2** | 38 |

Bold = against the intended direction. Content of each backward edge (every line opened):

| edge | files | content | class |
|---|---:|---|---|
| compute -> runtime | 31 | 16x `runtime/pdl.h`, 15x `runtime/process_diag.h`; `runtime/config.h` 0 (was 2 on 2026-07-29) | instrumentation |
| exec -> runtime | 20 | `pdl.h`, `storage_planner.h`, `vram_budget.h`, `graph_diag.h`, `process_diag.h`; `runtime/config.h` 0 (was 22; F-10 closed #1227) | instrumentation + init plumbing |
| memory -> runtime | 1 | `kv_cache.cu:5` -> `runtime/graph_diag.h` | instrumentation |
| memory -> model | 3 | `layer_offload.h`, `weight_snapshot.cpp`, `weight_cache_file.cpp` operate on `Model` | real coupling (07-29 verdict stands) |
| model -> exec | 1 | `weight_upload.cu:6` -> `exec/nvfp4_expert_offload.h` | NEW since 07-29; see G |
| model -> runtime | 4 | see G | |
| quant -> compute | 5 | `nvfp4_gemm.cu:3` -> `compute/gemm.h` (the one SETTLED keeps: FP16 fallback hands off to dense GEMM); 4x `compute/pdl_device.cuh` from `nvfp4_gemv_dense.cu`, `nvfp4_gemv_batched.cu`, `nvfp4_gemm_smallm_v2.cu`, `nvfp4_gemv_fused.cu` | 1 algorithmic (settled), 4 device-helper leakage (NEW since 08-03: SETTLED says "one edge remains") |
| quant -> runtime | 9 | `pdl.h`, `process_diag.h` | instrumentation |
| vision -> runtime | 2 | | instrumentation (see G) |
| core -> anything | 0 | the 07-29 `core <-> compute` cycle is closed (#1207) | |

Cycles by layer: `exec <-> runtime` (runtime -> exec 12 intended; exec -> runtime 20 instrumentation), `compute <-> runtime` (instrumentation), `model <-> exec` (exec -> model 10 intended; model -> exec 1 new), `quant <-> compute` (compute -> quant 16 intended; quant -> compute 5). No cycle carries a dispatch decision any more: `RuntimeConfig` is read from `exec/` zero times (`rg -c 'runtime/config.h' src/exec` = 0).

### Where the file-size gate is being ridden at the limit

`python3 tools/check_filesize.py`: `violations=0`, 34 allowlisted (two-way pinned), 44 warn-level. Function gate: 10 allowlisted bodies, 49 warn. Files within 10 % of their hard line that are NOT allowlisted (the hidden-god-file candidates):

| file | code LOC | hard | group |
|---|---:|---:|---|
| `src/runtime/engine_spec_ngram.cpp` | 790 | 800 | tu |
| `src/model/chat_template.cpp` | 786 | 800 | tu |
| `tools/imp-server/handlers_chat.cpp` | 781 | 800 | tu |
| `tools/imp-server/handlers_chat_core.cpp` | 780 | 800 | tu |
| `src/api/imp_api.cpp` | 764 | 800 | tu |
| `tools/imp-server/utils.cpp` | 730 | 800 | tu |
| `src/runtime/engine_kv_cache_init.cpp` | 721 | 800 | tu |
| `src/vision/vision_encoder.cu` | 712 | 800 | tu |
| `tools/imp-server/tool_call.cpp` | 705 | 800 | tu |
| `src/exec/executor_forward_moe_cutlass.cu` | 600 | 600 | kernel (at the line) |
| `src/compute/moe_routing.cu` | 586 | 600 | kernel |
| `src/compute/sampling_topk_topp.cu` | 585 | 600 | kernel |
| `src/exec/executor_elementwise.cu` | 585 | 600 | kernel |
| `src/exec/pre_dequant_phase3_moe.cu` | 585 | 600 | kernel |
| `src/runtime/engine.h` | 629 (raw 1502) | 700 | header |

Allowlisted and still growing past their pin is impossible by construction (two-way ceiling, #1526); the allowlist itself is the god-file list: `jinja.cpp` 2407, `weight_upload.cu` 2004, `tokenizer.cpp` 1818, `attention_fmha_sm120.cu` 1556, `executor_workspace_buffers.cu` 1534 (884 in one body), `executor_attention.cu` 1279 as a TU (1223 in one body across 3 `#include`d fragments, S-32 refused the split on measurement).

### 6-month churn (commits touching the file, `git log --since=6.months`, 1909 commits)

`src/runtime/engine.cpp` 254, `engine.h` 151, `config.h` 146, `config.cpp` 146, `tools/imp-server/handlers.cpp` 116, `engine_scheduler.cpp` 95, `src/exec/executor.h` 80, `src/api/imp_api.cpp` 75, `executor_workspace_buffers.cu` 71, `weight_upload.cu` 68, `tools/imp-cli/main.cpp` 66.

### CI shape (facts, `.github/workflows/ci.yml`, ruleset "Require CI")

Required status check: `Build` only. `Build` runs `scripts/ci_static_gates.sh` (file size, function size, determinism sites, dead inline accessors, log-fatal, test lanes, entrypoint, alloc sites + pairs, launch guards, docs sync + lint, doc citations, release hygiene) before compiling, then compiles on a GPU-less container and runs `ctest -L unit`. Not required: `Sanitizers` (path-gated), `Mock API contract`, `Real API contract (model-less)`, `Lint`, `clang-tidy` (advisory), `File size`, `PTX fallback`, `docs`, `hygiene`, `alloc-sites`, `launchguards`, `testlanes`. `Test` (full ctest, compute-sanitizer, perf gate) is gated on `vars.HAS_GPU_RUNNER`, unset by owner decision (2026-08-03, SETTLED F-5): never ran. Kernels run only in `make verify-fast` (pre-push hook, local).

# P1 - Per-axis audit

Each axis section is the scout's report verbatim (coverage, brief-vs-repo, findings, negatives, known-and-accepted, open questions). P2 verdicts that changed a severity are recorded in the P2 section and NOT edited into the axis text, so the record shows what was claimed and what survived.


## Axis A1 - Compute / kernels: dispatch coverage and legacy paths

Repo `<repo>`, branch `perf/engine-h-fanin-cut-and-attention-split-verdict`, HEAD `ef664dd8`, clean. READ-ONLY: no edits, no build, no GPU job.

### Coverage

**Read in full**
- `docs/audit/SETTLED.md` (652), `src/compute/CLAUDE.md`, `docs/internals/SM120.md`, root `CLAUDE.md`.
- `src/compute/dispatch_paths.h`, `dispatch_paths.cpp` (name switches), `dispatch_record.h`, `attention_dispatch.cu`, `attention_dispatch_decision.h`.
- `src/exec/executor_gemm_dispatch.cu` (700), `executor_attention_prefill.cu` (469), `moe_prefill_decision.h`, `attention_dispatch_rules.h`, `gemm_kernel_registry.h/.cu`, all nine `src/exec/gemm_kernel_*.cu` registration blocks.
- `tests/test_attention_dispatch_rules.cpp`.

**Read in part (named ranges opened and verified)**
- `src/exec/executor_forward_moe.cu:440-625`, `executor_forward_moe_batch.cu:224-397,510-520,606-748,1024-1310`.
- `src/compute/attention_fmha_sm120.cu:440-600,1989-2035`, `attention_blackwell.cu:385-405`, `attention_cublas.cu` (kernel list), `gemm_q4k.cu:230-246,575-653`, `gemm_grouped.cu:18-201,418-423`, `gemm_cutlass_mxfp4_sm120.cu:381-438`, `gemm_cutlass_sm120.cu:780-995`, `gemm_dp4a.cu:500-700`, `mmq_q8_imma.cu` (exports), `attention_mxfp4_prefill.cu:509-525`.
- `src/exec/executor_workspace.cu:320-352`, `executor_workspace_buffers.cu:580-605`, `src/runtime/engine_scheduler.cpp:190-241`, `engine_init_resolver.cpp:200-345`.
- `src/core/config/{attention,gemm,moe}.h` (defaults only), `tools/roofline/config.json` (class regexes), `tools/roofline/history/runs/1d5b9230_20260831_180644.json` (aggregated).
- `docs/audit/gemma4_attn_routing_2026_07_16/PERF_LOG.md` entry 1, `docs/audit/DEBT_LEDGER_2026_08_21.md` (OPEN grep + items 9/10), `docs/roadmap.md` lever rows 64/128/133/177/183-185/223/228/356.

**Delegated and spot-verified** - the paged-decode enumeration (KV dtype x launcher x head_dim x split-K x multitok, sparse gate, arch-dead kernels) was produced by a read-only sub-scout. I re-verified by hand: the two `sm_ver >= 90` else-arms (`attention_paged_fp8.cu:685/721`, `attention_paged_int4.cu:574/608`), `paged_attention_serves_head_dim` (`attention_paged.cu:1196-1216`), `paged_attention_unsupported_head_dim` being `[[noreturn]] throw` (`:1190`), and the resolver that consumes it (`engine_init_resolver.cpp:305-325`).

**Skipped** - kernel bodies of `attention_fmha_mxfp4_sm120.cu`, `attention_paged*.cu`, `gdn*.cu`, `sampling*.cu`, `ssm.cu`, constrainers, `gemv_dp4a_traits.cuh` internals. No SASS, no ncu, no build.

**Helper script** - the pre-existing `scratchpad/gemm_inventory.sh` was run and its output is **unreliable as-is**: its symbol pattern is `\b<fn>\s*\(`, which misses every function used as a **function pointer**. It reported the 15-symbol `gemv_*_q8_1_moe_{decode,gate_up_fused}` family as having zero callers; they are in fact live, taken by address in the ternary chains at `src/exec/executor_forward_moe_batch.cu:1217-1223,1241-1247,1303-1306`. Every "dead" claim below was re-derived with `rg -n -w <symbol>` over `src/ tools/ tests/ include/`.

### Brief vs repo

1. **"FA2 coverage vs the legacy `causal_softmax` + cuBLAS fallback ... the dispatch brief calls this 'the known MoE-prefill lever'"** - conflation of two different kernels. The roadmap's parked lever is `gemm_cublas`, and `gemm_cublas` in the roofline classifier is the **cuBLAS/cuBLASLt dense GEMM** class (`tools/roofline/config.json`, class `gemm_cublas`, regex `nvjet|ampere_|...|cublas|cutlass::Kernel|cutlass3x`), i.e. the **GDN projection GEMM on the hybrid**, not attention. `docs/roadmap.md:64` and `:185`: "hybrid pp512 `gemm_cublas` hole - PRICED, parked: 24.8% of roofline at 21.5% share = 2-3% of hybrid pp512". Attention's legacy class is `attn_legacy_softmax` (group `attention_legacy`), and it is **absent from every cell** of the newest roofline run (below). The two are unrelated.
2. **"multi-token (spec verify) path"** as a property of the `attention_paged_*_multitok.cu` files - wrong mapping. "multitok" in those filenames means **TOK KV tokens per warp iteration** (a bandwidth optimisation), one query row per CTA: `src/compute/attention_paged_nvfp4_multitok.cu:49`, `attention_paged_f16_multitok.cu:1-17`. The real M>1 (spec-verify) decode reshapes the chunk into M single-token sequences (`src/runtime/engine_spec_ngram.cpp:724-761`) and re-enters the **ordinary** decode block (`src/exec/executor_attention.cu:527,570`), so every KV dtype gets M>1 through its own single-token launcher with `batch = M`. There is no per-dtype M>1 kernel and no per-dtype M>1 decline.
3. **"marlin remnants"** - none. `rg -i marlin` over `src/ tools/ tests/ include/ cmake/` returns 6 hits, all prose/comment (`src/quant/nvfp4_gemm_smallm.cu`, `nvfp4_gemm_smallm_v2.cu`, `src/core/config/gemm.h`, `src/model/hf_config_loader.{h,cpp}` AWQ `version` field). No Marlin kernel, no Marlin file. Consistent with the sidecar PR never merging.
4. **SETTLED S-8 is still true, and the newest number confirms it** - see "Checked and NOT a finding".
5. **`gemm.use_kernel_registry`** (named in `src/exec/gemm_kernel_registry.h:13` as the switch that gates the registry) **does not exist**: `rg -n use_kernel_registry src/ tools/ tests/ docs/` returns exactly that one comment line. The registry is unconditional today; the header's design narrative is stale.

### Axis answers (Q1-Q4)

### Q1 - GEMM/GEMV family inventory

| family | file(s) | dispatched from | selected when | verdict |
|---|---|---|---|---|
| cuBLAS/cuBLASLt dense `gemm()` / `gemm_cublaslt()` | `compute/gemm.cu` (1121) | `weight_dispatch.cu`, `executor_forward.cu`, `executor_ffn.cu`, `executor_attention*.cu`, `executor_gemm_dispatch.cu`, `mtp_forward.cu`, `encoder_forward.cu` | FP16/BF16 weights; every dequant-to-FP16 fallback; FP8 prefill (`gemm_cublaslt`) | **load-bearing** |
| `gemm_capture_fp16_sm120` | `compute/gemm_capture_fp16_sm120.cu` | `compute/gemm.cu:784` | FP16 GEMM under graph capture where cuBLASLt is not capture-safe | live |
| dp4a GEMV (`gemv_*_q8_1`, `_fp32`, `_residual`, `gemv_qkv_fused_*`) | `compute/gemm_dp4a.cu` (802) + `gemv_dp4a_traits.cuh` (1677) | registry key `{FP16, <gguf qtype>, m_is_one=true}` via `exec/gemm_kernel_gguf.cu`; `executor_kernels.cu`; `executor_gemv_helpers.h`; `executor_attention_internal.h:72-90` | GGUF decode, M=1 | **load-bearing** |
| `ggml_mmvq_*` | `compute/ggml_mmvq.cu` (642) | `exec/gemm_kernel_gguf.cu` (same registry key, mmvq backend) + `executor_forward_moe_batch.cu:680-731` | GGUF decode where mmvq beats dp4a (`gemm.no_mmvq`, `force_mmvq`) | **load-bearing** |
| `quant_gemm_int4` | `quant/quant_gemm.h` | `exec/gemm_kernel_gguf.cu` | GGUF small-M IQ4 tier | live |
| `mmq_q8_imma_gemm` (Q8_0 INT8 IMMA) | `compute/mmq_q8_imma.cu` | `executor_gemm_dispatch.cu:~530` | `gemm.q8_imma_enabled` **default true**, Q8_0 source, M>=2 | **load-bearing** |
| `mmq_imma_moe_gemm` | `compute/mmq_q8_imma.cu` | `executor_forward_moe_batch.cu` | `gemm.moe_imma_prefill` **default true** | **load-bearing** |
| `mmq_q4k_imma_gemm` | `compute/mmq_q4k_imma_tile.cu` | `executor_gemm_dispatch.cu` (`gemm.q4k_imma_prefill`, default **false**) and `exec/gemm_kernel_q4k_imma.cu` (registry key `{FP16,Q4_K,false}`, **never dispatched**, A1-1) | opt-in | config-gated off |
| `mmq_q6k_imma_gemm` | `compute/mmq_q8_imma.cu` | tests only (`tests/test_mmq_q8_imma.cu`) | - | test-only, **documented refutation** at `executor_gemm_dispatch.cu:549-556` ("4.5k vs 6.6k pp512") |
| `mmq_q4k_hmma_gemm` / `try_q4k_hmma_dispatch` | `compute/mmq_q4k_hmma.cu`, `exec/gemm_kernel_q4k_hmma.cu` | `executor_gemm_dispatch.cu:424-431` | `gemm.q4k_hmma_enabled` **default false**, M>=32 | config-gated off |
| `gemm_q4k_dp4a_dense` / `gemm_q5k_dp4a_dense` | `compute/gemm_q4k.cu` | `executor_gemm_dispatch.cu:439-467` | prefill FP16 tier, M<=64, smem fits | **load-bearing** |
| `gemm_q4k_fused_moe_prefill` (scalar) | `compute/gemm_q4k.cu:234-608,614` | tests only | - | **production-dead** (A1-8) |
| `gemm_q4k_dp4a_moe_fused`, `gemm_q5k_dp4a_moe_fused`, `gemm_q6k_moe_fused` | `compute/gemm_q4k.cu`, `gemm_q6k.cu` (240) | `executor_forward_moe_batch.cu:303-309` inside `try_run_moe_q4k_prefill` | requires `!gemm.moe_imma_prefill` | **unreachable in default config** (A1-2) |
| `gemm_q6k_fused_moe_prefill`, `_tc` | `compute/gemm_moe_fused.cu` (175), `gemm_moe_fused_tc.cu` (283) | `executor_forward_moe_batch.cu:257-291` inside `try_run_moe_q6k_prefill` | requires `!gemm.moe_imma_prefill` | **unreachable in default config** (A1-2) |
| `gemm_moe_batched` | `compute/gemm_grouped.cu:201` | `executor_forward_moe_batch.cu`, `executor_forward_moe_legacy.cu` (7 sites) | FP16/FP8 MoE batch | live |
| `imp::gemm_grouped` (host-vector grouped) | `compute/gemm_grouped.cu:168` | **nobody** | - | **dead** (A1-4) |
| CUTLASS NVFP4 dense `gemm_nvfp4_cutlass_sm120` | `compute/gemm_cutlass_sm120.cu` (1027) | registry key `{CUTLASS_NVFP4,F16,false}` at `executor_gemm_dispatch.cu:594`; `weight_dispatch.cu`; `executor_forward.cu` | native NVFP4 prefill, M>1 | **load-bearing** (55.2% of nvfp4-dense pp4096 window) |
| stream-K / small-N variants | `gemm_cutlass_sm120.cu:983,990` | tests only as named entry points; the **shipped** stream-K is `args.scheduler.decomposition_mode = force_streamk ? DM::StreamK : DM::Heuristic` inside the mainline launchers at `:785,:811,:922` | `gemm.nvfp4_cutlass_streamk` (roadmap:182 SHIPPED default-on) | live via the flag, wrappers are harness entry points |
| CUTLASS grouped 3.x `gemm_grouped_cutlass_3x_nvfp4[_device_args]` | `compute/gemm_cutlass_grouped_3x.cu` (701) | `executor_forward_moe_cutlass.cu` | MoE prefill tiers GROUPED / DEVICE_ARGS | **load-bearing** |
| `gemm_grouped_nvfp4_smallM` | `compute/gemm_grouped_nvfp4_smallM.cu` (918) | `executor_forward_moe_cutlass.cu` | MoE prefill tier SMALL_M, `moe.nvfp4_smallM` **default false** (`src/core/config/moe.h:118`) | opt-in only (Q4) |
| dense small-M NVFP4 `gemm_nvfp4_smallm[_v2]_a4`, `_v2_pair_a4`, `_v2_stripes` | `quant/nvfp4_gemm_smallm*.cu` | `executor_gemm_dispatch.cu:411-418`, `executor_gemm_smallm.cu` | `gemm.nvfp4_smallm` **default true**, `_impl=2`, M<=32 | **load-bearing** |
| `gemm_nvfp4`, `gemm_nvfp4_batched[_acc]` | `quant/nvfp4_gemm.cu`, `nvfp4_gemv_batched.cu` | `executor_gemm_dispatch.cu:290-336,676-690`, `executor_forward_moe_legacy.cu` | spec-verify chunks; native-NVFP4 prefill safety net | live |
| NVFP4 GEMV (`gemv_nvfp4_kpar`, `_fused`, `_moe_*`, `_residual`) | `quant/nvfp4_gemv_{dense,fused,moe,batched}.cu` | `executor_ffn.cu`, `executor_attention_qkv.cu`, `executor_forward_moe_batch.cu`, `executor_gemm_dispatch.cu:189-227` | NVFP4 decode M=1 | **load-bearing** |
| MXFP4 GEMV (`gemv_mxfp4_*`) | `quant/mxfp4_gemm.cu` | `executor_ffn.cu`, `executor_attention_qkv.cu`, `executor_forward.cu`, `weight_dispatch.cu` | gpt-oss MXFP4 decode | live |
| CUTLASS MXFP4 `gemm_mxfp4_cutlass_sm120` | `compute/gemm_cutlass_mxfp4_sm120.cu` (658) | `exec/gemm_kernel_cutlass_nvfp4.cu` dual-cache branch, `attention_mxfp4_prefill.cu`, `weight_dispatch.cu` | `--mxfp4-prefill` | opt-in |
| `convert_nvfp4_to_mxfp4_hadamard` | `gemm_cutlass_mxfp4_sm120.cu:381-438` | **nobody** | - | **dead** (A1-4) |
| Marlin | - | - | - | **absent** |

### Q2 - prefill attention: what still takes a non-FA2 tier

Outer gate `src/exec/executor_attention_prefill.cu`; FMHA chain `src/compute/attention_dispatch.cu`. Defaults: `fmha_fa2="on"`, `fa2_fp16qk="on"`, `fa2_hd256=true`, `fp8_fmha="never"`, `fmha_sm120="auto"`, `attention.mxfp4="auto"` (`src/core/config/attention.h:37,39,47,60,92,139`). Auto `fmha_prefill_threshold = attn_scores_cap + 1` (`executor_workspace_buffers.cu:596-599`).

| (head_dim, mask/dtype/shape condition) | path taken | evidence |
|---|---|---|
| hd=128, F16 Q, no sinks, uniform or per-layer, any n, causal, SWA or not, chunked or single-shot | **FA2_FP16QK** | `executor_attention_prefill.cu:427-430,313-316`; kernel accepts at `attention_fmha_sm120.cu:1992,2013` |
| hd=256, F16 Q, no sinks, `fa2_hd256=true` (default) | **FA2_FP16QK** | `attention_dispatch_rules.h:35-37`; `attention_fmha_sm120.cu:2013` |
| hd=256 with `attention.fa2_hd256=false` | CUBLAS below threshold, FMHA_CHAIN -> FMHA_SM120 above | `attention_dispatch_rules.h:36`; `executor_attention_prefill.cu:433-437,455-464` |
| **hd=512 (Gemma-4 global layers), S-matrix fits** | **CUBLAS** by design and by measurement | `executor_attention_prefill.cu:397` (`prefer_fmha ... && hd != 512`), `:433-441`; measured 0.52x / 0.22x FMHA-vs-cuBLAS at pp512 / pp2048, `docs/audit/gemma4_attn_routing_2026_07_16/PERF_LOG.md` entry 1 |
| hd=512, S-matrix overflows | **CUBLAS_SLICED** (q-row slices), FMHA only if even a 16-row slice overflows | `executor_attention_prefill.cu:442-451`; 3.4-3.9x faster than FMHA hd=512 at Skv 8k/16k, same PERF_LOG entry 4 |
| **learned sinks (gpt-oss, hd=64), n < threshold** | **CUBLAS** (only the cuBLAS softmax folds sinks below the threshold) | `executor_attention_prefill.cu:426` excludes sinks from FA2; `:433` |
| learned sinks, n >= threshold | FMHA_CHAIN -> **FMHA_SM120** (sink-capable since #992); chain **throws** on decline | `attention_dispatch.cu:83-99` |
| hd in {64, 96}, no sinks, n < threshold | **CUBLAS**; above threshold FMHA_SM120 | `attention_dispatch_rules.h:28-37`; `executor_attention_prefill.cu:433,455` |
| **hd=192 (MLA, DeepSeek-V2-Lite)** | **CUBLAS only.** Neither family serves it; `max_safe_prefill_chunk` clamps n to the S-matrix cap so the chain is never reached | `tests/test_attention_dispatch_rules.cpp:22-30`; kernel switch instantiates only {64,96,128,256,512}, `attention_fmha_sm120.cu:544-600`; `docs/MODELS.md:52` |
| hd in {80, 160, 320} | CUBLAS only (no model in the zoo) | `tests/test_attention_dispatch_rules.cpp:39-44` |
| any hd, Q qtype != F16 | FA2 and FMHA both decline -> CUBLAS / throw | `attention_fmha_sm120.cu:1992,440` |
| chunk continuation with capture replay (`cap_replay`) | FA2-only; a decline **throws** rather than falling back | `executor_attention_prefill.cu:300-310` |
| KV dtype at chunked prefill | irrelevant to the tier: all six KV dtypes are gathered to FP16 first (`paged_kv_gather_*`), `executor_attention_prefill.cu:190-250`. The one exception is `attention.mxfp4_paged_kv` (**default false**), which reads NVFP4 paged KV directly at hd=128, `:129-152` |
| MXFP4 FMHA tier | unreachable unless `attention.mxfp4="always"` | `attention_mxfp4_prefill.cu:517-521` |
| FP8-QK FMHA tier | unreachable unless `attention.fp8_fmha="on"`; documented harmful (teacher-forced PPL 16.6 -> 549) | `attention_dispatch.cu:143-159`, `attention_dispatch_decision.h:70-75` |

**Quantified with existing numbers.** Newest roofline run, `tools/roofline/history/runs/1d5b9230_20260831_180644.json` (commit `1d5b9230`, 2026-08-31), aggregated per class over all `ncu_groups`:

| cell | total | attn_fa2 | attn_legacy_softmax |
|---|---:|---:|---:|
| nvfp4-dense pp4096 | 92.135 ms | 31.938 ms (34.7%) | **absent (0.0%)** |
| nvfp4-dense pp512 | 21.549 ms | 1.663 ms (7.7%) | **absent (0.0%)** |
| nvfp4-hybrid pp4096 | 37.967 ms | 4.319 ms (11.4%) | **absent (0.0%)** |
| nvfp4-hybrid pp512 | 15.347 ms | 0.287 ms (1.9%) | **absent (0.0%)** |

So **SETTLED S-8 holds today** on hd=128: the materialised path is 0.0% of the prefill window, and the exception (Gemma-4 hd=512) is deliberate and measured-faster. Nothing on this axis is a "legacy vestige" lever. The `gemm_cublas` class in the same hybrid cells (26.0% pp4096 / 18.9% pp512) is the GDN-projection GEMM, already priced and parked at `docs/roadmap.md:64,185`.

### Q3 - decode attention per KV dtype

| KV dtype | launcher | inner math | head_dims | split-K | KV-multitok |
|---|---|---|---|---|---|
| F16 | `paged_attention_decode`, `attention_paged.cu:1233` | scalar FMA (`:217,396,757`); no `mma.sync`/`wmma`/`dp4a` in the TU | 64/96/128/256/512 templated (`:1533-1548`); **any other hd falls to the untemplated generic kernel** `:1552` (this is how MLA hd=192 decodes) | yes, own heuristic, cap 64 splits (`:1284-1311,1296`) | yes, hd 128/256, GQA 1..8 (`attention_paged_f16_multitok.cu:374-377`) |
| FP8_E4M3 | `paged_attention_decode_fp8`, `attention_paged_fp8.cu:602` | scalar FMA (`:196-199`); tile `__fmaf_rn`; multitok `__hmul2` | 64/96/128/256/512 (`:771-799`); tile split-K **hd=128 only** (`attention_paged_fp8_tile.cu:513-514`) | yes, shared `compute_splitk_splits` cap 32 (`attention_paged_common.cuh:242-276`) | yes, hd 128 only (`attention_paged_fp8.cu:778-790`) |
| INT8 | `paged_attention_decode_int8`, `attention_paged_int8.cu:476` | **`__dp4a` for Q.K** (`:184,406`), P.V scalar float (`:230-236`) | 64/96/128/256, **no 512** (`:521-536,555-570`) | yes (`:502`) | no |
| INT4 | `paged_attention_decode_int4`, `attention_paged_int4.cu:539` | scalar FMA (`:135,298,500`); no dp4a | 64/96/128/256, no 512 | yes (`:562`) | no |
| NVFP4 | `paged_attention_decode_nvfp4`, `attention_paged_nvfp4.cu:422` | `__fmaf_rn` (`:224-225`); multitok `__hfma2` | 64/128/256/512, **no 96** (`:473-489`) | yes (`:441`) | yes, hd 128/256 + a GQA variant, ratio 2..16 (`attention_paged_nvfp4_multitok_gqa.cu:355-368`) |
| NVFP4 tensor-core | `paged_attention_decode_nvfp4_tc`, `attention_paged_nvfp4_tc.cu:1053` | **`nvcuda::wmma` 16x16x16 HMMA** for Q.K (`:268`) and P.V (`:403,492`). **No `mxf4nvf4` MMA in any paged TU** | 64/128/256/512, no 96 (`:1129-1145`) | yes, forced on when the FP16 residual ring is active (`:1096,1111-1114`) | n/a |
| MXFP4_KV | `paged_attention_decode_mxfp4_kv`, `attention_paged_nvfp4.cu:532` (NVFP4 templates with `ScaleDtype::UE8M0`) | scalar FMA, identical kernel | 64/128/256/512, no 96 | yes (`:552`) | **no** - the multitok launch call exists only in the NVFP4 launcher (`:446`) |

- **Scalar vs tensor-core**: only two decode kernels use tensor cores at all - INT8 (`__dp4a`, integer) and the opt-in NVFP4 TC (`wmma` HMMA, `kv_cache.bitdecoding_qk` default **false**, `src/core/config/kv_cache.h:52`). This is deliberate and documented: `docs/internals/SM120.md` "Why batch=1 decode is memory-bound" (memory:compute ~8,600:1 at FP32 accumulate). Not a finding.
- **M>1 (spec verify)** goes through the same single-token launchers with `batch = M` - see "Brief vs repo" item 2.
- **Sparse/top-k**: `attention.sparse_topk_tokens` (default 0, `src/core/config/attention.h:220`). Init gate accepts **F16 / FP8_E4M3 / NVFP4 only**, refuses MLA, token-recycling and persistent prefix cache (`src/runtime/engine_kv_cache_init.cpp:754-772`). Per-layer gate at `executor_attention_decode.cu:179-182` additionally requires no SWA, no sinks, `n == n_seq`, `nh/nkv <= 16`. No head_dim gate; head_dim is a runtime argument (`src/exec/sparse_attn_select.cu:534-547`). INT8 / INT4 / MXFP4_KV branches pass `layer_block_tables` unchanged and would ignore a sparse selection - harmless only because the init gate excludes them.

### Q4 - MoE ladders

**Prefill outer chain** (`src/exec/executor_forward_moe.cu:504-624`), in order:

| # | branch | gate | reachable with defaults? |
|---|---|---|---|
| 0 | `run_moe_decode_fast` (decode fast path, `goto moe_after_experts`) | `will_decode_fast` | yes (decode) |
| 1 | `try_run_moe_q6k_prefill` (fused Q6_K WMMA/scalar) | `!gemm.moe_imma_prefill` | **no** (`moe_imma_prefill` default **true**, `src/core/config/gemm.h:54`) |
| 2 | `try_run_moe_q4k_prefill` (fused Q4_K/Q5_K dp4a) | `eff<=640 && !gemm.moe_imma_prefill` | **no** |
| 3 | `try_run_moe_gemma4_ggml_prefill` | `!moe_imma_prefill && overrides.gemma4.ggml_prefill` (default **false**, `src/model/model_config.h:189`) | **no**, doubly gated |
| 4 | `FP16_BATCH` | FP16 batch dequant buffers present | yes |
| 5 | `FP8_BATCH` | `!can_fp16_batch && Q6_K experts` | yes |
| 6 | `CUTLASS3X` -> 4-tier ladder | `try_run_moe_cutlass3x_nvfp4_prefill_` | yes |
| 7 | `NVFP4_DEQUANT` | - | yes |
| 8 | `LEGACY` | - | yes |

**Inner 4-tier ladder** (`src/exec/moe_prefill_decision.h:47-76`, mirrored by `verify_against_moe_routing_model`):

| tier | gate | reachable with defaults? |
|---|---|---|
| DEVICE_ARGS | `moe.nvfp4_device_args` (default **true**), not gpt-oss, workspace ready | yes |
| SMALL_M | `moe.nvfp4_smallM` (default **false**, `src/core/config/moe.h:118`), not gpt-oss, kernel available, M <= 64 | **no - opt-in only** |
| GROUPED | `ws.grouped_ready` (gpt-oss lands here) | yes |
| LEGACY | fallthrough / host-offloaded experts | yes |

**Decode path** (`executor_forward_moe_batch.cu:1024-1310`): NVFP4-host, NVFP4 fused gate/up + decode + multirow, then the GGUF q8_1 GEMV family selected by function pointer per qtype (`:1217-1223,1241-1247,1303-1306`), then packed Q6_K/Q8_0 fused variants. **No `dispatch_record` enum or call site exists for MoE decode at all.**

### Findings

### [A1-1] The GemmKernelRegistry is a third dispatcher: 9 of its 19 registered strategies are unreachable from production, and one has drifted numerically from the live path
Axis: A1   Sev: S1   Confidence: high
Evidence: `GemmStrategy` is constructed in exactly three places in `src/ tools/` (`rg -l 'GemmKernelRegistry|GemmStrategy' src/ tools/` -> only `gemm_kernel_*.cu` + `executor_gemm_dispatch.cu`): `src/exec/executor_gemm_dispatch.cu:88` `{FP16, NONE, false}`, `:259` `{FP16, h.source_qtype, true}` (guarded at `:242-243` to exclude `NONE`/`F16`/`BF16`), `:594` `{CUTLASS_NVFP4, F16, false}`. Registered strategies, from the nine `register_kernel` blocks: `gemm_kernel_registry.cu:77,79` `{FP16,F16,false/true}`; `gemm_kernel_generic_dequant.cu:83` `{FP16,NONE,false}`; `gemm_kernel_gguf.cu:277-291` 8x `{FP16,<qtype>,true}`; `gemm_kernel_q4k_imma.cu:73` `{FP16,Q4_K,false}`; `gemm_kernel_fp8.cu:130,132` `{FP8,F16,false}`+`{FP8,NONE,false}`; `gemm_kernel_nvfp4_gemm.cu:56` / `nvfp4_gemv.cu:54` `{NVFP4,F16,false/true}`; `gemm_kernel_mxfp4.cu:105,107` `{MXFP4,F16,true/false}`; `gemm_kernel_cutlass_nvfp4.cu:141` `{CUTLASS_NVFP4,F16,false}`. 19 registered, 10 reachable, **9 unreachable**. Every handler is `static`, so the registry table is their only entry (`rg '^static GemmDispatchResult' src/exec/gemm_kernel_*.cu`). `tests/test_gemm_kernel_registry.cu` (1801 lines, allowlisted at 1259 code LOC in `tools/filesize_thresholds.toml:95`) is the only caller of the 9. The FP8 handler has **diverged**: `gemm_kernel_fp8.cu:52-56` quantises the activation to FP8 and runs W8A8 ("Mirror executor_kernels.cu:2278-2282 verbatim"), while the live FP8 prefill at `executor_gemm_dispatch.cu:494-501` calls `gemm_cublaslt(input /*FP16*/, fp8_w, ...)` = W8A16.
Current: three dispatchers coexist - `GraphExecutor::gemm_via_handle_` (`executor_gemm_dispatch.cu:157-698`, ~540 LOC of if-chain), the registry (19 entries, 10 live), and `imp::gemm_dispatch`/`gemv_dispatch` (`src/compute/weight_dispatch.cu`, 413 LOC, the tail fallthrough at `:696` and the MXFP4/default decode arms). The registry header still describes the migration as in progress behind a flag that no longer exists (`gemm_kernel_registry.h:13`, `:17-20` "once every dispatch site is covered the legacy path can be deleted").
Expectation: a strategy table is worth its indirection only when it is *the* dispatch. vLLM (`vllm/model_executor/layers/quantization/*` + `Fp8LinearMethod.apply`) and TensorRT-LLM (`tensorrt_llm/_torch/modules/linear.py` backend selection) both route every quant tier through the one registry; a registry that serves 10 of 19 keys while an if-chain serves the rest is the pre-migration state, not a design.
Delta: 9 unreachable registrations across 5 files (~455 LOC of wrapper) plus a 1801-LOC test pinning a contract that production only half uses; one of the 9 encodes different numerics than the live path it claims to mirror, so the test suite green-lights a kernel nobody runs.
Cost: either finish the migration (move the FP8 / NVFP4 / MXFP4 / FP16 arms of `gemm_via_handle_` onto the registry: ~5 files, ~250 LOC moved, high risk - these are the decode and prefill hot paths and there is no GPU CI lane) or retire the 9 dead registrations and their tests (~455 LOC + ~900 test LOC deleted, low risk, but discards the migration). Minimum honest action: correct `gemm_kernel_registry.h:13-20` and record the decision.
Falsifier: another file constructing `GemmStrategy`, or a non-static handler taken by address. Checked y - `rg -n -w 'GemmStrategy' src/ tools/` lists only `gemm_kernel_registry.h` and the 10 files above; `rg -n 'instance\(\)\.dispatch' src/ tools/` returns exactly `executor_gemm_dispatch.cu:89,260,595`.

### [A1-2] `gemm.moe_imma_prefill` defaults true and makes three whole MoE-prefill implementations (~1000 LOC) unreachable, with no death date
Axis: A1   Sev: S2   Confidence: high
Evidence: `src/core/config/gemm.h:54` `bool moe_imma_prefill = true;`. `src/exec/executor_forward_moe.cu:511-530`: `const bool moe_imma_pref = runtime_config().gemm.moe_imma_prefill;` then `if (!moe_imma_pref && try_run_moe_q6k_prefill(...))` / `else if (eff <= 640 && !moe_imma_pref && try_run_moe_q4k_prefill(...))` / `else if (!moe_imma_pref && cfg.overrides.gemma4.ggml_prefill && ... try_run_moe_gemma4_ggml_prefill(...))`. All three require `!moe_imma_pref`. The three functions are their kernels' only consumers: `executor_forward_moe_batch.cu:224-315` (92 LOC) is the sole caller of `gemm_q6k_fused_moe_prefill` and `_tc` (`compute/gemm_moe_fused.cu` 175 LOC + `gemm_moe_fused_tc.cu` 283 LOC); `:316-397` (82 LOC) is the sole caller of `gemm_q4k_dp4a_moe_fused`, `gemm_q5k_dp4a_moe_fused` and `gemm_q6k_moe_fused` (`compute/gemm_q6k.cu`, 240 LOC, whose only export it is); `:606-748` (143 LOC) is the sole caller of `ggml_mmvq_q4k_f32`. Verified per symbol with `rg -n -w`.
Current: ~317 LOC of executor + ~700 LOC of `src/compute` reachable only with `--set gemm.moe_imma_prefill=false`. Nothing in `docs/roadmap.md`, `docs/LIMITATIONS.md` or `docs/DESIGN_DECISIONS.md` names them as a retained A/B arm; the flag's own comment (`gemm.h:48-54`) presents IMMA as "lever #1 for the 2.4-2.6x GGUF-MoE prefill gap ... Default on" without saying what the superseded paths are still for. The `try_run_moe_q6k_prefill` block even carries a stale comment at `executor_forward_moe.cu:501-502` ("Scalar: disabled (FP16 batch path always wins)") describing a sub-variant of a function that no longer runs at all.
Expectation: a superseded implementation either has a stated A/B purpose with a death date (imp's own convention, `tools/filesize_thresholds.toml` `[allow]` reason strings, S-26) or it is removed. llama.cpp retires its superseded MMQ variants; vLLM deletes a kernel when a `_v2` lands.
Delta: 1000 LOC of hot-path CUDA that recompiles on every touch of its TU, is never exercised by the default configuration, is not in any CI lane that runs kernels (no GPU runner), and whose per-shape correctness therefore has no standing check.
Cost: decision, not code. Either (a) delete the three branches + `gemm_moe_fused.cu`, `gemm_moe_fused_tc.cu`, `gemm_q6k.cu` and the `moe_imma_prefill=false` arm (~1000 LOC, 6 files; risk: loses the fallback if the IMMA kernel ever declines a shape - check `mmq_imma_moe_gemm`'s decline paths first), or (b) keep them and write the reason plus a re-measure date next to the flag (~10 lines). Wrong-if-deleted: a Q6_K-expert MoE model whose IMMA path declines and whose FP16 batch buffer does not fit would lose a tier.
Falsifier: another caller of the three `try_run_moe_*` functions, or a resolver that flips `moe_imma_prefill` false for some model. Checked y - `rg -n -w try_run_moe_q6k_prefill|try_run_moe_q4k_prefill|try_run_moe_gemma4_ggml_prefill src/ tests/ tools/` returns only the declarations in `executor.h:1027,1032,1039`, the definitions, and the three gated call sites; `rg -n -w moe_imma_prefill src/ tools/ tests/` shows no writer outside config parsing.

### [A1-3] Pruning the PDL registration also silently dropped the max-L1 carveout for the whole dp4a GEMV family
Axis: A1   Sev: S2   Confidence: med
Evidence: `src/compute/gemm_dp4a.cu:659-669` - `gemv_pdl_register()` does two unrelated things per kernel: `pdl::enable_kernel(...)` **and** `SET_MAXL1(...)` = `cudaFuncSetAttribute(..., cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1)`, under the comment "GEMV kernels are bandwidth-bound with minimal SMEM: maximize L1 cache" (`:664-665`). It covers ~40 template instantiations (REG1 x19, REG2, REG3...). Commit `0af8f80d` (#1833, 2026-08-31) removed the call: `src/exec/executor_workspace.cu:338-343` now registers only `elementwise_add_fp16_kernel`, `nvfp4_gemv_pdl_register`, `layernorm_pdl_register`, `rope_pdl_register`, with the comment "The former blanket list (... activation/dp4a families) is gone: those kernels do not wait yet, and registering them raced". `gemv_pdl_register` is now decl+def only: `rg -n -w gemv_pdl_register src/ tools/ tests/ include/` -> `src/compute/gemm.h:313`, `src/compute/gemm_dp4a.cu:669`. Two lines below the pruned block, `mxfp4_gemv_set_l1_carveout()` is still called (`executor_workspace.cu:350`) under the comment "SMEM carveout: maximize L1 for bandwidth-bound GEMV kernels (**independent of PDL**)" - i.e. MXFP4 has a carveout-only entry point for exactly this reason, and NVFP4 keeps its carveout inside the PDL register that is still called (`src/quant/nvfp4_gemm.cu:297-325`). dp4a has neither. `rg -n 'cudaSharedmemCarveout' src/` confirms `gemm_dp4a.cu:666` is the only carveout site for the dp4a family and it is unreachable.
Current: every GGUF decode GEMV (`gemv_dp4a_kernel<Q4_K|Q5_K|Q6_K|Q8_0|Q4_0|Q2_K|Q3_K, NR, ...>` and the fp32/residual/qkv-fused variants) runs with the driver-default shared/L1 split instead of the explicit max-L1 the code asks for. This is the M=1 decode path for every GGUF model, including the perf-gate model.
Expectation: a per-kernel launch attribute set for a measured reason survives a change to an unrelated concern. The repo already knows the two are separable - it built `mxfp4_gemv_set_l1_carveout()` for that.
Delta: an unintended attribute regression on the GGUF decode hot path, invisible because #1833's measurements were on Qwen3.8-27B-**NVFP4** (whose family kept its carveout).
Cost: 3-4 lines - split `gemv_pdl_register()` into `gemv_set_l1_carveout()` (the SET_MAXL1 half) and call it beside `mxfp4_gemv_set_l1_carveout()` at `executor_workspace.cu:350`; leave the `pdl::enable_kernel` half unregistered. Risk near zero: the carveout is a scheduler hint, not a correctness change, and the racing kernels stay unregistered.
Falsifier: the carveout makes no measurable difference on these kernels, or the driver default is already max-L1 on sm_120. **NOT checked - needs a GPU A/B** (`tg128` on a Q8_0/Q4_K GGUF, carveout call restored vs not). The perf claim is therefore **HYPOTHESIS**; the code fact (the attribute is no longer set) is verified.

### [A1-4] Three dead host exports in the GEMM tree that the 2026-08-03 decl-only sweep could not see, and the mechanism that hid two of them
Axis: A1   Sev: S3   Confidence: high
Evidence: verified with `rg -n -w <symbol> src/ tools/ tests/ include/`.
- `imp::gemm_grouped` - `src/compute/gemm_grouped.h:13` (decl), `src/compute/gemm_grouped.cu:168-186` (def). Every other hit of the word is the include path `compute/gemm_grouped.h` or one of its own `IMP_LOG_ERROR` strings at `:30,71,132,144,152,175`. Its only consumer, the static `run_expert_matmul` (`:108-166`), is called nowhere else (`:184`). ~78 LOC dead. The live export of the file is `gemm_moe_batched` (`:201`).
- `convert_nvfp4_to_mxfp4_hadamard` - `src/compute/gemm_cutlass_mxfp4_sm120.h:41` (decl), `.cu:381-438` (def). The four other hits are its own `IMP_CHECK`/`IMP_LOG_DEBUG` message strings (`:383,386,389,401`). Added by `0b37bf6e` 2026-03-10, no caller ever. ~58 LOC dead. (`hadamard_transform_fp16` itself is live - `executor_forward.cu:739,829`, `executor_attention_qkv.cu:176`, `gemm_kernel_mxfp4.cu:59`.)
- `gemv_pdl_register` - see A1-3; dead since 2026-08-31, i.e. after the sweep.
Current: SETTLED §C states the non-kernel decl+def sweep "starts at 530 candidates ... 530 -> 201 **by occurrence count** -> 27 by the decl+def signature", and that the 27 "are fully resolved - do not re-open them". The first two above were never in the 27, and the reason is the filter: a function whose own name appears inside its `IMP_CHECK`/`IMP_LOG_*` message strings has 4-8 occurrences and passes the "more than decl+def" screen while still having zero callers. `gemm_grouped` has 12 occurrences; `convert_nvfp4_to_mxfp4_hadamard` has 6.
Expectation: SETTLED's own method note ("a call-graph tool reporting no callers is only evidence if the tool can see calls at all") applies in the other direction too - an occurrence-count screen must not count the symbol's own diagnostics.
Delta: ~136 LOC of dead exports, and a documented sweep whose "fully resolved" claim is narrower than it reads. This is new evidence, not a contradiction of the anchor: both files still exist and the 27 named candidates are indeed closed.
Cost: delete two functions + one static helper (2 files, ~136 LOC); `gemm_grouped_cleanup()` at `gemm_grouped.cu:421` is already an empty no-op and could go with them (1 more call site in `engine.cpp`). Risk: none for the two dead ones. If the sweep is re-run, exclude the symbol's own string literals before counting occurrences.
Falsifier: a caller reached through a macro or a function pointer. Checked y - word-boundary grep over `src/ tools/ tests/ include/`; the function-pointer trap that bit the helper script (see Coverage) is exactly what `-w` without a trailing `(` catches.

### [A1-5] `paged_attention_serves_head_dim` has no `MXFP4_KV` case, so the guard built for this class does not cover the dtype it was extended for
Axis: A1   Sev: S3   Confidence: high
Evidence: `src/compute/attention_paged.cu:1196-1216`. The switch handles `F16`, `FP8_E4M3`, `INT8`, `INT4`, `NVFP4`, then `default: // Not checked here; do not refuse it. return true;`. `MXFP4_KV` is a user-selectable KV dtype (`src/runtime/engine_init_resolver.cpp:210-211`, `kv_cache.dtype = "mxfp4"`), and its decode launcher shares the NVFP4 template set - 64/128/256/512, **no 96** (`src/compute/attention_paged_nvfp4.cu:570-618`), ending in `paged_attention_unsupported_head_dim(...)` at `:617`, which is `[[noreturn]] throw` (`attention_paged.cu:1190-1194`, decl `attention_paged.h:299`). The resolver that exists to prevent exactly this consults the function at `engine_init_resolver.cpp:305-325` and falls back to FP16 KV on a false - it gets `true` for MXFP4_KV at every head_dim. The function's own comment (`:1197-1202`) enumerates the launchers per file and simply omits `attention_paged_nvfp4.cu`'s MXFP4 arm.
Current: `kv_cache.dtype=mxfp4` on a head_dim-96 model passes init with the "MXFP4_KV ... ~3.6x compression" log line (`engine_init_resolver.cpp:340`) and throws at the first decode step instead of being refused at init.
Expectation: the #1674 guard's stated contract is "refuse the dtype at init rather than leaving the attention output unwritten". A dtype outside the switch defeats it.
Delta: one missing `case`. Not a silent-corruption class (the throw is loud), but the failure moves from init to mid-generation, which for a server means a 500 on a live request rather than a startup refusal.
Cost: one line (`case QType::MXFP4_KV:` sharing the `NVFP4` arm) plus one line in the comment; a CPU-lane test beside `tests/test_attention_dispatch_rules.cpp`. Risk: none. Wrong if MXFP4_KV is unreachable for hd=96 models for some other reason - the resolver applies it over `mcfg.head_dim` and `head_dim_per_layer`, so it is not.
Falsifier: an earlier gate refusing MXFP4_KV per head_dim. Checked y - `rg -n MXFP4_KV src/runtime/` shows only the selection (`:211`), the log line (`:339-342`) and the gather-kernel list (`engine_scheduler.cpp:433-437`), none of which look at head_dim.

### [A1-6] Two paged split-K kernels are pre-Hopper portability branches, compiled and unlaunchable on the only supported chip
Axis: A1   Sev: S3   Confidence: high
Evidence: `src/compute/attention_paged_fp8.cu:685` `} else if (sm_ver_fp8 >= 90) {` selects `paged_attention_splitk_fp8_pipeline_kernel`; the `else` at `:720` launches `paged_attention_splitk_fp8_kernel` (definition `:268-447`, ~180 LOC, 5 head_dim instantiations). `src/compute/attention_paged_int4.cu:574` `if (sm_ver_int4 >= 90) {` -> pipeline kernel; `else` at `:607` -> `paged_attention_splitk_int4_kernel` (definition `:178-331`, ~154 LOC, 4 instantiations). `get_device_sm_version()` returns `major*10+minor` (`src/compute/attention_dispatch.cu:23-33`) = 120 on every supported card, and `docs/internals/SM120.md` "SKU coverage" states every consumer-Blackwell part is compute capability 12.0. Neither kernel is referenced from `tests/`.
Current: ~334 LOC plus 9 template instantiations of dead `ptxas` work in two of the hottest decode TUs, in a repo whose `src/compute/CLAUDE.md` invariant reads "**No portability branches.** No other arch".
Expectation: an architecture-exclusive engine has one arm per decision. The equivalent FP16 and INT8 launchers do not carry this branch (`attention_paged.cu:1334-1408`, `attention_paged_int8.cu:521-536`), so the pattern is not even consistent inside the family.
Delta: recompile blast radius (`attention_paged_fp8.cu` 804 LOC, `attention_paged_int4.cu` 672 LOC - both single TUs re-`ptxas`ed on any touch) for code that can never run, and two kernels that no test and no gate covers.
Cost: delete both `else` arms and both kernels (2 files, ~340 LOC). Risk: low; the guarded pipeline kernels serve the identical head_dim sets. Wrong if imp ever targets sm_80/sm_89 - `CMakeLists.txt:48-50` emits only `sm_120a` SASS and `compute_120f` PTX, and `docs/internals/SM120.md` says the PTX arm targets GB203, also cc 12.0.
Falsifier: `get_device_sm_version()` returning < 90 on some supported device, or a test forcing the else arm. Checked y - grep of both kernel names over `src/ tests/ tools/` returns only the definition and the single guarded launch each.

### [A1-7] The resolved-dispatch dump (#1205) has blind spots exactly where the MoE ladder is least visible
Axis: A1   Sev: S3   Confidence: high
Evidence: `rg -n 'set_moe_prefill_outer|set_moe_prefill_tier|set_attn_decode|set_attn_prefill_*' src/` returns 24 call sites. Gaps:
- `MoePrefillOuter::FUSED_Q6K` (`src/compute/dispatch_paths.h:60`, name string `dispatch_paths.cpp:77`) has **zero producers** in `src/`; its only mention outside the enum is `tests/test_dispatch_record.cpp:48`. A tier that only a test reaches.
- The three MoE prefill branches at `executor_forward_moe.cu:513,521,528` set nothing, so under `--set gemm.moe_imma_prefill=false` the dump prints `moe_prefill=unset` (`moe_prefill_outer_name(UNSET)` -> `"unset"`, `dispatch_paths.cpp:63-64`) although a real path ran.
- `run_moe_decode_fast` (`executor_forward_moe_batch.cu:1024`) records nothing; there is no `MoeDecodePath` enum at all, although the decode side picks between the NVFP4-host, NVFP4-fused, NVFP4-multirow and seven GGUF q8_1 GEMV variants (`:1077-1108,1217-1247,1303-1306`).
Current: `Engine::log_resolved_dispatch_once_` (`src/runtime/engine_scheduler.cpp:204-241`) exists because "the prefill chain has six tiers and the MoE chain five, and every one of them declines by returning `false` with no log ... a model silently taking a slower or lower-quality path left no trace". The MoE decode chain, which is the whole decode hot path on an MoE model, is outside it.
Expectation: the mechanism's own stated contract - record the branch that won, at the point it wins.
Delta: an observability gate that does not cover the branch set it was built for. Consistent with SETTLED §E's "#1205's resolved-dispatch line never printed. A gate that cannot be shown to fire has not been validated."
Cost: 4 one-line `set_moe_prefill_outer` calls + a `MoeDecodePath` enum with ~6 enumerators and ~8 call sites (3 files, ~40 LOC). Risk: none (a thread_local store on a path already launching kernels). Wrong if the owner considers MoE decode routing already covered elsewhere - `rg dispatch_record src/exec/executor_forward_moe_batch.cu` returns nothing.
Falsifier: a producer of `FUSED_Q6K` reached through a macro. Checked y - `rg -n 'FUSED_Q6K' src/ tests/ tools/` returns 3 lines: the enumerator, its name string, and the test.

### [A1-8] `gemm_q4k_fused_moe_prefill` (scalar Q4_K MoE prefill) is reachable only from a test
Axis: A1   Sev: S3   Confidence: high
Evidence: `rg -n -w gemm_q4k_fused_moe_prefill src/ tools/ tests/ include/` -> `src/compute/gemm_q4k.h:13` (decl), `src/compute/gemm_q4k.cu:614` (def), `tests/test_gemm_q4k_fused_prefill.cu` (2). Its chain is exclusive: `launch_scalar<QKType>` (`gemm_q4k.cu:590-608`) and `gemm_qk_scalar_moe_prefill_kernel` (`:234-~325`, header comment "Scalar FP16 fallback (for benchmarking / small M_e)"), one instantiation (`QKType::Q4_K`). Production Q4_K/Q5_K MoE prefill runs `fused_dp4a_for_qtype` instead (`src/exec/executor_forward_moe_batch.cu:371,374,393` inside `try_run_moe_q4k_prefill`) - which is itself default-unreachable per A1-2.
Current: ~110 LOC (kernel + launcher + export) in the 653-LOC TU that also holds the live `gemm_q4k_dp4a_dense`/`gemm_q5k_dp4a_dense`, so every touch of the live dense kernels re-compiles the dead one. Unlike its sibling `mmq_q6k_imma_gemm` - also test-only, but carrying an explicit measured refutation in the dispatch (`executor_gemm_dispatch.cu:549-556`) - this one has no recorded verdict.
Expectation: an entry point kept for benchmarking says so where the dispatch would have used it, or lives in the bench tool.
Delta: 110 LOC of unexercised kernel with no stated reason, inside a hot TU.
Cost: delete the scalar arm + its test (2 files, ~130 LOC), or add the one-line reason. Risk: low - the test would go with it, and the test is the only user.
Falsifier: a caller via `tools/imp-bench`. Checked y - `rg -n -w gemm_q4k_fused_moe_prefill tools/` is empty.

### [A1-9] The MXFP4 prefill attention family is the largest default-off kernel block in `src/compute/` and carries no death date
Axis: A1   Sev: S3   Confidence: med
Evidence: `attention_mxfp4_available()` returns false unless `attention.mxfp4 == "always"` (`src/compute/attention_mxfp4_prefill.cu:509-521`); the config default is `"auto"` (`src/core/config/attention.h:139`), which the code comment at `:514-516` explicitly calls "OFF for FMHA". The second entry point, `fmha_sm120_mxfp4_prefill_paged`, is gated on `attention.mxfp4_paged_kv`, default false (`attention.h:167`, used at `src/exec/executor_attention_prefill.cu:132`). Files: `attention_fmha_mxfp4_sm120.cu` **2021 LOC** (2nd largest in `src/compute/`) + `attention_mxfp4_prefill.cu` 532 LOC. The measured verdict is on record: `CHANGELOG.md:3548-3552` "NVFP4-attention research knobs (#868, idea #846, **refuted**) ... residual noise compounds with context (+10 % NLL at 9k). All three default OFF."
Current: ~2550 LOC of prefill attention that no default configuration reaches, with the refutation recorded in the CHANGELOG but no row in `docs/roadmap.md`, `docs/LIMITATIONS.md` or `docs/DESIGN_DECISIONS.md`, and no re-measure trigger.
Expectation: imp's own convention for a refuted-but-kept path is a stated reason plus a condition under which it is re-tried (`docs/roadmap.md` lever ledger does this for every other refuted lever, e.g. rows 183-185).
Delta: the ledger that exists for exactly this class does not carry the largest instance of it.
Cost: one roadmap ledger row (~2 lines) if kept, or ~2550 LOC across 4 files if retired. Wrong if the FP4-attention program is expected to restart on a CUTLASS/PTX change - which is the argument to write down.
Falsifier: a roadmap or LIMITATIONS row I missed. Checked y - `rg -i 'mxfp4' docs/roadmap.md docs/LIMITATIONS.md docs/DESIGN_DECISIONS.md` returns only MXFP4 *weight-format* rows (roadmap:113, roadmap:369, LIMITATIONS:704/711/717), none about attention.

### Checked and NOT a finding

- **SETTLED S-8 (legacy cuBLAS prefill) holds today, with a fresh anchor.** `attn_legacy_softmax` (group `attention_legacy`, `tools/roofline/config.json`) is absent from all four cells of the newest roofline run `1d5b9230_20260831_180644` (nvfp4-dense/hybrid, pp512/pp4096) = 0.0%. The hd=512 exception is deliberate *and measured faster than the fused alternative* (0.52x/0.22x, `docs/audit/gemma4_attn_routing_2026_07_16/PERF_LOG.md` entry 1).
- **`causal_softmax` is not a stale roofline regex.** `rg -w causal_softmax src/` is empty, but the kernels are `causal_softmax_fp32_to_fp16_kernel` and `causal_softmax_inplace_kernel` (`src/compute/attention_cublas.cu:117,198`), which the substring regex matches. The 0.0% above is real, not a classifier miss.
- **`fmha_serves_head_dim` matches the kernel exactly.** `attention_fmha_sm120.cu:544-600` instantiates only {64,96,128,256,512} across Bq 128/64/32/16; hd=192 hits `default: break` and returns false, exactly as `src/exec/attention_dispatch_rules.h:28-30` claims. No over-conservatism in the MLA chunk clamp.
- **Stream-K CUTLASS NVFP4 is live, not test-only.** The named `gemm_nvfp4_cutlass_sm120_streamk` (`gemm_cutlass_sm120.cu:990`) is a test wrapper; the shipped path sets `decomposition_mode = force_streamk ? DM::StreamK : DM::Heuristic` inside the mainline launchers at `:785,:811,:922`. `docs/roadmap.md:182` SHIPPED default-on.
- **The 15 `gemv_*_q8_1_moe_{decode,gate_up_fused}` symbols are live**, taken by address in ternary chains (`executor_forward_moe_batch.cu:1217-1223,1241-1247,1303-1306`). The helper script called them dead; its pattern requires a trailing `(`.
- **No Marlin remnants.** 6 prose hits, no kernel, no file, no build entry.
- **`quantize_fp16_to_int8_subblock`, `pick_m_tile`, `capture_gemm_fp16_sm120_available`, `gemv`, `gemv_nvfp4`, `gemm_grouped_3x_nvfp4_cleanup`** all have in-file callers (`mmq_q4k_imma_tile.cu:419`, `gemm_grouped_nvfp4_smallM.cu`, `gemm_capture_fp16_sm120.cu:295`, `gemm_gemv_dtype.cu:53`, `nvfp4_gemm.cu:216`, own `.cu`). Not dead.
- **`mmq_q6k_imma_gemm` is test-only by a documented measurement**, not by neglect: `executor_gemm_dispatch.cu:549-556` records "4.5k vs 6.6k pp512 on Qwen3-14B-Q6_K" and why the MoE regime is different. Correctly retained.
- **Debt-ledger item 9 (`vhead_tiled_to_grouped`) is closed** - `rg -n -w` over `src/ tools/ tests/` returns only a mention in `tools/roofline/config.json:309`.
- **All seven paged decode launchers are reached from `executor_attention_decode.cu`**; no dtype family is test-only. The multitok/tile launchers are reached from inside the paged launchers.
- **B-section specialisation is not duplication** (SETTLED B re-checked): the per-KV-dtype decode kernels differ in the dequant inner step and the shared online-softmax rescale is in `attention_paged_common.cuh`; `compute_splitk_splits` (`:242-276`) is shared by five of the six dtypes. The one divergence, F16's own split-K heuristic (`attention_paged.cu:1284-1311`, cap 64 vs 32, target 2x vs 4x SMs), is a documented tuning difference, not a copy.
- **`select_attn_prefill_path` / `select_moe_prefill_path` are consulted at runtime**, not test-only (`attention_dispatch.cu:59-75` and its six call sites; `executor_forward_moe_cutlass.cu` per SETTLED F-3). No re-derivation needed.
- **`paged_attention_splitk_kernel`** (`attention_paged.cu:632`, launched `:1382`) is config-dead only under `attention.splitk_pipe=false` (default true) - a legitimate A/B arm, unlike A1-6's arch-dead pair.
- **`imp::gemm_dispatch` / `gemv_dispatch`** (`weight_dispatch.cu`, 413 LOC) are live tail fallthroughs (`executor_gemm_dispatch.cu:635`, `:237,:265,:269`), not a vestige.

### Known-and-accepted (restated)

- No GPU CI lane (SETTLED G/F-5, owner decision 2026-08-03) - the direct consequence for this axis is that every finding above about an unexercised kernel has no standing correctness check.
- Hybrid pp512 `gemm_cublas` hole: PRICED, parked, 2-3% of hybrid pp512 (`docs/roadmap.md:64,185`).
- `gemm_grouped_nvfp4` MoE prefill at ~60% of the weight floor: REFUTED twice, structural (`docs/roadmap.md:183`).
- Q4_K_M dense prefill gap vs llama.cpp: REFUTED, needs 2x weight VRAM (`docs/roadmap.md:356`, `docs/internals/SM120.md` "Open kernel work").
- `cublasLtMatmulGrouped` with NVFP4 returns zero grouped algos on sm_120 (`docs/internals/SM120.md`); CUTLASS is the primary NVFP4 GEMM path.
- FP8 prefill unavailable on sm_120 for the GDN projections: REFUTED e2e, 6/6 negative pairs (`docs/roadmap.md:184`).
- SETTLED S-11: the MoE 4-tier ladder is a designed ladder, not a twin, and "no death date exists for it".
- SETTLED S-22: the `throw` at the end of `attention_prefill_dispatch` is load-bearing; do not soften it. (A1-5 asks for an *init-time* refusal, not a softer throw.)

### Open questions

1. Does restoring the max-L1 carveout for the dp4a GEMV family move GGUF decode? Needs one GPU A/B on a Q8_0 or Q4_K GGUF (A1-3).
2. Is `gemm.moe_imma_prefill=false` still a needed escape hatch, or can the three superseded MoE prefill implementations go? Owner decision (A1-2).
3. Is the GemmKernelRegistry migration still intended, or should the 9 unreachable strategies and their ~900 test LOC be retired? Owner decision (A1-1).
4. `moe.nvfp4_smallM` is default off while the dense `gemm.nvfp4_smallm` is default on - was the MoE tier measured worse, or just never flipped? `docs/roadmap.md:183` refutes the *grouped* small-M design; whether that covers this tier is not stated.
5. Is the FP4-attention program (#846) expected to restart, or should `attention_fmha_mxfp4_sm120.cu` be retired? (A1-9)
6. Does any supported card report `sm_ver < 90`? If not, the two arch-dead split-K kernels (A1-6) can go without a compatibility argument.


## Axis A2 - Compute / kernels (hardware-feature consistency, block-scale numerics, launch config, cuBLAS autotune)

Repo `<repo>`, branch `perf/engine-h-fanin-cut-and-attention-split-verdict`, HEAD `ef664dd8`. READ-ONLY, no build, no GPU job.

### Coverage

**Read in full**
`src/compute/CLAUDE.md`, `docs/internals/SM120.md`, `docs/audit/SETTLED.md` (all 652 lines), `.claude/skills/sm120-cuda-expert/SKILL.md`, `tools/kernel_resource_baseline.txt`, `tools/kernel_resources.py`, `src/runtime/pdl.h`, `src/compute/pdl_device.cuh`, `docs/internals/KERNELS.md` sections 5-7, `src/compute/gemm_grouped_nvfp4_smallM.h`, `src/model/llm_compressor_loader.h`.

**Read in region (line ranges opened and verified)**
`src/compute/gemm.cu` (400-435, 596-690, 780-840, 1010-1030), `src/compute/gdn.cu` (380-412, 453-472), `src/compute/attention_fmha_sm120.cu` (1920-1960, 2040-2115), `src/compute/attention_paged_nvfp4.cu` (230-255), `src/compute/attention_paged_fp8.cu` (765-800), `src/compute/gemm_grouped_nvfp4_smallM.cu` (185-240, 290-520, 650-880), `src/compute/gemm_capture_fp16_sm120.cu` (1-45, 125-165), `src/compute/gemm_cutlass_sm120.cu` (70-175, 228-270, 870-905), `src/compute/activation.cu` (520-545), `src/compute/gemm_dp4a.cu` (655-800), `src/quant/nvfp4_gemm.cu` (110-340), `src/quant/mxfp4_gemm.cu` (530-560), `src/runtime/cuda_graph.cu` (90-190), `src/runtime/green_ctx.cu` (1-90), `src/exec/executor_workspace.cu` (315-350), `src/exec/executor_gemm_dispatch.cu` (30-120), `src/exec/executor_forward.cu` (760-810), `src/exec/executor_ssm_gdn.cu` (795-895), `src/core/config/attention.h` (85-125), `scripts/ci_static_gates.sh` (100-125), `.github/workflows/ci.yml` (195-220), `CMakeLists.txt` (280-300, 510-522).

**Swept by grep across `src/` + `tools/` + `tests/`** (every hit inspected)
`cp.async.bulk` / `CUtensorMap` / `cuTensorMapEncode` (2 files), `griddepcontrol` / `ProgrammaticStreamSerialization` / `pdl_wait` / `pdl_trigger` / `pdl::enable*` / `pdl::launch` (~200 hits), `__launch_bounds__` (120 hits), `mbarrier` (3 files), `cp.async` per file (17 files), `cluster_dim` / `__cluster_dims__` (0 hits), `H100|A100|B200|sm_90|sm_100|Hopper|Ampere|tcgen05|wgmma|TMEM` in `src/` (14 hits), `PreferredSharedMemoryCarveout` (9 hits), `dequant_gpu|dequantize_*` in `src/exec` + `src/compute`, `Modelopt|compressed-tensors|SfAtom`.

**Call-graph checks** `codegraph callers` on 4 symbols, each with a live control.

**Skipped** `third_party/`, CUTLASS-generated code, `src/vision/` kernels, the constrainer kernels (`json_constrain.cu`, `schema_constrain.cu`, `regex_constrain.cu`, `grammar_constrain.cu` - other axis), test bodies beyond signatures, `tools/standalone/`.

---

### Brief vs repo

| Axis-question statement | Repo | Evidence |
|---|---|---|
| Q4: "up to 2.6x prefill spread across container restarts", asks for a persist/pin mechanism | **Stale.** The magnitude was never measured; the instability is entirely in near-ties (top candidates within 5-10 %). Fixed 2026-08-04 by repairing the estimator; **R-16 (persist the algo) is REJECTED, not deferred**. The current code matches that description exactly (see "Checked and NOT a finding" N-1). Not reportable. | `SETTLED.md` G/F-9; `src/compute/gemm.cu:428-431, 602-635, 645-660` |
| Q1: "small-M `gemm_grouped_nvfp4_smallM.cu` and `src/quant/nvfp4_gemm_smallm_v2.cu`" implied as one family | Two unrelated kernels. `src/compute/gemm_grouped_nvfp4_smallM.cu` is the **MoE grouped** path (TMA + 4 producer/4 consumer warps, no PDL); `src/quant/nvfp4_gemm_smallm_v2.cu` is the **dense batched-decode** path (cp.async + mbarrier, 1 producer/4 consumer warps, PDL-instrumented and registered). | `gemm_grouped_nvfp4_smallM.cu:293,322,435`; `nvfp4_gemm_smallm_v2.cu:45,154,406` |
| Q1: "the v2 kernel" in the grouped file | There is no `smallM_kernel_v2` symbol in the tree. The kernel that the file's own header comment calls "smallM kernel v2" is named `smallM_kernel_v1` (see A2-6). | `gemm_grouped_nvfp4_smallM.cu:192` vs `:220` |
| Q2: "FP16 dequant fallbacks such as `src/quant/nvfp4_gemm.cu` calling `gemm()`" implied to sit on a decode path | It cannot. `M == 1` returns at `:214` into `gemv_nvfp4`; `M <= 16` returns at `:231-247` into the batched GEMV. The materialising fallback starts at `:251` and is prefill-only. | `src/quant/nvfp4_gemm.cu:214,231,251` |
| Q1: green contexts | Confirmed as documented, not re-flagged. `GreenContextManager::init` attempts `cudaDeviceGetDevResource` and falls back to priority streams + memSyncDomains; `has_green_contexts()` stays false on this chip. | `src/runtime/green_ctx.cu:73-90`; `docs/LIMITATIONS.md:79-82` |

---

### Q1 answer - hardware-feature treatment per hot kernel family

Legend: TMA = `cp.async.bulk.tensor` / `CUtensorMap`. WS = warp-specialised producer/consumer. PDL = `pdl_wait` / `pdl_trigger` present, and whether the kernel is *registered* (registration is what makes a graph edge programmatic, `cuda_graph.cu:127-155`). Graph = launched inside the captured decode graph.

| # | Family | File(s) | TMA | WS | PDL wait/trig | PDL registered | `__launch_bounds__` | In decode graph |
|---|---|---|---|---|---|---|---|---|
| 1 | FA2 prefill fp16/fp8 | `attention_fmha_sm120.cu` | no | no | no | no | `(SM120_BLOCK_THREADS,2)` :70, `(…,1)` :679, `_2cta (Bq/16*32,2)` :1949 | no (prefill) |
| 2 | FA2 MXFP4 prefill | `attention_fmha_mxfp4_sm120.cu` | no | no | no | no | `(MX_BLOCK_THREADS,1)` :190 | no |
| 3 | Paged decode F16 | `attention_paged.cu`, `attention_paged_f16_multitok.cu` | no (cp.async) | no | no | no | `(1024)` :93; `(BLOCK_THREADS)` :196,:263 | yes |
| 4 | Paged decode FP8 | `attention_paged_fp8.cu`, `_fp8_tile.cu`, `_fp8_multitok.cu` | no (cp.async) | no | no | no | `(BLOCK_THREADS)` :47 | yes |
| 5 | Paged decode NVFP4 | `attention_paged_nvfp4.cu`, `_tc.cu`, `_multitok.cu`, `_multitok_gqa.cu` | no | no | **trigger only, 1 of 10 kernels** (`:246`) | no | `(BLOCK_THREADS)` :120,:187,:170,:241; `_tc` deliberately none (`:53`) | yes |
| 6 | Paged decode INT4/INT8 | `attention_paged_int4.cu`, `_int8.cu` | no (cp.async) | no | no | no | none | yes |
| 7 | CUTLASS dense + grouped NVFP4 | `gemm_cutlass_sm120.cu`, `gemm_cutlass_grouped_3x.cu` | **yes** (CUTLASS TMA-WS, `cuTensorMapEncodeTiled` :875-900) | **yes** | no | no | n/a (CUTLASS) | grouped: yes (MoE fast path); dense: prefill |
| 8 | MoE grouped small-M | `compute/gemm_grouped_nvfp4_smallM.cu` | **yes** (`cp.async.bulk.tensor.2d` :71, `build_tma_2d_u8` :684) | **yes** (4 prod / 4 cons :322,:435) | no | no | none | no (MoE prefill) |
| 9 | Dense small-M v2 | `quant/nvfp4_gemm_smallm_v2.cu` | no (cp.async + mbarrier :16) | **yes** (1 prod / 4 cons :45) | yes :154,:276,:352 | **yes** :406,:416,:421,:470 | none | yes |
| 10 | NVFP4 GEMV dense + fused | `quant/nvfp4_gemv_dense.cu`, `_fused.cu` | no | no | yes | **yes** (`nvfp4_gemm.cu:304-318`) | `(kKparThreads,12)` / `(kMRThreads)` | yes |
| 11 | NVFP4 GEMV **batched** | `quant/nvfp4_gemv_batched.cu` | no | no | yes :43,:92,:147,:171,:193,:240 | **NO** - see A2-2 | `(kKparThreads)` / `(kMRThreads)` | yes |
| 12 | NVFP4 GEMV MoE | `quant/nvfp4_gemv_moe.cu` | no | no | no (stated: `nvfp4_gemm.cu:320`) | no, carveout kept | `(kKparThreads,12)`; `:98` records "no `__launch_bounds__` per sm120 perf testing" | yes |
| 13 | dp4a GEMV (GGUF decode) | `gemm_dp4a.cu`, `gemv_dp4a_traits.cuh` | no | no | no | **register fn is DEAD** - see A2-1 | none on the GEMV templates | yes |
| 14 | IMMA MMQ prefill | `mmq_q8_imma*.cu`, `mmq_q4k_imma_tile.cu` | no (cp.async) | no | no | no | `(kThreads)`; measurement recorded at `mmq_q8_imma.cu:319` | no |
| 15 | GDN scan fused | `gdn.cu` | no | no | yes :79,:293 | yes, at every launch site | `(HD*SPLIT,1)` :47 | yes |
| 16 | GDN chunkwise / TC | `gdn_scan.cu`, `gdn_scan_tc.cu` | no | no | no | no | `(HD,1)` | no |
| 17 | GDN chunk-parallel | `gdn_scan_chunkpar.cu`, `_pass.cu` | no | no | no | no | `(2*HD,1)` :157, `(kPassThreads,1)` :100 | no |
| 18 | SSM conv1d decode | `ssm.cu` | no | no | yes :169,:202 | yes :219,:232 | none | yes |
| 19 | RMSNorm | `layernorm.cu`, `layernorm_rowblock.cu` | no | no | fp16 yes; fp32 no | fp16 block/warp/residual + rowblock yes; fp32 no | exec-side `(512)` | yes |
| 20 | RoPE | `rope.cu` | no | no | fused qk-norm+rope yes :224,:293 | fused yes :407; `rope_forward_kernel` uses `pdl::launch` but is **not** registered (degrades to `<<<>>>`) | exec-side `(256)` | yes |
| 21 | Activation / quantize | `activation.cu` | no | no | **no** | `activation_pdl_register()` is **DEAD** - see A2-1 | none | yes |
| 22 | Sampling | `sampling.cu`, `sampling_penalties*.cu`, `sampling_topk_topp.cu` | no | no | argmax + penalties yes | argmax + penalties yes | exec-side `(256)` | yes |
| 23 | KV write | `exec/executor_kv_write_kernels.cu` | no | no | NVFP4 only :269 | NVFP4 only (`executor_kv_write.cu:95`) | `(256)` on all | yes |

**Reasons the repo itself states for the TMA/WS gaps** (so these are not findings):
- `docs/internals/KERNELS.md:101` - no TMEM/TMA-WS means the FA4 producer-warpgroup design is not buildable on sm_120a.
- `docs/internals/KERNELS.md:117-139` - TMA + warp specialisation was *built* as `tools/standalone/gemm_nvfp4_sm120a_tma.cu` and is the documented negative result; the follow-up occupancy hypothesis was refuted 2026-06-17 and the verdict is "keep CUTLASS TMA + warp-spec as the production dense NVFP4 GEMM; do not port cp.async + layout into prod".
- `docs/internals/SM120.md` "Why batch=1 decode is memory-bound" - "TMA for paged KV: `cp.async` already at near-peak" and "Warp specialization: all warps already stream KV at peak bandwidth".
- `src/exec/executor_workspace.cu:328-337` - the un-instrumented PDL families are un-registered on purpose ("those kernels do not wait yet, and registering them raced").
- An unmeasured TMA probe for the FMHA V-tile exists and has **no recorded verdict** in the tree: `tests/bench/fmha_v_load_bench.{cu,h}` (kernels `bench_separate` / `bench_tma_v_load` / `bench_fused`, pinned in `tools/kernel_resource_baseline.txt`). Listed under Open questions, not as a finding.

---

### Findings

### [A2-1] `gemv_pdl_register()` and `activation_pdl_register()` are dead; the dp4a GEMV family silently lost its MaxL1 carveout with them
Axis: A2   Sev: S2   Confidence: high (deadness), low (perf magnitude - HYPOTHESIS)
Evidence:
- `codegraph callers gemv_pdl_register` -> `No callers found`; control `codegraph callers nvfp4_gemv_pdl_register` -> 1 caller (`src/exec/executor_workspace.cu:163`). Same for `activation_pdl_register` (no callers) vs `mxfp4_gemv_set_l1_carveout` (1 caller, `executor_workspace.cu:350`).
- `grep -rn 'gemv_pdl_register' src/` -> declaration `src/compute/gemm.h:313`, definition `src/compute/gemm_dp4a.cu:669-798` (130 LOC), nothing else. Same shape for `activation_pdl_register` (`src/compute/activation.h:41`, `src/compute/activation.cu:531-534`).
- `git log -S 'gemv_pdl_register();' --oneline -- src/exec/` -> `0af8f80d perf(pdl): device half of programmatic dependent launch in the decode kernels (#1833)`. That commit removed the call site.
- The dead function does two things per kernel: `pdl::enable_kernel(...)` **and** `SET_MAXL1(...)` = `cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1)` (`gemm_dp4a.cu:664-668`), applied to ~40 dp4a GEMV instantiations (`REG1`..`REG5`, Q2_K/Q3_K/Q4_0/Q4_K/Q5_K/Q6_K/Q8_0 x NR).
- `grep -rn 'PreferredSharedMemoryCarveout' src/` returns 9 sites; none of the other 8 covers a `gemv_dp4a_*` kernel.
Current: NVFP4 and MXFP4 kept the carveout when PDL registration was withdrawn - `nvfp4_gemm.cu:320-325` explicitly drops registration for the MoE GEMVs while keeping `cudaFuncSetAttribute(..., MaxL1)`, and MXFP4 has a *separate* live `mxfp4_gemv_set_l1_carveout()` (`mxfp4_gemm.cu:538-551`). The dp4a family had both bundled into one function and lost both.
Expectation: a cache-carveout hint for a bandwidth-bound GEMV is orthogonal to PDL registration and should not be coupled to it; both NVFP4 and MXFP4 in this same tree decouple them.
Delta: the GGUF decode GEMV family (Q4_K / Q6_K / Q8_0 - the perf-gate model's own decode path per `SETTLED.md` F-9) runs without the L1 preference the sibling families keep. Secondary: the dead functions are loaded guns - re-adding either call registers kernels that have **no** `pdl_wait()` (`gemv_dp4a_kernel` reads `q8_1[blk].qs` at `gemv_dp4a_traits.cuh:819` and `d8[i]` at `:825` with no wait), which is precisely the race `executor_workspace.cu:332-336` records as having failed `DegenerationTest.GreedyDeterminism`.
Cost: delete the two dead functions and their declarations (4 files, ~140 LOC), or split the carveout out of `gemv_pdl_register` into a live `gemv_dp4a_set_l1_carveout()` called next to `mxfp4_gemv_set_l1_carveout()` (`executor_workspace.cu:350`), ~15 LOC. Risk of the carveout half: the dp4a `gemv_dp4a_kernel` family uses dynamic smem (`extern __shared__ char smem_q8[]`, `traits.cuh:812`) so MaxL1 is a *preference* the driver overrides; the kpar variants launch with `smem=0` and are the ones that would actually take it. If wrong: a neutral A/B, no correctness exposure.
Falsifier: another live call site sets the carveout for `gemv_dp4a_*`, or the carveout measures neutral on Q8_0 decode. Checked: call-site search y (none exists, with a control). Measurement: n (no GPU in this audit; the tree contains no A/B for this carveout).

### [A2-2] Batched-decode NVFP4 GEMVs pay for PDL instrumentation and never get it: `pdl_wait`/`pdl_trigger` present, kernel never registered
Axis: A2   Sev: S2   Confidence: high (mechanism), low (magnitude - HYPOTHESIS)
Evidence:
- `src/quant/nvfp4_gemv_batched.cu` instruments all three kernel families: `pdl_wait()` at `:43, :147, :193`, `pdl_trigger()` at `:92, :171, :240`, and launches every one through `pdl::launch` (`:278-440`, 16 sites).
- The registry `nvfp4_gemv_pdl_register()` (`src/quant/nvfp4_gemm.cu:292-328`) lists 14 kernels; **none** of `gemv_nvfp4_kpar_mb_fp32_kernel`, `gemv_nvfp4_kpar_mb_fp16_kernel`, `gemv_nvfp4_multirow_mb_kernel` appears. Repo-wide `pdl::enable*` sweep finds no other registration for them.
- `pdl::launch` on an unregistered kernel falls through to `func<<<...>>>` (`src/runtime/pdl.h:52-68`), and `apply_pdl_edges` converts a graph edge only when the **consumer** is registered (`src/runtime/cuda_graph.cu:148-153`). So both halves are inert here.
Current: the kernels execute `griddepcontrol.wait` / `griddepcontrol.launch_dependents` (no-ops without a programmatic dependency, `pdl_device.cuh:23-31`) and the edge stays `cudaGraphDependencyTypeDefault`.
Expectation: PDL is shipped default-on and measured to pay exactly in this regime - `docs/roadmap.md:139` "PDL device half (`griddepcontrol`) | SHIPPED default-on | @32 3/3 positive (+1.3% median), M=1 +1.7% median, idle 13.6 -> 10.8%". The `*_mb_*` kernels are the batched-decode (M<=32) path, i.e. the @32 serving regime that number was measured in.
Delta: the family whose regime produced the +1.3% is the one family that is instrumented but unregistered. The gain is not claimed here, only that the mechanism is disconnected.
Cost: 3 `NVFP4_REGISTER(...)` lines in `nvfp4_gemm.cu:292-328` (the template instantiations must be named, so ~10 lines for the `<1..4>` and `<NR,1..4,bool>` sets). Risk: a wrongly-registered kernel races; here the `pdl_wait()` is already present and is placed before the first global read, so the contract holds. Must be re-validated with `check-degeneration` (`GreedyDeterminism`) plus a 32-stream A/B, since #1833 records that this exact class raced when registration ran ahead of instrumentation.
Falsifier: the `*_mb_*` kernels are unreachable in production, or registration was withheld deliberately (the file/commit would say so - `nvfp4_gemm.cu:320` states the reason for the MoE GEMVs and says nothing about the batched ones). Checked: registration sweep y (absent); reachability y (`nvfp4_gemv_batched.cu:33` "consumer at batch>1", dispatched from the batched decode GEMM path).

### [A2-3] The single-sequence GDN scan runs the SPLIT=1 instance that spills; its batched sibling runs the measured, non-spilling SPLIT=2
Axis: A2   Sev: S2   Confidence: med   (perf delta at n_seq=1: HYPOTHESIS)
Evidence:
- `tools/kernel_resource_baseline.txt` pins four SPLIT=1 instances at the register ceiling with a local frame:
  `gdn_scan_fused_kernel<128,128,__half,1,float>` REG 255 STACK 96, `<128,128,float,1,float>` 255/96, `<128,128,__half,1,__nv_bfloat16>` 255/88, `<128,128,float,1,__nv_bfloat16>` 255/88. `tools/kernel_resources.py:18-20` defines STACK as "per-thread local frame ... spilled registers, or an indexed local array"; at REG 255 it is a spill.
- `src/compute/gdn.cu:396` `constexpr int SPLIT = 2;` with a 16-line measured table above it (`:380-395`) and `docs/plans/2026-08-24-qwen38-port.md:683-694`: SPLIT=1 = "255 + spill", 41.65 us at n_seq=16; **SPLIT=2 = 180 regs, 26.49 us** (-36 %); "SPLIT=1 would be perfectly coalesced and spills instead (128 B spill stores). SPLIT=2 is the one point where the spill is gone".
- That constant is used only in `gdn_scan_fused_f32_batched` (`:360`) and `gdn_scan_fused_bf16_batched` (`:421`). The four single-sequence entry points instantiate SPLIT=1 with no comment and no reason: `gdn_scan_fused_f32` `:462-463`, `gdn_scan_fused_bf16` `:504-505`, `gdn_scan_fused_fp32out` `:520-521`, `gdn_scan_fused_fp32out_bf16` `:567-568`.
- Those four are the batch=1 / short-prefill route: `src/exec/executor_ssm_gdn.cu:813-820` selects them when `rows < 128` (chunk-parallel needs `rows >= 128`), and `:883-890` is the single-sequence branch. This is the GDN model family's decode hot path (Qwen3.5 / 3.6 / 3.8).
Current: the same kernel template ships at 180 registers with no spill on the batched path and at 255 registers with an 88-96 B local frame on the single-sequence path.
Expectation: unclear as a general engine claim; the *repo's own* measurement is the expectation here, and it was applied to one of the two call sites.
Delta: an asymmetric application of a measured lever with no stated reason. The measured table covers n_seq=16/32/64 only, so n_seq=1 is genuinely unmeasured - `SPLIT` splits the state dimension across threads, not sequences, so the coalescing/occupancy trade at n_seq=1 is not obviously the same.
Cost: `src/compute/gdn.cu`, 4 functions, ~12 LOC (SPLIT constant, block dim `128*SPLIT`, smem term). Risk: SPLIT changes the intra-block reduction partition, so output is bit-different; a recurrent state path needs the numerics judge from `sm120-cuda-expert` (unit-test state diff + deterministic PPL on Qwen3.8-27B-NVFP4-vllm). The SPLIT=2 arithmetic already ships on the batched path, which bounds the numerics risk.
Falsifier: SPLIT=2 measures neutral or worse at n_seq=1 (plausible - 48 heads x 256 threads still underfills 170 SMs, and the coalescing loss is real). Checked: n. Needs a GPU A/B.

### [A2-4] FP8 paged decode never got the multitok treatment its F16 and NVFP4 siblings have: HD=128 only, no split-K, no Q-head grouping
Axis: A2   Sev: S2   Confidence: high (gap), low (magnitude - HYPOTHESIS)
Evidence:
- `src/compute/attention_paged_fp8_multitok.cu` contains exactly one kernel, `paged_attention_decode_fp8_multitok_kernel` (`:47`), with `static_assert(HEAD_DIM == 128, "multitok kernel is the HD=128 instance")` (`:52`), and one launcher `paged_attention_decode_fp8_multitok_hd128` (`:194`).
- `src/compute/attention_paged_f16_multitok.cu` has two kernels (`:196` decode, `:263` split-K), a Q-head-grouping helper `paged_attention_f16_multitok_heads_per_cta` (`:373`) and two launchers; the head-dim assert admits both HD=128 and HD=256 (`:202,:269`).
- `src/compute/attention_paged_nvfp4_multitok.cu` likewise has decode (`:120`) and split-K (`:187`) kernels, plus a whole separate GQA-grouped file `attention_paged_nvfp4_multitok_gqa.cu` (`:170, :241`).
- The dispatch falls back silently and states no reason: `src/compute/attention_paged_fp8.cu:771-793` - `case 128:` takes the multitok kernel when `process_diag_paged_fp8_multitok() > 1`; `case 256: LAUNCH_FP8_FALLBACK(256); break;` with no comment.
- `src/core/config/attention.h` documents the three knobs asymmetrically: `paged_fp8_multitok` "FP8 paged decode (HD=128)", `paged_nvfp4_multitok` "(HD=128/256)", `paged_f16_multitok` "(HD=128/256, GQA ratio 1..8) ... Split-K route too".
Current: an FP8-KV model with head_dim 256 (the Qwen3.5/3.6/3.8 shape) and any FP8-KV model on the split-K (long-context, single-stream) route runs the scalar per-token kernel.
Expectation: the sibling kernels in the same directory cover those shapes; the config comments record what the treatment was worth on the shapes it does cover - F16 `16/8 HD=256 665 -> 178 us` and split-K `1 x 32k 197 -> 109 us`, FP8 `209 -> 92 us` at HD=128 (`docs/PERF.md` refs in `attention.h:109-127`).
Delta: one of three KV dtypes covers one of three axes (head dim, split-K, Q-head grouping) that the other two cover.
Cost: an HD=256 instantiation plus a `case 256` dispatch arm is ~40 LOC in two files if the kernel generalises the way the F16 one does; the split-K variant is a second kernel (~80 LOC), the same shape as `attention_paged_f16_multitok.cu:263-372`. Risk: `attention_paged_fp8_multitok.cu:54 static_assert(ELEMS == 4, "one uint32 per lane")` says the lane layout is hard-wired to 4 FP8 bytes; HD=256 needs 8 (uint2), i.e. a real port, not a template widening. If wrong: attention output corruption, caught by `check-degeneration` + NIAH.
Falsifier: FP8 KV is never selected at HD=256 (e.g. the resolver forbids it), making the arm unreachable. Checked: partially - `attention.h:101-107` gates `fa2_hd256`'s "FP8-KV deterministic-cuBLAS skip for hd=256 models", which implies FP8 KV at hd=256 is a served configuration. A definitive check needs the resolver trace on a real hd=256 model.

### [A2-5] Only 1 of 10 paged-decode kernels calls `pdl_trigger()`, so the programmatic edge into the o_proj GEMV degrades to a default edge for every other KV dtype
Axis: A2   Sev: S2   Confidence: med   (magnitude: HYPOTHESIS)
Evidence:
- Repo-wide `pdl` grep per file: `attention_paged_nvfp4.cu` = 2 hits (`#include` at `:11`, `pdl_trigger()` at `:246`). All nine other paged-decode files (`attention_paged.cu`, `_f16_multitok.cu`, `_fp8.cu`, `_fp8_multitok.cu`, `_fp8_tile.cu`, `_int4.cu`, `_int8.cu`, `_nvfp4_multitok.cu`, `_nvfp4_multitok_gqa.cu`, `_nvfp4_tc.cu`) and `attention_paged_common.cuh` = 0 hits.
- Even inside `attention_paged_nvfp4.cu` the split-K sibling kernel (from `:255`) has no trigger; only the non-split-K scalar decode kernel does.
- The consumer of attention output at decode is the o_proj GEMV, which *is* registered (`nvfp4_gemm.cu:304-318`), so `apply_pdl_edges` converts the edge (`cuda_graph.cu:127-155`) and then finds a producer that never issues `griddepcontrol.launch_dependents` - the dependent grid launches only when every producer block exits, i.e. the default behaviour.
Current: the tail-overlap half of PDL exists for exactly one kernel in the decode-attention family, including the multitok kernels that are the shipped default at 32 streams (`attention.h`: `paged_f16_multitok = 4`, `paged_fp8_multitok = 4`, `paged_nvfp4_multitok = 4`).
Expectation: `src/runtime/pdl.h:33-42` states the design - "a registered kernel calls `pdl_wait()` before its first global access and `pdl_trigger()` after its last input read, so a programmatic edge really lets the consumer's blocks land on the SMs during the producer's tail". `docs/roadmap.md:139` records the shipped gain.
Delta: the treatment landed on one kernel and its siblings in the same directory did not get it. `executor_workspace.cu:328-337` states the reason for the *consumer* half being incremental ("those kernels do not wait yet"); nothing states a reason for the *producer* half.
Cost: one `pdl_trigger()` line per kernel after its last KV read (the pattern is `attention_paged_nvfp4.cu:246`, placed before the cross-warp reduce and the O store), ~10 files, ~10 LOC. Risk: LOW for a producer trigger - it changes scheduling only, never visibility (`pdl_device.cuh:11-14`), and the consumer's `pdl_wait()` still blocks until the producer grid completes. Still needs `GreedyDeterminism` + a 32-stream A/B because #1833 records this family as having raced once.
Falsifier: attention output is not consumed by a PDL-registered kernel at decode (then no edge is programmatic and the trigger is moot), or the attention kernel's tail is too short to overlap. Checked: consumer registration y (o_proj GEMV registered). Tail length: n, needs nsys.

### [A2-6] `tools/kernel_resource_baseline.txt` contradicts the register/spill numbers the FA2 tile levers were decided on
Axis: A2   Sev: S3   Confidence: high
Evidence:
- Baseline (ptxas, `cuobjdump -res-usage`, committed artifact):
  `fmha_sm120_fa2_kernel<64,256,true,false,64,true,false,false>` REG **255** STACK **96**
  `fmha_sm120_fa2_kernel<64,256,true,true,64,true,false,false>` REG **255** STACK **80**
  `fmha_sm120_fa2_kernel<64,256,true,false,32,true,false,false>` REG **246** STACK **0**
  `fmha_sm120_fa2_kernel<64,256,true,true,32,true,false,false>` REG **246** STACK **0**
  `fmha_sm120_fa2_kernel_2cta<128,128,true,true,64,true,true,false>` REG 128 STACK **40**
- Template order is `<Bq, HD, FP16QK, F16ACC, BKV, TWOSLOT, PVF16, FP8SCALED>` (`attention_fmha_sm120.cu:1932`), and the dispatch at `:2067-2073` shows `<64,256,true,false,64,true>` is the **shipped default** HD=256 instance (`fa2_hd256_bkv = 64`, `src/core/config/attention.h:99`).
- The record the lever was decided on says something else: `src/core/config/attention.h:95-99` "the 232 registers allow two", and `docs/plans/2026-09-04-lever-ledger-detail.md` "holds 232 regs x 128 threads = 29696 of 65536 (two CTAs fit) ... Bkv=32 halves the tile to 34816 B, **226 regs, 0 spills**". Neither mentions a spill on the default instance. For the 2-CTA wrapper, `attention.h:101-103` says "128 registers (137 unconstrained, **24 B of spill**)" against the baseline's STACK 40.
Current: two committed artifacts in the same tree disagree on the register count (232/226 vs 255/246) and on whether the shipped HD=256 instance has a local frame at all.
Expectation: `tools/kernel_resources.py:12-15` is explicit that the baseline is the ptxas-derived artifact and exists precisely because "no committed artifact carried a per-kernel register or spill number". A lever record that quotes different numbers for the same instance defeats that.
Delta: the ledger's conclusion ("smem is the binding limit, registers allow two CTAs") still survives at 255 regs x 128 threads = 32640 of 65536, so the verdict does not flip. What is lost is the *spill* on the default path: the record names Bkv=32's "0 spills" as an incidental detail rather than as a difference from the default. A next pass reading `attention.h` will not know the shipped kernel spills 96 B/thread.
Cost: text only. Reconcile in `src/core/config/attention.h:95-107` and `docs/plans/2026-09-04-lever-ledger-detail.md`, ~6 lines. Risk: none.
Falsifier: the ncu numbers describe a different instantiation than the baseline's (e.g. the `pv_f16` twin, which is below the 240 pin and therefore absent from the baseline). Checked: partially - the ledger names "the hybrid FA2 instance (`fa2<64,256,...,TWOSLOT>`)" without the F16ACC/PVF16 bits, so the mapping is ambiguous; that ambiguity is itself the defect.

### [A2-7] `gemm_grouped_nvfp4_smallM.cu`: the kernel's own header comment names the wrong version and the wrong warp split
Axis: A2   Sev: S3   Confidence: high
Evidence:
- `:192` "smallM kernel **v2** - TMA loads + 3-stage producer/consumer pipeline", `:220` `__global__ void smallM_kernel_**v1**(`. `grep -rn 'smallM_kernel_v2|smallM_v2|kernel_v2' src/ tests/ tools/` returns nothing - there is no v2 symbol.
- Same comment block, `:199-201` "Single producer thread (lane 0 of warp 0) issues TMA" and `:209` "Consumers (**all 8 warps**) wait on stage's mbarrier"; the code is `:293` "Warp-specialized split: **4 producer warps + 4 consumer warps**", `:322` `const bool is_producer = (warp_id < N_PRODUCER_WARPS);`, `:435` `const int consumer_warp = warp_id - N_PRODUCER_WARPS;`.
- `src/quant/nvfp4_gemm_smallm_v2.cu` is an unrelated kernel with a genuine `_v2` name, which is what makes the mislabel actively confusing (it caused a wrong premise in this audit's own brief - see "Brief vs repo").
Current: the file's navigation comment describes a design the file does not implement, in a 900-LOC TU whose static asserts and error strings all say `smallM_kernel_v1`.
Expectation: `src/compute/CLAUDE.md` "Numeric code is bit-sensitive: move a kernel verbatim, and say so"; a comment that names the wrong warp partition is worse than none for a producer/consumer mbarrier kernel.
Delta: comment/code drift on the only TMA + warp-specialised kernel imp wrote itself.
Cost: `src/compute/gemm_grouped_nvfp4_smallM.cu`, ~8 comment lines. Risk: none.
Falsifier: the comment describes an earlier revision kept deliberately. Checked: n; nothing in the file says so.

### [A2-8] The M=1 `beta != 0` uncached GEMM fallback materialises the whole weight per token, silently
Axis: A2   Sev: S2   Confidence: low   (reachability unverified)
Evidence:
- `src/exec/executor_gemm_dispatch.cu:42` "Uncached fallback: safety net for weights without a WeightHandle (kInvalidTensorID, budget-exhausted) **and for M=1 beta!=0 residual-add**".
- `:53-66`: on `beta != 0`, if the weight is not in `wc->fp16`, it runs `dequant_gpu(weight.data, qs->dequant, qtype, rows, cols, ctx.stream)` over the full `[rows, cols]` weight and then `gemm(...)`, per call.
- No first-use warning on that branch. The analogous NVFP4 fallback does warn once (`src/quant/nvfp4_gemm.cu:250-262` "using slow dequant-to-FP16 fallback"), and the `dropped_source` arm eight lines below warns once (`:93-101`).
- Entered from `gemm_via_handle_` when `h.primary_tier == StorageTier::Undefined` (`:168-176`) and as the final fallback (`:490`, `:556`).
Current: a decode step can pay a full-weight dequant-plus-copy per token with nothing in the log.
Expectation: this repo's own convention - `src/compute/CLAUDE.md` "A kernel that cannot serve its input fails loud", and every sibling degradation path here warns once.
Delta: the one degradation that lands on the per-token path is the one that is silent.
Cost: 5 LOC (a `static bool s_warned` one-shot naming the qtype and the byte count), `src/exec/executor_gemm_dispatch.cu:59`. Risk: none.
Falsifier: the branch is unreachable in production because every `beta != 0` weight is in `wcache_.fp16` by construction. Checked: n - the `wc->fp16.find` at `:54` is tried first, so reachability depends on the budget-exhaustion path; deciding it needs a run with a VRAM-constrained config.

### [A2-9] `generation.lm_dequant_fp16` is a live config key that runs a debug bisect on the lm_head, with a per-forward `cudaMallocAsync`
Axis: A2   Sev: S3   Confidence: high
Evidence:
- `src/exec/executor_forward.cu:783-800`: the comment is an investigation note - "DEBUG experiment (b): bypass fused rmsnorm_quantize_q8_1 + dp4a GEMV ... If this gives llama-matching top logit (~+2.07) then the dp4a path is buggy; if it still gives +8.83 then the bug is in hidden state or output_norm". The body does `cudaMallocAsync` of `vocab_size * d_model * 2` bytes, `dequant_gpu` of the whole lm_head, `gemm`, `cudaFreeAsync` - every forward.
- It is user-reachable: `src/core/config/generation.h:29 bool lm_dequant_fp16 = false;` and `src/runtime/config.cpp:319 B("generation.lm_dequant_fp16", ...)`, i.e. `--set generation.lm_dequant_fp16=true`.
Current: a finished bisect from a past defect hunt is a shipped runtime knob on the decode path.
Expectation: `docs/DESIGN_DECISIONS.md` / `SETTLED.md` C treats one-off debug scaffolding as removable; the surviving diagnostic knobs in this repo (`diagnostics.*`) are namespaced as diagnostics, not as `generation`.
Delta: knob namespace and lifetime. Not a correctness or default-path perf issue.
Cost: remove the branch and the key (3 files, ~20 LOC), or move it under `diagnostics.` with a comment saying it is a bisect. Risk: an external `imp.conf` setting the key would then warn on an unknown key.
Falsifier: the knob is documented as a supported diagnostic. Checked: y - `grep -rn 'lm_dequant_fp16' docs/` returns nothing.

---

### Checked and NOT a finding

- **N-1. F-9 (cuBLASLt algo selection) - current code matches `SETTLED.md` exactly; no new finding.** `src/compute/gemm.cu:428-431` holds `kBenchmarkRounds = 3`, `kTargetWindowMs = 0.5f`, `kMaxBenchIters = 512`, `kAlgoMargin = 0.10f`; `:590-596` sizes each candidate's rep count to the window; `:608-635` is the paired comparison ("Each round compares every candidate against `base` timed in that SAME round ... a candidate keeps its claim only by beating base by `kAlgoMargin` in every round"); `:645-660` takes the lowest-indexed candidate that won every round. There is no persistence, no on-disk cache, no version-keyed store anywhere in the file - `grep -n 'benchmark_and_select_algo' src/compute/gemm.cu` returns the definition (`:438`) and two call sites (`:832`, `:1025`), both on a cache miss inside the in-process `s_gemm_cache`. The rejected designs are documented in place at `:410-427`.
- **N-2. Which paths reach cuBLASLt.** The single entry is `gemm()` (`gemm.cu:~800`) plus the FP8-scaled variant (`:~1000`); both benchmark on a per-(shape, dtype) cache miss. F-9's measured statement ("only reachable from dense BF16/FP16 SafeTensors and the vision encoders; a Q8_0 GGUF never enters it") is about the perf-gate model, whose GGUF prefill takes the INT8-IMMA route (`mmq_q8_imma*`), not about a structural guard: any dequant-to-FP16 fallback that ends in `gemm()` (`executor_gemm_dispatch.cu:62-65, :106-110`, `gemm_kernel_generic_dequant.cu`, `quant/nvfp4_gemm.cu:283`) does enter cuBLASLt. Not a finding - it does not contradict F-9, which never claimed a guard.
- **N-3. cuBLASLt is not used inside a captured decode graph at all.** `src/compute/gemm_capture_fp16_sm120.cu:1-9` - "cuBLASLt fails with `CUBLAS_STATUS_INTERNAL_ERROR` on the first GEMM under `cudaStreamCapture` on sm_120"; `gemm.cu:784` substitutes the hand-rolled WMMA kernel under capture. So algorithm selection cannot affect graph-replayed decode.
- **N-4. The NVFP4 dequant-to-FP16 fallback is not on a decode path.** `src/quant/nvfp4_gemm.cu:214` returns to `gemv_nvfp4` at M==1; `:229-247` routes M<=16 through the batched GEMV with the reason recorded ("on Qwen3.6-27B MTP-only verify that was 49% of all GPU time, ~52 dequants x ~600 us per verify"); the materialising path starts at `:251` and throws rather than silently skipping under capture (`:268-278`).
- **N-5. Init-time dequant caches are init-time.** `src/exec/pre_dequant_phase0_nvfp4_loader.cu`, `phase1_fp16_cache.cu`, `phase2_fp8_cache.cu`, `phase3_cutlass.cu`, `phase3_fp8.cu`, `phase3_moe.cu`, `phase3_nvfp4_decode.cu`, `phase3c_mxfp4.cu`, `phase4_tensor_registry.cu` - all reached from `Engine::init` weight upload, none per request.
- **N-6. `gemm_fp16_kernel`'s 32-byte local frame is not a spill.** `gemm_capture_fp16_sm120.cu:150-151` declares `__half* A_stage[STAGES]; __half* B_stage[STAGES];` with STAGES=2 = 4 pointers x 8 B = exactly the 32 B the baseline pins - and it is 32 B for *both* the 243-register `<128,2>` and the 156-register `<64,2>` instance, which a register spill would not be. The `__launch_bounds__(THREADS_PER_BLOCK, 2)` at `:138` is therefore not forcing a spill. Same reading for `smallM_kernel_v1<128,128,128,3>` (REG 64, STACK 544) and `topk_*_kernel` (REG 39-44, STACK 1056): low registers with a large frame is an indexed local array.
- **N-7. No datacenter-SKU tuning anywhere in `src/`.** All 14 hits of `H100|A100|B200|sm_90|sm_100|Hopper|Ampere|tcgen05|wgmma|TMEM` are comments stating the feature is absent on sm_120a, or `sm_90+` guards on `griddepcontrol` / `cp.async` pipelined variants. No CUTLASS tile config imported from a datacenter default: `gemm_cutlass_sm120.cu:73-74` is `ThreadBlockShape = Shape<_128,_128,_128>` with `ClusterShape = Shape<_1,_1,_1>` and the comment "GeForce = no multicast"; the small-N variant `:154` is `Shape<_128,_64,_128>`. Confirms `SETTLED.md` C ("Zero stale-target code").
- **N-8. No cluster launch / distributed shared memory anywhere.** `grep -n 'cluster_dim|clusterDim|cudaLaunchAttributeClusterDimension|__cluster_dims__' src/` -> 0 hits, consistent with `ClusterShape<_1,_1,_1>`.
- **N-9. NVFP4 scale-layout format confusion (Modelopt vs compressed-tensors reciprocals) is closed.** `src/model/llm_compressor_loader.h:79-88` records the exact failure ("every weight comes out scaled by amax^2/36 with nothing failing ... perplexity 31.05 against 1.2e47") and the fix: detection now reads `config.json`'s `quantization_config` (`parse_compressed_tensors_config`, `llm_compressor_loader.cpp:353-422`) as well as `recipe.yaml`, with the precedence chain at `hf_config_loader.cpp:1185-1214`.
- **N-10. Block-scale / alpha arithmetic is tested.** `tests/test_cutlass_nvfp4_alpha.cu` (698 lines) asserts alpha is actually applied (`AlphaIsActuallyApplied`, mean output ratio must track the alpha ratio), the FP8-E4M3 scale-encoder clamp boundary, the Mistral L0 reproducer, and a byte-level prequant layout check against the compressed-tensors convention `global_scale = FP8_max * FP4_max / max(|W|)`. Plus `test_nvfp4_quant_ref.cu`, `test_nvfp4_compressed_tensors_ref.cu`, `test_nvfp4_quant_hw.cu`, `test_nvfp4_outlier_ref.cu`, `test_cutlass_grouped_ref.cu`, `test_quantize_fp16_nvfp4_moe_native.cu`. They are GPU tests, so they never run in CI - that is F-5, already accepted.
- **N-11. The `kernel-resources` gate is in the required `Build` job and is a real two-way ratchet.** `.github/workflows/ci.yml:212-213` runs `cuobjdump -res-usage build/libimp.a | python3 tools/kernel_resources.py -` after the compile step; `tools/kernel_resources.py:200-215` fails on NEW, IMPROVED **and** MOVED. `scripts/ci_static_gates.sh:104-116` skips it locally when no `libimp.a` exists, and says so.
- **N-12. `__launch_bounds__` is never applied blind on a GEMV/attention path.** `src/compute/attention_paged_nvfp4_tc.cu:53` records the deliberate absence; `src/quant/nvfp4_gemv_moe.cu:98` "no `__launch_bounds__` - per sm120 perf testing the override costs"; `src/compute/mmq_q8_imma.cu:319` carries the measured table for `(256,2)`; `src/compute/gdn.cu:380-395` the SPLIT table. This matches `src/compute/CLAUDE.md:68`.
- **N-13. The PDL wait-contract holds for every *live* registration.** Cross-checked all live registrations against the presence of `pdl_wait()`: `elementwise_add_fp16_kernel` (`executor_elementwise.cu:48`), `layernorm_pdl_register` -> `layernorm.cu:84,167,267`, `rope_pdl_register` -> `rope.cu:224`, `nvfp4_gemv_pdl_register` -> `nvfp4_gemv_dense.cu` / `_fused.cu` (14 kernels, all waiting), and the self-registering launch sites in `embedding.cu:65`, `ssm.cu:169`, `gdn.cu:79`, `layernorm_rowblock.cu:40,121`, `sampling.cu:171,180,309`, `sampling_penalties.cu:74`, `sampling_penalties_history.cu:94`, `gdn_gated_norm.cu:35`, `executor_kv_write_kernels.cu:269`, `nvfp4_gemm_smallm_v2.cu:154,352`, `gemm_gemv_dtype.cu:145`. The only registrations of non-waiting kernels are inside the two dead functions of A2-1.
- **N-14. `apply_pdl_edges` checks the consumer, not the producer, and clears the benign per-edge error.** `src/runtime/cuda_graph.cu:113-155` - the check moved from source to destination on 2026-08-31 with the reason stated in place, and the `cudaGraphKernelNodeGetParams` "invalid device function" is swallowed per edge so it cannot surface two frames later.
- **N-15. Green contexts: confirmed unavailable, not re-flagged.** `src/runtime/green_ctx.cu:73-90` attempts `cudaDeviceGetDevResource` / SM split and `goto fallback`s to priority streams; the sync-domain split (`kPrefillSyncDomain` / `kDecodeSyncDomain`, `:13-14`) is what actually ships. Matches `docs/LIMITATIONS.md:79-82`.
- **N-16. The MoE grouped small-M kernel really does use TMA + warp specialisation on sm_120a.** `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes` at `gemm_grouped_nvfp4_smallM.cu:71` (comment: "Emits UTMALDG on SM120"), `CUtensorMap` descriptors built through a runtime driver-entry lookup so no hard `libcuda.so.1` dependency (`:659-701`), SFA/SFB kept on `cp.async` because "their gmem stride is K/16 bytes - too small for TMA" (`:200-203`).
- **N-17. `SETTLED.md` C's `quant -> compute` edge is still exactly one and still justified.** `src/quant/nvfp4_gemm.cu:283` `gemm(B, A_fp16, C, 1.0f, beta, stream);` inside the M>16 fallback. Unchanged.
- **N-18. The bench kernels pinned in the baseline (`bench_separate`, `bench_tma_v_load`, `bench_fused`) are in `libimp.a` in CI.** `CMakeLists.txt:288` compiles `tests/bench/*.cu` into the compute sources under `IMP_BUILD_TESTS OR IMP_BUILD_BENCH`, and CI keeps `IMP_BUILD_TESTS=ON` for the `ctest -L unit` stage even with `IMP_BUILD_BENCH=OFF`. So the "IMPROVED = FAIL" arm does not misfire. (A build with both OFF would fail the gate; that build configuration is not gated anywhere, so it is a non-issue today.)

---

### Known-and-accepted (restated)

- No GPU CI lane (F-5, owner decision 2026-08-03). Every kernel-level test above, including all NVFP4 scale-layout tests, runs only under local `make verify-fast` / `make test-*`.
- Green contexts unavailable on sm_120 (`docs/LIMITATIONS.md:79-82`).
- FP8 prefill disabled on sm_120 (cuBLAS `NOT_SUPPORTED` at non-aligned M, `engine_init_resolver.cpp`); FP8 GDN-projection prefill re-measured 2026-09-01 and REFUTED (`docs/plans/2026-08-31-fp8-ssm-prefill.md`).
- Closed kernel classes not re-opened here: M=1 NVFP4 decode GEMV ceiling; MoE grouped GEMM at ~60 % of the weight floor; FA2 occupancy at hd=128 (16 warps/SM shipped, #1843); NVFP4 decode-attention traffic sharing; FP4-precision attention.
- `attention.fa2_hd256_bkv=32` is opt-in on a measured PPL trade (+0.53 %), not an oversight.
- `cublasLtMatmulGrouped` returns zero grouped algos with NVFP4 on sm_120 as of cuBLAS 13.4 (`docs/internals/SM120.md`, "Open kernel work").
- GGUF batched decode rows use the 4-bit NVFP4 overlay (#1897).
- `src/exec/executor_forward_moe_legacy.cu` is a reachable floor, not a twin (`SETTLED.md` C).
- `docs/roadmap.md` open items touched by this axis: none of A2-1..A2-9 appears in the Open table or the lever ledger.

---

### Open questions

1. `tests/bench/fmha_v_load_bench.{cu,h}` (cp.async vs TMA bulk for the FMHA V-tile, "Phase 1 gate for the Tier-2 lever LDGSTS -> TMA migration") has no recorded verdict anywhere in `docs/` or `SETTLED.md`. Was it ever run? If yes the number belongs in the lever ledger; if no, the three pinned bench kernels are carrying a gate entry for an unfinished experiment.
2. A2-3: does SPLIT=2 help, hurt or do nothing at n_seq=1 on the GDN single-sequence scan? Needs one GPU A/B on Qwen3.8-27B plus the state-diff numerics judge.
3. A2-1: is the MaxL1 carveout worth anything on the dp4a GEMV family? One two-image A/B on a Q8_0 GGUF at M=1 and at 32 streams settles it, and also says whether the loss at #1833 was inside the +1.7 % that commit reported.
4. A2-4: is FP8 KV at head_dim=256 a configuration the resolver actually grants? Needs an init trace on an hd=256 model with `kv_cache.dtype=fp8`.
5. A2-8: can `gemm_dispatch_uncached_fallback`'s `beta != 0` arm be reached at M=1 in a shipped configuration, or does the FP16 weight cache always cover it? Needs a VRAM-constrained run with the one-shot warning added.
6. A2-6: which exact template instantiation produced the ledger's "232 regs / 226 regs"? The ledger names `fa2<64,256,...,TWOSLOT>` without the F16ACC/PVF16 bits, so it cannot be matched against the baseline without re-running ncu.


## Axis B - Memory & KV (scout report)

Repo `<repo>`, branch `perf/engine-h-fanin-cut-and-attention-split-verdict`, HEAD `ef664dd8`, clean. Read-only, no build, no GPU job.

### Coverage

**Read in full**
- Priors: root `AUDIT.md` (header index of all 236 rows; Phase A CONFIRMED/REFUTED/GOTCHA/OPEN tables verbatim), `docs/internals/MEMORY.md` (0, A1.1-A1.7, A2, A3.1, A4, A5.1-A5.5, A7, Invariant compliance, Open questions, B0 Landed + Divergences + B1), `docs/audit/SETTLED.md` sections D, E, F, G (incl. F-12 and "The memory plan is a SHADOW"), `docs/audit/DEBT_LEDGER_2026_08_21.md:55-200`, `docs/LIMITATIONS.md:85-145`, root `CLAUDE.md`.
- Code: `src/memory/kv_cache.h`, `src/memory/kv_cache_manager.h`, `src/memory/vram_owned.h`, `src/core/config/kv_cache.h`, `src/exec/sparse_attn_geometry.h`, `src/core/cuda_static_reset.h`, `tools/alloc_allowlist.txt`, `tools/log_fatal_allowlist.txt`.

**Sampled (targeted ranges)**
`src/memory/kv_cache_manager.cpp` (reclaim/reuse/rollback/can_allocate/load_prefix_cache), `src/memory/recurrent_snapshot_store.h`, `src/runtime/engine_kv_cache_init.cpp`, `engine_init_resolver.cpp`, `engine_prefill.cpp`, `engine_prefill_ragged.cpp`, `engine_graph_decode.cpp`, `engine_scheduler.cpp`, `engine_spec_mtp.cpp`, `engine_sampling_stop.cpp`, `engine.cpp` (dtor), `src/compute/sampling_penalties.cu`, `sampling.cu`, `gemm.cu`, `gemm_moe_fused_tc.cu`, `ffn_sparsity_probe.cu`, `src/exec/executor_attention_prefill.cu`, `executor_attention.cu`, `executor_forward.cu`, `executor_ssm_gdn.cu`, `src/api/imp_api_suspend.cpp`, `tools/imp-server/handlers.cpp`, `handlers_misc.cpp`, `handlers_admin.cpp`, `metrics_memory.cpp`, `tests/test_prefix_cache_equiv.cpp`, `CMakeLists.txt` test lanes.

**Commands run** (output cited in findings): `tools/check_alloc_sites.py [--stats]`, `tools/check_alloc_pairs.py`, `tools/check_log_fatal.py`, `git log -S` on `kKVBlockSize` / the block-size rule, `gh pr list --search "block size in:title" --state merged`.

**Skipped**: `src/memory/{vmm_backend,block_pool,arena,plan,scratch_stack,layer_offload,weight_snapshot,library_reserve_cache}.cpp` internals, all CUDA kernel bodies, `src/quant/`, `src/model/weight_upload.cu`.

### Brief vs repo

| Axis question said | Repo | Evidence |
|---|---|---|
| "the default (find it: `kv_cache.block_size` ... in `src/runtime/config.h`, `src/core/config/*.h` ...)" | **There is no such config key.** `imp::cfg::KVCache` (`src/core/config/kv_cache.h:29-127`) has no block-size field; `src/runtime/config.cpp:174-183` registers ten `kv_cache.*` keys and none is a block size. The value lives on `EngineConfig::kv_block_size` (`src/runtime/engine.h:92`, default 0 = auto) and `ImpConfig` (the C API struct) has no field for it either. `imp-cli`/`imp-server` expose no flag. | grep for `kv_block_size` in `src/api/ include/ tools/imp-cli/` returns 0 hits |
| "`d_penalty_tokens_` grow path (`engine_sampling_stop.cpp`)" | Correct, `engine_sampling_stop.cpp:218-227`, but it grows through `VRAMAllocator`, not a raw `cudaMalloc`, and only when a request's `output_tokens` exceed the last high-water, so it is bounded by max output length per process, not per step. | file read |
| "the #1897 unplanned 19 MiB alloc under capture" | **Fixed.** `CHANGELOG.md:110-115`: the small-M GEMM scratch is "a planned T2 arena tenant taken before graph prewarm". | CHANGELOG |
| "B9 residual_meta_d_buf_ `cudaMallocAsync` every decode step" (DEBT_LEDGER) | **Closed.** The buffer is now allocated once at `src/runtime/engine_kv_cache_init.cpp:735` (`cudaMalloc`) and only *read* at `engine_scheduler.cpp:1081`; the free is in `~Engine` (`engine.cpp:148-150`). | grep `residual_meta_d_buf_` |
| "leak paths on error unwinding (a `throw` between a raw `cudaMalloc` and its owner assignment)" | Hunted in every allocating file that also throws; **none found** - see "Checked and NOT a finding" #2. | see below |
| "`IMP_LOG_FATAL`-then-return sites listed in `tools/log_fatal_allowlist.txt`" | The census is now 2 sites / 1 abort / 1 allowlisted, not 12. | `python3 tools/check_log_fatal.py` |
| Recurrent-state pinned host tier "since 2026-09-02" | Present and documented: `src/memory/recurrent_snapshot_store.h:38-63` (`server.recurrent_snapshot_host_mb`, `on_host` entries restored with `cudaMemcpyDefault`). Confirmed, not contradicted. | file read |

### Findings

### [B-1] `apply_logit_bias`'s arena buffers are never re-armed: the second engine in a process writes into freed VRAM
Axis: B   Sev: S0   Confidence: high
Evidence:
- `src/compute/sampling_penalties.cu:544-546` - `static int32_t* s_bias_tokens_buf`, `static float* s_bias_values_buf`, `static size_t s_bias_buf_cap` at file scope.
- `src/compute/sampling_penalties.cu:558-563` - `sampling_preallocate_logit_bias(int max_entries)` returns early when `cap <= s_bias_buf_cap`, i.e. **before** it nulls the pointers at `:565-567` and re-takes from the arena at `:569-570`.
- `src/runtime/engine_workspace_warmup.cpp:231` - the only caller, `sampling_preallocate_logit_bias(4096)`, a constant, once per `Engine::init`.
- `src/runtime/engine.cpp:84` - `~Engine` calls `engine_arena_close()`, which releases the arena region the buffers point into. `engine.cpp:109` calls `sampling_cleanup()`, which is `sampling_cleanup_cub() + sampling_cleanup_dry()` (`sampling_penalties.cu:669-672`) - **the bias pair has no cleanup function at all**; the only `s_bias_tokens_buf = nullptr` in the tree is at `:565`, behind the guard that just returned.
- `src/compute/sampling_penalties.cu:607` - the use site tests `s_bias_tokens_buf == nullptr || m > s_bias_buf_cap`; after the second init both are false, so `:621-627` `cudaMemcpyAsync` into the stale pointer and launch `apply_logit_bias_kernel` over it.
- Reachable by default: `tools/imp-server/handlers.cpp:655` `load_model_into_state()` does `imp_context_free` (`:664`) then `imp_context_create` (`:717`) in the same process, gated on `server.model_swap` which is `true` (`src/runtime/config.h:361`). Any subsequent request carrying `logit_bias` (`tools/imp-server/handlers_chat_params.cpp:279`) hits it.

Current: engine #1 takes 32 KiB from arena #1 and records `s_bias_buf_cap = 4096`; `~Engine` frees arena #1; engine #2's identical `sampling_preallocate_logit_bias(4096)` short-circuits; the first `logit_bias` request writes 2 host buffers into arena #1's freed address range and reads them from a kernel.
Expectation: `~Engine` already re-arms every *registered* arena tenant (`engine.cpp:93` `reset_static_cuda_state()`, whose in-place comment records this exact bug class being found and fixed for `gemm.cu`). This file is simply not in the registry.
Delta: a use-after-free on device memory on the default model-swap path, silent (no `IMP_CUDA_CHECK` can see a write into a re-mapped region) - corrupts whichever tenant of arena #2 landed on those bytes.
Cost: 1 file, ~8 LOC - add `sampling_cleanup_bias()` mirroring `sampling_cleanup_dry()` (`:660-667`) and call it from `sampling_cleanup()`, or register a `IMP_REGISTER_CUDA_STATIC_RESET` hook. Risk near zero; the DRY half already does exactly this.
Falsifier: a cleanup or reset path elsewhere that nulls `s_bias_buf_cap`. **Checked y** - `grep -n 's_bias_buf_cap' src/compute/sampling_penalties.cu` returns lines 546, 562, 567, 580, 607 only; nothing outside the file. Second falsifier: arena #2 landing at the same base with the same offsets would mask it, but that is luck, not a contract (arena capacity is model-dependent, `engine.cpp:748`).

### [B-2] Three more lazy device statics outside the reset registry, and nothing detects an unregistered one
Axis: B   Sev: S1   Confidence: high
Evidence:
- `src/compute/sampling.cu:202-208` - `static int32_t* d_result`, taken from `engine_arena()` (or `cudaMalloc` fallback) under `if (!d_result)`. Never nulled: `grep -n 'd_result' src/compute/sampling.cu` shows no assignment to null outside the failure branch. `sampling_cleanup()` does not touch it.
- `src/compute/gemm_moe_fused_tc.cu:245-272` - `static bool configured` / `static int* d_tile_counter`, arena-taken under `if (!configured)`, `cudaMemsetAsync` at `:274` on every Q6_K fused-MoE prefill call. No re-arm.
- `src/compute/ffn_sparsity_probe.cu:33-59` - file-scope `ProbeState g_state` with `initialized` + `d_counters` from the arena; `ensure_init_locked()` returns at `:36` on the second engine. Diagnostic-gated (`process_diag_ffn_sparsity_probe()`), so lower blast radius.
- `src/runtime/engine_spec_mtp.cpp:101-113` - `static void* s_norm_scratch`, raw `cudaMalloc`, capacity-checked so no OOB, but never freed and never re-armed; dangles after `imp_gpu_release(1)` (see B-3 for the path).
- The registry has 15 registering TUs (`grep -rln IMP_REGISTER_CUDA_STATIC_RESET src/`); `src/compute/{sampling_penalties,sampling,gemm_moe_fused_tc,ffn_sparsity_probe}.cu` and `src/runtime/{engine_scheduler,engine_spec_mtp}.cpp` are not among them.
- The only test is `tests/test_cuda_static_reset.cpp:23`, `EXPECT_GT(cuda_static_reset_hook_count(), 0)` - it proves the registry is non-empty, not that it is complete. There is no `tools/check_*.py` for it (`ls tools/check_*.py` = 10 gates, none about statics).

Current: `src/core/cuda_static_reset.h:18-24` states the failure mode in its own words - "a twelfth lazy static added without an entry dangled behind an armed guard - exactly the bug this file exists to prevent, with nothing to catch it" - and claims auto-registration removed it. Auto-registration removed the *call-site* half (a hand-maintained list in the aggregator). The *registration* half is still a convention with no gate, and six statics demonstrate it.
Expectation: this repo gates comparable conventions statically - `tools/check_alloc_sites.py` (a two-way ratchet, SETTLED S-27), `check_alloc_pairs.py`, `check_log_fatal.py`. A grep-shaped gate for "TU holds a device pointer static behind a null/capacity guard AND does not register a reset hook" is the same shape as those.
Delta: the mechanism is complete for the 15 TUs someone remembered, and silently absent for the rest.
Cost: fix the four device sites: 4 files, ~30 LOC. Gate: 1 new `tools/check_static_reset.py` + a `scripts/ci_static_gates.sh` line, ~80 LOC, allowlist for host-only statics (`model/weight_upload.cu` `g_warm`/`g_stager`, `memory/weight_snapshot.cpp` `g_armed_snapshot` are pinned-host by design, `weight_snapshot.h:16`). Risk: false positives on host statics, hence the allowlist.
Falsifier: one of the four is re-armed on some path I missed. **Checked y** for `sampling.cu` and `gemm_moe_fused_tc.cu` (no null assignment anywhere in the tree); `ffn_sparsity_probe.cu` and `engine_spec_mtp.cpp` read in full.

### [B-3] `s_h_normed` is sized from the first model's `d_model` and never resized, freed or re-armed
Axis: B   Sev: S1   Confidence: high
Evidence:
- `src/runtime/engine_scheduler.cpp:1635-1647`:
  ```
  static void* s_h_normed = nullptr;
  if (s_pre_norm_h) {
      if (s_h_normed == nullptr) {
          cudaMalloc(&s_h_normed, hidden_dim * sizeof(__half));
      }
      ... imp::rmsnorm(in_view, model_->output_norm(), out_view, ...)
  ```
  `hidden_dim = model_->config_.d_model` (`:1626`). No capacity variable, so the size of the first model that reached this line is permanent for the process.
- Gate `s_pre_norm_h = runtime_config_.diagnostics.mtp_prenorm_h`, whose code default is **`true`** (`src/core/config/diagnostics.h:153`).
- Model swap in one process is the default (`config.h:361`, `handlers.cpp:664/717`). Swap A -> B with `d_model(B) > d_model(A)` makes `rmsnorm` write `d_model(B)` halves into a buffer sized for `d_model(A)`: out-of-bounds device write.
- Also: never freed (leak across every teardown), and dangling after `imp_gpu_release(1)` (`src/api/imp_api_suspend.cpp:74-80`), which `/admin/suspend` calls by default (`handlers_admin.cpp:100`, `suspend.device_reset = true` at `config.h:416`). The armed `!= nullptr` guard then hands the next MTP step a pointer into a destroyed context - the exact scenario `cuda_static_reset.h:5-11` describes.
- Sibling drift: `imp.conf.example:849` ships `mtp_prenorm_h = false` while the code default is `true`, so the example config and the built-in default disagree about whether this path runs.

Current: a per-process device scratch on the MTP decode path with no size check, no owner and no reset hook.
Expectation: the same file's sibling at `engine_spec_mtp.cpp:103-110` gets the capacity check right (`if (need > s_norm_cap) { cudaFree; cudaMalloc; }`). The tier for a fixed engine-lifetime scratch is T2 (`MEMORY.md` A2), which is a `engine_arena().take_bytes()` call.
Delta: an unbounded-by-construction device write plus a suspend/resume dangling pointer, on a default-on diagnostic knob.
Cost: 1 file, ~10 LOC - move to the T2 arena (it is 2 bytes x `d_model`, well under any arena slack) or add `s_h_normed_cap`. Risk: none; the buffer has one reader in the same block. Also reconcile `imp.conf.example:849` with `diagnostics.h:153` (1 line, pick one).
Falsifier: two models with the same `d_model` would never trip the OOB half, and `mtp_prenorm_h=false` disables both halves. Neither makes the dangling-after-`device_reset` half wrong. **Checked y** - no other writer of `s_h_normed`.

### [B-4] I2 reads "met" in the design doc while the default-on ragged prefill path allocates 6 times per prefill step, and no shipping build can see it
Axis: B   Sev: S2   Confidence: high
Evidence:
- `src/runtime/engine_prefill_ragged.cpp:247-254` - six unconditional `cudaMallocAsync` (`d_tok`, `d_pos`, `d_bt`, `d_ctx`, `d_soff`, `d_slots`) per ragged prefill forward, freed by the `RaggedMeta` destructor at `:232-238`. There is no pool branch, unlike the serial path at `engine_prefill.cpp:315-322`, which uses `prefill_pool_` and falls back only when the pool is absent.
- Default on: `runtime.prefill_batch = true` (`src/runtime/config.h:230`); driven from `engine_prefill.cpp:141/189`.
- `docs/internals/MEMORY.md` "Invariant compliance" row I2 reads **"✓ - `0 cudaMalloc, 0 cudaMallocAsync, 0 pinned-host allocations while serving`, 15 requests, dense"**. That measurement is dated 2026-07-29/30 (`AUDIT.md` B35); ragged prefill shipped 2026-08-26 (`config.h:222` cites the measurement date; `#1780`).
- Why nothing catches it: `note_serving_allocation()` (`src/memory/backend.cpp:100`) is fed only by `Backend::acquire()` and by the `--wrap` interposer, and the interposer needs `IMP_ALLOC_INTERPOSE`, `OFF` in `CMakeLists.txt:67` and in no make target and no CI job (`DEBT_LEDGER_2026_08_21.md:61-77`, still OPEN). So `~Engine`'s I2 warning (`engine.cpp:75-80`) reads zero on every shipping build regardless.
- Same shape, also unpooled: `engine_graph_decode.cpp:171/179` (`cudaMallocAsync` block tables per `try_graph_loop_decode` call).

Current: the invariant table asserts a property the serving path stopped having, and the only instrument that could contradict it is compiled out.
Expectation: `MEMORY.md` A2 names this exact traffic as T4 (`ScratchStack`) and A7 step 5 as "The step that satisfies I2"; `engine_prefill.cpp` already shows the pooled shape for the identical buffers.
Delta: I2 is a claim, not a property; the drift happened within four weeks of the measurement and nothing announced it.
Cost: either (a) point ragged prefill at `prefill_pool_`/`ScratchStack` - 1 file, ~40 LOC, sizes already bounded by `max_seq_len x max_batch`; or (b) at minimum change the I2 row to `~` with the date and the two open sites. Risk of (a): the ragged buffers are variable-length per wave, so the pool must be sized for the worst case, which is what `workspace_sizes.h` already does for its neighbours.
Falsifier: the six calls could be outside the `Serving` phase, or a `cudaMallocAsync` from a pinned pool could be free. **Checked y** for the first - `step_prefill_ragged_` is called from the scheduler loop, after `set_alloc_phase(AllocPhase::Serving)` at `engine.cpp:818`. Not checked: whether the six cost anything measurable at 32 streams (needs a GPU run; filed as an open question).

### [B-5] The KV block size is an inherited constant with no operator surface, no A/B and no instrument that can vary it
Axis: B   Sev: S2   Confidence: high
Evidence:
- `src/memory/kv_cache.h:16` - `static constexpr int kKVBlockSize = 16;  // default tokens per block`. `git log --all -S'kKVBlockSize = ' -- src/memory/kv_cache.h` returns exactly one commit, `ff4d81d8 feat: full Blackwell (sm_120) optimized LLM inference engine` - the initial import. It has never been changed.
- The one refinement, `src/runtime/engine_init_resolver.cpp:827`, `config_.kv_block_size = (mcfg.n_kv_heads <= 4 && mcfg.n_kv_heads > 0) ? 32 : kKVBlockSize;`, traces to `505e9327` (2026-03-23) where its comment was the whole justification: *"Larger blocks (32) improve coalescing for GQA models with few KV heads"*. No measurement accompanies it, then or since.
- No A/B exists: `grep -n 'block size\|block_size' CHANGELOG.md docs/audit/PERF_LOG.md docs/roadmap.md` returns three PERF_LOG mentions that use `block_size` as a *denominator* for something else (`:330`, `:398`) and one kernel constraint (`:531`), plus the `#1819` CHANGELOG entry (`:429-444`) which is a sparse-budget **correctness** fix, not a block-size verdict. `gh pr list --search "block size in:title" --state merged` returns one unrelated PR (#134, `d_pf_block_tables_` sizing).
- No operator surface: `src/core/config/kv_cache.h` has ten keys and no block size; `src/runtime/config.cpp:174-183` registers none; `ImpConfig` and the CLI have no field. `EngineConfig::kv_block_size` (`src/runtime/engine.h:92`) is reachable only from in-tree C++. The comment at `engine_init_resolver.cpp:824` ("explicit imp.conf / --set value wins") describes a path that does not exist.
- No instrument: `tools/imp-bench/bench_attention.cu:283` hardcodes `const int block_size = 16;  // kKVBlockSize`, so even the attention microbenchmark cannot sweep it.
- What is coupled (all read): sparse decode geometry (`src/exec/sparse_attn_geometry.h:29-48`, every token->block conversion, and `#1819` is the record of what one wrong assumption there cost); the FP8 tile decode kernel's dispatch predicate (`src/compute/attention_paged_fp8_tile.cu:514`, `block_size >= 16 && block_size % 16 == 0`); the prefix cache's reuse granularity (`kv_cache_manager.cpp:516`, `is_full_block = (block_tokens == cache_->block_size())` - only whole blocks are cacheable, so a larger block coarsens every prefix hit); block-table width (`engine_kv_cache_init.cpp:107`, `blocks_per_seq = ceil(max_seq_len / kv_bs)`, which sizes `GPUBatchPool` and every H2D block-table copy); the SWA slack (`engine_kv_cache_init.cpp:162`, `max(2*kv_bs, spec_depth + kv_bs)`); the sparse row-table cap (`sparse_attn_geometry.h:33`, `+16`).
- Internal fragmentation is `block_size/2` tokens per sequence on average: at bs=16 and 32 concurrent sequences that is ~256 wasted token-slots, at bs=32 ~512. Concrete against a real pool: the dense server default plans 4096 blocks = 65 536 token-slots (`AUDIT.md` B1), so the loss is <1 %.

Current: a geometry constant that six subsystems read, chosen once and never revisited, that no operator can change and no bench can vary.
Expectation: vLLM v1 ships 16 as the default and exposes `--block-size {1,8,16,32,64,128}`; SGLang exposes `page_size`; TensorRT-LLM exposes `tokens_per_block`. All three treat it as a tunable because the trade (coalescing and table-walk length vs fragmentation and prefix-cache granularity) moves with head count and context length - and this engine's models span `n_kv_heads` 2..8 and contexts 4k..128k.
Delta: 16 is not wrong, but it is unevidenced, and the repo cannot cheaply produce the evidence because the parameter is unreachable from outside.
Cost: expose `kv_cache.block_size` - 1 key in `src/core/config/kv_cache.h`, 1 registration in `config.cpp`, 1 read in `init_resolve_kv_block_size_()`, plus a validity check (`bs % 16 == 0` for the FP8 tile path, `bs` a multiple of `kNVFP4Group`), ~25 LOC. Then a bench sweep. Risk: an operator setting a value the tile kernel declines silently falls back - guard it at resolve time, not at dispatch.
Falsifier: a measurement in an artifact I did not read. **Checked y** for `docs/PERF.md`, `docs/BENCHMARKS.md`, `docs/audit/PERF_LOG.md`, `docs/roadmap.md`, `CHANGELOG.md`, merged PR titles. Not checked: `tools/roofline/history/` entry bodies (grepped for `block_size`, no hit).

### [B-6] Nothing verifies the KV pool is resident rather than spilled, and both mitigations default to off
Axis: B   Sev: S2   Confidence: high
Evidence:
- Root `CLAUDE.md`: "A successful `cudaMalloc` proves nothing about free VRAM ... Measure *bandwidth* to tell resident from spilled: ~1530 vs ~237 GB/s. That 6.5x cliff is the mechanism behind #1103 (55 vs 391 tok/s), so '0 MiB free' is a correctness problem". The same fact is repeated in `AUDIT.md` G18 and `src/core/config/kv_cache.h:80-86`.
- No such measurement exists in the engine: `grep -rn 'GB/s\|bandwidth' src/runtime/ src/memory/ src/api/` returns two unrelated comments in `engine.h`. `grep -rln 'GB/s' src/ tools/ scripts/` finds it only in kernel comments and `tools/roofline/`. There is no post-init probe, no `/health` field, no metric.
- The two things that would prevent an overshoot are both off by default: `kv_cache.growable = false` (`src/core/config/kv_cache.h:75`) and `kv_cache.growable_initial_pct = 100` (`:87`) - and the second is inert without the first (`engine_kv_cache_init.cpp:568`, `kv_ceiling_blocks = growable ? kv_blocks_planned : 0`).
- The comment at `engine_kv_cache_init.cpp:572-575` states the intent exactly ("starting under it is the only way to be sure the pool is resident rather than spilled") for a path the default configuration never enters.
- `docs/LIMITATIONS.md:100-113` documents the consequence from the other end (the cold-start spill lottery), and `#1899` removed one *cause*; it did not add a *detector*.

Current: the engine can allocate a pool that WDDM has spilled into host memory, serve at ~1/6 the bandwidth, and report a successful load with no signal anywhere.
Expectation: the discriminator is one `cudaMemcpyDeviceToDevice` over a slice of the pool, timed with events, at the end of `init_kv_cache` - a few hundred microseconds, once. Compare against a threshold and log/refuse. `AUDIT.md` G13 already establishes `cudaMemGetInfo` is noise-free within a process, so the arithmetic side is trustworthy; it is the physical side that is unobserved.
Delta: the one platform fact the repo repeats most often has no runtime check.
Cost: ~60 LOC in `engine_kv_cache_init.cpp` plus one `/health` field and one metric. Risk: a threshold set from one box; make it a WARN plus a gauge, not a refusal, until it has been seen on more than one driver.
Falsifier: a residency check under another name (`resident`, `spill`, `wddm`). **Checked y** - `grep -rn 'spill' src/memory src/runtime` returns eight comment-only hits, no code.

### [B-7] `sparse_topk_tokens` and friends are the only block-size-derived quantities with a pure function and a test; the rest are open-coded
Axis: B   Sev: S3   Confidence: med
Evidence: `src/exec/sparse_attn_geometry.h:1-11` exists precisely because "every conversion used the compile-time `kKVBlockSize` (16) while a model with `n_kv_heads <= 4` runs a 32-token block" (#1819). The same class of open-coded conversion survives elsewhere: `src/exec/workspace_sizes.h:110` still carries `constexpr int kExecKVBlockSize = 16;  // kKVBlockSize, memory/kv_cache.h` - a second copy of the constant in a second header, used to size executor workspaces, that does not consult `config_.kv_block_size` at all. `src/runtime/engine.h:134` computes `streaming_kv_threshold` auto as `n_sinks + window + 2*kKVBlockSize`, again the compile-time 16 rather than the resolved value; on an `n_kv_heads<=4` model the real slack is half what the formula intends.
Current: one conversion family was lifted into a tested pure function; two more still read the constant where they should read the resolved value.
Expectation: the fix that shipped for the sparse family (a host-only pure function, CPU-testable at both block sizes) is the template.
Delta: the same defect shape, in two places the #1819 sweep did not reach.
Cost: 2 files, ~15 LOC - pass the resolved block size in, as `set_kv_block_size` already does for the executor (`src/exec/executor.h:516-517`, whose comment says "use this and not `kKVBlockSize` - the two differ on `n_kv_heads <= 4` models"). Risk: `kExecKVBlockSize` sizes a workspace, so changing it to 32 makes the workspace larger, not wrong.
Falsifier: `kExecKVBlockSize` might deliberately be an upper bound for sizing rather than a geometry constant. **Checked n** - `workspace_sizes.h:110`'s comment does not say, and I did not read every consumer.

### Checked and NOT a finding

1. **The "second engine reuses freed arena memory" bug is fixed for registered TUs.** `src/runtime/engine.cpp:85-93`: `reset_static_cuda_state()` runs in `~Engine` *after* `engine_arena_close()`, with a comment recording the original incident ("cuBLASLt matmul'd into it: status 14, fallback, illegal memory access"). My initial hypothesis that only `imp_gpu_release` calls it is refuted by that line. B-1/B-2 are what the registry does not cover, not the registry being unwired.
2. **No leak on error unwinding between a raw allocation and its owner.** For every allocating file that also contains `throw`: `executor_attention_prefill.cu` is the only interleaving candidate (`cudaMallocAsync` of `k_full`/`v_full` at `:191-192`, freed at `:357-358`) and `awk 'NR>=193 && NR<=356 && (/throw/||/IMP_CHECK/||/return/)'` returns exactly one line, the `:306` throw, which sits inside `if (cap_replay)` where `k_full = chunk_capture_k_` (a persistent buffer, `:165-167`) and the per-call allocation was never taken. `executor_attention.cu:346`, `executor_forward.cu:193`, `executor_ssm_gdn.cu:46/660/666/748` all precede their file's allocations.
3. **`tools/check_alloc_pairs.py`: `578 files, 204 pointer(s) with both halves in one file, 45 member(s) tree-wide, 0 mismatch(es), 0 allowlisted`.** The B10 class (allocate with `cudaMalloc`, free with `cudaFreeAsync`) is gated.
4. **`tools/check_log_fatal.py`: `2 IMP_LOG_FATAL site(s), 1 abort, 0 throw, 1 continue (1 allowlisted)`.** The 12-site census in the debt ledger is closed; the surviving allowlist entry (`expert_cache.cu`) carries its reason.
5. **A block in use cannot be evicted.** `cached_blocks_lru_` holds only blocks whose reference the cache itself owns (`CachedEntry{lru_it, BlockRef ref}`, `kv_cache_manager.h:483-487`); a sequence takes it by *moving* the ref out and erasing the entry (`kv_cache_manager.cpp:541-548`) or by `share_block` when another sequence already holds it (`:552`). `reclaim_cached_block()` (`:667-699`) pops only from that list and drops the cache's own ref. Pinned blocks are rotated past, bounded by list size (`:673-681`).
6. **The stale-hash class (#1044) has a load-bearing guard, a defence-in-depth guard, and a mutation-validated test.** Lookup-site guard: `kv_cache_manager.cpp:527-533` rejects a hash whose block is ref-0-and-not-cached and logs it. Eager cleanup: `drop_stale_hash_if_last()` (`:17-26`) from `rollback_partial_allocation` and `:1173`. Test: `tests/test_prefix_cache_equiv.cpp:407 RollbackOfPartialAllocationDropsItsHashes`, whose own comment records that stubbing the eager half out does *not* change the outcome, i.e. it names which of the two is load-bearing.
7. **Prefix-cache hit rate is computable from `/metrics`.** Numerator `imp_tokens_cached_total` (`handlers_misc.cpp:194`, fed at `handlers_chat_core.cpp:1181` and `stream_driver.cpp:667`), denominator `imp_tokens_prompt_total` (`:182`). Plus `imp_kv_blocks_cached` / `_reclaimable` / `_pinned` (`metrics_memory.cpp:89-104`) and `imp_prefix_cache_evictions_total` (`handlers_misc.cpp:233`).
8. **The prefix cache is a chained-hash + LRU, matching vLLM v1's automatic prefix caching, not a radix trie.** `block_hash_to_id_` / `block_id_to_hash_` (`unordered_map`) + `cached_blocks_lru_` (`std::list<int>`) + `cached_blocks_map_`, `kv_cache_manager.h:465-491`; chain built in `longest_cached_prefix_blocks` (`kv_cache_manager.cpp:681-702`), parent hash seeded by `content_salt`. SGLang's RadixAttention trie is a different design, not a missing one.
9. **`cache_control` and vision salting are both wired.** Anthropic `cache_control` maps to prompt-KV pinning (`tools/imp-server/anthropic.cpp:515-522` -> `cache_prompt` -> `pin_prefix`); the image salt is `req->vision_content_hash` (`handlers_chat_core.cpp:318-320,736` -> `scheduler.cpp:183-190`), with `cacheable = !has_image || vision_content_hash != 0` as the guard, and a dedicated test (`test_prefix_cache_equiv.cpp` `ContentSaltSeparatesIdenticalTokenPrefixes`) whose comment records that nothing exercised it before.
10. **All six KV dtypes have both a decode and a write kernel.** Decode dispatch: `src/exec/executor_attention_decode.cu:206 INT4`, `:216 NVFP4`, `:309 MXFP4_KV`, `:321 INT8`, `:331 FP8_E4M3`, F16 default. Write: `src/exec/executor_kv_write.cu:92 NVFP4`, `:109 MXFP4`, `:125 INT4`, `:141 INT8`, plus FP8/F16. Config surface: `kv_cache.dtype` accepts `auto|fp16|fp8|int8|int4|nvfp4|mxfp4` (`engine_init_resolver.cpp:200-243`).
11. **`load_prefix_cache` validates before uploading.** Magic, version, `n_layers`/head/dim geometry, model fingerprint and `scale_block_bytes`, each a `return -1` (`kv_cache_manager.cpp`, load function). The path is operator-supplied (`config_.prefix_cache_path`), never client-supplied.
12. **`tools/imp-server/` has zero entries in `tools/alloc_allowlist.txt`** - the HTTP layer allocates no device memory. The allowlist is 70 files: `src/exec` 27, `src/compute` 21, `src/runtime` 13, `src/quant` 4, `src/model` 2, `src/vision` 1, `src/lora` 1, `src/core` 1.
13. **Admission control does not read free VRAM.** `KVCacheManager::can_allocate` (`kv_cache_manager.cpp:438-455`) is `num_free_blocks() + reclaimable_cached_count_ - outstanding_reserved_blocks() >= num_blocks`, with an in-place comment explaining why the second source (live LRU sequences) was removed. No `cudaMemGetInfo` on the admission path. Where it can be wrong is upstream, in how many blocks the pool got - which is the LIMITATIONS entries restated below.
14. **`VramOwned` exists** (`src/memory/vram_owned.h`, so `AUDIT.md` R7 is superseded) but has two members tree-wide (`engine.h:696 d_banned_tokens_`, `:1155 d_spec_logits_`) against 215 acquisitions on the allowlist. Consistent with SETTLED F-12's re-scoping ("the residual is `VRAMAllocator`'s acquisition job for tiers sized after the upload"), so not re-opened as a migration backlog.
15. **The serial prefill metadata fallback is effectively dead.** `engine_prefill.cpp:315` takes the pooled branch whenever `prefill_pool_ && chunk_len <= config_.max_seq_len`; the five `cudaMallocAsync` at `:325-333` need the pool to be absent.
16. **The DRY penalty buffers, unlike the logit-bias pair in the same file, ARE re-armed.** `sampling_cleanup_dry()` (`sampling_penalties.cu:661-667`) is reached from `sampling_cleanup()` at `engine.cpp:109`. That asymmetry inside one file is what makes B-1 an oversight rather than a design choice.
17. **`I1: 70 files / 481 sites outside src/memory/ (allowlist: 70 / 464)` - `OK, no new direct allocation sites`.** The ratchet holds; 215 acquisitions / 286 releases, 10 release-only files.

### Known-and-accepted (restated)

- **No KV tier below VRAM: DO NOT BUILD** (`docs/roadmap.md:63`, decided 2026-08-01; the spill cliff and the ~165 us blocking transfer are the reasons, `AUDIT.md` B84/B36).
- **Recurrent-state snapshots do have a pinned host tier** (`server.recurrent_snapshot_host_mb`, `src/memory/recurrent_snapshot_store.h:38-63`) - the exception to the previous line, and deliberate.
- **VRAM planner's weight-cache reserve is an estimate with a floor; no retry when wrong** (`docs/LIMITATIONS.md:93-99`, #1631: projection alone plans 9977 blocks and OOMs, the shipped arm plans 7079).
- **The measured library reserve survives only if `vram.library_reserve_cache` outlives the process** (`LIMITATIONS.md:100-113`); a `docker run --rm` server charges the 3900 MiB constant, measured wrong in both directions (763 / 1366 / 3260 MiB across three checkpoints).
- **The memory plan is a shadow: `PlanInput::features` is mostly unfilled** and that is not a bug until A7 step 6 (`SETTLED.md` G; `plan_shadow.cpp:29-31` fills three fields; the authoritative number is the arena capacity at `engine.cpp:748`).
- **INT4 KV produces empty output on gpt-oss** (`LIMITATIONS.md:143`); **quantised KV is a default only for QWEN35** (`:709`).
- **No GPU CI lane** (owner decision 2026-08-03): the whole `gpu_kv` lane, which contains every prefix-cache and KV-cache test (`CMakeLists.txt:968-981,1122`), never runs in CI. `make verify-fast` locally is the only execution.
- **DEBT_LEDGER item 3 remains OPEN**: `IMP_ALLOC_INTERPOSE` is `OFF` and in no make target and no CI job, so `steady_state_allocations()` reads zero in every shipping build.
- **F-12 is re-scoped, not open**: the 48 remaining `VRAMAllocator` references are the acquisition path for tiers sized after the weight upload, not a migration backlog.
- **Prefix caching yields to SWA sizing when `kv_cache.swa_snapshot_mb == 0`** (`src/core/config/kv_cache.h:96-110`).

### Open questions

- Does arena #2 land at arena #1's base address on this box? If it usually does, B-1 and B-2 have been corrupting a *different tenant* rather than unmapped memory, which changes the symptom but not the fix. Needs one GPU run with two engines and a printed base.
- What do block sizes 32 and 64 cost/buy at 4k / 32k / 128k on `n_kv_heads` 2, 4 and 8? Cannot be answered today: the parameter has no operator surface (B-5) and `bench_attention.cu` hardcodes 16.
- Are the six ragged-prefill `cudaMallocAsync` per step measurable at 32 streams? Needs an `IMP_ALLOC_INTERPOSE=ON` build plus a burst, i.e. exactly the make target DEBT_LEDGER item 3 asks for.
- Owner question: is `diagnostics.mtp_prenorm_h` meant to default `true` (`diagnostics.h:153`) or `false` (`imp.conf.example:849`)? The answer changes B-3's reachability from "default path" to "opt-in".
- Would a residency (bandwidth) probe threshold generalise off this box? `AUDIT.md` G3 and G13 give the in-process measurement properties, but the 1530/237 GB/s pair is one driver on one card.


## Axis C - Scheduler & batching, speculative decoding, CUDA graph coverage

Repo `<repo>`, branch `perf/engine-h-fanin-cut-and-attention-split-verdict`, HEAD ef664dd8, clean. READ-ONLY, no build, no GPU job.

### Coverage

Read in full:
- `src/runtime/scheduler.h`, `src/runtime/scheduler.cpp` (309 LOC)
- `src/runtime/graph_eligibility.h`, `src/runtime/graph_eligibility.cpp`
- `tools/imp-server/batching_engine.h` (181 LOC)
- `src/runtime/CLAUDE.md`, `tools/imp-server/CLAUDE.md`
- `docs/internals/ARCHITECTURE.md` phases 3+4 and "Subsystems"/"Known limitations" (:79-145)
- `docs/audit/SETTLED.md` A (S-1..S-11), D (S-20..S-28), E, F
- `docs/roadmap.md` Open rows 1-12, Closed "per-request priority" + "speculation tree"
- `tests/test_scheduler.cpp` TEST list + :600-948

Read in targeted ranges (line spans opened, not whole file):
- `src/runtime/engine_scheduler.cpp` :84-250, :359-480, :600-944, :1400-1560 (2002 LOC file)
- `src/runtime/engine_prefill.cpp` :1-200, :270-350, :510-545, :690-750
- `src/runtime/engine_spec_ngram.cpp` :1-56 (header contract), :291 signature, :551-740, :1090-1130, plus a full grep of KV/rollback sites
- `src/runtime/engine_spec_capture.cpp` :1-100, :150-270
- `src/runtime/engine_graph_decode.cpp` symbol outline + :236-320
- `src/runtime/engine_decode_pipeline.cpp` :88-155
- `src/runtime/cuda_graph.cu` :410-460, :520-535, :1120-1150, :1319
- `src/runtime/engine_weight_upload.cpp` :285-360; `src/runtime/engine_init_resolver.cpp` :60-90, :525-540
- `src/runtime/engine.cpp` :790-830, :845-860; `src/runtime/config.h` :48-265
- `src/memory/kv_cache_manager.cpp` :1156-1200
- `tools/imp-server/batching_engine.cpp` :225-270, :396-462; `tools/imp-server/handlers.h` :120-175; `tools/imp-server/handlers_misc.cpp` :275-335; `tools/imp-server/tracing.cpp` :100-155; `tools/imp-server/responses.cpp` :60-195
- `src/compute/attention_paged_common.cuh` :60-80
- `docs/API.md` :50-110, :315-325

Skipped: `src/compute/` kernels, `src/exec/`, `engine_spec_mtp.cpp` bodies, `suffix_draft`/`ngram_draft`/`token_recycle_draft`, `engine_qwen3vl.cpp`, the constrained-pipeline half of `engine_graph_decode.cpp` (:446-900), `src/vision/`, `src/lora/`.

Commands run: `tools/check_function_size.py` (host python, as the hooks do), `rg`/`grep`, `sed -n`.

### Brief vs repo

| Axis-question premise | Repo | Evidence |
|---|---|---|
| "per-request speculative toggle ... the brief calls it a known gap" | A per-request toggle EXISTS: `"speculative": true/false`, tri-state `Request::spec_override` (-1/0/1), switches off all three drafters | `src/runtime/request.h:96-101`, `tools/imp-server/handlers_chat_params.cpp:315-316`, `handlers_chat.cpp:863-864`, `handlers_chat_core.cpp:745`, `anthropic.cpp:341-342`, `docs/API.md:53`. The real remainder is roadmap Open 4 (drafter choice and depth are global) plus the `/v1/responses` hole (C-9) |
| "`graph_max_*` keys" | No such config key exists anywhere | `grep -rn "graph_max" src/ tools/ docs/` -> 0 hits. The decode graph pool is `Engine::kMaxGraphPoolSize = 64` keyed by `n_sequences-1` (`src/runtime/engine.h:471-472`), with pow2 buckets on `max_blocks_per_seq` and `max_context_len` (`engine_scheduler.cpp:1445-1481`) |
| memory note "cudaGraphInstantiate 10-44 ms/req offen" | CLOSED. The parked exec is reused via `cudaGraphExecUpdate` | `docs/roadmap.md:216` (#1895: 31 of 34 setups updated in place in 0.1 ms, wall median 545 -> 532 ms; the 44 ms was CUPTI-inflated); code at `src/runtime/cuda_graph.cu:1130-1141` + `:1319`, `cuda_graph.h:296` |
| memory note "KV-pressure: Prefix-Cache blocks counted as used" | FIXED 2026-09-03 (#1879); the predicate now adds `num_reclaimable_cached_blocks()` | `src/runtime/engine_scheduler.cpp:895-903`, `docs/internals/MEMORY.md:552`, `docs/roadmap.md:170` |
| memory note "Graphs eager -40% each wave after first" | The TRIGGER is fixed; the one-way permanence is not (finding C-3) | `src/runtime/engine_scheduler.cpp:180-190` |
| memory note "cuda_graphs=never MoE only, dead on dense" | FIXED: the `never` check now runs before the MoE block, for every model | `src/runtime/engine_weight_upload.cpp:291-323` |
| memory note "first M>1 forward under capture (#1897)" | FIXED: small-M scratch is allocated before the graph prewarm | `src/runtime/engine.cpp:851-853` |
| "token-budget vs request-count scheduling" (implied either/or) | Both, at different points: admission is request-count + KV-block; the per-step prefill quota is token-charged | `scheduler.cpp:79` (`active_.size() < max_batch_size_`) vs `engine_prefill.cpp:88-93, 155-158` |
| ARCHITECTURE.md phase 4 "non-Gemma-4 MoE disables [graphs]" | Wrong; there is no MoE-generic demotion (finding C-8) | `docs/internals/ARCHITECTURE.md:123` vs the 8 `demote_graphs_()` call sites |

### Findings

### [C-1] Mid-flight KV exhaustion is untyped: the client is told "cancelled", the value that means "your client disconnected"
Axis: C   Sev: S1   Confidence: high
Evidence: `CancelReason::KvCapacity` is assigned at exactly ONE site, the admission path (`src/runtime/scheduler.cpp:165`); `grep -rn "cancel_reason" src/ tools/` returns 6 hits, 1 write and 4 reads. The four KV-pressure cancels that fire after admission leave it at `None`: `src/runtime/engine_scheduler.cpp:860-863` (decode `append_block` failure), `:876-879` (SWA `swa_prepare` failure), `src/runtime/engine_prefill.cpp:280-283` (chunk `allocate_blocks` failure), `:532-535` (SWA prepare at prefill). All four increment `kv_pressure_rejections_`, so `/metrics` sees them; the client does not. The consumers: `tools/imp-server/batching_engine.cpp:405-406` and `:453-455` map `cancel_reason != KvCapacity` to `finish_reason "cancelled"`, and `src/api/imp_api.cpp:772-774` / `:918-923` map it to `IMP_ERROR_OUT_OF_MEMORY` instead of `IMP_ERROR_CAPACITY`, so `handlers_chat_core.cpp:982` never sends the `503 capacity_error`.
Current: an admitted generation that runs the pool dry mid-decode ends with the same wire signal as a client disconnect. `docs/API.md:92` documents the counter as covering "admission or mid-decode", which is true for the counter and false for the client-visible typing.
Expectation: vLLM answers KV exhaustion mid-decode by preemption (recompute or swap) and only refuses when it cannot; imp deliberately has no preemption (roadmap Closed "per-request priority"), which makes the typed refusal the whole contract. `tools/imp-server/CLAUDE.md` states the invariant: "A request the server cannot honour is a 4xx, not a best-effort answer", and `scheduler.cpp:162-165` says this cancellation is "actionable, unlike every other cancellation".
Delta: the actionable reason is produced at 1 of 5 sites. It is worse on the flagship configs: the graceful valve (StreamingLLM auto-enable) is gated on `kv_cache_raw_->qtype() == QType::F16` (`engine_scheduler.cpp:898`), so on NVFP4-KV / FP8-KV there is no eviction fallback at all and cancel-newest is the only path - the one that loses its type here.
Cost: 4 one-line assignments in 2 files (`engine_scheduler.cpp`, `engine_prefill.cpp`), plus a test. Risk: near zero (widening an existing enum use). What breaks if wrong: a client that special-cases `"cancelled"` would start seeing `"capacity"` on these paths, which is the intent.
Falsifier: a fifth write of `cancel_reason` somewhere I did not grep, or a downstream that re-derives capacity from the counter. Checked y: the grep above is repo-wide (`src/ tools/`), and both `batching_engine.cpp` sites read the field directly.

### [C-2] Aging bounds queue-position starvation, not KV-blocked starvation: the head of the queue has no reservation
Axis: C   Sev: S2   Confidence: high (code) / med (impact, unmeasured)
Evidence: the sort (`src/runtime/scheduler.cpp:59-74`) lifts a request past `kAgingRounds = 32` (`scheduler.h:24`) to the front of its priority class. But the promotion loop below skips it on insufficient memory and keeps walking: `scheduler.cpp:170-171` (`++it; continue;` after `can_allocate` fails and the growth attempt did not help), and the same skip at `:205-207` (prefix-cache path returned `< 0`) and `:210-212` (plain `allocate_blocks` failed). Nothing marks the aged request as the reserve holder, so a shorter request behind it whose `admit_blocks` DOES fit is admitted in the same round. The infeasible case (`blocks_needed > cap`) is cancelled (`:157-167`) and is tested (`MemoryAwareSkipsLargeAdmitsSmall`, `AllRequestsTooLargeForMemory`, `tests/test_scheduler.cpp:603-662`); the transient case is not: none of the 6 aging/priority tests (`test_scheduler.cpp:801-947`) attaches a `KVCacheManager`, and none of the memory-aware tests advances the round counter.
Current: aging is a pure sort key. Under sustained short traffic that keeps the pool near-full but not full, an aged long prompt can be passed over indefinitely, exactly the unbounded case #1634 was written to close, on a different axis.
Expectation: vLLM's FCFS scheduler blocks the whole waiting queue behind the head request once the head cannot be scheduled (head-of-line blocking is deliberate, precisely as the anti-starvation property); SGLang's longest-prefix scheduler adds an explicit fairness term. imp's aging closes the sort-order half and leaves the allocator half open.
Delta: two different starvation mechanisms, one bounded and tested, one unbounded and untested.
Cost: ~20 LOC in `scheduler.cpp` (stop admitting past an aged request that failed `can_allocate`, or reserve blocks for it), 2-3 tests in `tests/test_scheduler.cpp`. Risk: medium - a strict head-block costs throughput exactly when the pool is tight; it must be gated on the aged flag, not applied to every skip.
Falsifier: the growable pool (`:120-122`, `:154`) always absorbing the pressure, making the skip unreachable in practice. Checked n (needs a GPU run at a pinned pool ceiling); it is reachable by construction once `ceiling_blocks() == total_blocks()`.

### [C-3] The mid-run graph demotion is permanent for the process; the trigger is transient
Axis: C   Sev: S2   Confidence: high
Evidence: `Engine::demote_graphs_` only ever writes `false` (`src/runtime/engine_scheduler.cpp:186`) and there is no re-promotion: `grep -rn "use_cuda_graphs" src/ tools/` shows the field written `true` only at its declaration (`engine.h:54`) and from the C API at construction (`src/api/imp_api.cpp:355`). `StreamingKvKvPressure` fires from a per-step predicate (`engine_scheduler.cpp:903-913`) whose input - pool occupancy - falls again when the wave drains, but `config_.streaming_kv_enabled` is likewise never cleared after that point (`:901`; the only `= false` writes are at init, `engine_weight_upload.cpp:108,127`).
Current: one wave that crosses "free + reclaimable < total/10" on an F16-KV model costs captured decode, and the StreamingLLM eviction stays armed, for the remaining process lifetime. The one-way property is deliberate and documented (`graph_eligibility.h:41-44`, `docs/API.md:94`), but it is documented as a consequence, not as an accepted cost with a number.
Expectation: an engine that turns a fast path off on transient pressure normally turns it back on when the pressure clears (vLLM re-enables CUDA graphs per batch shape; the graph cache is a cache, not a latch). The measured price here is large: `docs/roadmap.md:109` "gate asserts decode >= 1.3x, measures 2.28x", and the #1879 incident measured 2387 -> 1443-1485 tok/s for exactly this latch (`docs/roadmap.md:170`, `docs/internals/MEMORY.md:552`).
Delta: a latch on a transient signal, with a 2.28x fast path on the other side of it.
Cost: re-promotion is not a one-liner - the graph pool bakes block tables that StreamingLLM's `-1` sentinels invalidate, so it needs `streaming_kv_enabled` to be clearable first (drop the sentinels, re-materialise the evicted window is impossible - the KV is gone). Realistic scope: make the demotion recover only when `streaming_kv_enabled` was never actually acted on for any live sequence, ~40 LOC across `engine_scheduler.cpp` + `graph_eligibility.h`. Risk: high if done naively (a replayed graph over a sentinel table is the #948 IMA class).
Falsifier: eviction having already destroyed context on every live sequence by the time the pressure clears, making re-promotion meaningless. Checked partially: `evict_middle_blocks` only runs for sequences past `streaming_kv_threshold` (`engine_scheduler.cpp:920-932`), so a pool that spiked from many short sequences can trip the valve without any sequence being evicted - that is the recoverable case.

### [C-4] The KV-pressure predicate that cost 40% of decode has no test; the only graph-eligibility test checks enum names
Axis: C   Sev: S2   Confidence: high
Evidence: the #1879 defect and its fix are three lines of arithmetic at `src/runtime/engine_scheduler.cpp:895-903` (`st.free_blocks + reclaimable < pool_total / 10`), sitting inside `Engine::step_decode` (222 code LOC, `tools/check_function_size.py`). `tests/test_graph_eligibility.cpp` has 4 tests (`EveryReasonHasAName`, `NamesAreDistinct`, `OnlyKvPressureIsMidRun`, `NoneIsNotMidRunAndNamesItself`) - all of them bind the enum to its string, none of them evaluates the predicate. `grep -rn "num_reclaimable_cached_blocks" tests/` hits only `test_kv_cache.cpp:936` and `test_kv_cache_gpu.cpp:298-308`, which test the counter, not its consumer. The regression it guards is measured: 2387 -> 1443-1485 tok/s over waves 1-3 (`docs/plans/2026-09-04-lever-ledger-detail.md:52`).
Current: the fix is a comment plus a measurement. The next edit to that expression is unguarded, and the CI lane could not see it even if a test existed in the GPU battery (no GPU runner, per the common brief).
Expectation: this repo's own convention - `tools/alloc_allowlist.txt` is a two-way ratchet, `graph_eligibility.h` was made "deliberately header-only and dependency-free (no CUDA, no RuntimeConfig) so the CPU test lane can cover the enum-name binding without a GPU". The same argument applies one level up: the predicate is pure integer arithmetic over four ints.
Delta: the eligibility ENUM is CPU-testable and tested; the eligibility DECISION is neither.
Cost: extract `bool kv_pressure_demotes_graphs(int free, int reclaimable, int total)` into `graph_eligibility.h`, call it from `engine_scheduler.cpp:900`, add 4 CPU-lane cases. ~25 LOC, 2 files + 1 test file. Risk: near zero.
Falsifier: a GPU-lane test asserting the wave-2 throughput. Checked y: `grep -rln "streaming_kv\|kv_pressure\|demote" tests/` returns `test_graph_eligibility.cpp`, `test_engine_integration.cu` (a comment only, `:86`), and three python API tests that read the counter, none of which drives the predicate.

### [C-5] `imp_queue_time_seconds` measures HTTP-submit to worker-pickup, not the queue that `max_batch_size` creates
Axis: C   Sev: S2   Confidence: high
Evidence: `ServerRequest::queue_ms` is stamped in `BatchingEngine::worker_loop` at the moment the request is moved off `pending_queue_` and handed to `engine->add_request()` (`tools/imp-server/batching_engine.cpp:248-254`) - i.e. before `Scheduler::add_request` puts it in `pending_`, and long before `Scheduler::schedule` promotes it to `PREFILLING` under `active_.size() < max_batch_size_` (`src/runtime/scheduler.cpp:79`) and the KV admission test. The worker drains `pending_queue_` unconditionally every loop iteration (`batching_engine.cpp:230-231`, "Move all pending requests to active"), so this stamp is a loop-latency measurement. The histogram observes exactly this value (`handlers_chat_core.cpp:918-920`, `stream_driver.cpp:332-333`, `handlers_chat.cpp:273`), and `/metrics` publishes it as "Seconds from admission to the first decode step" (`tools/imp-server/handlers_misc.cpp:281`) - which is not what it contains. `imp_queue_depth` is `pending_queue_.size() + active_requests_.size()` (`batching_engine.cpp:123`, `handlers_misc.cpp:330-332`), and `active_requests_` holds everything handed to the engine, decoding or still in `Scheduler::pending_`, so the depth gauge cannot separate them either.
Current: at any concurrency above `max_batch_size` the wait that dominates client-observed TTFT lives in `Scheduler::pending_` and is invisible in both the histogram and the gauge; it surfaces only folded into `imp_ttft_seconds`.
Expectation: vLLM exports `vllm:num_requests_waiting` (scheduler-side) separately from `num_requests_running`, and its queue-time histogram is measured to scheduler admission. `docs/internals/BENCHMARKING.md:122` treats the same metric as the preemption/pressure signal.
Delta: the metric named after the queue does not measure the queue. `#1580`'s own comment claims it does ("i.e. it is the time spent waiting behind other requests").
Cost: stamp `queue_ms` when `Request::status` first becomes `PREFILLING` (the engine already owns that transition, `scheduler.cpp:264`) and expose it through the existing C API status read, or split `imp_queue_depth` into waiting/running. ~30 LOC across `scheduler.cpp`, `imp_api.cpp`, `batching_engine.cpp`; docs `API.md:104-108`. Risk: low; the histogram's meaning changes, which is the point, so it needs a CHANGELOG line.
Falsifier: the worker not being able to drain `pending_queue_` in one iteration under load (which would make the stamp meaningful). Checked y: the drain is an unconditional `while (!pending_queue_.empty())` with no cap.

### [C-6] Per-request speculation acceptance is counted and then dropped: no `usage` field, no span attribute, only process-global counters
Axis: C   Sev: S2   Confidence: high
Evidence: `Request` carries `spec_verifies`, `spec_drafted`, `spec_accepted` (`src/runtime/request.h:79-81`). `grep -rn "spec_drafted\|spec_accepted\|spec_verifies" src/ tools/` shows the only consumers outside `engine_spec_*.cpp` are the process-global Prometheus counters `imp_spec_drafted_total` / `imp_spec_accepted_total` / `imp_spec_verify_steps_total` / `imp_spec_miss_steps_total` (`tools/imp-server/metrics_memory.cpp:121-132`) and `tools/analysis/serving_kpi.py:204-219`, which differences them over a window. The OTLP span carries `imp.cached_tokens`, `imp.queue_ms`, `imp.ttft_ms` and the gen_ai token counts but nothing speculative (`tools/imp-server/tracing.cpp:104-128`, `:154`). No `usage` extension either: the request-level extras that DO exist are `prompt_tokens_details.cached_tokens` and `evicted_tokens` (`docs/API.md:94`).
Current: with `"speculative": true/false` settable per request (`request.h:101`) and `mtp_k=auto` adapting per request (roadmap Open 4), the counters average over a mix the operator chose to make heterogeneous. Attributing an acceptance collapse to a workload class requires a single-class server.
Expectation: TensorRT-LLM and vLLM v1 both report per-request accepted-draft-token counts in the response (`num_accepted_tokens` / `spec_token_acceptance_counts`); the data here already exists on the request object.
Delta: three integers that reach the end of the request and are never read.
Cost: one `usage.completion_tokens_details` extension in the three dialects plus two span int-attrs. ~40 LOC across `handlers_chat_core.cpp`, `handlers_messages.cpp`, `responses.cpp`, `tracing.cpp`, `imp_api.cpp` (the fields need a C-API read). Risk: low, additive fields. What breaks if wrong: a strict OpenAI schema validator on the client side.
Falsifier: an existing per-request surface I missed. Checked y: the grep above is repo-wide and `docs/API.md` documents no such field.

### [C-7] The decode-pipeline gate diagnostic prints `ssm_ok=1` for the hybrids it refuses
Axis: C   Sev: S3   Confidence: high
Evidence: the gate is unconditional - `if (ssm_state_) return false;` (`src/runtime/engine_decode_pipeline.cpp:112`), kept as a measured verdict after #1750 made the pipeline runnable on hybrids (`:104-111`: pipeline ON 862-914 vs OFF 940-953 tok/s at 32 streams). The once-per-process diagnostic still evaluates the pre-#1750 rule: `(int)!(ssm_state_ && !(runtime_config_.runtime.gdn_batched_decode && d_ssm_seq_slots_ != nullptr))` (`:145-146`). With `gdn_batched_decode = true` by default (`src/runtime/config.h:204`) and the slot table present, this prints `ssm_ok=1` while `ssm_state_` is what closed the gate. Every other flag in that line mirrors its gate exactly.
Current: the log line whose whole purpose is "say which gate closed" (`:124-128`, written against the same blind spot as #1646) names the wrong one on GDN/Mamba2 hybrids - the models where the question is most likely to be asked.
Expectation: this repo's own standard, SETTLED E: "A gate that cannot be shown to fire has not been validated" (#1205/#1210).
Delta: one boolean expression out of sync with the gate it reports.
Cost: 1 line in `engine_decode_pipeline.cpp:146`. Risk: none.
Falsifier: `gdn_batched_decode` being false or `d_ssm_seq_slots_` null in practice, which would make the printed value accidentally correct. Checked y: default is `true` (`config.h:204`) and the slot table is the shipped path (`engine_scheduler.cpp:759`).

### [C-8] ARCHITECTURE.md says MoE disables graph capture; the code has no MoE-generic demotion
Axis: C   Sev: S3   Confidence: high
Evidence: `docs/internals/ARCHITECTURE.md:123` - "Graph capture is enabled for most architectures; non-Gemma-4 MoE disables it because of host-side routing." The 8 `demote_graphs_()` call sites are `ConfigNever`, `DebugRaw`, `Gemma4NoGraphs`, `CalibrationActive`, `StreamingKvConfigured`, `ExpertsOnHost`, `PinnedSampleBufUnavailable`, `StreamingKvKvPressure` (`src/runtime/graph_eligibility.h:33-46`; call sites: `engine_workspace_warmup.cpp:217`, `engine_weight_upload.cpp:95,121,315,368`, `engine_init_resolver.cpp:69,797`, `engine_scheduler.cpp:913`). The only MoE-related one is `ExpertsOnHost`, and it is guarded on host-RESIDENT experts (`engine_weight_upload.cpp:325-346`), with gpt-oss explicitly exempted because its MXFP4 experts become on-device NVFP4 (`:326-332`).
Current: an on-device NVFP4 MoE (the shipped Qwen3.6-35B-A3B / Qwen3-Coder-30B class) captures decode graphs. The doc says it does not.
Expectation: `docs/internals/ARCHITECTURE.md` is the canonical narrative per root `CLAUDE.md`; L2 doc claims are citation-gated for `docs/`.
Delta: one stale sentence in the canonical narrative, on the axis (graph coverage) where readers use it to predict performance.
Cost: 1 sentence. Risk: none.
Falsifier: an MoE-specific graph refusal below `demote_graphs_` (e.g. a capture that always aborts). Checked partially: `moe_prefill_uncapturable()` gates the PREFILL graph only (`engine_prefill.cpp:706`); the decode gate at `engine_scheduler.cpp:1434` reads `config_.use_cuda_graphs` alone. A capture that aborts at runtime would set `capture_failed_` per runner (`cuda_graph.cu:410-449`), not a demotion - so the sentence would still be wrong about the mechanism.

### [C-9] `"speculative"` is a per-request field in 2 of 3 dialects; `/v1/responses` drops it while bridging `priority`
Axis: C   Sev: S3   Confidence: high
Evidence: `tools/imp-server/responses.cpp` maps `model`, `messages`, `tools`, `tool_choice`, `parallel_tool_calls`, `response_format`, `temperature`, `top_p`, `max_tokens`, `priority` (`:173`), `think_budget`, `stream` into the OpenAI body. `grep -rn "speculative" tools/imp-server/handlers_responses.cpp tools/imp-server/responses.cpp` -> 0 hits. The Anthropic dialect does bridge it (`anthropic.cpp:341-342`).
Current: an Agents-SDK / Codex client cannot switch speculation off per request; the same client CAN set `priority`. `docs/API.md:53` describes the field without naming the dialects that carry it.
Expectation: `tools/imp-server/CLAUDE.md` - "three wire-format adapters on a shared core"; `docs/roadmap.md:71` records `priority` as reaching "all three dialects", which is the standard this field misses.
Delta: one line in the Responses bridge.
Cost: 2 LOC in `responses.cpp`, 1 doc line, 1 mock-API assertion. Risk: none.
Falsifier: `/v1/responses` deliberately excluded from non-OpenAI extensions. Checked n: it already carries the non-OpenAI `priority` and `think_budget`, so the exclusion is not a policy.

### [C-10] `runtime.prefill_graph` is default-on and structurally dead on every quantized-KV config
Axis: C   Sev: S3   Confidence: high (code) / low (impact)
Evidence: the capture gate requires `kv_append_capturable = (config_.kv_cache_dtype == QType::F16)` (`src/runtime/engine_prefill.cpp:695`) and `offset == 0` (`:704`), on top of `pf_pool_used`, `use_cuda_graphs`, `!ends_at_snapshot`, `!nvfp4_dequant_uncapturable()`, `!moe_prefill_uncapturable()` (`:703-706`). The default is `prefill_graph = true` (`src/runtime/config.h:119`), flipped on 2026-05-17. The in-tree comment records that on Qwen3-8B-Q8_0 and Qwen3-Coder-30B-A3B-NVFP4 "neither ever captured a prefill chunk" and that finding out which gate closed cost "a source read plus three A/Bs" (`:728-734`); the resolution was a log line, not a gate change. There is one `prefill_graph_runner_`, invalidated whenever `chunk_len` or `block_table.size()` changes (`:709-713`), so even on F16 KV it re-captures per distinct prompt length.
Current: on the KV dtypes this engine ships for long context (NVFP4-KV, FP8-KV, MXFP4-KV, INT4/INT8 - all listed as supported by `supports_chunked_prefill_()`, `engine_scheduler.cpp:434-439`) the flag can never fire, by construction rather than by measurement.
Expectation: `find-stubs` category - a default-on knob whose precondition excludes the flagship configuration. The repo's own remedy for the same class was the `runtime.cuda_graphs="never"` fix (`engine_weight_upload.cpp:291-305`), which was about a setting "read, stored, and then never looked at".
Delta: a default-on capture path that is unreachable on the shipped KV dtypes, and per-prompt-length re-capture where it is reachable.
Cost: either resolve `prefill_graph` to `false` with a stated reason at init (like `engine_init_resolver.cpp`'s ~25 policy lines, S-25) or key the runner by `chunk_len` the way the decode pool is keyed by batch. Resolve-and-say-so is ~15 LOC in `engine_init_resolver.cpp`. Risk: low.
Falsifier: F16 being the resolved KV dtype for most deployments, making the path live. Checked n (needs the resolver's runtime decision on a real checkpoint, `init_resolve_kv_dtype_policy_`, `engine_init_resolver.cpp:183`) - listed under Open questions.

### Checked and NOT a finding

- **Rejected draft KV is dropped, not masked, and the drop is block-granular but safe.** `KVCacheManager::rollback(seq, p0 + 1 + matched)` keeps `ceil(new_seq_len / block_size)` blocks and leaves stale slots inside the last kept block (`src/memory/kv_cache_manager.cpp:1156-1197`); nothing reads them because every paged gather is bounded by `context_len`, and the next write overwrites them. Hash-registered blocks leave the prefix table on the same path (`:1174`), and the SWA table is trimmed in lockstep (`:1188-1196`).
- **Every spec-verify exit path rolls back.** 8 `kv_manager_->rollback(req->id, p0)` sites plus the accept-site `rollback(req->id, p0 + 1 + matched)` (`src/runtime/engine_spec_ngram.cpp:574, 585, 775, 801, 809, 827, 887, 927, 1111`), including the KV-exhaustion exit at `:570-575`, which trims back to the pre-step length instead of cancelling.
- **Multi-candidate private blocks are freed by the same rollback**, not by a separate path (`engine_spec_ngram.cpp:553-562`, `:1002`).
- **Padded verify rows cannot pollute the real rows.** Pads sit at positions after every real row, so causal masking hides them, and their KV falls to the rollback (`engine_spec_capture.cpp:19-23`); hybrids read the real chunk length from device (`d_chunk_len`) so the recurrent state stops at the real last row (`:25-30`).
- **Spec capture has a fidelity check with a number.** `diagnostics.spec_capture_fidelity` re-runs eager, restores the slab, replays the graph and diffs argmax (`engine_spec_capture.cpp:203-250`): 0/400 differing on Qwen3.8-27B-NVFP4 and Qwen3.6-35B-A3B-NVFP4, 45/400 on Nemotron-3.5-Lightning-30B-A3B-NVFP4 (`:205-209`).
- **The spec graph cache is keyed and bounded**, not per-request: `std::map<tuple<n_tokens, ctx_capacity, rec_slot, grouped_rows>, SpecVerifyGraph>` (`engine.h:1004`), with 8 chunk buckets (`engine_spec_capture.cpp:131`) and pow2 ctx tiers from 4096 (`:113-118`). It is invalidated wholesale on workspace move (`:181-189`) or launch failure (`:255-257`), never per entry - acceptable given the key space, but see Open questions.
- **Constrained decoding does NOT fall off the decode graph.** Only sampling is eager; `needs_constrained` gates the decode PIPELINE, not the capture (`engine_scheduler.cpp:1434`, `:1512`). The separate `step_constrained_pipeline` is a fast path, not a fallback.
- **The `#1643` prefill-rescheduling hole is closed and commented**: `prefill_offset >= 0`, not `> 0`, keeps a promoted-but-unserved request in the batch (`scheduler.cpp:272-292`).
- **The prefill rotor is id-based, not index-based**, because an index rotor systematically jumped a moving cohort - 5 of 32 burst requests starved to wave-end TTFT (`engine_prefill.cpp:96-107`).
- **The ragged prefill charges real rows, not the full chunk** (`engine_prefill.cpp:139-166`); the priced consequence is in `docs/plans/2026-09-04-lever-ledger-detail.md:43` (1094-token prompts 943.7 -> 1058.0 tok/s, ITL p95 46.2 -> 19.9 ms).
- **Admission reserves prompt + generation, not the prompt alone** (`scheduler.cpp:96-110`, `:227-236`), tested by `AdmissionReservesGeneration` and `AdmissionClampsReserveToPoolSize` (`test_scheduler.cpp:142-265`).
- **Decode-batch truncation cannot park rows in the default configuration**: admission is clamped to `min(EngineConfig::max_batch_size, runtime.max_batch_size)` with a log line (`src/runtime/engine.cpp:804-814`), the #1637 fix.
- **`max_batch_size` above the graph pool is announced, not silently clamped** (`engine_init_resolver.cpp:531-537`, cites 2.4x slower eager decode).
- **`cudaDeviceSynchronize()` in `CudaGraphConditionalRunner::cleanup()`** is device-wide, but the async loop only launches at `decode_batch.size() == 1` (`engine_scheduler.cpp:619`) and `prefill_overlap` requires `>= 2` decode rows (`engine_scheduler.cpp:120`), so no concurrent stream is stalled.
- **`imp_spec_*` metrics are documented**, just not in `docs/API.md`'s decision-counter table: `docs/internals/BENCHMARKING.md:121`, `docs/LIMITATIONS.md:259`.
- **Graph re-capture on context growth is monotonic and bucketed** (`engine_scheduler.cpp:1478-1481`), ~log2(max ctx) per process; the #948 IMA class it closes is documented in place.
- **`graph_prewarm` walks the batch pool at init** with a ~1000-token anchor so captures bake the 1024 ctx bucket (`config.h:59-69`), against 75 captures in the first wave of a 4-wave 32-stream run (704 vs 953-976 tok/s).

### Known-and-accepted (restated)

- Roadmap Open 2, paced serving prefill (dense): 31.4k prompt tokens in ~2 s under decode vs 26k tok/s standalone, vLLM ~1.1 s; `prefill_chunk_decode_cap` stays 1024 because 2048 buys +4.4% hybrid / +8.6% dense and costs the decoders +70% / +63% ITL during a foreign ingest.
- Roadmap Open 1, launch-coupled idle @32 (~8% real): both direct attacks closed; the next cut is fewer launches.
- Roadmap Open 4, speculation adapts per request: HALF CLOSED - chain depth adapts (#1801), drafter CHOICE is global, the chain saturates near 2.5 accepted/verify, the W=2 tree measured no gain (roadmap Closed "speculation tree").
- Roadmap Open 3: StreamingLLM eviction (`src/compute/attention_paged_common.cuh:71`) is the only answer under KV-pool pressure; no preemption anywhere (roadmap Closed "per-request priority").
- Roadmap Open 5, recurrent-state paging; Open 11, no KV tier below VRAM (DO NOT BUILD).
- `runtime.prefill_overlap` default OFF is a measured verdict, not caution: neutral at 32 streams on Qwen3.8-27B-NVFP4 both short-prompt (1771.3 vs 1777.7) and heavy-ingest (789.7 vs 790.6), because sm_120 has no green-context SM partitioning (`config.h:243-253`).
- Hybrid decode fairness is a time-slice, not a batch: without `gdn_batched_decode` the decode batch is resized to 1 and rotated per `hybrid_decode_quantum` (`engine_scheduler.cpp:760-801`).
- No GPU CI lane; the server streaming path is not in the perf gate (#1685).

### Open questions

- Does `init_resolve_kv_dtype_policy_` (`engine_init_resolver.cpp:183`) leave F16 KV as the resolved default on the reference checkpoints? That decides whether C-10 is "dead everywhere" or "dead on the long-context configs only". Needs one server boot with the resolved-dispatch log.
- How many entries does `spec_graphs_` actually reach on a 32-stream hybrid run, and what does each `SpecVerifyGraph` cost in VRAM? Key space is 8 buckets x ~4 ctx tiers x (max_batch+1) slots x 2; there is no per-entry eviction. Needs a GPU run with the spec-capture log.
- Is C-2's transient skip reachable in production, or does the growable pool always absorb it before the ceiling? Needs a run with `kv_cache` pinned at its ceiling under mixed short/long traffic.
- What is the real distribution of `Scheduler::pending_` residency at concurrency 64 with `max_batch_size = 32`? That is the number C-5 would expose.


## Axis D: Correctness and determinism (read-only audit, 2026-09-05, HEAD ef664dd8)

### Coverage

Read in full:
- `docs/determinism.md`, `tools/check_determinism_sites.py`, `tools/check_launch_guards.py` (docstring + scanner), `tools/log_fatal_allowlist.txt`, `tests/CLAUDE.md`, `src/compute/CLAUDE.md`, `docs/audit/SETTLED.md` sections A, B, C, C2, D, D2, E, F, G (lines 48-470), `docs/LIMITATIONS.md` "Untested code paths", "Gates that do not exist", "Known-bad" (lines 26-300).
- `src/core/cuda_raii.h` (all wrapper ctors/dtors), `src/core/logging.h:55-140`, `tests/test_capture_abort.cu`, `tests/test_determinism_e2e.cpp` (header + every assert), `tests/test_forward_pass.cu:255-455`, `tests/test_nvfp4_gemm_batched.cu:225-310`, `tools/imp-server/batching_engine.cpp:12-45, 275-366`, `src/exec/executor_gemm_dispatch.cu:60-300, 430-640`.

Sampled (the lines cited in findings were opened and verified):
- `src/compute/moe_routing.cu`, `moe_routing_permute.cu`, `gemm_moe_fused_tc.cu`, `gemm_cutlass_grouped_3x.cu`, `gemm_cutlass_sm120.cu`, `gemm.cu`, `sampling_topk_topp.cu`, `sampling_filters.cu`, `sampling_penalties*.cu`, `warp_reduce.cuh`, `src/quant/nvfp4_gemm_smallm*.cu`.
- `src/exec/executor_forward_moe*.cu`, `executor_forward.cu`, `executor.cu`, `executor_sampling.cu`, `executor_workspace.cu`, `sparse_attn_select.cu`.
- `src/runtime/engine.cpp`, `engine_scheduler.cpp`, `engine_spec_capture.cpp`, `engine_workspace_warmup.cpp`, `cuda_graph.cu`, `graph_diag.h`; `src/api/imp_api.cpp` (try/catch map + `imp_decode_step`); `src/model/model.cpp`, `src/memory/kv_cache.cu`, `vram_allocator.cu`.
- Test-name listings of ~45 test files under `tests/` (quant, gemm, attention, KV, e2e), `tests/refs/`, `Makefile` test/verify targets, `scripts/verify.sh` (grep), `CMakeLists.txt:995-1030`.
- Ran: `tools/check_determinism_sites.py --report`, `tools/check_launch_guards.py`, `tools/check_log_fatal.py`, `tools/check_test_lanes.py --report`, `gh pr view 1766`.

Skipped: `src/vision/`, `src/lora/`, `tools/imp-quantize/`, `tests/api/*.py`, attention-kernel numerics beyond test names, `docs/audit/SETTLED.md` per-area logs (lines 470-652).

### Brief vs repo

| Brief statement | Repo | Evidence |
|---|---|---|
| "memory says imp Forward NOT deterministic, 93.6 % floats differ, for the default mode" | The 93.6 % was measured on `--calibrate` activation-statistics files, dense Qwen3-0.6B/1.7B, PPL identical in both runs (33.7699); cause was cuBLASLt algo selection, fixed by `runtime.deterministic_gemm`; `--calibrate` now forces the flag. Not a logit measurement, not MoE. | `~/.claude/.../memory/awq_calibration_and_nondeterministic_calib_2026_07_31.md:26-38` |
| `tests/test_determinism_e2e.cpp` records drift magnitude | It asserts bit-equality only (`EXPECT_EQ(first, second)`, `EXPECT_EQ(ppl1, ppl2)`); every magnitude lives in `docs/determinism.md` | `tests/test_determinism_e2e.cpp:151,167,188,214,236` |
| MoE atomics: `src/compute/gemm_grouped*` | Zero atomics in `gemm_grouped.cu`, `gemm_grouped_nvfp4_smallM.cu` (only `cp.async`); CUTLASS grouped GEMM runs with default `TileSchedulerArguments` (no split-K, no stream-K) | `rg atomic src/compute/gemm_grouped*` = 0 hits; `gemm_cutlass_grouped_3x.cu:318,635` |
| `check_launch_guards.py` "what it exempts" | `ALLOWLIST = {}`; 437/437 in-scope launches guarded; 25 launches inside `#define` bodies are out of scope by design; the guard macro is log-only (`cudaPeekAtLastError`, no clear, no throw) | `tools/check_launch_guards.py:55`, gate output, `src/core/logging.h:108-116` |
| `tools/log_fatal_allowlist.txt` "FATAL-then-continue sites" | One entry; the function returns a verdict, its callers wrap it in `IMP_CHECK`. Gate: 2 FATAL sites, 1 abort, 1 allowlisted continue | `tools/log_fatal_allowlist.txt`, `tools/check_log_fatal.py` output |
| Batch invariance "GGUF not invariant by design (#1897); what about native NVFP4" | Native NVFP4 batched rows (M 2..32) are the same W4A4 kernel family; #1897's own text says so ("the numerics family of the native NVFP4 batched path (#1766)") | `docs/LIMITATIONS.md:71-76`, `src/exec/executor_gemm_dispatch.cu:356-415` |
| `(void)cuda*` swallowed checks | 33 sites, all in teardown / cleanup / capture-abort / pool-attribute paths; none on a request path | `rg '\(void\)\s*cuda' src/ tools/` |
| F-17 (SETTLED E) "CUTLASS grouped never consults the flag" | Still true; gate pins 6 reads in 4 files, none in `gemm_cutlass_grouped_3x.cu` | `check_determinism_sites.py --report` |

### Findings

### [D-1] A CUDA fault that does not throw is cleared once per forward and never surfaced: the server keeps serving stale tokens with /health ok
Axis: D   Sev: S1   Confidence: high
Evidence:
- `src/exec/executor_forward.cu:213-217`: every `forward_logits` starts with `cudaGetLastError()` and downgrades a pending error to `IMP_LOG_WARN("Cleared stale error before forward")`.
- Every CUDA check in the hot path is log-only: `rg -c IMP_CUDA_CHECK_LOG src/exec src/runtime` = 377; `IMP_CUDA_CHECK_BOOL|VOID` = 2 in all of `src/`; no throwing CUDA macro exists (`src/core/logging.h:83-126`). `IMP_CUDA_CHECK_LAUNCH` logs and continues (`logging.h:108-116`).
- `src/exec/executor_sampling.cu:497` and `:530`: `IMP_CUDA_CHECK_LOG(cudaStreamSynchronize)` / `cudaEventSynchronize`, then the pinned token buffer is returned regardless, so a failed step hands back the previous step's tokens.
- `src/runtime/engine_scheduler.cpp:70-160` (`Engine::step` / `step_impl_`): no CUDA-error path; `rg cudaError_t src/runtime/engine_scheduler.cpp src/runtime/engine_graph_decode.cpp` = 0.
- `src/api/imp_api.cpp:938-962`: `imp_decode_step` returns `IMP_SUCCESS` with whatever token the step produced; only "no token in 8 steps" is `IMP_ERROR_INTERNAL`.
- `tools/imp-server/batching_engine.cpp:312-366`: the poisoned-context detector (`cudaDeviceSynchronize` + `cudaGetLastError` + `cuda_error_is_unrecoverable`) runs only inside the two `catch` blocks; `faulted_` (`:335,364`) is set only there; `rg 'cudaPeekAtLastError|cudaGetLastError' tools/imp-server` = those two sites. The file's own comment (`:17-21`) names the mode: "every later launch returns the same sticky error, which forward() silently clears and ignores, so the engine would serve garbage forever".
- CUTLASS entry points turn a sticky error into a silent `return false` (`src/compute/gemm_cutlass_sm120.cu:877-881`, `gemm_cutlass_grouped_3x.cu:197-201`), which the MoE tier chain reads as "tier declined" and falls through (`src/exec/executor_forward_moe_cutlass.cu:678,705` per SETTLED G F-3).
- `graph_diag::check_post_launch` (`src/runtime/graph_diag.h:50-62`) is the only per-launch sync+check and is gated on `diagnostics.graph_diag` (default off).
Current: a device fault (illegal address, launch failure) raised inside a kernel never becomes a host exception. Launch guards log, syncs log, the next forward clears the sticky error with a WARN, the sampler returns stale pinned tokens, the C API returns `IMP_SUCCESS`, the server returns 200 and `/health` stays `ok` (the `engine_faulted` code in `tools/imp-server/utils.cpp:227-230` is reachable only through a host throw). #874's fix covered the throw path; the no-throw path is the common one for device faults.
Expectation: vLLM v1 `EngineCore` fails all in-flight requests and marks the engine dead on any exception from the model runner, and `torch.cuda` raises on the sticky error at the next sync, so the failure is a host exception by construction; TensorRT-LLM's executor checks the stream after each iteration and surfaces `cudaError` to the request. In imp nothing converts a sticky device error into a host signal.
Delta: no per-step probe of the sticky error state between `engine->step()` and token distribution; no promotion of an unrecoverable class to `faulted_` outside `catch`.
Cost: ~20 LOC in `tools/imp-server/batching_engine.cpp` (after `engine->step()`: `cudaPeekAtLastError()`, and on an unrecoverable class run the existing cancel + `faulted_` block) plus ~10 LOC in `Engine::step_impl_` or `collect_sampled_tokens` to return failure when the sync failed. Risk: false positives from the benign sticky errors the tree deliberately clears (green-context init `engine_workspace_warmup.cpp:52-59`, capture-invalidated `cuda_graph.cu:341`, `cudaGraphDebugDotPrint` on WSL2 `graph_diag.h:75`), so the probe must use `cuda_error_is_unrecoverable` and run after those clears. Breaks if wrong: a recoverable error stops the worker.
Falsifier: a per-step probe exists that I missed. Checked: y. `rg 'cudaPeekAtLastError|cudaGetLastError|cudaDeviceSynchronize' tools/imp-server/*.cpp` shows only the catch-block sites (`batching_engine.cpp:329-330,358-359`); the memory note `environment_vs_code_diagnosis_2026_08_22.md:13-18` records 73 red tests from one sticky CUDA 700 in a test process, the same mechanism.

### [D-2] Native-NVFP4 batched decode (2..32 rows) is W4A4 while single-stream is W4A16; no teacher-forced or logit-level instrument reaches the batched path, and the docs price it as "rounding only"
Axis: D   Sev: S2   Confidence: high
Evidence:
- Dispatch: M=1 takes `gemv_nvfp4_kpar` with FP16 activations (`src/exec/executor_gemm_dispatch.cu:183-237`); M<=32 with `gemm.nvfp4_smallm` (default on) quantizes the activation rows into `smallm_xq_` and calls `gemm_nvfp4_smallm_v2_a4` / `gemm_nvfp4_smallm_a4` (`:356-415`). `src/quant/nvfp4_gemm_smallm_v2.cu:1-5`: "both sides packed NVFP4 in the PLAIN layout ... no dequant anywhere".
- PR #1766 body (`gh pr view 1766`): "Not in here: PPL re-measurement. Teacher-forced PPL runs prefill (M=2048, CUTLASS) and cannot reach the batched-decode-only path ... Quality evidence is the server-level degen battery."
- `docs/LIMITATIONS.md:159-161` and `docs/determinism.md` ("Joining a batch may cost rounding, and only rounding ... max |delta| 3.1e-3 over a logit range of 1.41 (0.22 %)"): the number comes from `ForwardPassTest.DecodeLogitsInvariantToBatchComposition`, a 2-layer FP16 synthetic model (`tests/test_forward_pass.cu:372-455`), which never runs an NVFP4 kernel.
- Kernel tests compare each path against its own dequantised reference with tolerances (`NvFP4SmallMTest.A4MatchesHostReference` `tests/test_nvfp4_smallm.cu:238`, `BatchedSmallM.MatchesDequantisedReference` `tests/test_nvfp4_batched_smallm_equiv.cu:155`); none compares the M=1 kernel against the M=2 kernel on the same rows. `Nvfp4VerifyRowParity` (`tests/test_nvfp4_gemm_batched.cu:276-304`) pins verify-chunk vs decode bit-parity but for `gemm_nvfp4_batched` (W4A16), not the smallM A4 kernel.
- `docs/quantization.md:89`: the quantizer leaves `input_activations` null, "vLLM reads that as NVFP4A16".
Current: a request served at c>=2 has every linear layer's activation rounded to NVFP4 (per-16 micro-scales) before the MMA; the same request at c=1 keeps FP16 activations. GGUF sources with the decode overlay take the same kernel (#1897, documented). The only quality evidence is `degen_suite` 50/0 (coherence), not NLL, KL or argmax agreement.
Expectation: vLLM and TensorRT-LLM serve an NVFP4A16 checkpoint with W4A16 kernels at every batch size and use W4A4 only for checkpoints quantized with calibrated input scales (NVFP4A4). A dynamic per-row activation quantization on the serving path is a numerics change that the CUTLASS prefill already makes at M>=33, but there it is priced by teacher-forced PPL; here it is not priced at all.
Delta: the c=2..32 serving path has no quality number; the docs claim the solo/batched delta is rounding-sized on the strength of an FP16 harness.
Cost: measurement first, 0 code LOC: server-JSON logprobs at c=1 vs c=8 on identical prompts (`server.prefix_cache=false`, `speculative.mtp_k=0`, `speculative.ngram=false`), compare top-1 agreement and mean |delta logprob|; or extend `DecodeLogitsInvariantToBatchComposition` to an NVFP4 `DenseTestModel` (~150 LOC, GPU-only). If material: `gemm.nvfp4_smallm=false` is the existing switch (routes 2..32 rows to the CUTLASS/dequant prefill GEMM, W4A16), measured -11 % aggregate at c=32 per the v2 header (`nvfp4_gemm_smallm_v2.cu:8-14`). Breaks if wrong: nothing; the finding asks for a number and a doc correction.
Falsifier: a c=1 vs c>=2 NLL/logprob measurement exists. Checked: y. `docs/PERF.md` "Serving KPIs" holds throughput only; `docs/LIMITATIONS.md`, `docs/determinism.md`, `docs/plans/2026-08-24-qwen38-port.md` (grep `ppl|kl|nll` x `a4|smallm|batched`) and the PR body carry none.

### [D-3] The default-mode greedy-reproducibility gates run from no Makefile target
Axis: D   Sev: S2   Confidence: high
Evidence:
- The three asserts of default-mode (no `deterministic` flag) bit-identical greedy output: `GreedyLockTest.FrozenSequences` (`tests/test_e2e_greedy_lock.cpp:133`, header calls it "the single highest-leverage test class"), `DegenerationTest.GreedyDeterminism` (`tests/test_degeneration.cpp:215-233`), `PrefixCacheE2ETest.ControlNoCacheBackToBackIsDeterministic` (`tests/test_prefix_cache_e2e.cpp:110-121`). All three are in `test-e2e` (`CMakeLists.txt:1000-1027`) and `GTEST_SKIP` without `IMP_TEST_MODEL`.
- `make test-gpu` runs `imp-tests` through `DOCKER_RUN` with no model env (`Makefile:14,130`), then only `*DetEvalE2ETest*` with env (`:136-140`).
- `make test-e2e` filters to `PrimaryModelTest.*:GDNModelTest.*:EndToEndModelTest.*:Gemma4ModelTest.*:Gemma4GraphsTest.*:SpecCaptureFidelityTest.*:*DetEvalE2ETest*` (`Makefile:196`); none of the three suites matches.
- `scripts/verify.sh` runs two smoke prompts (`:820-827`) and the perf gate; `rg 'GreedyLockTest|FrozenSequences|PrefixCacheE2ETest|DegenerationTest'` over `Makefile`, `scripts/`, hooks = 0 invocations. The only recorded runs are by hand with `IMP_TEST_MODEL` exported (`docs/audit/PERF_LOG.md:292,386`).
- `tools/check_test_lanes.py --report` is per module (`test-e2e 203 macros, split`), so it cannot see a suite that is compiled, GPU-lane, and never selected.
- The lock table itself notes a default-mode nondeterminism it could not lock (`tests/refs/e2e_greedy_locks.h:65-71`: "NON-DETERMINISTIC across fresh contexts at temp=0 on a DENSE model ... likely an exact logit tie broken by a non-deterministic argmax reduction"), and the locks date from 2026-06-04 while the forward has changed since (FA2 softmax, NVFP4 GQA grouping, scale folding per CHANGELOG).
Current: default-mode greedy reproducibility and the frozen-token locks for Qwen3-8B-Q8_0 and Qwen3-8B-NVFP4-cortecs are asserted by tests that no target executes. `DetEvalE2ETest` (deterministic mode) does run; the default mode, which every server and CLI user gets, has no executed gate. AUDIT.md G9 records the silent-skip trap for two of the suites, not the filter gap.
Expectation: llama.cpp runs its model-backed greedy checks per PR; here the repo owner declined GPU CI (SETTLED G F-5), so `make test-e2e` and `verify-fast` are the only homes, and the lock test was written for exactly that ("From then on ANY token drift fails loudly").
Delta: one filter string omits three suites.
Cost: 1 line (`Makefile:196`, append `GreedyLockTest.*:DegenerationTest.*:PrefixCacheE2ETest.*`); +1-2 min per `make test-e2e`. Risk: the first run may fail on stale locks and force a regeneration with external verification (llama.cpp for the GGUF lock), which is the documented lifecycle (`test_e2e_greedy_lock.cpp:12-27`). Breaks if wrong: a flaky lock blocks the target; the header's rule is that a flaky lock is itself a finding.
Falsifier: the suites are selected by a glob I missed. Checked: y. `EndToEndModelTest` exists only in `tests/test_e2e.cpp`; no other target passes `IMP_TEST_MODEL` to `imp-tests`.

### [D-4] The documented mechanism for default-mode MoE drift ("MoE routing uses atomics: identical seeds can diverge") does not describe the default F16 path
Axis: D   Sev: S3   Confidence: med
Evidence:
- `docs/LIMITATIONS.md:163`: "MoE routing uses atomics: identical seeds can diverge". `docs/determinism.md` first bullet: "MoE token routing: atomic expert-bucket scatter ordering".
- Default prefill scatter is the fused token-centric kernel with no atomics when `token_to_expanded` is set and compute dtype is F16 (`src/exec/executor_forward_moe.cu:641-660`, `src/compute/moe_routing_permute.cu:325-331`); the FP32 `atomicAdd` scatter (`moe_routing_permute.cu:88`) is the fallback branch (`executor_forward_moe.cu:661-667`). Routing takes the buffered overload whenever `moe_.routing_buffers.pool` exists (`executor_forward_moe_batch.cu:803-807`), which sets `token_to_expanded` (`moe_routing.cu:756-760`). Decode combine `moe_weighted_sum_residual_kernel` (`moe_routing_permute.cu:295-317`) has no atomics.
- The remaining MoE atomics are shared-memory integer counters in `moe_fused_permute_kernel` (`moe_routing.cu:397,417`) that decide the row order inside an expert bucket; a per-row GEMM output does not depend on row position, so the order changes no FP value on the fused path. The deterministic permute kernel's own comment ties the effect to "the atomic scatter path" (`moe_routing.cu:431-436`).
- The measured default-mode cross-process drift in `docs/determinism.md` known limit 5 (NLL 1.3113 vs 1.2889, Qwen3.8-27B-NVFP4) is on a dense GDN hybrid, so its mechanism is cuBLASLt timing-based algo selection (`src/compute/gemm.cu:478-481`), not MoE atomics; `tests/refs/e2e_greedy_locks.h:65-71` attributes the dense flip to an argmax tie.
Current: a reader of LIMITATIONS is told MoE atomics are the source of seed divergence; the FP atomics on the default MoE path are gone, and the sources that remain in default mode are cuBLASLt algo selection, the FP32-compute MoE fallback, `softmax_sum_device_max_kernel`'s cross-block FP `atomicAdd` (`sampling_topk_topp.cu:611-616`), `typical_p` (known limit 3) and CUB top-k > 128 (known limit 2).
Expectation: expectation unclear for wording; the doc gate (`check_determinism_sites.py`) pins sites, not mechanisms.
Delta: one sentence in LIMITATIONS and the lead bullet in determinism.md name a retired mechanism ahead of the live ones.
Cost: 2 doc lines. Risk: none.
Falsifier: the default MoE prefill on gpt-oss / Qwen3-30B takes the non-buffered `moe_topk_gating` overload (`token_to_expanded = nullptr`, `moe_routing.cu:640`) so the FP32 atomic scatter is still the default. Checked: partially. `routing_buffers.pool` is the condition (`executor_forward_moe_batch.cu:803`); its allocation condition was not traced. Needs one log line on a real MoE run.

### [D-5] Numerics coverage per quant format: the seven LIMITATIONS "untested" formats have no kernel-level golden either, and two more paths sit on dispatch-vs-direct tests only
Axis: D   Sev: S3   Confidence: high
Evidence (file:TEST; "dispatch==direct" means two in-tree paths compared with each other, no independent reference):

| Format | Dequant vs CPU reference | GEMM/GEMV vs reference | End-to-end gate (target) |
|---|---|---|---|
| Q4_0 | `test_gguf_dequant_ref.cu` `GgufDequant/AllScaleModes` (typed) | GEMV: `GgufRef.Q4_0_GemvDp4a`; M>1: dequant + cuBLAS (`GemmTest.FP16_*`) | none (no Q4_0 checkpoint in `test-e2e`, `verify.sh`) |
| Q4_1 | none (kernel `dequant_gpu.cu:207`) | none; registry test asserts NoMatch only (`test_gemm_kernel_registry.cu:1121-1125`) | none |
| Q5_0 | none (`dequant_gpu.cu:244`) | none | none |
| Q5_1 | none (`dequant_gpu.cu:286`) | MoE IMMA only: `MmqQ8Imma.MoeGroupedQ51` (`test_mmq_q8_imma.cu:361`) | none |
| Q8_0 | `GgufDequant` typed | `GgufRef.Q8_0_Gemv{Fp16,Dp4a,Mmvq}`, `MMVQ.Q8_0_*`, `GemmDP4ATest.Q8_0_Q8_1_Basic`, `MmqQ8Imma.*` | `test-e2e` (Qwen3-4B-Q8_0: `PrimaryModelTest`, `DetEvalE2ETest`), `verify.sh` smoke + perf (Qwen3-8B-Q8_0), greedy lock (D-3) |
| Q2_K | none (`dequant_gpu.cu:457`) | none | none |
| Q3_K | none (`dequant_gpu.cu:517`) | none | none |
| Q4_K | `GgufDequant` typed, `QuantIntegrationTest.Q4_KDequantCorrectness` | `GgufRef.Q4_K_Gemv{Dp4a,Mmvq}`, `MMVQ.Q4_K_*`, `MmqQ8Imma.Q4KDenseNRMSE`, `MmqQ4kImmaGemm.*`, `GemmQ4k{Fused,Dp4a}MoePrefill.*` | `test-e2e` (gemma-4-26B-A4B Q4_K_M: `Gemma4ModelTest`, `Gemma4GraphsTest`) |
| Q5_K | `GgufDequant` typed, `Q5_KDequantCorrectness` | `GgufRef.Q5_K_GemvDp4a`, `MMVQ.Q5_K_*`, `QuantIntegrationTest.Q5_KDp4aDenseGemm` | none |
| Q6_K | `GgufDequant` typed | `GgufRef.Q6_K_Gemv{Fp16,Dp4a}`, `GemmDP4ATest.Q6K_Q8_1_Basic`, `MmqQ8Imma.Q6KDenseNRMSE` | `verify-north-star` only (Qwen3-14B-Q6_K, opt-in) |
| Q8_K | none (`dequant_gpu.cu:141`) | none | none |
| IQ4_NL / IQ4_XS | `GgufDequant` typed | none (no GEMV traits; decode runs on the NVFP4 overlay or dequant) | none (Llama-3.2-3B-IQ4_XS appears only in `docs/determinism.md` probes) |
| MXFP4 (gpt-oss layout) | `GptOssMxfp4ConvertRef.*` (conversion), `Mxf4nvf4ProbeTest.*` (MMA sanity) | dispatch==direct only: `WeightDispatchTest.MXFP4_Gemm/GemvMatchesDirect`; attention: `AttentionMxFP4Test.CompareWithFP16Reference` | `test-e2e` (Qwen3.5-4B-mxfp4 `GDNModelTest`; gpt-oss-20b `DetEvalE2ETest`), `verify.sh` smoke (Qwen3.5-4B MXFP4) |
| NVFP4 Modelopt layout | `Nvfp4QuantRefTest.*`, `NVFP4OutlierTest.QuantDequantGemvVsIndependentRef` (numpy golden) | `GemmNvfp4Batched.MatchesDequantRef*`, `NvFP4SmallM{,V2}Test.*MatchesHostReference`, `BatchedSmallM.MatchesDequantisedReference`, `CutlassGroupedRefTest.*`, `CutlassNvfp4StreamKTest.MatchesDataParallel`, `CutlassGrouped3xNvfp4Test.DeviceArgsMatchesHostArgs` (bit-exact) | `test-e2e` (Qwen3-Coder-30B FP4, Nemotron-3-Nano NVFP4), `test-spec-fidelity` (Qwen3.8-27B), greedy lock NVFP4-cortecs (D-3) |
| NVFP4 compressed-tensors layout | `NvFP4CompressedTensorsRef.*` (4 tests), `NvFP4PromoteWeightScale2.*` | same kernels as above after promotion | `LlmCompressorE2E.*` (test-e2e binary; env defaults `/models/Mistral...`, `/models/Qwen3-Coder-30B-A3B-FP4`; not in the `Makefile:196` filter) |
| FP8 E4M3 weights | `QuantTest.FP8RoundTrip`, `FP8_E4M3_DecodeMatchesIndependentLUT` | `FP8GemmTest.GemvFP8NonzeroMatchesReference`, `RowscaleGemvFromQ8SourceMatchesReference`; M>1 = dequant path (`Fp8CacheMissRegistryDispatchMatchesDirectPath`, dispatch==direct) | none |
| FP8 E5M2 | none (loader rejects: `test_quantize_fp8_source.cpp:146-147`) | none | none |
| FP16 KV | n/a | `PagedOracle/PathF16` sweeps HD128/256, `PagedF16Multitok`, `PagedAttentionTest.*` | every e2e |
| FP8 KV | `FP8KVCache.QuantDequantRoundtrip` | `PagedOracle/PathFP8`, `FP8KVCache.PagedAttentionDecodeFP8vsFP16`, `SplitK*Consistency`, `FmhaFP8Test.*` (prefill) | none by default (`kv_cache.dtype=auto` picks it on some models; no gate pins it) |
| INT8 KV | `KVCacheWriteTest.Int8PerHeadScale` | `PagedOracle/PathINT8`, `PagedAttentionTest.INT8_*`, `INT8KVCache.SplitKConsistency` | none |
| INT4 KV | `INT4QuantTest.PackUnpackRoundtrip` | `PagedOracle/PathINT4`, `PagedAttentionTest.INT4_*`, `PagedAttentionINT4Test.DecodeSingleHead` | none (LIMITATIONS: empty output on gpt-oss) |
| NVFP4 KV (+TC, MXFP4_KV) | none separate | `PagedOracle/PathNVFP4, PathNVFP4TC, PathMXFP4KV`, `PagedNvfp4Multitok.MatchesReferenceBothRoutes`, residual tests; `PagedAttentionNvfp4TCTest.LaunchSucceeds_HD128` is launch-only | none (sparse decode / long-context arms opt-in) |

Current: LIMITATIONS says of Q4_1/Q5_0/Q5_1/Q2_K/Q3_K/Q8_K "dequant paths, no gate reads such a checkpoint". The dequant kernels exist (`src/quant/dequant_gpu.cu:207-317, 457-565, 141`) and are in the typed-test's reach (`GgufDequantFormats`, `test_gguf_dequant_ref.cu:22-25` lists 7 formats), but none of the six is in the list; a wrong nibble order or scale unpack in those kernels is caught by nothing, and `GgufSmallm` registry comments claim Q2_K/Q3_K/Q5_1 "covered" by a strategy (`test_gemm_kernel_registry.cu:1095`) while the numerics are not.
Expectation: llama.cpp's `test-quantize-fns` / `test-backend-ops` run a dequant + dot-product reference for every quant type it ships, on every PR. Shipping a decode path for a format with no numerical check is below that bar; refusing the format at load would also meet it.
Delta: six shipped GGUF formats and one KV write path (NVFP4 KV write is only checked through the oracle read) have no independent numerical reference at any level; two weight GEMM paths (MXFP4, FP8 prefill dequant) are pinned only against another in-tree path.
Cost: extend `GgufDequantFormats` with six tags + a CPU reference per format (~40 LOC each in `test_gguf_dequant_ref.cu`, following the IQ4 additions at `:211-244`); or refuse Q4_1/Q5_0/Q2_K/Q3_K/Q8_K at load with a clear error (~10 LOC in the GGUF loader). Risk: a reference written from the same misunderstanding passes; use ggml's reference layout comments as the source.
Falsifier: a CPU-reference test for one of the six exists under another name. Checked: y. `rg -l -w` for each token over `tests/*.cu tests/*.cpp` returns only `test_tensor_kind_table.cpp` (name mapping), `test_gemm_kernel_registry.cu` (NoMatch asserts) and, for Q5_1, `test_mmq_q8_imma.cu`.

### Checked and NOT a finding

- CUTLASS grouped NVFP4 GEMM is deterministic by construction: default `TileSchedulerArguments`, no split-K/stream-K, no atomics (`gemm_cutlass_grouped_3x.cu:318,635`; `rg atomic` = 0). F-17 stands.
- Stream-K on the dense CUTLASS prefill is opt-in (`gemm.nvfp4_cutlass_streamk`, `gemm_cutlass_sm120.cu:833-858`) and pinned against data-parallel (`CutlassNvfp4StreamKTest.MatchesDataParallel`).
- `gemm_moe_fused_tc.cu:102` `atomicAdd(d_tile_counter)` is work distribution; each output tile is accumulated by one CTA, so the result is order-independent.
- `atomicMax` on float bit patterns is exact and order-independent: `quantize_fp16_nvfp4_moe_native.cu:107`, `attention_fmha_sm120.cu:675`, `sampling_topk_topp.cu:397`, `mtp_forward.cu:224`, `nvfp4_quant.cu:218`, `warp_reduce.cuh:24-28`.
- `moe_imbalance_kernel`, `moe_expert_hist_kernel`, `moe_expert_trace_kernel` (`executor_forward_moe_batch.cu:60-90, 753-790`): diagnostics counters, null-checked off the hot path.
- `sparse_attn_select.cu:383-454`: integer `atomicAdd`/`atomicOr` on histograms and bitmaps (opt-in sparse decode); `sampling_penalties_history.cu:38,50`: integer atomics.
- `executor_perplexity.cu:45-46`: NLL reduced in fixed order, no global atomic, so `imp_perplexity` is bit-reproducible in default mode (the doc's chosen A/B instrument).
- `nvfp4_gemm_smallm.cu:231`, `nvfp4_gemm_smallm_v2.cu:20`: split-K reduce is a fixed two-kernel reduction, no atomics.
- `sampling_penalties.cu:554` FP `atomicAdd` for `logit_bias`: two entries naming the same token sum commutatively; only >=3 duplicate entries in one request could vary. Not documented; below finding threshold.
- `typical_p` shared-memory FP `atomicAdd` (`sampling_filters.cu:170-176`) and CUB top-k > 128: documented as known limits 3 and 2.
- `check_determinism_sites.py`: 6 reads in 4 files, matches `docs/determinism.md` (ran it). `check_launch_guards.py`: 437/437 guarded, allowlist empty (ran it). `check_log_fatal.py`: 2 FATAL sites, 1 abort, 1 allowlisted (ran it).
- Deterministic-mode E2E gate executes: `*DetEvalE2ETest*` with `IMP_TEST_MODEL` and `IMP_TEST_MOE_MODEL` in `make test-gpu` (`Makefile:136-140`) and `make test-e2e` (`:196`); asserts are bit-equality on greedy bytes and PPL (`test_determinism_e2e.cpp:144-216`).
- `src/core/cuda_raii.h`: `CudaStream`, `CudaEvent`, `CudaGraph`, `CudaGraphExec` all have `noexcept` move ctor and move assignment, destructors call the CUDA destroy without throwing (`:26-44, 72-92, 114-141, 155-182`). S-24 stands.
- Destructors after context loss: `Model::~Model` counts failed `cudaFreeAsync`, warns once, clears the sticky error (`src/model/model.cpp:23-58`); `KVCache::~KVCache` uses log-only frees (`kv_cache.cu:421-445`); `VRAMAllocator::~VRAMAllocator` frees nothing (`vram_allocator.cu:13-16`); `Engine::~Engine` drains a leaked sticky error so it cannot cross a model boundary (`engine.cpp:53-69`). No throwing destructor found.
- Throw while a capture is open: `CudaGraphRunner::execute` catches `std::exception`, calls `abort_capture`, marks capture failed, runs eager (`cuda_graph.cu:416-433`); `tests/test_capture_abort.cu` covers the throw-mid-capture wedge (#874) and the failed-first-replay stale-logits case. The 11 `throw` sites in `src/exec/*.cu` are all `std::runtime_error`/`std::invalid_argument`.
- C API boundary: every engine-touching entry point in `src/api/imp_api.cpp` (25 exports) is wrapped in `try { } catch (bad_alloc) / (std::exception) / (...)` mapping to `IMP_ERROR_OUT_OF_MEMORY` / `IMP_ERROR_INTERNAL` (`:180-219, 275-294, 346-407, 560-578, 591-633, 647-665, 680-697, 710-789, 809-839, 848-868, 883-980`).
- Server: global exception handler maps `nlohmann::json::exception` to 400 and everything else to a JSON 500 (`tools/imp-server/main.cpp:393-404`); a host throw in `engine->step()` cancels active requests with `internal_error`, invalidates graphs, keeps the worker alive, and stops it only on an unrecoverable CUDA class (`batching_engine.cpp:312-366`); `/health` reports `engine_faulted` (`utils.cpp:227-230`); chat handlers return 503 when the worker is not running (`handlers_chat.cpp:59-63`). The gap is only the no-throw path (D-1).
- Batch-neighbour leakage: property 1 of `ForwardPassTest.DecodeLogitsInvariantToBatchComposition` is bit-exact and mutation-validated per its comment (`test_forward_pass.cu:404-425`).
- Verify-chunk vs decode parity on `gemm_nvfp4_batched`: `Nvfp4VerifyRowParity.MultirowShapeIsBitIdenticalToDecodeWhenOn` plus a diverging control (`test_nvfp4_gemm_batched.cu:276-304`).
- cuBLASLt in deterministic mode: candidates probed in stable heuristic order, first survivor picked (`gemm.cu:480-500`); mid-run re-pick refused (`gemm.cu:315-332`). Cross-process stability of the default mode is measured (36/36 on Qwen3-30B-A3B-NVFP4, `docs/determinism.md`).
- `moe_fused_permute_kernel` shared-memory integer atomics (`moe_routing.cu:397,417`): affect only row order within an expert bucket; the deterministic variant exists for the FP32 scatter fallback (`moe_routing.cu:431-436`).
- `(void)cuda*` (33 sites): all in teardown, cleanup, pool-attribute reset, or capture abort (`cuda_graph.cu:337,352`, `host_pinned.cpp:93`, `nvfp4_gemm.cu:122`, `mem_account.cu:127-132`, `executor_forward_moe_batch.cu:590,595`, `attention_fmha_sm120.cu:1979`, `gemm_grouped_nvfp4_smallM.cu:721`, `attention_cublas.cu:357`, `attention_mxfp4_prefill.cu:350`, `graph_diag.h:75`, `model.cpp:58`, `engine_weight_upload.cpp:146`). Bare `cudaGetLastError();` clears in `engine_spec_capture.cpp:255-387` and `engine_workspace_warmup.cpp:59,556` follow a logged failure and a fallback.
- Bit-identical prefix-cache hit: correctly not asserted (`test_prefix_cache_e2e.cpp:139-145`), the token-equality gate and its limit are documented (#1314).
- GDN batched vs sequential bit-identity: `test_gdn.cu:1688` asserts it for the recurrent scan.

### Known-and-accepted (restated)

- No correctness gate against a reference implementation (#1571), no soak test (#1642); no GPU CI lane (SETTLED G F-5).
- Untested formats Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q8_K, FP8 E5M2 (LIMITATIONS "Untested code paths"); D-5 only sharpens "no gate" to "no kernel golden either".
- GGUF batched decode rows read the 4-bit overlay with 4-bit activations (#1897); batched and solo decode not bit-identical, batch invariance out of scope (LIMITATIONS:159, determinism.md); D-2 is about the missing number for the native path, not the design.
- Prefix-cache hit not bit-equal to fresh prefill (#1314); cross-context-in-process reproducibility (determinism limit 4, `DISABLED_` gates); CUTLASS grouped GEMM outside `runtime.deterministic` (limit 5); `typical_p` and CUB top-k tie stability (limits 2, 3); `--use_fast_math` build envelope (limit 6).
- Speculation (n-gram corpus, MTP head cache) makes a second pass differ; hybrid prefix caching on recurrent models (LIMITATIONS:252-290).
- Server streaming path never in the perf gate (#1685); calibrated KV scales in checkpoints not read.

### Open questions

- D-2 price: logprob agreement c=1 vs c=8 on Qwen3.8-27B-NVFP4 via server JSON (`prefix_cache=false`, `mtp_k=0`, `ngram=false`); needs the GPU.
- D-1 demo: inject a device fault into a running `imp-server` (a diag-gated illegal-address kernel, or `set_fail_next_replay_for_test`-style hook) and record `/health`, the next request's status, and its tokens.
- D-4 residual: log whether `moe_.routing_buffers.pool` is non-null on gpt-oss-20b and Qwen3-30B-A3B default runs (decides whether the FP32 atomic scatter is ever the default).
- D-3: do the 2026-06-04 greedy locks still pass on today's tree for Qwen3-8B-Q8_0 and Qwen3-8B-NVFP4-cortecs?
- `gemm.cu:326-332`: when `reselect_algo_for_entry` refuses in deterministic mode, which kernel serves the shape afterwards, and is it the same across processes?
- Whether the `PagedOracle` NVFP4 KV write path (quantize) has any check beyond the oracle's read-side reference; the write kernel test list shows FP16/INT8 only.


## Axis E: Serving surface (imp-server) - audit report 2026-09-05

HEAD ef664dd8, read-only, no GPU jobs, no build.

### Coverage

Read in full: `tools/imp-server/handlers_messages.cpp`, `stream_driver.cpp`, `stream_driver.h`, `stream_pipeline.h`, `batching_engine.cpp`, `batching_engine.h`, `main.cpp`, `rate_limit.cpp`, `metrics_memory.cpp`, `tool_stream_filter.h`, `fuzz/fuzz_tool_stream.cpp`, `tests/test_server_logprobs.py`, `tests/test_server_metrics.py`, `tests/api/test_concurrency.py` (1-80), `tools/imp-server/CLAUDE.md`, `.claude/skills/server-api/SKILL.md`, `docs/roadmap.md` "The 2026 bar", `docs/LIMITATIONS.md`, `docs/audit/SETTLED.md` sections C (head), E, F.
Sampled (line ranges): `handlers.cpp` (31-125, 428-560), `handlers_admin.cpp` (15-120), `handlers_misc.cpp` (1-345), `handlers.h` (60-175), `handlers_chat.cpp` (150-170, 318-340, function map), `handlers_chat_core.cpp` (160-182, 300-330, 550-600, 686-745, 806-830), `anthropic.cpp` (100-122, 290-330, 430-470, 510-580), `reasoning_split.h` (100-235), `tool_call.cpp` (290-330, 390-430, 640-670), `src/runtime/engine_scheduler.cpp` (800-880 grep, 1263-1300, 1500-1560, 1795-1835), `src/runtime/engine.cpp` (296-311, 450-522 grep), `src/runtime/engine_internal.h` (40-95), `src/runtime/constraint_manager.{h,cpp}` (55-140; 150-185, 270-290, 350-375), `src/memory/kv_cache_manager.cpp` (455-520), `src/runtime/engine_prefill.cpp` (218-240), `src/exec/executor_sampling.cu` (40-100 grep), `src/compute/sampling_penalties.cu` / `sampling_filters.cu` (grep), `src/lora/lora_adapter.cpp` (35-36), `docs/API.md` (84-120, 269-283, 370-405), `docs/DEPLOYMENT.md` (170-230), `docs/archive/structural_debt_2026_07_07.md` (18-27), `CHANGELOG.md` (110-130), `tests/api/test_messages.py` (167-200), `tests/api/test_tools.py` (157-215), `tests/test_tool_stream_filter.cpp` / `test_tool_call.cpp` (grep of case names).
Skipped: `handlers_responses.cpp` (only accounting call sites), `handlers_rerank.cpp`, `image_fetch.cpp`, `tracing.cpp` (path grep only), `utils.cpp` (depth guard only), `webui/`, `src/compute/json_schema.cpp`, `schema_constrain.cu`, `gbnf_grammar.cpp` (only the cache surface in `constraint_manager`).
External: cpp-httplib v0.53.0 README (WebFetch) for ThreadPool `mqr` semantics; the header fetch truncated before the implementation.

### Brief vs repo

| Brief / memory said | Repo | Evidence |
|---|---|---|
| `/v1/messages` streaming is synthetic | Real per-token stream over the shared driver | `handlers_messages.cpp:287` calls `run_stream_loop_`; `stream_driver.cpp:241` `pop_token` per token; `handlers_messages.cpp:56-62` `emit_delta` -> `sink.write` per token. SETTLED F already says so |
| `/v1/completions` has no latency histogram (memory note 2026-09-04) | Fixed in #1896 | `handlers_chat.cpp:275,277,443-450,516-518,583-590`; `CHANGELOG.md:116-121`; gate `tests/test_server_metrics.py:67-85` |
| "per-key rate limit" (brief, `docs/roadmap.md:98`) | Per peer IP, one shared API key | `rate_limit.cpp:13-29` key = `remote_addr` or first XFF hop from a trusted proxy; `main.cpp:283-291` single `state.api_key` |
| GBNF "mask build 333 -> 12 ms" | Roadmap row exists, no compile-cost number for JSON schema classification anywhere | `docs/roadmap.md:77`; `rg -i 'classif.*ms' CHANGELOG.md docs/` empty |
| `tests/api/test_concurrency.py:37` "drives 10" | True, and below `max_concurrent` 64, so the 429 path is never exercised | `args.h:35`, `test_concurrency.py:36,61` |
| SCAN hold ~85 ms | Fixed for plain chat in #1894; the 256-token hold on tool requests remains | `stream_driver.cpp:89-98`, `reasoning_split.h:121-128`, `CHANGELOG.md:124-128` |

### Findings

### [E-1] Per-request LoRA switch is prefix-cache-blind and unlocked against the decode worker
Axis: E   Sev: S1   Confidence: high
Evidence: `tools/imp-server/handlers_chat_core.cpp:686-710` (`imp_lora_set` runs after the `state.mtx` block closes at 171-182); `src/api/imp_api.cpp:287-296` (no lock); `src/runtime/engine.cpp:296-311` (`lora_set`: `executor_->set_lora`, `invalidate_graphs`, no prefix-cache clear); `src/memory/kv_cache_manager.cpp:466,504` (hash chain seeded only by `content_salt`); `src/runtime/engine_prefill.cpp:230,943` (salt = `vision_content_hash` only); `src/lora/lora_adapter.cpp:35-36` (K and V projections are LoRA targets); `docs/API.md:54` ("PEFT adapter hot-swap, works with every quant path"); `docs/roadmap.md:96` (2026 bar: "Per-request adapter selection"); `rg -i lora docs/LIMITATIONS.md docs/DESIGN_DECISIONS.md` empty.
Current: A request naming adapter B after one with adapter A (a) reuses KV blocks the prefix cache computed under A when the token prefix matches (the cache key is tokens + image hash, never the adapter), and (b) flips the engine-global adapter and drops decode graphs from an HTTP thread while the worker may be inside `engine->step()` for A. Code comment at 690-694 declares mixed-adapter concurrency out of scope; no user-facing doc says so.
Expectation: vLLM keys prefix-cache block hashes with the LoRA id as an extra key and carries the adapter per sequence; llama.cpp's server tags the slot cache by adapter set. A hot-swap advertised as per-request must at least not serve adapter-A KV to adapter B.
Delta: silent cross-adapter KV reuse whenever two adapters share a prompt prefix (system prompt) with the prefix cache on by default; plus a data race on the executor's adapter pointer and graph pool.
Cost: fold `active_lora_` into `content_salt` at both call sites (`engine_prefill.cpp:230,943`) and carry it on `Request` (~20 LOC); make the switch safe by pausing the batching engine or refusing the switch while `queue_depth()>0` (~30 LOC in `handlers_chat_core.cpp`); risk low; breaks nothing if wrong beyond a cache miss.
Falsifier: something clears `block_hash_to_id_` on adapter switch, or the server disables the prefix cache when `--lora` is given. Checked: y. `rg 'block_hash_to_id_\.clear|clear_prefix' src tools` empty; `main.cpp:141-156` loads adapters and touches no cache setting; `tests/test_lora.cpp` has no prefix/cache case.

### [E-2] Backpressure ends at the worker pool: overflow connections are queued unbounded, never 429
Axis: E   Sev: S2   Confidence: high
Evidence: `tools/imp-server/main.cpp:189-191` (`new httplib::ThreadPool(max_concurrent + 8)`, no `mqr`); cpp-httplib v0.53.0 README: "optional parameter to limit the maximum number of pending requests ... Default limit is 0 (unlimited). Once the limit is reached, the listener will shutdown the client connection"; `main.cpp:233-276` (the 429 check runs inside the pre-routing handler, i.e. on a worker thread); `main.cpp:261-268` reads `queue_depth()` while `submit` happens later (`handlers_chat_core.cpp:815`, `handlers_messages.cpp:503-510`) with no re-check; `batching_engine.cpp:112-119` (`pending_queue_` unbounded); `main.cpp:182-188` ("+8 covers health checks and admin routes").
Current: 72 workers at the default `max_concurrent=64`. Connections 73..N sit in httplib's job list with no response and no timer (read timeout starts only when a worker picks the socket). Within the pool, N requests that read `queue_depth < 64` concurrently are all admitted (check-then-submit race). The "+8" is not reserved: any request type consumes any worker, so 72 inference connections leave `/health` without a thread.
Expectation: TGI `--max-concurrent-requests` answers 429 without holding a worker; vLLM/SGLang run an async server with `--max-num-seqs` on the engine and reject or queue with a bound. At 10x intended concurrency a 2026 server returns 429/503 within milliseconds.
Delta: at 10x concurrency 9/10 connections hang instead of receiving the documented 429; a liveness probe can hang with them.
Cost: pass `mqr` (1 line; upstream semantics = connection shutdown, no HTTP status) or make admission atomic by counting in-flight handlers in pre-routing (~30 LOC in `main.cpp`/`handlers.h`); reserve workers for `/health` needs a second listener or a mutex-free fast path (~50 LOC). Risk low.
Falsifier: httplib's pool grows dynamically or bounds the queue by default. Checked: y against the upstream README (unlimited by default); `max_n`/dynamic scaling with a custom `ThreadPool(n)` not verified in-tree (header fetch truncated). `tests/api/test_concurrency.py:36,61` drives 10, never reaches the bound.

### [E-3] `top_logprobs` are computed from the sampler-mutated logits, and only the temperature-0 case is asserted
Axis: E   Sev: S2   Confidence: med
Evidence: `src/runtime/engine_scheduler.cpp:1529-1533` (`sample_per_request(logits_out)` then `decode_logits_out = logits_out`, same device view); `:1799-1830` (logprob pass reads that view back); `src/exec/executor_sampling.cu:44-45,84` (`apply_pre_sample(last_logits, ...)` on the same tensor); `:46-62` (penalties, DRY, logit_bias, `apply_constraint_mask`, min_p, typical_p all write `lp` in place); `src/compute/sampling_penalties.cu:61,82,140,149` and `sampling_filters.cu:51,201` (in-place writes, bans to -1e30 / -FLT_MAX); `src/runtime/engine_internal.h:51-56` (comment says "from raw logits"); `tests/test_server_logprobs.py:64-69,100-105` (temperature 0, no penalties, no constraint); `docs/roadmap.md:97` ("at temperature 0 the emitted token IS top_logprobs[0]").
Current: with repetition/frequency/presence penalties, `logit_bias`, banned tokens, a constraint mask, `min_p` or `typical_p`, the reported distribution is the post-filter one (masked tokens read as probability 0) and the code comment claims raw. Under constraints logprobs also force the eager path (`engine_scheduler.cpp:1512`, `imp_constrained_eager_fallback_total`). Under speculation logprobs never exist: `engine_spec_ngram.cpp:170` disables drafting when `req.logprobs` (known, LIMITATIONS).
Expectation: vLLM documents the choice (`--logprobs-mode raw_logprobs|processed_logprobs`, default raw); OpenAI returns model log-probs of the sampled distribution. Either is defensible; undocumented and untested is not.
Delta: semantics differ from the comment and from vLLM's default; no test pins which one imp has.
Cost: snapshot the row's logits before `apply_pre_sample` when `req->logprobs` (one `cudaMemcpyAsync` per logprob row, ~15 LOC in `executor_sampling.cu` + scheduler), or document "processed" in `docs/API.md` and extend `tests/test_server_logprobs.py` with a penalty case (~20 LOC). Risk low.
Falsifier: the sampler works on a scratch copy. Checked: y for the n==1 path (`executor_sampling.cu:44` slices the caller's tensor, no copy); the n>1 path comment at `engine_scheduler.cpp:1275-1284` mentions "its own scratch slot" for the sampled token, not the logits. HYPOTHESIS marker on the numerical size of the effect (no measurement).

### [E-4] Tool-enabled requests still pay a 256-token first-token hold when thinking is off
Axis: E   Sev: S2   Confidence: med
Evidence: `tools/imp-server/stream_driver.cpp:74-75` (SCAN whenever `use_reasoning && think_budget > 0`, default `--think-budget` 0.5), `:89-90` (`scan_limit = has_tools ? 256 : 8`), `:98` (`set_release_on_plain_text(!has_tools)`), `:450-453` (release only when the held text is a tool opener); `reasoning_split.h:121-128` (the 8-token hold measured "~85 ms on Qwen3.8-27B"); `CHANGELOG.md:124-128` (#1894: 97-105 -> 32-62 ms after releasing the 8-token hold; "the hold stays on tool requests"); `anthropic.cpp:385-397` (thinking opt-in on `/v1/messages`, so Claude Code traffic lands in SCAN).
Current: an agent client (tools on every request, thinking off) whose answer is prose sees no delta until 256 tokens or EOS. Existing number: 8 tokens = ~85 ms on Qwen3.8-27B; 256 tokens at the same rate is ~2.7 s (HYPOTHESIS: arithmetic, no measurement of the 256 case in the tree).
Expectation: vLLM's Qwen3 reasoning parser streams immediately and treats text before `</think>` as reasoning only when the opener was in the prompt; no engine holds hundreds of tokens for leak protection.
Delta: TTFT for prose replies of agent harnesses is dominated by the hold, not the engine; the harnesses (`make test-agents-external`) do not measure TTFT.
Cost: for a pre-closed think block (the case the comment at 79-88 describes) treat the stream as CONTENT and reclassify retroactively only on a `</think>` closer, or release on plain text and route a late `</think>` to `reasoning` as the offline path already does (`split_last_think`); ~40 LOC in `reasoning_split.h` + `stream_driver.cpp`; risk: a chain of thought leaks as content on a model that reasons without an opener (the #1894 trade), covered by `tests/test_stream_reasoning_split.cpp`.
Falsifier: the 256 hold ends earlier because the model's first tokens are a tool opener (true for tool calls, not for prose) or because `reasoning_format != "deepseek"`. Not measured; needs a GPU (open question).

### [E-5] No readiness signal: `/health` is 200 while suspended, model-less, and mid-swap
Axis: E   Sev: S3   Confidence: high
Evidence: `tools/imp-server/handlers.cpp:100-108` (status "ok" unless faulted/kv-floored; "Suspended is HEALTHY"); `main.cpp:121-131` (model-less start); `handlers.cpp:676` (`publish_model_status(false, "")` at teardown), `:72` + `handlers.h:35` (250 ms timed lock falls back to the snapshot during a swap); `handlers.cpp:448-456` (inference is 503 while suspended); `docs/DEPLOYMENT.md:177` ("liveness. Answers before a model is loaded"); `rg -n 'ready' main.cpp` empty.
Current: readiness is a JSON field (`model_loaded`, `suspended`), never an HTTP status. An orchestrator readiness probe keyed on status code routes traffic to a server that answers 503 on every inference route.
Expectation: Kubernetes readiness convention (non-2xx while not serving); vLLM's `/health` is 200 only with a live engine and does not listen before the model is up.
Delta: no route that returns non-2xx for "cannot take inference traffic now".
Cost: `GET /ready` returning 503 when `!model_loaded() || suspended || swap in progress` (~30 LOC in `handlers.cpp`/`main.cpp`, exempt from auth like `/health`), plus a DEPLOYMENT.md row. Risk none.
Falsifier: `/health` already returns non-200 in those states. Checked: y, it does not (only `engine_faulted`, `kv_pool_floored`, `docs/API.md:370-405`).

### [E-6] Shutdown has no reject-new phase and no drain deadline
Axis: E   Sev: S3   Confidence: med
Evidence: `tools/imp-server/handlers.cpp:31-35` (`signal_handler` only calls `svr->stop()`); `main.cpp:504-517` (batching engine stopped after `listen_after_bind` returns, i.e. after httplib joined its workers); `batching_engine.cpp:56-84` (`stop()` cancels everything left); `rg -i 'stop_grace|STOPSIGNAL|SIGTERM' Dockerfile docker-compose.yml docs/DEPLOYMENT.md docker-entrypoint.sh` empty; `args.h:54` (write timeout 600 s, a stream may legitimately run that long).
Current: in-flight streams finish only because cpp-httplib's `ThreadPool::shutdown()` drains and joins (upstream behaviour, not controlled in-tree); connections already in httplib's job queue are still served after SIGTERM; nothing answers 503 to new arrivals; `docker stop` sends SIGKILL after 10 s while a 300 s `--request-timeout` generation is running.
Expectation: stop accepting, 503 or `Connection: close` for new requests, drain with a deadline (vLLM/Triton style), and a `stop_grace_period` in the compose file matching it.
Delta: the drain contract exists only for `/admin/suspend` and model swap (`batching_engine.cpp:86-104` `pause()` with a timeout), not for process exit.
Cost: a `draining` atomic checked in pre-routing -> 503 (~15 LOC), `pause(drain_ms)` before `stop()` (~10 LOC), `stop_grace_period` in `docker-compose.yml`. Risk none.
Falsifier: httplib v0.53.0 `Server::stop()` already refuses queued sockets or imp installs a drain elsewhere. Not verified (header fetch truncated); `rg -n 'drain' main.cpp handlers.cpp` shows only the swap path.

### [E-7] Prometheus series carry no `model`, `endpoint` or `status` labels
Axis: E   Sev: S3   Confidence: high
Evidence: `tools/imp-server/metrics_memory.cpp:33,63` (the only labels: `tier`, `layer`); `handlers_misc.cpp:157-337` (every other series unlabeled); `imp_requests_total` incremented from chat (`handlers_chat_core.cpp:1162`), completions (`handlers_chat.cpp:443,583`), stream driver (`stream_driver.cpp:664`), embeddings (`handlers_misc.cpp:455,602`), rerank (`handlers_rerank.cpp:165`); `main.cpp:451-458` (4xx/5xx split only); `tests/api/test_contract.py:278-286` (asserts names only).
Current: one TTFT/ITL/duration ladder for all dialects and endpoints; no per-endpoint error rate; the model name is not on any series (`imp_model_loaded` is a bare gauge).
Expectation: vLLM labels every series with `model_name` and splits `vllm:request_success_total` by `finished_reason`; SGLang the same with `name`.
Delta: a dashboard cannot separate `/v1/messages` from `/v1/completions`, or a swapped model's numbers from its predecessor's.
Cost: `endpoint` label on the 4 histograms and 3 request counters (~60 LOC in `handlers.h`, `handlers_misc.cpp`, the 8 observe sites), `model` on `imp_model_loaded`; `monitoring/grafana/dashboards/imp.json` queries. Cardinality: 6 endpoints. Risk none.
Falsifier: labels exist on a series I did not read. Checked: y, `rg '\{[a-z_]+="' tools/imp-server/*.cpp` hits only `metrics_memory.cpp:33,63`.

### [E-8] The Anthropic thinking `signature` is an unkeyed hash and is never verified on input
Axis: E   Sev: S3   Confidence: high
Evidence: `tools/imp-server/anthropic.cpp:554-565` (FNV-1a of the text, `imp_sig_<hex>`); `:106-118` (a prior-turn thinking block is mapped to `reasoning_content`, the signature field is not read); `handlers_messages.cpp:154-161` (`signature_delta` emitted on every thinking block).
Current: the field exists so SDKs round-trip blocks (comment at 618-620); it certifies nothing, and an edited or fabricated thinking block is accepted as the model's own prior reasoning.
Expectation: Anthropic's API verifies the signature server-side; a local server may skip verification, but then the block should be documented as unsigned, or signed with a per-process key so tampered history is refused (400) rather than replayed.
Delta: documentation gap plus a cheap integrity check not taken.
Cost: HMAC-SHA256 with a random per-process key (~30 LOC, no new dependency if a hash exists in tree, else document). Risk: clients that trim thinking text would get 400 on the next turn; upstream has the same behaviour.
Falsifier: a doc line already says the signature is decorative. Checked: y, `rg -n signature docs/API.md` empty.

### [E-9] Roadmap "2026 bar" overstates two serving rows
Axis: E   Sev: S3   Confidence: high
Evidence: `docs/roadmap.md:98` ("per-key rate limit") vs `rate_limit.cpp:13-29` (per peer IP) and `main.cpp:283-291` (one key); `docs/roadmap.md:96` ("Per-request adapter selection") vs E-1; `docs/roadmap.md:97` ("logprobs that agree with what was emitted") vs E-3 (asserted at temperature 0 only, which the row does say).
Current: the table is the reader-facing claim of what is met; two rows describe more than the code does.
Expectation: expectation unclear beyond "the table matches the tree".
Delta: two rows.
Cost: 3 lines in `docs/roadmap.md` (+ `docs/DEPLOYMENT.md` if it repeats "per key"). Risk none.
Falsifier: a second rate limiter keyed by API key exists. Checked: y, `rg -n 'rate_limit_key|RateLimiter' tools/imp-server` shows one limiter, keyed at `main.cpp:248-250`.

### Checked and NOT a finding

- `/v1/messages` streaming is real and incremental: worker `engine->step()` -> staged events -> `notify_loop_` (`batching_engine.cpp:126-146,375-482`) -> `push_token` -> `pop_token` (`stream_driver.cpp:241`) -> `think_split.feed` (`:441`) -> `emit_content_token`/`emit_text` (`:550,161`) -> `AnthropicSSE::emit_delta` -> `sink.write` (`handlers_messages.cpp:56-62`). One `sink.write` per token piece; the only holds are the SCAN buffer (E-4), UTF-8 stitch, stop-sequence holdback (`stream_pipeline.h:77-112`) and the close-tag guard in argument streaming.
- The hooks an incremental Anthropic stream needs exist on `StreamDialect` (`stream_driver.h:27-62`): `emit_reasoning` -> `thinking_delta`, `on_call_begin/on_call_args_delta/on_call_end` -> `tool_use` + `input_json_delta` (streamed for JSON layouts, chunked at 48 bytes on codepoint boundaries for XML/Gemma layouts, `handlers_messages.cpp:262-285`), `keepalive` -> `ping` every ~10 s, `error` event on timeout/capacity (`:304-310`). `message_start` no longer waits for admission (`:98-111`, measured 118.5 -> 11.4 ms, #1558).
- Metrics inventory (all in `handlers_misc.cpp:148-337` unless noted; type/labels):

| series | type | labels | fed by |
|---|---|---|---|
| `imp_uptime_seconds` | gauge | - | scrape |
| `imp_requests_total`, `_failed_total` (5xx), `_rejected_total` (4xx), `_cancelled_total`, `_timed_out_total` | counter | - | all generation paths, embeddings, rerank; 4xx/5xx from post-routing `main.cpp:451-458` |
| `imp_tokens_prompt_total`, `_completion_total`, `_cached_total` | counter | - | every generation path |
| `imp_request_duration_seconds`, `imp_ttft_seconds`, `imp_queue_time_seconds` | histogram, bounds 0.005..10 s (11 buckets, `handlers.h:81-83`) | - | chat stream+non-stream, completions stream+non-stream, messages, responses (`finish_stream_accounting_`), unadmitted cancel/timeout for queue |
| `imp_inter_token_seconds` | histogram, bounds 0.5 ms..0.5 s (`handlers.h:86-87`) | - | per token, all four loops |
| `imp_queue_depth`, `imp_model_loaded`, `imp_last_ttft_ms`, `imp_last_request_duration_ms` | gauge | - | scrape |
| `imp_decode_batch_steps_total`, `_rows_total` (counter), `_max`, `_last_rows` (gauge) | - | - | `batching_engine.cpp:294-307` |
| `imp_kv_blocks_total`, `_used`, `_cached`, `_reclaimable`, `_pinned`, `_live`, `_reserved` | gauge | - | `metrics_memory.cpp:70-114` |
| `imp_kv_pressure_rejections_total`, `imp_kv_pool_growths_total`, `imp_streaming_kv_auto_enables_total`, `imp_prefix_cache_evictions_total` | counter | - | engine counters read at scrape |
| `imp_spec_drafted_total`, `_accepted_total`, `_verify_steps_total`, `_miss_steps_total` | counter | - | `engine->spec_stats()` |
| `imp_memory_reserved_bytes`, `imp_memory_live_bytes` | gauge | `tier` | `memory_tier_stats()` |
| `imp_moe_expert_imbalance`, `imp_moe_expert_peak_rows` | gauge | `layer` | executor |
| `imp_vram_budget_bytes`, `_own_bytes`, `_own_peak_bytes` | gauge | - | `memory_budget_stat()` |
| `imp_otlp_spans_exported_total`, `_export_failures_total`, `_unsampled_requests_total` | counter | - | tracer |
| `imp_constrained_eager_fallback_total`, `imp_model_loads_total` | counter | - | |

- Prefix-cache hit rate is derivable: `imp_tokens_cached_total / imp_tokens_prompt_total` plus `imp_kv_blocks_cached` and `imp_prefix_cache_evictions_total`; vLLM's query/hit pair is block-level, imp's is token-level, same ratio.
- OTLP exports traces only: `tracing.cpp:210` defaults the path to `/v1/traces`; no `resourceMetrics`/`resourceLogs` anywhere (`rg` empty). Matches the roadmap Closed row.
- `cache_control`: any marker on system/messages/tools -> `cache_prompt`; the last marked system/message block bounds the pin via `cache_prefix_messages` (`anthropic.cpp:294-330,515-527`), re-rendered to a token count in `handlers_chat_core.cpp:556-571`; pins are FIFO-budgeted (`kv_cache_manager.cpp:732-740`, `server.prefix_pin_budget_pct` 25); `ttl` accepted, not modeled (known). Image salt: `handlers_chat_core.cpp:318-321` folds image hashes in order; an image request without a hash is excluded from reuse (`engine_prefill.cpp:218-227`).
- Grammar compile caching exists: schema keyed by the exact schema string, tool grammars by `tool_call_key` (tool set + envelope + dialect), GBNF by source (`constraint_manager.cpp:161-163,277-285,355-371`); the engine pools up to 8 managers and prefers one that already classified the same schema/grammar (`engine.cpp:498-522`). No LRU and no key normalisation, but no number in the tree says the cold classify cost matters (open question).
- Tool-call parsing: streamed args are cut at the first `close_tag_` occurrence regardless of string state (`tool_stream_filter.h:124,253-258`), identical to the buffered path (chunking-invariant by design); a second `<tool_call>` opener inside arguments streams as bytes (BODY looks only for the close tag); the non-stream parser accepts a drifted second opener as the delimiter (`tool_call.cpp:319-328`, test `test_tool_call.cpp:317`); string-aware brace counting (`:396-415`, test `:712`); unparseable bodies restore verbatim (`tool_stream_filter.h:147-152`); truncated streamed call records what went out and finishes "length" (`stream_driver.cpp:612-625`). Fuzz target asserts one invariant only, whole-UTF-8 argument deltas (`fuzz_tool_stream.cpp:43-58`), runs in the CPU lane over a fixed corpus (`tests/test_fuzz_corpus.cpp:266`, 1500 executions).
- Structural-debt items #888 (admission bypass), #889 (`/health` lock), #892 (SSE-loop drift) are closed in code: `main.cpp:28-32`, `handlers.cpp:72` timed lock, one `run_stream_loop_`.
- Untrusted-input guards present: 100 MiB body cap (`main.cpp:161`), JSON nesting cap 100 (`utils.cpp:52-58`), detokenize array cap 1e6 (`handlers_misc.cpp:121`), XFF believed only from `--trusted-proxy` (`rate_limit.cpp:14`), limiter map swept (`:40-47`), constant-time key compare (`main.cpp:278-286`), header echo sanitised (`main.cpp:446-450`).
- Model swap and `/admin/suspend` drain in-flight work via `pause()` with a timeout and restore on failure (`handlers.cpp:498-540`, `handlers_admin.cpp:37-46`); new requests block on `state.mtx` for the load duration (by design, one model).
- Keep-alive: `keep_alive_max` 100 requests/connection set (`main.cpp:174`); no `set_keep_alive_timeout` call, so httplib's default idle wait applies (a fact, not a finding).

### Known-and-accepted (restated)

- No soak test; largest driven load 10 requests (`docs/LIMITATIONS.md`, #1642).
- Generation half of the HTTP contract runs only in `make test-server` (no GPU CI); server streaming path never in the perf gate (#1685).
- `/admin/suspend`, `/admin/resume`, `server.model_swap` implemented, ungated.
- One model resident; a swap is paid by the requesting call; WSL2 never returns peak VRAM across swaps.
- DNS rebinding on `--allow-remote-images`.
- Speculation off for logprobs/constraints/tools (`engine_spec_ngram.cpp:170`, LIMITATIONS "Speculation is off for most real requests").
- `cache_control.ttl` accepted, not modeled (skill server-api).
- JSON Schema unsupported assertion keywords are a 400; `pattern` that does not compile is unenforced at 200 (LIMITATIONS).
- OTLP traces only (roadmap Closed).
- `cudaGraphInstantiate` 10-44 ms per request after token 1-2 (memory note, CHANGELOG "cudaGraphExecUpdate" entry addresses it).

### Open questions

- E-4 cost: measure client TTFT for a prose answer on `/v1/messages` with tools present and thinking off on Qwen3.8-27B (needs a GPU); expected to sit near 256 tokens of decode.
- E-3 size: diff `top_logprobs` with and without `repetition_penalty=1.3` on a fixed prompt to show the post-filter effect (needs a GPU).
- Cold schema/grammar classification cost for a 151k vocab (no number in tree; `constraint_manager.cpp:169-172`).
- Anthropic SDK behaviour on a `tool_use` block closed with non-JSON `partial_json` after `max_tokens` (client-side, no imp code).
- httplib v0.53.0 `ThreadPool(n)` with `max_n=0`: fixed at n or dynamic to 4n (README describes the default pool; the custom one was not verified).
- Owner decision: is mixed-adapter serving in scope (E-1 fix) or should `--lora` with `--max-concurrent > 1` be refused at startup?


## Axis F1 - Security: model-file and config parsers as UNTRUSTED input

Repo: <repo>, branch perf/engine-h-fanin-cut-and-attention-split-verdict, HEAD ef664dd8. READ-ONLY, no build, no GPU job.

### Coverage

Read in full:
- `src/model/gguf_parse.cpp`, `src/model/gguf_loader_internal.h`, `src/model/gguf_loader.h`, `src/model/model_limits.h`
- `src/model/json_util.h`, `src/model/json_util.cpp`
- `src/model/sentencepiece_loader.cpp`
- `src/lora/lora_adapter.cpp`, `src/lora/lora_adapter.h`, `src/exec/executor_lora.cu`
- `src/memory/weight_cache_file.cpp`, `src/memory/weight_cache_file.h`
- `src/core/tensor.h`, `src/core/tensor.cpp` (ctor / numel / strides)
- `src/model/hf_hub.cpp`
- `fuzz/README.md`, `fuzz/fuzz_targets.h`, `fuzz/fuzz_safetensors.cpp`
- `src/model/CLAUDE.md`

Read the whole risk surface by targeted range (line ranges opened, not grepped):
- `src/model/gguf_loader.cpp` 1-500, 620-900, 1020-1135
- `src/model/safetensors_loader.cpp` 1-120, 300-620, 609-660 (shard index)
- `src/model/hf_config_loader.cpp` 100-350, 455-660
- `src/model/tokenizer.cpp` 798-1160, 1130-1160, 2240-2340
- `src/model/jinja.cpp` 855-880, 1640-1740, 2020-2060, 2276-2400, 2700-2780
- `src/vision/vision_loader.cpp` 226-740
- `src/memory/weight_snapshot.cpp` 1-140, 228-299
- `src/memory/recurrent_snapshot_store.cpp` 1-80 (+ full grep for file I/O: none)
- `tools/imp-server/handlers.cpp` 395-530
- `src/compute/embedding.cu` 34-90

Sampled: `tests/test_gguf_fault_injection.cpp` (test list + tail), `tests/test_checkpoint_limits.cpp` (test list + 299-360), `docs/audit/SETTLED.md` (1-140 plus sections C2/D/S-28), `docs/DEPLOYMENT.md`, `docs/LIMITATIONS.md`, `Dockerfile` 140-170, `src/runtime/config.h` 355-410.

Skipped (not file parsers, or checked only by grep for sizing patterns and found clean): `chat_template.cpp` / `chat_template_families.cpp` (family selection over an already-parsed string), `safetensors_writer.cpp` (writer), `llm_compressor_loader.cpp` (name translation; grepped for `resize`/`reserve`/`memcpy`, only a `string_view` slice at :144), `weight_upload.cu`, `mtp_head.h` (a struct plus a size estimator; the MTP head is loaded through `load_shard`, which is the hardened SafeTensors path), all compute kernels other than `embedding.cu` and `executor_lora.cu`.

### Brief vs repo

| Brief statement | Repo |
|---|---|
| "recurrent-state snapshot store" is a parser to audit | It is not. `src/memory/recurrent_snapshot_store.cpp` has zero file I/O (no `open`/`read`/`mmap`/`ifstream`); it is a VRAM + pinned-host buffer pool. No F1 surface. |
| "the MTP head loader" is a separate parser | It is not. `mtp.*` tensors are diverted inside `load_shard` (`safetensors_loader.cpp:429-434, 585`), so they inherit the SafeTensors bounds checks. `probe_mtp_head` only reads a header. |
| "SentencePiece proto if any" | It exists and is hand-rolled: `src/model/sentencepiece_loader.cpp`, a full protobuf wire reader, no fuzz target. |
| "unbounded allocation / crash on a hostile file is S2 for a local engine, S1 if a server endpoint can trigger the load with a user-supplied path" | The endpoint exists and is worse than "a path in the models dir": `hf_hub.cpp:36` returns **any existing path** verbatim, so an unauthenticated request body reaches `load_gguf` on an arbitrary absolute path. See F1-4. `docs/DEPLOYMENT.md:97` states the opposite contract. |
| GGUF, jinja, hf_config, vision, weight cache, LoRA have no fuzz target | Confirmed. `fuzz/` holds exactly 6 targets; `tests/test_fuzz_corpus.cpp:255-275` drives 6. Add sentencepiece to the uncovered list. |

### Findings

### [F1-1] GGUF `n_dims > 4` writes past a 4-element stack array before the check that would reject it
Axis: F1   Sev: S0   Confidence: high
Evidence:
- `src/model/gguf_parse.cpp:416-427` reads `info.n_dims = reader.read_u32()` and **deliberately tolerates** `n_dims > 4`: `for (uint32_t d = 4; d < info.n_dims; d++) reader.read_u64();` ("Skip extra dims if n_dims > 4 (shouldn't happen)"). No cap, no `fail()`.
- `src/model/gguf_parse.cpp:442` (`gguf_tensor_byte_size`) and `:463` (`gguf_tensor_in_bounds`) both loop `d < info.n_dims && d < 4`, so the bounds check never looks at `n_dims` and cannot reject the tensor.
- `src/model/gguf_loader.cpp:663-667`:
  ```
  int ndim = static_cast<int>(info.n_dims);
  int64_t shape[4] = {1, 1, 1, 1};
  for (int d = 0; d < ndim; d++) {
      shape[d] = info.dims[ndim - 1 - d];
  }
  ```
  `shape` is 4 elements; `info.dims` is `int64_t dims[4]` (`gguf_loader.h:88`).
- `src/core/tensor.cpp:11` has the guard `IMP_CHECK(ndim >= 0 && ndim <= kMaxDims, ...)` - but it runs at line 669, **after** the loop.
Current: a hostile GGUF sets `n_dims = 1000` and supplies 1000 `u64` dim words (the reader stays clean, so no truncation error fires). The tensor passes `gguf_tensor_in_bounds` because dims[0..3] are small. Then the loop performs an 8000-byte write onto a 32-byte stack buffer, with source bytes taken by an out-of-bounds read of `info.dims[999..4]` (heap, past the `GGUFTensorInfo` element). `n_dims = 5` is the minimal case: one 8-byte stack write past `shape`.
Expectation: llama.cpp caps at `GGML_MAX_DIMS = 4` and rejects the tensor at parse; imp's own SafeTensors path flattens >4 dims instead of writing past `shape[kMaxDims]` (`safetensors_loader.cpp:495-540`). `src/model/model_limits.h` states the rule imp itself follows elsewhere: a declared count is refused before it sizes anything.
Delta: `n_dims` is the one GGUF header field with no ceiling and no post-read validation, and the only one the fault-injection battery never patches (`tests/test_gguf_fault_injection.cpp:115` writes `n_dims = 2` and records no offset for it; grep for `n_dims` in that file returns exactly that one line).
Cost: 2 files. Either `reader.fail()` in `parse_tensor_infos` when `n_dims > 4`, or clamp `ndim = std::min(ndim, 4)` at `gguf_loader.cpp:663`. ~6 LOC plus one fault-injection test. Risk: none; no real checkpoint has >4 dims (the GGUF spec's own max is 4).
Falsifier: a cap on `n_dims` somewhere between the read and the loop. Checked y: `grep -n 'n_dims' src/model/*.cpp src/model/*.h` returns 8 sites; none compares against 4 except the two `d < 4` loop guards, which only bound their own iteration.

### [F1-2] The vision (mmproj) GGUF loader has no tensor bounds check at all and repeats the `n_dims` overflow
Axis: F1   Sev: S0   Confidence: high
Evidence:
- `src/vision/vision_loader.cpp:536`: `const void* tensor_data = data + tensor_data_start + info.offset;` where `info.offset = reader.read_u64()` (`:414`), raw from the file. `grep -n 'file_size\|in_bounds\|bounds\|limit\|remaining()' src/vision/vision_loader.cpp` returns only `mmap`/`munmap` sites - there is **no** equivalent of `gguf_tensor_in_bounds`.
- The wild pointer is then dereferenced: `vision_loader.cpp:294` `std::memcpy(h_src.data(), src, static_cast<size_t>(n) * sizeof(half));` (host-to-host OOB read) and `:250` `cudaMemcpy(d_ptr, src, fp16_bytes, cudaMemcpyHostToDevice)`.
- `vision_loader.cpp:532-534`: `for (uint32_t d = 0; d < info.n_dims; d++) n_elements *= info.dims[d];` - no `d < 4` cap, so `n_dims > 4` reads past `int64_t dims[4]` (`:395`) and the product has no saturating guard.
- `vision_loader.cpp:539-541`: `int64_t shape[4] = {1,1,1,1}; for (uint32_t d = 0; d < info.n_dims; d++) shape[d] = info.dims[info.n_dims - 1 - d];` - the same stack overflow as F1-1, here with no `d < 4` anywhere in the file.
- `vision_loader.cpp:401`: `tensor_infos.reserve(static_cast<size_t>(tensor_count));` - unclamped, where `gguf_loader.cpp:146` clamps the identical reserve with a comment explaining why.
Current: an mmproj with `offset = 2^63` and `dims = [2^20, 1]` produces a read of 2 MiB from `mmap_base + 2^63`, i.e. a segfault or a silent read of unrelated process memory that is then uploaded to VRAM as weights. `tensor_count = 2^60` OOMs before a byte is parsed.
Expectation: same file format, same threat model, same checks as `gguf_loader.cpp`. `src/model/CLAUDE.md`: "A checkpoint is untrusted input, including the numbers it states about itself."
Delta: `gguf_tensor_in_bounds` and the clamped reserves exist and are exported from `gguf_loader_internal.h`; the vision loader is a hand-copied fork of the same parse loop from before those checks were added, and never picked them up. Result: the one loader that takes a file the operator usually downloads separately (a community `mmproj-*.gguf`) is the least checked.
Cost: 1 file, ~25 LOC: set `data_base`/`data_limit` on each `TensorInfo`, call `gguf_tensor_in_bounds`, clamp the reserve, cap `n_dims`. Risk: a malformed-but-currently-loading mmproj would start being refused; that is the stated invariant.
Falsifier: a bounds check inside `upload_tensor_fp16` / `engine_arena().take_bytes`. Checked y: `upload_tensor_fp16` (`:226-273`) validates only the wire *type*; `take_bytes` sizes the destination, never the source.

### [F1-3] Warm weight cache: `data_alloc` / `data_off` from the file index a 16-element vector unchecked
Axis: F1   Sev: S1   Confidence: high
Evidence:
- `src/memory/weight_cache_file.cpp:279-284` validates `rh.key_len` (1..256) and `rh.n_allocs` (1..16) and nothing else from `RecordHeader`.
- `:294-297` copies `rh.data_alloc` (int32), `rh.data_off` (uint64), `rh.scales_alloc`, `rh.scales_off` into the record verbatim.
- `:287` `take(&rec.tensor, sizeof(Tensor))` reads a whole `Tensor` POD - `ndim`, `shape[4]`, `stride[4]`, `qtype`, both pointers - straight out of the file.
- `src/memory/weight_snapshot.cpp:266-268`:
  ```
  weight = rec.tensor;
  weight.data = static_cast<char*>(new_allocs[rec.data_alloc]) + rec.data_off;
  if (rec.scales_alloc >= 0)
      weight.scales = static_cast<char*>(new_allocs[rec.scales_alloc]) + rec.scales_off;
  ```
  `new_allocs.size() <= 16`. `rec.data_alloc` is unvalidated, including negative.
Current: a cache file with `data_alloc = 0x7fffffff` performs an out-of-bounds read on a `std::vector<void*>`, and the garbage it yields, plus an unbounded `data_off`, becomes the device pointer every kernel then reads for that weight. `weight = rec.tensor` also installs an attacker-chosen `ndim` (e.g. 99), which `Tensor::numel()` (`tensor.cpp:27-35`) then loops over past `shape[4]`. `weight_cache.enabled = true` by default (`src/runtime/config.h:401`).
Expectation: a self-written cache is still a file on disk that another process may have replaced. The record framing here is carefully bounds-checked (`take`, `bytes > end - cur`); the index fields inside the record are not, which is an inconsistency inside one function rather than a design position.
Delta: the trust boundary is real in two configurations. `default_warm_cache_dir()` (`weight_cache_file.cpp:66-72`) falls back to **`/tmp/imp-warm-cache`** when both `XDG_CACHE_HOME` and `HOME` are unset (a systemd unit without `HOME`, `env -i`), which is world-writable. The shipped container sets `HOME=/home/imp` (`Dockerfile:149-155`), so the container is not exposed.
Cost: 1 file, ~6 LOC: reject `data_alloc < 0 || >= n_allocs`, same for `scales_alloc`, and `data_off >= allocs[data_alloc].bytes`; plus clamp `rec.tensor.ndim` to `[0, kMaxDims]`. Risk: none.
Falsifier: a validation of `data_alloc` inside `builder_add_views` or before `try_restore`. Checked y: `grep -n 'data_alloc' src/memory/*.cpp *.h` gives 5 sites - the write side (`weight_snapshot.cpp:70`, via `locate()`, which is correct), the file read (`weight_cache_file.cpp:294`), and the deref (`weight_snapshot.cpp:266`). Nothing between.

### [F1-4] An unauthenticated request body can name an arbitrary filesystem path to load
Axis: F1   Sev: S1   Confidence: high
Evidence:
- `src/model/hf_hub.cpp:32-38`: `resolve_model_path` starts with `if (fs::exists(model_id)) { return model_id; }`. The `looks_like_hf_repo` filter at `:46-54` runs only *after* that early return.
- `tools/imp-server/handlers.cpp:409-425` `find_model_path`: exact-name match inside `models_dir` first, then `if (name.find('/') != std::string::npos) resolve_model_auto(name, fmt)`. An absolute path contains `/`.
- `resolve_model_auto` (`hf_hub.cpp:146-181`) accepts a regular file whose name ends in `.gguf`, or any directory holding `model.safetensors` / any `.gguf`.
- `handlers.cpp:445-530` `ensure_model_loaded` is driven by the `"model"` field of an ordinary chat/completions body; the swap branch is `server.model_swap`, `= true` by default (`src/runtime/config.h:361`).
- Auth is opt-in: `--api-key` (`tools/imp-server/args.cpp:43`); with no flag every route is open.
Current: `POST /v1/chat/completions {"model": "/tmp/evil.gguf", ...}` unloads the serving model and runs `load_gguf` on an attacker-planted file. It is also a filesystem-existence oracle (503 "not found" vs a load attempt).
Expectation: `docs/DEPLOYMENT.md:97` documents the contract as "a request naming another model **in the directory** swaps to it", and `handlers.cpp:440-441` states "the name must resolve inside the models directory (or as a HuggingFace repo id)". The code does neither for an existing path.
Delta: this is the reachability multiplier that turns F1-1 (and the GGUF DoS class generally) from "operator points at a bad file" into "remote unauthenticated input reaches a memory-unsafe parser". A `find_model_path` that resolves only inside `models_dir` restores the documented contract.
Cost: 1 file, ~8 LOC: require the resolved path to be inside `models_dir` (`fs::weakly_canonical` + prefix), or gate the HF-repo branch on a config flag. Risk: an operator relying on absolute-path model selection over HTTP breaks; that behaviour is undocumented today.
Falsifier: a path check in `load_model_into_state`. Checked y: `handlers.cpp:655` takes `const std::string& path` and calls the loader; no containment test in the function.

### [F1-5] GGUF sizes containers from `block_count` before the cap that exists to stop that, and indexes a shorter array with it
Axis: F1   Sev: S1   Confidence: high
Evidence:
- `src/model/gguf_loader.cpp:327` `cfg.n_layers = static_cast<int>(get_uint("block_count"));`
- `:437` `cfg.swa_layers.resize(cfg.n_layers, 0);`, `:460` `cfg.head_dim_per_layer.resize(cfg.n_layers);`, `:493` `cfg.swa_layers.resize(cfg.n_layers, 0);` - all inside the gemma4 / gpt-oss branches.
- `:626` `if (!validate_declared_dimensions(cfg, &dim_err))` - the `kMaxModelLayers = 1024` cap runs **199 lines later**.
- `:429-433`: when the file supplies `gemma4.attention.sliding_window_pattern`, `cfg.swa_layers` gets exactly `int_array.size()` entries, an arbitrary file-chosen length, and the default-fill at `:436-440` is then skipped.
- `:462` `cfg.head_dim_per_layer[i] = cfg.swa_layers[i] ? key_len_swa : key_len;` for `i < cfg.n_layers`, **unguarded**. Every other read of `swa_layers` in the tree is size-guarded (`gguf_loader.cpp:877`, `model_profile.cpp:76`, `executor_attention.cu:265,311`).
Current: `block_count = 1024` with `sliding_window_pattern = [1]` gives a 1023-byte heap over-read of a `std::vector<uint8_t>`, and the bytes read decide each layer's `head_dim` - a silent wrong geometry as well as an OOB read. `block_count = 2147483647` reaches `resize(2147483647)` (about 8.6 GiB for `head_dim_per_layer`) plus a 2-billion-iteration over-read before `validate_declared_dimensions` is ever called.
Expectation: `src/model/model_limits.h` states its own contract in the header comment: "Check the counts a checkpoint declares about itself, **before anything is sized from them**." `tests/test_checkpoint_limits.cpp:260` asserts a two-billion-layer config is refused - via the SafeTensors path, where the call at `safetensors_loader.cpp:1182` does precede the sizing.
Delta: the GGUF call site is in the wrong place, and the one per-layer array read that is not size-guarded sits inside the window.
Cost: 1 file. Move the `validate_declared_dimensions` block from `:620-631` to just after `:327`, and guard `:462` with `i < cfg.swa_layers.size()` like its five siblings. ~10 LOC. Risk: a GGUF that today loads with `n_layers > 1024` would be refused; none exists.
Falsifier: an earlier cap on `block_count` in `get_uint` or `apply_arch_defaults`. Checked y: `get_uint` (`:307-315`) is a plain map lookup; `apply_arch_defaults` is called at `:565`, still after the resizes.

### [F1-6] A LoRA adapter's declared dims are never checked against the base model, and reach the kernels as extents
Axis: F1   Sev: S1   Confidence: high
Evidence:
- `src/lora/lora_adapter.cpp:205` `int64_t numel = shape[0] * shape[1];` - `shape` entries come from `d.as_int()` (`:195`), a double narrowed to int64. No negative check, no saturating product. `src/model/safetensors_loader.cpp:466-486` has exactly that guard, added by #1605, with a comment naming the failure mode.
- `:216-222` stores `w.r = shape[0]`, `w.K = shape[1]`, `w.N = shape[0]` verbatim.
- `src/runtime/engine.cpp:288-294` `lora_load` calls `a->load(path, n_layers)` and validates nothing else.
- `src/exec/executor_lora.cu:92-93`: `lora_gemv_a_kernel<<<w.r, 256>>>(A, x, t, w.r, w.K)` and the kernel reads `x[k]` for `k < K` (`:35-36`) where `x` is the projection input sized by the model's `d_model`.
- `:109,116`: prefill builds `y_shape = {n, w.N}` over the real output buffer and calls `gemm(tt, bt, yt, ...)`, i.e. `w.N` is the **write** extent.
- `src/lora/lora_adapter.cpp:65` `std::vector<uint16_t> h(numel);` allocates before the `nbytes != numel * 2` consistency check at `:67`/`:71`/`:79`.
Current: an adapter declaring `[r=8, K=1e6]` for a `d_model=4096` model reads ~2 MB past the activation buffer on device; an adapter declaring `N` larger than the projection output writes past it. `numel = 2^40` with a 4-byte payload attempts a 2 TiB host allocation before the check that would have rejected the tensor.
Expectation: `src/model/CLAUDE.md` invariant 2: a checkpoint this build cannot serve is refused at load. PEFT loaders in vLLM/HF validate `in_features`/`out_features` against the base module.
Delta: layer index is range-checked (`:182-185`) but the shapes are not. The path is operator-supplied (`--lora NAME=PATH`, `args.cpp:24`; requests select by *name* only, `handlers_chat_params.cpp:516`), so the file, not the path, is the untrusted part - but the file is typically a third-party download.
Cost: 1 file, ~10 LOC: reject `shape[i] <= 0`, use the same saturating product as `safetensors_loader.cpp:472-480`, and compare `w.K` / `w.N` against `d_model` and the projection's output width at attach. Risk: an adapter with a benign shape quirk starts being refused.
Falsifier: a shape check inside `set_lora` or `lora_delta_`. Checked y: `executor_lora.cu:62-82` only sizes scratch from `max_rank()`; `lora_delta_` has no comparison against `cfg`.

### [F1-7] `tokenizer.ggml.bos_token_id` is unbounded and lands in the embedding gather as a row index
Axis: F1   Sev: S1   Confidence: med
Evidence:
- `src/model/gguf_loader.cpp:1085-1090`: `bos_id = static_cast<int>(val_uint(it_bos->second));` - no range check against `vocab_size` or `tokens.size()`.
- `:1092` `tokenizer->load_vocab(tokens, scores, bos_id, eos_id);`
- `src/runtime/engine.cpp:930-932`: `if (tok->add_bos() && ...) tokens.insert(tokens.begin(), static_cast<int32_t>(tok->bos_id()));` (`add_bos` is also GGUF metadata, `gguf_loader.cpp:1049`).
- `src/compute/embedding.cu:42-45` and `:67-71`: `const int row = token_ids[token]; ... table[static_cast<int64_t>(row) * d_model + i]` - no comparison against `vocab_size` in any of the four embedding kernels.
- `grep -rn 'vocab_size' src/runtime/engine_prefill.cpp` returns nothing; there is no token-id clamp between tokenizer and forward pass.
Current: `bos_token_id = 0x40000000` with `add_bos_token = true` yields a device read at `table + 2^30 * d_model * 2` bytes, i.e. a CUDA illegal memory access that poisons the context for the whole process (the failure mode `gguf_loader.cpp:288-292` was written to prevent for a different cause). `0xFFFFFFFF` yields `row = -1`, a small negative-offset read.
Expectation: llama.cpp validates special token ids against the vocab at load. The decode side of imp's own tokenizer already range-checks (`tokenizer.cpp:2248`, `:2283`); the load side of the special ids does not, which is the same asymmetry #1606 fixed for `tokenizer.json`.
Delta: one unchecked metadata scalar reaches a device pointer computation with no intermediate bound.
Cost: 1 file, ~4 LOC: drop `bos_id`/`eos_id` outside `[0, tokens.size())` at `gguf_loader.cpp:1085-1090`. Risk: none.
Falsifier: a clamp in `Tokenizer::load_vocab` or in the prefill submit path. Checked y for `bos_id()` (`tokenizer.cpp:2328` returns the field raw) and for the prefill grep above. Not checked on GPU (would need a run), hence med.

### [F1-8] SentencePiece proto reader: `data_ + len > end_` is bypassable by pointer overflow
Axis: F1   Sev: S2   Confidence: high
Evidence:
- `src/model/sentencepiece_loader.cpp:68-80`:
  ```
  bool read_length_delim(const uint8_t** out_data, size_t* out_size) {
      uint64_t len = 0;
      if (!read_varint(&len)) return false;
      if (data_ + len > end_) { fail("length-delim past end"); return false; }
  ```
  `len` is a full 64-bit varint. `data_ + len` for `len` near `2^64` wraps to `data_ - k`, which is `< end_`, so the guard passes.
- `:97-101` `skip_field(2)` uses the same helper, so the cursor is then set to `data_ - k`: the `while (!r.at_end())` loops at `:144`, `:216`, `:276` re-parse the same bytes forever.
- Contrast `src/model/gguf_loader_internal.h:35-38`, whose comment states the correct form for exactly this reason: `bool check(size_t n) const { return n <= size_ - pos_; }` "Guards against `pos_ + n` overflowing on attacker-controlled u64 lengths".
Current: a `tokenizer.model` with a length-delimited field of `len = 2^64 - 100` hangs the loader (unbounded loop), or, on the `piece` branch (`:161`), throws `std::length_error` out of `std::string::assign`. Signed/pointer arithmetic overflow is UB either way.
Expectation: the overflow-safe form already used two files away; upstream protobuf compares against remaining bytes, never a computed end pointer.
Delta: the only hand-rolled binary parser in the tree that does not use the subtraction form.
Cost: 1 file, 1 line: `if (len > static_cast<uint64_t>(end_ - data_))`. Risk: none.
Falsifier: `size_t` being 32-bit (then `data_ + len` truncation differs). Not applicable: `sm_120a`, x86-64 only.

### [F1-9] The Jinja evaluator has a parse-depth cap but no evaluation-recursion or iteration budget
Axis: F1   Sev: S2   Confidence: high
Evidence:
- `src/model/jinja.cpp:861-877`: `kMaxParseDepth = 256` plus a `DepthGuard`, both members of `Parser`.
- The `Evaluator` class (`:1651` onwards) has no depth member. `call_macro` (`:2278-2308`) calls `render_node` on the macro body; `render_node` (`:1709`) dispatches to `eval` (`:1864`); `eval_call` (`:2310-2316`) looks the name up in `macros_` and calls `call_macro`. `{% macro f() %}{{ f() }}{% endmacro %}{{ f() }}` recurses with no counter.
- `:2326-2344` `range`: `int64_t n = eval(...).as_int(); for (int64_t i = 0; i < n; i++) arr.push_back(Value(i));` - no cap, and `Value` carries a string plus two containers.
- `tests/test_checkpoint_limits.cpp:334-356` `DeepChatTemplateIsRejectedRatherThanOverflowingTheStack` tests `Template::parse` only - `{{ ((((...)))) }}` and nested `{% if %}` - never a render.
Current: a chat template in `tokenizer.chat_template` (GGUF metadata, `gguf_loader.cpp:1110-1113`) or `tokenizer_config.json` gives an unbounded C++ recursion (SIGSEGV, not an exception, so the `imp_api.cpp` translation at `:213-220` does not catch it) or an OOM from `{% for i in range(10000000000) %}`.
Expectation: Jinja2 and minja both bound loop iterations and call depth; the parse cap in this same file shows the author accepts the class of guard.
Delta: half the guard shipped. The parse side is capped, the eval side is not, and the test only exercises the capped half.
Cost: 1 file, ~15 LOC: an `Evaluator` depth counter mirroring `Parser::DepthGuard`, plus a cap on `range` length and on total loop iterations. Risk: a legitimate template hitting the cap; the caps can sit at 1e6 iterations / 256 frames.
Falsifier: a guard in `render_for` or a template-render timeout. Checked y: `grep -n 'depth\|kMax\|budget\|limit\|max_iter' src/model/jinja.cpp` returns only the parser cap at `:861` and `kMaxJsonDepth` at `:2702` (a `value_to_json` serialiser guard).

### [F1-10] `config.json` `head_dim` sizes two vectors before any cap, and the shared JSON parser amplifies ~52x
Axis: F1   Sev: S2   Confidence: high
Evidence:
- `src/model/hf_config_loader.cpp:312-322`: `int hd = cfg.head_dim > 0 ? cfg.head_dim : ...; int rd = (cfg.rope_dim > 0) ? cfg.rope_dim : hd; int pairs = rd / 2; ... cfg.rope_short_factor.resize(pairs); cfg.rope_long_factor.resize(pairs);`
- `src/model/model_limits.h:58-70` `validate_declared_dimensions` caps `n_layers` and `n_experts` only; `head_dim`, `d_model`, `d_ff`, `vocab_size` have no ceiling anywhere.
- `src/model/json_util.h:26-32`: `JValue` holds `std::string` + `double` + two `std::vector`s, i.e. ~104 bytes per node. A 128 MiB SafeTensors header (the cap at `safetensors_loader.cpp:kMaxHeaderBytes`) of `0,0,0,...` is 64M nodes, roughly 6.6 GiB resident.
- `src/model/json_util.cpp:190` `v.num_val = std::stod(num_str);` with no `try`. `{"a": -}` gives `num_str == "-"` and `std::invalid_argument`; `1e999` gives `std::out_of_range`.
Current: `"head_dim": 2147483647` in a `config.json` reaches two `resize` calls of ~1.07e9 floats each (about 8.6 GiB) at load. The `std::stod` throw escapes `parse_json_file` / `load_shard`; it is translated to `IMP_ERROR_INTERNAL` at `src/api/imp_api.cpp:215-220`, so it is a clean refusal rather than a crash.
Expectation: `model_limits.h`'s own reasoning ("Every number a loader uses to size a container comes out of the file ... so each one that reaches a `resize` needs a ceiling") applies to `head_dim` exactly as it does to `n_layers`.
Delta: the ceiling list stopped at two fields. The header-size cap bounds the JSON *text* but not the tree it expands into.
Cost: `model_limits.h` + `hf_config_loader.cpp`, ~15 LOC: add `kMaxHeadDim` / `kMaxModelDim` to `validate_declared_dimensions`, and a node budget or a `try/catch` around `std::stod`. Risk: low.
Falsifier: a cap on `head_dim` elsewhere. Checked y: `grep -rn 'head_dim' src/model/model_limits.h` returns nothing; `validate_declared_dimensions` tests two fields.

### [F1-11] Truncated GGUF shard: `size() - stensor_count` underflows and the shard's tensors keep a null data base
Axis: F1   Sev: S3   Confidence: high
Evidence:
- `src/model/gguf_loader.cpp:254` `parse_tensor_infos(sreader, stensor_count, tensor_infos);` - unlike the primary shard (`:149-153`), there is **no** `if (sreader.failed())` check after it.
- `:262` `size_t shard_tensor_start = tensor_infos.size() - static_cast<size_t>(stensor_count);` - when the shard is truncated, fewer than `stensor_count` entries were appended, so this wraps to a huge `size_t`.
- `:264-267` the loop `for (size_t ti = shard_tensor_start; ti < tensor_infos.size(); ti++)` is then skipped, leaving those entries with `data_base = nullptr, data_limit = 0` (`gguf_loader.h:91-92`).
Current: no memory error - the skipped loop means the wrapped index is never dereferenced, and `gguf_tensor_in_bounds` rejects every tensor with `data_limit = 0` and a nonzero offset. The observable effect is a shard whose tensors are all silently dropped, reported only as `skipped` in the `"Weights: %d assigned, %d skipped"` line (`:709`).
Expectation: the primary shard's `reader.failed()` check exists three times in the same function; the shard branch should have the fourth.
Delta: an unsigned wrap that is currently harmless by accident, and a missing truncation check on the shard path. It is filed at S3 because the consequence today is a bad diagnostic, not unsafety - but it is one refactor away from being dereferenced.
Cost: 1 file, ~5 LOC.
Falsifier: `parse_tensor_infos` appending exactly `tensor_count` entries even on failure. Checked y: `gguf_parse.cpp:413` `for (uint64_t i = 0; i < tensor_count && !reader.failed(); i++)` exits early.

### [F1-12] Fuzz coverage: 6 targets cover 2 of the 10 file-parsing surfaces, and the two covered ones are the hardened ones
Axis: F1   Sev: S3   Confidence: high
Evidence: `ls fuzz/` gives `fuzz_gbnf/json_schema/regex/safetensors/tokenizer_json/tool_stream`. `tests/test_fuzz_corpus.cpp:255-275` drives the same six. `docs/audit/SETTLED.md:315` (S-28) records that "fuzzed in CI" was wrong until #1620 and that `fuzz/` is now the real answer.

| parser | fuzz target | corpus lane | hardened? |
|---|---|---|---|
| SafeTensors header + offsets | yes (`fuzz_safetensors`) | yes | yes - #1603/#1604/#1605 guards at `safetensors_loader.cpp:335-382, 466-486, 566-580` |
| `tokenizer.json` | yes (`fuzz_tokenizer_json`) | yes | yes - #1606 guards at `tokenizer.cpp:849-874, 920-956` |
| GGUF (`gguf_parse.cpp`, `gguf_loader.cpp`) | **no** | no (hand-written battery, `tests/test_gguf_fault_injection.cpp`, 20 cases, no `n_dims` case) | partly - F1-1, F1-5, F1-7, F1-11 |
| vision mmproj GGUF | **no** | no (no fault-injection test exists) | **no** - F1-2 |
| `config.json` (`hf_config_loader.cpp`) | **no** | no | partly - F1-10 |
| Jinja chat template | **no** | parse-depth unit test only | partly - F1-9 |
| SentencePiece `tokenizer.model` | **no** | no | no - F1-8 |
| warm weight cache `.impwcache` | **no** | no | partly - F1-3 |
| LoRA `adapter_model.safetensors` | **no** | no | no - F1-6 |
| shared `JsonParser` | indirectly (via the two above) | yes | depth yes, node budget no - F1-10 |
Current: the correlation is exact. Every parser with a fuzz target has the saturating-product, offset-window and narrowing guards; every parser without one is missing at least one of them, and the two S0s are both in unfuzzed parsers.
Expectation: the surfaces are the same shape and one `fuzz_common.h` `TempFile` harness already exists; `fuzz_gguf.cpp` and `fuzz_sentencepiece.cpp` are each ~20 lines by analogy with `fuzz_safetensors.cpp:14-33`.
Delta: `fuzz/fuzz_targets.h` states the inclusion rule as "a parser reachable from a file or a request body that a user does not control", which selects all ten rows above; six were built.
Cost: `fuzz/` + `CMakeLists` + `tests/test_fuzz_corpus.cpp`, ~40 LOC per target plus a seed corpus. GGUF and sentencepiece are pure-CPU and fit the lane; the vision loader needs a GPU for the upload half, but its parse half up to `upload_tensor_fp16` does not.
Falsifier: an existing GGUF fuzz target outside `fuzz/`. Checked y: `grep -rn 'LLVMFuzzerTestOneInput\|imp_fuzz' --include=*.cpp` finds them only under `fuzz/` and `tests/test_fuzz_corpus.cpp`.

Also worth one line, below the finding bar: `src/runtime/config.h:402-406` documents `warm_cache.dir` empty as "next to the model (`<file>.impwcache`)", but `weight_cache_file.cpp:74-92` resolves empty to `default_warm_cache_dir()` (`$XDG_CACHE_HOME/imp/warm`, `$HOME/.cache/imp/warm`, `/tmp/imp-warm-cache`). Doc drift, S3.

### Checked and NOT a finding

- **SafeTensors header size**: `validate_header_size` (`safetensors_loader.cpp:335-357`) uses the overflow-safe `declared > file_size - 8` form and caps at 128 MiB, with the wrap explained in the comment. Clean.
- **SafeTensors tensor offsets**: `validate_tensor_offsets` (`:359-382`) checks swap, `tensor_data_offset > file_size`, `offset_end > file_size - tensor_data_offset`, and `offset_end - offset_start == expected_nbytes`. The shape product is saturating (`:466-486`) and rejects `nelem > INT64_MAX` because `Tensor::numel()` redoes it in int64. Clean.
- **SafeTensors dtype/width confusion**: one table (`kSafeTensorsDtypes`, `:404-...`) supplies both the validation width and the QType, and an unservable dtype is refused rather than re-typed (#1604). The type-confusion case the brief asks about is closed here.
- **SafeTensors shard names from `index.json`**: `safetensors_shard_name_is_safe` (`:619`) requires a bare filename; `tests/test_checkpoint_limits.cpp:185-259` covers traversal and absolute names.
- **`JsonParser` recursion**: `kMaxDepth = 512` with an RAII `DepthGuard` (`json_util.h:47-63`, `json_util.cpp:88-93`), measured against the 30k/40k SIGSEGV point in the header comment, tested at `test_checkpoint_limits.cpp:299`. Clean.
- **`BinaryReader`** (`gguf_loader_internal.h:25-103`): `check(n)` is the subtraction form, `read_string` fails rather than returning "" on a length past EOF, `read<T>` uses `memcpy` so there is **no unaligned `reinterpret_cast<uint64_t>`** anywhere in the GGUF path. The brief's item (d) is clean for GGUF; SafeTensors reads its one `uint64_t` header field the same way (`safetensors_loader.cpp:378-379`).
- **GGUF zero/unknown type -> division by zero**: `gguf_tensor_byte_size` returns `SIZE_MAX` when `bs <= 0 || ts == 0` (`gguf_parse.cpp:451-454`); tested (`test_gguf_fault_injection.cpp:444`).
- **GGUF `general.alignment == 0`**: guarded at `gguf_loader.cpp:160-161` and `vision_loader.cpp:430-431`, tested (`test_gguf_fault_injection.cpp:457`).
- **GGUF dim product overflow / negative dims**: saturating, returns `SIZE_MAX` (`gguf_parse.cpp:440-459`), tested twice (`TensorDimOverflow`, `TensorDimNegative`).
- **GGUF `kv_count` / `tensor_count` reserves**: clamped to `remaining()/12` and `remaining()/24` with the reasoning in the comments (`gguf_loader.cpp:126, 146`); tested (`HugeKvCount`, `HugeTensorCount`, `MaxU64TensorCount`).
- **GGUF unknown array element type**: `r.fail()` rather than a zero-consumption spin (`gguf_parse.cpp:355-361`), tested.
- **GGUF split shard filename derivation**: built with `snprintf("%.*s-%05d-of-%05d.gguf", dash_pos, base_path, ...)` from the operator's own path (`gguf_loader.cpp:196-199`), so `split.count` cannot inject a path. A huge `split_count` stops at the first missing shard (`:202-207`).
- **`tokenizer.json` vocab and added_tokens**: `kMaxTokenId = 4 Mi` (`tokenizer.h:16`) applied on both loops, with the #1606 negative-index write recorded in the comments (`tokenizer.cpp:838-848, 920-930`). `build_special_pieces` bounds both vectors (`:1133`). Decode range-checks (`:2248, :2283`).
- **GGUF tokenizer arrays**: `scores.resize(tokens.size())` normalises a mismatched scores array (`gguf_loader.cpp:1081`); `token_type` goes through `load_token_types`.
- **`recurrent_snapshot_store`**: no file I/O at all; not an F1 surface.
- **MTP head**: loaded through `load_shard`, inherits the SafeTensors checks.
- **`llm_compressor_loader.cpp`**: name translation only; no `resize`/`reserve` driven by file numbers.
- **Warm cache record framing**: `take()` bounds every read, `key_len` capped at 256, `n_allocs` at 16, `bytes > end - cur` rejected (`weight_cache_file.cpp:251-311`). Only the index fields inside the record are unchecked (F1-3).
- **Exception handling at the boundary**: `imp_model_load_ex` catches `bad_alloc` / `exception` / `...` and maps to `ImpError` (`imp_api.cpp:213-220`); the server has `set_exception_handler` (`main.cpp:393-403`). So every "uncaught exception on a hostile file" in this axis is a clean 500, not a crash - which is why F1-8/F1-10 are S2/S3 and not higher.

### Known-and-accepted (restated)

- No GPU CI lane (`vars.HAS_GPU_RUNNER` unset); the fuzz corpus lane and the fault-injection batteries are CPU-only, so nothing in F1-2, F1-6 or F1-7 could be caught by CI even if a test existed for the device half.
- `/admin/suspend`, `/admin/resume`, `server.model_swap`: implemented, ungated (`docs/LIMITATIONS.md:39`, `docs/FEATURES.md:81`, #1680). F1-4 sits inside that ungated surface.
- Single-author project, no security response process (`docs/LIMITATIONS.md:24`).
- DNS-rebinding on `--allow-remote-images` is already an accepted item; F1-4 is a different endpoint (model path, not image URL) and is not on that list.
- GGUF is in maintenance mode (`gguf_loader.cpp:4-15`). That is a reason to prefer the cheap guard (a `n_dims > 4` refusal) over a rewrite, not a reason to leave the stack write.

### Open questions

- Does `bos_token_id` out of range actually reach the embedding kernel, or does some path clamp it at submit time? Needs one GPU run with a patched GGUF (F1-7, currently med confidence from code reading only).
- Is the `/tmp/imp-warm-cache` fallback reachable in any shipped deployment unit, or only under `env -i`? The container sets `HOME`; a systemd unit file is not in the repo. Owner question (F1-3).
- Was the absolute-path branch in `find_model_path` intentional (an operator convenience) or an accident of `resolve_model_path`'s early return? The two comments at `handlers.cpp:440-441` and `docs/DEPLOYMENT.md:97` both describe the narrower behaviour, which suggests accident. Owner question (F1-4).
- `fuzz_safetensors` deliberately stops at the header scan because no `config.json` sits beside the file (`fuzz/README.md`). A `fuzz_gguf` would have no such limit - it would reach `assign_tensor` and the whole config path without a GPU. Worth confirming that the corpus lane's ~0.7 s budget still holds with a seventh target.


## Axis F2 - Security: HTTP layer, API keys, fuzzing and sanitizers in CI

Repo: <repo>, branch `perf/engine-h-fanin-cut-and-attention-split-verdict`, HEAD ef664dd8. READ-ONLY, no build, no GPU job.

### Coverage

Read in full:
- `tools/imp-server/main.cpp` (522 L), `args.cpp` (142), `args.h` (69), `rate_limit.cpp` (61), `rate_limit.h` (48), `handlers_admin.cpp` (176), `image_fetch.cpp` (258)
- `fuzz/README.md`, `fuzz/fuzz_targets.h`, `fuzz/` listing (6 targets + `fuzz_common.h`)
- `tools/sanitizers/lsan.supp`, `tools/sanitizers/ubsan.supp`
- `tools/imp-server/CLAUDE.md`, `AUDIT_BRIEF_common.md`
- `.github/workflows/ci.yml` L640-850 (`sanitizers`, `test`), L296-340 (`tidy`), L30-45 (schedule)

Read in part (targeted ranges, cited below):
- `tools/imp-server/utils.cpp` L40-110 (nesting guard), L240-290 (auth compare), L475-545 (base64)
- `tools/imp-server/handlers.cpp` L282-330, L409-557 (model resolve/swap), L559-580
- `tools/imp-server/handlers_chat_params.cpp` L55-180, L400-480, L490-520
- `tools/imp-server/handlers_chat.cpp` L620-740, L770-810
- `tools/imp-server/handlers_chat_core.cpp` L55-130, L195-345, L695-740
- `tools/imp-server/handlers_misc.cpp` L25-80, L155-185, L340-370
- `src/model/hf_hub.cpp` L1-183 (full), `src/vision/image_processor.cpp` L140-177, `src/vision/qwen3vl_pipeline.cpp` L160-240
- `docs/LIMITATIONS.md` (security entries L24, L87-91), `docs/DEPLOYMENT.md` L118-170, `docker-compose.yml` L10-25, `docker-entrypoint.sh` L35-52
- `Makefile` (asan/sanitize/tidy/verify targets), `tests/test_fuzz_corpus.cpp` L1-60, L250-278
- `docs/audit/SETTLED.md` section D (S-20..S-28)

Skipped: `handlers_messages.cpp`, `handlers_responses.cpp`, `anthropic.cpp`, `tool_call*.cpp`, `stream_driver.cpp`, `tracing.cpp` read only via grep (auth/logging/parse-site queries); `webui/index.html`; the pinned cpp-httplib source (fetched by CMake, not present on this host - `find / -name httplib.h` returned nothing, so no claim about library internals is made here).

### Brief vs repo

| Axis-question premise | Repo says |
|---|---|
| "`n`/`best_of`/`logprobs`/`top_logprobs` bounds ... unbounded?" | All bounded. `n` <= `--max-n` (default 8) `handlers_chat_params.cpp:125-130`; `best_of>1` refused `handlers_chat.cpp:655-665`; `n>1` refused on `/v1/completions` `:643-647`; `top_logprobs` clamped to 20 `:722-725` and `handlers_chat_params.cpp:178-181`. |
| "JSON nesting depth on the request body (nlohmann parse of a 100 MB body with 1e6 nesting)" | Already closed (#1607). `reject_body_too_deep()` `utils.cpp:52-65`, cap 100, non-recursive scanner `utils.cpp:67-104`, called at **all 9** `json::parse(req.body)` sites (`handlers_chat_params.cpp:59`, `handlers_chat.cpp:625,938`, `handlers_messages.cpp:418`, `handlers_misc.cpp:31,87,338`, `handlers_responses.cpp:351`, `handlers_rerank.cpp:77`). Unit-tested `tests/test_sse_stream_utils.cpp:672-719`. The comment records the measurement: 50 000 nested arrays parse, 100 000 segfault. |
| "constant-time comparison or `std::string ==`" | Constant-time, both header forms. `utils.cpp:245-255` (Bearer), `:259-268` (`x-api-key`), `:270-279`. Tested `tests/test_server_auth.cpp:20-76`. |
| "does any log line print the Authorization header, the key, or prompt/response text at default log level" | No. Grep for `Authorization`/`api_key` in `tools/imp-server/*.cpp` yields only the read sites (`main.cpp:284-286`) and the flag parse (`args.cpp:99`). Prompt text is logged only under the opt-in `--log-requests` JSONL (`handlers_chat_core.cpp:96-101`), whose `--help` line says "prompt + response content" (`args.cpp:64`). |
| "are keys in `/metrics` labels" | No labels at all - `/metrics` emits bare counters, no label sets (`handlers_misc.cpp:155-185`). |
| "`max_tokens` bound (clamped to context / hard cap / unbounded)" | Clamped to remaining context: `handlers_chat_core.cpp:713-716`. |
| "LoRA `lora` body field -> adapter path lookup: raw path? `..` traversal?" | Name only, looked up in a map populated at startup from `--lora NAME=PATH`. `handlers_chat_params.cpp:516`, `handlers_chat_core.cpp:698-699`, `handlers.h:170-173`. No path from the request. |
| "message count and total prompt bytes bound" | Message count bounded (10000, `handlers_chat_params.cpp:87-95`). Prompt **bytes** bounded only by the 100 MiB payload cap; `--max-input-tokens` is *tokens* and defaults to 0 = off (`args.h:38`). |
| "`Sanitizers` ... path-gated on which files (cite the regex), required check or not (not)" | Correct, not required. Regex at `ci.yml:698`. It does cover all six fuzz-target sources (`RegexNfa` lives in `src/compute/json_schema.{h,cpp}`, matched by the `src/compute/json_schema` alternative). |
| "S-28 ... `fuzz/` driven by `tests/test_fuzz_corpus.cpp`" | Accurate as written in `SETTLED.md:315`; `fuzz/README.md` adds the correction history. |

### Findings

### [F2-1] `body["model"]` is a filesystem path, not a name resolved inside `--models-dir`
Axis: F2   Sev: S1   Confidence: high
Evidence: `tools/imp-server/handlers.cpp:409-426` (`find_model_path`): basename match against `scan_model_files(state.models_dir)`, then `if (name.find('/') != npos) resolve_model_auto(name, fmt)`. `src/model/hf_hub.cpp:36-38`: `if (fs::exists(model_id)) return model_id;` - step 1 of `resolve_model_path`, before any HF-repo-id shape check. `hf_hub.cpp:152-159`: a regular file ending `.gguf` is accepted; `:161-166`: a directory holding `model.safetensors` is accepted. Caller: `handlers_chat_params.cpp:495` -> `handlers_chat_core.cpp:172` -> `ensure_model_loaded` -> `handlers.cpp:491-541` swap (`server.model_swap` default `true`, `src/runtime/config.h:361`).
Current: `{"model": "/any/readable/path/x.gguf"}` (or a directory with `model.safetensors`) resolves and is loaded, tearing down the resident model. `handlers.cpp:440-441` states the opposite: "the name must resolve inside the models directory (or as a HuggingFace repo id)".
Expectation: vLLM's `--served-model-name` maps request names to one preloaded model and never resolves a request string to a path; llama.cpp's `--model-alias` / llama-swap resolve against a configured set. A request field that reaches `fs::exists` is a path-injection surface even when the extension check narrows it.
Delta: no containment check of the resolved path against `state.models_dir`; the comment documents a guarantee the code does not implement.
Cost: ~15 LOC in `find_model_path` - `std::filesystem::weakly_canonical` on both sides plus a prefix compare, gated so the HF-cache branch still works. Blast radius: 1 file. Risk: a legitimate operator flow that names an absolute path outside `--models-dir` breaks; mitigate by allowing it only when `--models-dir` was never set.
Falsifier: a containment check elsewhere on the path, or `model_swap` off by default. Checked y: `config.h:361` is `true`; `rg -n "models_dir" tools/imp-server/handlers.cpp` shows `models_dir` used only by `scan_model_files`, never to validate the resolved path.

### [F2-2] No connection-level backpressure: both guards run after the request is fully read
Axis: F2   Sev: S1   Confidence: high
Evidence: worker pool `= --max-concurrent + 8` = 72 by default (`main.cpp:189-191`, `args.h:35`). Both admission guards live in `set_pre_routing_handler` (`main.cpp:233-294`), which httplib invokes only once the request line, headers and body have been read. `main.cpp:168-169` states the mechanism itself: "A slow reader holding a socket open costs a worker thread either way". `main.cpp:182-188` states the other half: "a streamed completion holds its worker for the whole generation". Only `set_read_timeout` (60 s), `set_write_timeout` (600 s), `set_keep_alive_max_count` (100) and `set_tcp_nodelay` are configured (`main.cpp:161-191`) - no per-IP connection cap, no idle/keep-alive timeout of imp's own.
Current: 72 sockets that dribble one byte per 59 s occupy every worker. `--rate-limit` (default 0 anyway) counts *completed* requests and never sees them. `/health` and `/metrics` short-circuit before the guards (`main.cpp:242-244`) but still need a worker thread to be dispatched at all.
Expectation: vLLM/TGI run on an async server (uvicorn/hyper) where an idle connection costs a socket, not a thread; llama.cpp's server has the same thread-per-connection shape and the same exposure. The 2026-era answer for a blocking server is a per-peer concurrent-connection cap plus a short header/idle timeout.
Delta: the connection layer has numbers (#1622 wrote them down) but no *count* limit; the count limit that exists is measured in requests, one layer too high.
Cost: two options. (a) `--max-connections-per-peer` enforced at accept: httplib exposes no accept hook, so it would have to be a socket-option/`set_pre_routing` hybrid - awkward. (b) Document that a reverse proxy owns connection limits and add `limit_conn` to the nginx snippet in `docs/DEPLOYMENT.md:158-170`: ~10 lines, zero code risk. (b) is the honest fix for a single-author project.
Falsifier: httplib v0.53 refusing connections above the pool size instead of queueing them, or a per-peer cap I missed. Checked partially - `rg -n "set_" tools/imp-server/main.cpp` lists every setter used and none caps connections; httplib's own source is not on this host (`find / -name httplib.h` -> empty), so the queueing behaviour is stated as unverified.

### [F2-3] The `--max-concurrent` guard blocks on `state.mtx` with no timeout
Axis: F2   Sev: S1   Confidence: high
Evidence: `main.cpp:261-276`: `if (state.max_concurrent > 0 && is_inference_endpoint(req.path)) { ... std::lock_guard<std::timed_mutex> lock(state.mtx); ... queue_depth(); }`. The same mutex is held for the whole of a model swap (`handlers.cpp:500-540`) and the whole of `/admin/suspend` (`handlers_admin.cpp:23`, snapshot + teardown + `cudaDeviceReset`, minutes on a 27B model). Every other observability path deliberately uses a timed acquire - `kObservabilityLockTimeout{250}` (`handlers.h:35`, used at `handlers.cpp:72,216,288,349`, `handlers_misc.cpp:291,320`).
Current: during a swap or a suspend, every arriving inference request parks a worker thread on a blocking `lock_guard` *inside the load-shedding guard*. 72 workers, then the pool is gone. The one place that must not block is the only one that does.
Expectation: an admission controller reads queue depth from an atomic or a `try_lock`; it never waits on the control-plane mutex.
Delta: `queue_depth()` needs a lock that the request path cannot afford. `state.batching` is the pointer being protected, not the depth.
Cost: `main.cpp:261-276` -> `std::unique_lock lock(state.mtx, kObservabilityLockTimeout)`; on `!owns_lock()` treat as "engine busy" and answer 503/429 rather than waiting. ~8 LOC, 1 file. Risk: a spurious 429 during a legitimate swap - which is the correct answer anyway.
Falsifier: `state.mtx` never held long, or `queue_depth()` reachable lock-free. Checked n: `handlers_admin.cpp:23` holds it across `imp_weights_snapshot_capture` + `imp_gpu_release`; `handlers.cpp:518` holds it across `load_model_into_state`.

### [F2-4] Image decode is unbounded before the resize, and the VL path has no image-count cap
Axis: F2   Sev: S2   Confidence: high
Evidence: `handlers_chat_params.cpp:421-432` appends one `images` slot per `image_url` part with no count check. `handlers_chat_core.cpp:326-333` (Qwen-VL) loops `for (const auto& bytes : ctx.params.images)` calling `preprocess_image_qwen`; the mmproj branch at `:336-338` caps at 1, the Qwen branch does not. Decode: `src/vision/qwen3vl_pipeline.cpp:181` and `src/vision/image_processor.cpp:164` call `stbi_load_from_memory` on the request bytes at full resolution. `src/vision/image_processor.cpp:1` is the single `STB_IMAGE_IMPLEMENTATION` and defines no limits - `rg -n "STBI_MAX_DIMENSIONS|STBI_NO_" src/vision CMakeLists.txt` returns only the implementation define, so the defaults apply: `third_party/stb/stb_image.h:796` `STBI_MAX_DIMENSIONS (1 << 24)`, product bounded only by `stbi__mad3sizes_valid` (`:1031`, `:3298`) i.e. `w*h*3 <= INT_MAX`. `qwen3vl_pipeline.cpp:234-239` shows the budget check is applied to *patches*, after `smart_resize`, i.e. after the full-resolution buffer already exists.
Current: a ~700 KB PNG of a 26000x26000 constant image decodes to ~2 GiB of host RGB; the 100 MiB body cap admits many such parts, decoded sequentially, and every one of the 72 workers can be doing it. Reachable only when a Qwen-VL tower is loaded (`handlers_chat_core.cpp:205-220` refuses images otherwise).
Expectation: every serving stack that accepts images caps pixels before decode - vLLM `--limit-mm-per-prompt` plus a max-pixel guard, TGI a max image size. stb ships `STBI_MAX_DIMENSIONS` for exactly this.
Delta: neither a per-request image count nor a pre-decode pixel bound exists.
Cost: `#define STBI_MAX_DIMENSIONS 16384` above `src/vision/image_processor.cpp:1` (1 line, rejects before allocation) plus a `--max-images-per-request` check next to `handlers_chat_params.cpp:426` (~10 LOC). Risk: a legitimate very large photo is refused - 16384 is far above any VL tower's useful input.
Falsifier: a size check between the base64 decode and `stbi_load_from_memory`, or a count check I missed. Checked y: `rg -n "kMaxImages|images.size\(\)" tools/imp-server/*.cpp` -> only `handlers_chat_core.cpp:211` (error text) and `:336` (mmproj `>1`).

### [F2-5] The per-request caps added by #1616/#1617 have no test in any lane
Axis: F2   Sev: S2   Confidence: high
Evidence: `rg -n "max_logit_bias|max_batch_items|max_n" tests/` returns nothing outside `tests/api/` noise; `rg` for the cap strings ("10000 entries", "above the server limit of") in `tests/` returns nothing. `tests/test_server_request_limits.cpp` covers only `RateLimiter` + `sanitize_for_echo` + envelope shape (26 TESTs, listed). The 100 MiB payload cap, the 10000-message cap (`handlers_chat_params.cpp:87-95`), the 16-stop cap (`:162-166`), `--max-n` (`:125-130`), `--max-logit-bias` (`handlers_chat.cpp:740-745`) and `--max-batch-items` (`handlers_rerank.cpp:106`, `handlers_misc.cpp:357`) are untested.
Current: six documented limits (`docs/DEPLOYMENT.md:143-150`) that nothing gates. `rate_limit.h:69-70` states the repo's own rule for exactly this: "A limit whose test cannot run is a limit that regresses silently" - and the extraction that made the rate limiter testable was not repeated for these.
Expectation: same as the rate limiter - the cap check belongs in a free function testable without `ServerState`.
Delta: the caps sit inline in handlers that pull in `BatchingEngine`, so the CPU lane cannot construct them; the `Real API contract (model-less)` job could reach `--max-n` (validation runs before `ensure_model_loaded`) but does not exercise it.
Cost: cheapest path is the model-less API job: ~6 pytest cases in `tests/api/` posting `n=1000`, 20000 messages, 5000 `logit_bias` entries, 1000 rerank documents, asserting 400. ~60 LOC, no C++ change, no GPU. Risk: none.
Falsifier: a test file I did not grep. Checked y across `tests/` and `tests/api/`.

### [F2-6] No continuous fuzzing; the CI driver is 1500 iterations at a fixed seed with no coverage feedback
Axis: F2   Sev: S2   Confidence: high
Evidence: `rg -n "IMP_FUZZERS" .github/` -> nothing; the option exists only at `CMakeLists.txt:79,1206-1208`. CPU-lane counts: `tests/test_fuzz_corpus.cpp:254-277` - 1500 iterations for json_schema/regex/gbnf/tool_stream, 250 for safetensors/tokenizer_json, fixed seeds. `fuzz/README.md` says it plainly: "No coverage feedback in the CPU lane ... a few thousand executions ... catches a re-introduction of a known defect class, not a new one." The libFuzzer row of its own table says "on demand".
Current: six targets exist and are correct; nothing runs them with coverage feedback, ever, unless a human types the docker line in the README.
Expectation: OSS-Fuzz or a nightly libFuzzer job is the standard for a parser surface fed by unauthenticated request bodies. The targets are already `LLVMFuzzerTestOneInput`-shaped (`fuzz/fuzz_targets.h:7-10`) i.e. OSS-Fuzz ready.
Delta: the expensive half of the work (targets, corpus, entry points) is done; the cheap half (a scheduled job) is missing.
Cost: one `ubuntu-24.04` job, `silkeh/clang:18`, `if: github.event_name == 'schedule'`, the README's exact cmake line, `-max_total_time=600` per target over the four CPU-only targets = ~40 min nightly, `actions/cache` keyed on the corpus dir for growth, `actions/upload-artifact` for crashers. ~45 lines of YAML, 0 LOC of C++. The workflow already has `schedule: cron "17 3 * * *"` (`ci.yml:37-38`) and the note at `:36` says a nightly "costs nothing while HAS_GPU_RUNNER is unset". Risk: nightly flake noise on an advisory job.
Falsifier: an existing nightly fuzz job. Checked y: `.github/workflows/` grep for `IMP_FUZZERS` and `fuzzer` -> no match.

### [F2-7] `compute-sanitizer` has never run in any automated lane
Axis: F2   Sev: S2   Confidence: high
Evidence: three routes, all dead. (1) CI `test` job `ci.yml:801` `if: vars.HAS_GPU_RUNNER == 'true'`, unset by owner decision (brief, established fact) - and its memcheck step is `continue-on-error: true` (`:824-836`) even if it ran. (2) `make sanitize` (`Makefile:478-487`) carries its own obituary at `:455-459`: "DOES NOT WORK ON WSL2 ... compute-sanitizer reports 'Error: Failed to initialize' (verified 2026-06-04)". (3) `scripts/verify.sh` - `rg -n "sanitiz|asan|valgrind" scripts/verify.sh` -> no match, so neither `make verify-fast` nor `make verify` runs it.
Current: no device-memory checker has run against this tree's kernels in any automated lane. `make asan` / the `Sanitizers` job cover **host** code only; `Makefile:459-460` says so explicitly ("nvcc-compiled device code is NOT sanitized").
Expectation: unclear for a single-GPU WSL2 project - the tool genuinely cannot run on this driver model. Naming it is the finding, not proposing a fix that the hardware forbids.
Delta: `docs/audit/SETTLED.md` and the roadmap do not carry this as an open item; the Makefile comment is the only place it is written down.
Cost: 0 to record (one row in `docs/LIMITATIONS.md`). Non-zero to fix - it needs a native-Linux GPU box, which is the same blocker as the GPU CI lane already on the roadmap.
Falsifier: compute-sanitizer working on WSL2 now. Not checked (would need a GPU job, out of scope for this audit).

### [F2-8] `Sanitizers` path gate covers the six fuzz targets and nothing else that eats request bytes
Axis: F2   Sev: S2   Confidence: med
Evidence: `ci.yml:698` regex: `^(fuzz/|tests/test_fuzz_corpus|src/compute/json_schema|src/compute/gbnf|src/model/(safetensors_loader|tokenizer|json_util)|tools/imp-server/tool_stream_filter|CMakeLists.txt|\.github/workflows/ci\.yml)`. Not matched, yet parsing untrusted bytes: `tools/imp-server/utils.cpp` (`base64_decode` `:505-522`, `json_nesting_depth` `:67-104`, `sanitize_for_echo`, `dump_safe`), `src/vision/image_processor.cpp` (the only `STB_IMAGE_IMPLEMENTATION`, decoding request bytes), `src/model/gguf_*`, `tools/imp-server/handlers_chat_params.cpp`.
Current: an edit to `base64_decode` or to the stb configuration ships without the ASan lane ever running. The job is advisory anyway (`Build` is the sole required context), so this is a "read it after a merge" gap, not a merge blocker.
Expectation: the gate should list the files the sanitizer would actually catch a defect in, not only the files that have a fuzz target.
Delta: the regex tracks target *sources*, not the untrusted-input *surface*.
Cost: 1 line of YAML - add `|tools/imp-server/utils|src/vision/image_|src/model/gguf_`. Risk: the CUDA-less sanitizer build runs on more PRs (job is ~build time of test-core + test-text, no GPU).
Falsifier: those files already covered transitively. Checked n - the gate is a diff-path grep, it has no notion of transitive inclusion.

### [F2-9] `/tokenize`, `/detokenize` and `/v1/messages/count_tokens` are outside `--max-concurrent` and outside `--max-input-tokens`
Axis: F2   Sev: S3   Confidence: high
Evidence: `main.cpp:28-32` `is_inference_endpoint()` omits `/tokenize`, `/detokenize`, `/v1/messages/count_tokens`. `handlers_misc.cpp:30-71` (`handle_tokenize`): after the depth guard it takes `std::lock_guard<std::timed_mutex>` (`:54`, blocking) and calls `imp_tokenize` on `body["content"]` with a 256k-token buffer (`:68-71`) - no `state.max_input_tokens` check anywhere in the function. `--rate-limit` covers the path (`main.cpp:40-44`, deliberately) but defaults to 0 = off (`args.h:37`).
Current: at defaults, an unauthenticated caller posts a ~100 MiB `content` and pays a full BPE merge walk on a worker thread, repeatedly, with no concurrency guard. `main.cpp:34-39` names exactly this ("Tokenisation walks the whole prompt through the BPE merge table on a server thread") and routes it to a limit that is off by default.
Expectation: unclear whether tokenize should count against `--max-concurrent` (it uses no GPU). But `--max-input-tokens`, which exists to bound prompt work, not consulting it is an inconsistency.
Delta: the flag named "max input tokens" is enforced on `/v1/chat/completions` (`handlers_chat.cpp:806`) and not on the endpoint whose entire cost is tokenizing the input.
Cost: ~10 LOC - a byte-length precheck on `content` against `max_input_tokens * 4` before `imp_tokenize`, in 3 handlers. Risk: a legitimate long-prompt token count is refused; only when the operator set the flag.
Falsifier: a cap inside `imp_tokenize`. Checked partly - `tok_cap` bounds the *output* vector (`handlers_misc.cpp:68`), not the input walk.

### Checked and NOT a finding

1. API-key compare is constant time on both header forms and over the full expected length; no early-out. `utils.cpp:245-255, 259-268`, tested `tests/test_server_auth.cpp:20-76`.
2. `Authorization` / `x-api-key` are never logged. Only read sites exist (`main.cpp:284-286`).
3. Prompt/response text is not logged at default level. The three `IMP_LOG_INFO` request lines carry counts only (`handlers_chat.cpp:772`, `handlers_chat_params.cpp:490`, `stream_driver.cpp:662`). Content goes only into the opt-in `--log-requests` JSONL (`handlers_chat_core.cpp:96-101`), documented at `args.cpp:64`.
4. `/metrics` carries no labels, so no key or model can leak through a label set (`handlers_misc.cpp:155-185`). It does disclose model name and token counts in metric values - fenced behind `--metrics-require-auth`, documented `args.h:24-28`, `main.cpp:238-242`.
5. JSON body nesting is bounded at 100 before any recursive parser, at all 9 body-parse sites, with a non-recursive scanner and 14 unit tests. See "Brief vs repo".
6. `base64_decode` (`utils.cpp:505-522`) has no out-of-bounds path: index table lookup, `continue` on invalid, `push_back` only. Output is bounded by the input length, itself bounded by the 100 MiB payload cap.
7. `logit_bias` map size capped at 1024 (`handlers_chat.cpp:740-745`, `--max-logit-bias`), rerank documents and embedding inputs at 512 (`handlers_rerank.cpp:106`, `handlers_misc.cpp:357`), message count at 10000, stop sequences at 16.
8. `max_tokens` is clamped to remaining context (`handlers_chat_core.cpp:713-716`); `--request-timeout` 300 s is enforced on all three response paths (`handlers_chat.cpp:230,485`, `stream_driver.cpp:218`).
9. LoRA selection is a name looked up in a startup map; no request-controlled path (`handlers_chat_params.cpp:516`, `handlers_chat_core.cpp:698-699`).
10. `GET /v1/models/(.+)` - the only path-parameter route - touches the filesystem only through `scan_model_files(state.models_dir)` and never through the parameter (`handlers.cpp:282-330`). No static-file route exists; the web UI is baked in at build time (`main.cpp:301-303`, `webui_asset.h`).
11. `image_url` with `file://` or a bare path leaves the slot empty and is refused, not read (`handlers_chat_params.cpp:450-452`). Only `data:` and `http(s)://` are handled.
12. Remote image fetching: off by default, scheme-checked, userinfo stripped (`image_fetch.cpp:174-180`), destination classified over **all** A/AAAA records incl. IPv4-mapped v6 and 169.254.169.254 (`:18-160`), redirects disabled (`:211`), body capped, error string invariant to the destination so the endpoint is not a port scanner (`handlers_chat_params.cpp:454-468`). Only the rebinding race remains, which is a known LIMITATIONS entry.
13. 404 echo is sanitised and truncated before it enters a JSON body (`main.cpp:413-435`, `sanitize_for_echo`), and `X-Request-Id` echo turns CR/LF into `.` - the header-injection guard (`main.cpp:446-450`). Tested `tests/test_server_request_limits.cpp:105-123`.
14. Every escaping exception becomes a JSON envelope, `json::exception` -> 400 (`main.cpp:393-404`); no bare 500 with an empty body.
15. `X-Forwarded-For` is believed only from `--trusted-proxy` peers, and the derived key is truncated to 64 bytes; the tracker is swept every 256 admissions so a client cannot grow it without bound (`rate_limit.cpp:13-56`, tested `test_server_request_limits.cpp:23-103`).
16. The `Sanitizers` gate does cover all six fuzz-target sources - `RegexNfa::compile` lives in `src/compute/json_schema.{h,cpp}` (`rg -l "RegexNfa::compile" src/`), matched by the `src/compute/json_schema` alternative. My first hypothesis (a missed `regex_nfa.cpp`) was wrong.
17. The `Sanitizers` job cannot go green empty: it asserts `FuzzCorpus` is present in the binary before running (`ci.yml:754-766`), a guard written after the first version silently contained no tests.
18. `lsan.supp`'s broad `leak:<unknown module>` is justified in place (`tools/sanitizers/lsan.supp:7-10`): an imp leak always carries a symbolized imp frame. `ubsan.supp` suppresses one vendored-stb alignment class only.
19. `--allow-remote-images` over `https://` is dead code in the shipped build: `CPPHTTPLIB_OPENSSL_SUPPORT` is never defined (`grep -rn CPPHTTPLIB` returns only `image_fetch.cpp:245`; `CMakeLists.txt:535-541` sets no OpenSSL option), so `image_fetch.cpp:249` always returns "needs an imp built with OpenSSL". Narrows the SSRF surface further rather than widening it.
20. CORS wide open (`main.cpp:234-236`) is deliberate and documented three times (`tools/imp-server/CLAUDE.md:63-64`, `docs/DEPLOYMENT.md:152-155`, with the nginx front). Not a finding.
21. `clang-tidy` is advisory, changed `.cpp` only, `.cu` out of scope (`ci.yml:298-337`, `Makefile:565-578`). No security-specific checks configured; consistent with its stated advisory role.

### Known-and-accepted (restated)

- No GPU CI lane (`vars.HAS_GPU_RUNNER` unset) - the `Test` job with the full ctest, compute-sanitizer and the perf gate has never run. Roadmap/brief.
- DNS rebinding on `--allow-remote-images`: check and connect are two resolutions; the fix needs a connect-time callback httplib does not expose. `docs/LIMITATIONS.md:87-91`.
- "No default credential, no default refusal": without `--api-key` every endpoint is open to whoever reaches the port, admin routes included. `docs/DEPLOYMENT.md:129-132`; compose publishes on 127.0.0.1 and widening is a two-part change (`docker-compose.yml:14-19`), while the container itself binds 0.0.0.0 (`docker-entrypoint.sh:37-45`).
- No TLS in imp; terminate at a reverse proxy. `docs/DEPLOYMENT.md:154-155`.
- Single-author project: "no security response process". `docs/LIMITATIONS.md:24`.
- `/metrics` unauthenticated by default so a stock Prometheus scrape works; `--metrics-require-auth` folds it back (#1207).
- S-28 already corrected: the four "fuzzed" batteries were property/fault-injection tests; `fuzz/` is the real thing, CPU-lane driven. `docs/audit/SETTLED.md:315`.

### Open questions

- Does cpp-httplib v0.53 refuse or queue connections beyond the thread-pool size? Needs the fetched source (absent on this host) or an experiment; it decides whether F2-2 is "72 stalled workers" or "72 stalled workers plus an unbounded accept queue".
- Owner decision: is F2-1 (arbitrary path in `body["model"]`) intended operator convenience or a defect? The code comment says containment; the code does not.
- Is a nightly libFuzzer job (F2-6) acceptable as an always-advisory lane, given that `Build` is the only required check and nothing else here is gating?
- `--max-images-per-request`: what number does the Qwen-VL patch budget actually make sensible? Needs `max_patches_` at the configured tower, i.e. a model load.


## Axis G - Code architecture: layering, interfaces, config, dead code

Tree: `perf/engine-h-fanin-cut-and-attention-split-verdict`, HEAD `ef664dd8`, clean. READ-ONLY, no build, no GPU.

### Coverage

**Read in full**: `docs/internals/ARCHITECTURE.md`, `docs/audit/ARCHMAP.md`, `docs/audit/SETTLED.md` (652 lines), `docs/audit/AUDIT_ARCH_2026_07_29.md` sections 11.1-11.3, `docs/audit/DEBT_LEDGER_2026_08_21.md` sections 1, 2(a)-(c), `CONTRIBUTING.md` "Other rules", `docs/internals/CPP23.md`, `tools/check_function_size.py` (parser), `src/core/qtype.h/.cpp` (dtype tables), `src/api/imp_api_suspend.cpp`, `src/api/imp_api_vision.cpp`, `src/runtime/pdl.h`, `src/compute/pdl_device.cuh`, `src/exec/nvfp4_expert_offload.h` (header), `src/exec/pre_dequant_phase1_fp16_cache.cu`.

**Read in part (named ranges)**: `src/api/imp_api.cpp:142-175,326-420,983-1038`, `src/exec/executor_attention_prefill.cu:295-445`, `src/exec/executor_attention_decode.cu:150-360`, `src/runtime/engine_init_resolver.cpp:183-235,780-845`, `src/runtime/engine_scheduler.cpp:1920-1990`, `src/runtime/vram_budget.cpp:740-762`, `src/runtime/engine.cpp:224-262`, `src/model/model_config.h:176-192`, `tools/imp-server/handlers.cpp:555-630`, `src/exec/executor.h:25-35,236-246,505-535`, `src/exec/executor_forward_moe_cutlass.cu:54-80,840-851`.

**Machine-swept (whole tree)**: include graph over 842 files / 551 TUs (`include_fanin.py`, `fanin_drop.py`, `cycles.py` in the scratchpad; the first was reviewed and needed no fix); `rg` sweeps for `virtual|override`, `throw`, `return false;`, `std::expected`, `std::optional`, `IMP_LOG_FATAL`, `[[nodiscard]]`, `getenv`, statement-level discarded bool calls; `tools/check_filesize.py`, `tools/check_function_size.py` (default, `--list`, `--stats`, `--selftest`).

**Delegated and then spot-verified by hand**: file structure of the 12 warn-level files; config-surface enumeration; reachability sweep of `ModelArch` / `QType` / `src/lora` / `src/vision` / `tools/` binaries. Every claim I reproduce below was re-opened at its `path:line` before being written down.

**Skipped**: `tests/` internals, `webui/`, `fuzz/` bodies, kernel numerics, everything with a perf claim attached (no GPU).

**Hand-verified control for the include tool**: `src/runtime/engine.h` reverse-BFS = 40 direct includers, one of which (`src/runtime/engine_internal.h`) is a header whose 9 includers are already in the set, so 39 rebuilt TUs. `grep -rl '#include "runtime/engine.h"'` returns the same 40 paths. This matches S-33's post-#1907 figure of 39 exactly.

---

### Brief vs repo

| Statement in my brief | Repo says |
|---|---|
| "the only backward include edges in `src/` are ... quant->compute 5 ... vision->runtime 2" | Correct, and I add the classification: 4 of the 5 quant->compute are `compute/pdl_device.cuh` (created 2026-08-31, `0af8f80d` #1833), and 1 of the 2 vision->runtime is `src/vision/vision_encoder.h:3 -> runtime/cuda_graph.h` (since 2026-04-24, `976551c0` #53), which is a **type** edge, not instrumentation |
| Brief lists `src/core/config/*.h` as a fan-in target | It exists (10 files) but every one has **exactly 1 direct includer** (`core/dispatch_policy.h`), so all 10 measure the same 109 TUs. Measuring them separately is meaningless; measure `dispatch_policy.h` |
| `SETTLED.md` F-10: "`src/exec/` includes `runtime/config.h` **zero** times" | **False today.** `src/exec/pre_dequant_phase1_fp16_cache.cu:20` includes it (added 2026-08-12, `57fca0d1` #1388, 8 days after F-10 closed), and the include is **unused** (finding G-9) |
| `ARCHMAP.md:44` (`api` bullet): "all entry points wrap try/catch -> `ImpError`; nothing throws across the ABI" | 4 of 23 `ImpError`-returning entry points have no try/catch (finding G-10) |
| `CONTRIBUTING.md:97`: "Errors return codes (`ImpError` / `bool`)" | Contradicted by `docs/internals/CPP23.md:53` ("internal code throws"), by root `CLAUDE.md` ("don't convert them to status returns") and by 75 `throw` sites in `src/`. Listed as a negative below, not a finding, because the CONTRIBUTING clause is about CUDA errors and reads as sloppy rather than wrong |
| `docs/audit/AUDIT_ARCH_2026_07_29.md` section 11.1 "**Cycles:** `core <-> compute` ... `compute <-> model` ... `exec <-> runtime`" | Understated. `core <-> compute` is gone (#1207), but the actual SCC over `src/` layers is **8 nodes wide** (finding G-1) |

---

### P0. Module map

### Intended direction (`docs/audit/ARCHMAP.md:9-16`, "Layer DAG", quoted verbatim)

```
api ──▶ runtime ──▶ exec ──▶ compute ──▶ quant
                 │        └──▶ memory ──▶ core
                 ├──▶ model ──▶ core
                 └──▶ memory ──▶ core
vision ──▶ compute/model        lora ──▶ runtime/model
```

`ARCHMAP.md:43` also records the 2026-08-02 revision: the `compute/quant/memory -> runtime` edges are "mostly instrumentation", `exec -> runtime` "is not", and the durable fix is "a small `DispatchPolicy` POD in `core/`".

### Actual direction

Counts are `#include "<layer>/..."` lines, rows = includer. `OK` = forward per the DAG above, `BACK` = against it. `core` is the sink and `api` the source; neither appears as a violator.

| includer \ includee | api | core | memory | quant | compute | exec | model | runtime | vision | lora |
|---|---|---|---|---|---|---|---|---|---|---|
| **api** | - | OK 4 | OK 4 | . | . | OK 1 | OK 5 | OK 3 | . | . |
| **core** | . | - | . | . | . | . | . | . | . | . |
| **memory** | . | OK 20 | - | . | . | . | **BACK 3** | **BACK 1** | . | . |
| **quant** | . | OK 24 | OK 1 | - | **BACK 5** | . | . | **BACK 9** | . | . |
| **compute** | . | OK 124 | OK 16 | OK 16 | - | . | **BACK 7** | **BACK 31** | . | . |
| **exec** | . | OK 113 | OK 42 | OK 94 | OK 233 | - | OK 10 | **BACK 20** | **BACK 1** | OK 3 |
| **model** | . | OK 26 | OK 7 | OK 3 | . | **BACK 1** | - | **BACK 4** | **BACK 4** | . |
| **runtime** | . | OK 46 | OK 46 | . | OK 46 | OK 12 | OK 21 | - | OK 5 | OK 2 |
| **vision** | . | OK 16 | OK 8 | . | OK 1 | . | OK 3 | **BACK 2** | - | . |
| **lora** | . | OK 2 | . | . | . | . | OK 1 | . | . | - |

**88 backward include lines.** `core` includes nothing outside `core` (the one `core -> compute` edge from the 07-29 table is gone, #1207). `api` is a pure source.

### Cycles

`cycles.py` (Tarjan over the resolved include graph):

- **File-level SCCs: zero.** There is no real `#include` cycle anywhere in the tree.
- **Layer-level SCCs: one, of size 8**: `{compute, exec, lora, memory, model, quant, runtime, vision}`. Only `core` and `api` sit outside it.

Simulated repairs (same tool, edges relabelled/deleted):

| Scenario | SCC after |
|---|---|
| Move `pdl.h`, `process_diag.h`, `graph_diag.h`, `pdl_device.cuh` to `core/` (kills 64 of 88 backward lines) | unchanged, 8 nodes |
| the above + `storage_planner.h`/`vram_budget.h` to `exec/` + drop the vestigial `config.h` include (`exec -> runtime` = 0) | unchanged, 8 nodes |
| the above + `nvfp4_expert_offload.h` to `core/` (`model -> exec` = 0) | unchanged, 8 nodes |

The residual cycle-forming edges after all of that are `vision <-> runtime` (1 line), `memory <-> model` (3 vs 7), `model <-> vision` (4 vs 3) and `compute -> model -> quant -> compute` (7 / 3 / 16). Those are ordering decisions, not misplaced files.

### Classification of every backward edge

| Edge | n | Header(s) | Class | Note |
|---|---|---|---|---|
| compute -> runtime | 31 | `pdl.h` 16, `process_diag.h` 15 | **instrumentation** | both headers depend on nothing but `<cuda_runtime.h>` / `<string>` |
| exec -> runtime | 20 | `pdl.h` 12, `storage_planner.h` 4, `process_diag.h` 2, `vram_budget.h` 1, `config.h` 1 | 14 instrumentation, 5 **type leakage**, 1 **vestigial** | `config.h` at `pre_dequant_phase1_fp16_cache.cu:20` uses no symbol from it (G-9) |
| quant -> runtime | 9 | `pdl.h` 6, `process_diag.h` 3 | **instrumentation** | |
| quant -> compute | 5 | `pdl_device.cuh` 4, `gemm.h` 1 | 4 **type leakage** (`pdl_device.cuh` is 35 lines with **zero** `#include`s: a pure device inline helper, i.e. a `core/` thing), 1 **algorithmic** | the `gemm.h` one is `nvfp4_gemm.cu:3`, the NVFP4 dequant-to-FP16 fallback; SETTLED says leave it, and I agree |
| model -> runtime | 4 | `process_diag.h` 4 | **instrumentation** | |
| vision -> runtime | 2 | `cuda_graph.h` 1, `process_diag.h` 1 | 1 **type leakage**, 1 instrumentation | `vision_encoder.h:3` holds a `CudaGraph`; oldest backward edge in the tree (#53) |
| memory -> runtime | 1 | `graph_diag.h` (`kv_cache.cu:5`) | **instrumentation** | |
| compute -> model | 7 | `tokenizer.h` 4, `model.h` 1, `mtp_head.h` 1, `model_config.h` 1 | 6 **algorithmic** (settled), 1 **type leakage** | `gemm_cutlass_sm120.h:3` pulls the 800-LOC, 156-TU `model_config.h` for the 4-value enum `FFNActivation` (`model_config.h:14`). Identical shape to the `QType` case SETTLED records as fixed 2026-08-03 |
| model -> vision | 4 | `vision_model.h`, `qwen3vl_vision_config.h`, `qwen3vl_vision_load.h` | **algorithmic** | multimodal checkpoints carry a tower |
| memory -> model | 3 | `model/model.h` | **algorithmic** | these three genuinely operate on a `Model` |
| model -> exec | 1 | `exec/nvfp4_expert_offload.h` (`weight_upload.cu:6`) | **type leakage** | 193-line header whose only project include is `core/tensor.h`: a slot layout + predicates POD. Exactly the `WeightHandle` case moved to `core/` on 2026-08-03. Added 2026-08-13 `8a7bd8ca` #1407 |
| exec -> vision | 1 | `deepstack_inject.h` | **algorithmic** | inherent to the feature |

**Instrumentation total: 64 of 88 (73 %)** across four headers, none of which has a project dependency.

### Comparison against the 07-29 table (section 11.1) and SETTLED

| Edge | 07-29 | today | verdict |
|---|---|---|---|
| `exec -> runtime` | 27, incl. 22x `config.h` | 20, incl. 1x `config.h` | F-10 closed the 22; **one came back** (G-9) |
| `compute -> runtime` | 21, incl. 2x `config.h` | 31, 0x `config.h` | algorithmic half closed (#1227); instrumentation half grew +10 |
| `compute -> model` | 9 | 7 | 2 closed 2026-08-03 (the `QType` includes); 1 of the 7 is the same shape and still open |
| `core -> compute` | 1 | **0** | closed #1207 |
| `compute -> exec` | not listed | **0** | closed 2026-08-03 (`WeightHandle` -> `core/`) |
| `memory -> model` / `memory -> runtime` / `exec -> vision` | 3 / 1 / 1 | 3 / 1 / 1 | unchanged |
| `model -> vision` | 3 | 4 | +1 (`weight_map.cpp:2`) |
| `quant -> runtime` | 4 | 9 | +5 |
| `quant -> compute` | not listed | 5 | 1 was in SETTLED; **4 are new** (#1833, 2026-08-31) |
| `model -> exec` | not listed | 1 | **new** (#1407, 2026-08-13), never recorded anywhere |
| `vision -> runtime` | not listed | 2 | present since #53, never recorded |

The 07-29 table's own claim that it is "otherwise complete" is now false in three places (`quant -> compute`, `model -> exec`, `vision -> runtime`).

### `tools/` layer

| includer | api | core | memory | quant | compute | exec | model | runtime | vision | public `imp/*.h` |
|---|---|---|---|---|---|---|---|---|---|---|
| `tools/imp-server` | 8 | 6 | 8 | . | 3 | 1 | 11 | **28** | 6 | **2** |
| `tools/imp-cli` | 2 | . | 2 | . | . | . | 10 | 8 | . | **0** |
| `tools/imp-bench` | . | 5 | . | 2 | 8 | 1 | 1 | 2 | . | 0 |
| `tools/imp-quantize` | . | 5 | 1 | 6 | . | . | 7 | . | . | 0 |
| `tools/common` | . | 1 | . | . | . | . | . | 1 | . | 0 |

The only files that include `imp/imp.h` are `src/api/*`, `tools/imp-server/{handlers.h,utils.cpp}` and **17 test files**. See G-3.

### File-size warn ridge: conflation vs cohesion

Judged by top-level structure, not by size. `tools/check_filesize.py` today: `violations=0`, `warn=44`, `allowlisted=34`.

| File | code/raw | Top-level shape | Verdict |
|---|---|---|---|
| `src/runtime/engine_spec_ngram.cpp` | 790/1262 | 11 defs, all `Engine::spec_*`; `step_spec_verify_` is 972 raw lines and is allowlisted at 625 code LOC with a stated reason | **cohesive** |
| `src/model/chat_template.cpp` | 786/998 | 20 defs, all `ChatTemplate::*` + 4 file-statics; largest `init` 252 | **cohesive** |
| `tools/imp-server/handlers_chat.cpp` | 781/1003 | `handle_chat_completions` (35), `handle_completions` (623) + its two response builders in an anon ns (80-621), `handle_count_tokens` (929) | **CONFLATED**: three endpoints, two API dialects. Split line: 929 (Anthropic `count_tokens`) is a clean cut |
| `tools/imp-server/handlers_chat_core.cpp` | 780/1217 | `log_request_jsonl` (56), `collect_tool_enforcement_` (115), `snapshot_state_and_tokenize_` (162, 323 code LOC), `build_imp_request_` (726), `nonstream_chat_response_` (796, 287 code LOC) | **CONFLATED**; the file's own header comment names the three concerns |
| `src/api/imp_api.cpp` | 764/1038 | 34 defs, one per public C entry point, uniform validate -> call -> translate shape | **cohesive** (it is the ABI boundary by design) |
| `tools/imp-server/utils.cpp` | 730/914 | 41 free functions at global scope: UTF-8 sanitisation, dialect error responses, bearer/API-key auth, logprobs JSON, base64, think/Harmony channel parsing | **CONFLATED**, six unrelated concerns; the auth block (245-302) is the one with a security reason to live alone |
| `src/runtime/engine.h` | 629/1502 | one `class Engine` with ~150 member functions, ~150 data members, 10 nested structs, 27 section comments | **CONFLATED** by any normal reading, but **do not act on it**: F-24 refuted the extraction on churn and priced a full pimpl at a 42 % ceiling; S-33 took the cheap fan-in half instead |
| `src/runtime/engine_kv_cache_init.cpp` | 721/1064 | 2 defs; `init_kv_cache` is 983 raw lines, allowlisted at 658 with the #1103 ordering reason | **cohesive as a file**, and the function reason is load-bearing |
| `src/vision/vision_encoder.cu` | 712/973 | 13 `__global__` kernels (75-399) + `VisionEncoder` host code with two non-sharing forwards: `encode_impl` (529-798) and `encode_impl_gemma4v` (799-958) | **CONFLATED**: kernels vs orchestration, and two tower implementations. This is the clearest split candidate on the list |
| `tools/imp-server/tool_call.cpp` | 705/907 | prompt construction (8-207), four parser dialects (208-533), streaming tag scan (534-684), validation/formatting (685-907) | **CONFLATED**, four pipeline stages |
| `src/exec/executor_forward_moe_cutlass.cu` | 600/851 | 2 defs: a warn helper and one 779-raw-line function | **cohesive as a file**, and the function is a gate blind spot (G-6) |
| `src/runtime/engine_scheduler.cpp` (allowlisted, 1308) | -/2002 | 15 defs; `step_decode_forward` 621 raw + `step_decode` 350 raw = 971 of 2002 | accepted `(c)` reason, but the reason string itself names four sub-concerns |

Nothing on this list is being "ridden at the limit" in the gate sense: `violations=0` and the warn thresholds are soft. The three cheap, low-risk splits are `handlers_chat.cpp` (endpoint boundary), `utils.cpp` (concern boundary) and `vision_encoder.cu` (kernel/host boundary).

---

### G1. Header include weight and build-time hotspots

TU fan-in by reverse BFS over 551 TUs. Churn = commits in the last 6 months touching the file.

| header | TUs (src/tools/tests) | direct includers | commits/6mo | cost |
|---|---|---|---|---|
| `src/runtime/config.h` | **60** (22/23/15) | 36 | 146 | **8760** |
| `src/exec/executor.h` | **79** (53/12/14) | 51 | 80 | **6320** |
| `src/runtime/engine.h` | **39** (19/11/9) | 40 | 151 | **5889** |
| `src/model/model_config.h` | **156** (88/22/45) | 16 | 32 | **4992** |
| `src/model/model.h` | 123 | 34 | 25 | 3075 |
| `src/core/qtype.h` | 325 | 15 | 9 | 2925 |
| `src/runtime/request.h` | 61 | 24 | 45 | 2745 |
| `src/model/model_arch.h` | 176 | 13 | 15 | 2640 |
| `src/model/tokenizer.h` | 155 | - | 16 | 2480 |
| `src/core/logging.h` | 264 | 231 | 9 | 2376 |
| `src/core/dispatch_policy.h` | **109** (61/25/23) | 28 | **20 (in 32 days)** | 2180 |
| `src/core/tensor.h` | 303 | 140 | 7 | 2121 |
| `src/memory/kv_cache.h` | 98 | 40 | 21 | 2058 |
| `src/core/config/*.h` (each) | 109 | 1 (`dispatch_policy.h`) | 1-11 | - |

**Top 5 today: `config.h` 8760, `executor.h` 6320, `engine.h` 5889, `model_config.h` 4992, `model.h` 3075.**

Against the SETTLED 2026-08-03 table (its window and mine are both "6 months" but offset by a month, so a delta mixes churn drift with the structural change; direction is still readable):

| header | TUs then -> now | commits then -> now | cost then -> now |
|---|---|---|---|
| `config.h` | 85 -> 60 (**-29 %**, F-10) | 130 -> 146 | 11050 -> 8760 (**-21 %**) |
| `engine.h` | 41 -> 39 (S-33's number, reproduced) | 129 -> 151 | 5289 -> 5889 (**+11 %**) |
| `executor.h` | 77 -> 79 | 55 -> 80 (**+45 %**) | 4235 -> 6320 (**+49 %**) |
| `model_config.h` | 133 -> 156 (**+17 %**) | 31 -> 32 | 4123 -> 4992 (**+21 %**) |

Two things the SETTLED table could not have shown:

1. `src/core/dispatch_policy.h`, created by the F-10 fix on 2026-08-04 (`357b23a9` #1227), has **109 TU fan-in, higher than the 60 of the `config.h` it replaced in `exec/`**, and took **20 commits in its first 32 days**. Per month that is ~2045 TU-rebuilds against `config.h`'s ~1460, and only **2 of the 20** commits also touched `config.h`, so it is not merely riding along with config-key additions. The subjects show why: spec/MTP and MoE work edits the policy directly (`#1470`, `#1473`, `#1503`, `#1510`, `#1409`, `#1412`, `#1416`).
2. The accessor was **not** renamed. SETTLED F-10 design note (b) predicted `runtime_config().gemm.x -> dispatch_policy().gemm.x`; the shipped code kept the old name and only changed the return type (`src/exec/executor.h:529` returns `const DispatchPolicy&`; `src/exec/quant_pipeline.h:86` mirrors it). `rg` in `src/exec` finds **143** `runtime_config()` and **0** `dispatch_policy()`. Any future reader or grep will mis-model this the way F-10 records the last one was mis-modelled.

Two smaller measured items:

- `src/runtime/vram_budget.h` has 81 TU fan-in, and **40 of those reach it only through `src/exec/executor.h:31`**, where the single use is a reference parameter (`executor.h:241 void pre_dequant_weights(cudaStream_t, const VRAMBudget&)`). Cost of the edge: 15 commits/6mo x 40 TUs = 600.
- `src/compute/gemm_cutlass_sm120.h` (92 TUs) includes `model/model_config.h` (156 TUs) for `FFNActivation`. Deleting that one edge takes `model_config.h` from **156 to 144 TUs**; the 12 that leave are `compute/` TUs including `gemm.cu` and `gemm_cutlass_grouped_3x.cu`.

---

### G2. Interface boundaries

### Virtual dispatch

`rg -n 'virtual|override' src/exec src/compute src/memory src/runtime` returns 95 lines. Classified:

| Where | Real `virtual`/`override` | Hot path? |
|---|---|---|
| `src/exec` | **0** (all 6 hits are the words "override"/"virtual" in comments or config-key names) | - |
| `src/compute` | **0** (all hits are comments about the gpt-oss "virtual extra softmax column") | - |
| `src/runtime` | **0** (all hits are comments/config keys) | - |
| `src/memory` | `Backend` (`backend.h:168-228`, 8 virtuals) with 3 implementations (`backend.cpp:248`, `fake_backend.h:43-78`, `vmm_backend.cpp:130-247`); `HostPinnedAllocator` + `HostRegistrar` (`host_pinned.h:49-147`) with 1 implementation | **acquire/release: init only.** The one serving-time reach is `KVCache` growth: `kv_cache.cu:529-530` calls `be->commit_range(...)` -> virtual `do_commit_range`. That is one virtual call per pool growth event, next to a driver VMM call |

**S-23 holds exactly as written** ("zero virtual dispatch in `src/exec/` and `src/compute/`"). The `memory` hierarchy is deliberate (`fake_backend` is the CPU-lane test seam, per ARCHMAP) and is not on the per-token path.

### Error signalling per layer

Counts are `rg` line counts, so comments and strings inflate them slightly; the ratios are the point.

| layer | `throw` | `return false;` | `std::expected` | `std::optional` | `IMP_LOG_FATAL` | `IMP_LOG_ERROR` | `[[nodiscard]]` |
|---|---:|---:|---:|---:|---:|---:|---:|
| api | 0 | 0 | 0 | 0 | 0 | 27 | 0 |
| core | 3 | 10 | 0 | 2 | 2 | 5 | 3 |
| memory | 22 | 63 | 0 | 0 | 0 | 31 | 29 |
| quant | 2 | 25 | 0 | 0 | 0 | 23 | 0 |
| compute | 24 | 368 | 0 | 0 | 2 | 78 | 7 |
| exec | 16 | 151 | 0 | 0 | 12 | 58 | 9 |
| model | 3 | 243 | **4** | 2 | 0 | 88 | 3 |
| runtime | 5 | 209 | 0 | 0 | 0 | 91 | 22 |
| vision | 0 | 50 | **12** | 2 | 0 | 49 | 12 |
| lora | 0 | 13 | 0 | 0 | 0 | 9 | 0 |
| `tools/imp-server` | 22 | 147 | 0 | 0 | 0 | 4 | 1 |

`std::expected` exists in exactly 7 files, all Qwen3-VL / M-RoPE / image-placeholder code (`src/vision/qwen3vl_vision_{grid,load,upload,config}.{h,cpp}`, `src/model/{mrope_positions,image_placeholders}.{h,cpp}`). The convention is documented at `docs/internals/CPP23.md:45`: `expected` "replaces `bool f(..., T& out, std::string& err)`". That specific shape is essentially gone from the tree (4 residual matches, of which 2 are `answer_lost_to_reasoning`, a genuine predicate). So the migration hit its stated target; what remains is the much larger `bool` + `IMP_LOG_ERROR` population, which the convention does not claim.

`IMP_LOG_FATAL` is 14 sites and does **not** abort (`core/logging.h:62` logs only; `IMP_CHECK` at `:72-79` is the one that aborts). `src/exec` holds 12 of the 14.

### Ignored `bool` returns

Method: extracted 350 `bool`-returning free functions from headers, then searched for statement-level (result-discarded) calls. Four survived reading:

| Site | Verdict |
|---|---|
| `executor_attention_prefill.cu:310,425` (`try_fa2_fp16qk_prefill`) | **not discarded** - both are `if (...)` continuation lines. The chunked-capture path at `:302-306` even throws when FA2 declines (`"chunked_prefill: FA2 declined a capture-replay chunk"`), which is the S-22 pattern applied correctly |
| `executor_attention_prefill.cu:326,440` (`attention_cublas_prefill_sliced`) | **not discarded** - `else if` conditions |
| `engine_scheduler.cpp:1973` (`try_launch_async_graph_loop`) | **discarded**, and correct: it is inside `Engine::step_decode_process_outputs` (void); declining leaves the state untouched and the next `step()` runs an eager decode. Note the asymmetry with `engine_spec_ngram.cpp:476`, where the same call *is* tested and its `true` short-circuits the step. Not a defect, but the two call sites read as if they had different contracts |
| `weight_upload.cu:2782` (`weight_cache_write`) | **discarded and documented** at `:2779-2781` ("Best-effort") |

No `[[nodiscard]]`-worthy ignored failure found. This sweep is a negative.

### C-API boundary

There is **no macro**. `src/api/imp_api.cpp` hand-writes the same `try { ... } catch (const std::bad_alloc&) ... catch (const std::exception& e) ... catch (...)` block 15 times, `imp_api_vision.cpp` 4 times (2-arm), `imp_api_suspend.cpp` once (5-arm). See G-10 for the four that were missed.

---

### G3. Config surfaces

Nine ways one setting can be set, all present in the tree:

1. **`imp.conf` key** - 225 leaf keys, 54 declared directly in `src/runtime/config.h` (sections `runtime` 23, `server` 9, `rope` 6, `vram` 6, `warm_cache` 2, `suspend` 2, `constrained` 2, `calibration` 2, `bench` 1, `paths` 1) and 171 in the 9 lifted `src/core/config/*.h` (`attention` 36, `speculative` 33, `diagnostics` 33, `gemm` 22, `moe` 20, `kv_cache` 10, `gdn` 10, `generation` 5, `ffn` 2). All 225 are registered in `src/runtime/config.cpp:110-436`.
2. **`--set key=value`** - `tools/common/args_common.cpp:14-17`, applied by `apply_overrides` (`config.cpp:571-580`). Unknown key is an **error** (both tool mains `exit(1)` on the `rejected` vector), which is what `src/runtime/CLAUDE.md` promises.
3. **`--config path`** - same parse site; loaded at `config.cpp:593-598`.
4. **Dedicated CLI flag** - 26 in `tools/common/args_common.cpp` (12 value, 14 boolean), plus per-tool flags in `tools/imp-cli/args.cpp` and `tools/imp-server/args.cpp`.
5. **`IMP_*` env** - only **7** names are read by `getenv` in the C++ tree: `IMP_DETERMINISTIC`, `IMP_FMHA_FA2`, `IMP_SPEC_TRACE`, `IMP_JUMP_TRACE`, `IMP_PPL_DUMP` (`config.cpp:468-486`), `IMP_CONFIG` (`config.cpp:495`), and `IMP_WORKER_TIMING` (`tools/imp-server/batching_engine.cpp:182`). The 22 names in `docker-entrypoint.sh` are translated into CLI flags by the shell, so 21 of them correctly have no C++ reader. `IMP_WORKER_TIMING` is the one C++ reader that the entrypoint does not know about.
6. **Request body field** - no body field shares a name with a config key. The one real per-request shadow is `"speculative"` (bool) -> `Request::spec_override` (`src/runtime/request.h:96-101`), which collapses `speculative.ngram` + `speculative.mtp_k` + token recycling in a single tri-state.
7. **`ModelConfig::Overrides`** - one nested `Gemma4` struct, 7 bools (`src/model/model_config.h:181-191`). See G-7.
8. **Hardcoded default** - the member initialiser in each section struct.
9. **Resolver mutation at init** - 6 write sites in `src/runtime/engine_init_resolver.cpp`: `:71` `runtime.warmup=false`, `:73` `runtime.deterministic_gemm=true`, `:78` `moe.no_expert_cache=true`, `:82` `gdn.ref_kernel=true` (all in the `debug_raw` cascade), `:382` and `:785` `runtime.deterministic_gemm=true`. SETTLED F-10 (a) named two of these; there are six.

**Documented precedence** (`config.cpp:585-608`): `seed_from_env` -> `load_from_file` -> `apply_overrides`, i.e. **env < imp.conf < `--set`**. Dedicated CLI flags live on a separate struct (`EngineConfig`) and are merged later, in `Engine::init`.

### Five settings across surfaces

| setting | imp.conf | `--set` | CLI flag | `IMP_*` | request | Overrides | default | resolver | precedence documented | test |
|---|---|---|---|---|---|---|---|---|---|---|
| **kv cache dtype** | `kv_cache.dtype` (`config.h:35`) | y | `--kv-fp8/-int8/-int4/-nvfp4/-mxfp4` (`args_common.cpp:42-51`) | `IMP_KV_FP8`, `IMP_KV_INT8` (entrypoint 89-95) | n | n | `"auto"` | `engine_init_resolver.cpp:186-260`; string arm **validates and warns** on an unknown value (`:213-221`) | **yes**, `docker-entrypoint.sh:167-178` | **yes**, `tests/test_entrypoint.sh:124-132` + `tests/test_config.cpp:296-317` |
| **max_seq_len** | `runtime.max_seq_len` (`config.h:72`) | y | `--max-seq-len` (imp-cli only, `args.cpp:129` -> `main.cpp:179`) | via `IMP_SET` only | dead `config_overrides` JSON | n | `0` = auto | `engine_init_resolver.cpp:831-835` **overwrites the CLI value** | **no, and the comment at `:836-838` states the opposite** | **no** binding test |
| **prefill chunk size** | `runtime.prefill_chunk_size` (`config.h:149`) | y | `--prefill-chunk-size` (`args_common.cpp:32`) | `IMP_PREFILL_CHUNK_SIZE` (entrypoint 127) | dead JSON | n | `-1` = per-arch | `engine_scheduler.cpp:444-450`, **CLI wins** | **yes**, `config.h:148` | **yes**, `tests/test_config.cpp:443-451` |
| **deterministic** | `runtime.deterministic` (`config.h:43`) | y | none | `IMP_DETERMINISTIC` (`config.cpp:468`) | n | n | `false` | none for `deterministic`; `deterministic_gemm` forced at `:73,:382,:785` | general rule only | the `deterministic -> deterministic_gemm` cascade (`config.cpp:145-149`) has **no** test |
| **api key** | none | n/a | `--api-key` (`tools/imp-server/args.cpp:98`) | `IMP_API_KEY` (entrypoint 50) | `Bearer` / `x-api-key` header, `main.cpp:283-286` | n | `""` = auth off | none | n/a (single surface) | **yes**, `tests/test_server_auth.cpp` + `tests/test_entrypoint.sh` |

`imp.conf.example` lists **139** keys against **225** registered (62 %). Nothing in the example is stale (the example is a strict subset). **No script checks this** in either direction: `rg -l 'imp.conf.example' scripts tools .github tests` returns only usage strings (`tools/imp-cli/main.cpp:54`, `tools/imp-server/main.cpp:69`, `tools/imp-bench/main.cpp:106`, `tools/common/args_common.h:84`) and two tests that mention it in comments (`tests/test_config.cpp:407`, `tests/test_server_args.cpp:79,97`).

---

### G4. Dead code

Only classes the 2026-08-03 decl-only sweeps and `check_dead_inline_accessors.py` structurally cannot see. Everything below was re-verified by hand.

**Model architectures** (`src/model/model_arch.h`, 16 enumerators): every one has a producer. Two are GGUF-only: `QWEN35_MOE` (only `model.cpp:372`, the `"qwen35moe"` string table, no `hf_config_loader.cpp` class-name mapping) and `NOMIC_BERT` (only `model.cpp:383`; the SafeTensors path explicitly refuses encoder-only archs at `hf_config_loader.cpp:90-93,111-114`). `NOMIC_BERT` is by design. `QWEN35_MOE` is an asymmetry worth one line in `docs/MODELS.md`, not a finding.

**Quant types** (`src/core/qtype.h`, 27 enumerators):
- `Q8_1` (`qtype.h:23`): **producible** (`gguf_parse.cpp:183-184` maps GGUF wire type 9 to it, reached from `gguf_loader.cpp:669`), **no dequant** (`dequant_gpu_supported`, `dequant_gpu.cu:11-29`, returns false), **no GEMM registry entry**, **no load-time reject**. Finding G-8.
- `FP4_E2M1`: never assigned to a weight tensor anywhere in `src/`; reachable only through the C API (`imp_api.cpp:160-161`). Finding G-4.
- `FP8_E5M2`: `safetensors_loader.cpp:121-123` deliberately maps the SafeTensors `F8_E5M2` dtype to `QType::FP8_E4M3` with a comment. Already in LIMITATIONS as an untested format.
- Everything else has dequant, registry (or a documented registry-exempt KV path) and a producer.

**`src/lora/`** (328 LOC): fully reachable. `tools/imp-server/args.cpp:88` (`--lora NAME=PATH`) -> `main.cpp:150` -> `imp_lora_load` (`imp_api.cpp:272`) -> `LoraAdapter::load`; per-request selection via the `"lora"` body field (`handlers_chat_params.cpp:516` -> `handlers_chat_core.cpp:710` -> `imp_lora_set`, `imp_api.cpp:287`); kernels in `src/exec/executor_lora.cu`; test `tests/test_lora.cpp`. **No `imp-cli` flag exists**, so LoRA is server-only. Not dead.

**`src/vision/`**: two encoders, both selected, by presence rather than by arch switch. `VisionEncoder` (`vision_encoder.cu:504`) fires at `engine_workspace_warmup.cpp:193-195` when `config_.mmproj_path` is non-empty; `Qwen3VLEncoder` (`qwen3vl_encoder.cu:246`) at `:199-206` when `model_->vision_tower` exists. No unselected encoder.

**`tools/` binaries**: `imp-cli` (CMakeLists `:485`), `imp-quantize` (`:504`), `imp-bench` (`:518`), `imp-server` (`:562`), all default-on via `IMP_BUILD_TOOLS/BENCH/SERVER`; `fuzz_${t}` (`:1218`) gated `IMP_FUZZERS OFF`. Only `imp-bench` is absent from `README.md`, and it is documented in `docs/internals/KERNELS.md` and `docs/usage.md`. Nothing built-and-undocumented.

**Production-unreachable but test-referenced**: `dequant_int8_fp16` (`src/quant/dequant_int8.cu:71`) and `dequant_int4_fp16` (`src/quant/dequant_fp16.cu:49`) are compiled into `libimp` (CMakeLists `:180-181`) and their only callers are `tests/test_quant.cu:298` and `:175,242`. Zero references in `src/` or `tools/`. This is the `fp32_accum_add_fp16_kernel` class the debt ledger closed on 2026-08-21: invisible to the decl+def sweep (3 mentions), to the inline-accessor gate (`.cu` definitions) and to a caller query (a caller exists). Two files, ~120 LOC. Listed here rather than as a finding because the cost is one TU each and the ledger already named the class.

---

### Findings

### [G-1] The "Layer DAG" is a single 8-node cycle, and 73 % of the violations are two dependency-free headers
Axis: G   Sev: S1   Confidence: high
Evidence: `cycles.py` over 842 files: file-level SCCs = **0**, layer-level SCC = **`{compute, exec, lora, memory, model, quant, runtime, vision}`**. 88 backward include lines total; `runtime/pdl.h` accounts for 34 (compute 16, exec 12, quant 6), `runtime/process_diag.h` for 25 (compute 15, model 4, quant 3, exec 2, vision 1), `compute/pdl_device.cuh` for 4, `runtime/graph_diag.h` for 1 = **64**. `src/runtime/pdl.h` includes only `<cuda_runtime.h>` (`:3`); `src/runtime/process_diag.h` includes only `<string>` (`:19`); `src/compute/pdl_device.cuh` includes **nothing** (35 lines).
Current: `docs/audit/ARCHMAP.md:9-16` publishes a DAG. The tree is not one. Three headers with zero project dependencies sit in `runtime/` and `compute/` and are pulled *down* into every layer below them.
Expectation: a layer diagram in an audit doc is a checkable claim. llama.cpp keeps `ggml` strictly below its runtime; vLLM's `csrc/` has no import back into the scheduler. The fix pattern is already established *in this repo*: `WeightHandle` moved `exec/ -> core/` and `CutlassMxFP4Weight` moved `compute/ -> quant/` on 2026-08-03, each closing an edge class outright.
Delta: 64 of 88 backward lines are a file-placement accident, not coupling. Moving the four headers to `core/` (the `.cpp`/`.cu` bodies can stay where they are; `process_diag.cpp` is the only one that needs `runtime/config.h` and it is the installer) takes `compute -> runtime` 31 -> 0, `quant -> runtime` 9 -> 0, `model -> runtime` 4 -> 0, `memory -> runtime` 1 -> 0, `exec -> runtime` 20 -> 6, `quant -> compute` 5 -> 1.
Cost: 64 include lines + 4 file moves + CMake source paths. Zero semantic change (no symbol moves namespace). Blast radius is compile-only, and the gate that would catch a mistake is the build itself. Risk: low.
Falsifier: "moving them restores the DAG" - **checked, n**. Simulated in `cycles.py` by relabelling the four headers to `src/core`: the 8-node SCC is **unchanged**. Adding `storage_planner.h`/`vram_budget.h` -> `exec/`, dropping the vestigial `config.h` include, and moving `nvfp4_expert_offload.h` -> `core/` also leaves it unchanged. The residual cycles (`vision <-> runtime`, `memory <-> model`, `model <-> vision`, `compute -> model -> quant -> compute`) are ordering decisions, not misplaced files. So the honest claim is "cuts 73 % of the violations and makes four layers cleanly below `runtime`", **not** "restores the DAG".

### [G-2] `src/exec/executor.h` is the fastest-rising build-cost item and neither campaign touched it; the F-10 fix moved churn into a wider header
Axis: G   Sev: S1   Confidence: high
Evidence: fan-in x churn today: `config.h` 60x146=8760, **`executor.h` 79x80=6320**, `engine.h` 39x151=5889, `model_config.h` 156x32=4992. SETTLED 2026-08-03: `config.h` 85x130=11050, `engine.h` 41x129=5289, `executor.h` 77x55=4235, `model_config.h` 133x31=4123. `src/core/dispatch_policy.h`: 109 TUs, created `357b23a9` 2026-08-04, **20 commits in 32 days**, of which only **2** also touch `config.h`. `rg -c 'runtime_config()' src/exec` = 143; `rg -c 'dispatch_policy()' src/exec` = **0** (`src/exec/executor.h:529` returns `const DispatchPolicy&` under the old name).
Current: F-10 and F-24/S-33 both attacked the fan-in half of the cost, on `config.h` and `engine.h`. `executor.h` went 4235 -> 6320 (+49 %) in the same window with no attention, driven entirely by churn (55 -> 80 commits). `dispatch_policy.h` costs ~2045 TU-rebuilds/month against `config.h`'s ~1460.
Expectation: SETTLED's own framing - "cost is fan-in x churn, and fan-in was the cheap half" (F-24). The same arithmetic applied to today's table puts `executor.h` second and `dispatch_policy.h` on a per-month rate above the header it was extracted from.
Delta: the extraction relocated churn from a 60-TU header to a 109-TU one, and left the accessor named after the type it no longer returns. Nobody has priced `executor.h`'s fan-in half.
Cost: pricing is free (`fanin_drop.py`, one command per candidate edge). Two concrete edges already measured: dropping `executor.h:31 -> runtime/vram_budget.h` (one reference parameter at `:241`; forward-declare would do) removes 40 of `vram_budget.h`'s 81 TUs; dropping `compute/gemm_cutlass_sm120.h:3 -> model/model_config.h` takes `model_config.h` 156 -> 144. Risk: low, both are mechanical.
Falsifier: "the 20 dispatch_policy.h commits are mostly the #1227 landing and its immediate follow-ups" - **checked, n**. The commit subjects span 2026-08-10 to 2026-08-21 across MoE offload (#1409/#1412/#1416), spec/MTP (#1455/#1459/#1464/#1470/#1473/#1503/#1510) and diagnostics (#1356/#1359); only `357b23a9` is the landing. What is **not** checked: whether an `executor.h` edit really rebuilds 79 TUs in wall-clock terms (S-32 measured a ~5 s per-TU floor in `src/exec/`, so the wall-clock cost needs a build).

### [G-3] The ABI-stable C API has no first-party consumer
Axis: G   Sev: S1   Confidence: med
Evidence: `rg -l 'imp/imp\.h'` over `src tools tests`: `src/api/*`, `tools/imp-server/{handlers.h,utils.cpp}` (2 files) and **17 files under `tests/`**. `tools/imp-cli` includes it **zero** times. Direct `src/` includes instead: `tools/imp-server -> src/runtime` **28**, `src/model` 11, `src/memory` 8, `src/vision` 6, `src/api` 8; `tools/imp-cli -> src/model` 10, `src/runtime` 8.
Current: `CONTRIBUTING.md` and `docs/internals/ARCHITECTURE.md` describe `include/imp/{imp,types,error,config}.h` as the ABI-stable public boundary. Both shipping binaries bypass it and link against the internal C++ headers.
Expectation: llama.cpp's `llama-server` and `llama-cli` are consumers of `llama.h`; that is what keeps the C API honest. A public API whose only exercise is a test suite drifts from the product, and the drift is invisible until an external consumer hits it.
Delta: everything reachable only through the C API - the `map_dtype` translation (G-4), the try/catch contract (G-10), `ImpConfig` defaults - is validated by tests alone. Two of this report's findings sit exactly there.
Cost: not a refactor proposal. The cheap version is a CI-visible statement of the fact plus a contract test that drives the four config fields through `imp_context_create` with hostile values. ~1 test file. Risk: none.
Falsifier: "imp-server uses the C API for the parts that matter" - **checked, partly**. It calls `imp_lora_load`/`imp_lora_set` and `build_config` produces an `ImpConfig`, but the engine handle, KV manager, scheduler stats and vision pipeline are reached through `src/runtime/engine.h` directly (8 `tools/imp-server` files include it). Not checked: whether an external consumer exists outside this repo.

### [G-4] The C-API KV dtype is unvalidated where the identical `imp.conf` setting validates and warns
Axis: G   Sev: S2   Confidence: high
Evidence: `src/api/imp_api.cpp:358` `ecfg.kv_cache_dtype = map_dtype(config->kv_cache_dtype);` with `map_dtype` at `:142-169` mapping every `ImpDType` 1:1 to a `QType` (out-of-range falls to `F16`). No validation of `ecfg.kv_cache_dtype` exists anywhere: `rg -n 'kv_cache_dtype' src/runtime/engine.cpp src/api/imp_api.cpp` returns only `imp_api.cpp:78` (the default) and `:358`. `src/runtime/engine_init_resolver.cpp:200` gates the whole resolver on `if (config_.kv_cache_dtype == QType::F16)`, so any non-F16 C-API value skips it, including the `:213-221` unknown-value `IMP_LOG_WARN` that the `imp.conf` string path added deliberately. `src/exec/executor_attention_decode.cu:206-348` switches on `INT4 / NVFP4 / MXFP4_KV / INT8 / FP8_E4M3` and its final `else` (`:342`) launches the **FP16** paged kernel.
Current: 5 of the 11 `ImpDType` values are not valid KV dtypes (`IMP_DTYPE_FP32`, `_BF16`, `_FP8_E5M2`, `_INT32`, `_FP4_E2M1`). Passing one produces a `QType` the KV cache is sized for by `kv_block_bytes_per_layer` (`src/runtime/vram_budget.cpp:740-761`, which has a special packed-4-bit chain that `FP4_E2M1` is **not** in) and then read by the FP16 kernel.
Expectation: the same setting on the `imp.conf` surface already does the right thing, and the comment at `engine_init_resolver.cpp:213-217` says why it was added: "An unrecognised value used to fall through in silence and leave FP16 - so `kv_cache.dtype=mxfp4_kv` looked like it applied". That lesson was applied to one of the two surfaces.
Delta: one validation, one surface. The public, ABI-stable one is the unguarded one.
Cost: a `switch` in `map_dtype` (or a check in `Engine::init`) rejecting the 5 with `IMP_ERROR_INVALID_ARG`, plus a contract test. ~30 LOC, 2 files. What breaks if wrong: an embedder currently passing one of these and getting silent FP16 would start getting an error, which is the point.
Falsifier: "`kv_block_bytes_per_layer` or `KVCache`'s constructor rejects it first" - **checked, n**. `vram_budget.cpp:747` lists `INT4 | NVFP4 | MXFP4_KV` only; `FP4_E2M1` takes the `dtype_size()` branch, and `dtype_size(FP4_E2M1)` returns 1 (`core/qtype.cpp:23`), so the pool is sized 2x the real payload rather than zero. `rg 'qtype ==|switch (qtype' src/memory/kv_cache.cu` finds no dtype allowlist. Not checked on hardware (no GPU in this pass), so the *observable* symptom is unverified; the dispatch path is verified by reading.

### [G-5] `runtime.max_seq_len` silently outranks `--max-seq-len`, opposite to the documented rule and to its own sibling setting
Axis: G   Sev: S2   Confidence: high
Evidence: `src/runtime/engine_init_resolver.cpp:831-835`:
```
void Engine::init_compute_max_seq_len_() {
    const auto& mcfg = model_->config();
    if (int v = runtime_config_.runtime.max_seq_len; v > 0) {
        config_.max_seq_len = v;
```
`config_.max_seq_len` already holds the CLI value at this point (`tools/imp-cli/args.cpp:129` -> `tools/imp-cli/main.cpp:178-179` -> `ImpConfig` -> `EngineConfig`). The comment immediately below, `:836-838`, reads "Whoever set it before the auto resolver runs is the operator: the CLI flag, `runtime.max_seq_len`, or a C-API embedding" - describing a precedence the four lines above have just inverted. The sibling setting does the documented thing: `src/runtime/engine_scheduler.cpp:444-450` takes the CLI value when it is >= 0 and only then falls back to the config key, and `src/runtime/config.h:148` states the rule ("A CLI value wins over the file").
Current: two settings on the same axis, opposite precedence, one written-down rule, and the wrong one carries a comment asserting the right one.
Expectation: the general order established at `config.cpp:585-608` is env < file < `--set`, with dedicated flags on top (item 5 of G3). vLLM and llama.cpp both resolve CLI over file. Nothing in imp's docs proposes an exception for `max_seq_len`.
Delta: an operator who sets `runtime.max_seq_len` in `imp.conf` and then passes `--max-seq-len` on the command line gets the file value, with no log line saying the flag was dropped (`:834` logs `"max_seq_len: runtime.max_seq_len=%d"`, which reads like normal resolution).
Cost: 3 LOC (`if (config_.max_seq_len <= 0 && v > 0)`), 1 file, plus a `tests/test_config.cpp` binding test - there is currently none (`rg max_seq_len tests/test_config.cpp` = 0 key-binding assertions). What breaks if wrong: an operator relying on the file to override a hardcoded flag, which no doc promises.
Falsifier: "the CLI value has not landed in `config_.max_seq_len` yet when the resolver runs" - **checked, n**. `imp_context_create` copies `ImpConfig` into `EngineConfig` at `imp_api.cpp:326-410` before `Engine::init()`; the resolver runs inside `init()` (ARCHITECTURE's init table). Not checked: whether `imp-server` has an equivalent flag (it does not, so the exposure is imp-cli and C-API embedders).

### [G-6] The function-size gate stops counting at the first column-0 `}`, so a 557-LOC body reads as 176 and clears the hard limit
Axis: G   Sev: S2   Confidence: high
Evidence: `tools/check_function_size.py:140`:
```
while j < len(lines) and not lines[j].startswith("}"):
```
`src/exec/executor_forward_moe_cutlass.cu:72` defines `GraphExecutor::try_run_moe_cutlass3x_nvfp4_prefill_`, whose body runs to `:849`. The first column-0 `}` after `:72` is **`:360`**, inside the body (the file dedents inner block closes: `:845-847` are `}`, `}  // !smallM_done`, `}  // !device_args_done`). Non-blank non-comment lines 74-359: **176**. Lines 74-849: **557**. The hard limit is 500. `python3 tools/check_function_size.py` reports `violations=0`, and `--list` (everything over the 200 warn) does not mention the file at all. `--selftest` is **10/10**, and its cases include "lambda inside a body does not end it" but none for a column-0 brace inside a body.
Current: the largest function in `src/exec/` outside the allowlist is measured at ~1/3 of its size and is therefore neither a violation nor a warning.
Expectation: this is the third instance of the class SETTLED C2 already records twice - S-30 ("it measured files, not translation units") and S-31 ("a file's `[allow]` reason does not cover what is inside it"). Each was found the same way: read the gate's parser rather than its output.
Delta: one hard-limit violation is invisible. Repo-wide, an `awk` scan for a column-0 `}` followed by more indented code inside `src/` and `tools/` finds **exactly one file**, this one - so the current blast radius is 1, but the gate has no defence against the next one and its selftest does not cover the case.
Cost: brace-depth counting instead of column-0 matching in `functions_in()` (~15 LOC in one file), plus one selftest case. Then either split the function or allowlist it with a reason. Risk: the fix may surface other under-measured bodies; that is the point, and `--update` re-pins the allowlist.
Falsifier: "the gate finds the function under a different name / the 176 is the real code LOC" - **checked, n**. `--list` and the default output both contain zero lines matching `executor_forward_moe_cutlass`, and `check_filesize.py` independently scores the whole 851-line file at 600 code LOC, which cannot contain a 557-LOC function measured as 176 unless the measurement is truncated.

### [G-7] Six of the seven `ModelConfig::Overrides::Gemma4` flags have no writer and are read at seven dispatch sites
Axis: G   Sev: S2   Confidence: high
Evidence: `src/model/model_config.h:181-191` declares 7 bools. `rg -n 'overrides\.gemma4\.[a-z_0-9]+\s*=' src tools tests` returns **exactly one** assignment: `src/runtime/engine_init_resolver.cpp:801` `model_->config_.overrides.gemma4.force_mmvq = true;`. The other six are read and never written: `fp32_gemm_out` (`executor_attention.cu:704`), `no_graphs` (`engine_init_resolver.cpp:796`), `fp32_expert_down` (`executor_forward_moe.cu:226`), `no_decode_fast` (`executor_forward_moe.cu:334`), `ggml_prefill` (`executor_forward_moe.cu:524`), `no_post_ffw_1` (`executor_forward_moe_batch.cu:1400`). The header comment at `:178-180` says they are "Populated from imp.conf (legacy `[gemma4]` section seeds compat)"; `src/runtime/config.cpp:313-317` records that section was removed in Phase 5 Track A and nothing recreates it.
Current: six branches in the Gemma-4 MoE/attention dispatch are permanently `false`, and the header states the opposite.
Expectation: the `log_set_level` precedent in SETTLED C is the exact analogue - a knob whose only writer disappeared, where the fix was to *wire* it (`diagnostics.log_level`) rather than delete it, because removing it would have cemented the gap. `calibration.out_path` (debt ledger item 7) was closed the same way.
Delta: either these are diagnostics an operator should be able to reach (wire them to `diagnostics.*` keys) or they are dead (delete the fields and the six branches). Today they are neither, and the comment misleads.
Cost: wiring = 6 config keys + 6 registrations + example entries, ~40 LOC in 3 files. Deleting = 6 fields + 6 branches, ~30 LOC in 5 files. Either way the stale comment goes. Risk: low; `force_mmvq`, the one live field, is untouched.
Falsifier: "a test or a tool writes them" - **checked, n**. The `rg` above covers `src tools tests` and returns one hit. Not checked: whether the six branches were ever measured to matter (that would need a GPU A/B, and it is the question that decides wire-vs-delete).

### [G-8] `QType::Q8_1` is producible from a GGUF file with no dequant, no GEMM kernel and no load-time rejection
Axis: G   Sev: S2   Confidence: med
Evidence: `src/model/gguf_parse.cpp:183-184` maps GGUF wire type 9 to `QType::Q8_1`, reached from `gguf_loader.cpp:669`. `dequant_gpu_supported()` (`src/quant/dequant_gpu.cu:11-29`) lists 13 types and **not** `Q8_1`. `rg 'QType::Q8_1' src tools` outside `core/qtype.cpp` returns **only** `gguf_parse.cpp:184` - no registry entry in `src/exec/gemm_kernel_*.cu`, no branch anywhere. (The many other `Q8_1` mentions in `src/compute/gemm_q4k.cu` etc. are the dp4a *activation* format, a different thing.) `src/core/qtype.cpp:38,99` give it a row-byte size and a name, so it is a fully-formed type that nothing can compute with.
Current: a checkpoint declaring a Q8_1 weight tensor loads without complaint and then has no path at dispatch.
Expectation: the sibling gap for unknown `kv_cache.dtype` strings was closed with a warning (`engine_init_resolver.cpp:213-221`), and S-22 records the principle for the dispatch chain: "No tier accepted" is an error, not a degraded answer. A loader that accepts a type the compute layer cannot serve is the same shape one phase earlier.
Delta: `LIMITATIONS.md` lists Q4_1/Q5_0/Q5_1/Q2_K/Q3_K/Q8_K as *untested*; those at least have a dequant path and reach the generic catch-all kernel. `Q8_1` has neither, and is not on the list.
Cost: a reject in `gguf_parse.cpp` (or mapping it to `QType::NONE` the way unsupported i-quants are handled at `:206`), ~5 LOC. Risk: none, no real checkpoint should reach it.
Falsifier: "no GGUF file in the wild stores Q8_1 weights, so this is unreachable" - **not checked, and probably true**: llama.cpp uses Q8_1 as an intermediate activation format, not a storage type. That is why this is S2/med and not higher. What makes it worth the five lines is that the failure is silent at the point where it is cheap to catch.

### [G-9] `runtime/config.h` came back into `src/exec/` after F-10 drove it to zero, and the include is unused
Axis: G   Sev: S3   Confidence: high
Evidence: `src/exec/pre_dequant_phase1_fp16_cache.cu:20` `#include "runtime/config.h"`, added 2026-08-12 by `57fca0d1` (#1388), eight days after F-10 closed (`357b23a9` #1227, 2026-08-04). `grep -E 'RuntimeConfig|Rope|Vram|Server|WarmCache|Suspend|Bench|Paths|Constrained|Calibration|pending_runtime_config'` over that file (excluding line 20) returns **nothing**. Its only config read is `runtime_config().gemm.fp8_ssm_proj` at `:120`, and `runtime_config()` returns `const DispatchPolicy&` (`src/exec/executor.h:529`), reachable through `exec/executor.h` -> `core/dispatch_policy.h`. `SETTLED.md` F-10 still asserts "`src/exec/` includes `runtime/config.h` **zero** times".
Current: a 60-TU header is pulled into a `src/exec/` TU for no symbol, and a settled verdict is stale in the direction SETTLED's own section G warns about ("A stale *open* entry is worse than a stale closed one" - this is the mirror case, a stale *closed* one).
Expectation: `scripts/check-release.sh` section 1c already pins SETTLED's anchors so an entry "cannot quietly become the next stale prior". Anchors prove a file exists, not that a count is still zero. A zero that was worth a campaign is worth a gate.
Delta: one include line, and no mechanism preventing the next one.
Cost: delete the line (1 LOC); optionally add `src/exec/**` -> `runtime/config.h` to a layering check. There is no such check today - `ls tools/check_*.py` has 13 gates and none of them look at include direction. Risk: none.
Falsifier: "the TU needs `config.h` transitively for a type it names" - **checked, n**, by symbol grep over the full list of types declared in `runtime/config.h`. Not checked by compiling (no build in this pass), so the residual risk is a macro or a `using` I did not enumerate.

### [G-10] Four of 23 `ImpError` entry points have no try/catch, and the invariant is 19 hand-copied blocks with no macro
Axis: G   Sev: S3   Confidence: high
Evidence: unwrapped: `src/api/imp_api.cpp:983` `imp_context_reset`, `:1029` `imp_enable_mtp_spec_decode`, `src/api/imp_api_suspend.cpp:44` `imp_weights_snapshot_arm`, `:67` `imp_gpu_release`. Wrapped: 15 blocks in `imp_api.cpp` (`try` at 180, 275, 290, 346, 560, 591, 647, 680, 710, 809, 848, 883 plus delegating wrappers), 4 in `imp_api_vision.cpp` (51, 70, 90, 107), 1 in `imp_api_suspend.cpp` (21). `rg 'IMP_API_TRY|#define' src/api/imp_api.cpp` finds no wrapper macro; the only `#define` is `IMP_VERSION_STRING`. `docs/audit/ARCHMAP.md:44` claims "all entry points wrap try/catch".
Current: the ABI contract "nothing throws across the C boundary" is enforced by 20 copies of the same text. Four copies are missing. `imp_gpu_release` is the one with real fan-out: it calls `imp::reset_static_cuda_state()` (`imp_api_suspend.cpp:74`), which invokes every `IMP_REGISTER_CUDA_STATIC_RESET` hook across `compute/`.
Expectation: SETTLED S-7 states the repo's own rule for exactly this shape - `apply_constraint_mask()` is nine lines factored into one function because "four copies of this chain is exactly how an unmasked path ships". The same argument applies to a 6-line try/catch replicated 20 times.
Delta: 4 missing copies, and no mechanism that would notice the 5th.
Cost: an `IMP_API_GUARD(...)` macro or a `template <class F> ImpError guard(F&&)` helper, plus 23 call-site conversions. ~60 LOC in 3 files, mechanical. Risk: low; the catch arms differ only in whether `bad_alloc` is split out.
Falsifier: "nothing reachable from those four can throw, so it is cosmetic" - **checked, mostly true**. I traced `imp_context_reset`'s callees (`KVCacheManager::free_sequence`, `evict_cached_block`, `Engine::clear_recurrent_snapshots`, `reset_ssm_state`, `invalidate_graphs`, `reset_batch_pool_cache`, `mtp_accuracy_reset` which is `noexcept`) and found no `throw`; the 16 `throw` sites in `src/memory/kv_cache.cu` are all in constructors. What remains reachable is `std::bad_alloc` / `std::system_error` from the container and CUDA-handle work those functions do, which is UB across a C ABI. That is why this is S3 and not S2.

### [G-11] `imp.conf.example` covers 139 of 225 registered keys and no check exists in either direction
Axis: G   Sev: S3   Confidence: high
Evidence: `grep -cE '^\s*[a-z_]+\s*=' imp.conf.example` = **139**. Registered keys in `src/runtime/config.cpp:110-436` = **225** (54 declared in `src/runtime/config.h`, 171 in `src/core/config/*.h`). 86 registered keys are absent from the example, concentrated in `attention` (26 of 36), `gemm` (20 of 22), `diagnostics` (14 of 33), `speculative` (10 of 33). Nothing in the example is unregistered (strict subset). `rg -l 'imp.conf.example' scripts tools .github tests` returns only usage strings (`tools/imp-cli/main.cpp:54`, `tools/imp-server/main.cpp:69`, `tools/imp-bench/main.cpp:106`, `tools/common/args_common.h:84`) and two comment references in tests.
Current: 38 % of the tuning surface is discoverable only by reading `src/core/config/*.h`. The absent keys are exactly the perf-relevant ones the roadmap and CHANGELOG name (`attention.fa2_hd256`, `attention.fa2_hd256_bkv`, `gemm.fp8_ssm_proj`, `moe.nvfp4_smallM`).
Expectation: the repo already runs 13 static gates including `docs sync`, `doc citations` and `release hygiene`, and `tools/alloc_allowlist.txt` is explicitly a **two-way** ratchet (S-27) so a list "cannot go stale in either direction". The config example is the one operator-facing list with no such ratchet.
Delta: no gate, and a 62 % coverage that nobody can see moving.
Cost: a `tools/check_config_example.py` reading the registration table and the example, ~80 LOC, plus a decision about which keys are deliberately internal (a `# internal` marker in the header, or an allowlist). Adding all 86 to the example is *not* the recommendation - the decision per key is the work. Risk: none.
Falsifier: "`tests/test_config.cpp` already covers it" - **checked, n**. Its only mentions of the example are comments at `:407` (a negative control about boolean table values) and in `tests/test_server_args.cpp:79,97`. Neither compares key sets.

### [G-12] `build_config`'s `overrides` JSON is an 11-key config surface every caller passes empty
Axis: G   Sev: S3   Confidence: high
Evidence: `tools/imp-server/handlers.h:293-296` declares `build_config(..., const json& overrides = json::object())` and `load_model_into_state(..., const json& config_overrides = json::object())`. `handlers.cpp` reads 11 keys out of it: `max_batch_size` (`:563`), `max_seq_len` (`:582-583`), `kv_fp8/kv_int8/kv_int4/kv_nvfp4/kv_mxfp4` (`:599-603`), `prefill_chunk_size` (`:615`), `decode_nvfp4` (`:621`), `min_kv_tokens` (`:628`), plus `chat_template` (`:751`). Every call site passes an empty object or takes the default: `handlers.cpp:471`, `:518`, `:525` pass `json::object()`; `handlers_admin.cpp:138` and `main.cpp:134` omit the argument. `rg '"config"' tools/imp-server/*.cpp` = 0 hits, so no request body populates it.
Current: a twelfth config surface exists in code, is parsed, and can never be non-empty.
Expectation: the debt ledger's `calibration.out_path` entry (item 7, closed #1508) is the same shape - "parsed, documented, and never read" - and was closed by wiring it, on the `log_set_level` precedent. Here nothing documents it, so the cheaper answer is deletion.
Delta: ~40 LOC of dead parameter threading through 5 call sites, and one more place a future reader must check when tracing where a setting comes from.
Cost: delete the parameter and the 11 `overrides.value(...)` reads, or wire it to the admin model-swap endpoint if per-model config was the intent. 1 file, ~40 LOC. Risk: low; `handlers_admin.cpp:138` is the only caller with a plausible use.
Falsifier: "an admin or model-swap endpoint fills it from the request body" - **checked, n**. All five call sites listed above pass empty; no JSON parser in `tools/imp-server/*.cpp` produces the object.

---

### Checked and NOT a finding

- **S-23 (zero virtual dispatch in `exec`/`compute`) holds.** All 6 `exec` and all `compute` matches for `virtual|override` are comments or config-key names. The only real hierarchies are `Backend` (3 impls) and `HostPinnedAllocator` (1 impl) in `src/memory/`, both by design.
- **The one virtual call reachable during serving is `Backend::commit_range` -> `do_commit_range`** (`src/memory/kv_cache.cu:529-530`), on the growable-KV growth path, next to a driver VMM call. Not a hot-path cost.
- **No file-level `#include` cycle exists anywhere** in `src tools tests include fuzz` (Tarjan over 842 files, 0 SCCs of size > 1).
- **`try_fa2_fp16qk_prefill` results are never discarded.** The two statement-shaped matches at `executor_attention_prefill.cu:310,425` are `if`/`else if` continuation lines; the capture-replay arm at `:302-306` even throws when FA2 declines.
- **`try_launch_async_graph_loop`'s discarded return at `engine_scheduler.cpp:1973` is correct** - `Engine::step_decode_process_outputs` is `void` and declining falls through to the eager decode. The asymmetry with `engine_spec_ngram.cpp:476` (where it is tested) is a readability wrinkle, not a defect.
- **`weight_cache_write`'s discarded return** (`weight_upload.cu:2782`) is documented "Best-effort" at `:2779-2781`.
- **`CONTRIBUTING.md:97` vs `docs/internals/CPP23.md:53` and root `CLAUDE.md`.** CONTRIBUTING says "Errors return codes (`ImpError` / `bool`); CUDA errors are checked and logged, not thrown"; 75 `throw` sites exist in `src/` and S-22 records one as load-bearing. Reading the clause as scoped to *CUDA* errors makes it true (`IMP_CUDA_CHECK_*` at `core/logging.h:83-126` all log or return, none throw). Filed as a wording wrinkle, not a contradiction.
- **`std::expected` adoption is not inconsistent, it is scoped.** `CPP23.md:45` says it replaces `bool f(..., T& out, std::string& err)`; that signature has 4 residual matches in the tree, 2 of which are genuine predicates. The migration hit its stated target.
- **`imp.conf.example` contains no stale keys** - it is a strict subset of the 225 registered.
- **`--set` with an unknown key is a hard error**, not a warning (`config.cpp:571-580`, both tool mains `exit(1)`), exactly as `src/runtime/CLAUDE.md` promises. Unknown keys in `imp.conf` warn (`config.cpp:557-558`). The asymmetry is deliberate.
- **21 of the 22 `IMP_*` names in `docker-entrypoint.sh` have no C++ reader, and that is correct** - the shell translates them into CLI flags. The one C++ reader the entrypoint does not know about is `IMP_WORKER_TIMING` (`tools/imp-server/batching_engine.cpp:182`).
- **All 16 `ModelArch` enumerators have a producer.** `QWEN35_MOE` and `NOMIC_BERT` are GGUF-string-table-only; the second is by design (SafeTensors refuses encoder-only archs at `hf_config_loader.cpp:90-93`).
- **`src/lora/` is fully reachable** (`--lora` -> `imp_lora_load` -> `LoraAdapter::load`, per-request `"lora"` field, `executor_lora.cu`, `tests/test_lora.cpp`). No `imp-cli` flag, so it is server-only.
- **Both vision encoders are selected** (`engine_workspace_warmup.cpp:193-195` and `:199-206`), on disjoint conditions.
- **No `tools/` binary is built-by-default and documented nowhere.** `imp-bench` is absent from `README.md` but present in `docs/internals/KERNELS.md` and `docs/usage.md`.
- **`dequant_int8_fp16` / `dequant_int4_fp16`** (`src/quant/dequant_int8.cu:71`, `src/quant/dequant_fp16.cu:49`) have zero callers in `src/` or `tools/`; their only callers are `tests/test_quant.cu:175,242,298`. Same class as the `fp32_accum_add_fp16_kernel` the debt ledger closed 2026-08-21. Two files, ~120 LOC. Recorded rather than filed because the class is already named.
- **`out-of-range `ImpDType` is safe** - `map_dtype`'s `default:` returns `QType::F16` (`imp_api.cpp:167-168`). The problem in G-4 is in-range-but-invalid, not out-of-range.
- **`check_filesize.py` and `check_function_size.py` both report `violations=0` today**, and `check_function_size.py --selftest` is 10/10. G-6 is a parser gap, not a red gate.

---

### Known-and-accepted (restated)

- No GPU CI lane; `make verify-fast` locally is the only thing that runs a CUDA kernel against correctness or perf (SETTLED F-5, owner decision 2026-08-03). Every finding above that would need a runtime check inherits this.
- `GraphExecutor` is intrinsically forward-pass-coupled; do not re-attempt runner classes (SETTLED B). The `engine.h` god-header verdict stands as F-24 wrote it: extraction refuted on churn, pimpl priced at a 42 % ceiling.
- `src/exec/` per-TU compile floor is ~5 s and header-driven, which is why S-32 refused the `run_attention` split. Any split proposal in P0 above inherits that floor.
- Split on conflation, never on size (root `CLAUDE.md`, SETTLED B). The three splits named in P0 are conflation-based.
- `quant/nvfp4_gemm.cu:3 -> compute/gemm.h` stays: inverting it moves a dispatch decision on a GEMM path, unmeasurable without a GPU (SETTLED, 2026-08-03).
- Untested quant formats Q4_1/Q5_0/Q5_1/Q2_K/Q3_K/Q8_K and FP8 E5M2 (`docs/LIMITATIONS.md`). G-8 is about `Q8_1`, which is on none of those lists.
- `process_diag` is a process-wide config snapshot; two Engines with different diagnostics settings in one process fight over it (`docs/internals/ARCHITECTURE.md`, Known limitations). G-1's proposed move does not change that.

---

### Open questions

- Does an `executor.h` edit actually rebuild 79 TUs in wall-clock terms, given the ~5 s `src/exec` floor S-32 measured? Needs one timed `make dev` on a touched header.
- Are the six dead `Overrides::Gemma4` branches (G-7) worth wiring or deleting? The answer is a Gemma-4 A/B on each flag, which needs the card.
- Does `kv_cache_dtype = QType::FP4_E2M1` through the C API produce garbage or an early failure on hardware (G-4)? The dispatch path says garbage; only a run settles it.
- Should `runtime/pdl.h` and `runtime/process_diag.h` move to `core/`, or should `core/` gain a `diag/` subdirectory? The first is mechanical; the second is a naming decision for the owner.
- Is there an external consumer of `include/imp/` outside this repo (G-3)? Only the owner knows, and it changes the severity.
- Should a layering gate exist at all (`src/exec/**` must not include `runtime/**`)? There are 13 static gates and none checks include direction; G-9 is what its absence costs.


## Axis H - Build, CI, supply chain

Repo: <repo>, branch `perf/engine-h-fanin-cut-and-attention-split-verdict`, HEAD ef664dd8, clean. READ-ONLY, no build, no GPU job.

### Coverage

**Read in full**
`.github/workflows/ci.yml` (851), `.github/workflows/auto-merge.yml`, `.github/workflows/release-docker.yml`, `.github/workflows/roofline.yml`, `.github/dependabot.yml`, `Dockerfile` (164), `CMakePresets.json`, `cmake/CompilerFlags.cmake`, `cmake/imp-deps.cmake`, `scripts/ci_static_gates.sh`, `scripts/check_dep_pins.sh`, `scripts/bench_gate.sh`, `scripts/dep_build_args.sh`, `scripts/check_ptx_fallback.sh` (30-80), `.git/hooks/pre-push`, `.git/hooks/pre-commit`, `tools/roofline/rl_regress.py`, `tools/roofline/config.json`, `tests/perf_baseline.json`, `tests/perf_baseline_chunked.json`, `docs/internals/BENCHMARKING.md`, `.claude/skills/building-and-testing/SKILL.md`, `.claude/skills/benchmark-cuda/SKILL.md`, `.claude/skills/shipping-prs/SKILL.md`, `LICENSE`, `third_party/stb/` headers (heads + licence sections).

**Sampled**
`CMakeLists.txt` (1-140, 283-287, 460-482, 520-535, 630-645, plus grep for FetchContent/RDC/LTO), `Makefile` (1-25, 45-130, 361-441, 501-535, 565-580), `scripts/verify.sh` (100-250, 330-620, plus greps; 853 total), `scripts/check-release.sh` (header + section index; 529 total), `docker-entrypoint.sh` (grep only), `docs/PERF.md` (1-70 of 192), `tools/roofline/README.md` (1-80), `docs/audit/SETTLED.md` section G, `docs/LIMITATIONS.md` 35-75, `docs/audit/PERF_LOG.md` (grep), `CHANGELOG.md` (grep), `tests/perf_baseline_north_star.json` (1-25).

**Skipped** (out of axis): `src/**` except the attribution scan, `tests/**` except `test_entrypoint.sh` by reference, `tools/roofline/rl_*.py` other than `rl_regress.py`, `docs/BENCHMARKS.md` (grep only), `webui` (checked only for bundled third-party assets).

### Brief vs repo

| Axis-question premise | Repo |
|---|---|
| "`CMakePresets.json` exists?" | Yes, `CMakePresets.json` (74 lines, 5 configure presets, 4 build presets, 2 test presets). Referenced from exactly one place, `Makefile:573` (`make tidy`). Neither CI, `make build`, `make dev` nor the Dockerfile use `--preset`. |
| "is `tools/roofline/` wired to ANY gate or hook, or a manual tool" | Neither cleanly. `.github/workflows/roofline.yml:51` runs `rl_regress` against `history/BASELINE`, so it is not purely manual, but the workflow's only triggers are `paths: tools/roofline/**` (`:13-19`), it is not a required check, and measurement is manual (`Makefile:413-424`, `check-gpu` prerequisite). See H-2. |
| "no `NOTICE`/`THIRD_PARTY_LICENSES` file (none at root: verify)" | Confirmed: `find -iname 'LICEN*' -o -iname 'NOTICE*' -o -iname 'THIRD*'` returns only `./LICENSE`. But `third_party/stb/` exists (2 headers, licence text carried in-file). |
| "`scripts/verify.sh:358` '0.09 % back-to-back, 4.01 % same binary hours apart'" | Verbatim at `scripts/verify.sh:357-358`. Repeated at `:493-495`. |
| "the branch ruleset requires only `Build`" | Cited as given (not verifiable read-only). Consistent with `ci.yml:105-108`, `ci.yml:338-339`, `scripts/ci_static_gates.sh:5-9`, `shipping-prs/SKILL.md` rule 4. |
| "`Sanitizers`/`Mock API`/`Lint`/`File size` etc. are NOT required checks" | True of the **jobs**. NOT true of the **gates**: `ci.yml:124-125` runs `bash scripts/ci_static_gates.sh` with **no argument**, and `ci_static_gates.sh:62-64` turns an empty filter into `SELECT_ALL=1`. So filesize, lanes, entrypoint, alloc, kernels, launchguards, docs, citations and hygiene all block the required check. Only `Lint` (incl. the dependency-pin gate), `Mock API`, `Real API`, `clang-tidy`, `Sanitizers`, `PTX fallback` and `Roofline` are genuinely advisory. |
| "the brief's '2.6x autotune spread' is SETTLED F-9" | Restated, not re-derived. `SETTLED.md` G: F-9 FIXED by repairing the estimator (#1228, `src/compute/gemm.cu`), 4/8 shapes stable before and **7/8** after, R-16 (persist the algo) REJECTED. The 2.6x figure survives only as a 2026-05/06 snapshot in `docs/GOAL.md:117`, which that entry refutes. |
| "F-5 GPU lane declined" | `SETTLED.md` G: declined by the repo owner 2026-08-03, job and nightly trigger stay dormant. Consequence (their words): `make verify-fast` locally is the only thing that ever runs a CUDA kernel against correctness or perf. |

### Findings

### [H-1] The only dependency-pin check lives in a job nothing requires, and its offline half is hermetic
Axis: H   Sev: S1   Confidence: high
Evidence: `ci.yml:460-461` puts `bash scripts/check_dep_pins.sh --online` in the `Lint` job. `Lint`'s other step ends `exit 0` unconditionally (`ci.yml:453`, advisory clang-format), so a red `Lint` means exactly one thing: a broken pin. `Lint` is not a required check. `auto-merge.yml:37` arms `gh pr merge --auto --squash` on every non-draft owner PR, and the squash fires the instant `Build` is green. `scripts/ci_static_gates.sh:23-30` records the reason for the exclusion: "apt-installs clang-format and hits the network for upstream dependency tags. Adding an apt install and a network call to the one required check trades enforcement for flakiness."
Current: a PR that drifts `Dockerfile` ARG defaults away from `cmake/imp-deps.cmake`, or that pins a tag that does not exist upstream, merges with a red X. That is the exact 2026-08-14 defect the script was written for (`check_dep_pins.sh:6-18`: CUTLASS pinned `4.7.0` where every upstream tag carries `v`; cold builds died, cached builds stayed green).
Expectation: the drift half is a string comparison over two files in the checkout. `check_dep_pins.sh` gates the network behind `ONLINE=1` (`:31-32`, `:75`); with no flag it runs sed/grep only (`:39-60`, `:68-72`, `:87-89`). Hermetic, deterministic, sub-second: it meets `ci_static_gates.sh`'s own stated admission criteria (`:36-37`).
Delta: the stated objection applies to `--online` only, and it was used to exclude the whole gate. The class of failure that reaches `main` is the one a Docker layer cache hides for weeks.
Cost: one `if want deps` block in `scripts/ci_static_gates.sh` calling `bash scripts/check_dep_pins.sh` (no `--online`), ~6 LOC, 1 file. `Lint` keeps the `--online` variant. Risk: none - the check reads two tracked files and writes nothing. If wrong: a false red on the required check when someone edits `imp-deps.cmake` without the Dockerfile, which is precisely the intended behaviour.
Falsifier: `check_dep_pins.sh` needing network or a non-POSIX tool in the offline path. Checked y - lines 39-60 use `sed`/`grep` on `cmake/imp-deps.cmake` and `Dockerfile`; the only `git ls-remote` is inside `if [ "$ONLINE" = "1" ]` at `:75-83`.

### [H-2] The roofline regression gate is structurally unable to observe the code that would move it
Axis: H   Sev: S2   Confidence: high
Evidence: `.github/workflows/roofline.yml:12-19` triggers on `pull_request`/`push` with `paths: - "tools/roofline/**"` only. `:51` runs `python3 roofline.py regress --baseline "$(cat history/BASELINE)" --run latest --threshold 5` - both operands are committed JSON under `tools/roofline/history/`. Measurement is manual and GPU-gated: `Makefile:413-419` (`roofline-measure`, `roofline-pin`, both `check-gpu`). Given facts: `history/index.jsonl` holds 14 runs, newest `1d5b9230_20260831_180644`; `history/BASELINE` = `dca16b71_20260806_041710`. `tools/roofline/README.md` bills `regress` as "**Gate**". `Roofline` is not a required check.
Current: a `.cu` change triggers no roofline run, produces no new history entry, and is compared against nothing. The workflow only fires when someone edits the harness itself, at which point it re-parses a 5-day-stale run against a 25-day-older baseline.
Expectation: engines without a GPU CI lane (this repo's declined F-5) keep kernel-level regression out of CI too - vLLM's kernel benchmarks are a manual `benchmarks/` suite. The difference is that they are not named a gate.
Delta: the per-kernel signal that would catch a `%-of-roofline` regression exists, is versioned (`config.json` `config_version: 5`, `rl_regress.py:11-17` refuses cross-version comparison), has a restart-variance guard (`rl_regress.py:36-40`, fail only when the current max is below the baseline min), and never runs on the change it would judge.
Cost: 1 line in `docs/LIMITATIONS.md` to stop calling it a gate; a real gate needs the declined GPU runner. Risk: none for the doc fix.
Falsifier: a push touching only `src/` triggering the workflow, or a hook invoking `roofline-regress`. Checked y - the paths filter is on both triggers; `grep roofline` over `.git/hooks/`, `scripts/`, `.github/` finds only the three `Makefile` targets.

### [H-3] The perf gate's threshold sits inside the host's own same-tree spread, and no red has ever been a real regression
Axis: H   Sev: S2   Confidence: high
Evidence: `tests/perf_baseline.json` `thresholds`: `decode_regression_pct: 8`, `prefill_regression_pct: 8`, `vram_increase_pct: 10`. Measured spread on this box, from the repo's own artifacts: three runs back to back 0.09 %, the same binary hours apart 4.01 % (`scripts/verify.sh:357-358`); six quiet runs 278.59-289.77 (`scripts/verify.sh:482-484`); cross-day 287.63 vs 276.92 = -3.58 % (`docs/PERF.md`, PROV commit=2230e1c2); the same tree failed at -3.25 % and passed at +0.90 % on one day (`scripts/verify.sh:484-486`). The maintainer's release-day figures on one tree: 294.53 vs 277.73 tg128 = **-5.71 %**. `docs/internals/BENCHMARKING.md` "The gate": "A red gate has never been a regression on its own; the proof is a paired A/B against `main`, alternating the arms." The only demonstrated catch is the split-K mutant M29 at **-36 %** (same section; `docs/audit/MUTATION_BASELINE.md:85` records M29 as correctness-neutral with "no perf test in the suite sees it" before #1309).
Documented false negatives: (a) `docs/internals/BENCHMARKING.md` "Context-dependent changes" - #1270 shipped a split-count heuristic that cost **-7.30 % at 32k on Qwen3-30B-A3B-NVFP4** and passed `verify-fast` at +0.33 %, reverted in #1271; (b) `ci.yml:840-842` - the FP8-disable x FP16-widen interaction took Qwen3-8B Q8_0 tg128 from 284 to 146 and reached `main` because the gate then lived only in the local hook; (c) `docs/LIMITATIONS.md` - the server streaming path is never benched (#1685), so a `tools/imp-server/` change cannot move the pinned numbers by construction.
Current: detection window is roughly [8 %, 36 %]. The band 5.7-8 % is indistinguishable from the box, and everything below 8 % ships silently.
Expectation: a single-arm threshold gate against a weeks-old pin cannot resolve a signal smaller than its host noise. The repo already names the instrument that can (paired alternating A/B against `main` in one session) and already has the harness shape for it (`tools/analysis/two_image_conc_ab.sh`, `scripts/bench_longctx_ab.sh`, which "alternates the arms so host drift hits both equally").
Delta: no automated arm exists. `verify-fast` compares one arm to a calendar-old number; the repo's own doctrine says that is not a measurement.
Cost: a `make verify-ab` that builds `origin/main` into a second image and alternates 3 pairs. ~80 LOC of script plus one extra `make build` (3.5 min) and ~3x the bench wall on the paths that matter. Risk: doubles the pre-push gate for `PERF_RE` diffs; if wrong, the added time buys nothing that the 8 % gate does not already catch.
Falsifier: any recorded case of the single-arm gate catching a real regression below ~20 %. Checked y - grep over `CHANGELOG.md`, `docs/audit/PERF_LOG.md` and `gh pr list --search "perf gate"` (20 results) surfaces #1214 (median-of-trials, added because of a false positive), #1400 (widen 3/5 -> 8/8, "the old 3 % ... failed on docs-only changes") and #1309 (mutation proof at -36 %). No sub-20 % catch.

### [H-4] Three of the four pinned perf baselines are executed by nothing, and one of them is documented as a CI gate
Axis: H   Sev: S2   Confidence: high
Evidence: `tests/perf_baseline_chunked.json` (pinned `2026-07-15T16:47:04Z`) is reachable only through `make verify-chunked` (`Makefile:371-376`). `tests/perf_baseline_north_star.json` (pinned `2026-07-26T03:05:00Z`, Qwen3-14B Q6_K, the `docs/GOAL.md` north-star model) only through `make verify-north-star` (`Makefile:378-385`). Neither target appears in `.git/hooks/pre-push`, `.git/hooks/pre-commit`, `scripts/check-release.sh`, or any workflow (`grep -rn 'verify-chunked\|verify-north-star'` over `*.sh *.yml Makefile *.hook`: hits only in `Makefile`, the two skills, `CHANGELOG.md` and `scripts/repin_baselines_if_median.sh` prose).
Worse, the long-context band: `tests/perf_baseline_north_star.json` `_ttft_note` names `pp8192..pp65536` as "#1022 long-context TTFT gate" (8k=717 / 16k=1632 / 32k=4069 / 64k=11318 ms), and `docs/BENCHMARKS.md:656-658` calls it "the CI TTFT gate band". `scripts/verify.sh:563-565` loops `for PPLEN in 4096 8192 16384 32768 65536` and reads `.metrics.prefill_tps.pp${PPLEN} // empty` **from `$BASELINE`**; the default `$BASELINE` is `tests/perf_baseline.json`, which carries only `pp128`, `pp512`, `pp4096`. So on every real gate run, 8k/16k/32k/64k hit `[ -z "$BL_PP" ] && continue` and are silently skipped.
Current: the chunked-prefill path and the 32K-64K TTFT band have pinned baselines and no automatic consumer; the north-star headline (`tg128_at_ctx_2048`) is measured by hand or not at all.
Expectation: a pinned number with no runner is a comment. Either the target joins `scripts/check-release.sh` (which already runs four model-backed stages and states in its own header why each is irreplaceable) or the "CI TTFT gate" claim comes out of `docs/BENCHMARKS.md`.
Delta: `docs/BENCHMARKS.md` asserts CI coverage that does not exist, which is the class `scripts/docs_lint.py` and the `docs` gate were built to stop.
Cost: adding `verify-north-star` to `check-release.sh` = ~5 LOC and ~5 min of release wall (the model is already the GOAL.md hero); correcting the BENCHMARKS.md sentence = 1 line. Risk: the north-star pin is 41 days old, so the first run may be a red that is really H-3.
Falsifier: a hook, script or workflow invoking either target. Checked y (grep above).

### [H-5] Three configure sites, and the preset that claims to mirror CI does not
Axis: H   Sev: S2   Confidence: high
Evidence: `ci.yml:180-188` configures with default generator (no `-G`), `-DIMP_DISABLE_120F_FALLBACK=ON` and three ccache launchers. `Dockerfile:70-81` configures with `-G Ninja`, no `IMP_DISABLE_120F_FALLBACK` (default OFF, `CMakeLists.txt:47`), `IMP_EXTRA_CMAKE` empty by default (`:68`), no ccache (not in the apt list, `Dockerfile:29`). `CMakePresets.json:29-40` defines preset `ci` with `"displayName": "CI - mirrors the ci.yml Build job"`, Ninja + ccache launchers (inherited from `base`, `:5-16`) and **no** `IMP_DISABLE_120F_FALLBACK`. `Makefile:78-83` (`DEV_CMAKE_ARGS`) is a fourth set, also without ccache.
Current: the preset named after the CI job differs from it in generator, launcher and the one flag whose absence costs +53.1 % device-compile time and doubles every `.cu` diagnostic (`ci.yml:194-197`).
Expectation: a preset exists so one configure line is the truth. Here it is a fourth truth wearing the name of the second.
Delta: a reader who runs `cmake --preset ci` to reproduce a CI failure builds the second gencode CI deliberately skips, and gets a different diagnostic set.
Cost: 1 line - either add `"IMP_DISABLE_120F_FALLBACK": "ON"` to the `ci` preset or rename it. Risk: none. If wrong (CI moves to `--preset ci`), the flag is then in the right place anyway.
Falsifier: `--preset` used by CI or `make build`. Checked y - `grep -rn 'preset'` over `Makefile .github/ Dockerfile scripts/ docs/ .claude/`: `Makefile:573` (`make tidy`) and two doc mentions, nothing else.

### [H-6] Two remote-code fetches in the build path have no integrity check, and one is a moving branch
Axis: H   Sev: S2   Confidence: high
Evidence: `Dockerfile:30-32` - `wget -qO /tmp/cmake.sh https://github.com/Kitware/CMake/releases/download/v4.3.1/cmake-4.3.1-linux-x86_64.sh && sh /tmp/cmake.sh --skip-license --prefix=/usr/local`. No `sha256sum`, no signature (`grep -in 'sha256\|checksum\|gpg' Dockerfile .github/workflows/*.yml` returns nothing). `ci.yml:436-437` - `sudo curl -fsSL https://raw.githubusercontent.com/llvm/llvm-project/release/18.x/clang/tools/clang-format/git-clang-format -o /usr/local/bin/git-clang-format && sudo chmod +x`, i.e. a **branch** ref fetched and made executable on a runner that holds `GITHUB_TOKEN`.
Neighbouring non-determinism, same class: base images are tag-pinned not digest-pinned (`Dockerfile:21`, `Dockerfile:100`, and 5 `image:` lines in `ci.yml`); every `apt-get install` is unversioned (`Dockerfile:28-29`, `Dockerfile:110-113`, `ci.yml:76`, `:133`, `:280`, `:389`, `:582`, `:681`).
Current: the same tree builds from different bytes on different days, and two of those byte sources are fetched over the network without verification.
Expectation: GitHub's own hardening guidance and every 2026-era reproducible-build practice pin remote artefacts by digest and verify them. `release-docker.yml` publishes the result to GHCR as `ghcr.io/kekzl/imp:latest`, so this is a distribution path, not just a dev convenience.
Delta: `check_dep_pins.sh` guards the four FetchContent tags carefully and nothing guards the compiler-adjacent downloads.
Cost: `echo "<sha256>  /tmp/cmake.sh" | sha256sum -c -` = 1 line; `release/18.x` -> `llvmorg-18.1.8` = 1 token; `@sha256:` on the two Dockerfile base images = 2 lines. ~5 LOC across 2 files. Risk: the CMake checksum needs bumping with the version, which is the point.
Falsifier: an existing verification step elsewhere in the build. Checked y (grep above, zero hits).

### [H-7] Apache-2.0 code ships inside an MIT-labelled distribution with no Apache licence text anywhere
Axis: H   Sev: S2   Confidence: med
Evidence: `src/compute/nvfp4_quant_hw.cu:5-7`:
```
// Adapted from thu-ml/SageAttention3 (Apache-2.0 License),
// sageattention3_blackwell/sageattn3/quantization/fp4_quantization_4d.cu.
// Copyright (c) 2025 SageAttention team.
```
followed by an explicit modification list (`:9-15`) and a scale-layout formula transcribed "from lines 245-256 of the upstream file" (`:20-30`). The file is compiled into `libimp` unconditionally in the sm_120 branch (`CMakeLists.txt:285`), therefore into `imp-server`/`imp-cli` and the GHCR image (`release-docker.yml:80-91`). `LICENSE` is MIT, `Copyright (c) 2026 kekzl`. `Dockerfile:105-108` labels the published image `org.opencontainers.image.licenses="MIT"`. `find` for `LICEN*`/`NOTICE*`/`THIRD*` returns only `./LICENSE`.
Current: the only Apache-2.0 obligation met is the "state your changes" one (§4(b)). §4(a) (ship a copy of the License with every distribution, source or object) and §4(d) (reproduce the attribution notices) are not, and the OCI label asserts MIT for a build that is not purely MIT.
Expectation: llama.cpp, vLLM and FlashInfer all ship the licence text plus a `NOTICE`/per-file headers. Apache-2.0 §4 makes the licence text part of the artefact, not of the upstream repo.
Delta: no Apache-2.0 text in the tree or the image; nothing in `scripts/check-release.sh` checks for one.
Also verified in the same scan, and NOT a problem: `third_party/stb/stb_image.h` and `stb_image_resize2.h` carry the dual MIT / public-domain grant in-file (`stb_image.h:45-47` "See end of file for license information", `stb_image_resize2.h:382 LICENSE`), which satisfies MIT redistribution on its own. `src/vision/image_processor.h:23,43` says "Ported from transformers'" (Apache-2.0) but ports two rounding/reshape rules described in prose, not source text.
Cost: `THIRD_PARTY_LICENSES.md` at the root with the Apache-2.0 body, the SageAttention notice and a pointer to `third_party/stb/`; ~210 lines of licence text, 0 code, 0 risk. Optionally one `COPY` into the runtime image and a line in `check-release.sh`. The image label needs `MIT AND Apache-2.0` or the SPDX equivalent.
Falsifier: SageAttention3 not actually being Apache-2.0, or the adaptation being too small to be a derivative work. NOT checked - needs the upstream repo and an owner call. The only evidence in the tree is the file's own header, which asserts both the licence and a transcribed formula.

### [H-8] Every third-party pin in the project is a mutable ref
Axis: H   Sev: S3   Confidence: high
Evidence: `cmake/imp-deps.cmake:10-13` pins four deps by **tag** (`v1.18.0`, `v4.7.0`, `v0.53.0`, `v3.12.0`), never by commit SHA. Consumed by `CMakeLists.txt:94-100`, `:105-116`, `:536-549` with `GIT_SHALLOW TRUE` and no `URL_HASH`; and by `Dockerfile:48-51` as `git clone --depth=1 --branch ${TAG}`. GitHub Actions: 9 unique `uses:` lines, **0** pinned to a 40-hex SHA (`grep -rho 'uses: .*@[0-9a-f]\{40\}' .github/workflows/ | wc -l` = 0); all float on a major tag (`actions/checkout@v7`, `actions/cache@v6`, `docker/build-push-action@v7`). `tests/api/requirements.txt` is `httpx>=0.28.1` / `pytest>=9.1.1` - no upper bound, no lock file (`find` for `requirements*/pyproject/package*.json` returns exactly that one file). `roofline.yml:38` is the sole pinned Python dep (`matplotlib==3.11.0`).
Current: `check_dep_pins.sh --online` verifies only that a tag *resolves* (`:79-81`), not that it resolves to the same commit as last time. An upstream re-tag, or a compromised action release, changes what the published image contains with nothing in the repo moving.
Expectation: SHA-pinned actions plus content-addressed dependency pins are the 2026 baseline for anything that publishes a container. Dependabot (already configured, `.github/dependabot.yml`) rewrites SHA pins with a version comment, so the ergonomics cost is near zero for the actions half.
Delta: the repo pins carefully against *drift between its own two copies* and not at all against *upstream mutation*.
Cost: 28 action pins (mechanical, Dependabot-maintained) + four SHAs in `imp-deps.cmake`. The Dockerfile clone lines need reshaping, because `git clone --branch` takes a ref and not a SHA (fetch-then-checkout, ~4 lines). Total ~40 LOC over 3 files. Risk: the Dockerfile clone rework is the only non-mechanical part, and `check_dep_pins.sh`'s ARG-drift parser (`:53`) would need the new form.
Falsifier: a lock file or recorded SHA anywhere. Checked y - none found.

### [H-9] The perf pin's staleness warning is permanently on
Axis: H   Sev: S3   Confidence: high
Evidence: `scripts/verify.sh:370-376` computes `_age_days` and prints a four-line WARNING when it exceeds 30. `tests/perf_baseline.json` `"timestamp": "2026-07-26T01:48:33Z"` = **41 days** at HEAD's date, and `docs/PERF.md` confirms "Pinned 2026-07-26, thresholds widened 2026-08-13 (#1400)". Every `make verify-fast` / `make verify` run prints it.
Current: the warning fires on every gate run and has for eleven days, so it carries no information.
Expectation: the comment above it (`:353-368`) argues it is "a warning, not a failure" precisely so nobody re-pins to silence it. That reasoning holds for a warning that is usually quiet, not for one that is always on.
Delta: the repo's own record of what always-on gate output costs is #1664/#1666/#1689 (three wrong gate numbers posted in one day).
Cost: re-pin on a healthy day (`make gen-perf-baseline`, one GPU session, no LOC) or raise the threshold with a reason (1 line). Risk: re-pinning on a depressed day bakes in a low number - `benchmark-cuda` STOP #4 covers this.
Falsifier: the pin having been refreshed since 2026-07-26. Checked y - `docs/PERF.md` and the JSON both say 2026-07-26; the file's 2026-08-13 mtime is the `#1400` threshold edit.

### [H-10] Device LTO was never tried and never recorded, on a build that pays full RDC
Axis: H   Sev: S3   Confidence: high (on the absence) / HYPOTHESIS (on the gain)
Evidence: `grep -rniE '\blto\b|dlink-time|link.time optimi'` over `*.md *.cmake *.txt *.yml *.sh` = **zero hits**, including `docs/archive/`, `docs/audit/`, `CHANGELOG.md` and the roadmap lever ledger. Meanwhile the prerequisite is already paid: `CUDA_SEPARABLE_COMPILATION ON` + `CUDA_RESOLVE_DEVICE_SYMBOLS ON` on `imp` (`CMakeLists.txt:475-479`, comment "Enable separable compilation for device-side launches (PDL)"), on `imp-bench` (`:528-529`) and on every test target (`:637-638`). Release device flags are `-O3 --use_fast_math --extra-device-vectorization -Xptxas -O3` (`cmake/CompilerFlags.cmake:19`); no `-dlto`.
Current: relocatable device code without device LTO, i.e. the compile-time cost of separate compilation with none of the recovery. Whether that matters here is unmeasured.
Expectation: unclear. `-dlto` is nvcc's documented recovery for RDC-lost cross-TU device inlining; on a codebase whose hot kernels are large and mostly self-contained it may be worth nothing. **HYPOTHESIS - no number in this tree either way.**
Delta: the gap is the missing record, not a known loss. Every other build lever here carries a measurement (`-lineinfo` cost "~2x decode and ~4x prefill" at `CompilerFlags.cmake:20-26`, the second gencode "+53.1 % device-compile time" at `ci.yml:194-196`).
Cost to test: add `-dlto` to `IMP_SM120_FLAGS` and the device link, one `make build`, then `make kernel-resources` (reads `libimp.a`, no GPU) and one `make verify-fast`. ~2 LOC to try, ~20 min. Risk: `-dlto` conflicts with `-G`, so the Debug/sanitizer configs must be excluded, and the register/spill pin `tools/kernel_resource_baseline.txt` would move - which is also the instrument that would say whether it did anything.
Falsifier: an existing memo or PR that measured it. Checked y - zero grep hits repo-wide.

### Checked and NOT a finding

- **The static gates DO block the required check.** `ci.yml:124-125` runs `scripts/ci_static_gates.sh` with no argument; `:62-64` of that script turns an empty filter into `SELECT_ALL=1`, so filesize, lanes, entrypoint, alloc, kernels, launchguards, docs, citations and hygiene all fail `Build`. The named jobs re-run subsets purely for the check name (`:11-14`).
- **`Kernel resources` cannot pass on an empty pipe.** `ci.yml:211-213` pipes `cuobjdump -res-usage build/libimp.a` into `tools/kernel_resources.py -`; under `sh -e` a pipeline takes the last command's status, so a `cuobjdump` failure would only be caught by the script. Verified: `printf '' | python3 tools/kernel_resources.py -` prints "no resource records in the input" and exits **2**.
- **The `kernels` group inside the pre-build static-gates step correctly skips** (no `build/libimp.a` yet, `ci_static_gates.sh:104-116`); the post-build `Kernel resources` step is what actually runs it.
- **The shipped image keeps the compute_120f PTX fallback.** `CMakeLists.txt:47` defaults `IMP_DISABLE_120F_FALLBACK` OFF; `Dockerfile:66-81` passes no override and `IMP_EXTRA_CMAKE` is empty; `release-docker.yml:88-89` adds only `IMP_BUILD_BENCH=ON`. Only the CI `Build` job strips it (`ci.yml:185`).
- **`scripts/check_ptx_fallback.sh` cannot be fooled by the CI opt-out.** It reads `CMakeCache.txt` for `IMP_DISABLE_120F_FALLBACK:BOOL=ON` (`:45-50`) instead of inferring from the image count, and treats "zero PTX images without that flag" as FAIL (`:54-60`), which is the direction that matters.
- **The `--mount=type=cache` rule is honoured.** `grep -rn 'mount=type=cache'` finds two hits, both prose (`CLAUDE.md:112`, `AGENTS.md:84`). The Dockerfile uses none. `release-docker.yml:92-93` uses `type=gha` layer cache in the release lane only.
- **`--use_fast_math` is a recorded decision, not drift.** `cmake/CompilerFlags.cmake:19` sets it for Release and `:26` for RelWithDebInfo; `docs/determinism.md:242-247` names it as part of the determinism envelope, added in #1576.
- **`-lineinfo` is correctly RelWithDebInfo-only** with its measured justification (`CompilerFlags.cmake:20-26`: the CMake default `-O2 -g` cost ~2x decode and ~4x prefill). `docs/internals/BENCHMARKING.md` points profiling at the `relwithdebinfo` preset.
- **Warning flags are scoped correctly.** `-Wall -Wextra -Wpedantic` live on the `imp_warnings` INTERFACE target under `$<COMPILE_LANGUAGE:CXX>` (`CMakeLists.txt:27-29`), linked PRIVATE by first-party targets only, after a recorded incident of them leaking onto gtest/CUTLASS (`CompilerFlags.cmake:4-8`).
- **ccache is CI-only and that is deliberate.** `ci.yml:58-62,186-188,146-160` configure and cache it; the Dockerfile installs no ccache (`:29`) and `make dev` sets no launcher (`Makefile:78-83`), keeping the image build hermetic. `CMakePresets.json:12-14` sets launchers for the one preset consumer (`make tidy`).
- **HYPOTHESIS REFUTED: "the `clang-tidy` job dies on a bashism because the image ships no bash."** `ci.yml:669-673` asserts `nvidia/cuda:13.3.1-devel-ubuntu26.04` ships no bash, and the `tidy` job (`:264-330`) neither declares `shell: bash` nor installs bash while using `mapfile` (`:313`), process substitution and arrays, under `continue-on-error: true`. But `ci.yml:290-294` records that before #1626 the step **printed "nothing to lint"** - a line at `:314`, downstream of the `mapfile`. bash executes there. (The two notes contradict each other; see Open questions.)
- **`docker-entrypoint.sh` is gated.** `tests/test_entrypoint.sh` runs in the `entrypoint` group (`ci_static_gates.sh:89-92`), which the unfiltered `Build` step selects.
- **`check_dep_pins.sh` is bidirectional.** It fails both on a cmake pin with no Dockerfile ARG (`:68-69`) and on a Dockerfile ARG with no cmake pin (`:87-89`), so a fourth source of truth cannot appear silently.
- **Dependabot covers all three ecosystems** (github-actions, docker for `/` and `/tools/roofline`, pip for `/tests/api`) weekly, with a recorded reason for excluding `/tools/analysis` (`dependabot.yml:14-18`, run 29214173834).
- **The moving image tags are guarded.** `release-docker.yml:49-66` compares the published tag against every non-draft release with `sort -V` before letting `latest`/`{{major}}` move, after v0.22.0 took `latest` from v0.25.0.
- **The two perf-gate scripts now measure the same quantity.** Both pass `--set speculative.ngram=false` (`bench_gate.sh:40`, `verify.sh:426`, `:505`, `:635`, `:698`, `:701`), closing the #1625 divergence documented in `docs/internals/BENCHMARKING.md`.
- **The depressed-host FAIL->WARN degradation is bounded, not an escape hatch.** It fires only on a sampled signature (mem median <13801 MHz OR power max <400 W OR SM median <2000 MHz, `verify.sh:145-147`, `:216-221`), drops the first 2 cold-ramp samples, and fails **open** (plain FAIL) with fewer than 3 samples (`verify.sh:186-190`).
- **A gate run that measured nothing cannot report OK.** `MODEL_GATES_RUN` (`verify.sh:113-117`) plus the `#1474` summary rule; `scripts/check-release.sh:40-49` carries the same rule for `SKIP_VERIFY=1`.
- **No bundled third-party JS/CSS in the WebUI.** `tools/imp-server/webui/index.html` (49 KB, embedded by `cmake/embed_webui.cmake`) contains no CDN reference and no third-party copyright line.
- **`rl_regress.py` refuses invalid comparisons.** `config_version` mismatch raises rather than silently passing (`:11-17`, after "the 06-11 CI red was a v3 run gated against the long-stale v2 pin"), and a drop only fails when the restart ranges are disjoint (`:34-40`).
- **Docs-only PRs still pay the static gates.** The `Build` job's `Detect non-docs changes` (`ci.yml:90-103`) fails open to `code=true` on any uncertainty, and the gate step carries no `if:` (`ci.yml:124-125`), so the docs and release-hygiene gates - the ones a docs change can break - always run.

### Known-and-accepted (restated)

- No GPU CI lane (`SETTLED.md` G, F-5, declined by the owner 2026-08-03). Consequence: `make verify-fast` locally is the only thing in the project that runs a CUDA kernel against correctness or perf; the `Test` job with the full ctest, compute-sanitizer and `bench_gate.sh` stays dormant behind `vars.HAS_GPU_RUNNER`.
- No correctness gate against a reference implementation (#1571) and no soak test (#1642) - both blocked on the same absent runner (`docs/LIMITATIONS.md`).
- The generation half of the HTTP contract is deselected in CI (`Real API contract (model-less)`, #1600/#1559) and runs only in `make test-server`.
- The server streaming path is never in the perf gate (#1685): the gate benches `imp-cli`, which never enters the SSE writer.
- cuBLASLt algorithm selection is unpinned by design; `SETTLED.md` G, F-9 FIXED via the estimator (#1228), R-16 (persist) REJECTED with the two refuted alternative designs recorded.
- `main`'s reported CI status is stale by construction (auto-merge squashes as `GITHUB_TOKEN`, which starts no run); `workflow_dispatch` exists as the manual refresh (`ci.yml:8-18`).

### Open questions

- Is `thu-ml/SageAttention3` in fact Apache-2.0, and is `src/compute/nvfp4_quant_hw.cu` a derivative work requiring §4(a)/(d) compliance? Needs the upstream repo and an owner decision (H-7).
- Does anything real live in the 5.7-8 % band the perf gate cannot resolve, or is that band structurally noise on this box? Needs a paired alternating A/B against `main` on a healthy day (GPU).
- Would `-dlto` move `tools/kernel_resource_baseline.txt` at all? One `make build` plus `make kernel-resources` (CUDA toolkit, no GPU) settles it (H-10).
- `ci.yml:669-673` asserts the CUDA ubuntu26.04 image ships no bash; `ci.yml:290-294` records bashisms executing in the same image in the `tidy` job. One of the two notes is wrong, and which one decides whether the `Sanitizers`/`PTX fallback` bash installs are load-bearing or cargo. Needs an image pull.
- Should `perf_baseline_chunked.json` be deleted rather than gated? It is 52 days old, its own comment records that a model was "silently skipped for months" before being dropped, and no consumer exists (H-4).


## Axis I - Tests (scout report)

Repo `<repo>`, branch `perf/engine-h-fanin-cut-and-attention-split-verdict`, HEAD `ef664dd8`, clean. READ-ONLY: no edits, no build, no GPU job. Every number below comes from a command run in this session or from a cited file line.

### Coverage

**Read in full**
- `docs/audit/SETTLED.md` (652 lines), `tests/CLAUDE.md`, `tests/README.md`, `docs/audit/TEST_INVENTORY.md`, `docs/audit/MUTATION_BASELINE.md`, `tools/check_test_lanes.py`, `tools/check_unit_skips.py`, `scripts/check_e2e_lane_split.sh`, `scripts/check_verify_filter.sh`, `scripts/check_det_suite_filter.sh`, `scripts/pre-push.hook`, `scripts/pre-commit.hook`, `tests/api/pytest.ini`, `tests/api/requirements.txt`.
- `CMakeLists.txt:640-1232` (test modules, lane aggregates, ctest guards, fuzz targets).
- `Makefile:1-395` (all `test-*`, `bench*`, `verify*` targets).

**Sampled (targeted line ranges / greps)**
- `docs/audit/TEST_HARDENING_LOG.md` (headings + iterations 1 and 14), `docs/LIMITATIONS.md:26-68`.
- `scripts/verify.sh:280-360, 730-853`; `scripts/test_server.sh:90-125`; `.github/workflows/ci.yml` job list + `real-api` job (`:369-417`).
- Test bodies: `test_e2e_models.cpp`, `test_degeneration.cpp`, `test_determinism_e2e.cpp`, `test_e2e_greedy_lock.cpp`, `test_e2e_llm_compressor.cpp`, `test_prefix_cache_e2e.cpp`, `test_spec_capture_fidelity.cpp`, `test_forward_pass.cu`, `test_chunked_prefill.cu`, `test_lora.cpp`, `test_sampling.cu` (names only), `tests/api/test_perf_regression.py`, `tests/api/test_serving_kpi.py`.
- `tools/mutation/run.py` (`main`, `--verify-anchors`), `tools/mutation/catalogue.json` (aggregated).

**Skipped**
- Bodies of the ~200 remaining test files (counted, not read). `tools/mutation/run.py` execution paths beyond `--verify-anchors`. `loop/` (not committed, absent). `docs/audit/ESCAPE_ANALYSIS.md`, `docs/audit/DEBT_LEDGER_2026_08_21.md`.

**Commands whose output this report cites**
```
python3 tools/check_test_lanes.py --report        # 1617 laned / 1084 unlaned / 2701 total
python3 tools/mutation/run.py --verify-anchors    # anchors: 56/56 match exactly once
rg -c '^TEST' tests/*.cpp tests/*.cu             # per-file macro counts
md5sum .git/hooks/pre-commit scripts/pre-commit.hook   # identical
```

---

### Brief vs repo

| Axis question said | Repo says | Evidence |
|---|---|---|
| "pytest-rerunfailures in `requirements.txt`" | Not present. `tests/api/requirements.txt` is two lines: `httpx>=0.28.1`, `pytest>=9.1.1`. No rerun/flaky plugin anywhere. | file content |
| "`tests/api/conftest.py` `flaky`/`rerun`" | Neither word occurs. The only retry-shaped code is `wait_for_server` polling `/health` to a deadline (`conftest.py:34-45`). | `grep -n 'flaky\|rerun\|retry\|attempt' tests/api/conftest.py` -> 0 hits |
| "`test_perf_regression.py` tolerance widening" | No widening mechanism. It *reads* `thresholds.decode_regression_pct` from `tests/perf_baseline.json` so it shares one source with `verify.sh` (`test_perf_regression.py:57-60, 203-205`). The module fallbacks (5 % TTFT, 3 % decode) are *tighter* than the baseline's 8 %. | file |
| "retries in `scripts/verify.sh`" | None. `rg -n 'retry\|attempt\|RETRY' scripts/*.sh` -> 0 hits. `--gtest_repeat` occurs once in the tree, inside a comment (`tests/test_e2e.cpp:411`). | grep |
| "`tests/api/test_serving_kpi.py`" implied to be an E2E quality gate | It is a pure-arithmetic unit test of `tools/analysis/serving_kpi.py` (percentiles, histogram quantiles, `/metrics` parsing). Its own docstring: *"Lane-agnostic: no request is made."* 6 test functions, no server, no model. | `tests/api/test_serving_kpi.py:1-7` |
| "`scripts/validate_safetensors.py`" implied to be a gate | Exists (42 KB) and has **zero invocation sites** in `Makefile`, `scripts/`, `.github/`. Only mention outside itself is `scripts/check-release.sh:126` naming its *output* file as "written ... not read". | grep |
| `docs/audit/TEST_INVENTORY.md` (2026-08-08 prior) "2 125 gtest cases" | 2 701 `TEST/TEST_F/TEST_P` macros today. | `check_test_lanes.py --report` |
| TEST_INVENTORY §2.3 "the Stage-1 GPU hook is not installed on this box" | Installed. `.git/hooks/pre-commit` is byte-identical to `scripts/pre-commit.hook` (md5 `204b2bc0…`), dated 2026-09-02. Same for `pre-push`. | `md5sum` |
| TEST_INVENTORY §2.2 "61 test cases silently skip in CI" | Closed. `guard_unit_skips` (`CMakeLists.txt:1096-1098`, `tools/check_unit_skips.py`) fails the unit lane on any SKIPPED case; MUTATION_BASELINE records `unit-skips: 0 skipped of 1611`. | files |
| TEST_INVENTORY §2.4 quotes the verify filter as `AttentionTest.*` | Now `*Attention*` (`scripts/verify.sh:309`), changed in #1586. The consequence changed too - see [I-2]. | file |

The memory note *"DetEvalE2ETest.\* matches 0 tests and reports PASSED"* is **guarded**, three ways: `guard_det_suite_filter` (`CMakeLists.txt:1078-1081` + `scripts/check_det_suite_filter.sh`, asserts the `*DetEvalE2ETest*` literal resolves and covers both model rows), `guard_verify_filter` (`CMakeLists.txt:1090-1092`, asserts every pattern in `verify.sh`'s `FILTER=` line matches ≥1 test), `guard_e2e_lane_split` (`CMakeLists.txt:1069-1073`, asserts the unit filter resolves to a frozen 61-name set). All three carry the label `unit`, so they run in CI. The gap is that these guards check *pattern non-emptiness*, not *pattern coverage*, and one filter copy is guarded by nothing - [I-2] and [I-3].

The memory note *"mutation harness: 4 of 56 anchors dead"* is **fixed and still fixed today**: `python3 tools/mutation/run.py --verify-anchors` -> `anchors: 56/56 match exactly once`.

---

### Coverage by subsystem

Counts are `TEST/TEST_F/TEST_P` macros (`rg -c '^TEST' tests/<file>`), not gtest cases (a `TEST_P` expands per value row). Lane column from `check_test_lanes.py`: **unit** = `ctest -L unit` = the required `Build` job; **gpu** = executes only under `make verify-fast` / `make test-gpu` on a card. Mutants from `tools/mutation/catalogue.json` (56 anchors, 8 production files).

| Subsystem | Unit tests (file:macros) | GPU kernel tests | Integration test that loads a model | CI lane | Mutants |
|---|---|---|---|---|---|
| GGUF loader | `test_gguf_loader.cpp`:9, `test_gguf_fault_injection.cpp`:20 | - | `test_e2e_models.cpp` PrimaryModelTest:4 | unit | **4** (`gguf_loader.cpp`) |
| SafeTensors loader | `test_safetensors_loader.cpp`:24, `_writer`:7, `test_llm_compressor_loader.cpp`:45, `test_hf_config_loader.cpp`:21 | - | `test_e2e_llm_compressor.cpp`:4 | unit | 0 |
| Tokenizer | `test_tokenizer.cpp`:56, `_robustness`:20, `test_sentencepiece_loader.cpp`:9 | - | `test_tokenizer_qwen38.cpp`:4 (parity, `make test-gpu`), `test_tokenizer_compat.cpp`:1 (**no target**, [I-1]) | unit + gpu | 0 |
| Chat template | `test_chat_template.cpp`:62, `_goldens`:10, `test_jinja.cpp`:73, `_undefined`:6, `test_qwen38_chat_template.cpp`:6, `test_gpt_oss_harmony_golden.cpp`:4 | - | - (goldens are committed refs) | unit | 0 |
| Weight upload / decode cache | `test_weight_snapshot.cpp`:9, `test_tensor_kind_table.cpp`:14, `_matcher`:8, `test_weight_map_expert_gate.cpp`:5, `test_weight_registry_preservation.cpp`:14 | `test_weight_dispatch.cu`:13 | `test_quant_pipeline.cpp`:1, `test_tensor_kind_coverage.cpp`:1 (**no target**, needs `IMP_TEST_GGUF`) | unit + gpu | 0 |
| GEMM registry | - | `test_gemm_kernel_registry.cu`:41 | - | **gpu only** | 0 |
| FA2 prefill (FMHA) | - | `test_fmha_fp8.cu`:105, `test_attention_fmha_mxfp4.cu`:35, `_fmha_sm120`:24, `_mxfp4`:7, `_hd512`:9, `test_attention_crosspath.cu`:11 | - | **gpu only, and outside `verify-fast` [I-2]** | 0 |
| Paged decode / KV dtype | - | `test_paged_attention.cu`:26 (f16/int4), `test_attention_paged_oracle.cu`:7 (f16/fp8/nvfp4 multitok), `_reduce`:2, `_nvfp4_tc`:2, `_tc_residual`:5, `test_fp8_kv_cache.cu`:11 (fp8+int8), `test_kv_gather.cu`:5 | - | gpu, in `verify-fast` (`*Attention*`) | **18** (`attention_paged.cu`, `attention_paged_common.cuh`) |
| Sparse decode | `test_sparse_attn_geometry.cpp`:7 | `test_sparse_attn_select.cu`:12 | `make test-niah` (opt-in, no workflow) | unit + gpu | 0 |
| MoE routing + grouped GEMM | `test_routing_decision.cpp`:44, `test_expert_placement.cpp`:6, `test_nvfp4_expert_offload.cpp`:8 | `test_moe.cu`:13, `test_moe_executor.cu`:11, `test_expert_cache.cu`:18, `test_cutlass_grouped_*`:6, `test_nvfp4_smallm*`:12 | DetEvalE2ETest `moe` row (gpt-oss-20b) | unit + gpu | 0 |
| GDN / SSM | - | `test_gdn.cu`:17, `test_gdn_batched.cu`:10, `test_ssm.cu`:11 | `test_e2e_models.cpp` GDNModelTest:2 | gpu | 0 |
| KV cache mgr + prefix cache | `test_kv_cache.cpp`:51, `test_kv_accounting.cpp`:5, `test_kv_residual_sizing.cpp`:18 | `test_kv_cache_gpu.cpp`:11, `test_prefix_cache_equiv.cpp`:14, `test_kv_cache_write.cu`:4, `test_kv_block_copy.cu`:3 | `test_prefix_cache_e2e.cpp`:4 (**no target**, [I-1]) | unit + gpu | **11** (`kv_cache_manager.cpp`) |
| Memory plan / arena / growable pool | `test_memory_backend.cpp`:25, `_allocators`:22, `_plan`:20, `test_workspace_sizes.cpp`:28, `test_host_pinned.cpp`:20, `test_graph_slots.cpp`:12, `test_library_reserve_cache.cpp`:8, `test_pre_dequant_budget.cpp`:4 | `test_vram_budget_reserve.cpp`:19, `test_vram_query.cpp`:4 | - | unit + gpu | 0 |
| Scheduler (admission/priority/aging/chunked/ragged) | `test_scheduler.cpp`:30 | `test_prefill_ragged.cu`:3, `test_chunked_prefill.cu`:7 | `test_continuous_batching.cpp`:16 (split), `test_chunked_prefill.cu` (default `/models` paths) | unit + gpu | 0 |
| CUDA graph capture/replay | `test_graph_eligibility.cpp`:4, `test_graph_slots.cpp`:12 | `test_gemm_capture_fp16_sm120.cu`:5, `test_nvfp4_gemm_graph_capture.cu`:3, `test_capture_abort.cu`:2 | Gemma4GraphsTest:2, `test_warm_cache.cu`:2, `test_suspend_resume.cu`:3, `test_spec_capture_fidelity.cpp`:2 | unit + gpu | 0 |
| Spec decode (n-gram / MTP / adaptive k) | `test_ngram_draft.cpp`:8, `test_suffix_draft.cpp`:15, `test_token_recycle_draft.cpp`:17, `test_spec_gates.cpp`:7, `test_mtp_auto.cpp`:12, `test_mtp_presence.cpp`:9 | `test_mtp_feed_batch.cu`:3, `test_mtp_topw.cu`:2 | `test_spec_capture_fidelity.cpp`:2, `test_mtp_forward.cpp`:1, `test_real_checkpoints.cpp`:2 | unit + gpu | 0 |
| Sampling - top-k/top-p/temp/penalties | - | `test_sampling.cu`:25, `test_penalty_hist_append.cu`:1, `test_rowwise_topm.cu`:4 | - | **gpu only**, in `verify-fast` (`SamplingTest.*`) | **3** |
| Sampling - **DRY / mirostat / logit_bias** | **none** | **none** | **none** | **none** | 0 - [I-4] |
| Constrained decoding (JSON schema / regex / GBNF) | `test_json_constrain_property.cpp`:24, `test_schema_constrain_property.cpp`:35, `test_gbnf_grammar.cpp`:25, `test_regex_constrain.cpp`:19, `test_constraint_validation.cpp`:26, `fuzz/` x3 via `test_fuzz_corpus.cpp`:6 | `test_json_constrain.cu`:44, `test_schema_constrain.cu`:31, `test_tool_call_constrain.cu`:13, `test_grammar_regex_mask.cu`:2 | - | unit + gpu | **5** |
| C API | `test_e2e.cpp`:18 (EndToEndTest.* in unit), `test_c_api_enum_binding.cpp`:6, `test_exit_codes.cpp`:4 | - | `test_api_generate.cpp`:4, `test_engine_relaunch.cpp`:2 (default `/models` paths) | unit + gpu | 0 |
| Server HTTP (3 dialects, SSE, tools, cache_control, metrics, auth, rate limit, admin, model swap) | ~373 macros in `test-core`'s `IMP_BUILD_SERVER` block: `test_anthropic_transform.cpp`:71, `test_tool_call.cpp`:64, `test_sse_stream_utils.cpp`:63, `test_server_request_limits.cpp`:27, `test_tool_stream_filter.cpp`:25, `test_constraint_validation.cpp`:26, `test_regex_constrain.cpp`:19, `test_server_auth.cpp`:13, `test_logprobs_shapes.cpp`:13, `test_image_fetch.cpp`:13, `test_stream_reasoning_split.cpp`:13, `test_reasoning_reconcile.cpp`:13, `test_responses_transform.cpp`:11, `test_server_args.cpp`:9, `test_tracing.cpp`:7, `test_base64.cpp`:5 | - | pytest `tests/api/` 122 test fns: mock lane (`IMP_USE_MOCK=1`) + `-m nomodel` against the real binary; real model only via `make test-server` (12 batteries, opt-in) | unit (transforms) + none (handlers) | **5** (`anthropic.cpp`, `utils.cpp`) |
| Vision (2 towers) | `test_vision_preprocess.cpp`:17, `test_qwen3vl_vision_grid/load/config/map`:10+9+7+6, `test_vision_loader_check.cpp`:11, `test_image_placeholders.cpp`:12, `test_mrope_positions.cpp`:11 | `test_deepstack_inject.cu`:5, `test_vision_chunk_offset.cu`:5, `test_qwen3vl_encoder.cu`:2 | `test_vision_golden.cu`:2, `test_qwen3vl_pipeline.cu`:5 (`make test-vision`) | unit + gpu | 0 |
| **LoRA** | **none** | **none** | `test_lora.cpp`:1 (`LoraHotSwap`, default `/models/Llama-3.2-3B-Instruct-Q8_0.gguf`; `IMP_TEST_MODEL_LLAMA` is set by no target) | gpu | 0 |
| Quantizer tool (`imp-quantize`) | `test_quantize_checkpoint_out.cpp`:26, `test_quantize_policy.cpp`:21, `test_quantize_fp8_source.cpp`:10, `test_awq_calibration.cpp`:6, `test_checkpoint_limits.cpp`:12 | `test_quantize_fp16_nvfp4_moe_native.cu`:12 | **none** - no target runs the tool end to end; `scripts/validate_safetensors.py` has 0 invocation sites | unit + gpu | 0 |
| **`imp-bench`** | **none** | **none** | **none** - the only mention of `imp-bench` in `Makefile`/`CMakeLists.txt` is its build target | **none** | 0 |
| `imp-cli` | `tools/imp-cli/args.cpp` compiled into `test-core`; `test_mtp_auto.cpp`:12 (`apply_config_pins`), `test_exit_codes.cpp`:4, `tests/test_entrypoint.sh` (run by `ci_static_gates.sh:91`) | - | the `verify.sh` smoke prompt drives the real binary | unit | 0 |

**Zero-integration-test subsystems** (no test anywhere loads a model and exercises them end to end): GEMM registry, FA2 prefill, sparse decode, memory plan/arena, DRY/mirostat/logit_bias sampling, quantizer tool, `imp-bench`, server admin/model-swap.
**Zero-mutant subsystems** (16 of 25 rows): SafeTensors loader, tokenizer, chat template, weight upload, GEMM registry, FA2 prefill, sparse decode, MoE, GDN/SSM, memory plan, scheduler, CUDA graph, spec decode, vision, LoRA, quantizer.

---

### End-to-end: what actually asserts output quality

| Test / harness | What is asserted | Model | Target | Runs on a push? |
|---|---|---|---|---|
| `PrimaryModelTest.GenerateCoherentOutput` (`test_e2e_models.cpp:63-77`) | substring `"Paris"` + `len > 0` | Qwen3-4B Q8_0 | `make test-e2e` | no |
| `PrimaryModelTest.MultiTurnConversation` (`:100-128`) | substring `"4"` after reset | same | `make test-e2e` | no |
| `PrimaryModelTest.PrefillThenDecodeMultipleTokens` (`:144-164`) | `generated.size() >= 4` - **"does not crash"** | same | `make test-e2e` | no |
| `GDNModelTest.GenerateCoherentOutput` (`:201-244`) | `text.size() > 5` and unique-token ratio `>= 0.30` | Qwen3.5-4B mxfp4 | `make test-e2e` | no |
| `GDNModelTest.MultiTurnGDNState` (`:251-279`) | `text2.size() > 1` - **near-vacuous** | same | `make test-e2e` | no |
| `Gemma4ModelTest.AnswersCapitalOfFrance` (`:321-336`) | substring `"Paris"` | gemma-4-26B Q4_K_M | `make test-e2e` | no |
| `Gemma4ModelTest.RawCompletionProducesOutput` (`:339-355`) | `len > 0` - **"does not crash"** | same | `make test-e2e` | no |
| `DegenerationTest.*` (5, `test_degeneration.cpp:158-243`) | repetition ratio `< 0.5`/`0.6`, 3-gram `< 0.4`, 5-gram `< 0.3`, greedy `out1 == out2`, no leaked special tokens | default `/models/Qwen3-8B-Q8_0.gguf` | only unfiltered `make test-gpu` | no |
| `DetEvalE2ETest` (4 live + 2 `DISABLED_`, `test_determinism_e2e.cpp:144-237`) | greedy `first == second`; perplexity `ppl1 == ppl2` and `ppl1 > 0.0`. **No absolute PPL threshold** | MoE row (gpt-oss-20b) + dense row (Qwen3-4B) | `make test-e2e`, `make test-gpu` | no |
| `GreedyLockTest.FrozenSequences` (`test_e2e_greedy_lock.cpp:133-188`) | frozen 32-token greedy sequence per (model, prompt) - the strongest quality assertion in the tree | locks exist for `Qwen3-8B-Q8_0.gguf`, `Qwen3-8B-NVFP4-cortecs` (`tests/refs/e2e_greedy_locks.h`) | **none** - [I-1] | no |
| `LlmCompressorE2E` (4, `test_e2e_llm_compressor.cpp`) | `"Paris"` x2, `len > 5` x1, `IMP_SUCCESS` x1 | Gemma-4-26B-NVFP4, Mistral, Qwen3-Coder-30B-FP4 | only unfiltered `make test-gpu` | no |
| `SpecCaptureFidelityTest.CachedGraphMatchesEagerForward` (`:106-172`) | `checked >= 200` replays, divergence rate `< 2.0 %` | Qwen3.8-27B-NVFP4-vllm | `make test-spec-fidelity`, `make test-e2e` | no |
| `PrefixCacheE2ETest` (4) | cached vs uncached token-equality | needs `IMP_TEST_MODEL` | **none** - [I-1] | no |
| `tools/analysis/degen_suite.py` | 41 protocol-level degeneration checks, non-zero exit | server model | `make test-server` only (`test_server.sh:118`) | no |
| `make test-niah` (`Makefile:323-333`) | needle retrieved at 16 k/32 k x 3 depths | Qwen3-14B Q6_K | `make test-niah` | no |
| `scripts/validate_safetensors.py` | n/a - **zero invocation sites** | - | none | no |
| `tests/api/test_serving_kpi.py` (6 fns) | percentile/histogram arithmetic; **no server, no model** | - | mock + nomodel lanes | **yes** (as arithmetic) |
| `tests/api/test_perf_regression.py` (2 fns) | decode p50 `>= baseline * (1 - 8 %)`, TTFT p95 (skips when absent) | server model | `pytest -m perf` | no |
| **`scripts/verify.sh` smoke prompt** (`:731-838`) | `<8 distinct tokens in last 32` fails; token run `> 4` fails; 3-gram `> 3` fails; `< 5` tokens fails; NaN/Inf word fails; substring `"Paris"` | Qwen3-4B Q8_0, one prompt | `make verify-fast` | **YES - the only one** |

**What runs on a push**: the `pre-push` hook (`scripts/pre-push.hook`, installed, md5-identical) runs `scripts/ci_static_gates.sh filesize lanes entrypoint alloc launchguards docs citations`, then `make verify-fast` = `scripts/verify.sh fast`, which is:
1. the gtest filter at `verify.sh:309` - **359 of 2 701 macros (13.3 %)**, measured by applying the 12 patterns to every fixture/test name in `tests/`;
2. the perf gate vs `tests/perf_baseline.json` (skipped with `IMP_VERIFY_SKIP_PERF=1` when the diff misses `PERF_RE`);
3. the peak-VRAM gate; 4. graphs-ON vs graphs-OFF decode; 5. **one** smoke prompt.

**Tests that assert nothing, or only "no crash"** (by name):
- `Gemma4ModelTest.RawCompletionProducesOutput` - `EXPECT_GT(len, 0u)` only.
- `PrimaryModelTest.PrefillThenDecodeMultipleTokens` - `>= 4` tokens only.
- `GDNModelTest.MultiTurnGDNState` - `text2.size() > 1` only.
- `LlmCompressorE2E.Gemma4_LoadsWithoutIMA`, `.Modelopt_QwenCoder30B_StillWorks` - load succeeds / `len > 5`.
- `SamplingTest.NaNLogits` - `0 <= token < V` (recorded in TEST_INVENTORY §4, still true at `tests/test_sampling.cu`).
- `TmaBlockScaleBench.BothDescriptorsLaunch` (`tests/test_tma_block_scale_bench.cu:28`) - named as a check, asserts nothing (TEST_INVENTORY §4; the file still holds exactly 1 macro).

---

### Flaky / conditional-execution inventory

| Class | Count | Detail |
|---|---:|---|
| `DISABLED_` tests | **6** (was 4 in the 2026-08-08 baseline) | `DetEvalE2ETest.DISABLED_GreedyReproducibleAcrossFreshContexts` / `DISABLED_PerplexityBitIdenticalAcrossFreshContexts` (documented known limit), `FmhaHd512Test.DISABLED_BenchVsCublas` / `DISABLED_BenchLongCtxFallback`, **`SmallMV2Pair.DISABLED_M1PipelineVsGemvBench`** and **`BatchedSmallM.DISABLED_MarginalRowCost`** (`tests/test_nvfp4_batched_smallm_equiv.cu:285,414`) - both new since the baseline, both benchmarks |
| `GTEST_SKIP` call sites | 204 across `tests/` | top files: `test_attention_fmha_mxfp4.cu` 35, `test_e2e.cpp` 11, `test_gemm_grouped_nvfp4_smallM.cu` 10, `test_cutlass_grouped_ref.cu` 9, `test_chunked_prefill.cu` 8 |
| macro skips | 225 sites | `SKIP_IF_NO_CUDA()` 209, `SKIP_IF_NO_MODEL()` 16 |
| skips inside the CI lane | **0**, enforced | `guard_unit_skips` (`CMakeLists.txt:1096`) reads the three lanes' GTest JSON and fails on any `SKIPPED` |
| unlaned macros | **1 084**, pinned | `check_test_lanes.py PINNED = 1084`; verified matching today |
| `--gtest_repeat` | 0 real uses | one comment, `tests/test_e2e.cpp:411` |
| retry loops | 0 in `scripts/*.sh` | server-boot polls (`for i in $(seq 1 90); … sleep 2`) in `Makefile:292,317,327` are readiness waits, not retries |
| pytest reruns | none | `requirements.txt` = httpx + pytest; no `flaky`, no `rerun`, no `pytest-timeout` |
| failure tolerances | 1, documented | `tests/test_server_0token_battery.py:36` `FAIL_THRESHOLD = 0.10` - a `temp>0` case may return empty on up to 10 % of requests; `temperature == 0` is `empty > 0` -> fail (no tolerance) |
| timeouts that could mask a hang | `ctest --timeout 120` (`ci.yml:228,817`), `wait_for_server(timeout=120)` (`conftest.py:34`), `httpx.Client(timeout=60)` (`:97`), mutation harness `--timeout 2400` | The mutation harness is the only one that treats a timeout correctly: `run.py:242` records `TIMEOUT` separately and excludes it from the numerator ("a timeout is not an assertion"). ctest reports a timeout as a failure, so it does not mask. |
| empty-filter guards | 4 registered ctests, all `unit`-labelled | `guard_e2e_lane_split`, `guard_det_suite_filter`, `guard_verify_filter`, `guard_precommit_filter` (`CMakeLists.txt:1069-1093`) |

**Filters that no guard covers**: `scripts/verify.sh:334` `_LANE_FILTER` (a second hand copy of `_unit_e2e_filter`, currently stale - [I-3]); the six patterns of `make test-e2e`'s filter other than `*DetEvalE2ETest*` (`Makefile:196`); `make test-vision`'s `VisionGolden.*` and `*Qwen3VLPipeline*`; `make test-gpu`'s `*Qwen38TokenizerParity*`; `make test-spec-fidelity`'s literal test name. **Checked: all of them resolve to ≥1 test today** (applied each pattern to the 2 701 collected names) - so this is a latent guard gap, not a live outage.

**Mutation anchors**: `python3 tools/mutation/run.py --verify-anchors` -> `anchors: 56/56 match exactly once`. The 2026-09-02 repair (M22, M50, M54, M56) holds. The score-ignores-own-survivors half is also fixed: MUTATION_BASELINE records the CI-lane figure moving 15/20 -> 20/21 = 95.2 % after `KVCache::for_accounting()` and `tests/test_kv_accounting.cpp` landed, with M25 (equivalent) the only survivor.

---

### Findings

### [I-1] Five model-backed E2E suites, including the greedy regression locks, execute from no `make` target
Axis: I   Sev: S1   Confidence: high
Evidence: `tests/test_e2e_greedy_lock.cpp:61` and `tests/test_prefix_cache_e2e.cpp:34,45` skip unless `IMP_TEST_MODEL` is set. The only two sites in `Makefile`/`scripts/`/`.github/` that set `IMP_TEST_MODEL` for a gtest run are `Makefile:138` (filter `*DetEvalE2ETest*`) and `Makefile:190` (filter `PrimaryModelTest.*:GDNModelTest.*:EndToEndModelTest.*:Gemma4ModelTest.*:Gemma4GraphsTest.*:SpecCaptureFidelityTest.*:*DetEvalE2ETest*`). Neither filter contains `GreedyLockTest` or `PrefixCacheE2ETest`. `make test-gpu` runs everything unfiltered but sets no model variable (`Makefile:14` `DOCKER_RUN` has no `-e`), so both skip there. Same shape: `TokenizerCompatTest` (`test_tokenizer_compat.cpp:45`) and `TensorKindCoverage` (`test_tensor_kind_coverage.cpp:16`, needs `IMP_TEST_GGUF`, set by nothing - `rg IMP_TEST_GGUF Makefile scripts .github` -> 0).
Second half: even if `GreedyLockTest` were in the filter, `Makefile:190` points `IMP_TEST_MODEL` at `Qwen3-4B-Instruct-2507-Q8_0.gguf`, and `tests/refs/e2e_greedy_locks.h` holds rows only for `Qwen3-8B-Q8_0.gguf` and `Qwen3-8B-NVFP4-cortecs`, so `test_e2e_greedy_lock.cpp:186` would skip with "no greedy locks recorded".
Current: the file's own header calls it *"The single highest-leverage test class in the audit: a fixed prompt run …"*. It runs when a human types a `docker run` by hand.
Expectation: llama.cpp gates `test-backend-ops` + a greedy reference on every CI run; vLLM's `tests/basic_correctness` are wired to a marker a lane selects. A frozen-output lock that no lane can reach is the escape class this repo already fixed twice (#1299 `DetEvalE2ETest`, #1575 the pre-commit "full suite" running DetEval as a skip).
Delta: 11 macros of the strongest output-quality assertions in the tree (`GreedyLockTest` 1, `PrefixCacheE2ETest` 4, `TokenizerCompatTest` 1, `TensorKindCoverage` 1, plus the 4 `test-e2e` fixtures already covered) sit outside every target, and the lock table does not cover the model the one wired target uses.
Cost: `Makefile` only - add the two fixtures to the `test-e2e` filter and either point `IMP_TEST_MODEL` at Qwen3-8B-Q8_0 or record a Qwen3-4B lock row. ~6 lines. Risk: `make test-e2e` gets slower (a second checkpoint load); a Qwen3-4B lock row must be generated with `IMP_LOCK_PRINT=1` on a verified-idle card. What breaks if wrong: nothing - these tests skip today.
Falsifier: a target elsewhere that sets `IMP_TEST_MODEL` and runs the whole binary. Checked y - `rg -n 'IMP_TEST_MODEL[ =]' Makefile scripts/ .github/ | grep -v IMP_TEST_MODEL_` returns exactly 4 lines, two of them the server harnesses (`test_server.sh:90`, `coverage_server.sh:50`) which run no gtest.

### [I-2] The pre-push gate's `*Attention*` pattern covers paged decode and silently excludes the whole FA2/FMHA prefill family
Axis: I   Sev: S2   Confidence: high
Evidence: `scripts/verify.sh:309` `FILTER="TensorTest.*:GgufLoaderTest.*:Tokenizer*:ChatTemplate*:KVCache*:GemmTest.*:FP8GemmTest.*:SamplingTest.*:SoftmaxTest.*:*Attention*:VramBudget*:ForwardPassTest.*"`. Applying those 12 patterns to all 2 701 collected `Fixture.Test` names matches **359** (13.3 %). Matched attention fixtures are `PagedAttentionTest`, `PagedAttentionINT4Test`, `PagedAttentionNvfp4TC*`, `PagedAttentionReduce`, `AttentionCrossPathTest`, `AttentionChunkedTest`, `AttentionMxFP4Test`, `AttentionDispatchRules`. **Unmatched**: `FmhaFA2Test` (36), `FhmaMxFP4Test` (35), `FmhaSm120Test` (24), `FmhaFA2Hd256Bkv32Test` (17), `FmhaFA2Hd256Test` (15), `FmhaFA2PvF16Test` (13), `FmhaFP8Test` (12), `FmhaHd512*` - **152+ macros**. Their target `src/compute/attention_fmha_sm120.cu` is 1 556 code LOC and file-size-allowlisted (brief, §facts).
Current: `guard_verify_filter` (`scripts/check_verify_filter.sh:57-68`) asserts every pattern matches ≥1 test. `*Attention*` matches 359, so the guard is green while the FA2 family is uncovered.
Expectation: the gate that a repo calls "the only thing that ever puts a CUDA kernel in front of a correctness check" (`scripts/pre-push.hook:8-10`) should cover the kernel family with the most churn. #1586's own comment records the previous instance of exactly this defect (`AttentionTest.*` matched nothing for four months).
Delta: FA2 correctness on this branch rests on the perf gate's `pp512` throughput and one 5-token smoke prompt. A tile-boundary or sliding-window numerical regression at HD=256 / Bkv=32 changes neither.
Cost: one string in `verify.sh:309` (`+:Fmha*:Fhma*`), plus wall clock - `test-attention` is 279 s for 200 cases per TEST_INVENTORY §1, so the FA2 subset is the budget question, not the correctness one. Risk: `verify-fast` is documented at ~37 s of script time; adding the family could double the gate. What breaks if wrong: a slower push, nothing else.
Falsifier: the FA2 kernels are covered by the perf/smoke half of `verify-fast`. Checked partially - `verify.sh:731-838` runs one 5-token prefill on one dense model and asserts repetition/NaN/substring; it exercises one FA2 shape and cannot see a boundary-case numeric error. Not a full refutation, so the finding is scoped to "boundary-case correctness", not "FA2 is untested by the gate".

### [I-3] `make verify` (the documented full pre-merge gate) is red by construction: a second, unguarded copy of the unit filter went stale nine days ago
Axis: I   Sev: S2   Confidence: high
Evidence: `CMakeLists.txt:1057` `set(_unit_e2e_filter "BatchBuilderTest.*:SchedulerTest.*:RequestTest.*:EndToEndTest.*:StoragePlanner.*:WeightRegistryPreservation.*:StubModelTest.LoadStubModel:StubModelTest.TokenizeStub")`. `scripts/verify.sh:334` `_LANE_FILTER="BatchBuilderTest.*:SchedulerTest.*:RequestTest.*:EndToEndTest.*:StubModelTest.LoadStubModel:StubModelTest.TokenizeStub"` - missing `StoragePlanner.*` and `WeightRegistryPreservation.*`. `scripts/check_e2e_lane_split.sh` compares `--gtest_list_tests` under the *passed* filter against a frozen 61-name `EXPECTED` that contains 13 `StoragePlanner.*` and 1 `WeightRegistryPreservation.*` entry; `rg -n 'TEST(_F)?\(StoragePlanner|TEST(_F)?\(WeightRegistryPreservation' tests/` -> 14 macros. So the comparison cannot match, the script exits 1, and `verify.sh:339` calls `fail` -> `FAIL=1` -> `exit 1` (`verify.sh:108, 851`).
Provenance: `git log -S 'StoragePlanner.*:WeightRegistryPreservation' -- CMakeLists.txt` -> `3ce0c326` (2026-08-27, #1795). `verify.sh`'s copy last changed in `d25383c2` (2026-06-07).
Current: `verify.sh:330-333` states the copy is deliberate ("Can't source it here") and claims "Going stale is caught by the primary ctest guard … plus the frozen EXPECTED list". Both guards are real, and neither reads *this* string: `check_verify_filter.sh:38` greps only `^ *FILTER="`, and `guard_e2e_lane_split` is invoked by CMake with the CMake variable, not with `verify.sh`'s copy.
Expectation: a duplicated literal in a gate is either generated or guarded; the repo applies that rule to the other three filters (`guard_det_suite_filter`, `guard_verify_filter`, `guard_precommit_filter`).
Delta: the `full` lane of the project's own pre-merge gate has been failing on a bookkeeping mismatch for nine days and nobody noticed, which is itself the measurement of how often `make verify` is run.
Cost: 1 line in `scripts/verify.sh`, or extend `check_verify_filter.sh` to also diff `_LANE_FILTER` against `CMakeLists.txt`'s `_unit_e2e_filter` (~15 lines, CPU lane). Risk: none. What breaks if wrong: nothing.
Falsifier: run `scripts/check_e2e_lane_split.sh build/test-e2e "$_LANE_FILTER"` and see it pass. Checked n - no runnable binary on this host (`./build-dev/test-e2e` aborts with `GLIBC_2.43 not found`, and building is out of scope). The textual argument is exact: 14 EXPECTED names cannot appear in a listing produced by a filter that names neither fixture.

### [I-4] DRY, mirostat and `logit_bias` are shipped samplers with zero tests in any lane
Axis: I   Sev: S2   Confidence: high
Evidence: implementations - `src/exec/executor_sampling.cu` 12 `mirostat` mentions, `src/runtime/engine_scheduler.cpp` 12, `tools/imp-server/handlers_chat_params.cpp` 3; `src/api/imp_api.cpp` 9 `dry_*` mentions, `src/exec/inference_state.h` 3; `logit_bias` in `src/runtime/engine_scheduler.cpp` (6), `engine_decode_pipeline.cpp` (2), `engine_spec_ngram.cpp` (2), `handlers_chat_params.cpp` (8). Tests - `rg -lni 'mirostat|dry_multiplier|dry_base|logit_bias' tests/` returns exactly one path: `tests/api/test_repetition_compare.sh`, a manual comparison script whose only mention in the tree is its own usage line (`rg -n test_repetition_compare .` -> 1 hit, itself). `tests/test_sampling.cu`'s 25 macros cover greedy, top-k, top-p, temperature, seeding, batched slots and the two penalty kernels; none of the three.
Also uncovered by construction: `tools/imp-server/handlers_chat_params.cpp` (the JSON -> sampler-parameter parser) is compiled into `imp-server` only - the `test-core` `IMP_BUILD_SERVER` block (`CMakeLists.txt:796-846`) links `anthropic/responses/utils/tool_call/args/rate_limit/tracing/constraint_validation/image_fetch`, not the chat-param parser. The mock (`tests/api/mock_server.py`) reimplements the endpoints, so the CI API job does not execute it either.
Expectation: llama.cpp has `test-sampling` covering mirostat v1/v2 and DRY; vLLM has `tests/samplers/test_logits_processor.py` for logit bias. These are user-visible knobs that silently do nothing when broken.
Delta: three request fields can be accepted, plumbed and ignored (or applied wrongly) with no test anywhere; the mutation campaign already recorded `sampling` as its worst category (0/3 at baseline, `MUTATION_BASELINE.md`) and never added a mutant for these three.
Cost: one new `tests/test_sampling_advanced.cu` (~250 LOC, GPU lane) plus a host-side parser test if `handlers_chat_params.cpp` is added to `test-core` (that file pulls nlohmann, which the block already links). Risk: low. What breaks if wrong: nothing; new tests only.
Falsifier: a test under another name. Checked y - case-insensitive `rg -lni` over all of `tests/` for the three feature names returns only the shell script.

### [I-5] The mutation catalogue covers 8 production files; 16 of 25 subsystems have no mutant at all
Axis: I   Sev: S2   Confidence: high
Evidence: `tools/mutation/catalogue.json`, 56 mutants over `src/compute/attention_paged_common.cuh` (16), `src/memory/kv_cache_manager.cpp` (11), `src/compute/attention_paged.cu` (7), `src/compute/rope.cu` (4), `src/model/gguf_loader.cpp` (4), `src/quant/dequant_gpu.cu` (3), `src/compute/sampling_topk_topp.cu` (2), `tools/imp-server/anthropic.cpp` (3), `utils.cpp` (2), `src/compute/json_constrain.cu` (2), `schema_constrain.cu` (2), `gbnf_grammar.cpp` (1), `src/exec/executor_sampling.cu` (1). Nothing in `src/exec/` beyond one sampling anchor, nothing in `src/runtime/`, `src/model/weight_*`, `src/compute/gdn*`, `src/compute/attention_fmha*`, `src/exec/gemm_kernel_registry.cu`, `src/memory/plan.cpp`, `src/vision/`, `src/lora/`, `tools/imp-quantize/`.
Cross-check with churn (brief §facts): the top-churn files in the repo - `src/runtime/engine.cpp` 254 commits/6mo, `engine_scheduler.cpp` 95, `tools/imp-server/handlers.cpp` 116 - carry zero mutants.
Current: the headline "92.3 % kernel catalogue" is honest about what it measures (`TEST_HARDENING_LOG.md` iteration 14) but names a slice, and the slice is the one the campaign chose in August and has not widened since.
Expectation: mutation coverage should track churn x blast radius, not the subsystem that happened to be the campaign's first target.
Delta: the strongest quality instrument in the repo answers "are the paged-decode and KV-hash tests good", and says nothing about the scheduler, the executor, the graph capture path or the loaders - where the commits actually land.
Cost: catalogue entries only, no production change: ~10 new anchors at ~1 h each to author and verify, `--ci-only` runs are GPU-free and take ~2 s per mutant per iteration-14's measurement. Risk: none (the harness byte-restores in a `finally` and refuses a dirty tree, `run.py:311-313`).
Falsifier: a second catalogue elsewhere. Checked y - `tools/mutation/` holds one `catalogue.json`, and `--verify-anchors` enumerates 56.

### [I-6] `imp-quantize` and `imp-bench` have no integration test, and the one validation script that exists is never invoked
Axis: I   Sev: S2   Confidence: high
Evidence: `imp-bench` - `rg -c 'imp-bench' Makefile CMakeLists.txt` -> `CMakeLists.txt:10` (build target) and nothing in `Makefile`; no test file references it. `imp-quantize` - three of its sources are compiled into `test-core` (`CMakeLists.txt:764,772,780`: `tensor_policy.cpp`, `fp8_source.cpp`, `checkpoint_out.cpp`) giving 57 host-side macros, but no target runs the binary end to end. `scripts/validate_safetensors.py` (42 637 bytes) has zero invocation sites: `rg -n validate_safetensors Makefile scripts/ .github/` returns only `scripts/check-release.sh:126`, which names its *output* as "written by scripts/validate_safetensors.py, not read".
Current: `CMakeLists.txt:762-763` says the tensor-selection rule "has already shipped a broken checkpoint (#1159)", which is the argument for the unit test that exists - and the same argument for the round-trip that does not.
Expectation: a quantizer whose output is loaded by the engine is normally gated by a convert-then-load-then-generate round trip on one tiny checkpoint (llama.cpp gates `test-quantize-fns` + a convert smoke).
Delta: `imp-quantize` can emit a checkpoint that every unit test accepts and the engine rejects, and nothing between the tool and `make test-e2e` would say so.
Cost: a `make test-quantize` target: quantize Qwen3-0.6B (present in `~/models`) to NVFP4, load it through the C API, assert a substring. ~40 lines of Makefile + a small test. Risk: adds minutes to no automatic gate (it would be opt-in like `test-server`). What breaks if wrong: nothing.
Falsifier: an existing round-trip under another name. Checked y - `test_quant_integration.cu` (25 macros, `test-quant`) is a dequant/GEMV-vs-reference battery, not a tool round trip; it never invokes `imp-quantize`.

### [I-7] Four `make` gtest filters, and one of two hand-copied filters, are unguarded against the "matches nothing = PASSED" class
Axis: I   Sev: S3   Confidence: high
Evidence: guarded filters are `_unit_e2e_filter` (`guard_e2e_lane_split`), `*DetEvalE2ETest*` (`guard_det_suite_filter`), `verify.sh`'s `FILTER=` line (`guard_verify_filter`), and the pre-commit path filter (`guard_precommit_filter`) - `CMakeLists.txt:1069-1093`. Unguarded: `Makefile:196` (six patterns besides the DetEval one), `Makefile:221` `VisionGolden.*`, `Makefile:229` `*Qwen3VLPipeline*`, `Makefile:148` `*Qwen38TokenizerParity*`, `Makefile:208` `SpecCaptureFidelityTest.CachedGraphMatchesEagerForward`, and `verify.sh:334` `_LANE_FILTER` ([I-3]).
Current: all of them resolve today - I applied each pattern to the 2 701 collected names: 4, 2, 7, 4, 2, 2, 5, 2, 5, 4, 1 matches respectively. So this is latent.
Expectation: the repo has already paid for this class three times by its own record (#1299, #1575, #1586) and built a guard shape for it; extending the shape is cheaper than the fourth incident.
Delta: a fixture rename in `test_vision_golden.cu`, `test_qwen3vl_pipeline.cu` or `test_e2e_models.cpp` turns its target green while running nothing.
Cost: one generalised guard script reading every `--gtest_filter=` literal out of `Makefile` and asserting each pattern is non-empty (~50 lines, CPU lane, `--gtest_list_tests` only, no GPU). Risk: none.
Falsifier: a guard I missed. Checked y - `rg -n 'add_test\(NAME guard' CMakeLists.txt` returns exactly the five guards named above (incl. `guard_cancel_teardown`, which is a grep on source, not a filter guard).

### [I-8] The "full GPU suite" that the pre-commit hook runs sets no model env var, so most model-backed suites skip inside it
Axis: I   Sev: S3   Confidence: high
Evidence: `scripts/pre-commit.hook:74` `exec make -s test-gpu`; `Makefile:129-130` `test-gpu: build` / `$(DOCKER_RUN) imp-tests` with `DOCKER_RUN = docker run --rm --gpus all -v $(HOME)/models:/models $(DOCKER_IMG)` (`Makefile:14`) - no `-e` at all. `Makefile:131-149` then adds two explicit runs with env (DetEval, Qwen3.8 tokenizer parity), which is #1575's fix applied to two suites. Everything else that reads `IMP_TEST_MODEL*` via `getenv` with no default skips: `PrimaryModelTest`, `GDNModelTest`, `Gemma4ModelTest`, `Gemma4GraphsTest`, `EndToEndModelTest`, `GreedyLockTest`, `PrefixCacheE2ETest`, `Qwen3VLPipelineTest`, `VisionGolden`, `TokenizerCompatTest`, `TensorKindCoverage`. Suites with a hard-coded `/models/...` default *do* run there - `DegenerationTest` (`test_degeneration.cpp:22`), `LoraHotSwap` (`test_lora.cpp:36`), `ApiGenerateTest`, `EngineRelaunch`, `WarmCache`, `SuspendResume`, `QuantPipeline`, `ChunkedPrefillTest`, `LlmCompressorE2E`, `EncoderEmbed` - and all their default paths exist in `~/models` on this box (`ls ~/models`).
Current: which model-backed tests the strongest local gate runs is decided by whether the test author wrote `getenv(X)` or `env_cstr_or(X, "/models/...")`, not by any policy.
Expectation: one place lists the checkpoints the full suite runs against, the way `Makefile:189-195` already does for `test-e2e`.
Delta: `make test-gpu`'s "full suite" claim is true for macros and false for model coverage.
Cost: move the `test-e2e` env block into a Make variable and add it to `test-gpu`'s `DOCKER_RUN` (~10 lines). Risk: `test-gpu` gets much slower and hits the one-process VRAM ceiling TEST_INVENTORY §5 documents (38 failures loading three large checkpoints in one process). That risk is the reason not to do it naively.
Falsifier: `DOCKER_RUN` carrying env somewhere. Checked y - `Makefile:14` is its single definition.

### [I-9] `docs/audit/TEST_INVENTORY.md` is a month-old baseline that reads as current state
Axis: I   Sev: S3   Confidence: high
Evidence: it is linked as the entry point from `tests/CLAUDE.md:70` and `docs/audit/TEST_HARDENING_LOG.md:4` with no staleness marker beyond its date line. Four of its headline claims are now false: "2 125 gtest cases" (2 701 macros today), "§2.3 the Stage-1 GPU hook is not installed" (installed, md5-identical, 2026-09-02), "§2.2 61 test cases silently skip in CI" (0, enforced by `guard_unit_skips`), "§2.4 `AttentionTest.*`" (`*Attention*` since #1586). Its §3 `DISABLED_` table lists 4; there are 6.
Current: `SETTLED.md` §F exists precisely because the previous audit generated hypotheses from a brief with five wrong facts; this file is the same hazard one layer down, and it is the file the `tests/CLAUDE.md` router points at.
Expectation: either a regeneration command in-file (like `check_test_lanes.py --report`) or an explicit "baseline, superseded by" header.
Delta: the next audit re-derives four numbers, or worse, uses them.
Cost: a 6-line header block plus corrections, or fold the live numbers into `tests/CLAUDE.md` and mark the inventory historical. ~30 lines of docs. Risk: none. `scripts/docs_lint.py` gates the layer header.
Falsifier: a staleness marker I missed. Checked y - the file's only date context is its `Date:` line and the §1 "Provenance correction (2026-08-08)" note, which is about its own first pass.

---

### Checked and NOT a finding

- **CI unit lane has hidden skips** - no. `guard_unit_skips` reads the three lanes' GTest JSON and fails on any `SKIPPED` (`tools/check_unit_skips.py`, wired at `CMakeLists.txt:1096-1098` with `DEPENDS` on all three lanes).
- **`ctest -L unit` runs tests twice** - no. `CMakeLists.txt:640-645` explicitly refuses `gtest_discover_tests()` and documents why (R5/#580).
- **Mutation anchors have drifted again** - no. `--verify-anchors` -> 56/56 today.
- **pytest reruns / flaky plugins hide failures** - no. `requirements.txt` is httpx + pytest; no marker or plugin for reruns.
- **`test_perf_regression.py` widens tolerances** - no. It reads the threshold from `tests/perf_baseline.json` (one source with `verify.sh`); the hard-coded fallbacks are tighter.
- **The perf gate ignores prefill regressions silently** - known and deliberate: `scripts/bench_gate.sh:62-65` makes prefill a warning; decode fails. Restated, not filed.
- **`_unit_e2e_filter` can drift silently** - no. `check_e2e_lane_split.sh` freezes all 61 fully-qualified names and diffs both directions.
- **`*DetEvalE2ETest*` could match nothing** - no. `check_det_suite_filter.sh` additionally asserts both instantiated model rows (`moe`, `dense`) are present.
- **`verify.sh`'s `FILTER=` could contain a dead pattern** - no. `check_verify_filter.sh` fails on any pattern matching zero tests, and fails-open only when the binary lists nothing at all (no CUDA runtime), which it says out loud.
- **The pre-commit hook is not installed (TEST_INVENTORY §2.3)** - stale. Both hooks are installed and byte-identical to their `scripts/*.hook` sources.
- **Scheduler logic is GPU-gated** - no. `SchedulerTest` (30 macros, admission, priority, aging, chunked-prefill rescheduling, memory-aware skip) is inside `_unit_e2e_filter`, i.e. in the required CI job.
- **KV-manager bookkeeping is invisible to CI (TEST_INVENTORY §2.2)** - closed. `KVCache::for_accounting()` + `tests/test_kv_accounting.cpp` moved 5 mutants' worth of coverage into the unit lane; CI-lane mutation score on host-side mutants 15/20 -> 20/21.
- **Constrained decoding relies on GPU tests only** - no. The property batteries and the four fuzz targets run in the CPU lane and under ASan in `Sanitizers` (`CMakeLists.txt:711-717`, SETTLED S-28).
- **`tests/api` tests the real server in CI** - half. `Real API contract (model-less)` runs the real binary with `-m nomodel` and prints its own uncovered half (`ci.yml:404-417`); the generation contract runs only in `make test-server`.
- **`make test-server` batteries are advisory** - no. `scripts/test_server.sh` collects `fails[]` and exits 1; 12 batteries including `degen_suite.py` are hard-gated.
- **`test_server_0token_battery.py`'s 10 % tolerance hides a wedge** - no. `temperature == 0` uses `empty > 0` (zero tolerance); the 10 % applies only to `temp>0` (`tests/test_server_0token_battery.py:84`).
- **`--gtest_repeat` or retry loops mask flakiness** - no such construct in the tree.
- **INT8 KV decode is untested** - no. `INT8KVCache` has 2 macros in `tests/test_fp8_kv_cache.cu`.
- **`imp-cli` argument handling is untested** - no. `tools/imp-cli/args.cpp` is compiled into `test-core` (`CMakeLists.txt:768`) and `apply_config_pins` is covered by `test_mtp_auto.cpp`; `tests/test_entrypoint.sh` runs inside `ci_static_gates.sh:91`.

---

### Known-and-accepted (restated)

- **No GPU CI lane** - repo-owner decision 2026-08-03 (SETTLED F-5, WON'T FIX). Consequence: 1 084 of 2 701 macros run only under a human's `make verify-fast` / `make test-gpu`, pinned by `check_test_lanes.py`.
- **No correctness gate against a reference implementation** (#1571) - `docs/LIMITATIONS.md:57-63`. Confirms this axis's finding that no test asserts an absolute perplexity or KL bound: `DetEvalE2ETest` asserts only `ppl1 == ppl2` and `ppl1 > 0.0`.
- **No soak or endurance test** (#1642) - `docs/LIMITATIONS.md:64-68`; largest driven load is 10 concurrent requests.
- **Generation half of the HTTP contract untested in CI** (#1600, #1559) - `docs/LIMITATIONS.md:43-47`.
- **Server streaming path never in the perf gate** (#1685) - `docs/LIMITATIONS.md:48-51`.
- **`/admin/suspend`, `/admin/resume`, `server.model_swap` ungated** - `docs/LIMITATIONS.md:41`. Confirmed: `rg '/admin|model_swap' tests/` returns no test.
- **`/v1/rerank` vs llama.cpp is opt-in behind `COMPARE_URL=`** - `docs/LIMITATIONS.md:37-38`.
- **Untested quant formats Q4_1/Q5_0/Q5_1/Q2_K/Q3_K/Q8_K, FP8 E5M2, Llama-4, Phi-4** - `docs/LIMITATIONS.md:31-35`.
- **`make test-niah` exists, no workflow invokes it** - `docs/LIMITATIONS.md:57-63`.
- **Two `DISABLED_` determinism tests are a documented known limit** (#554) - `tests/test_determinism_e2e.cpp:15-22`.
- **`make test-gpu` sets no `IMP_TEST_MODEL*`** - partially recorded in `docs/audit/TEST_INVENTORY.md:§6` ("`make test-gpu` is not the right command"); [I-8] states what it costs today.
- **Property batteries are property + fault injection, not fuzzing** (#1620) - SETTLED S-28.

---

### Open questions

- Does `test-e2e` still produce 38 VRAM-induced failures when run as one process with every model variable set (TEST_INVENTORY §5)? Needs a free 5090; if it still does, [I-8]'s fix must shard by checkpoint, not just add env.
- What does adding `Fmha*`/`Fhma*` cost `make verify-fast` in wall clock on this box? `test-attention` is 279 s for the whole binary; the FA2 subset is unmeasured. Decides whether [I-2] is a filter change or a new `verify-medium` tier.
- Is `make verify` (full) actually red today? Needs one `--gtest_list_tests` run of `test-e2e` inside the container to confirm [I-3] by execution rather than by text.
- Would a Qwen3-4B greedy-lock row be stable enough to freeze, or should `make test-e2e` load Qwen3-8B for `GreedyLockTest`? Needs `IMP_LOCK_PRINT=1` on an idle card ([I-1]).
- Are DRY / mirostat / `logit_bias` reachable at all from the C API and both HTTP dialects, or is one of them already an accepted-and-ignored field? A `find-stubs` pass would settle it before writing tests for [I-4].


## Axis J - Docs (architecture audit imp, 2026-09-05)

Repo `<repo>`, branch `perf/engine-h-fanin-cut-and-attention-split-verdict`, HEAD `ef664dd8`, tree clean. READ-ONLY: no edit, no build, no GPU job. Host `python3` used only for `scripts/sync_docs.py --check` (the repo's own check script, as the hooks run it).

### Coverage

Read in full:
- `.claude/skills/docs-layers/SKILL.md`, `.claude/skills/README.md` (index table only).
- `scripts/docs_lint.py` (408 lines), `scripts/sync_docs.py` (104), `scripts/check_doc_citations.py` (97).
- `docs/audit/README.md`, root `CLAUDE.md`, `docs/PERF.md`, `docs/MODELS.md`, `docs/plans/README.md`, `docs/roadmap.md` Open + Closed tables (lines 47-88).
- `docs/internals/ARCHITECTURE.md` phase table (lines 38-132), `docs/internals/MEMORY.md` invariant table (lines 666-674).

Sampled (targeted `rg`/`sed -n`, not full read):
- `README.md` (lines 40-60, 125-230), `docs/BENCHMARKS.md` (section index + PROV lines), `docs/FEATURES.md` (10 rows), `docs/LIMITATIONS.md` (8 items), `docs/internals/KERNELS.md`, `docs/performance.md` (lines 1-40), `docs/internals/BENCHMARKING.md:94`, `AGENTS.md`, `src/{compute,model,runtime}/CLAUDE.md`, `tools/imp-server/CLAUDE.md`, `tests/CLAUDE.md`, `imp.conf.example`, `src/runtime/config.cpp`, `src/core/config/attention.h`, `src/model/model_arch.h`, `src/runtime/engine_init_resolver.cpp`, `CHANGELOG.md` (lines 93-236).

Skipped: `docs/archive/` (15 md), `docs/audit/*` except `README.md` and the SETTLED anchor list, `docs/MISSION_JOURNAL.md`, `docs/vram_audit.md`, `docs/API.md` beyond one grep, all `.github/` templates.

Gate re-runs today:
- `python3 scripts/sync_docs.py --check` -> `sync_docs: up to date` (exit 0).
- `lint.out` (earlier run this session): `docs_lint: OK (53 warning(s))`, 0 errors. All 53 are `edited Nx since commit` drift plus 11 L2/L3 unprovenanced-number warnings.
- `cit.out`: `PASS: 0 dead citation(s) across 33 living doc(s)`.

Scope arithmetic (`git ls-files '*.md'`): 118 tracked `.md`. 20 under dot-directories (out of lint scope by `in_scope`), 57 lint-exempt (`docs/audit/` 24, `docs/plans/` 13, `docs/archive/` 15, plus the 5 `EXCLUDED_FILES` `CHANGELOG.md` / `docs/MISSION_JOURNAL.md` / `docs/vram_audit.md` / `AUDIT.md` / `docs/roadmap.md`), 41 linted. Citation gate covers 33 of those 41.

`verified:` staleness: 48 files carry the field, **0 older than 30 days** (oldest `2026-08-13`, 23 days: `AGENTS.md:4`, `CLAUDE.md:4`, `CONTRIBUTING.md:4`, `docs/internals/QUANT_PIPELINE.md:4`, `docs/vision_gemma4v_spec.md:4`, `src/model/CLAUDE.md:4`, `tests/README.md:4`, plus 5 records and 3 tool READMEs). The lint's own threshold is 180 days, so nothing is stale by the gate either.

### Brief vs repo

| Brief statement | Repo |
|---|---|
| "`docs/STALE.md` if it exists" | Does not exist. `STALE.md` lives at `docs/audit/docs-rewrite/STALE.md`, regenerated by `scripts/docs_lint.py` on every local run (`docs_lint.py` main(), `stale` path) |
| "how many `verified:` dates are older than 30 days" | 0 of 48. The gate threshold is 180 days (`STALE_DAYS = 180`), not 30 |
| "`docs/audit/ARCHMAP.md` is a 2026-06-24 record: does anything living still cite it as current" | Nothing living does. `ARCHMAP` is named only in `docs/audit/README.md`, `docs/audit/AUDIT_ARCH_2026_07_29.md`, `docs/audit/arch_2026_07_29_evidence/progress.md` and `docs/archive/AUDIT_REPORT.md` - all records |
| "`docs/*.md` L1 operators, `docs/internals/*.md` L2" | Holds for 38 of 39 checked files. The exception is `docs/vision_gemma4v_spec.md`, which declares `layer: L2` while sitting in the L1 directory (finding J-7) |
| ARCHITECTURE phase names `step_prefill`, `step_decode_forward`, `init_kv_cache`, `forward_logits`, `prewarm_spec_scratch_`, `engine_arena_open` | All six exist with those exact names (`src/runtime/engine.h`, `engine_prefill.cpp`, `engine.cpp`, `engine_kv_cache_init.cpp`, `engine_spec_capture.cpp`, `src/memory/engine_arena.h`). No drift |
| "`docs/roadmap.md` Open rows whose `ref` path:line no longer says what the row says" | Both Open rows carrying a `path:line` are correct (see negatives). No drift found |

### Findings

### [J-1] ARCHITECTURE.md states one KV block size; the resolver picks two
Axis: J   Sev: S3   Confidence: high
Evidence: `docs/internals/ARCHITECTURE.md:72` "Paged blocks (block_size=16)". `src/runtime/engine_init_resolver.cpp:823-828`: `config_.kv_block_size = (mcfg.n_kv_heads <= 4 && mcfg.n_kv_heads > 0) ? 32 : kKVBlockSize;` with `kKVBlockSize = 16` (`src/memory/kv_cache.h:16`). `docs/API.md:391` shows `"kv_block_size": 32` in a live `/health` sample; `docs/BENCHMARKS.md:58` records the consequence: "budget_blocks uses kKVBlockSize=16 while this model runs block_size=32", the 2x sparse-budget defect fixed by #1819.
Current: the canonical L2 narrative asserts a constant that is a per-model auto-resolution.
Expectation: vLLM and SGLang document block size as a configurable with an auto rule, not a constant; the repo's own `docs/TROUBLESHOOTING.md:96` already writes `ceil(8192/block_size)` symbolically.
Delta: a kernel dev sizing anything off 16 is wrong on every `n_kv_heads <= 4` model (Qwen3.8-27B among them), which is exactly the class of bug #1819 was.
Cost: one table cell in `ARCHITECTURE.md`. Risk none.
Falsifier: a second place in the code that pins 16 unconditionally for the pool. Checked y: `kv_block_size` is a runtime config field, `kv_block_bytes_per_layer()` takes it as an argument (`docs/internals/MEMORY.md:736` documents that it is runtime).

### [J-2] 31 of 223 bound config keys are in no example and 8 in no document at all
Axis: J   Sev: S3   Confidence: high
Evidence: keys bound in `src/runtime/config.cpp` (lambdas `B/I/F/S` plus 2 special `dotted_key ==` cases) = 223; keys present in `imp.conf.example` = 192. Diff: 31 bound-but-undocumented, **0** documented-but-unbound. `imp.conf.example:1-13` claims the precedence chain ends at "embedded defaults (no file, **all values below**)". Of the 31, these appear in no doc, no plan and no record: `attention.sparse_min_ctx`, `diagnostics.moe_expert_hist`, `diagnostics.moe_expert_trace`, `diagnostics.spec_capture_fidelity`, `gemm.nvfp4_smallm_impl`, `speculative.batch_rr`, `speculative.min_history`, `speculative.shallow_draft_ctx`. The operator-visible one: `attention.sparse_min_ctx` default 12288 (`src/core/config/attention.h:224`, bound at `src/runtime/config.cpp:239`) gates the whole sparse-decode path (`src/exec/sparse_attn_geometry.h:43` "Identity below sparse_min_ctx"), while `README.md:161-163` and `docs/MODELS.md:66` tell operators to configure `attention.sparse_topk_tokens` with no mention of the engage threshold.
Current: the example file is the only key catalogue and it is 86 % complete; nothing gates the gap.
Expectation: llama.cpp and vLLM print every knob from one table (`--help` generated from the binder). A generated example, or a gate asserting binder == example, is the standard shape.
Delta: an operator who sets `sparse_topk_tokens` below 12288 context gets identity attention and no log line saying why; 7 further keys are reachable only by reading `config.cpp`.
Cost: either 31 example entries (~60 lines) or a ~40-line gate script `check_config_keys.py` in `ci_static_gates.sh`. Risk: the diagnostics keys may be deliberately unlisted - that is a decision to record, not a default.
Falsifier: a doc listing the keys elsewhere. Checked y: `rg` over `docs/`, `README.md`, `AGENTS.md`, `CLAUDE.md` finds the other 23 only in `docs/plans/*` records or the roadmap, and these 8 nowhere.

### [J-3] FEATURES.md source citations point at unrelated code, and the citation gate is existence-only by design
Axis: J   Sev: S3   Confidence: high
Evidence: `docs/FEATURES.md:34` cites `model.cpp:316` for "an alias onto the LLaMA path"; `src/model/model.cpp:315-316` is `bool kv_nvfp4_default_safe(ModelArch arch)`. The real alias table is `src/model/model.cpp:385,411-413`. `docs/FEATURES.md:78` cites `Makefile:290-291` for "`COMPARE_URL=` is opt-in"; `Makefile:289-292` is the `imp-agent-suite` `docker run`. The real lines are `Makefile:308,320`. Both are wrong at the doc's own pinned commit `c3e4db79`, so this is not post-pin drift. `scripts/check_doc_citations.py` passes both: its docstring says it "checks the line EXISTS, not what it says", and `cit.out` reports `PASS: 0 dead citation(s)`.
Current: 2 of 10 sampled FEATURES rows send the reader to unrelated code; the gate cannot see it.
Expectation: unclear that a content-checking gate is feasible in general; a cheap approximation is a symbol-anchored citation (`model.cpp:phi3_alias`) which `codegraph` could resolve.
Delta: 20 % sampled miss rate on the one doc whose job is "does this exist and where".
Cost: fixing the two rows is 2 lines. A symbol-anchored citation form is a gate rewrite (~80 LOC) plus a sweep of 33 living docs. Risk: line-number citations re-rot on the next split.
Falsifier: the cited lines being right at the pinned commit. Checked y: verified against `c3e4db79` as well as HEAD.

### [J-4] The README's headline competitive table exists nowhere else, and the doc it points at does not have it
Axis: J   Sev: S3   Confidence: high
Evidence: `README.md:137-154` publishes a six-model llama.cpp sweep dated 2026-08-30 (Qwen3-8B **385.4**, 14B **162.5**, gpt-oss-20b **382.7**, 30B-A3B 305.5). `rg --fixed-strings` over the whole tree finds `385.4`, `162.5`, `382.7`, `305.5` in `README.md` only. `README.md:194` then says "Per-model history with dates, commits and exact commands: `docs/BENCHMARKS.md`", whose newest GGUF competitive section is `docs/BENCHMARKS.md:77` "Competitive re-sweep 2026-08-21". `docs/PERF.md:10-11` claims to be the "single source of truth for every number about imp" and does not carry the sweep either.
Current: the most-read numbers in the repo live in the layer that is supposed to embed, not own.
Expectation: the repo's own rule (`docs-layers` SSoT map: "any number -> `docs/PERF.md`, README embeds a GENERATED extract").
Delta: the L0 headline is unmaintained-by-construction; the pointer to its history is dead in substance while passing every link and citation check.
Cost: one `docs/BENCHMARKS.md` section (~15 lines) plus a README link. Risk: none.
Falsifier: the sweep being recorded under different digits (rounding). Checked y: `rg` on all four values across `docs/`, `CHANGELOG.md`, `tools/roofline/` finds nothing.

### [J-5] Two L1 performance front doors, one of them self-declared historical and unindexed
Axis: J   Sev: S3   Confidence: high
Evidence: `docs/PERF.md` (L1, verified 2026-09-04) vs `docs/performance.md` (L1, 90 lines, verified 2026-08-28), whose own body says "Prefill + KV-cache tables below are **historical** (2026-05-27 era, CUDA 13.2.1)" and "llama.cpp / vLLM comparison from cross-engine bench 2026-05-24" (`docs/performance.md:28-33`). `docs/performance.md` is not linked from `docs/README.md` (the L1 hub) - it and `docs/vision_gemma4v_spec.md` are the only two `docs/*.md` absent from that index. Four living docs still route to it: `docs/MODELS.md:19` (for methodology), `docs/quantization.md:12` ("Benchmark numbers"), `docs/internals/KERNELS.md:108` ("the decode levers"), and `docs/internals/KERNELS.md:12`, whose link **text** is `` `performance.md` `` while its **target** is `../PERF.md`.
Current: an operator following `MODELS.md` for methodology lands on a May-era file; a kernel dev following `KERNELS.md:108` lands on the same one.
Expectation: one perf doc per layer; the skill's SSoT table names exactly one owner.
Delta: three docs point readers away from the maintained SSoT, and one link's text disagrees with its own target.
Cost: fold the two live paragraphs of `performance.md` into `PERF.md`/`BENCHMARKS.md`, retarget 4 links, delete or archive the file. ~120 LOC of markdown, 5 files. Risk: `performance.md` may still be the only home for the 2026-05 prefill/KV history - archive rather than delete.
Falsifier: `docs/README.md` linking it under another name. Checked y: `grep` for the basename in `docs/README.md` returns nothing.

### [J-6] MODELS.md publishes a superseded decode figure labelled "CI baseline"
Axis: J   Sev: S3   Confidence: med
Evidence: `docs/MODELS.md:41` `| Qwen3-8B | Q8_0 | 8.2 GB | **268** (tg128, CI baseline #540) | GGUF |`. The pinned baseline is `tests/perf_baseline.json` `metrics.decode_tps.tg128 = 287.19`, timestamp `2026-07-26T01:48:33Z`, and it is what `README.md:206` and `docs/PERF.md:64` publish. The `MODELS.md` header PROV is `date=2026-07-12`, i.e. two weeks older than the pin it names.
Current: an L1 doc calls a 2026-07-12 figure the CI baseline; it is 6.7 % below the actual pin.
Expectation: the repo's own rule 3b - a restated gated number does not inherit the gate; write the command or link the owner.
Delta: the one row a reader is most likely to cross-check against `make verify-fast` disagrees with it, with the wrong label attached.
Cost: one cell; either drop the parenthetical or link `PERF.md`. Risk none.
Falsifier: `#540`'s baseline still being live somewhere. Checked y: `perf_baseline.json` was re-pinned 2026-07-26 and its `_note` explains the re-pin.

### [J-7] Layer is self-declared, so a doc can downgrade its own provenance failures
Axis: J   Sev: S3   Confidence: med
Evidence: `scripts/docs_lint.py` check 2 selects severity from the frontmatter field: `if layer in ("L2", "L3"): warnings.append(msg) else: errors.append(msg)`. The field is read from the file, never derived from its path (`check_file()`, `lm = re.search(r"^layer:\s*(\S+)", fm)`). `docs/vision_gemma4v_spec.md:2` declares `layer: L2` while living in `docs/`, the L1 directory the skill's table assigns to operators. It is the only such case in 39 checked files.
Current: any `docs/*.md` can set `layer: L2` and turn "throughput figure with no PROV" from a build failure into a `STALE.md` line.
Expectation: the layer is a property of the path in the skill's own table (`L1 = docs/*.md`, `L2 = docs/internals/*.md`); deriving it removes the choice.
Delta: the check that fails L0/L1 numbers is opt-out by one word. Nothing exploits it today (`vision_gemma4v_spec.md` produces no number warning).
Cost: ~10 LOC in `docs_lint.py` (assert layer matches the path prefix) plus moving one file into `docs/internals/`. Risk: `AGENT_FILES_ALLOWLIST` and `DELIMITATION_ALLOWLIST_PREFIXES` are already path-based, so the two schemes would agree.
Falsifier: a deliberate reason for the L2 declaration in the L1 directory. Checked n: not stated in the file header or `docs/README.md`.

### [J-8] The CHANGELOG form rule is not met by 19 of the last 26 entries
Axis: J   Sev: S3   Confidence: high
Evidence: root `CLAUDE.md` "The CHANGELOG is a changelog, not a journal. One to three lines per entry". Counting bullet entries in `CHANGELOG.md:93-236` (releases 0.37.0, 0.36.0, 0.35.0): 26 entries, of which 19 exceed 3 lines. Longest 10 lines (`- Web UI at \`GET /\`: requests \`stream_options.include_usage\``), then 7 (`- Serving KPI harness tools/analysis/serving_kpi.py`), then two at 6.
Current: the median entry is a paragraph; the rule's escape hatch ("the investigation goes to `docs/` and the entry links there") is used by some entries and not by the long ones.
Expectation: keepachangelog form, which the file's own header follows.
Delta: 73 % non-compliance means the rule is not a rule; nothing checks it (`scripts/check-release.sh` gates release hygiene, not entry length).
Cost: a ~20 LOC gate in `ci_static_gates.sh` counting lines per bullet in the top release only, plus compression of the current top release. Risk: over-compression loses the checkable number the rule also demands.
Falsifier: the rule counting sentences rather than lines. Checked n - the wording says "lines".

### [J-9] The router has no row for the skill that owns the doc gate
Axis: J   Sev: S3   Confidence: high
Evidence: root `CLAUDE.md` "Where to start" names 12 skills; all 12 exist under `.claude/skills/` (`add-model-arch`, `benchmark-cuda`, `building-and-testing`, `check-degeneration`, `code-graph`, `codebase-audit`, `docs-sync`, `find-stubs`, `quant-formats`, `server-api`, `shipping-prs`, `sm120-cuda-expert`). The 13th skill, `docs-layers`, is named nowhere in the repo outside `.claude/skills/` (`rg -rn 'docs-layers'` excluding its own directory returns 0 hits outside `.claude/skills/README.md` and `docs-sync/SKILL.md` frontmatter). Root `CLAUDE.md` mentions `scripts/docs_lint.py` by path but routes "Keep docs in sync after a change" to `docs-sync` only.
Current: an agent whose `Build` job goes red on `docs` or `citations` has no router entry to the playbook that explains the header, PROV and generated blocks.
Expectation: the router's stated job ("This file is the router, not the manual").
Delta: one missing table row in the file every task starts from.
Cost: 1 line. Risk none.
Falsifier: `docs-sync` covering the lint rules itself. Checked y: `docs-sync`'s own description says "Do NOT use for layer/frontmatter/lint questions (docs-layers)".

### [J-10] The router's env-var list is incomplete
Axis: J   Sev: S3   Confidence: med
Evidence: root `CLAUDE.md` "The env vars seeded are `IMP_DETERMINISTIC`, `IMP_FMHA_FA2`, and the three trace knobs promoted to config keys in #1207 ... don't reintroduce ad-hoc env reads". `rg -o 'getenv\("IMP_[A-Z_]+"'` over `src/` and `tools/` finds one more live read: `tools/imp-server/batching_engine.cpp:182` `getenv("IMP_WORKER_TIMING")`, gating per-loop worker timing logs (`batching_engine.cpp:151`). It is bound to no config key, listed in no doc, and absent from `imp.conf.example`. (`IMP_NO_WARMUP` shows up in a grep only inside comments at `src/runtime/engine_init_resolver.cpp:61` and `engine_workspace_warmup.cpp:425` - it is genuinely gone.) The same `CLAUDE.md` sentence names `IMP_FMHA_FA2` twice.
Current: one ad-hoc env read exists that the rule says should not.
Expectation: the rule as written.
Delta: either promote it to `diagnostics.worker_timing` or add it to the router's list; today it is invisible to both.
Cost: ~8 LOC to bind a config key, or 1 line of doc. Risk none.
Falsifier: `IMP_WORKER_TIMING` being test-only. Checked y: it is in the server's decode worker loop, not in `tests/`.

### [J-11] MODELS.md has no Mixtral row while README and FEATURES mark it green
Axis: J   Sev: S3   Confidence: med
Evidence: `src/model/model_arch.h:9-25` has 16 enumerators. `docs/MODELS.md` carries a checkpoint row for 13 of the 14 real ones (`GENERIC` is not a model, `LLAMA4` is correctly yellow in `docs/FEATURES.md:46` and listed in `docs/LIMITATIONS.md:30`). `MIXTRAL` has none, yet `README.md:228` lists "LLaMA, Mistral, Mixtral" under a green marker and `docs/FEATURES.md:33` marks the row `✅` with an empty note column. Code and tests do exist (`src/model/model_arch.h:12`, `tests/test_moe_executor.cu`, `tests/test_chat_template.cpp`), so the marker is defensible; the "validated end to end" list is not.
Current: the one doc whose job is "which checkpoints were actually run" has no Mixtral entry, and `MODELS.md:29` says anything not on the list "is not verified end-to-end".
Expectation: FEATURES green = code path plus a gate; MODELS = a checkpoint someone ran. Those disagree here.
Delta: a reader picking a Mixtral checkpoint gets a green marker and no validated row.
Cost: either a `MODELS.md` row with the checkpoint that was run, or a note column on `FEATURES.md:33` naming the synthetic gate. ~2 lines.
Falsifier: a Mixtral checkpoint under another name in `MODELS.md`. Checked y: `rg -in mixtral docs/MODELS.md` returns nothing.

### [J-12] One measurement, four copies
Axis: J   Sev: S3   Confidence: high
Evidence: the host-noise pair "287.63 one day and 276.92 the next" appears in `README.md:216-218`, `docs/PERF.md:24-26`, `docs/internals/BENCHMARKING.md:94` and `tests/perf_baseline.json` (`_thresholds_note`). All four agree today. The `docs-layers` SSoT map assigns "any number" to `docs/PERF.md`; rule 3b permits restating a *gate threshold* (8/8/10), not the measurement behind it.
Current: four maintained copies of one 2026-08-13 measurement, only one of which (`PERF.md`) carries the PROV block.
Expectation: one owner, three links.
Delta: no error today; the next re-measurement has four places to reach and nothing that fails if it reaches three.
Cost: 3 edits to replace the restatement with a link. Risk: `perf_baseline.json`'s copy is arguably load-bearing (it explains the threshold to whoever refreshes the pin) - keep that one.
Falsifier: a gate that compares the four. Checked y: `docs_lint.py` has no cross-file number check; `check_doc_citations.py` checks paths only.

### Checked and NOT a finding

- `python3 scripts/sync_docs.py --check` -> `sync_docs: up to date`; the two generated `PERF:BEGIN/END` blocks in `README.md` and `docs/PERF.md` match `tests/perf_baseline.json` byte for byte.
- `docs_lint.py`: 0 errors, 53 warnings, all of them the `edited Nx since commit` drift class plus 11 L2/L3 unprovenanced-number warnings that the linter downgrades on purpose.
- `check_doc_citations.py .`: `PASS: 0 dead citation(s) across 33 living doc(s)` - no dead path, no dead line number, no dead bare `docs/*.md` reference.
- All 12 skills named in root `CLAUDE.md` "Where to start" exist under `.claude/skills/`.
- Every backticked path ending `.h/.cu/.cuh/.cpp/.py/.sh/.md/.toml/.json` in root `CLAUDE.md`, `AGENTS.md`, `src/{compute,model,runtime}/CLAUDE.md`, `tools/imp-server/CLAUDE.md`, `tests/CLAUDE.md` resolves. 10 are bare basenames that resolve elsewhere in the tree (e.g. `AGENTS.md:64` `executor_attention.cu` -> `src/exec/`, `tests/CLAUDE.md` `run.py` -> `tools/mutation/`); none dangle.
- All six ARCHITECTURE.md phase symbols exist under those exact names; `docs/internals/ARCHITECTURE.md:38` "four phases" matches the four `##` phase sections and their entry points.
- `docs/audit/ARCHMAP.md` (2026-06-24): cited only from records (`docs/audit/README.md`, `AUDIT_ARCH_2026_07_29.md`, `arch_2026_07_29_evidence/progress.md`, `docs/archive/AUDIT_REPORT.md`). No living doc treats it as current.
- The pinned gate numbers (`287.19`, `20716`, `12406.87`, `4885.13`, `15324.7`) appear only inside the two generated blocks plus `docs/MISSION_JOURNAL.md` (a record). No hand-typed copy of the gate exists in a living doc.
- `docs/BENCHMARKS.md`, `docs/GOAL.md`, `docs/MODELS.md` are the three `PROV_HEADER_ALLOWLIST` docs and all three still declare the per-row convention their header must declare (lint check 2 would error otherwise; it does not).
- `docs/LIMITATIONS.md`: 8 of 8 spot-checked items still true - untested GGUF quants (`tests/test_real_checkpoints.cpp` has 0 hits for Q4_1/Q5_0/Q5_1/Q2_K/Q3_K/Q8_K), `/admin/suspend` endpoint layer ungated, green contexts (`src/runtime/green_ctx.cu:97-133` unchanged), image-fetch DNS rebinding (`tools/imp-server/image_fetch.cpp:201,246,252`), JSON-Schema unenforceable keywords (`src/compute/json_schema.cpp:340-364`), `additionalProperties` as schema object (`json_schema.cpp:466-480`), calibrated KV scales unread (0 hits for `k_proj.k_scale`), INT4 KV on gpt-oss forced to F16 (`src/runtime/engine_init_resolver.cpp:289-297`). Nothing stale.
- `docs/FEATURES.md`: 8 of 10 sampled rows fully correct. Every marker classification checked was right, and every yellow row checked also appears in `LIMITATIONS.md` as the rule requires (rows 46, 66, 78, 109). The two defects are citation targets only (J-3).
- `docs/roadmap.md` Open table: both `path:line` refs say what the row says. Row 3 -> `src/compute/attention_paged_common.cuh:71` is the StreamingLLM context-range block; row 8 -> `src/model/weight_map.cpp:380` is the `model.embed_audio.` skip that increments the aggregate `skipped` counter. Row 5's `server.recurrent_snapshot_host_mb` is bound (`config.cpp:330`) and present in `imp.conf.example:619`; row 1's `tools/analysis/serving_idle_profile.sh` exists.
- Config binder: **0** keys documented in `imp.conf.example` that `config.cpp` does not bind. No phantom keys (the gap is one-directional, see J-2).
- `docs/plans/`: 12 plan docs plus `README.md`. 9 carry the terminal `## ROADMAP CLOSED`, 2 are "detail" records moved out of the roadmap (no closure expected), 1 is genuinely open (`2026-08-15-imp-quantize-roadmap.md`, items 2/3/4). `docs/plans/README.md` indexes all 12 with the correct state and the standing record - the #1904 pass holds.
- `MEMORY.md` invariants I1-I7: each names a gate and each gate exists. I1 `tools/check_alloc_sites.py` (in `ci_static_gates.sh:96`), I3 compile-time in `tests/test_memory_allocators.cpp` (unit lane), I4 `tests/test_memory_plan.cpp`, I6 `tests/test_api_generate.cpp` / `test_server_request_limits.cpp`, I7 `tests/test_library_reserve_cache.cpp` / `test_vram_budget_reserve.cpp`. I2's counter is real (`src/memory/backend.h:76-79 steady_state_allocations()`) and is asserted in `tests/test_memory_backend.cpp`, but its acceptance criterion "after a soak" has no soak - that is the already-accepted #1642, not a docs finding. Naming nit only: `MEMORY.md:334` calls the counter `steady_state_allocations_total`, a name that exists in no source file.
- Size budgets: `README.md` 325 lines (< 400), root `CLAUDE.md` within 2000 tokens, every per-directory `CLAUDE.md` within 800 (lint check 6 green).
- L0 vocabulary smell: exactly one hit, `README.md:228` "NVFP4 block-scaled `mma.sync` GEMM/GEMV", which the skill names as a smell but no gate checks. Left as a nit, not a finding - it sits in a feature matrix that links `FEATURES.md`.

### Known-and-accepted (restated)

- No GPU CI runner; "verified" in `FEATURES.md` means a gate that runs under local `make verify-fast`, not green in CI (`docs-layers` skill, status legend).
- `docs/roadmap.md`, `CHANGELOG.md`, `docs/MISSION_JOURNAL.md`, `docs/vram_audit.md`, `AUDIT.md` and the `docs/{archive,audit,plans}/` prefixes are records: append-only, lint-exempt, true of the day they name (`docs_lint.py` `EXCLUDED_FILES` / `EXCLUDED_PREFIXES`, `docs/audit/README.md`).
- `docs/LIMITATIONS.md` items confirmed still-true above are all already listed there: untested GGUF quants, FP8 E5M2, Llama-4 without a gate, green contexts unavailable on sm_120, DNS rebinding on remote images, JSON-Schema keyword subset, INT4 KV on gpt-oss.
- `docs/roadmap.md` Open 3 (long context half closed), Open 6 (`--calib` at wide GQA), Open 7 (3-D stacked experts), Open 8 (no audio), Open 11 (no KV tier below VRAM - DO NOT BUILD), Open 12 (hybrid pp512 `gemm_cublas` hole, parked).
- No soak test (#1642) and no correctness gate against a reference (#1571) - both bear on I2's acceptance criterion and on `FEATURES.md`'s green markers.

### Open questions

- Is the README's 2026-08-30 six-model sweep meant to supersede `BENCHMARKS.md`'s 2026-08-21 re-sweep, or should both stand as dated rows? (owner)
- `scripts/check-release.sh:491` runs `make bench-competitive` as release bar 2 on every release; its output is recorded nowhere in the tree. Should the bar's numbers land in `BENCHMARKS.md` automatically? (owner)
- Does `docs/performance.md` have a reader left, or is it archive material? (owner)
- Are the 11 `diagnostics.*` keys deliberately absent from `imp.conf.example`, or an omission? (owner)
- Whether the `attention.sparse_min_ctx` default of 12288 is right for `n_kv_heads <= 4` models (where blocks are 32 tokens) needs a GPU measurement, not a doc edit.

# P2 - Falsification pass (validator)

Method: every S0 and S1 claim was re-read against the code by the coordinating session with the opposite hypothesis ("what would make this wrong?"), opening the cited lines and running the falsifier grep myself, not trusting the scout's transcript. S2 claims that land in the top of the P3 queue were spot-checked the same way. Everything else stands as the scout's claim with the scout's own falsifier line. A pass with zero downgrades is suspicious; this one downgraded nine, merged six pairs, removed none.

## S0 / S1 verdicts

| ID | scout sev | verdict | what I opened | surviving sev |
|---|---|---|---|---|
| F1-1 GGUF `n_dims > 4` stack write | S0 | **CONFIRMED** | `gguf_parse.cpp:416-425` tolerates `n_dims > 4` (skips extra dims, no `fail()`); `gguf_loader.cpp:663-667` `int64_t shape[4]` written for `d < ndim`, source `info.dims[ndim-1-d]` reads past `dims[4]` (the next struct fields, `offset` among them, are attacker bytes); `grep -n n_dims src/model` = 8 sites, none caps at 4; `Tensor` ctor guard runs after the loop | S0 |
| F1-2 mmproj loader: no bounds check, same `n_dims` write | S0 | **CONFIRMED** | `vision_loader.cpp:404-415` same tolerant read; `:533` and `:540-541` loop `d < info.n_dims` with no `d < 4`; `grep 'in_bounds\|file_size' vision_loader.cpp` shows `file_size` used only by `mmap`/`munmap`; `:536` `data + tensor_data_start + info.offset` unchecked | S0 |
| B-1 logit-bias arena buffers never re-armed | S0 | **CONFIRMED** | `sampling_penalties.cu:544-546` statics; `:558-563` early `return` on `cap <= s_bias_buf_cap` before the nulling; `sampling_cleanup()` `:669-672` = cub + dry only, no bias half; `~Engine` closes the arena first (`engine.cpp:84`) and `ArenaAllocator::close()` `arena.cpp:30-38` does `region_.reset()` (VRAM released, `++generation_` that nobody here reads); use site `:607-627` writes into the stale pointer when non-null; no `IMP_REGISTER_CUDA_STATIC_RESET` in the file; `engine.cpp` includes `cuda_static_reset.h` but never runs the registry. Second engine in the process (model swap, default on) reproduces by construction | S0 |
| E-1 per-request LoRA: prefix cache adapter-blind, switch unlocked | S1 | **CONFIRMED** | salt = `req->vision_content_hash` only (`engine_prefill.cpp:230,943`); `kv_cache_manager.cpp:466,504` seed the chain from `content_salt`; `Engine::lora_set` `engine.cpp:296-311` clears graphs, not the prefix cache; K/V projections are adapter targets (`lora_adapter.cpp:35-36`); `imp_lora_set` runs after the `state.mtx` block (`handlers_chat_core.cpp:171-184` vs `:686-710`); the worker steps in its own thread (`batching_engine.cpp:311`); `prefix_cache = true` default (`config.h:348`); the in-code comment declares mixed adapters out of scope, no user doc does | S1 |
| D-1 sticky device fault never becomes a host signal | S1 | **CONFIRMED** | `logging.h:83-126` has `IMP_CUDA_CHECK_LOG/BOOL/LAUNCH/VOID`, no throwing variant; `executor_forward.cu:213-217` clears the sticky error with a WARN; `executor_sampling.cu:497,530` sync log-only then return the pinned buffer; every sync in `src/runtime/*.cpp` is `IMP_CUDA_CHECK_LOG` or bare; `batching_engine.cpp:17-21` names the mode in its own comment and `:329-331,358-360` probe only inside `catch` | S1 |
| F2-1 = F1-4 request `model` field resolves any existing path | S1 | **CONFIRMED, MERGED** (one finding, two axes) | `handlers.cpp:409-425` `find_model_path` falls through to `resolve_model_auto` when the name contains `/`; `hf_hub.cpp:36-38` `if (fs::exists(model_id)) return model_id;` before any shape check; `:152-166` accepts any `.gguf` file or safetensors dir; `config.h:361` `model_swap = true`; the comment at `handlers.cpp:440-441` promises containment the code does not implement; the `--api-key` guard (`main.cpp:281-291`) applies only when a key is configured | S1 |
| F1-5 GGUF `block_count` sizes vectors before `validate_declared_dimensions`; `swa_layers[i]` unguarded | S1 | **CONFIRMED** | `gguf_loader.cpp:327` raw `block_count`; `:429-433` `swa_layers` gets the file array's length; `:460-462` `head_dim_per_layer[i] = swa_layers[i] ? ...` for `i < n_layers` with no size guard; `:626` validation 199 lines later; `kMaxModelLayers = 1024` in `model_limits.h:35,65` | S1 |
| F1-6 LoRA adapter dims unchecked against the base model | S1 | **CONFIRMED** | `lora_adapter.cpp:193-222`: `shape` from JSON, `numel = shape[0]*shape[1]` no sign/overflow guard, `w.r/K/N` stored verbatim; `executor_lora.cu:92-93` launches `<<<w.r, 256>>>` with `w.K` as the read extent, `:109-116` `w.N` is the GEMM write extent | S1 |
| F1-7 `tokenizer.ggml.bos_token_id` unbounded to the embedding gather | S1 (med) | **CONFIRMED** | `gguf_loader.cpp:1085-1090` no range check; `tokenizer.cpp:1101` stores raw, `:2328` returns raw; `embedding.cu:42-45` `table[row * d_model + i]` with no vocab bound. Consequence is a device IMA (then D-1: served silently) | S1 |
| B-2 lazy device statics outside the reset registry, no gate | S1 | **CONFIRMED** | `sampling.cu:202-208` `static int32_t* d_result` arena-taken under `if (!d_result)`, never nulled; 15 registering TUs (`grep -rln IMP_REGISTER_CUDA_STATIC_RESET src/`), none of the four named files among them; `ls tools/check_*.py` has no statics gate | S1 |
| B-3 `s_h_normed` sized from the first model's `d_model`, raw `cudaMalloc`, never re-armed | S1 | **CONFIRMED** | `engine_scheduler.cpp:1635-1647` verbatim; `diagnostics.h:153` default `true`; `imp.conf.example:849` ships `false` (drift) | S1 |
| A1-1 GEMM registry serves 10 of 19 keys; three dispatchers | S1 | **CONFIRMED** | `instance().dispatch` at exactly `executor_gemm_dispatch.cu:89,260,595` constructing `{FP16,NONE,false}`, `{FP16,<gguf qtype>,true}` (guarded to exclude NONE/F16/BF16), `{CUTLASS_NVFP4,F16,false}`; 19 `register_kernel` calls across 9 files; the FP16/F16 pair, Q4_K/false, FP8 x2, NVFP4 x2, MXFP4 x2 have no producer. `gemm_kernel_registry.h:13` names a flag `gemm.use_kernel_registry` that exists nowhere else in `src/ tools/ imp.conf.example`. Does NOT contradict SETTLED S-2 (that entry is about duplicated dispatch across formats; this is about the migration being half-done and its header saying so) | S1 |
| F2-2 no connection-level backpressure | S1 | **DOWNGRADED S2** | `main.cpp:161-191`: pool `max_concurrent + 8`, read timeout 60 s, no per-peer cap, guards in `set_pre_routing_handler` `:233-294` after the body is read. All true. But `docs/DEPLOYMENT.md:154-170` already places a reverse proxy in front for TLS, the proposed fix is a `limit_conn` line there, and the shape is llama.cpp's. A reliability gap with a docs-sized fix is S2 by the rubric | S2 |
| F2-3 `--max-concurrent` guard takes `state.mtx` blocking | S1 | **DOWNGRADED S2, MERGED with E-5** | `main.cpp:261-265` `std::lock_guard<std::timed_mutex>` inside the load-shedding guard; `handlers_admin.cpp:23` holds the mutex across suspend; `handlers.cpp:518` across the swap load; `kObservabilityLockTimeout{250}` exists (`handlers.h:35`) and is used elsewhere. Exposure is only during swap/suspend and requests wait rather than fail; the fix is the 8-LOC timed lock plus a readiness signal, which is E-5. One dispatch item | S2 |
| C-1 mid-flight KV exhaustion untyped | S1 | **DOWNGRADED S2** | `cancel_reason` written at `scheduler.cpp:165` only; the four mid-flight cancels (`engine_scheduler.cpp:860-863,876-879`, `engine_prefill.cpp:280-283,532-535`) set status only; `cancel_sequence_` (`engine.cpp:383`) frees resources, sets no reason; consumers read the field directly (`batching_engine.cpp:406,455`, `imp_api.cpp:772,920`). Real, client-visible, 4 one-line assignments. Not architectural debt: S2, but first in its dispatch group on value-to-blast | S2 |
| H-1 dependency-pin check only in non-required `Lint` | S1 | **DOWNGRADED S2** | `ci_static_gates.sh` has no `deps` block (read in full in P0); `ci.yml:460-461` runs `check_dep_pins.sh --online` in `Lint`; `auto-merge.yml:37` arms squash on `Build` alone. True, 6-LOC fix, CI hygiene class | S2 |
| F1-3 warm weight cache `data_alloc`/`data_off` unvalidated | S1 | **DOWNGRADED S2** | `weight_cache_file.cpp:260-261` checks magic, version, `tensor_pod_size`; `:279-284` checks `key_len` and `n_allocs` only; `weight_snapshot.cpp:266-268` indexes `new_allocs[rec.data_alloc]` raw. True, but the file is self-written under the user's cache dir; an attacker who can rewrite it already has the user's privileges. Robustness (a truncated cache after a crash) not security | S2 |
| G-1 layer graph is one 8-node SCC; 64 of 88 backward lines are 4 dependency-free headers | S1 | **DOWNGRADED S3** | the counts match P0 (`pdl.h` 34, `process_diag.h` 25, `pdl_device.cuh` 4, `graph_diag.h` 1); `pdl.h` includes only `<cuda_runtime.h>`, `process_diag.h` only `<string>`, `pdl_device.cuh` nothing. The scout's own falsifier shows the move does NOT restore a DAG (the residual cycles are ordering decisions). No dispatch decision crosses any backward edge (`config.h` in `exec/` is one dead include, P0-1). An include-placement cleanup with a compile-only blast radius is hygiene, not architectural debt | S3 |
| G-2 `executor.h` cost 4235 -> 6320 (+49 %); F-10 moved churn into `dispatch_policy.h` (109 TUs) | S1 | **DOWNGRADED S2** | `git log` confirms `dispatch_policy.h` 20 commits in its 32-day life and `executor.h` 80 commits / 6 months; 29 direct includers of `dispatch_policy.h`; `executor.h:529` still names the accessor `runtime_config()` while returning `const DispatchPolicy&` (143 call sites, 0 `dispatch_policy()`). Real build cost, two mechanical include drops already priced by the scout; wall-clock not measured (S-32 floor ~5 s/TU). Bounded fix, no roadmap blocker | S2 |
| G-3 C API has no first-party consumer | S1 | **DOWNGRADED S3** | `tools/imp-server -> src/runtime` 28 includes vs `src/api` 8, `imp-cli -> api` 2 (P0 matrix agrees); imp-server does call `imp_context_create/free`, `imp_lora_set` (seen in E-1/F2-1 validation), so "no consumer" overstates it: the tools use the C API for lifecycle and reach `Engine` directly for the rest. The consequence the scout names (C-API-only paths validated by tests alone, G-4, G-10) is real; the fix is a contract test and a doc correction | S3 |
| I-1 = D-3 model-backed E2E suites run from no target | S1 / S2 | **CONFIRMED, MERGED, S2** | `Makefile:196` filter lacks `GreedyLockTest`, `DegenerationTest`, `PrefixCacheE2ETest`; `grep` over `Makefile scripts/ .git/hooks/pre-push` for those names is empty; the suites exist (`test_e2e_greedy_lock.cpp` 5 TESTs, `test_degeneration.cpp` 8). A one-line fix to a dead gate is S2, and it is one of the three best value-to-blast items in P3 | S2 |

## S2 spot checks (queue-relevant)

| ID | verdict | what I opened |
|---|---|---|
| I-3 `make verify` red by construction | CONFIRMED | `verify.sh:334` `_LANE_FILTER` lacks `StoragePlanner.*:WeightRegistryPreservation.*`; `CMakeLists.txt:1057` has them since `3ce0c326` (2026-08-27); `check_e2e_lane_split.sh:77-83` EXPECTED lists `StoragePlanner.*`, so the split check fails on every full `make verify` |
| F2-7 compute-sanitizer never in any lane | CONFIRMED | `Makefile:449-459` comment (fails on WSL2), `ci.yml:801` gated, `verify.sh` never calls it |
| A2-1 / A1-3 `gemv_pdl_register()` + `activation_pdl_register()` dead | CONFIRMED, MERGED | definitions `gemm_dp4a.cu:669`, `activation.cu:531`, declarations only elsewhere; the one live call is `nvfp4_gemv_pdl_register()` at `executor_workspace.cu:340`. Perf effect of the lost L1 carveout stays HYPOTHESIS |
| A2-2 batched NVFP4 GEMVs instrumented, unregistered | CONFIRMED (code fact) | `nvfp4_gemv_batched.cu:43,92,147,171,193` `pdl_wait/pdl_trigger`; no registration site names the `_mb_` kernels |
| C-3 graph demotion is a one-way latch | STANDS as documented behaviour | `demote_graphs_` only writes `false`; documented one-way in `graph_eligibility.h:41-44`, `docs/API.md:94`. Kept S2 because the recoverable case (pressure from many short sequences, none evicted) is real and unmeasured |
| C-10 `prefill_graph` dead on quantized KV | PARTIALLY CONFIRMED | `engine_init_resolver.cpp:221-243`: `auto` resolves to NVFP4 / FP8 for the arches `kv_nvfp4_default_safe` / `kv_fp8_no_hint_default_safe` name, so the capture gate `kv_cache_dtype == F16` (`engine_prefill.cpp:695`) is closed on those defaults. Which arches: not enumerated here |
| H-3 8 % threshold inside the box's spread | STANDS | every number is from `scripts/verify.sh:357-358,482-486`, `docs/PERF.md` PROV, `docs/audit/MUTATION_BASELINE.md:85`; the only demonstrated catch is -36 % |
| H-7 Apache-2.0 code, no licence text, image labelled MIT | CONFIRMED | `nvfp4_quant_hw.cu:5-7` "Adapted from thu-ml/SageAttention3 (Apache-2.0)"; `grep -rli apache` outside that pair hits only two records; `Dockerfile:108` `licenses="MIT"`; no NOTICE/THIRD_PARTY file at root. Whether the adaptation is a derivative work is an owner call, hence med |
| A1-2 `gemm.moe_imma_prefill=true` strands three MoE-prefill tiers | STANDS on the scout's grep | `gemm.h:54`; the three `try_run_moe_*` gated on `!moe_imma_pref` at `executor_forward_moe.cu:511-530`; no resolver writes the flag. Relation to SETTLED S-11: S-11 documents a 4-tier NVFP4 ladder; these are the GGUF-quant IMMA-vs-scalar tiers, a different family |
| A1-5 `paged_attention_serves_head_dim` no `MXFP4_KV` case | STANDS on the scout's read | `attention_paged.cu:1196-1216` default returns true; NVFP4 template set has no hd=96 |
| G-6 function-size gate stops at the first column-0 `}` | CONFIRMED | `tools/check_function_size.py:140` `while ... not lines[j].startswith("}")`; `src/exec/executor_forward_moe_cutlass.cu:845-846` has two column-0 braces followed by indented `return true;` at `:848`; `--list` names zero functions in that file although `check_filesize.py` scores it 600 code LOC. Third gate blind spot of the S-30/S-31 class |
| G-5 `runtime.max_seq_len` outranks `--max-seq-len` | CONFIRMED (code) | `engine_init_resolver.cpp:832-835` assigns `config_.max_seq_len = v` unconditionally when the file key is > 0, after the C API copied the CLI value in; the comment at `:836` lists both as "the operator". Which precedence the docs promise was not re-read by the validator; the scout cites it |
| G-9 = P0-1 | MERGED | same dead include, found independently by the coordinator and the scout |
| G-11 = J-2 | MERGED | same `imp.conf.example` gap, counted 139/225 (G) vs 31 missing of 223 (J); the two counts differ by the key-census method, both scouts list the keys |

## Kill list (downgraded, merged, removed)

| ID | from | to | reason |
|---|---|---|---|
| F2-2 | S1 | S2 | fix is a `limit_conn` line in an existing reverse-proxy doc; llama.cpp has the same shape |
| F2-3 | S1 | S2, merged into E-5 | exposure only during swap/suspend; requests wait, not lost; the fix and the readiness signal are one change |
| C-1 | S1 | S2 | four one-line assignments; a contract defect, not architectural debt |
| H-1 | S1 | S2 | 6-LOC addition to `ci_static_gates.sh`; CI hygiene class |
| F1-3 | S1 | S2 | self-written cache file, trusted by location; robustness not security |
| I-1 / D-3 | S1 / S2 | S2, merged | same defect reported by two axes; one Makefile line |
| F1-4 | S1 | merged into F2-1 | same defect, two axes |
| A1-3 | S2 | merged into A2-1 | same dead functions |
| E-4 | S2 | stays S2, marked HYPOTHESIS for the 256-token figure | the 8-token case is measured (~85 ms), the tool-request hold is not |
| G-1 | S1 | S3 | include placement, compile-only blast radius, and the move does not restore a DAG by the scout's own simulation |
| G-2 | S1 | S2 | measured build cost with two priced mechanical drops; no roadmap blocker |
| G-3 | S1 | S3 | overstated: the tools do use the C API for lifecycle; the residue is a contract test + a doc line |
| G-9 | S3 | merged into P0-1 | same finding |
| G-11 | S3 | merged into J-2 | same finding |
| removed | none | | every S0/S1 survived a code re-read; the seven downgrades are severity, not existence |

## Findings the coordinator added from P0 (not in any scout report)

| ID | sev | evidence | note |
|---|---|---|---|
| P0-1 `src/exec/pre_dequant_phase1_fp16_cache.cu:20` includes `runtime/config.h` | S3 | the file reads only `runtime_config().gemm.fp8_ssm_proj` (`:120`) and that accessor already returns `const DispatchPolicy&` (`executor.h:529`); no `RuntimeConfig` symbol in the file; added by #1388 (2026-08-12) after F-10 closed the edge at zero. SETTLED G "src/exec/ includes runtime/config.h zero times" is stale by one dead include | delete one line; re-verify the SETTLED anchor |
| P0-2 `src/vision/vision_encoder.h:3` includes `runtime/cuda_graph.h` for a `CudaGraphRunner` member | S3 | vision -> runtime at header level (every vision includer pulls `cuda_graph.h`); `CudaGraphRunner` is a generic wrapper with no `Engine` dependency | candidate for `core/` or `memory/`; measure fan-in before moving |
| P0-3 `runtime/storage_planner.h` included by `exec/executor.h:30` and 3 more `exec/` TUs; the header itself includes only `core/` | S3 | a type consumed by `exec/` living in `runtime/` (6 includers repo-wide); `exec -> runtime` today is otherwise `pdl.h` (12), `process_diag.h` (2), `vram_budget.h` (1) | move to `core/` if `runtime/` does not own it; else document |
| P0-4 `quant -> compute` is 5 edges, not the 1 SETTLED records | S3 | `nvfp4_gemv_dense.cu:13`, `nvfp4_gemv_batched.cu:21`, `nvfp4_gemm_smallm_v2.cu:36`, `nvfp4_gemv_fused.cu:13` include `compute/pdl_device.cuh` (device-side PDL helper) | `pdl_device.cuh` is a `core/` thing by content; SETTLED anchor for the "one edge remains" line needs re-pinning |
| P0-5 `model -> exec` edge new since 07-29 | S3 | `weight_upload.cu:6` -> `exec/nvfp4_expert_offload.h` (#1407, 2026-08-13) | axis G's simulation: moving that header to `core/` does not change the SCC; it is one of the 24 non-instrumentation backward lines |

## Summary of the pass

| | count |
|---|---:|
| S0 claimed / survived | 3 / 3 |
| S1 claimed / survived as S1 | 19 / 9 (E-1, D-1, F2-1, F1-5, F1-6, F1-7, B-2, B-3, A1-1) |
| downgraded | 9 (F2-2, F2-3, C-1, H-1, F1-3, I-1, G-1, G-2, G-3); D-3 folded into I-1 at S2 |
| merged pairs | 6 (F1-4/F2-1, I-1/D-3, A1-3/A2-1, F2-3/E-5, G-9/P0-1, G-11/J-2) |
| removed | 0 |
| coordinator-added | 5 (P0-1..P0-5) |

# P3 - Ranked dispatch queue

Dependency-ordered. One row = one dispatch. "Gate" = the measurable thing that must hold for the dispatch to be done. Blast = files / LOC touched. The three best value-to-blast-radius items are marked with `***`. Nothing here was started; this is read-only.

| # | dispatch | findings | sev | blast | gate ("done when") | depends on |
|---|---|---|---|---|---|---|
| 1 `***` | **Re-arm device statics across engine teardown.** Add `sampling_cleanup_bias()` mirroring `sampling_cleanup_dry()` and call it from `sampling_cleanup()`; register `sampling.cu` `d_result`, `gemm_moe_fused_tc.cu` tile counter, `ffn_sparsity_probe.cu` state and `engine_spec_mtp.cpp` `s_norm_scratch` with `IMP_REGISTER_CUDA_STATIC_RESET`; move `s_h_normed` to the T2 arena with a capacity field; reconcile `imp.conf.example:849` with `diagnostics.h:153`; add `tools/check_static_reset.py` (file-scope device pointers must appear in a registering TU or an allowlist) to `ci_static_gates.sh` | B-1, B-2, B-3 | S0, S1, S1 | 5 src files ~50 LOC; 1 gate ~80 LOC; 1 test | model swap A -> B in one process, then a `logit_bias` request: logits bit-identical to a fresh process on B (new test in `test-e2e` gpu lane); swap to a larger `d_model` with MTP on: no OOB (compute-sanitizer is dead on WSL2, so assert via a capacity check + the arena guard); the new gate lists every file-scope device pointer in `src/` and is green | none |
| 2 `***` | **Parser hardening, GGUF + mmproj.** `reader.fail()` on `n_dims > 4` in `gguf_parse.cpp` and `vision_loader.cpp`; port `gguf_tensor_in_bounds` + clamped reserve into the vision loader; move `validate_declared_dimensions` (or a `block_count <= kMaxModelLayers` pre-check) ahead of the first `resize`; size-guard `swa_layers[i]`; range-check `bos_token_id`/`eos_token_id` against the vocab; `failed()` check + underflow guard on the shard path; extend `test_gguf_fault_injection.cpp` with the `n_dims`, `block_count`, `sliding_window_pattern`, `bos_token_id` cases; add `fuzz/fuzz_gguf.cpp` and `fuzz/fuzz_mmproj.cpp` driven by `test_fuzz_corpus.cpp` | F1-1, F1-2, F1-5, F1-7, F1-11, F1-12 | S0, S0, S1, S1, S3, S3 | 4 src files ~60 LOC; 2 fuzz targets; ~8 tests | a GGUF with `n_dims = 5` and an mmproj with `offset = 2^63` are refused with a typed error, in the CPU lane; the fault-injection battery fails on the pre-fix tree for each new case (mutation-validated); both fuzz targets run in the corpus lane and under the `Sanitizers` job | none (lands before #3 so the endpoint never reaches an unhardened parser) |
| 3 | **Contain the request `model` field and make admission non-blocking.** `find_model_path` resolves only inside `--models-dir` (weakly_canonical + prefix) or as an HF id from the cache, never an arbitrary existing path; the `--max-concurrent` guard takes `state.mtx` with `kObservabilityLockTimeout` and answers 503/429 on timeout; add a readiness signal (`/health` reports `ready=false` while suspended, model-less or mid-swap, or a separate `/ready`); document the reverse-proxy `limit_conn` line in `DEPLOYMENT.md` | F2-1 (= F1-4), F2-3 + E-5, F2-2, E-6 | S1, S2, S2, S3 | 3 server files ~60 LOC; 1 doc | `{"model": "/tmp/x.gguf"}` -> 404 with the models dir set (mock-API test); during a swap a burst of 100 inference requests never exhausts the 72-thread pool and `/health` answers within 250 ms (test-server); `/health` shows not-ready during `/admin/suspend` | #2 |
| 4 `***` | **Repair the dead gates.** `Makefile:196` adds `GreedyLockTest.*:DegenerationTest.*:PrefixCacheE2ETest.*` (and the two other model-backed suites I-1 names); `verify.sh:334` `_LANE_FILTER` synced with `CMakeLists.txt:1057` and a guard that a gtest filter matching zero tests fails (5 filters, I-7); `make test-gpu` sets `IMP_TEST_MODEL*` or documents that it does not; `check_dep_pins.sh` (offline) added to `ci_static_gates.sh`; `perf_baseline_chunked.json` / `perf_baseline_north_star.json` wired to a target or deleted; `check_function_size.py` counts brace depth instead of stopping at the first column-0 `}` (+1 selftest case), then the 557-LOC body in `executor_forward_moe_cutlass.cu` is split or allowlisted with a reason | I-1 = D-3, I-3, I-7, I-8, H-1, H-4, G-6 | S2 x6, S3 | `Makefile`, `scripts/verify.sh`, `scripts/ci_static_gates.sh`, `tools/check_function_size.py`, 2 json; ~50 lines | `make verify` green on an unchanged tree (it is red today by I-3); `make test-e2e` output lists `GreedyLockTest.*` as RUN, not 0 matched; a filter edited to match nothing turns the target red (mutation check); the function gate reports the 557-LOC body and its selftest plants the column-0-brace case | none; lands before #5, #6, #10, #12 so their gates can run |
| 5 | **Surface a device fault as a host signal.** After `engine->step()` in `batching_engine.cpp`, `cudaPeekAtLastError()` classified by `cuda_error_is_unrecoverable`; on unrecoverable: cancel in-flight, set `faulted_`, stop the worker (the existing `catch` block body); `collect_sampled_tokens` returns failure when its sync failed; `/health` reflects `faulted_` | D-1 | S1 | `batching_engine.cpp` ~20 LOC, `executor_sampling.cu` ~10 LOC, `utils.cpp` health | a test kernel that faults on purpose (gpu lane): the request gets 5xx, `/health` reports faulted, no later request returns 200 with stale tokens; the three benign sticky-error clears (`engine_workspace_warmup.cpp:52-59`, `cuda_graph.cu:341`, `graph_diag.h:75`) still pass | #4 (the gpu-lane suite must be runnable) |
| 6 | **Make per-request LoRA honest.** Fold `active_lora_` into `content_salt` at `engine_prefill.cpp:230,943` (carry the adapter id on `Request`); serialize `imp_lora_set` against the worker (refuse the switch while `queue_depth() > 0`, or pause the batching engine); validate adapter `r/K/N` against `d_model` and the projection sizes in `Engine::lora_load`, refuse on mismatch; saturating `numel` and a `nbytes` check before the host allocation; state the single-adapter-at-a-time contract in `docs/API.md:54` and `LIMITATIONS.md` | E-1, F1-6, E-9 (adapter row) | S1, S1, S3 | `engine_prefill.cpp`, `request.h`, `engine.cpp`, `lora_adapter.cpp`, `handlers_chat_core.cpp`; ~80 LOC; 2 docs | two adapters sharing a system prompt produce different KV for the shared prefix (test in `test_lora.cpp`, mutation-validated by dropping the salt); an adapter declaring `K = 1e6` on a 4096-d model is refused at load; concurrent requests naming different adapters get a typed 409/400 instead of a race | #4 |
| 7 | **Typed capacity cancel + per-request speculation telemetry + queue metric semantics.** Set `CancelReason::KvCapacity` at the four mid-flight sites; surface `spec_drafted/accepted/verifies` per request in `usage` and as span attributes; make `imp_queue_time_seconds` measure the admission queue and split `imp_queue_depth` into waiting/running; the `speculative` field on `/v1/responses` | C-1, C-6, C-5, C-9 | S2, S2, S2, S3 | `engine_scheduler.cpp`, `engine_prefill.cpp`, `batching_engine.cpp`, `handlers_*.cpp`, `metrics_*.cpp`; ~90 LOC; tests | a decode that exhausts the pool ends with `finish_reason: "capacity"` on the stream and 503 `capacity_error` on the non-stream path (test-server with a tiny pool); `usage` carries acceptance on a spec-on request; `/metrics` shows waiting != running under a burst | #4 |
| 8 | **Decide the GEMM registry, then act.** Owner decision recorded in SETTLED: (a) finish the migration (move the FP8 / NVFP4 / MXFP4 / FP16 arms of `gemm_via_handle_` onto the registry, high risk without a GPU lane) or (b) retire the 9 unreachable registrations + their tests and fix `gemm_kernel_registry.h:13-20`. In the same dispatch: `gemm.moe_imma_prefill` either gains a death date for the three stranded tiers or a resolver reason; the three dead exports (A1-4) and the two `sm_ver >= 90` split-K arms (A1-6) go; `gemm_q4k_fused_moe_prefill` gets a verdict; `MXFP4_KV` gets its `paged_attention_serves_head_dim` case | A1-1, A1-2, A1-4, A1-6, A1-8, A1-5, A1-7 | S1, S2, S3, S3, S3, S3, S3 | (b): ~455 src LOC + ~900 test LOC deleted, ~1000 LOC MoE tiers, ~470 LOC dead arms; (a): ~250 LOC moved across 5 hot files | (b): `rg -n 'instance\(\)\.dispatch'` producers == registered keys, `test_gemm_kernel_registry.cu` pins only live keys, `make verify-fast` green and perf gate unchanged (paired A/B, #10 helps); (a): all 19 keys have a production producer and the dispatch record names the registry tier; either way `moe_imma_prefill` has a recorded verdict and `mxfp4` KV + hd=96 is refused at init not mid-decode | #4; #10 preferred first if (a) |
| 9 | **Third-party licence file.** `THIRD_PARTY_LICENSES.md` at root with the Apache-2.0 text, the SageAttention3 notice, the stb licence pointer; OCI label `MIT AND Apache-2.0`; `check-release.sh` asserts the file exists; owner confirms the derivative-work reading | H-7, H-8 (attribution half) | S2 | 1 new file ~210 lines, `Dockerfile` 1 line, `check-release.sh` ~5 lines | file present in the repo and in the runtime image; `check-release.sh` section fails when it is removed | none |
| 10 | **Calibrate the perf gate.** `make verify-ab`: build `origin/main` into a second image, alternate 3 pairs, report the paired delta; trigger `roofline.yml` (or a documented manual step) on `src/compute/**` and `src/exec/**` changes, not only `tools/roofline/**`; record the threshold rationale next to the number in `perf_baseline.json` | H-3, H-2, H-9 | S2, S2, S3 | `Makefile`, `scripts/verify.sh` ~80 LOC, 1 workflow | a planted -5 % mutant (e.g. one fewer thread block on the decode GEMV) is caught by the paired gate and is invisible to the single-arm 8 % gate (both measured on the same tree the same hour); N=6 unchanged-tree runs produce 0 paired reds | none; GPU time |
| 11 | **Re-wire PDL registration and the L1 carveout, measured.** Restore the max-L1 carveout for the dp4a GEMV family without PDL registration (the NVFP4 pattern at `nvfp4_gemm.cu:320-325`); register the `_mb_` batched NVFP4 GEMVs; audit the 9 paged-decode kernels without `pdl_trigger`; delete the two dead `*_pdl_register()` functions; fix the SPLIT=1 single-sequence GDN scan if the batched SPLIT=2 measures ahead | A2-1 (= A1-3), A2-2, A2-5, A2-3, A2-4 | S2 x5 | `gemm_dp4a.cu`, `activation.cu`, `nvfp4_gemv_batched.cu`, `executor_workspace.cu`, `gdn.cu`; ~60 LOC | alternating A/B @32 and M=1 on Qwen3-8B-Q8_0 (carveout) and Qwen3.8-27B-NVFP4 (batched PDL), 3 pairs each, delta reported with sign; `kernel_resource_baseline.txt` re-pinned; `DegenerationTest.GreedyDeterminism` green (the race `executor_workspace.cu:332-336` records) | #4, #10 |
| 12 | **Instrument batch invariance and close the numerics-golden gaps.** A teacher-forced NLL / top-1 agreement instrument that runs the native-NVFP4 batched decode path at M=1 vs M=32 on the same prompts; kernel-level goldens for Q4_1, Q5_0, Q2_K, Q3_K, Q8_K, E5M2 against the CPU dequant reference, or an explicit "unsupported" refusal at load for formats nobody will gate (Q8_1 refusal lands in #15); fix the FP8 registry copy (W8A8 vs live W8A16) or delete it under #8 | D-2, D-5, A1-1 (numerics half) | S2, S3 | `tools/analysis/` script ~150 LOC; `tests/test_quant_integration.cu` +6 cases | a number: NLL delta and top-1 agreement M=1 vs M=32 on Qwen3-14B-NVFP4, recorded in `docs/PERF.md` with PROV; every `QType` the loader accepts has either a golden or a load-time refusal | #4, #8 |
| 13 | **Doc drift pack.** `ARCHITECTURE.md` block-size and MoE-graph claims (J-1, C-8); 31 missing keys in `imp.conf.example` (J-2); `MODELS.md:41` 268 vs pinned 287.19 (J-6); one measurement copied into 4 places (J-12); `kernel_resource_baseline.txt` vs `attention.h` (A2-6); the `smallM` header naming v2 for `_v1` (A2-7); `gemm_kernel_registry.h:13-20` flag that does not exist; roadmap 2026-bar "per-key rate limit" and adapter rows (E-9); `TEST_INVENTORY.md` re-dated or marked record (I-9); `decode-pipeline` diagnostic `ssm_ok=1` (C-7); `LIMITATIONS.md` "MoE routing uses atomics" (D-4); `IMP_WORKER_TIMING` ad-hoc env (J-10); CHANGELOG entries over 3 lines (J-8); a `docs-layers` row in the root router (J-9); SETTLED re-pins for P0-1 and P0-4 | J-1..J-12, C-7, C-8, A2-6, A2-7, D-4, E-9, I-9, P0-1, P0-4 | S3 | docs and comments only, ~40 hunks | `docs_lint`, `sync_docs --check`, `check_doc_citations` green; every cited line reads what the row says (spot-check list in the axis reports) | none |
| 14 | **Layering and build-cost leftovers.** Delete the dead `runtime/config.h` include in `pre_dequant_phase1_fp16_cache.cu`; move the four dependency-free headers (`runtime/pdl.h`, `runtime/process_diag.h`, `compute/pdl_device.cuh`, `runtime/graph_diag.h`) to `core/` (64 of 88 backward include lines; the scout's simulation shows this does NOT restore a DAG, so the claim is "four layers cleanly below runtime", nothing more); take the two priced drops (`executor.h:31 -> runtime/vram_budget.h`, forward-declare: -40 TUs of 81; `gemm_cutlass_sm120.h:3 -> model/model_config.h`: 156 -> 144); rename `GraphExecutor::runtime_config()` to `dispatch_policy()` (143 sites, prefix rename); decide `runtime/storage_planner.h` and `CudaGraphRunner` placement by the S-33 method; correct `ARCHMAP.md:9-16` to say SCC, not DAG; re-pin the SETTLED anchors for F-10 "zero" and "one `quant -> compute` edge" | P0-1..P0-5, G-1, G-2, G-9 | S3, S2 | ~70 include lines, 4 file moves, CMake paths, 1 rename | `rg -c 'runtime/config.h' src/exec` = 0; `compute/quant/model/memory -> runtime` = 0; reverse-BFS TU counts for `executor.h`, `vram_budget.h`, `model_config.h` at or below the priced numbers; `ci_static_gates.sh` gains a layering rule (allowed edge list) so the count cannot drift back | none |
| 15 | **Config precedence and dead flags.** `runtime.max_seq_len` yields to a CLI/C-API value already set (3 LOC + a `test_config.cpp` binding test); `imp_context_create` rejects the 5 invalid `ImpDType` values for `kv_cache_dtype` and runs the resolver on the valid ones; the 6 writer-less `Overrides::Gemma4` flags either gain a writer (config key or loader) or go, with their 7 read sites; `build_config`'s overrides JSON either gets a caller or is removed; `QType::Q8_1` gets a load-time refusal (no dequant, no registry entry); a C-API contract test drives `ImpConfig` through `imp_context_create` with hostile values; the 4 unwrapped `ImpError` entry points get the try/catch (or a macro for all 24) and `ARCHMAP.md` stops saying "all" | G-4, G-5, G-7, G-8, G-12, G-3, G-10 | S2 x5, S3 x2 | `engine_init_resolver.cpp`, `imp_api.cpp`, `model_config.h`, `tools/imp-server/config*.cpp`, `qtype.cpp`; ~80 LOC; 2 tests | `--max-seq-len 8192` with `runtime.max_seq_len = 4096` in the file resolves to 8192 and logs the override; `imp_context_create` with `kv_cache_dtype = <invalid>` returns `IMP_ERROR_INVALID_ARG`; a Q8_1 GGUF is refused at load with a typed error; every Gemma4 override flag has a `rg` writer outside its declaration | none; before #13 so the doc pack records the final state |

## The three best value-to-blast-radius items

1. **#1** B-1 alone: an S0 device use-after-free on the default model-swap path, fixed by one 8-LOC cleanup function that already exists for its sibling (`sampling_cleanup_dry`). The rest of #1 is the gate that stops the next one.
2. **#2** F1-1 + F1-2: two S0 stack writes on hostile files, ~6 LOC each, plus the fuzz targets that would have found them (the two S0s are exactly in the two unfuzzed parsers).
3. **#4** the dead gates: `make verify` is red on an unchanged tree today and the greedy regression locks have never run from a target; ~30 lines of Makefile/script restore the only output-quality gates the repo has and unblock #5, #6, #7, #12.

Runner-up: **#3** (~60 LOC) closes the only unauthenticated remote path into the parsers of #2.

## What is deliberately NOT in the queue

- Green contexts, KV tier below VRAM, GPU CI lane, reference-correctness gate, soak test: owner decisions or absent hardware, restated in P4.
- Splitting `run_attention`, pimpl on `engine.h`, forward-declaring `config.h` consumers: refuted on measurement (SETTLED C2, G).
- A registry/template rewrite of `weight_map.cpp` or the loader ladders: SETTLED needs a concrete bug first; #2 supplies bounds checks, not a rewrite.
- Block size 32/64 A/B (B-5): a real question with no instrument; it needs #10's paired harness first, then one dispatch with a number. Listed as an open question, not a queue item, until then.

# P4 - Known-and-accepted (restated, not re-argued)

Already on the roadmap, in `docs/LIMITATIONS.md`, `docs/DESIGN_DECISIONS.md` or `docs/audit/SETTLED.md`. Listed so nobody files them again as new findings.

| item | where it is recorded | touched by |
|---|---|---|
| No GPU CI lane; `Test` job dormant on `vars.HAS_GPU_RUNNER`; `make verify-fast` locally is the only kernel gate | SETTLED G F-5 (owner decision 2026-08-03), `docs/LIMITATIONS.md` "The five" #3 | every axis |
| No correctness gate against a reference implementation (#1571); no soak test (#1642) | `docs/LIMITATIONS.md` "Gates that do not exist" | D, I, J |
| Generation half of the HTTP contract deselected in CI (#1600/#1559); server streaming path never in the perf gate (#1685) | `docs/LIMITATIONS.md` "Untested code paths" | E, H, I |
| `/admin/suspend`, `/admin/resume`, `server.model_swap` implemented, ungated | `docs/LIMITATIONS.md`, `docs/FEATURES.md` (#1680) | B, E, F1, F2 |
| Launch-coupled idle ~8 % @32; paced serving prefill on dense (`prefill_chunk_decode_cap` 1024) | `docs/roadmap.md` Open 1, 2 | C |
| Long context half closed (sparse decode opt-in, NIAH price); MLA, prefill sparsity, StreamingLLM as the only KV-pressure answer | `docs/roadmap.md` Open 3 | B, C |
| Speculation adapts per request only in chain depth; drafter choice global; W=2 tree measured no gain | `docs/roadmap.md` Open 4, Closed "speculation tree" | C |
| Recurrent-state paging (pinned host tier since 2026-09-02) | `docs/roadmap.md` Open 5 | B |
| `--calib` at wide GQA; quantizer refuses 3-D stacked experts; no audio; no video; one VL tower family | `docs/roadmap.md` Open 6-10 | A1, F1 |
| No KV tier below VRAM: DO NOT BUILD (6.5x spill cliff, ~165 us blocking transfer) | `docs/roadmap.md` Open 11, `AUDIT.md` B84/B36 | B |
| Hybrid pp512 `gemm_cublas` hole, priced and parked | `docs/roadmap.md` Open 12 | A1 |
| Green contexts unavailable on sm_120 (`cudaDevResourceGenerateDesc` fails); SM reconfiguration in `step_schedule()` unreachable | `docs/LIMITATIONS.md` | A2, C |
| DNS rebinding on `--allow-remote-images` | `docs/LIMITATIONS.md` | F2 |
| VRAM planner reserve is an estimate with a floor, no retry when wrong (#1631); library reserve constant when the cache path does not outlive the process | `docs/LIMITATIONS.md` | B |
| Memory plan is a shadow until A7 step 6; F-12 re-scoped (48 `VRAMAllocator` refs are the post-upload acquisition path, not a backlog) | SETTLED G | B |
| GGUF batched decode rows (M 2..32) read the 4-bit overlay with 4-bit activations (#1897); batched vs solo not bit-identical | `docs/LIMITATIONS.md` "Known-bad" | D |
| Untested formats Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q8_K, FP8 E5M2; Llama-4 and Phi-4 without a gate | `docs/LIMITATIONS.md` | D, I |
| Prefix-cache hit not bit-equal to fresh prefill (#1314); cross-context reproducibility; CUTLASS grouped GEMM outside `runtime.deterministic` (F-17) | `docs/determinism.md`, SETTLED E | D |
| Speculation off for logprobs / constraints / tools; `cache_control.ttl` accepted, not modeled; OTLP traces only | `docs/LIMITATIONS.md`, skill server-api, `docs/roadmap.md` Closed | E |
| JSON Schema assertion keywords imp cannot enforce are a 400 (#1567); `pattern` that does not compile is unenforced at 200 | `docs/LIMITATIONS.md` | E |
| cuBLASLt algorithm selection unpinned by design: estimator fixed (#1228), persistence R-16 REJECTED with two refuted designs | SETTLED G F-9 | A2, H |
| `runtime.prefill_overlap` default OFF is a measured verdict (neutral at 32 streams) | `docs/roadmap.md` lever ledger | C |
| `attention.fa2_hd256_bkv=32` opt-in on a measured PPL trade (+0.53 %) | `docs/roadmap.md` lever ledger | A2 |
| FP8 prefill unavailable on sm_120 (cuBLAS `NOT_SUPPORTED`); FP8 GDN-projection prefill REFUTED 2026-09-01 | `docs/plans/2026-08-31-fp8-ssm-prefill.md` | A2 |
| Legacy cuBLAS attention tier is a deliberately retained tier for Gemma-4 hd=512 (S-8); MoE prefill ladder is a designed 4-tier ladder (S-11); `executor_forward_moe_legacy.cu` is a reachable floor | SETTLED A, C | A1 |
| `run_attention` split refused on measurement (S-32); `engine.h` fan-in cut instead of pimpl (S-33, F-24); `DispatchPolicy` extraction done (F-10) | SETTLED C2, G | G |
| DEBT_LEDGER item 3: `IMP_ALLOC_INTERPOSE` OFF everywhere, so `steady_state_allocations()` reads zero in every shipping build | `docs/audit/DEBT_LEDGER_2026_08_21.md` | B |
| No default credential: without `--api-key` every endpoint is open; no TLS in imp, terminate at a reverse proxy; `/metrics` unauthenticated by default (#1207) | `docs/DEPLOYMENT.md` | F2 |
| Single-author project, no security response process | `docs/LIMITATIONS.md` "The five" #5 | F1, F2 |
| Records (`docs/roadmap.md`, `CHANGELOG.md`, `AUDIT.md`, `docs/{archive,audit,plans}/`) are append-only and lint-exempt | `scripts/docs_lint.py` | J |

# P5 - Dispatch log

Appended as the P3 queue is worked, one row per dispatch. The P3 table above stays as
written on 2026-09-05; this is the only part of the file that moves. `SETTLED.md` section H
carries the per-finding verdicts and must list the same closed set.

| # | findings | status | landed | gate outcome | left out |
|---|---|---|---|---|---|
| 1 | B-1, B-2, B-3 | ✅ CLOSED | dispatch #1 PR | `SamplingTest.LogitBiasRearmsAfterStaticReset` + `GreedyScratchRearmsAfterStaticReset` green, red with the hook removed (mutation run); `tools/check_static_reset.py` 25/38 candidate TUs re-arm, 1 allowlisted, selftest 11/11; seven TUs re-arm (the six named plus `mmq_q4k_imma_tile.cu`'s weight-plane cache) | the two-model swap bit-identity test (needs two models in one `test-e2e` process); `s_h_normed` OOB asserted by the capacity field, not by a test |
