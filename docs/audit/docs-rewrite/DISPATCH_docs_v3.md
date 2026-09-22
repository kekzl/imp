<!--
layer: L3
audience: agents
verified: 2026-09-23
commit: 9cbb8004
-->

# DISPATCH docs v3: fact log

Every section removed, moved or corrected in the prose cleanup. Base `9cbb8004`. Inputs: [`DOCS_INVENTORY.md`](DOCS_INVENTORY.md), [`TARGET_STRUCTURE.md`](TARGET_STRUCTURE.md). Fact check: every token class (code span, flag, issue ref, path, decimal, integer >= 3 digits, URL, KEY=VALUE) of the old file must appear in the new file set or have a row below.

## Orchestrator

| file | deleted or moved section | fact preserved where |
|---|---|---|
| `docs/usage.md` | whole file | `docs/CONFIG.md` (CLI, `imp.conf`, C API), `CONTRIBUTING.md` (requirements, build) |
| `docs/architecture.dot`, `.svg`, `.png` | whole files, unreferenced after the Mermaid diagram | `docs/internals/ARCHITECTURE.md` |
| `AUDIT.md` B25 | citation `docs/usage.md` | `docs/CONFIG.md` (`vram.kv_fraction` row) |
| `docs/README.md` | layer prose, "Contracts" list | index tables; `usage.md` row -> `CONFIG.md`, `API_FEATURES.md` row added |
| 61 paragraphs > 2 sentences (15 files) | prose form | same file, bullets or tables; tokens checked by `fc.sh` |
| `docs/internals/MEMORY.md` | A2-A6, invariant compliance, open questions (23 tokens absent from the rewrite: `Owned<T, Tier::ModelResident>`, `BlockPool<Stride>`, `+256.00 MiB`, #1939, #1940, ...) | verbatim in `docs/archive/memory_census_2026.md` "Design draft A2-A6" |
| `docs/internals/ARCHITECTURE.md` | `src/compute/sampling.{h,cu}`, `<\|image_pad\|>` | formatting only: component table lists `sampling.cu`, `sampling.h`; pipe escaped inside the table |

| file | claim | was | now | source |
|---|---|---|---|---|
| `tools/imp-cli/args.cpp` `--help` | `--max-tokens` default | 256 | 8192 (16384 with `--interactive`) | `tools/common/args_common.h:28`, `tools/imp-cli/mode_interactive.cpp:22` |
| `docs/CONFIG.md` | calibration flag | `--calibrate-out` | `--calibrate <out>` | `imp-cli --help`, `tools/imp-cli/args.cpp:123` |
| `CONTRIBUTING.md` | bug report version command | `./build/imp-cli --version` | `git rev-parse --short HEAD` | `imp-cli --version` exits "Unknown argument: --version" |

Phase 3 checks on the rewritten set: 34 `make` targets exist (`make -n`), 39 `scripts/`/`tools/` paths exist, 100 config keys resolve in `src/runtime/config.cpp`, 150 `--flags` checked against `imp-cli`/`imp-server`/`imp-quantize --help` (remaining unmatched ones belong to nsys, ncu, docker, vLLM, HF CLI, scripts), 46 `IMP_*` names found in code.

## Agent A log - README.md, docs/QUICKSTART.md, docs/FEATURES.md, docs/MODELS.md, docs/GOAL.md

### 1. Deleted or moved sections

| file | deleted or moved section | fact preserved where |
|---|---|---|
| README.md | "Is imp for you" (yes/no table + llama.cpp/vLLM prose) | hardware facts -> README "Requirements"; competitive framing -> link `docs/PERF.md#competitive-standing` |
| README.md | "How it works in five minutes" (full narrative) | link `docs/internals/ARCHITECTURE.md`; removed: the class-name term `Request` (single low-value token, narrative not restated) |
| README.md | "Where imp loses" (5-point list) | link `docs/LIMITATIONS.md` - same 5 points already present there as "Five things to weigh first" |
| README.md | "Build from source" full section | kept, condensed to one code block + link `CONTRIBUTING.md` |
| README.md | "How this was built" | condensed to 2 lines, kept links `docs/audit/`, `docs/MISSION_JOURNAL.md` |
| README.md | llama.cpp six-model competitive table ("How fast is it, really") | duplicate of `docs/BENCHMARKS.md` "Competitive sweep 2026-08-30" (verbatim, including the Gemma-4 row README had omitted "for length") |
| README.md | vLLM streams comparison table (6 rows) | rows 1,3,4,6 match `docs/BENCHMARKS.md` "imp vs vLLM at concurrency"; row 8 (dense Qwen3-14B, 3948.9 vs 3817.6, +3.4%, vLLM flags, llama.cpp digest `c49f4d48`) was dropped by the concurrent BENCHMARKS.md rewrite (882->306 lines) - restored as a compact provenance line in README so it is not lost from the corpus |
| README.md | NVFP4 KV capacity paragraph (126 432 / 86 848 tokens, PROV commit=3921547d) | moved to `docs/MODELS.md` Qwen3.8-27B row, own PROV block kept; not duplicated in PERF/BENCHMARKS/MODELS at base, so kept rather than linked away |
| README.md | embedded `webui.png` screenshot (`<img>` + caption) | removed: screenshot kept only in `docs/QUICKSTART.md` (relative path `webui.png` = same asset `docs/webui.png`), README links there |
| docs/QUICKSTART.md | links to `usage.md` (CLI reference, C API) | renamed per campaign: `docs/CONFIG.md` (`#c-api`), file confirmed present in worktree |
| docs/FEATURES.md | "Model architectures" table (14 rows, arch-level ✅/🟡 status) | link `docs/MODELS.md`; unique facts folded into MODELS.md: Mixtral/Llama-4 untested (`tests/test_moe_executor.cu`, #1680), Phi-4 alias note, nomic-embed bidirectional/no-KV/mean-pooled, DeepSeek-V2 "latent-KV decode is opt-in", gpt-oss "learned attention sinks", Gemma-3 SigLIP encoder, GDN family shares the `qwen3_5` arch path |
| docs/FEATURES.md | Qwen3/Qwen3.6-MoE/Gemma-3/Gemma-4/Nemotron-H/DeepSeek rows (✅, no extra detail) | removed: duplicate of `docs/MODELS.md` per-checkpoint rows, which already imply the arch works |
| docs/FEATURES.md | Qwen3-VL / Qwen3.6-35B-A3B vision `make test-vision` command + #1680 | removed: test-infra detail, not a model fact; the "not tested for vision" substance is already in `docs/MODELS.md` Vision table |
| docs/GOAL.md | "What imp is NOT" (6 prose bullets) | condensed to table, link `docs/DESIGN_DECISIONS.md`. Multi-GPU and CPU-engine rows kept inline since DESIGN_DECISIONS.md (rewritten concurrently to a single table, no per-decision headings any more) still carries the same two facts; training/mobile/model-zoo/research-playground rows kept inline because DESIGN_DECISIONS.md does **not** carry them (checked, not present) |
| docs/GOAL.md | "What best on 5090 requires" (4 subsections, prose bullets) | converted to a 4-row table, no facts dropped (fact-check clean) |
| docs/GOAL.md | "Agentic surface" (6 prose bullets) | converted to a 6-row table, no facts dropped |
| docs/GOAL.md | grouped-GEMM #374 narrative (14,562->17,521 pp512, 1.14-1.32x pre-#374) | folded into the "Definition of best" table, row 2 cell (kept, not dropped) |
| docs/GOAL.md | Mission / Release bar / North-star prose | converted to tables/shorter prose throughout for the 120-line budget; every dated number verified present via `fc.sh` |

### 2. Corrected facts

| file | claim | was | now | source |
|---|---|---|---|---|
| docs/MODELS.md (fact moved from docs/FEATURES.md) | Phi-4 alias path:line citation | `src/model/model.cpp:299` | `src/model/model.cpp:328` | `rg -n phi src/model/model.cpp`: line 299 is unrelated (`ModelSamplingDefaults` temperature/top_p init); the actual `{"phi3", ModelArch::LLAMA}` alias entry is at line 328 |
| docs/FEATURES.md | rerank cross-check Makefile citation | `Makefile:335` | `Makefile:365` | `sed -n` on `Makefile`: line 335 is the `bench-agentic` concurrency target; `COMPARE_URL` first appears at line 365 |

### 3. Removed unverifiable claims

None. Every fact dropped from these five files was either a duplicate of an owner doc (table 1) or a deliberate budget/dedup cut with the fact preserved elsewhere; nothing was removed for being unverifiable. Two stale `path:line` citations were corrected in place (table 2) rather than removed, since `rg` located the right line.

### Note: MODELS.md em/en dash cleanup

`docs/MODELS.md` carried 25 em dashes (` - `) in prose copied near-verbatim from the base file. Replaced all with ` - ` (brief "Do not" rule + user's global no-em-dash rule); no other file in this assignment had any.

## Agent B log: usage.md / API.md / DEPLOYMENT.md rewrite

### 1. Deleted or moved sections

| file | deleted or moved section | fact preserved where |
|---|---|---|
| docs/usage.md | `## Requirements` | merged into `CONTRIBUTING.md` `## Prerequisites` |
| docs/usage.md | `## Build` (CMake option table, Docker/host build commands) | merged into `CONTRIBUTING.md` `## Build` |
| docs/usage.md | `## Configuration - imp.conf` | `docs/CONFIG.md` `## imp.conf` |
| docs/usage.md | `## CLI - imp-cli` (all flag docs, `<details>` block) | `docs/CONFIG.md` `## CLI - imp-cli` (converted prose/`<details>` block to a table) |
| docs/usage.md | `### Machine-readable output - --json` | `docs/CONFIG.md` `## --json - machine-readable output` |
| docs/usage.md | `## Server - imp-server` intro, endpoints list, warm cache, suspend-to-RAM, model identity, context-window auto-detection | endpoints list -> `docs/API.md` `## Endpoints` (already owner); warm cache -> `docs/DEPLOYMENT.md` `## Compose`; suspend-to-RAM detail -> `docs/DEPLOYMENT.md` `## Health, metrics, lifecycle` table; model identity / context-window -> `docs/DEPLOYMENT.md` `## Health, metrics, lifecycle` (new "Context-window reporting" paragraph) |
| docs/usage.md | Server-only flags table | `docs/CONFIG.md` `## Server flags` (per-request caps table intentionally left owned by `docs/DEPLOYMENT.md` `## Auth and exposure`, as in the original - no duplication introduced) |
| docs/usage.md | curl / streaming / OpenAI-SDK examples | `docs/CONFIG.md` `## Server flags` section |
| docs/usage.md | `### LoRA adapters` | `docs/CONFIG.md` `## LoRA adapters` |
| docs/usage.md | `## C API` | `docs/CONFIG.md` `## C API` |
| docs/usage.md | `## Project Structure` | `docs/CONFIG.md` `## Project structure` (per BRIEF fallback: ARCHITECTURE.md is owned by another agent, CONTRIBUTING.md's 150-line budget could not fit it) |
| docs/API.md | `## Constrained decoding`, `## Tool calling` (incl. thinking on/off, reasoning budget), `## Images` | `docs/API_FEATURES.md`, verbatim except header/cross-links |
| docs/DEPLOYMENT.md | none removed, only reorganised | n/a |
| CONTRIBUTING.md | `## Prerequisites`, `## Build` (existing content) | rewritten in place, merged with usage.md's Requirements/Build (see corrections below) |

### 2. Corrected facts

| file | claim | was | now | source |
|---|---|---|---|---|
| docs/CONFIG.md (from usage.md), CONTRIBUTING.md | canonical CUDA toolchain | `13.3` | `13.4.1` (minimum stays 13.2, CMake-enforced) | `Dockerfile:25,132` (`FROM nvidia/cuda:13.4.1-...`) |
| CONTRIBUTING.md | host compiler standard | "C++20 host code, CUDA C++20 device code", "GCC 12+, Clang 15+" | "C++23 host code, CUDA C++23 device code", "GCC 13+, Clang 16+" | `CMakeLists.txt:12-13,20-21` (`CMAKE_CXX_STANDARD 23`, `CMAKE_CUDA_STANDARD 23`); matches root `CLAUDE.md` ("C++23/CUDA") which usage.md already had right |
| docs/CONFIG.md (from usage.md) | `--max-tokens` default | `256` | `8192` (shared with `imp-server` since #1209's `CommonArgs` merge) | `tools/common/args_common.h:28` (`int max_tokens = 8192;`, comment: "8192 for both tools: imp-cli's old default of 256 predates reasoning models") |
| docs/CONFIG.md (from usage.md) | CLI flag list | missing `--prompt-file`, `--calibrate`, `--mirostat-tau`, `--mirostat-eta`, `--token-trace`, `--no-fp8-prefill`, `--dual-path-quant` | all seven added to the CLI flags table | `tools/imp-cli/args.cpp`, `tools/imp-cli/args.h`, `tools/common/args_common.cpp` |
| docs/CONFIG.md (from usage.md) | test suite size | "GTest suite (2125 cases across 8 binaries)" | "GTest suite, 8 module binaries" (specific case count dropped, see removed-claims table) | `CMakeLists.txt:691-1078` (8 `imp_add_test_module` calls: test-core/text/compute/attention/quant/kv/moe-gdn/e2e) |

### 3. Removed unverifiable / stale claims

| file | claim | why unverifiable |
|---|---|---|
| docs/usage.md (test count) | "2125 cases across 8 binaries" | The raw `TEST(`/`TEST_F(`/`TEST_P(` macro count in `tests/` is now 3015 (42% higher), and `TEST_P` instantiation multiplies at runtime, so an exact live case count needs a full GPU rebuild + `--gtest_list_tests` per binary, out of scope for a docs-only fact-check. The "8 binaries" part is verified and kept; the specific case count is dropped rather than replaced with another guess. |

### 4. Intentional renames (not defects) - sanctioned by BRIEF "Renames in this campaign"

| file | old reference | new reference |
|---|---|---|
| docs/DEPLOYMENT.md | `[usage.md](usage.md#configuration--impconf)` | `[CONFIG.md](CONFIG.md#impconf)` |
| all 5 files | (n/a - no file linked to a moved `API.md#anchor`, see fc.sh missing-token list: `usage.md` is the only surviving miss and it is this rename) | |

### 5. Formatting-only fc.sh artifacts (fact present, token literal differs)

| src token | dst location | why it doesn't match verbatim |
|---|---|---|
| `` `--max-images-per-request <n>` `` (docs/usage.md, one combined code span) | `docs/DEPLOYMENT.md` Auth-and-exposure table: `` `--max-images-per-request` `` (flag) + `8` (default) in separate table columns | Same fact (flag name + default 8), split into table columns instead of one prose code span. Not duplicated in `docs/CONFIG.md`'s server table by design - matches the pre-rewrite split (usage.md's own server-flags table never had this flag either, only DEPLOYMENT.md did). |

### Per-file line counts and fc.sh result

| file | before | after | fc.sh missing (post-log) | corrections |
|---|---|---|---|---|
| docs/usage.md -> docs/CONFIG.md + CONTRIBUTING.md + docs/DEPLOYMENT.md + docs/API.md | 487 | deleted (`rm docs/usage.md`) | 3 (all logged above: `--max-images-per-request <n>` formatting, `13.3` correction, `2125` stale) | 3 |
| docs/API.md -> docs/API.md + docs/API_FEATURES.md | 473 | API.md 276, API_FEATURES.md 208 | 0 | 1 (added missing `store: false` /v1/responses detail to the Endpoints table, was present in usage.md's Server section, not previously in API.md's own endpoint row) |
| docs/DEPLOYMENT.md | 254 | 250 | 1 (`usage.md` -> intentional rename to CONFIG.md, see table 4) | 0 |
| CONTRIBUTING.md | 143 | 136 | 1 (`13.3` correction, same as CONFIG.md row above, logged once) | 2 (C++ standard, CUDA version) |

Orchestrator note: nothing left for another agent to fix inside these five files. Cross-file link inventory (files outside my assignment that link to `usage.md` or would need `API.md#anchor` updates) reported separately in my final message, not edited here per BRIEF.

## Log: agent C (LIMITATIONS.md, TROUBLESHOOTING.md, DESIGN_DECISIONS.md, determinism.md)

### 1. Deleted or moved sections

| file | deleted or moved section | fact preserved where |
|---|---|---|
| docs/LIMITATIONS.md | "Speculative decoding is not universally profitable" thinking-traffic ledger + adaptive-chain-depth detail | condensed row + `docs/archive/limitations_detail_2026.md#speculative-decoding-economics` |
| docs/LIMITATIONS.md | "MTP released for one model class" two-accept-rate reconciliation + third-row-defect paragraph | model-class table kept inline; reconciliation moved to `docs/archive/limitations_detail_2026.md#nemotron-mtp-defect` |
| docs/LIMITATIONS.md | "MTP speculation truncates answers" full 4-pass investigation (~185 lines) | condensed row + `docs/archive/limitations_detail_2026.md#mtp-truncation` |
| docs/LIMITATIONS.md | "Speculative decoding does not reproduce non-speculative greedy output on a GDN hybrid" full investigation incl. 5-hypothesis table | condensed row + `docs/archive/limitations_detail_2026.md#mtp-vs-greedy-divergence` |
| docs/LIMITATIONS.md | "A speculative arm is not byte-stable across processes" full detail | condensed row + `docs/archive/limitations_detail_2026.md#mtp-cross-process-stability` |
| docs/LIMITATIONS.md | "MTP head accepts 75.0% of its first draft" full detail incl. 6 ruled-out hypotheses | condensed row + `docs/archive/limitations_detail_2026.md#mtp-acceptance-rate` |
| docs/LIMITATIONS.md | "RESOLVED (2026-08-18)" MTP GDN launch-defect fix, pre-snapshot superseded block | `docs/archive/limitations_detail_2026.md#mtp-gdn-hybrid-launch-fix-resolved` (row 17's numbers already carry the outcome inline) |
| docs/LIMITATIONS.md | Nemotron-3.5 MTP economics narrative (41%/39% accept, 51% decode cost, DSpark -42%) duplicated in DESIGN_DECISIONS.md's "Speculative decoding stays opt-in" | removed: duplicate of owner `docs/DESIGN_DECISIONS.md`; LIMITATIONS row 16 links there instead of restating |
| docs/LIMITATIONS.md | "Batched and solo decode disagree" full measurement narrative (already fuller in determinism.md's batch-invariance section) | removed: duplicate of owner `docs/determinism.md`; row links there and to `PERF.md` |
| docs/LIMITATIONS.md | "Identical seeds can diverge" mechanism list | kept as condensed row, full list stays owned by `docs/determinism.md` (already was, per original doc's own cross-reference) |
| docs/TROUBLESHOOTING.md | none removed outright; "content empty" 74-turn / 54-turn corroboration tables condensed into prose within the row | numbers folded into the same table row (74/74, 53/54, character counts, `5 005` token ablation) |
| docs/DESIGN_DECISIONS.md | per-decision `##` headings and narrative prose framing | converted to one table, one row per decision; all reason/measurement text kept in the row |
| docs/determinism.md | per-limit `###` headings (1,2,3,5,4,6) and narrative framing | converted to one known-limits table, renumbered 1-6 in table order (plus 2 un-numbered #1314 rows for the prefix-cache-hit / batch-invariance scope items, which the source itself did not number) |
| docs/determinism.md | "Two probes" paragraph + 4-arm probe result table (#1314) | kept, moved into a short supporting block directly under the known-limits table |
| docs/determinism.md | `imp.conf.example` retraction note ("An earlier revision recommended `runtime.deterministic_gemm`... that recommendation was wrong and is retracted") | removed: historical self-correction about a past doc revision that this rewrite does not carry forward; nothing in the new doc repeats the retracted claim, so the retraction is moot (style rule: delete "previously"/history) |
| docs/DESIGN_DECISIONS.md | old-state PDL detail: "the audit's summary rows no longer count PDL as working sm_120a idiom while its evidence file records `griddepcontrol: 0`", and `pdl.h`'s old claim | removed: superseded by the 2026-08-31 device-half fix now documented in the same row; `griddepcontrol` is no longer 0 after that fix, so restating the old evidence-file value would be misleading without heavy extra framing |

### 2. Corrected facts

| file | claim | was | now | source |
|---|---|---|---|---|
| docs/LIMITATIONS.md | file:line for the six-request-features speculation gate | `src/runtime/engine_spec_ngram.cpp:281-283` | `src/runtime/engine_spec_ngram.cpp:143-144` | verified with `grep -n` against the worktree at `9cbb8004`; code moved since the doc's 2026-09-13 `verified:` date |
| docs/LIMITATIONS.md | file:line for recurrent-snapshot restore-on-prefix-hit | `src/runtime/engine_sampling_stop.cpp:262` | `src/runtime/engine_sampling_stop.cpp:338` | same |
| docs/LIMITATIONS.md | file:line for JSON Schema `additionalProperties`/pattern-compile-failure warn | `json_schema.cpp:535` | `src/compute/json_schema.cpp:624` (the `compile_patterns()` warn site) | same |
| docs/LIMITATIONS.md | file:line for enum-as-quoted-string FSM logic | `schema_constrain.cu:733` | `src/compute/schema_constrain.cu:288` | same |
| docs/LIMITATIONS.md | file:line for the soak-test comment in the metrics file | `tools/imp-server/metrics_memory.cpp:45` | `tools/imp-server/metrics_memory.cpp:79` | same |
| docs/LIMITATIONS.md | file:line for the soak-test comment in the memory-backend test | `tests/test_memory_backend.cpp:215` | `tests/test_memory_backend.cpp:236` | same |
| docs/LIMITATIONS.md | file:line for the soak-test comment in alloc_interpose | `src/memory/alloc_interpose.cpp:111` | `src/memory/alloc_interpose.cpp:136` | same |

All other cited config keys (`runtime.deterministic`, `runtime.deterministic_gemm`, `runtime.warmup`,
`runtime.decode_burst`, `runtime.cuda_graphs`, `runtime.gdn_batched_decode`, `runtime.no_pdl`,
`kv_cache.swa_snapshot_mb`, `kv_cache.max_blocks`, `kv_cache.dtype`, `server.prefix_cache`,
`server.green_contexts`, `server.recurrent_snapshot_mb`, `server.recurrent_snapshot_host_mb`,
`vram.library_reserve_cache`, `gemm.nvfp4_smallm`, `gemm.nvfp4_gdn_proj_prefill`,
`moe.force_host_experts`, `moe.skip`, `speculative.mtp_k`, `speculative.mtp_adaptive_k`,
`speculative.verify_nvfp4_gemm`, `speculative.mtp_econ_min_emit`, `speculative.miss_burst`,
`speculative.mtp_nvfp4_head`, `attention.no_qknorm_fused`, `gdn.chunkwise_scan`,
`diagnostics.spec_trace`, `diagnostics.dump_hidden_dir`, `diagnostics.mtp_prenorm_h`,
`runtime.think_answer_reserve`, `runtime.max_seq_len`, `runtime.max_batch_size`), the server default
`max_tokens=8192` (`tools/imp-server/handlers.h:233`), `default_think_budget=0.5f`
(`tools/imp-server/handlers.h:262`), `--allow-remote-images` (`tools/imp-server/args.h:55-60`),
Makefile targets (`dev`, `dev-test`, `dev-clean`, `test-unit`, `verify`, `verify-fast`, `verify-ab`,
`roofline-measure`, `asan`, `sanitize`), and `process_diag_deterministic_gemm()`'s "5 files, 7 reads"
count (confirmed 2+1+1+2+1=7 by grep) were checked against the worktree and found correct as
written; no change needed.

### 3. Removed unverifiable claims

| file | claim | why unverifiable |
|---|---|---|
| (none) | - | every claim in these four files was either confirmed against the worktree's `src/`, `tools/`, `Makefile`, `scripts/`, or is a dated PROV-tagged measurement (kept verbatim as a record per the brief: "measurements are records, keep them with their provenance, do not verify them") |

### Regex-artifact false positives in fc.sh (not real gaps)

`fc.sh`'s tokenizer reads backtick spans without excluding embedded newlines correctly across the
*original* file's own line-wrapped PROV blocks. Two fragments it extracts from `docs/LIMITATIONS.md`
at base commit (`"; answer bytes compared with"`, `"; divergence offsets from"`) are byte-ranges
between two backticks that straddle a hand-wrapped line break in the ORIGINAL text, not real
citable facts. Both PROV blocks are preserved verbatim (reflowed) in
`docs/archive/limitations_detail_2026.md` under `#mtp-vs-greedy-divergence` and
`#mtp-cross-process-stability`; the actual content (`cmp`, `/metrics` deltas, card idle, kernel
identity from the dispatch log) is all present. No fix applied; re-wrapping to byte-identical line
breaks was judged not worth chasing.

### Path-style non-issues (not real gaps)

`docs/determinism.md` and `docs/internals/BENCHMARKING.md` are flagged missing from
`docs/LIMITATIONS.md`/`docs/determinism.md` only because the rewritten files use relative links
(`determinism.md`, `internals/BENCHMARKING.md`) from within `docs/`, correctly, instead of the
original's `docs/`-prefixed inline-code references. Both targets are linked and resolve; this is a
style improvement, not a lost fact. No fix applied.

### Table-syntax fix (not a content change)

`docs/LIMITATIONS.md`'s Gemma-4 turn-framing template
(`<|turn>user\n...<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`) contains literal `|`
characters that broke the 4-column table (11 fields instead of 6). Moved out of the table cell into
a fenced code block directly below the "Model-specific blockers" table; same fix applied to one
archive-file table row (`<|im_end|>` -> `EOS (\`248046\`)`, the literal token is preserved elsewhere
in the same file inside fenced blocks, so no fact was lost).

### Return

| file | lines before | lines after | fc.sh missing (final) | corrections |
|---|---:|---:|---:|---:|
| docs/LIMITATIONS.md | 792 | 107 | 11 (2 regex artifacts, 2 path-style, 7 line-number corrections, all logged) | 7 |
| docs/TROUBLESHOOTING.md | 234 | 48 | 0 | 0 |
| docs/DESIGN_DECISIONS.md | 232 | 24 | 2 (both deliberate: superseded PDL detail) | 0 |
| docs/determinism.md | 295 | 75 | 2 (1 path-style, 1 deliberate retraction-note drop) | 0 |
| docs/archive/limitations_detail_2026.md (new) | - | 492 | (destination file, not checked as its own source) | - |

New file: `docs/archive/limitations_detail_2026.md` (layer L1, audience records), holding the full
MTP speculative-decoding investigation series under 7 headings:
`#speculative-decoding-economics`, `#nemotron-mtp-defect`, `#mtp-truncation`,
`#mtp-vs-greedy-divergence`, `#mtp-cross-process-stability`, `#mtp-acceptance-rate`,
`#mtp-gdn-hybrid-launch-fix-resolved`. Linked from 6 rows in the rewritten `LIMITATIONS.md`.

Anchors and relative links checked programmatically (custom script, not `scripts/check-release.sh`
since that needs the full repo's link graph): all links in the 5 files resolve, all anchors exist.
No links point INTO these files from elsewhere in the repo were checked (out of scope: other agents
own the referring files); orchestrator should verify no other doc links to
`docs/LIMITATIONS.md#known-bad-and-known-limited-behaviour`-style old per-item anchors, since the
old file used bullet lists (no anchors) for those items, so no such inbound anchor links should have
existed. `TARGET_STRUCTURE.md` (read via `git show 4bcb25a9:docs/audit/docs-rewrite/TARGET_STRUCTURE.md`
since it is not yet merged into this worktree's base `9cbb8004`) confirms `docs/LIMITATIONS.md` is
the sole canonical owner of "Limitations, untested paths" and `docs/DESIGN_DECISIONS.md` of
"Deliberate absences" - both cross-reference each other and `docs/determinism.md` per that map.

No other files touched.

## Agent D log: docs/PERF.md, docs/BENCHMARKS.md, docs/quantization.md, docs/internals/QUANT_PIPELINE.md

Note: `docs/audit/docs-rewrite/TARGET_STRUCTURE.md` did not exist in the worktree (checked at
base commit `9cbb8004` and on disk); proceeded on the explicit per-file assignment given in the
task message (budgets, split points, archive targets), which is more specific than BRIEF.md's
generic pointer to that file. Flagging for the orchestrator in case another agent still needs it.

### 1. Deleted or moved sections

| file | deleted or moved section | fact preserved where |
|---|---|---|
| docs/BENCHMARKS.md | entire pre-rewrite file (all sweeps 2026-06 through 2026-09-08: three competitive re-sweeps, imp-vs-vLLM concurrency progression + nsys profiling attribution, batched-decode optimization history, full agentic-reliability tables) | moved verbatim to new `docs/archive/benchmarks_pre_v0.44.md`; current BENCHMARKS.md keeps only the latest number per section with date/commit/model/quant/command |
| docs/quantization.md | `#### What --calib does` (full AWQ transform/fold mechanism, calibration-file provenance) | new `docs/archive/quantization_awq_findings.md` |
| docs/quantization.md | `#### Head-to-head against a Modelopt export` | archive; one-line summary (9.9252 vs 10.0301, Qwen3-14B) kept in quantization.md Quality section |
| docs/quantization.md | `#### Calibrating a model that will not fit, and what it exposed` | archive |
| docs/quantization.md | "Why it flips between 1.7B and 14B" group-interaction tables | archive; production rule (`--calib-groups BD` on wide-GQA) kept in quantization.md |
| docs/quantization.md | "Splitting the tie... REFUTED" + "second variant... REFUTED" | archive |
| docs/quantization.md | `#### Fused layers share one tensor scale` deep measurement (29.42/30.40/31.05 PPL) | archive; one-row mention kept in quantization.md Roles table |
| docs/quantization.md | `#### Roles that stay full precision, and why` (RMSNorm-offset root cause investigation, gate-share PPL tables) | archive; short Roles table kept in quantization.md |
| docs/quantization.md | `#### MoE, and two roles that must stay full precision` (bisection table, gpt-oss/Gemma-4 destack quality numbers) | archive; one Roles-table row kept in quantization.md |
| docs/quantization.md | `#### Refuted: micro-scale search` | archive; one-line pointer kept in "Choosing a quant" |
| docs/quantization.md | `### NVFP4 internal pipeline` (dense/MoE pipeline diagrams) | merged into `docs/internals/QUANT_PIPELINE.md` "## NVFP4 walkthrough" |
| docs/quantization.md | `#### What the loader enforces` deep example (Nemotron-3.5-Lightning case, `config_groups`, `targets`, `NVFP4 inventory:` log line) | condensed in quantization.md; full mechanics moved to `docs/internals/QUANT_PIPELINE.md` "## Loader enforcement (NVFP4 checkpoints)" |
| docs/quantization.md | Modelopt CLI workflow (`pip install`, `python -m modelopt.llm.ptq`, quant-modes table) | moved to `docs/internals/QUANT_PIPELINE.md` "## Producing NVFP4 outside imp-quantize" |

### 2. Corrected facts

| file | claim | was | now | source |
|---|---|---|---|---|
| docs/PERF.md | `moe.expert_cache_budget_pct` default | "Default stays 15" | Default is `0` (automatic: free VRAM minus allocator headroom and a state/KV floor), since #2069; flat 15% pre-#2069 starved a 56 GiB host-resident model | `src/core/config/moe.h:26` comment; `1383e87d` (#2069) confirmed ancestor of base commit `9cbb8004` via `git merge-base --is-ancestor` |
| docs/PERF.md | line citation for MoE host-offload measurement (23.3 vs 384.0 tok/s) | `LIMITATIONS.md:108-110` | `LIMITATIONS.md:64` | `grep -n "23.3\|384.0" docs/LIMITATIONS.md` |
| docs/PERF.md | line citation for #1669 withdrawn-row story | `roadmap.md:655` | `roadmap.md` (no line number) | `docs/roadmap.md` is 448 lines total, `:655` is out of range; could not locate the exact current line, dropped the number per instructions |
| docs/quantization.md | code citation for the AWQ tie-statistic inflation mechanism | `awq_plan.cpp:266-277` | `awq_plan.cpp:163-174` | read `tools/imp-quantize/awq_plan.cpp`; the max-reduction "tie" logic (`site_statistic`/`spread` lambda) is at 135-179, not 266-277 (which is mid-`search_group_scale` call, unrelated) |
| docs/quantization.md | KV-cache chunked-prefill gather function names | `paged_kv_gather_*_to_fp16` (glob) | `paged_kv_gather_{nvfp4_to_fp16,mxfp4_kv_to_fp16}` (exact names) | `grep -n paged_kv_gather src/compute/kv_gather.h`; the MXFP4 symbol is `paged_kv_gather_mxfp4_kv_to_fp16`, the glob does not match it |
| docs/quantization.md | link text for the models doc | `` `supported-models.md` `` (text) linking to `MODELS.md` | `` `MODELS.md` `` | the file is `docs/MODELS.md`; old link text named a file that does not exist |
| docs/internals/QUANT_PIPELINE.md | link text for the architecture doc | `architecture.md` | `ARCHITECTURE.md` | actual file is `docs/internals/ARCHITECTURE.md` |
| docs/internals/QUANT_PIPELINE.md | TurboQuant retirement doc pointer | `turboquant_retired_2026_05_17.md` (named as a file) | "(PR #251)" | `find . -iname "*turboquant*"` found no such file anywhere in the repo; `docs/archive/README.md` records the retirement as the `turboquant_fp8_gap` plan memo referencing PR #251, not a standalone file at that path |

### 3. Removed unverifiable claims

| file | claim | why unverifiable |
|---|---|---|
| - | - | none: every fact cut from the source files was either kept in the rewritten file, moved verbatim to an archive/internals doc, or corrected against source (table 2). Nothing was deleted outright. |

### Verification performed

- `tests/perf_baseline.json` vs PERF.md pinned-gate table: byte-identical (tg128 299.61, pp128
  5031.75, pp512 12707.44, pp4096 15776.32, own_peak_mb 20642, commit `6fe00f81`, cuda 13.4).
- Config keys/defaults checked against `src/core/config/*.h` and `src/runtime/config.cpp`:
  `moe.expert_cache_budget_pct` (0, was wrong in doc), `moe.force_host_experts`,
  `gemm.nvfp4_lm_head` (auto), `gemm.nvfp4_smallm` (true), `gemm.nvfp4_gdn_proj_prefill` ("in"),
  `gemm.fp8_ssm_proj` (true), `kv_cache.dtype` (auto, enum verified), `vram.library_reserve_mb`,
  `server.prefix_cache`, `speculative.min_match` (6, line citation was stale), `speculative.draft_ctx_cap`.
- CLI flags checked against `tools/imp-quantize/main.cpp` and `tools/imp-cli/args.cpp`: `--calib`,
  `--calib-weight abs|sq`, `--calib-groups`, `--format modelopt|vllm`, `--lm-head`,
  `--keep-attn-gate`, `--keep-gdn-proj`, `--dry-run`, `--calibrate`, `--kv-fp8/int8/int4/nvfp4/mxfp4`.
- File paths spot-checked to exist: all `src/exec/pre_dequant_phase*.cu`, `src/quant/*.cu`,
  `gemm_kernel_*.cu`, `executor_gemm_smallm.cu`, `executor_forward_moe_nvfp4_host.cu`,
  `awq_plan.cpp`, `kv_gather.{h,cu}`, `calibration_stats.h`, `model_arch.h`/`model.cpp`
  (`kv_fp8_hint_default_safe` / `kv_fp8_no_hint_default_safe`).
- `GemmKernelRegistryTest.RegistryHoldsExactlyTheProducedKeys` pins the registry at 10 keys:
  matches QUANT_PIPELINE.md's claim (`tests/test_gemm_kernel_registry.cu:45`).
- `docs/audit/AUDIT_ARCH_2026_07_29.md` answer 5 / 3.50% citation in PERF.md: confirmed present.
- Ran `scripts/docs_lint.py` (in `imp:toolchain` container) after every edit pass: fixed all
  paragraph->2-sentence violations in the four files (PERF.md 8, BENCHMARKS.md 13,
  quantization.md 8, QUANT_PIPELINE.md 3) via semicolon-merging or converting to
  tables/bullets; zero forbidden tokens (tcgen05/TMEM/wgmma/sm_100/sm_90/Hopper); zero
  em/en dashes in any of the four rewritten files (archive files kept verbatim, not scrubbed,
  per the "move verbatim, not rewritten" instruction).
- `fc.sh` fact-loss check against base `9cbb8004`: 0 missing tokens for BENCHMARKS.md; the
  remaining "missing" tokens for PERF.md (2), quantization.md (3) and QUANT_PIPELINE.md (2) are
  exactly the 8 corrections in table 2 above (old, now-wrong strings no longer appear, by design).

### Return

| file | lines before -> after | fc.sh missing after logging | corrections |
|---|---|---|---|
| docs/PERF.md | 246 -> 250 (budget <=250) | 2 (both logged corrections) | 3 |
| docs/BENCHMARKS.md | 882 -> 297 (budget <=300) | 0 | 0 (all facts preserved via archive) |
| docs/quantization.md | 552 -> 200 (budget <=200) | 3 (all logged corrections) | 5 |
| docs/internals/QUANT_PIPELINE.md | 67 -> 147 (budget <=250) | 2 (both logged corrections) | 2 |
| docs/archive/benchmarks_pre_v0.44.md | new, 890 lines | n/a (record, no lint) | 0 |
| docs/archive/quantization_awq_findings.md | new, 299 lines | n/a (record, no lint) | 1 (fixed a truncation bug in my own first draft: the Roles-full-precision section had been cut one line short mid-sentence, restored) |

Nothing for the orchestrator to fix in other agents' files from my side: I did not touch or add
links into `docs/usage.md`/`docs/API.md`/the not-yet-created `docs/CONFIG.md`/`docs/API_FEATURES.md`.
My files link to `docs/MODELS.md`, `docs/PERF.md`, `docs/BENCHMARKS.md`, `docs/roadmap.md`,
`docs/LIMITATIONS.md`, `docs/QUICKSTART.md`, `../README.md`, `internals/ARCHITECTURE.md`,
`internals/BENCHMARKING.md`, `archive/performance_2026_05.md` (pre-existing), and the two new
archive files - all verified to exist. `docs/internals/QUANT_PIPELINE.md` gained two new
top-level sections (Loader enforcement, Producing NVFP4 outside imp-quantize) and a renamed
"NVFP4 walkthrough" section (was "NVFP4 internal pipeline" inside quantization.md); if another
agent's file links to the old `quantization.md#nvfp4-internal-pipeline` anchor, it needs
updating to `internals/QUANT_PIPELINE.md#nvfp4-walkthrough`. `docs/quantization.md`'s "Formats
and where they show up" table combined the five GGUF Q*-quant rows into one row - if another
file counts on individual anchors/rows there, none existed (plain table, no per-row anchors).

## Agent E log - docs/internals/{MEMORY,ARCHITECTURE,KERNELS,PROFILING,SM120,ATTENTION_DISPATCH,BENCHMARKING,CPP23}.md

fc.sh baseline: 9cbb8004. Final missing-token counts (full DST set): MEMORY.md 95/850, ARCHITECTURE.md 35/187, KERNELS.md 2/110, PROFILING.md 0/77, SM120.md 3/106, ATTENTION_DISPATCH.md 3/136, BENCHMARKING.md 0/116, CPP23.md 1/96. Every SM120/ATTENTION_DISPATCH/CPP23/KERNELS remaining "missing" token is an intentional correction (table 2) except KERNELS.md's FA2 template-parameter list (table 1). MEMORY.md/ARCHITECTURE.md remainders are grouped below by source subsection.

### 1. Deleted or moved sections

| file | deleted or moved section | fact preserved where |
|---|---|---|
| MEMORY.md | `0. The defect` | moved verbatim to `docs/archive/memory_census_2026.md` |
| MEMORY.md | `A1. Current-state inventory` (A1.1-A1.7) | moved verbatim to `docs/archive/memory_census_2026.md` |
| MEMORY.md | `A7. Migration plan` | moved verbatim to `docs/archive/memory_census_2026.md` |
| MEMORY.md | `B0. Implementation log` (Landed, I1 allowlist baseline, Divergences D1-D16, B1-B8) | moved verbatim to `docs/archive/memory_census_2026.md` |
| MEMORY.md | A3.1 VMM WSL2 spike raw probe table (`0.4`, `0x2000000000`, `13030`, `2016`, `VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED=1`, `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`, `commit(new_total)`) | condensed to one summary table row ("24 GiB reserve costs 0 MiB physical; commit granularity 2 MiB..."); full probe table is in the pre-campaign doc history (git blame on this section) - not otherwise archived, logged here as the record |
| MEMORY.md | A3.2 growth-guard implementation detail (`do_acquire()`, `do_commit()`, `do_commit_range()`, the `--wrap` interposer wrapping `cudaMalloc*`) | condensed to "commit()/commit_range() are guarded the same way as acquisition"; mechanism kept, wrapper-function names and interposer detail dropped for budget |
| MEMORY.md | A4.3 example `PlanFailure::report()` output box (`35952`, `3345`, `25600`, `6144`, `3072`, `1680`, `mode-2`) | condensed to the properties table ("Fails at load time with an itemised report... plus the three largest levers"); the illustrative numbers were a worked example, not a measurement, so dropping them loses no fact - PlanFailure's `report()`/`lines()`/`levers` shape is still stated in the code block |
| MEMORY.md | A5.1 KV refcount implementation detail (`evict_cached_block()`, `unpin_prefix()`, `block_hash_to_id_`, `cache_->inc_ref`, `allocate_blocks`, `total_blocks()`) | condensed to the ownership-rule paragraph and the pressure-valve paragraph; the rule and the #1879 fix are kept, individual method/field names dropped |
| MEMORY.md | A5.2 graph pool bucketing detail (`std::tuple<n_tokens, ctx_capacity, rec_slot>`, `spec_capture_bucket_`, `spec_capture_ctx_tier_`, `speculative.capture_ctx_cap`, `free_spec_buffers_`, `max_batch + 1`, `used=0.0 / reserved=0.0 / high_since_serving=0.0 MiB`, `cudaDeviceGetGraphMemAttribute`, `cuda_graph.cu:339,1291`) | condensed to the pool table + one measured-zero sentence; bucketing formula and dead-trim-call citation dropped for budget |
| MEMORY.md | A5.3 pre-migration static identifiers (`gemm.cu:s_workspace`, `gemm.cu:s_bench_scratch`, `gemm_cutlass_sm120.cu:s_cutlass_workspace`, `gemm_nvfp4_cutlass_sm120_impl`, `executor_perplexity.cu`) | mechanism and the 488 MiB / 152 320 B figures kept; the specific pre-move static-variable names and one co-caller file dropped (`CutlassWorkspaceContract` and `executor_forward.cu`/`gemm_nvfp4_cutlass_sm120_workspace(M, N, K)` were restored) |
| MEMORY.md | A5.4 (`VisionPipeline::init()`, `engine_qwen3vl.cpp`, `vision_pipeline.cpp:97`) | condensed to one paragraph naming the design decision and the +1610 MiB figure; the raw-`cudaMalloc`-fallback hot-path-hole callout and its file:line dropped |
| MEMORY.md | A6 (`HostPinnedAllocator`, `JsonConstrainer::init`, `mtp_k=0`, `AllocPhase::Serving` as a joined token, `RequestStatus::CANCELLED`) | `FakeBackend` capability list and V1-V9 table kept; specific I2-measurement call-site names dropped (belong to the archived B7 log) |
| MEMORY.md | Invariant-compliance sub-measurements (`499`, `275`, `582`, `638`, `696`, `7460`, `864`, `984`, `309`, `2239`, `27.7`, `98.3`, `82.5`, `102.0`, `292.7`, `293.4`, `32607`, `4182`, `report_library_reserve()`, `engine_weight_upload.cpp:278`, `engine_kv_cache_init.cpp:370`) | verdict + target kept per invariant (I1-I7 table); the B-numbered measurement trail behind each "partial" is in the archived B0 log (`docs/archive/memory_census_2026.md`, "Invariants now under test" / per-invariant B-refs) |
| MEMORY.md | Provenance branch name `fix/1104-json-number-grammar` | the measurement conditions sentence is kept without the exact branch name; branch name is in the archived A1 Provenance section verbatim |
| MEMORY.md | `src/memory/vram_query.{h,cpp}` as a joined brace token | file is cited twice in the rewritten A4 section as `src/memory/vram_query.h` (no `.cpp` in the brace) - same file, narrower citation |
| KERNELS.md | "REFUTED (2026-06-17)" occupancy follow-up narrative | moved verbatim to `docs/archive/kernels_refuted_2026.md` |
| KERNELS.md | FA2 kernel template parameter list `<Bq, HD, FP16QK, F16ACC, BKV, TWOSLOT, PVF16, …>` and "dispatcher bands the tile config by grid-fill" | dropped; section 2's tiling spec (Bq/Bkv/HD/`__launch_bounds__`) and the kernel table's file/role already state the shape this summarized |
| PROFILING.md | Mission, Phase 0-5, Deliverables, Constraints (the whole phase-plan narrative) | moved verbatim to `docs/plans/2026-06-profiling-campaign.md` (one-line header comment added, no lint there) |
| ARCHITECTURE.md | inline attention-dispatcher decision snippet and vars (`attention-dispatch.md` display text, `attention.fa2_hd256_bkv=32`, `hd ∉ {128, 256}`, `0.53`) | removed: duplicate of owner `ATTENTION_DISPATCH.md`, which already states (its own text) that this exact inline snippet was wrong for six weeks - see corrected-facts table |
| ARCHITECTURE.md | "Re-rendering the diagram" docker+dot recipe (`architecture.dot`, `architecture.svg`, `d/architecture.dot`, `d/architecture.png`, `d/architecture.svg`) | removed: the file now carries an inline Mermaid diagram per the target spec, no build step needed. `docs/architecture.dot/.svg/.png` become unreferenced assets - **flag for orchestrator**: decide whether to delete them or move the recipe to `CONTRIBUTING.md` (not in this agent's file set; grep confirms no other `.md` references them) |
| ARCHITECTURE.md | Phase-2 engine-init granular private-method table (`RuntimeConfig::load()`, `init_resolve_*`, `init_resolve_kv_dtype_policy_`, `init_resolve_ssm_dtype_`, `init_resolve_fp8_prefill_`, `init_resolve_quant_flags_`, `init_compute_max_seq_len_`, `engine_arena_open`, `graph_slot_pool_open_for`, `exec_t2_demand()`, `warmup()`, `prewarm_spec_scratch_`, `upload_expert_weights`, `570` LOC) | removed by design: target spec asks for "Mermaid + component table", not the old 9-row init-step table. The Mermaid pipeline diagram + the component table's "Engine init orchestrator" row cover the same ground at component grain |
| ARCHITECTURE.md | Vision subsystem deep mechanics (`<\|image_pad\|>`, `deepstack_inject.cu`, `smart_resize`, `patchify`, `mmproj.gguf`, `model/mrope_positions.cpp`, `compute/rope.cu`, `model/image_placeholders.cpp`) | condensed to one component-table row (`src/vision/`, SigLIP/gemma4v vs Qwen3-VL one-liner); no other file in this campaign owns vision internals, so this is a real reduction, not a dedup - flagged here rather than silently dropped |
| ARCHITECTURE.md | "Known limitations" section content (S-matrix workspace cap, `process_diag` process-wide snapshot) | kept in ARCHITECTURE.md (2-row table) **and** flagged as duplicate of owner `docs/LIMITATIONS.md` per `TARGET_STRUCTURE.md`'s fact-owner table - `docs/LIMITATIONS.md` does not yet contain either fact (checked); kept rather than deleted so the fact survives regardless of ordering between parallel agents. Orchestrator should dedupe once `LIMITATIONS.md` is rewritten |

### 2. Corrected facts

| file | claim | was | now | source |
|---|---|---|---|---|
| MEMORY.md | `BlockPool` template shape | `template <size_t Stride> class BlockPool` | stride is a runtime constructor argument, not a template parameter | `src/memory/block_pool.h:70` (`class BlockPool {`); matches the tree's own D2 divergence note |
| MEMORY.md | Cross-tier ownership type | `Owned<T, Tier>` compile-time-tagged handle exists | not implemented; `StableSpan<T>` + a runtime `RegionTag` + `generation()` counter on the arena is what exists, cross-tier smuggling caught at runtime not compile time | `grep -rln "class Owned" src/memory/` = 0 hits; `src/memory/arena.h:76-81` (`RegionTag tag()`, `generation()`); matches the tree's own D1 divergence note |
| MEMORY.md | Debug-build phase-guard behaviour | `IMP_ASSERT_FAIL` with tag and backtrace | `IMP_LOG_ERROR` + `std::abort()` (no such macro as `IMP_ASSERT_FAIL` in the tree) | `src/memory/backend.cpp:113-116` |
| MEMORY.md | Release-build phase-guard counter name | `steady_state_allocations_total{tag}` | `steady_state_allocations(tag)` | `src/memory/backend.h:70-73`; `docs/audit/AUDIT_arch_2026.md:2916` already flags the old name as "a name that exists in no source file" |
| MEMORY.md | `kv_cache.growable` default | "off by default" | on by default (`growable_initial_pct` default 25, commits a fraction at startup and grows at admission) | `src/core/config/kv_cache.h:35-44` ("Default on") |
| MEMORY.md | `--vram-budget` / `vram_query.h` status | "the rewriting view is retired with the heuristics that needed it" | still live: `vram_budget_install()` still rewrites what `cudaMemGetInfo` returns; `plan_memory()`'s `budget_bytes` is fed from the same distributable-VRAM probe rather than replacing it | `src/memory/vram_query.h:9-23`; `src/runtime/plan_shadow.cpp:20` |
| ARCHITECTURE.md / SM120.md | TMA warp-specialized grouped GEMM on `sm_120a` | "unresolved and deliberately not claimed either way" (`OPEN_QUESTIONS.md` Q1 cited as open) | present on `sm_120a` native SASS, verified by `cuobjdump` on the shipped grouped-NVFP4 cubin (#1543); only the `compute_120f` PTX fallback loses it | `docs/audit/docs-rewrite/OPEN_QUESTIONS.md` CLOSED Q1 ("ANSWERED by #1543"); `CHANGELOG.md` #1543 entry |
| ATTENTION_DISPATCH.md | Learned-sinks gate line | `attention_dispatch.cu:34` | `attention_dispatch.cu:65` | `grep -n "if (has_sinks) {" src/compute/attention_dispatch.cu` = 65 |
| ATTENTION_DISPATCH.md | FP16 decode launcher line | `attention_paged.cu:1115` | `attention_paged.cu:1121` | `grep -n "void paged_attention_decode(" src/compute/attention_paged.cu` = 1121 |
| ATTENTION_DISPATCH.md | MXFP4-KV decode launcher line | `attention_paged_nvfp4.cu:424` | `attention_paged_nvfp4.cu:492` | `grep -n "void paged_attention_decode_mxfp4_kv(" src/compute/attention_paged_nvfp4.cu` = 492 |
| SM120.md | Two-image gencode citation | `CMakeLists.txt:46-47` | line range dropped (now 58-71 depending on the `IMP_DEVICE_LTO` branch); cited as the file only, plus the two build options that were added since (`IMP_DISABLE_120F_FALLBACK`, `IMP_DEVICE_LTO`) | `CMakeLists.txt:47-72` |
| SM120.md | f32-accumulate cross-reference | `KERNELS.md:20` | `KERNELS.md:31` | line moved when `KERNELS.md` was rewritten in this same pass |
| SM120.md / KERNELS.md | Self-referential link display text | `sm120_optimal_kernel.md` (KERNELS.md link) / `sm120.md` (SM120.md's own link into KERNELS.md) | `KERNELS.md` / `SM120.md` - display text now matches the real filename | same-file check |
| CPP23.md | Reproduction command for "no more `std::string& err` out-params" | `grep -rIn "std::string& err" src tools include` returns nothing | that exact command returns 3 false-positive hits (`error()`, `error_reason()` accessors contain the substring "err"); `grep -rIn "std::string& err[,)]" src tools include` returns nothing and is the accurate reproduction | ran both commands against the tree |

### 3. Removed unverifiable claims

None. Every fact removed from these 8 files in this pass was either condensed-for-budget (table 1, with its replacement or archive location stated) or checked and corrected (table 2); nothing was dropped for being unverifiable.

