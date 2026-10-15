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

## 2026-10-15 Phase 3: baseline (validator)

Baseline tree `5a30cb5f` (origin/main), detached worktree `imp-baseline`, image `imp:test-imp-baseline-016f0c60` (`imp.tree=d1913866`, build args `IMP_BUILD_TESTS=ON`). No source edit. Scratch artifacts live in `/tmp/claude-1000/cleanup/` (tmpfs: copy before a reboot).

### Tests

Test list: [`BASELINE_TESTLIST_2026_10_15.txt`](BASELINE_TESTLIST_2026_10_15.txt), `<binary> <Suite.Test>` from `--gtest_list_tests`, 3696 entries (10 `DISABLED_`), plus 88 `SKIP` lines (runtime skips of `make test-gpu`).

| binary | listed | ran | passed | skipped | failed | disabled |
|---|---:|---:|---:|---:|---:|---:|
| test-core | 2004 | 2004 | 2004 | 0 | 0 | 0 |
| test-text | 272 | 272 | 272 | 0 | 0 | 0 |
| test-compute | 227 | 226 | 225 | 1 | 0 | 1 |
| test-attention | 309 | 306 | 306 | 0 | 0 | 3 |
| test-quant | 275 | 273 | 269 | 4 | 0 | 2 |
| test-kv | 113 | 113 | 113 | 0 | 0 | 0 |
| test-moe-gdn | 212 | 212 | 210 | 2 | 0 | 0 |
| test-e2e | 284 | 280 | 199 | 81 | 0 | 4 |
| total | 3696 | 3686 | 3598 | 88 | 0 | 10 |

| suite | result |
|---|---|
| `make test-gpu` (wall 15m19.8s, exit 0) | table above; `*DetEvalE2ETest*` 6/6 passed; `*Qwen38TokenizerParity*` 4/4 passed |
| `make dev-test` (`ctest -L unit`, wall 1m59.6s incl. `make dev`) | 13/13 passed (`unit_core`, `unit_text`, `unit_e2e_subset`, 10 guards) |

### Reference dumps

Harness `dump_refs.sh` (sha256 `1b85922761472004...`): fresh `imp-server` container per run, `--set runtime.deterministic=true`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `/v1/chat/completions` with `temperature 0`, `seed 1234`, `max_tokens 64`, `logprobs true`, `top_logprobs 5`, 3 prompts per model, 2 runs per model. Raw JSON: `/tmp/claude-1000/cleanup/baseline_dumps/base_<model>_p<i>_run<r>.json`; diff key `.choices` (jq `-cS`) in `*.choices.json` (raw files differ in `id`/`created` only). Hash lists: `/tmp/claude-1000/cleanup/baseline_dumps_choices.sha256`, `baseline_dumps_raw.sha256`.

| key | model | prompt tokens p0/p1/p2 | completion tokens | run1 vs run2 |
|---|---|---|---|---|
| dense_q8 | Qwen3-4B-Instruct-2507-Q8_0.gguf | 19/22/18 | 64/64/64 | byte-identical 3/3 |
| moe_nvfp4 | Qwen3-30B-A3B-NVFP4-Modelopt | 21/24/20 | 64/64/64 | byte-identical 3/3 |
| dense_nvfp4 | Qwen3-8B-NVFP4-cortecs | 21/24/20 | 64/64/64 | byte-identical 3/3 |
| gguf_q4km | gemma-3-12b-it-Q4_K_M.gguf | 19/23/19 | 64/64/64 | byte-identical 3/3 |
| vision | gemma-3-4b-vl/gemma-3-4b-it-Q4_K_M.gguf + mmproj-F16.gguf, images `tests/fixtures/vision_test_64.png` (p0), `vision_test_green_bar.png` (p1, p2) | 274/275/274 | 64/57/64 | byte-identical 3/3 |

No known-nondeterministic path under `runtime.deterministic=true`: MoE matched too. Runs without the switch were not taken.

| key | p | sha256 of `base_<key>_p<i>_run1.choices.json` (run2 identical) |
|---|---|---|
| dense_q8 | p0 | `e69f8bfa22ea7e17cdc6f54727a548dc1c50fba7de32e41bdb029acfdc3ae1b2` |
| dense_q8 | p1 | `6721998ea5e2b09936d9d73107576173bb11693ed2fc0212d606aa007a597fd0` |
| dense_q8 | p2 | `f8874ef6eba3247c4c6e23932c7416273bf1159c0099d6ceefb3daf21c86fb6d` |
| moe_nvfp4 | p0 | `0f64cf71e8d5348a60b692b1d1130a5446adabc2d51d5edf4daa845d54b5f877` |
| moe_nvfp4 | p1 | `a507a651afbda492ed3cea256943864225ac5c4f4d7a20258916bc12ca64adfb` |
| moe_nvfp4 | p2 | `8a7c47302ddf91b21f67107f730e12de6998f6451a4b54fcb40a5f1bbce713ab` |
| dense_nvfp4 | p0 | `e21960135cc12f127eb27f99e7faa4f914333edee44379ebd38a9bfc3fc2322b` |
| dense_nvfp4 | p1 | `181f93b11f7f7c61d586fb9bd9f6e9a370f711a0081e880d8ed990ab3989868b` |
| dense_nvfp4 | p2 | `485757cfd1d83ad265e9ea28afe10415c9d255f0911f354bd21f70d88423e09f` |
| gguf_q4km | p0 | `6b528a29f22b1f653f4521a15a6b74e87d54795f5e0a4bd12ca3d0b6309bab8c` |
| gguf_q4km | p1 | `97b8d2d3803d2762f04241acd3a297ba8bb6eaf0bcd7b8240e585bf125ab289a` |
| gguf_q4km | p2 | `58a63f8dc407b616a2b881ac7b4ff98cf1d62047c6bc9d55088332e14f7f3859` |
| vision | p0 | `f29f646cd85d4937a57d4623a0190100d9b116ecb3110dbf30910497b5981763` |
| vision | p1 | `d05e9408f0107ba857a7f715b55c42b9fbfe7caee5caa0fe716ce2d50e12e890` |
| vision | p2 | `f6d65b5e779ea901c5ca32f50df472a3873a75e39be85a1cdf9b60690fef9bdb` |

### Kernel resources

Method of [`PTXAS_BASELINE_2026_10_15.md`](PTXAS_BASELINE_2026_10_15.md) (`cuobjdump -res-usage` + `-sass` counts in `imp:toolchain`) on `/usr/local/bin/imp-server` of the baseline image: 1024 instances in 132 ELF images. Table: `/tmp/claude-1000/cleanup/ptxas/baseline_res.tsv` (sha256 `7fe90b2d92a4b295...`, script `res_table.py`).

| diff vs PTXAS_BASELINE (v0.53.0 release binary), unique rows | count | cause |
|---|---:|---|
| identical REG/STACK/SHARED/LOCAL/ACQ/PRE/STL/LDL | 1010 | |
| `mmq_imma_kernel` renamed by a 5th template parameter (`int` 128), resources unchanged | 4 | #2730 |
| new `mmq_imma_kernel<32, *, 0, 0, 32>` (REG 40, SHARED 12032) | 2 | #2730 |
| new `control_vector_add_fp16_kernel` (REG 20), `control_vector_add_fp32_kernel` (REG 16) | 2 | #2728 |
| new `smallM_smoke_single_mma_kernel` (REG 16) | 1 | `SMALLM_TEST_HOOKS`, compiled only with `IMP_BUILD_TESTS=ON`; release image is `OFF` |

No existing kernel changed REG, STACK or SASS counts. Diff baseline for later batches: this image's table, not the release table.

### Binaries and build

| binary (`/usr/local/bin`, static, no shipped `.so`) | bytes |
|---|---:|
| imp-server | 42524880 |
| imp-cli | 37248664 |
| imp-quantize | 37691256 |
| test-core | 58799504 |
| test-e2e | 41980600 |
| test-quant | 39635048 |
| test-moe-gdn | 37708592 |
| test-kv | 35821280 |
| test-attention | 23113160 |
| test-compute | 22450288 |
| test-text | 4452416 |
| image `imp:test-imp-baseline-016f0c60` | 5614294099 |

| build fact | value |
|---|---|
| `make build` wall (warm ccache, not cold) | 29.4 s; compile step 18.2 s; ccache 823/826 hits |
| per-TU times from `make build` | not available: `.ninja_log` stays in the builder stage, and a 99.6 % ccache-hit build has no compile times |
| per-TU proxy | `build-dev/.ninja_log` of the `make dev` inside `make dev-test` (fresh `build-dev`, `-j16`, Release): 864 steps, 723 < 1 s (ccache hits), 64 >= 5 s; top 20 below are ccache misses only and ran under `-j16` contention |

Top 20 TUs (s): `imp-server/handlers_chat_core.cpp` 30.3, `tests/test_sampling_params_golden.cpp` 29.8, `tests/test_constraint_validation.cpp` 29.5, `imp-server/handlers.cpp` 26.6, `tests/test_quant_integration.cu` 25.5, `imp-server/handlers_chat.cpp` 24.7, `imp-server/handlers_chat_params.cpp` 24.5, `imp-server/handlers_messages.cpp` 24.3, `tests/test_spec_usage.cpp` 24.3, `tests/test_engine_integration.cu` 24.2, `imp-server/handlers_completions_params.cpp` 24.0, `imp-server/handlers_responses.cpp` 23.9, `imp-server/handlers_chat_stream.cpp` 23.7, `imp-server/stream_driver.cpp` 23.6, `tests/test_prefill_ragged.cu` 23.6, `tests/test_moe_executor.cu` 23.3, `imp-server/handlers_misc.cpp` 22.9, `imp-server/handlers_completions.cpp` 22.6, `imp-server/main.cpp` 22.6, `imp-server/constraint_validation.cpp` 22.3.

### Perf: Qwen3-Coder-30B-A3B-Instruct-FP4

Harness `perf.sh` (sha256 `80dd437268d3870a...`): per container 5 `imp-cli --bench --bench-reps 5 --max-tokens 256 --temperature 0` processes, `--bench-pp 512` and `--bench-pp 2048` in separate containers, 2 containers per shape, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. All 20 processes: `attn_prefill=fa2_fp16qk attn_decode=paged_fp8 moe_prefill=cutlass3x graphs=1`, KV `FP8_E4M3`, no `imp.conf`. Values = per-process average tok/s; n = processes. Logs: `/tmp/claude-1000/cleanup/perf/`.

| metric | container 1 median (min..max) | container 2 median (min..max) | pooled median (min..max), n=10 | pooled spread |
|---|---|---|---|---|
| tg256 after pp512 (decode gate) | 371.57 (371.22..372.00) | 371.69 (371.54..372.11) | 371.63 (371.22..372.11) | 0.24 % |
| pp512 | 30480.93 (30331.12..30529.83) | 30326.40 (30125.23..30453.44) | 30444.14 (30125.23..30529.83) | 1.33 % |
| pp2048 | 48971.56 (48738.65..49024.48) | 48817.70 (48741.00..49048.72) | 48919.10 (48738.65..49048.72) | 0.63 % |
| tg256 after pp2048 | 345.44 (345.20..345.58) | 345.26 (345.18..345.51) | 345.41 (345.18..345.58) | 0.12 % |

Clocks (1 Hz, samples at util >= 50 %): SM 2861..2918 MHz, mem 13801 MHz, 307..329 W per container. Gate for later batches (DISPATCH rules): decode 371.63 tok/s +-2 % = 364.20..379.06; prefill inside 30125.23..30529.83 (pp512) and 48738.65..49048.72 (pp2048).

### Roofline

`tools/roofline/roofline --config <copy of config.json with docker.image=imp:test-imp-baseline-016f0c60> measure --models q8-dense,nvfp4-moe --shapes tg256 --restarts 1`: run `5a30cb5f_20261015_081707` (11m7s), appended to `tools/roofline/history/` (run json, `index.jsonl`, raw `*.csv.gz` + `nsys_extract.json`). Full sweep (all 7 models x 5 shapes, 3 restarts) not run: hours of GPU; `gptoss-moe` model directory `gpt-oss-20b` is missing.

| class | cell | time share % | %-roofline |
|---|---|---:|---:|
| gemv_nvfp4 | q8-dense/tg256 | 74.8 | 61.1 |
| attn_decode_paged | q8-dense/tg256 | 7.8 | 5.1 |
| gemv_nvfp4 | nvfp4-moe/tg256 | 54.4 | 39.9 |
| moe_routing | nvfp4-moe/tg256 | 14.6 | 2.4 |
| attn_decode_paged | nvfp4-moe/tg256 | 10.6 | 2.3 |

`roofline regress --baseline dca16b71_20260806_041710` (pinned `BASELINE`, q8-dense/tg256 only): exit 1, one FAIL `kv_write` 0.2 -> 0.2 %-roofline (-33.8 %, 1.3 % time share); `gemv_nvfp4` 62.6 -> 61.1. Not a cleanup finding: the pin predates this tree by two months.

### Not obtained, deviations

| item | state |
|---|---|
| GPU lock | `gpu_lock.sh` keeper (TTL 4 h) expired between 00:24Z and 06:11Z (no job ran in that window); dumps for four models and the perf runs ran with a FREE busy check before every container but without the lock; re-acquired 06:16:55Z for the roofline, released after |
| `imp-bench` | not in the image (`IMP_BUILD_BENCH=OFF`) |
| cold `make build` | not measured (warm ccache) |
