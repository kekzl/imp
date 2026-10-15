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

## 2026-10-15 Phase 4 B1 (builder)

Branch `cleanup/b1-ownership-odr` off `origin/main` 8d4c4148. Compile only, no GPU run. `make build` (image, `IMP_BUILD_TESTS=ON`): 13 unique warning lines on origin/main, 13 on the branch, diff empty. `make dev` (adds `IMP_BUILD_BENCH`, compiles `bench_e2e.cpp`) green; `make dev-test` 13/13.

| row | files | change | lines +/- | compile result |
|---|---|---|---|---|
| H-018 | `memory/kv_cache.h`, `memory/ssm_state.h`, `memory/recurrent_snapshot_store.h`, `model/ngram_table.cpp`, `compute/sampling_topk_topp.cu`, `tools/imp-quantize/awq.cu` | copy ctor + copy assignment `= delete` on 6 of 7 types; `= default` ctor where the deleted copy ctor suppressed the implicit one; `OwnedSampleScratch`: two-arg ctor (was aggregate init), const members (copy assignment implicitly deleted, file at the 600-LOC hard-review line) | +17 / -2 | green, 0 copies exposed in the 6 |
| H-019 | `tools/imp-bench/bench_e2e.cpp` | 14 leaked `new uint16_t[]` -> `std::vector<std::vector<uint16_t>>` declared before the model in `bench_e2e()` | +10 / -6 | green (dev build) |
| H-023 | `exec/executor_gemv_helpers.h` | `static` -> `inline`; comment names the 4 includers | +2 / -2 | green |
| H-024 | `model/jinja.cpp`, `compute/json_schema.cpp`, `compute/gemm.cpp`, `model/weight_upload.cpp`, `model/safetensors_loader.cpp`, `model/model.cpp`, `runtime/engine_spec_batch_verify.cpp`, `imp-server/constraint_validation.cpp`, `imp-server/handlers.cpp`; citations in `docs/CONFIG.md`, `AUDIT.md`, `docs/LIMITATIONS.md` | 41 of 46 `misc-use-internal-linkage` host findings into anonymous namespaces (`Expr`, `Node` stay outside: `jinja.h` forward-declares them) | +38 / -5 | green |
| H-026 | `compute/gbnf_grammar.h`, `compute/sampling.h`, `exec/executor_attention_internal.h`, `exec/pre_dequant_phase3_cutlass.cpp`, `memory/vram_allocator.h`, `memory/weight_cache_file.cpp`, `model/tokenizer_pretok.cpp`, `model/weight_upload.cpp`, `runtime/engine.h`, `vision/image_decode.cpp` | default member initializers on all 14 findings; `FileHeader` writer sets all 8 fields before `f.write` | +35 / -35 | green |

| finding / skip | evidence | proposed next step |
|---|---|---|
| H-018 real copy: `MoeRoutingBuffers` | deleting its copy assignment fails `tests/test_mtp_attn_guard.cpp:60` `ws = MtpDraftWorkspace{}`: the implicit move assignment of `MtpDraftWorkspace` falls back to the `MoeRoutingBuffers` copy (user-declared dtor, no move ops). Benign there (both `pool` null); with an allocated `routing_buf` the target leaks its pool and keeps the source's, which the temporary frees: double free | owner call: move ctor/assignment on `MoeRoutingBuffers` (free, steal, null the source), then the `= delete` copy ops; not applied here |
| H-024 skip: `bench_e2e` | `tools/imp-bench/main.cpp:21` declares and calls it from another TU | a `bench_e2e.h` header (out of scope) |
| H-024 skip: `PrefixCacheHeader` | `memory/kv_cache_manager.cpp` code LOC 1425 = its filesize ceiling; the wrap adds 2 | with the next shrink of that file |
| H-024 kept external | `gemm_reset_static_cuda_state`, `gemm_grouped_reset_static_cuda_state`, `qwen3vl_encoder_reset_static_cuda_state` | SETTLED R-11 |

## 2026-10-15 Phase 4 B1 (orchestrator decision)

| row | decision | why |
|---|---|---|
| H-018 `MoeRoutingBuffers` | GO: move ops (free own pool, take the source's, null the source), copy ops deleted | the builder's finding: `ws = MtpDraftWorkspace{}` fell back to the implicit copy; with a live pool that leaks the target pool and double-frees the source's. Current callers move empty pools only, so outputs are unchanged; `make build` green |

## 2026-10-15 Phase 4 B1 (validator)

Branch head `e4b4134e`, image `imp:b1-final`, against the Phase 3 baseline `5a30cb5f`. Verdict: all 7 gates PASS.

| gate | baseline | B1 | result |
|---|---|---|---|
| build warnings | 13 unique lines | 13, diff empty | PASS |
| `make test-gpu` | 0 FAILED; per-binary PASSED counts 2004, 306, 272, 269, 225, 210, 199, 113, 6, 4 | identical counts, 0 FAILED, `[ RUN ]` list diff 0 lines | PASS |
| reference dumps (`runtime.deterministic=true`, 5 models x 3 prompts x 2 runs) | `*.choices.json` | 30/30 byte-identical | PASS |
| sanitizer | BLOCKED on WSL2 (Phase 1) | not run | n/a |
| ptxas (`cuobjdump -res-usage` + SASS counts) | 1024 rows imp-server / imp-cli, 1026 imp-quantize | identical rows in all three | PASS |
| decode tg256 pooled median, Qwen3-Coder-30B-A3B FP4 | 371.63 tok/s (gate 364.20..379.06) | 372.26 (+0.17 %) | PASS |
| decode tg256 after pp2048 | 345.41 | 344.82 (-0.17 %) | PASS |
| prefill pp512 | 30444 (30125..30530) | 30387 | PASS |
| prefill pp2048 | 48919 (48739..49049) | 48929 | PASS |
| `make preflight` (255 static gates incl. file size, clang-tidy on 17 files) | - | 0 failures | PASS |

| fact | value |
|---|---|
| imp-server binary | 42524880 -> 42440960 B (-83920, internal linkage) |
| GPU lock | held for the whole run (08:35-08:53 local) |

## 2026-10-15 Phase 4 B2 (builder)

Branch `cleanup/b2-stream-order` off `origin/main` 7462b738. Compile only, no GPU run. `make build DOCKER_IMG=imp:b2-branch`: 13 unique warning lines, main 13, diff empty. `make dev` green; `make dev-test` 13/13.

| row | files | change | lines +/- | compile result |
|---|---|---|---|---|
| C-020 | `compute/sampling_penalties.cu`, `tests/test_sampling_advanced.cu` | fallback logit read: legacy `cudaMemcpy` -> `cudaMemcpyAsync(stream)` + `cudaStreamSynchronize(stream)`; new GPU test `SamplingAdvancedTest.LogitBiasFallbackReadsOnTheCallerStream` (host callback holds a non-blocking stream 20 ms before the logits land) | +33 / -3 | green |
| C-021 | `exec/expert_cache.cu`, `exec/expert_cache.h`, `tools/sync_device_write_allowlist.txt` | `invalidate_host_state(cudaStream_t)`; `cudaMemset(d_lookup_)` -> `cudaMemsetAsync(..., stream)`; allowlist key count 2 -> 1 (the runtime site was classed `init`) | +4 / -5 | green |
| C-022 | `compute/gemm.cpp` | FP8 fallback scale reads: 2x legacy `cudaMemcpy` -> static `read_gemm_scales` (2x `cudaMemcpyAsync(stream)` + `cudaStreamSynchronize(stream)`); helper keeps `gemm_cublaslt` CCN at 25 (inline version hit 26, ratchet #2210) | +12 / -5 | green |

| finding / skip | evidence | proposed next step |
|---|---|---|
| C-020 reach | server default `--max-logit-bias` 1024 (`tools/imp-server/args.h:56`) < 4096 arena slots (`engine_workspace_warmup.cpp:261`): fallback only after an arena decline or `--max-logit-bias` > 4096 | GPU: new test + decode A/B with logit_bias (kernel path, expected bit-exact) |
| C-021 reach | `d_lookup_` allocated only with `moe.expert_cache_debug_parity` (default false, `expert_cache.cu:127`); sole reader `check_parity()` | production unchanged; GPU: expert-cache tests with debug parity |
| C-025 needs orchestrator go | fixes: throw (behaviour change: request fails instead of the slow path) or `cudaMallocAsync` / arena (moves allocation onto `stream` in the MoE layer path, decode-graph region); `MoeRoutingBuffers::allocate` failure is silent (`moe_routing.cu:504-507`, no log) | owner call; minimum: a WARN on pool allocation failure |
| C-026 refuted (production) | `decode_tier` is `tier`, FP8 or NVFP4 (`pre_dequant_phase4_tensor_registry.cpp:78-88`); executor `default:` (`executor_gemm_dispatch.cpp:547-548`) needs decode FP32, so `primary_tier` FP32 and `gemv_dispatch` throws in its FP32 case; FP8 case reached only by `tests/test_weight_dispatch.cu:306` | none; a `host_scale` field would grow the `WeightHandle` union for a test-only path |
| C-027 skip: KV growth, smallM dummy, constrain categories, `executor_workspace_buffers.cpp:1710` | each site has an explicit `cudaStreamSynchronize(0)` / `nullptr` after the legacy write (#1652, #2275): correct; stream-ordered version needs a stream threaded into `KVCache` growth (KV path, check-degeneration scope) | needs orchestrator go if wanted; not L |
| C-027 refuted: `executor_workspace_buffers.cpp:1200` | `cudaMemset(qscratch_.cutlass_act_sf)` without a sync, but allowlisted `init` ("before the warmup device syncs", `tools/sync_device_write_allowlist.txt:26`); `cudaDeviceSynchronize` at `engine_workspace_warmup.cpp:563`, `:572` | stale comment "Sync because ..." with no sync in the code; comment fix only |

## 2026-10-15 Phase 4 B2 (validator)

Code head `0e0d354b`, image `imp:b2-final`, against the Phase 3 baseline `5a30cb5f`. Verdict: gates PASS; the C-020 test fails mutation validation.

| gate | baseline | B2 | result |
|---|---|---|---|
| build warnings | 13 unique lines | 13, diff empty | PASS |
| `make test-gpu` | 0 FAILED | 0 FAILED; `[ RUN ]` list diff = 1 added line (`LogitBiasFallbackReadsOnTheCallerStream`) | PASS |
| reference dumps (5 models x 3 prompts x 2 runs) | `*.choices.json` | 30/30 byte-identical | PASS |
| ptxas rows, imp-server / imp-cli / imp-quantize | 1024 / 1024 / 1026 | identical | PASS |
| decode tg256 pooled median | 371.63 tok/s (gate 364.20..379.06) | 371.76 (+0.03 %) | PASS |
| decode tg256 after pp2048 | 345.41 | 344.85 (-0.16 %) | PASS |
| prefill pp512 / pp2048 | 30444 (30125..30530) / 48919 (48739..49049) | 30437 / 48911 | PASS |

| C-020 mutation (fallback read back to legacy `cudaMemcpy`) | runs | result |
|---|---|---|
| writer = H2D copy on the held stream (builder test) | 1 | mutant PASSES (26 ms) |
| writer = kernel on the held stream | 3 fixed + 3 mutant | all PASS, 22 ms each |
| conclusion | - | legacy `cudaMemcpy` D2H waits for the held non-blocking stream on this WSL2 host; race not observable here; code change kept (CUDA stream semantics), test comment says it covers the fallback path only |

## 2026-10-15 Phase 4 B2 (orchestrator decision)

| row | decision | why |
|---|---|---|
| C-020, C-021, C-022 | ship | gates above |
| C-025 | DEFER to the deferred list | reachable only after a failed pool allocation; both fixes change behaviour or move an allocation into the decode-graph region |
| C-026, C-027 | closed as logged by the builder | refuted / explicit syncs in place |

## 2026-10-15 Phase 4 B3 (builder)

Branch `cleanup/b3-launch-checks` off `origin/main` b87b0364, head 32213359. Compile only, no GPU run. `make build DOCKER_IMG=imp:b3-branch`: 13 unique warning lines (main: 13), all in untouched `tests/*.cpp` and `stl_vector.h`, 0 in a changed file. `make dev` green; `make dev-test` 13/13. Every commit passed `make preflight` (final: 256 static gates ok). Branch diff: 23 files, +375 / -258.

| row | files | change | lines +/- | result |
|---|---|---|---|---|
| C-014 | `core/pdl_launch.cuh` | `IMP_CUDA_CHECK_LAUNCH()` after the `<<<>>>` fallback (cudaPeekAtLastError, log-only, capture-safe) | +1 / -0 | applied, covers all 106 `pdl::launch` sites |
| C-015 | `exec/executor_forward_moe_kernels.cuh`, `tools/check_launch_guards.py`, `scripts/ci_static_gates.sh` | gate collects `*.cu` + `*.cuh` (475 -> 500 launches), `--selftest` 7 cases; `IMP_CUDA_CHECK_LAUNCH` on `sanitize_fp16_kernel` | +53 / -5 | applied |
| C-015 apa | `compute/apa/apa.cuh`, `tools/check_launch_guards.py` | vendored kekzl/apa v0.6.1: local patch reverted (7e445f93); file-level ALLOWLIST entry, its 2 `reset_stats_kernel` launches are checked by the `cudaPeekAtLastError` 5 lines below (gate window 4) | +4 / -5 | applied as allowlist; upstream fix needs orchestrator go |
| H-016 | `core/cublas_check.h` (new), `compute/gemm.cpp`, `gemm_batched.cpp`, `gemm_grouped.cpp`, `gemm_gemv_dtype.cu`, `attention_cublas.cu`, `attention_mxfp4_prefill.cu`, `vision/vision_encoder.cu`, `qwen3vl_encoder.cpp`, `internvl_encoder.cpp`; `AUDIT.md` B14 citation | `IMP_CUBLAS_CHECK_LOG` on 79 sites; `(void)` + reason on 3 `cublasLtMatmulAlgoGetHeuristic` (status carried by `nresults`); file-local `CUBLASLT_CHECK` (10 uses) replaced | +215 / -179 | applied |
| H-016 gate | `tools/check_cuda_discards.py`, `tools/cuda_discard_allowlist.txt`, `scripts/ci_static_gates.sh` | matcher `cuda<Upper>(` or `cublas<Upper>(`; selftest 22 -> 28; CEILING stays 1 | +14 / -8 | applied |
| H-017 | `tools/imp-server/handlers_admin.cpp`, `tools/imp-bench/bench_e2e.cpp` | `IMP_CUDA_CHECK_LOG` on 2 `cudaMemGetInfo` + 19 bench_e2e calls (row line numbers shifted by B1 H-019) | +25 / -21 | applied |
| H-017 gate | `tools/check_cuda_discards.py` | scan roots `src` + `tools/imp-server`; selftest 29; imp-server allowlist entries need a reason | +17 / -6 | applied |
| H-008 | `tools/check_dead_inline_accessors.py`, `tools/dead_inline_allowlist.txt`, `scripts/ci_static_gates.sh` | wrapped signature joined (JOIN_MAX 8); `--selftest` 6 cases; 3 findings: 2 deleted, `apa.cuh:pass2` allowlisted (vendored public API) | +46 / -8 | applied |
| H-008 dead code | `model/loader_assign.h`, `compute/ptx92_utils.cuh` | deleted `assign_quant_with_scales` (H-007, pulled from B4), `cvt_e4m3x4_to_f32x4`, its only callee `cvt_e4m3x2_to_f16x2` (all `inline` / `__forceinline__`, 0 callers) | +0 / -30 | applied; B4 drops H-007 |

| gate | fixture | old gate | new gate |
|---|---|---|---|
| `check_launch_guards.py` | `src/bad.cuh`: `k<<<1, 32, 0, s>>>(p);` unguarded | rc 0, `0/0` scanned, OK | rc 1, `FAIL src/bad.cuh:3` |
| `check_cuda_discards.py` (cuBLAS) | `src/compute/bad.cpp`: bare `cublasSetStream`, `cublasGemmEx` | rc 0, 0 found | rc 1, `2 discarded status > allowed 0 (lines 2,3)` |
| `check_cuda_discards.py` (imp-server) | `tools/imp-server/bad.cpp`: bare `cudaMemGetInfo` | rc 0, 0 found | rc 1, `1 discarded status > allowed 0 (lines 3)` |
| `check_dead_inline_accessors.py` | `src/x.h`: `inline void wrapped_dead(...,\n ...) {` | rc 0, 0 found | rc 1, `DEAD src/x.h:2 wrapped_dead()` |

| finding / skip | evidence | proposed next step |
|---|---|---|
| apa.cuh is vendored ("change there first, then copy") | header line 4; edited in ddf4f20e, reverted in 7e445f93 | add a peek after `reset_stats_kernel` in kekzl/apa, re-vendor, drop the ALLOWLIST entry (a stale entry fails the gate) |
| tools/ outside the discard gate | same matcher over `tools/imp-bench`, `imp-quantize`, `imp-cli`: 171 (bench_attention.cu 89, bench_gemm.cu 63, bench_q8_imma.cu 12, imp-quantize 7) | next batch |
| GPU-only checks | no GPU in this batch | validator list in the B3 report |

## 2026-10-15 Phase 4 B3 (validator)

Code head `32213359`, image `imp:b3-final`, against the Phase 3 baseline `5a30cb5f`. Verdict: all gates PASS, both mutations caught.

| gate | baseline | B3 | result |
|---|---|---|---|
| build warnings | 13 unique lines | 13, diff empty | PASS |
| `make test-gpu` | 0 FAILED | 0 FAILED, `[ RUN ]` list diff 0 lines | PASS |
| reference dumps (5 models x 3 prompts x 2 runs) | `*.choices.json` | 30/30 byte-identical | PASS |
| ptxas rows, imp-server / imp-cli / imp-quantize | 1024 / 1024 / 1026 | identical | PASS |
| decode tg256 pooled median | 371.63 tok/s (gate 364.20..379.06) | 371.73 (+0.03 %) | PASS |
| decode tg256 after pp2048 | 345.41 | 345.60 (+0.06 %) | PASS |
| prefill pp512 / pp2048 | 30444 (30125..30530) / 48919 (48739..49049) | 30430 / 48979 | PASS |
| `cuBLAS error` lines, test-gpu + 10 dump servers | 0 / 0 | 0 / 0 | PASS: no silently failing cuBLAS call on the success path |
| `kernel launch failed` lines, test-gpu | 7 | 9 | PASS: the 2 new lines are the new `pdl_launch.cuh:38` check reporting the error `GgufQ8_0Dp4aRegistryDispatchMatchesDirectPath` already raised in the baseline |

| mutation | binary, filter | fixed | mutant |
|---|---|---|---|
| C-014: dp4a row-parallel GEMV `pdl::launch` block 2048 | test-compute `GemmDP4ATest.Q8_0_Q8_1_Basic` | exit 0, 0 log lines | exit 1, 1 `CUDA kernel launch failed at` line |
| H-016: `vision_gemm` lda 0 | test-e2e `VisionGolden.Gemma3SigLIP` | exit 0, 0 lines, 0 skipped | exit 1, 164 `cuBLAS error: cublasGemmEx` lines |

| finding (not B3) | evidence | next step |
|---|---|---|
| test passes while raising CUDA errors | `GemmKernelRegistryTest.GgufQ8_0Dp4aRegistryDispatchMatchesDirectPath`: `cudaMemsetAsync` `invalid argument` at `gemm_dp4a.cu:143-144` and a failed launch at `gemm_dp4a.cu:135`, in the baseline and in B3 | find-stubs pass: assert no CUDA error in that test |

## 2026-10-15 Phase 4 B3 (orchestrator decision)

| row | decision | why |
|---|---|---|
| C-014, C-015, H-016, H-017 | ship | gates above |
| `apa.cuh` launch-guard allowlist entry | GO | vendored from kekzl/apa; the entry goes when upstream adds the check |
| H-008 incl. H-007 (3 dead inline functions), `apa.cuh:pass2` dead-inline allowlist entry | GO | no callers, ptxas rows identical; B4 drops H-007 |
| 171 discarded results in `tools/imp-bench`, `imp-quantize`, `imp-cli` | next batch | outside the shipped binaries' hot paths |

## 2026-10-15 Phase 4 B4 (builder)

Branch `cleanup/b4-dead-code` off `origin/main` 1ab58bc7, code head 60d148cd. Compile only, no GPU run. `make build DOCKER_IMG=imp:b4-branch`: 13 unique warning lines (main: 13), all in untouched `tests/*.cpp` and `stl_vector.h`, 0 in a changed file. `make dev` green; `make dev-test` 13/13. Every commit passed `make preflight` (final: 0 failures). Code diff: 26 files, +104 / -285. Dead-symbol evidence: `codegraph sync` then `codegraph callers <sym>` (control `write_kv_cache`: callers found) plus `rg -n '\b<sym>\b' src tools tests include fuzz docs scripts cmake`.

| row | commit | files | change | lines +/- | evidence | result |
|---|---|---|---|---|---|---|
| H-001 | 26092a1c | `core/tensor_view.h` | header deleted | +0 / -74 | `rg tensor_view` in code = 0 includes; `TensorView` only in its own file; codegraph callers 0 | applied |
| H-003 | bcd89b16 | `exec/executor.h`, `exec/executor_workspace_buffers.cpp`, `exec/executor_workspace.cpp` | `GraphExecutor::release_moe_batch_buf` deleted; locator comment names `.cpp` (was `.cu`) | +2 / -17 | codegraph callers 0; rg = decl + def + 1 comment | applied |
| H-004 | 698952e7 | `memory/mem_account.h`, `memory/mem_account.cpp`, `runtime/engine.cpp` | `MemAccount::unattributed_bytes` deleted; 2 comments name the live `unattributed_signed_bytes` (`engine_workspace_warmup.cpp:512`) | +6 / -11 | codegraph callers 0; rg = decl + def + 2 comments | applied |
| H-005 | 62a4d1e1 | `model/hf_config_loader.h`, `model/hf_config_loader.cpp` | `HFConfigLoader::load_added_tokens` + its `AddedToken` struct deleted | +0 / -42 | codegraph callers 0; rg = decl + def; `HFConfigLoader::AddedToken` used only there | applied |
| H-006 | 10fca744 | `core/process_diag.h`, `core/process_diag.cpp` | setter deleted, getter kept | +0 / -2 | codegraph callers 0; rg = decl + def | applied |
| H-007 | - | - | - | - | done in B3 (#2735) | skipped |
| H-010 | 4ea1f86b | `compute/gdn.h`, `compute/gdn.cu`, `tests/test_gdn.cu` (comment) | host launcher `gdn_scan_prefill_f32` deleted; `gdn_scan_decode_kernel` stays (launched by `gdn_scan_decode_f32`) | +1 / -24 | codegraph callers 0; rg = decl + def + 1 test comment | applied |
| C-008 | e6e872c4 | `compute/attention_paged.cu` | `static sm_ver` and the `sm_ver >= 90` term deleted; non-pipe arm still reachable via `attention.splitk_pipe=false` | +3 / -4 | value 120 on every device the fatbin loads on | applied (guard) |
| C-008 rest | - | `compute/attention_dispatch.cpp`, `attention.h` | `get_device_sm_version` kept | - | REFUTED as dead: live caller `tools/imp-bench/bench_attention.cu:168` (`sm >= 120` gate, 108-line block); codegraph callers = `bench_attention` | next batch |
| C-009 | 17140d24 | `compute/mmq_q8_imma.cu`, `mmq_q8_imma_q4k.cu`, `mmq_q8_imma_q51.cu`, `mmq_q8_imma_q6k.cu`, `mmq_q4k_imma_tile.cu`, `gemm_cutlass_mxfp8_sm120.cu` | `#if __CUDA_ARCH__ >= 800` x4, `>= 750` x1, `>= 890` x1 (dead `return 0` else) removed | +0 / -14 | `__CUDA_ARCH__` = 1200 in both device passes | applied |
| C-010 | - | `core/pdl.cpp:53`, `gemm_grouped_nvfp4_smallM.cu:716`, `gemm_capture_fp16_sm120.cu:270`, `gemm_cutlass_grouped_3x.cu:96` | not changed | - | can flip: each returns false (uncached) when `cudaGetDevice` / `cudaGetDeviceProperties` fails (no device, CPU lane) and false on a cc < 12.0 device; removal turns both into true | needs orchestrator go |
| C-011 | 5e168a96 | `core/pdl_device.cuh` | guard `>= 900` -> `defined(__CUDA_ARCH__)`; comment no longer claims a 120f no-op | +5 / -4 | host pass: undefined before and after; device passes: 1200 | applied |
| C-003 | 093e6a5b | `compute/constrain_common.h`, `compute/json_constrain.cu` | `__global__ inline` body moved verbatim to `json_constrain.cu`; header declares it | +35 / -27 | launchers `json_constrain.cu:708`, `schema_constrain.cu:367,417`, `tests/test_json_constrain.cu:172`; non-template kernel, extern host stub, no `-rdc` | applied |
| H-033 | ac422517 | `CMakeLists.txt` | stale TurboQuant / sm_120f note deleted | +0 / -3 | TurboQuant retired 2026-05-17; 0 sm_120f SASS emitted | applied |
| C-013 + H-034 | 60d148cd | `CMakeLists.txt`, `compute/attention_mxfp4_prefill.h` (comment) | option and 4 `if(IMP_USE_CUTLASS)` blocks removed, bodies dedented; `IMP_USE_CUTLASS=1` define on `imp` and `test-quant` dropped | +52 / -63 | `rg IMP_USE_CUTLASS` in src/tools/tests = 1 comment; 0 `#if` readers | applied |

| SASS check (no GPU) | imp:b3-final | imp:b4-branch | result |
|---|---|---|---|
| per-kernel rows, imp-server / imp-cli / imp-quantize (`cuobjdump -sass` in `imp:toolchain`, sha1 of each function body without addresses and encodings) | 1024 / 1024 / 1026 | 1023 / 1023 / 1025 | -1 each: the duplicate `constrain_mask_allow_kernel` row (C-003) |
| unique (kernel, SASS hash) pairs | - | - | identical in all 3 binaries |
| hash tool discriminates | - | 1014 distinct hashes over 1023 kernels, 0 empty bodies | - |
| `compute_120f` PTX | not compared | not compared | validator |

| finding / skip | evidence | proposed next step |
|---|---|---|
| `get_device_sm_version` + cache + INFO line | only caller after C-008: `tools/imp-bench/bench_attention.cu:168`, whose `sm >= 120` gate is always true | drop the bench gate (dedent 108 lines), then delete the function |
| C-010 | see row | owner call: fold the 4 checks into one cached `device_query_ok()` that keeps the false-on-error result, or accept true on a cc < 12.0 device |
| GPU-only checks | no GPU in this batch | validator: reference dumps, test-gpu (`*JsonConstrain*`, `*SchemaConstrain*` exercise the moved kernel), PTX diff |

## 2026-10-15 Phase 4 B4 (validator)

Code head `60d148cd`, image `imp:b4-final`, against the Phase 3 baseline image `imp:test-imp-baseline-016f0c60`. Verdict: all gates PASS; perf judged by a paired control.

| gate | baseline | B4 | result |
|---|---|---|---|
| build warnings | 13 unique lines | 13, diff empty | PASS |
| `make test-gpu` | 0 FAILED | 0 FAILED; `[ RUN ]` diff = the B2 test only | PASS |
| reference dumps (5 models x 3 prompts x 2 runs) | `*.choices.json` | 30/30 byte-identical | PASS |
| ptxas rows imp-server / imp-quantize | 1024 / 1026 | 1023 / 1025: only `constrain_mask_allow_kernel` (the header duplicate) gone | PASS |
| SASS hash per (kernel, arch), 3 binaries | - | 1 removed row (the duplicate); 2 rows change only the ELF arch label (`sm_120a` <-> `sm_120f`) with identical hashes | PASS |
| PTX, imp-server | 998 entries | 997; diff = 66 removed lines of the duplicate entry, 0 added | PASS |

| perf, Qwen3-Coder-30B-A3B FP4, pooled n=10 | decode tg256 | pp512 | pp2048 | tg256 after pp2048 |
|---|---|---|---|---|
| Phase 3 baseline, 06:xx | 371.63 | 30444 | 48919 | 345.41 |
| B4 run 1, 11:26 | 369.52 (-0.57 %) | 30210 | 48592 (below the baseline min 48739) | 343.06 |
| control: baseline image, 11:31 | 366.75 | 29916 | 48495 | 341.00 |
| B4 run 2, directly after the control | 368.65 (+0.52 % vs control) | 30179 | 48318 (-0.37 % vs control) | 340.16 |

| perf verdict | evidence |
|---|---|
| host drift, not B4 | the unchanged baseline image lost 1.3 % decode since the Phase 3 run; B4 vs control inside each other's min..max; SASS identical |

## 2026-10-15 Phase 4 B4 (orchestrator decision)

| row | decision | why |
|---|---|---|
| H-001, H-003..H-006, H-010, C-003, C-008 guard, C-009, C-011, C-013 + H-034, H-033 | ship | gates above |
| C-010 | DEFER to the deferred list | removal flips the result to true on a failed device query (CPU lane) and on cc < 12.0 |
| C-008 `get_device_sm_version` | next batch | live caller `tools/imp-bench/bench_attention.cu:168` |

## 2026-10-15 Phase 4 B7 (builder)

Branch `cleanup/b7-dup-helpers` off `origin/main` 1ab58bc7. Compile + CPU only, no GPU run. `make build DOCKER_IMG=imp:b7-branch`: 13 unique warning lines (main: 13), 0 in a changed file. `make dev` green; `make dev-test` 13/13. Every commit passed `make preflight`.

| row | files | change | lines +/- | result |
|---|---|---|---|---|
| H-012 | `model/model_limits.h`, `model/gguf_tensor_assign.cpp` (2), `vision/vision_loader.cpp`, `lora/lora_adapter.cpp`, `vision/qwen3vl_vision_map.cpp`, `exec/pre_dequant_phase0_nvfp4_loader.cpp` (2), `tests/test_checkpoint_limits.cpp` | `parse_index(s, IndexParse)`: `kStrict` (default, unchanged), `kAtoi` = `(int)strtol` (glibc `atoi`), `kStoi` = `std::stoi` with throw -> -1; 3 `stoi` sites -> `kStoi`, 4 `atoi` sites -> `kAtoi`; every site rejects < 0 after the parse, so throw and -1 take the same branch | +70 / -27 | applied, behaviour-neutral for every input |
| H-014 | `core/cache_dir.h` (new), `memory/weight_cache_file.cpp`, `memory/library_reserve_cache.cpp`, `tests/test_library_reserve_cache.cpp` | `imp_cache_dir(leaf)`: XDG, then HOME, then ""; fallbacks stay at the callers (`/tmp/imp-warm-cache`, "") | +74 / -12 | applied |

| test | cases |
|---|---|
| `CheckpointLimits.IndexParsingStrictEdgeCases` | "007" -> 7, INT_MAX, "x", "+3", " 3", "3 ", "-0" -> 0, "-5" |
| `CheckpointLimits.IndexParsingLegacyPoliciesMatchAtoiAndStoi` | 13 inputs against `std::atoi` / `std::stoi` as oracle; pinned "x" -> 0 (kAtoi), "3x" -> 3 (kStoi), "4294967296" -> 0 (kAtoi), 2 overflows -> -1 (kStoi) |
| `CheckpointLimits.IndexParsingRejectsOverflowAndJunk` (existing) | empty, "1x", "-1", 3 overflows |
| `LibraryReserveCache.CacheDirResolvesXdgThenHomeThenEmpty` | XDG set; XDG "" + HOME; XDG unset + HOME "" |

| mutation (one `make dev-test` run, files restored from scratchpad backups) | failing tests | rerun after restore |
|---|---|---|
| kStrict: `return 0` instead of `return -1` on a parse error | `IndexParsingRejectsOverflowAndJunk`, `IndexParsingStrictEdgeCases`, `AnAliasingLayerIndexDoesNotOverwriteLayerZero` | 13/13 |
| kStoi: drop the `end == s.c_str()` no-digits check | `IndexParsingLegacyPoliciesMatchAtoiAndStoi` | 13/13 |
| `imp_cache_dir`: accept an empty `XDG_CACHE_HOME` | `CacheDirResolvesXdgThenHomeThenEmpty` | 13/13 |

| needs orchestrator go (switch to `kStrict`) | legacy malformed mapping | input source |
|---|---|---|
| `model/gguf_tensor_assign.cpp:78` layer (`kStoi`) | "3x", " 3", "+3" -> layer 3 | GGUF tensor name |
| `model/gguf_tensor_assign.cpp:327` expert (`kStoi`) | "3x" -> expert 3 | GGUF tensor name |
| `vision/vision_loader.cpp:469` layer (`kStoi`) | "3x" -> layer 3 | mmproj GGUF tensor name |
| `lora/lora_adapter.cpp:32` layer (`kAtoi`) | "x" -> layer 0; overflow -> truncated | PEFT adapter key |
| `vision/qwen3vl_vision_map.cpp:22` block (`kAtoi`) | digit run > INT_MAX -> low 32 bits ("4294967296" -> block 0) | SafeTensors name, digits pre-checked |
| `exec/pre_dequant_phase0_nvfp4_loader.cpp:288,300` layer, expert (`kAtoi`) | "x" -> 0 | internal keys built by `std::to_string` (`weight_map.cpp`, `weight_upload.cpp`): unreachable from a checkpoint |

| finding / skip | evidence | proposed next step |
|---|---|---|
| per-loader malformed-index tests | legacy policies keep the old mapping, so a per-loader test today pins the wrong mapping; qwen3vl already has `MalformedIndexIsUnknownNotZero` | add with each `kStrict` go |
| GPU-only checks | no GPU in this batch | validator list in the B7 report |

## 2026-10-15 Phase 4 B7 (validator)

Code head `2ac22739`, image `imp:b7-final`, against the Phase 3 baseline image `imp:test-imp-baseline-016f0c60`. Verdict: all gates PASS.

| gate | baseline | B7 | result |
|---|---|---|---|
| build warnings | 13 unique lines | 13, diff empty | PASS |
| `make test-gpu` | 0 FAILED | 0 FAILED; `[ RUN ]` diff = 3 new B7 tests + the B2 test | PASS |
| reference dumps (5 models x 3 prompts x 2 runs; GGUF layer, vision `v.blk.`, NVFP4 dense + MoE parse sites) | `*.choices.json` | 30/30 byte-identical | PASS |
| Qwen3-VL-4B-Instruct dumps (vision-map parse site), 3 image prompts x 2 runs, baseline image vs B7 image | - | 6/6 byte-identical | PASS |
| LoRA key parse site | - | `LoraHotSwap.*` 3/3 OK in test-gpu | PASS |
| ptxas rows, 3 binaries | 1024 / 1024 / 1026 | identical | PASS |
| SASS hash per kernel, 3 binaries | - | 0 removed, 0 added | PASS |
| decode tg256 pooled median | 371.63 tok/s (gate 364.20..379.06) | 372.06 (+0.12 %) | PASS |
| decode tg256 after pp2048 | 345.41 | 345.12 (-0.08 %) | PASS |
| prefill pp512 / pp2048 | 30444 (30125..30530) / 48919 (48739..49049) | 30360 / 48958 | PASS |

| not exercised | why |
|---|---|
| GGUF expert parse site (`gguf_tensor_assign.cpp:327`, 5-part names) | local MoE GGUFs use packed `_exps` tensors; covered by the `kStoi`/`kAtoi` reference unit test only |

## 2026-10-15 Phase 4 B7 (orchestrator decision)

| row | decision | why |
|---|---|---|
| H-012, H-014 | ship | gates above |
| `kStrict` at the 7 parse sites | DEFER to the deferred list | behaviour change on malformed tensor names ("3x" -> layer 3 today); belongs in a fix PR with one malformed-name test per loader |

## 2026-10-15 Phase 4 B5 (validator)

Branch `cleanup/b5-magic-numbers` head `5a6b6e1c` (H-029, C-032, C-033 rename part, C-034, H-031, H-032; builder log on the branch), image `imp:b5-final`. Verdict: FAIL on test-gpu; not shipped.

| gate | reference | B5 | result |
|---|---|---|---|
| build warnings | 13 | 13, diff empty | PASS |
| SASS hash per kernel vs `imp:b4-final`, 3 binaries | - | 0 removed, 0 added | PASS |
| PTX imp-server vs `imp:b4-final` | 886193 lines | diff 0 | PASS |
| reference dumps | `*.choices.json` | 30/30 byte-identical | PASS |
| decode tg256 pooled median | 371.63 tok/s | 370.72 (-0.24 %) | PASS |
| `make test-gpu`, run 1 | 0 FAILED | `DeviceFaultSignalTest.FaultInsideAStepCancelsFaultsAndRefusesLaterRequests` FAILED (child exit 7: "the fault kernel did not poison the context"); suite stopped, DetEval + Qwen38 tokenizer not run | FAIL |
| `make test-gpu`, run 2 | - | same test FAILED (child exit 8: `AsyncGraphLoop: no token for 30 s`) | FAIL |
| control: `make test-gpu` on `imp:b4-final` (main without B5), between the two | - | test OK (3106 ms), suite green | - |

| isolated runs of the test (`--gtest_filter='DeviceFaultSignalTest.*'`) | failed / runs |
|---|---|
| baseline image `imp:test-imp-baseline-016f0c60` | 0 / 60 |
| `imp:b1-base` (main 8d4c4148, pre-cleanup) | 0 / 20 |
| `imp:b1-final`, `b2-final`, `b3-final`, `b4-final` | 1 / 20, 1 / 20, 2 / 20, 1 / 20 |
| `imp:b5-final` | 2 / 40 |
| B5 with a DIAG print: dev build / image | 0 / 40, 0 / 40 |
| all isolated failures | 7, all within the first 120 runs; 0 / 160 after them, every image |

## 2026-10-15 Phase 4 B5 (orchestrator decision)

| item | decision | why |
|---|---|---|
| B5 | REVERT: not merged; branch `cleanup/b5-magic-numbers` kept | dispatch rule: any gate failure reverts, no fix forward; full suite red 2/2 with B5, green on the control |
| B5 re-validation | later: 3 x `make test-gpu` on a rebuilt B5 image vs 3 x main, interleaved | separate a B5 cause from the timing flake |
| `DeviceFaultSignalTest` timing flake | new finding, needs a GPU repro | isolated failures on 5 images with unrelated code; exit 7 and exit 8 are two failure modes |
| C-033 runtime `device_sm_count()` | DEFER | turns `kMinBlocks` into a runtime value; retry-on-error is a behaviour change |

## 2026-10-15 Phase 4 B6 (builder)

Branch `cleanup/b6-headers-build` off `origin/main` 887edce4, code head d85a30e2. Compile + CPU only, no GPU run. `make build DOCKER_IMG=imp:b6-branch`: 13 unique warning lines (main `imp:b6-main`: 13), diff empty. `make dev` green; `make dev-test` 13/13. Every commit passed `make preflight` (0 failures). Code diff: 14 files, +74 / -46.

| row | commit | files | change | result |
|---|---|---|---|---|
| H-020 | 9bc8c234 | `exec/executor.h` | 2 includes (`json_constrain.h`, `schema_constrain.h`) removed | applied; build named 0 TUs that needed them through executor.h |
| H-021 (Tokenizer) | cb561ce4 | `model/model.h`, `model/model.cpp`, `model/gguf_loader.cpp`, `vision/vision_pipeline.cpp`, `docs/CONFIG.md` | `class Tokenizer;` forward declaration; `set_tokenizer` out of line; 2 TUs named by the build include `tokenizer.h`; CONFIG.md citation `model.cpp:191` -> `:194` | applied |
| H-021 (GenerationConfig) | bb3af8ae | `model/generation_config.h` (new), `model/hf_config_loader.h`, `model/model.h`, `runtime/presets.h`, `runtime/presets.cpp`, `tests/test_sampling_defaults_parity.cpp` | struct moved verbatim to namespace scope; `HFConfigLoader::GenerationConfig` kept as alias; 1 test TU named by the build includes `hf_config_loader.h` | applied |
| H-009 | d85a30e2 | `CMakeLists.txt`, `Makefile`, `.github/workflows/ci.yml` | 10 TUs out of libimp into static `imp_test_kernels` (built only with `IMP_BUILD_TESTS`, linked by test-quant, PIC, no `-rdc`); kernel-resources dump (Makefile + CI step) reads `libimp_test_kernels.a` too; stale CMake + ci.yml comments removed | applied |

| header (build-dev `ninja -t deps`, 844 targets) | TUs main | TUs B6 |
|---|---:|---:|
| `compute/json_constrain.h` | 130 | 78 |
| `compute/schema_constrain.h` | 127 | 75 |
| `model/tokenizer.h` | 274 | 170 |
| `model/hf_config_loader.h` | 218 | 18 |
| `model/generation_config.h` (new) | - | 218 |
| `exec/executor.h`, `model/model.h` (control) | 119, 201 | 119, 201 |

| `make dev` after one header change (indicative: other builders ran; load avg 3.9 main, 11.6 B6) | ninja steps main | ninja steps B6 | `touch` ms main / B6 (ccache hits) | content edit ms main / B6 (ccache misses) |
|---|---:|---:|---:|---:|
| `compute/json_constrain.h` | 156 | 104 | 3523 / 3489 | 74955 / 51067 |
| `compute/schema_constrain.h` | 153 | 101 | 3262 / 3522 | 68545 / 50197 |
| `model/tokenizer.h` | 300 | 196 | 3684 / 3624 | 130790 / 100673 |
| `model/hf_config_loader.h` | 244 | 44 | 3707 / 3393 | 102928 / 9940 |

| H-009 check | result |
|---|---|
| 10 moved objects, build-dev `imp.dir` vs `imp_test_kernels.dir` | byte-identical (`cmp`) |
| tests OFF + bench ON (release-docker shape) | imp-bench, imp-server, imp-cli, imp-quantize link; libimp.a 354 objects, 0 of the 10 |
| kernel-resources pin (libimp.a + libimp_test_kernels.a) | 92 at-risk kernels match, baseline unchanged; libimp.a alone reports 3 IMPROVED (`bench_fused`, `bench_separate`, `bench_tma_v_load`) |
| test-quant `--gtest_list_tests` | 275 (imp:b6-main) / 275 (build-dev), names identical |

| `make check-test-lanes` | main | B6 |
|---|---|---|
| `ctest -L unit` / no CI lane | 2357 / 1286 | 2357 / 1286 |
| test-quant (no CI) | 262 | 262 |

| SASS check (no GPU) | imp:b6-main | imp:b6-branch | result |
|---|---|---|---|
| per-kernel rows imp-server / imp-cli / imp-quantize | 1023 / 1023 / 1025 | 1023 / 1023 / 1025 | diff 0 lines in all 3 |
| distinct SASS hashes | - | 1014 / 1014 / 1016 | - |

| host symbol diff (`nm -C --defined-only`, 3 shipped binaries) | rows |
|---|---|
| mangling renamed (`HFConfigLoader::GenerationConfig` -> `GenerationConfig`) | `load_generation_config` (T, t), `resolve_sampling_defaults` (T, not in imp-quantize) |
| new out-of-line | `Model::set_tokenizer` |
| gone weak instantiations | `Tokenizer::~Tokenizer()`, `__uniq_ptr_impl<Tokenizer>::reset` |
| compiler label | `CSWTCH.149` -> `CSWTCH.150` |

| finding | evidence | proposed next step |
|---|---|---|
| `json_constrain.h` / `schema_constrain.h` still reach 78 / 75 TUs | other includers (`constraint_manager.h` path) | price with the SETTLED reverse-BFS before acting (H-022) |
| `tokenizer.h` still reaches 170 TUs | `json_constrain.h` / `schema_constrain.h` include it | same |
| GPU-only checks | no GPU in this batch | validator: reference dumps, test-gpu (test-quant runs the moved kernels) |

## 2026-10-15 Phase 4 B6 (validator)

Code head `d85a30e2`, image `imp:b6-final`; paired control `imp:main-ctrl` built from main `36fa6317`. Verdict: all gates PASS.

| gate | main control | B6 | result |
|---|---|---|---|
| build warnings | 13 | 13, diff empty | PASS |
| SASS hash per kernel, imp-server / imp-cli / imp-quantize | 1023 / 1023 / 1025 | diff 0 / 0 / 0 | PASS |
| `make test-gpu` | 2/2 green, 3700 `[ RUN ]` each | 2/2 green, 3700 each, list diff 0; `DeviceFaultSignalTest.FaultInside...` OK in all 4 runs | PASS |
| reference dumps vs the Phase 3 baseline | - | 30/30 byte-identical | PASS |
| decode tg256 pooled median, paired run | 371.36 tok/s | 370.93 (-0.12 %) | PASS |
| prefill pp512 / pp2048, paired run | 30298 / 48885 | 30330 / 48799 | PASS |

| measurement note | value |
|---|---|
| host noise during the paired perf | spread up to 8.72 % (main) and 7.58 % (B6) on pp512; medians compared |
| first B6 perf run (unpaired) | SM clock 2778..2855 MHz, pp512 spread 9.78 %; discarded for the paired run |
| first two B6 suites | not run: the GPU lock was taken from the main worktree, `make test-gpu` in the B6 worktree refused it (validator script error) |

## 2026-10-15 Phase 4 B6 (orchestrator decision)

| row | decision | why |
|---|---|---|
| H-020, H-021, H-009 | ship | gates above |
| kernel-resources pin reads `libimp_test_kernels.a` too | accept | the 3 moved bench kernels would otherwise read as IMPROVED and fail the gate; baseline unchanged |
| rest fan-in via `constraint_manager.h`, `tokenizer.h` | with H-022 (deferred list) | M risk |

## 2026-10-15 Phase 4 B8 (builder)

Branch `cleanup/b8-warnings` off `origin/main` 36fa6317, code head d5c0c5d1. Compile + CPU only, no GPU run. `make build DOCKER_IMG=imp:b8-branch` (524 CXX + 302 CUDA compile steps): 13 unique warning lines (main: 13), diff empty, 0 `-Wshadow`. `make dev` green; `make dev-test` 13/13. Every commit passed `make preflight`. Code diff: 24 files, +211 / -209. origin/main moved to 32b66557 during the batch; the branch is not rebased.

| row | commit | files | change | lines +/- | evidence | result |
|---|---|---|---|---|---|---|
| H-027 shorten (6 host) | bf732436 | `exec/executor_attention.cpp`, `exec/executor_forward_moe_legacy.cpp` | `static_cast<int>` on `NvFP4QuantResult::N/K` (`int64_t`) at the `int` GEMV args: attention:655,656; moe_legacy:282,329 (N and K each) | +6 / -6 | same implicit conversion spelled out; host objects identical | applied |
| H-027 shadow (5 listed host) | e3c9eb70 | `exec/executor_attention.cpp` (`eps` -> `post_eps`), `exec/executor_attention_qkv.cpp:257` (`h` -> `kv_h`), `runtime/engine_kv_cache_init.cpp:506` (`blocks_per_seq` -> `blocks_per_req`), `model/tokenizer.cpp:657` (`n` -> `entry`), `imp-server/handlers_misc.cpp:651` (`req` -> `embed_req`) | rename | +34 / -34 | g++ `-Wshadow` reports each; host objects identical | applied |
| g++ `-Wshadow`, 4 more sites | e3c9eb70 | `exec/executor_attention_qkv.cpp:152` (`h` -> `wh`), `exec/executor_forward_moe_cutlass.cpp:289` (`n` -> `n_ex`), `imp-quantize/awq_plan.cpp:473` (`f` -> `fs`), `tests/test_determinism_e2e.cpp:208` (`info` -> `param_info`) | rename | (same commit) | not in the clang lane (g++ only) | applied |
| g++ `-Wshadow`, `Tensor` ctors (11) | c260f775 | `core/tensor.cpp` | ctor params `ptr, qt, nd, dims, strides, dev` in the definition; `this->` dropped | +11 / -11 | `tensor.h` keeps member-named params: 120 `/*on_device=*/` argument comments (bugprone-argument-comment, WarningsAsErrors) | applied |
| g++ `-Wshadow`, attention lambdas (13) | cd6be0ab | `exec/executor_attention.cpp`, `exec/executor_attention_decode.cpp`, `exec/executor_attention_prefill.cpp` | `decode_attend` params -> `sub_state, sub_n, sub_q, sub_k, sub_v, sub_ao, sub_block_tables`; `prefill_attend_seq` -> `seq_n, seq_q, seq_k, seq_v, seq_ao, seq_block_tables`; both fragments get `// clang-format off` | +120 / -119 | object identical before the marker; after it only 21 line-number immediates (`__LINE__`, +-1) | applied |
| H-027 flag | b08e8895 | `CMakeLists.txt` | `-Wshadow` added to `imp_warnings` under `$<COMPILE_LANGUAGE:CXX>` | +2 / -1 | 0 findings in deps (FetchContent targets do not link `imp_warnings`; no header warning seen) | applied |
| H-027 `-Wsign-conversion`, `-Wimplicit-int-float` | - | - | not enabled | - | inventory: 1850 + 473 host findings, L effort | out of scope (row says leave off) |
| H-025 | d5c0c5d1 | 4 `compute/*_constrain.h`, 4 `compute/*_constrain.cu` (host methods), `runtime/constraint_manager.cpp`, `tests/test_regex_constrain.cpp`, `tests/test_tool_call_constrain.cu` | `{Json,Schema,Regex,Grammar}Constrainer::reset()` -> `rewind()`; `ConstraintManager` `->reset()` x10 -> `->rewind()`; `.reset()` x3 on the `unique_ptr` unchanged | +38 / -38 | preflight tidy on `constraint_manager.cpp`: 0 `readability-ambiguous-smartptr-reset-call` (inventory: 13); old name gone, so a missed caller is a compile error | applied |

| `-Wshadow` count (g++, `make dev`) | value |
|---|---|
| main, flag off | 0 (flag absent); 13 unique warning lines total |
| main + flag, before renames | 33 unique sites: tensor.cpp 11, executor_attention.cpp 14, executor_attention_qkv.cpp 2, 1 each in moe_cutlass, tokenizer, engine_kv_cache_init, awq_plan, handlers_misc, test_determinism_e2e |
| branch + flag | 0; `make build` 13 unique lines, diff vs main empty |

| behaviour check (no GPU) | reference | B8 | result |
|---|---|---|---|
| host `objdump -d -r` of all 522 first-party `.cpp.o` in `build-dev`, before vs after bf732436..cd6be0ab (pre-marker) | main 36fa6317 | 522 / 522 identical | PASS |
| same, after the `clang-format off` lines | - | 521 identical; `executor_attention.cpp.o`: 21 `mov`/`push` immediates = log `__LINE__` +-1 | PASS |
| same, H-025 (`Constrainer::reset()` -> `rewind()` normalised in symbol names) | cd6be0ab | `constraint_manager.cpp.o`, `test_regex_constrain.cpp.o`: 0 diff lines | PASS |
| SASS per kernel, `imp:main-ctrl` (36fa6317) vs `imp:b8-branch`, imp-server / imp-cli / imp-quantize | 1023 / 1023 / 1025 rows | 1023 / 1023 / 1025, 0 hash diffs; 1014 distinct hashes in imp-server | PASS |

| finding / skip | evidence | proposed next step |
|---|---|---|
| `readability-inconsistent-declaration-parameter-name` on `tensor.h:47,50` | 2 new advisory tidy findings (not WarningsAsErrors) from c260f775 | rename the header params together with the `/*on_device=*/` comments, or accept |
| `IMP_CUDA_CHECK_LOG` text in `executor_attention_prefill.cpp:202,205` | `#call` stringifies `kk.data`/`vv.data`, now `seq_k.data`/`seq_v.data`: CUDA-error message text only | none |
| `// clang-format off` in the two attention fragments | `git clang-format` formats a body fragment as top-level code: every changed line of a top-level block is dedented | none; applies to any later edit of those fragments |
| GPU-only checks | no GPU in this batch | validator: test-gpu (`*JsonConstrain*`, `*SchemaConstrain*`, `*Regex*`, `*Gbnf*`, `*ToolCall*`), reference dumps (attention lambdas), PTX diff |
