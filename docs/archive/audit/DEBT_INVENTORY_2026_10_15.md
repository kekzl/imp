# Debt inventory 2026-10-15 (DISPATCH_cleanup Phase 1)

Scout output, read-only, release binary v0.53.0 (cb591619) and tree 4aca9671. Host scout and CUDA scout tables verbatim.

## Host and build (H-)

Tree: `4aca9671` (branch `feat/webui-imp-redesign`; `origin/main` = `afecddf7`). Binaries: `ghcr.io/kekzl/imp:latest` (imp-cli, imp-server, imp-quantize, imp-bench, symtab present, no `--gc-sections`).
Ledgers read first: `docs/audit/SETTLED.md`, `docs/audit/DEBT_LEDGER_2026_08_21.md`. Nothing below re-reports a SETTLED/REFUTED entry; overlaps are cited.

Evidence tools:
- `rg` over `src tools tests include fuzz`, comments stripped where counted.
- codegraph (synced 2026-10-15): functions/methods with no incoming `calls/references/instantiates` edge = 734, of which 79 have <= 2 tokens tree-wide; kernels and macro-bound hooks filtered by hand.
- Linker: `readelf -s` FILE symbols (440 objects) and `nm -C --defined-only` (15607 symbols) over the 4 shipped binaries. Without `--gc-sections` a linked object keeps every function, so "present in nm" proves the object was pulled, not that the function is reachable; "absent in nm" for a non-inline function proves no shipped binary links it.
- clang-tidy: full lane (`scripts/tidy_lane.sh --all`, 282 .cpp + 144 .cu host side = 426 TUs, 0 clang errors) in `imp:lint`, on a scratch copy with `modernize-*` added and `--extra-arg=-Wall -Wextra -Wshadow -Wconversion`. 16012 unique findings, 6155 in .cpp/.h.

| id | category | file:line | evidence | risk | effort | proposed action |
|---|---|---|---|---|---|---|
| H-001 | Dead code | `src/core/tensor_view.h:9` | Whole header (74 lines, class `TensorView`) included by 0 files (`rg 'tensor_view.h'` = 0); `TensorView` has 0 mentions outside its file; last touched by the #2306 nodiscard sweep | L | S | Delete the header |
| H-002 | Dead code (test-only production code) | `src/memory/scratch_stack.h:1`, `src/memory/scratch_stack.cpp:1` | T4 `ScratchStack` (213 lines) used only by `tests/test_memory_allocators.cpp` (6 tests) + a friend decl in `span.h:22,36`; `scratch_stack.cpp` has no FILE symbol in any of the 4 binaries; `docs/internals/MEMORY.md:96,170` claim "one `ScratchStack::Mark` per forward", false (0 users in `src/exec`). AUDIT.md B2 recorded this for all tier types in 2026-07; the others were adopted since, this one was not | L (doc) / M (removal touches `span.h`, SETTLED S-20 load-bearing) | S | Owner call: correct MEMORY.md to "designed, not adopted", or adopt; do not drop the `span.h` passkey friend blindly |
| H-003 | Dead code | `src/exec/executor.h:559`, `src/exec/executor_workspace_buffers.cpp:1408` | `GraphExecutor::release_moe_batch_buf`: decl + def + one comment (`executor_workspace.cpp:710`), 0 calls; codegraph 0 incoming edges; in nm (object linked, not reachability) | L | S | Delete; `executor.h` edit needs the forced rebuild (codebase-audit skill) |
| H-004 | Dead code | `src/memory/mem_account.h:74`, `src/memory/mem_account.cpp:167` | `MemAccount::unattributed_bytes`: decl + def, 0 calls; comments at `engine.cpp:781` and `mem_account.cpp:95` describe it as a live number | L | S | Delete and fix the two comments, or wire it into `--mem-report` |
| H-005 | Dead code | `src/model/hf_config_loader.h:52`, `src/model/hf_config_loader.cpp:250` | `HFConfigLoader::load_added_tokens`: decl + def, 0 calls (src/tools/tests) | L | S | Delete |
| H-006 | Dead code | `src/core/process_diag.h:172`, `src/core/process_diag.cpp:77` | `process_diag_set_nvfp4_cublaslt_min_m`: 0 callers; the only other of 23 `process_diag_set_*` with 0 callers outside process_diag; value installed directly at `process_diag_install.cpp:69`. Same shape as SETTLED section C `process_diag_set_mxfp4_blockscale` (redundant mutator, removed) | L | S | Delete the setter, keep the getter |
| H-007 | Dead code | `src/model/loader_assign.h:19` | `assign_quant_with_scales`: 1 occurrence in tree (its definition); 0 in nm | L | S | Delete |
| H-008 | Dead code (gate blind spot) | `tools/check_dead_inline_accessors.py:61` | `DEF_RE` needs `name(` ... `)` ... `{` on ONE line; gate reports "0 header-inline definition(s) with no other mention" while `loader_assign.h:19` (H-007) and `compute/ptx92_utils.cuh:30` `cvt_e4m3x4_to_f32x4` each have exactly 1 mention (both wrap their parameter list) | L | S | Join continuation lines before matching; add a wrapped-signature selftest case |
| H-009 | Build/CMake, dead code in shipped lib | `CMakeLists.txt:389-399` | `if(IMP_BUILD_TESTS OR IMP_BUILD_BENCH)` appends 10 TUs to `libimp` (8 `tests/bench/*.cu`, `src/compute/nvfp4_quant_ref.cu`, `src/compute/mxf4nvf4_qkt_validate.cu`); release image builds `IMP_BUILD_BENCH=ON` (`release-docker.yml:108`); nm: 0 of their exported symbols (`probe_mxf4nvf4_*`, `nvfp4_quant_linear_fp16`, `nvfp4_dequant_linear_fp16`, `qkt_mxf4nvf4_validate`, `run_peak`, `fmha_v_load_bench`) in any shipped binary incl. imp-bench; callers only in `tests/`. Comment `:387` "still serve internal call sites" is false (0 src/tools callers); `ci.yml:325` prices these TUs at ~30-60 s | L | S | Gate on `IMP_BUILD_TESTS` only (or a test-only OBJECT lib linked by the test targets); release build time measured in the build step |
| H-010 | Dead code (overlaps kernel scout) | `src/compute/gdn.h:170`, `src/compute/gdn.cu:794` | Host launcher `gdn_scan_prefill_f32`: 0 callers; only mention outside decl/def is a comment `tests/test_gdn.cu:284` ("Legacy") | L | S | Delete launcher + its kernel if the kernel scout confirms the kernel has no other launcher |
| H-011 | Duplicated helpers (dtype conversion) | `src/model/weight_upload_traits.h:22,44`; `src/model/ngram_table.cpp:28,58,72`; `src/model/mxfp8_widen.cpp:15`; `tools/imp-quantize/fp8_source.cpp:47`; inline BF16 widen loops `src/model/weight_upload.cpp:577,1189,1874`, `weight_upload_traits.h:157` | `src/core/fp_bits.h:3` declares itself the "one copy" of host half/bf16 conversion; 5 more half/bf16 and 3 more E4M3 decoders exist. Deltas: `wupload::fp16_to_float` maps Inf/NaN to 0 (deliberate, `:36`); `tools/imp-quantize/quant_report.cpp:24` is deliberately independent ("a shared mistake between writer and reader cancels") and stays | M (weight bits) | M | Add E4M3 decode to `fp_bits.h`; replace copies after an exhaustive CPU equality test (65536 fp16 / 256 e4m3 inputs); keep the Inf/NaN->0 variant as a named wrapper |
| H-012 | Duplicated helpers / inconsistent validation | `src/model/gguf_tensor_assign.cpp:79,334`; `src/vision/vision_loader.cpp:471`; `src/lora/lora_adapter.cpp:31`; `src/exec/pre_dequant_phase0_nvfp4_loader.cpp:287,299`; `src/vision/qwen3vl_vision_map.cpp:22` | 7 tensor-name index parses bypass `model_limits.h:29` `parse_index` (from_chars, rejects non-digits and overflow): `std::stoi`+`catch(...)` accepts "3x" as 3; `atoi` maps ".layers.x." to layer 0 and is UB on overflow. `src/model/CLAUDE.md`: "index parsed out of a name: drop" | L | S | Route all 7 through `parse_index`; one unit test per loader with a malformed index |
| H-013 | Inconsistent validation | `tools/imp-cli/args.cpp` (25), `tools/imp-server/args.cpp` (14), `tools/common/args_common.cpp:31` (5) | 44 `atoi/atof` on CLI numbers while `imp.conf`/`--set` use strict `src/runtime/config.cpp:58` `parse_int` (trailing garbage rejected, mutation M60): `--prefill-chunk abc` = 0 silently | M (malformed CLI input changes from silent 0 to error) | S | Shared strict parser in `tools/common/args_common.cpp`, error on garbage |
| H-014 | Duplicated helpers | `src/memory/weight_cache_file.cpp:66`, `src/memory/library_reserve_cache.cpp:22` | Same XDG_CACHE_HOME -> HOME/.cache/imp/<leaf> resolution twice; fallbacks differ (`/tmp/imp-warm-cache` vs empty) | L | S | One `imp_cache_dir(leaf)` in `src/core`; keep both fallbacks explicit |
| H-015 | Duplicated helpers (library handles) | `src/compute/gemm.cpp:65,77`; `src/compute/gemm_grouped.cpp:28`; `src/compute/attention_cublas.cu:44`; `src/compute/attention_mxfp4_prefill.cu:322`; `src/vision/vision_encoder.cu:25`; `src/vision/qwen3vl_encoder.cpp:28`; `src/vision/internvl_encoder.cpp:24` | 7 lazily created cuBLAS handles + 1 cuBLASLt, each a file static with its own reset hook; 4 of 7 discard the `cublasCreate` status (`attention_mxfp4_prefill.cu:322`, the 3 vision sites); math mode differs per handle (TF32 vs DEFAULT, `gemm_grouped.cpp:33`) | M (math mode / stream binding per call site) | M | Shared create helper with status check first; a single handle with per-call math mode only after an A/B |
| H-016 | Inconsistent error handling | `src/compute/gemm.cpp` (31), `src/compute/gemm_batched.cpp` (14), `src/vision/vision_encoder.cu` (12), `src/compute/attention_cublas.cu` (7), `src/vision/qwen3vl_encoder.cpp` (5), `src/vision/internvl_encoder.cpp` (5), `src/compute/attention_mxfp4_prefill.cu` (4), `src/compute/gemm_grouped.cpp` (3), `src/compute/gemm_gemv_dtype.cu` (1) | 82 statement-level `cublas*`/`cublasLt*` calls discard `cublasStatus_t`, incl. GEMM launches (`internvl_encoder.cpp:37`, `qwen3vl_encoder.cpp:41`) whose failure leaves the output stale. `tools/check_cuda_discards.py` matches only `cuda<Upper>(`, so cuBLAS sits outside the #2211 gate; `CUBLASLT_CHECK` is a file-local macro (`gemm.cpp:33`, 10 uses) next to the `IMP_CUDA_CHECK_*` family (`logging.h:77-118`) | L (success path unchanged) | M | `IMP_CUBLAS_CHECK_*` beside `cuda_errors.h`; extend `check_cuda_discards.py` to `cublas\w*\(` with a ratchet starting at 82 |
| H-017 | Inconsistent error handling | `tools/imp-server/handlers_admin.cpp:39,94`; `tools/imp-bench/bench_e2e.cpp:114,176-179,191,197` | Discarded CUDA runtime results in `tools/` (the #2211 gate scans `src/` only); admin path reports VRAM deltas from a possibly failed `cudaMemGetInfo` (values pre-zeroed) | L | S | Check or `(void)`-annotate; widen the gate to `tools/imp-server` |
| H-018 | Ownership (rule of five) | `src/memory/kv_cache.cu:306`, `src/memory/ssm_state.cpp:13`, `src/compute/moe_routing.cu:484`, `src/memory/recurrent_snapshot_store.cpp:7`, `src/model/ngram_table.cpp:85`, `src/compute/sampling_topk_topp.cu:654`, `tools/imp-quantize/awq.cu:56` | 7 types whose destructor frees (cudaFree / close / pool) with an implicit copy ctor (no `= delete`, no non-copyable member). `MoeRoutingBuffers` is held BY VALUE in `exec/moe_workspace.h:30` and `compute/mtp_forward.h:139`; `RecurrentSnapshotStore::lru_map_` stores iterators into its own `lru_`, so a copy dangles. A copy today = double free | L (compile-time only; a compile error exposes a real copy) | S | `T(const T&) = delete; T& operator=(const T&) = delete;` on all 7 |
| H-019 | Ownership | `tools/imp-bench/bench_e2e.cpp:40,51` | `new uint16_t[n]` handed to a non-owning `Tensor`; 0 `delete[]` in the file; 14 call sites per run | L | S | `std::vector<uint16_t>` storage kept alive for the run |
| H-020 | Header hygiene (fan-in) | `src/exec/executor.h:13-14` | Includes `compute/json_constrain.h` and `compute/schema_constrain.h`; 0 uses of `JsonConstrainer`, `SchemaConstrainer`, `JsonGrammar`, `SchemaFrame`, `JsonState`, `SchemaPhase` in executor.h. Transitive fan-in 128 / 125 TUs, almost all via executor.h (117 TUs); 20 / 27 commits since 2026-04-15. Not covered by SETTLED's refuted "trim includes" (that was about executor.h's includers) | L (compile-only) | S | Move the two includes to the TUs that use them; let the compiler name them (build step) |
| H-021 | Header hygiene (fan-in) | `src/model/model.h:3-9` | `tokenizer.h` reaches 258 TUs (21 commits/6 mo) through model.h, which uses `Tokenizer` only via `unique_ptr`/pointer (`:67,68,133`); `hf_config_loader.h` (209 TUs) is included only for `GenerationConfig` held by value (`:115`) | L | M | Forward-declare `Tokenizer`, move `set_tokenizer` out of line (unique_ptr assignment needs the complete type); split `GenerationConfig` into its own header; build step names the real includers |
| H-022 | Header hygiene (build cost re-measure) | `src/exec/executor.h`, `src/runtime/config.h`, `src/runtime/engine.h` | fan-in x commits since 2026-04-15 over 825 TUs (src+tools+tests): config.h 92 x 170 = 15640, executor.h 117 x 132 = 15444, model_config.h 236 x 40 = 9440, engine.h 63 x 170 = 10710. SETTLED build-cost table (2026-08-03, 450 TUs) read 11050 / 4235 / 4123 / 5289; G-2 read executor.h 6320. TU sets differ, so not a like-for-like regression yet | L | S | Re-run SETTLED's reverse-BFS method on the same TU set before acting; H-020/H-021 are the cheap levers it would price |
| H-023 | Header hygiene / ODR | `src/exec/executor_gemv_helpers.h:12` | `static` (non-inline) `dispatch_gemv_residual` defined in a header included by 4 TUs (`executor_attention.cpp`, `executor_perplexity.cu`, `executor_ffn.cu`, `executor_forward.cu`): one copy per TU; clang `-Wunused-function` fires at `:12:13` in the tidy lane; comment names `executor_attention.cu` (file is `.cpp`) | L | S | `inline`; fix the comment |
| H-024 | Header hygiene / ODR | `src/model/jinja.cpp:250-1715` (26), `src/compute/json_schema.cpp:16`, `src/compute/gemm.cpp:242-275`, `src/model/weight_upload.cpp:166,312,403`, `src/model/safetensors_loader.cpp:101,134,322`, 46 host sites total | `misc-use-internal-linkage`: external-linkage types with generic names (`imp::jinja::Token`, `Lexer`, `Parser`, `Evaluator`, `*Node`) in .cpp files; a same-named class with inline members in another TU of the same namespace would link silently (ODR) | L | S | Anonymous namespace per file; `*_reset_static_cuda_state` (3 of the 46) stay external (macro-bound, SETTLED R-11) |
| H-025 | clang-tidy readability | `src/runtime/constraint_manager.cpp:128-363` | 13 `readability-ambiguous-smartptr-reset-call`: `unique_ptr` and pointee both have `reset()`; `.reset()` destroys the constrainer, `->reset()` rewinds it, one character apart | L | S | Rename the constrainer method (`rewind()`) or `= nullptr` for the pointer |
| H-026 | clang-tidy (uninitialized members) | `src/compute/sampling.h:111,237`, `src/runtime/engine.h:944`, `src/memory/weight_cache_file.cpp:29`, `src/memory/vram_allocator.h:67`, `src/model/weight_upload.cpp:403,2092`, `src/exec/pre_dequant_phase3_cutlass.cpp:113`, `src/compute/gbnf_grammar.h:111,116`, `src/model/tokenizer_pretok.cpp:221`, `src/vision/image_decode.cpp:72`, `src/exec/executor_attention_internal.h:254` | 14 host `cppcoreguidelines-pro-type-member-init`; `weight_cache_file.cpp:29` is the on-disk cache header POD (magic, version, ...) | L | S | Default member initializers; check the cache header is fully written before persisting |
| H-027 | Warnings (-Wshadow, -Wconversion) | `CMakeLists.txt:43`; host sites `src/exec/executor_attention.cpp:655,656,708`, `src/exec/executor_attention_qkv.cpp:257`, `src/exec/executor_forward_moe_legacy.cpp:282,329`, `src/runtime/engine_kv_cache_init.cpp:506`, `src/model/tokenizer.cpp:657`, `tools/imp-server/handlers_misc.cpp:650` | `imp_warnings` = `-Wall -Wextra -Wpedantic` only. Tidy lane with the flags: shadow 15 (5 host), shorten-64-to-32 11 (6 host), implicit-int-conversion 19 (all .cu), sign-compare 21 (all .cu), implicit-int-float 543 (473 host; top `executor_workspace_buffers.cpp` 54, `weight_upload.cpp` 38, `mem_account.cpp` 32), sign-conversion 4694 (1850 host; top `tokenizer.cpp` 201) | L | S (shadow + shorten) / L (sign-conversion) | Fix the 11 host shadow/shorten sites, add `-Wshadow` to `imp_warnings`; leave `-Wsign-conversion` off |
| H-028 | clang-tidy modernize-* (not enabled in `.clang-tidy`) | top host files: `src/model/tokenizer.cpp` (247 all checks), `src/model/weight_upload.cpp` (235), `src/model/jinja.cpp` (235) | Host counts: use-designated-initializers 649, avoid-c-arrays 342, use-std-print 340, use-nodiscard 280, avoid-c-style-cast 245, readability-container-contains 230, use-auto 165, return-braced-init-list 160, use-scoped-lock 151, use-integer-sign-comparison 150, use-starts-ends-with 98, use-ranges 84, use-emplace 42, raw-string-literal 31 | L | M | Enable only auto-fixable, behaviour-neutral ones (container-contains, starts-ends-with, use-emplace) with `--fix` per file; rest stays off per the `.clang-tidy` calibration rule (> 20 findings or judgment -> off) |
| H-029 | Magic numbers (hardware) | `src/exec/executor_gemm_dispatch.cpp:697` | sm_120 opt-in smem `101376` as a local `constexpr kSmemLimit` in a host dispatch predicate; same value re-declared kernel-side `gemm_q4k.cu:380` (`kDenseSmemLimit`), `gemm_q8_0_dense.cu:19` (`kSmemLimit`), and as bare literals in `fmha_sm120_tile_select.h:53-56`, `fmha_fp8_tile_select.h:70-71`; no shared constant (`rg 'kSm120\w*Smem'` = 0) | L | S | `inline constexpr size_t kSm120SmemOptinBytes = 101376;` in `src/core`, used by all 5 sites |
| H-030 | Magic numbers / config drift | `src/runtime/engine_kv_cache_init.cpp:146`, `src/runtime/engine_scheduler.cpp:479,488` | Prefill-chunk default 2048 encoded twice. `resolve_prefill_chunk_size_()` also honours `runtime.prefill_chunk_size` from imp.conf/`--set` (`:479`); the SWA burst cap reads only `config_.prefill_chunk_size` (CLI/C API), so `--set runtime.prefill_chunk_size=4096` with `kv_cache.swa_sizing` active sizes the live window for 2048-token chunks. Candidate SWA undersize, not reproduced (needs GPU) | M (KV sizing) | S | Use `resolve_prefill_chunk_size_()` if its inputs are resolved at that point, else one named default; then check-degeneration on an SWA model |
| H-031 | Magic numbers (duplicated literals) | `src/runtime/engine_init_resolver.cpp:105,396,850-851`; `src/runtime/constraint_manager.cpp:78,230,290,325` | `CUBLAS_WORKSPACE_CONFIG` value `":4096:8"` 4x; constrained-decoding preamble budget `8192` 4x | L | S | One named constant each |
| H-032 | Magic numbers (host mirror of kernel geometry) | `src/exec/workspace_sizes.cpp:265-276` | `exec_apa_workspace_bytes` re-derives APA kernel geometry from literals (64-row KV tiles, 192-row q block, 12 warps, `72*d`, `136*d+16` at `:260`); the anti-drift check `tests/test_attention_ra2.cu:319` sits in `test-attention`, a GPU-lane target (CI has no GPU) | L | S | Share the constants with `apa.cuh` via a host-visible header, or move the equality test to a CPU-lane target |
| H-033 | Stale comment (build) | `CMakeLists.txt:407-408` | "TurboQuant ... resolved by using sm_120f instead": the build emits no sm_120f SASS; `cuobjdump imp-server` = 131 `sm_120a` ELF (130 TUs + 1 device link) + 130 `compute_120f` PTX, nothing else | L | S | Delete or restate the comment |
| H-034 | Build/CMake (dead option) | `CMakeLists.txt:139` | `IMP_USE_CUTLASS=OFF` cannot link: it drops 12 CUTLASS TUs (`:374-386`), while `src/` has 0 `IMP_USE_CUTLASS` guards and 10 unconditional TUs/headers include their headers (`exec/executor.h:18-19`, `executor_forward_moe.cu`, `gemm_kernel_cutlass_nvfp4.cpp`, ...). Description says "requires sm_90+" (pre-sm_120 wording). Static evidence; OFF was not built | L | S | Remove the option, make CUTLASS unconditional |
| H-035 | Stale TODO/FIXME/HACK/XXX | n/a | 0 markers in `src tools include cmake scripts CMakeLists.txt Dockerfile` (code files); 1 hit `tools/eval/niah/data/filler.txt:77` is the roman numeral "XXX." in eval text data | L | S | None; classification below |

## clang-tidy counts (unique file:line:check, full lane, scratch config = repo `.clang-tidy` + `modernize-*` + -Wall -Wextra -Wshadow -Wconversion)

| check | all | .cu/.cuh | host .cpp/.h | top files |
|---|---:|---:|---:|---|
| clang-diagnostic-sign-conversion | 4694 | 2844 | 1850 | tokenizer.cpp 201, mtp_forward.cu 115, vision_encoder.cu 89 |
| readability-math-missing-parentheses | 4129 | 3531 | 598 | attention_fmha_mxfp4_sm120.cu 160, attention_fmha_sm120.cu 125 |
| modernize-avoid-c-arrays | 1206 | 864 | 342 | mtp_forward.cu 51, ssm_scan_ssd.cu 38 |
| modernize-avoid-c-style-cast | 1192 | 947 | 245 | attention_fmha_mxfp4_sm120.cu 95, gemm_moe_gemv.cu 77 |
| modernize-use-designated-initializers | 712 | 63 | 649 | |
| modernize-use-auto | 593 | 428 | 165 | |
| clang-diagnostic-implicit-int-float-conversion | 543 | 70 | 473 | executor_workspace_buffers.cpp 54, weight_upload.cpp 38 |
| misc-use-internal-linkage | 472 | 426 | 46 | jinja.cpp 26 (host); device helpers in .cu |
| modernize-use-std-print | 343 | 3 | 340 | |
| modernize-use-nodiscard | 280 | 0 | 280 | chat_template.h 27, jinja.h 20, host_pinned.h 16 |
| readability-container-contains | 246 | 16 | 230 | |
| modernize-use-integer-sign-comparison | 202 | 52 | 150 | |
| modernize-use-scoped-lock | 171 | 20 | 151 | |
| clang-diagnostic-unused-variable | 73 | 73 | 0 | attention_fmha_mxfp4_sm120.cu 27 (host pass of device code) |
| clang-diagnostic-unused-parameter | 59 | 59 | 0 | |
| cppcoreguidelines-pro-type-member-init | 57 | 43 | 14 | H-026 |
| clang-diagnostic-shadow | 15 | 10 | 5 | H-027 |
| clang-diagnostic-shorten-64-to-32 | 11 | 5 | 6 | H-027 |
| readability-ambiguous-smartptr-reset-call | 13 | 0 | 13 | H-025 |
| performance-inefficient-string-concatenation | 10 | 0 | 10 | json_schema.cpp 2, handlers_rerank.cpp 2 |
| bugprone-* / clang-analyzer-* (WarningsAsErrors) | 0 | 0 | 0 | pins in `tools/tidy_baseline.toml` hold |

## Stale-marker classification

| hit | classification |
|---|---|
| `tools/eval/niah/data/filler.txt:77` "XXX." | not a marker (eval text data); nothing to do |

No TODO/FIXME/HACK/XXX in code. Nearest equivalents are stale comments, filed as H-002 (MEMORY.md), H-004, H-009 (`CMakeLists.txt:387`), H-023, H-033.

## Checked, not a new finding

- Arch emission: only `sm_120a` SASS + `compute_120f` PTX (`CMakeLists.txt:81-90`, `CMAKE_CUDA_ARCHITECTURES OFF`; cuobjdump counts in H-033). `IMP_DISABLE_120F_FALLBACK=ON` in the `ci` preset (SETTLED H-5).
- Dockerfile `:88` `sed -march=native` is a documented deliberate guard, not dead.
- Raw `new`/`delete` in `src/`: C-API handles (`imp_api.cpp:221,248,424,449`, `imp_api_suspend.cpp:23,57`), custom deleters (`engine.cpp:78-79`, `recurrent_snapshot_store.cpp:132-146`): all owned by design.
- `shared_ptr`: 93 of 185 are `shared_ptr<Request>` shared between server and engine; rest (snapshot entries, jinja objects, compiled NFA/split sequences) are genuinely shared. No unique_ptr-would-do case found.
- Raw `cudaFree*` outside `src/memory`: 230 sites; tracked by I1 (`tools/alloc_allowlist.txt`, 69 files, two-way ratchet, SETTLED S-27) and F-12 re-scope (SETTLED G). Not re-reported.
- Logging bypass: 0 `printf`/`fprintf(stderr)`/`std::cerr` in `src/` (all hits are `snprintf`).
- `catch(...)` outside `src/api`: 10 sites, all parse-with-fallback or forward-throw capture; none swallows an engine error.
- getenv: env reads in `src/` are in `config.cpp` (seeded list) or are standard paths (HOME, XDG_CACHE_HOME, HF_*); `engine_init_resolver.cpp:849` reads `CUBLAS_WORKSPACE_CONFIG`, which cuBLAS owns.
- Dead header-type sweep (types never named outside their header): 79 raw, all live after reading except `TensorView` (H-001).
- `tools/analysis/`: 9 of 107 files referenced nowhere; research scripts, out of host-code scope.

## Deferred to build step

- Per-TU compile time. Proxy (own LOC + transitive project-header LOC, system/CUDA/CUTLASS headers excluded, so CUTLASS .cu TUs are under-ranked):

| TU | proxy LOC | own LOC | project headers |
|---|---:|---:|---:|
| `tools/imp-server/handlers.cpp` | 12293 | 723 | 144 |
| `tools/imp-server/stream_driver.cpp` | 12268 | 518 | 141 |
| `tools/imp-server/handlers_misc.cpp` | 12262 | 798 | 142 |
| `tools/imp-server/handlers_chat_core.cpp` | 12231 | 780 | 143 |
| `src/runtime/engine.cpp` | 11530 | 771 | 144 |
| `src/runtime/engine_scheduler.cpp` | 11278 | 1387 | 128 |
| `src/runtime/engine_prefill.cpp` | 10695 | 647 | 128 |
| `src/api/imp_api.cpp` | 10545 | 770 | 124 |
| `src/runtime/engine_init_resolver.cpp` | 10468 | 683 | 123 |
| `src/runtime/engine_kv_cache_init.cpp` | 10345 | 774 | 122 |

- H-020 / H-021: removing includes needs a compile to name the real includers (SETTLED S-33 method).
- H-009: release build-time saving (ci.yml claims ~30-60 s).
- H-034: confirm `-DIMP_USE_CUTLASS=OFF` fails to link before removing the option.
- `-Wconversion` with nvcc host compiler (g++) may differ from the clang counts above; the counts here are clang host-pass.
- H-030: GPU repro of the SWA burst cap with `--set runtime.prefill_chunk_size=4096` on an SWA model.

## Out of scope seen (handed to the kernel scout / other owners)

- Dead device helpers: `fp8_bits_to_float` (`attention_paged_fp8_multitok.cu:19`), `cp_async_cg_16_local` (`gemm_grouped_nvfp4_smallM.cu:21`), `mbarrier_invalidate` (`gemm_grouped_nvfp4_smallM.cu:44`), `cvt_e4m3x4_to_f32x4` (`ptx92_utils.cuh:30`), `apa::pass2` (`apa/apa.cuh:313`), `ldcs_lane<Uint4x2>` specialization (`attention_paged_f16_multitok.cu:50`): 0-1 tree mentions, clang `-Wunused-function` on the first three.
- Same-name external-linkage `__device__` functions in several TUs under `-rdc` builds (device ODR risk): `ue8m0_to_float` (`gemm_cutlass_mxfp4_sm120.cu:411`, `quant/mxfp4_gemm.cu:21`, identical signature), `ue4m3_decode` (`attention_paged_nvfp4.cu:23`, `attention_paged_nvfp4_tc.cu:22`, identical), `fp4_byte_to_half2`, `block_reduce_max/sum`, `quantize_micro_block_nvfp4`, `mtp_logit_to_float` (`mtp_forward.cu:47`, `mtp_topw.cu:19`).
- Smem literals `96 * 1024` at 5 `cudaFuncSetAttribute` sites (`gdn.cu:702`, `gdn_scan_tc.cu:614,643`, `gdn_scan.cu:401,507`).
- 10 `.cu` misc-use-internal-linkage hits on device helpers (426 total in .cu).
- Not analysed per brief: MoE atomics nondeterminism, `/v1/messages` streaming, FA2 coverage gaps, PDL, syncs, arch guards.

## CUDA (C-)

Repo: kekzl/imp at `4aca9671` (branch `feat/webui-imp-redesign`, clean).
Binary: `ghcr.io/kekzl/imp:latest` v0.53.0, revision `cb591619`, `/usr/local/bin/imp-server` (device code statically linked: 131 sm_120a cubins, 130 compute_120f PTX images).
Tools: `cuobjdump` / `cu++filt` from `imp:toolchain` (CUDA 13.4, V13.4.92); no build, no GPU job.
Read first: `docs/audit/SETTLED.md` (A, B, C, D, E, G, H), `docs/internals/SM120.md`, `docs/internals/KERNELS.md`, `src/compute/CLAUDE.md`, skills codebase-audit, sm120-cuda-expert, code-graph (no launch edges: launch questions answered with `rg`, per its rule 4).
Per-kernel resource table: `/tmp/claude-1000/cleanup/PTXAS_BASELINE.md`.

## Counts

| metric | value |
|---|---:|
| kernel instantiations in release binary (STO_ENTRY, unique) | 1018 |
| distinct kernel base names | 466 |
| base names named directly at a launch site (`<<<`, `pdl::launch`, `cudaLaunchKernelEx`) | 423 |
| base names launched through a variable / macro / table / library (verified one level) | 43 |
| base names with no non-definition reference in `src/` `tools/` `include/` | 0 |
| `<<<` sites in src | 590 |
| `pdl::launch` sites in src | 106 |
| instantiations in sm_120a SASS but absent from compute_120f PTX | 26 |

## Findings

Risk = risk of the change breaking behavior or perf. Effort S < 1 h, M < 1 day, L more.

| id | category | file:line | evidence | risk | effort | proposed action |
|---|---|---|---|---|---|---|
| C-001 | unreferenced kernels | release binary vs `src/` | 466 base names from `cuobjdump -symbols` STO_ENTRY; each has a launch or address-taken use in src (423 at a launch site, 43 via pointer/macro/table: CUB 5, CUTLASS `device_kernel`, dequant table 14, gated-activation dispatch 9, FMHA `KFN` macro 3, `auto* k = ...` 11). One-level check only (SETTLED C: `ccg coverage` caveat) | - | - | Negative result: record in SETTLED C with this method. No removal candidate |
| C-002 | dead template instantiations | `src/compute/attention_fmha_mxfp4_sm120.cu`, `attention_mxfp4_prefill.cu` | `fmha_sm120_mxfp4_kernel` = 51 of 1018 instantiations (largest family), plus `mxfp4_attn_softmax_kernel`, `mxfp4_k_channel_mean_kernel`, `mxfp4_tile_mean_paged_kernel`; default-off research family | M | M | NOT new: SETTLED H A1-9 (owner's call, re-measure trigger recorded). Listed for size only; no action without owner decision |
| C-003 | dead template instantiations (duplicate SASS) | `src/compute/constrain_common.h:152` | `__global__ inline constrain_mask_allow_kernel` in a header: same mangled symbol in 2 cubins (1019 STO_ENTRY rows, 1018 unique); launched from `json_constrain.cu:708`, `schema_constrain.cu:367,417` | L | S | Define once in one `.cu`, declare in the header |
| C-004 | kernel variant explosion | `src/compute/layernorm.cu:35`, `:382`, `:426` | `rmsnorm_fp32_kernel` vs `rmsnorm_fp32_in_fp32_out_kernel`: 24/25 body lines, diff = signature + 1 line (`weight[i]` vs `__half2float(weight[i])`); `rmsnorm_fp32_in_fp16_out_kernel` vs `_in_fp32_out`: 11 diff lines (output convert) | M | S | One `template <typename TW, typename TO>`; accept only if `cuobjdump -sass` per instance is byte-identical to today's three kernels (bit-sensitive, all three `pdl::launch`ed) |
| C-005 | kernel variant explosion | `src/compute/kv_gather.cu` (`paged_kv_gather_nvfp4_to_fp16_kernel`, `paged_kv_gather_mxfp4_kv_to_fp16_kernel`) | 67/65 body lines, 14 diff lines, all in the scale decode (UE4M3 via `__nv_fp8_e4m3` vs UE8M0 `<< 23`) | L | S | Template on a scale-decode functor; SASS-identical gate. Not covered by SETTLED B (B names the paged DECODE kernels, not the gather kernels) |
| C-006 | kernel variant explosion | `src/compute/layernorm.cu:202` | `rmsnorm_residual_fp32_kernel` vs `rmsnorm_fp32_kernel`: 27/24 lines, 11 diff lines | M | S | Fold into the C-004 template only if SASS stays identical; else leave |
| C-007 | kernel variant explosion | `src/exec/executor_elementwise.cu` (`residual_add_rmsnorm_kernel`, `add_rmsnorm_inplace_kernel`) | 52/52 lines, 24 diff lines | M | M | Candidate only; decide after a SASS diff. Checked and NOT candidates (diff >= 40 % of body): swiglu/geglu/gpt_oss_glu vec4, `gemv_q8_0_kernel` vs `gemv_q6k_kernel`, MoE decode q8_0 vs q6k, `dequant_iq4_nl` vs `_xs`, `hadamard_64` vs `_128`, `gelu_fp32` vs `_fp16` |
| C-008 | dead arch guard | `src/compute/attention_paged.cu:1240`, `:1252`; `src/compute/attention_dispatch.cpp:19-35`; `src/compute/attention.h:22` | `static int sm_ver = get_device_sm_version(); ... else if (sm_ver >= 90 && !force_non_pipe)`: always true on sm_120. F16 sibling of SETTLED H A1-6 (fp8/int4 `sm_ver >= 90` selectors already deleted). `get_device_sm_version()` has no other caller (`rg`) | L | S | Drop the `sm_ver` term (non-pipe arm stays reachable via `attention.splitk_pipe`), then delete `get_device_sm_version` + its cache + INFO line |
| C-009 | dead arch guard | `src/compute/gemm_cutlass_mxfp8_sm120.cu:119-125`; `mmq_q8_imma.cu:233`, `mmq_q8_imma_q4k.cu:240`, `mmq_q8_imma_q51.cu:192`, `mmq_q8_imma_q6k.cu:257`; `mmq_q4k_imma_tile.cu:157` | `#if __CUDA_ARCH__ >= 890 ... #else` (dead else), `>= 800` x4, `>= 750` x1: always true for compute_120a and compute_120f | L | S | Remove guards; prove with SASS diff (expected identical) |
| C-010 | dead runtime capability check | `src/core/pdl.cpp:53`; `src/compute/gemm_grouped_nvfp4_smallM.cu:716-721`; `gemm_capture_fp16_sm120.cu:270-275`; `gemm_cutlass_grouped_3x.cu:96-101` | `major >= 9` and `cc >= 120` via `cudaGetDeviceProperties`; the fatbin carries only sm_120a SASS + compute_120f PTX, so it cannot load on a device where these fail | L | S | Delete or fold into one cached helper; keep the log line that names the device |
| C-011 | dead arch guard + wrong comment | `src/core/pdl_device.cuh:3`, `:17`, `:23` | Comment: device half is a "no-op ... for the compute_120f fallback". False: `__CUDA_ARCH__` = 1200 for compute_120f; the PTX dump has `griddepcontrol.wait` in 80 sm_120f entries. `>= 900` guard is dead | L | S | Fix the comment; guard to `defined(__CUDA_ARCH__)` |
| C-012 | arch list / fatbin gap | `CMakeLists.txt:627-634` (`-rdc=true` on `nvfp4_gemv_dense.cu`, `nvfp4_gemv_fused.cu`, `mmq_q8_imma_q4k/_q51/_q5k/_q6k.cu`), `:83-85` | 26 instantiations (18 bases: 14 dense NVFP4 decode GEMVs `gemv_nvfp4_kpar/_fp32/multirow/_fp32/residual/_mr/qkv_fused/_mr/gate_up_fused/_mr/swiglu_residual/_mr/geglu_residual/_mr`, 4 `mmq_imma_*_raw_kernel`) are in sm_120a SASS and ABSENT from the compute_120f PTX (992 PTX `.entry` vs 1018 SASS). `docs/internals/SM120.md` says GB203 runs the 120f PTX: on GB203 these kernels have no image | M | M | Owner decision: (a) default `IMP_DISABLE_120F_FALLBACK=ON` and correct SM120.md, or (b) drop rdc (instantiate in the launcher TU) so PTX is emitted, or (c) document the gap. Untestable here (no GB203, #1651) |
| C-013 | dead build option + stale comment | `CMakeLists.txt:139`, `:619`; `CMakeLists.txt:407-408` | `IMP_USE_CUTLASS` "(requires sm_90+)": the define it sets is read by no source (`rg IMP_USE_CUTLASS src` = 1 comment, `attention_mxfp4_prefill.h:14`); OFF removes CUTLASS sources that src calls unconditionally. Line 407-408: TurboQuant "resolved by using sm_120f", but no sm_120f SASS is built | L | S | Delete the option (CUTLASS is mandatory) or prove OFF builds; delete the stale comment |
| C-014 | missing post-launch check | `src/core/pdl_launch.cuh:37` | Fallback `func<<<grid, block, smem, stream>>>(args...)` has no check; 68 of 106 `pdl::launch` sites have no guard in the gate's window: `nvfp4_gemv_batched.cu` 20, `gemv_dp4a_traits.cuh` 11, `nvfp4_gemv_dense.cu` 10, `layernorm.cu` 7, `layernorm_rowblock.cu` 6, `nvfp4_gemv_moe.cu` 4, `nvfp4_gemv_fused.cu` 4, `rope.cu` 3, `executor_elementwise.cu` 3, `gemm_dp4a.cu` 2, 1 each in `gdn.cu`, `ffn_sparsity_mask.cu`, `activation.cu`, `nvfp4_gemv_gdn_input.cu`. Unregistered kernels (dp4a GEMV family, SETTLED H A1-3) always take this branch | L | S | `IMP_CUDA_CHECK_LAUNCH()` after the `<<<>>>` inside `pdl::launch` (one line covers all 68); the `cudaLaunchKernelEx` branch already logs |
| C-015 | launch-guard gate blind spot | `tools/check_launch_guards.py:128` | Gate scans `src/**/*.cu` only; 24 `<<<` live in `.cuh`. Unguarded by the gate's own `scan()`: `src/compute/apa/apa.cuh:216`, `:459` (`reset_stats_kernel`), `src/exec/executor_forward_moe_kernels.cuh:25` (`sanitize_fp16_kernel`), `src/core/pdl_launch.cuh:37` | L | S | `rglob("*.cuh")` too; add the 3 guards |
| C-016 | launch-guard gate blind spot | 23 macro-body launches: `attention_paged*.cu` 16, `attention_fmha_mxfp4_sm120.cu` 2, `attention_fmha_sm120.cu` 1, `attention_blackwell.cu` 1, `ssm.cu` 1, `ssm_scan_reg.cu` 1, `quant/dequant_gpu.cu` 1 | Gate counts these and does not decide them (by design, docstring) | L | S | Manual pass per macro call site (deferred, see below) |
| C-017 | inconsistent launch-config helpers | `src/compute/ra2/ra2.cuh:110`; `apa/apa.cuh:303`; `attention_fmha_sm120.cu:453`, `:973`, `:1981`; `src/core/pdl.cpp:18`; per-launch `pdl::enable_kernel` at `gdn.cu:366,371,414,474,532`, `layernorm_rowblock.cu:229-289`, `rope.cu:210,217`, `executor_elementwise.cu:337,389`, `executor_kv_write.cu:108,202`, `moe_routing.cu:459,562`, `ssm.cu:203,216` | `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` on every launch in some families, cached by a static in others (53 call sites); PDL registration once at init (`nvfp4_gemm.cu:279`, `nvfp4_gemv_batched.cu:231`) vs an `unordered_set::insert` on every launch elsewhere. The registry is a global, unsynchronized `std::unordered_set` mutated from launch sites | M | M | One init-time registration table (PDL + smem attribute); measure host cost first. Thread-safety: confirm only one host thread launches |
| C-018 | PDL consistency | `src/quant/nvfp4_gemv_batched.cu:226-241`; callers `src/exec/executor_gemm_dispatch.cpp:575`, `:600` | Comment: `gemv_nvfp4_multirow_mb_kernel` and `gemv_nvfp4_kpar_mb_fp16_kernel` are "test oracles only" and stay unregistered. They are production: `gemm_nvfp4_batched(_acc)` serves speculative verify chunks (GGUF overlay M <= 33, ST-NVFP4 M <= 4) with `speculative.verify_smallm=false` default (`src/core/config/speculative.h:45`). SASS: all 16 instantiations contain ACQBULK + PREEXIT, i.e. they wait/trigger but launch without the programmatic attribute. Disproves the anchor of SETTLED H A2-2 | M | S | Audit wait placement (contract `core/pdl_device.cuh`), register in `nvfp4_gemv_batched_pdl_register()`, A/B on an MTP/n-gram verify run + check-degeneration; update SETTLED A2-2 |
| C-019 | PDL consistency (negative) | all `pdl::enable` sites | Every instantiation of every registered family (50 names) has ACQBULK in SASS: 0 registered-without-wait (the race class in sm120-cuda-expert). 27 bases trigger-only (PREEXIT, no ACQBULK): harmless | - | - | Record as checked |
| C-020 | hidden sync + ordering race (hot, per decode step with logit_bias) | `src/compute/sampling_penalties.cu:604-613` | Fallback when the T2 arena gave no slots (`:567-573`) or `m > s_bias_buf_cap`: per entry a synchronous legacy-stream `cudaMemcpy` D2H of `d_logits[t]`, then an H2D write back. Logits come from the engine's `cudaStreamNonBlocking` stream (`src/runtime/engine.cpp:827`): legacy stream 0 does not order against it (#2275 class), so the read can see stale logits; 2 host round trips per entry | L | S | Always use `apply_logit_bias_kernel` (grow a device buffer when the arena declines), or `cudaMemcpyAsync(stream)` + `cudaStreamSynchronize(stream)` |
| C-021 | ordering race (runtime) | `src/exec/expert_cache.cu:684` (called from `:228-229` in `acquire_slot_`) | `cudaMemset(d_lookup_, 0xFF, ...)` on the legacy stream with no sync, inside a runtime path that holds `stream`; kernels on the non-blocking stream may read or write `d_lookup_` before it lands (#1652 / #2275 class) | L | S | `cudaMemsetAsync(d_lookup_, 0xFF, bytes, stream)` (thread the stream into `invalidate_host_state`) |
| C-022 | ordering race (fallback) | `src/compute/gemm.cpp:1176-1178` | cuBLASLt-FP8-unsupported fallback reads `aScale` / `bScale` with legacy `cudaMemcpy`; `aScale` can be an activation scale written on `stream` | L | S | `cudaMemcpyAsync(..., stream)` + `cudaStreamSynchronize(stream)` |
| C-023 | pageable transfer (hot, per decode step, constrained requests) | `src/compute/json_constrain.cu:700`; `schema_constrain.cu:364`, `:408`; `grammar_constrain.cu:147`; `regex_constrain.cu:279` | Vocab-sized allow mask (`std::vector<uint8_t> token_allow_`, `json_constrain.h:187`, `schema_constrain.h:253`) uploaded by `cudaMemcpyAsync` from pageable memory each step: staged copy, host-blocking. `executor_ple.cpp:110-111` cites 245 us for one pageable D2H on WSL2 | L | S | Pinned staging buffer per constrainer; measure step latency with a schema request first |
| C-024 | hidden sync + pageable transfer (hot, per MoE layer) | `src/exec/executor_forward_moe_batch.cu:134-139`, `:458-470`, `:516-520`, `:975-980`; `executor_forward_moe_cutlass.cpp:378-391`; `executor_forward_moe_legacy.cpp:81-86` | D2H of expert offsets / indices into a fresh `std::vector` + `cudaStreamSynchronize` per MoE layer per prefill chunk (and per step on the host-experts decode branch `:975-980`). Structural host dependency (SM120.md: "blocked - D2H expert routing"); the pageable landing zone is not | M | M | Establish which tiers are default (`src/exec/moe_prefill_decision.h`), then pinned landing buffers; keep the sync |
| C-025 | hidden sync (fallback) | `src/compute/moe_routing.cu:400-450`; freed `src/exec/executor_forward_moe.cu:779-783`; chosen at `executor_forward_moe_batch.cu:688-693` | Allocating `moe_topk_gating` overload: 4 `cudaMalloc` + 4 `cudaFree` (device-wide sync) per MoE layer when `moe_.routing_buffers.pool` is null (pool allocation failed) | L | S | Throw or take from the arena; reachable only after a failed pool allocation |
| C-026 | hidden sync (edge path) | `src/compute/weight_dispatch.cpp:279-292` | `gemv_dispatch` FP8: D2H of the weight scale + `cudaStreamSynchronize` per call; under capture the sync fails and the function returns with `y` unwritten. Executor reaches it only via `default:` (`executor_gemm_dispatch.cpp:547-548`) with `primary_tier` FP8 and a decode tier outside the switch; tests reach it directly (`tests/test_weight_dispatch.cu`) | L | S | Cache the host scale at upload (as `wcache_.fp8` `host_scale` already does); confirm reachability |
| C-027 | legacy-stream init / growth (not hot) | `src/memory/kv_cache.cu:462-471`; `kv_cache_scales.cu:96-105`; `kv_cache_minmax.cu:105-108`; `src/compute/gemm_grouped_nvfp4_smallM.cu:749-751`; `constrain_device_buffers.h:59-62`; `src/exec/executor_workspace_buffers.cpp:1200`, `:1710` | Legacy `cudaMemset` / `cudaMemcpy` + `cudaStreamSynchronize(0)`; each documented (#1652, #2275); runs on KV growth or init | L | S | Optional: `cudaMemsetAsync` on the consuming stream, drops the stream-0 sync |
| C-028 | hidden sync (negative) | `attention_cublas.cu:101`; `executor_perplexity.cu:586`; `pre_dequant_phase3_moe.cpp:181`; `executor_forward_moe_batch.cu:205`; `activation_calibrator.cu:171`; `executor_debug.h` (7); `executor_forward.cu:115,120,149,1106` | Every `cudaDeviceSynchronize` in scope is prewarm (SETTLED E, F-16), tool (PPL, calibration), load, or behind a debug / profile flag | - | - | None |
| C-029 | graph lifecycle | `src/exec/ffn_graph_cache.cpp:44-62`, `:126`; `ffn_graph_cache.h:18-21` | Prefill FFN graphs keyed (layer, row bucket), at most 8 buckets; eviction destroys every layer's exec of the victim and a later sighting re-captures and re-instantiates (no `cudaGraphExecUpdate`). Buckets: 16-row steps to 64, 64 to 1024, 256 to 4096 (~31), so mixed prompt lengths churn. `cuda_graph.h:265` prices one instantiate at 10-44 ms (decode graph) | M | M | Update the victim exec via `cudaGraphExecUpdate` instead of destroy + instantiate; measure instantiate cost and churn rate first |
| C-030 | stream / event lifecycle (negative) | `src/core/cuda_raii.h` | 0 raw `cudaStreamCreate*` / `cudaEventCreate*` in `src/` outside `cuda_raii.h` (RAII, SETTLED D S-24) | - | - | None |
| C-031 | local memory / spills | see `/tmp/claude-1000/cleanup/PTXAS_BASELINE.md` | 84 instantiations with STACK > 0 (all with STL in SASS), 11 at REG >= 240, LOCAL 0 everywhere. Already ratcheted by `tools/kernel_resource_baseline.txt` (87 rows = these 84 + 3 bench kernels not in the release binary): no drift. Largest frames: `topk_partial/_finalize(_rows)_kernel` 1056 B (sampling, per decode step), `smallM_kernel_v1` 528 / 272 / 144 B (MoE prefill), `rmsnorm_quantize_q8_1_kernel` 128 B (decode, PDL-registered), `paged_attention_decode_kernel_generic` / `paged_attention_gqa_kernel` 128 B, `fmha_sm120_fa2_kernel<64,256,...>` 255 regs / 112 B, `gdn_scan_chunkwise_kernel<128,128,64>` 255 / 80 B, `apa::apa_kernel` 168 / 56 B, `gemm_fp16_kernel<128,2>` 244 / 32 B, Q3_K dp4a GEMV family (DPQTag 6) 16 B x 19 | M | M | Per kernel: split spill vs indexed local array with `-Xptxas -v` (deferred). Decode-path candidates first: topk 1056 B, `rmsnorm_quantize_q8_1` 128 B. Any `__launch_bounds__` change needs `make verify-ab` (src/compute/CLAUDE.md) |
| C-032 | magic number (hardware) | `src/exec/executor_gemm_dispatch.cpp:697`; `src/compute/gemm_q4k.cu:380`; `src/compute/gemm_q8_0_dense.cu:19` | 101376 (sm_120 opt-in smem per block) as three independent literals; also in `static_assert`s `fmha_sm120_tile_select.h:53-56`, `fmha_fp8_tile_select.h:70-71`; 8 other sites query `cudaDevAttrMaxSharedMemoryPerBlockOptin` | L | S | One named constant in `src/core/`, asserts and the three literals use it |
| C-033 | magic number (hardware) | `src/compute/gemv_fp8_rowscale_fp32.cu:249`; fallbacks `gemm_cutlass_sm120.cu:847`, `gdn_scan_chunkpar.cu:472`; `src/quant/mxfp4_gemm.cu:26-34` | `constexpr int kMinBlocks = 2 * 170` hardcodes the SM count; 12 independent SM-count queries; `mxfp4_gemm.cu` uses `cudaGetDeviceProperties` and retries every call on failure | L | S | One cached `device_sm_count()` in `src/core/`; `kMinBlocks` from it |
| C-034 | magic number | `src/compute/gemv_dp4a_traits.cuh:1153`, `:1167` | `48 * 1024` default dynamic-smem limit literal, twice | L | S | Named constant |
| C-035 | compute-sanitizer | n/a | Does not run on this WSL2 host (`src/compute/CLAUDE.md`; SETTLED H F2-7, `docs/LIMITATIONS.md`) | - | - | BLOCKED, not attempted |

### Per category

| category | ids | count |
|---|---|---:|
| unreferenced kernels / dead instantiations | C-001 (negative), C-002 (recorded), C-003 | 3 |
| variant explosion | C-004..C-007 | 4 |
| dead arch guards / arch lists / build options | C-008..C-013 | 6 |
| launch checks / launch-config helpers | C-014..C-017 | 4 |
| PDL | C-018, C-019 (negative) | 2 |
| hidden sync / ordering | C-020..C-028 | 9 |
| stream / event / graph lifecycle | C-029, C-030 (negative) | 2 |
| spills / local memory | C-031 | 1 |
| magic numbers | C-032..C-034 | 3 |
| compute-sanitizer | C-035 (BLOCKED) | 1 |

Risk of the change (actionable rows): L 20, M 10, H 0; 5 rows are negative / blocked records (C-001, C-019, C-028, C-030, C-035).

## Legacy causal_softmax + cuBLAS attention

No removal proposed; no call-site table built. SETTLED A S-8: the materialised cuBLAS prefill is a deliberately retained tier (Gemma-4 hd=512 global layers, gated at `src/exec/executor_attention_prefill.cpp`).

## Deferred to build step

| item | what is needed | rows |
|---|---|---|
| spill vs local-array split | `-Xptxas -v` per TU (spill stores / loads per instance) on the 84 STACK > 0 instantiations | C-031 |
| SASS identity for merges | build each candidate, `cuobjdump -sass` diff per instance; accept only byte-identical | C-004..C-007, C-009, C-011 |
| per-instantiation reachability | which of `fmha_sm120_fa2_kernel` (22), `fmha_sm120_kernel` (14), `gdn_scan_fused_kernel` (10), `fp8_head_mma_kernel` (10), `gemm_qk_dp4a_dense_kernel` (9), `ssm_scan_reg_kernel` (8), `paged_attention_*` multitok (11 + 11) instances ever launch: `dispatch_record` trace over a model battery (GPU); not provable statically | C-001 |
| host cost of per-launch setup | nsys host timeline of `cudaFuncSetAttribute` + `pdl::enable` per decode step | C-017 |
| transfer latency | per-step latency of the constrained-decode mask upload and the MoE offset readbacks (pageable vs pinned) | C-023, C-024 |
| macro-body launches | read the 23 macro call sites for a post-switch guard | C-016 |
| compute_120f gap | rebuild without `-rdc` on the 6 TUs (or with the fallback off) and re-run `cuobjdump -ptx` | C-012 |
| PDL on batched verify | wait-placement audit + MTP verify A/B + check-degeneration | C-018 |
| graph churn | count FFN-graph evictions and instantiate time on a mixed-length prefill run | C-029 |

## Out of scope seen

- `src/runtime/`: `cudaDeviceSynchronize` at `engine_scheduler.cpp:551`, `engine_workspace_warmup.cpp:563,572,649,672`, `engine.cpp:84`, `engine_kv_swap.cpp:80`, `cuda_graph.cu:1314`; `cudaGraphInstantiate` per spec-capture bucket `engine_spec_capture.cpp:287`, `:352`; cc check `engine_init_resolver.cpp:658`. Runtime scout's area.
- `src/api/imp_api.cpp:980`, `imp_api_suspend.cpp:70`, `src/memory/weight_cache_file.cpp:169`, `weight_snapshot.cpp:291`, `mem_account.cpp:431`: device syncs at reset / suspend / snapshot / accounting.
- MoE atomics nondeterminism: not analyzed (`moe_fused_permute_deterministic_kernel` exists).
- `/v1/messages` streaming path: not touched.
- FA2 coverage gaps: not analyzed.
