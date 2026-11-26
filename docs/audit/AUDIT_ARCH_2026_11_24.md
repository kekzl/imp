# imp - architecture audit - 2026-11-24

Read-only audit of `imp` at `41225f0b` (main after #3279). Three axes, one reader each, ledger
first ([`SETTLED.md`](SETTLED.md), [`DEBT_LEDGER_2026_08_21.md`](DEBT_LEDGER_2026_08_21.md),
[`AUDIT_ARCH_2026_07_29.md`](AUDIT_ARCH_2026_07_29.md), the backlog ledger in
[`../ROADMAP.md`](../ROADMAP.md)). No build, no GPU job: every number is DERIVED from source or
from `git log` / `gh run list` on this date. The memory axis (invariants I1-I7) is not in here:
[`../internals/MEMORY.md`](../internals/MEMORY.md).

Status vocabulary:

| status | meaning |
|---|---|
| CONFIRMED | the audit reader read both sides and the lead re-read the sites (C-1, C-3, D-1 only) |
| CONFIRMED-BY-READER | the audit reader read the definition and the callers; not re-read by the lead |
| SUSPECTED | mechanism read, effect or reachability not shown |
| SETTLED-SIZING | already settled in a ledger; only the numbers are new |
| FIXED (#n) | fixed on main by that PR after the audit date; claim and evidence stay as read at `41225f0b`, the status the audit gave follows as "was" |

Backlog rows: B-98 .. B-128 in [`../ROADMAP.md`](../ROADMAP.md), one per actionable finding; B-129 .. B-134 added 2026-11-25 for C-9, C-10, C-13, C-14, D-11, X-10. D-8, X-3 and X-11 are SETTLED-SIZING and have none.

## 1. What to fix first

| rank | ID | reason |
|---|---|---|
| 1 | C-1 | silent cross-request state corruption on every hybrid (Qwen3.5 / 3.6 / 3.8): a cancel of a queued request zeroes another live sequence's GDN / SSM state |
| 2 | C-2 | use-after-free window at model swap: the tokenizer is freed while a handler holds a raw pointer; `server.model_swap` defaults to true |
| 3 | X-1 | 15 model env vars are wired into no target: port e2e suites skip green everywhere, Inkling alone is 17 tests in 3 files |
| 4 | D-1 | a void GEMM wrapper leaves its output unwritten behind one log line: 37 sites by pattern in 19 files, the class DEBT_LEDGER row 8 closed |
| 5 | D-4 + C-6 | Inkling is a private runtime (9378 LOC, 44 files) with no capability table: the reason #3270 broke it with `verify-fast` green |

## 2. Control plane (`src/runtime/`, `src/api/`, `tools/imp-server/`, `src/core/config/`)

| ID | claim | evidence | status | blast radius | first slice |
|---|---|---|---|---|---|
| C-1 | Cancelling a request that holds no recurrent slot zeroes another live sequence's GDN / SSM state | `Engine::reset_ssm_state` (`src/runtime/engine.cpp`) falls back to slot `seq_id % cap` when `recurrent_slot_of_` has no entry and resets it; the worker loop in `tools/imp-server/batching_engine.cpp` calls it for every cancelled active request; requests enter `active_requests_` before the scheduler admits them (slots are taken at prefill, `src/runtime/engine_sampling_stop.cpp`); `ssm_state.cpp` memsets the slot | FIXED (#3284), was CONFIRMED | every hybrid: a client disconnect while queued silently corrupts another client's stream | in `reset_ssm_state`, return when `recurrent_slot_of_` has no entry; the same modulo fallback in `engine_sampling_stop.cpp` throws |
| C-2 | Model swap and `/admin/suspend` free the tokenizer while other handlers still hold a raw pointer; submit never checks model identity | snapshot `ctx.snap.tok = state.tok` under `state.mtx`, then `tok->encode` unlocked (`handlers_chat_core.cpp`); swap frees the model (`handlers.cpp`); `pause()` waits only for worker-active requests (`batching_engine.cpp`); submit checks only `is_running()` (`handlers_chat.cpp`); `Model::tokenizer_` is a `unique_ptr` (`src/model/model.h`). Threads: handler A before submit or after finish, handler B in swap | FIXED (#3291), was CONFIRMED-BY-READER (race window) | use-after-free, or old-vocab tokens submitted to the new model | swap waits for `state.inflight` to drop to the swapper itself (idle unload already checks it, `handlers_admin.cpp`), plus a load epoch compared at submit |
| C-3 | Channel-marker ids survive a swap to a non-channel model | the only writer of `state.channel_open_id` / `channel_close_id` / `channel_newline_id` sits inside `if (co >= 0 && cc >= 0)` in `load_model_into_state` (`handlers.cpp`), no reset; read by `stream_driver.cpp` and `handlers_completions.cpp` | FIXED (#3284), was CONFIRMED | after Gemma-4 then any other model, the new model's token with that id is filtered as a channel header | reset the three ids to -1 before the detect; longer term one per-model struct replaced wholesale in `load_model_into_state` |
| C-4 | Request teardown exists as six hand-synced lists | `finish_request_release_`, `cancel_sequence_` (`engine.cpp`), server cancel and server fault (`batching_engine.cpp`), two C API sites (`imp_api.cpp`). The last four skip `spec_suffix_idx_.erase`: an entry created in `engine_spec_ngram.cpp` leaks per disconnected request (`speculative.ngram` and `.suffix` default true). 31 writes of CANCELLED (`rg -c 'status = (imp::)?RequestStatus::CANCELLED' src/runtime src/api tools/imp-server`) against 9 `cancel_sequence_` call sites | FIXED (#3295), was CONFIRMED-BY-READER | host memory grows with every disconnect; each new per-request table must be added in six places | public `Engine::cancel_request(id)` wrapping `cancel_sequence_`, used by server and C API |
| C-5 | Server cancel frees KV and slot while a device-side lane still runs for that request; one of three lanes is drained first | the worker loop drains `bd_pipe_` only; `step_async_graph_resume` (`engine_scheduler.cpp`) has no status check, keeps appending tokens, and `finish_request` overwrites CANCELLED with FINISHED; `cpipe_` tears down one step later (`engine_graph_decode.cpp`) | FIXED (#3295), was CONFIRMED-BY-READER structure, effect SUSPECTED | safe today only because `step_impl_` skips scheduling during a burst; a lane that interleaves scheduling reuses freed blocks | status check at the top of `step_async_graph_resume`, covered by C-4's entry point; the comment "polled between bursts" in `engine_scheduler.cpp` is stale |
| C-6 | No capability table: features hooked into the executor layer loop are silent no-ops on a model with its own forward | Inkling replaces the loop (`executor_forward.cu`); `rg 'lora\|cvec\|control_vector\|lens\|sae_\|calib_' src/exec/inkling_forward*.cu src/exec/executor_inkling.cu src/exec/inkling_forward.h` = 0 hits; `logit_lens_supported()`, `phrase_ban_supported()`, `cfg_supported()` say yes and the server publishes them (`handlers.cpp`) | FIXED (#3301), was CONFIRMED-BY-READER absence, API effect SUSPECTED | lens, SAE, LoRA and control-vector requests on Inkling answer 200 without the feature; phrase-ban rewind does not reach Inkling's own K/V and conv state | one `EngineCapabilities` struct derived at init, read by the `*_supported()` calls and the server |
| C-7 | Graph invalidation is by convention across at least eight families | 9 capture sites in 8 files (`rg -n 'cudaStreamBeginCapture' src`, minus the string in `ffn_graph_cache.cpp`); the triple `invalidate_graphs(); prefill_graphs_.invalidate_all();` + pool invalidate is pasted in `engine.cpp` (`lora_set`) and `engine_control_vector.cpp`; `invalidate_graphs()` skips the decode pool by design and never reaches `mtp_head_graphs_`, the dspark graph or the Inkling graphs; the workspace generation is consumed only by spec capture and the FFN cache | FIXED (#3289), was CONFIRMED-BY-READER | the next forward-topology switch has to find every family by hand | `Engine::invalidate_forward_graphs(reason)` owning the full list; the counted form is I3's G2 (`tools/check_graph_stability.py`, 6 of 8 captures read no generation) |
| C-8 | Decode-lane eligibility is a hand-kept request-field matrix, and `logprobs` doubles as "force eager" | five gate sites (`rg -n 'host_logit_shaping\(\)' src/runtime`): two in `engine_scheduler.cpp`, `engine_decode_pipeline.cpp`, `engine_spec_ngram.cpp`, `engine_spec_batch_verify.cpp`; lens and SAE set `req->logprobs = true` to stay off fast lanes (`engine.cpp`). `Request` has about 132 fields, `Engine` about 210 members (regex counts) | FIXED 2026-11-26, was CONFIRMED-BY-READER | a new sampling or readout feature missing from one predicate is silently skipped inside a captured lane | a per-request `LaneNeeds` bitset computed once in `add_request`, tested by each lane |
| C-9 | `EngineConfig` and `RuntimeConfig` overlap, bridged by sentinel-equals-default, and config doubles as runtime state | 11 of 35 `EngineConfig` fields have a same-named `RuntimeConfig` key (name match); bridge in `Engine::init` compares `== 25`, `== 1.0f`; mutated while serving at 4 sites in `engine_scheduler.cpp`; Inkling rewrites operator config (`engine_inkling.cpp`) | PARTIAL 2026-11-26 (B-129: all 8 bridged fields are `std::optional`, serving flags in `EngineRuntimeState`; Inkling's config rewrites open), was CONFIRMED-BY-READER | an embedder passing 25 or 1.0 explicitly loses to `imp.conf`; logs of "the config" show mutated values | `std::optional` for the bridged fields; a separate `EngineRuntimeState` for the mutable flags |
| C-10 | Inkling limits bypass the graph-demotion answer site and carry a dead branch | `engine_inkling.cpp` writes `use_cuda_graphs` directly, so `graph_demotion_` stays `None`; its `if (scheduler_)` is always false there (called before the scheduler is created in `Engine::init`) | FIXED 2026-11-26, was CONFIRMED-BY-READER | the resolved-dispatch summary shows graphs off with no reason; the next arch with its own forward copies the pattern | add a demotion reason, call `demote_graphs_` |
| C-11 | Results discarded where failure leaves a half-initialised object | `(void)stream_.create(...)` in `Engine::init`; `(void)executor_->resize_workspace` at 6 sites (`rg -n '\(void\)executor_->resize_workspace' src/runtime`), where failure nulls the shared workspace (`executor_workspace_config.cpp`, documented unreachable today) | FIXED 2026-11-26 (B-106: resize throws, stream create and `allocate_workspaces` fail `init`), was CONFIRMED-BY-READER | a failed stream create serves on the legacy default stream; the workspace branch goes live the day something exceeds the init-time maximum | fail `init` on stream create; throw on resize failure |
| C-12 | Eight "log once per process" latches are static, not per engine | 2 in `engine_scheduler.cpp`, `engine_decode_pipeline.cpp`, `engine_prefill.cpp`, `engine_spec_batch_verify.cpp`, `engine_spec_sampled.cpp`, `engine_spec_mtp.cpp`, `dspark_drafter.cu` | PARTIAL 2026-11-25 (B-107): 5 of 8 are `LogOnce` latches (`src/core/log_once.h`) that `imp_context_create` and `~Engine` re-arm; 3 left in `engine_scheduler.cpp` (2) and `engine_prefill.cpp` (1), listed in `tools/static_reset_allowlist.txt`; was CONFIRMED-BY-READER | after a swap the new model never logs why speculation, pipeline or prefill graph is off | move to Engine members (with D-10) |
| C-13 | `/metrics` reads engine state the worker mutates under no common lock | handler thread: `metrics_memory.cpp` (`num_cached_blocks()` is `cached_blocks_lru_.size()`, `spec_stats()` a plain struct) under `state.mtx`; the worker never takes `state.mtx` | FIXED 2026-11-25 (worker publishes an `EngineMetricsSnapshot` per step and idle tick; the scrape reads it under a model lease; the handler also held no `state.mtx` and no lease, so a swap could free the engine mid-scrape), was CONFIRMED-BY-READER, benign so far | torn counters today; a crash once a metric walks a container | publish a snapshot struct from the worker per step |
| C-14 | Nothing tests C-1 to C-6 | disconnect tests assert only the cancel counter (`tests/api/test_lifecycle.py`); `rg -il inkling tests` shows kernel, quality and draft tests only; no channel-id reference in `tests` | FIXED (#3309), was CONFIRMED-BY-READER | these paths need a GPU, a hybrid or two models: nothing catches a regression | CPU test of `reset_ssm_state` with a slotless id; an API test that swaps models mid-request |

## 3. Data plane (`src/exec/`, `src/compute/`, `src/quant/`, `src/model/`)

| ID | claim | evidence | status | blast radius | first slice |
|---|---|---|---|---|---|
| D-1 | A void GEMM / launch wrapper logs ERROR and returns with the output tensor unwritten; the class DEBT_LEDGER row 8 closed has recurred | three exits of `gemm_dispatch_uncached_fallback` (`src/exec/gemm_dispatch_uncached.cpp`): "beta but no FP16 path", "dropped weight at final fallback" (WARN once), "no route for a ... weight" (blame 2026-10-23, after the closure). 37 sites in 19 files in exec + compute + quant via `rg -U 'IMP_LOG_ERROR\([^;]*;\s*\n(\s*//[^\n]*\n)?\s*return;'`, among them 5 in `src/compute/weight_dispatch.cpp` and one in `src/exec/executor_gemm_dispatch.cpp` | FIXED (#3288) for 31 exits and (B-108 rest) 6 more; FIXED 2026-11-26 for the 19 designed fallbacks (one WARN per model load + `imp_fallback_total{site}`, a reason per allowlist row); open for the 5 Inkling shape guards; was CONFIRMED (3 sites read, 34 by pattern only) | a new format with no route serves stale device memory behind one log line; under graph capture the node is simply absent | make the 3 `gemm_dispatch_uncached` exits throw like the no-route arms of `weight_dispatch.cpp`; add a static gate on the pattern |
| D-2 | The dense GEMM route ladder (about 20 exits over M, beta, qtypes, tier, cache presence, scratch size, 3 policy flags) has no route record, no pure decision model and no route-level test | `GraphExecutor::gemm_via_handle_` ladder in `src/exec/executor_gemm_dispatch.cpp`; `rg -c dispatch_record` there = 0; `src/compute/dispatch_record.h` records attention and MoE prefill only; siblings have models (`src/exec/moe_prefill_decision.h`, `src/compute/attention_dispatch_decision.h`); `rg -c gemm_via_handle tests` = 2 in 1 file | CONFIRMED-BY-READER | a port landing on a slower or different-numerics rung is invisible; SETTLED F-3's "one half of a symmetric problem" applies to this third half | a `DensePath` enum plus `dispatch_record::set_dense(kind, path)` at each `return` |
| D-3 | The "one-tier-truth" StoragePlan is not the truth; tiers are re-inferred from which pointer-keyed map holds the source, then re-decided at runtime and again at call sites | plan read only in `pre_dequant_phase1_fp16_cache.cpp` and `pre_dequant_phase3_cutlass.cpp`; `storage_planner.h` calls the gap "informational"; inference in `pre_dequant_internal.h`, written in `pre_dequant_phase4_tensor_registry.cpp`; 48 runtime map lookups in 13 files (`rg 'wcache_?(\.\|->)(fp16\|fp8\|nvfp4\|...)\.(find\|contains\|count\|at)\(' -g '!pre_dequant*'`); call-site pre-routing in `executor_ffn.cu`, `executor_attention_qkv.cpp` | PARTIAL 2026-11-26 (B-110): every handle carries `WeightHandle::entries` (fp16 / fp8 / nvfp4 entry pointers keyed by `source_data`, bound in Phase 4 and again when layer offload rebinds `source_data`); Phase 4 throws when an NVFP4 primary / decode tier lacks its nvfp4 entry or an FP8 decode / prefill tier its fp8 entry (`missing_tier_entry`, `src/exec/weight_tier_entries.h`); the 16 handle-keyed lookups (15 lines) of those 3 maps in GEMM dispatch, the released-source GEMM, the small-M weight and the GDN / FP8 M=1 helpers read the handle: runtime map lookup lines outside `pre_dequant*` 45 -> 30 in 12 -> 11 files (fp16 / fp8 / nvfp4 / nvfp4_moe / cutlass_* / released_fp8_view / q8_planes_only / fused_*); left: the 20 lookups keyed by a layer tensor pointer, the prefill-copy maps, tier inference from map membership; was CONFIRMED-BY-READER | deciders can disagree silently: a decode tier of NVFP4 with a cache miss `break`s into the prefill ladder (`executor_gemm_dispatch.cpp`); Phase 4b/4c keep freed pointers as keys (`weight_caches.h`) | assert at the end of Phase 4 that every handle's decode / prefill tier has its cache entry; store the entry pointer in the handle |
| D-4 | Inkling is a private runtime (9378 LOC, 44 files, `fd -i inkling src tools tests`) whose sequence identity is the first block id of a KV pool it stores no data in | `src/exec/inkling_slot_map.h`, `src/exec/executor_inkling.cu`; no `WeightRegistry\|registry_\|gemm_via_handle_\|wcache_\|KVCache` hits in `inkling_forward.*`, `inkling_draft.cu`, `inkling_weights.cu`; engine prefix cache off (`src/runtime/engine_inkling.cpp`); #3270 (KV idle release) broke it, #3272 fixed it | PARTIAL (ROADMAP B-111): S1 slots bound on the request id, S2a/S2b the 9 full-attention layers in the paged pool, S3a (2026-11-26, GPU acceptance open) conv state and sliding rings in one engine-owned per-slot slab (`Engine::inkling_slab_`, 1178238976 B per slot, 0 B change); S3b (2026-11-26, GPU acceptance open) the draft's 2 local rings and the lookahead buffers in a 196458752 B lookahead region per lookahead slot of the same slab, planned at init (2 x 81920: `used_after_start` +374.7 MiB, peak unchanged); open S4 (reuse across slots, draft weights and draft scratch at the first forward, host conv checkpoints grown on the request path); was CONFIRMED-BY-READER | every KV / BlockPool / scheduler change can break Inkling with `verify-fast` green; LoRA, calibration, dispatch record and KV dtypes need a second implementation | carry a sequence id in `InferenceState` for decode rows and bind slots on it |
| D-5 | The LM-head route is decided in 4 places, two of them verbatim copies | the route block in `executor_forward.cu` equals the one in `executor_perplexity.cu`; further ladders in both files, `executor_draft_draws.cpp`, `executor_prompt_logprobs.cu` (which logs "LM head route changed") | FIXED (B-112), was CONFIRMED-BY-READER | PPL and logprobs judge a ladder serving does not share; a new head format means 4 edits | one `lm_head_route_()` returning an enum, called by all four |
| D-6 | Cross-op fusion state is a pointer-tag handshake on the executor (17 members: `smallm_xq_src_*`, `prenorm_*`, `router_prenorm_*`, `fp32_tail_layer_`, `norm_fold_*`) | tag check in `executor_gemm_smallm.cpp`; consumer in `executor_gemm_dispatch.cpp`; invalidation is manual (`executor.h`), called from 2 files (`rg -c 'smallm_xq_forget_\('`: `executor_forward_moe.cu` 2, `executor_fp32_residual.cpp` 5) | mechanism CONFIRMED-BY-READER, stale read SUSPECTED | a new in-place writer to a normed buffer that skips the forget call lets the next GEMM reuse stale quantized activations | bump a generation counter in the buffer-view helper instead of manual forget calls |
| D-7 | Per-forward mode is ambient executor state, not parameters | set at the top of the forward (`executor_forward.cu`); `cur_spec_verify_` 23 refs in 11 files, `cur_force_fp16_` 13 in 8, `cur_decode_rows_` 16 in 6; save / flip / restore in `executor_gemm_smallm.cpp` | CONFIRMED-BY-READER | any re-entrant forward (draft inside verify, lookahead) inherits the outer mode | move the 4 flags into `GemmContext`, which already carries `spec_verify_small_m` |
| D-8 | GraphExecutor holds 219 data members and 309 methods; about 70 are nullable per-architecture or tooling state | class body from `awk '/^class GraphExecutor/{f=1} f' src/exec/executor.h`; members via `rg -c '^\s+[A-Za-z_][\w:<>,\*& ]*\s[\*&]?[a-z]\w*_(\[[^\]]*\])?(\s*=\s*[^;]+\|\{[^;]*\})?;'`, methods via the analogous `name(` pattern. By prefix: qsa 12, mla 10, ssm 9, ple 7, hc 6, expert_cache 5, longrope / yarn 5, inkling 2, sae / lens / cvec / lora / calib 18 | SETTLED-SIZING (SETTLED B / R-080; that anchor measured recompiles, not port cost) | most entangled trio: GEMM dispatch, norm / FFN producers, spec-verify (D-6, D-7) | none until D-6 / D-7 land |
| D-9 | Three executor-to-kernel parameters travel through globals | `g_q8_epi` (`attention_paged_reduce.cu`, armed in `executor_attention.cpp`, consumed by the next eligible reduce); `s_splitk_scratch` (`attention_paged.cu`, set in `executor_attention_decode.cpp` and `executor_qsa.cpp`); the NVFP4 dequant provider holding an executor `this` (`nvfp4_gemm.cu`, `executor_lazy_scratch.cpp`) | FIXED 2026-11-26 (B-115: epilogue, NVFP4 dequant workspace and split-K scratch are per-call arguments), was CONFIRMED-BY-READER; safe today only because of one context per process | an armed epilogue with no following reduce leaks to the next call; a second forward thread breaks it | pass scratch and epilogue as launcher arguments |
| D-10 | 64 once-per-process log flags in exec + compute + quant (49 + 14 + 1), 39 at INFO, none re-armed on model swap; the dense route has no metric | `rg 'static (std::atomic<bool>\|bool\|std::once_flag) \w*(logged\|warned\|once\|said\|noted\|reported)'`; 12 reset hooks exist, none for these; `rg -c 'metrics\|counter_inc\|prom_'` in exec + compute = 2 | PARTIAL 2026-11-25 (B-116): 60 of the 63 matches in exec + compute + quant at `fc1bb711` are `LogOnce` latches; left: 2 `attr_once` (`gdn_scan_chunkpar*.cu`, a kernel attribute, not a log) and `json_constrain.cu` `once` (as `LogOnce` it trips the log-and-return gate, B-108); `tools/check_static_reset.py` fails a new flag. The dense-route metric is not addressed; was CONFIRMED-BY-READER | after a swap, model B's fallback never logs if model A tripped the flag | clear the flags in one registered reset hook |
| D-11 | 73 launcher declarations take 14 or more positional parameters, with runs of same-typed ints | awk comma count over `src/{compute,quant,exec}/*.h`. Worst: `gdn_scan_fused_bf16_batched` 26 (`gdn.h`), `lt_gemm_strided_batched` 23 (`gemm_lt_ex.h`), `write_kv_cache_rope_fused` 22 (`executor_kernels.h`), `paged_attention_nvfp4_multitok_gqa_launch` 22 (`attention_paged.h`), `qknorm_rope_fused` 21 (`rope.h`) | PARTIAL 2026-11-26 (B-133): the 10 raw-pointer `paged_attention*` launchers take `PagedDecodeArgs`, the 6 Tensor-level paged entry points and `paged_attention_decode_mla_padded` take `PagedDecodeTensorArgs` (62 call sites); 63 launchers left; was CONFIRMED-BY-READER | swapped `n_heads` / `n_kv_heads` / `head_dim` / `block_size` compiles | one `PagedDecodeArgs` for the 26 `paged_attention*` launchers; 27 Args structs already exist in newer code |
| D-12 | CUTLASS was removed (2026-10-31) but its predicate and vocabulary remain | `cutlass_sm120_nvfp4_available() { return true; }` (`nvfp4_sfatom_quant.cu`) guards 7 sites; 95 distinct identifiers, 925 mentions in 107 files (`rg -ci cutlass src`); `StorageTier::CUTLASS_NVFP4` means native NVFP4, `StorageTier::NVFP4` means GGUF overlay | predicate FIXED 2026-11-25 (definition, declaration and 7 guards deleted; the `cutlass` vocabulary stays, rename is the next slice), was CONFIRMED-BY-READER; names admitted historical in `gemm_cutlass_sm120.h` (#2938) | readers pick the wrong tier enum; 7 unreachable else arms | delete the predicate and its dead arms |
| D-13 | The WY GDN scan kernels are reachable only from tests | `gdn_scan_chunkwise_wy_f32` (`gdn_scan.cu`) and all of `gdn_scan_tc.cu` (660 lines); callers only in `tests/test_gdn.cu`; production calls fused / chunkpar / chunkwise from `executor_ssm_gdn.cu` | FIXED 2026-11-25 (owner: delete; 660 + 258 kernel lines, 3 tests removed), was CONFIRMED-BY-READER by `rg -w` over src, tests, tools, scripts | maintenance on unserved kernels (#2325 fixed a race in them); about 1000 LOC compiled | owner call: delete, or mark as chunkpar oracle |
| D-14 | Shared primitives are re-pasted in the drafters | `dspark_head_norm_rope_kernel` (`src/runtime/dspark_drafter.cu`) and `draft_norm_rope_k` (`src/exec/inkling_draft.cu`) are near line-for-line; RMSNorm again in `inkling_ops.cu` and `dspark_drafter.cu`; kernels live in `src/runtime/` | PARTIAL (B-119: shared kernel in compute, Inkling callers open), was CONFIRMED-BY-READER | a numerics fix lands in one copy | move the norm + RoPE kernel to `src/compute/` with a `w == nullptr` arm |
| D-15 | Architecture lists live outside ModelProfile in the loaders | 5 arches for gamma = 1 + W in `Model::upload_weights_gpu` (`weight_upload.cpp`); 8 per-arch refusals in `gguf_loader.cpp` (Flint and Inkling each added one) | FIXED (#3294), was CONFIRMED-BY-READER | a new Qwen GDN arch missing from the list loads norms off by 1 with no error | add `norm_gamma_is_delta` and `gguf_supported` to the profile row |

Sizing, same reader:

| fact | number | source |
|---|---|---|
| `ModelArch::` reads | exec 2, compute 0, runtime 8, model 266 | `rg -c 'ModelArch::'` per directory |
| trait / prefix hits in exec | 1043 in 56 files; `executor_qsa.cpp` 118, `executor.h` 113, `executor_sampling.cu` 96 | `has_gdn_?\|has_moe_?\|has_ssm_?\|is_moe\|inkling\|mamba\|\bmla_\|swa_\|\bqsa_\|ple_\|ModelArch::\|AttnVariant::\|profile` |
| Flint port (`553e4e7a`) | 54 files: 12 in src/model, 1 in runtime, 0 in exec | `git show --stat` |
| Inkling port (`b22e5f10`) | 57 files: 11 model, 6 exec, 5 runtime, 2 compute, 6 imp-server | `git show --stat` |
| public entry points never named in `tests/` | 133 of 382 in compute / quant headers (name level; indirect coverage not excluded); `gemv_fp8_block_moe` and `write_kv_cache_rope_fused` 0 references | name grep |
| not done | no full builder / reader / freer table per weight-cache family: only `fused_kv`, `fused_gate_up`, `lm_head_cutlass_` traced, none built but unread | |

## 4. Cross-cutting (layering, gates, tests, build, docs)

| ID | claim | evidence | status | blast radius | first slice |
|---|---|---|---|---|---|
| X-1 | 15 model env vars are set by no make target, script or workflow, so their suites skip green everywhere; Inkling alone is 17 tests in 3 files | `rg -c "\b$v\b" Makefile scripts .github` = 0 for `IMP_TEST_MODEL_{INKLING,DSPARK,DSPARK_TARGET,DFLASH,MTP,MTP_QWEN4EXP,KIMI,LING3,GEMMA4_E4B,GEMMA4_AUDIO,FP8,LLAMA,MISTRAL,DEEPSEEK,QWEN4B}` and `IMP_MTP_MODEL`; `tests/test_inkling_draft.cu` `GTEST_SKIP`; the `Makefile` names 9 vars | FIXED (#3290), was CONFIRMED-BY-READER; third recurrence of SETTLED I-1, no gate | every new port: its e2e gate exists and never runs | static gate: every `IMP_TEST_MODEL*` name read in `tests/` appears in `Makefile`, or in an allowlist with a reason |
| X-2 | Only `Build` is required; clang-tidy went red on main on 3 consecutive pushes and each merged | ruleset 14716423 `required_status_checks: [Build]`; runs 45259686168, 45259954018, 45270975645 (2026-11-23) `clang-tidy failure`; 8 failed main push runs in the last 200 (since 2026-11-09) | FIXED (#3287), was CONFIRMED-BY-READER | a header finding fails every later PR touching an includer; nobody is stopped | join the tidy result into `Build` the way static gates are, or add a "main last run" row to `scripts/preflight.sh` |
| X-3 | 1380 of 4082 tests are GPU-only and run in no automated lane; the hook GPU step is opt-in | pattern `^\s*(TEST\|TEST_F\|TEST_P\|TYPED_TEST)\(` per `imp_add_test_module` source list: compute 295, attention 196, quant 313, kv 121, moe-gdn 227, e2e 305 minus 77 unit subset; CPU 2313 + 312 + 77; `scripts/hook_lib.sh` | SETTLED-SIZING (F-5, owner: no GPU runner) | all kernel, KV and forward correctness | none beyond X-1; keep `IMP_HOOK_GPU=1` in the session env |
| X-4 | F-10 closed on the include edge, but the rebuild cost moved to `core/`: config sections are the hottest headers behind the widest fan-in | `git log --since=30.days -- src/core/config src/core/dispatch_policy.h` = 80 of 395 commits; `src/exec/executor.h` and `src/runtime/config.h` include `core/dispatch_policy.h`; `rg -l 'core/dispatch_policy.h\|runtime/config.h' src tools tests` = 99 files | PARTIAL 2026-11-26 (B-123): `executor.h` and 2 more headers forward-declare the policy, TUs reaching `core/dispatch_policy.h` 197 -> 160; slice 2: TUs reaching `runtime/config.h` 117 -> 98; `engine.h` (68 TUs) left; was CONFIRMED-BY-READER (counts); rebuild cost after #1227 not measured | each new config key recompiles most of exec + runtime + tests | `executor.h` holds only a pointer: move the accessor out of line, forward-declare, include per-section headers as `dispatch_policy.h` already says |
| X-5 | The Sanitizers PR path filter is a hand list that misses 3 untrusted-input parsers | the filter regex in `.github/workflows/ci.yml` has no match for `src/compute/regex_nfa.cpp` (a fuzz source, `cmake/imp-fuzzers-cpu.cmake`), `src/vision/vision_loader.cpp`, `src/audio/audio_wav.cpp` ("Untrusted input" in `audio_wav.h`); `fuzz/` has 8 targets, none for WAV | FIXED (#3287), was CONFIRMED-BY-READER; the nightly run covers it within 24 h, advisory | new parsers (audio, future loaders) skip ASan at PR time | derive the filter from the fuzz source list plus a tagged-file list; add `fuzz_wav` |
| X-6 | PDL enablement is a hand list in two places with no gate; the class already recurred 3 times | `src/exec/executor_workspace.cpp` calls 4 `*_pdl_register()`; `rg -l 'pdl_wait\(\|pdl_trigger\(\|pdl_launch_completion'` = 45 TUs, `rg -l 'enable_kernel\('` = 28; SETTLED A1-3, A2-2, A2-5 | FIXED (#3292): gated, 0 unregistered; was SUSPECTED (no unregistered `pdl_wait` kernel proven today) | silent perf loss on a new kernel, never a failure | gate: each kernel with `pdl_wait()` has an `enable_kernel` reference, selftested |
| X-7 | `check_static_reset.py` judges per TU: one hook passes a TU however many lazy statics it has | docstring "re-arms = `IMP_REGISTER_CUDA_STATIC_RESET(` in the TU"; in `src/quant/nvfp4_gemm.cu` statics sit outside the hook (safe today: `executor_lazy_scratch.cpp` clears them) | FIXED (#3292): gated by name; was SUSPECTED hole, no live defect | device use-after-free after model swap when a second static is added to a registered TU | gate compares names of acquiring statics against names assigned in the hook body |
| X-8 | The dormant CI `Test` job has drifted while never executing | `ci.yml` comment "decode 3% / prefill 5%" vs `tests/perf_baseline.json` 8 / 8; "baseline GGUF" vs the SafeTensors-only rule; `Test skipped` in every run | FIXED (#3287), was CONFIRMED-BY-READER, low | wrong gate the day a runner appears | delete the job or reduce it to `ctest -L gpu` |
| X-9 | `ARCHITECTURE.md` omits two source directories and the Inkling path; L2 docs have no staleness gate | `rg -ci 'audio\|lora\|inkling' docs/internals/ARCHITECTURE.md` = 0; `src/audio` 12 files, `src/lora` 4, `src/*/inkling*` 6318 lines; the layer order in `docs/audit/ARCHMAP.md` lacks `audio` (`check_layering.py` ranks it with vision) | FIXED (#3292), was CONFIRMED-BY-READER | a new port reads a map without the newest forward path | one row per missing directory; apply the 14-day `verified:` rule to L2 |
| X-10 | `imp-bench` is not compiled by the required check | `ci.yml` `-DIMP_BUILD_BENCH=OFF`, preset `ci` same; `tools/imp-bench/` 5 files | FIXED (#3293), was CONFIRMED-BY-READER, low (dev and image builds compile it) | bench breakage surfaces at measurement time | build the target in the nightly run |
| X-11 | imp-server and imp-cli link `src/` directly instead of the C API | `tools/imp-server`: 112 internal layer includes vs 2 `imp/` includes | SETTLED-SIZING (G-3, PARTLY) | the C API stays without a first-party consumer | none |

Where gates run:

| lane | gates |
|---|---|
| required (`Build`) | all 26 gates of `scripts/ci_static_gates.sh`, Kernel resources, `ctest -L unit`; the same script runs in preflight on every commit and push |
| advisory | clang-tidy, Sanitizers (path-filtered), PTX fallback, Mock / Real API contract, Lint, CodeQL, the named gate jobs, Roofline (triggers only on `tools/roofline/**`) |
| local only | `check-alloc-interpose` (otherwise only `check-release.sh`; a row of `verify` and `verify-fast` since 2026-11-26), `verify`, `verify-fast`, `verify-ab`, `verify-chunked`, `verify-north-star`, `test-gpu`, `test-e2e`, `test-vision`, `test-quantize`, `test-server`, `test-agents`, `test-niah`, `sanitize`, `vram-touch-audit` |
| scheduled | the nightly run (includes Fuzz) 10 of 10 green; 1 of the 8 main failures was a network flake in Real API contract (run 45320504156, curl exit 35) |

Include graph, `rg -c '^\s*#\s*include\s+"<to>/' src/<from>`; `^` = upward edge; rank core < memory <
quant < compute < model < lora < exec < vision = audio < runtime < api:

| from | to: count |
|---|---|
| core | none |
| memory | core 40, model 3^ |
| quant | core 55, memory 6, compute 1^ |
| compute | core 277, memory 36, quant 24, model 7^ |
| model | core 93, memory 15, quant 8, compute 4, vision 7^, audio 2^, exec 1^ |
| lora | core 2, memory 2, model 4 |
| exec | compute 299, core 215, quant 126, memory 84, model 33, lora 6, runtime 6^, vision 1^ |
| vision | core 28, model 16, memory 11, compute 7, runtime 1^ |
| audio | core 8, memory 2, model 2, compute 1 |
| runtime | memory 86, core 78, compute 63, model 42, exec 39, vision 13, quant 4, lora 3, audio 2 |
| api | core 6, model 6, runtime 6, memory 5, exec 1 |

- Upward lines: 29 = the sum of the 16 pins in `tools/layering_pins.txt` (13 pins 60 days ago).
- Cycles: memory / model, quant / compute, compute / model, model / exec, model / vision, exec / runtime, vision / runtime.
- `tools/` and `tests/` are outside the gate.

Top fan-in (direct includes over `src tools tests include`) with commits in 30 days:

| header | fan-in | churn |
|---|---|---|
| `core/logging.h` | 422 | 0 |
| `core/tensor.h` | 199 | 2 |
| `compute/gemm.h` | 109 | 8 |
| `model/model.h` | 81 | 4 |
| `exec/executor.h` | 79 | 49 |
| `runtime/engine.h` | 68 | 39 |
| `quant/nvfp4_quant.h` | 64 | 0 |
| `memory/device_alloc_all.h` | 63 | 3 |
| `runtime/request.h` | 61 | 9 |
| `quant/nvfp4_gemm.h` | 61 | 3 |
| `memory/kv_cache.h` | 61 | 7 |

Hot outside the top 10: `core/process_diag.h` 54 / 19, `runtime/config.h` 59 / 13, `core/config/gemm.h` 27
commits, `core/config/speculative.h` 24. `executor.h` and `engine.h` are priced in SETTLED (S-33, R-080).

## 5. Memory-axis inventories of the same day (I3, I5)

| invariant | count | source |
|---|---|---|
| I3 | `StableSpan` / `DeviceSpan` in 0 function parameters outside `src/memory`; 87 `take` / `take_bytes` sites in 31 files unwrap at once; 8 graph capture sites in 7 files, 6 read no buffer generation; 11 free-then-reallocate sites in 9 files (9 unreachable by reading, 2 small-M scratches, 1 banned-token cache not traced) | `rg "StableSpan\|DeviceSpan" src --glob '!src/memory/**'`; `rg '\.take(_bytes)?(<[^>]+>)?\('`; `rg cudaStreamBeginCapture` |
| I3 signatures | 759 header declarations take `cudaStream_t`: 435 with a raw device pointer, 111 `Tensor` only, 0 spans (compute 362 / 294 / 37, exec 249 / 73 / 59, quant 72 / 62 / 6, runtime 76 / 6 / 9) | header grep, pattern in the I3 inventory |
| I5 | 238 hand-written release sites in 78 files outside `src/memory/` (exec 124, runtime 36, compute 32, quant 22, model 20, vision 2, lora 2) against 53 RAII declarations; 8 owner-or-view `bool *owned*` flags | F1 `\b(device_free_all\|device_free_all_async\|device_free_all_n\|device_free_all_async_n\|device_free_async_checked\|vram_free\|weights_free)\s*\(`, F2 `\b(vram_alloc_\|alloc_\|alloc\|va_\|va)\s*(\.\|->)\s*free\s*\(` |
| I5 leaks | no release anywhere: `d_penalty_hist_` (64 x 131072 slots = 32 MiB), five `moe_.layer_stage_*` buffers and `d_slot_idx`, the Q4_K IMMA `WeightCache` planes on a shape mismatch (test-only entry) | alloc and every free site read |

## 6. Checked and fine

Control plane:

- Engine has no internal locking by design: one worker drives it; exclusive access goes through `pause()` / `resume()` under `state.mtx` (`batching_engine.cpp`, embeddings in `handlers_misc.cpp`).
- `VRAMAllocator::bytes_by_tag()` is mutex-guarded.
- The FFN graph cache refuses LoRA, calibration and offload, and keys on the workspace generation (`executor_ffn_graph.cpp`); the control-vector add sits outside the FFN phase.
- No capture uses global mode, so the KV grow-ahead thread cannot break a capture.
- `register_block_hashes` on an already freed sequence is a no-op (`kv_cache_manager.cpp`).
- Idle unload checks `state.inflight` before suspending (`handlers_admin.cpp`).
- One live context per process is enforced (`imp_api.cpp`).
- Sampling fields have one parser table (`sampling_fields.h`); chat, messages and responses share `build_imp_request_`; `/v1/completions` builds its own text-only request but reuses `apply_to` and `apply_spec_contract_`.

Data plane:

- S-1 holds: architecture dispatch stays in `model_profile`, 2 `ModelArch::` reads in exec, 0 in compute.
- Static reset: 12 registered hooks, `tools/static_reset_allowlist.txt` has 0 entries, one context per process is documented (`include/imp/imp.h`, `src/api/imp_api.cpp`).
- GDN scan f32 / bf16 variants are templates, not copies (`gdn_scan_chunkpar.cu`, `gdn_scan_chunkpar_pass.cu`).
- The MoE prefill model is replayed against the live chain (`executor_forward_moe_cutlass.cpp`); the attention model likewise (`attention_dispatch.cpp`).
- `weight_dispatch.cpp` no-route arms throw.
- A conventional MoE port needs no exec change (Flint: 0 exec files).
- `fused_kv` has builder, reader and freer (`pre_dequant_phase1_fp16_cache.cpp`, `executor_attention_qkv.cpp`, `executor_workspace_buffers.cpp`).
- The SafeTensors NVFP4 path pays at most 2 map probes above M 32 before the tier switch (`executor_gemm_dispatch.cpp`).

Cross-cutting:

- Layering is gated in the required check: an unpinned, over-ceiling or stale backward edge fails (`tools/check_layering.py`); core has no outgoing edge.
- Allowlists shrink: `alloc_allowlist.txt` 69 to 0 entries in 60 days, `sync_device_write_allowlist.txt` 66 to 49 in 45 days, `complexity_baseline.toml` 193 to 143 pins, `static_reset_allowlist.txt` 1 to 0.
- Dev, image and CI builds use the same `Release` flags (`Makefile`, `Dockerfile`, `ci.yml`); `rg IMP_DEBUG` finds 5 uses in 4 files, no other compile-time feature macro in `src/`.
- Every model path named in `Makefile`, `verify.sh` and the hooks exists in the host model directory `$HOME/models` (11 of 11); `verify.sh` fails, not skips, on a missing model directory.
- ROADMAP backlog: 12 rows sampled (B-29b/c/e, B-33, B-59, B-60, B-61, B-64, B-85, B-86, B-95, B-97); none is open with a merged fix.
- `check_filesize.py` charges `#include`d `.cpp` fragments to their includer, so the `executor_attention.cpp` TU is still measured after the `.cu` to `.cpp` move.

Not completed: OPEN items in root `AUDIT.md` older than 30 days (only O1 / O2 and the Phase B list were read, without age); tests that assert nothing (the grep heuristic was too noisy for a count).

## 7. Ledger drift

Amended in place on 2026-11-24 where the file is a ledger; listed only where it is not.

| where | says | code today | amended |
|---|---|---|---|
| `SETTLED.md` "A `VramOwned` type does not exist" | no such type | `src/memory/vram_owned.h` defines it (#1530) | yes |
| `SETTLED.md` S-1 | `ModelArch::` in 15 files, 9 in `src/model/` | 22 and 16 (`rg -c 'ModelArch::' src -g '*.cpp' -g '*.h' -g '*.cu' -g '*.cuh' \| wc -l`) | yes |
| `SETTLED.md` S-5 | `step()` chain in `src/runtime/engine.cpp` | `engine_scheduler.cpp` (`rg 'Engine::step\(\)\|step_decode_forward' src/runtime/engine.cpp` = 0); the early-return lanes in `step_impl_` are parallel paths | yes |
| `SETTLED.md` S-6 | one RMSNorm / RoPE / top-k each under `src/compute/` | copies in `inkling_ops.cu`, `inkling_draft.cu`, `src/runtime/dspark_drafter.cu` (D-14) | yes |
| `SETTLED.md` S-9 | private mini-runtime refuted outside vision | Inkling is one (D-4) | yes |
| `SETTLED.md` deliberate specialisations | `gdn_scan.cu` vs `gdn_scan_tc.cu` | `gdn_scan_tc.cu` has no production caller (D-13) | yes |
| `SETTLED.md` section E, F-16 | `attention_cublas_prewarm()` called from `engine.cpp` | symbol gone (`rg attention_cublas_prewarm src` = 0); the verdict holds, the file has no `cudaDeviceSynchronize` | yes |
| `SETTLED.md` F-10 | CLOSED | the cost it cited (85 TUs x 130 commits per 6 months) lands on `core/config/*` at 80 commits per 30 days (X-4) | yes |
| `SETTLED.md` I-1 | FIXED for five suites | the class is open again with 15 unwired env vars (X-1); `docs/LIMITATIONS.md` records it for Ling3 and Gemma4-E4B only | yes |
| `DEBT_LEDGER_2026_08_21.md` row 8 | CLOSED | the class is live in `gemm_dispatch_uncached.cpp` (3 exits) and 5 sites of `weight_dispatch.cpp` (D-1) | yes |
| `graph_eligibility.h` and audit F-14 | every site calls `demote_graphs_` | `engine_inkling.cpp` wrote the flag directly (C-10, FIXED 2026-11-26) | holds again |
| `src/runtime/CLAUDE.md` entry points | suspend / resume and the top-level step in `engine.cpp`; "no status returns" | step in `engine_scheduler.cpp`, suspend in `src/api/imp_api_suspend.cpp`; `Engine::init` and `lora_set` return bool | no: L3 file, own PR |
| `src/compute/CLAUDE.md` | "a kernel that cannot serve its input fails loud" | does not hold for the D-1 sites | no: closes with D-1 |
| `.claude/skills/building-and-testing/SKILL.md` | `/models/Qwen3-8B-Q8_0.gguf` | file does not exist (`Qwen3-8B-GGUF/` holds a Q4_K_M); contradicts the SafeTensors-only test rule | no: skill file |
| `tools/filesize_thresholds.toml` reasons | `tests/test_gdn.cu` "12 tests"; `attention_paged.cu` "cluster" kernel; `executor_attention.cpp` "three #include'd .cu fragments"; `executor.h` "H-022" | 17 tests; 0 matches; the fragments are `.cpp`; H-022 only under `docs/archive/`; the other 6 sampled reasons hold | no: gate file |
| `tests/test_models.h` | `kEnvGolden` | 0 users | no |
