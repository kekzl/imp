# Debt ledger - 2026-08-21

What is actually open in imp today, measured against the tree at `d40cd394` (v0.29.0,
clean), not against a doc. Every OPEN item carries a `path:line` verified on this date, the
failure mode it produces, and the command that would prove it closed.

Read with [`SETTLED.md`](SETTLED.md), [`../LIMITATIONS.md`](../LIMITATIONS.md),
[`../DESIGN_DECISIONS.md`](../DESIGN_DECISIONS.md), [`../../AUDIT.md`](../../AUDIT.md) and
[`../roadmap.md`](../roadmap.md). Section 3 lists what those already killed, so the next
pass does not re-chase it.

Convention: **OPEN** = verified against the code today. **CLOSED** = the doc said open, the
code says otherwise, with the evidence. **NOT A FINDING** = swept, nothing there - recorded
because a negative result is what stops the next sweep.

---

## 1. Top 10, ranked by cost if ignored

| # | Item | Where | Cost if ignored |
|---|---|---|---|
| 1 | Per-decode-step `cudaMallocAsync` whose address is baked into a replayed CUDA graph, with no invalidation | `src/runtime/engine_scheduler.cpp:1695` | Silently wrong KV residual metadata on `residual + batch>1` decode; survives only because the async pool's release threshold is pinned to `UINT64_MAX` |
| 2 | ~~`cudaMalloc` freed through `cudaFreeAsync`~~ **CLOSED `556f3b8d` (#1505)** | `src/compute/mtp_forward.cu:610` / `:619` | Undefined behaviour per the CUDA API, once per MTP draft step, on every MoE model (the device-chain arm requires `n_experts == 0`) Fixed, and three further pairs this ledger did not have came out with it. |
| 3 | ~~Invariant I2 has no gate in any shipping build~~ **CLOSED: `make check-alloc-interpose`, a two-way pin at 19 calls over 5 named sites** | `CMakeLists.txt:67` | The counter that is supposed to catch items 1 and 2 reads zero in every build anyone runs; 469 direct allocation sites sit outside `src/memory/` where it cannot see them First run found **46 device allocations while serving**. 27 were the MTP workspace sitting on the wrong side of the phase flip (fixed). The remaining 19 are enumerated in (g) and the pin only ever goes down. |
| 4 | ~~829 of 2288 GTest macros run in no CI lane~~ **CLOSED, and corrected to 968** (`tools/check_test_lanes.py`) | `CMakeLists.txt:952-957` | The entire GPU correctness surface reaches `main` on the strength of one human running `make verify-fast`; the required check is `Build`, which runs `ctest -L unit` Pinned now, and it fails when it moves. 829 was an undercount, see (e). |
| 5 | ~~The file-size allowlist has no ceiling, and 16 of its 29 reasons are numerically stale~~ **CLOSED: every entry carries a measured `code_loc`, drift either way fails** | `tools/filesize_thresholds.toml:60` | An allowlisted file is exempt forever: `engine_scheduler.cpp` went 1074 → **1962** code LOC (+83 %) with the gate green the whole way. The gate is blind exactly where recompile blast radius is worst Reasons re-derived and the LOC figure taken out of the prose, because that is the half that went stale. |
| 6 | `vram.library_reserve_mb` default is a constant measured to be wrong in both directions | `src/memory/plan.h:100` | On a cold cache the plan charges 3900 MiB where the model needs 0 (Qwen3-4B IQ4_NL) or 7458 (Qwen3-8B Q8_0) - AUDIT B41. Either 3.9 GiB of KV pool set aside for nothing, or a 3.5 GiB under-charge |
| 7 | ~~`calibration.out_path` is parsed, documented, and never read~~ **CLOSED `66a1f633` (#1508)** | `src/runtime/config.h:362` | An operator who sets it in `imp.conf` gets no file and no warning; the real path comes from `--calibrate-out` Wired rather than removed (the `log_set_level` precedent, SETTLED §C); verified end to end. |
| 8 | ~~A dispatch case that logs an error and returns **without writing its output tensor**~~ **CLOSED `66a1f633` (#1508)** | `src/compute/weight_dispatch.cu:342` | Unreachable today, and the shape of the failure is a whole projection emitting stale device memory with one ERROR line. Its comment still describes a "phase-2 shim" whose phase 3 landed elsewhere It throws now. Two siblings with the identical defect remain at `weight_dispatch.cu:278` and `:378` - `IMP_LOG_FATAL` only logs, `IMP_CHECK` is what aborts - and a test was pinning the old behaviour as the contract (`EXPECT_NO_THROW`, "output buffer is unchanged"). |
| 9 | ~~Dead FP16 twin of a live FP32 kernel~~ **CLOSED `66a1f633` (#1508)** | `src/compute/gdn.cu:273` + `src/compute/gdn.h:152` | ~35 LOC and a `__global__` in a `.cu` already 44 code LOC over its warn threshold; only `vhead_tiled_to_grouped_f32` runs |
| 10 | ~~20 header-inline accessors with no caller anywhere~~ **CLOSED: 28 removed, `tools/check_dead_inline_accessors.py` added** | `src/exec/executor.h:187` and 19 others | Weak on its own. It is here because it is the *class the 2026-08-03 decl-only sweep could not see*: that sweep filtered on decl+def (2 occurrences); a header-inline definition is 1 20 was an undercount and the sweep cascaded, see (c). |

Item 10 is the only one on this list whose justification is maintenance surface rather than a
failure mode or a blind gate. It is ranked last for that reason, and it is stated rather than
dressed up.

---

## 2. The six categories

### (a) Memory / allocation invariants I1-I7 - `AUDIT.md` B8, B9, B10, O1

```
$ grep -rn "set_alloc_phase" src/ include/ tools/ | wc -l
8
$ python3 tools/check_alloc_sites.py
I1: 67 files / 469 sites outside src/memory/ (allowlist: 67 files / 469 sites)
OK — no new direct allocation sites.
$ python3 tools/check_alloc_sites.py --stats | sed -n 2p
  acquisitions 210  (device 210, pinned-host 0)   releases 279
```

**B8 - CLOSED.** The entry's headline ("`set_alloc_phase()` appears only in `tests/`; the
allocation-phase guard was never armed - acceptance criterion 3 is currently vacuous") is
false against the tree. `src/runtime/engine.cpp:789` sets `Planning` before the sub-phases,
`:818` sets `Serving` after `prewarm_spec_scratch_()`, `:75` resets to `Loading` in the
destructor and logs the count. The CUDA-side watermarks are armed at the same transition
(`engine.cpp:822`). B35 records the measured result: `steady state clean: 0 cudaMalloc,
0 cudaMallocAsync, 0 pinned-host allocations while serving`.

**B8's second half - OPEN, and it is item 3 above.** `note_serving_allocation()`
(`src/memory/backend.cpp:100`) is called from `Backend::acquire()` and from the `--wrap`
interposer. The interposer compiles only under `IMP_ALLOC_INTERPOSE`, which is `OFF`
(`CMakeLists.txt:67`) and appears in **no make target and no CI job**:

```
$ grep -rn "IMP_ALLOC_INTERPOSE" Makefile .github/workflows/ scripts/ | wc -l
0
```

So in every build that ships, `steady_state_allocations()` sees only what routes through
`Backend` - which is the arena at init - while 469 direct sites sit outside `src/memory/`.
B9 and B10 below both allocate while serving and both read as zero.
*Proof it is closed:* a `make` target that builds `-DIMP_ALLOC_INTERPOSE=ON`, drives ≥15
requests on a config that exercises residual+batch>1 and an MoE MTP chain, and fails on a
non-zero `[alloc-interpose] steady state` line.

**B9 - OPEN, structurally unchanged, line numbers moved 1511 → 1695 and 1873 → 2057.**

```
$ sed -n '1695p' src/runtime/engine_scheduler.cpp
            if (cudaMallocAsync(&residual_meta_d_buf_, meta_bytes, dec_stream) == cudaSuccess) {
$ sed -n '1703,1705p' src/runtime/engine_scheduler.cpp
                state.d_residual_seq_slots = base + static_cast<ptrdiff_t>(0) * N;
                state.d_residual_counts = base + static_cast<ptrdiff_t>(1) * N;
                state.d_residual_write_idxes = base + static_cast<ptrdiff_t>(2) * N;
$ sed -n '2023p' src/runtime/engine_scheduler.cpp
        graph_runner.execute(dec_stream);
$ sed -n '2056,2058p' src/runtime/engine_scheduler.cpp
    if (residual_meta_d_buf_ != nullptr) {
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(residual_meta_d_buf_, dec_stream));
        residual_meta_d_buf_ = nullptr;
```

The three invalidation triggers around `graph_runner.execute()` are `bucketed_max_blocks`
(`:1981`), `bucketed_max_ctx` (`:2002`) and the recurrent state slot (`:2015`). None of them
observes `residual_meta_d_buf_`. **Failure mode:** the captured `forward_logits` replays with
the previous step's device address baked in as a kernel parameter; if the stream-ordered
allocator ever hands back a different address, the residual metadata read is a stale pointer.
**Nothing tests it** - `grep -rn "TEST.*[Rr]esidual" tests/` returns 17 tests, all of them
kernel-level or sizing-level; `test_attention_paged_nvfp4_tc_residual.cu:242
MultiSeqBatchArrayForm_HD64` is the closest and it never captures a graph.
*Proof it is closed:* token-for-token equality between `use_cuda_graphs=false` and the default,
on a run with `kv_cache.bitdecoding_residual_tokens>0`, `kv_cache_dtype=NVFP4` and `batch>1`,
over ≥200 tokens.

**B10 - CLOSED `556f3b8d` (#1505).** The allocation is gone rather than re-typed: `ws.d_tok`
is a persistent workspace slot beside the existing `ws.d_argmax`, whose comment already
recorded the same fix for the output side of the same function. Three further mismatched
pairs came out of that sweep and were **not** in this ledger: `chunk_eager_k_`/`_v_`
(128 MiB, `cudaMallocAsync` at `executor_attention_prefill.cu:176`/`:177` against `cudaFree`
at `executor_workspace_buffers.cu:1584`/`:1588` - the reverse direction of B10, recorded
nowhere) and `shared_workspace_`'s resize grow branch. `tools/check_alloc_pairs.py` gates
the class.

**What #1505 did NOT reach**, stated so "an allocation gate landed" is not read as "I2 is
gated": the pairing gate is a *static* check over source text. Item 3 - that no shipping
build can observe a serving-phase allocation - is untouched by it. A future per-request
`cudaMallocAsync`/`cudaFreeAsync` pair is correctly paired *and* an I2 violation, and
nothing in the tree would say so.

The entry as it read before the fix:

**B10 - was OPEN, lines moved 606 → 610 and 615 → 619.**

```
$ sed -n '609,610p;619p' src/compute/mtp_forward.cu
        int32_t* d_tok = nullptr;
        if (cudaMalloc(&d_tok, sizeof(int32_t)) != cudaSuccess) {
        cudaFreeAsync(d_tok, stream);
```

Reachability, which the entry did not state: the arm runs when `d_prev_token == nullptr`,
which is the default of `mtp_draft_step` (`src/compute/mtp_forward.h:212`). Four call sites
take it - `engine_spec_mtp.cpp:94`, `:143`, `engine_scheduler.cpp:2199`, `:2232` - and the
device-chain arm that avoids it is gated on `ws->n_experts == 0`
(`src/runtime/engine_spec_mtp.cpp:80`), so **every MoE model takes the host arm on every
draft step**. *Proof it is closed:* the allocation is gone (arena or workspace slot), and
`check_alloc_sites.py --stats` shows `src/compute/mtp_forward.cu` one acquisition lighter.

**O1 - CLOSED, and the answer is bigger than the question.** "What claims the ~3.9 GiB" was
answered by AUDIT B41: it is the cuBLAS/CUTLASS reserve claimed on the two warmup forward
passes, it is not a constant, and it was measured at 0 / 4182 / 7458 MiB across three
checkpoints. B42/B43 shipped the measurement and a warning; the auto-persist follow-up B42
called for now exists as `src/memory/library_reserve_cache.h` and is wired at
`src/runtime/engine_kv_cache_init.cpp:196-227`.

**What is still open from O1 (item 6):** the cold path.

```
$ sed -n '98,100p' src/memory/plan.h
// Re-measure after a driver or CUDA bump; imp.conf `vram.library_reserve_mb`
// overrides it per host.
constexpr size_t kMeasuredLibraryReserveBytes = 3900ull * 1024 * 1024;
$ sed -n '195,196p' src/runtime/engine_kv_cache_init.cpp
    // vram.library_reserve_mb always wins; a miss leaves the constant in place.
    if (config_.library_reserve_mb < 0 && runtime_config_.vram.library_reserve_cache != "off") {
```

A first run on any new model/quant charges 3900 by construction. *Proof it is closed:* a
first-run boot on a model whose measured reserve is 0 (Qwen3-4B IQ4_NL) plans a KV pool
within 1 % of the same model's second run.

**NEW, OPEN - the shared-workspace fallback loses its bytes from the accounting.**

```
$ sed -n '638p;666p' src/exec/executor_workspace.cu
        cudaError_t err = cudaMalloc(&shared_workspace_, max_shared);
        vram_free(vram_alloc_, shared_workspace_);
$ sed -n '99p' src/memory/vram_allocator.cu
    IMP_CUDA_CHECK_LOG(cudaFree(ptr));
```

`allocate_shared_workspace` falls back to a raw `cudaMalloc` when `VRAMAllocator` rejects
the request (the Nemotron-30B case its own comment names), and `free_workspace` returns
that pointer through `vram_free` -> `allocator->free()`, which never handed it out.
**Memory-safe**: `VRAMAllocator::free` misses the map, skips the bookkeeping and still
calls `cudaFree`. **Not accounted for**: the `allocated_` counter and the `MemAccount`
pool note are both skipped, so the largest single workspace allocation in the process is
invisible to the accounting exactly on the configurations where it was hardest to get.
Same shape as item 3: a counter reads clean because the traffic never reaches it.
*Proof it is closed:* a run that forces the fallback (a budget tight enough for
`vram_alloc` to decline) shows `--mem-report` charging `shared_workspace` the same bytes
it charges on the non-fallback path.

**NEW, CLOSED - `IMP_LOG_FATAL` does not stop anything, and ten of twelve sites
assumed it did.**

```
$ sed -n '58p' src/core/logging.h
#define IMP_LOG_FATAL(...) ::imp::log_message(::imp::LogLevel::FATAL, __FILE__, __LINE__, __VA_ARGS__)
$ python3 tools/check_log_fatal.py     # on the pre-fix tree
log-fatal: 12 IMP_LOG_FATAL site(s), 1 abort, 0 throw, 11 continue (1 allowlisted)
```

`IMP_CHECK` (`logging.h:68-74`) is the only thing in the tree that reaches
`std::abort()`, and its own comment says so. So the macro's name promised what
only its sibling delivered. Census and verdicts:

| site | after the log | verdict |
|---|---|---|
| `pre_dequant_phase4_tensor_registry.cu:550` | `std::abort()` | correct |
| `weight_handle.h:26` | `return self.handles_[id]` | **out-of-bounds read after reporting the id is out of range** |
| `weight_dispatch.cu:281`, `:392` | `return` | output tensor left holding whatever it held; `:392` is the `default:` |
| `expert_cache.cu` x4 | falls through / returns the slot | fills a slot it just said does not fit; returns a pointer it just called unusable |
| 3 MoE staging sites | falls through | each says in its own comment that continuing hands a HOST pointer to a device kernel |
| `expert_cache.cu` parity check | `return false` | correct in itself, **and both callers discarded the verdict** |

The last row is the one worth keeping: `check_parity()` returning `false` is the
right contract for a function that reports agreement, but `get_or_load()` called
it as a bare statement. So the debug facility detected the host/device
divergence, logged it, returned false into nothing, and carried on - while
`expert_cache.h:273` documented the behaviour as "aborts on mismatch". Found
while correcting that comment, not by the census.

Two throw (dispatch failures the API boundary turns into `ImpError`), eight
abort via `IMP_CHECK` (state corruption or a documented abort contract), one was
already correct, one is allowlisted with the reason. `tools/check_log_fatal.py`
keeps it at zero, keyed on the message rather than the line number - the first
version keyed on lines and rotted immediately when four comment lines moved a
site by six.

### (b) Stubs, ignored request fields, dead kernels, tests that assert nothing

```
$ rg -n -i --glob '!build*' -e '\b(STUB|NOT[_ ]?IMPLEMENTED|UNIMPLEMENTED)\b' src include tools | wc -l
8
$ grep -rn "TODO\|XXX\|FIXME\|HACK" src/ include/ tools/ --include=*.cpp --include=*.h --include=*.cu --include=*.cuh | wc -l
6
$ rg -n 'DISABLED_' tests | wc -l
8
$ awk -f .claude/skills/find-stubs/tests_without_assertions.awk $(find tests -name '*.cpp' -o -name '*.cu') | wc -l
44
```

Rung 1 is **8, down from the skill's stated baseline of 12** - `gemm_grouped_dispatch`, the
2026-08-19 find, is gone from the tree. Of the 8: three are the `mmq_q4k_hmma` "Phase 0 stub"
header/comment set, live behind `gemm.q4k_hmma_enabled` (default `false`, dispatched at
`src/exec/executor_gemm_dispatch.cu:340`); one (`src/compute/gdn.cu:393`) is a comment
*describing a fixed defect*, not a stub; one is a tokenizer note; two are the item-8 branch.

**OPEN - `src/compute/weight_dispatch.cu:342`, item 8.**

```
$ sed -n '342,353p' src/compute/weight_dispatch.cu
        case StorageTier::CUTLASS_NVFP4: {
            // CUTLASS_NVFP4 is a prefill tier (M>1).  Decode falls through
            // to NVFP4 GEMV in the consumer (executor_kernels.cu line 1951).
            // gemv_dispatch is only called for decode (M=1); in phase-2 the
            // consumer still uses the wcache_ NVFP4 entry directly.
            // Stub: log error and do nothing so tests can verify routing.
            IMP_LOG_ERROR(
                "gemv_dispatch CUTLASS_NVFP4: not directly callable for decode "
                "(no FP8 micro_scales in payload); consumer should use NVFP4 tier");
            return;
        }
```

`gemv_dispatch` switches on `w.primary_tier`; its four callers
(`src/exec/executor_gemm_dispatch.cu:235,263,266`) switch on `decode_tier`. Traced:
`decode_tier` is assigned in exactly one place - `src/exec/pre_dequant_phase4_tensor_registry.cu:90,96,98,100` -
to `tier`, `FP8` or `NVFP4` and nothing else, so `primary_tier == CUTLASS_NVFP4` with a
`decode_tier` that routes here is not producible today. It is reported because the branch
returns **without writing `y`**: reached, it emits a projection of stale device memory behind
one ERROR line. *Proof it is closed:* the branch either writes a correct result or throws
(the `#654` precedent, `SETTLED.md` S-22).

**NOT A FINDING - accepted-but-ignored request fields.** Rung 5 reproduces the recorded
baseline exactly: 53 fields, 10 candidates, no name with 0 uses, no new name in the list.

```
SUSPECT cache_prompt (2)  enable_thinking_requested (2)  image_error (2)  include_usage (1)
SUSPECT json_schema_str (2)  max_stop_len (1)  rep_pen_explicit (1)  requested_model (1)
SUSPECT top_k_explicit (1)  top_p_explicit (1)
```

**NOT A FINDING - tests that assert nothing.** 44, the recorded baseline, dominated by
harness-delegating bodies (`run_case(...)`, `run_arch(...)`). 8 `DISABLED_`, all with a
stated reason in the file; two of them (`test_determinism_e2e.cpp:177,221`) are deliberately
the gate for a known limit.

**find-stubs rung 4 ran, 2026-08-21. One finding, and the census is worth less
than the cross-check.**

```
$ nm build-dev/imp-server | grep -oP '__device_stub__\S+' | sed 's/.*__device_stub__/_/; s/\.cold//' \
    | c++filt | sed 's/(.*//; s/<.*//; s/.*:://' | grep '_kernel$' | sort -u
285 kernels present
$ <launched, from the four nsys runs of assignments 4 and 4b:
   dense + MoE x 8k + 32k, prefill and decode>
61 launched
249 dark
```

**249 dark does not mean 249 dead**, and this is the limit of the method as the
skill describes it: four workloads cover no vision, no constrained decoding, no
MoE host offload, no KV dtype but FP8, and no speculation. Cross-checking each
dark kernel against its mentions in `src/` collapses 249 to **7**, and reading
those 7 collapses it to **1**:

| | |
|---|---|
| 6 of 7 | live benchmark/probe kernels that exist only under `tests/bench/` and are launched by their own harnesses |
| **1 of 7** | `fp32_accum_add_fp16_kernel` - declared in `executor_kernels.h`, defined in `executor_elementwise.cu`, and its **only caller is its own test** |

That last one is the finding, and its shape matters more than its size: **every
existing check reads it as covered.** The 2026-08-03 decl-only sweeps
(`SETTLED.md` §C) filtered on decl+def with nothing else, and this has a third
mention. `check_dead_inline_accessors.py` (#1506) filters on header-inline
definitions, and this is a `.cu` definition. A code-graph caller query finds a
caller. A kernel whose only caller is a test is invisible to all three, and it
is the same class as the `add_fp16_bias_to_fp32_kernel` §C records as "never
launched at all". Removed, with its test.

The second half of rung 4 - a *live* kernel behind a condition that is never
true, the `ssm_graph_ban` class - is **not** answered by this. That needs to
know which condition gates each kernel, which a launch census cannot see, and
the four workloads here would report such a kernel as merely dark. Stated so the
next pass does not read this entry as having closed it.

### (c) Dead code - symbols with no caller

The graph was 76 files stale; `codegraph sync` cost 1.7 s. **`ccg enrich` does not run
against this DB** - it aborts on `sqlite3.IntegrityError: UNIQUE constraint failed:
index 'idx_edges_identity'` after reporting `already in graph: 225`, so there are no
`launches` edges and every `__global__` kernel appears uncalled. The sweep was therefore run
graph-first and filtered by the mandatory grep, per `code-graph` trap 2.

```
$ codegraph sync
Synced 76 changed files — Added: 4, Modified: 72 — 1,895 nodes in 817ms
$ <sql: functions/methods in src|tools with no incoming calls edge>
581
$ <filter: ≤2 textual occurrences across src tools tests include>
110 candidates, of which 33 are Python (dynamic dispatch, false positives)
```

Verified against the code, the 77 C++ candidates resolve to:

- **False-positive family, refuted:** 10 `*_reset_static_cuda_state` functions, every one
  bound by `IMP_REGISTER_CUDA_STATIC_RESET` on the line after its definition. This is
  `SETTLED.md` §C's R-11 mechanism, which says it "looks deadest of all". Not dead.
- **False-positive family, refuted:** every `__global__` kernel checked
  (`paged_attention_decode_kernel`, `paged_attention_splitk_pipeline_kernel`,
  `paged_attention_splitk_fp8_pipeline_kernel`, `paged_attention_splitk_int4_kernel`,
  `paged_attention_decode_int8_kernel`, `gguf_q*_kernel`, `q4k_imma_kernel`) has its launch
  or registration in the same file, invisible to the graph without launch edges.
- **OPEN, one genuine dead function (item 9):**

```
$ grep -rwn "vhead_tiled_to_grouped" src tools tests include
src/compute/gdn.cu:273:void vhead_tiled_to_grouped(const half* src, half* dst, ...
src/compute/gdn.h:152:void vhead_tiled_to_grouped(const half* src, half* dst, ...
$ grep -rwn "vhead_tiled_to_grouped_f32" src tools tests include
src/compute/gdn.h:160:void vhead_tiled_to_grouped_f32(...
src/compute/gdn.cu:308:void vhead_tiled_to_grouped_f32(...
src/exec/executor_ssm_gdn.cu:438:        vhead_tiled_to_grouped_f32(v_scratch_tiled, v_scratch_grouped, ...
```

  The single consumer of `gdn.vhead_reorder` (`src/exec/executor_ssm_gdn.cu:421-422`) calls
  only the FP32 variant. The FP16 twin plus `vhead_tiled_to_grouped_kernel` is dead.
  *Proof it is closed:* both symbols removed, `make dev-test` green, `check_filesize.py`
  shows `src/compute/gdn.cu` below its 544 code LOC.

- **OPEN, item 10 - 20 header-inline accessors with exactly one occurrence in the tree**
  (their own definition):

```
src/exec/executor.h:187 sample_slots          src/exec/executor.h:200 sample_parity
src/exec/executor.h:390 streaming_n_sinks     src/memory/kv_cache_manager.h:147 pin_budget_blocks
src/memory/kv_cache_manager.h:282 residual_n_kv_heads
src/memory/kv_cache_manager.h:283 residual_head_dim
src/core/weight_handle.h:77 is_populated      src/core/qtype.h:51 is_block_quant
src/core/qtype.h:59 is_compute_dtype          src/core/buffer.h:32 is_device
src/compute/json_schema.h:68 empty_set_dead   src/compute/json_constrain.h:94 closing_tokens_needed
src/runtime/engine.h:182 lora_active          src/vision/vision_pipeline.h:64 boi_id
src/vision/vision_pipeline.h:65 eoi_id        src/exec/activation_calibrator.h:53 skipped_non_fp16
src/memory/graph_slots.h:176 host_bytes       src/lora/lora_adapter.h:59 set_name
src/memory/weight_snapshot.h:145 builder_set_identity
src/model/loader_assign.h:22 assign_quant_with_scales
```

  **CORRECTION, 2026-08-21: 20 was an undercount and the true figure is 28.** Two reasons,
  both mine rather than the tree's. (1) I counted `hidden_state` as having two occurrences;
  the second is inside a *comment* in `src/model/mtp_head.h:19`, and the purpose-built gate
  strips comments before counting. (2) Four device `__forceinline__` helpers
  (`cp_async_cg_8` in `attention_paged_common.cuh`, three `tq_fp4_*` in `turboquant_fp4.cuh`)
  were in my candidate list and I did not carry them into the written list, because I was
  selecting accessor-shaped names by eye - a filter applied after the measurement and never
  stated. (3) The remaining three appeared only when the first 25 were removed: deleting
  `tq_fp4_pack_pair`/`_unpack_lo`/`_unpack_hi` orphaned `tq_fp4_quantize_signed` and
  `tq_fp4_dequant_nibble`, and removing those orphaned one more. Dead code hiding dead code,
  so the sweep has to run to a fixpoint (three rounds: 25, 2, 1).
  `turboquant_fp4.cuh` goes from ~100 lines to 44. `attention_paged_common.cuh` is included
  by 10 TUs, so `cp_async_cg_8` is the one with real recompile fan-out.

  Why they survived the sweep `SETTLED.md` §C calls "fully resolved": that sweep filtered on
  the decl+def signature, i.e. **2** occurrences. A header-inline definition has **1**. Dated
  with `git log -S`, all six sampled predate 2026-08-03, so this is an uncovered class, not a
  re-run. *Proof it is closed:* the list is empty under the same query, and the query becomes
  a gate.

### (d) `[allow]` entries in `tools/filesize_thresholds.toml` whose reason no longer holds

```
$ python3 tools/check_filesize.py | tail -1
scanned 739 files in src, tools, tests | warn=38 allowlisted=29 violations=0
```

The gate is green and the allowlist is the reason. Every entry states a code-LOC figure as
its justification; **16 of 29 files are now larger than the figure that justified them**, and
the gate has no mechanism to notice:

| file | reason says | measured today | drift |
|---|---|---|---|
| `src/runtime/engine_scheduler.cpp` | 1074 | **1962** | +83 % |
| `src/compute/attention_fmha_mxfp4_sm120.cu` | 810 | **1520** | +88 % |
| `src/exec/executor_workspace_buffers.cu` | 906 | **1346** | +49 % |
| `src/memory/kv_cache_manager.cpp` | 852 | **1200** | +41 % |
| `src/model/weight_upload.cu` | 1658 | **1999** | +21 % |
| `src/compute/mtp_forward.cu` | 651 | **853** | +31 % |
| `src/runtime/cuda_graph.cu` | 863 | **1046** | +21 % |
| `src/exec/executor_forward_moe_batch.cu` | 872 | **1061** | +22 % |
| `src/compute/gemm.cu` | 613 | **727** | +19 % |
| `src/model/weight_map.cpp` | 966 | **1068** | +11 % |
| `src/model/hf_config_loader.cpp` | 810 | **943** | +16 % |
| `src/compute/attention_paged.cu` | 1134 | **1196** | +5 % |
| `src/compute/attention_fmha_sm120.cu` | 1436 | **1480** | +3 % |
| `tests/test_paged_attention.cu` | 982, "14 tests" | **1202**, **26 tests** | +22 % / +86 % |
| `tests/test_kv_cache.cpp` | 951, "56 tests" | **1159**, **59 tests** | +22 % |
| `tests/test_gdn.cu` | 942, "11 tests" | **1026**, **12 tests** | +9 % |

Two entries shrank below their stated figure (`src/compute/json_schema.cpp` 1158 → 1034,
`src/model/safetensors_loader.cpp` 1147 → 1088) and one is exact
(`src/model/gguf_loader.cpp` 812). Test counts from
`grep -cE '^\s*(TEST|TEST_F|TEST_P)\(' <file>`.

**Failure mode (item 5):** `SETTLED.md` S-26 blesses this allowlist as "manages file size
instead of silencing it", and it does - for the *decision*. It does not manage the *number*.
An allowlisted file has no ceiling at all, so the metric the gate exists to control
(recompile blast radius, one `.cu` = one `ptxas` TU) grows unobserved; `engine_scheduler.cpp`
nearly doubled. One reason is additionally wrong in substance rather than arithmetic:
`schema_constrain.cu` says "split candidate if a third grammar lands", and it now carries
`init_grammar_for_test` plus two body grammars around one `sim_advance` - the condition it
names is at its boundary, not past it.
*Proof it is closed:* each `[allow]` entry carries a measured `code_loc` the gate compares
against, failing on growth in either direction - the same two-way ratchet
`tools/alloc_allowlist.txt` already has (`SETTLED.md` S-27).

### (e) Untested paths - `docs/FEATURES.md` 🟡, and gates reachable from no target

```
$ grep -c "🟡" docs/FEATURES.md
4
```

One of the four is the legend line. The three real ones - Llama-4 (`:43`), FP8 E5M2 (`:61`),
MoE host offload for NVFP4 experts (`:102`) - are each recorded in `LIMITATIONS.md` at
`:38`, `:39` and `:54`. **The contract FEATURES.md states about itself holds.** NOT A FINDING.

**No orphan test file.** Every `tests/*.cpp|cu` is a source of one of the eight
`imp_add_test_module` targets:

```
$ comm -23 <(ls tests/*.{cpp,cu} | sed 's|tests/||' | sort) \
           <(sed -n '580,930p' CMakeLists.txt | grep -oE '[A-Za-z0-9_]+\.(cpp|cu)' | sort -u)
(empty)
```

(A case-sensitive first pass wrongly flagged `test_gemm_grouped_nvfp4_smallM.cu`; it is at
`CMakeLists.txt:828`.)

**OPEN - item 4. The gap is the lane, not the file.**

| binary | files | `TEST*()` macros | ctest lane |
|---|---|---|---|
| test-core | 59 + 13 via `target_sources` (`CMakeLists.txt:734-749`) | 1086 | `unit` |
| test-text | 6 | 197 | `unit` |
| test-e2e | 28 | 176 | `unit` (6-pattern filter) + `gpu` (its complement) |
| test-compute | 23 | 200 | `gpu` only |
| test-attention | 13 | 210 | `gpu` only |
| test-quant | 34 | 196 | `gpu` only |
| test-kv | 11 | 75 | `gpu` only |
| test-moe-gdn | 9 | 148 | `gpu` only |
| **total** | **195** | **2288** | |

Macro counts, not instantiations: a `TEST_P` counts once here and runs once per value. The
13 `target_sources` files are the trap in this table - an `imp_add_test_module`-only parse
misses them and undercounts test-core by 271.

CI's required check is the job named `Build` (`.github/workflows/ci.yml:42`), whose test step
is `cd build && ctest -L unit` (`:189`). **829 macros in the five GPU-only binaries run in no
CI lane at all**, plus test-e2e's complement. The `perf` lane covers compute/attention/quant/e2e
and omits `test-kv` and `test-moe-gdn` (`CMakeLists.txt:962-967`) - harmless, since the `gpu`
lane runs those binaries unfiltered, but it means `*Perf*`/`*Bench*` there are never run in
isolation.

**CORRECTION, 2026-08-21: the figure above is a macro count and it is also incomplete.**
829 counts `TEST`/`TEST_F`/`TEST_P` macros in the five GPU-only binaries and omits
`test-e2e`'s GPU half. Three numbers, all honest, all reconciled:

| quantity | value | how |
|---|---|---|
| macros, five GPU-only binaries | 829 | what this ledger first said |
| macros, + `test-e2e`'s GPU half (139) | **968** | what `tools/check_test_lanes.py` pins |
| listed tests, same set | 995 | `--gtest_list_tests` on a clean build |

968 -> 995 is 27 `TEST_P` value rows (compute +9, attention +7, quant +6, e2e +5). The gate
pins the **macro** count because that is the one derivable from sources, and it says so in
its own failure message so the two can never be silently compared. Both were re-derived from
two independent states after a stray uncommitted file in the shared working tree was found
compiled into `test-quant`, moving the listed count from 202 to 205 and the total from 995
to 998. A gate whose first-ever value is its own pin has nothing to disagree with, which is
why it is derived twice here.

This is downstream of a recorded decision (`DESIGN_DECISIONS.md:72`, "No GPU runner in CI"),
so the decision is not the finding. The finding is that nothing in the repo *states the
number*, and `make verify-fast` is the single point of failure for all 829.
*Proof it is closed:* the count is asserted somewhere that fails when it grows - the same
shape as `guard_e2e_lane_split` and `guard_det_suite_filter`, which already guard the two
filters that could silently shift a test into the wrong lane.

### (f) `RuntimeConfig` knobs parsed but never read, read but never acted on, documented and absent

```
$ grep -cE '^\s*[BIFS]\("' src/runtime/config.cpp
184
$ <per-leaf read count outside config.{h,cpp}, sorted ascending, head -1>
0 calibration.out_path out_path
$ comm -23 <(imp.conf.example leaf keys) <(config.cpp leaf keys)
0 auto cublas_fp16_acc deterministic dp4a hd
```

**Documented and absent: none.** The five names the diff surfaces are false positives -
`cublas_fp16_acc` and `deterministic` are parsed by the special-case branches at
`config.cpp:229` and `:103` rather than the `B/I/F/S(...)` ladder; `auto`, `dp4a`, `hd` and
`0` come from prose in comment lines. NOT A FINDING.

**Parsed but never read: one (item 7).**

```
$ sed -n '358,363p' src/runtime/config.h
    struct Calibration {
        bool enabled = false;
        // Where imp_calibration_write() puts the file. Empty means the caller
        // supplies the path.
        std::string out_path;
    } calibration;
$ grep -rn "out_path" src/ tools/ --include=*.cpp --include=*.cu --include=*.h | grep -v "config.cpp\|config.h"
(no output)
$ sed -n '377,378p' tools/imp-cli/main.cpp
        if (!args.calibrate_out.empty()) {
            ImpError ce = imp_calibration_write(ctx, args.calibrate_out.c_str());
```

`imp_calibration_write` (`src/api/imp_api.cpp:821`) takes the path as its argument. The
sibling `calibration.enabled` *is* live (`src/runtime/engine_weight_upload.cpp:87`), which is
what makes the dead one look wired. *Proof it is closed:* either the key is removed, or a run
with `calibration.out_path` set and `--calibrate-out` unset writes that file.

**Read but never acted on: none found.** Six leaves whose only textual consumer is
`src/runtime/process_diag.cpp` were checked one by one; five reach a real decision through a
`process_diag_*` accessor (`attention_paged_fp8.cu:619`, `attention_paged.cu:1487`,
`weight_upload.cu:1725`, `nvfp4_gemv_moe.cu:185`). The sixth, `attention.fp8_qk_scaled`,
**looked** dead under a name-symmetric grep and is live at
`src/compute/attention_fmha_sm120.cu:1980` - the accessor is `process_diag_fp8_qk_scaled()`,
not `process_diag_attention_fp8_qk_scaled()`. Recording the near-miss: this is the G1 shape,
and the naming asymmetry is what would make the next sweep report it.

---

## 3. Already dead - do not re-chase

| Considered | Killed where |
|---|---|
| "Architecture dispatch / quant dispatch / loader mapping / KV cache / execution paths / layer primitives / sampling are duplicated" | `SETTLED.md` S-1…S-7, all REFUTED 2026-07-29 |
| "The legacy cuBLAS prefill is a vestige" | `SETTLED.md` S-8 - 0.0 % on hd=128/256; Gemma-4 hd=512 takes it by design |
| "NVFP4 grouped-GEMM has two competing paths" | `SETTLED.md` S-11 - a designed 4-tier ladder, `src/exec/moe_prefill_decision.h` |
| `#if 0` blocks, `_v2`/`_new`/`_old` pairs, stale `wgmma`/`tcgen05`/`sm_100` code, a `VramOwned` type | `SETTLED.md` §C - hunted, absent |
| The `executor_kernels.h` decl-only kernel sweep; the non-kernel decl+def sweep (27 candidates) | `SETTLED.md` §C - fully resolved 2026-08-03, explicitly "do not re-open" |
| `*_reset_static_cuda_state` as dead code | `SETTLED.md` §C R-11 - macro-bound self-registration |
| "Engine teardowns leak ~15 GiB" | `AUDIT.md` B5 → B36 - WSL2/WDDM never returns peak commitment; not imp's to fix |
| `cudaDeviceGraphMemTrim` | `AUDIT.md` B27 - dead, already recorded |
| The six grow-on-demand statics that freed a live graph parameter (B13) | `AUDIT.md` B13 - CLOSED 2026-07-31 by #1139/#1140 |
| `runtime.cuda_graphs=never` inert on dense models | `AUDIT.md` G1 - FIXED 2026-08-20 (#1502) |
| `--vram-budget` overrun; the peak-VRAM gate that was declared and never implemented | `AUDIT.md` B37/B38/B39 - measured and shipped |
| "Device sync per layer in `attention_cublas.cu`"; F-17; F-4's count | `SETTLED.md` §E - the findings were themselves wrong |
| MTP: six drafter-accuracy hypotheses, MoE draft head, unfused verify chunk, the repair forward, the async conditional-graph loop | `roadmap.md` "Buried, so they are not re-run" (`:407-461`) |
| Launch-count framing; the 22.3 % CUTLASS share | `roadmap.md` "Withdrawn by their author" (`:462`) |
| Draft-model speculation, FFN contextual sparsity, BitDecoding, NVFP4 GEMV tuning, FMHA rewrites, cuTile, CompileIQ ptxas tuning | `roadmap.md` "Investigated and shelved" (`:923`) |
| "No GPU runner in CI is debt" | `DESIGN_DECISIONS.md:72` - a decision with a stated cost, not debt. Item 4 is about the *unstated count*, not the decision |
| Splitting a file because it is large | `codebase-audit` skill - split on conflation, and only with a concrete recompile cost |

---

## 4. Provenance

Tree `d40cd394`, working copy clean, 2026-08-21. No GPU was used: every number here comes
from `grep`, `sed`, `python3 tools/check_filesize.py`, `python3 tools/check_alloc_sites.py`,
`codegraph`, or reading the file. Nothing was built and nothing was run on the device, so no
claim here is a measurement of behaviour - every OPEN item's "proof it is closed" is written
as the experiment that would settle it, precisely because this pass could not run one.

---

## (g) The 19 serving-phase allocations the I2 gate found

`make check-alloc-interpose` builds `-DIMP_ALLOC_INTERPOSE=ON`, drives 20 requests at
batch 4 on a config with NVFP4 residual KV and an MTP chain, and reads the interposer's
report. `scripts/check_alloc_interpose.sh` pins the count at **19** and fails in both
directions, so this list is a work queue with a gate behind it rather than a note.

| calls | bytes | site | what it would take |
|---|---|---|---|
| 15 | ~0 | `Engine::try_launch_async_graph_loop`, `src/runtime/engine_graph_decode.cpp:408-412`, `:434` | `d_bt` / `d_token` / `d_pos` / `d_ctx` / `d_banned` are allocated and torn down **per request**. Making `cpipe_` persistent needs it sized from the KV plan at init and the teardown path reworked with it. This is the real violation of the five, and it is the same family as item 1. |
| 2 | 128 MiB | `GraphExecutor::run_attention` -> `chunk_eager_k_` / `_v_`, `src/exec/executor_attention_prefill.cu:176-177` | Grow-only gather scratch for the eager chunked prefill path, sized from the live context on first use. Pre-size it from `ctx_capacity` the way its sibling `chunk_capture_k_`/`_v_` already is. |
| 1 | ~0 | `Engine::banned_tokens_device_`, `src/runtime/engine_graph_decode.cpp:29` | Lazy first-use upload of a list that is known at engine init. The cheapest of the five. |
| 1 | 0.001 MiB | `imp::VRAMAllocator::allocate` | The engine arena growing after the phase flip. Whether that is a defect or a plan one slab short is its own question. |

Two things the first run got wrong, both worth keeping because both are the campaign's
recurring shape:

**The violation report was `IMP_LOG_DEBUG`.** Someone who opted into the measurement build
still saw nothing at default log level. It is `IMP_LOG_WARN` now, and the clean line is
`IMP_LOG_INFO`, because the gate asserts that **one of the two appears at all**: absent
both, the binary was built without the flag and a grep for violations passes for the wrong
reason.

**The gate itself first reported 2 allocations when there were 19.** The report's banner
and its first class were on the same line (no `\n` after `while serving:`), and the parser
anchored at the start of a line, so it skipped `cudaMalloc` entirely and summed only the
async and pinned rows. Both halves are fixed: the format has its newline, and the parser
matches the class name anywhere in the line rather than depending on that.

The gate also refuses to judge before it can prove it reached the path it claims to
exercise: it fails if `residual buffer enabled` is absent from the log.

---

## (h) The release blocker is defined over seven heroes and observed on two

`GOAL.md` says: *"If a hero model regresses against any competitor on the primary
metric, that is a release blocker."* The hero set has seven entries. Two of them are
pinned by a gate:

```
tests/perf_baseline.json             -> Qwen3-8B-Q8_0.gguf
tests/perf_baseline_north_star.json  -> Qwen3-14B-Q6_K.gguf
tests/perf_baseline_chunked.json     -> no model pinned
```

Qwen3-Coder-30B-A3B, Qwen3.6-35B-A3B, **Gemma-4-26B-A4B**, Nemotron-H and gpt-oss-20b are
measured by nothing. The only instrument that has ever looked at them is the competitive
sweep, which before 2026-08-21 had last run on 2026-07-12.

This is a coverage gap, not a calibration one, and the distinction decides where the fix
goes. The 8 % threshold never entered into it: no gate ran on the model, so no threshold
was applied. Recording it as "5.3 % slipped under 8 %" would send the next reader to argue
about a number that was never consulted.

`make bench-competitive` (#1518) is the instrument that closes it, and it is wired into
`scripts/check-release.sh` as stage 9 rather than into `verify-fast`, because the blocker is
defined at release scope and a per-PR sweep is not warranted for it. `RELEASE_BAR=1` fails
on any hero under a 5 % decode lead and names which and by how much.

**Five of the seven heroes are measurable against llama.cpp; two are not, and the gate says
so.** Qwen3-Coder-30B-A3B and Nemotron-H are NVFP4-only on this host and llama.cpp has no
NVFP4 path on sm_120, so no shared-quant comparison exists. The script prints them with
that reason on every run. That is deliberate: the failure this section records is five
heroes being unmeasured *without anything saying so*, and a stage that quietly omitted the
two it cannot reach would reproduce it at smaller scale.

### Two open shapes recorded here rather than chased

**An unexplained +5 tok/s on Gemma-4 between window index 237 and 476.** The scan that
located the `63df2d30` trade also mapped the rest of the window, and it is flat to the
resolution of the instrument for 178 commits and then recovers:

| window index | commit | Gemma-4 tg128 |
|---|---|---|
| 0 | `7811658a` | 258.96 |
| 59 | `c5936db7` | 241.04 |
| 119 | `5c9f6c07` | 241.37 |
| 178 | `5408eb5c` | 240.41 |
| 237 | `9f0322c8` | 240.21 |
| 476 | `6e503cca` | 245.24 |

Indices 59 through 237 span 1.16 tok/s, 0.48 %, against a within-arm spread of 0.05 to
0.4 % on this model. Something after index 237 gives back ~5 tok/s and nothing records
what. Not chased: the question it belonged to ("why is Gemma-4 slow") dissolved when the
drop turned out to be a priced trade.

**The `is_dense=false` arm has now been checked on a second model and was right there.**
`pre_dequant_internal.h` records that this arm "silently voided that flag on quantized
hybrids and cost the hero -5% decode", repaired for GDN hybrids only. Gemma-4 falls through
the same arm and is neither dense nor GDN, so it looked like a second instance of the same
defect. It is not: the trade is genuinely losing for it (+7.4 % decode against +9.0 % PPL).
The generalisation still stands for the members of that category nobody has measured, but
it now has one instance where the categorical call was correct.

---

## (j) Every gate this campaign shipped is advisory, and one of them proved it

`main` went red on 2026-08-21 and stayed red: #1523's `File size` check **FAILED and the PR
merged anyway**.

```
$ gh pr checks 1523 --json name,state
FAILURE  File size
SUCCESS  Build
SUCCESS  Alloc sites, Docs, Lint, Launch guards, clang-tidy,
         Mock API contract, Real API contract, Release hygiene, enable-auto-merge
SKIPPED  Test
```

Ruleset `14716423` ("Require CI", enforcement active) requires exactly **one** context:

```
$ gh api repos/kekzl/imp/rulesets/14716423 \
    --jq '.rules[] | select(.type=="required_status_checks")
          | .parameters.required_status_checks[].context'
Build
```

| check | required | state on #1523 |
|---|---|---|
| **Build** | **yes** | SUCCESS |
| **File size** | no | **FAILURE** |
| Alloc sites | no | SUCCESS |
| Docs | no | SUCCESS |
| Lint | no | SUCCESS |
| Launch guards | no | SUCCESS |
| clang-tidy | no | SUCCESS |
| Mock API contract | no | SUCCESS |
| Real API contract | no | SUCCESS |
| Release hygiene | no | SUCCESS |
| Test | no | SKIPPED |
| enable-auto-merge | no | SUCCESS |

Eleven of twelve are advisory, and auto-merge squashes on mergeability plus the one required
context. So `check_alloc_pairs.py` (#1505), `check_test_lanes.py` and
`check_dead_inline_accessors.py` (#1506), `check_log_fatal.py` (#1511) and
`check_alloc_interpose.sh` (#1520) are all correct, all red-tested, and **none of them can
stop a merge.** A step that blocks its own job blocks nothing when the job is not required.

This is the campaign's own defect shape one level up: the property that was verified is
"does the gate go red", and the property that matters is "does red stop anything". Every one
of these gates had the first tested and none had the second.

**Deliberately not fixed here.** Adding required contexts changes what can merge for every
future PR in this repo, which is the repo owner's decision. Until it is made, this entry is
the honest statement of what these gates are: **documentation with a red light, not
enforcement.** `.claude/skills/shipping-prs/SKILL.md` already records the same fact from the
other direction ("every other job can go red and the PR still merges", with the two 2026-08
`Alloc sites` incidents); what is new here is the enumeration and a case where it happened
to a gate written the same day.

Query note, because the obvious one is wrong: `gh api repos/kekzl/imp/branches/main/protection`
returns **404 "Branch not protected"**. That is not the answer. Protection here is configured
via **rulesets**, which the legacy protection endpoint does not see.
