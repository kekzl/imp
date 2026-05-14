# MTP Phase 2.2 + 3.5 + 5.5 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take MTP from "end-to-end scaffolding works" (post PR #172) to "production-grade speculative decoding with measured net win" — implement the full MTP transformer block, auto-invoke draft+verify from the decode loop, validate acceptance rate.

**Architecture:** PR #172 shipped weight loading + reduced FC-only forward + engine API + CLI flag + smoke. Three concrete open work items remain. Phase 2.2 fills in the transformer block in `mtp_forward.cu`. Phase 3.5 wires drafts into the decode loop with verify-accept logic. Phase 5.5 measures.

**Tech Stack:** Same as before — C++20 + CUDA sm_120a, reuses existing attention + MoE kernels where feasible, no new third-party deps.

**Spec:** `docs/superpowers/specs/2026-05-14-mtp-wiring-design.md`

**Predecessor work (shipped on main):**
- PR #171 (`536af79`) — Phase 1.A: detection + scaffolding
- PR #172 (`b1f74e0`) — Phases 1.B + 1.C + 2.1 + 3 + 4 + 5: full scaffolding, reduced forward

**Memory:** [`mtp_phase2_open_2026_05_14`](../../../memory/mtp_phase2_open_2026_05_14.md)

---

## Phase 2.2 — Full Transformer Block (multi-week)

**Outcome:** `src/runtime/mtp_forward.cu:186-190` no-op placeholder replaced with full attention + 256-expert MoE block. Draft logits become close-to-optimal (matching DeepSeek-V3 paper §4.5 expectations).

### Design fork

The shipped `MtpHead` uses **flat individual tensors** (q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, router, experts_gate_up_packed [256,1024,2048], experts_down_packed [256,2048,512], shared_expert_*). The main model's `TransformerLayer` uses per-expert vectors `expert_w_gate[256]` + `expert_w_up[256]` + `expert_w_down[256]`. Two paths:

**Path A: TransformerLayer-view adapter**
- Add `TransformerLayer mtp_layer_view_` field to `MtpHead`, populated as Tensor views of the same device buffers (no extra allocation).
- Slice `experts_gate_up_packed[256, 1024, 2048]` into 256 × `expert_w_gate[512, 2048]` + 256 × `expert_w_up[512, 2048]` views.
- Refactor `GraphExecutor::run_attention(int layer, ...)` and `run_moe_ffn(int layer, ...)` to a `_on(const TransformerLayer&, ...)` overload that takes the layer by reference.
- MTP forward calls `executor_->run_attention_on(mtp.layer_view, state, stream)` then `run_moe_ffn_on(...)`.
- Cost: medium refactor of two existing functions + their internal layer-index-driven callers.

**Path B: From-scratch fused kernels**
- Write new attention kernel that reads from `mtp.q_proj` / `mtp.k_proj` / `mtp.v_proj` / `mtp.o_proj` directly.
- Write new 256-way MoE forward against the packed 3D `experts_gate_up_packed` / `experts_down_packed`.
- Cost: high — the 256-expert MoE forward is the bulk of multi-week budget.

**Recommended path:** A (adapter). It reuses existing battle-tested kernels (Qwen3-Coder NVFP4 MoE 266 tok/s with graphs is the reference) and limits new code to a view-population helper.

### Task list (Path A)

#### Task 2.2.A.1 — Add `TransformerLayer mtp_layer_view_` to MtpHead

**Files:**
- Modify: `src/model/mtp_head.h`

- [ ] **Step 1: Add the field**

```cpp
// In MtpHead struct, after final_norm:

// TransformerLayer view: populated by populate_layer_view() so the existing
// run_attention/run_moe_ffn kernels can be invoked against MTP weights.
// All Tensor handles inside are device-side views of the flat fields above —
// no extra allocations.
TransformerLayer mtp_layer_view;
```

- [ ] **Step 2: Add `populate_layer_view()` declaration**

```cpp
// Free function (not method — TransformerLayer lives elsewhere). Builds
// view Tensors that point into the flat MtpHead fields, including slicing
// the 256-expert packed tensors. Idempotent. Returns false if any expected
// flat field is unloaded.
bool mtp_populate_layer_view(MtpHead& head);
```

- [ ] **Step 3: Implement in mtp_head.cpp (new file)**

(Implementation details TBD by executor — reuse the slicing pattern from `src/model/mtp_loader.cpp` in the abandoned fix/open-bugs branch, lines 226-249, or the existing slicing in `src/model/weight_upload.cu:1590-1647` for Gemma-4 fused gate_up.)

- [ ] **Step 4: Call from safetensors_loader after MTP load succeeds**

- [ ] **Step 5: Unit test** — populate, then verify `head.mtp_layer_view.wq.data == head.q_proj.data`, expert views point at expected offsets within `head.experts_gate_up_packed.data`.

- [ ] **Step 6: Commit + verify-fast green**

#### Task 2.2.A.2 — Refactor `run_attention` to take `const TransformerLayer&`

**Files:**
- Modify: `src/graph/executor.h` — add `run_attention_on(const TransformerLayer&, const InferenceState&, cudaStream_t)`
- Modify: `src/graph/executor_attention.cu` — extract body of `run_attention(int layer, ...)` into the new overload; the old form becomes `run_attention_on(model_->layer(layer), ...)`

- [ ] Refactor signature, run all attention tests green, verify-fast green.

#### Task 2.2.A.3 — Refactor `run_moe_ffn` to take `const TransformerLayer&`

Mirror of 2.2.A.2 for the MoE function. `src/graph/executor_forward_moe.cu`.

- [ ] Refactor, all MoE tests + Qwen3-Coder NVFP4 / Gemma-4 NVFP4 / Qwen3.6-NVFP4 smoke green, verify-fast green.

#### Task 2.2.A.4 — Wire MTP forward to use the layer view

**Files:**
- Modify: `src/runtime/mtp_forward.cu:186-190`

Replace the no-op Step 5 with:

```cpp
// Step 5: full transformer block (Phase 2.2).
// fc_out is the layer input. We need an InferenceState-like context for
// the attention call — but MTP runs on a single token in a single-step
// batched form. Build a minimal state and reuse the existing executor.
{
    // Construct or borrow an InferenceState for n_tokens=1.
    InferenceState mtp_state;
    mtp_state.n_tokens = 1;
    // ... populate position_ids, kv_block_table_, etc. from a dedicated
    //     MTP KV slot (allocated separately to avoid polluting the main
    //     model's KV cache).
    executor_->run_attention_on(mtp.mtp_layer_view, mtp_state, stream);
    executor_->run_moe_ffn_on(mtp.mtp_layer_view, stream);
}
```

This task requires designing the **MTP KV cache** — see Task 2.2.A.5.

#### Task 2.2.A.5 — Allocate MTP KV cache slot

**Files:**
- Modify: `src/memory/kv_cache_manager.{h,cpp}` — extend with an auxiliary single-layer pool sized for `max_seq_len × n_kv_heads × head_dim`

Rationale: the MTP layer has its own attention. If we reuse the main model's KV cache, drafted tokens pollute the main cache and require rollback on rejection. A separate MTP KV slot keeps draft + main state cleanly isolated.

#### Task 2.2.A.6 — Golden-value test

**Files:**
- Create: `scripts/mtp_reference.py` — HF transformers reference that runs the Qwen3.6 MTP head against 32 random hidden states + tokens and dumps expected output tensors.
- Create: `tests/test_mtp_forward_golden.cu` — load reference dump, run imp's `mtp_draft_step`, assert cosine-sim ≥ 0.99 on output logits.

Tests Path A is correctly mirroring the upstream model.

#### Task 2.2.A.7 — verify-fast + Qwen3.6-NVFP4 smoke

`make verify-fast` green; `imp-cli --model /home/kekz/models/Qwen3.6-35B-A3B-NVFP4 --mtp-spec-decode 2 --bench` produces coherent output; no main-model perf regression.

---

## Phase 3.5 — Auto-invoke Draft + Verify from Decode Loop (~3-5 days)

**Outcome:** When `--mtp-spec-decode K` is set, the decode loop automatically calls `mtp_draft_one` K times then runs a batched verify forward and accepts the longest prefix matching argmax.

**Files:**
- Modify: `src/runtime/engine.cpp` — `step_decode_continuous()` branches when `mtp_spec_decode_enabled()`
- Possibly modify: `src/graph/executor.h` — add `forward_batch_for_verify(...)` variant that returns per-position logits
- Modify: `src/memory/kv_cache_manager.{h,cpp}` — accept-prefix-only KV commit primitive (see Task 2.2.A.5)

### Task list

#### Task 3.5.1 — Draft loop helper

`std::vector<int32_t> Engine::mtp_draft_k(int prev_token, const void* d_h_prev)` — calls `mtp_draft_one` K times, feeding back the previous drafted token. Returns the K draft tokens.

#### Task 3.5.2 — Batched verify forward

The main model needs to run on `[prev_token, draft_0, ..., draft_{K-1}]` (K+1 positions) in one batched call. The current `forward_logits` does this; we just need to capture per-position logits.

#### Task 3.5.3 — Accept-prefix selection

For greedy: accept draft_i iff `argmax(verify_logits[i]) == draft_i`. Stop at first mismatch. Sample the verify-position-N+1 token from `verify_logits[accepted_count]` as the bonus.

#### Task 3.5.4 — KV state machine

Only commit KV entries for accepted positions (`prev_token` + accepted drafts). Reject KV entries for unaccepted drafts. Probably easiest: write all K+1 KV entries during verify forward to a scratch slot, then memcpy only the accepted prefix to the real cache.

#### Task 3.5.5 — Sampling support (non-greedy)

DeepSeek-V3 paper §4.5 describes the probabilistic accept rule for sampling. Initial ship targets greedy only; extend later.

#### Task 3.5.6 — Multi-sequence support

Continuous-batching path. Initial ship can gate spec-decode to batch=1; multi-seq is a follow-up.

---

## Phase 5.5 — Acceptance-Rate Validation (~3-5 days)

**Outcome:** Measured acceptance rate, tok/s vs baseline, coherence vs baseline. Default-on/off decision.

### Task list

#### Task 5.5.1 — A/B matrix harness

`scripts/mtp_bench.sh` wrapping `imp-bench` over the matrix:
- K ∈ {1, 2, 3}
- prompts ∈ {factual, verbose-think, code, instruction}
- length ∈ {64, 256, 1024}

Captures tok/s + accept-rate per cell. Output CSV.

#### Task 5.5.2 — Coherence smoke

`check-degeneration` skill against MTP-on and MTP-off; outputs must be coherent in both modes.

#### Task 5.5.3 — Decide default

Threshold: net throughput ≥ 1.3× baseline AND accept rate ≥ 60% on ≥ 3/4 prompt classes → default-on. Else: keep opt-in via `--mtp-spec-decode K`.

#### Task 5.5.4 — Document findings

`memory/mtp_phase5_validation_<date>.md` regardless of outcome. Update `MEMORY.md` index.

---

## Cross-cutting concerns

### Performance gates
- All phases gated by `make verify-fast` (no Qwen3-4B Q8 baseline regression)
- Phase 5.5 also runs Qwen3.6-NVFP4 tg256 baseline (current 117–142 tok/s post-PR #95)

### Memory updates
- After Phase 2.2 → `mtp_phase2_2_forward_<date>.md`: which kernels reused, golden-value RMS, perf
- After Phase 3.5 → `mtp_phase3_5_verify_<date>.md`: accept-rule, KV state machine
- After Phase 5.5 → `mtp_phase5_validation_<date>.md`: A/B numbers, default-on/off decision

### Out-of-scope
- DeepSeek V3 / V3.1 specifically (different attention architecture)
- Multimodal MTP variants (image/audio conditioning)
- Distributed/multi-GPU
- Sampling-mode accept rule (greedy first; sampling is 3.5 follow-up)

---

## Self-review

**Spec coverage:**
- ✅ Phase 2.2 full transformer block → Tasks 2.2.A.1 – 2.2.A.7
- ✅ Phase 3.5 auto-invoke → Tasks 3.5.1 – 3.5.6
- ✅ Phase 5.5 validation → Tasks 5.5.1 – 5.5.4

**Placeholder scan:**
- Task 2.2.A.1 Step 3 (`populate_layer_view` implementation) and Task 3.5.4 (KV state machine) are intentionally outlined-not-detailed because the right shapes depend on existing code surfaces that should be re-inspected when the task is picked up. Path A vs B is the load-bearing decision; once that's made, the detailed steps follow mechanically.

**Risk markers:**
- Path A's `run_attention_on` refactor touches a hot function with many internal callers. Mitigation: keep the int-layer overload as a thin shim around the by-ref form.
- MTP KV cache (Task 2.2.A.5) adds VRAM cost: `max_seq_len × n_kv_heads × head_dim × 2 bytes`. For Qwen3.6 16K context: 16384 × 2 × 256 × 2 = 16 MiB. Negligible.
- Phase 3.5 batched verify may need graph capture work — currently graphs run on n=1 fast-path; n=K+1 may hit a different code path with different perf characteristics.
