# Chunked Prefill — Hybrid GDN+MoE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable chunked prefill on hybrid GDN+MoE / Mamba2+MoE models (Qwen3.5/3.6, Nemotron-H) so prompts longer than the executor's per-step `max_tokens` cap (256/512) no longer hit `RequestStatus::CANCELLED`.

**Architecture:** Two-part change. (1) Patch the Mamba2 plain conv1d prefill kernel (`src/compute/ssm.cu:196-233`) to read the previous chunk's `conv_state` instead of zero-padding — mirrors the chunked-prefill branch already present in the f32_silu fused variant. (2) Drop the four hybrid-arch carve-outs in `Engine::supports_chunked_prefill_()` (`src/runtime/engine.cpp:1697-1700`). Everything else (SSM/GDN scan state persistence, attn_scores capacity guard, KV write offset awareness) is already chunk-safe — verified by code reading in the spec.

**Tech Stack:** C++20, CUDA 13.2 (sm_120a), GTest, Docker (`make build`/`make test-gpu`).

**Spec:** `docs/superpowers/specs/2026-05-09-chunked-prefill-hybrid-gdn-moe-design.md`

---

## File Map

| File | Role |
|---|---|
| `tests/test_ssm.cu` (modify) | Append `SSMConv1dTest.ChunkedPrefillEquivalence` — fails on current code, passes after kernel patch |
| `src/compute/ssm.cu` (modify, lines 196-233) | Add `conv_state` read branch for `src_t < 0` to `ssm_conv1d_prefill_kernel` |
| `src/runtime/engine.cpp` (modify, lines 1697-1700) | Drop four hybrid arch returns from `supports_chunked_prefill_()` |
| `src/runtime/engine.h` (modify, lines ~244-247) | Update doc-comment on `supports_chunked_prefill_()` to reflect new scope |
| `docs/roadmap.md` (modify, lines 20-32) | Remove "Hybrid models" from out-of-scope list; mention shipped in CHANGELOG |
| `tests/perf_baseline.json` (regenerate) | Refresh hybrid-model decode/prefill numbers |
| `CHANGELOG.md` (append) | Add entry for this change |

---

## Task 1: Mamba2 conv kernel chunked-prefill — failing test (TDD red)

**Why first:** Pure CUDA kernel, smallest unit, fastest feedback loop. Test reproduces the bug before we know it exists in CI.

**Files:**
- Modify: `tests/test_ssm.cu` (append after existing `SSMConv1dTest.StateConsistency` at line ~259)

- [ ] **Step 1.1: Append the failing test**

Append to the end of `tests/test_ssm.cu` (before the closing `}  // namespace` and `}  // namespace imp`):

```cpp
// ===========================================================================
// Test: Chunked prefill equivalence — splitting a sequence across two
// ssm_conv1d_prefill calls (with conv_state threaded between them) must
// produce identical output to a single full-sequence call. Catches the
// zero-pad-instead-of-conv_state-read bug at chunk boundary.
// ===========================================================================
TEST(SSMConv1dTest, ChunkedPrefillEquivalence) {
    SKIP_IF_NO_CUDA();

    constexpr int channels = 4;
    constexpr int kernel_size = 4;
    constexpr int n_chunk_a = 5;
    constexpr int n_chunk_b = 5;
    constexpr int n_total = n_chunk_a + n_chunk_b;

    // Random-ish input
    std::vector<float> h_x(n_total * channels);
    for (int i = 0; i < n_total * channels; i++)
        h_x[i] = std::sin(static_cast<float>(i) * 0.7f) * 2.0f;

    // Non-uniform weight (so any boundary-zero bug shows up)
    std::vector<float> h_w(channels * kernel_size);
    for (int i = 0; i < channels * kernel_size; i++)
        h_w[i] = (i % 2 == 0) ? 0.3f : -0.5f;

    // ---- Reference: single full-sequence prefill ----
    float* d_state_full;
    cudaMalloc(&d_state_full, channels * kernel_size * sizeof(float));
    cudaMemset(d_state_full, 0, channels * kernel_size * sizeof(float));

    Tensor d_x_full = make_fp16_gpu(h_x.data(), {n_total, channels});
    Tensor d_w_full = make_fp16_gpu(h_w.data(), {channels, kernel_size});
    Tensor d_out_full = alloc_fp16_gpu({n_total, channels});
    Tensor d_bias = make_empty_tensor();

    ssm_conv1d_prefill(d_state_full, d_x_full, d_w_full, d_bias, d_out_full, kernel_size, nullptr);
    cudaDeviceSynchronize();

    auto out_full = read_fp16(d_out_full);
    std::vector<float> state_full(channels * kernel_size);
    cudaMemcpy(state_full.data(), d_state_full, channels * kernel_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    // ---- Chunked: chunk A then chunk B, threading conv_state ----
    float* d_state_chunked;
    cudaMalloc(&d_state_chunked, channels * kernel_size * sizeof(float));
    cudaMemset(d_state_chunked, 0, channels * kernel_size * sizeof(float));

    Tensor d_x_a = make_fp16_gpu(h_x.data(), {n_chunk_a, channels});
    Tensor d_w_a = make_fp16_gpu(h_w.data(), {channels, kernel_size});
    Tensor d_out_a = alloc_fp16_gpu({n_chunk_a, channels});
    ssm_conv1d_prefill(d_state_chunked, d_x_a, d_w_a, d_bias, d_out_a, kernel_size, nullptr);
    cudaDeviceSynchronize();

    Tensor d_x_b = make_fp16_gpu(h_x.data() + n_chunk_a * channels, {n_chunk_b, channels});
    Tensor d_w_b = make_fp16_gpu(h_w.data(), {channels, kernel_size});
    Tensor d_out_b = alloc_fp16_gpu({n_chunk_b, channels});
    ssm_conv1d_prefill(d_state_chunked, d_x_b, d_w_b, d_bias, d_out_b, kernel_size, nullptr);
    cudaDeviceSynchronize();

    auto out_a = read_fp16(d_out_a);
    auto out_b = read_fp16(d_out_b);
    std::vector<float> state_chunked(channels * kernel_size);
    cudaMemcpy(state_chunked.data(), d_state_chunked, channels * kernel_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    // ---- Compare ----
    // First chunk_a tokens of full == out_a
    for (int t = 0; t < n_chunk_a; t++) {
        for (int ch = 0; ch < channels; ch++) {
            EXPECT_NEAR(out_full[t * channels + ch], out_a[t * channels + ch], 1e-2f)
                << "Chunk A mismatch at t=" << t << " ch=" << ch;
        }
    }
    // Tokens [n_chunk_a, n_total) of full == out_b — this is where the bug bites
    for (int t = 0; t < n_chunk_b; t++) {
        for (int ch = 0; ch < channels; ch++) {
            EXPECT_NEAR(out_full[(n_chunk_a + t) * channels + ch], out_b[t * channels + ch], 1e-2f)
                << "Chunk B mismatch at t=" << t << " ch=" << ch;
        }
    }
    // Final state should match
    for (int i = 0; i < channels * kernel_size; i++) {
        EXPECT_NEAR(state_full[i], state_chunked[i], 1e-2f)
            << "State mismatch at i=" << i;
    }

    cudaFree(d_state_full);
    cudaFree(d_state_chunked);
    free_tensor(d_x_full);
    free_tensor(d_w_full);
    free_tensor(d_out_full);
    free_tensor(d_x_a);
    free_tensor(d_w_a);
    free_tensor(d_out_a);
    free_tensor(d_x_b);
    free_tensor(d_w_b);
    free_tensor(d_out_b);
}
```

- [ ] **Step 1.2: Build and run the test (expect FAIL)**

Run (via the canonical Makefile target — discovers the test binary that hosts `SSMConv1dTest`):
```bash
make build && make test-gpu GTEST_FILTER='SSMConv1dTest.ChunkedPrefillEquivalence'
```

If the Makefile doesn't accept `GTEST_FILTER`, fall back to:
```bash
make build && docker run --rm --gpus all -v "$PWD":/imp -w /imp imp:test \
    bash -c "cmake --build build -j\$(nproc) && \
             ctest --test-dir build -R SSMConv1dTest.ChunkedPrefillEquivalence --output-on-failure"
```

Expected: FAIL with "Chunk B mismatch at t=0 ch=..." (kernel zero-pads at chunk boundary; chunk B's first `kernel_size-1` tokens see zeros instead of chunk A's trailing values).

---

## Task 2: Mamba2 conv kernel chunked-prefill — implementation (TDD green)

**Files:**
- Modify: `src/compute/ssm.cu` lines 196-233 (`ssm_conv1d_prefill_kernel`)

- [ ] **Step 2.1: Patch `ssm_conv1d_prefill_kernel`**

In `src/compute/ssm.cu`, locate the inner conv loop in `ssm_conv1d_prefill_kernel` (around line 207-218):

Current code:
```cuda
for (int ch = threadIdx.x; ch < channels; ch += blockDim.x) {
    float sum = 0.0f;

    for (int k = 0; k < kernel_size; k++) {
        int src_t = token - (kernel_size - 1) + k;
        float val = 0.0f;
        if (src_t >= 0) {
            val = __half2float(x_in[src_t * channels + ch]);
        }
        // Weight layout: [channels, kernel_size] — kernel_size is contiguous per channel
        sum += val * __half2float(weight[ch * kernel_size + k]);
    }
```

Replace the inner `for (int k...)` with the conv_state-aware version (mirror of `ssm_conv1d_prefill_f32_silu_kernel` lines 262-277):

```cuda
for (int k = 0; k < kernel_size; k++) {
    int src_t = token - (kernel_size - 1) + k;
    float val;
    if (src_t >= 0) {
        val = __half2float(x_in[src_t * channels + ch]);
    } else if (conv_state) {
        // Chunked prefill: read trailing context from previous chunk's
        // conv_state instead of zero-padding. conv_state[ch*K + s] holds
        // the input at global position (chunk_offset - K + s).
        int state_idx = src_t + kernel_size;  // maps to [1..K-1]
        val = (state_idx >= 0 && state_idx < kernel_size)
                  ? conv_state[ch * kernel_size + state_idx]
                  : 0.0f;
    } else {
        val = 0.0f;
    }
    // Weight layout: [channels, kernel_size] — kernel_size is contiguous per channel
    sum += val * __half2float(weight[ch * kernel_size + k]);
}
```

The state-write block at lines 224-231 needs no change — it already writes the trailing `kernel_size` tokens of the current chunk to `conv_state` using `x_in`, which becomes the chunk-A trailing context for chunk B.

- [ ] **Step 2.2: Run the test (expect PASS)**

Run:
```bash
make build && docker run --rm --gpus all -v "$PWD":/imp -w /imp imp:test \
    bash -c "cmake --build build -j\$(nproc) && \
             ctest --test-dir build -R SSMConv1dTest --output-on-failure"
```

Expected: all 5 SSMConv1dTest tests PASS, including the new `ChunkedPrefillEquivalence`. The pre-existing `PrefillCausal` test still passes because it starts from a zeroed `conv_state` and `state_idx ∈ [0, K-1]` reads `0.0f` from the zeroed state (numerically identical to zero-pad fallback).

- [ ] **Step 2.3: Commit kernel fix + test**

```bash
git add src/compute/ssm.cu tests/test_ssm.cu
git commit -m "$(cat <<'EOF'
fix(ssm): chunked-prefill conv_state read in Mamba2 conv1d kernel

ssm_conv1d_prefill_kernel zero-padded input when src_t < 0, losing the
last (kernel_size - 1) tokens of the previous chunk on chunk N>0. Port
the conv_state read branch from ssm_conv1d_prefill_f32_silu_kernel:
when src_t < 0 and conv_state is present, read from the persisted
trailing-context buffer instead of zero-padding.

Adds SSMConv1dTest.ChunkedPrefillEquivalence regression test: split a
sequence across two prefill calls threading conv_state, compare to
single full-sequence call element-wise.

Required for chunked prefill on Nemotron-H Mamba2+MoE; no effect on
single-chunk prefill (pre-existing PrefillCausal test still passes).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Engine carve-out removal

**Files:**
- Modify: `src/runtime/engine.cpp` lines 1689-1709
- Modify: `src/runtime/engine.h` doc-comment near line 244

- [ ] **Step 3.1: Update `supports_chunked_prefill_()` body**

In `src/runtime/engine.cpp` around line 1689:

Current code:
```cpp
bool Engine::supports_chunked_prefill_() const {
    if (!model_)
        return false;
    const auto& cfg = model_->config();
    // Out-of-scope archs: SWA / dual-head_dim / hybrid (GDN / Mamba2).
    if (cfg.arch == ModelArch::GEMMA3) return false;       // SWA (sliding_window_pattern=6)
    if (cfg.arch == ModelArch::GEMMA4) return false;       // SWA + dual head_dim
    if (cfg.arch == ModelArch::LLAMA4) return false;       // MoE + SWA, untested
    if (cfg.arch == ModelArch::QWEN35) return false;
    if (cfg.arch == ModelArch::QWEN35_MOE) return false;
    if (cfg.arch == ModelArch::QWEN36_MOE) return false;
    if (cfg.arch == ModelArch::NEMOTRON_H_MOE) return false;
    // KV dtypes wired through paged_kv_gather: FP16, FP8_E4M3, NVFP4. Others
    // (INT4/INT8/TurboQuant) would need their own gather kernels.
    if (kv_cache_raw_) {
        QType kvt = kv_cache_raw_->qtype();
        if (kvt != QType::F16 && kvt != QType::FP8_E4M3 && kvt != QType::NVFP4)
            return false;
    }
    return true;
}
```

Replace with:
```cpp
bool Engine::supports_chunked_prefill_() const {
    if (!model_)
        return false;
    const auto& cfg = model_->config();
    // Out-of-scope archs: SWA / dual-head_dim variants. Hybrid GDN+MoE /
    // Mamba2+MoE archs (QWEN35*, QWEN36_MOE, NEMOTRON_H_MOE) ARE supported —
    // their attention layers share one (nkv, hd) geometry, the existing
    // chunked-attention path handles them, and SSM/GDN/Mamba2 forward kernels
    // persist state across chunks.
    if (cfg.arch == ModelArch::GEMMA3) return false;       // SWA (sliding_window_pattern=6)
    if (cfg.arch == ModelArch::GEMMA4) return false;       // SWA + dual head_dim
    if (cfg.arch == ModelArch::LLAMA4) return false;       // MoE + SWA, untested
    // KV dtypes wired through paged_kv_gather: FP16, FP8_E4M3, NVFP4. Others
    // (INT4/INT8/TurboQuant) would need their own gather kernels.
    if (kv_cache_raw_) {
        QType kvt = kv_cache_raw_->qtype();
        if (kvt != QType::F16 && kvt != QType::FP8_E4M3 && kvt != QType::NVFP4)
            return false;
    }
    return true;
}
```

- [ ] **Step 3.2: Update `engine.h` doc-comment**

In `src/runtime/engine.h` around line 244:

Current comment:
```cpp
// Returns true for full-attention models (Qwen3, Llama, Mistral) with FP16
// or FP8 KV cache. Returns false for Gemma-4 (SWA + dual head_dim), hybrid
// models (GDN/Mamba2), and sub-byte KV dtypes.
bool supports_chunked_prefill_() const;
```

Replace with:
```cpp
// Returns true for full-attention models (Qwen3, Llama, Mistral) and
// hybrid GDN+MoE / Mamba2+MoE models (Qwen3.5/3.6, Nemotron-H) with FP16,
// FP8, or NVFP4 KV cache. Returns false for SWA archs (Gemma-3/4, Llama-4)
// and sub-byte KV dtypes lacking gather kernels (INT4, TurboQuant).
bool supports_chunked_prefill_() const;
```

- [ ] **Step 3.3: Build and run unit suite**

```bash
make build && docker run --rm --gpus all -v "$PWD":/imp -w /imp imp:test \
    bash -c "cmake --build build -j\$(nproc) && \
             ctest --test-dir build -R '(SSM|Engine)' --output-on-failure"
```

Expected: all green. (No new unit test for the engine change — verified end-to-end in Task 4.)

- [ ] **Step 3.4: Commit engine carve-out removal**

```bash
git add src/runtime/engine.cpp src/runtime/engine.h
git commit -m "$(cat <<'EOF'
feat(engine): enable chunked prefill on hybrid GDN+MoE archs

Drop QWEN35 / QWEN35_MOE / QWEN36_MOE / NEMOTRON_H_MOE returns from
supports_chunked_prefill_(). The roadmap framing ("needs per-layer-shape
aware paged-prefill kernel") was Gemma-4 specific. For these four hybrid
archs:

- attention layers share one (nkv, hd) geometry — no n_kv_heads_per_layer
  / head_dim_per_layer overrides
- existing chunked-attention path in executor_attention.cu (post PR #149)
  supports them — kvt_ok && !sliding && !per_layer_shapes all hold
- ssm_scan_kernel / gdn_scan_fused_kernel / ssm_conv1d_prefill_f32_silu
  persist state across chunks naturally
- ssm_conv1d_prefill (Mamba2 plain conv) chunked-prefill conv_state read
  is fixed in the previous commit

The dead cancellation block at engine.cpp:1772 stays in place — it now
only fires for Gemma-3/4/Llama-4, which remain out of scope.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: End-to-end smoke verification on real models

**Goal:** Confirm long prompts on hybrid models (a) no longer get CANCELLED and (b) produce coherent output that survives multi-chunk SSM-state continuation.

**Files:** None modified. This task is gated by external model files in `/home/kekz/models/` and `$REPO/models/`.

- [ ] **Step 4.1: Smoke run — Qwen3.5-4B-Q8_0 long prompt**

Build a >256-token prompt to a temp file (the SSM/MoE workspace cap), then run via `imp-cli`:

```bash
# Build a prompt of ~350 tokens (50 repeats of 7-word sentence ≈ 350 words)
yes 'The quick brown fox jumps over the lazy dog. ' | head -50 | tr -d '\n' > /tmp/long_prompt.txt
echo $'\nContinue this story:' >> /tmp/long_prompt.txt

docker run --rm --gpus all \
    -v "$PWD/models":/models -v "$PWD":/imp -v /tmp:/tmp -w /imp imp:test \
    ./build/tools/imp-cli/imp-cli \
        --model /models/Qwen3.5-4B-Q8_0.gguf \
        --prompt-file /tmp/long_prompt.txt \
        --max-tokens 64 --temperature 0
```

Note: the `imp-cli` binary path may be `./build/tools/imp-cli` or `./build/tools/imp-cli/imp-cli` depending on CMake target layout — check `ls build/tools/imp-cli/` if the first form fails.

Expected:
- No `RequestStatus::CANCELLED` in stderr
- No `chunked_prefill: unsupported config` abort
- No `attn_scores_ capacity` abort
- Coherent continuation generated (decode produces sensible English)

If any abort: capture stderr, identify the failing guard, and revisit Task 2/3.

- [ ] **Step 4.2: Smoke run — Nemotron-H Mamba2+MoE long prompt** (only if NVFP4 model present locally)

```bash
# Skip with explicit log if file missing:
test -f /home/kekz/models/nemotron-3-nano-30b-a3b-nvfp4/model.safetensors || echo "SKIP: Nemotron-H NVFP4 not local"
```

If present:
```bash
docker run --rm --gpus all \
    -v /home/kekz/models:/models -v "$PWD":/imp -w /imp imp:test \
    bash -c "./build/tools/imp-cli --model /models/nemotron-3-nano-30b-a3b-nvfp4 \
             --prompt-file <(yes 'Lorem ipsum dolor sit amet. ' | head -40 | tr -d '\n') \
             --max-tokens 32 --temperature 0"
```

Expected: coherent output. Specifically validates Task 2's conv_state fix (the only path that exercises the patched plain `ssm_conv1d_prefill_kernel`).

- [ ] **Step 4.3: Run validate_safetensors.py battery on hybrid SafeTensors models**

Pick the available model(s):
```bash
for m in Qwen3.6-35B-A3B-NVFP4 Nemotron-H-NVFP4; do
    if [ -d "/home/kekz/models/$m" ]; then
        IMP_DOCKER_IMG=imp:test IMP_MODELS_DIR=/home/kekz/models \
            python3 scripts/validate_safetensors.py --model "$m" 2>&1 | tail -30
    fi
done
```

Expected: `phase4` count >= the value in the relevant memo's "current battery score" (e.g., Qwen3.6-NVFP4 was 16/20 per memory `qwen36_nvfp4_antthinking_2026_05_04`). The `long_context_recall` prompt specifically must no longer fail with CANCELLED.

If a model that was passing now fails: hold off committing perf baseline + roadmap update; investigate before continuing.

- [ ] **Step 4.4: Note results in a session log file (no commit yet)**

Capture results to `/tmp/chunked_prefill_hybrid_smoke.log` for reference; don't commit. The next task's perf baseline run is the thing that gets committed.

---

## Task 5: Refresh perf baseline for hybrid models

**Files:**
- Modify: `tests/perf_baseline.json` (regenerated by script)

- [ ] **Step 5.1: Regenerate baseline**

```bash
docker run --rm --gpus all \
    -v "$PWD/models":/models -v "$PWD":/imp -w /imp imp:test \
    bash scripts/gen_perf_baseline.sh
```

This benchmarks Qwen3-4B / Qwen3-8B / Qwen3.5-4B / Qwen3.5-9B / Llama-3.2-3B at the existing tg256 / pp512 entries. Hybrid models (Qwen3.5-4B, Qwen3.5-9B) should land within 3% decode / 5% prefill of pre-change numbers (chunking only affects prefill of prompts > effective_chunk; pp512 ≤ 256/512 cap = single chunk, no overhead change).

- [ ] **Step 5.2: Check the diff**

```bash
git diff tests/perf_baseline.json
```

Expected: small numeric movements (within thresholds). If a hybrid model regresses >5% on pp or >3% on tg, investigate before committing — chunking shouldn't affect prompts that fit in one chunk.

- [ ] **Step 5.3: Commit refreshed baseline**

```bash
git add tests/perf_baseline.json
git commit -m "$(cat <<'EOF'
perf: refresh baseline post chunked-prefill hybrid enablement

Confirms chunked-prefill enablement on hybrid GDN+MoE / Mamba2+MoE archs
does not regress short-prompt (≤ effective_chunk) prefill or decode.
Long-prompt prefill (> 256/512 cap) on hybrid models was previously
CANCELLED — no comparable baseline to refresh.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Roadmap + CHANGELOG documentation

**Files:**
- Modify: `docs/roadmap.md` lines 20-32 (Chunked prefill scope section)
- Modify: `CHANGELOG.md` (top of file, new entry)

- [ ] **Step 6.1: Update `docs/roadmap.md`**

Open `docs/roadmap.md`. Locate the section starting at line 20 (`### Chunked prefill scope (full-attention + FP16/FP8 KV)`).

Find the bullet list at lines 26-29:
```
- Gemma-3 / Gemma-4 (SWA — Gemma-4 also has dual head_dim 256/512)
- Llama-4 (MoE + SWA)
- Hybrid models with non-attention layers (Qwen3.5/3.6 GDN, Nemotron-H Mamba2)
- Sub-byte KV cache dtypes (INT4, NVFP4, TurboQuant variants)
```

Replace with (drop the "Hybrid" bullet, update NVFP4 status):
```
- Gemma-3 / Gemma-4 (SWA — Gemma-4 also has dual head_dim 256/512)
- Llama-4 (MoE + SWA)
- Sub-byte KV cache dtypes (INT4, TurboQuant, TurboQuant Lite)
```

Find the section header at line 20 and update to reflect the new scope. Replace:
```
### Chunked prefill scope (full-attention + FP16/FP8 KV)
```
with:
```
### Chunked prefill scope (full-attention + hybrid GDN/Mamba2; FP16/FP8/NVFP4 KV)
```

Find the paragraph starting at line 22 (`Default \`prefill_chunk_size = 512\`...`). Update the lead phrase:
```
Default `prefill_chunk_size = 512` for full-attention models (Qwen3, Llama, Mistral) with FP16 or FP8 KV cache.
```
to:
```
Default `prefill_chunk_size = 512` for full-attention models (Qwen3, Llama, Mistral) and hybrid GDN+MoE / Mamba2+MoE models (Qwen3.5/3.6, Nemotron-H) with FP16, FP8, or NVFP4 KV cache.
```

- [ ] **Step 6.2: Append CHANGELOG entry**

Add to the top of `CHANGELOG.md` under the current unreleased / dated section:

```markdown
- **Chunked prefill on hybrid GDN+MoE / Mamba2+MoE archs**: Qwen3.5/3.6 GDN
  and Nemotron-H Mamba2 models now ingest prompts longer than the executor's
  per-step `max_tokens` cap (256/512). Previously such prompts were rejected
  with `RequestStatus::CANCELLED`. Mamba2 plain conv1d prefill kernel reads
  trailing-context from `conv_state` at chunk boundary instead of zero-padding;
  engine `supports_chunked_prefill_()` carve-out for the four hybrid archs
  removed.
```

(Match the existing CHANGELOG.md style — check the file before appending.)

- [ ] **Step 6.3: Commit docs**

```bash
git add docs/roadmap.md CHANGELOG.md
git commit -m "$(cat <<'EOF'
docs(roadmap): chunked prefill now supports hybrid GDN+MoE / Mamba2+MoE

Removes hybrid models from the chunked-prefill out-of-scope list — they
now route through the same chunked-attention path as full-attention
models with their SSM/GDN state persistence handled by the existing
forward kernels.

Out-of-scope list now: Gemma-3/4 SWA, Llama-4 MoE+SWA, sub-byte KV
gather kernels (INT4, TurboQuant variants).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Final verification + PR push

- [ ] **Step 7.1: Run `make verify-fast`**

```bash
make verify-fast
```

Expected: green (build + filtered tests + perf gate + smoke prompt within ~90s).

- [ ] **Step 7.2: Optional — `make verify` for the full pre-merge gate**

```bash
make verify
```

Expected: green within ~5min. Skip if `verify-fast` covers the changed surfaces and no other recent changes to verify.

- [ ] **Step 7.3: Push branch and open PR**

```bash
git push -u origin "$(git branch --show-current)"
gh pr create --title "feat(engine): chunked prefill on hybrid GDN+MoE / Mamba2+MoE archs" \
    --body "$(cat <<'EOF'
## Summary

- Patch `ssm_conv1d_prefill_kernel` to read trailing context from
  `conv_state` at chunk boundary (mirror of f32_silu fused variant)
- Drop `QWEN35` / `QWEN35_MOE` / `QWEN36_MOE` / `NEMOTRON_H_MOE` carve-outs
  in `Engine::supports_chunked_prefill_()`
- Long prompts on hybrid models no longer rejected with
  `RequestStatus::CANCELLED`

## Test plan

- [x] `SSMConv1dTest.ChunkedPrefillEquivalence` — kernel-level chunk-boundary
      correctness (split sequence vs single-shot)
- [x] `make verify-fast` green
- [x] Smoke prompt (>256 tokens) on Qwen3.5-4B-Q8_0 — no CANCELLED
- [x] `validate_safetensors.py` battery on Qwen3.6-NVFP4 / Nemotron-H-NVFP4
      where available — no regression vs prior baseline

## Out of scope (separate work)

- Gemma-3/4 SWA chunked prefill
- Llama-4 MoE+SWA
- Sub-byte KV gather kernels (INT4, TurboQuant, TurboQuant Lite)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review checklist

- All spec components mapped to tasks: ✓
  - Mamba2 conv kernel patch → Tasks 1+2
  - Engine carve-out removal → Task 3
  - Engine doc-comment update → Task 3.2
  - End-to-end validate_safetensors → Task 4
  - Perf baseline refresh → Task 5
  - Roadmap update → Task 6.1
  - CHANGELOG entry → Task 6.2
- No placeholders / TBDs: ✓
- Type / function name consistency: ✓ (`supports_chunked_prefill_`, `resolve_prefill_chunk_size_`, `ssm_conv1d_prefill_kernel`, `ssm_conv1d_prefill_f32_silu_kernel` — all match the codebase)
- TDD ordering: failing test before implementation (Task 1 → Task 2.1)
- Each step is bite-sized (<5 min): ✓
- Exact file paths + line numbers given: ✓
- Risk handling: 4.1 / 4.3 are gates that hold up subsequent tasks if they regress

