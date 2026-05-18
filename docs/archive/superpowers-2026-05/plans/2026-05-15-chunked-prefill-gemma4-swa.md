# Chunked Prefill for Gemma-4 (SWA + dual head_dim) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the "Gemma-3 / Gemma-4 (SWA — Gemma-4 also has dual head_dim 256/512)" out-of-scope item in `docs/roadmap.md` (lines 20-32) so Gemma-4 prompts longer than `executor->max_tokens()` no longer hit `RequestStatus::CANCELLED`.

**Architecture:** Three-stage change.
1. Add a `sliding_window` parameter to the three `causal_softmax_*` kernels inside `src/compute/attention_cublas.cu`. The mask zero-outs `j < (q_offset + i) - sliding_window + 1` in addition to the existing causal mask `j > q_offset + i`. This is what the naive reference attention already does; the cuBLAS prefill path is simply missing it.
2. Thread `sliding_window` through `attention_cublas_prefill` and call sites in `src/graph/executor_attention.cu`. Drop the `!sliding_active` gate on the non-chunked cuBLAS dispatch and the `sliding_active` reject in the chunked-prefill branch.
3. Loosen `Engine::supports_chunked_prefill_()` so Gemma-4 is allowed; per-layer head_dim (256 SWA / 512 global) is *already* dispatched per-layer through the rectangular cuBLAS prefill — the gate is defensive.

**Tech Stack:** C++20, CUDA 13.2 (sm_120a), cuBLAS, GTest, Docker (`make build`/`make test-gpu`).

**Test model:** `/home/kekz/models/gemma-4-26B-A4B-it-Q8_0.gguf` and `/home/kekz/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`.

---

## File Map

| File | Role |
|---|---|
| `src/compute/attention_cublas.cu` (modify, lines 92-300) | Add `sliding_window` to 3 softmax kernels + apply window mask alongside causal |
| `src/compute/attention_cublas.cu` (modify, lines 380-520) | Add `sliding_window` to `attention_cublas_prefill` signature, pass through |
| `src/compute/attention_cublas.h` (modify) | Public signature update — `int sliding_window = 0` default |
| `tests/test_attention_cublas.cu` (modify/create) | New tests: bit-exactness at sw=0, correctness vs naive at sw>0, chunked sw+q_offset |
| `src/graph/executor_attention.cu` (modify, lines 681-869) | Pass `layer_sliding_window`; drop sliding/per-layer gates in chunked branch |
| `src/runtime/engine.cpp` (modify, lines 1908-1951) | Drop `GEMMA4` from `supports_chunked_prefill_()`, loosen `attn_shapes_vary` for Gemma-4 |
| `tests/perf_baseline.json` (regenerate) | Refresh Gemma-4 baselines after the change |
| `docs/roadmap.md` (modify, lines 20-32) | Remove Gemma-3/4 from out-of-scope list |
| `CHANGELOG.md` (append) | Entry under Unreleased |

---

## Task 1: SWA softmax kernel — failing test (TDD red)

**Why first:** the math should be a 5-line change to each kernel. A test against the naive reference catches off-by-ones on the `(q_offset + i) - sliding_window + 1` mask boundary.

**Files:**
- Modify: `tests/test_attention_cublas.cu` (append a new TEST)

- [ ] **Step 1.1: Read current test file shape**

```bash
ls tests/test_attention_cublas.cu 2>&1 || echo "NEW FILE NEEDED"
head -40 tests/test_attention_cublas.cu 2>/dev/null
```

If the file is missing, search for the closest existing attention test:
```bash
ls tests/test_attention*.cu tests/test_*attn*.cu 2>&1
```

- [ ] **Step 1.2: Append SWA equivalence test**

Add a new GTest case `AttentionCublasTest.SlidingWindowMatchesNaive` that:
- Generates Q/K/V FP16 in [-1, 1] for n_heads=2, n_kv_heads=2, head_dim=64, q_len=kv_len=128
- Runs `attention_cublas_prefill(..., sliding_window=32, causal=true)` 
- Runs `naive_attention_prefill(..., sliding_window=32)`
- Asserts max abs-diff ≤ 2e-2 (cuBLAS FP32-S vs naive FP32 reference)

```cpp
TEST(AttentionCublasTest, SlidingWindowMatchesNaive) {
    SKIP_IF_NO_CUDA();
    constexpr int q_len = 128, kv_len = 128;
    constexpr int n_heads = 2, n_kv_heads = 2, head_dim = 64;
    constexpr int sliding_window = 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> h_q(q_len * n_heads * head_dim);
    std::vector<float> h_k(kv_len * n_kv_heads * head_dim);
    std::vector<float> h_v(kv_len * n_kv_heads * head_dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : h_q) x = dist(rng);
    for (auto& x : h_k) x = dist(rng);
    for (auto& x : h_v) x = dist(rng);

    Tensor Q = make_fp16_gpu(h_q.data(), {q_len, n_heads * head_dim});
    Tensor K = make_fp16_gpu(h_k.data(), {kv_len, n_kv_heads * head_dim});
    Tensor V = make_fp16_gpu(h_v.data(), {kv_len, n_kv_heads * head_dim});
    Tensor O_cublas = alloc_fp16_gpu({q_len, n_heads * head_dim});
    Tensor O_naive  = alloc_fp16_gpu({q_len, n_heads * head_dim});
    int64_t s_shape[3] = {n_heads, q_len, kv_len};
    Tensor S = alloc_fp16_gpu_3d(s_shape);

    attention_cublas_prefill(Q, K, V, O_cublas, S, n_heads, n_kv_heads, head_dim,
                             scale, /*causal=*/true, /*softcap=*/0.0f,
                             /*q_offset=*/0, /*stream=*/nullptr,
                             /*sliding_window=*/sliding_window);
    naive_attention_prefill(static_cast<const half*>(Q.data),
                            static_cast<const half*>(K.data),
                            static_cast<const half*>(V.data),
                            static_cast<half*>(O_naive.data),
                            q_len, n_heads, n_kv_heads, head_dim,
                            scale, /*softcap=*/0.0f, /*stream=*/nullptr,
                            sliding_window);
    cudaDeviceSynchronize();
    auto a = read_fp16(O_cublas);
    auto b = read_fp16(O_naive);
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); i++)
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    EXPECT_LT(max_diff, 2e-2f) << "cuBLAS SWA != naive SWA";
}
```

The exact helper names (`make_fp16_gpu`, `alloc_fp16_gpu`, `read_fp16`) follow the conventions used in `tests/test_ssm.cu`. If the helpers don't exist for `_3d`, allocate manually via `cudaMalloc` and wrap in `Tensor(ptr, QType::F16, 3, shape, true)`.

- [ ] **Step 1.3: Build, expect failure**

```bash
cmake --build build --target test_attention_cublas -j$(nproc) 2>&1 | tail -10
```
Expected: build fails because `attention_cublas_prefill` signature does not yet accept `sliding_window`. Stop here — that's the red.

---

## Task 2: SWA softmax kernel — implementation

**Files:**
- Modify: `src/compute/attention_cublas.cu` (3 kernels at lines 92-145, 151-199, 212-300)
- Modify: `src/compute/attention_cublas.cu` (function `attention_cublas_prefill` at lines 380-520)
- Modify: `src/compute/attention_cublas.h` (declaration)

- [ ] **Step 2.1: Update kernel signatures**

For each of the three softmax kernels (`causal_softmax_fp32_to_fp16_kernel`, `causal_softmax_fp32_inplace_kernel`, `causal_softmax_inplace_kernel`):

1. Add a trailing `int sliding_window` parameter (after `bool causal`).
2. Compute `int window_lo = (sliding_window > 0) ? (abs_row - sliding_window + 1) : 0;` after the existing `int abs_row = q_offset + row;`.
3. Replace `(causal && j > abs_row)` masks with `(causal && j > abs_row) || (sliding_window > 0 && j < window_lo)`.

There are three mask sites per kernel (max-pass, sum-pass, final write-pass) — update all three in each kernel.

- [ ] **Step 2.2: Update `attention_cublas_prefill` signature**

In `src/compute/attention_cublas.h`, change the function signature:

```cpp
void attention_cublas_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, Tensor& S,
                              int n_heads, int n_kv_heads, int head_dim, float scale, bool causal,
                              float softcap = 0.0f, int q_offset = 0, cudaStream_t stream = nullptr,
                              int sliding_window = 0);
```

In `src/compute/attention_cublas.cu`, mirror this on the definition, and pass `sliding_window` as the trailing argument of every softmax kernel launch (4 launches: MHA FP16, MHA FP32, GQA FP16, GQA FP32 — lines 459, 462, 510, 513).

- [ ] **Step 2.3: Build host-side and run the test**

```bash
cmake --build build -j$(nproc) 2>&1 | tail -5
./build/tests/test_attention_cublas --gtest_filter=AttentionCublasTest.SlidingWindowMatchesNaive 2>&1 | tail -10
```
Expected: PASS. If fails, dump `max_diff` and check off-by-ones at the window edge.

- [ ] **Step 2.4: Bit-exact regression check at sliding_window=0**

```bash
make test-gpu 2>&1 | tail -10
```
Expected: same totals as before, all green. The new parameter defaults to 0 (off) on every existing call site, so behavior must be unchanged.

- [ ] **Step 2.5: Commit**

```bash
git add src/compute/attention_cublas.cu src/compute/attention_cublas.h tests/test_attention_cublas.cu
git commit -m "$(cat <<'EOF'
feat(attn): sliding_window mask in cuBLAS prefill softmax

Add sliding_window parameter to attention_cublas_prefill (default 0 = off).
Three softmax kernels (FP16-inplace, FP32-inplace, FP32→FP16 fused) now
mask j < (q_offset + i) - sliding_window + 1 alongside the existing
causal j > q_offset + i.

Bit-exact at sliding_window=0; new GTest verifies SWA output matches the
naive FP32 reference within 2e-2 max abs diff (cuBLAS FP32-S precision).

Unblocks routing Gemma-4 SWA layers through cuBLAS (replacing the naive
workaround at executor_attention.cu:835-851) and chunked prefill for
SWA models — both follow in separate commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Route Gemma-4 SWA layers through cuBLAS (non-chunked)

**Files:**
- Modify: `src/graph/executor_attention.cu` (lines 819-869 — cuBLAS-vs-naive-vs-FMHA dispatch)

- [ ] **Step 3.1: Pass `layer_sliding_window` to the cuBLAS call**

At line 868 (the cuBLAS dispatch in the non-chunked branch), change:
```cpp
attention_cublas_prefill(qv, kk, vv, ao, attn_scores_, nh, nkv, hd, scale, /*causal=*/true,
                         cfg.attn_logit_softcap, /*q_offset=*/0, stream);
```
to:
```cpp
attention_cublas_prefill(qv, kk, vv, ao, attn_scores_, nh, nkv, hd, scale, /*causal=*/true,
                         cfg.attn_logit_softcap, /*q_offset=*/0, stream,
                         layer_sliding_window);
```

- [ ] **Step 3.2: Drop the `!sliding_active` gate**

At line 853, change:
```cpp
} else if ((force_cublas_attn || !no_cublas_attn) && attn_scores_buf_ &&
           n <= static_cast<int>(attn_scores_.shape[1]) && !sliding_active) {
```
to:
```cpp
} else if ((force_cublas_attn || !no_cublas_attn) && attn_scores_buf_ &&
           n <= static_cast<int>(attn_scores_.shape[1])) {
```
The cuBLAS path now handles sliding window directly.

- [ ] **Step 3.3: Tighten the naive SWA workaround**

At line 837 (`use_naive_for_swa` predicate), narrow to only when cuBLAS attn_scores would overflow (n > attn_scores cap). Replace:
```cpp
bool gemma4_swa_broken = (cfg.arch == ModelArch::GEMMA4 && sliding_active);
bool gemma4_global_too_long = (cfg.arch == ModelArch::GEMMA4 && !sliding_active && n > cublas_cap);
bool use_naive_for_swa = ((gemma4_swa_broken || gemma4_global_too_long) && n <= 8192 &&
                          !RuntimeConfig::current().attention.no_naive_swa);
```
with:
```cpp
// cuBLAS path now supports sliding_window natively (PR adds sw to softmax kernels).
// Fall back to naive only when attn_scores buffer is too small for the FP16 S matrix.
bool gemma4_overflow_cublas = (cfg.arch == ModelArch::GEMMA4 && n > cublas_cap);
bool use_naive_for_swa = (gemma4_overflow_cublas && n <= 8192 &&
                          !RuntimeConfig::current().attention.no_naive_swa);
```

- [ ] **Step 3.4: Smoke Gemma-4 short prompts**

```bash
make build 2>&1 | tail -5
docker run --rm --gpus all -v /home/kekz/models:/models imp:test \
  ./build/tools/imp-cli/imp-cli \
  --model /models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
  --prompt "The capital of France is" --max-tokens 16 2>&1 | tail -20
```
Expected: coherent answer ("Paris" or similar). If it outputs `"own owners"` or random tokens, the cuBLAS SWA path has a bug — debug before continuing.

- [ ] **Step 3.5: Run the full degeneration check**

Invoke the `check-degeneration` skill for Gemma-4-26B Q4_K_M (multi-turn) and Q8_0 (single-turn smoke). Expected: no degeneration.

- [ ] **Step 3.6: Commit**

```bash
git add src/graph/executor_attention.cu
git commit -m "$(cat <<'EOF'
perf(attn): route Gemma-4 SWA layers through cuBLAS

attention_cublas_prefill now supports sliding_window (previous commit).
SWA layers no longer need the naive FP32 reference workaround when the
attn_scores buffer fits the FP16 S matrix. Naive path is kept as a
fallback for n > attn_scores capacity (≤ 8192).

Removes the !sliding_active gate on cuBLAS dispatch and the
gemma4_swa_broken branch from the naive-fallback predicate.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Enable chunked prefill for Gemma-4

**Files:**
- Modify: `src/runtime/engine.cpp` (lines 1908-1951 — `supports_chunked_prefill_`)
- Modify: `src/graph/executor_attention.cu` (lines 681-810 — chunked prefill branch)

- [ ] **Step 4.1: Drop GEMMA4 from `supports_chunked_prefill_`**

At line 1918 in `engine.cpp`, delete:
```cpp
if (cfg.arch == ModelArch::GEMMA4) return false;       // SWA + dual head_dim
```
Keep the `GEMMA3` and `LLAMA4` gates (different scope — Gemma-3 has no test model in repo, Llama-4 has MoE + SWA which is a separate work item).

Also relax the `head_dim_per_layer` uniformity check at line 1938-1942 to allow Gemma-4 specifically:
```cpp
if (!cfg.head_dim_per_layer.empty()) {
    int ref = first_nonzero_int(cfg.head_dim_per_layer);
    if (ref > 0 && any_nonzero_differs(cfg.head_dim_per_layer, ref)) {
        // Gemma-4 (256 SWA / 512 global): each layer call uses its own hd,
        // and the cuBLAS rectangular path is per-layer-shape-aware. Allow.
        if (cfg.arch != ModelArch::GEMMA4) return false;
    }
}
```

- [ ] **Step 4.2: Drop the SWA + attn_shapes_vary aborts in the chunked branch**

In `executor_attention.cu` lines 727-733, change the abort guard:
```cpp
if (!kvt_ok || sliding_active || attn_shapes_vary) {
    IMP_LOG_ERROR(
        "chunked_prefill: unsupported config (kv=%d swa=%d attn_shapes_vary=%d) at L%d — "
        "engine should have prevented this",
        (int)kvt, (int)sliding_active, (int)attn_shapes_vary, layer);
    std::abort();
}
```
to:
```cpp
if (!kvt_ok) {
    IMP_LOG_ERROR(
        "chunked_prefill: unsupported KV dtype %d at L%d — engine should have prevented this",
        (int)kvt, layer);
    std::abort();
}
// sliding_active + attn_shapes_vary (Gemma-4 dual hd) are now both supported
// via attention_cublas_prefill's sliding_window param and per-layer dispatch.
```

- [ ] **Step 4.3: Pass `layer_sliding_window` to the chunked cuBLAS call**

At line 801, change:
```cpp
attention_cublas_prefill(qv, k_full_t, v_full_t, ao, attn_scores_, nh, nkv, hd, scale,
                         /*causal=*/true, cfg.attn_logit_softcap, q_offset, stream);
```
to:
```cpp
attention_cublas_prefill(qv, k_full_t, v_full_t, ao, attn_scores_, nh, nkv, hd, scale,
                         /*causal=*/true, cfg.attn_logit_softcap, q_offset, stream,
                         layer_sliding_window);
```

- [ ] **Step 4.4: Build + chunked smoke**

```bash
cmake --build build -j$(nproc) 2>&1 | tail -5
docker run --rm --gpus all -v /home/kekz/models:/models imp:test \
  ./build/tools/imp-cli/imp-cli \
  --model /models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
  --prefill-chunk-size 512 \
  --prompt "$(cat tests/data/long_prompt_1500_tokens.txt)" \
  --max-tokens 32 2>&1 | tail -20
```
If no 1500-token prompt file exists, synthesize one inline: `"Write a long story. " repeated 100 times`.

Expected: coherent continuation. The earlier behavior was `RequestStatus::CANCELLED` for any Gemma-4 prompt > `max_tokens`.

- [ ] **Step 4.5: Run `make test-gpu`**

```bash
make test-gpu 2>&1 | tail -15
```
Expected: same totals as before, all green (or only previously-known skipped tests).

- [ ] **Step 4.6: Commit**

```bash
git add src/runtime/engine.cpp src/graph/executor_attention.cu
git commit -m "$(cat <<'EOF'
feat(prefill): chunked prefill for Gemma-4 (SWA + dual head_dim)

Drop the GEMMA4 arch gate and the sliding_active + attn_shapes_vary
aborts in the chunked-prefill branch. Both prerequisites are now wired:

- sliding_window is threaded through attention_cublas_prefill's softmax
  (previous PR) and into the chunked branch's call site here.
- Per-layer head_dim (256 SWA / 512 global) is per-layer-dispatched
  through the rectangular cuBLAS path; the uniformity check is relaxed
  for Gemma-4 specifically.

Gemma-3 and Llama-4 stay out-of-scope: no Gemma-3 model in the test set,
Llama-4 needs MoE + SWA which is a separate work item.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Long-prompt validation + perf

- [ ] **Step 5.1: Long-prompt coherence (Gemma-4 Q4_K_M + Q8_0)**

Generate or fetch a ~2000-token prompt. Run with `--prefill-chunk-size 512` on both quants. Compare decode output against single-chunk reference (no chunked) using `IMP_PREFILL_CHUNK_SIZE=0`. Outputs should be semantically equivalent (token-by-token may differ due to FP rounding through different code paths; perplexity should be within 1%).

- [ ] **Step 5.2: `make verify-fast`**

```bash
make verify-fast 2>&1 | tail -30
```
Expected: green.

- [ ] **Step 5.3: Refresh perf baseline**

```bash
scripts/gen_perf_baseline.sh 2>&1 | tail -30
```
Compare `tests/perf_baseline.json` before/after. The change is decode-neutral (SWA mask only affects prefill); prefill may shift but stays within ±5% on Gemma-4 (the existing 3%/5% gate stays).

- [ ] **Step 5.4: Commit baseline refresh**

```bash
git add tests/perf_baseline.json
git commit -m "$(cat <<'EOF'
test(perf): refresh baselines after Gemma-4 chunked prefill enable

Decode-neutral by design; prefill numbers shift slightly because Gemma-4
now goes through the cuBLAS path for SWA layers (previously naive FP32).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Roadmap, CHANGELOG, memory, PR

- [ ] **Step 6.1: Update `docs/roadmap.md`**

In lines 20-32 ("Chunked prefill scope"), edit the out-of-scope list. Remove Gemma-4 (keep Gemma-3 and Llama-4 with a note). Add a "SHIPPED" mention referencing this PR.

- [ ] **Step 6.2: Append CHANGELOG entry**

Under `## Unreleased` add:
```markdown
- feat(prefill): chunked prefill for Gemma-4 (SWA + dual head_dim 256/512).
  cuBLAS prefill softmax kernels now accept a sliding_window parameter.
  Gemma-4 SWA layers no longer fall back to naive FP32 attention when
  attn_scores fits. Prompts > max_tokens on Gemma-4-26B no longer cancel.
```

- [ ] **Step 6.3: Write project memory**

Create `~/.claude/projects/-home-kekz-github-com-kekzl-imp/memory/gemma4_chunked_prefill_2026_05_15.md` with:
- One-line description in `description` frontmatter
- Body: shipped scope, what cuBLAS SWA path replaces, perf delta, references to PR + relevant files

Then add a pointer line in `MEMORY.md` index under "Shipped root-cause fixes".

- [ ] **Step 6.4: Push branch + open PR**

```bash
git push -u origin $(git branch --show-current)
gh pr create --title "feat(prefill): chunked prefill + cuBLAS SWA for Gemma-4" --body "$(cat <<'EOF'
## Summary
- Adds `sliding_window` to `attention_cublas_prefill`'s three softmax kernels.
- Routes Gemma-4 SWA layers through cuBLAS (replacing the naive FP32 workaround for n ≤ attn_scores cap).
- Enables chunked prefill for Gemma-4 (drops the SWA + dual head_dim gates).

Closes the Gemma-4 entry in the "Chunked prefill out-of-scope" list in `docs/roadmap.md`.

## Test plan
- [x] New GTest: cuBLAS SWA softmax matches naive reference within 2e-2 (FP32-S precision).
- [x] `make test-gpu` — no regressions, bit-exact at sliding_window=0.
- [x] `make verify-fast` — green.
- [x] Gemma-4-26B Q4_K_M long-prompt smoke with `--prefill-chunk-size 512` (no degeneration).
- [x] `check-degeneration` skill on Gemma-4 multi-turn.
- [x] Perf baseline refresh (decode-neutral; prefill within ±5%).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review checklist (run after writing)

- [x] All steps reference real file paths from the repo (`src/compute/attention_cublas.cu`, `src/graph/executor_attention.cu`, `src/runtime/engine.cpp`).
- [x] Each code change shows the BEFORE and AFTER blocks (not just "modify line X").
- [x] No TBD / TODO / "fill in" placeholders.
- [x] Per-task commit and verification commands present.
- [x] Builds on the existing `naive_attention_prefill(... sliding_window)` reference — no new reference implementation needed.
- [x] Tests come first (Task 1 = red, Task 2 = green).
