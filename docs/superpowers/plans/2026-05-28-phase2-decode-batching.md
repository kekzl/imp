# Phase 2 — Decode Batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** imp-server handles up to 4 concurrent decode requests with batched weight reads. Prefill stays serial.

**Architecture:** Most infrastructure already exists (scheduler, KV manager, BatchBuilder, GPUBatchPool, per-batch-size graph pool). The work is removing artificial batch=1 guards in the decode path, fixing the server's per-request cancellation on new arrivals, and extending penalty upload to multi-sequence.

**Tech Stack:** C++20, CUDA 13.2, GTest, cpp-httplib.

**Spec:** `docs/superpowers/specs/2026-05-28-phase2-decode-batching-design.md`

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `src/runtime/engine_scheduler.cpp` | Modify | Remove batch=1 decode guard; extend penalties to multi-seq |
| `tools/imp-server/batching_engine.cpp` | Modify | Remove single-request cancellation on new arrival; keep graph invalidation per batch-size change |
| `src/runtime/config.h` | Modify | Add `runtime.max_batch_size` (default 4) |
| `src/runtime/engine.cpp` | Modify | Read `max_batch_size` from runtime config |
| `tests/test_continuous_batching.cpp` | Modify | Add multi-decode scenario test |
| `tests/test_e2e.cpp` | Modify | Add multi-request server integration test |

---

## Task 1: Remove single-request cancellation in BatchingEngine

**Files:**
- Modify: `tools/imp-server/batching_engine.cpp:80-110`

The server currently cancels the previous active request and invalidates all CUDA graphs whenever a new request arrives (lines 87-103). This prevents concurrent requests.

- [ ] **Step 1: Read the current code**

Read `tools/imp-server/batching_engine.cpp:60-115` to understand the current `worker_loop` intake logic.

- [ ] **Step 2: Remove the stale-request cancellation block**

Replace lines 84-103 (the `if (ctx_->active_request)` block + `invalidate_graphs()` + `reset_batch_pool_cache()`) with just the request add:

```cpp
            while (!pending_queue_.empty()) {
                auto sr = std::move(pending_queue_.front());
                pending_queue_.pop_front();

                sr->notified_count = sr->request->output_tokens.size();
                engine->add_request(sr->request);
                active_requests_.push_back(std::move(sr));
            }
```

The `ctx_->active_request` field is a C API concept that the server shouldn't touch. Graph invalidation will be handled by the graph pool's per-batch-size indexing (already done at `engine_scheduler.cpp:1010-1037`).

- [ ] **Step 3: Build and verify**

Run: `make build`
Expected: Clean build.

- [ ] **Step 4: Commit**

```bash
git add tools/imp-server/batching_engine.cpp
git commit -m "fix: stop cancelling previous request on new server submission"
```

---

## Task 2: Remove batch=1 decode guard in scheduler

**Files:**
- Modify: `src/runtime/engine_scheduler.cpp:722-725`

- [ ] **Step 1: Read the current guard**

At line 722-725:
```cpp
    // SSM/GDN: limit decode batch to 1 sequence
    if ((ssm_state_ || gdn_state_) && decode_batch.size() > 1) {
        decode_batch.resize(1);
    }
```

This guard correctly limits SSM/GDN models to batch=1 (their recurrent state isn't batched). But there's no guard for standard transformer/MoE models — the batch=1 constraint comes entirely from the server's request cancellation (Task 1) and the async graph loop guard (line 1262: `valid_decode.size() == 1`).

- [ ] **Step 2: Verify no hidden batch=1 constraint exists for transformer models**

Grep for any other place that forces single-sequence decode:

```bash
grep -n 'decode_batch.*resize\|decode_batch.*=.*1\|valid_decode.*resize' src/runtime/engine_scheduler.cpp
```

The only hit should be the SSM/GDN guard at line 722. If there are others, they need evaluation.

- [ ] **Step 3: Add max_batch_size enforcement**

After the SSM/GDN guard, add a general cap:

```cpp
    // SSM/GDN: limit decode batch to 1 sequence (recurrent state not batched)
    if ((ssm_state_ || gdn_state_) && decode_batch.size() > 1) {
        decode_batch.resize(1);
    }

    // Cap at configured max batch size
    const int max_bs = runtime_config_.runtime.max_batch_size;
    if (max_bs > 0 && static_cast<int>(decode_batch.size()) > max_bs) {
        decode_batch.resize(max_bs);
    }
```

- [ ] **Step 4: Build**

Run: `make build`
Expected: Clean build.

- [ ] **Step 5: Commit**

```bash
git add src/runtime/engine_scheduler.cpp
git commit -m "feat: enforce max_batch_size cap on decode batch"
```

---

## Task 3: Extend penalty upload to multi-sequence

**Files:**
- Modify: `src/runtime/engine_scheduler.cpp:865-868`

- [ ] **Step 1: Read the current penalty gate**

At lines 865-868:
```cpp
    // Penalties (single-sequence only)
    if (gpu_batch.n_sequences == 1) {
        upload_penalties(*valid_decode[0], state, dec_stream);
    }
```

When batch > 1, penalties are uploaded per-sequence inside `sample_per_request` (line 982-993), but only if the request needs them. The single-sequence fast path at line 866 uploads penalties into `state.penalty_tokens` once for the whole step.

- [ ] **Step 2: Extend to upload penalties for the first request in multi-seq**

The multi-seq sampling lambda at lines 973-1000 already handles per-request penalty upload. The gate at line 866 is an optimization for the single-seq case. For multi-seq, we can just skip it — the lambda handles it.

No code change needed here. The existing code already works: when `n_sequences > 1`, the penalty gate at line 866 is skipped, and `sample_per_request` uploads penalties per-request at lines 982-993.

**Verify:** Read lines 973-1000 and confirm that `d_penalty_tokens_` is uploaded per-request in the multi-seq path with correct synchronization (each `cudaMemcpyAsync` overwrites the same buffer, but `sample_single_from_logits` consumes it before the next iteration).

- [ ] **Step 3: Commit (documentation-only if no change needed)**

If verification confirms no change needed, commit a comment clarifying the multi-seq penalty path:

```bash
git add src/runtime/engine_scheduler.cpp
git commit -m "docs: clarify multi-seq penalty upload path in decode_build_inference_state"
```

---

## Task 4: Add max_batch_size to RuntimeConfig

**Files:**
- Modify: `src/runtime/config.h`
- Modify: `src/runtime/config.cpp` (TOML parser section for `[runtime]`)

- [ ] **Step 1: Add field to RuntimeConfig**

In `src/runtime/config.h`, find the `Runtime` struct and add:

```cpp
struct Runtime {
    // ... existing fields ...
    int max_batch_size = 4;  // Max concurrent decode sequences (0 = unlimited)
    // ... existing fields ...
};
```

- [ ] **Step 2: Wire up TOML parsing**

In `src/runtime/config.cpp`, in the `[runtime]` parsing section, add:

```cpp
if (key == "max_batch_size") { cfg.runtime.max_batch_size = std::stoi(val); continue; }
```

- [ ] **Step 3: Wire up CLI override**

If `--set runtime.max_batch_size=N` already works via the generic TOML override path, no change needed. Verify by checking how `--set` routes to config fields.

- [ ] **Step 4: Build and verify**

Run: `make build`

- [ ] **Step 5: Commit**

```bash
git add src/runtime/config.h src/runtime/config.cpp
git commit -m "feat: add runtime.max_batch_size config (default 4)"
```

---

## Task 5: Multi-decode correctness test

**Files:**
- Modify: `tests/test_continuous_batching.cpp`

- [ ] **Step 1: Read existing test patterns**

Read `tests/test_continuous_batching.cpp` to understand the existing BatchBuilder test infrastructure.

- [ ] **Step 2: Add a multi-decode test that verifies output isolation**

```cpp
TEST(ContinuousBatching, MultiDecodeOutputIsolation) {
    // Verify that decoding 2 requests simultaneously produces the same
    // per-request output as decoding them sequentially.
    // This requires a real model — skip if unavailable.
    const char* model_path = getenv("IMP_TEST_MODEL");
    if (!model_path) GTEST_SKIP() << "Set IMP_TEST_MODEL";

    // Run request A alone, record its first 16 tokens
    // Run request B alone, record its first 16 tokens
    // Run both simultaneously, verify A's tokens match and B's tokens match

    // Implementation: use Engine directly with add_request + step loop
    // (not imp_generate, which is single-request blocking)
}
```

The full implementation depends on the Engine API for multi-request — read `engine.h` to find `add_request()` and `step()` signatures, then write the test.

- [ ] **Step 3: Run test**

Run: `make test-gpu TEST_FILTER="ContinuousBatching.MultiDecodeOutputIsolation"`
Expected: PASS (or SKIP if no model).

- [ ] **Step 4: Commit**

```bash
git add tests/test_continuous_batching.cpp
git commit -m "test: multi-decode output isolation correctness test"
```

---

## Task 6: Server multi-request integration test

**Files:**
- Modify: `tests/test_e2e.cpp` or create a script

- [ ] **Step 1: Write a multi-curl test script**

Create `tests/test_server_concurrent.sh`:

```bash
#!/bin/bash
# Test concurrent requests against imp-server
# Requires: imp-server running on port 8080 with a loaded model

set -e

BASE="http://localhost:8080/v1/chat/completions"

# Fire 4 concurrent requests
for i in 1 2 3 4; do
    curl -s "$BASE" \
        -H "Content-Type: application/json" \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Count from $i to $((i+5))\"}],\"max_tokens\":32}" \
        -o "/tmp/imp_concurrent_$i.json" &
done

# Wait for all
wait

# Verify all got responses
for i in 1 2 3 4; do
    f="/tmp/imp_concurrent_$i.json"
    if [ ! -s "$f" ]; then
        echo "FAIL: request $i got empty response"
        exit 1
    fi
    content=$(jq -r '.choices[0].message.content' "$f" 2>/dev/null)
    if [ -z "$content" ] || [ "$content" = "null" ]; then
        echo "FAIL: request $i got no content: $(cat $f)"
        exit 1
    fi
    echo "OK: request $i: ${content:0:60}..."
done

echo "PASS: all 4 concurrent requests completed"
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x tests/test_server_concurrent.sh
git add tests/test_server_concurrent.sh
git commit -m "test: concurrent server request integration test"
```

---

## Task 7: Degeneration check with concurrent decode

**Files:** None (manual verification)

- [ ] **Step 1: Start server with a model**

```bash
docker run --gpus all -v /home/kekz/models:/models -p 8080:8080 \
    imp:test imp-server --model /models/Qwen3-8B-Q8_0.gguf
```

- [ ] **Step 2: Fire 4 concurrent requests**

```bash
tests/test_server_concurrent.sh
```

Expected: All 4 responses are coherent, non-degenerate text.

- [ ] **Step 3: Verify decode throughput improvement**

Run single-request baseline:
```bash
time curl -s http://localhost:8080/v1/chat/completions \
    -d '{"messages":[{"role":"user","content":"Write a 200-word story"}],"max_tokens":200}'
```

Run 4 concurrent:
```bash
time (for i in 1 2 3 4; do curl -s http://localhost:8080/v1/chat/completions \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Write a 200-word story about topic $i\"}],\"max_tokens\":200}" &; done; wait)
```

Expected: 4-concurrent wall time < 2x single-request wall time (weight amortization kicks in).

- [ ] **Step 4: Document results**

Note the throughput numbers for the PR description.

---

## Risk assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Penalty buffer race condition in multi-seq sampling | Medium | The `d_penalty_tokens_` buffer is overwritten per-seq in a serial loop — each `sample_single_from_logits` call completes before the next overwrites. Safe if sampling is synchronous on the stream. |
| Graph invalidation when batch size changes | Low | Graph pool already indexes by n_sequences; `invalidate_for_update()` handles topology changes. |
| KV eviction cascade under 4 active requests | Medium | LRU eviction fires per-block; with 4 requests the eviction threshold is hit sooner. Max_batch_size=4 is conservative. |
| Constraints/JSON schema with batch>1 | Low | Currently gated on `valid_decode.size() == 1` (lines 887-898). With batch>1, constraints are skipped — acceptable for Phase 2, can be extended later. |
| SSM/GDN models accidentally run batch>1 | None | Existing guard at line 722-725 correctly prevents it. |
