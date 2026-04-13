# Decode Drift Root Cause Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find the exact decode step, layer, and operation where Gemma-4's output diverges from llama.cpp, causing degeneration after ~15 tokens.

**Architecture:** Add a step counter to imp's debug dumps so each decode step is tagged. Capture llama.cpp reference dumps for the same prompt. Compare with a Python script that produces a step × layer error matrix.

**Tech Stack:** C++/CUDA (debug infra), Python 3 (diff script), llama.cpp custom build (reference)

---

### Task 1: Add decode step counter to imp's debug dumps

**Files:**
- Modify: `src/graph/executor_forward.cu:170-175` (add step counter)
- Modify: `src/graph/executor_forward.cu:228-234` (tag token dumps)
- Modify: `src/graph/executor_forward.cu:304-337` (tag layer dumps)

- [ ] **Step 1: Add step counter member and increment logic**

In `src/graph/executor_forward.cu`, find `cur_n_tokens_ = n;` (around line 170). Add a static step counter right after:

```cpp
    cur_n_tokens_ = n;
    // Decode step counter for debug dump tagging
    static int s_decode_step = 0;
    if (n == 1) s_decode_step++;  // increment only for decode (n=1), not prefill
    const int decode_step = (n == 1) ? s_decode_step : 0;
```

- [ ] **Step 2: Tag the input_tokens dump with step number**

Find the `fprintf(stderr, "[DEBUG_FWD] input_tokens (%d):"` line (around line 228-230). Change to:

```cpp
    if (debug_forward_enabled()) {
        fprintf(stderr, "[DEBUG_FWD] [step=%d] input_tokens (%d):", decode_step, n);
```

- [ ] **Step 3: Tag the per-layer debug dumps with step number**

Find the layer loop debug blocks. There are two blocks to modify:

**Block 1 — after attention (around line 306):**
Find: `snprintf(buf, sizeof(buf), "after_layer%02d_%s", i,`
Change to: `snprintf(buf, sizeof(buf), "[step=%d] after_layer%02d_%s", decode_step, i,`

**Block 2 — after FFN/MoE (around line 328):**
Find the second `snprintf(buf, sizeof(buf), "after_layer%02d_%s", i,`
Change to: `snprintf(buf, sizeof(buf), "[step=%d] after_layer%02d_%s", decode_step, i,`

**Block 3 — after out_scale (around line 377):**
Find: `snprintf(buf, sizeof(buf), "L%02d_after_out_scale", i);`
Change to: `snprintf(buf, sizeof(buf), "[step=%d] L%02d_after_out_scale", decode_step, i);`

- [ ] **Step 4: Tag the l_out row dumps with step number**

Find: `snprintf(rbuf, sizeof(rbuf), "attn_out-%d", i);`
Change to: `snprintf(rbuf, sizeof(rbuf), "[step=%d] attn_out-%d", decode_step, i);`

Find: `snprintf(rbuf, sizeof(rbuf), "moe_out-%d", i);`
Change to: `snprintf(rbuf, sizeof(rbuf), "[step=%d] moe_out-%d", decode_step, i);`

Find: `snprintf(rbuf, sizeof(rbuf), "l_out-%d", i);`
Change to: `snprintf(rbuf, sizeof(rbuf), "[step=%d] l_out-%d", decode_step, i);`

- [ ] **Step 5: Dump ALL layers (not just hardcoded subset)**

Find the `dump_this_layer` condition (around line 310):
```cpp
const bool dump_this_layer = (i <= 6 || i == 10 || i == 15 || i == 17 || i == 20 || i == 23 || i == 25 || i == 26 || i == 27 || i == 28 || i == 29);
```

Change to dump every layer when `IMP_DEBUG_FORWARD` is set:
```cpp
const bool dump_this_layer = true;  // dump all layers for drift analysis
```

Do the same for the second occurrence of this condition (after FFN/MoE, around line 331).

- [ ] **Step 6: Build and verify**

```bash
docker compose build imp-server 2>&1 | tail -5
```
Expected: `Image imp:latest Built`

Quick test:
```bash
docker compose run --rm -e IMP_DEBUG_FORWARD=1 --entrypoint "" imp-server \
  imp-cli --model /models/gemma-4-26B-A4B-it-Q4_K_M.gguf \
  --prompt "Hi" --max-tokens 2 --temperature 0 --chat-template gemma 2>&1 | \
  grep '\[step=' | head -5
```
Expected: Lines starting with `[step=0]` (prefill) and `[step=1]`, `[step=2]` (decode)

- [ ] **Step 7: Commit**

```bash
git add src/graph/executor_forward.cu
git commit -m "gemma4: add step counter to debug dumps for decode drift analysis

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Capture imp decode dumps

**Files:** None (capture only)

- [ ] **Step 1: Run imp with debug dumps for 15 decode steps**

```bash
docker compose run --rm -e IMP_DEBUG_FORWARD=1 -e IMP_NO_WARMUP=1 --entrypoint "" imp-server \
  imp-cli --model /models/gemma-4-26B-A4B-it-Q4_K_M.gguf \
  --prompt "What is the capital of France?" --max-tokens 15 --temperature 0 \
  --chat-template gemma 2>&1 > /tmp/imp_decode_drift.log
echo "Lines: $(wc -l < /tmp/imp_decode_drift.log)"
```

- [ ] **Step 2: Verify step tagging is present**

```bash
grep '\[step=' /tmp/imp_decode_drift.log | grep 'l_out-29' | head -10
```

Expected: Lines like `[DEBUG_FWD_ROW] [step=1] l_out-29[0] L2=... ...`

- [ ] **Step 3: Count steps captured**

```bash
grep -o '\[step=[0-9]*\]' /tmp/imp_decode_drift.log | sort -u
```

Expected: `[step=0]` through `[step=15]` (0=prefill, 1-15=decode)

---

### Task 3: Capture llama decode dumps

**Files:** None (capture only)

- [ ] **Step 1: Run llama with tensor dumps for same prompt and token count**

```bash
LD_LIBRARY_PATH=/tmp/llama-build/build/bin LLAMA_TENSOR_DUMP=1 \
  /tmp/llama-build/build/bin/llama-completion \
  -m /home/kekz/models/gemma-4-26B-A4B-it-Q4_K_M.gguf \
  -p "What is the capital of France?" -n 15 --temp 0 -no-cnv \
  2>&1 > /tmp/llama_decode_drift.log
echo "Lines: $(wc -l < /tmp/llama_decode_drift.log)"
```

- [ ] **Step 2: Verify decode dumps are present**

```bash
grep '\[DUMP\].*l_out-29' /tmp/llama_decode_drift.log | head -10
```

Expected: Multiple `l_out-29` lines — one per prefill + one per decode step. Shape changes from `[2816,N,1]` (prefill) to `[2816,1,1]` (decode).

- [ ] **Step 3: Count evaluations**

```bash
grep '\[TOKENS\]\|l_out-29.*shape' /tmp/llama_decode_drift.log | head -20
```

Expected: One `[TOKENS]` line, then repeated `l_out-29` entries for each eval.

---

### Task 4: Write comparison script

**Files:**
- Create: `scripts/compare_decode_dumps.py`

- [ ] **Step 1: Create the comparison script**

Create `scripts/compare_decode_dumps.py`:

```python
#!/usr/bin/env python3
"""Compare imp vs llama.cpp per-layer hidden states across decode steps.

Usage:
  python3 scripts/compare_decode_dumps.py /tmp/imp_decode_drift.log /tmp/llama_decode_drift.log

Output: Table of step × layer → relative L2 error (%).
Highlights the first (step, layer) where error exceeds threshold.
"""
import sys
import re
from collections import defaultdict

def parse_imp_l_out(path):
    """Parse imp's [step=N] l_out-L[0] L2=X lines into {(step, layer): L2}."""
    data = {}
    with open(path) as f:
        for line in f:
            # Match: [DEBUG_FWD_ROW] [step=3] l_out-5[0] L2=123.456  ...
            m = re.search(r'\[step=(\d+)\]\s+l_out-(\d+)\[0\]\s+L2=([\d.]+)', line)
            if m:
                step, layer, l2 = int(m.group(1)), int(m.group(2)), float(m.group(3))
                data[(step, layer)] = l2
    return data

def parse_llama_l_out(path):
    """Parse llama's [DUMP] l_out-L lines into {(eval_idx, layer): L2}.

    llama dumps are sequential — each l_out-0 starts a new eval.
    eval 0 = warmup (skip), eval 1 = prefill, eval 2+ = decode steps.
    """
    data = {}
    eval_idx = -1
    with open(path) as f:
        for line in f:
            m = re.search(r'\[DUMP\]\s+l_out-(\d+)\s+.*?L2=([\d.]+)', line)
            if m:
                layer = int(m.group(1))
                l2 = float(m.group(2))
                if layer == 0:
                    eval_idx += 1
                data[(eval_idx, layer)] = l2
    return data

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <imp_log> <llama_log>")
        sys.exit(1)

    imp_data = parse_imp_l_out(sys.argv[1])
    llama_data = parse_llama_l_out(sys.argv[2])

    # Map llama eval indices to imp step indices
    # llama: eval 0 = warmup (2 tokens), eval 1 = warmup (2 tokens),
    #         eval 2 = prefill (6 tokens), eval 3+ = decode
    # imp:   step 0 = prefill, step 1+ = decode
    # Heuristic: find the prefill eval (largest n_tokens) and map from there
    # For simplicity: assume llama eval N+offset maps to imp step N

    imp_steps = sorted(set(s for s, l in imp_data.keys()))
    llama_evals = sorted(set(e for e, l in llama_data.keys()))

    print(f"imp: {len(imp_steps)} steps, llama: {len(llama_evals)} evals")

    # Try to align: find prefill in both
    # imp step 0 = prefill (should have largest token count)
    # llama: last warmup eval before decode evals
    # Use L2 similarity at layer 0 to auto-align
    imp_prefill_l2 = imp_data.get((0, 0), 0)
    best_offset = 0
    best_diff = float('inf')
    for e in llama_evals:
        ll2 = llama_data.get((e, 0), 0)
        diff = abs(imp_prefill_l2 - ll2) / max(imp_prefill_l2, 1e-6)
        if diff < best_diff:
            best_diff = diff
            best_offset = e

    print(f"Auto-aligned: imp step 0 ↔ llama eval {best_offset} (L0 L2 diff: {best_diff*100:.1f}%)")

    # Print comparison table
    layers = [0, 5, 10, 15, 20, 25, 29]
    max_step = min(15, len(imp_steps))

    print(f"\n{'step':>5}", end="")
    for l in layers:
        print(f"  L{l:02d}%", end="")
    print("  | first >10%")
    print("-" * (6 + 7 * len(layers) + 15))

    for s in range(max_step):
        print(f"{s:5d}", end="")
        first_bad = None
        for l in layers:
            imp_l2 = imp_data.get((s, l), 0)
            llama_l2 = llama_data.get((s + best_offset, l), 0)
            if llama_l2 > 1e-6:
                rel_err = abs(imp_l2 - llama_l2) / llama_l2
            else:
                rel_err = 0
            marker = " *" if rel_err > 0.10 else ""
            print(f"  {rel_err*100:5.1f}{marker}", end="")
            if rel_err > 0.10 and first_bad is None:
                first_bad = f"L{l}"
        print(f"  | {first_bad}" if first_bad else "  | ok")

    # Summary
    print(f"\nLegend: % = relative L2 error between imp and llama")
    print(f"  * = error > 10% (potential drift point)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x scripts/compare_decode_dumps.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/compare_decode_dumps.py
git commit -m "scripts: add decode drift comparison tool (imp vs llama l_out L2)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Run comparison and identify drift point

**Files:** None (analysis only)

- [ ] **Step 1: Run the comparison**

```bash
python3 scripts/compare_decode_dumps.py /tmp/imp_decode_drift.log /tmp/llama_decode_drift.log
```

Expected: Table showing per-step, per-layer error percentages. Look for:
- Step where error first crosses 10% at any layer
- Layer that consistently has the highest error
- Whether the error grows in attention layers vs MoE layers

- [ ] **Step 2: If comparison fails due to format mismatch, debug**

Check that imp's dumps have the expected `[step=N] l_out-L[0] L2=X` format:
```bash
grep 'l_out-0\[0\]' /tmp/imp_decode_drift.log | head -5
```

Check llama's dumps have `[DUMP] l_out-0 ... L2=X` format:
```bash
grep '\[DUMP\] l_out-0 ' /tmp/llama_decode_drift.log | head -5
```

Adjust regex patterns in the script if needed.

- [ ] **Step 3: Document findings**

Create a brief summary of where the drift starts. Example:
```
Decode step 3, Layer 5 (global attention): imp L2=45.2, llama L2=48.7 (7.6% error)
Decode step 5, Layer 5: imp L2=38.1, llama L2=52.3 (27.2% error) ← DIVERGENCE POINT
All other layers at step 5: <5% error
→ Root cause: global attention layer (hd=512) RoPE or paged attention
```

- [ ] **Step 4: Commit analysis**

```bash
git add -A
git commit -m "gemma4: decode drift analysis — identified divergence point

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
