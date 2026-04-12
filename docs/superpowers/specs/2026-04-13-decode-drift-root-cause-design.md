# Design: Decode-Drift Root Cause Analysis for Gemma-4

## Context

Gemma-4 26B-A4B produces correct first sentences ("Capital of France is **Paris**.") but decode degenerates after ~15 tokens. The model never generates the `<turn|>` stop token (106). We need to find the exact decode step, layer, and operation where imp diverges from llama.cpp.

## Current State

- Prefill: correct (first token matches llama)
- Decode steps 1-8: mostly correct (thematically relevant tokens)
- Decode steps 9+: degeneration ("own", "-", "```", repeated patterns)
- All MMVQ unit tests pass (< 0.1% error)
- dp4a GEMV matches llama router logits within 0.7% at L0

## Approach: Step-Tagged Tensor Dump + Automated Diff

### Phase 1: Add step counter to imp debug dumps

Modify `executor_debug.h` or `executor_forward.cu` to track a global decode step counter. Every `[DEBUG_FWD]` line gets tagged with `[step=N]` so we can distinguish decode step 1 from step 10.

Add 4 dump checkpoints per layer per step:
1. `attn_out` — after attention (post O-projection + post_attn_norm + residual)
2. `moe_out` — after MoE experts + shared expert + post_ffn_norm + residual
3. `l_out` — after layer_out_scale (= llama's `l_out-N`)

Format per line: `[step=S layer=L] name sum=X L2=Y [0..2]=a b c`

### Phase 2: Capture llama reference dumps

Run llama's `llama-completion` with `LLAMA_TENSOR_DUMP=1` for the same prompt ("What is the capital of France?" with Gemma chat template) for 15 decode steps. Parse output into a structured format (one file per decode step, or one CSV with step/layer/tensor/values columns).

Key tensors from llama: `attn_out-N`, `ffn_moe_combined-N`, `l_out-N` for each layer 0-29.

### Phase 3: Diff script

Python script that:
1. Parses imp dump files (grep for `[step=S layer=L]` pattern)
2. Parses llama dump files (grep for tensor names + row indices)
3. For each (step, layer): computes relative error between imp and llama L2 norms
4. Reports the first (step, layer) where relative error exceeds a threshold (e.g., 10%)
5. Outputs a summary table: step × layer → error%

### Phase 4: Root cause identification

With the error map, identify:
- Does error grow uniformly across layers? → systematic issue (all GEMMs)
- Does error spike at specific layers? → layer-specific bug (e.g., global attention layers 5/11/17/23/29)
- Does error spike at a specific step? → decode-specific bug (KV cache write, paged attention)
- Does error grow in MoE but not attention? → MoE routing amplification

## Success Criteria

Identify: "At decode step X, layer Y, operation Z, imp diverges from llama by >10%. All earlier steps are <3%." This pinpoints where to focus the fix.

## Files

| File | Change |
|------|--------|
| `src/graph/executor_forward.cu` | Add step counter, pass to debug dumps |
| `src/graph/executor_debug.h` | Add step tag to dump format |
| `scripts/compare_decode_dumps.py` | New: parse + diff imp vs llama dumps |
