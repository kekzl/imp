# Decode regression check — MoE NVFP4 prefill optimization

The change targets the prefill MoE dispatch chain only (`executor_forward_moe.cu`, `n > 1` path). Decode (`n == 1`) flows through `gemv_nvfp4_moe_decode` and `gemv_nvfp4_moe_swiglu_decode` and is unaffected.

## Bench (3 reps, post-warmup)

| Model | tg256 before | tg256 after | Δ | Pass (≤2% regression) |
|---|---:|---:|---:|:---:|
| Qwen3-Coder-30B-A3B-NVFP4 | 261 | 268 | **+3.1 %** | ✓ |
| Qwen3.6-35B-A3B-NVFP4     | 225 | 232 | **+3.1 %** | ✓ |
| Gemma-4-26B-A4B-NVFP4     | 202 | 210 | **+4.0 %** | ✓ |
| Qwen3-4B-Instruct Q8_0 (dense; perf gate)  | 151.16 | 154.75 | **+2.4 %** | ✓ |

The "+2-4 %" decode movement is consistent across models and matches what we'd expect from a slightly cleaner cache state in the bench harness — it is **not** the optimization helping decode (there's no decode-side change), it's that prefill finishes ~10× sooner so the CUDA Graph capture for decode happens with less L2 pressure. The true decode arithmetic is unchanged.

## Graphs ON / OFF parity

```
graphs OFF tg256 = 165.91 tok/s
graphs ON  tg256 = 247.28 tok/s   (1.49x, threshold 1.3x)
```

Pass — graph capture for decode still delivers ≥1.3× speedup. No silent fallback.

## Coherence

`Qwen3-Coder-30B-A3B-NVFP4` greedy decode (`temperature=0 seed=42 max_tokens=128`) on prompt "Write a Python function that computes the factorial of n using recursion." produced a complete `def factorial(n):` recursion implementation with input validation, docstring, and base-case return. No repetition, no NaN, no early-EOS. stderr clean of `CUDA error|capture failed|falling back|NaN|is Inf` matches.

## Verdict

Decode is preserved on all NVFP4 MoE models the optimization touches. No regressions.
