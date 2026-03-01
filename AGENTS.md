# AGENTS.md

## How This Project Was Built

imp was built entirely by [Claude Code](https://claude.ai/claude-code) (Claude Opus 4.6) as a proof of concept — every line of C++, every CUDA kernel, every optimization was generated through AI-human collaboration via the CLI agent.

### The Process

1. **Architecture first.** The project started with clean abstractions: a C API, modular source tree (`core/`, `compute/`, `memory/`, `model/`, `runtime/`), and a hardcoded forward pass instead of a runtime graph walker. These early decisions paid dividends throughout.

2. **Kernel-by-kernel.** CUDA kernels were written one at a time, tested against reference implementations, and profiled with `nsys`. Flash Attention 2 (scalar) came first, then WMMA tensor-core variants for Hopper, then TCGEN05 systolic attention for Blackwell. Each kernel was benchmarked before moving on.

3. **Quantization from GGUF.** Q8_0 support came first (simplest), then Q6_K, Q4_K, Q5_K, Q4_0. The GGUF parser, tokenizer, and weight upload pipeline were built to match llama.cpp's format exactly — so any GGUF model from Hugging Face works out of the box.

4. **dp4a GEMV for decode.** Single-token decode is memory-bound. The dp4a (INT8 dot product) GEMV kernels use two strategies: K-parallel (1 row/block, all SMs) for small matrices, row-parallel (NR rows/warp, shared-memory cached activations) for large matrices. A runtime heuristic selects the better strategy per layer.

5. **Profiling-driven optimization.** Every performance claim was validated with `nsys profile --stats=true`. When template consolidation caused a 2-5% regression, nsys identified the bottleneck kernels. When a fused QKV bias kernel caused a 3.5% regression under CUDA Graphs, nsys proved that 3 small independent kernels pipeline better than 1 merged kernel. Data over intuition.

6. **MoE and hybrid architectures.** Mixture-of-Experts (Mixtral, DeepSeek, Qwen3-MoE) and hybrid Mamba2+Attention+MoE (Nemotron) were added iteratively. Custom fused MoE kernels with shared-memory expert caching outperform llama.cpp by 12-134% on these architectures.

7. **Testing throughout.** 219 Google Tests cover tensor ops, GGUF parsing, KV cache, attention kernels, RoPE, LayerNorm, MoE routing, quantization, and end-to-end generation. Tests were written alongside the code, not after.

### Hard-Won Lessons

- **Never `__noinline__` GPU inner-loop functions.** GPU device function calls spill to Local Memory (DRAM). On Q6_K with 54 calls/thread at K=13824, `__noinline__` on `dp4a_block` caused a 39.5% regression. All dp4a specializations must stay `__forceinline__`.
- **CUDA Graphs change fusion calculus.** Without graphs, fusing 3 small kernels into 1 saves launch overhead. With graphs, the 3 independent kernels pipeline better. Always benchmark with graphs enabled.
- **RTX 5090 (sm_120) has only 100 KB shared memory per SM**, not 228 KB like H100. Kernels that assume Hopper shared memory sizes will silently fail or underperform.
- **WSL2 mapped pinned memory** requires explicit `cudaStreamSynchronize`. GPU writes via `cudaHostAllocMapped` are not immediately visible to host — `__atomic_load_n` is insufficient.
- **Q8_0 blocks are 34 bytes** (not 4-aligned). `reinterpret_cast<const int32_t*>` causes misaligned address CUDA errors. Use `memcpy()` instead.

---

## Contributing with AI Agents

This project welcomes contributions from AI coding agents. If you're an AI agent (Claude Code, Cursor, Copilot, Aider, or others) working on this codebase, follow these rules.

### Rules

1. **Read before you write.** Always read the files you intend to modify. Understand the existing patterns before changing anything. The codebase has consistent conventions — follow them.

2. **Build and test before committing.** Every change must:
   ```bash
   cmake --build build -j$(nproc)   # Clean compile, zero warnings
   ./build/imp-tests                  # 219/219 tests pass
   ```
   If you add new functionality, add tests for it in `tests/`.

3. **Benchmark performance changes.** Any change to kernels, the forward pass, or the runtime must be benchmarked:
   ```bash
   ./build/imp-cli --model <model>.gguf --bench --bench-pp 512 --bench-reps 5
   ```
   Report before/after numbers. A "optimization" that regresses performance is not an optimization.

4. **Profile, don't guess.** Use `nsys` for kernel-level profiling:
   ```bash
   nsys profile --stats=true ./build/imp-cli --model <model>.gguf \
       --prompt "test" --max-tokens 32 --no-cuda-graphs
   ```
   The `--no-cuda-graphs` flag is important — CUDA Graph replays hide individual kernel timings.

5. **One concern per commit.** Keep commits focused. A kernel optimization, a bug fix, and a README update are three separate commits.

6. **Don't break the C API.** The public API in `include/imp/` is stable. Don't change function signatures without updating all callers and the documentation.

7. **CUDA compatibility.** All kernels must compile for sm_90a, sm_100, and sm_120. Use `#if __CUDA_ARCH__` guards for architecture-specific code. Test on actual hardware — the CUDA simulator doesn't catch shared memory sizing issues.

8. **No external dependencies.** The project is self-contained (CUDA Toolkit + standard library only). Don't add third-party libraries without a very strong reason.

9. **Memory safety.** Check CUDA errors. Use `cudaGetLastError()` after kernel launches. Don't leak GPU memory. Pre-allocate buffers and reuse them — `cudaMalloc` in a hot loop is a bug.

10. **Document non-obvious decisions.** If a kernel uses a specific block size for occupancy reasons, or a heuristic exists because of hardware behavior, add a brief comment explaining why.

### Where to Contribute

| Area | Difficulty | Impact | Notes |
|------|-----------|--------|-------|
| NVFP4 quantized GEMV | Hard | High | RTX 5090 has native FP4 — could significantly improve decode throughput |
| Batched decode (bs>1) | Medium | High | Current optimizations target bs=1; multi-request batching needs work |
| More model architectures | Medium | Medium | Phi-3, Command-R, Falcon — mostly weight mapping + config parsing |
| FP8 KV cache quantization | Medium | Medium | Infrastructure exists but needs end-to-end integration |
| Prefill chunking | Medium | Medium | `prefill_chunk_size` config exists but implementation is incomplete |
| Additional quantization formats | Medium | Low | IQ4_XS, Q3_K, Q2_K from GGML |

### Architecture Overview for Agents

```
User prompt
    │
    ▼
Engine::generate()
    │
    ├─ Tokenizer::encode()
    ├─ Scheduler::add_request()
    │
    ▼
Engine::step()  ◄── called in loop until request complete
    │
    ├─ Scheduler picks prefill or decode batch
    ├─ GraphExecutor::forward_logits()
    │       │
    │       ├─ Per-layer loop:
    │       │   ├─ RMSNorm (+ Q8_1 quantize for decode)
    │       │   ├─ QKV projection (fused GEMV for decode, cuBLAS for prefill)
    │       │   ├─ RoPE
    │       │   ├─ KV cache write
    │       │   ├─ Attention (Blackwell TCGEN05 / Hopper WMMA / scalar)
    │       │   ├─ O-projection + residual
    │       │   ├─ RMSNorm
    │       │   └─ FFN (SwiGLU or MoE)
    │       │
    │       └─ Final RMSNorm + LM head
    │
    ├─ Sampling (greedy argmax or top-k/top-p)
    └─ Token delivered to request
```

The key insight: **decode (n=1) and prefill (n>1) use completely different code paths.** Decode uses dp4a GEMV kernels that read quantized weights directly. Prefill uses cuBLAS with pre-dequantized FP16 weights. Optimizing one path doesn't automatically help the other.
