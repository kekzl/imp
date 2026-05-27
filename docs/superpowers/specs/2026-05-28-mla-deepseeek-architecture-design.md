# MLA (Multi-head Latent Attention) Architecture Design Spec

**Goal:** Add DeepSeek-V2/V3 architecture support to imp, enabling MLA which replaces full K/V with latent vectors for ~93% KV VRAM reduction.

**Status:** Design spec. Blocked on no local MLA model. Implementation estimated at 3-4 weeks.

---

## What is MLA

Multi-head Latent Attention (DeepSeek-V2, arxiv:2405.04434) compresses KV representations:
- Instead of storing full K[n_kv_heads, head_dim] and V[n_kv_heads, head_dim] per token...
- ...MLA stores a single latent vector c[latent_dim] per token (latent_dim << n_kv_heads × head_dim)
- At attention time, K and V are reconstructed from the latent: K = c @ W_K^T, V = c @ W_V^T
- The KV cache stores c instead of K/V → 93% compression

## DeepSeek-V2/V3 Architecture Details

```
n_heads = 128
head_dim = 128  
n_kv_heads = 1 (MLA)
latent_dim = 512 (V2) or 1024 (V3)
rope_head_dim = 64 (separate from value head)

Per-token KV cache: 512 × 2B = 1 KB (vs 128 × 128 × 2 × 2B = 64 KB for MHA)
Compression: 64x
```

## What imp Needs

### 1. Model Loading
- New arch detection: `DeepSeek-V2`, `DeepSeek-V3` in GGUF/SafeTensors
- Weight layout: `W_UK` (up-projection for K), `W_UV` (up-projection for V), separate from standard Q/K/V projections
- Rope: partial rope on a subset of K dimensions (rope_head_dim < head_dim)

### 2. KV Cache
- Store latent vectors instead of full K/V: `[n_layers, max_blocks, block_size, latent_dim]`
- Block size stays 16 tokens, but each block is much smaller
- No separate K and V caches — one unified latent cache

### 3. Attention Kernel
- Prefill: Q @ (latent @ W_K)^T — can be fused as Q @ W_K @ latent^T
- Decode: same structure but paged over latent blocks instead of K/V blocks
- RoPE applied only to the rope_head_dim subset of K
- Absorb W_K into Q projection: Q_absorbed = Q @ W_K, then attention is Q_absorbed @ latent^T

### 4. Rope Handling
- DeepSeek uses "decoupled rope": rope applied to a separate set of dimensions
- The rope'd portion of K is stored separately from the latent
- Cache layout: `[latent_dim + rope_head_dim]` per token

## Gating Conditions

This work should proceed when:
1. A DeepSeek-V2/V3 model is available in GGUF or NVFP4 SafeTensors format
2. The model fits on RTX 5090 (32 GB)
3. There's a reference engine (llama.cpp or vLLM) producing correct output for A/B validation

## Files (Preliminary)

| File | Action |
|---|---|
| `src/model/model_arch.h` | Add `DEEPSEEK_V2`, `DEEPSEEK_V3` arch enums |
| `src/model/gguf_loader.cpp` | MLA weight name mapping |
| `src/memory/kv_cache.h` | Latent-mode block layout variant |
| `src/exec/executor_attention.cu` | MLA attention dispatch |
| `src/compute/attention_mla.cu` | New kernel: absorbed-Q × latent paged attention |
| `src/compute/rope.cu` | Decoupled rope for partial K dims |
| `tests/test_mla.cu` | Correctness tests |

## References

- DeepSeek-V2: arxiv:2405.04434
- DeepSeek-V3: arxiv:2412.19437
- Retrofitting MLA: arxiv:2502.14837 (theoretical, not yet practical)
