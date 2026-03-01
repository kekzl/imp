# imp — GEMV Dispatch Reference

Complete map of quantization type support across all decode (batch_size=1) dispatch points
for Dense and MoE model architectures.

Last updated: 2026-03-01

---

## 1. GEMV Variant Matrix

All dp4a GEMV function variants declared in `src/compute/gemm.h`, grouped by quant type.

| Variant               | Q6_K | Q8_0 | Q4_0 | Q4_K | Q5_K | Q2_K | Q3_K | Purpose                          |
|-----------------------|------|------|------|------|------|------|------|----------------------------------|
| `basic`               |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   | Standard GEMV                    |
| `_residual`           |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   | GEMV + residual add fused        |
| `_qkv_fused`          |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   | Fused Q/K/V triple projection    |
| `_fp32`               |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   | FP32 output (LM head)            |
| `_inline_quant`       |  Y   |  Y   |  —   |  Y   |  Y   |  Y   |  Y   | Inline FP16→Q8_1 (unused)        |
| `_moe_decode`         |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   | MoE per-expert GEMV              |
| `_moe_gate_up_fused`  |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   | MoE fused gate+up projection     |

Additional non-dp4a (FP16 dequant) variants exist for Q6_K and Q8_0 only:
`gemv_q6k`, `gemv_q8_0`, `gemv_q6k_moe_decode`, `gemv_q8_0_moe_decode`,
`gemv_q6k_moe_gate_up_fused`, `gemv_q8_0_moe_gate_up_fused`.

Note: `_inline_quant` variants are registered with PDL but never called by the executor.
The executor uses separate `quantize_fp16_to_q8_1` + K-parallel GEMV for higher occupancy.

---

## 2. Dense Decode Dispatch

All dispatch points in `src/graph/executor.cu` for dense transformer decode (n=1).
Every dp4a path requires `q8_1_buf_ != nullptr && d8_buf_ != nullptr && input.dtype == FP16`.

### 2.1 QKV Fused (line ~2235)

Single kernel reads input once, computes Q, K, V projections simultaneously.
Requires all three weight matrices to share the same quant type.

Preprocessing: `rmsnorm_quantize_q8_1()` — fused RMSNorm + Q8_1 quantization.

| QType | Function                       |
|-------|--------------------------------|
| Q6_K  | `gemv_qkv_fused_q6k_q8_1`     |
| Q4_0  | `gemv_qkv_fused_q4_0_q8_1`    |
| Q4_K  | `gemv_qkv_fused_q4_k_q8_1`    |
| Q5_K  | `gemv_qkv_fused_q5_k_q8_1`    |
| Q2_K  | `gemv_qkv_fused_q2_k_q8_1`    |
| Q3_K  | `gemv_qkv_fused_q3_k_q8_1`    |
| Q8_0  | `gemv_qkv_fused_q8_0_q8_1`    |

Fallback: 3 separate calls to `gemm_dispatch()` for Wq, Wk, Wv.

### 2.2 Wo Residual (line ~2505)

Fused output projection + residual addition. Uses separate `quantize_fp16_to_q8_1`
followed by K-parallel `_residual` GEMV that writes `y = W·x + residual` directly.

Condition: `!has_post_attn_norm && n==1 && dp4a buffers available`.

| QType | Function                       |
|-------|--------------------------------|
| Q6_K  | `gemv_q6k_q8_1_residual`      |
| Q4_0  | `gemv_q4_0_q8_1_residual`     |
| Q4_K  | `gemv_q4_k_q8_1_residual`     |
| Q5_K  | `gemv_q5_k_q8_1_residual`     |
| Q2_K  | `gemv_q2_k_q8_1_residual`     |
| Q3_K  | `gemv_q3_k_q8_1_residual`     |
| Q8_0  | `gemv_q8_0_q8_1_residual`     |

Fallback paths (in priority order):
1. **beta=1 cuBLAS** (`will_fuse_o_beta1`): `gemm(ao, wo_fp16, h, 1.0, 1.0)` when fp16_cache hit and batch > 1
2. **gemm_dispatch()**: standard routing (see section 4)
3. **Post-attn norm + FP32 accum**: `rmsnorm_fp32_accum_to_fp16_kernel` (Gemma-3 path)

### 2.3 FFN Gate+Up Fused (line ~2635)

Single kernel computes both gate and up projections. Input dispatched by `w_gate_qtype`.

Preprocessing: `rmsnorm_quantize_q8_1()` — fused FFN norm + Q8_1 quantization.

| QType | Dispatcher                     |
|-------|--------------------------------|
| Q6_K  | `gemv_gate_up_fused` (internal Q6_K_Traits) |
| Q8_0  | `gemv_gate_up_fused` (internal Q8_0_Traits) |
| Q4_0  | `gemv_gate_up_fused` (internal Q4_0_Traits) |
| Q4_K  | `gemv_gate_up_fused` (internal Q4_K_Traits) |
| Q5_K  | `gemv_gate_up_fused` (internal Q5_K_Traits) |
| Q2_K  | `gemv_gate_up_fused` (internal Q2_K_Traits) |
| Q3_K  | `gemv_gate_up_fused` (internal Q3_K_Traits) |

`gemv_gate_up_fused()` (`gemm.cu:1736`) takes `GGMLQuantType` and dispatches internally
via `DISPATCH_GATE_UP` macro with K-par vs row-par heuristic.

Fallback: 2 separate calls to `gemm_dispatch()` for w_gate and w_up.

### 2.4 FFN Down Residual (line ~2676)

Fused activation + Q8_1 quantization + GEMV + residual addition.

Condition: `!has_post_ffn_norm && n==1 && dp4a buffers available`.

Preprocessing: `swiglu_quantize_q8_1()` or `geglu_quantize_q8_1()` — fused activation + quantization.

| QType | Function                       |
|-------|--------------------------------|
| Q6_K  | `gemv_q6k_q8_1_residual`      |
| Q4_0  | `gemv_q4_0_q8_1_residual`     |
| Q4_K  | `gemv_q4_k_q8_1_residual`     |
| Q5_K  | `gemv_q5_k_q8_1_residual`     |
| Q2_K  | `gemv_q2_k_q8_1_residual`     |
| Q3_K  | `gemv_q3_k_q8_1_residual`     |
| Q8_0  | `gemv_q8_0_q8_1_residual`     |

### 2.5 FFN Down FP32 Accum (line ~2721)

Post-FFN-norm path with FP32 accumulation (Gemma-3).
Same activation fusion, but basic GEMV (no residual) followed by
`rmsnorm_fp32_accum_to_fp16_kernel` for fused post-norm + accumulate.

Supported QTypes: Q6_K, Q8_0, Q4_0, Q4_K, Q5_K, Q2_K, Q3_K.

### 2.6 LM Head / Output Projection (line ~4347)

FP32 output for sampling precision. Used for both prefill-last-token and decode.

Preprocessing: `rmsnorm_quantize_q8_1()`.

| QType | Function                       |
|-------|--------------------------------|
| Q6_K  | `gemv_q6k_q8_1_fp32`          |
| Q4_0  | `gemv_q4_0_q8_1_fp32`         |
| Q4_K  | `gemv_q4_k_q8_1_fp32`         |
| Q5_K  | `gemv_q5_k_q8_1_fp32`         |
| Q2_K  | `gemv_q2_k_q8_1_fp32`         |
| Q3_K  | `gemv_q3_k_q8_1_fp32`         |
| Q8_0  | `gemv_q8_0_q8_1_fp32`         |

Fallback: `rmsnorm()` + cuBLAS `gemm()` (FP16 output, cast to FP32).

---

## 3. MoE Decode Dispatch

All dispatch points for Mixture-of-Experts decode in `src/graph/executor.cu`.
MoE layers replace the dense FFN with routed expert projections.

### 3.1 Router / Gate Logits (line ~2903)

Computes expert selection scores. Always FP16 weights (small matrix: n_experts x d_model).

| Path    | Function              | Output |
|---------|-----------------------|--------|
| Decode  | `gemv_gate_fp32()`    | FP32   |
| Prefill | cuBLAS `gemm()` + cast| FP32   |

### 3.2 Expert Gate+Up Fused — dp4a (line ~3012)

Gated experts (SwiGLU): single kernel computes both gate and up projections
across top_k selected experts.

Preprocessing: `rmsnorm_quantize_q8_1()` — shared across all experts.

| QType | Function                              |
|-------|---------------------------------------|
| Q6_K  | `gemv_q6k_q8_1_moe_gate_up_fused`    |
| Q4_K  | `gemv_q4_k_q8_1_moe_gate_up_fused`   |
| Q5_K  | `gemv_q5_k_q8_1_moe_gate_up_fused`   |
| Q4_0  | `gemv_q4_0_q8_1_moe_gate_up_fused`   |
| Q2_K  | `gemv_q2_k_q8_1_moe_gate_up_fused`   |
| Q3_K  | `gemv_q3_k_q8_1_moe_gate_up_fused`   |
| Q8_0  | `gemv_q8_0_q8_1_moe_gate_up_fused`   |

Parameters: `q8_1_stride=0, d8_stride=0` (shared input, not per-expert).

### 3.3 Expert Up Only — dp4a (line ~3057)

Non-gated experts (Nemotron: relu² activation instead of SwiGLU).
Up projection only, dispatched via `_moe_decode` variant.

| QType | Function                         |
|-------|----------------------------------|
| Q6_K  | `gemv_q6k_q8_1_moe_decode`      |
| Q4_0  | `gemv_q4_0_q8_1_moe_decode`     |
| Q4_K  | `gemv_q4_k_q8_1_moe_decode`     |
| Q5_K  | `gemv_q5_k_q8_1_moe_decode`     |
| Q2_K  | `gemv_q2_k_q8_1_moe_decode`     |
| Q3_K  | `gemv_q3_k_q8_1_moe_decode`     |
| Q8_0  | `gemv_q8_0_q8_1_moe_decode`     |

### 3.4 Expert Down — dp4a (line ~3109)

Down projection after activation. Input is per-expert (different activations per expert).

Preprocessing (fused activation + Q8_1 quantization):
- Gated: `swiglu_quantize_q8_1()` over `top_k * d_ff` elements
- Non-gated: `relu_sqr_quantize_q8_1()` over `top_k * d_ff` elements

| QType | Function                         |
|-------|----------------------------------|
| Q6_K  | `gemv_q6k_q8_1_moe_decode`      |
| Q4_0  | `gemv_q4_0_q8_1_moe_decode`     |
| Q4_K  | `gemv_q4_k_q8_1_moe_decode`     |
| Q5_K  | `gemv_q5_k_q8_1_moe_decode`     |
| Q2_K  | `gemv_q2_k_q8_1_moe_decode`     |
| Q3_K  | `gemv_q3_k_q8_1_moe_decode`     |
| Q8_0  | `gemv_q8_0_q8_1_moe_decode`     |

Parameters: `q8_1_stride=d_ff/32, d8_stride=d_ff/32` (per-expert input offsets).

### 3.5 Expert Gate+Up — FP16 Fallback (line ~3076)

When dp4a buffers are unavailable. Uses raw FP16 input (no Q8_1 quantization).

| QType | Function                         |
|-------|----------------------------------|
| Q6_K  | `gemv_q6k_moe_gate_up_fused`    |
| Q8_0  | `gemv_q8_0_moe_gate_up_fused`   |

Other quant types: not supported on this path (dp4a always available in practice).

### 3.6 Expert Down — FP16 Fallback (line ~3155)

| QType | Function                  |
|-------|---------------------------|
| Q6_K  | `gemv_q6k_moe_decode`    |
| Q8_0  | `gemv_q8_0_moe_decode`   |

### 3.7 Shared Expert FFN (line ~3885)

Optional shared expert (Nemotron, DeepSeek V3). Uses standard `gemm_dispatch()` —
same dispatch logic as dense FFN (section 4). Supports all quant types.

| Weight         | Dispatch               |
|----------------|------------------------|
| w_up_shared    | `gemm_dispatch()`      |
| w_gate_shared  | `gemm_dispatch()`      |
| w_down_shared  | `gemm_dispatch()`      |

### 3.8 Weighted Sum + Residual (line ~3164)

Final step: weighted combination of expert outputs + residual addition.
Always FP16, handled by `moe_weighted_sum_residual()`.

---

## 4. MoE Prefill Dispatch

Multiple prefill strategies, selected by model size and quant type.

| Path                          | Condition                 | QTypes Supported                |
|-------------------------------|---------------------------|---------------------------------|
| **Q6_K Fused** (line ~3214)   | `can_fused_q6k && n>1`   | Q6_K only                       |
| **FP8 Batch** (line ~3339)    | Q6_K→FP8 E4M3            | Q6_K only (FP8 dequant)        |
| **FP16 Batch** (line ~3274)   | dequant → grouped GEMM   | All `dequant_gpu_supported`     |
| **Serial Fallback** (line ~3780) | per-expert loop        | All (dequant → cuBLAS per expert) |

---

## 5. `gemm_dispatch()` — Generic Routing

Central dispatch function (`executor.cu:785`) used for all non-fused GEMV/GEMM.
Called for individual Q/K/V projections, Wo fallback, FFN fallback, shared experts.

Dispatch order (first match wins):

| Priority | Condition                          | QType    | Action                            |
|----------|------------------------------------|----------|-----------------------------------|
| 1        | `n==1 && q8_buf`                   | Q4_0     | `quantize_q8_1` → `gemv_q4_0_q8_1` |
| 2        | fp16_cache hit                     | Q4_1     | cuBLAS `gemm()` with cached FP16  |
| 3        | no cache                           | Q4_1     | `quant_gemm_int4()`               |
| 4        | `n==1 && q8_buf`                   | Q6_K     | `quantize_q8_1` → `gemv_q6k_q8_1` |
| 5        | `n==1 && q8_buf`                   | Q8_0     | `quantize_q8_1` → `gemv_q8_0_q8_1` |
| 6        | `n==1 && q8_buf`                   | Q4_K     | `quantize_q8_1` → `gemv_q4_k_q8_1` |
| 7        | `n==1 && q8_buf`                   | Q5_K     | `quantize_q8_1` → `gemv_q5_k_q8_1` |
| 8        | `n==1 && q8_buf`                   | Q2_K     | `quantize_q8_1` → `gemv_q2_k_q8_1` |
| 9        | `n==1 && q8_buf`                   | Q3_K     | `quantize_q8_1` → `gemv_q3_k_q8_1` |
| 10       | `n==1 && dequant_scratch`          | Q6_K     | `gemv_q6k()` (FP16 dequant GEMV) |
| 11       | `n==1 && dequant_scratch`          | Q8_0     | `gemv_q8_0()` (FP16 dequant GEMV)|
| 12       | `fp16_cache && dequant_supported`  | *many*   | cuBLAS `gemm()` with cached FP16 |
| 13       | `dequant_scratch && dequant_supported` | *many* | `dequant_gpu()` → cuBLAS `gemm()` |
| 14       | Fallback                           | FP16/BF16| cuBLAS `gemm()`                   |

`dequant_gpu_supported` types: Q6_K, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q8_K.

---

## 6. Summary: QType Coverage per Dispatch Point

### Dense

| Dispatch Point        | Q6_K | Q8_0 | Q4_0 | Q4_K | Q5_K | Q2_K | Q3_K | Q4_1 |
|-----------------------|------|------|------|------|------|------|------|------|
| QKV Fused             |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |  —   |
| Wo Residual           |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |  —   |
| Gate+Up Fused         |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |  —   |
| Down Residual         |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |  —   |
| Down FP32 Accum       |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |  —   |
| LM Head (FP32 out)    |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |  —   |
| gemm_dispatch (dp4a)  |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |  —   |
| gemm_dispatch (other) |  Y*  |  Y*  |  Y*  |  Y*  |  Y*  |  Y*  |  Y*  |  Y   |

*via fp16_cache or on-the-fly dequant → cuBLAS

Q4_1 is handled exclusively through `quant_gemm_int4()` or the fp16_cache path
(no dp4a GEMV variants exist for Q4_1).

### MoE

| Dispatch Point          | Q6_K | Q8_0 | Q4_0 | Q4_K | Q5_K | Q2_K | Q3_K |
|-------------------------|------|------|------|------|------|------|------|
| Gate+Up Fused (dp4a)    |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |
| Up Only (dp4a)          |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |
| Down (dp4a)             |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |
| Gate+Up (FP16 fallback) |  Y   |  Y   |  —   |  —   |  —   |  —   |  —   |
| Down (FP16 fallback)    |  Y   |  Y   |  —   |  —   |  —   |  —   |  —   |
| Router                  | FP16 | FP16 | FP16 | FP16 | FP16 | FP16 | FP16 |
| Shared Expert FFN       |  Y*  |  Y*  |  Y*  |  Y*  |  Y*  |  Y*  |  Y*  |

*Shared experts use `gemm_dispatch()` — full dense-path support.

MoE FP16 fallback only covers Q6_K and Q8_0. Other quant types require dp4a
(always available when `q8_1_buf_` is allocated, which is the default).

### MoE Prefill

| Path             | Q6_K | Q8_0 | Q4_0 | Q4_K | Q5_K | Q2_K | Q3_K |
|------------------|------|------|------|------|------|------|------|
| Q6_K Fused       |  Y   |  —   |  —   |  —   |  —   |  —   |  —   |
| FP8 Batch        |  Y   |  —   |  —   |  —   |  —   |  —   |  —   |
| FP16 Batch       |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |
| Serial Fallback  |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |  Y   |

---

## 7. Type Conversions

All dtype conversions in the inference pipeline, organized by where they occur.

### 7.1 Init-time: Eager Weight Dequantization (line ~1626)

Runs once at engine init via `dequant_all_weights_to_fp16()`. Converts all supported
quantized weights to FP16 and caches them on GPU for zero-overhead prefill.

```
Quantized (GGUF bytes) ──dequant_gpu()──▸ FP16 (fp16_cache_)
```

**Supported types**: Q6_K, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q8_K

**Priority order**: Attention (Wq, Wk, Wv, Wo) first, then SSM, shared experts, dense FFN.
Budget-limited by free VRAM minus 1 GiB reserve.

After individual weight caching, fused weight tensors are assembled (D2D copy):
- **Fused KV** (line ~1688): `[Wk; Wv]` → `[2*nkv*hd, d_model]` for strided batched GEMM
- **Fused Gate+Up** (line ~1722): `[w_gate; w_up]` → `[2*d_ff, d_model]` for strided batched GEMM

### 7.2 Per-token: Input Quantization

These kernels convert FP16 activations to Q8_1 for dp4a GEMV consumption.

| Kernel | Conversion | Where used | Line |
|--------|-----------|------------|------|
| `rmsnorm_quantize_q8_1` | FP16 → RMSNorm → Q8_1 + scales | QKV fused, FFN gate+up fused, MoE input, LM head | ~2228, ~2642, ~2860, ~4348 |
| `quantize_fp16_to_q8_1` | FP16 → Q8_1 + scales | Wo residual, gemm_dispatch dp4a paths | ~2514, ~796-861 |
| `swiglu_quantize_q8_1` | FP16 gate + FP16 up → SwiGLU → Q8_1 | Dense down residual, MoE down (dp4a) | ~2687, ~3115 |
| `geglu_quantize_q8_1` | FP16 gate + FP16 up → GEGLU → Q8_1 | Dense down (GEGLU models) | ~2691 |
| `relu_sqr_quantize_q8_1` | FP16 → relu² → Q8_1 | MoE down non-gated (Nemotron) | ~3124 |

**Q8_1 block format**: 36 bytes per 32 elements — `d` (FP32 scale) + `qs[32]` (INT8) + `ds` (FP32 sum).
Stored in `q8_1_buf_` + `d8_buf_` (pre-allocated at init).

### 7.3 Per-token: KV Cache Write

Converts FP16 K/V projections to the configured KV cache dtype.

| Cache dtype | Kernel | Conversion | Scale storage |
|------------|--------|------------|---------------|
| **FP16** | `write_kv_cache_fused_kernel` | FP16 → FP16 (copy to paged block) | None |
| **FP8 E4M3** | `write_kv_cache_fp8_kernel` | FP16 → FP8 E4M3 (per-layer scale) | `kv_scales_[layer]` (calibrated once from first batch) |
| **INT8** | `write_kv_cache_int8_kernel` | FP16 → INT8 (per-head scale) | `scale_pool_` — `[block, slot, kv_head]` as FP16 |

FP8 calibration (line ~1999): `calibrate_fp8_scale()` computes `absmax / 448.0` from
K and V data on the first write to each layer. Reused for all subsequent writes.

INT8 quantization (line ~461): Warp-level absmax reduction per head → `scale = absmax / 127`,
`int8 = round(val * 127 / absmax)`. No calibration step needed.

### 7.4 Per-token: KV Cache Read (Attention Decode)

Dequantization happens inside the paged attention kernels.

| Cache dtype | Q·K inner loop | V accumulation | Scale load |
|------------|----------------|----------------|------------|
| **FP16** | FP16 FMA | FP16 FMA | None |
| **FP8 E4M3** | `fp8_e4m3_to_float()` → FP32 FMA | `fp8_e4m3_to_float()` → FP32 FMA | Per-layer `kv_scale` (1 float mul after warp reduce) |
| **INT8** | `__dp4a(K_int8x4, Q_int8x4, acc)` → INT32 | `(float)(int8_t)byte` → FP32 FMA | Per-head-per-token `half` from `scale_pool_` |

INT8 Q·K: Q vector quantized to INT8 once in registers at kernel start. dp4a computes
4 multiply-adds per instruction. Final dot = `(float)sumi * q_scale * k_scale * softmax_scale`.

### 7.5 Per-token: On-the-fly Weight Dequantization

When fp16_cache misses (budget exhausted) or during MoE prefill.

| Path | Conversion | Where |
|------|-----------|-------|
| `dequant_gpu()` | GGUF quant → FP16 | gemm_dispatch fallback (line ~887), MoE prefill FP16 batch (line ~3351) |
| `dequant_gpu_fp8()` | GGUF quant → FP8 E4M3 | MoE prefill FP8 batch (line ~3416) |
| `dequant_expert()` | GGUF quant → FP16 (single expert) | MoE serial fallback (line ~3560) |

`dequant_gpu` kernels per type (all in `src/quant/dequant_gpu.cu`):

| Kernel | Block format | Dequant formula |
|--------|-------------|-----------------|
| `dequant_q4_0_kernel` | 18B/32elem: `d[2] + qs[16]` | `d * (nibble - 8)` |
| `dequant_q4_1_kernel` | 20B/32elem: `d[2] + m[2] + qs[16]` | `d * nibble + m` |
| `dequant_q5_0_kernel` | 22B/32elem: `d[2] + qh[4] + qs[16]` | `d * (q5 - 16)` |
| `dequant_q5_1_kernel` | 24B/32elem: `d[2] + m[2] + qh[4] + qs[16]` | `d * q5 + m` |
| `dequant_q8_0_kernel` | 34B/32elem: `d[2] + qs[32]` | `d * qs[i]` |
| `dequant_q8k_kernel` | 292B/256elem: `d[4] + qs[256] + bsums[32]` | `d * qs[i]` |
| `dequant_q6k_v2_kernel` | 210B/256elem: interleaved 6-bit | `d * sc[i] * q6val` |
| `dequant_q2k_kernel` | 84B/256elem: 2-bit + scales/mins | `d * sc * q2 - dmin * m` |
| `dequant_q3k_kernel` | 110B/256elem: 3-bit + scales | `d * sc * q3` |
| `dequant_q4k_kernel` | 144B/256elem: 4-bit + scales/mins | `d * sc * q4 - dmin * m` |
| `dequant_q5k_kernel` | 176B/256elem: 5-bit + scales/mins | `d * sc * q5 - dmin * m` |

### 7.6 MoE Prefill: FP8 Batch Path (line ~3395)

Three-step conversion pipeline for FP8 grouped GEMM:

```
1. Expert weights: GGUF quant ──dequant_gpu_fp8()──▸ FP8 E4M3
2. Activations:    FP16 ──calibrate_fp8_scales_per_expert()──▸ per-expert scales
                   FP16 ──quantize_fp16_to_fp8_e4m3_per_expert()──▸ FP8 E4M3
3. GEMM:           FP8 × FP8 ──cublasGemmGroupedBatchedEx──▸ FP16 output
```

### 7.7 FP32 Accumulation Path (Gemma-3)

Post-norm models maintain a parallel FP32 residual stream alongside FP16 hidden state.

| Kernel | Conversion | Where |
|--------|-----------|-------|
| `fp16_to_fp32_kernel` | FP16 embedding → FP32 residual | After embedding (line ~4258) |
| `rmsnorm_fp32_accum_to_fp16_kernel` | FP16 norm output + FP32 accum → FP32 add → FP16 | Post-attn norm, post-FFN norm |
| `fp32_to_fp16_rowscale_kernel` | FP32 accum → FP16 hidden | Before LM head (line ~4316) |

### 7.8 Router + Sampling Precision

| Conversion | Where |
|-----------|-------|
| `fp16_to_fp32_kernel` | Router gate logits (prefill): FP16 GEMM output → FP32 for softmax/sigmoid | line ~2921 |
| `gemv_*_fp32` | LM head dp4a GEMV: INT8 dp4a → FP32 logits directly | line ~4347 |
| cuBLAS FP16 → `fp16_to_fp32_kernel` | LM head fallback: FP16 GEMM → FP32 for sampling | line ~4382 |

### 7.9 Conversion Pipeline Summary (Decode)

Complete dtype flow for a single decode token through a dense transformer layer:

```
FP16 hidden
  │
  ├─▸ rmsnorm_quantize_q8_1 ──▸ Q8_1 ──▸ dp4a QKV fused ──▸ FP16 Q, K, V
  │                                                              │
  │                                            K,V: write_kv_cache ──▸ FP16/FP8/INT8 cache
  │                                            Q: paged_attention ◂── FP16/FP8/INT8 cache
  │                                                              │
  │                                                         FP16 attn_out
  │                                                              │
  ├─▸ quantize_fp16_to_q8_1(attn_out) ──▸ Q8_1 ──▸ dp4a Wo residual ──▸ FP16 hidden (updated)
  │
  ├─▸ rmsnorm_quantize_q8_1 ──▸ Q8_1 ──▸ dp4a gate+up fused ──▸ FP16 gate, FP16 up
  │                                                                    │
  │                                              swiglu_quantize_q8_1 (fused act+quant)
  │                                                                    │
  │                                                                  Q8_1
  │                                                                    │
  └───────────────────── dp4a down residual ◂──────────────────────────┘
                              │
                         FP16 hidden (updated, ready for next layer)
```

MoE decode replaces the FFN portion:

```
FP16 hidden
  │
  ├─▸ rmsnorm_quantize_q8_1 ──▸ Q8_1 (shared across experts)
  │                                │
  │                    dp4a moe_gate_up_fused ──▸ FP16 gate[top_k], FP16 up[top_k]
  │                                                        │
  │                              swiglu_quantize_q8_1 (or relu_sqr_quantize_q8_1)
  │                                                        │
  │                                                    Q8_1 per expert
  │                                                        │
  │                                     dp4a moe_decode (down) ──▸ FP16 down[top_k]
  │                                                                     │
  └────── moe_weighted_sum_residual(down, weights, residual) ──▸ FP16 hidden (updated)
```
