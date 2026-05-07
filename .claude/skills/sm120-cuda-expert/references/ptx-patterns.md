# sm_120a PTX Inline Assembly Patterns

Heavy reference for the `sm120-cuda-expert` skill. Templates verified on RTX 5090 / GB202 under CUDA 13.2.1, `compute_120a / sm_120a`.

Guard all sm_120 code: `#if __CUDA_ARCH__ >= 1200`.

---

## NVFP4 block-scaled MMA — `kind::mxf4nvf4` (peak path)

The peak NVFP4 path on consumer Blackwell. `mma.sync.aligned.kind::mxf4nvf4.block_scale` with FP32 accumulator in registers (FA2-style + block-scaling). Hardware applies the per-16-element UE4M3 scale **inside** the MMA — no manual scale-apply. K=64 per MMA (vs k=32 for f8f6f4) → half the MMA count for the same tile. Raw MMA speedup measured **2.60×** on RTX 5090 (`mxf4nvf4_mma_bench`); end-to-end attention gain expected 1.5–2.5× post softmax + P·V.

```cuda
// scale_vec::4X.m16n8k64, e2m1 inputs, e2m1 inputs, f32 accumulator, ue4m3 scales
asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3},"          // D fragments (FP32)
    "{%4,%5,%6,%7},"          // A fragments (NVFP4 packed)
    "{%8,%9},"                 // B fragments (NVFP4 packed)
    "{%10,%11,%12,%13},"      // C fragments (FP32 accumulator-in)
    "{%14},"                   // sfa_in (UE4M3 scale, packed)
    "{%15,%16},"              // bidA, tidA (block/thread id within scale group)
    "{%17},"                   // sfb_in
    "{%18,%19};\n"            // bidB, tidB
    : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1),
      "f"(c0),"f"(c1),"f"(c2),"f"(c3),
      "r"(sfa_in), "h"(bidA), "h"(tidA),
      "r"(sfb_in), "h"(bidB), "h"(tidB));
```

**Requires `compute_120a`** (NOT `compute_120` / `compute_120f` — `block_scale` modifier and TMA-WS-grouped-GEMM are gated on the `a` arch suffix).

References in repo:
- `src/compute/attention_mxf4nvf4_probe.cu` — canned-input probe
- `src/compute/attention_fmha_mxf4nvf4_sm120.h` — FMHA upgrade target

---

## FP8 MMA — `kind::f8f6f4` (legacy / fallback)

Used when block-scaled NVFP4 isn't applicable (FP8 weights, FP8 KV).

```cuda
asm volatile(
    "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
      "f"(c0),"f"(c1),"f"(c2),"f"(c3));
```

---

## FP16 → FP8 packed conversion

```cuda
asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=r"(fp8x2) : "r"(fp16x2));
```

---

## FP4 (E2M1) ↔ FP16 / FP32 / BF16 packed conversion

The naive `cvt.rn.satfinite.e2m1x2.{f32,f16x2,bf16x2}` instruction looks unsupported on sm_120 if you target a `.b32` register, but it works when you route the FP4 byte through a `.b8` register. SASS confirms hardware emission: `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C`.

```cuda
// f16x2 -> e2m1x2 packed (1 byte holds 2 FP4 values)
asm volatile(
  "{ .reg .b8 b;"
  "  cvt.rn.satfinite.e2m1x2.f16x2 b, %1;"
  "  cvt.u32.u8 %0, b; }"
  : "=r"(fp4x2_u32) : "r"(fp16x2));
```

Verified 2026-05-04 under CUDA 13.2.1 on both `sm_120f` and `sm_120a`. See `references/known-issues.md` "Resolved" section for history.

---

## NVFP4 prmt register LUT (replaces SMEM dequant table)

The `prmt.b32` instruction permutes 4 bytes from two source registers into a result, indexed by 4 nibbles in a selector register. Two uint32 constants encode the 8 FP4-decoded FP16 values; one `prmt` decodes 4 NVFP4 values into 4 packed FP16. Zero SMEM, zero L2, register-only.

```cuda
constexpr uint32_t kLutLo = 0x3E3C3800u;  // FP16: [0.0, 0.5, 1.0, 1.5]
constexpr uint32_t kLutHi = 0x46444240u;  // FP16: [2.0, 3.0, 4.0, 6.0]
asm volatile("prmt.b32 %0, %1, %2, %3;\n"
             : "=r"(out) : "r"(kLutLo), "r"(kLutHi), "r"(selector));
```

---

## `cp.async` for paged KV (16-byte vector load)

```cuda
asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(glob));
asm volatile("cp.async.commit_group;\n");
asm volatile("cp.async.wait_group 0;\n");
__syncthreads();  // REQUIRED before reading smem — race on SMEM otherwise
```

---

## Warp-XOR shuffle reduce

Single-warp sum/max reduction in 5 instructions (32-lane warp).

```cuda
#pragma unroll
for (int o = 16; o >= 1; o >>= 1)
    val += __shfl_xor_sync(0xFFFFFFFF, val, o);
```

---

## KV cache streaming load (`__ldcs`, bypass L1)

KV reads are one-shot per generated token — caching them in L1 evicts useful weights. `__ldcs` issues a streaming load that bypasses L1 and uses evict-first L2.

```cuda
const float4 kv = __ldcs(reinterpret_cast<const float4*>(kv_ptr));
```

**KV only.** Do NOT use on weights — weights are reused across batches and benefit from L1 caching.
