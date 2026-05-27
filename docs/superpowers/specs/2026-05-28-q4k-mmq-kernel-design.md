# Q4_K_M Custom MMQ Kernel Design Spec

**Goal:** Close the -38% Q4_K_M prefill gap vs llama.cpp with a custom tiled GEMM kernel that works directly on Q4_K packed weights without dequant-to-FP16.

**Status:** Design spec. Implementation estimated at 2-3 weeks.

---

## Problem

Q4_K_M prefill flows through dequant→cuBLAS: each weight block is dequantized to FP16 (8.3x bandwidth overhead), then cuBLAS runs FP16×FP16 GEMM. llama.cpp uses an in-register MMQ kernel that decodes nibbles in shared memory and feeds them directly to tensor cores (dp4a on Ampere, IMMA on Hopper/Blackwell).

Prior attempts:
- **dp4a MMQ** (PR #189): +13-56% at M=2..16, loses to cuBLAS at M≥32. dp4a peak ~50 TOPS vs FP16-TC ~838 TFLOPS = 16x ceiling gap.
- **FP16 HMMA v2** (PRs #193 shipped, then retired): microbench 4.87x dp4a at M=512, but e2e -4% on Qwen3.6 MoE (MIN_M gate, fp16_cache hits, dispatch overhead).
- **INT8 IMMA** (PRs #254-#269, deferred): raw MMA 931 TOPS, but tile kernel plateaued at 40 TOPS (4.3% peak). 3.8x slower e2e than dequant→cuBLAS.

## Approach: In-SMEM Q4_K Decode + FP16 HMMA

Follow llama.cpp's proven architecture but target sm_120's FP16 HMMA (`mma.sync.m16n8k16`):

1. **Load Q4_K blocks into shared memory** (packed, no dequant)
2. **In-SMEM nibble decode**: unpack 4-bit nibbles → unsigned 0-15 integers
3. **Scale + zero-point apply**: per-block min/d correction → FP16 values in registers
4. **Feed to mma.sync.m16n8k16**: FP16 × FP16 → FP32 accumulator
5. **Write FP32 output** (or FP16 downcast)

Key differences from the retired v2:
- No pre-materialized FP16 weight buffer (decode in-SMEM, not in a separate pass)
- No fp16_cache dependency (operates on raw Q4_K packed data)
- Single kernel launch (no dispatch overhead)

## Tile Design

```
TILE_M = 64 (activation rows)
TILE_N = 64 (weight cols — one column-group of the weight matrix)
TILE_K = 64 (reduction dimension, 2× Q4_K block size)

Threads: 128 (4 warps × 32 lanes)
SMEM: Q4_K weight tile (64×64 packed = 2048 bytes)
      + activations (64×64 × 2B = 8 KB)
      + scales/mins (64 × 2 floats = 512 bytes)
      = ~10.5 KB per tile (fits easily in 100 KB)

MMA: 4 warps × m16n8k16 = covers 64×64 tile in 4 MMA calls
```

## Critical Implementation Details

### Q4_K Block Layout
```
struct block_q4_K {
    half d;           // super-block scale
    half dmin;        // super-block min
    uint8_t scales[12]; // 6-bit scales + 6-bit mins packed for 8 sub-blocks
    uint8_t qs[128];    // 256 4-bit values packed as 128 bytes (2 per byte)
};
// 144 bytes per 256 elements = 4.5 bits/element
```

### Nibble Decode in SMEM
```cuda
// Load packed Q4_K block into SMEM
__shared__ uint8_t qs_smem[128];  // 256 nibbles packed

// Unpack per-thread: each thread handles 2 nibbles
uint8_t packed = qs_smem[lane_id];
uint8_t lo = packed & 0xF;        // lower nibble
uint8_t hi = (packed >> 4) & 0xF; // upper nibble

// Apply scale + min: val = d * scale * nibble - dmin * min
half w0 = __float2half(d * sc * lo - dmin * m);
half w1 = __float2half(d * sc * hi - dmin * m);
```

### Stream-K Scheduling
For large M (prefill), partition the K dimension across CTAs to improve tail utilization. Each CTA computes a partial sum and atomicAdd to the output.

## Dispatch Integration

In `src/exec/gemm_kernel_generic_dequant.cu` (the fallback path), add a check:
```cpp
if (qtype == QType::Q4_K && M >= 32 && config.gemm.q4k_mmq_enabled) {
    return mmq_q4k_fp16hmma_gemm(packed, activations, output, M, N, K, stream);
}
```

For M < 32, the existing dp4a GEMV path (PR #436) is already faster than cuBLAS.

## Expected Performance

- **Target:** within 20% of llama.cpp at pp512 (currently -38%)
- **Theoretical ceiling:** 838 TFLOPS FP16-TC. At Q4_K 4.5 bits/elem, the kernel is compute-bound above M=64 (arithmetic intensity > 100 FLOPs/byte)
- **Realistic:** 200-400 TFLOPS effective (shared memory decode overhead, scale apply, tile boundaries)

## Files

| File | Action |
|---|---|
| `src/compute/mmq_q4k_hmma.cu` | Create — the kernel |
| `src/compute/mmq_q4k_hmma.h` | Create — public interface |
| `src/exec/gemm_kernel_q4k_mmq.cu` | Modify — dispatch handler |
| `tests/test_mmq_q4k.cu` | Create — correctness vs dequant reference |
| `CMakeLists.txt` | Modify — register new sources |
