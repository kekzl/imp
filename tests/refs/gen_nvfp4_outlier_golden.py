#!/usr/bin/env python3
"""Generate fp64 spot-check goldens for tests/test_nvfp4_outlier_ref.cu.

TEST_AUDIT (retired) risk #2: the NVFP4 quantize pipeline (the product's core decode
path) had only round-trip-via-own-code tests on benign Gaussian data. The
Gemma mode-2 collapse (#514/#516) — a single outlier weight inflates the
per-tensor scale until normal micro-blocks underflow the 1/512 FP8-E4M3 floor
and dequant to zero, ultimately NaN logits at decode step 2 — was invisible to
those tests because nothing fed an adversarial (outlier / all-tiny) weight
distribution through quantize -> dequant against an INDEPENDENT reference.

This generator is that independent reference. It reimplements NVFP4's
two-level dequant in fp64 FROM THE FORMAT DEFINITION (not by calling imp):

  Level 1 (tensor scale):  tensor_scale = global_absmax / 6     (FP32)
  Level 2 (micro scale):   ms = local_absmax / (tensor_scale * 6)  per 16 vals
                           clamp ms to [1/512, 448], round-trip through FP8 E4M3
  Quantized value:         code = RNE_E2M1( v / (tensor_scale * ms_actual) )
  Dequantized value:       code_mag * tensor_scale * ms_actual    (signed)

E2M1 magnitudes (NVFP4 standard): {0, .5, 1, 1.5, 2, 3, 4, 6}, RNE between
adjacent representable values (the same midpoint thresholds imp's
float_abs_to_fp4_e2m1 uses). FP8 E4M3: 4-exp/3-man, bias 7, min subnormal
2^-9 = 1/512, max normal 448 — the imp/NVIDIA fp8_e4m3 definition.

The C++ test generates IDENTICAL inputs with the same integer-LCG + f32
multiply-only recipe (bit-exact across numpy/C++, no libm), runs imp's real
quantize_fp16_to_nvfp4 + gemv_nvfp4_kpar on the GPU, and asserts 48 spot
values per distribution against this golden at the NVFP4 1e-1 rel class
tolerance (tests/refs/README.md), PLUS a hard no-NaN/Inf guard on every
distribution (the actual Gemma-class assert). The all-tiny / extreme-outlier
distributions are the floor-collapse triggers.

Regenerate (host stays clean — run in a container with numpy):
  docker run --rm -v $PWD:/work -w /work python:3.12-slim \
    sh -c "pip install -q numpy && python3 tests/refs/gen_nvfp4_outlier_golden.py"
"""

import numpy as np

OUT = "tests/refs/nvfp4_outlier_golden.h"
N_SPOTS = 48
MICRO = 16

# (name, N, K, amp, outlier_mult, outlier_period, tiny)
#   amp           base cubed-uniform amplitude (per element, f32)
#   outlier_mult  multiplier applied to 1/outlier_period of elements
#   tiny          if set, scales EVERY element down to the underflow regime
#
# (a) gaussian-ish: N(0,1)-class heavy-tailed, no injected outliers.
# (b) gemma_outlier_64x: same + 1/256 weights at 64x RMS (Gemma activation
#     class — single weights ~64x the typical magnitude).
# (c) extreme_outlier_512x: one weight per tensor at 512x — the explicit
#     floor-collapse trigger: tensor_scale balloons, normal blocks underflow
#     micro_scale below 1/512 and clamp.
# (d) all_tiny: every |w| < 1e-4 — the underflow side; with no large element
#     the tensor_scale itself is tiny and the whole tensor risks collapse.
CONFIGS = [
    ("gaussian",             64, 256, 1.0,   1.0,    1, False),
    ("gemma_outlier_64x",    64, 256, 1.0,  64.0,  256, False),
    ("extreme_outlier_512x", 64, 256, 1.0, 512.0, 4096, False),  # ~1 per tensor (64*256/4096=4)
    ("all_tiny",             64, 256, 1.0,   1.0,    1, True),
]

E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
# RNE midpoints between adjacent magnitudes (imp float_abs_to_fp4_e2m1 thresholds).
E2M1_THR = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float64)
FP4_MAX = 6.0
FP8_MIN_SUB = 1.0 / 512.0   # 2^-9
FP8_MAX = 448.0


def lcg_fill(seed: int, n: int, amp: float, outlier_mult: float, outlier_period: int,
             tiny: bool) -> np.ndarray:
    """Bit-exact mirror of the C++ generator (f32 ops only, stored as f16).

    Recipe (cubed-uniform heavy tails, like the attention crosspath golden):
      x_{n+1} = x_n*1664525 + 1013904223           uint32 LCG
      v   = ((x>>8) & 0x3FFF) - 8192               in [-8192, 8191]
      f   = v / 8192.0f                            f32 in [-1,1]
      val = f*f*f * amp                            cubed -> heavy tails
      if outlier_period>1 and (x % period)==0: val *= outlier_mult
      if tiny: val *= (1/16384)f                   push everything sub-1e-4
    """
    x = np.uint32(seed)
    out = np.empty(n, dtype=np.float16)
    inv = np.float32(1.0 / 8192.0)
    amp32 = np.float32(amp)
    om = np.float32(outlier_mult)
    tinyf = np.float32(1.0 / 16384.0)
    period = np.uint32(outlier_period)
    for i in range(n):
        x = np.uint32((np.uint64(x) * 1664525 + 1013904223) & 0xFFFFFFFF)
        v = np.int32((x >> np.uint32(8)) & np.uint32(0x3FFF)) - np.int32(8192)
        f = np.float32(v) * inv
        val = f * f * f * amp32
        if outlier_period > 1 and (x % period) == np.uint32(0):
            val = val * om
        if tiny:
            val = val * tinyf
        out[i] = np.float16(val)
    return out


def fp8_e4m3_round(x: float) -> float:
    """Round a non-negative f64 to the nearest FP8 E4M3 value (RNE), return f64.

    Mirrors imp float_to_fp8_e4m3 / fp8_e4m3_to_float: bias 7, 3 mantissa
    bits, min subnormal 2^-9, max normal 448, saturate (no Inf). Below the
    min subnormal -> 0.
    """
    if x <= 0.0:
        return 0.0
    if x >= FP8_MAX:
        return FP8_MAX
    if x < FP8_MIN_SUB:
        # imp returns 0 for |val| < 1/512 (sign-only byte).
        return 0.0
    # Decompose into the E4M3 grid. e_field in [0,15], 3-bit mantissa.
    # Normal: value = (1 + m/8) * 2^(e-7), e in [1,15] (e=15,m in 0..6).
    # Subnormal: value = (m/8) * 2^-6, e=0, m in 1..7.
    import math
    e = math.floor(math.log2(x))
    e = max(e, -6)  # smallest normal exponent is 2^-6 (e_field=1)
    # candidate spacing at this binade
    if e >= -6:
        # try normal binade [2^e, 2^(e+1)); mantissa step = 2^e / 8
        step = (2.0 ** e) / 8.0
        # but if x < 2^-6 we are in the subnormal binade with step 2^-9
        if x < 2.0 ** -6:
            step = 2.0 ** -9
            base = 0.0
        else:
            base = 0.0
    q = round(x / step)  # RNE (python round is banker's rounding -> RNE)
    val = q * step
    if val > FP8_MAX:
        val = FP8_MAX
    return float(val)


def e2m1_rne(mag: float) -> float:
    """Round a non-negative f64 magnitude to nearest E2M1 representable (RNE).

    Uses the same midpoint thresholds as imp's branchless code. Saturates to 6.
    """
    code = int(np.sum(mag >= E2M1_THR))  # 0..7
    return float(E2M1_MAG[code])


def dequant_ref(w_f16: np.ndarray) -> np.ndarray:
    """Independent fp64 NVFP4 quantize->dequant of a [N,K] f16 weight tensor."""
    N, K = w_f16.shape
    w = w_f16.astype(np.float64)
    global_absmax = float(np.max(np.abs(w)))
    tensor_scale = global_absmax / FP4_MAX
    if tensor_scale == 0.0:
        tensor_scale = 1.0  # imp: all-zero tensor -> scale 1
    out = np.zeros_like(w)
    nmb = K // MICRO
    for r in range(N):
        for g in range(nmb):
            block = w[r, g * MICRO:(g + 1) * MICRO]
            local_absmax = float(np.max(np.abs(block)))
            ms = local_absmax / (tensor_scale * FP4_MAX)
            if ms < FP8_MIN_SUB:
                ms = FP8_MIN_SUB          # imp floor clamp
            if ms > FP8_MAX:
                ms = FP8_MAX
            ms_actual = fp8_e4m3_round(ms)
            if ms_actual == 0.0:
                ms_actual = FP8_MIN_SUB   # imp guards 0 -> 1/512
            combined = tensor_scale * ms_actual
            inv = 1.0 / combined
            for j in range(MICRO):
                v = block[j]
                code_mag = e2m1_rne(abs(v) * inv)
                dq = code_mag * combined
                out[r, g * MICRO + j] = -dq if v < 0.0 else dq
    return out


def spot_indices(total: int) -> list:
    return [int((np.uint64(k) * np.uint64(2654435761)) % np.uint64(total)) for k in range(N_SPOTS)]


def gemv_ref(w_dq: np.ndarray, x_f16: np.ndarray) -> np.ndarray:
    """fp64 y[M] = sum_k dequant(W)[m,k] * x[k], from f16-rounded x."""
    x = x_f16.astype(np.float64)
    return w_dq @ x


def main():
    lines = [
        "// AUTO-GENERATED by tests/refs/gen_nvfp4_outlier_golden.py — do not edit.",
        f"// numpy {np.__version__}; fp64 NVFP4 quantize->dequant + GEMV references.",
        "// Independent reimplementation of NVFP4 (E2M1 + UE4M3 microscale + f32",
        "// tensorscale) from the format definition — NOT produced by imp code.",
        "// Regenerating must reproduce this file (see tests/refs/README.md).",
        "#pragma once",
        "",
        "namespace imp_refs {",
        "",
    ]
    for idx, (name, N, K, amp, om, period, tiny) in enumerate(CONFIGS):
        w = lcg_fill(0x2468 + idx * 7, N * K, amp, om, period, tiny).reshape(N, K)
        xv = lcg_fill(0xACE0 + idx * 7, K, 1.0, 1.0, 1, False)  # GEMV input, benign
        w_dq = dequant_ref(w)
        y = gemv_ref(w_dq, xv)

        dq_flat = w_dq.reshape(-1)
        dq_idx = spot_indices(dq_flat.size)
        y_idx = spot_indices(y.size)

        gmax = float(np.max(np.abs(w)))
        ts = gmax / FP4_MAX if gmax > 0 else 1.0
        lines.append(f"// config {name}: N={N} K={K} amp={amp} outlier={om}x/1per{period} tiny={int(tiny)}")
        lines.append(f"//   global_absmax={gmax:.6g} tensor_scale={ts:.6g} "
                     f"(normal-block micro_scale ~ block_absmax/global_absmax; "
                     f"floor=1/512={1/512:.6g})")
        lines.append(f"inline constexpr int {name}_dq_spot_idx[{N_SPOTS}] = {{")
        lines.append("    " + ", ".join(str(i) for i in dq_idx) + "};")
        lines.append(f"inline constexpr double {name}_dq_spot_val[{N_SPOTS}] = {{")
        lines.append("    " + ", ".join(f"{dq_flat[i]:.17g}" for i in dq_idx) + "};")
        lines.append(f"inline constexpr int {name}_gemv_spot_idx[{N_SPOTS}] = {{")
        lines.append("    " + ", ".join(str(i) for i in y_idx) + "};")
        lines.append(f"inline constexpr double {name}_gemv_spot_val[{N_SPOTS}] = {{")
        lines.append("    " + ", ".join(f"{y[i]:.17g}" for i in y_idx) + "};")
        lines.append("")
    lines.append("}  // namespace imp_refs")
    lines.append("")
    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
