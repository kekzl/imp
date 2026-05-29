"""Python cuTile FA2 (causal, fp16, D=128) + autotune ceiling vs native.

Measures the autotuned cuTile attention perf on sm_120a — the decisive
productionize-vs-shelve experiment for the Tile backend (GOAL/DISPATCH).
Mirrors tools/analysis/tile_fa2_bench.cu (native naive = 24 eff-TFLOPS).
"""
import sys
from itertools import product
import numpy as np
import cupy as cp
import cuda.tile as ct

S, D = 2048, 128
HEADS, REPS = 32, 50
SCALE = 1.0 / (D ** 0.5)
NEG = -1e30


@ct.kernel
def fa2(Q, K, V, O, scale, TM: ct.Constant[int], TN: ct.Constant[int]):
    qb = ct.bid(0)
    qv = Q.tiled_view((TM, D), padding_mode=ct.PaddingMode.ZERO)
    kv = K.tiled_view((TN, D), padding_mode=ct.PaddingMode.ZERO)
    vv = V.tiled_view((TN, D), padding_mode=ct.PaddingMode.ZERO)

    q = qv.load((qb, 0))                       # [TM, D]
    acc = ct.zeros((TM, D), ct.float32)
    m = ct.full((TM, 1), NEG, ct.float32)
    l = ct.zeros((TM, 1), ct.float32)

    row = ct.expand_dims(qb * TM + ct.arange(TM, dtype=ct.int32), 1)   # [TM,1]
    for j in range(kv.num_tiles(0)):
        k = kv.load((j, 0))                    # [TN, D]
        kt = ct.transpose(k)                   # [D, TN]
        qk = ct.mma(q, kt, ct.zeros((TM, TN), ct.float32))   # [TM, TN]
        qk = qk * scale
        col = ct.expand_dims(j * TN + ct.arange(TN, dtype=ct.int32), 0)   # [1,TN]
        masked = ct.greater(ct.broadcast_to(col, (TM, TN)),
                            ct.broadcast_to(row, (TM, TN)))
        qk = ct.where(masked, ct.full((TM, TN), NEG, ct.float32), qk)

        rmax = ct.max(qk, 1, keepdims=True)    # [TM,1]
        mij = ct.maximum(m, rmax)
        alpha = ct.exp2((m - mij) * 1.442695)
        acc = acc * ct.broadcast_to(alpha, (TM, D))
        p = ct.exp2((qk - ct.broadcast_to(mij, (TM, TN))) * 1.442695)
        l = l * alpha + ct.sum(p, 1, keepdims=True)
        m = mij
        v = vv.load((j, 0))                    # [TN, D]
        acc = ct.mma(ct.astype(p, ct.float16), v, acc)

    out = acc / ct.broadcast_to(l, (TM, D))
    ct.store(O, (qb, 0), ct.astype(out, O.dtype))


def make_inputs():
    rng = cp.random.default_rng(0)
    q = (rng.integers(-6, 7, (S, D)).astype(cp.float32) * 0.02).astype(cp.float16)
    k = (rng.integers(-6, 7, (S, D)).astype(cp.float32) * 0.02).astype(cp.float16)
    v = (rng.integers(-6, 7, (S, D)).astype(cp.float32) * 0.02).astype(cp.float16)
    return q, k, v


def ref_causal(q, k, v):
    # host numpy reference (container has CUDA 13 cublas, cupy-cuda12x can't matmul)
    qf, kf, vf = (cp.asnumpy(x).astype(np.float32) for x in (q, k, v))
    s = (qf @ kf.T) * SCALE
    mask = np.triu(np.ones((S, S), bool), 1)
    s = np.where(mask, NEG, s)
    s = s - s.max(1, keepdims=True)
    p = np.exp(s); p = p / p.sum(1, keepdims=True)
    return (p @ vf)


def main():
    q, k, v = make_inputs()
    o = cp.zeros((S, D), cp.float16)
    stream = cp.cuda.get_current_stream()
    TM = TN = 64
    grid = (ct.cdiv(S, TM),)
    ct.launch(stream, grid, fa2, (q, k, v, o, SCALE, TM, TN))
    stream.synchronize()
    ref = ref_causal(q, k, v)
    oh = cp.asnumpy(o).astype(np.float32)
    rel = float(np.max(np.abs(oh - ref) / np.maximum(1.0, np.abs(ref))))
    print(f"correctness TM=TN=64: max_rel_err={rel:.4f} {'OK' if rel < 0.05 else 'FAIL'}")
    if rel >= 0.05:
        sys.exit(1)

    # ---- autotune tile sizes ----
    space = [dict(TM=tm, TN=tn) for tm, tn in product((32, 64, 128), (32, 64, 128))]
    grid_fn = lambda c: (ct.cdiv(S, c['TM']),)
    args_fn = lambda c: (q, k, v, cp.zeros((S, D), cp.float16), SCALE, c['TM'], c['TN'])
    res = ct.tune.exhaustive_search(space, stream, grid_fn, fa2, args_fn)
    best_us = res.best.mean_us
    flop = 2.0 * S * S * D            # QK + PV (causal ~0.5 absorbed into 2x macs)
    tflops = flop / (best_us * 1e-6) / 1e12
    print(f"BEST config={res.best.config}  {best_us:.1f} us  -> {tflops:.1f} eff-TFLOPS "
          f"({100*tflops/838:.1f}% of 838 FP16 roofline; native naive C++=24 TFLOPS)")


if __name__ == "__main__":
    main()
