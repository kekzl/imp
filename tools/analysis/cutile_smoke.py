"""Smoke test: does a trivial cuTile kernel execute on sm_120a (WSL2)?
Uses cupy for device arrays + stream. Validates the JIT/driver path before
investing in an FA2 kernel + autotuner."""
import cupy as cp
import cuda.tile as ct


@ct.kernel
def vadd(X, Y, Out, tn: ct.Constant[int]):
    i = ct.bid(0)
    xv = X.tiled_view((tn,), padding_mode=ct.PaddingMode.ZERO)
    yv = Y.tiled_view((tn,), padding_mode=ct.PaddingMode.ZERO)
    tx = xv.load((i,))
    ty = yv.load((i,))
    ct.store(Out, (i,), ct.add(tx, ty))


N = 4096
TN = 256
x = cp.arange(N, dtype=cp.float32)
y = cp.ones(N, dtype=cp.float32) * 3.0
out = cp.zeros(N, dtype=cp.float32)
stream = cp.cuda.get_current_stream()
grid = (ct.cdiv(N, TN),)
ct.launch(stream, grid, vadd, (x, y, out, TN))
stream.synchronize()
err = float(cp.max(cp.abs(out - (x + y))))
print(f"cuTile vadd on sm_120a: max_abs_err={err}  out[:4]={cp.asnumpy(out[:4])}")
print("EXEC OK" if err == 0.0 else "EXEC MISMATCH")
