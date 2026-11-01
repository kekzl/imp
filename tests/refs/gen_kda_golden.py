#!/usr/bin/env python3
"""Generate KDA (Kimi Delta Attention) goldens for tests/test_kda_reference.cpp.

Independent reference: fla-org/flash-linear-attention's torch code, NOT imp.
Copied verbatim (MIT) from commit 14bef3d9ec98ee359c9a9c7860795d91675dd091:
  fla/ops/kda/naive.py  naive_recurrent_kda, naive_chunk_kda
  fla/ops/kda/gate.py   naive_kda_lowerbound_gate
q/k L2 norm follows fla's fused kernels (use_qk_l2norm_in_kernel): x / sqrt(sum(x^2) + 1e-6).

Pipeline per Ling-3.0 (BailingMoeV3KimiDeltaAttention, kda_safe_gate, kda_lower_bound=-5):
  q, k = l2norm(q), l2norm(k)
  g    = lower_bound * sigmoid(exp(A_log[h]) * (f + dt_bias[h, :]))
  beta = sigmoid(b)
  o, S = naive_recurrent_kda(q, k, v, g, beta, scale=K^-0.5, initial_state=S0)
naive_chunk_kda (chunk 16) must agree with the recurrent form; the generator asserts it.

Inputs come from a 31-bit LCG so the C++ test regenerates them bit-exactly
(tests/kda_reference.h kda_lcg_fill). Output: tests/refs/kda_golden.h.

Regenerate (host stays clean, CPU container):
  docker run --rm -v $PWD:/work -w /work python:3.12-slim sh -c \
    "pip install -q torch==2.5.1 einops==0.8.0 --index-url https://download.pytorch.org/whl/cpu \
     --extra-index-url https://pypi.org/simple && python3 tests/refs/gen_kda_golden.py"
"""

import torch
import torch.nn.functional as F
from einops import rearrange

OUT = "tests/refs/kda_golden.h"

T, H, K, V = 32, 2, 16, 16
LOWER_BOUND = -5.0
SEED = 20261101


class Lcg:
    """state = (state * 1103515245 + 12345) mod 2^31; value in [-1, 1) as float32."""

    def __init__(self, seed):
        self.s = seed % (1 << 31)

    def next(self):
        self.s = (self.s * 1103515245 + 12345) % (1 << 31)
        return float(self.s) / float(1 << 30) - 1.0

    def fill(self, n, amp):
        return torch.tensor([self.next() * amp for _ in range(n)], dtype=torch.float32)


# ---- fla/ops/kda/naive.py (verbatim) ----
def naive_recurrent_kda(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False):
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    if scale is None:
        scale = K ** -0.5

    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    q = q.repeat_interleave(G, dim=2) * scale   # [B, T, HV, K]
    k = k.repeat_interleave(G, dim=2)           # [B, T, HV, K]

    S = k.new_zeros(B, HV, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)
    if not output_final_state:
        S = None
    return o.to(dtype), S


def naive_chunk_kda(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False, chunk_size=64):
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    BT = chunk_size
    NT = T // BT
    if scale is None:
        scale = K ** -0.5
    assert T % BT == 0

    q, k = [rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).to(torch.float) for x in [q, k]]
    v, g, beta = [rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).to(torch.float) for x in [v, g, beta]]
    q = q.repeat_interleave(G, dim=1) * scale  # [B, HV, NT, BT, K]
    k = k.repeat_interleave(G, dim=1)          # [B, HV, NT, BT, K]
    g = g.cumsum(-2)

    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)

    A = torch.zeros(*g.shape[:-1], BT, dtype=torch.float, device=q.device)
    for i in range(BT):
        k_i = k[..., i, :]
        g_i = g[..., i:i+1, :]
        A[..., i] = torch.einsum('... c d, ... d -> ... c', k * (g - g_i).exp(), k_i)
    A = A * beta[..., None]

    A = -A.masked_fill(mask, 0)
    for i in range(1, BT):
        A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :, None].clone() * A[..., :, :i].clone()).sum(-2)
    A = (A + torch.eye(BT, dtype=torch.float, device=q.device)) * beta[..., None, :]

    w = A @ (g.exp() * k)
    u = A @ v

    S = k.new_zeros(B, HV, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, NT):
        q_i = q[:, :, i]
        k_i = k[:, :, i]
        u_i = u[:, :, i]
        g_i = g[:, :, i]
        w_i = w[:, :, i]
        Aqk = torch.zeros(B, HV, BT, BT, dtype=torch.float, device=q.device)
        for j in range(BT):
            k_j = k[:, :, i, j]
            g_j = g[:, :, i, j:j+1, :]
            Aqk[..., j] = torch.einsum('... c d, ... d -> ... c', q_i * (g_i - g_j).exp(), k_j)
        Aqk = Aqk.masked_fill(mask, 0)
        v_i = u_i - w_i @ S
        o[:, :, i] = (q_i * g_i.exp()) @ S + Aqk @ v_i
        S = S * rearrange(g_i[:, :, -1].exp(), 'b h k -> b h k 1')
        S += rearrange((g_i[:, :, -1:] - g_i).exp() * k_i, 'b h c k -> b h k c') @ v_i
    if not output_final_state:
        S = None
    return rearrange(o, 'b h n c d -> b (n c) h d').to(dtype), S


# ---- fla/ops/kda/gate.py (verbatim) ----
def naive_kda_lowerbound_gate(g, A_log=None, dt_bias=None, lower_bound=-5.0, output_dtype=torch.float32):
    H, _ = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    if A_log is not None:
        g = A_log.view(H, 1).float().exp() * g
    g = lower_bound * F.sigmoid(g)
    return g.to(output_dtype)


def l2norm_kernel(x):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def fmt(name, t):
    vals = t.reshape(-1).tolist()
    body = ",\n".join("    " + ", ".join(f"{v:.9g}f" for v in vals[i:i + 8]) for i in range(0, len(vals), 8))
    return f"inline constexpr float {name}[{len(vals)}] = {{\n{body}\n}};\n"


def main():
    torch.manual_seed(0)
    rng = Lcg(SEED)
    # Fill order is the contract with kda_reference.h: q, k, v, f, b, A_log, dt_bias, S0.
    q = rng.fill(T * H * K, 1.0).view(1, T, H, K)
    k = rng.fill(T * H * K, 1.0).view(1, T, H, K)
    v = rng.fill(T * H * V, 1.0).view(1, T, H, V)
    f = rng.fill(T * H * K, 3.0).view(1, T, H, K)
    b = rng.fill(T * H, 3.0).view(1, T, H)
    A_log = rng.fill(H, 1.5)
    dt_bias = rng.fill(H * K, 1.0)
    S0 = rng.fill(H * K * V, 0.5).view(1, H, K, V)

    qn, kn = l2norm_kernel(q), l2norm_kernel(k)
    g = naive_kda_lowerbound_gate(f, A_log, dt_bias, LOWER_BOUND)
    beta = torch.sigmoid(b)
    o, S = naive_recurrent_kda(qn, kn, v, g, beta, initial_state=S0.clone(), output_final_state=True)
    o_c, S_c = naive_chunk_kda(qn, kn, v, g, beta, initial_state=S0.clone(), output_final_state=True,
                               chunk_size=16)
    assert torch.allclose(o, o_c, atol=1e-4, rtol=1e-4), (o - o_c).abs().max()
    assert torch.allclose(S, S_c, atol=1e-4, rtol=1e-4), (S - S_c).abs().max()

    with open(OUT, "w") as fh:
        fh.write("// GENERATED by tests/refs/gen_kda_golden.py - do not edit.\n")
        fh.write(f"// torch {torch.__version__}, fla naive KDA @14bef3d9, LCG seed {SEED}.\n")
        fh.write("#pragma once\n\n// clang-format off\nnamespace imp::kda_golden {\n\n")
        fh.write(f"inline constexpr int kT = {T}, kH = {H}, kK = {K}, kV = {V};\n")
        fh.write(f"inline constexpr unsigned kSeed = {SEED}u;\n")
        fh.write(f"inline constexpr float kLowerBound = {LOWER_BOUND}f;\n\n")
        fh.write("// o [T, H, V] from naive_recurrent_kda.\n")
        fh.write(fmt("kOut", o))
        fh.write("\n// Final state [H, K, V].\n")
        fh.write(fmt("kState", S))
        fh.write("\n}  // namespace imp::kda_golden\n// clang-format on\n")
    print(f"wrote {OUT}: max|o-o_chunk|={(o - o_c).abs().max().item():.3g}")


if __name__ == "__main__":
    main()
