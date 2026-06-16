# The Optimal sm_120a Attention Kernel

Canonical design reference for the imp hot-path attention kernel on RTX 5090
(GB202, **sm_120a** — consumer Blackwell). This is the spec future FA2 levers
must measure themselves against. It is grounded in the profiling ground-truth
and the empirical refutations accumulated through 2026-06, not in datacenter
(B200 / FA4) assumptions that do not port.

Companion docs: [`sm120.md`](sm120.md) (kernel notes), [`performance.md`](performance.md)
(baselines + methodology). Hot-path source, both in
`src/compute/attention_fmha_sm120.cu`: `fmha_sm120_fa2_kernel` (the primary
register-resident FA2, dispatched by `fmha_sm120_fa2_prefill`) and
`fmha_sm120_kernel` (the slow tiled-FMHA **fallback** — not an optimization
target). The FA2 kernel is templated on `<Bq, HD, FP16QK, F16ACC, BKV, TWOSLOT,
PVF16, …>`; the dispatcher bands the tile config by grid-fill (`blocks_128` vs
`sm_count`).

---

## 1. Design thesis: what are we optimizing against?

Profiling ground-truth (issue #597, post-#609): FA2 is **tensor-pipe busiest at
52.8 %, occupancy smem-capped at 16.7 %, 0.75 waves → wait-latency-limited**,
with a flat SOL < 37 % across all units. That is an **instruction-mix +
dependency-chain** signature — NOT a bandwidth gap and NOT an occupancy gap. The
optimal kernel therefore attacks exactly four things, in priority order:

1. **f32-accumulate in QK^T** runs at **¼ TC rate** → the single largest compute loss.
2. **Softmax (exp / max / rescale) on the critical path** between the QK store and
   the PV load serializes the tensor pipe.
3. **Synchronous K/V loads** stall the MMA. With smem-capped 1-block/SM, occupancy
   *cannot* hide that latency — software pipelining must.
4. **`O_acc` in shared memory** eats the budget needed for larger tiles / deeper
   async rings.

## 2. Full spec (hd=128, NVFP4 model, long context)

### Tiling & occupancy
- **Bq = 128, Bkv = 64**, 8 warps / 256 threads, **1 block/SM** (smem-capped).
- `__launch_bounds__(256, 1)` — correct for an SMEM-limited kernel; allows maximum
  register allocation (the documented FMHA exception to the no-`__launch_bounds__`
  rule). Do **not** write `,2`: at hd=128 the smem budget never admits 2 blocks/SM,
  so the hint is a lie that only costs register headroom.

### Shared-memory budget (target ≤ 99 KB optin; query `sharedMemPerBlockOptin`)
```
Q_tile      half[128 × 128]      = 32 KB   (loaded once, kernel-resident)
K/V ring    half[2 × 64 × 128]   = 32 KB   (2–3-stage cp.async double-buffer)
S/P overlay float[128 × 64]      = 32 KB   (f32 scores; half-P aliases the bytes)
row_m,row_l float[2 × 128]       ≈  1 KB
                                  ─────────
                                   ~97 KB → fits, exactly 1 block/SM
O_acc       → REGISTERS, not smem (0 KB)
```
**The lever that finances Bq=128:** `O_acc` leaves shared memory and lives as
**MMA accumulator fragments in registers**, held by each warp across the *entire*
KV loop (true FA2 register-resident). The tiled fallback keeps `O_acc` as a
`float[Bq×HD]` = 64 KB smem block — which is precisely why it cannot fit Bq=128 in
99 KB and degrades to Bq=64.

### MMA: dual-precision, both f16-accumulate
- **QK^T: `mma.sync.m16n8k16.f16.f16.f16.f16`** (f16 accumulator). Online-softmax
  subtracts the row max, so f16 dynamic range on the scores is safe. Cost +0.37 %
  PPL (the `attention.fa2_f16acc` knob). This is the **¼-rate → full-rate** jump.
- **PV: also f16-accumulate** (`mma.sync.m16n8k16`, default-on since PR #674; the
  "O sum needs f32 range" objection was refuted).

### Async pipeline (the missing overlap)
- Synchronous `float4` copies → a **3-stage `cp.async.cg.shared.global` 16-byte
  ring**. Producer lanes prefetch K tile *j+1* and V tile *j* while consumers run
  QK/PV on tile *j*. `commit_group` / `wait_group(N-1)` + `__syncthreads()` before
  any smem read.
- **Rationale:** at 1 block/SM (0.75 waves) more occupancy cannot hide GDDR7
  latency — so deep software pipelining (more in-flight async per wave) must. This
  is the only correct response to smem-capped occupancy.

### Take softmax off the tensor-pipe critical chain
- `exp` via **`ex2.approx.f32` on the MUFU/SFU pipe** — runs **parallel to the
  tensor pipe**. While warp A runs PV-MMA for tile *j*, warp B computes the softmax
  for tile *j+1*. **No forced producer/consumer specialization** — the evidence is
  unambiguous: the cross-tile pipeline (both combined and split-K/V-prefetch
  variants) **regressed +9 % / +15 %**; warps deliver phase diversity themselves.
  Enough warps + the scheduler suffices.
- Keep running max/sum in register lanes; the **O rescale (`O *= α`) is a register
  op on the accumulator fragments**, not a smem read-modify-write.

### The NVFP4 precision boundary (the honest line)
QK and PV stay **f16**. The `mxf4nvf4.block_scale` MMA (k=64, 2.6× raw) is
tempting, but it requires Q/K in NVFP4 → the **format-intrinsic quality cliff**
(e4m3-QK PPL 5722 vs 6.12, #511 — 3 mantissa bits × 36-layer compounding). **FP4
MMA is the weapon for the projection GEMMs** (q/k/v/o_proj, FFN), **not** for
QK^T/PV *inside* attention. Attention math wants f16 mantissa; only the linear
layers may go FP4.

## 3. Steady-state pipeline (per KV tile)

```
Tensor pipe:   [QK mma j ][ PV mma j-1      ][QK mma j+1]   ← never idle
SFU pipe:               [exp/max softmax j  ]               ← parallel, hidden
Async copy:    [cp.async K_{j+1}, V_j  ......]              ← hidden behind mma
Barriers:      1× __syncthreads / KV tile (cp.async.wait)
```

## 4. Lever status — the honest punchline

| Lever | Status | Evidence |
|-------|--------|----------|
| f16-acc QK^T + PV | **shipped** | `fa2_f16acc` / #674, +3–4 % pp, +0.37 % PPL |
| Register-resident O | **shipped** | primary FA2 (`mxf4nvf4_sm120.h`) |
| cp.async K/V double-buffer | **shipped** | −11.6 % kernel, long ctx |
| Smem row-stride padding | **shipped** | 1.54× kernel, PR #484 |
| Sawtooth L2 locality | **shipped** | in the fallback source today |
| Deeper async ring / cross-tile pipe | **REFUTED** | both variants **+9 % / +15 % regression** — phase-chain hypothesis false |
| Bq=128 / 2-CTA / occupancy push | **REFUTED** | reg-squeeze succeeded (16.5→30.6 %) but dense **+11 % regression**, SOL stayed flat |
| FP4-QK inside attention | **REFUTED** | #511 PPL 5722, format-intrinsic |

**Punchline:** the optimal sm_120 attention kernel and imp's primary FA2 have
**converged**. Every un-refuted lever is in; every remaining design move (deeper
pipeline, more occupancy, FP4-QK) has been empirically refuted. The 52 %→100 %
roofline gap is **architecture, not implementation debt.**

> No open re-litigation. An earlier draft of this doc speculated that the Bq=128
> and deeper-pipeline refutations predated the register-resident-O envelope and
> were worth re-running. Reading the actual dispatcher (`fmha_sm120_fa2_prefill`)
> refutes that: the register-resident-O + Q-in-registers + **Bq=128** config is
> already the *shipped* large-seq path (selected when `blocks_128 >= sm_count`),
> with f16-acc QK^T and PV-f16-acc as config-gated variants on top. The 2-CTA
> refutation is baked into the design — the Bq=128 path explicitly forgoes 2-CTA
> residency in favor of deeper cp.async overlap at Bkv=64 (see the comment at the
> top of `fmha_sm120_fa2_prefill`), and the underfill band (`sm_count/2 <=
> blocks_128 < sm_count`) deliberately drops to **Bq=64 + TWOSLOT** to put 2
> CTAs/SM resident where the grid would otherwise underfill the 170 SMs. The
> refuted levers were measured against this exact kernel family (#597 / #648 /
> #653 / #674); they are closed, not pending.

## 5. The wall (silicon, not code)

What would take the kernel from ~52 % to ~100 % roofline, we **cannot build**:

- **No `tcgen05` / async MMA** → the MMA always blocks the issuing warp. We can
  *emulate* an FA4-style pipeline with `cp.async` + warp diversity, but never hide
  the MMA itself behind async.
- **No TMEM / TMA-WS** → no producer-warpgroup-TMA-into-mbarrier-ring + consumer-
  warpgroup-MMA-into-TMEM + softmax-warpgroup-on-the-side (FA4 on B200).
- **FP4 mma.sync = ½ datasheet, f32-acc = ¼** → the TC peak is half the marketing
  number.

On a B200 the optimal kernel is FA4 with three *hardware* async pipelines. On
sm_120a the optimal kernel is **register-resident FA2 with f16-QK + cp.async
double-buffer + SFU-overlapped softmax** — and that *is* imp's primary FA2. We
lose to datacenter Blackwell on architecture, not implementation; we win decode
(uncontested NVFP4) and MoE-pp2048 in exchange.

## 6. Decode counterpart (one paragraph — HBM-bound)

The optimal decode "kernel" is not a compute design but **traffic elimination**:
NVFP4 GEMVs at the GDDR7 ceiling (~1.5 GB/ms = 86 % datasheet) + a conditional
CUDA graph + PDL. It is built, at the limit, and its only remaining wall-breaker
is algorithmic (speculation), not kernel-technical. See the decode levers in
`MEMORY.md` and `docs/performance.md`.
