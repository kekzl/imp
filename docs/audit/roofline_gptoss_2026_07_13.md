# gpt-oss-20b Decode Roofline Cell — Run `1d7bbf5a_20260713_011911` (#984)

Investigation cell for the gpt-oss decode tie vs llama.cpp b9976 (imp ~344 tok/s,
llama.cpp 329–345, mid-estimate +2% — below the ≥5% bar-2 lead). Analog to the
35B hybrid cell from the cv4 re-pin (`docs/audit/roofline_2026_07_11.md`).

- Commit: `1d7bbf5a` · config_version: 4 (additive: `gptoss-moe` model entry +
  regexes that only match gpt-oss-only kernels — cv4 runs stay comparable)
- Model: `/models/gpt-oss-20b` (MXFP4 SafeTensors: BF16 dense + NVFP4-converted
  experts), shapes pp512 + tg256, 2 restarts, methodology per `tools/roofline/README.md`.
- Two pipeline blockers fixed on the way (both in this branch):
  1. `~Model` double-freed all 216 converted MoE expert buffers (24 layers × 9
     pointers registered in both `wcache_->nvfp4_moe` and `gpu_allocations_`) —
     harmless standalone, SIGSEGV under nsys's CUDA interception.
  2. The nsys pass carried `--cuda-memory-usage=true`, which nothing in the
     pipeline reads; combined with gpt-oss's teardown free burst it crashed the
     injection on this WSL2 driver (app died before the CUPTI flush → rep had
     zero kernel rows). Dropped.

## Module 1 — decode (tg256) kernel classes

| Kernel class | Time share % (nsys) | achieved (med) | %-roofline (med) | note |
|---|---|---|---|---|
| gemv_nvfp4 (MoE experts + lm_head) | 40.8 | 1,070 GB/s | 59.7 | gate_up 1135 GB/s, down 960 GB/s — healthy |
| **gemv_fp (dense q/k/v/o, FP16)** | **33.5** | 980 GB/s | 54.7 | **kernel healthy — the BYTES are the lever** |
| moe_routing | 9.9 | 20 GB/s | 1.1 | ~7 tiny launches/layer/step, latency-bound |
| attn_decode_paged | 7.7 | 35 GB/s | 2.0 | 4-way splitk at ctx 64–320 = over-split |
| rmsnorm | 2.7 | 12 GB/s | 0.7 | grid (1,1,1) single block |
| rope / kv_write / elementwise | ~3 | <10 GB/s | <0.5 | bias chains: 3 add-bias kernels/layer/step |

Decode GPU busy ≈ 2.68 ms/token (no-graphs nsys window) vs ~2.9 ms/token wall
with graphs (344 tok/s) → with graphs ON decode is GPU-busy-bound and the
shares above are the correct lever weights.

## Per-launch decomposition (ncu steady-state, clocks locked to base)

| Launch (per layer per step) | avg time | bytes | GB/s |
|---|---|---|---|
| q-proj `gemv_fp16` (4096 rows) | 21.2 µs | 23.7 MB | 1,121 |
| o-proj `gemv_fp16` (2880×4096) | 21.1 µs | 24.3 MB | 1,154 |
| k-, v-proj `gemv_fp16` (512 rows each) | 2 × 6.5 µs | 3.0 MB | 469 |
| MoE gate_up `gemv_nvfp4_moe` (4 experts) | 33.1 µs | 37.5 MB | 1,135 |
| MoE down `gemv_nvfp4_moe` | 19.5 µs | 18.8 MB | 960 |
| paged-attn splitk + reduce | 12.4 + 4.5 µs | 0.6 MB | 41 / 19 |
| router chain (gate GEMV, top-k, 3× bias, weighted_sum) + 2× rmsnorm + rope + kv_write | ~29 µs | — | 5–35 |

Root cause of the FP16 class: gpt-oss dense weights are BF16 SafeTensors and
`nvfp4_beneficial()` (src/core/qtype.h) only covers GGUF qtypes → no decode
cache is built; q/k/v/o decode as 2 B/elem FP16 GEMVs while every GGUF/NVFP4
hero decodes its dense projections at ≤1 B/elem.

## Lever list (decode, prioritized; window share × byte/latency headroom)

1. **FP8 sidecar for dense q/k/v/o (33.5% share).** q/o run at 1.12–1.15 TB/s —
   bandwidth-bound, so bytes ≈ time. FP8-per-row (reuse the #949/#962 SSM
   sidecar infra) halves the bytes → est. **+17–20% decode**. NVFP4 (÷3.6 bytes)
   would be +~28% but carries the known attn-weight precision risk
   (pre_dequant_phase2 comment; #982 lm_head lesson) — must be PPL-gated
   against bar-1 parity either way.
2. **Router-chain fusion (9.9% at 1.1% roofline).** gate GEMV + logit bias +
   top-k + 3× expert bias + weighted_sum are ~7 latency-bound launches;
   fusing the scalar chain into 1–2 kernels ≈ halves the class → est. +4–5%.
3. **qkv fusion (within the 33.5%).** k/v at 512 rows reach only 469 GB/s;
   one fused qkv GEMV (5120 rows) runs at q/o efficiency → est. +2–3%,
   composes with lever 1.
4. **Decode splitk tuning (7.7% at 2.0%).** 4 splits at ctx 64–320 is
   over-split for hd=64/GQA-8 (same failure class as the 35B padded-batch
   split lesson) → est. +1–2% at short ctx, more at long ctx.
5. rmsnorm/bias micro-fusions (~5% combined at <1% roofline) → est. +1–2%.

Sum of 1–4 ≈ **+25–30% decode** — enough to turn the 344-tok/s tie into a
clear bar-2 lead if lever 1 passes the PPL gate.

## Prefill finding (bonus, not #984 scope)

`attn_legacy_softmax+cublas` carries **92.4% of pp512 attention** (8.8% of the
prefill window): gpt-oss prefill attention runs the legacy materialized cuBLAS
path, not FA2 — presumably the attention-sink path was never routed to FA2.
Report lever estimate ~4.4% pp window; worth a separate issue.

*(Raw: `tools/roofline/history/raw/1d7bbf5a_20260713_011911/` — ncu CSV +
nsys extracts committed, binaries local-only. Every number traceable to the
run id.)*
