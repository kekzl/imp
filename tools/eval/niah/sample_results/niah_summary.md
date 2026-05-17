# NIAH Phase 2 results

Cells = pass-rate over depth × seed.

| Config | 4096 tokens | 16384 tokens |
|---|---:|---:|
| fp16 | 100.0% | 60.0% |
| fp8 | 100.0% | 60.0% |
| nvfp4 | 100.0% | 66.7% |
| mxfp4_kv | 100.0% | 60.0% |
| tq_qjl_on | 100.0% | 0.0% |
| tq_qjl_off | 100.0% | 0.0% |

## Path A verdict (16K context, nvfp4 vs mxfp4_kv)
- nvfp4:    66.7%
- mxfp4_kv: 60.0%
- Δ = -6.7 pp

🟡 **PASS WITH CAVEAT** — investigate per-depth pattern (Δ 5-10 pp)

## QJL-stripping (TQ) verdict (16K context, kept for comparison)
- tq_qjl_on:  0.0% (typically 0%: TQ engine-limited to ~4K BPE tokens, see Phase 2 findings)
- tq_qjl_off: 0.0%
- Δ = +0.0 pp
