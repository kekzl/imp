# AWQ default groups at n_rep 4

Roadmap row 109, closed 2026-10-04 (#2474). Question: on dense models with n_rep 4, does `--calib` do better with all groups (`ABCDEG`) or with attention groups A and C off (`BDEG`)?

| Model (n_rep 4) | `ABCDEG` | `BDEG` | delta |
|---|---:|---:|---:|
| Qwen3-4B | 14.9701 | **14.1954** | -5.18 % |
| Qwen3-8B | 11.7296 | **11.4263** | -2.59 % |
| Phi-4 | - | - | `phi3` refused by `--calib` (no block layout) |

[PROV: commit=a566c761 date=2026-10-04 hw=RTX5090 model=Qwen3-4B,Qwen3-8B quant=NVFP4-AWQ cuda=13.4.2
       path=imp-quantize+imp-cli-perplexity n=1-per-arm-deterministic harness_md5=cc173414
       cmd=`imp-cli --calibrate` on the BF16 source over `fetch_calib_corpus.sh` output, `imp-quantize --calib --calib-groups <arm>`, `imp-cli --perplexity ppl_corpus_45k.txt --set runtime.deterministic_gemm=true --set kv_cache.dtype=fp16`]

Decision: `kAwqWideGqaRep` 5 -> 4 (`tools/imp-quantize/awq_sites.h`). n_rep 3 has no local model and keeps all groups.
