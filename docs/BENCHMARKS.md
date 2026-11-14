<!--
layer: L1
audience: operators
verified: 2026-11-01
commit: d8550086
-->

# Benchmarks

Reproducibly anchored measurements: every row states **when**, **on what commit**, **with which
CUDA version and quant**, and **the exact command**; re-run the command on the stated commit to
reproduce. The commit SHA is authoritative, tagged releases snapshot a SHA; every sweep this file
superseded (older competitive re-sweeps, the imp-vs-vLLM concurrency progression, the
batched-decode optimization history) moved verbatim to
[`archive/BENCHMARKS_PRE_V0.44.md`](archive/BENCHMARKS_PRE_V0.44.md); the GGUF comparisons
(GGUF is legacy since 2026-10-30) to [`archive/BENCHMARKS_GGUF.md`](archive/BENCHMARKS_GGUF.md).

**Hardware (constant across all runs):** single RTX 5090 (GB202, 32 GB GDDR7, water-cooled,
never thermally throttled), Ryzen host, WSL2, Docker; **method:** greedy (temp = 0), CUDA Graphs
on, 10 repetitions, isolated run (one model per process, no concurrent GPU work), clocks warmed
before timing, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Decode (tg) is the reliable A/B signal; prefill
(pp) varies up to 2.6x across container restarts (cuBLAS autotuning) and is not tabulated for
comparisons.

**Toolchain (current: `v0.66.0`):** C++23, Ubuntu 26.04 / GCC 15.2, CUDA 13.5.0.

The CI-gated canonical baseline lives in [`tests/perf_baseline.json`](../tests/perf_baseline.json)
(8% decode / 8% prefill regression gate, 10% peak-VRAM ceiling); pinned values, thresholds and
methodology: [`PERF.md`](PERF.md). Refresh via `scripts/gen_perf_baseline.sh`.

## Decode vs vLLM (SafeTensors heroes, release bar 2)

Same checkpoint on both engines, each served from a fresh container, one stream, 300-token greedy
answers with `ignore_eos`, median of 3 waves (`tools/analysis/conc_client.py`). The bar reads the
spec-off column (vLLM runs no speculation by default); imp's default column is reported only:
greedy answers that fall into a loop let the n-gram drafter accept everything (Qwen3-8B FP8).

| Model | imp default | imp spec-off | vLLM 0.31.0 | imp lead (spec-off) |
|---|---:|---:|---:|---:|
| Qwen3-8B NVFP4 (cortecs) | 306.9 | **273.9** | 186.5 | +46.9 % |
| Qwen3-8B FP8 | 2415.2 (loop) | **173.6** | 112.0 | +55.0 % |
| Qwen3-14B NVFP4 | 170.2 | **169.2** | 128.8 | +31.4 % |
| Qwen3-Coder-30B-A3B NVFP4 | 392.6 | **357.6** | 220.6 | +62.1 % |
| Qwen3.6-35B-A3B NVFP4 | 329.4 | **334.1** | 183.9 | +81.7 % |
| Gemma-4-26B-A4B NVFP4 | 428.0 | **274.4** | 222.6 | +23.3 % |
| Nemotron-3-Nano-30B-A3B NVFP4 | 617.6 | **414.5** | engine init fails | - |
| Qwen3.8-27B NVFP4 (vllm export) | 99.4 | **96.4** | 70.7 | +36.4 % |

[PROV: commit=cff927db date=2026-10-30 hw=RTX5090 model=per-row quant=per-row cuda=13.4.3
       path=server-api cmd=`RELEASE_BAR=1 bash scripts/bench_competitive.sh` n=3 waves x 1 run per arm
       harness_md5=b531179dfc496b75ccdb2a7f5fbda93b note=vllm/vllm-openai v0.31.0 sha256 c1c9f6fd;
       that run gated the default column, the lead column is recomputed from its TSV on spec-off;
       gpt-oss-20b not contested (SafeTensors not staged)]

## Feature validations (decode)

One-off measurements for shipped decode features, each still the only sweep on record for that
feature.

| Date | Commit | Model (quant) | Feature | Result |
|---|---|---|---|---|
| 2026-07-11 | `fp8-ssm-gguf` | Qwen3.6-35B-A3B hybrid (Q4_K_M) | `gemm.fp8_ssm_proj` on Q8_0-kept GDN projections | tg256 224.4 -> **272.0** (+21%), PPL +1.8% (201-tok corpus) |
| 2026-08-28 | `899301c6` | Qwen3-8B (Q8_0, fp8 KV) | sparse decode attention, 32k ctx | tg136 160.3 -> **199.5** (+24.5%, 3/3 alternating) |
| 2026-08-29 | `sparse-serving` | Qwen3-8B (Q8_0, fp8 KV) | sparse attn, 3 streams x 25k ctx | 155.6 -> **197.7** (+27%, `tools/analysis/serving_sparse_ab.sh`) |
| 2026-08-30 | `nvfp4-loadwidth` | Qwen3.8-27B (NVFP4, nvfp4 KV) | server decode @77k ctx, batch 8 | 64.1 -> **74.2** (+15.7%, 18/18 runs at 125 chunks) |
| 2026-08-30 | `nvfp4-sparse` | Qwen3.8-27B (NVFP4, nvfp4 KV) | + sparse attn @77k ctx, batch 8 | 74.3 -> **100.2** (+35%, NIAH 8-9/10 vs dense 10/10) |

Commands: `imp-cli --bench` for the first three, `imp-server --max-batch 8
[--set attention.sparse_topk_tokens=N]` plus a forced-length client for the last two, all CUDA 13.3.

## NVFP4 SafeTensors decode (tg256)

Baseline rows 2026-06-09, commit `ec9145b3`, CUDA 13.3, warm isolated runs (one model per
process, 10 reps), command `imp-cli --model <dir>/ --bench --bench-pp 16 --bench-reps 10
--max-tokens 256`. Superseding measurements noted per row.

| Model | Params (active) | tok/s |
|---|---|---:|
| Qwen3-8B-cortecs | 8.2B | 270 |
| Qwen3-14B | 14B | 159 |
| Qwen3-30B-A3B-Modelopt | 30B (3B) | 305 |
| Qwen3-Coder-30B-A3B | 30B (3B) | 338 |
| Qwen3.6-35B-A3B | 35B (3B) | 257 -> **320** (2026-07-10, `80864b06`, `gemm.fp8_ssm_proj` sidecar, +19.2% spec-off) |
| Gemma-4-26B-A4B | 26B (4B) | 266 |
| Nemotron-3-Nano-30B | 30B (3B) | 148 -> **386** (2026-08-12, PR #1389: CUDA graphs were wrongly demoted for pure-SSM layers, not an arch limit) |
| gpt-oss-20b | 21B (3.6B) | 325 -> **391** (2026-07-13, `63df2d30`, `gemm.fp8_attn_proj` sidecar, +12%, teacher-forced PPL unaffected by construction) |

gpt-oss experts convert SafeTensors MXFP4 to NVFP4 at load (bit-exact nibbles, power-of-two
scales) and register for the grouped NVFP4 prefill GEMM (pp512 ~16-19k tok/s, measured on the
CUTLASS kernel; the own grouped kernel since #2929); attention stays on cuBLAS. vLLM 0.31.0 runs
NVFP4 natively on `sm_120` (`FlashInferCutlassNvFp4LinearKernel`, FP8 KV with the XQA decode
backend on Qwen3-14B-NVFP4, 2026-10-31 server log);
per-row superseding evidence: [archive](archive/BENCHMARKS_PRE_V0.44.md).

## NVFP4 prefill (FP16-QK FA2 primary hd=128 prefill, #687)

imp rows 2026-06-13, commit `290a163a`, CUDA 13.3, median of 3 isolated trials x 40 reps;
command: `imp-cli --model <dir> --bench --bench-pp <n> --bench-reps 40 --max-tokens 256`. vLLM
reference: 0.22.1 FlashInfer-NVFP4 (fp8 KV), same host, measured 2026-06-11 (older than the imp
rows, ratios carry that cross-day caveat).

| Model | pp | imp tok/s | vLLM tok/s (06-11) | verdict |
|---|---|---:|---:|---|
| Qwen3-30B-A3B MoE | 2048 | **43646** | 34500 | imp +27% |
| Qwen3-30B-A3B MoE | 4096 | **37639** | 36200 | imp +4% |
| Qwen3-14B dense | 2048 | **26918** | 26600 | ~tie (+1.2%) |
| Qwen3-14B dense | 4096 | **24232** | 25300 | 1.04x behind |

imp also wins TTFT/pp512 outright (2.1-3.4x, vLLM has a flat-cost small-M regime). Decode
(tg256 @ctx2048): 14B 159, 30B-A3B ~317.

## Qwen3.8-27B, imp vs vLLM

Fresh server per arm, engine defaults (imp `--max-concurrent N --set runtime.max_batch_size=N`, vLLM `--max-num-seqs N --max-model-len 16384`), 300-token greedy answers, `ignore_eos`, completion tokens / wall, median of 3 trials (range).

[PROV: commit=ca8780ef date=2026-10-27 hw=RTX5090 model=Qwen3.8-27B quant=NVFP4 cuda=13.4.3 path=server-api cmd=`tools/analysis/engine3_conc.sh`
       n=3 trials x 3 waves note=imp ghcr v0.59.0 sha256 7a52810d, vllm/vllm-openai:v0.31.0 sha256 c1c9f6fd;
       raw logs docs/audit/launch_2026_10_27/]
[PROV: commit=ed4422b7 date=2026-10-31 hw=RTX5090 model=Qwen3.8-27B quant=NVFP4 cuda=13.4.3 path=server-api cmd=`tools/analysis/engine3_conc.sh`
       n=3 trials x 3 waves note=imp v0.60.0 tree 0937bac4 (imp:test), vllm/vllm-openai:v0.31.0; harness_md5=5525cd20;
       raw logs docs/audit/launch_2026_10_31/]

| engine | checkpoint | speculative decoding | 1 stream tok/s | 32 streams tok/s |
|---|---|---|---:|---:|
| imp v0.60.0 | Qwen3.8-27B-NVFP4-vllm | off (`speculative.mtp_k=0 speculative.ngram=false`) | **95.5** (95.4-95.7) | - |
| imp v0.60.0 | Qwen3.8-27B-NVFP4-vllm | default: none at 32 | - | **1974.8** (1963.7-1976.6) |
| vLLM 0.31.0 (2026-10-31) | Qwen3.8-27B-NVFP4-vllm | default: none | 70.4 (70.0-70.9) | 1380.1 (1378.1-1394.3) |
| imp v0.59.0 | Qwen3.8-27B-NVFP4-vllm | off (`speculative.mtp_k=0 speculative.ngram=false`) | **96.3** (95.4-96.4) | - |
| imp v0.59.0 | Qwen3.8-27B-NVFP4-vllm | default: MTP k=2 at 1 stream, none at 32 | 116.2 (113.9-121.8) | **1979.1** (1978.6-1980.1) |
| vLLM 0.31.0 | Qwen3.8-27B-NVFP4-vllm | default: none | 69.9 (68.7-70.4) | 1414.2 (1404.7-1415.6) |
| vLLM 0.31.0 | Qwen3.8-27B-NVFP4-vllm | MTP k=2 (`--speculative-config`) | 124.4 (119.4-126.9) | 913.6 (906.0-941.5) |

- MTP k=2 on both: these template completions put the two within prompt noise (imp waves 95.7-132.5); 6 real chat prompts x 400 tokens, 3 starts (`tools/analysis/mtp_vs_vllm.sh`): imp **126.49** vs vLLM 116.84 tok/s. Speculation off: imp leads by 35.7 % (v0.60.0; 37.8 % on v0.59.0). imp keeps the KV cache in NVFP4 on this family. The 2026-10-11 sweep (imp 91.0 / 93.7 one stream): [archive](archive/BENCHMARKS_PRE_V0.44.md).

### MTP on a default server (concurrent GDN hybrid)

Default `imp-server` (no `--max-batch`, auto batch), fresh server per arm, 3 alternating pairs; 1 stream = 6 real chat prompts x 400 greedy tokens x 2 rounds (`seq_stream_client.py`, thinking on), c4 / c32 = `conc_client.py` 300-token bursts, median of waves 1-3. Harness `tools/analysis/mtp_concurrent_ab.sh`.
[PROV: commit=d1765b68 date=2026-11-03 hw=RTX5090 model=Qwen3.8-27B quant=NVFP4 cuda=13.4.3 path=server-api cmd=`tools/analysis/mtp_concurrent_ab.sh` n=3 pairs note=main d1765b68 vs this branch, images built from each tree; harness_md5=d264afa75dd4; raw summaries docs/audit/qwen38_mtp_concurrent_2026_11_03/]

| arm | max batch | 1 stream tok/s | c4 tok/s | c32 tok/s | own peak MiB | KV plan blocks |
|---|---:|---:|---:|---:|---:|---:|
| main, default (head not loaded) | 32 | 96.87 (96.86-96.90) | 341.3 (340.1-341.8) | 2043.5 (2040.7-2046.5) | 22034 | 15651 |
| main, `--set speculative.mtp_k=2 --set speculative.ngram=false` (1 trial) | 17 | 130.82 | 334.9 | 1116.7 | 22276 | 16954 |
| this branch, default (auto takes the head) | 32 | **131.28** (131.10-131.43) | 339.7 (339.1-340.1) | 2042.0 (2039.9-2042.0) | 24807 | 13230 |

- 1 stream +35.5 %, 70.0 % of drafts accepted (2552 / 3644); c4 -0.47 % (all 3 pairs lower, 900-token c4 +0.4 % in one pair), c32 -0.07 % (pairs mixed sign). Forcing the head on main cost c32 45 %: the auto batch charged it at recurrent-state price (32 -> 17 slots); without that charge the head comes out of the KV pool: 15651 -> 13230 blocks (all 32 slots fit 7825 -> 6615 tokens), own peak +2773 MiB.

### MTP across chat turns (`speculative.mtp_history_slots`)

Default `imp-server`, fresh server per arm, 3 alternating pairs, 6 user turns x up to 1500 tokens per conversation (`mtp_multiturn_probe.py`, thinking on, the assistant turn sent back as `content` only), follow-up = turns 2-6, tok/s = (completion tokens - 1) / stream time. On main the side request or the other chat overwrote the one head KV slot and the next turn's prefix-cache resume (464-2272 cached tokens) unbound the head (`hidden gap before chunk`: 34 per arm on main, 0 here). [PROV: commit=eb178985 date=2026-11-03 hw=RTX5090 model=Qwen3.8-27B quant=NVFP4 cuda=13.4.3 path=server-api cmd=`tools/analysis/mtp_multiturn_ab.sh` n=3 pairs note=main vs this branch, images built from each tree; harness_md5=bd8d987066a2 probe_md5=d742d48b6488; raw summaries docs/audit/qwen38_mtp_history_slots_2026_11_03/]

| conversation | main follow-up tok/s | this branch | turns drafting, main -> this branch |
|---|---:|---:|---|
| greedy | 119.93 (119.75-120.08) | 119.60 (118.73-120.07) | 15/15 -> 15/15 |
| sampled | 112.60 (112.25-112.90) | 112.71 (112.16-113.00) | 15/15 -> 15/15 |
| greedy + UI side request | 101.88 (101.27-102.25) | **122.28** (121.84-122.53) | 3/15 -> 15/15 |
| two chats alternating | 101.52 (101.16-101.75) | **117.43** (117.23-117.68) | 6/30 -> 30/30 |
| 6 fresh chats x 400, 1 stream (`mtp_concurrent_ab.sh`, 5 pairs) | 131.06 (130.34-131.41) | 130.95 (130.49-131.53) | c4 338.64 -> 338.14, c32 2022.6 -> 2018.2 (pairs mixed sign), own peak +162 to +219 MiB |
| MTP carry vs 6bca2fe2: 2 overlapping chats, all 12 turns [PROV: commit=6bca2fe2 date=2026-11-03 hw=RTX5090 model=Qwen3.8-27B quant=NVFP4 cuda=13.4.3 path=server-api cmd=`tools/analysis/mtp_multiturn_ab.sh` + `tools/analysis/mtp_concurrent_ab.sh` n=3 pairs note=harness_md5=5c087000f7fb overlap_md5=ca7d262552dc concurrent_md5=f8ac61ec6cb4; raw summaries docs/audit/qwen38_mtp_carry_2026_11_03/] | 93.78 (93.76-93.80) | **98.38** (97.74-99.00) | 6/36 -> 35/36 |
| `mtp_carry`: greedy + UI side request / 6 fresh chats x 400, 1 stream | 120.37 (120.08-120.60) / 131.50 (131.42-131.61) | 119.97 (117.60-124.27) / 131.59 (131.53-131.65) | 15/15 -> 15/15; c4 339.37 -> 338.60 (3/3 pairs lower), c32 2024.17 -> 2034.10 (3/3 higher), own peak 25059-25119 -> 25043-25073 MiB |
| `speculative.mtp_batch_verify` vs 288b9c52: 2 / 4 overlapping chats (`CHATS=2` / `4`), all turns [PROV: commit=288b9c52 date=2026-11-04 hw=RTX5090 model=Qwen3.8-27B quant=NVFP4 cuda=13.4.3 path=server-api cmd=`tools/analysis/mtp_multiturn_ab.sh` + `tools/analysis/mtp_concurrent_ab.sh` n=3 pairs note=this branch with `--set speculative.mtp_batch_verify=4` (the new default); harness_md5=1407282084d4 overlap_md5=6bd477985f61 concurrent_md5=f8ac61ec6cb4; pair 3 of overlap_ab_summary.txt overlapped a dockerd restart, replaced by overlap_ab_pair3_rerun_summary.txt; raw summaries docs/audit/qwen38_mtp_batch_verify_2026_11_04/] | 94.21 (92.28-95.18) / 83.72 (78.77-86.47) | **117.06** (113.63-119.18) / **117.45** (114.84-120.91) | 35/36 -> 36/36, 13/72 -> 72/72 |
| `mtp_batch_verify`, quiet host: 6 fresh chats x 400 x 2, 1 stream / c4 / c32 (`controls_quiet_host_summary.txt`) | 131.13-131.56 / 337.0-338.1 / 2020.2-2030.6 | 131.06-131.52 / **457.2-461.8** / 2029.8-2034.7 | key off on this image 131.21-131.60 / 337.7-338.5 / 2030.7-2031.8; own peak 24541-24551 -> 24447-24771 MiB, KV plan 4812 -> 4531 blocks |
| `speculative.mtp_batch_verify_k` 2 vs daa62a92: 2 / 4 overlapping chats, all turns [PROV: commit=daa62a92 date=2026-11-04 hw=RTX5090 model=Qwen3.8-27B quant=NVFP4 cuda=13.4.3 path=server-api cmd=`tools/analysis/mtp_multiturn_ab.sh` + `tools/analysis/mtp_concurrent_ab.sh` n=3 pairs note=this branch, tree 2388cba1, k 2 the new default; harness_md5=1407282084d4 overlap_md5=6bd477985f61 concurrent_md5=f8ac61ec6cb4; raw summaries docs/audit/qwen38_mtp_batch_verify_k2_2026_11_04/] | 119.44 (118.81-120.67) / 121.59 (121.10-121.91) | **132.79** (130.44-134.13) / **134.17** (131.51-135.86) | 36/36, 72/72 both; k 1 on this branch 119.44 / 120.81; 6 fresh chats x 400, 1 stream / c2 / c4 / c32 (`concurrent_ab_summary.txt`) 130.70-131.73 / 241.3-255.0 / 457.6-469.6 / 2022.3-2031.5 -> 131.28-131.58 / **279.2-291.2** / **485.8-507.8** / 2030.7-2035.7, own peak 24430-25148 -> 25336-25350 MiB, KV plan 12305 -> 11380 blocks |
| Batched drafts on the NVFP4 draft head vs 15d35035: 2 / 4 overlapping chats, all turns [PROV: commit=15d35035 date=2026-11-04 hw=RTX5090 model=Qwen3.8-27B quant=NVFP4 cuda=13.4.3 path=server-api cmd=`tools/analysis/mtp_multiturn_ab.sh` + `tools/analysis/mtp_concurrent_ab.sh` n=3 pairs note=this branch 0a5e17cf; harness_md5=1407282084d4 overlap_md5=6bd477985f61 concurrent_md5=f8ac61ec6cb4; raw summaries docs/audit/qwen38_mtp_batched_draft_head_2026_11_04/] | 129.67 (127.82-132.98) / 133.07 (130.50-135.92) | **135.18** (132.87-138.60) / **138.15** (137.86-138.74) | 36/36, 72/72 both; head pass 2 / 3 / 4 requests 3.74-3.93 / 3.77-3.97 / 3.78-4.02 -> 2.73-3.07 / 2.91-3.19 / 3.14-3.31 ms (`step_price.txt`); 6 fresh chats x 400, 1 stream / c2 / c4 / c32 130.94-131.63 / 257.2-281.1 / 487.7-514.5 / 2020.7-2027.1 -> 131.26-131.34 / **286.0-300.6** / **529.8-538.4** / 2031.2-2035.6, own peak 25336-25350 -> 25334-25336 MiB |
| `speculative.mtp_kv_window` + `mtp_batch_verify_solo` vs b3c0087d: 1 chat at 8k / 20k / 40k / 80k tokens; 2 overlapping chats at 20k / 40k / 80k (`mtp_longctx_ab.sh`, repo docs + source prompts, greedy 384 tokens). On main a prompt above 16384 tokens never binds the head; here the head KV is a 16384-row ring and a request decoding alone verifies in the batched step (the single verify chunk's FA2 call: 0.99 ms per layer at 20k) [PROV: commit=b3c0087d date=2026-11-04 hw=RTX5090 model=Qwen3.8-27B quant=NVFP4 cuda=13.4.3 path=server-api cmd=`tools/analysis/mtp_longctx_ab.sh` + `tools/analysis/mtp_concurrent_ab.sh` n=3 pairs note=branch tree 17c80274, frozen b3c0087d prompt corpus; harness_md5=5454cd8bd776 client_md5=c26973421a02 concurrent_md5=f8ac61ec6cb4; raw summaries docs/audit/qwen38_mtp_long_context_2026_11_04/] | 104.10-105.00 / 89.53-90.45 / 88.55-89.01 / 86.56-87.44; 152.40-153.54 / 150.70-151.54 / 146.48-147.71 | **153.14-154.98** / **143.35-143.67** / **133.68-134.76** / **140.99-141.46**; **240.98-263.89** / **243.78-245.33** / **235.31-238.06** | accepted / drafted per 1-chat answer 232/270, 0, 0, 0 -> 230/302, 231/300, 224/314, 233/296; 6 fresh chats x 400, 1 stream / c4 / c32 127.55-128.05 / 503.2-550.4 / 1937.9-1943.5 -> **149.14-149.85** / 511.2-546.6 / 1936.9-1958.0 (c4, c32 pairs mixed sign); TTFT 80k 7.25-7.31 -> 7.58-7.61 s; ring 16384 / full attention / ring 4096 at 40k 222/276 / 222/277 / 215/282 (`window_arms_c1.txt`); greedy text vs MTP off diverges after 6-103 words here, 5-82 on main's verify chunk; `mtp_batch_verify_solo=false` byte-identical to main at 1k / 4k / 8k |
| Factor row (2 drafts, draft 1 as a factor row, always on since 2026-11-14) vs 1 draft, the concurrent default of e713db1a: 1 chat at 20k / 40k / 80k tokens; 2 chats at 20k / 40k / 80k, all streams (`mtp_longctx_ab.sh`, greedy 384 tokens) [PROV: commit=e713db1a date=2026-11-11 hw=RTX5090 model=Qwen3.8-27B quant=NVFP4 cuda=13.5.0 path=server-api cmd=`tools/analysis/mtp_longctx_ab.sh` n=2 pairs note=one image of this branch, arms by `--set`: `speculative.mtp_batch_verify_k=1` / defaults / `speculative.mtp_batch_verify_factor_row=false` (key removed 2026-11-14); frozen prompt corpus (19773 prompt tokens at 20k); harness_md5=1fb90481a6c3 client_md5=e207c7b27faf; raw summaries docs/audit/qwen38_mtp_verify_factor_row_2026_11_11/] | 137.08-137.79 / 133.81-133.86 / 128.41-128.80; 248.94-249.24 / 243.31-244.16 / 240.15-240.44 | **165.83-166.20** / **156.86-156.98** / **154.30-154.34**; **287.77-288.06** / **288.09-292.06** / **284.98-286.72** | second spare per row instead of the factor row: 149.76-151.26 / 155.09-155.70 / 149.54-149.57; 289.06-297.90 / 266.97-272.43 / 270.32-271.41. 4 chats at 8k / 20k: 476.19-482.75 / 450.19-451.67 -> 575.17-583.27 / 509.62-509.81; 8 chats at 8k 569.98-571.44 -> 570.43-573.75. KV plan 11239 -> 10901 blocks (second spare 10313) |
- `mtp_overlap_probe.py`: chat c starts after chat c-1's first 300 deltas. `mtp_batch_verify`: while <= 4 decode, every bound chat forwards its last token plus its MTP draft (2 rows each); 4 rows = 12.57-13.04 ms forward + argmax plus 2.00-2.64 ms head feed and draft for 3.66 tokens, 8 rows 13.12-13.65 + 2.02-2.65 ms for 7.15-7.20 tokens, plain decode on main 11.34 / 11.72 ms for 2 / 4 rows (`verify_step_pricing.txt`, `pricing_main_summary.txt`); UI side request (decodes alone) 124.59 / 116.62 / 120.24 -> 110.96 / 117.18 / 122.30 with equal drafts; c2 166.6 / 166.8 / 165.3 -> 242.4 / 230.6 / 225.9 (first control run, beside a foreign 1-core process). Rationale and the two `batch_verify` defects fixed on the way: [plan](plans/2026-09-11-batched-mtp-verify.md#bound-verify-at-2-4-rows-2026-11-04-speculativemtp_batch_verify-default-4). `mtp_batch_verify_k` 2: every bound chat forwards its last token plus a 2-token MTP chain (3 rows each); 6 / 9 / 12 rows = 12.75-12.76 / 13.12-13.15 / 13.87-13.93 ms forward + argmax plus 3.71-3.80 ms head pass for 4.59-4.74 / 6.66-6.98 / 9.24-9.42 tokens, against 12.14-12.15 / 12.41-12.43 / 12.86-12.89 ms + 1.96-2.03 ms for 3.39-3.48 / 5.30-5.42 / 6.87-7.16 at k 1 (`overlap_and_step_price.txt`); fresh servers: greedy and side-request transcripts 6/6 byte-identical to main (`identity_summary.txt`). [Plan](plans/2026-09-11-batched-mtp-verify.md#two-drafts-per-bound-request-2026-11-04-speculativemtp_batch_verify_k-default-2). `mtp_carry` overlap (`mtp_overlap_probe.py`): chat 1 starts after chat 0's first 300 deltas, 2 s / 6 s think time; on main a chat bound at its lone prefill unbound at its first lone step after a batched one (`head cache behind the context after batched decode`: 9 per arm, 0 with the carry) and the second chat never bound. Cost per batched step: one 10 KiB hidden-row copy per bound row, one 256-row head pass per 256 steps, 10 MiB scratch; c4 / c32 bind nothing (spec counters equal in both arms); fresh servers: greedy and side-request transcripts 6/6 byte-identical to main (`gates_final_identity.txt`); multiturn own peak 22856 MiB on main, 22856-24053 here (2 of 4 runs higher, as on one main pair of the history-slots A/B, 24485).

## imp vs vLLM at concurrency

Same 19.6 GiB compressed-tensors NVFP4 checkpoint on both engines (`imp-quantize --format vllm`
from `Qwen/Qwen3.8-27B-FP8`), same client (`tools/analysis/conc_client.py`, 300-token greedy
gens, `/v1/completions`, aggregate = completion tokens / wall, median of 3 waves), fresh server
per arm, 3 alternating trials, imp pinned (`--max-concurrent CONC --set
runtime.max_batch_size=32 --set runtime.max_seq_len=4096 --set kv_cache.max_blocks=2387`). Harness `tools/analysis/vllm_conc_ab.sh`.

[PROV: commit=a44298cb date=2026-09-02 hw=RTX5090 model=Qwen3.8-27B-NVFP4-vllm quant=NVFP4-CT
       cuda=13.3 path=server-api cmd=tools/analysis/vllm_conc_ab.sh n=3-trials-x-3-waves-per-arm
       vllm=v0.27.1 / v0.28.0]

| run | shape | imp tok/s (median) | vLLM tok/s (median) | imp vs vLLM |
|---|---|---:|---:|---:|
| 1 | 32 streams, 38-tok prompts, vLLM 0.27.1 | **1807.9** | 1447.8 | **+24.9%** |
| 2 | 32 streams, 38-tok prompts, vLLM 0.28.0 | **1833.8** | 1410.7 | **+30.0%** |
| 3 | 8 streams, 36-tok prompts, vLLM 0.27.1 | **573.0** | 495.8 | **+15.6%** |
| 4 | 32 streams, 1082-tok prompts, vLLM 0.27.1 | **873.4** | 497.8 | **+75.5%** |
| 5 | dense Qwen3-14B-NVFP4, 32 streams, 38-tok, vLLM 0.27.1 | 3480.4 | **3767.9** | -7.6% |
| 6 | dense Qwen3-14B-NVFP4, 8 streams, 36-tok, vLLM 0.27.1 | **1085.4** | 1005.8 | **+7.9%** |
| 7 | dense Qwen3-14B-NVFP4, 32 streams, 982-tok, vLLM 0.27.1, imp KV pool 8192 blocks | 1470.6 | **2492.6** | -41.0% |
| 10 | run 9 with `attention.paged_fp8_multitok=4` (key removed 2026-11-14, 4 is the only value), paced serving prefill (#1950-#1952), grouped FP8 decode attention (#1953) | 2491.2 | 2490.9 | +0.0% (2/3 pairs) |
| 11 | run 5 on imp v0.60.0 vs vLLM 0.31.0 (2026-10-31) | **4373.4** (4355.8-4380.5) | 4080.8 (4033.1-4107.9) | **+7.2%** (3/3 pairs) |
| 12 | run 6 on imp v0.60.0 vs vLLM 0.31.0 (2026-10-31) | **1236.5** (1233.7-1238.8) | 1086.5 (1025.8-1116.8) | **+13.8%** (3/3 pairs) |
| 13 | run 7 on imp v0.60.0 vs vLLM 0.31.0 (2026-10-31), 1000-token prompts | 2547.3 (2545.6-2548.9) | **2658.4** (2631.1-2668.2) | -4.2% (0/3 pairs) |

[PROV: commit=e08db9d2 date=2026-10-31 hw=RTX5090 model=Qwen3-14B-NVFP4 quant=NVFP4 cuda=13.4.3 path=server-api
       cmd=`MODEL_NAME=Qwen3-14B-NVFP4 IMG_VLLM=vllm/vllm-openai:v0.31.0 CONC={32|8|32} PLEN={0|0|1000} KV_BLOCKS={2387|2387|8192} tools/analysis/vllm_conc_ab.sh`
       n=3-trials-x-3-waves-per-arm note=runs 11-13; imp:test tree 0937bac4 (v0.60.0); harness_md5=1e9b1463 (client moved into imp:toolchain); raw logs docs/audit/launch_2026_10_31/q14_run{5,6,7}.txt]

The GDN-hybrid checkpoint (runs 1-4) leads vLLM at every shape tried; the dense counter-probe
(runs 5-10) crossed over between 8 and 32 streams on 2026-09-02; on v0.60.0 vs vLLM 0.31.0
(runs 11-13) imp leads at 8 and 32 streams with short prompts and trails only with 1000-token
prompts at 32 streams (-4.2%; a streamed re-run on main `ce149da7`, 3 waves per engine: TTFT max
1.19-1.23 vs 0.95-1.02 s, wall minus TTFT max 2.51-2.57 vs 2.63-2.64 s, so the gap is prefill);
single-stream imp leads 17-22% on both
checkpoints. The 32-stream GDN-hybrid
aggregate moved 81.5 -> 1807.9 tok/s across this branch's levers; runs 8-9 and the profiling
breakdown that attributed the gap (GEMM class, GPU idle, launch density):
[archive](archive/BENCHMARKS_PRE_V0.44.md).

## Long context (pp8192 / tg512 @ 16k ctx)

All rows 2026-07-11, commit `905630e2` (re-measured after #967/#968/#969), CUDA 13.3, isolated
runs; command: `imp-cli --model <m> --bench --bench-pp {8192|16384} --bench-reps {5|3}
--max-tokens {64|512} --max-seq-len {9216|17408}`. pp carries the usual restart variance; tg is
the signal.

| Model | Quant | pp8192 tok/s | tg512 @16k (defaults) |
|---|---|---:|---:|
| Qwen3-8B | Q8_0 | 13268 | **208.9** (FP8 KV auto since #977; 151.9 on FP16 KV) |
| Qwen3-Coder-30B-A3B | NVFP4 | 35516 | 269.5 |
| Qwen3.6-35B-A3B | NVFP4 | 14887 | **264.3** |
| Qwen3.6-35B-A3B | Q4_K_M (GGUF) | 9436 (pp16384) | **234.2** |

FP8-KV is worth **+39..41%** at 16k on Qwen3-8B; auto-engages on hint-less GGUF since #977
(`kv_fp8_no_hint_default_safe` allowlist). **128K single-chunk prefill** (2026-07-24, `d8bc45a8`):
Qwen3-14B NVFP4 `pp131072 = 3792 tok/s` (34.6s TTFT, `--prefill-chunk-size 0
--max-seq-len 140000`); auto `max_seq_len` ceiling is 128K since that commit; discovery-day
(pre-fix) numbers and the #963/#964/#967/#968/#969 root causes: [archive](archive/BENCHMARKS_PRE_V0.44.md).

## Concurrent serving throughput (batched decode)

Aggregate throughput = sum of completion tokens / wall-clock across N concurrent
`POST /v1/chat/completions` on `imp-server --model Qwen3-14B-NVFP4 --max-batch 16`
(`max_tokens=200`, `temperature=0.7`).

| Date | Commit | Model | Concurrency | Aggregate tok/s |
|---|---|---|---:|---:|
| 2026-07-12 | batched-sampling PR | Qwen3-Coder-30B-A3B-FP4 | 16 | **1173** sustained closed-loop median, above vLLM's published 1157 reference (cloudrift.ai) while keeping 5.4x its own single-stream decode (396 tok/s) |
| 2026-06-23 | pre-`#745` | Qwen3-14B-NVFP4 | 16 | 472 (single-block sampler + per-row LM head, the pre-fix floor) |

**+62% from two fixes** (#745 sampler, #746 LM head): top-k/top-p sampler 36% -> 6% of decode GPU
time; NVFP4 LM head 18% -> 7% (per-sequence M=1 GEMV loop -> batched-M). Single-stream decode is
unchanged by design, both fixes touch only the n>1 path; full optimization sequence (pipelined
batched decode, row-parallel sampling, CUTLASS LM head) and the vLLM reference details:
[archive](archive/BENCHMARKS_PRE_V0.44.md).

### F16 KV decode attention at 32 streams (2026-09-03)

Models whose KV stays FP16 under `kv_cache.dtype=auto` (GGUF without an FP8 hint: Llama, Mistral,
Gemma, Phi) moved from a cooperative GQA kernel (22% of DRAM bandwidth) to a four-tokens-per-warp
kernel (#1880/#1882). Aggregate tok/s at 32 concurrent streams, 1000-token prompts, 300-token
completions, `ignore_eos`, harness `tools/analysis/prefill_cap_conc_ab.sh`.

[PROV: commit=f70e072a date=2026-09-03 hw=RTX5090 model=three-model-sweep quant=per-row cuda=13.3
       path=paged-attention-f16 cmd=tools/analysis/prefill_cap_conc_ab.sh
       n=2-trials-x-3-waves flags=CONC=32,PLEN=1000,KV_BLOCKS=3000,IGNORE_EOS=1]

| Model | KV | cooperative | multitok (new default) | delta |
|---|---|---:|---:|---:|
| Llama-3.2-3B-Instruct-Q8_0 | F16 (auto) | 1623.9 | **2407.6** | +48.3% |
| Phi-4-reasoning-plus-NVFP4 | F16 (forced) | 1099.1 | **1812.0** | +64.9% |
| gemma-3-12b-it-Q4_K_M | F16 (auto) | 218.4 | **252.8** | +15.8% |

Single-stream long-context (`scripts/bench_longctx_ab.sh`, tg128, #1882 split-K vs the #1880
tree), Llama-3.2-3B-Instruct-Q8_0: 8k 366.2 -> **422.3** (+15.3%), 32k 183.1 -> **237.2**
(+29.5%), 64k 110.8 -> **153.1** (+38.1%).

## Multi-turn TTFT (hybrid prefix caching, #831 / v0.15.0)

Setup: `imp-server --model Qwen3.6-35B-A3B-NVFP4 --set runtime.max_seq_len=12288`, 6-turn
growing-history replay (~2k tokens per turn), `max_tokens=60`, `temperature=0`, streaming.
2026-07-02, CUDA 13.3: `v0.14.0` = `2316f2fd`, `v0.15.0` = `e80a26a4`.

| Turn | History | v0.14.0 TTFT | v0.15.0 TTFT | `cached_tokens` |
|---|---|---:|---:|---:|
| 1 | fresh | 1.62s | 1.89s | - |
| 2 | ~2k | 2.85s | 1.41s | 2016 |
| 4 | ~6k | 4.31s | 1.40s | 6080 |
| 6 | ~10k | 6.70s | **1.94s** | 10176 |

v0.14.0 grows linearly with history; v0.15.0 stays flat (**3.5x at ~10k tokens**, gap widening
with context). Recurrent-state snapshots (`server.recurrent_snapshot_mb`, default 256) enable
prefill-only deltas on SSM/GDN models; `usage.prompt_tokens_details.cached_tokens` reports the
hybrid hits.

### Attention-model replay (prefix cache)

Same growing-transcript shape on the standard attention path (paged prefix cache, default-on).
2026-07-15, CUDA 13.3, harness `tools/agent_replay_bench.py`, TTFT to first content token.

| Model | Turn 0 (~360 tok) | Turn 15 (~5.2k tok) cache-ON | Turn 15 cache-OFF | speedup |
|---|---:|---:|---:|---:|
| Qwen3-8B Q8_0 (dense) | 32ms | **19-35ms flat** | 439ms | **23x** |
| Qwen3-Coder-30B-A3B-FP4 (MoE) | 20ms | **33ms** | 168ms | **5.2x** |

`make bench-agentic MODEL=<name>` boots the server and runs this plus the concurrency harness.

## Output-quality gate

Every perf-relevant change must pass `python3 tools/analysis/degen_suite.py` against a running server.
