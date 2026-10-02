# Roadmap

Last reviewed: 2026-10-02

Single-author, single-GPU experiment: "roadmap" means current focus, not
schedule. Shipped work is in [`CHANGELOG.md`](../CHANGELOG.md), competitive
numbers in [`BENCHMARKS.md`](BENCHMARKS.md), limitations in
[`LIMITATIONS.md`](LIMITATIONS.md).

| House rule | |
|---|---|
| Row form | fact + number + decision + ref, one row each |
| Investigation | goes to `docs/plans/`, the PR body or `LIMITATIONS.md`, never into a table cell |
| Lifecycle | entries are closed, corrected or superseded in place, never deleted |
| Proposing | open an issue with label `roadmap`; a row moves Now -> closed only with its acceptance met |
| Citations | `scripts/check_doc_citations.py` needs `path:N anchor`: anchor on N = ok, moved = `DRIFT` (exit 0, `--fix`), gone or ambiguous = `DEAD`, bare basename on 2 files = `AMBIGUOUS` (passes), so cite the path; [history](archive/roadmap_ledger_2026_09_28.md#house-rule-history) |

## Direction

| | |
|---|---|
| Goal | fastest local engine for AI agent workloads on consumer Blackwell |
| Workload | 20k-100k+ tokens per session, context accumulates, streams run in parallel |
| Working regime | aggregate throughput at tens of concurrent streams; batch=1 is settled |
| Foundations (2026-05) | chunked-prefill FMHA + 256 MiB S-matrix, ctx ~4-6k to 32k+ (#453), multi-request decode batching (#454), StreamingLLM auto-enable on full KV (#455) |
| Serving ground (2026-08) | warm weight cache (#956), suspend-to-RAM (#954), request-order independence (#957), gemma-3 IMA fix (#959) |

## Standing position

| axis | state | measured | ref |
|---|---|---|---|
| GDN hybrid @32 vs vLLM | AHEAD. Qwen3.8-27B 1807.9 vs 1447.8 tok/s (+24.9%, vLLM 0.27.1), 1833.8 vs 1410.7 (+30.0%, 0.28.0), @8 573.0 vs 495.8 (+15.6%), @32 x 1082-token prompts 873.4 vs 497.8 (+75.5%), 3 trials each | 2026-09-02 | [`BENCHMARKS.md`](BENCHMARKS.md#imp-vs-vllm-at-concurrency) runs 1-4 |
| dense NVFP4 @32 vs vLLM | PARITY OR AHEAD. Qwen3-14B 38-token prompts 3948.9 vs 3817.6 (+3.4%, 2026-09-03); 982-token prompts +0.2/+1.0/-0.6% (was 0.75x before #1953) | 2026-09-08 | [`benchmarks_pre_v0.44.md`](archive/benchmarks_pre_v0.44.md) runs 8-10 |
| batch=1 | 99.5 tok/s spec-off (`gdn.m1_fused`) vs the ~112 tok/s roofline (14.5 GB/token at 1628 GB/s resident), was 87.4 = 78% on 2026-08-27; past it only through the MTP verify | 2026-09-10 | `CHANGELOG.md` 0.39.0, [archive](archive/roadmap_ledger_2026_09_28.md#batch1) |
| raw-speed half of [`GOAL.md`](GOAL.md) | MET: batch=1 decode +13-48% vs llama.cpp on every hero, MoE prefill leads vLLM single-seq, cross-engine PPL parity measured | 2026-07-12 | [`PERF.md`](PERF.md), [`GOAL.md`](GOAL.md) |

Method: both engines on one client (`tools/analysis/vllm_conc_ab.sh`), 3 alternating trials, same checkpoint; PROV rows in [`BENCHMARKS.md`](BENCHMARKS.md#imp-vs-vllm-at-concurrency). Superseded rows: [archive](archive/roadmap_ledger_2026_09_28.md#standing-position-rows-superseded-2026-10-01).

## Now

Rows 1-15 are closed (index below); new rows start at 16, ranked by what an agent workload notices first.

| # | outcome | now -> target | size | ref |
|---|---|---|---|---|
| 16 | long sessions decode faster with no flag | sparse decode opt-in, Qwen3-8B 32k 160.3 -> 199.5 tok/s (+24.5 %) when on -> default on; flips only after NIAH + PPL on every hero model (GPU gate) | S | `docs/archive/roadmap_ledger_2026_09_28.md:18 199.5`, #2405 |
| 17 | long-prompt ingest finishes sooner with no flag | sparse prefill opt-in, Qwen3.8-27B pp77824 6835.55 -> 10529.05 tok/s (1.54x), NIAH 10/10, PPL +0.26 % -> default on; flips only after NIAH + PPL on every hero model (GPU gate) | S | `CHANGELOG.md:34 10529.05`, #2406 |
| 18 | an agent's next turn keeps its prefix hit while other sessions fill the KV pool | prefix eviction is session-blind -> request `session_id` pins that session's prefix blocks | S | [SGLang #29173](https://github.com/sgl-project/sglang/pull/29173), [TRT-LLM #16115](https://github.com/NVIDIA/TensorRT-LLM/pull/16115), #2407 |
| 19 | overload is rejected by queued prompt tokens, not only by request count | admission gate is `max_concurrent` requests, 429 when full -> plus a `max_queued_tokens` bound | S | `tools/imp-server/handlers.h:300 max_concurrent`, `tools/imp-server/main.cpp:312 stays 429`, [vLLM #49445](https://github.com/vllm-project/vllm/pull/49445), #2408 |
| 20 | parallel agents sharing one system prompt hit the hybrid prefix cache | recurrent snapshots at block boundaries -> snapshot at the branch point; upstream hit rate 43.8 -> 60.8 % | M | [vLLM #37898](https://github.com/vllm-project/vllm/pull/37898), [SGLang #34565](https://github.com/sgl-project/sglang/pull/34565), #2409 |
| 21 | Qwen3-Coder-Next runs on imp | arch falls back to GENERIC -> registry `qwen3_next` + MoE host offload | S | [HF](https://huggingface.co/Qwen/Qwen3-Coder-Next), `src/model/hf_config_loader.cpp:70 falling back to GENERIC`, #2410 |
| 22 | Devstral-Small-2 stays correct on long contexts | `Mistral3ForConditionalGeneration` maps to MISTRAL, `llama_4_scaling_beta` unread -> scaling applied; wrong long-context output suspected from code, not run | S | [HF](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512), `src/model/hf_config_loader.cpp:28 Mistral3ForConditionalGeneration`, #2411 |
| 23 | Granite 4.2 runs on imp | arch falls back to GENERIC -> own arch with attention and residual multipliers | S | [HF](https://huggingface.co/ibm-granite/granite-4.2-30b), `src/model/hf_config_loader.cpp:70 falling back to GENERIC`, #2412 |
| 24 | toolkit fixes reach the image, grouped NVFP4 cuBLASLt re-probed | CUDA 13.4.1 -> 13.4.2; `cublasLtMatmulGrouped` NVFP4 returns zero algos on sm_120 -> probe result recorded | S | `Dockerfile:23 13.4.1-devel`, `docs/internals/SM120.md:113 cublasLtMatmulGrouped`, [13.4.2](https://hub.docker.com/r/nvidia/cuda/tags?name=13.4.2), #2413 |
| 57 | server prefill of short prompts reaches the graphed path | server ragged prefill never graphed; Qwen3-30B-A3B-NVFP4 pp512 eager 20024.43 vs graphed 24708.99 tok/s, gap 19.0 % -> length-bucketed graphs for the ragged path | M | `tools/roofline/PERF_LOG.md:147 20024.43`, `tools/roofline/PERF_LOG.md:162 never reaches it`, #2435 |
| 58 | a fresh container plans the measured library reserve | `--rm` server plans the 3900 MiB constant vs measured 1930 MiB, Qwen3.8-27B 975 vs 2025 KV blocks, gap 51.9 % -> measured reserve at first start | S | `docs/LIMITATIONS.md:57 3900 MiB constant`, #2436 |
| 69 | Gemma-4 MoE prefill routing runs on more than one CTA | Gemma 4 forces deterministic GEMM, its permute launches `<<<1>>>` with a per-token rank loop: 12.1 % of nvfp4-gemma4-26b pp4096, 212.8 us/call (inventory 679866b6, untracked) -> multi-CTA permute, same layout | S | `src/compute/moe_routing.cu:723 moe_fused_permute_deterministic_kernel<<<1`, `src/compute/moe_routing.cu:470 rank +=`, `src/runtime/engine_init_resolver.cpp:814 Gemma 4: enabling`, #2465 |
| 70 | gpt-oss MoE prefill adds expert biases inside the fused activation | two `moe_add_expert_bias_sorted` launches before the clamped GLU, the fused act+quantize has neither: 13.4 % of mxfp4-gptoss-20b pp4096 (inventory 679866b6, untracked) -> bias + clamped GLU + quantize in one kernel | S | `src/exec/executor_forward_moe_cutlass.cpp:327 fused kernel has neither`, #2466 |
| 78 | HF checkpoints and GGUF resolve architectures from one table | HF class names in `parse_model_arch` (32 entries, GGUF path only) and `map_architecture` (35 entries) plus a model_type map (computed: entry counts) -> one table both loaders read | S | `src/model/model.cpp:331 HuggingFace architecture class names`, `src/model/gguf_loader.cpp:317 parse_model_arch(arch_str)`, `src/model/hf_config_loader.cpp:25 arch_map`, `src/model/hf_config_loader.cpp:123 type_to_class`, #2457 |
| 81 | MoE prefill refuses a weight format it has no kernel for | decode refuses since #2445; prefill `fused_dp4a_for_qtype` default arm launches nothing, the IMMA `qkind` chain ends in Q8_0 (0), both guarded upstream only -> no default arm, throw as decode does | S | `src/exec/executor_forward_moe_batch.cu:348 default:`, `src/exec/executor_forward_moe_batch.cu:566 : 0;`, `src/compute/moe_decode_select.cu:21 default: return nullptr`, #2459 |
| 83 | every endpoint reads sampling fields through one parser | 29 request keys parsed in both the chat and the completions handler (computed: rg key intersection) -> one parser for chat, completions, messages, responses | S | `tools/imp-server/handlers_chat_params.cpp:130 body.value("temperature", 0.7f)`, `tools/imp-server/handlers_completions.cpp:607 body.value("temperature", 0.7f)`, #2461 |
| 86 | tool calls parse per dialect from one registry | `ChatTemplateFamily` in 169 sites / 20 files (computed: rg), no `arg_key` (row 33) or Cohere (row 34) parser -> registry: template family -> tool-call parser | M | `src/model/chat_template.h:14 enum class ChatTemplateFamily`, #2464 |
| 88 | the arch -> profile mapping is pinned by a CPU test | `derive_model_profile` called by 1 test for 17 `ModelArch` values (computed: rg) -> table test, every arch -> expected profile flags | S | `tests/test_mla.cpp:144 derive_model_profile`, `src/model/model_profile.cpp:47 p.is_gpt_oss`, #2456 |
| 90 | the first start sizes the KV ceiling from its own measured reserve | mismatch only warns; Qwen3.8-27B charged 3900 vs measured 1930 MiB, +7004 blocks of ceiling (computed: 1970 MiB / 288 KiB NVFP4 block) -> apply the measurement in-process; same issue as row 58 | S | `src/runtime/engine_workspace_warmup.cpp:404 library reserve MISMATCH`, `docs/LIMITATIONS.md:57 3900 MiB constant`, #2436 |
| 96 | default requests are admitted up to the KV ceiling | admission reserves max_tokens / block + 1, server default 8192 -> 513 blocks per request, 32 x 513 = 16416 > 12425-block ceiling on Qwen3.8-27B (computed), 24 of 32 admitted -> reserve an expected length; trade-off #1635 | M | `src/runtime/scheduler.cpp:243 (req->max_tokens + bs - 1) / bs + 1`, `tools/imp-server/handlers.h:241 8192`, `docs/plans/2026-10-01-recurrent-state-paging-measured.md:27 8090-8533`, #2486 |
| 104 | exports ship the LM head the runtime already runs | runtime default is an FP8 per-row head, the quantizer writes BF16: Qwen3.8-27B 2425 MiB -> FP8 per-row head, 1213.4 MiB (computed: 248320 x 5120 x 1 B + 4 B per row) | S | `docs/GOAL.md:109 FP8 per-row head from a 16-bit head`, `docs/plans/2026-08-24-qwen38-port.md:208 2425.00 MiB`, #2479 |
| 105 | exports declare the FP8 KV hint | `kv_cache_quant_algo` written as null; the hint saves ~768 MiB KV on a 3.9k-token context at +1.07 % PPL (Qwen3-14B) -> FP8 hint for allowlisted families | S | `tools/imp-quantize/checkpoint_out.cpp:422 kv_cache_quant_algo`, `docs/quantization.md:227 768 MiB`, #2480 |
| 106 | an export records its full recipe and rebuilds byte-identical | checkpoint says `awq` or `none`; Qwen3-14B `awq` arms span 9.9068 to 12.2634 PPL, +23.8 % (computed) under one label -> groups, weight mode, corpus hash in the checkpoint + byte-identity test | S | `tools/imp-quantize/checkpoint_out.cpp:419 "awq" : "none"`, `docs/quantization.md:193 12.2634`, #2481 |

## Next

| # | outcome | now -> target | size | ref |
|---|---|---|---|---|
| 25 | dense models decode faster at 32 streams | small-M GEMM 5.49 ms per decode step -> 4.56 ms resident-bandwidth floor (Qwen3-14B-NVFP4) | M | `docs/archive/roadmap_ledger_2026_09_28.md:100 5.49`, #2414 |
| 26 | more models get the longer context on a 32 GB card | NVFP4 KV default only for QWEN35, max_model_len 48512 -> 131072 at +0.29..0.35 % PPL -> same default on families that pass PPL | M | `docs/LIMITATIONS.md:91 48512`, #2415 |
| 27 | Q8_0 GGUF long prompts take the faster route with no flag | M=2048 q_o IMMA 361.44 us vs FP16 route 288.96 us -> IMMA <= FP16 route | M | `docs/LIMITATIONS.md:81 361.44`, #2416 |
| 28 | hybrids ingest prompts faster without the NVFP4 PPL cost | NVFP4 `all` pp4096 +20.9 % at +4.25 % PPL, MXFP8 `all` +11.6 % -> W8A8 FP8 with act-quant fused into the norm | M | `docs/archive/roadmap_ledger_2026_09_28.md:12 20.9`, #2417 |
| 29 | follow-up turns on a hybrid reuse more of a long prompt | snapshots at block boundaries -> checkpoints inside a chunked prefill; upstream TTFT 9..25 % lower | M | [vLLM #52789](https://github.com/vllm-project/vllm/pull/52789), #2418 |
| 30 | more hybrid sessions keep a cached prefix in the same budget | snapshots at state precision -> int8 snapshots, upstream about 2x slots | S | [SGLang #28185](https://github.com/sgl-project/sglang/pull/28185), #2419 |
| 31 | an agent harness cuts a long reasoning phase mid-request | no control endpoint -> endpoint closes the think block, the answer continues | S | [llama.cpp #23971](https://github.com/ggml-org/llama.cpp/pull/23971), #2420 |
| 32 | code agents get speculation hits on repo text absent from the prompt | prompt-lookup drafts from the request's context -> drafts from a loaded corpus | M | `src/runtime/config.h:383 prompt-lookup`, [ExLlamaV3 v1.5.3](https://github.com/turboderp-org/exllamav3/releases/tag/v1.5.3), #2421 |
| 33 | GLM-4.7-Flash runs with tool calls | arch falls back to GENERIC -> MLA head-dim variant + `arg_key` tool format | M | [HF](https://huggingface.co/zai-org/GLM-4.7-Flash), `src/model/hf_config_loader.cpp:70 falling back to GENERIC`, #2422 |
| 34 | North-Mini-Code runs with tool calls | `CohereForCausalLM` maps to LLAMA -> cohere2_moe, parallel block, SWA, Cohere tool format | M | [HF](https://huggingface.co/CohereLabs/North-Mini-Code-1.0), `src/model/hf_config_loader.cpp:62 CohereForCausalLM`, #2423 |
| 35 | Mistral-Small-4 runs on imp | arch falls back to GENERIC -> mistral4 MLA + MoE, host offload | M | [HF](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603), `src/model/hf_config_loader.cpp:70 falling back to GENERIC`, #2424 |
| 36 | Nemotron-3-Super NVFP4 runs on imp | only `NemotronHForCausalLM` mapped -> LatentMoE, host offload | M | [HF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4), `src/model/hf_config_loader.cpp:44 NemotronHForCausalLM`, #2425 |
| 37 | LFM2 runs on imp | arch falls back to GENERIC -> short-conv block | M | [HF](https://huggingface.co/LiquidAI/LFM2-24B-A2B), `src/model/hf_config_loader.cpp:70 falling back to GENERIC`, #2426 |
| 38 | side-image base drift gets a Dependabot PR | Docker ecosystem covers `/` and `/tools/roofline` only, no compose -> tools/ Dockerfiles, compose files, digest pins | S | `.github/dependabot.yml:19 docker`, #2427 |
| 39 | users can verify the released image | no SBOM or attestation in the release workflow -> SBOM + signed provenance per tag | S | [`release-docker.yml`](../.github/workflows/release-docker.yml), #2428 |
| 40 | vision agents can pass https image URLs | `CPPHTTPLIB_OPENSSL_SUPPORT` never defined, https fails -> httplib built with OpenSSL | M | `tools/imp-server/image_fetch.cpp:249 needs an imp built with OpenSSL`, #2429 |
| 41 | build image carries current CMake fixes | CMake 4.3.1 -> 4.3.5 or 4.4.3 | S | `Dockerfile:36 cmake-4.3.1`, [v4.3.5](https://github.com/Kitware/CMake/releases/tag/v4.3.5), [v4.4.3](https://github.com/Kitware/CMake/releases/tag/v4.4.3), #2430 |
| 59 | MoE long prompts keep the GPU busy | pp4096 nvfp4-q36-35b wall 185.5 vs kernel sum 125.5 ms, idle 32.4 %, gemma4-26b 27.7 %, cause unattributed -> re-measure on current tree first (inventory predates the serial prefill graph change), then close the idle | M | `tools/roofline/inventory/KERNELS.md:80 32.4 %`, `tools/roofline/inventory/KERNELS.md:81 27.7 %`, `tools/roofline/PERF_LOG.md:155 offset-0 chunk captured`, #2437 |
| 60 | Gemma-4-26B Q4_K_M ingests prompts at llama.cpp speed | pp512 8946 vs llama.cpp 10749 tok/s, gap 16.8 %, Q5_1 raw IMMA 19.9 % of the cell; archive Q4_K refutation covers dense only -> pp512 >= 10749 | M | `docs/BENCHMARKS.md:84 10749`, `tools/roofline/inventory/KERNELS.md:58 19.9 %`, `docs/archive/roadmap_ledger_2026_09_28.md:355 Q4_K_M prefill gap`, #2438 |
| 61 | Q4_K MoE decode reaches the NVFP4 rate | Qwen3-30B-A3B tg128 Q4_K_M 324.65 vs NVFP4 389.79 tok/s, gap 16.7 %, `moe_gate_up<Q4_K>` DRAM active 60.3 %, L2 prefetch and NR=2/4 refuted -> tg128 >= 370; distinct from row 56 (launch time) | L | `docs/scoreboard.tsv:21 324.65`, `docs/scoreboard.tsv:20 389.79`, `tools/roofline/PERF_LOG.md:26 60.3 %`, `tools/roofline/PERF_LOG.md:22 L2 prefetch`, `tools/roofline/PERF_LOG.md:23 NR = 2 / 4`, #2439 |
| 62 | NVFP4 KV decode attention reads at the F16 rate | NVFP4 paged decode 614 GB/s vs F16 multitok 1644 GB/s, gap 62.7 %, conversion-bound -> >= 1200 GB/s | M | `docs/archive/roadmap_ledger_2026_09_28.md:143 614 GB/s`, `docs/plans/2026-09-04-lever-ledger-detail.md:49 1644 GB/s`, `docs/plans/2026-09-04-lever-ledger-detail.md:46 conversion-bound`, #2440 |
| 63 | Q4_K MoE prefill IMMA kernel leaves the issue bound | `mmq_imma_q4k_raw_kernel` 47.0 % of q4k-q3-30b pp512, tensor pipe ~25 % of 969.3 TOPS, 229 regs -> scale epilogue, PTX `ldmatrix.m8n16.x?.s8.s4` (ISA 9.4 sec 9.7.16.5.15) for the nibble unpack; risk: ldmatrix.x4 on Q8 IMMA -6.7 % at 240 regs | M | `tools/roofline/inventory/KERNELS.md:36 47.0 %`, `tools/roofline/PERF_LOG.md:59 tensor pipe ~25 %`, `tools/roofline/peaks/PEAKS.md:58 969.3`, `src/compute/mmq_q8_imma_q4k.cu:202 __vsub4`, `tools/roofline/PERF_LOG.md:77 -6.7 %`, #2441 |
| 64 | gpt-oss FA2 at hd=64 nears the FP16 tensor peak | FA2 hd=64 119.5 of 246 TFLOPS, 13.8 % of pp4096, occupancy 16.3 %, 151 regs -> 85 % of peak; distinct from the hd=128 price-out | M | `tools/roofline/PERF_LOG.md:175 119.5`, `tools/roofline/inventory/KERNELS.md:67 13.8 %`, `docs/archive/roadmap_ledger_2026_09_28.md:159 deeper in-CTA FA2 pipelining`, #2442 |
| 71 | Gemma-4 hd=512 decode attention runs on the F16 multitok path | multitok takes head_dim 64/128/256 only, hd=512 runs the split-K pipeline kernel: 10.2 % of nvfp4-gemma4-26b tg128 at 8k (inventory 679866b6, untracked) -> hd=512 multitok variant | M | `src/compute/attention_paged_f16_multitok.cu:369 head_dim != 64 && head_dim != 128 && head_dim != 256`, #2467 |
| 72 | uncached Q8_0 GEMMs skip the per-call dequant | uncached fallback has an IMMA in-place arm for Q4_K only, Q8_0 dequants to FP16 + cuBLAS: `dequant_q8_0_kernel` 5.8 % of q4k-q36-35b pp512 (inventory 679866b6, untracked) -> Q8_0 IMMA arm; distinct from row 27 (cached route) | S | `src/exec/executor_gemm_dispatch.cpp:69 Q4_K with no FP16 cache`, `src/exec/executor_gemm_dispatch.cpp:146 dequant to FP16 then cuBLAS`, #2468 |
| 73 | MoE IMMA prefill quantizes the shared gate/up input once | `imma_quantize_act` runs per GEMM call, gate and up pass the same input: `quantize_act_fast_kernel` 10.9 % of q4k-q3-30b pp4096 (inventory 679866b6, untracked) -> one quantize, reused | S | `src/compute/mmq_q8_imma.cu:341 imma_quantize_act(x_f16, act_rows, K, stream)`, `src/exec/executor_forward_moe_batch.cu:596 gathered_base`, #2469 |
| 74 | dense NVFP4 prompts skip the separate SwiGLU and quantize | producer-side swiglu+quant only for 2 <= M <= 32; above it `swiglu_fp16_kernel` 6.8 % + NVFP4 quantize 4.0 % of nvfp4-14b pp4096 (inventory 679866b6, untracked) -> fused producer for M > 32 | M | `src/exec/executor_gemm_smallm.cpp:113 M > 32`, #2470 |
| 75 | split-K decode attention needs no second reduce launch | `paged_attention_reduce_kernel` launched after every split-K decode: 6.3 % of nvfp4-q3-30b tg128 at 8k (inventory 679866b6, untracked) -> reduce in the last finishing CTA | M | `src/compute/attention_paged.cu:1430 paged_attention_reduce_kernel<<<grid2`, #2471 |
| 79 | a new arch sets profile traits instead of executor branches | 70 `is_gemma4` / `is_gpt_oss` sites in 21 files (computed: rg), arch #17 (#2057) touched 70 files (computed: git show --stat) -> capability traits; partly reopens SETTLED S-1, disprove its anchor first | M | `src/model/model_profile.h:31 is_gemma4`, `docs/audit/SETTLED.md:52 S-1`, #2458 |
| 82 | a new KV dtype is one descriptor entry | `MXFP4_KV` in 52 sites / 18 files, NVFP4-or-MXFP4_KV predicate in 13 single-line copies (computed: rg) -> descriptor (bits, group, scale plane) read by allocator, kernels, planner | S | `src/memory/kv_cache.cu:32 QType::MXFP4_KV`, `src/runtime/vram_budget.cpp:620 QType::MXFP4_KV`, #2460 |
| 84 | the CLI and the server sample a model the same way by default | CLI takes the arch preset (Qwen3 0.6 / top_k 20), server hard-codes 0.7 / 40 -> one ImpConfig mapper, preset in both; changes server defaults for clients that omit fields | S | `src/model/model.cpp:184 0.6f, 0.95f, 20`, `tools/imp-cli/main.cpp:170 get_sampling_defaults`, `tools/imp-server/handlers_chat_params.cpp:130 0.7f`, `tools/imp-server/handlers_chat_params.cpp:138 40`, #2462 |
| 85 | a new vision family declares its prompt layout once | vision token literals in 39 sites / 13 files outside the placeholder module (computed: rg) -> per-family layout registry | S | `src/model/image_placeholders.h:22 One video's prompt layout`, #2463 |
| 91 | the allocator headroom is a measured number | constant 5 % of total, never measured: 1630 MiB on 32 GB = 5796 NVFP4 KV blocks on Qwen3.8-27B (computed: 1630 MiB / 288 KiB) -> headroom from a measured margin | S | `src/memory/vram_query.h:77 kAllocatorHeadroomPct = 5`, `src/model/mtp_head.h:266 1630 MiB on 32 GB`, #2482 |
| 92 | the quantized-KV scale plane grows with the KV pool | data pool grows to the ceiling, the scale plane is allocated at the ceiling up front: 388.3 MiB on Qwen3.8-27B (computed: 16 layers x 12425 blocks x 2 x 1024 B) -> scale plane grows with the pool | S | `src/memory/kv_cache.cu:81 max_blocks_ * 2 * scale_block_bytes_`, `docs/plans/2026-10-01-recurrent-state-paging-measured.md:38 12425 blocks`, #2483 |
| 94 | untied models keep the token embedding in host memory | `tok_emb` always uploaded: 2425 MiB on Qwen3.8-27B (248320 x 5120 x 2 B) -> host-resident table, per-step gather; tied models keep it; overlaps row 103 | M | `src/model/weight_upload.cpp:423 upload_unquantized_weight(tok_emb`, `docs/plans/2026-08-24-qwen38-port.md:212 248320 × 5120 × 2 B`, #2484 |
| 95 | concurrent hybrid sessions keep recurrent snapshots on device | device snapshot budget fixed 256 MiB = 3 slots of 79.5 MiB on Qwen3.8-27B, evictions go to the 2048 MiB host tier -> budget sized from the stream count | S | `src/runtime/config.h:309 recurrent_snapshot_mb = 256`, `docs/plans/2026-09-04-lever-ledger-detail.md:67 79.5 MiB each`, #2485 |
| 98 | hybrid exports store recurrent projections in MXFP8 | GDN projections NVFP4 or BF16 (`--keep-gdn-proj`), Mamba mixers have no keep flag; Qwen3.8-27B BF16 10605 vs MXFP8 5468 MiB (computed: 48 GDN layers x 115.8 M elem) -> MXFP8/FP8 on disk + Mamba keep flag | M | `tools/imp-quantize/tensor_policy.cpp:54 .linear_attn.`, `docs/plans/2026-09-12-factored-verify-spare.md:36 48 value heads`, #2475 |
| 99 | calibrated MoE exports calibrate the experts | experts stay round-to-nearest; Gemma-4-26B imp-quantize 26.99 vs reference export 26.06 PPL, +3.6 % (computed) -> per-expert AWQ groups | M | `tools/imp-quantize/awq_plan.cpp:442 MoE experts NOT calibrated`, `docs/archive/quantization_awq_findings.md:292 26.99`, #2476 |
| 100 | `--calib` covers the shipped MoE and hybrid families | norm-convention table knows 9 model_types (computed: 4 plain + 5 unit-offset), gemma4, gpt_oss, nemotron_h, deepseek, mistral4, glm refused -> conventions for those | M | `tools/imp-quantize/awq_sites.cpp:35 kPlain[]`, `tools/imp-quantize/awq_sites.cpp:40 kUnitOffset[]`, `tools/imp-quantize/awq_plan.cpp:377 --calib does not support`, #2477 |
| 103 | exports can store the token embedding as FP8 rows | embeddings always excluded from quantization: 2425 MiB on Qwen3.8-27B -> opt-in FP8 per-row embedding, PPL gate per family; overlaps row 94 | M | `src/model/nvfp4_module_policy.h:174 embed_tokens`, `docs/plans/2026-08-24-qwen38-port.md:208 2425.00 MiB`, #2478 |
| 107 | stacked-expert checkpoints beyond gpt-oss and Gemma-4 quantize | 2 stacked layouts (computed: table rows), any other 3-D expert stack such as Llama-4 is refused -> shape-verified layout per family | M | `tools/imp-quantize/expert_destack.cpp:32 stacked_expert_layout`, `tools/imp-quantize/main.cpp:201 refusing:`, #2472 |
| 108 | scalar-scale FP8 sources quantize | E4M3 weights without `weight_scale_inv` copy through unquantized, block-scaled FP8 only -> per-tensor-scale FP8 widened and quantized | S | `tools/imp-quantize/main.cpp:596 scalar-scale FP8 export`, `docs/quantization.md:112 Block-scaled FP8 sources`, #2473 |
| 109 | default calib groups at n_rep 3-4 rest on a measurement | threshold measured at n_rep 5; n_rep 3-4 (Qwen3-8B, Qwen3-4B, Phi-4) unmeasured, keeps all groups -> PPL A/B ABCDEG vs BDEG on the three | S | `docs/quantization.md:203 n_rep 3-4`, #2474 |

## Later

| # | outcome | now -> target | size | ref |
|---|---|---|---|---|
| 42 | short streams keep their ITL during an ingest on hybrids | mixed prefill+decode step refuted, -4.6 % aggregate, ITL max 126-128 -> 181-184 ms -> reopens with a head-dim-256 prefill kernel | L | `docs/archive/roadmap_ledger_2026_09_28.md:35 181-184`, no issue |
| 43 | stability under long load is measured | largest driven load 10 concurrent requests, no soak -> soak test | M | `docs/LIMITATIONS.md:45 No soak`, no issue |
| 44 | server-side per-request cost is gated | perf gate benches `imp-cli` only -> server-side benchmark harness | M | `docs/LIMITATIONS.md:36 Perf gate benches`, no issue |
| 45 | quantisation drift is caught before release | no KL or PPL-drift gate vs a reference forward -> drift gate | M | `docs/LIMITATIONS.md:44 KL divergence`, no issue |
| 46 | JSON and tool turns get speculation | drafter choice has no per-request target, n-gram acceptance 11.6 % -> drafter per request class | M | `docs/archive/roadmap_ledger_2026_09_28.md:32 11.6`, no issue |
| 47 | MoE models past 32 GB decode faster | host offload 20.99 tok/s at the 15 % default expert cache budget -> faster offload | L | `docs/archive/roadmap_ledger_2026_09_28.md:322 20.99`, no issue |
| 48 | more streams fit by evicting KV | KV eviction shelved (H2O memo) -> TriAttention eviction; reopens the memo, needs accuracy proof | L | [TRT-LLM #16957](https://github.com/NVIDIA/TensorRT-LLM/pull/16957), `docs/archive/README.md:131 k5_h2o_eviction`, no issue |
| 49 | dense targets without MTP get a block drafter | DSpark measured -42 % on Nemotron -> DFlash/DSpark on dense targets without MTP only; measure verify cost first | L | [llama.cpp #25173](https://github.com/ggml-org/llama.cpp/pull/25173), `docs/DESIGN_DECISIONS.md:21 -42 %`, no issue |
| 50 | video requests reach first token sooner | no video token pruning -> EVS-style pruning; upstream TTFT -33 % | M | [vLLM #48912](https://github.com/vllm-project/vllm/pull/48912), no issue |
| 51 | Kimi-Linear runs on imp | KDA kernel missing | L | [HF](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct), no issue |
| 52 | Ling-3.0 runs on imp | KDA kernel missing | L | [HF](https://huggingface.co/inclusionAI/Ling-3.0-tiny), no issue |
| 53 | Olmo-3.1 runs on imp | post-norm and SWA missing | S | [HF](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct), no issue |
| 54 | GPU tests run in CI | no GPU runner -> owner decision: self-hosted runner runs fork code | M | owner decision, no issue |
| 55 | users get a binary without Docker | releases ship the Docker image only -> release binaries | M | [`release-docker.yml`](../.github/workflows/release-docker.yml), no issue |
| 56 | GGUF MoE decode spends less host launch time | host graph launch -> device graph launch for GGUF MoE | L | `src/runtime/engine.h:1026 graph launch when possible`, no issue |
| 65 | Q8_0 prompts get more of the INT8 tensor peak | pp512 `mmq_imma_kernel` tensor pipe 25.1 / 23.3 / 25.7 % vs 969.3 TOPS -> re-measure first, counters predate the +11.4 % change; related row 27 | M | `tools/roofline/PERF_LOG.md:69 25.1 / 23.3 / 25.7`, `tools/roofline/peaks/PEAKS.md:58 969.3`, `tools/roofline/PERF_LOG.md:84 +11.4 %`, no issue |
| 66 | 32 streams reach first token without a p99 tail | Qwen3-8B-NVFP4 c=32 TTFT p99 238 ms, no ceiling measured -> ceiling measured | M | `docs/PERF.md:116 232 / 238`, no issue |
| 67 | Mamba2 hybrids ingest long prompts faster | `ssm_scan_reg_kernel` 25.8 % of Nemotron pp4096 -> chunked SSD on tensor cores, gives up bit identity | L | `tools/roofline/inventory/KERNELS.md:69 25.8 %`, `tools/roofline/PERF_LOG.md:237 chunked SSD form`, no issue |
| 68 | Flash-Next serves sooner after start | ready-to-serve 97.7 s, pinning step 67.15 s, no ceiling measured -> ceiling measured | M | `docs/QUICKSTART.md:143 97.7 s`, `CHANGELOG.md:69 67.15 s`, no issue |
| 76 | decode kernels overlap their launch tails | `gemv_fp8_e4m3`, `topk_gating`, `paged_attention_reduce`, `rmsnorm_quantize_q8_1` launch without PDL; max shares 24.0 % and 8.0 / 6.3 / 5.5 % (inventory 679866b6, untracked), e2e gain unmeasured -> PDL on the four after an A/B | S | `tools/roofline/inventory/KERNELS.md:52 24.0 %`, `src/compute/gemm_gemv_dtype.cu:557 gemv_fp8_e4m3_kernel<true>`, `src/compute/moe_routing.cu:711 topk_gating_kernel<<<`, no issue |
| 77 | the build drops M=1 GGUF GEMV kernels no cell runs | `mmvq_kernel`, `gemv_q6k_kernel`, `gemv_q8_0_kernel`, `gemv_f16_moe_decode_kernel`: 0 calls in 52 cells (inventory 679866b6, untracked; computed: 13 models x 4 workloads), Q6_K/Q8_0/F16 MoE absent from the matrix -> retire or prove a caller | M | `tools/roofline/inventory/KERNELS.md:15 13 models`, `src/compute/ggml_mmvq.cu:410 mmvq_kernel`, no issue |
| 80 | model-specific block state leaves the executor header | `executor.h` 1254 lines, 64 includers (computed: wc, rg) -> per-family state structs; owner decision: conflicts SETTLED B | M | `docs/audit/SETTLED.md:64 Deliberate specialisation`, no issue |
| 87 | a dead or mistyped config key is caught | 144 of 260 bound keys appear in no test (computed: rg over tests/); the key gate checks the example file only -> binding test per key + expiry field | S | `tools/check_config_keys.py:2 imp.conf.example lists every key`, no issue |
| 89 | per-arch HF config parsing leaves `load_config` | `load_config` CCN 189, 7 per-arch blocks (computed: rg) -> per-arch hooks | M | `tools/complexity_baseline.toml:149 = 189`, `src/model/hf_config_loader.cpp:623 cfg.arch == ModelArch::GEMMA4`, no issue |
| 93 | dense Q8_0 weights live once in VRAM | raw Q8_0 stays beside the IMMA planes and the NVFP4 decode overlay: 2.75 B/elem (computed: 1.0625 raw + 1.125 planes + 0.5625 NVFP4) -> one resident copy per path | L | `src/compute/mmq_q8_imma_scratch.cu:25 qs plane[N][K] s8`, `src/exec/pre_dequant_phase4_tensor_registry.cpp:533 near-zero sources freed today`, no issue |
| 97 | /metrics splits VRAM by allocator tag | the allocator keeps per-tag bytes and logs them; /metrics exports tiers and own bytes only -> per-tag gauge plus an untracked residual series | S | `src/memory/vram_allocator.cpp:143 by_tag`, `tools/imp-server/metrics_memory.cpp:21 imp_memory_reserved_bytes`, no issue |
| 101 | NVFP4 and MXFP4 exports close more of the BF16 gap | best `--calib` export still +8.7 % vs BF16 on Qwen3-1.7B -> GPTQ / MR-GPTQ / rotation after AWQ | L | `docs/quantization.md:186 +8.7%`, no issue |
| 102 | the BF16 gap is known for 14B and larger exports | gap measured on Qwen3-0.6B and Qwen3-1.7B only; a 14B BF16 source does not fit for `--calibrate` -> gap rows for 14B and up | M | `docs/quantization.md:185 Qwen3-0.6B`, `docs/quantization.md:198 BF16 14B does not fit`, no issue |

## Not gaps

Continuous batching, prefix caching, per-request LoRA, embeddings, the three API dialects, `/metrics`, suspend/resume, sampler surface.

## Closed rows and archive

| closed row or section | closed or moved | archive anchor | detail record |
|---|---|---|---|
| Open 1, 2, 4, 7, 13, 14, 15 | 2026-09-16 | [Closed](archive/roadmap_ledger_2026_09_28.md#closed) | [`plans/2026-09-16-open-rows-closed-detail.md`](plans/2026-09-16-open-rows-closed-detail.md) |
| Open 12 | 2026-09-16 | [Open rows closed before the move](archive/roadmap_ledger_2026_09_28.md#open-rows-closed-before-the-move) | `gemm_cublas` pricing in [`plans/2026-09-04-lever-ledger-detail.md`](plans/2026-09-04-lever-ledger-detail.md), closure evidence in the row |
| Open 3 | 2026-10-01 | [Open rows closed after the move](archive/roadmap_ledger_2026_09_28.md#open-rows-closed-after-the-move) | [`plans/2026-08-28-sparse-decode-attention.md`](plans/2026-08-28-sparse-decode-attention.md) |
| Open 5 | 2026-10-01 | [Open rows closed after the move](archive/roadmap_ledger_2026_09_28.md#open-rows-closed-after-the-move) | [`plans/2026-10-01-recurrent-state-paging-measured.md`](plans/2026-10-01-recurrent-state-paging-measured.md) |
| Open 6 | 2026-10-01 | [Open rows closed after the move](archive/roadmap_ledger_2026_09_28.md#open-rows-closed-after-the-move) | [`quantization.md`](quantization.md), #2359 |
| Open 8 | 2026-10-01 | [Open rows closed after the move](archive/roadmap_ledger_2026_09_28.md#open-rows-closed-after-the-move) | owner decision, [`LIMITATIONS.md`](LIMITATIONS.md#model-specific-blockers), #2359 |
| Open 9 | 2026-10-01 | [Open rows closed after the move](archive/roadmap_ledger_2026_09_28.md#open-rows-closed-after-the-move) | [`API_FEATURES.md`](API_FEATURES.md#video), #2363, #2370 |
| Open 10 | 2026-10-01 | [Open rows closed after the move](archive/roadmap_ledger_2026_09_28.md#open-rows-closed-after-the-move) | [`MODELS.md`](MODELS.md#vision), #2375 .. #2389 |
| Open 11 | 2026-10-01 | [Open rows closed after the move](archive/roadmap_ledger_2026_09_28.md#open-rows-closed-after-the-move) | owner decision, AUDIT B84, B36, #2359 |
| The 2026 bar | 2026-09-28 | [The 2026 bar](archive/roadmap_ledger_2026_09_28.md#the-2026-bar-assessed-2026-08-21) | same section |
| Lever ledger | 2026-09-28 | [Lever ledger](archive/roadmap_ledger_2026_09_28.md#lever-ledger) | [`plans/2026-09-04-lever-ledger-detail.md`](plans/2026-09-04-lever-ledger-detail.md) (serving and kernel rows, 08-25 .. 09-04) |
| Batch=1, MTP verify on a GDN hybrid | 2026-09-28 | [Batch=1](archive/roadmap_ledger_2026_09_28.md#batch1) | same section |
| MoE host offload | 2026-09-28 | [MoE host offload](archive/roadmap_ledger_2026_09_28.md#moe-host-offload) | same section |
| First-party NVFP4 quantizer | 2026-09-28 | [NVFP4 quantizer](archive/roadmap_ledger_2026_09_28.md#first-party-nvfp4-quantizer-experimental-calibration-ships) | same section |
| Closed competitive records | 2026-09-28 | [records](archive/roadmap_ledger_2026_09_28.md#closed-competitive-records) | same section |
| Known limitations | 2026-09-28 | [Known limitations](archive/roadmap_ledger_2026_09_28.md#known-limitations) | [`LIMITATIONS.md`](LIMITATIONS.md) |
| Investigated and shelved | 2026-09-28 | [shelved](archive/roadmap_ledger_2026_09_28.md#investigated-and-shelved) | same section |
| Citations rule history (#2231, #2185) | 2026-10-01 | [House rule history](archive/roadmap_ledger_2026_09_28.md#house-rule-history) | same section |
| Standing position (2026-09-04) table | 2026-10-01 | [Standing position rows superseded 2026-10-01](archive/roadmap_ledger_2026_09_28.md#standing-position-rows-superseded-2026-10-01) | same section |
| Everything moved out on 2026-08-31 | 2026-08-31 | - | [`plans/2026-08-31-roadmap-ledger-detail.md`](plans/2026-08-31-roadmap-ledger-detail.md) |

Archive moves keep text verbatim, relative links re-based with `../`.
