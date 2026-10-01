# Roadmap

Last reviewed: 2026-10-01

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
| 17 | long-prompt ingest finishes sooner with no flag | sparse prefill opt-in, Qwen3.8-27B pp77824 6835.55 -> 10529.05 tok/s (1.54x), NIAH 10/10, PPL +0.26 % -> default on; flips only after NIAH + PPL on every hero model (GPU gate) | S | `CHANGELOG.md:18 10529.05`, #2406 |
| 18 | an agent's next turn keeps its prefix hit while other sessions fill the KV pool | prefix eviction is session-blind -> request `session_id` pins that session's prefix blocks | S | [SGLang #29173](https://github.com/sgl-project/sglang/pull/29173), [TRT-LLM #16115](https://github.com/NVIDIA/TensorRT-LLM/pull/16115), #2407 |
| 19 | overload is rejected by queued prompt tokens, not only by request count | admission gate is `max_concurrent` requests, 429 when full -> plus a `max_queued_tokens` bound | S | `tools/imp-server/handlers.h:299 max_concurrent`, `tools/imp-server/main.cpp:311 429`, [vLLM #49445](https://github.com/vllm-project/vllm/pull/49445), #2408 |
| 20 | parallel agents sharing one system prompt hit the hybrid prefix cache | recurrent snapshots at block boundaries -> snapshot at the branch point; upstream hit rate 43.8 -> 60.8 % | M | [vLLM #37898](https://github.com/vllm-project/vllm/pull/37898), [SGLang #34565](https://github.com/sgl-project/sglang/pull/34565), #2409 |
| 21 | Qwen3-Coder-Next runs on imp | arch falls back to GENERIC -> registry `qwen3_next` + MoE host offload | S | [HF](https://huggingface.co/Qwen/Qwen3-Coder-Next), `src/model/model.cpp:375 loading as GENERIC`, #2410 |
| 22 | Devstral-Small-2 stays correct on long contexts | `Mistral3ForConditionalGeneration` maps to MISTRAL, `llama_4_scaling_beta` unread -> scaling applied; wrong long-context output suspected from code, not run | S | [HF](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512), `src/model/model.cpp:357 Mistral3ForConditionalGeneration`, #2411 |
| 23 | Granite 4.2 runs on imp | arch falls back to GENERIC -> own arch with attention and residual multipliers | S | [HF](https://huggingface.co/ibm-granite/granite-4.2-30b), `src/model/model.cpp:375 loading as GENERIC`, #2412 |
| 24 | toolkit fixes reach the image, grouped NVFP4 cuBLASLt re-probed | CUDA 13.4.1 -> 13.4.2; `cublasLtMatmulGrouped` NVFP4 returns zero algos on sm_120 -> probe result recorded | S | `Dockerfile:25 13.4.1-devel`, `docs/internals/SM120.md:113 cublasLtMatmulGrouped`, [13.4.2](https://hub.docker.com/r/nvidia/cuda/tags?name=13.4.2), #2413 |

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
| 33 | GLM-4.7-Flash runs with tool calls | arch falls back to GENERIC -> MLA head-dim variant + `arg_key` tool format | M | [HF](https://huggingface.co/zai-org/GLM-4.7-Flash), `src/model/model.cpp:375 loading as GENERIC`, #2422 |
| 34 | North-Mini-Code runs with tool calls | `CohereForCausalLM` maps to LLAMA -> cohere2_moe, parallel block, SWA, Cohere tool format | M | [HF](https://huggingface.co/CohereLabs/North-Mini-Code-1.0), `src/model/model.cpp:363 CohereForCausalLM`, #2423 |
| 35 | Mistral-Small-4 runs on imp | arch falls back to GENERIC -> mistral4 MLA + MoE, host offload | M | [HF](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603), `src/model/model.cpp:375 loading as GENERIC`, #2424 |
| 36 | Nemotron-3-Super NVFP4 runs on imp | only `NemotronHForCausalLM` mapped -> LatentMoE, host offload | M | [HF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4), `src/model/model.cpp:345 NemotronHForCausalLM`, #2425 |
| 37 | LFM2 runs on imp | arch falls back to GENERIC -> short-conv block | M | [HF](https://huggingface.co/LiquidAI/LFM2-24B-A2B), `src/model/model.cpp:375 loading as GENERIC`, #2426 |
| 38 | side-image base drift gets a Dependabot PR | Docker ecosystem covers `/` and `/tools/roofline` only, no compose -> tools/ Dockerfiles, compose files, digest pins | S | `.github/dependabot.yml:19 docker`, #2427 |
| 39 | users can verify the released image | no SBOM or attestation in the release workflow -> SBOM + signed provenance per tag | S | [`release-docker.yml`](../.github/workflows/release-docker.yml), #2428 |
| 40 | vision agents can pass https image URLs | `CPPHTTPLIB_OPENSSL_SUPPORT` never defined, https fails -> httplib built with OpenSSL | M | `tools/imp-server/image_fetch.cpp:249 needs an imp built with OpenSSL`, #2429 |
| 41 | build image carries current CMake fixes | CMake 4.3.1 -> 4.3.5 or 4.4.3 | S | `Dockerfile:38 cmake-4.3.1`, [v4.3.5](https://github.com/Kitware/CMake/releases/tag/v4.3.5), [v4.4.3](https://github.com/Kitware/CMake/releases/tag/v4.4.3), #2430 |

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
