# Roadmap

Last reviewed: 2026-10-15

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
| Citations | `scripts/check_doc_citations.py` needs `path:N anchor`: anchor on N = ok, moved = `DRIFT` (exit 0, `--fix`), gone or ambiguous = `DEAD`, bare basename on 2 files = `AMBIGUOUS` (passes), so cite the path; [history](archive/ROADMAP_LEDGER_2026_09_28.md#house-rule-history) |

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
| dense NVFP4 @32 vs vLLM | PARITY OR AHEAD. Qwen3-14B 38-token prompts 3948.9 vs 3817.6 (+3.4%, 2026-09-03); 982-token prompts +0.2/+1.0/-0.6% (was 0.75x before #1953) | 2026-09-08 | [`BENCHMARKS_PRE_V0.44.md`](archive/BENCHMARKS_PRE_V0.44.md) runs 8-10 |
| batch=1 | 99.5 tok/s spec-off (`gdn.m1_fused`) vs the ~112 tok/s roofline (14.5 GB/token at 1628 GB/s resident), was 87.4 = 78% on 2026-08-27; past it only through the MTP verify | 2026-09-10 | `CHANGELOG.md` 0.39.0, [archive](archive/ROADMAP_LEDGER_2026_09_28.md#batch1) |
| raw-speed half of [`GOAL.md`](GOAL.md) | MET: batch=1 decode +13-48% vs llama.cpp on every hero, MoE prefill leads vLLM single-seq, cross-engine PPL parity measured | 2026-07-12 | [`PERF.md`](PERF.md), [`GOAL.md`](GOAL.md) |

Method: both engines on one client (`tools/analysis/vllm_conc_ab.sh`), 3 alternating trials, same checkpoint; PROV rows in [`BENCHMARKS.md`](BENCHMARKS.md#imp-vs-vllm-at-concurrency). Superseded rows: [archive](archive/ROADMAP_LEDGER_2026_09_28.md#standing-position-rows-superseded-2026-10-01).

## Now

No open row; closed rows are indexed below. A new row takes the next free number (122), ranked by what an agent workload notices first.

## Next

| # | outcome | now -> target | size | ref |
|---|---|---|---|---|
| 100 | `--calib` covers the shipped MoE and hybrid families | CLOSED 2026-10-17: glm4_moe_lite (MLA q_lora: groups Q, S) default `ABCDEGKQS` PPL 10.4179 / 13.5082 vs RTN 10.4301 / 13.7470; nemotron_h (Mamba2 G/E, A/C, relu^2 S) default `G` 10.1743 / 12.4981 vs RTN 10.2391 / 12.5052, E +12 % / +16 % stays opt-in; 14 model_types. Out: gpt_oss (owner 2026-10-11: no exact fold, `clamp(up, -7, 7)`), mistral4 (no imp arch, row 35). Table: `docs/plans/2026-10-04-awq-moe-experts.md` | M | `tools/imp-quantize/awq_sites.cpp:140 kPlain[]`, `tools/imp-quantize/awq_plan.cpp:398 --calib does not support`, `src/compute/gpt_oss_glu.cuh:8 gpt_oss_glu_elem`, #2477, #2670 |

## Later

| # | outcome | now -> target | size | ref |
|---|---|---|---|---|
| 42 | short streams keep their ITL during an ingest on hybrids | mixed prefill+decode step refuted, -4.6 % aggregate, ITL max 126-128 -> 181-184 ms -> reopens with a head-dim-256 prefill kernel | L | `docs/archive/ROADMAP_LEDGER_2026_09_28.md:35 181-184`, no issue |
| 46 | JSON and tool turns get speculation | measured 2026-10-05, Qwen3.8-27B-NVFP4 server, one client (auto loads no MTP head at max_batch 32): tool calls already gain from n-gram (213.1 / 226.8 vs 166.2 / 173.9 tok/s with mtp_k=0), JSON-schema turns get nothing (90.7 / 90.9 vs 90.4 / 90.7) and a forced `mtp_k=2` costs them 27 % (65.6 / 67.0); prose flat. The existing jump-ahead (`constrained.jump_ahead`, #844) never runs on GDN hybrids (gate below) and on dense Qwen3-14B loses: unordered required keys leave 1..3-char forced spans, and a one-key schema's long spans exit at 1/3 because the canonical draft (`_the`) misses the model's own split (`_`): 159.6 -> 157.0 tok/s; the model flips between `_` and `_the` against either draft, so forced spans now take the draft's split (`constrained.jump_ahead_trust`, drafts from the request's own earlier split first): 159.9 -> 163.1 tok/s, jump-ahead still opt-in; keys in schema order measured 2026-10-05 on Qwen3-14B-NVFP4, five-key array: spans stay 3 tokens (`,"` `age` `":`), under the 4-token jump minimum, 160.5 vs 160.5 tok/s, and the output changes (624 vs 732 tokens), not shipped -> hybrid rollback via the MTP verify path | M | `tools/analysis/spec_request_class.sh:2 Roadmap row 46`, `src/runtime/engine_graph_decode.cpp:753 ssm_state_`, `src/runtime/jump_draft.cpp:7 jump_draft_from_history`, no issue |
| 47 | MoE models past 32 GB decode faster | NVFP4 done by the device expert cache: Qwen3-30B-A3B-NVFP4 all 48 layers host-resident tg128 317.08 vs 360.54 resident (2026-10-05); GGUF still the host LRU: Qwen3-30B-A3B-Q4_K_M 56.27 vs 339.01 -> device expert cache for GGUF experts | L | `docs/archive/ROADMAP_LEDGER_2026_09_28.md:322 20.99`, no issue |
| 48 | more streams fit by evicting KV | KV eviction shelved (H2O memo) -> TriAttention eviction; reopens the memo, needs accuracy proof | L | [TRT-LLM #16957](https://github.com/NVIDIA/TensorRT-LLM/pull/16957), `docs/archive/README.md:132 k5_h2o_eviction`, no issue |
| 49 | dense targets without MTP get a block drafter | DSpark measured -42 % on Nemotron; verify cost 2026-10-18: Qwen3-14B NVFP4 11.87 ms per 2-token verify vs 5.94 ms decode step (2.0x), Q6_K 14.14 ms per 4.12-token verify vs 5.8 ms (2.4x); widths 8..16 need a forced-width harness (n-gram drafts 1 token there whatever `speculative.k`) -> that harness, then DFlash/DSpark on dense targets without MTP | L | [llama.cpp #25173](https://github.com/ggml-org/llama.cpp/pull/25173), `docs/DESIGN_DECISIONS.md:21 -42 %`, no issue |
| 50 | video requests reach first token sooner | no video token pruning -> EVS-style pruning; upstream TTFT -33 % | M | [vLLM #48912](https://github.com/vllm-project/vllm/pull/48912), no issue |
| 51 | Kimi-Linear runs on imp | KDA kernel missing | L | [HF](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct), no issue |
| 52 | Ling-3.0 runs on imp | KDA kernel missing | L | [HF](https://huggingface.co/inclusionAI/Ling-3.0-tiny), no issue |
| 54 | GPU tests run in CI | no GPU runner -> owner decision: self-hosted runner runs fork code | M | owner decision, no issue |
| 55 | users get a binary without Docker | releases ship the Docker image only -> release binaries | M | [`release-docker.yml`](../.github/workflows/release-docker.yml), no issue |
| 56 | GGUF MoE decode spends less host launch time | host graph launch -> device graph launch for GGUF MoE | L | `src/runtime/engine.h:1104 graph launch when possible`, no issue |
| 65 | Q8_0 prompts get more of the INT8 tensor peak | re-measured 2026-10-18 (Qwen3-8B Q8_0 pp512, ncu, unlocked clocks): q/o and gate-up now run `mmq_imma_q8k128_kernel`, 117.3 / 215.4 us at tensor pipe 30.9 / 31.6 % (was 25.1 / 23.3 / 25.7 % on `mmq_imma_kernel`), k/v `mmq_imma_kernel<32>` 34.8 us at 12.6 % (was 6.4 %); still a third of 969.3 TOPS -> raise the K128 kernel; related row 27 | M | `tools/roofline/PERF_LOG.md:69 25.1 / 23.3 / 25.7`, `tools/roofline/peaks/PEAKS.md:58 969.3`, `tools/roofline/PERF_LOG.md:84 +11.4 %`, no issue |
| 77 | the build drops M=1 GGUF GEMV kernels no cell runs | CLOSED 2026-10-18: decided overlay vs MMVQ on Gemma-4-26B-A4B Q4_K_M: NVFP4 decode overlay 256.48 tg tok/s / PPL 231.17, MMVQ (`diagnostics.no_nvfp4_decode_cache=true`) 116.85 / 223.72, overlay stays the default; none of the four kernels is unreachable without a knob (`mmvq_kernel` = Gemma-4 parity arm, `gemv_q6k` / `gemv_q8_0` behind `gemm.no_dp4a_gemv`, `gemv_f16_moe_decode` = Nemotron MTP experts), so none is retired; the resolver log no longer promises MMVQ for all weight GEMMs | M | `src/runtime/engine_init_resolver.cpp:861 MMVQ for weight GEMMs outside`, no issue |
| 80 | model-specific block state leaves the executor header | `executor.h` 1254 lines, 64 includers (computed: wc, rg) -> per-family state structs; owner decision: conflicts SETTLED B | M | `docs/audit/SETTLED.md:64 Deliberate specialisation`, no issue |
| 101 | NVFP4 and MXFP4 exports close more of the BF16 gap | best `--calib` export still +8.7 % vs BF16 on Qwen3-1.7B; plain GPTQ (`--gptq`, shipped) adds nothing measurable on Qwen3-1.7B, -1.5 % on Qwen3-0.6B -> rotation (MR-GPTQ) or sequential calibration | L | `docs/QUANTIZATION.md:214 +8.7%`, [GPTQ record](plans/2026-10-17-gptq-nvfp4.md) |
| 112 | more long agent sessions run than their KV fits in VRAM | the host spill holds blocks of freed sequences, no attention kernel reads host KV -> cold pages of live sessions in pinned host RAM, sparse decode fetches only its selected pages over PCIe; a host round trip costs 1.05 ms per layer on WDDM (row 111), so the fetch needs the mailbox path | L | `src/memory/kv_cache_manager_spill.cpp:31 before its sequence was freed`, `src/core/config/attention.h:137 sparse_topk_tokens`, no issue |
| 113 | a chat branches at any token without a new prefill | CLOSED 2026-10-17: assistant prefill in all three dialects (`continue_final_message`, trailing assistant turn on `/v1/messages`, #2764); Web UI token loom: click a token in the token view, pick a `top_logprobs` alternative, a new chat sends the prefix with `continue_final_message` and continues on the prefix cache, the source chat stays. `drive.py features`: 4 branch checks, 35 checks, 0 false | M | `tools/imp-server/webui/index.html:2175 function branchAt`, `src/model/chat_template_continue.cpp` |
| 114 | a request steers style or persona without a finetune | CLOSED 2026-10-15: `"control_vector": NAME` or `{name, scale}`, llama.cpp cvector GGUF via `--control-vector NAME=PATH`, h += scale * v_l after every layer the file carries, inside the captured graphs. Qwen3-4B Q8_0, " banana" embedding on layers 18-33, temp 0: scale 0 == base text, scale 4 steers, base after a steered request == base | S | `src/exec/executor_control_vector.cu`, `tests/test_control_vector.cpp` |
| 115 | the Web UI shows at which layer a model settles on a token | CLOSED 2026-10-17: `"logit_lens": N` on chat, the residual after every layer is stashed and read out after sampling by one LM-head pass (rank and logprob of the sampled token, top-N, entropy per layer); Web UI "layer readout" grid. temp 0, last layer rank 1 on every token, gap to the sampler's logprob: Qwen3-4B Q8_0 36 layers 1.9e-6, Qwen3.8-27B NVFP4 64 layers 7.7e-7, gemma-3-12b Q4_K_M 48 layers 6.1e-6, text unchanged | M | `src/exec/executor_logit_lens.cu`, `src/runtime/engine_logit_lens.cpp` |
| 116 | the Web UI shows which concepts fire per token | CLOSED 2026-10-17: `--sae DIR` (Gemma Scope 2 JumpReLU), `"sae_features": N` per token, `"sae_clamp": {id: value}` steers at every position, Web UI "Concepts" panel + "steer" field. gemma-3-12b-it Q4_K_M, layer 24 16k: L0 58.5, err 0.055 (declared at layer 10: L0 2193, err 99.4); clamp 451 at 2000 / 5000 / 10000 rewrites / changes / breaks the answer | M | `src/exec/executor_sae.cu`, `src/runtime/engine_sae.cpp`, `docs/API_CONTROL.md`, no issue |
| 117 | an answer follows its prompt harder than sampling alone allows | CLOSED 2026-10-17: `cfg_scale` (1..10) on chat, a shadow request decodes the conversation without its system message (or with `cfg_negative_prompt`) in the same step, owner row = l_b + s (l_a - l_b) before sampling, the shadow takes the owner token. Qwen3-4B Q8_0 temp 0, system "exactly three words", "favorite food": unguided "Sushi is my favorite.", cfg 1.5 "Sushi, fresh, delicious"; identical shadow prompt: cfg 1.5 == cfg 3 bit for bit | M | `src/runtime/engine_cfg.cpp`, `src/compute/cfg_combine.cu` |
| 118 | a server-side ban holds for multi-token phrases | CLOSED 2026-10-17: dense: `banned_strings` rewinds output + KV to the phrase start and bans its first token there, token 0 resamples from saved prefill logits, the stream holds back the last max-phrase bytes. Qwen3-4B Q8_0 temp 0, "Eiffel Tower" banned: same text up to "iconic", then "Notre-Dame Cathedral"; hybrids done 2026-10-17: recurrent slab after the prefill and every 32 tokens (`kPhraseSnapEvery`), a rewind restores the latest one before the phrase and re-forwards the kept tokens; Qwen3.8-27B NVFP4, Nemotron-3-Nano NVFP4: post-ban token = unbanned rank-2, 3 of 3 each, a no-restore mutant fails 1 of 3; `banned_regex` (PCRE2, leftmost match ending in new text inside the last 256 bytes): Qwen3-4B Q8_0 and Nemotron-3-Nano, `\b1[0-9]{3}\b` on five Paris buildings with years: 5 matches -> 0, post-ban token = rank-2 | M | `src/runtime/engine_phrase_ban.cpp`, `src/runtime/phrase_ban.h`, `src/runtime/phrase_regex.cpp`, no issue |
| 119 | dense models past 32 GB decode at usable speed | `--gpu-layers` streams host layers once per forward; measured 2026-10-18 on Qwen3-14B Q6_K `--gpu-layers 0`: one pass 188.6 ms at width 1 and 189.71 ms at width 64, so verify width is free; n-gram now drafts on through misses there (code rewrite 5.30 -> 6.83 tok/s); multi-candidate verify (width 8, 17 rows, `src/runtime/engine_spec_ngram.cpp:418 const int width = std::min(8`) is the flattened tree -> lift its caps under offload, then a best-first tree drafter (SpecExec) | L | `src/model/layer_host_keep.h:2 --gpu-layers Pass 1`, no issue |
| 120 | Gemma-4 takes audio input | the loader reads `audio_config`, no encoder, input type or tokenizer route exists -> audio encoder plus an `input_audio` content part | L | `src/model/model_config.h:150 imp has no audio path at all`, no issue |
| 121 | the Web UI shows watts and joules per answer | CLOSED 2026-10-15: `/metrics` `imp_gpu_power_watts` + `imp_gpu_energy_joules_total` (NVML, #2727); the Web UI reads the counter around each turn and shows J and W average next to tok/s and TTFT, plus a power sparkline in its Server tab. Real server, Qwen3-4B Q8_0, 512-token answer: 558 J, 340 W avg | S | `tools/imp-server/metrics_power.cpp`, `tools/imp-server/webui/index.html` |

## Not gaps

Continuous batching, prefix caching, per-request LoRA, embeddings, the three API dialects, `/metrics`, suspend/resume, sampler surface.

## Closed rows and archive

| closed row or section | closed or moved | archive anchor | detail record |
|---|---|---|---|
| Now 16-24, 57, 58, 69, 70, 78, 81, 83, 86, 88, 90, 96, 104-106 (all) | 2026-10-15 | [Now](archive/ROADMAP_LEDGER_2026_10_15.md#now-23-rows) | PR numbers in each row |
| Next: 42 of 43 rows (all but 100) | 2026-10-15 | [Next](archive/ROADMAP_LEDGER_2026_10_15.md#next-42-rows) | PR numbers in each row |
| Later: 15 closed or refuted rows | 2026-10-15 | [Later](archive/ROADMAP_LEDGER_2026_10_15.md#later-15-rows) | PR numbers in each row |
| Open 1, 2, 4, 7, 13, 14, 15 | 2026-09-16 | [Closed](archive/ROADMAP_LEDGER_2026_09_28.md#closed) | [`plans/2026-09-16-open-rows-closed-detail.md`](plans/2026-09-16-open-rows-closed-detail.md) |
| Open 12 | 2026-09-16 | [Open rows closed before the move](archive/ROADMAP_LEDGER_2026_09_28.md#open-rows-closed-before-the-move) | `gemm_cublas` pricing in [`plans/2026-09-04-lever-ledger-detail.md`](plans/2026-09-04-lever-ledger-detail.md), closure evidence in the row |
| Open 3 | 2026-10-01 | [Open rows closed after the move](archive/ROADMAP_LEDGER_2026_09_28.md#open-rows-closed-after-the-move) | [`plans/2026-08-28-sparse-decode-attention.md`](plans/2026-08-28-sparse-decode-attention.md) |
| Open 5 | 2026-10-01 | [Open rows closed after the move](archive/ROADMAP_LEDGER_2026_09_28.md#open-rows-closed-after-the-move) | [`plans/2026-10-01-recurrent-state-paging-measured.md`](plans/2026-10-01-recurrent-state-paging-measured.md) |
| Open 6 | 2026-10-01 | [Open rows closed after the move](archive/ROADMAP_LEDGER_2026_09_28.md#open-rows-closed-after-the-move) | [`QUANTIZATION.md`](QUANTIZATION.md), #2359 |
| Open 8 | 2026-10-01 | [Open rows closed after the move](archive/ROADMAP_LEDGER_2026_09_28.md#open-rows-closed-after-the-move) | owner decision, [`LIMITATIONS.md`](LIMITATIONS.md#model-specific-blockers), #2359 |
| Open 9 | 2026-10-01 | [Open rows closed after the move](archive/ROADMAP_LEDGER_2026_09_28.md#open-rows-closed-after-the-move) | [`API_FEATURES.md`](API_FEATURES.md#video), #2363, #2370 |
| Open 10 | 2026-10-01 | [Open rows closed after the move](archive/ROADMAP_LEDGER_2026_09_28.md#open-rows-closed-after-the-move) | [`MODELS.md`](MODELS.md#vision), #2375 .. #2389 |
| Open 11 | 2026-10-01 | [Open rows closed after the move](archive/ROADMAP_LEDGER_2026_09_28.md#open-rows-closed-after-the-move) | owner decision, AUDIT B84, B36, #2359 |
| The 2026 bar | 2026-09-28 | [The 2026 bar](archive/ROADMAP_LEDGER_2026_09_28.md#the-2026-bar-assessed-2026-08-21) | same section |
| Lever ledger | 2026-09-28 | [Lever ledger](archive/ROADMAP_LEDGER_2026_09_28.md#lever-ledger) | [`plans/2026-09-04-lever-ledger-detail.md`](plans/2026-09-04-lever-ledger-detail.md) (serving and kernel rows, 08-25 .. 09-04) |
| Batch=1, MTP verify on a GDN hybrid | 2026-09-28 | [Batch=1](archive/ROADMAP_LEDGER_2026_09_28.md#batch1) | same section |
| MoE host offload | 2026-09-28 | [MoE host offload](archive/ROADMAP_LEDGER_2026_09_28.md#moe-host-offload) | same section |
| First-party NVFP4 quantizer | 2026-09-28 | [NVFP4 quantizer](archive/ROADMAP_LEDGER_2026_09_28.md#first-party-nvfp4-quantizer-experimental-calibration-ships) | same section |
| Closed competitive records | 2026-09-28 | [records](archive/ROADMAP_LEDGER_2026_09_28.md#closed-competitive-records) | same section |
| Known limitations | 2026-09-28 | [Known limitations](archive/ROADMAP_LEDGER_2026_09_28.md#known-limitations) | [`LIMITATIONS.md`](LIMITATIONS.md) |
| Investigated and shelved | 2026-09-28 | [shelved](archive/ROADMAP_LEDGER_2026_09_28.md#investigated-and-shelved) | same section |
| Citations rule history (#2231, #2185) | 2026-10-01 | [House rule history](archive/ROADMAP_LEDGER_2026_09_28.md#house-rule-history) | same section |
| Standing position (2026-09-04) table | 2026-10-01 | [Standing position rows superseded 2026-10-01](archive/ROADMAP_LEDGER_2026_09_28.md#standing-position-rows-superseded-2026-10-01) | same section |
| Everything moved out on 2026-08-31 | 2026-08-31 | - | [`plans/2026-08-31-roadmap-ledger-detail.md`](plans/2026-08-31-roadmap-ledger-detail.md) |

Archive moves keep text verbatim, relative links re-based with `../`.
