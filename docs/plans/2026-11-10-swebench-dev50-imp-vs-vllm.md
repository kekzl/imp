# SWE-bench dev50: imp vs vLLM on the same agent (2026-11-10)

ANSWERED 2026-11-10: the engine does not cap the solve rate. vLLM resolves 32/50, inside imp's
31-33 band over 9 runs; imp finishes the agent phase ~3x faster at twice the concurrency.

## Setup

| | imp | vLLM (control) |
|---|---|---|
| checkpoint | Qwen3.8-27B-NVFP4-vllm | same files |
| engine | `imp-server` (images `imp:m3079`, `imp:mtpspill`, earlier builds) | `vllm/vllm-openai:v0.31.0` |
| server line | `--max-batch 8`, MTP auto, NVFP4 KV | `--kv-cache-dtype fp8 --gpu-memory-utilization 0.90 --max-model-len 163840 --max-num-seqs 8 --enable-prefix-caching --max-num-batched-tokens 8192 --tool-call-parser qwen3_coder --reasoning-parser qwen3`, no MTP |
| agent | familiar, Anthropic `/v1/messages`, 160-call `max_steps`, 6000 s timeout | familiar 3a3c0be, `--context 131072` |
| concurrent agents | 4 (measured in dev50j, dev50k) | 2 |
| set | 50 instances (django, sphinx), `instances.jsonl` of the familiar dev set | same |
| card | RTX 5090 32 GB, WSL2 | same |

## Result

| run | date | engine | resolved | non-zero exits | agent phase |
|---|---|---|---|---|---|
| dev50m | 2026-11-07 | imp | 33 | 0 | - |
| dev50p | 2026-11-07 | imp | 32 | 0 | - |
| dev50a | 2026-11-08 | imp | 31 | 0 | - |
| dev50b | 2026-11-08 | imp | 33 | 0 | - |
| dev50s | 2026-11-08 | imp | 33 | 0 | - |
| dev50h | 2026-11-09 | imp | 32 | 0 | - |
| dev50i | 2026-11-09 | imp | 31 | 0 | - |
| dev50j | 2026-11-09 | imp `m3079` | 32 | 1 | 163 min, 10.67 agent-h |
| dev50k | 2026-11-10 | imp `mtpspill` + `kv_cache.host_spill_mb=16384` | 33 | 0 | 143 min, 9.18 agent-h |
| dev50v | 2026-11-10 | vLLM 0.31.0 | 32 | 1 (sphinx-8638, 124 at 5676 s) | 475 min, 15.46 agent-h |

Per instance over the 9 imp runs: 27 always resolved, 12 never, 11 flaky. vLLM's differences sit in
the flaky band: gained sphinx-7985 (imp 0/9) and sphinx-10435 (1/9), lost sphinx-10466 (9/9) and
sphinx-9281 (7/9). Remaining losses are agent-side (160-call `max_steps` exploration, a patch in the
wrong file), with 0 tool-call faults in dev50k.

## vLLM line: why 0.90, no MTP, 2 agents

| try | line | outcome |
|---|---|---|
| BF16 KV, 131072 | default dtype | refused: 8.33 GiB KV needed, 8.07 GiB free |
| 0.95 | fp8 KV | refused: 30.2 of 31.82 GiB free < 30.23 desired |
| 1 | 0.93 + MTP k=2, 4 agents | 32056 of 32607 MiB used, WDDM paged to host RAM (decode 0.3-47 tok/s), 16/50 in 4 h 37, 6 timeouts; discarded |
| 2 | 0.90, no MTP, 163840, 2 agents | KV 216977 tokens; memory max 30608 MiB in the 3-chat probe, 31518 MiB read during dev50v at ~110 tok/s aggregate (resident) |

- `/v1/messages` in vLLM 0.31.0 does not clamp `max_tokens`: prompt + 32768 > `max_model_len` is a 400, hence 163840 with familiar `--context 131072`.
- Hybrid KV in vLLM ("Mamba cache mode align", 1568-token attention blocks) costs ~1.37x per token: 2 x 52k chats fill 65.8 % of the pool.
- 3 concurrent 50k chats: 4 preemptions, 17.5-22.1 tok/s per chat; 2 chats: 0 preemptions, 118.2 tok/s aggregate. The control ran 2 agents.

## Files

`~/.cache/familiar-swebench/dev50{a,b,h,i,j,k,m,p,s,v}/` (`score.log`, `runs.tsv`, `familiar-<run>.<run>.json`
`resolved_ids`); the discarded try 1 in `dev50v-paged/`. Concurrency probe: `vllm_conc.py <url> <n> <reps> <max_tokens>`
(N unique long chats on `/v1/messages`, streaming, per-chat TTFT and tok/s), not in the tree.

## ROADMAP CLOSED (2026-11-10)

Row 124: ANSWERED, solve rate is model- and agent-bound on this set; imp's lever is wall time.
