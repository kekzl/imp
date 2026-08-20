# Roadmap

Single-author, single-GPU experiment -- "roadmap" means "current focus," not "schedule." Shipped work lives in [`CHANGELOG.md`](../CHANGELOG.md); competitive numbers live in [`docs/BENCHMARKS.md`](BENCHMARKS.md).

## Direction: local inference for AI agents

The goal is making imp the fastest local engine for AI agent workloads on consumer Blackwell. Agents generate far more tokens per session (20k-100k+), accumulate context fast, and often run in parallel. This demands long context, concurrent request handling, and high decode throughput.

### Foundations (shipped 2026-05)

- **Long context** (#453) -- chunked-prefill FMHA (`q_offset`), S-matrix 1024→256 MiB, auto `fmha_prefill_threshold`. Context ceiling ~4-6k → 32k+.
- **Concurrent requests** (#454) -- multi-request decode batching (`runtime.max_batch_size`).
- **KV streaming** (#455) -- StreamingLLM auto-enables when the KV cache runs full: sink tokens + sliding window, agent sessions effectively unlimited.

## Current focus: operational robustness for agent workloads

The engine is past the raw-speed land-grab; current work is making it boringly reliable to *operate* under agent load:

- **Fast (re)starts** -- on-disk warm weight cache (cold boots skip weight conversion, #956) and suspend-to-RAM (`/admin/suspend`/`resume`: free the GPU in seconds, resume without re-reading weights, #954).
- **Determinism as a product property** -- greedy request-order independence (decode-graph pool pre-armed in warmup, `runtime.warmup` default-on, #957); see [`determinism.md`](determinism.md).
- **Model-support debt burn-down** -- last hard crash (gemma-3-12b GGUF decode IMA) fixed in #959; remaining blockers under "Known limitations".
- **MLA family expansion** -- DeepSeek-V2-Lite is validated (#802/#803 latent-KV decode, opt-in); DeepSeek-V3 / GLM / Kimi / Ling reuse the same path once weights are staged locally.

## Open gaps to the mission (assessed 2026-07-26)

A ranked audit of what still separates imp from "best agentic engine on a 5090". This is a gap list, not a schedule -- nothing here is committed. The raw-speed half of [`GOAL.md`](GOAL.md) is met (batch=1 decode leads llama.cpp by +13-48% on every hero in the 2026-07-12 re-sweep, MoE prefill leads vLLM single-seq, cross-engine PPL parity measured). Every item below sits on the *agentic* half of the mission.

**Status as of 2026-07-26** (one line each; detail in the entries below):

| # | Gap | State |
|---|---|---|
| 1 | First-party NVFP4 quantizer | **partial** — `imp-quantize` converts, the output runs, and **AWQ calibration ships** (`--calib`): Qwen3-0.6B +25.1% → +18.3% PPL vs BF16. Modelopt head-to-head done (imp ahead on one model). 3-D stacked experts are **refused** rather than mangled, and supporting them is **not** the cheap job it looked. **A model too large to run can now be calibrated off a quantized twin — and that showed `--calib` HURTS at 14B** (both measured 2026-08-01, below) |
| 2 | Vision beyond Gemma | **largely closed** — Qwen3-VL runs end to end (#1163-#1180): dynamic resolution, DeepStack, three-axis M-RoPE, images over `/v1/chat/completions` and `imp-cli --image`, several images per request. Qwen3.6-35B-A3B-NVFP4 joined on the same tower (#1379 + this PR). What remains: no video, and no VL family with a genuinely different tower |
| 3 | One server, one model | **closed** — `server.model_swap` (#1080) |
| 4 | Constrained decoding is JSON-only | **closed** — `response_format: regex` / `guided_regex` (#1091) and GBNF grammars (#1095) ship; `/v1/rerank` remains a separate item |
| 5 | No speculation **tree** | **half closed, and the other half is no longer negative** (2026-08-19). A trained draft head is not missing: the MTP head pays **+21.3 %** at `speculative.mtp_k=1` since `ea547a53` (see the re-measurement below). No EAGLE/Medusa/multi-candidate **tree** exists, and that part stands. The −7 % that used to close this row was `token_recycling`, re-measured on the current build at **−0.27 %** — neutral, not a loss |
| 6 | Context VRAM-capped, no host spill | **shelved on measurement, not size** (2026-08-01) — the "silent context loss" half closed 2026-07-31. The remaining half, a tier below VRAM, has **no reproducible trigger on this box** and would land on a 6.5x bandwidth cliff; see below |
| 7 | Agentic quality unmeasured vs competitors | **closed** — measured across three model families, published in [`BENCHMARKS.md`](BENCHMARKS.md); vLLM/SGLang deliberately out of scope |
| 8 | No GBNF/EBNF grammar surface | **closed** — GBNF via `response_format: grammar`, llama.cpp's `grammar`, vLLM's `guided_grammar` (#1095) |
| 9 | `/v1/rerank` absent | **closed** — `POST /v1/rerank` scores query+document jointly with a causal-LM cross-encoder (Qwen3-Reranker); validated against llama.cpp on the same GGUF |
| 10 | Agent-harness batteries are imp-internal | **closed** — real aider, Claude Code AND the OpenAI Agents SDK drive imp in `make test-agents-external`, one per dialect (chat-completions / `/v1/messages` / `/v1/responses`) |

Shipped alongside, not from this list: the live web UI at `GET /` (#1078) and the streamed non-ASCII corruption fix that building it exposed.

1. **First-party NVFP4 quantizer — EXPERIMENTAL, but calibration now ships.** `imp-quantize` (2026-07-26) converts a dense BF16/FP16 SafeTensors checkpoint to NVFP4 in-tree, so a model with no published export can reach the NVFP4 path at all. **AWQ-class activation calibration landed 2026-07-31** and closes the half of this entry that was open: `imp-cli --perplexity <corpus> --calibrate <file>` collects the mean |activation| per input channel off the single GEMM dispatch every weight goes through, and `imp-quantize --calib <file>` searches a per-group scale against it. Measured over `ppl_corpus_45k.txt` (13 537 tokens, calibrated on general prose that is *not* the scoring corpus), on both dense Qwen3 sizes staged locally — Qwen3-0.6B: BF16 24.06, round-to-nearest 30.10 (+25.1%), **AWQ 28.48 (+18.3%)**; Qwen3-1.7B (sharded, so the multi-shard write path too): BF16 17.22, round-to-nearest 20.43 (+18.6%), **AWQ 19.21 (+11.5%)**. So 5-6% of the perplexity removed, a quarter to two fifths of the gap to BF16, with `degen_suite.py` reading 45/45 on every checkpoint involved. Detail in [`quantization.md`](quantization.md). **Still experimental** — a Modelopt export beats this, and it should not be used to produce checkpoints anyone relies on. **Five things measurement settled, worth keeping:** (a) 2026-07-26, picking micro-scales by minimizing reconstruction error instead of `absmax` moved PPL 30.10 → 29.88 (0.7%) for ~6× the cost — the micro-block is 16 values, where `absmax` is already near-optimal, and the dominant error is the FP4 grid itself, which no scale improves. That is exactly why the lever had to be *moving* the error (AWQ) rather than picking better scales. (b) 2026-07-31, folding `o_proj`'s scale into `v_proj` writes into the tensor the KV cache stores, which on the default FP8_E4M3 KV looked like it must cost more than it wins — refuted: the FP8-vs-FP16-KV penalty is 0.300 PPL on the calibrated checkpoint and 0.595 on the round-to-nearest one, so the scaled `v_proj` is if anything friendlier to FP8 KV. (c) **A calibration run must be deterministic or the checkpoint is not reproducible** — without `runtime.deterministic_gemm` two runs of the same command differed on 94% of the recorded floats and moved the quantized model's perplexity by 1.6%, enough to swamp per-group attribution; checkpoints built from those files also each flipped one degeneration probe, a different one each time, which the deterministic build does not. `--calibrate` now forces it. (d) 2026-07-31, **"MoE is not supported" was wrong in the dangerous direction.** Checkpoints storing experts the HF-standard way (one 2-D tensor per expert) were never skipped — they were quantized into a checkpoint that loaded and then emitted garbage. Bisection on DeepSeek-V2-Lite showed the experts are fine (4992 of them quantize correctly); what breaks is the **MLA latent projections** and the **MoE router**, both now refused. With those excluded the model quantizes 3.28x and `degen_suite` reads 3 FAIL/32 against the BF16 source's 5 — a strict subset, so quantization adds none. Detail in [`quantization.md`](quantization.md). (e) 2026-07-31, **the head-to-head against a Modelopt export is done, and it went the other way.** `Qwen3-14B-NVFP4` is a real Modelopt export whose untouched tensors are bit-identical to the `Qwen/Qwen3-14B` BF16 source, so both quantizers demonstrably started from the same weights; both quantize the same 280 tensors and exclude `lm_head`. Over `ppl_corpus_45k.txt`, reproduced to four decimals: Modelopt **10.0301**, `imp-quantize` without `--calib` **9.9252** — the uncalibrated in-tree quantizer 1.05% ahead. One model on one corpus, so it is not a claim that imp-quantize is the better quantizer; it does retire the blanket "prefer a published export". Mechanism, half confirmed: the export ships 280 `input_scale` + 40 `k_scale`/`v_scale` tensors (it targets a recipe that quantizes activations and KV too), and imp **verifiably does not apply them** — `input_scale` is read by no GEMM kernel. So imp runs W4A16 against weights rounded for W4A4. Whether that is the whole reason, versus Modelopt's calibration corpus simply sitting further from this one, would need a second corpus to separate. **(g) 2026-08-01: the "cannot calibrate a model that does not fit" blocker was not real, and removing it showed that `--calib` hurts at 14B.** A calibration file is keyed by (layer, tensor kind) and the recording hook sits before the tier switch, so nothing ties it to the checkpoint it came from: the statistics can be collected from any quantization of the same model. Controlled on Qwen3-0.6B, where both routes are possible — stats from imp's own round-to-nearest twin score **28.8868** against **28.4782** from the BF16 source and **30.0979** uncalibrated, so the detour recovers three quarters of the gain. Applied to Qwen3-14B, which cannot be run in BF16 at all, it produced the first `--calib` result at that size and the result is **negative**: **9.9252 round-to-nearest vs 12.6016** (twin: imp's own checkpoint) and **12.2853** (twin: NVIDIA's Modelopt export). Two quantizers sharing no code agree, so the statistics source is not the variable — the calibration is. Ruled out: incomplete plan (both runs scaled 160 groups — 4 per layer across all 40, so none were skipped), degenerate statistics (280 entries, all 40 layers, no zero or non-finite channel), a magnitude effect (the search normalises by the group mean; the floor is relative), the FP8 KV path (fp8 and fp16 score identically to four decimals). What remains is that the search minimises a *local* proxy — per-group weight-reconstruction error, which improved on every group — and a checkpoint whose weights each reconstruct better can still be a worse model. `imp-quantize` now says both things when `--calib` is passed. Detail in [`quantization.md`](quantization.md). **(h) 2026-08-05: why it flips between 1.7B and 14B is ANSWERED — it is the ATTENTION half that fails, and `--calib` at 14B is fixed by dropping it.** `--calib-groups` runs any subset of the planner's four groups, so the damage can be attributed rather than guessed. Against each model's own round-to-nearest baseline, Qwen3-14B (`n_rep=5`): **`BD` (the two FFN groups) −0.1330, the best measured configuration of all and better than round-to-nearest**; `BCD` −0.08; `C` +0.02; `A` +0.65; `ABD` +0.76; `ABCD` **+2.68**. The harm is entirely in the attention pair and mostly in its *interaction*: A × C is **+1.36** and C × ABD is **+1.90, i.e. 71% of ABCD's total damage**, while BD × C is +0.03. On Qwen3-0.6B (`n_rep=2`) the same C × ABD interaction is **+0.05**, forty times smaller, so the effects add and the full set wins (ABCD −1.21). So no group is "broken": C alone is nearly neutral on the 14B, and blaming it — the obvious reading of the GQA tie — would have been wrong. The `n_rep` dependence is C's tie: its statistic must be constant across the query heads sharing a KV head, the tie is a `max`, and it inflates a channel's weight by a median 1.346 at `n_rep=5` versus 1.000 at `n_rep=2` — and that statistic *is* the weight in the error term, so the search optimises the wrong thing faithfully. C and D also run first and rewrite `v_proj`/`up_proj`, members of A and B, which is the path the interaction travels. **Spin-off: group A hurts both models** (+0.28 / +0.65, nothing to do with `n_rep`). Practical consequence: **`--calib-groups BD` on wide-GQA models, default `ABCD` on narrow-GQA ones** — so `--calib-groups` is a production switch, not only a diagnostic. `n_rep` is 8 on most 70B-class checkpoints, so `ABCD` is the thing to avoid there; whether `BD` still pays at that size is untested. **The quantizer has no VRAM ceiling either way** (it uploads one group at a time: ~0.7 GiB for a 14B, ~1.8 GiB for a 70B); only calibration and scoring must run the model. **(i) 2026-08-16: the output is no longer imp-only, and the fused-layer scale was wrong for every reader including imp.** `--format vllm` writes compressed-tensors `nvfp4-pack-quantized`; vLLM 0.27.1 loads it and generates (verified on Qwen3-0.6B and on Qwen3.8-27B, 51.8 → 19.2 GiB). Two things that had to be right first, both silent when wrong: the tensor scale is stored **inverted** between the two layouts, and engines merge q/k/v and gate/up into one linear that keeps **one** scale — so three independently calibrated scales left two matrices dequantized against the third's, with an amax spread up to 3.7× inside a group. Sharing the scale per fused group is also the better quantization (Qwen3-0.6B **30.40 → 29.42**), and on Qwen3.8-27B it is neutral (4.6158 vs 4.6124), which is what a GDN hybrid with 16 attention layers of 64 should look like. **Refuted in passing:** scaling by `absmax/(6×448)` so the FP8 micro-scales fill their range — the convention published exports use, and verifiably effective at what it claims (0 % of micro-scales left subnormal) — measures **31.05**, worse than `absmax/6`. Spin-off fix: imp detected compressed-tensors from `recipe.yaml` alone, so any export published without one was read as Modelopt and inverted; PPL **1.2e47**. Detail in [`quantization.md`](quantization.md).

    **(f) 2026-08-01: 3-D stacked experts are refused, and supporting them is not the cheap job it looked.** The refusal came first, because the old one never fired — the rank check sat behind a `.weight` name test and no real stacked checkpoint names its experts that way, so they were copied through as BF16 while `hf_quant_config.json` announced NVFP4 (#1188). The obvious next step, de-stacking `[ne, N, K]` into the per-expert 2-D layout the loader already reads, was then evaluated against a real `gpt-oss-20b` BF16 checkpoint rather than from the shapes, and it does not work as stated:

    - **The fused layout is not one layout.** `model_config.h:249` documents gpt-oss's `expert_gate_up_bias_fused` as `[ne, 2*d_ff]` **interleaved**, while the Gemma-4 split in `weight_upload.cu` is **concatenated** (rows `[0, n_ff)` gate, `[n_ff, 2*n_ff)` up). A de-stacker has to know which model it is looking at; there is no shape that tells it.
    - **Expert biases have nowhere to go.** gpt-oss carries `experts.{gate_up,down}_proj_bias`, and the forward applies them as a *stacked* `[ne, …]` tensor (`moe_add_expert_bias_sorted`). The generic per-expert branch in `weight_map.cpp` matches `.weight` only, so a de-stacked checkpoint would be a hybrid — per-expert weights, stacked biases — that no loader branch reads.

    So "the loader already handles the target format" is true only for weights, and only for models without expert biases. Doing this properly means a per-model layout descriptor plus per-expert bias support in the loader and the MoE forward, not a 200-line change in the quantizer. Deferred on that basis, not on effort alone. (The same checkpoint immediately paid for itself another way: imp *loaded* it and generated garbage, because nothing refused a MoE model whose experts all failed to map — fixed separately.)

2. **Vision beyond Gemma — Qwen3-VL shipped 2026-07-31 (#1163-#1180); covered by a target since 2026-08-11.**

    *A measurement debt this entry carried silently until now:* `Qwen3VLPipelineTest` — the only test that puts image bytes through the real checkpoint — was runnable from **no make target at all**. `IMP_TEST_MODEL_QWEN3VL` was set in exactly one place, `tools/mutation/run.py`, and pointed at a **GGUF file** while the test calls `load_safetensors()` on a directory, so even there it failed rather than ran. The test also resolves its fixture by a relative path, so it needs the repo mounted as the working directory (the image ships no `tests/`) — the trap [`docs/audit/SETTLED.md`](audit/SETTLED.md) already recorded as "the run that looked greenest tested nothing". Run correctly, all five pass; `make test-vision` now runs them alongside the Gemma goldens. Until that line existed, "Qwen3-VL runs end to end" rested on one manual run from 2026-07-31.
 `Qwen3-VL-4B-Instruct` describes an image end to end, from `imp-cli --image` and from `/v1/chat/completions`. The three pieces the 2026-07-31 re-scope identified all landed, and the re-scope is worth keeping because the *original* one-line assessment ("dynamic resolution + M-RoPE in the encoder, so this is encoder work") was wrong about where two of them lived:

    - **Encoder: dynamic resolution.** Buffers are sized from a patch *budget* (`runtime.vision_max_patches`, default 4096 ≈ 1024x1024) rather than a fixed `image_size`, and a larger image is scaled down to fit rather than refused. A 1795x2397 photo becomes 972 image tokens.
    - **Text model: M-RoPE, which was never encoder work.** `mrope_section` is read from the text config (`hf_config_loader.cpp`) and drives the main forward (`rope.cu`); the hardcoded `[11, 11, 10]` in the MTP head is now only a fallback. Per-token (t,h,w) ids come from `mrope_positions.cpp`. This was the piece the original assessment misfiled as encoder work, and it has now genuinely run with three distinct axes.
    - **DeepStack, which the original assessment missed entirely.** Taps are added after each of the first `n_deepstack` LM layers at the image-token positions (`executor_forward.cu`, `deepstack_inject.cu`) — a change to the text forward pass, not to the encoder.

    Text-only models and text-only prompts are bit-identical to before. Tensor inventory, the DeepStack index spaces and the traps found on the way are in [`plans/2026-07-31-qwen3-vl-vision.md`](plans/2026-07-31-qwen3-vl-vision.md).

    **What is left, now that the hard parts are done, is narrower and mostly plumbing:**

    - ~~**One image per request.**~~ **Closed** — several `image_url` parts in one request are encoded in prompt order into one concatenated embedding, each placeholder expanding to its own picture's token count. The kernels needed no change: they address embeddings by "the k-th image token in the prompt", which is already a global index across pictures. `imp-cli --image` repeats, `/image` stacks before a turn, and the C API gained `imp_add_image{,_from_memory}`. An `image_url` that cannot be read is now a 400 rather than a skipped picture — dropping one would slide every later image onto the wrong placeholder. The mmproj tower still takes one image and says so.
    - **No video.** `temporal_patch_size` is parsed and used, but only as a still-image repeat. Video needs a decoder (only `stb` is vendored), a frame axis on `QwenPatches`, a real temporal axis in M-RoPE (today every image token in a run shares one `t`), a `<|video_pad|>` convention, and a budget that is not per-image.
    - **No VL family with a *different* tower.** There is still no vision arch registry; what exists is an allowlist of `vision_config.model_type` values that name the *same* Qwen3-VL layout (`vision_tower_supported()`). InternVL/Pixtral each need a config parser, a tensor name map, a loader and an encoder forward; `Qwen2VLForConditionalGeneration` / `Qwen2_5_VLForConditionalGeneration` are not even in the text arch map. Qwen2.5-VL would additionally need windowed encoder attention (`window_size` + `fullatt_block_indexes`). InternVL's tiling produces several crops per image; that dependency is now satisfied — the multi-image path above treats N pictures as one concatenated embedding, which is exactly what a tiled image needs.

    **A second *model* on the existing tower cost two gates, not a port (#1379 + this PR).** The entry below used to claim it "needs a checkpoint staged plus an encoder forward of roughly the size of the Qwen3-VL port itself (fifteen PRs)". Both halves were wrong for Qwen3.6-35B-A3B-NVFP4: the checkpoint had been staged all along with a complete 333-tensor tower, and no encoder was needed because it *is* the Qwen3-VL tower. What blocked it was a literal string compare on `model_type`, and an unconditional `model.visual.*` skip in the llm-compressor loader that also made the shard-drop discard `model_visual.safetensors` whole. The estimate was right about the *class* of work (a genuinely new tower) and wrong about this instance — worth remembering as an instance of costing a task by its category rather than by reading the checkpoint.

    So one piece remains a project rather than a task: **video** needs a container/codec dependency in a tree that vendors only `stb`. A genuinely new tower (InternVL/Pixtral) still carries the port-sized estimate above.

3. ~~**One server, one model.**~~ **Closed** — `server.model_swap` (default on) serves a model other than the loaded one by swapping to it: in-flight generations drain first and are never cancelled (the `/admin/suspend` contract), and a failed load restores the previous model rather than leaving the server empty. Those two were exactly why the first-generation auto-swap had been removed. `/v1/models` now lists the rest of the models directory alongside the loaded one, so a harness can see what it may ask for. Still serial by nature: 32 GB fits one model at a time and the requesting call pays one load (the warm weight cache, #956, makes repeats cheap).

4. **Regex-constrained decoding — shipped (GBNF followed as item 8; rerank remains).** `response_format: {"type":"regex"}` (and vLLM's `guided_regex`) constrain the whole reply to a pattern, so a diff header, an ID format, an enum or a small DSL is enforceable without prompting and hoping. Built on the `RegexNfa` already in the tree for JSON-Schema `pattern` — a second engine was written and discarded after measuring identical behaviour. What this needed was the decode-time wrapper: `RegexConstrainer` with the JSON constrainers' `apply_mask` contract, a per-state-set mask cache, EOS gated on an accepting state, and — the part that actually took the time — closing every path that bypasses the mask (the spec-ngram and graph-loop routers, two further `apply_mask` call sites, thinking-default suppression, and pooled-manager state that leaked between requests). The full grammar surface followed as item 8; `/v1/rerank` remains open as item 9.

5. **Speculation has no tree.** No EAGLE / Medusa / multi-candidate path exists in the tree. **"And no trained draft head" is retired (2026-08-19):** the MTP head is one, and it pays +21.3 % at `speculative.mtp_k=1` — see "Re-measured on the fixed build" below. Prompt-lookup n-gram only drafts spans that already appear in the context, so it contributes nothing on free-form reasoning output, and the verify-in-loop experiment was removed after a nine-class sweep found no prompt class where it won (see `CHANGELOG.md`, Unreleased).

    **Re-measured 2026-08-19: the −7 % is gone, and `token_recycling` is now
    neutral.** Same model and prompt class as the 2026-07-27 run below,
    alternating arms, fresh process per arm:

    | `speculative.token_recycling` | tok/s (r1, r2) | mean | drafted | accepted | verifies |
    |---|---|---:|---:|---:|---:|
    | off | 155.95, 155.07 | 155.51 | 48 | 4 (8.3 %) | 3 |
    | on | 154.76, 155.41 | 155.08 | 77 | 9 (11.7 %) | 27 |

    **−0.27 %, against −7.0 % three weeks earlier.** The cause is the same one
    that flipped MTP: `ea547a53` made the verify chunk cheaper, and this path
    runs through the same `greedy_argmax_all` on the same chunk. It is *neutral*,
    not a win — on this prompt class it drafts 77 spans and gets 9 of them, so
    there is almost nothing to win. What changed is that being wrong stopped
    costing anything.

    Note the level, not only the delta: **156 tok/s where that measurement read
    99.37**, because #1102 (below) sits between the two.

    ```
    [PROV: commit=d374df1b date=2026-08-19 hw=RTX5090 model=Qwen3-14B-Q6_K
           quant=Q6_K cuda=13.3 path=imp-server n=3 reasoning prompts x 2
           alternating rounds cmd=`imp-server --think-budget 0 --set
           speculative.token_recycling=true|false --set server.prefix_cache=false`,
           tokens from usage.completion_tokens, counters from /metrics]
    ```

    **Re-measured 2026-07-27 on the build of that day** (the re-evaluation trigger in [`plans/2026-07-22-token-recycling-spec-tree.md`](plans/2026-07-22-token-recycling-spec-tree.md) was "only if the verify ratio drops below ~1.3×"). It has not. Qwen3-14B-Q6_K, three fresh reasoning prompts, best of 3 each, warm clocks, server path — `speculative.token_recycling` **off 99.37 tok/s vs on 92.46 tok/s (−7%)**, accept 1.65–1.79 tok/verify at ~40 ms/verify against a ~10 ms decode step. A verify step costs ~4 decode steps and returns 1.7 tokens; break-even would need an accept near 4, which no published tree result reaches. Route (b) (true tree mask) stays unwarranted on these numbers.

    **And it was not the largest batch=1 lever — that one has since been fixed.** The same measurement session found that decode throughput was set by the CONFIGURED context capacity rather than the live sequence length: the same 280-token request ran at 160 tok/s with `runtime.max_seq_len=1024` and 99 tok/s at the server's default, with the captured decode body growing 44 → 137 kernels (issue #1100, repro `tools/analysis/ctx_capacity_decode_sweep.sh`). The cause was not the capacity itself but the pre-dequant VRAM budget: the NVFP4 decode cache's reservation was subtracted from the shared budget that the cache itself spends from, so every byte the (already-allocated) KV pool took came out of the cache a second time. At full context that left 100 of 280 weight tensors without an NVFP4 overlay, decoding from Q6_K source instead. Fixed in #1102 — the sweep is now flat at 162–163 tok/s across every capacity from 1024 to 40960, a 42-kernel body throughout. Speculation is once again the open batch=1 question, and on the numbers above it is still not worth building.

6. **Context is VRAM-capped with no host spill — but the "silent context loss" half of this entry is stale (measured 2026-07-31).** No KV offload to host RAM and no general layer offload (only the MoE expert cache) — that part stands, and it is the actual gap. What does NOT happen is silent truncation. Measured on Qwen3-8B-Q8_0 with the window pinned to 2048:
    - a prompt past the window is a **typed refusal**, not a truncation: `Prompt exceeds context window (2539 tokens >= 2048 max)`, `invalid_request_error`;
    - a generation that reaches the window stops at exactly `total_tokens = 2048` with **`finish_reason: "length"`** — the OpenAI-standard signal, which every client already handles;
    - StreamingLLM eviction, the thing this entry called silent, is gated twice: it auto-enables only below 10 % free KV blocks AND only on the **FP16** KV path (`engine_scheduler.cpp`), and it emits a WARN naming the sink count and window when it does. The default KV dtype on this class of model is FP8_E4M3 (auto), so the path is not reachable without opting out of it.
    The honest remaining item is therefore capacity, not silence: past the pool there is nowhere to spill to. On 32 GB the auto ceiling is 128K since #1004, and the pool now serves the full requested context up to that (measured: 4k/32k/128k all granted, and holding a 128k pool at 81 MiB free costs 1 % of decode — AUDIT B84). **The client-visible signal shipped 2026-07-31**: when StreamingLLM evicts, the reply carries `usage.prompt_tokens_details.evicted_tokens` (chat-completions and `/v1/responses`' `input_tokens_details`; `usage.imp_evicted_tokens` on `/v1/messages`, whose schema has no slot for "we dropped context" and should not be guessed at). Absent unless eviction actually fired, so its presence *is* the signal. Verified end to end by forcing the path — FP16 KV, `kv_cache.max_blocks=320`, a prompt that starts past the 4132-token threshold — and reading the field back on all three dialects. So what remains of this entry is only the first half: **a tier BELOW VRAM**. Past the pool there is still nowhere to spill.

    **Scoped 2026-08-01, and the result is: do not build it.** Not because it is large — it is roughly 800-1000 LOC across the KV manager, a new host tier and the scheduler — but because three measurements already in this repo say it would not pay:

    - **There is no reproducible trigger on this box.** AUDIT B84 (2026-07-31): Qwen3-8B-Q8_0 asked for 4096 / 32768 / 131072 tokens of context gets *exactly* what it asked for each time, and holding a 128K pool at 81 MiB free costs 1 % of decode (288.02 vs 290.97 tok/s). The pool is never clamped below the request, so a spill tier would sit unexercised behind the 128K auto ceiling.
    - **Where it would fire, it lands on a cliff.** AUDIT B36: resident 1531 GB/s vs host-spilled 237 GB/s — 6.5x. Spilling is not graceful degradation on a memory-bound decode; it is the mechanism behind 55 vs 391 tok/s in #1103.
    - **Each transfer blocks the host ~165 µs** on this WSL2/WDDM box (`executor_elementwise.cu:409`). Decode steps are ~3.5 ms, so one blocking call is 4.7 % of a step *regardless of bytes moved* — and unlike the D2D case there is no kernel to replace it with, because H2D over PCIe cannot be a kernel launch.

    There is also a structural constraint worth writing down, because it is what makes the design expensive rather than merely tedious: **spill cannot be expressed in the block table.** A block id is an `int`, negative values already mean "skipped" to the FP16 paged kernel, and the quantized kernels do not check the sign at all. So a spilled block must be restored *before* `batch.cpp` builds the table — the latency cannot be hidden behind the kernels, only in front of them. Prefetch has exactly one good hook (`prefill_allocate_kv_blocks_`, where the whole block list is known at admission) and one useless one (decode, which knows only the next single block, ~3.5 ms ahead).

    Revisit if any of those change: a model whose pool *is* clamped below the request, a measured H2D figure that makes the cliff shallower, or a workload that needs context past the 128K ceiling.

7. ~~**Agentic quality is unmeasured against competitors.**~~ **Closed.** `tools/analysis/agentic_compare.py` measures the checks an agent harness depends on against any OpenAI-compatible server, and the results are published in [`BENCHMARKS.md`](BENCHMARKS.md): three model families, four budgets, 8-turn sessions. The headline is a defaults difference, not a capability one — at a 200-token budget imp keeps every contract while llama.cpp needs ~800 because it lets a think-capable model reason first; on a non-thinking model llama.cpp's `json_object` and `tool_choice=auto` have gaps of their own. It also earned its keep immediately by finding a REAL imp bug (Llama-3.2 bare-JSON tool calls were dropped, fixed in #1088) that our own batteries never saw, because they run Qwen. **Not covered: vLLM/SGLang** — different weight format and more VRAM than is free while serving; a deliberate scope cut, not an oversight. Extending to them, or to more families, is now a matter of running the harness, not building one.

8. ~~**No GBNF/EBNF grammar surface.**~~ **Closed** — GBNF ships as `response_format: {"type":"grammar"}`, llama.cpp's top-level `grammar`, and vLLM's `guided_grammar`, so a client written against either server works unchanged. The gap was real and structural, not cosmetic: a regex covers the formats agents pin most often, but it is regular *by definition*, and a nested expression language or a bracket-balanced DSL needs a stack that `RegexNfa` cannot have. So this is the one constrainer that could not be another wrapper over the existing engine — it is a nondeterministic pushdown simulator (`src/compute/gbnf_grammar.cpp`, parser split into `gbnf_parser.cpp`) behind the same `apply_mask` contract as the other three. Two things dominated the work, and neither was the grammar compiler. **First, cost:** the first correct cut spent 333 ms building ONE token mask inside a JSON string (the vocabulary walk simulates 151k tokens per new state), which made a plain request appear to hang. Interning parse stacks into integers and memoising each stack's successor set — the target of a transition does not depend on which character took it, only *whether* it matched — brought the same mask to 12 ms. **Second, refusal discipline:** left recursion (direct, indirect through a nullable prefix, and the star-over-nullable spelling), undefined rules, a missing `root` and grammar-bomb repetition bounds are rejected at compile time, because a grammar the simulator cannot expand must not become a hung request or a silently-different language. UTF-8 is assembled across token boundaries with overlong encodings rejected — otherwise a token could spell a forbidden character in a longer form and walk past the mask. The bypass checklist held: the spec-ngram and graph-loop routers, all four `apply_mask` sites (now one shared helper), the thinking default and the pooled-manager state all had to be closed, and a mutation run found one that the tests had not: a recompile cleared the stack arena but not its memoised transitions, so a second grammar on a pooled manager would have been decoded with the first one's.

9. ~~**`/v1/rerank` is absent**~~ **Closed.** `POST /v1/rerank` (and `/rerank`) ships in the Cohere/Jina/vLLM shape. The quality bar held — query and document are scored JOINTLY in one forward, never recomposed from two embeddings — but **the premise of this entry was wrong, and worth recording as such**: it assumed a reranker needs sequence-classification architecture support, because that is what a reranker was (a BERT cross-encoder with a classification head) when the gap was written. The current generation — Qwen3-Reranker, bge-reranker-v2-gemma — are CAUSAL LMs that read the pair and answer a yes/no question, with the relevance score being the softmax over those two logits. That is a cross-encoder by the definition that matters, and imp could already run it: the work was an endpoint plus a prefill-only path that reads two specific logits at the last position, not a second architecture family for a 22M-parameter model. It also lands on the fast stack (NVFP4, FA2, paged KV) and reuses the prefix cache, since every document in one call shares the system+instruct+query prefix. Validated the way item 7 validates things — against llama.cpp serving the *same GGUF*: top-1 agreement on 3/3 queries, median per-document score delta 0.0014, the one ordering difference being a tie between two documents both scored below 0.025. Gate: `make test-rerank`. **Known characteristic:** the first call after load scores ~1e-3 differently from later ones, because a cold prefill and one reusing cached prefix blocks are not the same arithmetic; ordering is unaffected, and the gate asserts the real contract rather than pretending otherwise.

10. ~~**The agent-harness batteries only probe our own server.**~~ **Closed.** `tools/analysis/agent_loop_suite.py` (#1007 stage 1) covers the Anthropic tool loop, the OpenAI loop and `/v1/responses`, but by construction it asserts what imp *thinks* correct looks like. Stage 2 now runs REAL binaries: aider over the OpenAI dialect and **Claude Code over the Anthropic one** (`ANTHROPIC_BASE_URL`), both in `make test-agents-external`, each having to land an actual edit in a throwaway repo — an assertion only a real tool call satisfies. **It paid for itself on the first run.** Claude Code is the demanding client (one request carries a ~20K system prompt, 25 tool definitions, `cache_control`, extended-thinking fields and streaming) and it printed the model's chain of thought instead of doing the edit: with tools present, the STREAMING path was handing reasoning to the user as the answer while the non-streaming path put the same bytes in `reasoning_content`. Nothing in our own batteries saw it, because they compared imp against imp. Fixed, and pinned at three levels — a CPU unit battery on the splitter, a `reasoning-channel` check in the hard gate, and the agent binary itself. **The third leg now closes it: the OpenAI Agents SDK over `/v1/responses`** — the dialect Codex and the SDK speak by default, and the one nothing outside our own probes had ever driven. Same contract as the other two: a real function call has to land an actual edit in a throwaway repo, and the run has to contain a `function_call` item rather than a description of one. **What it turned up is a measurement rather than a bug, and it is worth stating because it looks like one:** imp emits `reasoning` + `function_call` correctly, but *the model* decides whether to call. On Qwen3-8B-Q8_0 with this exact request, a 400-token budget yields the call (232 tokens used) while a 1400-token budget yields a bare prose `message` (511) — given room, it reasons its way past the tool. The leg pins temperature 0 and 400 tokens so it tests the dialect, not the model's appetite for deliberation. OpenHands stays out (docker-in-docker, disproportionate for one smoke task).

Explicitly **not** gaps: continuous batching, prefix caching, per-request LoRA, embeddings, the OpenAI / Anthropic / Responses APIs, `/metrics`, suspend/resume, and the sampler surface (DRY, mirostat, typical_p, logit_bias) all ship today. Multi-GPU remains a non-goal.

### Built-in live UI -- shipped

`imp-server` serves a single-page UI at `GET /` (assessed feasible 2026-07-26, shipped the same day). It renders the SSE stream live and draws one bar per token, so inter-token latency is visible while the answer is written. The page is embedded into the binary at build time (`cmake/embed_webui.cmake`), so there is no asset path to locate at runtime. Source: `tools/imp-server/webui/index.html`.

The assessment that preceded it, kept because it is the reason this cost almost nothing -- no engine or protocol work was required:

- Streaming is the real OpenAI wire format -- `text/event-stream` via `set_chunked_content_provider`, `data: {...}\n\n` chunks, terminating `data: [DONE]` (`handlers_chat_stream.cpp`).
- CORS is wide open for any origin, preflight included -- `Access-Control-Allow-Origin: *` plus an `OPTIONS` catch-all (`main.cpp`), so a page served from anywhere can call the API directly, without a proxy.
- Client disconnect is detected on the token loop (`is_writable`), so closing the tab cancels generation instead of leaving the GPU running.
- `reasoning_content` and tool calls already arrive as separate stream channels, so a collapsible thinking pane and streamed tool calls need no server change.

The only client-side constraint: `EventSource` is GET-only, so the page consumes the stream with `fetch()` + `ReadableStream` -- the standard approach.

Note this is **not** on the `GOAL.md` surface commitment (HTTP server, C API, CLI): it is a convenience with a maintenance tail, not a mission item. It stays deliberately small -- one file, no build step, no dependencies -- and it is not a reason to grow a frontend stack. Anything beyond a thin client belongs in Open WebUI or another external front end.

## Performance work

The batch=1 *competitive campaigns* are closed as programs -- every lever they left open either shipped or was refuted by measurement -- but targeted wins keep landing where new levers appear:

- **FA2 hd=256 prefill default-on** (#930/#932) -- Qwen3.6/Qwen3.5 hybrids, pp4096 +26% over the WMMA path it replaced.
- **FP8 tile attention** (#899/#900) -- FP8-KV decode tiles + GQA batching, long-context decode +14%.
- **FP8 SSM projection sidecar** (#949) -- per-row-scale FP8 for GDN in/out projections; Qwen3.6-35B NVFP4 decode +19% (tg ~320). Extended to GGUF hybrids' Q8_0-kept GDN projections (dequant→FP8 at init): 35B UD-Q4_K_M decode +21% (tg 272, ahead of llama.cpp) -- closed the last decode combo where llama.cpp led.
- **Speculative decoding economics** (#852/#862-#866) -- hybrid-safe verify + MTP drafts; echo-heavy agent workloads up to +156% on 27B.

### The MTP verify on a GDN hybrid, decomposed (2026-08-17)

Prompted by a report that an RTX 5060 Ti gets +85% from MTP-2 on Qwen3.8-27B
while imp measured a loss on a 5090. The comparison is not the one it looks
like: his 47.4 tok/s is *with* MTP and his baseline is 25.7, while imp's ~84 is
*without*. Both engines sit near their memory roofline (84% and 93%); the 4x
bandwidth is the difference. What is real is that the same absolute draft
overhead amortises against his 39 ms step and not against our 11.8 ms one -- a
faster card makes an inefficient drafter relatively more expensive.

Measured per verify, Qwen3.8-27B-NVFP4, k=2, guard off, over 300 verifies:

| phase | before #1455 | after | note |
|---|---:|---:|---|
| chunk forward (captured, 3 rows) | 17.8 ms | 17.8 ms | against an 11.8 ms decode step |
| replay of the accepted prefix | 25.1 ms | 17.2 ms | fires in ~37-48% of verifies |
| argmax + D2H + rollback | 16.4 ms | 10.7 ms | includes the replay |
| decode | 64.3 tok/s | 84.4 tok/s | 83.7 without MTP |

**The verify price is the wrong lever, and one measurement settles it.** The
same server, same model, speculation at its default settings, differing only in
how predictable the continuation is:

| workload | tok/s | draft acceptance | emitted per verify |
|---|---:|---:|---:|
| ordinary prose, no speculation | 89.1 | — | — |
| ordinary prose, n-gram default | 89.4 | 11.2 % | — |
| ordinary prose, MTP k=2 | 87.9 | 58 % | 2.3 |
| **repeat a text verbatim** | **876.5** | **98.3 %** | **36.6** |

**RESOLVED (2026-08-18), and the section below is a record of the broken build.**
The dispute is settled by fixing the engine: two launch defects, a grid over
tokens and a `GemmContext` built without `cur_spec_verify_`, kept every GDN
projection off the small-M batched GEMV. Fixed in `ea547a53`. MTP k=2 now runs
**104.06 tok/s against 86.21 without speculation**, from 75.26 against 84.47,
and kernel time per emitted token falls 11.35 to 8.93 ms. Every "speculation
does not pay" statement below described imp, not speculation, and no longer
describes imp either. Detail and the remaining levers:
[`LIMITATIONS.md`](LIMITATIONS.md).

Not a cross-engine claim: vLLM was measured here for MTP **acceptance** only
(59.7 % against imp's 58-64 %, which is what cleared the drafter), and no
trustworthy throughput comparison between the two engines exists.

Speculation is worth **ten times** on the workload it was built for, and nothing
on the one it was not. Where it pays, acceptance is near-total and the
partial-acceptance repair path never runs, so the verify work below is inert
there; where that work helps, speculation does not pay in the first place.
~~Acceptance is the lever: 11 % to 98 % is 10x, and the entire verify cost is a
few percent. Spend on drafters, not on the repair path.~~

**Retracted 2026-08-18: that sentence does not survive k=1.** It was written
from the k=2 row above, where a poor drafter and an expensive verify are
confounded. Measured at k=1 on a mixed corpus, the MTP head already accepts
**75.0 %** and emits 1.748 per verify, and it still loses: 84.7-85.8 tok/s
against ~88 without speculation. From those numbers a verify costs 20.6 ms
against an 11.36 ms decode step, and the two levers price out as:

| close this gap | result |
|---|---|
| acceptance 75 % → 87 % (the published figure) | 90.8 tok/s, **+3.2 %** |
| verify 20.6 ms → 13.6 ms (1.2x a decode step) | 128.2 tok/s, **+45.7 %** |

So the verify price is worth an order of magnitude more than the acceptance
gap at k=1, and the retracted sentence had it backwards. The two are not
independent either: verify cost grows **5.23 ms per extra draft token**, which
is 46 % of a full decode step, and that slope is what caps chain length, which
caps emitted-per-verify, which caps what any acceptance number can be worth.
Where the slope lives — the forward, or the recurrent state machinery around
it — is unmeasured and is the open question.

### Re-measured on the fixed build (2026-08-19), and the answer is the forward

**Everything above this heading describes the build before `ea547a53`, and the
verdict has flipped.** That fix landed *after* the k=1 numbers were taken, so the
table above prices a verify that no longer exists. Re-run on the current build,
`speculative.ngram=false` so the MTP head is the only drafter, guard off, prefix
cache off, `--think-budget 0` (the 0.5 default disables speculation inside a
think block, and this is a reasoning model), three prompts, two rounds with the
arm order reversed in the second:

| k | chunk rows | tok/s (r1, r2) | mean | vs k=0 | acceptance | emitted/verify | ms/verify |
|---|---:|---|---:|---:|---:|---:|---:|
| 0 | 1 | 86.04, 86.02 | 86.03 | — | — | — | — |
| **1** | 2 | 104.44, 104.19 | **104.31** | **+21.3 %** | 76.0 % | 1.76 | 16.89 |
| 2 | 3 | 103.86, 97.78 | 100.82 | +17.2 % | 59.9 % | 2.20 | 21.82 |
| 3 | 4 | 88.08, 87.34 | 87.71 | +2.0 % | 49.9 % | 2.50 | 28.52 |

**MTP pays now.** The k=0 arm reproduces to 0.02 % across rounds, so the +21 %
is far outside the measurement's own noise.

**It is still not shippable as a default, and the blocker is correctness, not
speed (2026-08-19).** At `mtp_k=1`, 2 of 6 prompts end after ~40 tokens with a
re-statement of the question, deterministically (4 of 4 fresh processes
byte-identical under `deterministic_gemm`), against 0 of 6 for the same prompts
without speculation. Detail, trace and the exact output in
[`LIMITATIONS.md`](LIMITATIONS.md). That also puts a caveat on the +21.3 %: the
arms do not generate the same text.

**Two measurement notes this cost, both worth carrying.** The sweep above sets
`speculative.ngram=false` to isolate the MTP head — a correct control that also
*hid* this defect, because the arm that first showed it was the one run at
defaults. And the degenerate answer is what `deterministic_gemm` pins: the pin
does not remove the state, it stabilises it, so a run that looks reproducible
can be reproducibly wrong.

Three things follow, and the first two say what did *not* cause it:

- **Acceptance did not move.** 76.0 % here against the 75.0 % measured before the
  fix. The drafter is the same drafter; the external-parity finding below still
  stands.
- **The slope survives.** Fitting cost per verify against chunk rows gives
  `4.96 ms + 5.82 ms x rows`, against `5.36 + 6.53` before — an extra row still
  costs **50 %** of an 11.62 ms decode step (was 57 %). So chain length is still
  not a lever: k=3 buys 2 %.
- **What changed is the verify price at two rows**, 20.6 → 16.89 ms. The
  retracted paragraph's own model predicted this: it priced 20.6 → 13.6 ms at
  +45.7 %, and we got half that distance for +21.3 %. The model was right; only
  its input number was stale.

**The open question is answered: the slope lives in the forward, not in the
recurrent state machinery.** nsys on k=1 against k=3 (same prompts, same binary,
`--cuda-graph-trace=node` — without it graph-replayed kernels are not attributed
at all):

| kernel | share of the k=3 − k=1 growth | per launch k=1 → k=3 |
|---|---:|---|
| `gemv_nvfp4_kpar_mb_fp16` (the batched verify GEMM) | **65.1 %** | 26.95 → 32.87 us, **+22.0 %** |
| `gemv_fp16` | 11.5 % | 64.02 → 66.23 us, +3.4 % |
| `gemv_nvfp4_multirow_fp32` | 8.4 % | 440.10 → 442.90 us, +0.6 % |
| `gdn_scan_fused` (the recurrent state) | **2.3 %** | 8.72 → 10.03 us, +15.0 % |

The GDN scan — the candidate the question named — carries 2.3 % of the growth.
Two thirds sit in the batched NVFP4 GEMV, and **it is not purely a count**: the
same kernel costs 22 % more *per launch* at four rows than at two. A batched GEMV
exists to amortise one weight read across M rows; this one amortises well
(doubling the rows costs 22 %, not 100 %) but not freely, and that residual is
the slope.

Reported per launch on purpose: greedy decoding is deterministic *within* a
process but not across them, and the same prompt returned 700 tokens in one run
and 92 in another. Any per-token normalisation across two profiled processes
would be reporting that, not the kernel.

```
[PROV: commit=3c3e9ac9 date=2026-08-19 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=imp-server n=3 prompts x 2 alternating rounds
       cmd=`imp-server --think-budget 0 --set speculative.ngram=false
       --set speculative.mtp_k=0|1|2|3 --set speculative.mtp_econ_min_emit=0
       --set server.prefix_cache=false`; throughput on the `make build` image,
       tokens from usage.completion_tokens, verifies from /metrics. nsys arms on
       the dev build (nsys ships only in imp:toolchain) — sound because both arms
       are the same binary and the claim is relative, not an absolute rate]
```

**Three levers remain in the verify, each worth 4-6 ms of a 28.5 ms verify.** MTP needs the
verify below 26.7 ms to pay at 2.26 emitted per verify, so any one of them
would flip it from break-even to a win:

1. **Kill the remaining replay.** On `matched == 0` the state needed is the one
   after row 0, which the GDN scan already holds in registers -- it commits at a
   device-specified row (`d_real_n`) for padding. Writing a second snapshot at
   row 0 costs one extra 151 MiB slab and removes the forward entirely for
   ~37% of verifies. The commit point in `gdn_scan_fused_kernel` is six lines.
2. **The 4.3 ms of argmax + D2H + rollback** is a host round-trip per verify.
   Deciding acceptance on device would remove it, at the cost of a
   conditional-graph redesign of the accept path.
3. ~~**The chunk forward at 17.8 ms for 3 rows** against 11.8 ms for one.~~
   **Profiled 2026-08-17, and it dissolves the premise of this list.** Two nsys
   runs on Qwen3.8-27B-NVFP4, `speculative.hybrid=true` forced in both (an
   `imp-cli --bench` default pins it off, which silently disables speculation
   outright on a GDN model), k=0 against k=2, both arms verified quiet and the
   k=2 arm asserted to show drafting in its own log before being read:

   | | k=0 (n-gram only) | k=2 (n-gram + MTP) |
   |---|---:|---:|
   | total GPU kernel time | 34.27 s | 38.67 s |
   | `gemv_nvfp4_kpar_mb_fp16` | 8919 ms (26.0 %) | 17045 ms (44.1 %) |
   | per launch | 38.7 us | 38.6 us |

   The per-launch cost is identical, so the chunk is not slower at three rows.
   ~~There are 92 % more launches [per emitted token].~~

   **WITHDRAWN 2026-08-18 by its author.** The per-token normalisation behind
   that claim divided by 16384 tokens, taken from the benchmark's
   `tg 8192 tokens avg 0.00 ms ( 0.00 tok/s) [2 reps]` line. That line is the
   REQUEST, and its `0.00 ms` says the phase produced no timing. At the measured
   ~256 launches per forward, 230400 launches is about 1100 forwards, so the
   arms emitted on the order of a thousand tokens, not 16384. The absolute
   kernel times stand; every per-token figure derived from them does not, and
   neither does the direction of the comparison. Superseded by the four-arm
   table below, which counts tokens from the API responses.

   **The accounting error underneath all three items: a verify replaces a decode
   step only when the draft is accepted.** On rejection it is additional. So the
   cost of speculation is not "the verify price minus what it saves", it is a
   full weight sweep per verify whether or not that verify emits anything. That
   is why the repair path is not where the money is, and why the chain-length
   saturation in [`LIMITATIONS.md`](LIMITATIONS.md) is a consequence rather than
   a separate result.

### The four-arm table (2026-08-18)

Four server arms, `speculative.ngram=false` so the MTP head is the only drafter,
`server.prefix_cache=false`, same three prompts, nsys per arm, **tokens counted
from the API responses**:

| arm | rows in chunk | tokens | verifies | kernel ms/token | emitted/verify | cost/verify |
|---|---:|---:|---:|---:|---:|---:|
| k=0 (no speculation) | 1 | 723 | 0 | 11.39 | n/a | 11.39 ms |
| k=1 | 2 | 729 | 424 | 11.33 | 1.715 | 19.43 ms |
| k=2 | 3 | 721 | 334 | 11.35 | 2.153 | 24.44 ms |
| k=3 | 4 | 768 | 291 | 11.98 | 2.629 | 31.50 ms |

Two facts, neither depending on a fit:

- **Speculation buys no GPU time.** Kernel time per emitted token is flat across
  k=0..2 and worse at k=3, while emitted-per-verify climbs 1.72 → 2.63. Each
  verify costs in proportion to what it emits.
- **Cost per verify is linear in chunk rows**: `5.36 ms + 6.53 ms x rows`, fitted
  on four points against two parameters, residuals -0.50 / +1.01 / -0.52 / +0.01.
  **An extra row costs 6.53 ms, 57 % of a full decode step**, where on a
  bandwidth-bound batch-1 decode it should be near-free.

At k=1 the accepted-prefix repair is structurally unreachable (it is gated on
`0 < matched < K`), so that row is free of repair effects: a 2-row verify costs
19.43 ms against an 11.39 ms decode, i.e. **the second row alone costs 8.04 ms**.

```
[PROV: commit=196a3384 date=2026-08-18 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=imp-server n=3 prompts x 256 greedy tokens per
       arm cmd=`--set speculative.ngram=false --set speculative.mtp_k=0|1|2|3
       --set speculative.mtp_econ_min_emit=0 --set server.prefix_cache=false`,
       nsys -t cuda --cuda-graph-trace=node; kernel time = cuda_gpu_kern_sum
       total, tokens from usage.completion_tokens, verifies from /metrics]
```

### Externally measured, and it closes the drafter question

**vLLM 0.27.1 reaches 59.7 % MTP acceptance (418/700) on this box, this card and
this checkpoint family** (`method: qwen3_next_mtp, num_speculative_tokens: 2`,
resolved architecture `Qwen3_5MTP`), against imp's 58.0-64.1 % at k=2. Parity.
Another engine gets the same acceptance from the same head, so the drafter is not
imp's problem and the published 87 % figure describes a regime nobody has pinned.

### Buried, so they are not re-run

- Six drafter-accuracy hypotheses: draft lm_head precision, quantised head
  weights, a missing `gamma = 1 + W` offset, the hidden-state convention, a RoPE
  defect, an uninitialised MTP KV cache. All measured, all dead; detail in
  [`LIMITATIONS.md`](LIMITATIONS.md).
- **MoE draft head**: this checkpoint's MTP head has no experts and no router,
  only `mlp.gate_proj/up_proj/down_proj`. The per-expert host loop cannot run.
- **An unfused verify chunk**: per-launch cost is flat at 32.5 us across 2 and 3
  rows, so the batched GEMV amortises its weight read as designed.
- **The repair forward as the main cost**: unreachable at k=1, and consistent
  with ~21 % of verifies at k=2.
- **The async conditional-graph loop**, which MTP switches off: worth 1.0 %
  measured (87.28 against 86.44 tok/s), not the 27-45 % assumed. The condition at
  `engine_scheduler.cpp` that trades it for telemetry coverage is still wrong on
  its own terms, but it is a 1 % wrong.

Added 2026-08-18, on the speculative greedy divergence:

- **The recurrent-state mechanism for it**, i.e. "the verify advances the
  recurrent state through the chunk kernels while plain decode advances it
  through the single-token path". Disproven: `--set gdn.chunkwise_scan=false` is
  byte-inert on both arms and the divergence survives at the same offsets. That
  sentence stood in [`LIMITATIONS.md`](LIMITATIONS.md) until this measurement
  and is gone from it now.
- **Five decode-side kernel hypotheses**: GDN chunkwise scan
  (`gdn.chunkwise_scan=false`), fused QK-norm + RoPE
  (`attention.no_qknorm_fused=true`), the NVFP4 `use_multirow` K-partition split
  (patched out), the fused NVFP4 FFN at decode (two-site patch in
  `executor_ffn.cu`), the attention kernel family (mirror flip plus
  `speculative.capture=false`). Each instrument was proven live by its dispatch
  log before the null result was believed; all five leave the speculative output
  where it was. They cannot be the cause for a structural reason: in a
  speculative arm every emitted token comes out of the multi-row verify chunk
  and none out of the single-token decode step, so no decode-side kernel is on
  the emitting path at all. Offsets and the per-hypothesis evidence in
  [`LIMITATIONS.md`](LIMITATIONS.md).
- **Chunk-side kernel choice as the fix.** Not dead, but not a lever either:
  `--set speculative.verify_nvfp4_gemm=false` moves the first difference to
  bytes 58 / 130 / 150 instead of 79 / 332 / 243 and still never reaches the
  non-speculative reference. No chunk-side kernel choice tried so far closes the
  gap.
- **Correction to "Two of three prompts diverge"**, in the *Two findings that
  are not speed* list further down this file: all three prompts diverge, at
  bytes 79 / 332 / 243 with `mtp_k=2`. Two of three was the count on the first
  pass only. The control is unchanged and now holds across eleven
  no-speculation processes.
- **Cross-process reproducibility of a speculative arm at temperature 0.** It
  does not hold, and it is not a kernel hypothesis to chase here: 8 of 9
  processes at `mtp_k=2` agree byte for byte and the ninth does not, and the two
  processes measured at `mtp_k=1` disagree with each other. That is also why two
  fresh processes were seen agreeing in #1457 while #1467's commit body asserted
  they do not: both were looking at the same effect with too few samples to see
  it. Documented for callers in [`LIMITATIONS.md`](LIMITATIONS.md).

### Withdrawn by their author

- "0.72 forwards per emitted token, so speculation moves fewer bytes": the launch
  count fell and the GPU time did not, so counting launches never measured cost.
- The launch-count framing generally, including the 22.3 % CUTLASS share quoted
  from a broken CSV parse (nsys kernel names contain commas inside template
  arguments; the field must be read with a real CSV reader).

Two findings that are not speed:

- **Speculation does not reproduce the non-speculative greedy output on this
  model.** Two of three prompts diverge, first at character 79 of 1026, with
  `deterministic_gemm` on both arms and a stable control. Predates #1455. See
  [`LIMITATIONS.md`](LIMITATIONS.md).
- **The economics guard is right and its constant is too coarse.** 4.0 emitted
  per verify was derived when the verify ran eagerly; the measured break-even is
  now 2.42. It should be derived from `verify_wall_ms / decode_step_ms` rather
  than pinned, which is what the comment beside it already asks for.

**Candidate, not committed: CPU-resident cold experts (no measurement yet).**

A MoE with a small active set touches only a fraction of its weights per token; the rest occupies VRAM without ever being read on that step. The idea is not to *stream* those experts over PCIe — at ~55 GB/s effective that is far too slow once 20B+ is active per token — but to compute the cold ones **on the CPU with AVX-512** while the hot ones stay resident on the GPU. ktransformers demonstrates the shape carries in practice. On this box (9800X3D, DDR5-6000) it is the difference between comfortably running a 30B-A3B and reaching the 80B-120B class — [`GOAL.md`](GOAL.md) now names that as the ambition for MoE — which no amount of kernel work on the GPU side can buy.

The pieces imp would need mostly exist: grouped GEMM, expert routing and paged memory are all in place, and `expert_overhead_pct` already models the on-device/off-device split. So this is engineering, not a breakthrough.

Two things have to be said plainly before anyone starts:

- **It collides with a stated non-goal.** [`GOAL.md`](GOAL.md) says "Not a CPU engine. GPU only. No AVX kernels." Taking this on means amending that line deliberately — the scope decision comes first, the code second.
- **DDR5 bandwidth is a hard ceiling, and whether it lands at usable tok/s is a measurement question, not a matter of belief.** The honest first step is a bandwidth-and-latency budget for one decode step with a realistic hot/cold split, measured on this host, before a line of kernel code is written. If that budget says the CPU half cannot keep up with the GPU half, the idea is dead and the measurement is what kills it.

**That budget was measured 2026-08-10. It does not kill the idea — and it moves the bottleneck somewhere else than this entry assumed.**

*Host bandwidth.* Streaming read, 16 threads, 24 GiB buffer, 512-bit non-temporal loads: **62.5 GB/s**, three runs within 0.2 % (4 threads already reach 62.4 — the bandwidth saturates well before the core count does). That is ~65 % of the DDR5-6000 dual-channel theoretical, which is what a real streaming load gets.

*Routing skew*, the input this entry never had. `diagnostics.moe_expert_hist` records a per-layer expert-activation histogram; `tools/analysis/moe_routing_skew.py` turns it into the coverage curve. Three prompts x 512 tokens, decode-dominated:

| model | experts | top_k | coverage at 40 % resident | vs flat |
|---|---|---|---|---|
| Qwen3-30B-A3B-NVFP4 | 128 | 8 | **84.7 %** | 2.12x |
| gpt-oss-20b-mxfp4 | 32 | 4 | 71.6 % | 1.79x |

So the router is genuinely skewed, and **more so at the higher expert count** — which is the favourable direction, since the 80B-120B targets carry 128 (gpt-oss-120b) to 512 (Qwen3-Next-80B) experts.

*The catch, and it is the reason this is not a green light.* That coverage is **in-sample**: it assumes the resident set is chosen for the workload being served. Cross-validated — resident set picked on prompt 0, applied to the others — the 30B loses 15.2 and 29.5 points against those prompts' own oracles (78.8 % and 58.7 % against 94.0 % and 88.1 %). **The hot expert set is prompt-dependent**, so a static split calibrated once does not transfer, and an adaptive one pays PCIe traffic this budget does not model. Three prompts is a small sample and two of them are topically far apart, so treat the 29.5-point figure as a magnitude, not a constant.

*The transfer term, measured rather than assumed.* The 165 µs figure quoted above for a blocking transfer is what a *host call* costs, not what a completed round trip costs, and using it twice per layer overstates the real thing by ~3.8x. Measured directly on this box at 8 KiB (one token's activations at d_model 4096), medians over 300 reps, idle GPU:

| | µs |
|---|---|
| single D2H + synchronize | 45.4 |
| **D2H + H2D, no kernel between — the cold-expert shape** | **86.2** |
| D2H + kernel + H2D | 126.5 |
| bare kernel launch + synchronize | 34.6 |

The three add up, so the split is clean: a GPU kernel between the transfers costs its own launch, and the cold-expert path has none there — the host is what computes. The cost is also **size-independent** (8 KiB, 64 KiB and 1 MiB all land within noise of each other), which is what makes it latency and not bandwidth. So one MoE layer's round trip is 86 µs and 48 of them are **4.1 ms/token**, not 15.8.

A trap worth recording, because it is what made the old figure look plausible: a small **pageable** H2D returns before the transfer has completed — CUDA stages it — so timing `cudaMemcpy` on pageable memory measures the staging, not the arrival. Pinned memory times *slower* here (45 vs 22 µs) for exactly that reason, not because pinning is worse. Only synchronized numbers are comparable.

*The resulting band*, for a 120B-A5B shape (4.3B active routed params, 0.53 B/param, 48 MoE layers):

| assumption | cold/token | bandwidth | transfer | total | ceiling |
|---|---|---|---|---|---|
| pooled / matched workload | 0.35 GB | 5.6 ms | 4.1 ms | 9.7 ms | **103 tok/s** |
| held-out prompt | 0.94 GB | 15.1 ms | 4.1 ms | 19.2 ms | **52 tok/s** |
| flat, i.e. this entry's own prior | 1.37 GB | 21.9 ms | 4.1 ms | 26.0 ms | 38 tok/s |

**So the binding constraint is the cold-weight stream after all, and it is set by how well the resident set matches the workload** — not by the round trip, which is a fifth of the budget at the pessimistic end. An intermediate revision of this entry claimed the opposite ("the round trip is 74 % of the best case, faster memory buys nothing"); that rested on the 165 µs estimate and is withdrawn by the measurement above.

Three caveats on the band, none of them small: it is measured on an **idle** GPU, so queueing under real decode load can only make the transfer term worse; it assumes the host compute is fully overlapped, which the sequential layer dependency makes non-trivial; and the residency figure is in-sample. Treat 52 tok/s as the honest end and 103 as the ceiling nobody should plan against.

*Can a calibrated resident set survive workload drift?* Measured, since that was the question this entry ended on. Nine prompts on Qwen3-30B-A3B (essay, code, arithmetic, lists, dialogue, explanation, history, science, verse, networking), 512 tokens each; resident set pooled from k of them, coverage evaluated on the held-out rest, 300 random splits per k:

| calibration prompts | held-out coverage | ceiling |
|---|---|---|
| 0 (flat, no calibration) | 40.0 % | 38 tok/s |
| **1** | **67.6 %** | **63 tok/s** |
| 3 | 70.3 % | 67 tok/s |
| 8 | 72.6 % | 71 tok/s |
| oracle, set chosen per prompt | 92.1 % | 142 tok/s |

**The first prompt buys almost all of it, and then the curve goes flat.** Calibration is worth 38 → 63 tok/s; eight times the calibration data adds 8 more. What it does *not* buy is the last ~20 points: the gap to the oracle is genuine per-prompt variation in which experts run hot, and no static set closes it. So the 103 tok/s "matched workload" row above is not an operating point — 63-71 is, and 142 is what an oracle would get.

*Is an adaptive resident set worth its eviction traffic?* Measured with a per-token expert trace (`diagnostics.moe_expert_trace`), and the answer redirects this entry away from its own premise.

**Expert selection has strong temporal locality.** Median reuse distance is **2 tokens** — 45 % of selections repeat at the very next token, 80 % within eight. An LRU at the same 40 % residency therefore needs almost no traffic once warm: over three prompts on Qwen3-30B-A3B, steady-state hit rates are **94.7 % / 90.0 % / 95.3 %**, i.e. only **0.38-0.80 experts per layer per token** are new. The cache warms in ~64 tokens (13.8 % miss over the first 64, 5.3 % thereafter).

That is 4-5x fewer misses than the corpus-calibrated static set achieves, and it needs no calibration at all — the cache converges on the prompt it is actually serving.

**Which makes host-side compute the weaker design.** Two ways to serve a miss, priced for a 120B-A5B shape (~5.9 MB per expert, 48 layers):

| design | per token | cost |
|---|---|---|
| static calibrated split, host computes the cold half | 105 experts, 620 MB from host RAM @ 62.5 GB/s | 9.9 ms + 4.1 ms round trips = **14.0 ms** |
| LRU, stream the missing expert into VRAM, GPU computes | 20-39 experts, 120-228 MB over PCIe @ 25.6 GB/s | **4.7-8.9 ms** |

Streaming also deletes the per-layer host round trip entirely, because the GPU never stops being the thing that computes. Measured PCIe H2D is 25.6 GB/s at a single expert's 6 MiB but 50.6 GB/s at 64 MiB, so batching several promotions is worth more than any host-side kernel would be.

**And imp already has that mechanism**: `ExpertCache::get_or_load()` (`executor_forward_moe_legacy.cu`), with `moe.no_expert_cache` to switch it off. So the work this entry was reaching for is not an AVX kernel and not a `GOAL.md` amendment — it is whether that cache holds this hit rate for a model whose experts do not fit, and whether a promotion can be issued early enough. The latter is the same structural constraint entry 6 records for KV spill: routing for layer N is known only at layer N, so the fetch sits in front of the kernels rather than behind them.

*What a promotion actually costs, and whether it can be hidden.* Measured with `tools/analysis/expert_promotion_overlap.cu`: 48 layers, one 6 MiB expert, a spin kernel standing in for per-layer compute, medians over 20 reps.

| per-layer compute | baseline | promotion on the compute stream | prefetched one layer ahead on a copy stream |
|---|---|---|---|
| 100 µs | 4.80 ms | 15.47 ms (**+10.67**) | 5.45 ms (+0.66) |
| 200 µs | 9.50 ms | 20.02 ms (**+10.53**) | 9.85 ms (+0.36) |
| 400 µs | 18.86 ms | 29.56 ms (**+10.70**) | 19.19 ms (+0.32) |

So copy/compute overlap **does** work on this box: given ≥100 µs of compute per layer, a prefetched promotion is all but free, while the same transfer issued in front of its consumer costs the full ~10.7 ms and hides nothing.

**But the prefetch column is not reachable, and the reason is not engineering.** Issuing a fetch a layer early means knowing layer N+1's routing at layer N, and routing depends on that layer's own attention output. The obvious substitute — prefetch whatever the previous token used — was evaluated on the traces and does not carry: **42-47 % of a token's selections repeat from the token before** (63-68 % from a four-token window, at the price of holding ~19 experts per layer). Worse, it fails precisely where it would be needed: a miss is by definition an expert that was *not* recently used, which is the one case a recency predictor cannot supply.

So promotions land in the first column, at the measured LRU rate rather than one per layer: 0.42 experts/layer/token is **~4.4 ms/token** of unhidden PCIe traffic for a 120B shape. Still ~3x better than the 14.0 ms of the host-compute design, and now the largest single term in it.

Caveats: one model as proxy, three prompts, 512 tokens each; expert size and layer count assumed for the 120B shape rather than measured; and the spin kernel is a stand-in for real per-layer compute.

*Does the existing `ExpertCache` hold up when the experts genuinely do not fit? Measured 2026-08-11, and the answer is yes — but it was never the binding constraint.* `moe.force_host_experts=N` pins the last N MoE layers to host regardless of fit, which makes the whole range reachable on a model that otherwise fits. Qwen3-30B-A3B-Q4_K_M (48 layers, 128 experts, top_k 8, 1.23 MiB per expert matrix), all 48 MoE layers host-resident, `--bench-reps 3`:

| arm | pp512 tok/s | tg256 tok/s | hit rate |
|---|---|---|---|
| experts fully resident (`N=0`) | 10580 | **311.24** | — |
| host-resident, LRU cache | 254.20 | **24.98** | 88.7 % |
| host-resident, staging buffer only | 308.11 | **6.63** | — |

**The cache holds its hit rate and earns its keep: 88.7 % at full offload, ~0.9 new experts/layer/token — the same order as the 0.38-0.80 the traces predicted — and 3.8x the decode of the staging fallback it replaces.** The design this entry reached for is therefore not missing; it is in the tree and it works.

**What the measurement moves is the ceiling.** Full offload still costs 12.5x against resident decode (311.24 → 24.98), and residual PCIe explains only part of it: at 88.7 % the misses move ~4.7 GB/s averaged over the run, well under the 25.6 GB/s this box reaches at a single expert's size. The rest is the dispatch shape — a host-resident layer fails the `on_device` test that selects the grouped GEMM, so it runs the serial per-expert path at 3·top_k = 24 launches per layer per token. **So the next lever is the dispatch, not the cache.**

*Profiled 2026-08-11, and it says which half of "the dispatch" — the host's, not the GPU's.* CUDA graphs are already off under host offload (the `engine.cpp` guard), so an nsys run of this path is the real thing rather than the usual no-graphs overstatement. Qwen3-30B-A3B-Q4_K_M, 48 layers host-resident, `--bench-pp 8 --max-tokens 32`:

| | value |
|---|---|
| `cudaLaunchKernel` | 197 809 calls, **3091 per token**, median 4.58 µs CPU → **~14 ms/token** |
| `cudaMemcpyAsync` | 632 per token, median 8.4 µs → ~5 ms/token |
| `dequant_q4k_kernel` | 47.1 % of GPU kernel time (74 098 launches, 5.07 µs) |
| `gemv_fp16_kernel` | 16.2 % (67 749 launches, 1.91 µs) |

So ~19 ms of a ~40 ms token is the **host issuing work**, and the two kernels that dominate GPU time are a Q4_K expert being materialised as FP16 in VRAM purely to be read once by the GEMV that follows it.

**The obvious lever off that profile was built and is refuted.** Routing Q4_K experts through the fused ggml MMVQ kernel (`ggml_mmvq_q4k`, which the Gemma-4 MoE path already uses) instead of dequant→FP16→GEMV does exactly what it should to the GPU: `dequant_q4k` drops 74 098 → 6334 launches and the expert path's kernel time falls **504.8 → 288.7 ms, −43 %**. End to end it moves **nothing** — 3 paired rounds, signs mixed on both pp and tg. The reason is visible in the same profile: the fused wrapper issues *two* kernels of its own (quantise the activation to Q8_1, then MMVQ), so the launch count is unchanged at 197 824, and on an issue-bound path only the launch count is the currency. **Kernel time fell 43 % and throughput did not move — which is the cleanest evidence available that this path is bound by the host, not the device.** Reverted rather than shipped.

~~**What that leaves is a launch-count lever with a concrete target.** `quantize_fp16_to_q8_1_ggml_kernel` fires 67 764 times…~~ **Withdrawn 2026-08-11, same day: that kernel only exists in the reverted experiment.** The profile it was read off had the fused MMVQ enabled; shipped `main` has no per-expert activation quantise at all, so "hoist it out of the loop" describes code that does not exist. What follows replaces it, measured against a proper control.

**Where the offload penalty actually comes from — a decomposition.** The comparison that settles it needs care: with experts resident the engine captures decode into CUDA graphs, and nsys does not attribute kernels replayed inside a graph unless asked, so a naive resident-vs-offload launch count reads 18 052 vs 197 809 and is meaningless. Forcing `runtime.cuda_graphs=never` on the resident arm makes the two comparable (offload has graphs disabled by the `engine.cpp` guard anyway). Same model and command, `--bench-pp 8 --max-tokens 32`:

| arm | `cudaLaunchKernel` | per token | tg tok/s |
|---|---|---|---|
| experts resident, graphs ON | 18 052 (graph-hidden) | — | **286.94** |
| experts resident, graphs OFF | 52 024 | ~790 | **116.13** |
| experts host-resident (graphs forced off) | 197 809 | ~3000 | **~14.9** |

So the 19x splits cleanly and multiplicatively: **2.47x from losing CUDA graphs** (host-resident experts disable capture outright) and **7.8x from the rest**, of which the visible mechanism is a **3.8x launch-count increase**.

**And the mechanism is a kernel family the offload path cannot reach.** Resident decode runs `gemv_dp4a_moe_gate_up_kernel<DequantTraits<Q4_K>>` — **one** launch per layer covering gate *and* up across all `top_k` experts (3168 launches = one per layer per token). Host-resident decode runs 24 `(expert, projection)` pairs at two kernels each: 48 launches per layer per token, ~16x more. The whole family exists and Q4_K is in it (`gemv_q4_k_q8_1_moe_gate_up_fused`, `gemv_q4_k_q8_1_moe_decode`, plus Q6_K/Q8_0/Q5_K/Q4_0/Q2_K/Q3_K siblings) — it is simply unreachable, because those kernels address an expert as `base + expert_id * stride` inside one contiguous device tensor, and the LRU cache holds experts in arbitrary slots.

**That makes the fix concrete, and it is not a new kernel.** Two ways to hand the existing kernels a contiguous view: gather the `top_k` active experts into a staging buffer per (layer, projection) and call them with `stride = expert_raw` and indices `0..top_k-1` — no kernel change at all, at the cost of ~10 MiB of D2D per layer-projection — or add a variant taking `const void* const*` expert pointers, which removes the copy but touches the family. Either way the target is the launch count, not the kernel time: the refutation above already showed that cutting 43 % of GPU kernel time on this path buys nothing.

**Except neither is needed, and this is the part worth acting on: the cache pool already IS the contiguous tensor those kernels want.** Three facts line up, each checked against the code rather than assumed:

- `ExpertLRUCache` partitions its pool per layer at a **fixed stride** — `pool_ + (layer * slots_per_layer + slot) * slot_size_` (`expert_cache.h`, `slot_ptr`). A layer's slots are exactly the `base + i * stride` array `gemv_dp4a_moe_gate_up_kernel` indexes.
- That kernel's only use of identity is `expert_id = expert_indices[expert_slot]` followed by `base + expert_id * stride_bytes`. Feed it **slot** indices with `stride = slot_size_` and it needs no change at all.
- The slot indices already exist on the device: `d_lookup_[layer][proj][expert]` is defined to hold the **layer-relative slot index** or -1, maintained on every insert and eviction. Its header says Phase 5 would read it from inside the dispatch kernels; this is that read, one level up.

So the shape is: make a layer's active experts resident (at the measured 88.7 % hit rate that is ~2.7 H2D per layer per token, not 24), gather `slot_idx[k] = d_lookup[layer][proj][expert_indices[k]]` in one small kernel, then call the existing fused kernel against the pool. Per layer that is ~3-4 launches against today's 48 — the 16x — with **no new kernel, no staging copy and no change to the cache's data structures**. The gate shipped in #1365 already guarantees the precondition, since it only uses the cache when `3 · n_active ≤ slots_per_layer`.

**Built and measured 2026-08-11 (#1370) — it holds.** Qwen3-30B-A3B-Q4_K_M with all 48 MoE layers host-resident:

| | before | after |
|---|---|---|
| decode tg256 | 22.9 tok/s (median of 17 runs, 21.0-25.0) | **48.3** (median of 5, 45.8-50.0) |
| `cudaLaunchKernel` | 197 809 | **61 585** (−69 %; 52 024 for the same model resident) |
| prefill pp512 | ~300-340 | unchanged — `n>1` still takes the legacy path |

**2.1x on decode, and the launch count now lands within 18 % of the fully-resident model** — the gap this entry opened is closed as far as the dispatch shape can close it.

**Correcting the correctness argument this entry shipped with.** #1370 cited perplexity — 10.9616 host against 10.9637 resident — as the evidence. **That measurement cannot see the change.** `--perplexity` is teacher-forced, so it runs prefill (`n=2048`, and the log says `legacy FP16 fallback path`), while the new path is gated on `n == 1`. The number is real and worth keeping for what it does show — prefill is untouched — but it is not evidence about decode. Textbook E6: right test, right property, an input that cannot reach the code.

What does support it, gathered afterwards:

- **The kernels and the bytes are not new.** Offload decode now runs the same dp4a kernels every on-device Q4_K MoE decode already uses, against a byte copy of the same weights. The only new thing is the *addressing*.
- **Addressing failure is loud, not subtle.** A wrong slot index reads a different expert's weight matrix, which degenerates; it does not drift.
- **Old vs new decode with prefill held constant** (`moe.no_expert_cache` toggled, so both arms take the legacy prefill): the two agree token-for-token for ~25 tokens and then diverge at a tie, both coherent and correct — the signature of a small numerical difference, which is expected because the old path was dequant→FP16→GEMV and the new one is dp4a Q4_K×Q8_1.
- **Two model families** on the offload decode path — Qwen3-30B-A3B-Q4_K_M and Gemma-4-26B-A4B-Q4_K_M — give coherent single-turn, multi-turn and long-decode output with a clean `stderr`.

Note the trap the first attempt walked into: **resident-vs-offload generated text is NOT a decode test either**, because prefill still differs between those two arms, so greedy decoding diverges from the first token for reasons that have nothing to do with the change.

What did **not** materialise is the rest of the 19x. Establishing residency needs the routing on the host, so the path still pays one D2H + sync per layer — the serial fallback paid it too, which is why this is 2.1x rather than the ~16x the launch ratio alone suggested. Removing it means predicting routing a layer ahead, which the trace study above already refuted. The other factor, **CUDA graphs (2.47x), is now the largest single remaining term on this path — and `moe.allow_graphs_under_offload` does not reach it.** Measured 2026-08-11: with the flag on, capture is attempted and **aborts every time** (three attempts, three aborts, per-step decode throughout), because every MoE path that serves host-resident experts reads routing on the host and `moe_host_args_capture_guard` throws under capture. That was already true of the serial fallback before #1370, so the flag has never delivered captured decode; its old description — "correct only when prefetch coverage matches router selection" — oversold it, since capture never gets far enough for coverage to be the question. The guard does its job: the abort is clean and falls back, so this was never a correctness risk, only a promise the flag could not keep. Descriptions corrected in #1373; the flag is kept as the escape hatch for the day the blocker lifts.

**A boundary this whole campaign has, stated because none of its entries said it: every number above is Q4_K_M GGUF.** The two models measured are Qwen3-30B-A3B-Q4_K_M and Gemma-4-26B-A4B-Q4_K_M. **NVFP4-prequant experts have no host-offload path at all** — `weight_upload.cu` calls them "mandatory on-device" and sets `overhead_pct = 0` to force them up, because host-resident they stay INT8-packed and reach the generic cuBLAS GEMM as raw bytes (status 15 → repeated-token garbage + IMA on Qwen3.6-35B-A3B-NVFP4). So the 52-103 tok/s band, the sizing rule and the 2.1x all describe the **GGUF** path. Whether any of it transfers to an NVFP4 checkpoint is untested, and it matters: a staged 80B-120B is at least as likely to arrive as NVFP4 or MXFP4 as as GGUF. That refusal predates this campaign (#925, 2026-07-09) and was never re-checked against it.

**Re-checked 2026-08-13, and the "mandatory on-device" rule was not enforced anywhere.** Measured on Qwen3-30B-A3B-NVFP4-Modelopt, the NVFP4 twin of this campaign's own model, greedy and same prompt: resident is coherent at 361.97 tok/s; `moe.force_host_experts=8` (8 of 48 layers) answers *"the capital of France is the city of the same name, France itself. 2: France is not a city"* at 88.77 tok/s; `=48` repeats `ftp` forever. **All three exit 0.** Three guards disagreed and each let it through: Phase 0 promotes the scale sidecars only onto weights that are already `on_device`, and says so at `IMP_LOG_DEBUG`, which the default log level does not print; `gemm()` then recognises the scale-less packed weight, logs an ERROR and **returns without multiplying** (bounded to 20 lines, so a long run prints nothing after the first few tokens); and the `weight_upload.cu` warning that names this hazard is on the *budget* branch, which `moe.force_host_experts` does not pass through. The predicate now sits where the placement is decided, in `model/expert_placement.h`, so it does not care which route produced the placement, and the load is refused with the remedy named. The GGUF path is explicitly out of scope for the refusal and is measured unchanged: the same prompt with all 48 layers host-resident is coherent at 41.06 tok/s, expert cache 64.3 % on a cold run.

So the entry above stands, with one correction: the symptom is not only "repeated-token garbage + IMA". At a *partial* offload the model stays fluent and is simply wrong, which is the harder failure to notice. Building the path itself is still open, and the shape is the same one #1370 used: a cache slot would hold one expert's packed bytes and its micro-scales concatenated, so `gemv_nvfp4_moe_*` can be fed slot indices with `expert_stride_packed = expert_stride_ms = slot_size_`, needing no kernel change. The one piece that does not fall out for free is `tensor_scales`, which those kernels index with the same id as the weight, so it needs a per-slot mirror rather than the per-expert array that exists today.

**Built 2026-08-13, and the decode half was exactly that. The estimate still missed roughly half the work, in a way worth recording.** Measured on Qwen3-30B-A3B-NVFP4-Modelopt, greedy, same prompt as the refusal above:

| arm | before (#1403) | after |
|---|---|---|
| experts resident | 361.97 tok/s, coherent | **384.03**, coherent |
| 8 of 48 layers on host | 88.77 tok/s, **wrong answer** | **44.54**, correct |
| all 48 layers on host | repeats `ftp` forever | **23.03**, correct |

Note the middle row got *slower*, which is the point: those eight layers were fast because they were skipping their GEMMs. Both host arms now emit the same answer as the resident one, and the 8-layer arm stops at the identical token count (119). GGUF offload is unchanged (39.77 tok/s at full offload, 65.7 % hit rate), and `make verify-fast` is flat (decode −1.05 %, prefill +0.95 %, peak VRAM +0.01 %).

**What the estimate missed: the prefill.** #1370 could ignore `n > 1` — its own entry says "prefill unchanged, n>1 still takes the legacy path" — because GGUF experts reach `dequant_expert`'s staging buffer there. Per-expert NVFP4 tensors do not: the M>1 fallback handed the 593 MiB expert matrix to `gemm_nvfp4`, which dequantises on the device, so the first working decode still died on an IMA. The fix is small once located (stage one expert per `expert_gemm` call through the same slot pool, three slots regardless of how many experts the prompt activates), but it is a second path, not a detail of the first.

**And the promotion had a blast radius nobody costed.** Letting Phase 0 label host-resident experts NVFP4 makes them visible to every consumer that keys off `qtype == NVFP4` and then does device work. Four had to learn the difference, each found by a separate crash or wrong answer: Phase 0b registered them for the CUTLASS grouped prefill (host pointer into a kernel); Phase 3 tried to copy them into a contiguous device buffer with `cudaMemcpyDeviceToDevice` (fails on a host source, leaving the buffer uninitialised — and it would have pulled back into VRAM exactly what was moved out); the micro-scales were still uploaded to VRAM, so the promotion's own both-on-host guard never fired; and `can_decode_fast` keys off `expert_up_packed`, which is only stamped for device-resident experts, so the decode sat in the serial legacy path at 35 tok/s while the new one went unused. **The generalisable rule: `qtype` says how to decode bytes, never where they live. A predicate that reads one and means the other is the #1384 / #1403 shape, and it appeared four more times here.**

Two costs remain, both measured rather than assumed. Prefill runs the serial per-expert fallback, and it evicts the decode working set — 74.6 % hit rate against the 88.7 % the GGUF path reaches, which is the same thrashing #1365's working-set gate fixed there and which nothing yet fixes here. Neither is a correctness issue and both are ordinary follow-on work.

**The second of those is now fixed, and the measurement is worth more than the fix.** #1365's rule already existed in `run_moe_legacy_fallback_` as `use_expert_cache` — a dispatch touching more than `slots_per_layer` cells bypasses the cache for the single staging buffer, same H2D bytes, no eviction. The NVFP4 staging path simply did not read it. Honouring it, on Qwen3-30B-A3B-NVFP4 with all 48 MoE layers host-resident, `--bench-pp 512 --bench-reps 3`, three runs per arm (medians):

| | cache misses | pp512 | tg256 |
|---|---|---|---|
| gate off | 106 108 | 285.57 | 54.77 |
| **gate on** | **30 451** | **285.15** | **57.31** |

**Misses drop 3.5x and throughput barely moves.** That is the informative part. It says the prefill transfers were never the cost — they move the same bytes either way, which is why `pp512` is flat to 0.15 % — and that what the eviction destroyed was only the *decode* cache's contents, worth +4.6 % on `tg256`. That figure sits inside this path's own spread (the arms overlap: 50.8-59.7 against 57.3-60.3), so treat the miss count as the result and the throughput as directionally consistent with it, not as a measured win.

**GGUF's +5.6 % on pp512 does not reproduce here, and the reason is structural rather than a failure to reproduce it.** There the bypass replaces a dequant-into-scratch with a straight copy; here both routes are a copy into a slot and the GEMM reads NVFP4 either way. Same rule, different arithmetic underneath, so only the decode half of the benefit transfers.

Two caveats on the numbers: the arms were not alternated (each switch is a full rebuild), and the first run of any arm is cold — 35.06 tok/s against 57-60 warm on identical code, the same warm/cold gap this entry records for the GGUF path.

*What the remaining serial prefill actually costs, profiled 2026-08-13.* Host-offloaded prefill is **61x** slower than resident on this model (pp512 17508 vs 285 tok/s), against only 6.9x on decode — so prefill, not decode, is where this path loses most. nsys says why, and it is not the GEMMs: `cudaMemcpyAsync` is **53.7 %** of API time at 90 759 calls, and the host sits in those calls for **4.13 s** while the GPU spends **795 ms** actually transferring. The gap is the driver staging an mmap source synchronously, because WSL2 cannot page-lock a mapping in place.

Isolated at one expert's size (768 KiB weights + 96 KiB micro-scales), medians over 300 reps:

| source | host time in the call | GPU time | rate |
|---|---|---|---|
| pageable mmap, 2 calls (what it did) | 76.2 µs | 92.3 µs | 9.6 GB/s |
| pinned, 1 call | 2.8 µs | 27.3 µs | 32.4 GB/s |

**`moe.pin_host_experts` (default off) copies those experts into pinned host memory at load**, which is what the GGUF packed path already does at `weight_upload.cu` Path A1 — the per-expert NVFP4 tensors simply never reached it, the same "one path got the treatment, the other did not" shape as the rest of this entry. Six alternating paired rounds, all 48 MoE layers host-resident:

| | pp512 | tg256 | model load |
|---|---|---|---|
| off | 276.6 | no effect | 5.1 s |
| **on** | **317.6 (+14.8 %, 6/6 pairs)** | 3 up / 3 down | 22.6 s |

Decode does not move because its cache hits 96-98 % and barely transfers; prefill touches every expert. It stays **off by default** because 4.4x model-load time is too visible a cost to impose silently, and break-even is around 32k prompt tokens.

**Two method notes, both of which cost a wrong conclusion first.** A per-expert pinned buffer was the obvious implementation and is the wrong one: nsys measured **36 877 `cudaHostAlloc` calls costing 24.7 s** plus 6.4 s of `cudaFreeHost`, to save 0.5 s of transfer. One slab per (layer, projection) is 144 allocations for the same result. And **three paired rounds read decode as −33 % and that was noise** — this path's decode spread is wider than the effect (the off arm alone spanned 34.7 to 66.1 tok/s), which is exactly the "only paired, alternating rounds decide anything here" rule already recorded above, needing six pairs rather than three.

*Where the offload prefill stands after that, and what the next lever is not.* Re-profiled with pinning on, the two halves are now about even: **784 ms of H2D against 670 ms of kernel time**, and the largest single kernel is `dequantize_nvfp4_kernel` at **51.9 % of kernel time** (17 085 instances, 20.4 µs each). That is the cost of the M>1 route in `gemm_nvfp4`: dequantise the staged expert to FP16 in VRAM, then cuBLAS. The resident path never pays it, because CUTLASS grouped reads NVFP4 directly.

**The obvious lever off that profile was tested and is refuted.** `gemm_nvfp4` sends M<=16 through the batched NVFP4 GEMV, which skips the dequant; at pp512 an expert sees ~32 tokens and therefore misses it. Raising `kSmallMBatchedGemv` 16 → 48 measured **316.09 against 317.6 tok/s median** — no change, well inside this path's spread. The reason is in the threshold's own arithmetic: the GEMV re-reads the NVFP4 weight once per MR=4 tile, so at M=32 it moves ~2.0x FP16-equivalent bytes against the dequant route's ~2.25x. The two are a wash there, which is where the 16 was placed to begin with.

So the dequant is not waste a threshold can recover; it is the price of having no grouped NVFP4 GEMM on this path. Closing it means feeding CUTLASS grouped from the slot pool in chunks, which is a project rather than a tuning change, and it is the honest next step for whoever picks this up.

*Scoped 2026-08-13, not built. Read this before starting it.* The shape that fits is **per-layer staging**, not chunking: copy one layer's experts into a contiguous device buffer, run the ordinary resident prefill against it, reuse the buffer for the next layer. It moves the same 15.8 GB per prefill pass as today, but as 144 large transfers instead of 18 432 small ones, and the resident path it hands off to has no dequant. Budget: **~331 MiB** for 128 experts x 3 projections at this model's sizes, one layer live at a time.

Four things that decide the work, each checked against the tree rather than assumed:

- **The staging copy is nearly free to write, because #1409 already made the sources contiguous.** `pin_host_experts` lays every expert of a projection into one pinned slab back to back, so a whole projection is *one* `cudaMemcpyAsync`. Without that flag the mmap offsets are not guaranteed adjacent and it needs a per-expert fallback.
- **`run_proj` (the smallM branch) consumes the NATIVE `[ne, N, K/16]` layout**, addressing `packed_data + e*stride` and `micro_scales + e*stride`. A staged buffer satisfies it as-is: **no SfAtom conversion needed**. But `moe.nvfp4_smallM` defaults to **off** and the branch only fires up to `max_M <= 64`.
- **The main CUTLASS 3.x path does NOT take that layout.** It wants SfAtom scale factors plus `wcache_->cutlass_nvfp4` entries, which host-resident experts deliberately do not have (registering a host pointer there is what caused the illegal access this entry opens with). Using it means running `convert_nvfp4_moe_scales_to_sfatom` per staged layer — the converter exists — and building the weight entries per layer rather than once at load.
- **The injection point is small**: the prefill reads `ly.nvfp4_moe_{gate,up,down}_ptr` in exactly **9 places** in `executor_forward_moe_cutlass.cu`, all in one function. A staged `NvFP4MoEQuantResult` carried on `MoeFfnContext` covers all of them.

So the decision to make first is which GEMM it targets: smallM is a far smaller change and already accepts the layout, but is opt-in, unproven here and capped at M<=64; the main path is the fast one and costs a per-layer SfAtom conversion plus weight-entry bookkeeping. Either way the transfer batching is worth having on its own — it is the larger half of the 784 ms, and it does not depend on which GEMM wins.

**Built 2026-08-13, and the transfer half alone is worth 2.5x.** The GEMM decision above is still open and was not needed: staging a whole layer into a device buffer and leaving the dequant→cuBLAS route untouched moves prefill from **317.6 to 790.8 tok/s** on Qwen3-30B-A3B-NVFP4 with all 48 MoE layers host-resident (medians of 3, ranges 298-345 against 766-798, nowhere near overlapping). Decode is unchanged at ~59, as intended — the buffer is only used at `n > 1`, because staging 128 experts to reach 8 would be more traffic than the slot cache it already uses. 324 MiB, one layer live at a time, reused across layers.

**The finding that matters more than the number: layer staging and pinning only work together.** With `moe.pin_host_experts` off, the same build measures **252-286 tok/s — the staging buys exactly nothing.** Two reasons compound. A pageable source is staged inside the driver no matter how large the request, so a big transfer is not a fast one; and it is pinning that makes a projection's experts contiguous in the first place (the per-projection slabs of #1409), which is what allows one memcpy instead of 128. The staging buffer is therefore allocated only when pinning is on — spending 324 MiB to gain nothing is worse than not having it.

That also re-prices `moe.pin_host_experts` itself, and it is now a much better trade than when it shipped: **276.6 → 790.8 tok/s prefill (2.9x) for 17.5 s of model-load time**, so break-even falls from ~32k prompt tokens to **~7.5k**. It stays off by default because the load cost is still the first thing an operator would notice, but for a long-lived server on this path it is close to a straight win. Against resident prefill the gap narrows from 61x to 22x.

*Re-profiled after it, and the remaining lever is unchanged in kind but now dominant.* Kernel time is where it was — `dequantize_nvfp4_kernel` is still **52 %** of it (344 ms over 17 049 instances, i.e. one per expert per projection per layer) — but it is now ~53 % of the prefill rather than one term among several. Staging fixed the transfers and did nothing for the dequant, exactly as intended.

**One correction to the scoping above, found by running it rather than reading it: the smallM branch is unreachable, so it is not the cheap option.** Resident prefill logs `CUTLASS 3.x device-args full path`, which is selected at `executor_forward_moe_cutlass.cu:159` — *before* the smallM branch at :405 is ever consulted. So "smallM already accepts the native layout" is true and irrelevant: nothing reaches it on this model. Closing the dequant means the device-args path, which needs SfAtom scale factors and `wcache_->cutlass_nvfp4` entries built per staged layer. The converter exists (`convert_nvfp4_moe_scales_to_sfatom`) and the staged buffer is the right shape to feed it; the work is the per-layer weight-entry bookkeeping, not the conversion.

**Built 2026-08-13 behind `moe.staged_cutlass_prefill` (default off), and it is the largest single step on this path — with a decode question attached that is the reason it is opt-in.** Converting a staged layer's scales to SfAtom and building the per-expert pointer arrays lets a host-resident layer enter the same CUTLASS grouped prefill a resident one uses. Six alternating paired rounds, all 48 MoE layers host-resident, `moe.pin_host_experts` on in both arms:

| | pp512 | tg256 |
|---|---|---|
| off (staged + dequant to cuBLAS) | 663.2 | 59.4 |
| **on (staged + CUTLASS grouped)** | **1563.9 (+136 %, 6/6)** | **37.7 (-36 %, 6/6)** |

The prefill arm is remarkably tight (1558-1568 across six runs, under 1 % spread), which is what a path that stopped waiting on transfers and dequants looks like. Cumulatively this path's prefill has gone **285 to 1564 tok/s** across this campaign, and the gap to resident from 61x to **11x**.

**The decode figure is real and NOT understood, which is why this is opt-in rather than default.** All six pairs are negative, so it is not the spread this entry keeps warning about. But it reverses with context: with `--bench-pp 8` instead of 512, the same two arms measure **25.5 to 30.6 tok/s, i.e. the staged path is FASTER**. This code only runs at `n > 1` and cannot touch the decode kernels, so what differs is the state decode inherits: the expert cache's hit rate is 91 % against 92.9-98.4 % after a long prefill, and 84.8 % against 80.6 % after a short one. A 2.4x prefill win does not get to impose an unexplained decode cost by default; explaining it is the next task here, ahead of further GEMM work.

Not a regression risk elsewhere: the resident NVFP4 path measures unchanged (pp512 17612 against 17508, tg 380 against 395), the default arm still takes the legacy fallback, and `verify-fast` is green.

*Where the path stands after all of it, re-profiled 2026-08-11 on the shipped build.* The entry opened by calling this path issue-bound; after #1370 and #1376 **it is not any more, and that retires the framing rather than confirming it.** Same config (256 cold decode tokens, all experts host-resident), tg phase 7.06 s:

| term | per token | share |
|---|---|---|
| H2D expert traffic | 186 786 ops, **150 GB** total, ~51 GB/s achieved | **41 %** of the step |
| kernel time | — | ~26 % |
| `cudaLaunchKernel` | ~1341 | ~24 % |

The remaining memcpys are no longer bookkeeping — median size 0.885 MB, i.e. expert weights — and they move at ~51 GB/s, which is the batched PCIe figure this entry measured earlier, not a stalled one. **So the path is now transfer-bound, which is the regime the PCIe budget at the top of this entry modelled all along.**

That narrows the remaining lever to one thing: **move fewer bytes**, which means a higher hit rate, which means cache capacity — already exposed as `moe.expert_cache_budget_pct` (#1374). Overlapping the transfers instead does not pay here: per-layer compute is 3 GEMVs at ~6.4 µs, far under the ~100 µs this entry measured as the point where a prefetched promotion becomes free. And the launch term that remains is dominated by the ~1100 non-MoE launches per token, which only CUDA graphs collapse — blocked as recorded above.

*And the largest lever on this path is neither of those — it is cache capacity, which was a hardcoded constant.* The pool is a share of free VRAM, fixed at 15 % since it was written, and that number decides how many tokens of routing history the cache holds: 73 slots/layer on this model is ~3 tokens, which catches the ~45 % next-token reuse the traces found but not the ~80 %-within-8 band. `moe.expert_cache_budget_pct` makes it measurable (#1374); swept on Qwen3-30B-A3B-Q4_K_M with all experts host-resident, 256 **cold** decode tokens (`--bench-pp 8 --bench-reps 1`, so no cross-rep cache warmth):

| budget | slots/layer | hit rate | tg tok/s |
|---|---|---|---|
| 5 % | 24 | 36.6 % | 10.51 |
| **15 % (default)** | **73** | **74.1 %** | **20.99** |
| 30 % | 146 | 89.4 % | 30.51 |
| 50 % | 244 | 96.2 % | **51.86** |

**2.47x from a config value** — the same factor CUDA graphs are worth, available today and without a line of kernel code. Monotonic in both columns, and the bottom row is the #1365 gate threshold (`3 · top_k = 24`) where the cache barely holds one token.

The default is deliberately **not** raised. This sweep runs a model that fits, forced off-GPU, so the room is free; on a model that genuinely does not fit, the same VRAM is what the KV pool and weight caches want, and the existing comment ("expert cache must not over-commit") is the reason the constant is conservative. What changes is that it is now a measured trade an operator can make, not an invisible constant.

*And the sweep has a floor, which is the number the 80B-120B ambition actually has to clear.* Below `3 · top_k` slots per layer the cache is bypassed by the #1365 gate, and decode collapses to the staging path at ~6.8 tok/s regardless of anything else. Probed 2026-08-11 whether that gate is too strict at the margin — a pool holding 79 % of the working set ought to keep most of the reuse — and **it is not**: measured hit rate is **0.0-0.4 %** at 14 and 19 slots against a 24-cell working set, and switching the cache on there is *slower* than bypassing it (5.60-6.28 vs 6.84-6.88 tok/s). Below full fit the cache retains nothing, so the threshold is exact rather than conservative.

That turns the floor into a sizing rule: **the offload path needs at least `n_layers × kExpertProjCount × top_k × slot_size` of VRAM for the cache before it beats the staging fallback at all.** For the 120B-A5B shape this entry keeps using (48 layers, top_k 4, ~6 MB per expert matrix) that is **~3.5 GB**, and for a Qwen3-Next-80B shape with top_k 10 it is nearer 8.6 GB — VRAM that has to be found *after* the resident weights and the KV pool. That, not the kernel work, is what decides whether the band above is reachable on a 32 GB card.

A caveat this sweep also exposed, worth carrying: **decode throughput on this path depends strongly on cache warmth, so a `--bench-reps 3` figure is not a cold-start figure.** The same build reads 20.99 tok/s cold (`pp8`, 1 rep) and 49.60 warm (`pp512`, 3 reps). The #1370 A/B is unaffected — both arms used the identical warm configuration — but absolute numbers from this path must state which they are.

**What lifting it would take**, so the next attempt starts from the constraint rather than the flag: routing *and* expert residency both resolved device-side. Routing is already there (`expert_indices` is a device array, and `d_lookup_` holds the slot indices), so the gather is capturable. Residency is not: a miss needs a host-issued H2D, and there is no device-side way to ask for one. A capturable path therefore has to either guarantee zero misses — which contradicts the premise that the experts do not fit — or tolerate a miss producing a defined, detectable result rather than a wrong expert. The 52-103 tok/s band for a 120B shape is no longer obviously out of reach: a 30B at 48 tok/s with 3.8x the parameters to stream is the same order.

Reproduce the profile: `nsys profile --sample=none --cpuctxsw=none --backtrace=none -t cuda --stats=true` around `imp-cli --set moe.force_host_experts=48`. **Put `--set runtime.cuda_graphs=never` on any resident arm you compare against**, or nsys hides the graph-replayed kernels and the launch counts are not comparable.

**Two things worth keeping, because neither was predicted.** *First, the budget runs the wrong way and it does not matter.* The pool is 15 % of **free** VRAM, so moving layers to host makes the cache **bigger** — 32 slots/layer at N=2 up to 73 at N=48, hit rate climbing 37.3 % → 89.4 % with it. The starvation this looked like it would produce never happens. *Second, prefill was thrashing the cache.* One dispatch touches `kExpertProjCount` cells per **active** expert, and at pp512 essentially every expert is active: 3·128 = 384 cells against 73 slots, i.e. a 73/384 = **19 % structural ceiling**, and a prefill-dominated run measures 24.3 % against it. Below that threshold the cache retains nothing and is strictly more work than the single-slot staging buffer for identical H2D bytes. Gating the cache on working-set fit is worth a median **+5.6 % pp512 (5/5 paired rounds positive)** at no decode cost, and lifts decode's own hit rate 88.7 % → 95.7 % once the thrashing accesses stop evicting it. Shipped.

**A measurement trap on this path, recorded because it invalidated a first pass:** prefill throughput under host-resident experts varies ~15 % between two runs of the *same* arm, which is larger than every effect above. Only paired, alternating rounds decide anything here — and the decode number moves with how long the prefill was, because prefill is what warms the cache (pp8 → 72.7 % → 13.82 tok/s; pp512 → 88.7 % → 24.98). Two runs with different prompt histories are not an A/B.

Reproduce: `tools/analysis/expert_cache_offload_sweep.sh` (`MODE=ab ROUNDS=5` for the paired comparison).

Sample caveats: one model, 512 tokens per prompt, and the k=8 row has a single held-out prompt per split (spread +-8.8 points against +-1.7 at k=3), so the flat middle of the curve is the trustworthy part, not its ends.

Reproduce: `bash tools/analysis/moe_routing_skew.sh` for the skew, `tools/analysis/host_transfer_latency.cu` for the transfer term.

Closed competitive records (kept for the record, not active work):

- **NVFP4 prefill vs vLLM -- CLOSED** (re-measured 2026-06-13, commit `290a163a`). FP16-QK FA2 as primary hd=128 prefill lifted pp4096 +21-24%: MoE pp4096 +4% ahead of vLLM, MoE pp2048 +27%, dense pp2048 ~tie. The lone residual gap -- dense pp4096 at ~1.04× -- is structural: every bounded kernel idea (cross-tile pipeline, grouped-GEMM tile axis, chunk-4096, occupancy/2-CTA, fp8-QK, scaled fp8-KV) was measurement-refuted; at pp4096 FA2 sits at ~5% DRAM and the dominant cost is the NVFP4 GEMMs (~59%), a separately-refuted ceiling.
- **kv-fp8 storage default-on -- SHIPPED** for Qwen3 dense/MoE, Llama (Phi-4), Nemotron-H MoE (`kv_cache.dtype=auto` honors the model's FP8 hint where the long-context quality gate passes; ~768 MiB KV saved on dense). Remaining families are blocked, not actionable: Qwen3.6-35B / Qwen3.5 declare no FP8 hint; Gemma-4's baseline PPL on the gate corpus is broken. These stay FP16 (or `--kv-fp8` opt-in).
- **Q4_K_M prefill gap (-38% vs llama.cpp) -- evidence-refuted.** The in-SMEM Q4_K MMQ + HMMA kernel was built (`feat/q4k-mmq-hmma`) and ncu-proved decode-throughput-bound, tying cuBLAS -- closing the gap needs beating cuBLAS or paying 2× weight VRAM (rejected). Practical resolution: use NVFP4 SafeTensors for fast Q4_K-class prefill. Details: [`plans/2026-05-28-q4k-mmq-kernel-design.md`](plans/2026-05-28-q4k-mmq-kernel-design.md).
- **Sawtooth wavefront reordering (#456) -- refuted** (measured 2026-05-29: only lives in the WMMA fallback, unreachable on the hot path; force-routed A/B flat-to-negative). (Harness `tools/analysis/sawtooth_ab.sh` was deleted with the rotted-script sweep in #1030.)

## Known limitations

- **MTP speculative decoding on Nemotron-3.5: loaded, correct, and uneconomic — because the draft path is not captured (measured 2026-08-12).** The head is wired (a miniature Nemotron: attention in `layers.0`, MoE in `layers.1`, per-expert 2-D non-gated squared-ReLU experts, DeepSeek-style router bias, NoPE attention, no `attn_output_gate`) and drafts well: **43.9 % top-1 accept** at depth 1, 17.3 % / 9.7 % at depths 2-3. It still loses badly:

    | k | tok/s | vs no spec |
    |---|---:|---:|
    | 0 | 364 | — |
    | 1 | 216 | -41 % |
    | 2 | 166 | -54 % |
    | 3 | 129 | -65 % |

    **Re-measured 2026-08-19, and the cost is now ~2 %, for a reason the table
    above cannot show.** `ea547a53` (2026-08-18) touched `ssm.cu` and
    `executor_ssm_gdn.cu`, which this Mamba2 hybrid runs through, so the numbers
    above price a build six days older than that fix:

    | k | tok/s (r1, r2) | mean | vs k=0 | drafted | accepted | verifies |
    |---|---|---:|---:|---:|---:|---:|
    | 0 | 366.11, 363.45 | 364.78 | — | — | — | — |
    | 1 | 357.45, 357.63 | 357.54 | **-2.0 %** | 24 | 2 (8 %) | 24 |
    | 2 | 353.59, 349.88 | 351.74 | -3.6 % | 48-80 | 1-7 (2-9 %) | 24-40 |

    **But the mechanism changed, not the drafter.** Speculation is not cheap
    here — it *stops*: with 0 of the first 8 drafts accepted, the acceptance-poor
    floor unbinds all speculation for the request after 8 verifies
    (`spec-ngram: req 2 gave up (acceptance-poor: verifies=8 accepted=0/8)`), so
    24 verifies over 2100 tokens is the whole of it. The -2 % is what 8 wasted
    verifies per request cost, not what MTP costs. **The guard is doing exactly
    its job**, and the honest reading of this entry is now "the head does not
    draft usefully on this model", not "speculation is expensive on this model".

    **Two things that did not add up, recorded rather than guessed at.** The
    first is resolved below (2026-08-20); the second closed as #1497.

    - This entry documents **43.9 % top-1 accept at depth 1**; the serving path
      measures **0-9 %**. Those may not be the same quantity — the 43.9 % came
      from `--mtp-spec-decode` through `mtp_accuracy_bench.sh`, which scores the
      draft offline, while this counts what the verify chunk accepts. Nobody has
      shown they agree, and the gap is too large to assume they do.
    - The load emits **270 `WeightMap: unrecognised weight name: mtp.*`
      warnings** for this checkpoint and **zero** for Qwen3.8-27B-NVFP4, whose
      head accepts 76 %. Both are Model Optimizer exports; the `divert_to_mtp`
      branch in `safetensors_loader.cpp` is gated on `llm_compressor_format`, so
      a Modelopt checkpoint's `mtp.*` tensors reach the generic mapper instead.
      The head still uploads (272 allocations, 2.49 GiB) so the warnings look
      cosmetic — but a warning that fires 270 times on a working load is noise
      that would hide a real one, and the 0 % acceptance sits right next to it.

    ```
    [PROV: commit=02872bdf date=2026-08-19 hw=RTX5090
           model=NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 quant=NVFP4 cuda=13.3
           path=imp-server n=3 prompts x 2 alternating rounds
           cmd=`tools/analysis/mtp_k_sweep.sh` with MTP_MODEL set, counters from
           /metrics, give-up line from the server log]
    ```

    **Resolved 2026-08-20: the 0-9 % was a defect, and the two numbers are the
    same quantity.** A fully rejected verify chunk takes the cheap path in
    `engine_spec_ngram.cpp:1072-1079` — it adopts `spec_snap_slab`, the recurrent
    state the chunk forward wrote as of its first row, instead of restoring the
    pre-chunk slab and re-forwarding. `run_gdn` has written that slab since #847.
    `run_ssm` never did: `ssm_scan_prefill` had no snapshot parameter at all and
    the `ssm_conv1d_prefill` call passed none, so on a Mamba2 hybrid the slab was
    never written, and `vram_alloc_` does not zero it. Every fully rejected verify
    therefore committed uninitialised VRAM as the recurrent state. Measured
    device-side with the slab poison-filled at allocation: **0 of 26 378 240 bytes
    written by the chunk forward without the wiring, 26 302 836 (99.71 %) with
    it.** The 0.29 % gap is the bytes that legitimately land on the poison value.

    What that cost, on the same commit and checkpoint:

    | | offline top-1 accept | serving accept | Nemotron k=1 decode |
    |---|---:|---:|---:|
    | before | 851/2097 = **40.6 %** | 0/24 = **0.0 %** | 354.80, 356.54 tok/s |
    | after | 861/2097 = **41.1 %** | 590/1507, 587/1510 = **39.2 / 38.9 %** | 177.17, 175.12 tok/s |

    The offline counter (`engine_scheduler.cpp:2069-2071`) scores, per eager
    decode step, whether the head's depth-1 draft equals the token the main model
    then emits. The serving counter (`engine_spec_ngram.cpp:1119-1120`, exported
    at `metrics_memory.cpp:85,88`) counts, per verify chunk, how many of the K
    drafts equal the verify forward's argmax at their row. At k=1 those are the
    same question, and they now answer it the same way. Before the fix they could
    not: the first fully rejected verify destroyed the state, the emitted stream
    degenerated (`Here's` then 300 x `0`), and nothing could match afterwards —
    which is also why the acceptance-poor floor fired after exactly 8 verifies and
    made the feature look like it cost only 2 %.

    **The economics verdict does not change, only its reason.** With the guard
    disabled, k=1 now runs speculation for the whole generation and costs **-51 %**
    (176 vs 363 tok/s); with the shipped guard it lands at 258-341 tok/s, because
    1.41 emitted per verify sits on the 1 + 0.40k break-even and the verdict flips
    between runs. The drafter was never the problem; the verify chunk is.

    **Not MTP-specific.** The same branch runs for any drafter. With
    `speculative.mtp_k=0` and n-gram/suffix drafting on this model, the pre-fix
    build derails into unrelated prose after the first fully rejected verify
    (`Here's a thinking process:\n\n1.  **0.5\n- 1.0,The, 2015). The first step in
    the process of creating a new product ...`, 1/79 accepted) where the fixed
    build stays on task (13/120 accepted).

    ```
    [PROV: commit=8a7f2763 date=2026-08-20 hw=RTX5090
           model=NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 quant=NVFP4 cuda=13.3
           path=spec-verify/mtp-draft
           cmd=`imp-server --think-budget 0 --set speculative.ngram=false --set
                speculative.mtp_k=1 --set speculative.mtp_econ_min_emit=0 --set
                server.prefix_cache=false`, 3 prompts x 700 max_tokens; offline arm
                `imp-cli --mtp-spec-decode 1 --set speculative.hybrid=false` on the
                same 3 prompts
           n=2 per arm, arms alternated, fresh process per arm]
    ```

    **The reason is now unambiguous, and it changed with #1389.** Measured before that fix, when the main decode was graph-demoted at 126 tok/s, MTP cost only -1.3 % at k=1 — draft and decode were both eager, so a draft step was priced like a decode step. With graphs restored the main decode is ~2.75 ms/token while the draft still runs eager (`mtp_forward.cu`: "drafts run outside graph capture for now", three `cudaStreamSynchronize` per draft token). A draft step now costs roughly a *whole* decode step despite the head being a fraction of the model, so every speculative token is a bad trade.

    **The draft MoE is now device-side (`gemv_f16_moe_decode`, added for this)** — no D2H of the routing, no host loop, one fewer sync per draft token. It bought +14…+23 % *on the speculative path* (k=1 216→247, k=2 166→194, k=3 129→159) and did **not** change the verdict: k=1 is still −32 % against no speculation. That matches the bound computed before building it — with a *free* draft the ceiling is +10.7 % at best and negative if the verify chunk really costs 2× a decode step. **The remaining cost is the verify chunk, not the draft.** Capturing the draft path is now unblocked (that was the point of the kernel) but is not worth doing for its own sake. Until then the built-in tree probe caps the other direction — a top-4 tree reaches E[accept] 1.08 vs 0.52, nowhere near the break-even. `speculative.mtp_econ_min_emit` detects the loss after 8 verifies and unbinds, so the default costs nothing and `--mtp-spec-decode` is opt-in. For context, NVIDIA's own DSpark drafter measured **-42 % in vLLM** on this card (351 -> 202), and the model card recommends no speculation at all for H100-class bandwidth.

- **Single GPU only.** No tensor parallelism, no multi-GPU.
- **Blackwell only.** No Hopper, Ada, Ampere. No AMD, Intel, Apple, CPU.
- **Qwen3.5-27B MXFP4 untested** -- the old wording ("fails at load, blocked on no public MXFP4 GGUF + NaN bug") is stale in each part. The alpha/beta NaN is moot: GDN alpha/beta are pinned `FP16_ONLY` (`tensor_kind_table.cu`) and dequantised at load, so that kernel is unreachable for them. The GGUF framing is wrong too — the SafeTensors loader no longer refuses MXFP4, it warns; the real blocker is that no MXFP4 SafeTensors *decode* path exists outside gpt-oss. And the original load failure was VRAM, not NaN, for which `attention.mxfp4_fp16_cache_policy = "pruned"` (#244) exists but has never been verified on this model. Net: blocked on a checkpoint imp can decode, not on a bug.
- **Gemma-4 Q4_K_M code-gen drift** -- no longer reproduces (verified 2026-06-13, and again 2026-08-11 when the same UD-Q4_K_M ran coherent single-turn, multi-turn and long-decode output on the host-offload path; the original file is gone, so it can't be A/B'd). If some other Q4_K_M quant of this model degenerates, fall back to Q5_K_M or Q8_0.
- ~~**Native-FP8 weights decode through their FP16 companion**~~ -- **closed 2026-08-12, measured.** Decode now serves those weights from the checkpoint's own FP8 bytes; the FP16 companion is prefill-only. The mechanism was already there: phase 4 has a `fp8_decode_sidecar` rule that demotes an FP8 cache entry out of the primary tier and sets `decode_tier = FP8`, and it simply did not recognise a native-FP8 source (its two tests are `qtype == F16` and `dequant_gpu_supported`, neither of which matches). An explicit `FP8CacheEntry::native_source` flag now drives that rule, and phase 3 reads the same flag to *keep* the FP16 copy rather than treating the entry as an alternative to it.

    **The two traps, for whoever touches this next.** A first attempt registered the entry without the phase-4 rule; `infer_tier_from_wcache` derives a tier from *which cache holds a weight*, so prefill moved to FP8 as well and died on `status 15`. Phase 3 then read the entry as "has an FP8 alternative" and freed the FP16 copy prefill depends on — an illegal access that **only appears under decode mode 2**: the plain CLI run passed, `--bench` is what exposed it. Both are the same underlying shape: one lookup answering two different questions.

    **Measured** (`gemm.fp8_ssm_proj` on/off, same build, paired, 12 in-process reps each): **median +7.5 %** decode over 27 pairs, t=3.33. Order matters and was balanced deliberately — measuring ON first reads +8.2 % median, OFF first +3.8 %, so a single-order A/B would have overstated it. The independent prediction from bytes/bandwidth (890 MB/token instead of 1780 across the 46 tensors, at 1792 GB/s) is +6.9 %, which the median matches. Do not quote the mean (+11.1 %): this host's per-process spread on this model is wide enough that outliers pull it.
- ~~**No dequant path for native FP8 weights**~~ -- **closed.** Native-FP8 weights get an FP16 companion at load, which is what `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` needed (Modelopt `MIXED_PRECISION`: 5935 MoE expert tensors NVFP4, 46 Mamba `in_proj`/`out_proj` FP8). sm_120 has no FP8 prefill GEMM, so the raw bytes used to reach cuBLAS as `dtB=CUDA_R_8F_E4M3` and abort with `status 15`. Phase 0 now records the per-tensor scalar scale instead of mis-promoting the weight to NVFP4, and Phase 1 expands it with the existing `dequantize_fp8_e4m3_to_fp16`. Costs 1698 MiB of FP16 cache on this model; init lands at 24.4 of 32.6 GB. See `docs/supported-models.md`.

## Investigated and shelved

- **Draft-model speculative decoding** -- separate draft models don't amortize weight reads on a single bandwidth-bound GPU. What *did* ship instead: prompt-lookup n-gram speculation (default-on for batch-1 greedy dense, #668-#670) and MTP self-drafts with hybrid-safe verify (#852) -- the drafts are free, so the economics work.
- **FFN contextual sparsity** -- warp-cooperative layout masks the skip. +0-1% measured.
- **BitDecoding (TC KV decode) -- shelved, and the scope now stated** (#1268, 2026-08-07). The original entry read "decode is weight-bound, not attention-bound, 0% gain" with no context length attached, and that omission is what made it misleading: it was measured at `tg256`, which prefills 64 tokens. Paged attention is 4.3% of the decode window there and **31.1% at 8k, 45.1% at 32k** -- at long context it is the second-largest class, not noise. Still shelved, but for a different reason than "attention doesn't matter": the two levers built on the finding are dead, and the third reading it rested on did not survive measurement.
  - *Split-count boost* (#1270, reverted #1271): +10.0% at 32k on Qwen3-8B (`n_kv_heads=8, g=4`), **-7.30% at 32k on Qwen3-30B-A3B** (`n_kv_heads=4, g=8`). One model is not a heuristic; the condition separating the two was never established.
  - *KV block size 16 -> 32*: neutral everywhere (-0.48% .. +0.07%), with the 30B as a null control that came out null.
  - *"Latency/occupancy-bound at 192 GB/s"*: **retracted by measurement.** The same kernel reaches **629.6 GB/s at 32k -- 3.4x -- at unchanged 16-17% occupancy** (roofline runs `dca16b71_20260806_041710` and `120bc0d7_20260807_091356`). The low bandwidth at 8k is a kernel short of work, not one held back by occupancy; the occupancy figure itself is a deliberate smem-for-L2 trade the tile dispatch documents.
  - Amdahl, re-estimated where the class actually weighs most (32k): closing 629.6 -> 1127 GB/s (what the GEMV reaches at that length) is ~20% of the decode window -- against a kernel already at 35% of roofline, not the 3.8x gap the 8k figure suggested.
  - Re-open on a mechanism, not on the share: the share is real and will keep growing with context.
- **NVFP4 GEMV tuning** -- 6 approaches refuted; structurally bandwidth-bound. The "64-73% of HBM peak" this used to quote is a 2026-05 figure the kernel has since outgrown: [`sm120_optimal_kernel.md`](internals/KERNELS.md) measures the decode GEMV at the GDDR7 ceiling, ~1.5 GB/ms (~84% of datasheet, ~98% of the 1531 GB/s a resident buffer actually reaches). The refutation is unaffected — there is less headroom than the old number implied, not more.
- **FMHA rewrites** -- cluster, TMA bulk and the long-context heuristic were each A/B-refuted, and that still stands. **The "cuBLAS wins" conclusion does not**: since #597/#930 the register-resident FA2 kernel is the DEFAULT prefill path for hd=128/256 (`attention.fmha_fa2` / `fa2_fp16qk` both `"on"`), and cuBLAS is the fallback for configs FA2 declines.
- **MoE offload + CUDA Graphs** -- **no longer shelved; moved to "CPU-resident cold experts" above, where it is now an active, measured entry.** This line said "full kernel-driven slot resolution deferred (multi-week, marginal user impact)" and every part of that is now false: it shipped in a day (#1370), is worth 2.1x decode, and it is not kernel-driven at all — the device-side mirror it was designed around turned out to have no reader and was removed (#1376). The CUDA-graphs half is open and blocked for a stated reason (#1373).
- **CUDA Tile (cuTile C++) -- benchmarked on sm_120, shelved** (2026-05-29). A correct cuTile FA2 autotuned to 26.5 eff-TFLOPS = 3.2% of roofline, order-of-magnitude below imp's hand-written FMHA -- confirms the published 0.53×-FA2 result on this arch (vs 2.5× on B200). Re-evaluate only on a new toolkit showing ≥parity on sm_120. Harness: `tools/analysis/cutile_fa2.py` + `Dockerfile.cutile`.
- **CompileIQ ptxas auto-tuning -- refuted** (2026-05-29). The ptxas search space is flat on imp's hotspots: FA2 is smem-occupancy + barrier-bound, NVFP4 decode is HBM-bound -- codegen touches neither (all sweep points within ±0.4%). Reusable harness: `tools/analysis/Dockerfile.ciq` + `tools/analysis/ptxas_sweep.sh`.
