<!--
layer: L1
audience: operators
verified: 2026-08-13
commit: 81ffa573
-->

# Limitations

What imp does not do, does badly, or does without a test behind it. This file
exists so a reader can decide against imp quickly. The top five are repeated in
the README on purpose.

Things that are absent *by decision* rather than by omission are in
[`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md), with the measurement that made the
decision.

## The five a new reader should weigh first

1. **One GPU, one chip.** `sm_120a` only: no multi-GPU, no tensor parallelism,
   and no other GPU vendor or generation. What consumer Blackwell has and lacks
   is in [`internals/ARCHITECTURE.md`](internals/ARCHITECTURE.md).
2. **One model resident at a time.** 32 GB fits one; serving a second means a
   swap, and the requesting call pays the load.
3. **No GPU in CI.** Every kernel-level correctness and performance check runs on
   the maintainer's machine before a push. If that matters to your risk model,
   it should count against imp.
4. **Decode measurements move several percent between sessions on this host.**
   Any single number you read anywhere, including ours, is one sample of a
   distribution. See [`PERF.md`](PERF.md).
5. **Single-author project.** No support rotation, no SLO, no security response
   process.

## Untested code paths (every 🟡 from `FEATURES.md`)

These have a code path and no gate. They may work; nothing proves it.

- **Llama-4** architecture: loads, no dedicated gate.
- **FP8 E5M2**: the type exists, nothing exercises it.
- **logprobs**: present in the parameter surface, no dedicated test.

## Known-bad and known-limited behaviour

- **INT4 KV cache produces empty output on gpt-oss.** Its sink term is correct
  and is unit-tested against a sink-aware reference; 4 bits per value on a
  64-wide head is simply not enough. It falls back to FP16 rather than pretending.
- **Host-offloaded NVFP4 MoE experts are slow, and nothing gates them.** They run
  correctly (Qwen3-30B-A3B-NVFP4, all 48 MoE layers on host: 23.3 tok/s against
  384.0 resident), but the price is steep and the only checks that exercise the
  path are manual. The CPU lane covers the slot arithmetic, not the kernels.
  The remaining cost is that prefill goes through the serial per-expert
  fallback, one expert per GEMM. A placement the expert cache cannot hold at all
  is still refused at load rather than served wrong (#1403).

  [PROV: commit=8a7bd8c date=2026-08-13 hw=RTX5090 model=Qwen3-30B-A3B-NVFP4-Modelopt
         quant=NVFP4 cuda=13.3 path=nvfp4-moe-host-offload
         cmd=`imp-cli --max-tokens 220 --temperature 0 --set moe.force_host_experts=48`
         n=1 note=single greedy run per arm; resident arm is the same command
         without the --set. Cold cache, short prompt — not a benchmark figure]
- **Batched and solo decode are not bit-identical.** Joining a batch costs
  rounding, measured at 0.22 % of the logit range, with identical greedy argmax.
  A neighbour's *content* provably cannot reach another row. No flag makes the
  two bit-equal; pin batch composition if you need that.
- **MoE routing uses atomics**, so identical seeds can diverge.
- **Speculative decoding is not universally profitable.** On Nemotron-3.5 the MTP
  head drafts well (43.9 % top-1 accept) and still costs 32 % of decode, because
  the draft path runs outside CUDA graphs while the main decode does not. It is
  opt-in and self-disables after 8 verifies.
- **Speculation is off for most real requests, by two rules that are easy to
  miss.** It requires greedy sampling (`temperature: 0` or `top_k: 1`), so any
  request with a temperature gets none; and a think budget disables it inside
  the think block, which on a reasoning model is most of the answer. The server
  sets `think_budget` to 0.5 by default, so on such a model speculation never
  runs out of the box. Penalties are **not** a blocker at the default
  `repeat_last_n: 0`: the verify replicates them for the unbounded window.
  Other engines have neither rule, because rejection sampling makes a verify
  distribution-preserving at any temperature and none of them force `</think>`.
- **The same request can produce different output on its second pass through
  one process.** Two things speculation keeps across requests carry it: the
  n-gram drafter's corpus, and the MTP head's own KV cache, which resumes over
  the longest common prefix of the previous history. Draft acceptance therefore
  depends on what the process has served before, and accepted drafts change
  which tokens are emitted. Observed here as three prompts run twice in one
  server, where the first differed on the second pass while two fresh processes
  running the same three prompts in the same order were byte-identical.
  A peer's conformance tier had been carrying this as an unexplained sporadic
  divergence. Not a defect on its own, but a golden-output test has to pin
  `speculative.ngram` and `speculative.mtp_k`, or it is testing the history too.

  **Only on the greedy path.** Speculation requires `temperature: 0` or
  `top_k: 1`, so a sampled request never drafts and none of this reaches it —
  measured as `imp_spec_drafted_total` not moving at all on a
  `temperature: 0.8` request with the drafter enabled. A sampled request that
  diverges between identical runs is a different mechanism: without
  `runtime.deterministic_gemm` the forward is not bit-reproducible, and at a
  temperature a last-bit difference in the logits moves the sampled token
  whenever the top candidates are close.
- **Speculative decoding does not reproduce the non-speculative greedy output on
  a GDN hybrid.** Its contract is that it changes speed and not tokens; here it
  changes tokens. Measured on Qwen3.8-27B-NVFP4 with
  `runtime.deterministic_gemm=true` on both arms and a stable control (two
  no-MTP processes, byte-identical): with `mtp_k=2`, two of three prompts
  diverge from the no-MTP answer, the first at character 79 of 1026 — an early
  token flip, not a rounding tail. Both answers are coherent and correct, but
  they are different generations. The verify advances the recurrent state
  through the chunk kernels while plain decode advances it through the
  single-token path, and the two do not agree bit for bit. Predates the
  2026-08-17 verify work: the same two prompts diverge with the older eager
  replay.
- **MTP loses on a GDN hybrid at every chain length, and the guard is right to
  disable it.** Measured on Qwen3.8-27B-NVFP4, greedy, 256 tokens, thinking off,
  economics guard disabled so it runs throughout, two alternating rounds with a
  fresh process per arm:

  | `mtp_k` | tok/s (round 1 / 2) | emitted per verify | draft acceptance |
  |---|---|---|---|
  | 0 | 89.2 / 87.2 | | |
  | 2 | 83.3 / 80.1 | 2.22 / 2.25 | 58.8 / 54.3 % |
  | 4 | 64.0 / 61.9 | 2.56 / 2.46 | 36.7 / 34.5 % |
  | 6 | 53.8 / 52.9 | 2.55 / 2.44 | 24.6 / 23.9 % |

  A longer chain does not buy its way out. Emitted-per-verify saturates near
  2.5 from `mtp_k=4` on while the verify chunk grows linearly in k, so the
  fourth and later draft tokens are paid for on every verify and almost never
  accepted: acceptance falls from 58.8 % to 23.9 % across the sweep. Break-even
  needs the verify below 2.25 emitted tokens times the decode step, and no
  chain length on prose reaches it. This is the chain-length form of the same
  result recorded in `docs/roadmap.md`: acceptance is the lever, chain length
  is not.

```
[PROV: commit=3cf2af24 date=2026-08-17 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=imp-server n=3 runs of 256 greedy tokens per
       arm, 2 alternating rounds
       cmd=`--set speculative.mtp_k=0|2|4|6 --set speculative.mtp_econ_min_emit=0
       --set runtime.max_seq_len=8192`, request `temperature 0, think_budget 0`;
       rates and acceptance from /metrics deltas, card idle at ~0.9 GiB]
```

  An earlier measurement on the pre-snapshot build (commit `a82dec8f`, before
  the mid-chunk recurrent snapshot of #1459) put `mtp_k=2` at 65.5 tok/s and
  found that enabling MTP cost ~17 % even when the drafter never fired. The
  snapshot work lifted `mtp_k=2` to the numbers above; the ~17 % figure has not
  been re-measured since.
- **`imp-cli --prompt` prints only about 10 tokens.** Byte-level comparisons have
  to go through the server's JSON.
- **Prefill graph capture is disabled per model** when one NVFP4 weight exceeds
  the dequant-workspace cap. Decode graphs are unaffected.
- **No video input.** Images work; video needs a decoder this tree does not vendor.
- **No vision architecture registry.** The tower allowlist covers Qwen3-VL-shaped
  encoders and Gemma. InternVL and Pixtral each need a port.

## Model-specific blockers

- **Qwen3.5-27B MXFP4**: blocked on a checkpoint imp can decode, not on a bug.
  No MXFP4 SafeTensors decode path exists outside gpt-oss.
- **Gemma-4 on the FP8-KV quality gate**: its baseline perplexity on the gate
  corpus is broken, so it stays FP16 KV unless you opt in.
- **Qwen3.6-35B / Qwen3.5**: declare no FP8 KV hint, so they stay FP16 KV by
  default.

## Operational sharp edges

- **A successful `cudaMalloc` proves nothing about free VRAM on WSL2.** The
  driver oversubscribes into host memory and returns success; 28 GiB allocates
  with 22.6 GiB reported free. The tell is bandwidth, ~1530 vs ~237 GB/s, and the
  symptom is a 6.5x throughput cliff.
- **Free VRAM only ever decreases within a process** on WSL2/WDDM, however
  cleanly CUDA released it. Anything sized from `cudaMemGetInfo` reads a moving
  floor.
- **`kv_cache.swa_snapshot_mb` set below one snapshot silently disables prefix
  caching** — worse than setting it to zero. It warns since #1092.
