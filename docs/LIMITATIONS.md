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
- **NVFP4 MoE experts cannot be host-offloaded.** A checkpoint whose experts do
  not fit is refused at load instead of served with those experts silently
  skipped (#1403). GGUF experts do have a working host path.
- **Batched and solo decode are not bit-identical.** Joining a batch costs
  rounding, measured at 0.22 % of the logit range, with identical greedy argmax.
  A neighbour's *content* provably cannot reach another row. No flag makes the
  two bit-equal; pin batch composition if you need that.
- **MoE routing uses atomics**, so identical seeds can diverge.
- **Speculative decoding is not universally profitable.** On Nemotron-3.5 the MTP
  head drafts well (43.9 % top-1 accept) and still costs 32 % of decode, because
  the draft path runs outside CUDA graphs while the main decode does not. It is
  opt-in and self-disables after 8 verifies.
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
