<!--
layer: L1
audience: operators
verified: 2026-10-30
commit: a4070e6a
-->

# How imp works

- **One chip generation, no compromises for others.** Every Blackwell `sm_120a` card multiplies 4-bit numbers (NVFP4) directly; imp keeps weights in that format and computes on it, where portable engines unpack to 16 bits or skip the format.
- **Everything is its own:** file loaders, tokenizer, the conversation memory (a paged KV cache shared between requests), the GPU kernels. No framework underneath.
- **Replay, not re-launch.** Each answer step is recorded once as a CUDA graph and replayed, so the CPU does not hand the GPU thousands of small jobs per token.
- **Guess, then check.** A cheap drafter proposes the next tokens, the model checks them all in one pass and keeps what it agrees with: same answer, fewer passes.
- **Many users at once:** requests share the card (continuous batching), a repeated prompt prefix is read once (prefix caching).
- **Models bigger than the card:** experts live in host RAM, the busiest are cached on the card ([`BIG_MODELS.md`](BIG_MODELS.md)).

| Next | Read |
|---|---|
| what exists and what is tested | [`FEATURES.md`](FEATURES.md) |
| why it is this fast, or this slow | [`PERF.md`](PERF.md) |
| what it deliberately does not do | [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md) |
| the whole design, kernels included | [`internals/ARCHITECTURE.md`](internals/ARCHITECTURE.md) |
