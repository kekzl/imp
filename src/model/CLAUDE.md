<!--
layer: L3
audience: agents
verified: 2026-09-13
commit: 81d22eff
-->

# src/model - loaders, architectures, weight upload

GGUF and SafeTensors loading, the architecture registry, tensor-name mapping, upload and placement.

## Invariants

- Architecture registry = `model_arch.h` + the string map in `model.cpp`. A new arch needs both, a loader branch and a chat template. Phi (`PhiForCausalLM`, `Phi3ForCausalLM`) maps onto `LLAMA` in `hf_config_loader.cpp`, no own arch.
- A checkpoint this build cannot serve is refused at load, never loaded and served wrong (#1403).
- A drop predicate and the function that executes it see the same condition (#1384, #1403).
- Tensor names are translated in one place: two translation paths that disagree discard whole towers.
- A checkpoint is untrusted input, including the counts it states about itself. Anything reaching a `resize`, a recursion or an `open()` needs a bound from `model_limits.h` (declared count: refuse; index parsed out of a name: drop) (#1611).

## Entry points

- `model_arch.h`: the architecture enum
- `model.cpp`: string map, per-arch sampling defaults, KV-FP8 safety lists
- `gguf_loader.cpp` / `safetensors_loader.cpp`: the two formats
- `hf_config_loader.cpp`: `config.json` parsing, arch detection
- `weight_map.cpp`, `tensor_kind_matcher.cpp`: tensor name -> role
- `weight_upload.cu`: device placement, expert offload decisions
- `expert_placement.h`: the pure predicate for a servable MoE placement

## Test

`make test-vision`: the only lane that puts image bytes through a real checkpoint. Model-level e2e runs against the `make build` image, not `make dev`. Lane rules: `tests/CLAUDE.md`.

## Pitfalls

- Cost a loader task from the checkpoint (`config.json`, tensor names), not from its model category.
- The failure mode here is a silently skipped tensor: count assigned vs total, refuse on a shortfall.
- `IMP_LOG_DEBUG` is invisible at the default log level: a skip reported only there is not reported.

## Do not touch

`third_party/` GGUF headers.

## See also

[`docs/MODELS.md`](../../docs/MODELS.md) (what loads), [`docs/internals/QUANT_PIPELINE.md`](../../docs/internals/QUANT_PIPELINE.md) (quant layers). Skills `add-model-arch`, `quant-formats`.
