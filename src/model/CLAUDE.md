<!--
layer: L3
audience: agents
verified: 2026-10-24
commit: 28185525
-->

# src/model - loaders, architectures, weight upload

GGUF/SafeTensors loading, arch registry, tensor-name mapping, upload and placement.

## Invariants

- Arch registry = `model_arch.h` + `arch_registry.cpp`: the one table of GGUF / HF class / HF model_type spellings (#2457). New arch: both, a `kArchRegistry` row in `model.cpp`, a `tests/test_model_profile_table.cpp` row, a chat template. Phi (`PhiForCausalLM`, `Phi3ForCausalLM`) maps to `LLAMA`.
- A checkpoint this build cannot serve is refused at load (#1403).
- A drop predicate and its executor see the same condition (#1384, #1403).
- Tensor names are translated in one place.
- Checkpoints are untrusted, including their own counts. Anything reaching `resize`, recursion or `open()` is bounded by `model_limits.h`: declared count -> refuse, index from a name -> drop (#1611).

## Entry points

- `model_arch.h`: arch enum
- `arch_registry.cpp`: checkpoint spelling -> `ModelArch` (`parse_model_arch`, HF loader)
- `model.cpp`: per-arch config, sampling defaults, KV-FP8 safety lists
- `gguf_loader.cpp` / `safetensors_loader.cpp`: the two formats
- `hf_config_loader.cpp`: `config.json` detection + `load_config`; generic keys `hf_config_generic.cpp`, per-arch `hf_config/<arch>.cpp`, one `kHfConfigHooks` row each in `arch_registry.cpp` (#2537)
- `weight_map.cpp`, `tensor_kind_matcher.cpp`: tensor name -> role
- `weight_upload.cpp`: device placement, expert offload
- `expert_placement.h`: pure predicate for a servable MoE placement

## Test

`make test-vision`: only lane with image bytes through a real checkpoint. Model e2e runs against `make build`, not `make dev`. Lanes: `tests/CLAUDE.md`.

## Pitfalls

- Cost a loader task from the checkpoint (`config.json`, tensor names), not the model category.
- Main failure: silently skipped tensor. Count assigned vs total, refuse on shortfall.
- `IMP_LOG_DEBUG` is invisible at default log level: a skip logged only there is unreported.

## See also

[`docs/MODELS.md`](../../docs/MODELS.md), [`docs/internals/QUANT_PIPELINE.md`](../../docs/internals/QUANT_PIPELINE.md). Skills `add-model-arch`, `quant-formats`.
