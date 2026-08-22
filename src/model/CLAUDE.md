<!--
layer: L3
audience: agents
verified: 2026-08-13
commit: 81ffa573
-->

# src/model — loaders, architectures, weight upload

GGUF and SafeTensors loading, the architecture registry, tensor-name mapping,
and the upload/placement decisions that follow.

## Invariants

- **The architecture registry is `model_arch.h` plus the string map in
  `model.cpp`.** Adding an architecture means both, plus a loader branch, plus a
  chat template. Phi-4 is an *alias* onto LLaMA, not its own arch.
- **A checkpoint this build cannot serve must be refused at load**, not loaded
  and served wrong. Refusing costs a user one clear error; serving wrong costs
  them a wrong answer at exit code 0 (#1403).
- **A drop predicate and the function that executes it must see the same
  condition.** #1384 and #1403 were both this shape: a gate standing in front of
  the check meant to catch the problem.
- **Tensor names are translated in one place.** Two translation paths disagreeing
  is how a vision tower gets discarded whole.
- **A checkpoint is untrusted input, including the numbers it states about
  itself.** Anything out of a file that reaches a `resize`, a recursion or an
  `open()` needs a bound: `model_limits.h` holds the caps and the rule for which
  treatment applies (a declared count is refused, an index parsed out of a name
  is dropped). One tensor name reached 18.9 TiB before #1611.

## Entry points

- `model_arch.h` — the architecture enum
- `model.cpp` — string map, per-arch sampling defaults, KV-FP8 safety lists
- `gguf_loader.cpp` / `safetensors_loader.cpp` — the two formats
- `hf_config_loader.cpp` — `config.json` parsing, arch detection
- `weight_map.cpp`, `tensor_kind_matcher.cpp` — tensor name → role
- `weight_upload.cu` — device placement, expert offload decisions
- `expert_placement.h` — the pure predicate deciding a servable MoE placement

## Build & test

```
make dev && make dev-test
make test-vision            # the only lane that puts image bytes through a real checkpoint
```

Model-level end-to-end work must run against the `make build` image, not
`make dev`.

## Pitfalls

- **Cost a loader task by reading the checkpoint, not by its category.** A second
  vision family was estimated at fifteen PRs and took two gates, because the
  checkpoint already carried the tower.
- A silently skipped tensor is the failure mode here. Count assigned vs total
  and refuse on a shortfall; 247 of 316 once "succeeded".
- `IMP_LOG_DEBUG` is invisible at the default log level. A skip reported only
  there is not reported.
- Test-model env vars that point at the wrong path make the battery skip
  **silently**.

## Do not touch

`third_party/` GGUF headers.

## See also

[`docs/MODELS.md`](../../docs/MODELS.md) for what loads,
[`docs/internals/QUANT_PIPELINE.md`](../../docs/internals/QUANT_PIPELINE.md) for
the quant layers. Skills `add-model-arch` and `quant-formats`.
