<!--
layer: L3
audience: agents
verified: 2026-08-13
commit: 81ffa573
-->

# imp test suite

> Comments across this tree cite `TEST_AUDIT (retired) §N` or `risk #N`. That
> document existed twice, as `docs/TEST_AUDIT.md` and `tests/TEST_AUDIT.md`, and
> both copies were deleted (#805, #946). What the section and risk numbers meant
> is recorded in [`../docs/archive/README.md`](../docs/archive/README.md). The
> citations are kept because they are why those tests exist; they are not a
> pointer to a file you can open.

How to build, run, and extend the tests. The engine targets **one chip**
(NVIDIA Blackwell `sm_120a`, RTX 5090), so the GPU tests assume that device and
**skip-with-reason** when a required device feature, model, or golden is absent —
they never fail for a missing prerequisite.

A prerequisite is *absent* when its env var is **unset**. A var that is **set**
to a path that is not there is a different thing: a misconfiguration, and it
fails loudly (`imp_test::require_readable`). Treating the two alike is what let
`make test-e2e` mount a directory of dangling symlinks and report green having
loaded no model at all.

## Running

The host has no CUDA toolkit by design — everything builds inside Docker
(`make build` → image `imp:test`, CUDA 13.3). See the `building-and-testing`
notes and the root `Makefile`.

```
make build         # build the image (compiles all test binaries)
make test-unit     # CPU-only tests (no GPU)         — ctest -L unit
make test-gpu      # full GPU suite                  — runs every binary
make verify-fast   # ~90s pre-push gate
ctest -L unit|gpu|perf      # on a host build, partition by label
```

Tests are split into per-module binaries so a kernel change relinks only the
affected one:

| binary | scope | GPU? |
|---|---|---|
| `test-core` | tensors, loaders (GGUF/SafeTensors/HF/SPM), config, server transforms (anthropic, SSE, **tool_call**), vision preprocess, **memory subsystem** (backend, tier allocators, `plan_memory`, graph slots — CPU-only via the `fake_backend` seam) | no |
| `test-text` | tokenizers, chat templates, jinja | no |
| `test-compute` | rope, norm, activation, embedding, GEMM/GEMV, softmax, sampling, reduce, **fp8 gemm/gemv** | yes |
| `test-attention` | FA2 / paged / chunked / fp8 / mxfp4 attention, **crosspath parity** | yes |
| `test-quant` | quant round-trip + **dequant/GEMV vs fp64 reference**, CUTLASS grouped GEMM | yes |
| `test-kv` | KV-cache write/gather, FP8 KV, prefix cache, VRAM query/accounting | yes |
| `test-moe-gdn` | MoE routing, GDN/SSM, json-schema FSM | yes |
| `test-e2e` | forward pass, batching, determinism, greedy locks, vision golden, VRAM budget reserve | yes (some CPU-stub) |

Run one binary / filter:

```bash
docker run --rm --gpus all -v $HOME/models:/models imp:test \
  test-quant --gtest_filter='GgufDequant/*:GgufRef.*'
```

### Three-stage gate (CI has no GPU runner)

GPU correctness cannot run in CI (no GPU runner), so the suite is gated in
stages — install the hooks with `make install-hooks`:

- **Stage 1 — pre-commit (GPU), local.** `scripts/pre-commit.hook` runs
  `make test-gpu` (the full GTest suite, which includes the CPU binaries too)
  when staged changes touch `src/ include/ tools/ tests/ CMakeLists/ *.cmake`.
  This is where the kernel oracles below are actually gated. (`pre-push` keeps
  the `make verify-fast` perf + peak-VRAM + smoke regression gate — the VRAM gate
  is the one that fails on a memory regression with flat throughput; it needs `jq`
  and skips with `IMP_VERIFY_SKIP_VRAM=1`.)
- **Stage 2 — CI (CPU).** `.github/workflows/ci.yml` builds everything and runs
  `ctest -L unit` (`test-core` incl. tool-call + Bearer-auth, `test-text`, the
  CPU subset of `test-e2e`) plus the Python mock-API suite. The `gpu`/`perf`
  lanes are skipped (the self-hosted GPU job runs only if `HAS_GPU_RUNNER`).
- **Stage 3 — server (GPU), local, opt-in.** `make test-server`
  (`scripts/test_server.sh`) boots a real `imp-server` against a live model and
  GATES on the OpenAI+Anthropic wire batteries — the only place `handlers.cpp` /
  `batching_engine.cpp` and the SSE protocols run end-to-end (CI's `mock-api`
  suite is a contract stub, not the real handlers). It is opt-in (a model load +
  boot costs minutes), not hooked; run it before relying on a server change.
  Batteries, all hard-gated:
  `exercise_all_endpoints.py` (no 5xx), `test_server_robustness.py` (#712 bad
  input → 4xx+envelope), `test_server_logprobs.py` (logprob sum + top-k
  descending order), `test_server_messages_stream.py` (Anthropic `/v1/messages`
  event sequence), `test_server_thinking_toggle.py` (thinking actually
  disables/enables on both dialects), `test_server_embed_chat_interleave.sh` +
  `test_server_0token_battery.py` (#710 no empty-completion wedge). `make
  coverage` runs the same set ungated to measure `tools/imp-server/` line coverage.

When adding a kernel correctness test, prefer expressing the oracle on the CPU
side where feasible so a regression is reproducible in CI without the 5090.

## Tags / partitions

The CTest registration uses three label aggregates (`CMakeLists.txt`):
`unit` (CPU), `gpu` (needs the 5090), `perf` (`*Perf*`/`*Bench*`/`*Throughput*`
filters). There is no per-test GTest label; the partition is the binary it lives
in (plus the e2e gtest filter `_unit_e2e_filter`, guarded by
`guard_e2e_lane_split` so a rename can't silently move a test to the wrong lane).

Device/weight gating is done in-test:

```cpp
if (sm_ < 120) GTEST_SKIP() << "requires sm_120a";
// SKIP_IF_NO_CUDA(); / SKIP_IF_NO_MODEL();   (see tests/test_cuda_skip.h, test_models.h)
```

A `GTEST_SKIP` that is `sm_`-gated still **runs** on the RTX 5090 target — it
only skips off-target. Don't convert a skip into a hard failure for a missing
model: the suite must stay green on a clean checkout.

## Oracles & tolerances

Correctness tests compare against an **independent** reference, never imp-vs-imp
(which is tautological — a bug shared by both paths passes). In order of
preference:

1. **Closed-form / CPU-naive fp64 reference computed in the test.** Slow but
   trustworthy. Examples: `test_gguf_dequant_ref.cu` re-derives each GGUF block
   format's dequant in fp64 from the format definition; `test_attention_crosspath.cu`
   computes attention in fp64.
2. **cuBLAS** for dense GEMM/attention shapes (already a dependency).
3. **Committed golden tensors** generated by a `tests/refs/` script from a pinned
   model + PyTorch/numpy. The generator command is documented at the top of each
   golden so it is regenerable, not magic (e.g. `tests/refs/gen_*.py`,
   `IMP_VISION_GOLDEN_DUMP=1` for vision).
   `test_attention_crosspath.cu` additionally cross-checks its in-test fp64
   reference against a committed numpy golden at 1e-9 — proving the test and the
   generator compute the same math.

Tolerances are **documented and justified per path** (see the header comment of
each file). Starting points:

| path | bound | rationale |
|---|---|---|
| fp32-accum dequant (decode only) | 1e-3 rel | one f16 round of a 2–3-factor product |
| fp16-dequant GEMV (fp32 dot) | 1e-2 rel (rms-normalized) | fp32-vs-fp64 dot over K + one f16 store |
| dp4a / MMVQ GEMV | 1.5e-2 rms / 5e-2 worst | + Q8_1 activation quant (~0.4% RMS/elem, correlated within a block) |
| FP8 E4M3 decode | exact | every E4M3 value is representable in f16 |
| FP8 E4M3 / NVFP4 MMA | per-block rel-err vs a **dequantized** reference | not vs fp32 ground truth |
| f16 attention vs fp64 | 1e-2 rel | f16 QK + f32 softmax |
| fp8/mxfp4 attention | characterized envelope + **bias guard** | 4-bit score noise; assert no *systematic* bias (signed mean ≈ 0), not just bounded abs error |

The relative metric for a dot product / attention output is normalized by
`rms(reference)`, not per-element — genuine sign cancellation drives some outputs
to ~0 where a per-element ratio explodes without indicating a real error.

## Adding a test

- Add the source to the right binary in `CMakeLists.txt` (under the matching
  `imp_add_test_module(...)`; server-transform tests go under the
  `IMP_BUILD_SERVER` block of `test-core`).
- Make it **deterministic**: fixed seeds, pinned tolerances. A flaky test is
  worse than no test — quarantine and document instead of merging it.
- Use an **independent** oracle; print the measured error as a characterization
  record even when the assertion is loose.
- If a test exposes a real engine bug, **file it** (a GitHub issue; for the memory
  subsystem, append the finding to root `AUDIT.md` under its CONFIRMED / REFUTED /
  OPEN convention) with a
  minimal repro and quarantine the assertion — do not silently patch the kernel
  or weaken the bound to go green.

## Regenerating goldens

```bash
# attention crosspath fp64 golden
python tests/refs/gen_attention_crosspath_golden.py > tests/refs/attention_crosspath_golden.h
# tokenizer parity golden (HF AutoTokenizer) - committed header, not JSON
python tests/refs/gen_tokenizer_golden.py <model_dir> tests/refs/tokenizer_golden_qwen3.h
# vision encoder golden
IMP_VISION_GOLDEN_DUMP=1 make test-vision
```

See `tests/refs/README.md` for the per-golden tolerance/oracle policy.
