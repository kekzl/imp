<!--
layer: L1
audience: operators
verified: 2026-09-22
commit: 9cbb8004
-->

# Contributing to imp

Single-author / single-target project (NVIDIA Blackwell `sm_120a` only) - contribution overhead is intentionally low, but a few things save us both time.

## Prerequisites

- An NVIDIA RTX 5090, RTX PRO 5000 or RTX PRO 6000 Blackwell (`sm_120a`). Other architectures are not supported and will not be added.
- CUDA Toolkit 13.2+ (minimum enforced by CMake); the canonical, tested toolchain is 13.4.1 - what `Dockerfile`'s `imp:toolchain`/`imp:builder` stages and CI build with.
- CMake 3.25+ and a C++23 host compiler (GCC 13+, Clang 16+) - `CMAKE_CXX_STANDARD 23` is required, not a preference. CUDA libs: `cudart`, `cuda_driver`, `cublas`, `cublasLt`.
- Docker with GPU passthrough for the canonical build/test workflow. The host needs none of the above directly - `make build` runs everything in a CUDA 13.4.1 container.
- CUTLASS v4.6.2 and Google Test v1.17.0 fetch automatically via `FetchContent`. `stb_image`/`stb_image_resize2` are vendored in `third_party/stb/`.

## Build

```bash
# Host build (toolchain installed natively)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Docker build (canonical)
make build     # -> imp:test image, full GPU passthrough
make dev       # incremental container build (seconds, Ninja) - iterate here
```

| CMake option | Default | Effect |
|---|---|---|
| `IMP_BUILD_TESTS` | `ON` | GTest suite, 8 module binaries |
| `IMP_BUILD_TOOLS` | `ON` | `imp-cli`, `imp-quantize` |
| `IMP_BUILD_BENCH` | `ON` | `imp-bench` |
| `IMP_BUILD_SERVER` | `ON` | `imp-server` |
| `IMP_SANITIZERS` | `OFF` | ASAN + UBSAN (host C++ only) |
| `IMP_ALLOC_INTERPOSE` | `OFF` | Wrap `cudaMalloc`/`cudaMallocAsync` to attribute steady-state allocations - never benchmark with it on, costs ~3% decode (`AUDIT.md` G16) |
| `IMP_FUZZERS` | `OFF` | libFuzzer binaries from `fuzz/` (clang only) |
| `CMAKE_BUILD_TYPE` | - | `Release` / `RelWithDebInfo` / `Debug` |
| `CMAKE_CUDA_ARCHITECTURES` | `OFF` | pinned instead via raw `--generate-code=arch=compute_120a,code=sm_120a` (+ `compute_120f` PTX fallback), a CMake < 3.31 workaround - don't override it |

## Test

```bash
make dev-test      # CPU unit lane against the dev build (seconds); mirrors CI's `ctest -L unit`
make test-gpu       # Full CUDA suite (~4-5 min; test-attention alone ~241s)
make test-unit      # CPU-only filter against the full-image build (~5s) - a DIFFERENT binary from dev-test/CI
make verify-fast    # Build + filtered tests + perf gate + peak-VRAM gate + smoke prompt
make verify         # Full pre-merge gate (~5 min)
```

`make install-hooks` installs two hooks: pre-commit runs `make test-gpu` when staged changes touch buildable sources; pre-push runs `verify-fast` when `src/`, `include/`, `tools/`, `tests/` or `scripts/` change, with the perf gate only when the diff touches a path that can move it (`src/{compute,exec,quant,runtime,model}/`, any `.cu`/`.cuh`, the build definition, or a baseline). A release always runs everything. Skip a single commit: `git commit --no-verify`.

## Benchmark

The gate uses `tests/perf_baseline.json` (8% decode / 8% prefill regression thresholds, plus a 10% peak-VRAM ceiling over the pinned `metrics.memory_mb.own_peak_mb`). After a change that intentionally moves perf *or* peak VRAM, refresh it:

```bash
scripts/gen_perf_baseline.sh
```

For ad-hoc kernel work: `imp-cli --bench` or `nsys profile --stats=true` (with `--no-cuda-graphs`, since graph replays hide individual kernel timings):

```bash
./build/imp-cli --model <model>.gguf --bench --bench-pp 512 --bench-reps 5
nsys profile --stats=true ./build/imp-cli --model <model>.gguf \
    --prompt "test" --max-tokens 32 --no-cuda-graphs
```

Prefill numbers vary up to 2.6x across container restarts because of cuBLAS algorithm selection. Decode is the reliable A/B signal.

## Code style

- C++23 host code, CUDA C++23 device code.
- Public API in `include/imp/` is C-compatible (`extern "C"`) and treated as stable.
- Internal types live in the `imp::` namespace.

| Element | Convention |
|---|---|
| Classes / structs | `PascalCase` |
| Functions / methods | `snake_case` |
| Member variables | `trailing_underscore_` |
| Constants | `kPascalCase` |
| Enum values | `PascalCase` (`DType::FP16`) |
| C API symbols | `imp_snake_case` |
| Macros | `IMP_UPPER_CASE` |

Other rules:

- **English only.** PRs (title + body), commit messages, code comments, docs and `.md` files. Deliberate non-English *test data* (tokenizer Unicode fixtures, multilingual probes in `tools/analysis/degen_suite.py`) is exempt and should carry a comment saying so.
- `#pragma once` in headers (no include guards).
- `.cu` for CUDA, `.cpp` for plain C++, `.h` for headers (CUDA or not).
- File names are `snake_case`. Known exception: `gemm_grouped_nvfp4_smallM.{h,cu}` mirrors the user-facing config key `moe.nvfp4_smallM`, which can't change without breaking configs.
- Errors return codes (`ImpError` / `bool`); CUDA errors are checked and logged, not thrown.
- Don't add third-party dependencies without a strong reason - the only runtime deps are the CUDA toolkit, CUTLASS (vendored via `FetchContent`) and `stb_image` for vision.
- **Serving allocates nothing** - the measured steady state is `0 cudaMalloc, 0 cudaMallocAsync, 0 pinned-host allocations while serving`. Acquire memory through `src/memory/backend.h` and the tier allocators (`arena`, `block_pool`, `scratch_stack`, `graph_slots`) rather than raw `cudaMalloc`/`cudaFree`, and resolve capacity at init instead of at first use. The blocking `Alloc sites` CI job (`tools/check_alloc_sites.py` against `tools/alloc_allowlist.txt`) rejects new direct allocation sites. See [`docs/internals/MEMORY.md`](docs/internals/MEMORY.md).
- Don't `__noinline__` GPU inner-loop functions; spills go to local memory and tank performance.

## Commit messages

One concern per commit. First line is a Conventional-Commits-style summary:

```
fix(nvfp4): clamp encoder output to FP16 range
docs: rewrite README for public release
chore: remove dead code and personal benchmark scripts
```

Body explains *why*, not *what* - the diff already says what changed.

## Pull requests

- Run `make verify-fast` (or `make verify`) before pushing. CI is the source of truth, but failing local first wastes everyone's time.
- Release-touching PRs: `scripts/check-release.sh` runs the same gate plus a doc-link / secret / personal-path scan.
- Perf-sensitive changes: include before/after numbers in the PR description (model, quant, `tg256` and/or `pp512`, hardware).
- Don't reintroduce SM 8.0 / 9.0 / 10.0 code paths - removed deliberately, the build pins `arch=compute_120a,code=sm_120a` (`IMP_SM120_FLAGS` in `CMakeLists.txt`).
- Don't break the C API. If a public function in `include/imp/` needs to change, update every caller and call it out in the PR.

## Filing bugs

Useful bug reports include: commit SHA (`git rev-parse --short HEAD`), driver version (`nvidia-smi`) and CUDA toolkit version (`nvcc --version`), model identity (file path or HF repo + quantization), the exact command that reproduces, and whether you ran in Docker or on the host.

For decode quality regressions (degenerate output, repetition loops), the `check-degeneration` workflow in `scripts/` is a useful first triage.

## AI-agent contributions

imp is built end-to-end with [Claude Code](https://claude.ai/claude-code). PRs from AI agents are welcome on the same terms as human PRs, with two extra rules:

- **Read before writing.** The codebase has consistent conventions; follow them rather than inventing parallel ones.
- **Profile, don't guess.** A "performance optimization" that regresses `tg256` is not a performance optimization. Use `nsys` and report numbers.

## License

By contributing, you agree your contribution is licensed under the MIT License (see `LICENSE`).
