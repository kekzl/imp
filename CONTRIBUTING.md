# Contributing to imp

Thanks for taking a look. imp is a single-author / single-target experiment, so contribution overhead is intentionally low — but a few things will save us both time.

## Prerequisites

- An NVIDIA RTX 5090 (Blackwell, `sm_120a`). Other architectures are not supported and will not be added.
- CUDA Toolkit 13.2 or newer (minimum enforced by CMake; the canonical, tested toolchain is 13.3 — what Docker and CI build with).
- CMake 3.25 or newer.
- A C++20 host compiler (GCC 12+, Clang 15+).
- Docker with GPU passthrough if you want to use the canonical build/test workflow.

The host doesn't need any of these directly — `make build` runs everything in `imp:test`, a CUDA 13.3 container with the toolchain pre-installed.

## Build

Inside the container or with the toolchain on the host:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Useful build options:

| Option | Default | Effect |
|---|---|---|
| `IMP_BUILD_TESTS` | `ON` | Build the GTest suite |
| `IMP_BUILD_TOOLS` | `ON` | Build `imp-cli` and `imp-bench` |
| `IMP_BUILD_SERVER` | `ON` | Build `imp-server` |
| `IMP_BUILD_BENCH` | `ON` | Build the benchmark binary |
| `IMP_SANITIZERS` | `OFF` | ASAN + UBSAN (host C++ only) |
| `CMAKE_BUILD_TYPE` | — | `Release` / `RelWithDebInfo` / `Debug` |

## Test

```bash
make test-gpu          # Full CUDA suite (~4-5 min; test-attention alone ~241s)
make test-unit         # CPU-only filter (~5s)
make verify-fast       # Build + filtered tests + perf gate + smoke prompt (~90s)
make verify            # Full pre-merge gate (~5min)
```

`make install-hooks` installs a pre-push hook that runs `verify-fast` whenever `src/`, `include/`, `tools/`, or `tests/` change.

## Benchmark

The CI gate uses `tests/perf_baseline.json` (3% decode / 5% prefill regression thresholds). After an intentional perf change, refresh it:

```bash
scripts/gen_perf_baseline.sh
```

For ad-hoc kernel work, use `imp-cli --bench` or `nsys profile --stats=true` (with `--no-cuda-graphs`, since graph replays hide individual kernel timings):

```bash
./build/imp-cli --model <model>.gguf --bench --bench-pp 512 --bench-reps 5
nsys profile --stats=true ./build/imp-cli --model <model>.gguf \
    --prompt "test" --max-tokens 32 --no-cuda-graphs
```

Prefill numbers vary up to 2.6× across container restarts because of cuBLAS algorithm selection. Decode is the reliable A/B signal.

## Code style

- C++20 host code, CUDA C++20 device code.
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

- **English only.** All PRs (title + body), commit messages, code comments, docs, and `.md` files are written entirely in English. (Deliberate non-English *test data* — tokenizer Unicode fixtures, multilingual probes in `tools/analysis/degen_suite.py` — is exempt and should carry a comment saying so.)
- `#pragma once` in headers (no include guards).
- `.cu` for CUDA, `.cpp` for plain C++, `.h` for headers (CUDA or not).
- File names are `snake_case`. Known intentional exception: the `smallM` fragment (`gemm_grouped_nvfp4_smallM.{h,cu}`) — it mirrors the user-facing config key `moe.nvfp4_smallM`, which can't change without breaking configs.
- Errors return codes (`ImpError` / `bool`); CUDA errors are checked and logged, not thrown.
- Don't add third-party dependencies without a very strong reason — the only runtime deps are the CUDA toolkit, CUTLASS (vendored via FetchContent), and `stb_image` for vision.
- Keep `cudaMalloc` / `cudaFree` out of hot loops — pre-allocate and reuse.
- Don't `__noinline__` GPU inner-loop functions; spills go to local memory and tank performance.

## Commit messages

One concern per commit. The first line is a Conventional-Commits-style summary:

```
fix(nvfp4): clamp encoder output to FP16 range
docs: rewrite README for public release
chore: remove dead code and personal benchmark scripts
```

Body explains *why*, not *what* — the diff already says what changed.

## Pull requests

- Run `make verify-fast` (or `make verify`) before pushing. CI is the source of truth, but failing local first wastes everyone's time.
- For release-touching PRs, `scripts/check-release.sh` runs the same gate plus a doc-link / secret / personal-path scan.
- For perf-sensitive changes, include before/after numbers in the PR description (model, quant, `tg256` and/or `pp512`, hardware).
- Don't reintroduce SM 8.0 / 9.0 / 10.0 code paths. They were removed deliberately and the build pins `arch=compute_120a,code=sm_120a` (see CMakeLists.txt:23).
- Don't break the C API. If a public function in `include/imp/` needs to change, update every caller and call it out in the PR.

## Filing bugs

Useful bug reports include:

- Output of `./build/imp-cli --version` (or the commit SHA).
- Driver version (`nvidia-smi`) and CUDA toolkit version (`nvcc --version`).
- Model identity (file path or HF repo + quantization).
- Exact command that reproduces.
- Whether you ran in Docker or on the host.

For decode quality regressions (degenerate output, repetition loops), the `check-degeneration` workflow in `scripts/` is a useful first triage.

## AI-agent contributions

imp is built end-to-end with [Claude Code](https://claude.ai/claude-code). PRs from AI agents are welcome on the same terms as human PRs, with two extra rules:

- **Read before writing.** The codebase has consistent conventions; follow them rather than inventing parallel ones.
- **Profile, don't guess.** A "performance optimization" that regresses `tg256` is not a performance optimization. Use `nsys` and report numbers.

## License

By contributing, you agree your contribution is licensed under the MIT License (see `LICENSE`).
