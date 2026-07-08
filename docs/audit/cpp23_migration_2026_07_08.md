# C++23 migration audit (2026-07-08)

Question: where can moving imp from C++20 → C++23 break? Every finding below was
**verified against the real toolchain / code**, not inferred. Verification used
nvcc 13.3 (V13.3.33) + GCC 15.2 (the production `nvidia/cuda:13.3.0-devel-ubuntu26.04`
base), compiling the actual TUs with the real flags from `build/compile_commands.json`.

## TL;DR

Migration is **low-risk and technically clear**. The two things that could have been
show-stoppers — nvcc dialect support and CUTLASS-under-c++23 — both pass a live compile.
No `-Werror`, so any residual C++23 deprecation is a warning, not a build break.
The only real gotcha is a **toolchain-version coupling** (below).

## Verified: no blockers

| Check | Result | How verified |
|---|---|---|
| nvcc 13.3 accepts `-std=c++23` | YES | `nvcc --help` lists `c++23`; forwards dialect to host compiler |
| CUTLASS v4.5.2 device TUs under c++23 | **all 5 compile clean** (0 err / 0 warn) | live `nvcc -std=c++23 -c` of `gemm_cutlass_sm120`, `gemm_cutlass_mxfp4_sm120`, `gemm_cutlass_grouped_3x`, `gemm_grouped_nvfp4_smallM`, `attention_fmha_sm120` w/ GCC 15.2 host |
| First-party host TUs under c++23 | clean, **zero new diagnostics** vs c++20 | `g++-15 -std=gnu++23 -fsyntax-only -Wall -Wextra -Wpedantic` on tensor/model/config/vision/lora TUs |
| `--expt-relaxed-constexpr` set | YES (`cmake/CompilerFlags.cmake:17`) | covers C++23's expanded `constexpr` stdlib in device code |
| CMake supports `CUDA_STANDARD 23` | YES | Dockerfile installs CMake **4.3.1** (`Dockerfile:24`) |
| Removed features in use (`std::aligned_storage`/`aligned_union`, `<codecvt>`, `unary_function`, `throw()` dyn-spec, GC `declare_reachable`) | **none present** | grep `src/ include/ tools/ tests/` |
| `u8""` / `char8_t` literals | none | grep |
| `[=]` implicit-`this` captures (deprecated) | **none** — code uses explicit captures | grep (0 hits) |
| `volatile` compound assignment (deprecated) | none | grep (83 `volatile` uses, all plain load/store in kernels) |
| `-Werror` anywhere | **no** — only `-Wall -Wextra -Wpedantic`, CXX-only, not nvcc, not deps (`CMakeLists.txt:18-20`) | grep CMake + `.github/` |

## The actual blocker (found during migration) — CMake has no CUDA23 dialect

`set(CMAKE_CUDA_STANDARD 23)` fails at configure time:

```
Target "imp" requires the language dialect "CUDA23".  But the current compiler
"NVIDIA" does not support this, or CMake does not know the flags to enable it.
```

Root cause is **CMake, not the toolchain**: `Compiler/NVIDIA.cmake` in CMake 4.3.1
(`__compiler_nvidia_cxx_standards`) defines `..._STANDARD_COMPILE_OPTION` only up to
c++20 — there is no `CMAKE_CUDA23_STANDARD_COMPILE_OPTION`. nvcc 13.3 *does* accept
`-std=c++23` (verified), CMake just doesn't emit it for the CUDA language yet. Fix is
to teach CMake the flag before targets are defined (in `CMakeLists.txt`):

```cmake
if(NOT DEFINED CMAKE_CUDA23_STANDARD_COMPILE_OPTION)
  set(CMAKE_CUDA23_STANDARD_COMPILE_OPTION  "-std=c++23")
  set(CMAKE_CUDA23_EXTENSION_COMPILE_OPTION "-std=c++23")
endif()
```

Verified in a minimal CUDA project (configure + compile of a c++23 template-lambda TU
pass) and in the full imp build. `CMAKE_CUDA_EXTENSIONS OFF` alone does **not** fix it
(there's no non-extension CUDA23 mapping either). Remove the shim once the CMake module
ships a native CUDA23 entry.

## The other gotcha — host-compiler coupling

`nvcc --std=c++23` is **silently ignored** if the host compiler is judged too old.
Observed live: on the stale `impdev:ncu` image (GCC 13.3 / Ubuntu 24.04), nvcc emits

```
nvcc warning : The -std=c++23 flag is not supported with the configured host compiler. Flag will be ignored.
```

and compiles as the default dialect **without failing**. The production builder
(Ubuntu 26.04 / GCC 15.2) is fine. Implication: bumping the standard requires the
**profiling / ncu image (`impdev:ncu`) to be rebuilt on the 26.04/GCC-15 base too**,
or profiling/roofline builds quietly compile non-c++23 and diverge from CI. This is
the single action item; it's the same "impdev:ncu stale" trap already noted in memory.

## What the migration actually is

1. `CMakeLists.txt:4-7` — flip `CMAKE_CXX_STANDARD 20`→`23` and `CMAKE_CUDA_STANDARD 20`→`23`.
2. Rebuild `impdev:ncu` (and any other dev/profiling image) on `ubuntu26.04` / GCC 15.2.
3. Full `make build` + `make test-unit` / `test-gpu` (this audit compiled representative
   TUs, not the whole tree — the residual unknown is only the ~110 host TUs and ~200 `.cu`
   not individually sampled; risk is low given no `-Werror` and c++23 ≈ superset here).

## Residual low-risk items to watch during the full build

- **P2266 implicit-move on return** (c++23 hardens `return local;` to move): can turn a
  copy-only-from-lvalue local return into an error. None seen in sampled TUs; a full
  build is the real check. Hard error if it hits, not silent.
- **libstdc++15 header transitivity**: GCC 15 already tightened this at c++20 (the
  `<algorithm>`/`<numeric>` sweep, PR #906). c++23 mode may expose a few more missing
  includes → hard error, trivial fix.
- CUTLASS deprecation warnings under c++23 would be **invisible anyway** — deps and nvcc
  TUs are not compiled with `-Wall` (`imp_warnings` is CXX-only, first-party-only).

## Refuted (do not re-chase)

- "nvcc can't do c++23" — false, 13.3 supports it.
- "CUTLASS 4.5.2 won't build under c++23" — false, all 5 TUs clean.
- "new C++23 warnings will break the build" — false, no `-Werror`.
