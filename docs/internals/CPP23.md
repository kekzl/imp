<!--
layer: L2
audience: kernel-devs
verified: 2026-09-22
commit: 9cbb8004
-->

# C++23 in imp

The build targets C++23 since 2026-07-08 (`CMakeLists.txt:4-16`, migration record in [`../archive/cpp23_migration_2026_07_08.md`](../archive/cpp23_migration_2026_07_08.md)).

- This file says which of the language the tree actually uses and where the line runs between host and device code.
- The 2026-07-29 architecture audit called it "C++17 with C++23 spelling ... nvcc constrains what is usable in `.cu`"; the first half was accurate, the second half was never measured and is false.

## What nvcc 13.3 actually accepts

Probed against the production toolchain (nvcc 13.3.33 / GCC 15.2, `imp:toolchain`, the real `CMAKE_CUDA_FLAGS` including `--expt-relaxed-constexpr`), compiling and running on `sm_120a`.

Compiled in a `.cu`, launched on the card, output checked:

| in `__device__` code | |
|---|---|
| `std::span`, `std::array`, `std::bit_cast`, `std::to_underlying` | works |
| concepts / `requires` on a device template | works |
| multidimensional `operator[](r, c)` | works |
| deducing this (`this auto&& self`) | works |
| static `operator()`, `[[assume]]`, `1uz`, `auto(x)` | works |

One kernel using all of the above compiled with **zero diagnostics** under the production flags and produced the expected values on an RTX 5090.

Compiled and run on the host, both as `.cpp` and as the host side of a `.cu` (identical results, so nvcc's host path is not the constraint):

| | |
|---|---|
| `std::expected`, `std::format`, `std::print` | works |
| `std::ranges::to`, `views::zip`, `views::enumerate`, `views::chunk` | works |
| `std::byteswap`, `std::unreachable`, `std::flat_map`, `std::generator`, `std::stacktrace`, `std::move_only_function`, `std::spanstream`, `std::out_ptr` | present |
| `if consteval`, `__cpp_size_t_suffix`, `__cpp_auto_cast` | works |
| `std::mdspan` | **absent** |
| `std::start_lifetime_as` | **absent** |

Not probed in device code, and not used there: the ranges views, `std::format`, the allocating or OS-backed containers. Host facilities with no reason to appear in a kernel, so the first table is "measured to work on the card", not "everything that could".

The two absences are libstdc++ 15.2 gaps, not language ones.

- No `std::mdspan` → tensor views stay hand-rolled.
- No `std::start_lifetime_as` → reading a POD out of a mapped byte buffer stays `std::memcpy` rather than `std::bit_cast` (bit_cast needs an object on both sides; a `const uint8_t*` into an mmap is not one).

## The rules

| Use | For | Detail |
|---|---|---|
| `std::expected<T, E>` | errors a caller must handle | replaces `bool f(..., T& out, std::string& err)`, whose contract lived in a comment and depended on every caller reading the bool - a half-filled result is not a value that exists with `expected`. Where the refusal carries more than a sentence, the error type is a struct (`Qwen3VLVisionLoadError` carries the counts the caller logs) |
| `std::optional<T>` | absence that is not an error | `log_level_from_string` returns `nullopt` for an unknown word: a value the caller does not have, not a failure it reports |
| `std::span<T>` | host buffers | any host-side (pointer, length) pair in a C++ signature is a span; removes a callable state - `ngram_draft(nullptr, 6, ...)` and `SuffixDraftIndex::append(nullptr, 5)` were real call shapes needing a defensive null check, both tests exercising them are gone because the state is now unrepresentable |
| raw `const half*` + extent, NOT `std::span` | device pointers | a `std::span` says "you may index and iterate this"; on a device pointer that is a silent host segfault at the first `s[0]`. The one place the C++17 shape is correct - not a defect to fix wholesale |
| `std::string_view` | strings a function only reads | except where the callee needs a null-terminated `c_str()` for a C API, where `const std::string&` stays and says so. Rule for new/touched code only: the 591 existing `const std::string&` parameters have NOT been swept |
| `std::format`, not `snprintf` into a fixed buffer | building a string | the memory-plan failure report was seven `snprintf` calls into one `char buf[256]` whose truncation only a user with a refused engine would see |
| printf-style, NOT `std::format` | logging | `IMP_LOG_*` is a variadic macro over `log_message(..., const char* fmt, ...)` with `__attribute__((format(printf)))`, across 1431 call sites; format-string checking is already compile-time, `std::format` would buy type safety at the cost of touching every site. Deliberately not converted |
| `std::bit_cast` | bit patterns | constexpr, so a conversion can be checked by `static_assert` instead of a test run (`src/core/fp_bits.h` does exactly that) |
| exceptions, unchanged | internal error propagation | out of scope here: internal code throws, `src/api/imp_api.cpp` translates to `ImpError` at the C ABI boundary; `expected` is for the layers below that boundary that returned a bool |
| NOT `std::unreachable()` / `[[assume]]` | the two branches commented "statically unreachable" (`engine_decode_pipeline.cpp`, split out of `engine_scheduler.cpp` 2026-08-26) | both carry a safe fallback (log and abandon the half-enqueued step, or re-run the row through the legacy collect path); replacing a fallback with undefined behaviour is a bet that the comment is right |

## What stays C ABI

`include/imp/imp.h` is a C header: raw pointers, lengths, `ImpError` returns, no templates.

- None of these rules apply to it.
- The span is constructed one level in, in `src/api/imp_api_vision.cpp`.

## Where this is done, and where it is not

**`bool f(..., std::string& err)` no longer exists anywhere in `src/`, `tools/` or `include/`.** Was 36 sites, 15 of them header declarations, across the Qwen3-VL loader chain, image placeholder expansion, M-RoPE position building and the whole `imp-quantize` surface; `grep -rIn "std::string& err[,)]" src tools include` returns nothing (plain `"std::string& err"` also matches `error()`/`error_reason()` accessors - false positives, not the out-param pattern).

Spans replaced the host (pointer, length) pairs in the drafters (`ngram_draft`, `SuffixDraftIndex`, `TokenRecycleTable`), the image byte path from the C ABI down to `stbi_load_from_memory`, `BatchBuilder`'s token and block tables, the perplexity and encoder-embed entry points, `BinaryReader`, `make_weight_key` and (as `string_view`) `JsonParser` and `log_level_from_string`.

Not converted, each for a reason:

- kernel launch wrappers: a span over a device pointer is a lie;
- `IMP_LOG_*`: the printf attribute already checks it, the change is 1431 call sites wide;
- tensor views: libstdc++ 15.2 has no `std::mdspan`;
- the *reads inside* `BinaryReader` and the SafeTensors header parse stay `std::memcpy`: no `std::start_lifetime_as` to give a POD a lifetime inside a mapped byte buffer (the constructors now take spans);
- `GGUFValue` (a tag plus seven always-present payload fields `std::variant` would express as one): a data-structure change with its own blast radius, not a signature change.
