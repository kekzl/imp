<!--
layer: L2
audience: kernel-devs
verified: 2026-08-21
commit: 551011a3
-->

# C++23 in imp

The build has targeted C++23 since 2026-07-08 (`CMakeLists.txt:4-16`, migration
record in [`../archive/cpp23_migration_2026_07_08.md`](../archive/cpp23_migration_2026_07_08.md)).
This file says which of the language it actually uses, and where the line runs
between host and device code. It exists because the split used to be undocumented
and the answer people assumed was wrong: the 2026-07-29 architecture audit put
it as "C++17 with C++23 spelling ... a defensible choice for CUDA host code,
nvcc constrains what is usable in `.cu`". The first half was accurate. The second
half was never measured, and it is false.

## What nvcc 13.3 actually accepts

Probed against the production toolchain (nvcc 13.3.33 / GCC 15.2, `imp:toolchain`,
the real `CMAKE_CUDA_FLAGS` including `--expt-relaxed-constexpr`), compiling and
running on `sm_120a`:

Compiled in a `.cu`, launched on the card, output checked:

| in `__device__` code | |
|---|---|
| `std::span`, `std::array`, `std::bit_cast`, `std::to_underlying` | works |
| concepts / `requires` on a device template | works |
| multidimensional `operator[](r, c)` | works |
| deducing this (`this auto&& self`) | works |
| static `operator()`, `[[assume]]`, `1uz`, `auto(x)` | works |

One kernel using all of the above compiled with **zero diagnostics** under the
production flags and produced the expected values on an RTX 5090.

Compiled and run on the host, both as `.cpp` and as the host side of a `.cu`
(identical results, so nvcc's host path is not the constraint):

| | |
|---|---|
| `std::expected`, `std::format`, `std::print` | works |
| `std::ranges::to`, `views::zip`, `views::enumerate`, `views::chunk` | works |
| `std::byteswap`, `std::unreachable`, `std::flat_map`, `std::generator`, `std::stacktrace`, `std::move_only_function`, `std::spanstream`, `std::out_ptr` | present |
| `if consteval`, `__cpp_size_t_suffix`, `__cpp_auto_cast` | works |
| `std::mdspan` | **absent** |
| `std::start_lifetime_as` | **absent** |

Not probed in device code, and not used there: the ranges views, `std::format`
and the allocating or OS-backed containers. They are host facilities and the
tree has no reason to want them in a kernel, so the first table is "what was
measured to work on the card", not "everything that could".

The two absences are libstdc++ 15.2 gaps, not language ones. `std::mdspan` is why
tensor views here are still hand-rolled; `std::start_lifetime_as` is why reading a
POD out of a mapped byte buffer still goes through `std::memcpy` rather than
`std::bit_cast` (bit_cast needs an object on both sides, and a `const uint8_t*`
into an mmap is not one).

## The rules

**Errors that a caller must handle: `std::expected<T, E>`.** The pattern it
replaces is `bool f(..., T& out, std::string& err)`, whose contract lived in a
comment ("returns false, leaving `out` UNTOUCHED") and depended on every caller
reading the bool before the output. With `expected` a half-filled result is not a
value that exists. Where the refusal carries more than a sentence, the error type
is a struct: `Qwen3VLVisionLoadError` carries the counts the caller logs.

**Absence that is not an error: `std::optional<T>`.** `log_level_from_string`
returns `nullopt` for a word it does not know; that is not a failure the caller
reports, it is a value it does not have.

**Host buffers: `std::span<T>`.** Any host-side (pointer, length) pair in a C++
signature is a span. This removes a state the old signatures could be called in:
`ngram_draft(nullptr, 6, ...)` and `SuffixDraftIndex::append(nullptr, 5)` were
both real call shapes that needed a defensive null check, and both tests that
exercised them are gone because the state is unrepresentable.

**Device pointers are NOT spans.** A `std::span` says "you may index and iterate
this", and on a device pointer that is a segfault on the host, silently, at the
first `s[0]`. Kernel launch wrappers keep raw `const half*` + extent, and the
extent stays a separate parameter. This is the one place the C++17 shape is the
correct one, and it is the reason the audit's "span is not used for the raw
pointer + length pairs that dominate every kernel launch wrapper" is not a defect
to fix wholesale.

**Strings a function only reads: `std::string_view`.** Except where the callee
needs a null-terminated `c_str()` for a C API, in which case `const std::string&`
stays and says so. This is the rule for new and touched code: the 591 existing
`const std::string&` parameters have NOT been swept, and doing that is a
separate change that has to look at each callee.

**Building a string: `std::format`.** Not `snprintf` into a fixed buffer. The
memory-plan failure report was seven `snprintf` calls into one `char buf[256]`,
whose truncation nobody would see except a user whose engine had just refused to
start.

**Logging stays printf-style.** `IMP_LOG_*` is a variadic macro over
`log_message(..., const char* fmt, ...)` with `__attribute__((format(printf)))`,
across 1431 call sites. The format-string checking is already
compile-time; `std::format` would buy type safety over a hazard the attribute
already covers, at the cost of touching every site. Deliberately not converted.

**Bit patterns: `std::bit_cast`.** It is constexpr, so a conversion written with
it can be checked by `static_assert` instead of by a test run
(`src/core/fp_bits.h` does exactly that).

**Exceptions.** Unchanged and out of scope here: internal code throws, and
`src/api/imp_api.cpp` translates to `ImpError` at the C ABI boundary. `expected`
is for the layers below that boundary that already returned a bool.

**`std::unreachable()` is deliberately absent.** The tree has exactly two
branches commented "statically unreachable" (`engine_scheduler.cpp:2747` and
`:2801`), and both carry a safe fallback: log and abandon the half-enqueued
step, or re-run the row through the legacy collect path. Replacing a fallback
with undefined behaviour is not a modernisation, it is a bet that the comment is
right. `[[assume]]` is absent for the same reason.

## What stays C ABI

`include/imp/imp.h` is a C header: raw pointers, lengths, `ImpError` returns, no
templates. Nothing in this document applies to it. The span is constructed one
level in, in `src/api/imp_api_vision.cpp`.

## Where this is done, and where it is not

**`bool f(..., std::string& err)` no longer exists anywhere in `src/`, `tools/`
or `include/`.** It was 36 sites, 15 of them declarations in headers, across the
Qwen3-VL loader chain, image placeholder expansion, M-RoPE position building and
the whole `imp-quantize` surface; `grep -rIn "std::string& err" src tools
include` now returns nothing.

Spans replaced the host (pointer, length) pairs in the drafters
(`ngram_draft`, `SuffixDraftIndex`, `TokenRecycleTable`), the image byte path
from the C ABI down to `stbi_load_from_memory`, `BatchBuilder`'s token and
block tables, the perplexity and encoder-embed entry points, `BinaryReader`,
`make_weight_key` and (as `string_view`) `JsonParser` and
`log_level_from_string`.

Not converted, and each for a reason rather than for lack of time:

- the kernel launch wrappers, because a span over a device pointer is a lie;
- `IMP_LOG_*`, because the printf attribute already checks it and the change is
  1431 call sites wide;
- the tensor views, because libstdc++ 15.2 has no `std::mdspan`;
- the *reads inside* `BinaryReader` and the SafeTensors header parse, which stay
  `std::memcpy` because there is no `std::start_lifetime_as` to give a POD a
  lifetime inside a mapped byte buffer (the constructors now take spans);
- `GGUFValue`, a tag plus seven always-present payload fields that
  `std::variant` would express as one, because that is a data-structure change
  with its own blast radius rather than a signature change.
