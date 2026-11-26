#!/usr/bin/env python3
"""Lazy device statics must re-arm across an engine teardown.

WHY THIS EXISTS
---------------
Several TUs hold file- or function-scope statics (arena slices, cudaMalloc'd
scratch, cuBLAS handles) behind a lazy `if (!ptr)` / capacity guard. ~Engine
closes the T2 arena and imp_gpu_release() resets the device; both then run the
hooks registered with IMP_REGISTER_CUDA_STATIC_RESET (core/cuda_static_reset.h)
so every such guard re-arms. A TU that holds such a static and registers no
hook keeps its guard armed over released memory, and the SECOND engine in the
process (server.model_swap, default on) uses it. That is a silent device
use-after-free: no IMP_CUDA_CHECK sees a write into a re-mapped region.

Registration is a convention. Measured 2026-09-05 (AUDIT_ARCH_2026 B-1..B-3):
15 TUs registered, 6 did not, one of them the logit_bias arena slots that every
`logit_bias` request after a model swap wrote through.

WHAT IT CHECKS
--------------
For each TU under src/ (outside src/memory/, which owns the driver):

  candidate  = declares a file-scope or function-scope static, or a file-scope
               `g_*` / `s_*` global, whose type is a pointer, a CUDA/cuBLAS
               handle, or a class instance (not std::, not a primitive)
  acquires   = takes from engine_arena() or calls cudaMalloc* / cublasCreate /
               cublasLtCreate / cublas(lt)_handle_create / cudaStreamCreate /
               cudaEventCreate / cudaGraphInstantiate
  re-arms    = IMP_REGISTER_CUDA_STATIC_RESET( in the TU, or an
               engine_arena().generation() check (the other accepted pattern:
               the tenant compares the arena generation instead of trusting
               its cached pointer)

A TU that is a candidate AND acquires AND does not re-arm fails unless it is
in tools/static_reset_allowlist.txt with a reason. A listed TU that no longer
matches is a stale entry and fails too, so the list cannot rot either way.

Per NAME (AUDIT_ARCH_2026_11_24 X-7: one hook used to pass a TU however many
statics it held). In a TU that acquires and re-arms, every candidate static
must be covered:

  by the hook   = its name occurs in the body of a function registered with
                  IMP_REGISTER_CUDA_STATIC_RESET, or of a function of the same
                  TU that the hook calls (transitively)
  by the arena  = its name occurs in a function of the TU that reads
                  engine_arena().generation(), or in one that calls such a
                  function directly (the caller of a `valid()` helper), or its
                  type is a struct of the TU with that read in a member

An uncovered static fails unless the allowlist carries `<path>::<name>` with a
reason. A static that another TU resets names the function:
`<path>::<name>  reset-by=<function>  <reason>`; the function must be defined
under src/ outside that TU. A name entry is stale when the static is gone or
covered, or when its reset-by function no longer exists.

Log-once latches (AUDIT_ARCH_2026_11_24 C-12, D-10): a `static bool`, `std::atomic<bool>`
or `std::once_flag` named *logged* / *warned* / *once* / *said* / *noted* / *reported*
anywhere under src/ stays set across a model swap, so model B never logs the fallback model
A tripped. Use core/log_once.h `LogOnce`, which every model load re-arms. A remaining one
needs `<path>::<name>  log-once: <reason>` in the allowlist; a listed latch that is gone fails.

Usage:
    python3 tools/check_static_reset.py             # check (CI)
    python3 tools/check_static_reset.py --list      # every candidate TU + status
    python3 tools/check_static_reset.py --selftest  # planted cases
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

from check_pdl_registration import balanced, strip_comments

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src"
ALLOWLIST = REPO / "tools" / "static_reset_allowlist.txt"

EXEMPT_PREFIXES = ("src/memory/",)
SUFFIXES = (".cu", ".cpp")

ACQUIRE = re.compile(
    r"\b(engine_arena\(\)\s*\.\s*take(_bytes)?\s*\(|cudaMalloc\w*\s*\(|cublasCreate|cublasLtCreate"
    r"|cublas(lt)?_handle_create\s*\(|cudaStreamCreate|cudaEventCreate|cudaGraphInstantiate)"
)
REARM_HOOK = re.compile(r"\bIMP_REGISTER_CUDA_STATIC_RESET\s*\(")
REARM_GEN = re.compile(r"\bengine_arena\(\)\s*\.\s*generation\s*\(")
HOOK_ARG = re.compile(r"\bIMP_REGISTER_CUDA_STATIC_RESET\s*\(\s*(\w+)\s*\)")
RESET_BY = re.compile(r"reset-by=(\w+)")
NAMESPACE_HEAD = re.compile(r"(?:\bnamespace\b[^()=;]*|\bextern\s*\"[^\"]*\"\s*)$")
FUNC_NAME = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
CALL = FUNC_NAME
TYPE_HEAD = re.compile(r"\b(?:struct|class)\s+(\w+)[^;{()]*\{")
NOT_A_NAME = {"__launch_bounds__", "__attribute__", "alignas", "decltype", "noexcept", "requires"}

# `static <type> <name> [= init];` at any indentation. Function declarations
# and definitions have a `(` before any `=`, which the initializer check drops.
STATIC_DECL = re.compile(
    r"^\s*static\s+(?!constexpr\b|const\b|inline\b|thread_local\b|__device__|__constant__|__global__"
    r"|__shared__|__forceinline__|__host__)(?P<type>[\w:<>,\s*&]+?)\s+(?P<stars>\**)\s*(?P<name>\w+)"
    r"\s*(?P<init>=[^;]*|\{[^;]*\}|\[[^;]*)?;"
)
# File-scope globals by the repo's own naming: `g_` / `s_` at column 0
# (anonymous-namespace globals are not `static` and would otherwise be missed;
# ffn_sparsity_probe.cu's `ProbeState g_state;` was one of the six).
GLOBAL_DECL = re.compile(
    r"^(?P<type>[\w:]+(?:<[^;]*>)?(?:\s*\*+)?)\s+(?P<stars>\**)\s*(?P<name>[gs]_\w+)\s*"
    r"(?P<init>=[^;]*|\{[^;]*\})?;"
)
HANDLE_TYPES = re.compile(
    r"\b(cublasHandle_t|cublasLtHandle_t|cudaStream_t|cudaEvent_t|cudaGraphExec_t|cudaGraph_t"
    r"|CUtensorMap|cudnnHandle_t)\b"
)
LOG_ONCE_LATCH = re.compile(
    r"^\s*static\s+(?:std::atomic<bool>|std::atomic_bool|bool|std::once_flag)\s+"
    r"(?P<name>\w*(?:logged|warned|once|said|noted|reported)\w*)\b",
    re.M,
)
LOG_ONCE_TAG = "log-once:"
PRIMITIVES = re.compile(
    r"^(unsigned\s+)?(int|bool|float|double|size_t|char|long|short|uint\d+_t|int\d+_t|uintptr_t"
    r"|ptrdiff_t|std::\w+(<.*>)?)$"
)


def device_relevant(typ: str, stars: str) -> bool:
    """A pointer, a CUDA handle, or a class instance; never a primitive or std::."""
    t = " ".join(typ.split())
    if stars or "*" in t:
        return True
    if HANDLE_TYPES.search(t):
        return True
    if PRIMITIVES.match(t):
        return False
    head = t.split("<", 1)[0].split("::")[-1]
    return head[:1].isupper() and head != "LogOnce"  # host latch, core/log_once.h re-arms it


def candidates(text: str) -> list[str]:
    """Names of the lazy-static declarations a TU holds (empty = not a candidate)."""
    found = []
    for line in text.splitlines():
        stripped = line.split("//", 1)[0]
        m = STATIC_DECL.match(stripped) or GLOBAL_DECL.match(stripped)
        if not m:
            continue
        init = m.group("init") or ""
        # `static int foo(int x);` and `static void bar() {` are functions.
        head = stripped.split("=", 1)[0] if "=" in init else stripped
        if "(" in head:
            continue
        if device_relevant(m.group("type"), m.group("stars")):
            found.append(m.group("name"))
    return found


def classify(text: str) -> tuple[str, list[str]]:
    """-> (status, names). status: 'ok' | 'not-candidate' | 'no-rearm'."""
    names = candidates(text)
    if not names or not ACQUIRE.search(text):
        return "not-candidate", names
    if REARM_HOOK.search(text) or REARM_GEN.search(text):
        return "ok", names
    return "no-rearm", names


def function_bodies(text: str) -> list[tuple[str, str]]:
    """(name, body) of every namespace-scope function definition in comment-stripped text."""
    out = []
    i, n, mark = 0, len(text), 0
    while i < n:
        ch = text[i]
        if ch in ";}":
            mark = i + 1
        elif ch == "{":
            header = "\n".join(ln for ln in text[mark:i].splitlines() if not ln.lstrip().startswith("#"))
            if NAMESPACE_HEAD.search(header):
                mark = i + 1  # transparent: its members are namespace scope too
            else:
                end = balanced(text, i, "{", "}")
                if end < 0:
                    break
                names = [nm for nm in FUNC_NAME.findall(header) if nm not in NOT_A_NAME]
                if names and "=" not in header.split("(", 1)[0]:
                    out.append((names[0], text[i:end]))
                i = mark = end
                continue
        i += 1
    return out


def uncovered(text: str, names: list[str]) -> list[str]:
    """The candidate names neither a registered hook nor an arena-generation check reaches."""
    code = strip_comments(text)
    funcs = function_bodies(code)
    bodies: dict[str, list[str]] = {}
    for name, body in funcs:
        bodies.setdefault(name, []).append(body)
    reach: list[str] = []
    todo = sorted(set(HOOK_ARG.findall(code)))
    seen: set[str] = set()
    while todo:
        fn = todo.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for body in bodies.get(fn, []):
            reach.append(body)
            todo.extend(sorted(c for c in set(CALL.findall(body)) if c in bodies and c not in seen))
    gen_funcs = {name for name, body in funcs if REARM_GEN.search(body)}
    for name, body in funcs:
        if name in gen_funcs or gen_funcs & set(CALL.findall(body)):
            reach.append(body)
    covered = "\n".join(reach)
    # A static whose own type checks the generation in a member function (CubSortScratch::ensure).
    gen_types = set()
    for m in TYPE_HEAD.finditer(code):
        end = balanced(code, m.end() - 1, "{", "}")
        if end > 0 and REARM_GEN.search(code[m.end():end]):
            gen_types.add(m.group(1))

    def by_type(nm: str) -> bool:
        return any(re.search(r"\b" + re.escape(t) + r"\s+" + re.escape(nm) + r"\b", code) for t in gen_types)

    return [nm for nm in names
            if not re.search(r"\b" + re.escape(nm) + r"\b", covered) and not by_type(nm)]


def scan() -> dict[str, tuple[str, list[str], list[str]]]:
    """rel -> (status, candidate names, names no hook and no generation check covers)."""
    out = {}
    for p in sorted(SRC.rglob("*")):
        if p.suffix not in SUFFIXES or not p.is_file():
            continue
        rel = p.relative_to(REPO).as_posix()
        if rel.startswith(EXEMPT_PREFIXES):
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        status, names = classify(text)
        if names:
            out[rel] = (status, names, uncovered(text, names) if status == "ok" else [])
    return out


def read_allowlist() -> dict[str, str]:
    allowed = {}
    if not ALLOWLIST.exists():
        return allowed
    for raw in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        path, _, reason = line.partition("  ")
        allowed[path.strip()] = reason.strip()
    return allowed


def defined_outside(function: str, rel: str) -> bool:
    """True when `function` is defined in a src/ file other than `rel`."""
    call = re.compile(r"\b" + re.escape(function) + r"\s*\(")
    for p in sorted(SRC.rglob("*")):
        if p.suffix not in SUFFIXES + (".h", ".cuh") or not p.is_file():
            continue
        if p.relative_to(REPO).as_posix() == rel:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if call.search(text) and any(nm == function for nm, _ in function_bodies(strip_comments(text))):
            return True
    return False


def name_verdict(results: dict[str, tuple[str, list[str], list[str]]], allowed: dict[str, str],
                 defined=defined_outside) -> list[str]:
    """Failure lines of the per-name rule; `allowed` holds the `<path>::<name>` entries."""
    fails = []
    for rel, (_, _, missing) in sorted(results.items()):
        for name in missing:
            if f"{rel}::{name}" not in allowed:
                fails.append(f"{rel}::{name}: lazy device static that no registered hook and no "
                             f"engine_arena().generation() check of its TU reaches")
    for key, reason in sorted(allowed.items()):
        rel, _, name = key.partition("::")
        if name not in results.get(rel, ("", [], []))[2]:
            fails.append(f"{key}: stale allowlist entry (the static is gone, or its TU now resets it)")
            continue
        m = RESET_BY.match(reason)
        if m and not defined(m.group(1), rel):
            fails.append(f"{key}: stale allowlist entry, reset-by={m.group(1)} is not defined under src/ "
                         f"outside {rel}")
        if not (reason[m.end():] if m else reason).strip():
            fails.append(f"{key}: allowlist entry without a reason")
    return fails


def log_once_latches(text: str) -> list[str]:
    """Names of the process-static log-once flags in comment-stripped text."""
    return sorted({m.group("name") for m in LOG_ONCE_LATCH.finditer(strip_comments(text))})


def scan_log_once() -> dict[str, list[str]]:
    out = {}
    for p in sorted(SRC.rglob("*")):
        if p.suffix not in SUFFIXES + (".h", ".cuh") or not p.is_file():
            continue
        names = log_once_latches(p.read_text(encoding="utf-8", errors="replace"))
        if names:
            out[p.relative_to(REPO).as_posix()] = names
    return out


def log_once_verdict(found: dict[str, list[str]], allowed: dict[str, str]) -> list[str]:
    """Failure lines: an unlisted latch, a listed one that is gone, a listed one without a reason."""
    fails = [f"{rel}::{nm}: process-static log-once flag, use core/log_once.h LogOnce (re-armed per model load)"
             for rel, names in sorted(found.items()) for nm in names if f"{rel}::{nm}" not in allowed]
    for key, reason in sorted(allowed.items()):
        rel, _, name = key.partition("::")
        if name not in found.get(rel, []):
            fails.append(f"{key}: stale allowlist entry (the log-once flag is gone)")
        elif not reason[len(LOG_ONCE_TAG):].strip():
            fails.append(f"{key}: allowlist entry without a reason")
    return fails


def check_log_once(allowed: dict[str, str]) -> tuple[int, str]:
    """-> (failures, summary) of the log-once rule; prints the failure lines."""
    found = scan_log_once()
    fails = log_once_verdict(found, allowed)
    if fails:
        print("check_static_reset: log-once rule (C-12/D-10):")
        for line in fails:
            print(f"  {line}")
    n = sum(len(v) for v in found.values())
    return len(fails), f"log-once flags: {n}, listed {len(allowed)}"


def check(list_all: bool) -> int:
    results = scan()
    everything = read_allowlist()
    allowed = {k: v for k, v in everything.items() if "::" not in k}
    log_once = {k: v for k, v in everything.items() if "::" in k and v.startswith(LOG_ONCE_TAG)}
    by_name = {k: v for k, v in everything.items() if "::" in k and k not in log_once}
    if list_all:
        for rel, (status, names, missing) in results.items():
            tag = status if rel not in allowed else f"{status} (allowlisted)"
            shown = [nm + ("" if nm not in missing else
                           " (listed)" if f"{rel}::{nm}" in by_name else " (UNCOVERED)") for nm in names]
            print(f"  {tag:<26} {rel}  [{', '.join(shown)}]")
    failures = 0
    unlisted = [rel for rel, (s, *_) in results.items() if s == "no-rearm" and rel not in allowed]
    stale = [rel for rel in allowed if results.get(rel, ("not-candidate", [], []))[0] != "no-rearm"]
    missing_reason = [rel for rel, reason in allowed.items() if not reason]
    name_fails = name_verdict(results, by_name)
    if name_fails:
        failures += len(name_fails)
        print("check_static_reset: per-name rule (reset the static in the TU's hook, or list")
        print(f"  <path>::<name> in {ALLOWLIST.relative_to(REPO)} with reset-by=<function> or a reason):")
        for line in name_fails:
            print(f"  {line}")
    if unlisted:
        failures += len(unlisted)
        print("check_static_reset: lazy device static with no re-arm (register a hook with")
        print("  IMP_REGISTER_CUDA_STATIC_RESET, check engine_arena().generation(), or add the")
        print(f"  file to {ALLOWLIST.relative_to(REPO)} with a reason):")
        for rel in unlisted:
            print(f"  {rel}  [{', '.join(results[rel][1])}]")
    if stale:
        failures += len(stale)
        print("check_static_reset: stale allowlist entries (the TU now re-arms, or holds no")
        print("  lazy device static): delete them:")
        for rel in stale:
            print(f"  {rel}")
    if missing_reason:
        failures += len(missing_reason)
        print("check_static_reset: allowlist entries without a reason:")
        for rel in missing_reason:
            print(f"  {rel}")
    n_ok = sum(1 for s, *_ in results.values() if s == "ok")
    n_cand = len(results)
    n_names = sum(len(names) for s, names, _ in results.values() if s == "ok")
    n_missing = sum(len(missing) for _, _, missing in results.values())
    log_once_fails, log_once_tail = check_log_once(log_once)
    failures += log_once_fails
    tail = (f"{n_ok}/{n_cand} candidate TUs re-arm; {n_names} statics in them, {n_names - n_missing} "
            f"covered by a hook or a generation check, {n_missing} not; {log_once_tail}")
    if failures:
        print(f"check_static_reset: {failures} failure(s) ({tail})")
        return 1
    print(f"check_static_reset: ok ({tail}; allowlist: {len(allowed)} TUs, {len(by_name)} statics)")
    return 0


def selftest() -> int:
    cases = [
        # (label, text, expected status)
        ("static ptr + arena take, no hook",
         "static int32_t* s_buf = nullptr;\nvoid f() { auto s = engine_arena().take_bytes(4); }\n",
         "no-rearm"),
        ("static ptr + arena take + hook",
         "static int32_t* s_buf = nullptr;\nvoid f() { auto s = engine_arena().take_bytes(4); }\n"
         "IMP_REGISTER_CUDA_STATIC_RESET(reset);\n",
         "ok"),
        ("static ptr + cudaMalloc + generation check",
         "float* g_scratch = nullptr;\nbool ensure() {\n  if (g == engine_arena().generation()) return true;\n"
         "  cudaMalloc(&g_scratch, 4);\n}\n",
         "ok"),
        ("function-local static ptr behind a null guard (sampling.cu's d_result)",
         "int f() {\n    static int32_t* d_result = nullptr;\n    if (!d_result) cudaMalloc(&d_result, 4);\n}\n",
         "no-rearm"),
        ("anonymous-namespace struct global with a device member (ffn probe)",
         "namespace {\nProbeState g_state;\nvoid init() { auto s = engine_arena().take_bytes(8); }\n}\n",
         "no-rearm"),
        ("host-only static (counter, string) is not a candidate",
         "static int s_calls = 0;\nstatic std::string s_name;\nvoid f() { cudaMalloc(&p, 4); }\n",
         "not-candidate"),
        ("a LogOnce latch is host state, not a device static",
         "static LogOnce s_warned;\nvoid f() { cudaMalloc(&p, 4); }\n",
         "not-candidate"),
        ("static function declaration is not a variable",
         "static void ensure_ptr_arrays(int n);\nstatic int* make(int n) { return nullptr; }\n"
         "void f() { cudaMalloc(&p, 4); }\n",
         "not-candidate"),
        ("device static without any acquisition is not a candidate",
         "static cublasHandle_t s_handle = nullptr;\nvoid f() { use(s_handle); }\n",
         "not-candidate"),
        ("cuBLAS handle + cublasCreate, no hook",
         "static cublasHandle_t s_handle = nullptr;\nvoid f() { if (!s_handle) cublasCreate(&s_handle); }\n",
         "no-rearm"),
        ("cuBLAS handle via compute/cublas_handle.h, no hook",
         "static cublasHandle_t s_handle = nullptr;\n"
         "void f() { if (!s_handle) s_handle = cublas_handle_create(\"x\", CUBLAS_DEFAULT_MATH); }\n",
         "no-rearm"),
        ("cuBLASLt handle via compute/cublas_handle.h, no hook",
         "static cublasLtHandle_t s_lt = nullptr;\nvoid f() { if (!s_lt) s_lt = cublaslt_handle_create(\"x\"); }\n",
         "no-rearm"),
        ("constexpr / thread_local statics are skipped",
         "static constexpr int* kNull = nullptr;\nstatic thread_local std::vector<int32_t> h_tokens;\n"
         "void f() { cudaMalloc(&p, 4); }\n",
         "not-candidate"),
        ("std::unordered_map of device planes is a pointer holder",
         "std::unordered_map<const void*, WeightCache> g_w_cache;\n"
         "bool f() { cudaMalloc(&c.w, 4); }\n",
         "no-rearm"),
    ]
    failures = 0
    for label, text, want in cases:
        got, names = classify(text)
        ok = got == want
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got} {names}" + ("" if ok else f" (want {want})"))
    hook = ("static float* s_a = nullptr;\nstatic float* s_b = nullptr;\n"
            "void fill() { cudaMalloc(&s_a, 4); cudaMalloc(&s_b, 4); }\n"
            "namespace {\nvoid drop_b() { s_b = nullptr; }\n"
            "void reset() { s_a = nullptr; %s}\nIMP_REGISTER_CUDA_STATIC_RESET(reset);\n}\n")
    gen = ("static float* s_a = nullptr;\nstatic float* s_b = nullptr;\nstatic uint64_t s_gen = 0;\n"
           "static bool valid() { return s_gen == engine_arena().generation(); }\n"
           "float* a() { if (!valid()) { auto s = engine_arena().take_bytes(4); s_a = s.data(); } return s_a; }\n"
           "float* b() { if (!s_b) cudaMalloc(&s_b, 4); return s_b; }\n")
    name_cases = [
        # (label, text, expected uncovered names)
        ("both statics reset in the hook", hook % "s_b = nullptr; ", []),
        ("second static not in the hook", hook % "", ["s_b"]),
        ("second static reset by a function the hook calls", hook % "drop_b(); ", []),
        ("a name only in a comment of the hook does not count", hook % "/* s_b */ ", ["s_b"]),
        ("generation check covers the static of its caller, not the other one", gen, ["s_b"]),
        ("generation check in a member function of the static's own type",
         "struct Scratch {\n  float* p = nullptr;\n  uint64_t gen = 0;\n"
         "  bool ensure() { if (gen == engine_arena().generation()) return true;\n"
         "    p = engine_arena().take_bytes(4).data(); return true; }\n};\n"
         "static Scratch s_scratch;\nstatic float* s_other = nullptr;\n"
         "void f() { s_scratch.ensure(); cudaMalloc(&s_other, 4); }\n", ["s_other"]),
    ]
    for label, text, want in name_cases:
        status, names = classify(text)
        got = uncovered(text, names) if status == "ok" else ["<status " + status + ">"]
        ok = got == want
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: uncovered {got}" + ("" if ok else f" (want {want})"))
    res = {"src/a.cu": ("ok", ["s_a", "s_b"], ["s_b"])}
    foreign = lambda fn, rel: fn == "other_tu_clear"  # noqa: E731
    list_cases = [
        # (label, results, allowlist, expected failure count)
        ("uncovered static, not listed", res, {}, 1),
        ("static reset via a listed foreign function",
         res, {"src/a.cu::s_b": "reset-by=other_tu_clear  the executor clears it"}, 0),
        ("listed with a reason and no reset-by", res, {"src/a.cu::s_b": "host callback, no device memory"}, 0),
        ("reset-by names a function that does not exist",
         res, {"src/a.cu::s_b": "reset-by=gone_fn  the executor clears it"}, 1),
        ("reset-by without a reason", res, {"src/a.cu::s_b": "reset-by=other_tu_clear"}, 1),
        ("stale entry: the hook now resets the static",
         {"src/a.cu": ("ok", ["s_a", "s_b"], [])}, {"src/a.cu::s_b": "reset-by=other_tu_clear  x"}, 1),
        ("stale entry: the TU is gone", {}, {"src/a.cu::s_b": "reason"}, 1),
    ]
    for label, results, allowed, want in list_cases:
        got = len(name_verdict(results, allowed, foreign))
        ok = got == want
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got} failure(s)" + ("" if ok else f" (want {want})"))
    n_log_once, log_once_failures = selftest_log_once()
    failures += log_once_failures
    total = len(cases) + len(name_cases) + len(list_cases) + n_log_once
    print(f"selftest: {total - failures}/{total} cases")
    return 1 if failures else 0


def selftest_log_once() -> tuple[int, int]:
    """-> (cases, failures) of the log-once rule."""
    latch_cases = [
        # (label, text, expected names)
        ("plain static bool", "void f() {\n    static bool logged = false;\n}\n", ["logged"]),
        ("atomic, braced init", "    static std::atomic<bool> s_warned{false};\n", ["s_warned"]),
        ("per-key array", "    static bool s_logged[128] = {};  // once per qtype\n", ["s_logged"]),
        ("once_flag", "    static std::once_flag attr_once;\n", ["attr_once"]),
        ("prefix and suffix around the word", "static bool logged_sparse_active = false;\n",
         ["logged_sparse_active"]),
        ("LogOnce is the fix, not a finding", "    static LogOnce logged;\n", []),
        ("unrelated static bool", "    static bool smem_configured = false;\n", []),
        ("commented out", "// the previous plain `static bool warned` raced\n", []),
        ("non-static local", "    bool logged = false;\n", []),
    ]
    failures = 0
    for label, text, want in latch_cases:
        got = log_once_latches(text)
        ok = got == want
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} log-once {label}: {got}" + ("" if ok else f" (want {want})"))
    found = {"src/a.cu": ["logged"]}
    verdict_cases = [
        # (label, found, allowlist, expected failure count)
        ("unlisted latch", found, {}, 1),
        ("listed with a reason", found, {"src/a.cu::logged": "log-once: owned by another slice"}, 0),
        ("listed without a reason", found, {"src/a.cu::logged": "log-once:"}, 1),
        ("stale: the latch is gone", {}, {"src/a.cu::logged": "log-once: x"}, 1),
    ]
    for label, fnd, allowed, want in verdict_cases:
        got = len(log_once_verdict(fnd, allowed))
        ok = got == want
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} log-once {label}: {got} failure(s)" + ("" if ok else f" (want {want})"))
    return len(latch_cases) + len(verdict_cases), failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="print every candidate TU with its status")
    ap.add_argument("--selftest", action="store_true", help="run the planted classifier cases")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    return check(args.list)


if __name__ == "__main__":
    sys.exit(main())
