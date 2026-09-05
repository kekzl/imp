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

Registration is a convention. Measured 2026-09-05 (AUDIT_arch_2026 B-1..B-3):
15 TUs registered, 6 did not, one of them the logit_bias arena slots that every
`logit_bias` request after a model swap wrote through.

WHAT IT CHECKS
--------------
For each TU under src/ (outside src/memory/, which owns the driver):

  candidate  = declares a file-scope or function-scope static, or a file-scope
               `g_*` / `s_*` global, whose type is a pointer, a CUDA/cuBLAS
               handle, or a class instance (not std::, not a primitive)
  acquires   = takes from engine_arena() or calls cudaMalloc* / cublasCreate /
               cublasLtCreate / cudaStreamCreate / cudaEventCreate /
               cudaGraphInstantiate
  re-arms    = IMP_REGISTER_CUDA_STATIC_RESET( in the TU, or an
               engine_arena().generation() check (the other accepted pattern:
               the tenant compares the arena generation instead of trusting
               its cached pointer)

A TU that is a candidate AND acquires AND does not re-arm fails unless it is
in tools/static_reset_allowlist.txt with a reason. A listed TU that no longer
matches is a stale entry and fails too, so the list cannot rot either way.

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

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src"
ALLOWLIST = REPO / "tools" / "static_reset_allowlist.txt"

EXEMPT_PREFIXES = ("src/memory/",)
SUFFIXES = (".cu", ".cpp")

ACQUIRE = re.compile(
    r"\b(engine_arena\(\)\s*\.\s*take(_bytes)?\s*\(|cudaMalloc\w*\s*\(|cublasCreate|cublasLtCreate"
    r"|cudaStreamCreate|cudaEventCreate|cudaGraphInstantiate)"
)
REARM_HOOK = re.compile(r"\bIMP_REGISTER_CUDA_STATIC_RESET\s*\(")
REARM_GEN = re.compile(r"\bengine_arena\(\)\s*\.\s*generation\s*\(")

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
    return head[:1].isupper()


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


def scan() -> dict[str, tuple[str, list[str]]]:
    out = {}
    for p in sorted(SRC.rglob("*")):
        if p.suffix not in SUFFIXES or not p.is_file():
            continue
        rel = p.relative_to(REPO).as_posix()
        if rel.startswith(EXEMPT_PREFIXES):
            continue
        status, names = classify(p.read_text(encoding="utf-8", errors="replace"))
        if names:
            out[rel] = (status, names)
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


def check(list_all: bool) -> int:
    results = scan()
    allowed = read_allowlist()
    if list_all:
        for rel, (status, names) in results.items():
            tag = status if rel not in allowed else f"{status} (allowlisted)"
            print(f"  {tag:<26} {rel}  [{', '.join(names)}]")
    failures = 0
    unlisted = [rel for rel, (s, _) in results.items() if s == "no-rearm" and rel not in allowed]
    stale = [rel for rel in allowed if results.get(rel, ("not-candidate", []))[0] != "no-rearm"]
    missing_reason = [rel for rel, reason in allowed.items() if not reason]
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
    n_ok = sum(1 for s, _ in results.values() if s == "ok")
    n_cand = len(results)
    if failures:
        print(f"check_static_reset: {failures} failure(s) ({n_ok}/{n_cand} candidate TUs re-arm)")
        return 1
    print(f"check_static_reset: ok ({n_ok}/{n_cand} candidate TUs re-arm, {len(allowed)} allowlisted)")
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
    print(f"selftest: {len(cases) - failures}/{len(cases)} cases")
    return 1 if failures else 0


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
