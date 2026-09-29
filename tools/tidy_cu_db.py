#!/usr/bin/env python3
"""Rewrite the nvcc entries of compile_commands.json into clang CUDA host-only commands.

clang-tidy cannot read nvcc flags. Kept: -I/-isystem/-D/-std; added: `-x cuda
--cuda-host-only`, so the host half of each .cu TU is analysed (#2210).

Usage: python3 tools/tidy_cu_db.py <build/compile_commands.json> <out_dir>
Exit 1 when the input holds no .cu entry: an empty database lints nothing (#1626).
"""
import json
import os
import shlex
import sys

CUDA_PATH = os.environ.get("CUDA_PATH", "/usr/local/cuda")
CLANG_CUDA = ["clang++", "-x", "cuda", "--cuda-host-only", "--cuda-gpu-arch=sm_120a",
              f"--cuda-path={CUDA_PATH}", "-nocudalib", "-Wno-unknown-cuda-version",
              # clang 21 + CUDA 13.4: math_functions.hpp:2994 uses this macro before it is defined.
              "-D_NV_RSQRT_SPECIFIER=noexcept(true)"]
# clang's CUDA wrapper still includes texture_fetch_functions.h, which CUDA 13 dropped.
TEXTURE_STUB = "// Empty: CUDA 13 dropped this header, clang's CUDA wrapper still includes it.\n"
# Overloads nvcc provides and clang's wrapper lacks; without them 20 of 174 TUs error out.
COMPAT = """#pragma once
// Lint-only (tools/tidy_cu_db.py): nvcc overloads missing from clang's CUDA wrapper.
__host__ inline int min(int a, int b) { return a < b ? a : b; }
__host__ inline int max(int a, int b) { return a > b ? a : b; }
__host__ inline unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }
__host__ inline unsigned max(unsigned a, unsigned b) { return a > b ? a : b; }
__host__ inline long long min(long long a, long long b) { return a < b ? a : b; }
__host__ inline long long max(long long a, long long b) { return a > b ? a : b; }
__host__ inline unsigned long long min(unsigned long long a, unsigned long long b) { return a < b ? a : b; }
__host__ inline unsigned long long max(unsigned long long a, unsigned long long b) { return a > b ? a : b; }
__device__ void __stcs(unsigned short* p, unsigned short v);
__device__ void __stcs(float* p, float v);
"""


def expand_rsp(args, directory):
    """Inline `--options-file F` / `-optf F` / `@F`: the Makefiles generator puts -I there."""
    out, i = [], 0
    while i < len(args):
        a, path = args[i], None
        if a in ("--options-file", "-optf") and i + 1 < len(args):
            path, i = args[i + 1], i + 2
        elif a.startswith("@"):
            path, i = a[1:], i + 1
        else:
            out.append(a)
            i += 1
            continue
        with open(os.path.join(directory, path), encoding="utf-8") as f:
            out += expand_rsp(shlex.split(f.read()), directory)
    return out


def host_flags(args):
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if a in ("-I", "-isystem", "-D"):
            out += [a, args[i + 1]]
            i += 2
            continue
        if a.startswith(("-I", "-D", "-std=")):
            out.append(a)
        i += 1
    return out


def convert(entries, stub_dir):
    out = []
    for e in entries:
        if not e["file"].endswith(".cu"):
            continue
        args = expand_rsp(e["arguments"] if "arguments" in e else shlex.split(e["command"]),
                          e["directory"])
        out.append({"directory": e["directory"], "file": e["file"],
                    "arguments": CLANG_CUDA + ["-isystem", stub_dir, "-include",
                                               os.path.join(stub_dir, "tidy_nvcc_compat.h")]
                    + host_flags(args[1:]) + ["-c", e["file"]]})
    return out


def main():
    if len(sys.argv) != 3:
        sys.stderr.write(__doc__)
        return 2
    stub_dir = os.path.join(os.path.abspath(sys.argv[2]), "stub")
    with open(sys.argv[1], encoding="utf-8") as f:
        cu = convert(json.load(f), stub_dir)
    if not cu:
        sys.stderr.write(f"no .cu entry in {sys.argv[1]}\n")
        return 1
    # Every imp TU has project -I paths; none means an unexpanded response file.
    bare = [e["file"] for e in cu if not any(a.startswith("-I") for a in e["arguments"])]
    if bare:
        sys.stderr.write(f"{len(bare)} .cu entries without an -I path, e.g. {bare[0]}\n")
        return 1
    os.makedirs(stub_dir, exist_ok=True)
    for name, text in (("texture_fetch_functions.h", TEXTURE_STUB), ("tidy_nvcc_compat.h", COMPAT)):
        with open(os.path.join(stub_dir, name), "w", encoding="utf-8") as f:
            f.write(text)
    with open(os.path.join(sys.argv[2], "compile_commands.json"), "w", encoding="utf-8") as f:
        json.dump(cu, f, indent=1)
    print(f"tidy_cu_db: {len(cu)} .cu entries -> {sys.argv[2]}/compile_commands.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
