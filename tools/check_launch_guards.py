#!/usr/bin/env python3
"""Every CUDA kernel launch in src/ must be followed by a post-launch error check.

WHY THIS EXISTS
---------------
A kernel launch that fails at launch time (invalid configuration, too much
shared memory, missing kernel image) does NOT raise where it happens — the
error goes sticky and surfaces at the next synchronizing call, arbitrarily far
away, or is swallowed entirely and the kernel simply never runs. The output is
then silently wrong rather than obviously broken. `IMP_CUDA_CHECK_LAUNCH()`
(core/logging.h) reports it at the launch site via `cudaPeekAtLastError()`
without clearing the sticky error, so downstream handling still sees it.

The convention was ~99% adopted and 0% enforced, which is exactly what erosion
looks like from the outside: uniform adherence in mature code and a clean miss
in the newest file. The 2026-08-02 architecture audit found
src/vision/qwen3vl_encoder_kernels.cu at 9 launches / 0 checks — the whole
Qwen3-VL tower — where a launch failure would have produced silently wrong
image embeddings.

WHAT COUNTS
-----------
A launch is a `<<<` outside a comment/string in a `.cu` or `.cuh`. It is guarded when one of
IMP_CUDA_CHECK_LAUNCH / cudaPeekAtLastError / cudaGetLastError appears within
LOOKAHEAD lines after the statement the launch belongs to (launches spanning
several lines are followed to their terminating `;`).

A launch inside a `#define` body (C-016) is guarded when the body checks after the
`<<<`, or when EVERY expansion of the macro in the file is followed by a check before
its enclosing function closes (the first later line starting with `}` in column 0).
The second form is the shared check after a `switch` over head_dim
(paged_attention_splitk in compute/attention_paged.cu: one cudaGetLastError after
both macros, with a single-split fallback). A macro with no expansion is unguarded.

Exit 0 when clean, 1 when a launch is unguarded and not allowlisted.
`--selftest` runs the matcher and the file collection on fixtures.
"""

import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = ["src"]
SUFFIXES = ("*.cu", "*.cuh")  # .cuh: 24 launches were outside the gate (C-015)
LOOKAHEAD = 4  # lines after the launch statement ends

GUARD = re.compile(r"IMP_CUDA_CHECK_LAUNCH|cudaPeekAtLastError|cudaGetLastError")
MACRO_NAME = re.compile(r"#define\s+(\w+)")

# Launch sites that are deliberately unguarded, each with a reason.
# THIS LIST ONLY SHRINKS — an entry that no longer matches is an error too, so
# it cannot go stale silently.
ALLOWLIST: dict[str, str] = {
    "src/compute/apa/apa.cuh": "vendored kekzl/apa v0.8.1, patched upstream only; the 2 reset_stats_kernel "
    "launches are checked by the cudaPeekAtLastError 5 lines below, after the 4-launch redo group",
}


def strip_comments_and_strings(src: str) -> str:
    """Blank out comments and string/char literals, preserving line structure."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif c == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(ch if ch == "\n" else " " for ch in src[i:j]))
            i = j
        elif c in "\"'":
            q, j = c, i + 1
            while j < n and src[j] != q:
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, n)
            out.append("".join(ch if ch == "\n" else " " for ch in src[i:j]))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def launch_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for d in SCAN_DIRS:
        for pattern in SUFFIXES:
            files.extend((root / d).rglob(pattern))
    return sorted(files)


def scan(path: Path) -> tuple[list[tuple[int, str]], int]:
    """Return ([(line_no, source_line)] unguarded, count of macro-body launches)."""
    return scan_text(path.read_text(encoding="utf-8", errors="replace"))


def macro_spans(clean: list[str]) -> list[tuple[str, int, int]]:
    """(name, first line, last line) of every #define, continuation lines included."""
    spans = []
    i = 0
    while i < len(clean):
        stripped = clean[i].strip()
        if stripped.startswith("#define"):
            j = i
            while j + 1 < len(clean) and clean[j].rstrip().endswith("\\"):
                j += 1
            m = MACRO_NAME.match(stripped)
            spans.append((m.group(1) if m else "", i, j))
            i = j + 1
        else:
            i += 1
    return spans


def macro_launch_guarded(clean: list[str], name: str, start: int, end: int, launch: int) -> bool:
    """Body checks after the launch, or every expansion is checked before its function closes."""
    if GUARD.search("\n".join(clean[launch : end + 1])):
        return True
    call = re.compile(r"\b%s\s*\(" % re.escape(name))
    sites = [
        k for k, line in enumerate(clean)
        if not start <= k <= end and call.search(line) and not line.lstrip().startswith("#")
    ]
    if not name or not sites:
        return False
    for k in sites:
        close = k + 1
        while close < len(clean) and not clean[close].startswith("}"):
            close += 1
        if not GUARD.search("\n".join(clean[k:close])):
            return False
    return True


def scan_text(raw: str) -> tuple[list[tuple[int, str]], int]:
    clean = strip_comments_and_strings(raw).split("\n")
    raw_lines = raw.split("\n")
    spans = macro_spans(clean)
    span_of = {i: sp for sp in spans for i in range(sp[1], sp[2] + 1)}

    bad = []
    macro_launches = 0
    for idx, line in enumerate(clean):
        if "<<<" not in line:
            continue
        if idx in span_of:
            macro_launches += 1
            name, start, end = span_of[idx]
            if not macro_launch_guarded(clean, name, start, end, idx):
                bad.append((idx + 1, raw_lines[idx].strip()))
            continue
        # Follow the statement to its terminating ';' (launches wrap often).
        end = idx
        while end < len(clean) and ";" not in clean[end]:
            end += 1
        window = "\n".join(clean[idx : min(end + 1 + LOOKAHEAD, len(clean))])
        if not GUARD.search(window):
            bad.append((idx + 1, raw_lines[idx].strip()))
    return bad, macro_launches


def selftest() -> int:
    cases = {  # source -> (unguarded, macro-body launches)
        "k<<<1, 32, 0, s>>>(p);\n": (1, 0),
        "k<<<1, 32, 0, s>>>(p);\nIMP_CUDA_CHECK_LAUNCH();\n": (0, 0),
        "k\n    <<<1, 32, 0, s>>>(p,\n        q);\nif ((e = cudaPeekAtLastError()) != cudaSuccess)\n": (0, 0),
        "a<<<1, 1>>>(p);\nb<<<1, 1>>>(p);\nc<<<1, 1>>>(p);\nd<<<1, 1>>>(p);\ne<<<1, 1>>>(p);\n"
        "IMP_CUDA_CHECK_LAUNCH();\n": (1, 0),
        "#define L(D) \\\n    k<D><<<1, 32>>>(p)\n": (1, 1),  # never expanded
        "#define L(D) \\\n    k<D><<<1, 32>>>(p); \\\n    IMP_CUDA_CHECK_LAUNCH()\n": (0, 1),
        "#define L(D) \\\n    k<D><<<1, 32>>>(p)\nvoid f() {\n    L(64);\n    cudaGetLastError();\n}\n": (0, 1),
        "#define L(D) \\\n    k<D><<<1, 32>>>(p)\nvoid f() {\n    L(64);\n    cudaGetLastError();\n}\n"
        "void g() {\n    L(32);\n}\n": (1, 1),
        "// k<<<1, 32>>>(p);\n": (0, 0),
    }
    bad = 0
    for text, want in cases.items():
        unguarded, in_macro = scan_text(text)
        got = (len(unguarded), in_macro)
        if got != want:
            print(f"selftest: {text!r}: want {want}, got {got}")
            bad += 1
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src" / "a").mkdir(parents=True)
        for name in ("k.cu", "k.cuh", "k.h", "k.cpp"):
            (root / "src" / "a" / name).write_text("k<<<1, 32>>>(p);\n", encoding="utf-8")
        got_files = sorted(p.name for p in launch_files(root))
        if got_files != ["k.cu", "k.cuh"]:
            print(f"selftest: collected {got_files}, want ['k.cu', 'k.cuh']")
            bad += 1
    total = len(cases) + 1
    print(f"selftest: {total - bad}/{total} cases ok")
    return 1 if bad else 0


def main() -> int:
    if "--selftest" in sys.argv[1:]:
        return selftest()
    total_launches = 0
    macro_total = 0
    findings: list[tuple[str, int, str]] = []
    files = launch_files(ROOT)

    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        clean = strip_comments_and_strings(f.read_text(encoding="utf-8", errors="replace"))
        total_launches += clean.count("<<<")
        unguarded, in_macro = scan(f)
        macro_total += in_macro
        for line_no, text in unguarded:
            findings.append((rel, line_no, text))

    unexpected = [(f, l, t) for f, l, t in findings if f not in ALLOWLIST]
    stale = [f for f in ALLOWLIST if not any(f == g for g, _, _ in findings)]

    guarded = total_launches - len(findings)
    print(f"launch guards: {guarded}/{total_launches} kernel launches carry a post-launch check "
          f"({macro_total} of them in #define bodies, decided per expansion site)")

    if unexpected:
        print(f"\nFAIL — {len(unexpected)} unguarded kernel launch(es):\n")
        for f, l, t in unexpected:
            print(f"  {f}:{l}\n      {t[:100]}")
        print(
            "\nAdd IMP_CUDA_CHECK_LAUNCH(); after the launch (core/logging.h).\n"
            "A launch-time failure otherwise surfaces at the next sync, or not at all,\n"
            "and the kernel's output is silently wrong."
        )
        return 1

    if stale:
        print(f"\nFAIL — {len(stale)} stale allowlist entry/entries (now guarded — remove them):")
        for f in stale:
            print(f"  {f}")
        return 1

    print("OK — every kernel launch is guarded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
