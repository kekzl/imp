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
A launch is a `<<<` outside a comment/string. It is guarded when one of
IMP_CUDA_CHECK_LAUNCH / cudaPeekAtLastError / cudaGetLastError appears within
LOOKAHEAD lines after the statement the launch belongs to (launches spanning
several lines are followed to their terminating `;`).

Launches written inside a `#define` body are OUT OF SCOPE and are reported as
such rather than guessed at. There the `<<<` is not the launch *site* — the
macro's call sites are, typically a `switch` over head_dim expanding it 5x with
a single shared check after the switch (paged_attention_splitk in
compute/attention_paged.cu guards both its macros with one cudaGetLastError 67
lines below, and falls back to the single-split path on failure). Deciding
guardedness for those needs the enclosing function, not a line window; a
line-distance heuristic would either flag working code or silently stop
catching real misses once someone's switch grows. The gate covers what it can
decide and prints the count it cannot, so the blind spot is visible instead of
implied.

Exit 0 when clean, 1 when a launch is unguarded and not allowlisted.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = ["src"]
LOOKAHEAD = 4  # lines after the launch statement ends

GUARD = re.compile(r"IMP_CUDA_CHECK_LAUNCH|cudaPeekAtLastError|cudaGetLastError")

# Launch sites that are deliberately unguarded, each with a reason.
# THIS LIST ONLY SHRINKS — an entry that no longer matches is an error too, so
# it cannot go stale silently.
ALLOWLIST: dict[str, str] = {}


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


def scan(path: Path) -> tuple[list[tuple[int, str]], int]:
    """Return ([(line_no, source_line)] unguarded, count of macro-body launches)."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    clean = strip_comments_and_strings(raw).split("\n")
    raw_lines = raw.split("\n")

    # Line indices that belong to a #define body (the directive line and every
    # backslash-continued line after it).
    in_macro = set()
    cont = False
    for i, line in enumerate(clean):
        stripped = line.strip()
        if cont or stripped.startswith("#define"):
            in_macro.add(i)
        cont = line.rstrip().endswith("\\")

    bad = []
    macro_launches = 0
    for idx, line in enumerate(clean):
        if "<<<" not in line:
            continue
        if idx in in_macro:
            macro_launches += 1
            continue  # out of scope — see the module docstring
        # Follow the statement to its terminating ';' (launches wrap often).
        end = idx
        while end < len(clean) and ";" not in clean[end]:
            end += 1
        window = "\n".join(clean[idx : min(end + 1 + LOOKAHEAD, len(clean))])
        if not GUARD.search(window):
            bad.append((idx + 1, raw_lines[idx].strip()))
    return bad, macro_launches


def main() -> int:
    total_launches = 0
    macro_total = 0
    findings: list[tuple[str, int, str]] = []
    files = []
    for d in SCAN_DIRS:
        files.extend(sorted((ROOT / d).rglob("*.cu")))

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

    in_scope = total_launches - macro_total
    guarded = in_scope - len(findings)
    print(f"launch guards: {guarded}/{in_scope} in-scope kernel launches carry a post-launch check "
          f"({macro_total} in #define bodies — guarded at their call sites, not checked here)")

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
