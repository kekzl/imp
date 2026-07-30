#!/usr/bin/env python3
"""I1 gate: exactly one module talks to the CUDA driver about memory.

docs/MEMORY_ARCHITECTURE.md invariant I1 — no cudaMalloc / cudaMallocAsync /
cudaFree / cuMemCreate / thrust::device_vector / pinned-host allocation outside
src/memory/. Everything else receives typed views.

The engine cannot get there in one commit, so the gate runs against an explicit
allowlist that shrinks monotonically. Two hard failures:

  1. A file allocates and is NOT on the allowlist  -> the invariant regressed.
  2. A file IS on the allowlist but no longer allocates -> the entry is stale
     and must be deleted. This is what makes "the allowlist shrinks
     monotonically" true by construction rather than by good intentions: you
     cannot migrate a file and leave its entry behind, so the list is always an
     accurate picture of the remaining debt.

Scope is src/ — the engine. tools/imp-bench, tools/standalone and tools/analysis
are separate probe binaries that legitimately allocate directly; they are not
part of the engine's memory subsystem and are not gated.

Usage:
    python3 tools/check_alloc_sites.py            # check (CI)
    python3 tools/check_alloc_sites.py --update   # rewrite the allowlist
    python3 tools/check_alloc_sites.py --stats    # progress only, never fails
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src"
ALLOWLIST = REPO / "tools" / "alloc_allowlist.txt"

# The module that IS allowed to do this.
EXEMPT_PREFIXES = ("src/memory/",)

SUFFIXES = (".cpp", ".cu", ".h", ".cuh", ".hpp")

# Driver/runtime memory APIs. Ordered longest-first so the alternation reports
# the most specific name (cudaMallocAsync, not cudaMalloc).
# Split on purpose. Invariant I1 is titled "single ACQUISITION point", and the
# two halves need different work: an acquisition needs a tier to move to, while
# a release follows for free once the owner is a RAII type (Region, BlockRef,
# GraphSlotLease) — it is a consequence of migrating, not separate work.
# Reporting one number for both overstated the distance to criterion 2 by more
# than half: 696 total is 309 acquisitions and 407 releases, and EIGHT files
# contain only releases, so no allocation migration can ever remove them.
APIS_ACQUIRE = [
    "cudaMallocAsync",
    "cudaMallocManaged",
    "cudaMallocPitch",
    "cudaMallocHost",
    "cudaMalloc3D",
    "cudaMalloc",
    "cudaHostAlloc",
    "cudaHostRegister",
    "cuMemAddressReserve",
    "cuMemCreate",
    "cuMemMap",
    "cuMemSetAccess",
    "thrust::device_vector",
]
APIS_RELEASE = [
    "cudaFreeAsync",
    "cudaFreeHost",
    "cudaFree",
    "cudaHostUnregister",
    "cuMemAddressFree",
    "cuMemRelease",
    "cuMemUnmap",
]
APIS = APIS_ACQUIRE + APIS_RELEASE
# Word-boundary on the left so `imp_cudaMalloc_wrapper` is not a false hit, and
# a '(' or '<' on the right so a mention in an identifier is not either.
PATTERN = re.compile(r"\b(" + "|".join(re.escape(a) for a in APIS) + r")\s*[(<]")
PATTERN_ACQUIRE = re.compile(r"\b(" + "|".join(re.escape(a) for a in APIS_ACQUIRE) + r")\s*[(<]")
PATTERN_RELEASE = re.compile(r"\b(" + "|".join(re.escape(a) for a in APIS_RELEASE) + r")\s*[(<]")

# A line whose first non-space run starts a comment. Cheap and good enough:
# the alternative is a real preprocessor, and a banned call hidden inside a
# block comment that starts mid-line is not a failure mode worth handling.
COMMENT_LINE = re.compile(r"^\s*(//|\*|/\*)")


def scan() -> dict[str, list[tuple[int, str]]]:
    """rel_path -> [(line_no, api)] for every non-exempt file that allocates."""
    hits: dict[str, list[tuple[int, str]]] = {}
    for path in sorted(SRC.rglob("*")):
        if path.suffix not in SUFFIXES or not path.is_file():
            continue
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith(EXEMPT_PREFIXES):
            continue
        found: list[tuple[int, str]] = []
        with path.open(encoding="utf-8", errors="replace") as fh:
            for n, line in enumerate(fh, 1):
                if COMMENT_LINE.match(line):
                    continue
                m = PATTERN.search(line)
                if m:
                    found.append((n, m.group(1)))
        if found:
            hits[rel] = found
    return hits


def read_allowlist() -> dict[str, int]:
    """rel_path -> budgeted site count.

    The trailing `# N` the writer emits is the BUDGET, not a comment: a file
    may not exceed it. Without that the gate is file-granular, and a step that
    removes six of a file's seventeen sites shows zero progress and cannot
    regress — engine_scheduler.cpp alone would hide six per-request
    allocations behind an unchanged entry.
    """
    if not ALLOWLIST.exists():
        return {}
    out: dict[str, int] = {}
    for line in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        path, _, rest = stripped.partition("#")
        path = path.strip()
        if not path:
            continue
        try:
            out[path] = int(rest.strip())
        except ValueError:
            out[path] = -1  # legacy entry without a budget: count not enforced
    return out


def write_allowlist(files: dict[str, list[tuple[int, str]]]) -> None:
    total = sum(len(v) for v in files.values())
    body = [
        "# I1 allowlist — files outside src/memory/ that still call the CUDA",
        "# driver about memory directly. Generated by tools/check_alloc_sites.py.",
        "#",
        "# THIS LIST ONLY SHRINKS. Adding an entry is a deliberate act that has to",
        "# be justified in review; the gate fails on any file that allocates and is",
        "# not listed, and equally on any listed file that no longer allocates.",
        "#",
        f"# remaining: {len(files)} files, {total} sites",
        "",
    ]
    for rel in sorted(files):
        body.append(f"{rel}  # {len(files[rel])}")
    ALLOWLIST.write_text("\n".join(body) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="rewrite the allowlist from the tree")
    ap.add_argument("--stats", action="store_true", help="print progress, never fail")
    args = ap.parse_args()

    hits = scan()
    total_sites = sum(len(v) for v in hits.values())

    if args.update:
        write_allowlist(hits)
        print(f"allowlist updated: {len(hits)} files, {total_sites} sites")
        return 0

    allowed = read_allowlist()
    offenders = sorted(set(hits) - set(allowed))
    stale = sorted(set(allowed) - set(hits))
    budget_total = sum(v for v in allowed.values() if v >= 0)

    print(f"I1: {len(hits)} files / {total_sites} sites outside src/memory/ "
          f"(allowlist: {len(allowed)} files / {budget_total} sites)")

    if args.stats:
        # Acquisition / release split. I1 is titled "single ACQUISITION point",
        # and the halves need different work: an acquisition needs a tier to move
        # to, a release follows for free once the owner is a RAII type. Reporting
        # one number overstated the distance to criterion 2 by more than half.
        acq = rel_ = 0
        only_release = []
        for relpath, lines in hits.items():
            text = (REPO / relpath).read_text(errors="ignore").splitlines()
            a = sum(len(PATTERN_ACQUIRE.findall(t)) for t in text
                    if not t.strip().startswith("//"))
            r = sum(len(PATTERN_RELEASE.findall(t)) for t in text
                    if not t.strip().startswith("//"))
            acq += a
            rel_ += r
            if a == 0 and r > 0:
                only_release.append((r, relpath))
        print(f"  acquisitions {acq}   releases {rel_}")
        if only_release:
            print(f"  {len(only_release)} file(s) contain ONLY releases — no allocation "
                  f"migration can remove them:")
            for r, f in sorted(only_release, reverse=True):
                print(f"    {r:4d}  {f}")
        print("  largest by total calls:")
        for relpath in sorted(hits, key=lambda r: -len(hits[r]))[:15]:
            budget = allowed.get(relpath, 0)
            mark = "" if budget < 0 else f"  (budget {budget})"
            print(f"  {len(hits[relpath]):4d}  {relpath}{mark}")
        return 0

    rc = 0

    # Per-site budget. Over is a regression; under means the list is stale and
    # has to be refreshed in the same commit that earned the reduction, so the
    # remaining debt is always an accurate number rather than a stale ceiling.
    over, under = [], []
    for rel, sites in hits.items():
        budget = allowed.get(rel, -1)
        if budget < 0:
            continue
        if len(sites) > budget:
            over.append((rel, len(sites), budget))
        elif len(sites) < budget:
            under.append((rel, len(sites), budget))
    if over:
        rc = 1
        print("\nFAIL: these files gained allocation sites.")
        for rel, actual, budget in sorted(over):
            print(f"  {rel}: {actual} sites, budgeted {budget}")
    if under:
        rc = 1
        print("\nFAIL: these files shrank — refresh the budget in this commit "
              "(python3 tools/check_alloc_sites.py --update).")
        for rel, actual, budget in sorted(under):
            print(f"  {rel}: {actual} sites, budgeted {budget}  (-{budget - actual})")
    if offenders:
        rc = 1
        print("\nFAIL: these files allocate but are not on the I1 allowlist.")
        print("Route them through src/memory/ (docs/MEMORY_ARCHITECTURE.md A3),")
        print("or, if that is genuinely not possible yet, justify the addition in review.")
        for rel in offenders:
            for line_no, api in hits[rel][:5]:
                print(f"  {rel}:{line_no}: {api}")
            if len(hits[rel]) > 5:
                print(f"  {rel}: ... and {len(hits[rel]) - 5} more")

    if stale:
        rc = 1
        print("\nFAIL: these allowlist entries no longer allocate — delete them.")
        print("(The list must stay an accurate picture of the remaining debt.)")
        for rel in stale:
            print(f"  {rel}")

    if rc == 0:
        print("OK — no new direct allocation sites.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
