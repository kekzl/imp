#!/usr/bin/env python3
"""Fail when a device pointer is freed through a different allocator than the one
that produced it.

WHY THIS EXISTS
---------------
`cudaFreeAsync` on a pointer from plain `cudaMalloc` (and the reverse) is not valid
CUDA. It is also invisible: the driver does not have to complain, the pointer is the
right width, and the program keeps running. `AUDIT.md` B10 recorded exactly one such
pair in July at `mtp_forward.cu:606/615`; by August the lines had moved to 610/619 and
the defect had not. A defect that survives its own bug report by moving two lines is a
defect that needs a gate, not another report.

`tools/check_alloc_sites.py` is the sibling of this file. It counts sites and diffs them
against an allowlist, which answers "who talks to the driver" (invariant I1). It cannot
answer "does this pointer come home to the same allocator", because that is a property of
a *pair*, not of a site.

WHAT IT CHECKS
--------------
Three families, each of which must be matched:

    cudaMalloc / cudaMallocManaged        <->  cudaFree
    cudaMallocAsync                       <->  cudaFreeAsync
    cudaMallocHost / cudaHostAlloc        <->  cudaFreeHost

Two passes, because the two real defects in this tree had different shapes:

1. **In-file**, keyed on the pointer expression. Catches the `mtp_forward.cu` `d_tok`
   shape: a local allocated and freed a few lines apart through two allocators.
2. **Cross-file, member variables only** (names ending in `_`). Catches the
   `chunk_eager_k_` shape: `cudaMallocAsync` in the grow path in one TU, `cudaFree` in
   `free_buffers()` in another. An in-file check is structurally blind to it, and that
   is precisely where a teardown drifts away from its allocator - nobody reads the two
   files together. Restricted to member-looking names because a bare local called `p`
   means something different in every function; a trailing underscore is this
   codebase's member convention and makes the name meaningful tree-wide.

The pointer expression is normalised (casts, `this->`, `[i]` subscripts and whitespace
stripped) so `cudaFree((void*)m_buf_)` pairs with `cudaMalloc(&m_buf_, n)`.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not flag a name that is only allocated, or only freed. Half a pair inside one
file is the normal shape of an owning class whose destructor lives in another TU, and
flagging it would bury the real finding under hundreds of them. The gate fires only when
BOTH halves are present in one file and they disagree - which is the case where the
mismatch is provable from the text alone.

ALLOWLIST
---------
`tools/alloc_pairs_allowlist.txt`, one `path:name  # reason` per line (cross-file
entries use `member:name`). A justified
mismatch is a pointer that genuinely comes from either API depending on a branch, and
whose free branch matches. Empty reasons are rejected, and a stale entry (the mismatch is
gone) fails the gate too, so the list cannot rot in either direction - the same two-way
ratchet `tools/alloc_allowlist.txt` uses.

Usage:
    python3 tools/check_alloc_pairs.py            # gate: non-zero on any unlisted mismatch
    python3 tools/check_alloc_pairs.py --list     # every pair found, matched or not
"""
import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ROOTS = ("src", "tools", "include")
EXTS = (".cpp", ".cu", ".h", ".cuh", ".hpp")
ALLOWLIST = ROOT / "tools" / "alloc_pairs_allowlist.txt"

# family -> (alloc APIs, free API)
FAMILIES = {
    "sync": (("cudaMalloc", "cudaMallocManaged"), "cudaFree"),
    "async": (("cudaMallocAsync",), "cudaFreeAsync"),
    "host": (("cudaMallocHost", "cudaHostAlloc"), "cudaFreeHost"),
}
ALLOC_FAMILY = {a: f for f, (allocs, _) in FAMILIES.items() for a in allocs}
FREE_FAMILY = {free: f for f, (_, free) in FAMILIES.items()}

# `cudaMalloc` is a prefix of `cudaMallocAsync`/`cudaMallocHost`/`cudaMallocManaged`, so
# the alternation must try the longest first and the boundary must be explicit.
#
# The regex finds the CALL, not the argument. Reading the first argument with
# `[^,)]+` looks equivalent and is not: this codebase wraps most calls in
# IMP_CUDA_CHECK_LOG and clang-format then breaks the line right after the open
# paren, so `cudaMallocAsync(\n    &d_alpha, ...)` yields an empty first argument
# and the pointer silently drops out of the census. Three matched MoE pairs
# disappeared that way on the first run of this tool. `first_arg()` below walks
# the real parentheses instead, across newlines.
API_RE = re.compile(
    r"\b(cudaMallocAsync|cudaMallocManaged|cudaMallocHost|cudaHostAlloc|cudaMalloc"
    r"|cudaFreeAsync|cudaFreeHost|cudaFree)\s*\(")

CAST_RE = re.compile(r"(?:reinterpret_cast|static_cast|const_cast)\s*<[^>]*>\s*")
CSTYLE_RE = re.compile(r"\(\s*void\s*\*+\s*\)|\(\s*[A-Za-z_][A-Za-z0-9_:]*\s*\*+\s*\)")


def strip_comments(text: str) -> str:
    """Blank out // and /* */ without touching line numbering."""
    out, i, n = [], 0, len(text)
    state = None  # None | 'line' | 'block' | 'str' | 'chr'
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if state is None:
            if c == "/" and nxt == "/":
                state = "line"; out.append("  "); i += 2; continue
            if c == "/" and nxt == "*":
                state = "block"; out.append("  "); i += 2; continue
            if c == '"':
                state = "str"
            elif c == "'":
                state = "chr"
            out.append(c); i += 1; continue
        if state == "line":
            if c == "\n":
                state = None; out.append(c)
            else:
                out.append(" ")
            i += 1; continue
        if state == "block":
            if c == "*" and nxt == "/":
                state = None; out.append("  "); i += 2; continue
            out.append(c if c == "\n" else " "); i += 1; continue
        # inside a string/char literal
        out.append(c)
        if c == "\\":
            if i + 1 < n:
                out.append(text[i + 1]); i += 2; continue
        elif (state == "str" and c == '"') or (state == "chr" and c == "'"):
            state = None
        i += 1
    return "".join(out)


def normalise(expr: str) -> str:
    """`(void**)&this->buf_[i]` -> `buf_`. Returns "" for anything not a plain name."""
    e = expr.strip()
    e = CAST_RE.sub("", e)
    e = CSTYLE_RE.sub("", e)
    e = e.strip().lstrip("&").strip()
    e = re.sub(r"\[[^\]]*\]", "", e)          # drop subscripts
    e = re.sub(r"^\(+|\)+$", "", e).strip()
    e = re.sub(r"^this\s*->\s*", "", e)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\s*(?:\.|->)\s*[A-Za-z_][A-Za-z0-9_]*)*", e):
        return ""
    return re.sub(r"\s+", "", e)


def first_arg(text: str, open_paren: int) -> str:
    """Text of the first argument of the call whose '(' is at `open_paren`.

    Walks real parentheses/brackets so a nested cast or a line break inside the
    argument list does not truncate it. Returns "" if the parens do not close.
    """
    depth, start, i, n = 0, open_paren + 1, open_paren, len(text)
    while i < n:
        c = text[i]
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
            if depth == 0:
                return text[start:i]
        elif c == "," and depth == 1:
            return text[start:i]
        i += 1
    return ""


def scan(path: pathlib.Path):
    """-> {name: {'alloc': {family: [(line, api)]}, 'free': {family: [(line, api)]}}}"""
    text = strip_comments(path.read_text(errors="ignore"))
    starts = [0]
    for ch in text:
        starts.append(starts[-1] + (1 if ch == "\n" else 0))

    names = {}
    for m in API_RE.finditer(text):
        api = m.group(1)
        name = normalise(first_arg(text, m.end() - 1))
        if not name:
            continue
        lineno = text.count("\n", 0, m.start()) + 1
        slot = names.setdefault(name, {"alloc": {}, "free": {}})
        if api in ALLOC_FAMILY:
            slot["alloc"].setdefault(ALLOC_FAMILY[api], []).append((lineno, api))
        else:
            slot["free"].setdefault(FREE_FAMILY[api], []).append((lineno, api))
    return names


def load_allowlist():
    entries = {}
    if not ALLOWLIST.exists():
        return entries
    for lineno, raw in enumerate(ALLOWLIST.read_text().split("\n"), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, _, reason = line.partition("#")
        key = key.strip()
        if not reason.strip():
            print(f"ERROR {ALLOWLIST.name}:{lineno}: entry {key!r} has no reason", file=sys.stderr)
            sys.exit(2)
        entries[key] = reason.strip()
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print every pair, not just mismatches")
    args = ap.parse_args()

    files = sorted(p for r in ROOTS for p in (ROOT / r).rglob("*")
                   if p.suffix in EXTS and "build" not in p.parts)
    allow = load_allowlist()

    mismatches, pairs, scanned = [], 0, 0
    # name -> {'alloc'|'free': {family: [(rel, line, api)]}}, members only
    members = {}
    for path in files:
        rel = str(path.relative_to(ROOT))
        for name, use in scan(path).items():
            if name.endswith("_") and "." not in name and "->" not in name:
                slot = members.setdefault(name, {"alloc": {}, "free": {}})
                for half in ("alloc", "free"):
                    for fam, hits in use[half].items():
                        slot[half].setdefault(fam, []).extend(
                            (rel, ln, api) for ln, api in hits)
            if not use["alloc"] or not use["free"]:
                continue
            pairs += 1
            afam, ffam = set(use["alloc"]), set(use["free"])
            if args.list:
                a = ",".join(f"{ln}:{api}" for f in use["alloc"] for ln, api in use["alloc"][f])
                fr = ",".join(f"{ln}:{api}" for f in use["free"] for ln, api in use["free"][f])
                mark = "OK " if afam == ffam else "BAD"
                print(f"{mark} {rel}:{name}  alloc[{a}]  free[{fr}]")
            if afam != ffam:
                mismatches.append((rel, name, use))
        scanned += 1

    # Pass 2: members whose allocate and free halves disagree ACROSS files. Only
    # members already reported in pass 1 for a single file are skipped, so the same
    # defect is never counted twice.
    seen_in_file = {(rel, name) for rel, name, _ in mismatches}
    cross = 0
    for name, use in sorted(members.items()):
        if not use["alloc"] or not use["free"]:
            continue
        cross += 1
        afam, ffam = set(use["alloc"]), set(use["free"])
        if afam == ffam:
            continue
        files_seen = {r for half in ("alloc", "free") for fam in use[half]
                      for r, _, _ in use[half][fam]}
        if len(files_seen) == 1 and (next(iter(files_seen)), name) in seen_in_file:
            continue  # pass 1 already has it
        use2 = {half: {fam: [(ln, f"{api} [{r}]") for r, ln, api in hits]
                       for fam, hits in use[half].items()} for half in ("alloc", "free")}
        mismatches.append(("member", name, use2))

    unlisted = [m for m in mismatches if f"{m[0]}:{m[1]}" not in allow]
    listed = {f"{m[0]}:{m[1]}" for m in mismatches}
    stale = [k for k in allow if k not in listed]

    print(f"alloc-pairs: {scanned} files, {pairs} pointer(s) with both halves in one file, "
          f"{cross} member(s) with both halves tree-wide, "
          f"{len(mismatches)} mismatch(es), {len(allow)} allowlisted")

    for rel, name, use in unlisted:
        pfx = "" if rel == "member" else f"{rel}:"
        a = ", ".join(f"{pfx}{ln} {api}" for f in use["alloc"] for ln, api in use["alloc"][f])
        fr = ", ".join(f"{pfx}{ln} {api}" for f in use["free"] for ln, api in use["free"][f])
        print(f"MISMATCH {name}\n    allocated: {a}\n    freed:     {fr}")
    for k in stale:
        print(f"STALE allowlist entry (mismatch is gone, remove it): {k}")

    if unlisted or stale:
        print("\nA pointer must go home to the allocator it came from: cudaMalloc<->cudaFree, "
              "cudaMallocAsync<->cudaFreeAsync, cudaMallocHost/cudaHostAlloc<->cudaFreeHost.")
        return 1
    print("OK — every in-file allocate/free pair uses one allocator.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
