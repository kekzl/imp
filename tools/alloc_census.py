#!/usr/bin/env python3
"""Group the remaining I1 sites by the BUFFER they touch, not by the file.

`check_alloc_sites.py` answers "which files still allocate" — the gate question.
This answers the planning one: *what* is still allocated, and where its
acquisition and its releases live. They are different questions, and the file
view actively misleads on the second (AUDIT B59): the KV block tables are 8
acquisitions against 23 releases spread over five files, and one allocation is
freed at eight sites depending on which path unwinds. A file-at-a-time migration
cannot close a buffer like that, because moving the allocation means moving
frees out of files you were not editing.

What it cannot do, stated so nobody reads more into the output than is there.
It matches on the TEXT of the allocated expression, which fails in both
directions, and the two worked examples are in the tree today:

  - FALSE MERGE. `d_token_allow_` reports as one family across four files. It is
    four *different* buffers — grammar_constrain.h:94, regex_constrain.h:101,
    json_constrain.h:172 and schema_constrain.h:215 each declare their own
    member of that name. Sibling classes with identical member names are the
    common case in this codebase, so **a cross-file row is a hypothesis to
    check, not a finding**. `d_block_tables` is the opposite case: genuinely one
    buffer, threaded through four files, verified by hand in B59.
  - FALSE SPLIT. A buffer renamed between its acquisition and its release
    splits into two families. B59 hit exactly that: `d_bt` is acquired as a
    local in engine_graph_decode.cpp and freed as `async_d_block_tables_` in
    three other files. No textual tool can join them.

It also says nothing about lifetime or hot-path-ness. Criterion 3 already
measures that (0 allocations while serving); this is about ownership.

Usage:
    python3 tools/alloc_census.py            # families, largest first
    python3 tools/alloc_census.py --pairs    # only families that look unpaired
    python3 tools/alloc_census.py --file F   # one file's families
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src"
EXEMPT_PREFIXES = ("src/memory/",)
SUFFIXES = (".cpp", ".cu", ".h", ".cuh", ".hpp")

ACQUIRE = ("cudaMallocAsync", "cudaMallocManaged", "cudaMallocPitch", "cudaMallocHost",
           "cudaMalloc3D", "cudaMalloc", "cudaHostAlloc", "cudaHostRegister")
RELEASE = ("cudaFreeAsync", "cudaFreeHost", "cudaFree", "cudaHostUnregister")

# The first argument of the call, with the address-of and any cast peeled off.
# The call plus its first argument. `\s` spans newlines because four sites in
# executor_forward_moe_cutlass.cu put the argument on the next line, and a
# line-based match silently dropped them — the census is only useful if it can
# say it saw everything.
CALL = re.compile(
    r"\b(" + "|".join(ACQUIRE + RELEASE) + r")\s*\(\s*"
    r"(?:reinterpret_cast\s*<[^>]*>\s*\(\s*)?"
    r"&?\s*\(?\*?\s*"
    r"([A-Za-z_][A-Za-z0-9_]*(?:\s*(?:\.|->)\s*[A-Za-z_][A-Za-z0-9_]*)*)",
    re.S)
ANY_CALL = re.compile(r"\b(" + "|".join(ACQUIRE + RELEASE) + r")\s*\(")
COMMENT = re.compile(r"^\s*(//|\*|/\*)")

# Container prefixes that say where a buffer lives, not what it is. Stripping
# them is what merges `ws.h_expert_indices` and `ctx.h_expert_indices` into one
# family; keeping them would split every workspace member by call site.
CONTAINER_PREFIX = re.compile(r"^(ws|ctx|p|c|buf|entry|w|e|slot|cache|wcache_|moe_|qscratch_)\s*(\.|->)\s*")

# Names that carry no identity: the same `p` in twelve files is twelve unrelated
# locals, and grouping them produced the largest "family" in the first run of
# this tool — an artefact that would have sent the next migration at a coincidence.
# These get qualified by file, so they still show up but as what they are.
GENERIC = {"p", "ptr", "buf", "tmp", "out", "dst", "src", "data", "mem", "h", "d",
           "d_a", "d_b", "d_c", "raw", "hp", "result", "scratch"}


def is_generic(name: str) -> bool:
    return name in GENERIC or len(name) <= 3


def normalize(expr: str) -> str:
    expr = re.sub(r"\s+", "", expr)
    prev = None
    while prev != expr:
        prev = expr
        expr = CONTAINER_PREFIX.sub("", expr)
    return expr


unparsed: list[tuple[str, int]] = []


def scan():
    """buffer -> {'acquire': [(file, line)], 'release': [(file, line)]}"""
    fams: dict[str, dict[str, list]] = collections.defaultdict(
        lambda: {"acquire": [], "release": []})
    for path in sorted(SRC.rglob("*")):
        if path.suffix not in SUFFIXES or not path.is_file():
            continue
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith(EXEMPT_PREFIXES):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        keep = []
        offsets = []
        pos = 0
        for line in lines:
            if not COMMENT.match(line):
                keep.append(line)
                offsets.append(pos)
            pos += len(line) + 1
        body = "\n".join(keep)
        line_of = {}
        run = 0
        for i, line in enumerate(keep):
            line_of[run] = i
            run += len(line) + 1

        def lineno(off: int) -> int:
            starts = [s for s in line_of if s <= off]
            return lines.index(keep[line_of[max(starts)]]) + 1 if starts else 0

        seen = set()
        def in_string(off: int) -> bool:
            bol = body.rfind("\n", 0, off) + 1
            seg = body[bol:off]
            return (seg.count('"') - seg.count('\\"')) % 2 == 1

        for m in CALL.finditer(body):
            if in_string(m.start()):
                continue
            api, expr = m.group(1), normalize(m.group(2))
            kind = "acquire" if api in ACQUIRE else "release"
            key = f"{expr}  [{rel}]" if is_generic(expr) else expr
            fams[key][kind].append((rel, lineno(m.start())))
            seen.add(m.start())
        for m in ANY_CALL.finditer(body):
            if m.start() not in seen and not in_string(m.start()):
                unparsed.append((rel, lineno(m.start())))
    return fams


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", action="store_true",
                    help="only families whose acquisitions and releases do not line up")
    ap.add_argument("--file", help="restrict to families touching this file")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    fams = scan()
    if args.file:
        fams = {k: v for k, v in fams.items()
                if any(args.file in f for f, _ in v["acquire"] + v["release"])}

    rows = []
    for name, sites in fams.items():
        a, r = len(sites["acquire"]), len(sites["release"])
        files = {f for f, _ in sites["acquire"] + sites["release"]}
        rows.append((a + r, a, r, len(files), name, sites))

    if args.pairs:
        # An acquisition with no release, or releases with no acquisition, is
        # either a leak, a process-lifetime buffer, or — most often here — a
        # buffer whose two halves are spelled differently (B59).
        rows = [x for x in rows if x[1] == 0 or x[2] == 0 or x[3] > 1]

    rows.sort(reverse=True)
    total_a = sum(x[1] for x in rows)
    total_r = sum(x[2] for x in rows)
    print(f"{len(rows)} buffer families  ({total_a} acquisitions, {total_r} releases)")
    if unparsed:
        # Silence here would be the same bug the gate had: a number that omits
        # what it could not read.
        print(f"  {len(unparsed)} call(s) whose target this tool could not name:")
        for f, n in unparsed:
            print(f"      {f}:{n}")
    print(f"{'total':>5} {'acq':>4} {'rel':>4} {'files':>5}  buffer")
    for total, a, r, nfiles, name, sites in rows[:args.top]:
        mark = "  <- cross-file: verify, may be same-named siblings" if nfiles > 1 else ""
        print(f"{total:5d} {a:4d} {r:4d} {nfiles:5d}  {name}{mark}")
        if nfiles > 1:
            where = collections.Counter(f for f, _ in sites["acquire"] + sites["release"])
            for f, c in where.most_common():
                print(f"{'':22}{c:3d}  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
