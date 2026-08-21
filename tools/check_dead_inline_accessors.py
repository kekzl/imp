#!/usr/bin/env python3
"""Fail on a function defined inline in a header that nothing anywhere calls.

WHY THIS EXISTS
---------------
`docs/audit/SETTLED.md` §C records two decl-only sweeps from 2026-08-03 that
removed thirteen never-called functions and closed the class. It did not close
this one, and the reason is a detail of the filter rather than of the tree: that
sweep matched the decl+def signature, i.e. a name occurring exactly TWICE (once
in a header, once in a .cpp). A function DEFINED in the header occurs once, so it
scored as "already unique" and fell out of the candidate set. Twenty of them
survived, all predating that sweep.

Nothing here is a runtime defect: an uncalled `constexpr` accessor costs nothing
at execution. What it costs is the reading of the header - an accessor is an
assertion that some caller needs this value, and twenty untrue assertions is a
header that describes an API nobody uses. This gate keeps the count at zero so
the next reader can trust the ones that remain.

WHAT IT CHECKS
--------------
For every function DEFINED inline in a header under `src/` (the body opens on
the same line as the signature), count word-boundary occurrences of its name
across `src tools tests include`. Exactly one occurrence means the definition is
the only mention: no call, no address taken, no test.

`include/` is deliberately out of scope: it is the public C API, whose consumers
are outside this repo, and "no caller in the tree" says nothing there.

KNOWN LIMITS, stated rather than papered over
---------------------------------------------
This is a text scan, not a compiler. It cannot see a name produced by token
pasting, so a macro-generated caller reads as no caller. It skips constructors,
destructors and `operator`s (a constructor's name is its class, so the count is
never 1). If a legitimate entry ever appears, put it in the allowlist with a
reason - do not widen the regex until it stops finding things.

Usage:
    python3 tools/check_dead_inline_accessors.py           # gate
    python3 tools/check_dead_inline_accessors.py --list    # candidates + counts
"""
import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
HEADER_ROOTS = ("src",)
SEARCH_ROOTS = ("src", "tools", "tests", "include")
HEADER_EXTS = (".h", ".hpp", ".cuh")
SEARCH_EXTS = (".cpp", ".cu", ".h", ".hpp", ".cuh")
ALLOWLIST = ROOT / "tools" / "dead_inline_allowlist.txt"

# An inline definition: optional specifiers, a return type, a name, an argument
# list, optional trailing const/noexcept/override, then `{` on the SAME line.
# Requiring the brace on the line is what separates a definition from a
# declaration, and it is why this finds a class the decl+def sweep could not.
DEF_RE = re.compile(
    r"^[ \t]*"
    r"(?:\[\[nodiscard\]\][ \t]*)?"
    r"(?:(?:static|constexpr|inline|virtual|explicit|friend)[ \t]+)*"
    r"(?:[A-Za-z_][\w:]*(?:[ \t]*<[^;{}]*>)?(?:[ \t]*(?:const|\*|&))*[ \t]+)+"
    r"([A-Za-z_]\w*)[ \t]*\("
    r"[^;{}]*\)[ \t]*"
    r"(?:const[ \t]*)?(?:noexcept[ \t]*)?(?:override[ \t]*)?(?:const[ \t]*)?"
    r"\{")

SKIP_NAMES = {"if", "for", "while", "switch", "return", "catch", "sizeof", "operator"}


def strip_comments(text):
    out, i, n, state = [], 0, len(text), None
    while i < n:
        c, nxt = text[i], text[i + 1] if i + 1 < n else ""
        if state is None:
            if c == "/" and nxt == "/":
                state = "line"; out.append("  "); i += 2; continue
            if c == "/" and nxt == "*":
                state = "block"; out.append("  "); i += 2; continue
            if c == '"': state = "str"
            elif c == "'": state = "chr"
            out.append(c); i += 1; continue
        if state == "line":
            if c == "\n": state = None; out.append(c)
            else: out.append(" ")
            i += 1; continue
        if state == "block":
            if c == "*" and nxt == "/": state = None; out.append("  "); i += 2; continue
            out.append(c if c == "\n" else " "); i += 1; continue
        out.append(c)
        if c == "\\":
            if i + 1 < n: out.append(text[i + 1]); i += 2; continue
        elif (state == "str" and c == '"') or (state == "chr" and c == "'"):
            state = None
        i += 1
    return "".join(out)


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
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    # One pass over the corpus, so the occurrence count is a dict lookup rather
    # than a grep per candidate (which was ~90 s on this tree).
    corpus = []
    for r in SEARCH_ROOTS:
        for p in (ROOT / r).rglob("*"):
            if p.suffix in SEARCH_EXTS and "build" not in p.parts:
                corpus.append((p, strip_comments(p.read_text(errors="ignore"))))

    counts = {}
    for _, text in corpus:
        for tok in re.findall(r"[A-Za-z_]\w*", text):
            counts[tok] = counts.get(tok, 0) + 1

    candidates = []
    for p, text in corpus:
        rel = str(p.relative_to(ROOT))
        if p.suffix not in HEADER_EXTS or not any(rel.startswith(r + "/") for r in HEADER_ROOTS):
            continue
        # Class name in scope, so a constructor is not mistaken for a function.
        classes = set(re.findall(r"\b(?:class|struct)\s+([A-Za-z_]\w*)", text))
        for lineno, line in enumerate(text.split("\n"), 1):
            m = DEF_RE.match(line)
            if not m:
                continue
            name = m.group(1)
            if name in SKIP_NAMES or name in classes or name.startswith("operator"):
                continue
            if counts.get(name, 0) == 1:
                candidates.append((rel, lineno, name))

    allow = load_allowlist()
    unlisted = [c for c in candidates if f"{c[0]}:{c[2]}" not in allow]
    listed = {f"{c[0]}:{c[2]}" for c in candidates}
    stale = [k for k in allow if k not in listed]

    if args.list:
        for rel, lineno, name in candidates:
            print(f"{rel}:{lineno}  {name}  (occurrences: {counts.get(name, 0)})")

    print(f"dead-inline: {len(corpus)} files scanned, {len(candidates)} header-inline "
          f"definition(s) with no other mention, {len(allow)} allowlisted")
    for rel, lineno, name in unlisted:
        print(f"DEAD {rel}:{lineno}  {name}()  - defined here and mentioned nowhere else")
    for k in stale:
        print(f"STALE allowlist entry (no longer dead, remove it): {k}")
    if unlisted or stale:
        print("\nAn accessor nobody calls is a claim that some caller needs the value. "
              "Delete it, or allowlist it with the reason it must stay.")
        return 1
    print("OK - every header-inline definition under src/ is mentioned somewhere else.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
