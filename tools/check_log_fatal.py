#!/usr/bin/env python3
"""Fail when a site logs at FATAL and then carries on as if nothing happened.

WHY THIS EXISTS
---------------
`IMP_LOG_FATAL` sounds like it ends the process. It does not. `logging.h:58`
makes it a plain `log_message(LogLevel::FATAL, ...)` call; `IMP_CHECK`
(`logging.h:68-74`) is the only thing in the tree that reaches `std::abort()`,
and its own comment says so. So the macro's name promises what only its sibling
delivers, and every site that wrote `IMP_LOG_FATAL(...)` expecting the process
to stop got a log line instead.

Measured 2026-08-21: of 12 sites, **ten continued wrongly**. Three of them said
so in their own comments first - "Continuing would hand a host pointer to a
device kernel", "say so loudly rather than fall through" - and then fell
through. One reported `WeightRegistry::handle: id out of range` and indexed out
of range on the next line. Two returned from a dispatch leaving the output
tensor holding whatever it held before, which is the shape #654 removed from
attention_prefill_dispatch (SETTLED.md S-22).

The failure is not that people wrote the wrong macro. It is that the wrong macro
is indistinguishable from the right one at the call site, so a reviewer reading
"FATAL" reads "stops here".

WHAT IT CHECKS
--------------
For each `IMP_LOG_FATAL(...)` call outside `logging.h`, what the code does next:

    aborts   -> the statement is followed by std::abort()
    throws   -> ... by a throw
    returns  -> ... by a return, which is only correct where the function's
                contract IS to report a verdict rather than to stop
    falls    -> nothing: control continues into the operation just declared
                impossible

`aborts` and `throws` pass. `returns` and `falls` must be justified in
`tools/log_fatal_allowlist.txt` with a reason, and a stale entry fails too, so
the list cannot rot in either direction.

Entries are keyed on `path:<first words of the message>`, NOT on a line number.
The first version used line numbers and immediately rotted: adding four comment
lines above a site moved it 587 -> 593 and the gate reported both an
unjustified site and a stale entry for the same unchanged code. A message
prefix survives edits above it and says at a glance which site is meant.

The check is deliberately syntactic and shallow. It cannot know whether a
`return` is correct; that is what the allowlist reason is for. What it CAN do is
make a new one impossible to add silently, which is the whole defect.

Usage:
    python3 tools/check_log_fatal.py           # gate
    python3 tools/check_log_fatal.py --list    # every site with its verdict
"""
import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ROOTS = ("src", "tools", "include")
EXTS = (".cpp", ".cu", ".h", ".cuh", ".hpp")
ALLOWLIST = ROOT / "tools" / "log_fatal_allowlist.txt"


def classify(lines, i):
    """What happens after the IMP_LOG_FATAL statement starting at line index i."""
    depth, j, started = 0, i, False
    while j < len(lines):
        depth += lines[j].count("(") - lines[j].count(")")
        if "(" in lines[j]:
            started = True
        if started and depth <= 0:
            break
        j += 1
    for k in range(j + 1, min(j + 6, len(lines))):
        t = lines[k].strip()
        if not t or t.startswith("//"):
            continue
        if t.startswith("std::abort") or t.startswith("abort()"):
            return "aborts"
        if t.startswith("throw"):
            return "throws"
        if t.startswith("return"):
            return "returns"
        if t == "}":
            continue  # closing the `if` the log sits in; keep looking
        return "falls"
    return "falls"


def message_key(lines, i):
    """First few words of the log message: a key that survives edits above it."""
    depth, j, started, text = 0, i, False, []
    while j < len(lines):
        text.append(lines[j])
        depth += lines[j].count("(") - lines[j].count(")")
        if "(" in lines[j]:
            started = True
        if started and depth <= 0:
            break
        j += 1
    m = re.search(r'"([^"]{4,})"', "\n".join(text))
    return " ".join(m.group(1).split()[:5]) if m else f"line{i + 1}"


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

    sites = []
    for r in ROOTS:
        for p in sorted((ROOT / r).rglob("*")):
            if p.suffix not in EXTS or "build" in p.parts or p.name == "logging.h":
                continue
            lines = p.read_text(errors="ignore").split("\n")
            for i, l in enumerate(lines):
                if "IMP_LOG_FATAL(" in l and not l.lstrip().startswith("//"):
                    sites.append((str(p.relative_to(ROOT)), i + 1, classify(lines, i),
                                  message_key(lines, i)))

    allow = load_allowlist()
    bad = [s for s in sites if s[2] in ("returns", "falls") and f"{s[0]}:{s[3]}" not in allow]
    listed = {f"{s[0]}:{s[3]}" for s in sites if s[2] in ("returns", "falls")}
    stale = [k for k in allow if k not in listed]

    if args.list:
        for rel, ln, v, key in sites:
            print(f"  {v:8s} {rel}:{ln}  key={key!r}")

    print(f"log-fatal: {len(sites)} IMP_LOG_FATAL site(s), "
          f"{sum(1 for s in sites if s[2] == 'aborts')} abort, "
          f"{sum(1 for s in sites if s[2] == 'throws')} throw, "
          f"{len(listed)} continue ({len(allow)} allowlisted)")
    for rel, ln, v, key in bad:
        print(f"CONTINUES-WRONGLY {rel}:{ln}  logs at FATAL and then {v}")
        print(f"                  allowlist key if this is correct: {rel}:{key}")
    for k in stale:
        print(f"STALE allowlist entry (the site no longer continues, remove it): {k}")
    if bad or stale:
        print("\nIMP_LOG_FATAL only LOGS. Use IMP_CHECK to abort, or throw "
              "(imp_api.cpp translates to ImpError), or justify the site in "
              f"{ALLOWLIST.name} with the reason continuing is correct.")
        return 1
    print("OK - no site logs at FATAL and then carries on unjustified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
