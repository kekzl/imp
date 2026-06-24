#!/usr/bin/env python3
"""File-size gate for imp.

Classifies every source file (header / kernel-.cu / normal TU) and compares its
CODE LOC against per-category thresholds from tools/filesize_thresholds.toml.

The metric is a proxy for RECOMPILE BLAST RADIUS, not readability — see the toml
header and docs/AGENTS.md "File Layout & Size".

Exit codes:
  0  no hard-review violation (warn-level smells may still be printed)
  1  at least one NON-allowlisted file exceeds its hard-review threshold
With --warn-only the gate always exits 0 (advisory display step).

Usage:
  python3 tools/check_filesize.py                 # blocking hard gate
  python3 tools/check_filesize.py --warn-only     # advisory, never fails
  python3 tools/check_filesize.py --config X.toml  # alternate config (tests)
  python3 tools/check_filesize.py --root src/compute  # restrict scan roots
"""
import argparse
import os
import sys

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    sys.stderr.write("check_filesize.py needs Python 3.11+ (tomllib)\n")
    sys.exit(2)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "filesize_thresholds.toml")
SRC_EXT = (".cu", ".cuh", ".cpp", ".hpp", ".h")


def code_loc(text):
    """Return (raw_lines, code_lines).

    code_lines = non-blank lines remaining after stripping C/C++ comments with a
    char state machine that does NOT treat // or /* inside string/char literals as
    comment starts (so the count is honest in both directions).
    """
    out = []
    i, n = 0, len(text)
    in_block = in_line = in_str = in_chr = False
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if in_line:
            if c == "\n":
                in_line = False
                out.append(c)
            i += 1
        elif in_block:
            if c == "*" and nxt == "/":
                in_block = False
                i += 2
            else:
                if c == "\n":
                    out.append(c)
                i += 1
        elif in_str:
            out.append(c)
            if c == "\\" and nxt:
                out.append(nxt)
                i += 2
            else:
                if c == '"':
                    in_str = False
                i += 1
        elif in_chr:
            out.append(c)
            if c == "\\" and nxt:
                out.append(nxt)
                i += 2
            else:
                if c == "'":
                    in_chr = False
                i += 1
        elif c == "/" and nxt == "/":
            in_line = True
            i += 2
        elif c == "/" and nxt == "*":
            in_block = True
            i += 2
        elif c == '"':
            in_str = True
            out.append(c)
            i += 1
        elif c == "'":
            in_chr = True
            out.append(c)
            i += 1
        else:
            out.append(c)
            i += 1
    raw = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    code = sum(1 for line in "".join(out).split("\n") if line.strip())
    return raw, code


def load_config(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def classify(relpath, cfg):
    ext = os.path.splitext(relpath)[1]
    if ext in tuple(cfg["classify"]["header_ext"]):
        return "header"
    if ext == ".cu" and any(relpath.startswith(d) for d in cfg["classify"]["kernel_dirs"]):
        return "kernel"
    return "tu"


def scan(cfg, roots):
    rows = []
    for r in roots:
        base = os.path.join(REPO_ROOT, r)
        if not os.path.isdir(base):
            continue
        for dp, _, fns in os.walk(base):
            if "__pycache__" in dp:
                continue
            for fn in fns:
                if os.path.splitext(fn)[1] not in SRC_EXT:
                    continue
                full = os.path.join(dp, fn)
                rel = os.path.relpath(full, REPO_ROOT)
                with open(full, "r", errors="replace") as fh:
                    raw, code = code_loc(fh.read())
                rows.append({"path": rel, "group": classify(rel, cfg), "raw": raw, "code": code})
    return rows


def main():
    ap = argparse.ArgumentParser(description="imp file-size gate")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--warn-only", action="store_true", help="print smells but always exit 0")
    ap.add_argument("--root", action="append", help="override scan roots (repeatable)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    th = cfg["thresholds"]
    allow = cfg.get("allow", {})

    # Validate allowlist: every entry needs a non-empty reason (anti-cheat).
    bad = [p for p, reason in allow.items() if not str(reason).strip()]
    if bad:
        print("ERROR: allowlist entries without a reason:")
        for p in bad:
            print(f"  {p}")
        return 2

    roots = args.root if args.root else cfg["classify"]["roots"]
    rows = scan(cfg, roots)

    warns, hards, allowed = [], [], []
    for r in rows:
        g = r["group"]
        warn_t, hard_t = th[g]["warn"], th[g]["hard"]
        if r["code"] > hard_t:
            r["delta"] = r["code"] - hard_t
            r["limit"] = hard_t
            if r["path"] in allow:
                allowed.append(r)
            else:
                hards.append(r)
        elif r["code"] > warn_t:
            r["delta"] = r["code"] - warn_t
            r["limit"] = warn_t
            warns.append(r)

    def table(title, items, limit_label):
        if not items:
            return
        print(f"\n{title}")
        print(f"  {'code':>5} {'raw':>5} {'+/-':>6}  {limit_label:<6} group    file")
        for r in sorted(items, key=lambda x: -x["delta"]):
            print(f"  {r['code']:>5} {r['raw']:>5} {r['delta']:>+6}  {r['limit']:<6} {r['group']:<8} {r['path']}")

    table(f"WARN ({len(warns)}) — soft smell, not blocking", warns, "warn>")
    table(f"ALLOWLISTED ({len(allowed)}) — over hard-review but accepted in baseline", allowed, "hard>")
    table(f"HARD-REVIEW VIOLATIONS ({len(hards)}) — NOT allowlisted", hards, "hard>")

    print(f"\nscanned {len(rows)} files in {', '.join(roots)} | "
          f"warn={len(warns)} allowlisted={len(allowed)} violations={len(hards)}")

    # Flag stale allowlist entries (path listed but no longer over hard) — advisory.
    over_paths = {r["path"] for r in allowed}
    stale = [p for p in allow if p not in over_paths]
    if stale:
        print(f"\nNOTE: {len(stale)} allowlist entr(y/ies) no longer exceed hard-review "
              f"(safe to remove from [allow]):")
        for p in sorted(stale):
            print(f"  {p}")

    if hards and not args.warn_only:
        print("\nFAIL: hard-review threshold exceeded. Split the file, or — if it is "
              "legitimately monolithic — add it to [allow] in tools/filesize_thresholds.toml "
              "WITH a reason.")
        return 1
    print("\nOK" + (" (warn-only)" if args.warn_only else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
