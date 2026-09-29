#!/usr/bin/env python3
"""Verify `path:line anchor` and bare `docs/*.md` citations in the LIVING docs.

Cite format: `path:LINES anchor`, one backtick span.
  LINES  = N | N-M (range) | N,M,... or N/M/... (list)
  anchor = literal text that must appear on the cited line (range: within it;
           list: on every listed line). No anchor = DEAD (#2185: existence-only
           checks passed two Makefile cites that pointed 16 and 81 lines off).
Verdict per span (#2231: line shifts alone turned main red 3x on 2026-09-29):
  anchor on the cited line(s)                         -> ok
  anchor elsewhere, exactly once in the file          -> DRIFT (warning, exit 0)
  anchor several times, exactly one within WINDOW     -> DRIFT to that one
  anchor several times, none or 2+ within WINDOW      -> DEAD (ambiguous)
  anchor not in the file                              -> DEAD
`--fix` rewrites drifted line numbers in place; a second run reports 0 drift.
A bare basename is resolved by the anchor when several files share the name.
Records (docs/archive/, docs/plans/, docs/audit/) are out of scope.
"""
import os
import re
import shutil
import sys
import tempfile

WINDOW = 25  # max distance (lines) from the cited span to pick one of several anchor hits

EXT = r"cpp|cu|cuh|h|hpp|c|py|sh|md|txt|cmake|json|toml|yml|yaml|conf"
PATH = r"((?:[\w./-]+/)?((?:[\w.-]+\.(?:" + EXT + r"))|Makefile|Dockerfile))"
CITE_START_RE = re.compile(PATH + r":\d")
CITE_RE = re.compile(PATH + r":(\d+(?:[-–]\d+)?(?:[,/]\d+(?:[-–]\d+)?)*)(?: (.*))?")
PART_RE = re.compile(r"(\d+)(?:([-–])(\d+))?")


def _spans(spec):
    return [(int(m.group(1)), int(m.group(3) or m.group(1))) for m in PART_RE.finditer(spec)]


def _respec(spec, spans):
    """spec with each N / N-M replaced by the new spans, separators kept."""
    it = iter(spans)

    def sub(m):
        a, b = next(it)
        return f"{a}{m.group(2)}{b}" if m.group(2) else str(a)
    return PART_RE.sub(sub, spec)


def _lines(path, cache={}):
    if path not in cache:
        with open(path, encoding="utf-8", errors="replace") as fh:
            cache[path] = fh.read().split("\n")
    return cache[path]


def _resolve(full, spans, anchor):
    """(new_spans, reason). new_spans == spans: ok; other list: drift; None: dead."""
    src = _lines(full)
    n = len(src) - (1 if src and src[-1] == "" else 0)
    hits = [i + 1 for i, line in enumerate(src[:n]) if anchor in line]
    out = []
    for a, b in spans:
        if 1 <= a <= b <= n and anchor in "\n".join(src[a - 1:b]):
            out.append((a, b))
            continue
        where = f"line {a}{'-' + str(b) if b != a else ''}"
        if not hits:
            return None, f"anchor '{anchor}' not in the file (cited {where})"
        pick = hits
        if len(hits) > 1:
            pick = [h for h in hits if (a - h if h < a else h - b if h > b else 0) <= WINDOW]
        if len(pick) != 1:
            return None, (f"anchor '{anchor}' ambiguous: {len(hits)} hits (lines "
                          f"{', '.join(map(str, hits[:8]))}), {len(pick)} within {WINDOW} of {where}")
        h = pick[0]
        d = h - a if h < a else h - b  # smallest shift that puts the hit inside the span
        out.append((a + d, b + d))
    return out, ""


def _index(root):
    index = {}
    for base, _dirs, files in os.walk(root):
        if any(x in base for x in (".git", "build", "third_party", "node_modules")):
            continue
        for f in files:
            index.setdefault(f, []).append(os.path.join(base, f))
    return index


def check(doc, root, index=None, fix=False):
    """(dead, ambiguous, drift) message lists; fix=True rewrites drifted specs in doc."""
    index = index if index is not None else _index(root)
    text = open(doc, encoding="utf-8").read()
    bad, ambiguous, drift, edits = [], [], [], []

    for sm in re.finditer(r"`([^`\n]+)`", text):
        span, ln = sm.group(1), text.count("\n", 0, sm.start()) + 1
        if not CITE_START_RE.match(span):
            continue
        m = CITE_RE.fullmatch(span)
        if not m:
            bad.append(f"{ln}: {span} - malformed cite: write `path:N anchor`, N as N, N-M, N,M or N/M")
            continue
        path, basename, spec, anchor = m.group(1), m.group(2), m.group(3), (m.group(4) or "").strip()
        cite = f"{path}:{spec}"
        if not anchor:
            bad.append(f"{ln}: {cite} - no anchor: write `{cite} <text on that line>`")
            continue
        spans = _spans(spec)
        cands = [p for p in (os.path.join(root, path), os.path.join(os.path.dirname(doc), path)) if os.path.isfile(p)]
        cands = cands[:1] or index.get(basename, [])
        if not cands:
            bad.append(f"{ln}: {cite} - no file of that name in the tree")
            continue
        results = [(c,) + _resolve(c, spans, anchor) for c in cands]
        exact = [c for c, new, _ in results if new == spans]
        moved = [(c, new) for c, new, _ in results if new is not None and new != spans]
        if len(exact) > 1:
            ambiguous.append(f"{ln}: {cite} - anchor matches {len(exact)} files named {basename}; cite the path")
        if exact:
            continue
        if len(moved) == 1:
            new_spec = _respec(spec, moved[0][1])
            drift.append(f"{ln} {cite} -> {new_spec} ({anchor})")
            edits.append((sm.start() + 1 + m.start(3), sm.start() + 1 + m.end(3), new_spec))
        elif len(cands) == 1:
            bad.append(f"{ln}: {cite} - {results[0][2]}")
        elif not moved:
            bad.append(f"{ln}: {cite} - anchor '{anchor}' matches none of the {len(cands)} files named {basename}; cite the path")
        else:
            bad.append(f"{ln}: {cite} - anchor moved in {len(moved)} files named {basename}; cite the path")

    # bare docs/*.md names (markdown links are already covered elsewhere)
    for m in re.finditer(r"`(docs/[\w./-]+\.md)`", text):
        if not os.path.exists(os.path.join(root, m.group(1))):
            bad.append(f"{text.count(chr(10), 0, m.start()) + 1}: {m.group(1)} - referenced file does not exist")

    if fix and edits:
        for s, e, new in sorted(edits, reverse=True):
            text = text[:s] + new + text[e:]
        with open(doc, "w", encoding="utf-8") as fh:
            fh.write(text)
    return bad, ambiguous, drift


def living_docs(root):
    """Every doc whose citations must stay live. Records are excluded on
    purpose: docs/archive/, docs/plans/ and docs/audit/ cite the line numbers
    of the commit they describe, and rewriting those would destroy the record
    (same reason docs_lint.py excludes roadmap.md)."""
    import glob
    docs = [os.path.join(root, "docs/roadmap.md")]
    docs += sorted(glob.glob(os.path.join(root, "docs/*.md")))
    docs += sorted(glob.glob(os.path.join(root, "docs/internals/*.md")))
    for extra in ("README.md", "CONTRIBUTING.md", "AGENTS.md", "AUDIT.md"):
        p = os.path.join(root, extra)
        if os.path.exists(p):
            docs.append(p)
    seen, out = set(), []
    for d in docs:
        rp = os.path.realpath(d)
        if rp not in seen:
            seen.add(rp)
            out.append(d)
    return out


def run(root, docs, fix=False):
    """Prints the report; returns the exit code (1 only on dead citations)."""
    total_bad = total_drift = 0
    index = _index(root)
    for doc in docs:
        bad, ambiguous, drift = check(doc, root, index, fix)
        rel = os.path.relpath(doc, root)
        for b in sorted(set(bad)):
            print(f"  DEAD      {rel}:{b}")
        for a in sorted(set(ambiguous)):
            print(f"  AMBIGUOUS {rel}:{a}")
        for d in drift:
            print(f"  {'FIXED' if fix else 'DRIFT'}     {rel}:{d}")
        total_bad += len(set(bad))
        total_drift += len(drift)
    hint = "rewritten" if fix else "warning, `--fix` rewrites"
    print(f"{'FAIL' if total_bad else 'PASS'}: {total_bad} dead citation(s), "
          f"{total_drift} drifted ({hint}) across {len(docs)} living doc(s)")
    return 1 if total_bad else 0


def selftest():
    src = "\n".join(["// filler"] * 10 + ["int alpha_ = 0;", "int beta_ = 1;", "int twin_;"]
                    + ["// gap"] * 60 + ["int twin_;", "int gamma_ = 2;"]) + "\n"
    cases_doc = {  # name -> (doc line, expected (dead, drift))
        "exact line -> ok": ("`src/a.h:11 alpha_`", (0, 0)),
        "anchor moved +1 -> DRIFT": ("`src/a.h:10 alpha_`", (0, 1)),
        "range drift keeps its width": ("`src/a.h:1-2 beta_`", (0, 1)),
        "anchor deleted -> DEAD": ("`src/a.h:11 deleted_`", (1, 0)),
        "anchor twice, cited line far from both -> DEAD": ("`src/a.h:43 twin_`", (1, 0)),
        "anchor twice, one within WINDOW -> DRIFT": ("`src/a.h:20 twin_`", (0, 1)),
        "cite past EOF, anchor present -> DRIFT": ("`src/a.h:500 gamma_`", (0, 1)),
        "list cite, one part drifted -> DRIFT": ("`src/a.h:11,70 alpha_`", (0, 1)),
        "no anchor -> DEAD": ("`src/a.h:11`", (1, 0)),
    }
    cases, tmp = [], tempfile.mkdtemp(prefix="citesel")
    try:
        os.makedirs(os.path.join(tmp, "src"))
        with open(os.path.join(tmp, "src/a.h"), "w") as fh:
            fh.write(src)
        index = _index(tmp)
        for i, (name, (line, want)) in enumerate(cases_doc.items()):
            doc = os.path.join(tmp, f"d{i}.md")
            with open(doc, "w") as fh:
                fh.write(f"# t\n\n{line}\n")
            bad, _amb, drift = check(doc, tmp, index)
            cases.append((f"{name} {(len(bad), len(drift))}", (len(bad), len(drift)) == want))
        doc = os.path.join(tmp, f"d{len(cases_doc)}.md")
        body = "`src/a.h:10 alpha_` and `src/a.h:1-2 beta_` and `src/a.h:500 gamma_` and `src/a.h:11,70 alpha_`\n"
        with open(doc, "w") as fh:
            fh.write(body)
        _, _, d1 = check(doc, tmp, index, fix=True)
        fixed = open(doc).read()
        bad2, _, d2 = check(doc, tmp, index, fix=True)
        cases.append((f"--fix rewrites {len(d1)} spans", len(d1) == 4 and fixed ==
                      "`src/a.h:11 alpha_` and `src/a.h:11-12 beta_` and `src/a.h:75 gamma_` and `src/a.h:11,11 alpha_`\n"))
        cases.append((f"second run: {len(d2)} drift, {len(bad2)} dead, file unchanged",
                      not d2 and not bad2 and open(doc).read() == fixed))
    finally:
        shutil.rmtree(tmp)
    for name, ok in cases:
        print(f"  {'ok ' if ok else 'BAD'} {name}")
    nbad = sum(not ok for _, ok in cases)
    print(f"selftest: {len(cases) - nbad}/{len(cases)} planted cases pass")
    return 1 if nbad else 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    root = args[0] if args else "."
    sys.exit(run(root, [args[1]] if len(args) > 1 else living_docs(root), fix="--fix" in sys.argv))
