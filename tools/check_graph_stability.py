#!/usr/bin/env python3
"""I3 gate: a captured graph never holds an address that can move.

docs/internals/MEMORY.md invariant I3: a CUDA graph bakes device pointers, so a buffer it can
bake is either held as a StableSpan from mint to launch, or its owner bumps a generation that
every graph runner checks before replay. This gate counts, in src/ outside src/memory/, the
three places where that is not yet true, against tools/graph_stability_allowlist.txt:

  G1 regrow sites      a release (vram_free, device_free_all*, cudaFree*) of a member or static
                       X followed within 25 lines of the same function by an allocation into X.
                       The owner spelling counts too: X.reset(); then X = DeviceOwned<T>::allocate*
                       or X.adopt*, or X.set_owned / X.set_view (ViewOrOwned); `(void)` and `->` spellings too.
                       A site is cleared only by what this tool reads in the source:
                         guarded       a bump_graph_epoch( / bump_graph_epoch_if_moved( / bump_generation(
                                       call in the same function body, or a comment at the site
                                       `// I3-bumped-by: <function>` naming a function defined in src/
                                       whose body holds that call;
                         not bakeable  a comment at the site `// I3-not-bakeable: <reason>` (3+ words),
                                       counted and listed per file as G1n and printed on every run.
                       "At the site" = on the release line or in the comment block directly above it.
                       G1 counts the rest (unguarded, target 0); a marker without a reason, naming
                       no bumping function, or at no unguarded regrow site fails outright.
  G2 unkeyed captures  cudaStreamBeginCapture* calls in a file that reads no generation
                       (a generation getter call, a comparison against a stored generation, or
                       a read of the process graph epoch, graph_epoch() of core/graph_epoch.h).
  G3 discarded proofs  take / take_bytes call lines whose result is not declared StableSpan<T>
                       or passed to X.set_view( on the same line (ViewOrOwned keeps the span).

The list only shrinks: a file over its budget fails, a file that is not listed fails, and an
entry above what the file still holds fails as stale.

Usage:
    python3 tools/check_graph_stability.py            # check (CI)
    python3 tools/check_graph_stability.py --update   # rewrite the allowlist
    python3 tools/check_graph_stability.py --selftest
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src"
ALLOWLIST = REPO / "tools" / "graph_stability_allowlist.txt"
EXEMPT_PREFIX = "src/memory/"
SUFFIXES = (".cpp", ".cu", ".h", ".cuh", ".hpp")
METRICS = ("G1", "G1n", "G2", "G3")
NAMES = {
    "G1": "buffer(s) freed and reallocated in place with no epoch bump",
    "G1n": "regrow site(s) marked I3-not-bakeable",
    "G2": "graph capture(s) in a file that reads no generation",
    "G3": "take site(s) not kept as a StableSpan",
}
WINDOW = 25

COMMENT_LINE = re.compile(r"^\s*(//|\*|/\*)")
STRING = re.compile(r'"(?:[^"\\]|\\.)*"')
RELEASE = re.compile(r"\b(?:vram_free|device_free_all\w*|cudaFree\w*)\s*\(([^;]*)\)\s*;")
LVALUE = re.compile(r"[A-Za-z_]\w*(?:(?:->|\.)\w+|\[\w+\])*")
# Storage that outlives the call: a member (trailing underscore, -> or .) or a static (s_, g_).
PERSISTENT = re.compile(r"_$|->|\.|^(?:s_|g_)")
ALLOC_ASSIGN = r"\s*=[^=;]*\b(?:vram_alloc|vram_alloc_force|weights_alloc|cudaMalloc\w*)\s*\("
ALLOC_OUT = r"\b(?:device_alloc_all\w*|device_alloc_pool_async|cudaMalloc\w*)\s*\([^;]*(?:\(|&|,\s*)\s*"
# The owner spelling of the same regrow: X.reset(); then X = DeviceOwned<T>::allocate*(...) or X.adopt*(...).
# `(void)X->reset();` and `bool moved = X.reset();` are the same release (ViewOrOwned reports whether the address moved).
OWNER_RESET = re.compile(r"^\s*(?:\(void\)\s*|(?:const\s+)?bool\s+\w+\s*=\s*)?([A-Za-z_]\w*(?:(?:->|\.)\w+|\[\w+\])*)(?:\.|->)reset\(\)\s*;")
OWNER_ALLOC = r"(?:\s*=[^=;]*\bDeviceOwned\s*<|(?:\.|->)adopt(?:_async)?\s*\(|(?:\.|->)set_(?:owned|view)\s*\()"
FUNCTION_END = re.compile(r"^}")
BUMP = re.compile(r"\bbump_graph_epoch(?:_if_moved)?\s*\(|\bbump_generation\s*\(")
MARK_NOT_BAKEABLE = re.compile(r"//.*?\bI3-not-bakeable:(.*)$")
MARK_BUMPED_BY = re.compile(r"//.*?\bI3-bumped-by:\s*([A-Za-z_][\w:]*)?")
MARK_ANY = re.compile(r"//.*?\bI3-(?:not-bakeable|bumped-by):")
REASON_WORDS = 3
CAPTURE = re.compile(r"\bcudaStreamBeginCapture\w*\s*\(")
GENERATION = re.compile(r"\w*generation\w*\s*\(\s*\)|\w*generation\w*\s*!=|!=\s*\w*generation\w*"
                        r"|\bgraph_epoch\s*\(\s*\)")
TAKE = re.compile(r"\.take(?:_bytes)?(?:<[^>]+>)?\(")
KEPT = re.compile(r"\bStableSpan\s*<|(?:\.|->)set_view\s*\(")


def code_lines(text: str) -> list[str]:
    """Source lines with comment lines, trailing // comments and string literals blanked."""
    out = []
    for line in text.splitlines():
        if COMMENT_LINE.match(line):
            out.append("")
            continue
        out.append(STRING.sub('""', line).split("//", 1)[0])
    return out


def split_args(args: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in args:
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    return parts


def owner_regrow(lines: list[str], i: int) -> bool:
    m = OWNER_RESET.match(lines[i])
    if not m or not PERSISTENT.search(m.group(1)):
        return False
    again = re.compile(r"(?<![\w.>])" + re.escape(m.group(1)) + OWNER_ALLOC)
    for j in range(i + 1, min(len(lines), i + 1 + WINDOW)):
        if FUNCTION_END.match(lines[j]):
            return False
        if again.search(lines[j] + " " + (lines[j + 1].strip() if j + 1 < len(lines) else "")):
            return True
    return False


def regrow_sites(lines: list[str]) -> list[int]:
    """Line index of the release of every regrow site."""
    sites = []
    for i, line in enumerate(lines):
        m = RELEASE.search(line)
        if not m:
            # Adjacent owner resets are one site, as one multi-argument free call is.
            if owner_regrow(lines, i) and not (i > 0 and owner_regrow(lines, i - 1)):
                sites.append(i)
            continue
        hit = False
        for arg in split_args(m.group(1)):
            if not LVALUE.fullmatch(arg) or not PERSISTENT.search(arg):
                continue
            x = re.escape(arg)
            assign = re.compile(r"(?<![\w.>])" + x + ALLOC_ASSIGN)
            outp = re.compile(ALLOC_OUT + x + r"\b")
            for j in range(i + 1, min(len(lines), i + 1 + WINDOW)):
                if FUNCTION_END.match(lines[j]):
                    break
                two = lines[j] + " " + (lines[j + 1].strip() if j + 1 < len(lines) else "")
                if assign.search(two) and re.match(r"\s*" + x + r"\s*=", lines[j]) or outp.search(lines[j]):
                    hit = True
                    break
            if hit:
                break
        if hit:
            sites.append(i)
    return sites


def count_regrow(lines: list[str]) -> int:
    return len(regrow_sites(lines))


def function_bumps(name: str, files: dict[str, list[str]]) -> bool | None:
    """True if a definition of `name` in files holds a bump call, False if none does, None if undefined."""
    head = re.compile(r"^[A-Za-z_].*(?<!\w)" + re.escape(name) + r"\s*\(")
    found = None
    for lines in files.values():
        for i, line in enumerate(lines):
            if not head.match(line):
                continue
            j = i
            while j < len(lines) and "{" not in lines[j] and ";" not in lines[j]:
                j += 1
            if j == len(lines) or "{" not in lines[j].split(";", 1)[0]:
                continue  # a declaration
            end = next((k for k in range(j + 1, len(lines)) if FUNCTION_END.match(lines[k])), len(lines))
            if any(BUMP.search(l) for l in lines[j:end]):
                return True
            found = False
    return found


def site_marker(raw: list[str], i: int) -> tuple[int, str] | None:
    """The I3 marker line at site i: on the release line, or in the comment block directly above."""
    if MARK_ANY.search(raw[i]):
        return i, raw[i]
    j = i - 1
    while j >= 0 and COMMENT_LINE.match(raw[j]):
        if MARK_ANY.search(raw[j]):
            return j, raw[j]
        j -= 1
    return None


def classify(raw: list[str], lines: list[str], files: dict[str, list[str]]) -> dict:
    """Every regrow site as unguarded, guarded or marked not bakeable; errors fail the gate outright."""
    out = {"unguarded": [], "guarded": [], "marked": [], "errors": []}
    used = set()
    for i in regrow_sites(lines):
        begin = next((j + 1 for j in range(i - 1, -1, -1) if FUNCTION_END.match(lines[j])), 0)
        end = next((j for j in range(i + 1, len(lines)) if FUNCTION_END.match(lines[j])), len(lines))
        if any(BUMP.search(l) for l in lines[begin:end]):
            out["guarded"].append(i)
            continue
        mark = site_marker(raw, i)
        if mark is None:
            out["unguarded"].append(i)
            continue
        used.add(mark[0])
        by = MARK_BUMPED_BY.search(mark[1])
        if by:
            bumps = function_bumps(by.group(1), files) if by.group(1) else None
            if bumps:
                out["guarded"].append(i)
                continue
            out["unguarded"].append(i)
            out["errors"].append((mark[0], "I3-bumped-by names no function" if not by.group(1) else
                                  f"I3-bumped-by: {by.group(1)} is "
                                  + ("defined nowhere in src/" if bumps is None else "defined without a bump call")))
            continue
        reason = MARK_NOT_BAKEABLE.search(mark[1]).group(1).strip()
        if len(re.findall(r"\w+", reason)) < REASON_WORDS:
            out["unguarded"].append(i)
            out["errors"].append((mark[0], f"I3-not-bakeable needs a reason of {REASON_WORDS}+ words on its line"))
            continue
        # The printed claim is the whole comment from the marker down to the site.
        more = [re.sub(r"^\s*//+\s?", "", raw[k]).strip() for k in range(mark[0] + 1, i)]
        out["marked"].append((i, " ".join([reason, *more]).strip()))
    for j, line in enumerate(raw):
        if MARK_ANY.search(line) and j not in used:
            out["errors"].append((j, "I3 marker at no unguarded regrow site: delete it"))
    return out


def count_unkeyed_captures(lines: list[str]) -> int:
    captures = sum(len(CAPTURE.findall(line)) for line in lines)
    if any(GENERATION.search(line) for line in lines):
        return 0
    return captures


def count_captures(lines: list[str]) -> int:
    return sum(len(CAPTURE.findall(line)) for line in lines)


def count_discarded(lines: list[str]) -> int:
    return sum(1 for line in lines if TAKE.search(line) and not KEPT.search(line))


def scan() -> tuple[dict[str, dict[str, int]], int, dict]:
    """Per-metric hits, the capture total, and the G1 detail: guarded count, claims, marker errors."""
    hits: dict[str, dict[str, int]] = {m: {} for m in METRICS}
    total_captures = 0
    raw: dict[str, list[str]] = {}
    for path in sorted(SRC.rglob("*")):
        if path.suffix not in SUFFIXES or not path.is_file():
            continue
        raw[path.relative_to(REPO).as_posix()] = path.read_text(encoding="utf-8", errors="replace").splitlines()
    # A named bumping function may live in src/memory/; regrow sites there stay exempt.
    files = {rel: code_lines("\n".join(text)) for rel, text in raw.items()}
    g1 = {"guarded": 0, "regrow_files": 0, "claims": [], "errors": []}
    for rel, lines in files.items():
        if rel.startswith(EXEMPT_PREFIX):
            continue
        total_captures += count_captures(lines)
        c = classify(raw[rel], lines, files)
        g1["guarded"] += len(c["guarded"])
        g1["regrow_files"] += bool(c["unguarded"] or c["guarded"] or c["marked"])
        g1["claims"] += [f"{rel}:{i + 1}: {reason}" for i, reason in c["marked"]]
        g1["errors"] += [f"FAIL {rel}:{j + 1}: {msg}" for j, msg in c["errors"]]
        for metric, n in (("G1", len(c["unguarded"])), ("G1n", len(c["marked"])),
                          ("G2", count_unkeyed_captures(lines)), ("G3", count_discarded(lines))):
            if n:
                hits[metric][rel] = n
    return hits, total_captures, g1


def read_allowlist() -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {m: {} for m in METRICS}
    if not ALLOWLIST.exists():
        return out
    for line in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        entry, _, rest = s.partition("#")
        metric, path = entry.split()
        out[metric][path] = int(rest.strip())
    return out


def write_allowlist(hits: dict[str, dict[str, int]]) -> None:
    body = [
        "# I3 allowlist: where a captured graph could hold an address that moves (src/ outside src/memory/).",
        "# Generated by tools/check_graph_stability.py --update. THIS LIST ONLY SHRINKS:",
        "# a new buffer comes from a tier allocator and stays a StableSpan, a new capture site reads",
        "# the generation of what it bakes (docs/internals/MEMORY.md, invariant I3).",
        "# G1 is the unguarded regrow sites (target 0: bump the graph epoch where the address moves).",
        "# G1n is the sites carrying `// I3-not-bakeable: <reason>`: a new one needs this list changed, in review.",
        "#",
    ]
    body += [f"# {m} {NAMES[m]}: {len(hits[m])} files, {sum(hits[m].values())}" for m in METRICS]
    body.append("")
    for m in METRICS:
        body += [f"{m} {rel}  # {hits[m][rel]}" for rel in sorted(hits[m])]
    ALLOWLIST.write_text("\n".join(body) + "\n", encoding="utf-8")


def verdict(hits: dict[str, int], allowed: dict[str, int], metric: str) -> list[str]:
    what = NAMES[metric]
    fails = []
    for rel in sorted(hits):
        if rel not in allowed:
            fails.append(f"FAIL {metric} {rel}: {hits[rel]} {what}, file not on the allowlist")
        elif hits[rel] > allowed[rel]:
            fails.append(f"FAIL {metric} {rel}: {hits[rel]} {what}, budget {allowed[rel]}")
    for rel in sorted(allowed):
        if rel not in hits:
            fails.append(f"FAIL {metric} {rel}: listed but clean now; delete the entry (--update)")
        elif hits[rel] < allowed[rel]:
            fails.append(f"FAIL {metric} {rel}: {hits[rel]}, budget {allowed[rel]}: lower it (--update)")
    return fails


def selftest() -> int:
    results: list[tuple[str, bool]] = []
    for name, hits, allowed, want in [
        ("not listed", {"a.cpp": 1}, {}, True),
        ("over budget", {"a.cpp": 3}, {"a.cpp": 2}, True),
        ("stale entry", {}, {"a.cpp": 2}, True),
        ("under budget", {"a.cpp": 1}, {"a.cpp": 2}, True),
        ("exact", {"a.cpp": 2, "b.cu": 1}, {"a.cpp": 2, "b.cu": 1}, False),
    ]:
        results.append((name, bool(verdict(hits, allowed, "G1")) == want))

    def g1(src: str) -> int:
        return count_regrow(code_lines(src))

    results += [
        ("G1 member regrow", g1("void f() {\n    vram_free(a_, ws_);\n    ws_ = vram_alloc(a_, n, \"t\");\n}\n") == 1),
        ("G1 cast and force", g1("    vram_free(a_, c->buf);\n    c->buf = static_cast<char*>(vram_alloc_force(a_, n, \"t\"));\n") == 1),
        ("G1 assignment on the next line", g1("    vram_free(a_, ws_);\n    ws_ =\n        vram_alloc(a_, n, \"t\");\n") == 1),
        ("G1 out-parameter", g1("    device_free_all(s_buf);\n    e = device_alloc_all(dev_req(s_buf, n));\n") == 1),
        ("G1 local is not a regrow", g1("    vram_free(a_, tmp);\n    tmp = vram_alloc(a_, n, \"t\");\n") == 0),
        ("G1 another buffer", g1("    vram_free(a_, ws_);\n    xq_ = vram_alloc(a_, n, \"t\");\n") == 0),
        ("G1 reset to null", g1("    vram_free(a_, ws_);\n    ws_ = nullptr;\n") == 0),
        ("G1 next function", g1("    vram_free(a_, ws_);\n}\nvoid g() {\n    ws_ = vram_alloc(a_, n, \"t\");\n") == 0),
        ("G1 past the window", g1("    vram_free(a_, ws_);\n" + "    x++;\n" * WINDOW + "    ws_ = vram_alloc(a_, n, \"t\");\n") == 0),
        ("G1 comment", g1("    // vram_free(a_, ws_);\n    ws_ = vram_alloc(a_, n, \"t\");\n") == 0),
        ("G1 owner regrow", g1("    ws_.reset();\n    ws_ = DeviceOwned<void>::allocate(a_, n, \"t\");\n") == 1),
        ("G1 owner adopt", g1("    k_.reset();\n    n_ = 0;\n    k_.adopt_async(raw, n, s);\n") == 1),
        ("G1 owner pair is one site", g1("    k_.reset();\n    v_.reset();\n    k_.adopt_async(a, n, s);\n    v_.adopt_async(b, n, s);\n") == 1),
        ("G1 view-or-owned regrow", g1("    ws_.reset();\n    moved = ws_.set_owned(DeviceOwned<void>::allocate(a_, n, \"t\"));\n") == 1),
        ("G1 static view-or-owned regrow", g1("    (void)s_ws->reset();\n    (void)s_ws->set_view(arena.take_bytes(n));\n") == 1),
        ("G1 owner reset only", g1("    ws_.reset();\n    ws_bytes_ = 0;\n") == 0),
        ("G1 view-or-owned regrow, moved kept", g1("    const bool moved = ws_.reset();\n    if (moved)\n        bump();\n    (void)ws_.set_owned(DeviceOwned<void>::allocate(a_, n, \"t\", true));\n") == 1),
        ("G1 owner local", g1("    tmp.reset();\n    tmp = DeviceOwned<void>::allocate(a_, n, \"t\");\n") == 0),
        ("G1 owner another buffer", g1("    ws_.reset();\n    xq_ = DeviceOwned<void>::allocate(a_, n, \"t\");\n") == 0),
    ]

    def cls(src: str) -> tuple[int, int, int, int]:
        lines = code_lines(src)
        c = classify(src.splitlines(), lines, {"a.cpp": lines})
        return len(c["unguarded"]), len(c["guarded"]), len(c["marked"]), len(c["errors"])

    grow = "    ws_.reset();\n    ws_ = DeviceOwned<void>::allocate(a_, n, \"t\");\n"
    f_grow = "void f() {\n" + grow + "}\n"
    g_bumps = "void E::g() {\n    f();\n    bump_graph_epoch();\n}\n"
    results += [
        ("G1 new unguarded site", cls(f_grow) == (1, 0, 0, 0)),
        ("G1 a new unguarded site fails", bool(verdict({"a.cpp": 1}, {}, "G1"))),
        ("G1 guarded by a bump in the same function",
         cls("void f() {\n    const void* b = ws_.get();\n" + grow + "    bump_graph_epoch_if_moved(b, ws_.get());\n}\n") == (0, 1, 0, 0)),
        ("G1 guarded by a workspace generation bump",
         cls("void f() {\n    if (ws_.reset())\n        w.bump_generation();\n" + grow + "}\n") == (0, 1, 0, 0)),
        ("G1 bump in another function, no marker", cls(f_grow + g_bumps) == (1, 0, 0, 0)),
        ("G1 bump in a comment is no guard", cls("void f() {\n    // bump_graph_epoch();\n" + grow + "}\n") == (1, 0, 0, 0)),
        ("G1 marker names the bumping caller",
         cls("void f() {\n    // I3-bumped-by: E::g\n" + grow + "}\n" + g_bumps) == (0, 1, 0, 0)),
        ("G1 marker names a function without a bump",
         cls("void f() {\n    // I3-bumped-by: E::g\n" + grow + "}\nvoid E::g() {\n    f();\n}\n") == (1, 0, 0, 1)),
        ("G1 marker names an undefined function",
         cls("void f() {\n    // I3-bumped-by: E::h\n" + grow + "}\nvoid E::h();\n" + g_bumps) == (1, 0, 0, 1)),
        ("G1 marker names nothing", cls("void f() {\n    // I3-bumped-by:\n" + grow + "}\n" + g_bumps) == (1, 0, 0, 1)),
        ("G1 not bakeable with a reason",
         cls("void f() {\n    // I3-not-bakeable: both readers run eager\n    // after the forward.\n" + grow + "}\n") == (0, 0, 1, 0)),
        ("G1 not bakeable on the release line",
         cls("void f() {\n    ws_.reset();  // I3-not-bakeable: refuses under a capture\n"
             "    ws_ = DeviceOwned<void>::allocate(a_, n, \"t\");\n}\n") == (0, 0, 1, 0)),
        ("G1 not bakeable without a reason", cls("void f() {\n    // I3-not-bakeable:\n" + grow + "}\n") == (1, 0, 0, 1)),
        ("G1 not bakeable with a one-word reason", cls("void f() {\n    // I3-not-bakeable: eager\n" + grow + "}\n") == (1, 0, 0, 1)),
        ("G1 marker not directly above the site",
         cls("void f() {\n    // I3-not-bakeable: both readers run eager\n    n_ = 0;\n" + grow + "}\n") == (1, 0, 0, 1)),
        ("G1 marker at a guarded site is stale",
         cls("void f() {\n    bump_graph_epoch();\n    // I3-not-bakeable: both readers run eager\n" + grow + "}\n") == (0, 1, 0, 1)),
        ("G1 marker at no regrow site", cls("void f() {\n    // I3-not-bakeable: both readers run eager\n    n_ = 0;\n}\n") == (0, 0, 0, 1)),
        ("G1n a new not-bakeable site fails", bool(verdict({"a.cpp": 2}, {"a.cpp": 1}, "G1n"))),
        ("G1n listed and unchanged", not verdict({"a.cpp": 1}, {"a.cpp": 1}, "G1n")),
    ]

    def g2(src: str) -> int:
        return count_unkeyed_captures(code_lines(src))

    begin = "    cudaStreamBeginCapture(s, mode);\n"
    results += [
        ("G2 unkeyed", g2(begin + "    cudaStreamBeginCaptureToGraph(s, g, 0, 0, 0, mode);\n") == 2),
        ("G2 getter", g2(begin + "    if (gen_ != ws.generation()) drop();\n") == 0),
        ("G2 comparison", g2(begin + "    if (generation != generation_) drop();\n") == 0),
        ("G2 kv getter", g2(begin + "    key = ex->kv_calibration_generation();\n") == 0),
        ("G2 graph epoch", g2(begin + "    if (captured_epoch_ != graph_epoch())\n        drop();\n") == 0),
        ("G2 epoch bump is no read", g2(begin + "    bump_graph_epoch();\n    epoch_ = other_graph_epoch_;\n") == 1),
        ("G2 word in a comment or string", g2(begin + "    // generation() here\n    log(\"x.generation() != y\");\n") == 1),
        ("G2 capture name in a string", g2("    disable_(\"cudaStreamBeginCapture(\", err);\n") == 0),
    ]

    def g3(src: str) -> int:
        return count_discarded(code_lines(src))

    results += [
        ("G3 unwrapped", g3("    p_ = arena.take_bytes(n).data();\n    auto s = arena.take<float>(n);\n") == 2),
        ("G3 kept", g3("    StableSpan<float> s = arena.take<float>(n);\n") == 0),
        ("G3 kept in a view-or-owned", g3("    moved = ws_.set_view(arena.take_bytes(n));\n    p_ = set_view_of(arena.take_bytes(n));\n") == 1),
        ("G3 kept through a static handle", g3("    (void)s_ws->set_view(arena.take_bytes(n));\n") == 0),
        ("G3 other call",g3("    q.take_all();\n    undertake(n);\n") == 0),
    ]
    bad = [name for name, ok in results if not ok]
    for name in bad:
        print(f"selftest FAIL: {name}")
    print(f"selftest: {len(results) - len(bad)}/{len(results)} cases")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    hits, total_captures, g1 = scan()
    if args.update:
        write_allowlist(hits)
        print("allowlist updated: " + ", ".join(f"{m} {sum(hits[m].values())}" for m in METRICS))
        return 0
    allowed = read_allowlist()
    unguarded, marked = sum(hits["G1"].values()), sum(hits["G1n"].values())
    print(f"I3: G1 {unguarded} unguarded regrow sites, {g1['guarded']} guarded, {marked} marked not bakeable "
          f"({unguarded + g1['guarded'] + marked} sites in {g1['regrow_files']} files), "
          f"G2 {sum(hits['G2'].values())} of {total_captures} graph captures read no generation "
          f"({len(hits['G2'])} files), G3 {sum(hits['G3'].values())} take sites in {len(hits['G3'])} files "
          f"not kept as StableSpan (allowlist: "
          + " / ".join(f"{m} {sum(allowed[m].values())}" for m in METRICS) + ")")
    for claim in g1["claims"]:
        print("  G1n " + claim)
    fails = g1["errors"] + [f for m in METRICS for f in verdict(hits[m], allowed[m], m)]
    for f in fails:
        print("  " + f)
    if fails:
        print("FAIL: a graph-baked address can move unseen (invariant I3).")
        return 1
    print("OK: no new unguarded or not-bakeable regrow site, unkeyed capture or discarded StableSpan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
