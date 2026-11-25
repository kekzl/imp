#!/usr/bin/env python3
"""I3 gate: a captured graph never holds an address that can move.

docs/internals/MEMORY.md invariant I3: a CUDA graph bakes device pointers, so a buffer it can
bake is either held as a StableSpan from mint to launch, or its owner bumps a generation that
every graph runner checks before replay. This gate counts, in src/ outside src/memory/, the
three places where that is not yet true, against tools/graph_stability_allowlist.txt:

  G1 regrow sites      a release (vram_free, device_free_all*, cudaFree*) of a member or static
                       X followed within 25 lines of the same function by an allocation into X.
                       The owner spelling counts too: X.reset(); then X = DeviceOwned<T>::allocate*
                       or X.adopt*, or X.set_owned / X.set_view (ViewOrOwned).
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
METRICS = ("G1", "G2", "G3")
NAMES = {
    "G1": "buffer(s) freed and reallocated in place",
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
OWNER_RESET = re.compile(r"^\s*([A-Za-z_]\w*(?:(?:->|\.)\w+|\[\w+\])*)\.reset\(\)\s*;")
OWNER_ALLOC = r"(?:\s*=[^=;]*\bDeviceOwned\s*<|\.adopt(?:_async)?\s*\(|\.set_(?:owned|view)\s*\()"
FUNCTION_END = re.compile(r"^}")
CAPTURE = re.compile(r"\bcudaStreamBeginCapture\w*\s*\(")
GENERATION = re.compile(r"\w*generation\w*\s*\(\s*\)|\w*generation\w*\s*!=|!=\s*\w*generation\w*"
                        r"|\bgraph_epoch\s*\(\s*\)")
TAKE = re.compile(r"\.take(?:_bytes)?(?:<[^>]+>)?\(")
KEPT = re.compile(r"\bStableSpan\s*<|\.set_view\s*\(")


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


def count_regrow(lines: list[str]) -> int:
    n = 0
    for i, line in enumerate(lines):
        m = RELEASE.search(line)
        if not m:
            # Adjacent owner resets are one site, as one multi-argument free call is.
            if owner_regrow(lines, i) and not (i > 0 and owner_regrow(lines, i - 1)):
                n += 1
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
        n += hit
    return n


def count_unkeyed_captures(lines: list[str]) -> int:
    captures = sum(len(CAPTURE.findall(line)) for line in lines)
    if any(GENERATION.search(line) for line in lines):
        return 0
    return captures


def count_captures(lines: list[str]) -> int:
    return sum(len(CAPTURE.findall(line)) for line in lines)


def count_discarded(lines: list[str]) -> int:
    return sum(1 for line in lines if TAKE.search(line) and not KEPT.search(line))


def scan() -> tuple[dict[str, dict[str, int]], int]:
    hits: dict[str, dict[str, int]] = {m: {} for m in METRICS}
    total_captures = 0
    for path in sorted(SRC.rglob("*")):
        if path.suffix not in SUFFIXES or not path.is_file():
            continue
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith(EXEMPT_PREFIX):
            continue
        lines = code_lines(path.read_text(encoding="utf-8", errors="replace"))
        total_captures += count_captures(lines)
        for metric, n in (("G1", count_regrow(lines)), ("G2", count_unkeyed_captures(lines)),
                          ("G3", count_discarded(lines))):
            if n:
                hits[metric][rel] = n
    return hits, total_captures


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
        ("G1 owner reset only",g1("    ws_.reset();\n    ws_bytes_ = 0;\n") == 0),
        ("G1 owner local", g1("    tmp.reset();\n    tmp = DeviceOwned<void>::allocate(a_, n, \"t\");\n") == 0),
        ("G1 owner another buffer", g1("    ws_.reset();\n    xq_ = DeviceOwned<void>::allocate(a_, n, \"t\");\n") == 0),
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
    hits, total_captures = scan()
    if args.update:
        write_allowlist(hits)
        print("allowlist updated: " + ", ".join(f"{m} {sum(hits[m].values())}" for m in METRICS))
        return 0
    allowed = read_allowlist()
    print(f"I3: G1 {sum(hits['G1'].values())} regrow sites in {len(hits['G1'])} files, "
          f"G2 {sum(hits['G2'].values())} of {total_captures} graph captures read no generation "
          f"({len(hits['G2'])} files), G3 {sum(hits['G3'].values())} take sites in {len(hits['G3'])} files "
          f"not kept as StableSpan (allowlist: "
          + " / ".join(f"{m} {sum(allowed[m].values())}" for m in METRICS) + ")")
    fails = [f for m in METRICS for f in verdict(hits[m], allowed[m], m)]
    for f in fails:
        print("  " + f)
    if fails:
        print("FAIL: a graph-baked address can move unseen (invariant I3).")
        return 1
    print("OK: no new regrow site, unkeyed capture or discarded StableSpan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
