#!/usr/bin/env python3
"""PDL gate: a kernel that waits is registered, a register function is called.

WHY THIS EXISTS
---------------
A kernel opts into Programmatic Dependent Launch in two halves (src/core/pdl_device.cuh,
src/core/pdl_launch.cuh): its body calls pdl_wait(), and host code registers its address
with pdl::enable_kernel() / pdl::enable(). pdl::launch() and the graph edge conversion
only use the PDL attribute for a registered kernel, so a kernel with the first half and
not the second launches serialized: a silent performance loss, never a failure. The class
recurred three times (docs/audit/SETTLED.md A1-3, A2-2, A2-5); AUDIT_ARCH_2026_11_24 X-6.

WHAT IT CHECKS
--------------
1. Kernels. A kernel is a `__global__` function definition under src/. It WAITS when its
   body calls pdl_wait(), or calls a `__device__` function defined in the same file whose
   body does (transitively). Every waiting kernel needs a registration reference: its name
   as an identifier in src/
     a. inside the argument list of enable_kernel(...), pdl::enable(...), or of a macro
        whose #define body contains one of the two, or
     b. on the right-hand side of `X = ...;` in a file where X is, by rule a or b, a
        registered name (`auto* k = some_kernel<..>; pdl::enable_kernel(k);`), or
     c. in the argument list of a call `F(...)` in a file where F is a parameter declared
        `Name&& F` and a generic lambda in that file registers one of its `auto`
        parameters (the by_m/by_cfg dispatch of nvfp4_gemm_smallm_ws.cu).
   A kernel name defined in several files is matched per file first (the reference must
   sit in a file that defines it or includes one that does); a unique name matches
   anywhere in src/.
2. Register functions. Every function defined under src/ whose name ends in
   `_pdl_register` is called from the central list (CENTRAL below) or from the body of a
   register function that is (transitively).
3. The allowlist (tools/pdl_registration_allowlist.txt) names waiting kernels without a
   registration, each with a reason. An entry whose kernel is registered, no longer waits
   or no longer exists is stale and fails.

4. The reverse (the contract of src/core/pdl_launch.cuh): a `__global__` name that rules
   a-c register and whose body never reaches pdl_wait() fails, and so does a pdl_wait()
   call outside every function body the gate parses.

Not checked: template arguments (one registered instantiation covers the name) and
whether the registration runs before the first launch.

Usage:
    python3 tools/check_pdl_registration.py             # check (CI)
    python3 tools/check_pdl_registration.py --list      # every waiting kernel + status
    python3 tools/check_pdl_registration.py --selftest  # planted cases
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src"
ALLOWLIST = REPO / "tools" / "pdl_registration_allowlist.txt"
CENTRAL = "src/exec/executor_workspace.cpp"
WAIT_HOME = "src/core/pdl_device.cuh"  # defines pdl_wait()
SUFFIXES = (".cu", ".cuh", ".cpp", ".h", ".hpp")

IDENT = re.compile(r"[A-Za-z_]\w*")
WAIT = re.compile(r"\bpdl_wait\s*\(")
ENABLE = re.compile(r"\b(?:enable_kernel|pdl\s*::\s*enable)\s*\(")
DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(\w+)\(([^)]*)\)((?:.*\\\n)*.*)$", re.M)
FUNC_ATTR = re.compile(r"\b(__global__|__device__)\b")
SKIP_NAMES = {"__launch_bounds__", "__attribute__", "alignas"}
NAME_PAREN = re.compile(r"[\s\w:*&<>,]*?\b([A-Za-z_]\w*)\s*\(")
REGISTER_DEF = re.compile(r"\bvoid\s+(\w+_pdl_register)\s*\(\s*(?:void)?\s*\)\s*\{")
REGISTER_CALL = re.compile(r"\b(\w+_pdl_register)\s*\(\s*\)\s*;")
INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]+"([^"]+)"', re.M)
FORWARD_PARAM = re.compile(r"\b[A-Z]\w*\s*&&\s*(\w+)\s*[,)]")
AUTO_PARAM = re.compile(r"\bauto\s*[*&]*\s*(\w+)\s*[,)]")
KEYWORDS = {"reinterpret_cast", "static_cast", "const", "void", "auto", "true", "false", "half", "float",
            "int", "__half", "pdl", "imp", "uint8_t", "nullptr"}


def strip_comments(text: str) -> str:
    """Blank out // and /* */ comments and string literals, keeping every newline."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            # A `\` at the end of a // comment inside a #define keeps the macro going.
            out.append("\\" if text[i:j].rstrip().endswith("\\") else "")
            i = j
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j]))
            i = j
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"' and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            out.append('"' + " " * max(0, j - i - 1) + '"')
            i = min(n, j + 1)
        else:
            out.append(c)
            i += 1
    return "".join(out)


def balanced(text: str, start: int, open_ch: str, close_ch: str) -> int:
    """Index just past the bracket that closes text[start] (== open_ch), or -1."""
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def functions(text: str) -> list[tuple[str, str, str, int, int, int]]:
    """-> [(kind, name, body, line, body start, body end)] for each __global__ / __device__ function DEFINITION."""
    found = []
    for m in FUNC_ATTR.finditer(text):
        pos = m.end()
        name = None
        paren_end = -1
        while True:
            nm = NAME_PAREN.match(text, pos)
            if not nm:
                break
            paren_end = balanced(text, nm.end() - 1, "(", ")")
            if nm.group(1) in SKIP_NAMES and paren_end > 0:
                pos = paren_end  # `__global__ void __launch_bounds__(256, 2) name(`
                continue
            name = nm.group(1)
            break
        if name is None or paren_end < 0:
            continue
        rest = re.match(r"[\s\w]*", text[paren_end:])
        brace = paren_end + (rest.end() if rest else 0)
        if brace >= len(text) or text[brace] != "{":
            continue  # a declaration
        end = balanced(text, brace, "{", "}")
        if end < 0:
            continue
        found.append((m.group(1), name, text[brace:end], text.count("\n", 0, m.start()) + 1, brace, end))
    return found


def waiting_helpers(text: str) -> set[str]:
    """Names of the __device__ functions of this file whose own body calls pdl_wait()."""
    return {name for kind, name, body, *_ in functions(text) if kind == "__device__" and WAIT.search(body)}


def waiting_kernels(text: str, helpers: frozenset[str] = frozenset()) -> dict[str, int]:
    """Kernel name -> line, for each __global__ whose body reaches pdl_wait().

    helpers: waiting __device__ functions of other files (a shared .cuh body).
    """
    funcs = functions(text)
    waits = {name for _, name, body, *_ in funcs if WAIT.search(body)} | set(helpers)
    changed = True
    while changed:
        changed = False
        for _, name, body, *_ in funcs:
            if name in waits:
                continue
            if any(re.search(r"\b" + re.escape(w) + r"\s*[<(]", body) for w in waits):
                waits.add(name)
                changed = True
    return {name: line for kind, name, _, line, *_ in funcs if kind == "__global__" and name in waits}


def unattributed_waits(text: str) -> list[int]:
    """Lines with a pdl_wait() call outside every parsed __global__ / __device__ body.

    The gate cannot name the kernel such a call belongs to, so it fails instead of passing.
    """
    spans = [(start, end) for *_, start, end in functions(text)]
    return [text.count("\n", 0, w.start()) + 1 for w in WAIT.finditer(text)
            if not any(a <= w.start() < b for a, b in spans)]


def call_args(text: str, pattern: re.Pattern) -> list[str]:
    out = []
    for m in pattern.finditer(text):
        end = balanced(text, m.end() - 1, "(", ")")
        if end > 0:
            out.append(text[m.end():end - 1])
    return out


def registered_names(text: str) -> set[str]:
    """Identifiers this file registers, by rules a, b and c of the docstring."""
    macros = [m.group(1) for m in DEFINE.finditer(text) if ENABLE.search(m.group(3))]
    patterns = [ENABLE] + [re.compile(r"\b" + re.escape(name) + r"\s*\(") for name in macros]
    names: set[str] = set()
    for pat in patterns:
        for args in call_args(text, pat):
            names.update(IDENT.findall(args))
    names -= KEYWORDS
    # Rule b: follow `X = expr;` for every registered X, to a fixed point.
    seen: set[str] = set()
    while names - seen:
        x = sorted(names - seen)[0]
        seen.add(x)
        for m in re.finditer(r"\b" + re.escape(x) + r"\s*=(?!=)([^;]*);", text):
            names.update(set(IDENT.findall(m.group(1))) - KEYWORDS)
    # Rule c: a generic lambda registers its `auto` parameter; kernels reach it as the
    # arguments of a call through a forwarded callable parameter.
    if names & set(AUTO_PARAM.findall(text)):
        for fwd in sorted(set(FORWARD_PARAM.findall(text))):
            for args in call_args(text, re.compile(r"(?<![\w.>:])" + re.escape(fwd) + r"\s*\(")):
                names.update(set(IDENT.findall(args)) - KEYWORDS)
    return names


def register_graph(files: dict[str, str]) -> tuple[dict[str, str], dict[str, set[str]], set[str]]:
    """-> (defined: name -> file, calls: name -> callees, roots called from CENTRAL)."""
    defined: dict[str, str] = {}
    calls: dict[str, set[str]] = {}
    for rel, text in files.items():
        for m in REGISTER_DEF.finditer(text):
            end = balanced(text, m.end() - 1, "{", "}")
            body = text[m.end():end] if end > 0 else ""
            defined[m.group(1)] = rel
            calls[m.group(1)] = set(REGISTER_CALL.findall(body))
    roots = set(REGISTER_CALL.findall(files.get(CENTRAL, "")))
    return defined, calls, roots


def analyse(files: dict[str, str]) -> tuple[dict[tuple[str, str], tuple[int, bool]], list[tuple[str, str]],
                                               list[str]]:
    """files: rel -> comment-stripped text.

    -> (kernels: (file, name) -> (line, registered), uncalled: [(register function, file)],
        other: failure lines for a pdl_wait() the gate cannot attribute and for a registered
        kernel that never waits)
    """
    helpers = frozenset(h for rel, text in files.items() if rel != WAIT_HOME for h in waiting_helpers(text))
    kernels = {rel: waiting_kernels(text, helpers) for rel, text in files.items()}
    registered = {rel: registered_names(text) for rel, text in files.items()}
    includes = {rel: {inc for inc in INCLUDE.findall(text)} for rel, text in files.items()}
    definers: dict[str, list[str]] = {}
    for rel, text in files.items():
        for kind, name, *_ in functions(text):
            if kind == "__global__" and rel not in definers.setdefault(name, []):
                definers[name].append(rel)
    out: dict[tuple[str, str], tuple[int, bool]] = {}
    for rel, ks in kernels.items():
        for name, line in ks.items():
            if len(definers.get(name, [])) <= 1:
                ok = any(name in r for r in registered.values())
            else:
                tail = rel.split("/", 1)[1] if "/" in rel else rel
                ok = name in registered[rel] or any(
                    name in registered[other] and (tail in includes[other] or rel in includes[other])
                    for other in files)
            out[(rel, name)] = (line, ok)
    defined, calls, roots = register_graph(files)
    reach = set()
    todo = sorted(roots)
    while todo:
        fn = todo.pop()
        if fn in reach:
            continue
        reach.add(fn)
        todo.extend(sorted(calls.get(fn, ())))
    uncalled = sorted((fn, rel) for fn, rel in defined.items() if fn not in reach)
    other = [f"FAIL {rel}:{line}: pdl_wait() outside every __global__ / __device__ body the gate parses; "
             f"it cannot name the kernel"
             for rel, text in sorted(files.items()) if rel != WAIT_HOME for line in unattributed_waits(text)]
    waiting = {name for ks in kernels.values() for name in ks}
    other += [f"FAIL {rel} {name}: registered for PDL, its body never reaches pdl_wait() (a registered "
              f"kernel may start before its predecessor finished: {WAIT_HOME})"
              for rel in sorted(files) for name in sorted(registered[rel])
              if name in definers and name not in waiting]
    return out, uncalled, other


def read_allowlist(path: pathlib.Path = ALLOWLIST) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, _, reason = line.partition("  ")
        rel, _, name = key.strip().partition("::")
        out[(rel, name)] = reason.strip()
    return out


def verdict(kernels: dict[tuple[str, str], tuple[int, bool]], uncalled: list[tuple[str, str]],
            other: list[str],
            allowed: dict[tuple[str, str], str]) -> list[str]:
    fails = []
    for (rel, name), (line, ok) in sorted(kernels.items()):
        if not ok and (rel, name) not in allowed:
            fails.append(f"FAIL {rel}:{line} {name}: calls pdl_wait(), no enable_kernel / pdl::enable "
                         f"reference in src/ (it launches without the PDL attribute)")
    for fn, rel in uncalled:
        fails.append(f"FAIL {rel} {fn}(): defined, not called from {CENTRAL} or a register function "
                     f"that is")
    fails.extend(other)
    for (rel, name), reason in sorted(allowed.items()):
        hit = kernels.get((rel, name))
        if hit is None:
            fails.append(f"FAIL allowlist {rel}::{name}: stale, no such waiting kernel; delete the entry")
        elif hit[1]:
            fails.append(f"FAIL allowlist {rel}::{name}: stale, the kernel is registered; delete the entry")
        elif not reason:
            fails.append(f"FAIL allowlist {rel}::{name}: no reason")
    return fails


def load() -> dict[str, str]:
    files = {}
    for path in sorted(SRC.rglob("*")):
        if path.suffix in SUFFIXES and path.is_file():
            files[path.relative_to(REPO).as_posix()] = strip_comments(
                path.read_text(encoding="utf-8", errors="replace"))
    return files


def selftest() -> int:
    kern = "__global__ void __launch_bounds__(256, 2) {name}(float* y) {{ pdl_wait(); y[0] = 1.f; }}\n"
    central = "void init() { a_pdl_register(); }\n"
    reg_a = "void a_pdl_register() { pdl::enable_kernel(a_kernel); }\n"

    def run(files: dict[str, str], allowed: dict[tuple[str, str], str] | None = None) -> list[str]:
        stripped = {rel: strip_comments(text) for rel, text in files.items()}
        kernels, uncalled, other = analyse(stripped)
        return verdict(kernels, uncalled, other, allowed or {})

    base = {CENTRAL: central, "src/a.cu": kern.format(name="a_kernel") + reg_a}
    cases = [
        ("waiting kernel, registered, register function called", base, None, 0),
        ("waiting kernel with no enable_kernel",
         {**base, "src/b.cu": kern.format(name="b_kernel")}, None, 1),
        ("enable_kernel only in a comment does not count",
         {**base, "src/b.cu": kern.format(name="b_kernel") + "// pdl::enable_kernel(b_kernel);\n"}, None, 1),
        ("kernel without pdl_wait needs no registration",
         {**base, "src/b.cu": "__global__ void b_kernel(float* y) { y[0] = 1.f; }\n"}, None, 0),
        ("pdl_wait through a __device__ helper counts",
         {**base, "src/b.cu": "template <int N> __device__ void body(float* y) { pdl_wait(); }\n"
                              "__global__ void b_kernel(float* y) { body<4>(y); }\n"}, None, 1),
        ("registered through a local alias (auto* k = kernel<..>)",
         {**base, "src/b.cu": "template <int N>\n" + kern.format(name="b_kernel")
                              + "void go() { auto* k = b_kernel<4>; pdl::enable_kernel(k); }\n"}, None, 0),
        ("registered through pdl::enable(reinterpret_cast<const void*>(&kernel))",
         {**base, "src/b.cu": kern.format(name="b_kernel")
                              + "void go() { pdl::enable(reinterpret_cast<const void*>(&b_kernel)); }\n"},
         None, 0),
        ("registered through a macro that expands to enable_kernel",
         {**base, "src/b.cu": kern.format(name="b_kernel")
                              + "#define REG(k) pdl::enable_kernel(k)\nvoid go() { REG(b_kernel); }\n"},
         None, 0),
        ("registered through a forwarded callable and a generic lambda",
         {**base, "src/b.cu": kern.format(name="b_kernel")
                              + "template <typename L> void by(L&& go) { go(b_kernel); }\n"
                              + "void f() { by([&](auto k) { pdl::enable_kernel(k); }); }\n"}, None, 0),
        ("a register function nobody calls",
         {**base, "src/b.cu": kern.format(name="b_kernel")
                              + "void b_pdl_register() { pdl::enable_kernel(b_kernel); }\n"}, None, 1),
        ("a register function called from a called register function",
         {CENTRAL: central,
          "src/a.cu": kern.format(name="a_kernel")
                      + "void a_pdl_register() { pdl::enable_kernel(a_kernel); b_pdl_register(); }\n",
          "src/b.cu": kern.format(name="b_kernel")
                      + "void b_pdl_register() { pdl::enable_kernel(b_kernel); }\n"}, None, 0),
        ("same kernel name in two files: the registration of one does not cover the other",
         {**base, "src/c.cu": kern.format(name="a_kernel")}, None, 1),
        ("unregistered kernel on the allowlist passes",
         {**base, "src/b.cu": kern.format(name="b_kernel")}, {("src/b.cu", "b_kernel"): "reason"}, 0),
        ("stale allowlist entry: the kernel is registered",
         base, {("src/a.cu", "a_kernel"): "reason"}, 1),
        ("stale allowlist entry: no such kernel",
         base, {("src/gone.cu", "gone_kernel"): "reason"}, 1),
        ("registered kernel that never waits",
         {**base, "src/b.cu": "__global__ void b_kernel(float* y) { y[0] = 1.f; }\n"
                              "void go() { pdl::enable_kernel(b_kernel); }\n"}, None, 1),
        ("pdl_wait through a __device__ helper of another file, registered",
         {**base, "src/b.cuh": "__device__ void shared_body(float* y) { pdl_wait(); }\n",
          "src/b.cu": "__global__ void b_kernel(float* y) { shared_body(y); }\n"
                      "void go() { pdl::enable_kernel(b_kernel); }\n"}, None, 0),
        ("pdl_wait in a function the gate cannot parse as a kernel or helper",
         {**base, "src/b.cu": "#define BODY(y) pdl_wait(); y[0] = 1.f;\n"}, None, 1),
        ("allowlist entry without a reason",
         {**base, "src/b.cu": kern.format(name="b_kernel")}, {("src/b.cu", "b_kernel"): ""}, 1),
    ]
    failures = 0
    for label, files, allowed, want in cases:
        got = len(run(files, allowed))
        ok = got == want
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got} failure(s)" + ("" if ok else f" (want {want})"))
    print(f"selftest: {len(cases) - failures}/{len(cases)} cases")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="print every waiting kernel with its status")
    ap.add_argument("--selftest", action="store_true", help="run the planted cases")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    files = load()
    kernels, uncalled, other = analyse(files)
    allowed = read_allowlist()
    if args.list:
        for (rel, name), (line, ok) in sorted(kernels.items()):
            tag = "registered" if ok else ("allowlisted" if (rel, name) in allowed else "UNREGISTERED")
            print(f"  {tag:<12} {rel}:{line} {name}")
    defined, _, _ = register_graph(files)
    n_reg = sum(1 for _, ok in kernels.values() if ok)
    n_files = len({rel for rel, _ in kernels})
    print(f"PDL: {len(kernels)} kernels in {n_files} files call pdl_wait(), {n_reg} registered, "
          f"{len(kernels) - n_reg} not (allowlist: {len(allowed)}); {len(defined)} register functions, "
          f"{len(uncalled)} uncalled")
    fails = verdict(kernels, uncalled, other, allowed)
    for f in fails:
        print("  " + f)
    if fails:
        print("FAIL: register the kernel (pdl::enable_kernel), call the register function, or list the "
              f"kernel in {ALLOWLIST.relative_to(REPO)} with a reason.")
        return 1
    print("OK: every waiting kernel is registered or listed, every register function is called.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
