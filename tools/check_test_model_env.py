#!/usr/bin/env python3
"""X-1 gate: a model environment variable read under tests/ reaches its suite.

A gtest suite that reads IMP_TEST_MODEL_<X> takes its GTEST_SKIP branch when no target sets the
name, and a skip is green (docs/audit/SETTLED.md I-1, AUDIT_ARCH_2026_11_24 X-1). This gate
reads every model variable out of the C++ test sources and holds each one to one of two states:

  wired   a non-comment line of the Makefile, scripts/ or .github/workflows/ sets NAME=...
  listed  tools/test_model_env_allowlist.txt names it WITH a reason

A read variable that is neither fails. A listed variable that is wired, or that no test reads
any more, fails as stale. A listed variable without a reason fails. A kEnv constant in
tests/test_models.h that no test uses fails as dead plumbing.

Usage:
    python3 tools/check_test_model_env.py            # check (CI)
    python3 tools/check_test_model_env.py --list     # one line per variable: state, readers
    python3 tools/check_test_model_env.py --update   # drop stale entries, add new ones WITHOUT a reason
    python3 tools/check_test_model_env.py --selftest
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
TESTS = REPO / "tests"
REGISTRY = "tests/test_models.h"
ALLOWLIST = REPO / "tools" / "test_model_env_allowlist.txt"
SUFFIXES = (".cpp", ".cu", ".h", ".cuh", ".hpp")
# IMP_TEST_MODEL, IMP_TEST_MODEL_<X>, IMP_TEST_MOE_MODEL, IMP_MTP_MODEL: a checkpoint path.
NAME = r"IMP_(?:[A-Z0-9]+_)*MODEL(?:_[A-Z0-9]+)*"
LITERAL = re.compile(r'"(' + NAME + r')"')
CONSTANT = re.compile(r'\b(kEnv[A-Za-z0-9_]+)\s*=\s*"(' + NAME + r')"')
NOT_A_PATH = ("IMP_ERROR_",)
COMMENT_LINE = re.compile(r"^\s*(//|\*|/\*)")
SETTER_COMMENT = re.compile(r"^\s*@?#")


def code_lines(text: str) -> list[str]:
    return [line.split("//", 1)[0] for line in text.splitlines() if not COMMENT_LINE.match(line)]


def scan_sources(sources: dict[str, str]) -> tuple[dict[str, set[str]], list[str]]:
    """{variable: reader files} and the registry constants no test uses."""
    reads: dict[str, set[str]] = {}
    constants: dict[str, str] = {}
    for rel, text in sources.items():
        for line in code_lines(text):
            if rel == REGISTRY:
                for ident, name in CONSTANT.findall(line):
                    constants[ident] = name
                continue
            for name in LITERAL.findall(line):
                if not name.startswith(NOT_A_PATH):
                    reads.setdefault(name, set()).add(rel)
    dead = []
    for ident, name in sorted(constants.items()):
        use = re.compile(r"\b" + ident + r"\b")
        users = {rel for rel, text in sources.items()
                 if rel != REGISTRY and any(use.search(line) for line in code_lines(text))}
        if users:
            reads.setdefault(name, set()).update(users)
        else:
            dead.append(f"{ident} ({name})")
    return reads, dead


def wired_names(names: set[str], setters: dict[str, str]) -> dict[str, str]:
    """{variable: first file that sets it} over the non-comment lines of the setter files."""
    out: dict[str, str] = {}
    for name in sorted(names):
        pat = re.compile(r"(?<![A-Za-z0-9_])" + name + r"=")
        for rel in sorted(setters):
            if any(pat.search(line) for line in setters[rel].splitlines() if not SETTER_COMMENT.match(line)):
                out[name] = rel
                break
    return out


def read_tree() -> tuple[dict[str, str], dict[str, str]]:
    sources = {p.relative_to(REPO).as_posix(): p.read_text(encoding="utf-8", errors="replace")
               for p in sorted(TESTS.rglob("*")) if p.suffix in SUFFIXES and p.is_file()}
    paths = [REPO / "Makefile"]
    paths += sorted((REPO / "scripts").rglob("*.sh"))
    paths += sorted((REPO / ".github" / "workflows").glob("*.y*ml"))
    setters = {p.relative_to(REPO).as_posix(): p.read_text(encoding="utf-8", errors="replace")
               for p in paths if p.is_file()}
    return sources, setters


def parse_allowlist(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        name, _, reason = s.partition("#")
        out[name.strip()] = reason.strip()
    return out


HEADER = [
    "# X-1 allowlist: model environment variables read under tests/ that no target sets.",
    "# One line per variable: NAME  # reason (what is missing, or why the suite runs without it).",
    "# tools/check_test_model_env.py fails on an entry without a reason, on an entry for a",
    "# variable that is wired by now, and on an entry no test reads any more.",
    "",
]


def write_allowlist(entries: dict[str, str]) -> None:
    body = HEADER + [f"{name}  # {entries[name]}".rstrip() for name in sorted(entries)]
    ALLOWLIST.write_text("\n".join(body) + "\n", encoding="utf-8")


def verdict(reads: dict[str, set[str]], wired: dict[str, str], allowed: dict[str, str],
            dead: list[str]) -> list[str]:
    fails = []
    for name in sorted(reads):
        if name in wired or name in allowed:
            continue
        files = ", ".join(sorted(reads[name]))
        fails.append(f"FAIL {name}: read by {files}, set by no target and not on the allowlist. Add "
                     f"`-e {name}=/models/<dir>` to the Makefile target that runs the suite, or list it "
                     f"with a reason in tools/test_model_env_allowlist.txt")
    for name in sorted(allowed):
        if name not in reads:
            fails.append(f"FAIL {name}: listed, but no test reads it any more; delete the entry (--update)")
        elif name in wired:
            fails.append(f"FAIL {name}: listed, but {wired[name]} sets it now; delete the entry (--update)")
        elif not allowed[name]:
            fails.append(f"FAIL {name}: listed without a reason; write `{name}  # <what is missing>`")
    for d in dead:
        fails.append(f"FAIL {d}: constant in {REGISTRY} that no test uses; delete it")
    return fails


def selftest() -> int:
    r = {"IMP_TEST_MODEL_A": {"tests/a.cpp"}}
    cases = [
        ("read, neither wired nor listed", r, {}, {}, [], 1),
        ("stale entry: wired by now", r, {"IMP_TEST_MODEL_A": "Makefile"}, {"IMP_TEST_MODEL_A": "x"}, [], 1),
        ("stale entry: no longer read", {}, {}, {"IMP_TEST_MODEL_A": "no local checkpoint"}, [], 1),
        ("entry without a reason", r, {}, {"IMP_TEST_MODEL_A": ""}, [], 1),
        ("dead registry constant", {}, {}, {}, ["kEnvA (IMP_TEST_MODEL_A)"], 1),
        ("wired", r, {"IMP_TEST_MODEL_A": "Makefile"}, {}, [], 0),
        ("listed with a reason", r, {}, {"IMP_TEST_MODEL_A": "no local checkpoint: A.gguf"}, [], 0),
    ]
    ok = 0
    total = len(cases)
    for name, reads, wired, allowed, dead, want in cases:
        got = len(verdict(reads, wired, allowed, dead))
        if got == want:
            ok += 1
        else:
            print(f"selftest FAIL: {name}: {got} failure(s), wanted {want}")

    def check(label: str, got: object, want: object) -> None:
        nonlocal ok, total
        total += 1
        if got == want:
            ok += 1
        else:
            print(f"selftest FAIL: {label}: got {got!r}, wanted {want!r}")

    sources = {
        REGISTRY: 'inline constexpr const char* kEnvA = "IMP_TEST_MODEL_A";\n'
                  'inline constexpr const char* kEnvDead = "IMP_TEST_MODEL_DEAD";\n'
                  'inline constexpr const char* kEnvOther = "IMP_TEST_GGUF";\n',
        "tests/a.cpp": 'auto p = env(imp_test::kEnvA);\n// getenv("IMP_TEST_MODEL_C")\n'
                       'auto q = std::getenv("IMP_MTP_MODEL");  // "IMP_TEST_MODEL_D"\n'
                       'EXPECT_STREQ(s, "IMP_ERROR_INVALID_MODEL");\n',
    }
    reads, dead = scan_sources(sources)
    check("scan: literals, constants, comments", {k: sorted(v) for k, v in reads.items()},
          {"IMP_TEST_MODEL_A": ["tests/a.cpp"], "IMP_MTP_MODEL": ["tests/a.cpp"]})
    check("scan: unused constant", dead, ["kEnvDead (IMP_TEST_MODEL_DEAD)"])
    setters = {"Makefile": "# IMP_TEST_MODEL_A=<dir> to override\n\t@# -e IMP_TEST_MODEL_A=/x\n"
                           "\t\t-e IMP_TEST_MODEL_AB=/models/ab \\\n\t\t-e XIMP_MTP_MODEL=/m \\\n",
               "scripts/t.sh": 'export IMP_MTP_MODEL="$M"\n'}
    check("wired: comments, prefixes and longer names do not count",
          wired_names({"IMP_TEST_MODEL_A", "IMP_MTP_MODEL", "IMP_TEST_MODEL_AB"}, setters),
          {"IMP_MTP_MODEL": "scripts/t.sh", "IMP_TEST_MODEL_AB": "Makefile"})
    check("allowlist: reason parsing", parse_allowlist("# c\nA  # no checkpoint: x\nB\nC  #\n"),
          {"A": "no checkpoint: x", "B": "", "C": ""})
    print(f"selftest: {ok}/{total} cases")
    return 0 if ok == total else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    sources, setters = read_tree()
    reads, dead = scan_sources(sources)
    wired = wired_names(set(reads), setters)
    allowed = parse_allowlist(ALLOWLIST.read_text(encoding="utf-8")) if ALLOWLIST.exists() else {}
    if args.update:
        kept = {n: allowed.get(n, "") for n in reads if n not in wired}
        write_allowlist(kept)
        missing = sorted(n for n in kept if not kept[n])
        print(f"allowlist updated: {len(kept)} entries, {len(missing)} without a reason"
              + (": " + " ".join(missing) if missing else ""))
        return 0
    if args.list:
        for name in sorted(reads):
            state = f"wired ({wired[name]})" if name in wired else (
                f"listed ({allowed[name]})" if name in allowed else "UNWIRED")
            print(f"{name}\t{state}\t{' '.join(sorted(reads[name]))}")
        return 0
    print(f"X-1: {len(reads)} model variables read under tests/, {len(wired)} wired, "
          f"{len(allowed)} on the allowlist")
    fails = verdict(reads, wired, allowed, dead)
    for f in fails:
        print("  " + f)
    if fails:
        print("FAIL: a suite gated on an unset variable skips, and a skip is green (SETTLED I-1).")
        return 1
    print("OK: every model variable read under tests/ is wired or listed with a reason.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
