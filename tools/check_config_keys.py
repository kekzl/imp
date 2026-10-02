#!/usr/bin/env python3
"""imp.conf.example lists every key src/runtime/config.cpp binds, and nothing else.

WHY THIS EXISTS
---------------
The example file is the only key catalogue an operator has (`imp-cli --help` does
not print the binder). On 2026-09-05 it carried 192 of 223 bound keys; the 31
missing ones included `attention.sparse_min_ctx`, the threshold that silently
turns `attention.sparse_topk_tokens` into a no-op below 12288 context
(AUDIT_arch_2026 J-2). Nothing gated the gap. The reverse direction (a key in
the example that no binder reads) is what `imp.conf: unknown key` reports at
runtime; this gate reports it at commit time.

WHAT IT CHECKS
--------------
Bound keys are the string literals of the `B("...")` / `I("...")` / `F("...")` /
`S("...")` binder calls plus the `dotted_key == "..."` special cases in
src/runtime/config.cpp. Example keys are `name = value` lines under the last
`[section]` header in imp.conf.example, joined as `section.name`. The two sets
must be equal.

Expiry: a key whose comment block says "experimental" carries `expires: YYYY-MM-DD`
in that block (a block covers the run of key lines below it, up to a blank line);
a date before today fails. tests/test_config_bindings.cpp checks each key's binding.

Usage:
    python3 tools/check_config_keys.py             # check (CI)
    python3 tools/check_config_keys.py --list      # every key with its side(s)
    python3 tools/check_config_keys.py --selftest  # planted cases
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_CPP = ROOT / "src" / "runtime" / "config.cpp"
EXAMPLE = ROOT / "imp.conf.example"

BIND_RE = re.compile(r'^\s*[BIFS]\("([A-Za-z0-9_.]+)"', re.M)
DOTTED_RE = re.compile(r'dotted_key\s*==\s*"([a-z0-9_.]+)"')
SECTION_RE = re.compile(r"^\[([a-z0-9_]+)\]")
KEY_RE = re.compile(r"^([A-Za-z0-9_]+)\s*=")
EXPIRES_RE = re.compile(r"expires:\s*(\d{4}-\d{2}-\d{2})")
EXPERIMENTAL_RE = re.compile(r"\bexperimental\b", re.I)


def bound_keys(text: str) -> set[str]:
    return set(BIND_RE.findall(text)) | set(DOTTED_RE.findall(text))


def example_keys(text: str) -> set[str]:
    keys: set[str] = set()
    section = None
    for line in text.splitlines():
        m = SECTION_RE.match(line)
        if m:
            section = m.group(1)
            continue
        m = KEY_RE.match(line)
        if m and section:
            keys.add(f"{section}.{m.group(1)}")
    return keys


def key_comments(text: str) -> list[tuple[str, str]]:
    """(section.key, comment block above its run of key lines); keys with no blank line between
    them share the block written above the first."""
    out: list[tuple[str, str]] = []
    section = None
    block: list[str] = []
    prev_key = False
    for line in text.splitlines():
        m = SECTION_RE.match(line)
        if m:
            section, block, prev_key = m.group(1), [], False
            continue
        if line.startswith("#"):
            if prev_key:
                block = []
            block.append(line)
            prev_key = False
            continue
        m = KEY_RE.match(line)
        if m and section:
            out.append((f"{section}.{m.group(1)}", "\n".join(block)))
            prev_key = True
            continue
        block, prev_key = [], False
    return out


def expiry_failures(text: str, today: dt.date) -> tuple[list[str], int]:
    """An experimental key's block carries `expires: YYYY-MM-DD`, not before today."""
    fails: list[str] = []
    dated = 0
    for key, block in key_comments(text):
        m = EXPIRES_RE.search(block)
        if m:
            dated += 1
            if dt.date.fromisoformat(m.group(1)) < today:
                fails.append(f"{key} expired {m.group(1)}: remove the key or extend with a reason")
        elif EXPERIMENTAL_RE.search(block):
            fails.append(f"{key} is marked experimental without `expires: YYYY-MM-DD`")
    return fails, dated


def diff(bound: set[str], example: set[str]) -> tuple[list[str], list[str]]:
    return sorted(bound - example), sorted(example - bound)


def check(list_all: bool) -> int:
    bound = bound_keys(CONFIG_CPP.read_text(encoding="utf-8"))
    example_text = EXAMPLE.read_text(encoding="utf-8")
    example = example_keys(example_text)
    missing, unbound = diff(bound, example)
    expired, dated = expiry_failures(example_text, dt.date.today())
    for f in expired:
        print(f"FAIL: {f}")
    if list_all:
        for k in sorted(bound | example):
            side = "both" if k in bound and k in example else ("binder only" if k in bound else "example only")
            print(f"{k:45s} {side}")
    for k in missing:
        print(f"FAIL: {k} is bound in src/runtime/config.cpp but absent from imp.conf.example")
    for k in unbound:
        print(f"FAIL: {k} is in imp.conf.example but no binder in src/runtime/config.cpp reads it")
    if missing or unbound or expired:
        print(f"check_config_keys: {len(missing)} missing, {len(unbound)} unbound, {len(expired)} expiry "
              f"({len(bound)} bound, {len(example)} in the example)")
        return 1
    print(f"PASS: imp.conf.example lists all {len(bound)} bound keys and nothing else; "
          f"{dated} experimental key(s) dated, none expired")
    return 0


def selftest() -> int:
    cpp = '''
    B("runtime.alpha", cfg.runtime.alpha);
    I("runtime.beta", cfg.runtime.beta);
    F("gemm.gamma", cfg.gemm.gamma);
    S("paths.delta", cfg.paths.delta);
    if (!matched && dotted_key == "runtime.special") {
'''
    example = '''
[runtime]
# comment
alpha = true
beta  = 3
special = false
[gemm]
gamma = 1.5
[paths]
extra = "x"
'''
    bound = bound_keys(cpp)
    ex = example_keys(example)
    missing, unbound = diff(bound, ex)
    cases = [
        ("all four binder forms parse", bound == {"runtime.alpha", "runtime.beta", "gemm.gamma",
                                                  "paths.delta", "runtime.special"}),
        ("section prefix joins the example key", "gemm.gamma" in ex),
        ("a bound key missing from the example is reported", missing == ["paths.delta"]),
        ("an example key with no binder is reported", unbound == ["paths.extra"]),
        ("a comment line is not a key", "runtime.#" not in ex and len(ex) == 5),
        ("a binder key with upper case parses", bound_keys('    B("moe.nvfp4_smallM", x);') == {"moe.nvfp4_smallM"}),
    ]
    dated = '''
[runtime]
# Experimental fast path.
fast = false
# Experimental, expires: 2026-01-31.
old = false
old_twin = 1

# experimental, expires: 2027-06-30
fresh = false
# Stable knob.
plain = 1
'''
    today = dt.date(2026, 10, 2)
    fails, n_dated = expiry_failures(dated, today)
    cases += [
        ("experimental without a date is reported", any(f.startswith("runtime.fast ") for f in fails)),
        ("a past date is reported", any(f.startswith("runtime.old expired") for f in fails)),
        ("a date covers its whole run of keys", any(f.startswith("runtime.old_twin expired") for f in fails)),
        ("a future date passes", not any("runtime.fresh" in f for f in fails)),
        ("an undated stable key passes", not any("runtime.plain" in f for f in fails) and len(fails) == 3),
        ("dated keys are counted", n_dated == 3),
    ]
    bad = [name for name, ok in cases if not ok]
    for name, ok in cases:
        print(f"  {'ok ' if ok else 'BAD'} {name}")
    if bad:
        print(f"selftest: {len(bad)} of {len(cases)} planted cases failed")
        return 1
    print(f"selftest: {len(cases)}/{len(cases)} planted cases pass")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return check(a.list)


if __name__ == "__main__":
    sys.exit(main())
