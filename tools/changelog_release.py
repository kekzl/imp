#!/usr/bin/env python3
"""Write a release's CHANGELOG.md section from the squash subjects since the previous tag.

PRs do not edit CHANGELOG.md: 263 of 374 commits in the 30 days before 2026-11-15 touched it,
and every merge dirtied every open PR. The squash subject already is the entry
(`perf(sparse): ... 45.54 -> 32.93 us per call (#3107)`), so the release cut collects them:

    python3 tools/changelog_release.py --version 0.68.0 --since v0.67.0            # write
    python3 tools/changelog_release.py --version 0.68.0 --since v0.67.0 --dry-run  # print
    python3 tools/changelog_release.py --selftest

`feat` -> Added, `fix` -> Fixed, `perf` / `refactor` / `build` / `revert` -> Changed, `docs` / `test`
/ `ci` / `chore` / `style` -> Internal, `release` skipped. Entries written by hand under
`## [Unreleased]` (breaking changes, a user-facing note a subject cannot carry) move into the new
section first; a PR number they cite gets no generated line.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"
ORDER = ["Added", "Changed", "Fixed", "Removed", "Internal"]
TYPE_TO_HEADING = {
    "feat": "Added",
    "fix": "Fixed",
    "perf": "Changed",
    "refactor": "Changed",
    "build": "Changed",
    "revert": "Changed",
    "docs": "Internal",
    "test": "Internal",
    "ci": "Internal",
    "chore": "Internal",
    "style": "Internal",
}
SUBJECT = re.compile(r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]*)\))?(?P<bang>!)?:\s*(?P<text>.+)$")
PR_REF = re.compile(r"#(\d+)")


def entry_for(subject: str) -> tuple[str, str] | None:
    """(heading, '- line') for one squash subject; None for a release commit."""
    m = SUBJECT.match(subject.strip())
    if not m:
        return "Changed", f"- {subject.strip()}"
    kind = m.group("type")
    if kind == "release":
        return None
    heading = TYPE_TO_HEADING.get(kind, "Changed")
    scope = m.group("scope")
    text = m.group("text").strip()
    bang = "**Breaking:** " if m.group("bang") else ""
    return heading, f"- {bang}{scope + ': ' if scope else ''}{text}"


def split_unreleased(text: str) -> tuple[str, dict[str, list[str]], str]:
    """(text before [Unreleased] incl. its heading, hand entries by heading, text from the next '## ')."""
    lines = text.splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.startswith("## [Unreleased]"))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    hand: dict[str, list[str]] = {}
    heading = "Changed"
    for ln in lines[start + 1 : end]:
        if ln.startswith("### "):
            heading = ln[4:].strip()
        elif ln.startswith("- "):
            hand.setdefault(heading, []).append(ln.rstrip("\n"))
        elif ln.strip() and ln[0].isspace() and hand.get(heading):
            hand[heading][-1] += "\n" + ln.rstrip("\n")
    return "".join(lines[: start + 1]), hand, "".join(lines[end:])


def cite_introducing_pr(hand: dict[str, list[str]]) -> None:
    """A hand entry written before its PR had a number gets ' (#N)' of the squash that added it."""
    for entries in hand.values():
        for i, e in enumerate(entries):
            if PR_REF.search(e):
                continue
            needle = e.splitlines()[0][2:62]
            out = subprocess.run(["git", "log", "--first-parent", "--reverse", "--format=%s", "-S", needle,
                                  "--", "CHANGELOG.md"], cwd=ROOT, capture_output=True, text=True)
            refs = PR_REF.findall(out.stdout.splitlines()[0]) if out.stdout.strip() else []
            if refs:
                entries[i] = e + f" (#{refs[-1]})"


def build_section(version: str, date: str, subjects: list[str], hand: dict[str, list[str]]) -> str:
    cited = {n for entries in hand.values() for e in entries for n in PR_REF.findall(e)}
    out: dict[str, list[str]] = {h: list(v) for h, v in hand.items()}
    for s in subjects:
        refs = PR_REF.findall(s)
        if refs and refs[-1] in cited:
            continue
        e = entry_for(s)
        if e:
            out.setdefault(e[0], []).append(e[1])
    parts = [f"## [{version}] - {date}\n"]
    for h in ORDER + sorted(k for k in out if k not in ORDER):
        if out.get(h):
            parts.append(f"\n### {h}\n" + "\n".join(out[h]) + "\n")
    return "".join(parts)


def render(text: str, version: str, date: str, subjects: list[str], use_git: bool = False) -> str:
    head, hand, tail = split_unreleased(text)
    if use_git:
        cite_introducing_pr(hand)
    return head + "\n" + build_section(version, date, subjects, hand) + "\n" + tail


def git_subjects(since: str) -> list[str]:
    out = subprocess.run(["git", "log", "--first-parent", "--reverse", "--format=%s", f"{since}..HEAD"],
                         cwd=ROOT, capture_output=True, text=True, check=True)
    return [s for s in out.stdout.splitlines() if s.strip()]


def selftest() -> int:
    base = ("# Changelog\n\n## [Unreleased]\n\n### Removed\n- C API: `X.y` is gone (ABI break). (#7)\n\n"
            "## [0.1.0] - 2026-01-01\n\n### Added\n- old\n")
    subjects = [
        "feat(api): /v1/foo answers bar (#5)",
        "fix(server): model id is org/name (#6)",
        "refactor(config): 47 keys leave the config (#7)",
        "perf(sparse)!: score 45 -> 33 us (#8)",
        "docs(audit): lever refuted (#9)",
        "release: v0.1.0 (#4)",
        "an unprefixed subject (#10)",
    ]
    got = render(base, "0.2.0", "2026-02-02", subjects)
    want = ("# Changelog\n\n## [Unreleased]\n\n## [0.2.0] - 2026-02-02\n"
            "\n### Added\n- api: /v1/foo answers bar (#5)\n"
            "\n### Changed\n- **Breaking:** sparse: score 45 -> 33 us (#8)\n- an unprefixed subject (#10)\n"
            "\n### Fixed\n- server: model id is org/name (#6)\n"
            "\n### Removed\n- C API: `X.y` is gone (ABI break). (#7)\n"
            "\n### Internal\n- audit: lever refuted (#9)\n"
            "\n## [0.1.0] - 2026-01-01\n\n### Added\n- old\n")
    if got != want:
        print("selftest FAIL\n--- got\n" + got + "--- want\n" + want)
        return 1
    print("selftest: hand entry kept, its PR not duplicated, release skipped, 5 headings in order: ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version")
    ap.add_argument("--since", help="previous release tag, e.g. v0.67.0")
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.version and a.since):
        ap.error("--version and --since are required")
    text = CHANGELOG.read_text(encoding="utf-8")
    new = render(text, a.version, a.date, git_subjects(a.since), use_git=True)
    if a.dry_run:
        _, _, tail = split_unreleased(text)
        sys.stdout.write(new[: len(new) - len(tail)])
        return 0
    CHANGELOG.write_text(new, encoding="utf-8")
    print(f"CHANGELOG.md: [{a.version}] - {a.date} written from {a.since}..HEAD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
