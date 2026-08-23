#!/usr/bin/env python3
"""Documentation linter. Six checks fail the build; staleness only warns.

That split was undocumented until #1683, which found the header claiming all
seven were blocking while check 7 appended to `warnings` and the exit code read
`errors` alone. Staleness stays a warning on purpose - a date does not tell you
whether the content moved - but it says so now, and the commit marker it sits
next to is checked and reported (see check 3).

The point of each check is that it catches a class of defect this repo has
actually shipped before, not that it enforces a style:

  1. forbidden tokens  - a datacenter-Blackwell feature named as an imp feature.
                         The delimitation itself is legitimate and lives in ONE
                         allowlisted place.
  2. unprovenanced     - a throughput number without a [PROV: ...] block near
     numbers             it. "decode 287" with no model and no date is how a
                         figure outlives the measurement it came from.
  3. frontmatter       - every in-scope doc declares exactly one layer, so a
                         reader knows whether it may assume CUDA knowledge.
  4. generated drift   - a PERF block hand-edited away from its source.
  5. dead links        - an internal link to a file that does not exist.
  6. size budgets      - README and the CLAUDE.md hierarchy.
  7. staleness         - `verified:` older than 180 days is a warning, listed
                         in docs/audit/docs-rewrite/STALE.md.

Records (archives, append-only ledgers, journals) are EXCLUDED. A record is a
statement about one dated afternoon; demanding a freshness refresh on it would
be demanding that history be rewritten.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# --- scope -----------------------------------------------------------------

EXCLUDED_PREFIXES = (
    "docs/archive/",
    "docs/audit/",
    "docs/plans/",
    "third_party/",
    "build/",
    "build-dev/",
)
EXCLUDED_FILES = {
    "CHANGELOG.md",          # append-only
    "docs/MISSION_JOURNAL.md",  # append-only
    "docs/vram_audit.md",    # append-only, says so in its own title
    "AUDIT.md",              # running findings log
    # The roadmap is a dated research record ("a gap list, not a schedule"):
    # its numbers are the narrative of how each gap was measured or refuted,
    # and stripping them for provenance blocks would destroy the record. The
    # reader-facing distillation lives in LIMITATIONS.md / DESIGN_DECISIONS.md
    # / PERF.md, which ARE linted.
    "docs/roadmap.md",
}
EXCLUDED_SUBSTRINGS = (".pytest_cache/", "node_modules/")

# The single place allowed to spell out what consumer Blackwell does NOT have.
#
# docs/internals/ is allowlisted as a whole, and the distinction is the point of
# the rule rather than a hole in it. What must not be repeated is the
# *delimitation* ("sm_120a is not a small B200"), because a reader who meets it
# in eight places cannot tell which one is maintained. What an L2 kernel
# document does instead is derive from it: "no tcgen05, therefore the MMA blocks
# the issuing warp" is design rationale, and deleting it would leave the kernel's
# shape unexplained. L0 and L1 have no business doing either; they link.
DELIMITATION_ALLOWLIST = {"docs/internals/ARCHITECTURE.md"}
DELIMITATION_ALLOWLIST_PREFIXES = ("docs/internals/",)

# Agent-layer files may name the architecture boundary as a guardrail; that is
# their job. They are size-budgeted instead.
AGENT_FILES_ALLOWLIST = {"CLAUDE.md", "AGENTS.md"}

FORBIDDEN = [
    (r"\btcgen05\b", "tcgen05"),
    (r"\bTMEM\b", "TMEM"),
    (r"\bwgmma\b|\bWGMMA\b", "wgmma"),
    (r"\bsm_100\b", "sm_100"),
    (r"\bsm_90a?\b", "sm_90"),
    (r"\bHopper\b", "Hopper"),
    (r"1258\s*tok/s", "the stale 1258 tok/s figure"),
    # (?<![\d.]) so a legitimate ratio like "1.20x behind" is not mistaken for
    # the stale "20x behind vLLM" headline this rule exists to keep out.
    (r"(?<![\d.])20\s*[x×]\s+behind", "the stale '20x behind vLLM' claim"),
]

# A throughput/ratio figure that must carry provenance.
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:k\s*)?(?:tok/s|tokens/s|x faster|x behind)\b")
PROV_RE = re.compile(r"\[PROV:")

# Documents that carry provenance per row in a convention they declare in their
# own header, predating this linter. The rule is "no number without provenance",
# not "no number without this particular syntax": BENCHMARKS.md already states
# that every row names its date, commit, CUDA version, quant and command, and
# GOAL.md dates each re-measurement inline. Rewriting either into [PROV:] blocks
# would move the provenance without adding any. They must keep declaring it,
# which is what PROV_HEADER_RE checks.
PROV_HEADER_ALLOWLIST = {"docs/BENCHMARKS.md", "docs/GOAL.md", "docs/MODELS.md"}
PROV_HEADER_RE = re.compile(r"commit|re-measured|measured", re.I)

# The metadata header is an HTML COMMENT, not YAML frontmatter, and that is
# deliberate: GitHub renders YAML front matter as a visible table at the top of
# the page, so the README - the one file that exists for first contact - opened
# with `layer / audience / verified / commit` instead of with what imp is.
# Machine-readable and invisible beats machine-readable and in the reader's way.
# The legacy `---` form is still accepted so a file is never silently unchecked.
FRONTMATTER_RE = re.compile(r"\A(?:<!--\n(.*?)\n-->|---\n(.*?)\n---)\n", re.S)
VALID_LAYERS = {"L0", "L1", "L2", "L3"}

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")

README_MAX_LINES = 400
CLAUDE_ROOT_MAX_TOKENS = 2000
CLAUDE_DIR_MAX_TOKENS = 800
STALE_DAYS = 180
# A `commit:` marker is reported when THE FILE ITSELF changed after it, not at
# some commit count. The count was the first attempt and it was arbitrary: 200
# did not fire on the case that motivated the check (40 files pinned at
# 81ffa573 with main 133 commits ahead), and any number that did fire would
# have been picked to fit that one case. "Was this file edited since it was
# last verified" needs no threshold and is the question the field claims to
# answer (#1683).
_COMMIT_DEPTH_CACHE: dict = {}


def _edits_since(sha: str, path: str):
    """Commits touching `path` after `sha`. None if `sha` is not in history."""
    key = (sha, path)
    if key in _COMMIT_DEPTH_CACHE:
        return _COMMIT_DEPTH_CACHE[key]
    import subprocess
    val = None
    try:
        if subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"],
                          capture_output=True, timeout=10).returncode == 0:
            out = subprocess.run(["git", "rev-list", "--count", f"{sha}..HEAD", "--", path],
                                 capture_output=True, text=True, timeout=10)
            if out.returncode == 0 and out.stdout.strip():
                val = int(out.stdout.strip())
    except Exception:
        val = None
    _COMMIT_DEPTH_CACHE[key] = val
    return val


BASELINE = ROOT / "tests" / "perf_baseline.json"


def _ignored_paths(candidates: list[str]) -> set[str]:
    """The subset of `candidates` that .gitignore excludes (#1663).

    Without this the linter walks whatever happens to be in the tree. A local
    scratch directory - `_audit/` during the 2026-08 audit - produced 160
    errors on every run, and the working answer became `| grep -v '^FAIL
    _audit/'`. A gate whose output has to be filtered by hand is one command
    away from not being read at all.

    `git check-ignore` rather than `git ls-files`: a doc that is written but
    not yet added is still in scope, and would otherwise pass locally and fail
    in CI. Falls back to "nothing is ignored" when git is unavailable, which is
    the old behaviour.
    """
    if not candidates:
        return set()
    try:
        r = subprocess.run(
            ["git", "check-ignore", "--stdin"],
            input="\n".join(candidates),
            capture_output=True, text=True, cwd=ROOT, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    # exit 0 = some paths ignored, 1 = none, 128 = not a git repo
    if r.returncode not in (0, 1):
        return set()
    return {line.strip() for line in r.stdout.splitlines() if line.strip()}


def in_scope(rel: str) -> bool:
    # Dot-directories are agent state, skills and GitHub templates. They are
    # not repository documentation: `.claude/skills/sm120-cuda-expert` names
    # tcgen05 precisely because its job is to stop an agent reaching for it.
    if rel.startswith("."):
        return False
    if rel in EXCLUDED_FILES:
        return False
    if any(s in rel for s in EXCLUDED_SUBSTRINGS):
        return False
    return not rel.startswith(EXCLUDED_PREFIXES)


def approx_tokens(text: str) -> int:
    """Cheap, stable proxy: ~4 chars per token. Exactness is not the point,
    catching a file that doubled in size is."""
    return len(text) // 4


def check_file(path: pathlib.Path, rel: str, errors: list, warnings: list) -> None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    layer = None

    # 3. frontmatter
    m = FRONTMATTER_RE.match(text)
    if not m:
        errors.append(f"{rel}: missing frontmatter (needs layer/audience/verified/commit)")
    else:
        fm = m.group(1) or m.group(2)
        # The error message above promises four fields and this validated one
        # until #1683: `audience:` and `commit:` were read by no line in the
        # file, so 40 documents carried `commit: 81ffa573` unnoticed while main
        # moved 133 commits past it.
        for field in ("audience", "commit"):
            if not re.search(rf"^{field}:\s*\S+", fm, re.M):
                errors.append(f"{rel}: frontmatter has no `{field}:`")
        cm = re.search(r"^commit:\s*([0-9a-f]{7,40})", fm, re.M)
        if cm:
            edits = _edits_since(cm.group(1), rel)
            if edits is None:
                warnings.append(f"{rel}: commit {cm.group(1)} is not in this history")
            elif edits > 0:
                warnings.append(
                    f"{rel}: edited {edits}x since the commit it says it was verified "
                    f"against ({cm.group(1)})")
        lm = re.search(r"^layer:\s*(\S+)", fm, re.M)
        if not lm:
            errors.append(f"{rel}: frontmatter has no `layer:`")
        elif lm.group(1) not in VALID_LAYERS:
            errors.append(f"{rel}: unknown layer {lm.group(1)!r}, expected one of {sorted(VALID_LAYERS)}")
        else:
            layer = lm.group(1)
        # 7. staleness
        vm = re.search(r"^verified:\s*(\d{4}-\d{2}-\d{2})", fm, re.M)
        if vm:
            age = (dt.date.today() - dt.date.fromisoformat(vm.group(1))).days
            if age > STALE_DAYS:
                warnings.append(f"{rel}: verified {age} days ago")

    # 1. forbidden tokens
    if (rel not in DELIMITATION_ALLOWLIST
            and rel not in AGENT_FILES_ALLOWLIST
            and not rel.startswith(DELIMITATION_ALLOWLIST_PREFIXES)):
        for i, line in enumerate(lines, 1):
            for pat, name in FORBIDDEN:
                if re.search(pat, line):
                    errors.append(
                        f"{rel}:{i}: names {name}. The sm_120a delimitation belongs in "
                        f"docs/internals/ARCHITECTURE.md; link there instead of restating it."
                    )
                    break

    # 2. numbers without provenance.
    #
    # Severity depends on the layer, deliberately. In L0/L1 a number is a claim
    # made TO a reader who cannot check it, so it must carry its referent or go.
    # In L2 a number is usually the result of an experiment the surrounding
    # paragraph describes, often one that was refuted; making that a build
    # failure would push the next author to delete the figure rather than
    # document it, which is the opposite of what this linter is for.
    if rel in PROV_HEADER_ALLOWLIST and not PROV_HEADER_RE.search("\n".join(lines[:40])):
        errors.append(
            f"{rel}: is allowlisted for inline provenance but its header no longer "
            f"declares that convention. Either restore the declaration or move to [PROV:] blocks."
        )

    for i, line in enumerate(lines, 1):
        if not NUMBER_RE.search(line):
            continue
        if rel in PROV_HEADER_ALLOWLIST:
            continue
        window = "\n".join(lines[max(0, i - 12): i + 12])
        if PROV_RE.search(window):
            continue
        msg = f"{rel}:{i}: throughput figure with no [PROV: ...] block within 12 lines"
        if layer in ("L2", "L3"):
            warnings.append(msg)
        else:
            errors.append(msg)

    # 5. dead internal links
    for i, line in enumerate(lines, 1):
        for target in LINK_RE.findall(line):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                errors.append(f"{rel}:{i}: dead link -> {target}")


def check_generated_blocks(errors: list) -> None:
    """4. The PERF block must match tests/perf_baseline.json."""
    if not BASELINE.exists():
        errors.append("tests/perf_baseline.json missing; cannot verify generated blocks")
        return
    data = json.loads(BASELINE.read_text())
    decode = data["metrics"]["decode_tps"]["tg128"]
    for rel in ("README.md", "docs/PERF.md"):
        path = ROOT / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        block = re.search(r"<!-- PERF:BEGIN -->(.*?)<!-- PERF:END -->", text, re.S)
        if not block:
            errors.append(f"{rel}: no <!-- PERF:BEGIN -->/<!-- PERF:END --> block")
            continue
        if str(decode) not in block.group(1):
            errors.append(
                f"{rel}: PERF block drifted from tests/perf_baseline.json "
                f"(expected decode {decode}). Run scripts/sync_docs.py."
            )


def check_refs_generators_listed(errors: list) -> None:
    """7. Every generator in tests/refs/ has a row in tests/refs/README.md.

    That table is the only index of which golden a generator writes and which
    test consumes it, and rule 1 of that README ("every golden value traces to a
    committed generator") is only checkable through it. Two generators had
    drifted out of it - gen_tokenizer_golden.py and gen_chat_goldens.py - which
    is the repo's recurring shape: the artefact exists, nothing references it,
    and its absence reads like absence of the thing itself.
    """
    refs = ROOT / "tests" / "refs"
    readme = refs / "README.md"
    if not refs.is_dir() or not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    missing = sorted(p.name for p in refs.glob("gen_*.py") if p.name not in text)
    if missing:
        errors.append(
            "tests/refs/README.md: generator(s) with no row in the table: "
            + ", ".join(missing)
            + " — add one naming the golden it writes and the test that consumes it"
        )


def check_budgets(errors: list) -> None:
    """6. README and CLAUDE.md size budgets."""
    readme = ROOT / "README.md"
    n = len(readme.read_text(encoding="utf-8").splitlines())
    if n > README_MAX_LINES:
        errors.append(f"README.md: {n} lines > {README_MAX_LINES}")

    root_claude = ROOT / "CLAUDE.md"
    if root_claude.exists():
        t = approx_tokens(root_claude.read_text(encoding="utf-8"))
        if t > CLAUDE_ROOT_MAX_TOKENS:
            errors.append(f"CLAUDE.md: ~{t} tokens > {CLAUDE_ROOT_MAX_TOKENS}")

    claude_files = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("CLAUDE.md")]
    ignored = _ignored_paths(claude_files)
    for rel in claude_files:
        p = ROOT / rel
        if rel == "CLAUDE.md" or rel in ignored or rel.startswith(EXCLUDED_PREFIXES):
            continue
        t = approx_tokens(p.read_text(encoding="utf-8"))
        if t > CLAUDE_DIR_MAX_TOKENS:
            errors.append(f"{rel}: ~{t} tokens > {CLAUDE_DIR_MAX_TOKENS}")


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    candidates = [p.relative_to(ROOT).as_posix() for p in sorted(ROOT.rglob("*.md"))]
    ignored = _ignored_paths(candidates)
    for rel in candidates:
        if rel in ignored or not in_scope(rel):
            continue
        check_file(ROOT / rel, rel, errors, warnings)

    check_generated_blocks(errors)
    check_refs_generators_listed(errors)
    check_budgets(errors)

    if warnings:
        stale = ROOT / "docs" / "audit" / "docs-rewrite" / "STALE.md"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text(
            "# Stale docs\n\nGenerated by scripts/docs_lint.py. "
            f"Threshold {STALE_DAYS} days.\n\n"
            + "".join(f"- {w}\n" for w in warnings),
            encoding="utf-8",
        )
        for w in warnings:
            print(f"WARN  {w}")

    for e in errors:
        print(f"FAIL  {e}")

    if errors:
        print(f"\ndocs_lint: {len(errors)} error(s), {len(warnings)} warning(s)")
        return 1
    print(f"docs_lint: OK ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
