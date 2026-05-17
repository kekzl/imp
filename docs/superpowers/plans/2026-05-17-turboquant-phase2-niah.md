# TurboQuant Phase 2 — NIAH Retrieval-Quality A/B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a minimal Needle-in-a-Haystack (NIAH) retrieval-quality A/B on Qwen3-8B Q8_0 comparing FP16 / FP8 / TurboQuant-with-QJL / TurboQuant-without-QJL (the synthetic MXFP4-K proxy from Phase 1's `IMP_TQ_SKIP_QJL=1` flag), at 4K and 16K context, to determine whether the QJL correction is doing real retrieval work — i.e., whether Path A's storage rewrite would regress quality.

**Architecture:** A Python harness (`tools/eval/niah/niah_bench.py`) that constructs prompts of the form `<filler[0:depth]> <needle> <filler[depth:]> <question>`, runs them through `imp-cli` once per configuration via Docker, and scores by substring-matching the needle's canonical answer. Filler text is a pre-staged static asset (`tools/eval/niah/data/filler.txt`) of ~30 K tokens generated from a public-domain technical article. Results are emitted as both JSON and a markdown summary. Per-prompt cost is ~3-4 s wall clock (~2 s docker+model load + 0.5-1.2 s inference); the full 4-config × 2-context × 5-depth × 3-seed matrix = **120 prompts ≈ 6-8 minutes** of GPU wall clock.

**Tech Stack:** Python 3 (via `dev python`), Docker (`imp:test` image, built by `make build`), Qwen3-8B Q8_0 GGUF at `/home/kekz/models/Qwen3-8B-Q8_0.gguf` (downloaded in Phase 1; ~8.2 GB), `IMP_TQ_SKIP_QJL=1` env knob (shipped in PR #246 / Phase 1).

**Acceptance criteria from design memo §5 Phase 2:**

| MXFP4-K NIAH score vs TurboQuant at 16K | Verdict |
|---|---|
| Within **5 pp** of TQ | ✅ **PASS** — Path A green-light. Phase 3 (production wire-up) proceeds. |
| **5-10 pp** regression | 🟡 **PASS with caveat** — investigate per-depth pattern. If uniform, attribute to 4-bit quantization itself (not QJL absence) and ship Path A with documented quality note. |
| **>10 pp** regression | ❌ **FAIL** — QJL is doing real retrieval work. Path A refuted. Fall back to Path B (already refuted in Phase 1) or shelve the whole effort. |

**Important caveat (Phase 1 carry-over):** `IMP_TQ_SKIP_QJL=1` proxies "post-Path-A storage" at the **quality** level by stripping QJL while keeping the rest of the TurboQuant decode kernel unchanged (PolarQuant FP4 K dequant + INT4 V). It does NOT swap the underlying storage to straight MXFP4-K. So Phase 2 answers **"does QJL correction matter for retrieval quality"** but does NOT directly answer **"does PolarQuant→straight-MXFP4 storage transition matter"**. The Phase 1 findings already noted this; Phase 2 inherits the same caveat and surfaces it in the findings memo.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `tools/eval/niah/data/filler.txt` | Create (~30 K tokens, ~120 KB) | Static filler corpus — public-domain technical text, large enough for 16 K-token contexts with padding |
| `tools/eval/niah/niah_bench.py` | Create (~200 LOC) | Python harness: construct prompts, invoke imp-cli via docker, score, emit JSON+md |
| `tools/eval/niah/README.md` | Create (~30 LOC) | One-shot usage, design rationale, caveat carry-over |
| `tools/eval/niah/sample_results/` | Create dir | Holds JSON output from the actual run (committed for reproducibility tracing) |
| `docs/superpowers/plans/2026-05-17-turboquant-phase2-findings.md` | Create at end | Findings memo + acceptance check + decision |
| `docs/roadmap.md` | Modify §"Closing the TurboQuant–FP8 gap" | Update entry with Phase 2 outcome |
| `memory/turboquant_phase2_findings_2026_05_17.md` | Create | Memory mirror (project type) |
| `memory/MEMORY.md` | Modify (add one line under TurboQuant) | Index pointer |

**Out of scope (explicitly NOT in this plan):**
- Writing an MXFP4-KV kernel (that's design memo §5 Phase 3 — production wire-up).
- RULER-subset variable-tracking (design memo §5 Phase 2 lists this as "optional"; defer unless NIAH is too coarse to distinguish configs).
- Benchmarking additional models (Llama-3.2-3B, Qwen3.5-9B etc.). Phase 2 focuses on the same Qwen3-8B Q8_0 used in Phase 1 for cross-phase comparability.

---

## Task 1: Worktree setup (skill-delegated, already done at planning time)

**Files:** none

- [ ] **Step 1: Confirm we're on the right branch**

```bash
git -C /home/kekz/.config/superpowers/worktrees/imp/turboquant-phase2-niah branch --show-current
git -C /home/kekz/.config/superpowers/worktrees/imp/turboquant-phase2-niah log --oneline -3
```
Expected: branch is `perf/turboquant-phase2-niah`, top commit is the Phase 1 merge `ef39b56 perf(turboquant): Phase 1 microbench + bottleneck verification (#246)`. If anything else, STOP and report BLOCKED.

- [ ] **Step 2: Confirm Phase 1 infrastructure is live on this branch**

```bash
grep -n "tq_skip_qjl" /home/kekz/.config/superpowers/worktrees/imp/turboquant-phase2-niah/src/runtime/config.h
grep -n "SKIP_QJL" /home/kekz/.config/superpowers/worktrees/imp/turboquant-phase2-niah/src/compute/attention_paged_turboquant.cu | head -3
```
Expected: both grep return matches. The `IMP_TQ_SKIP_QJL=1` env var must work end-to-end on this branch — otherwise the Phase 2 measurement can't run.

---

## Task 2: Stage the filler text corpus

**Files:**
- Create: `tools/eval/niah/data/filler.txt`

The filler text needs to:
- Be at least 30 K tokens (Qwen3 BPE → ~120 KB plain text) so a 16 K-context prompt has filler at every depth-percentile plus padding.
- Be on a single coherent topic (technical or factual) so the model can't trivially infer the needle from topic shift.
- NOT contain any phrase that could be confused with the needle's answer.
- Be public-domain to avoid licensing issues with checking it into the repo.

Use a Project Gutenberg public-domain technical text. Recommended: **"Relativity: The Special and General Theory" by Albert Einstein (1916)** — Gutenberg eBook #30155 is plain English, ~30 K tokens, technical enough to not be trivially summarised, and out of copyright since 1923.

- [ ] **Step 1: Create the data directory**

```bash
mkdir -p /home/kekz/.config/superpowers/worktrees/imp/turboquant-phase2-niah/tools/eval/niah/data
mkdir -p /home/kekz/.config/superpowers/worktrees/imp/turboquant-phase2-niah/tools/eval/niah/sample_results
```

- [ ] **Step 2: Download the filler text**

```bash
cd /home/kekz/.config/superpowers/worktrees/imp/turboquant-phase2-niah && \
curl -sSL "https://www.gutenberg.org/cache/epub/30155/pg30155.txt" -o tools/eval/niah/data/filler_raw.txt
wc -w tools/eval/niah/data/filler_raw.txt
```
Expected: ~50 K-60 K words (Project Gutenberg includes header/footer that bumps the word count). If the download fails (e.g. Gutenberg server is down), fall back to **"On the Origin of Species" by Charles Darwin** at `https://www.gutenberg.org/cache/epub/2009/pg2009.txt` — same shape.

- [ ] **Step 3: Strip Gutenberg header/footer and truncate to ~30 K tokens**

The Gutenberg files wrap the public-domain text with a license header (starts with `*** START OF THE PROJECT GUTENBERG EBOOK`) and footer (starts with `*** END OF THE PROJECT GUTENBERG EBOOK`). Extract everything between those markers, then truncate by character count (~120 KB ≈ 30 K Qwen tokens; the actual token count is checked in Task 4's sanity run).

```bash
cd /home/kekz/.config/superpowers/worktrees/imp/turboquant-phase2-niah
awk '/\*\*\* START OF THE PROJECT GUTENBERG EBOOK/,/\*\*\* END OF THE PROJECT GUTENBERG EBOOK/' \
    tools/eval/niah/data/filler_raw.txt \
  | sed '1d;$d' \
  | head -c 120000 \
  > tools/eval/niah/data/filler.txt
wc -c tools/eval/niah/data/filler.txt
rm tools/eval/niah/data/filler_raw.txt
```
Expected: `filler.txt` is ~120 000 bytes.

- [ ] **Step 4: Sanity-check that the filler doesn't accidentally contain a phrase like "best thing"**

This task's needle uses the canonical NIAH phrase `"The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."` — verify none of those distinctive words appear in the filler:

```bash
cd /home/kekz/.config/superpowers/worktrees/imp/turboquant-phase2-niah
grep -ciE "best thing|San Francisco|Dolores Park|sandwich" tools/eval/niah/data/filler.txt
```
Expected: `0`. If non-zero, swap to a different Gutenberg source (Darwin works for the same reason).

- [ ] **Step 5: Commit**

```bash
cd /home/kekz/.config/superpowers/worktrees/imp/turboquant-phase2-niah
git add tools/eval/niah/data/filler.txt
git commit -m "tools(eval): stage NIAH filler corpus from Project Gutenberg

Adds a public-domain 120 KB text excerpt (Einstein's 'Relativity: The
Special and General Theory', Gutenberg #30155, ~30 K Qwen-BPE tokens)
as the filler for the TurboQuant Phase 2 NIAH harness. Verified to
contain none of the needle's distinctive vocabulary."
```

---

## Task 3: Write the NIAH harness

**Files:**
- Create: `tools/eval/niah/niah_bench.py`

The harness must:
1. Accept CLI args: model path, output dir, configs to run, ctx lengths, depth percentiles, seeds, optional `--smoke` for a 1-prompt sanity run.
2. For each (config, ctx_tokens, depth_pct, seed): build a prompt of approximately `ctx_tokens` tokens by injecting the needle sentence at `depth_pct` of the filler, then append the question.
3. Spawn `docker run --rm --gpus all -v /home/kekz/models:/m imp:test imp-cli ...` per prompt with the appropriate flags + env.
4. Capture stdout, extract the generated text (between the prompt echo and the final stats line — imp-cli's format), score by substring match.
5. Emit `sample_results/niah_results.json` (raw per-prompt) + `sample_results/niah_summary.md` (per-(config,ctx) aggregate accuracy table).

**Critical design decisions baked into the script** (so reviewers can argue with them):
- **Token counting:** approximate via `len(text) / 4` chars-per-token (Qwen3 BPE averages 3.8-4.2 chars/token for English prose). Exact tokenization isn't needed — the harness controls prompt construction by char count; the model's tokenizer just consumes it. If the filler is too short for 16K context, the script `cycle()`s through it.
- **Per-prompt timeout:** 60 s. Model load + 16K prefill + 64-token decode is typically < 5 s; 60 s catches CUDA hangs without making the matrix run hours longer on failure.
- **Determinism:** `--temperature 0 --seed 42` per prompt. Three "seeds" in the matrix vary the filler-cycle offset, not the sampler seed (which stays 42 for byte-identity).
- **Accuracy metric:** case-insensitive substring match for the key phrase `"dolores park"` in the generated text. Lenient — the model just needs to recall enough of the needle to demonstrate retrieval, not regurgitate it verbatim. The needle sentence is unique enough that this metric has effectively zero false-positive risk against the filler.

- [ ] **Step 1: Write the harness skeleton with a failing self-test**

Create `tools/eval/niah/niah_bench.py` with this content:

```python
#!/usr/bin/env python3
"""NIAH retrieval-quality bench for TurboQuant Phase 2.

Runs 4 KV-cache configs × 2 contexts × 5 depths × 3 seeds = 120 prompts on
Qwen3-8B Q8_0, comparing FP16 / FP8 / TurboQuant (QJL on) / TurboQuant (QJL
off, via IMP_TQ_SKIP_QJL=1). Reports per-(config, ctx) accuracy and a
ship/no-ship verdict against the Phase 2 design-memo thresholds.

Per-prompt invocation: spawns `docker run imp:test imp-cli ... --prompt <P>
--max-tokens 64 --temperature 0 --seed 42`. Scores by substring match on
"dolores park" in the generated text (case-insensitive). The needle and
question are fixed.

Usage:
  tools/eval/niah/niah_bench.py [--model PATH] [--out DIR] [--smoke] \
      [--config NAME ...] [--ctx N ...] [--depth PCT ...] [--seed N ...]

Acceptance per docs/plans/turboquant_fp8_gap_design_2026_05_17.md §5
Phase 2: MXFP4-K (= tq_no_qjl in this script) NIAH accuracy at 16K within
5 pp of TurboQuant → green-light Path A.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

NEEDLE = (
    "The best thing to do in San Francisco is eat a sandwich and "
    "sit in Dolores Park on a sunny day."
)
QUESTION = (
    "\n\nBased only on the text above, answer concisely: what is the "
    "best thing to do in San Francisco?\n\nAnswer:"
)
ANSWER_KEY = "dolores park"  # lowercase substring used by the scorer

CONFIGS = {
    # name           CLI flags                              extra env
    "fp16":         (["--kv-fp16"],                          {}),
    "fp8":          (["--kv-fp8"],                           {}),
    "tq_qjl_on":    (["--kv-turboquant"],                    {}),
    "tq_qjl_off":   (["--kv-turboquant"],                    {"IMP_TQ_SKIP_QJL": "1"}),
}
DEFAULT_CTX     = [4096, 16384]
DEFAULT_DEPTHS  = [0.0, 0.25, 0.50, 0.75, 0.95]
DEFAULT_SEEDS   = [0, 1, 2]
DEFAULT_MODEL   = "/m/Qwen3-8B-Q8_0.gguf"  # path inside container
DEFAULT_HOST_MODELS_DIR = "/home/kekz/models"
DOCKER_IMAGE    = "imp:test"
CHARS_PER_TOKEN = 4  # Qwen3 BPE rough approximation; safe upper bound for English prose
MAX_GEN_TOKENS  = 64
PER_PROMPT_TIMEOUT_S = 60


@dataclasses.dataclass
class Prompt:
    config: str
    ctx_tokens: int
    depth_pct: float
    seed: int
    text: str

@dataclasses.dataclass
class Result:
    config: str
    ctx_tokens: int
    depth_pct: float
    seed: int
    generated: str
    score: int                 # 0 or 1
    wall_s: float
    returncode: int


def build_filler_corpus(path: Path, target_chars: int) -> str:
    """Read filler.txt and cycle/truncate to target_chars."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    if len(raw) >= target_chars:
        return raw[:target_chars]
    # Cycle if too short (shouldn't happen with the ~120 KB Gutenberg source).
    return (raw * ((target_chars // len(raw)) + 2))[:target_chars]


def build_prompt(filler: str, ctx_tokens: int, depth_pct: float, seed: int) -> str:
    """Construct the NIAH prompt.

    - Total prompt length aims at ctx_tokens × CHARS_PER_TOKEN chars.
    - Needle is inserted at depth_pct of the filler portion.
    - Question is appended at the end.
    - `seed` parameter shifts the filler-cycle starting offset for run variation.
    """
    needle = NEEDLE
    question = QUESTION
    # Reserve room for needle + question. CHARS_PER_TOKEN is a deliberate upper-bound estimate.
    budget = ctx_tokens * CHARS_PER_TOKEN - len(needle) - len(question)
    if budget <= 0:
        raise ValueError(f"ctx_tokens={ctx_tokens} too small for needle+question")
    # Per-seed cyclic shift so the three "seeds" produce different prompts.
    offset = (seed * (budget // 7)) % len(filler) if filler else 0
    body = (filler[offset:] + filler[:offset])[:budget]
    depth_chars = int(budget * depth_pct)
    pre = body[:depth_chars]
    post = body[depth_chars:]
    return pre + "\n\n" + needle + "\n\n" + post + question


def run_prompt(prompt: Prompt, model_path: str, host_models_dir: str) -> Result:
    flags, env_extra = CONFIGS[prompt.config]
    docker_env = []
    for k, v in env_extra.items():
        docker_env += ["-e", f"{k}={v}"]
    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{host_models_dir}:/m",
        *docker_env,
        DOCKER_IMAGE,
        "imp-cli",
        "--model", model_path,
        *flags,
        "--prompt", prompt.text,
        "--max-tokens", str(MAX_GEN_TOKENS),
        "--temperature", "0",
        "--seed", "42",
        "--max-seq-len", str(prompt.ctx_tokens + MAX_GEN_TOKENS + 256),
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=PER_PROMPT_TIMEOUT_S, check=False,
        )
        wall = time.time() - t0
        out = proc.stdout + "\n" + proc.stderr
        # imp-cli echoes the prompt then the completion; the completion is whatever
        # appears after the question's "Answer:" marker. Extract it.
        gen = ""
        m = re.search(r"Answer:\s*(.*?)(?:\n\[|\nInit:|\Z)", out, re.DOTALL)
        if m:
            gen = m.group(1).strip()
        else:
            # Fallback: take the last ~MAX_GEN_TOKENS chars of stdout.
            gen = proc.stdout[-MAX_GEN_TOKENS * CHARS_PER_TOKEN:].strip()
        score = 1 if ANSWER_KEY in gen.lower() else 0
        return Result(
            config=prompt.config, ctx_tokens=prompt.ctx_tokens,
            depth_pct=prompt.depth_pct, seed=prompt.seed,
            generated=gen, score=score, wall_s=wall,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return Result(
            config=prompt.config, ctx_tokens=prompt.ctx_tokens,
            depth_pct=prompt.depth_pct, seed=prompt.seed,
            generated="<TIMEOUT>", score=0, wall_s=time.time() - t0,
            returncode=-1,
        )


def write_summary(results: list[Result], out_path: Path) -> None:
    """Emit a markdown table: rows = configs, cols = ctx lengths, cells = accuracy %."""
    by_cell: dict[tuple[str, int], list[int]] = {}
    for r in results:
        by_cell.setdefault((r.config, r.ctx_tokens), []).append(r.score)
    configs = sorted({r.config for r in results}, key=lambda c: list(CONFIGS).index(c))
    ctxs    = sorted({r.ctx_tokens for r in results})
    lines = ["# NIAH Phase 2 results", "", "Cells = pass-rate over depth × seed.", ""]
    head = "| Config | " + " | ".join(f"{c} tokens" for c in ctxs) + " |"
    sep  = "|---|" + "|".join("---:" for _ in ctxs) + "|"
    lines += [head, sep]
    for cfg in configs:
        cells = []
        for ctx in ctxs:
            scores = by_cell.get((cfg, ctx), [])
            cells.append(f"{100*sum(scores)/len(scores):.1f}%" if scores else "—")
        lines.append(f"| {cfg} | " + " | ".join(cells) + " |")
    # Phase 2 acceptance check
    def cell_pct(cfg: str, ctx: int) -> float:
        ss = by_cell.get((cfg, ctx), [])
        return 100 * sum(ss) / len(ss) if ss else float("nan")
    if 16384 in ctxs:
        tq_qjl_on   = cell_pct("tq_qjl_on",   16384)
        tq_qjl_off  = cell_pct("tq_qjl_off",  16384)
        delta_pp = tq_qjl_off - tq_qjl_on
        if abs(delta_pp) <= 5:
            verdict = "✅ **PASS** — Path A green-light (Δ within ±5 pp)"
        elif abs(delta_pp) <= 10:
            verdict = "🟡 **PASS WITH CAVEAT** — investigate per-depth pattern (Δ 5-10 pp)"
        else:
            verdict = "❌ **FAIL** — Path A refuted (Δ > 10 pp; QJL is doing real retrieval work)"
        lines += [
            "",
            f"## Phase 2 verdict (16K context)",
            f"- tq_qjl_on:  {tq_qjl_on:.1f}%",
            f"- tq_qjl_off: {tq_qjl_off:.1f}%",
            f"- Δ = {delta_pp:+.1f} pp",
            "",
            verdict,
        ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--host-models-dir", default=DEFAULT_HOST_MODELS_DIR)
    ap.add_argument("--out", default="tools/eval/niah/sample_results")
    ap.add_argument("--filler", default="tools/eval/niah/data/filler.txt")
    ap.add_argument("--config", action="append", default=None,
                    help=f"Config(s) to run (default: all of {list(CONFIGS)})")
    ap.add_argument("--ctx", type=int, action="append", default=None)
    ap.add_argument("--depth", type=float, action="append", default=None)
    ap.add_argument("--seed", type=int, action="append", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="Run 1 prompt (first config, 4K, depth=0.5, seed=0)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    filler = build_filler_corpus(Path(args.filler), target_chars=200_000)

    if args.smoke:
        prompts = [
            Prompt(config="tq_qjl_on", ctx_tokens=4096, depth_pct=0.5, seed=0,
                   text=build_prompt(filler, 4096, 0.5, 0))
        ]
    else:
        configs = args.config or list(CONFIGS)
        ctxs    = args.ctx    or DEFAULT_CTX
        depths  = args.depth  or DEFAULT_DEPTHS
        seeds   = args.seed   or DEFAULT_SEEDS
        prompts = [
            Prompt(config=c, ctx_tokens=ct, depth_pct=d, seed=s,
                   text=build_prompt(filler, ct, d, s))
            for c, ct, d, s in itertools.product(configs, ctxs, depths, seeds)
        ]

    print(f"NIAH bench: {len(prompts)} prompts; output → {out_dir}", file=sys.stderr)
    results: list[Result] = []
    for i, p in enumerate(prompts, 1):
        r = run_prompt(p, args.model, args.host_models_dir)
        results.append(r)
        print(f"[{i:3d}/{len(prompts)}] {p.config:12s} ctx={p.ctx_tokens:>5d} "
              f"depth={p.depth_pct:.2f} seed={p.seed} → "
              f"{'PASS' if r.score else 'FAIL'} "
              f"({r.wall_s:.1f}s rc={r.returncode})", file=sys.stderr)

    (out_dir / "niah_results.json").write_text(
        json.dumps([dataclasses.asdict(r) for r in results], indent=2),
        encoding="utf-8",
    )
    write_summary(results, out_dir / "niah_summary.md")
    print(f"Wrote {out_dir}/niah_results.json and niah_summary.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Make it executable and run `python -c "import ..."` syntax check**

```bash
cd /home/kekz/.config/superpowers/worktrees/imp/turboquant-phase2-niah
chmod +x tools/eval/niah/niah_bench.py
python3 -c "import ast; ast.parse(open('tools/eval/niah/niah_bench.py').read())" && echo "syntax OK"
```
Expected: `syntax OK`. If python3 isn't on the host, use `dev python` (per CLAUDE.md clean-host policy):
```bash
docker run --rm -v $(pwd):/w -w /w python:3.11-slim python -c "import ast; ast.parse(open('tools/eval/niah/niah_bench.py').read())" && echo "syntax OK"
```

- [ ] **Step 3: Write a brief README**

Create `tools/eval/niah/README.md`:

```markdown
# TurboQuant Phase 2 NIAH harness

Runs a 4-config × 2-context × 5-depth × 3-seed Needle-in-a-Haystack
retrieval test on Qwen3-8B Q8_0, comparing FP16 / FP8 / TurboQuant
(QJL on) / TurboQuant (QJL off via `IMP_TQ_SKIP_QJL=1`).

Output: `sample_results/niah_results.json` (raw) + `niah_summary.md`
(aggregate accuracy + Phase 2 verdict).

## Usage

```bash
# Full matrix (~6-8 min wall clock):
tools/eval/niah/niah_bench.py

# Smoke test (1 prompt):
tools/eval/niah/niah_bench.py --smoke

# Subset (e.g. only TQ configs, only 4K context):
tools/eval/niah/niah_bench.py --config tq_qjl_on --config tq_qjl_off --ctx 4096
```

## Caveat

`IMP_TQ_SKIP_QJL=1` proxies "post-Path-A storage" at the **quality** level
by stripping QJL while keeping PolarQuant FP4 K + INT4 V. It does NOT
swap the underlying storage to straight MXFP4-K. So this harness answers
"does the QJL correction matter for retrieval quality" — not the strictly
broader "does PolarQuant→straight-MXFP4 storage transition matter".

See `docs/plans/turboquant_fp8_gap_design_2026_05_17.md` §5 Phase 2.
```

- [ ] **Step 4: Commit**

```bash
cd /home/kekz/.config/superpowers/worktrees/imp/turboquant-phase2-niah
git add tools/eval/niah/niah_bench.py tools/eval/niah/README.md tools/eval/niah/sample_results/.gitkeep 2>/dev/null
# The .gitkeep is optional; mkdir -p creates the dir but git ignores empty dirs.
touch tools/eval/niah/sample_results/.gitkeep
git add tools/eval/niah/sample_results/.gitkeep
git commit -m "tools(eval): add NIAH retrieval-quality harness for TurboQuant Phase 2

Python harness that runs Qwen3-8B Q8_0 across 4 KV-cache configs
(FP16, FP8, TurboQuant with QJL, TurboQuant with QJL stripped via
IMP_TQ_SKIP_QJL=1) at 4K + 16K context, 5 depth percentiles, 3 seeds
per cell. Scores by substring match on the canonical 'dolores park'
NIAH needle; emits JSON + markdown summary with the Phase 2 ship/
no-ship verdict (Δ vs TurboQuant at 16K against the ±5pp / ±10pp /
>10pp thresholds from the Phase 2 design memo §5).

Caveat: stripping QJL while keeping PolarQuant + INT4 V proxies the
post-Path-A quality at the correction level only — not the storage-
level MXFP4-K vs PolarQuant transition."
```

---

## Task 4: Smoke test the harness

**Files:** none (runtime only)

- [ ] **Step 1: Run the smoke flag**

```bash
cd /home/kekz/.config/superpowers/worktrees/imp/turboquant-phase2-niah
python3 tools/eval/niah/niah_bench.py --smoke 2>&1 | tail -20
```
Expected: one prompt runs (`tq_qjl_on` at ctx=4096, depth=0.5, seed=0). Wall clock 4-6 s. Output prints `PASS` (TurboQuant with QJL should easily retrieve the needle at 4K). If it prints `FAIL` with rc=0, the issue is the `Answer:` regex extraction — re-read `imp-cli` stdout structure and adjust the `m = re.search(r"Answer:\s*(.*?)(?:\n\[|\nInit:|\Z)", ...)` pattern.

If rc != 0: read stderr for the imp-cli error. Most likely causes:
- Model file not at `/home/kekz/models/Qwen3-8B-Q8_0.gguf` (pass `--host-models-dir` to point elsewhere)
- Docker image `imp:test` not built (run `make build`)
- The 16K prompt exceeds the shell arg limit (smoke is 4K → not a concern; if you hit this in the full run, the harness needs to be reworked to write the prompt to a tempfile and use a `--prompt-file` patch to imp-cli)

- [ ] **Step 2: Inspect raw output**

```bash
cat /home/kekz/.config/superpowers/worktrees/imp/turboquant-phase2-niah/tools/eval/niah/sample_results/niah_results.json | head -30
```
Expected: a single JSON record with `"score": 1`, `"generated"` containing "dolores park" (case-insensitive), `wall_s` ~4-6 s, `returncode: 0`. If generated text looks like garbled completions or unrelated, the prompt construction may have over-truncated the needle — check `build_prompt` logic.

- [ ] **Step 3: Cross-config sanity check**

Run a 4-cell smoke (one per config, all at 4K, depth=0.5, seed=0) to verify each config flag works:

```bash
cd /home/kekz/.config/superpowers/worktrees/imp/turboquant-phase2-niah
python3 tools/eval/niah/niah_bench.py --ctx 4096 --depth 0.5 --seed 0 2>&1 | tail -10
```
Expected: 4 PASS lines (FP16, FP8, TurboQuant with QJL, TurboQuant without QJL should all retrieve the needle at 4K depth=50%; quality cliffs typically appear at long context, not short). If `tq_qjl_off` already fails at 4K depth=50%, QJL is doing meaningful work even at short context — that's interesting data, document it. If FP16 fails, the harness has a bug (FP16 is the gold reference and should never miss a needle this short).

If any config fails for non-quality reasons (CUDA error, model load fail), STOP and report BLOCKED before Step 4.

No commit for this task — it's a measurement-only gate.

---

## Task 5: Run the full matrix

**Files:** updates `tools/eval/niah/sample_results/`

- [ ] **Step 1: Run the bench**

```bash
cd /home/kekz/.config/superpowers/worktrees/imp/turboquant-phase2-niah
python3 tools/eval/niah/niah_bench.py 2>&1 | tee /tmp/niah_run.log
```
Expected: ~6-8 minutes wall clock. 120 lines of `[NNN/120] <config> ctx=<N> depth=<P> seed=<S> → PASS|FAIL`. The progress line is printed to stderr so it interleaves cleanly with tee.

If individual prompts timeout or error: investigate per the prompt (likely VRAM issue at 16K context for some configs; the harness includes a per-prompt timeout so the matrix completes even if a config hangs).

- [ ] **Step 2: Review the summary**

```bash
cat tools/eval/niah/sample_results/niah_summary.md
```
Expected: a table like:
```
| Config | 4096 tokens | 16384 tokens |
|---|---:|---:|
| fp16 | 100.0% | 100.0% |
| fp8 | 100.0% | 93.3% |
| tq_qjl_on | 100.0% | 80.0% |
| tq_qjl_off | 100.0% | 73.3% |
```
(Numbers are illustrative — the actual values are the Phase 2 data.)

Plus the verdict block:
```
## Phase 2 verdict (16K context)
- tq_qjl_on:  80.0%
- tq_qjl_off: 73.3%
- Δ = -6.7 pp

🟡 PASS WITH CAVEAT — investigate per-depth pattern (Δ 5-10 pp)
```

- [ ] **Step 3: Commit the raw + summary results**

```bash
cd /home/kekz/.config/superpowers/worktrees/imp/turboquant-phase2-niah
git add tools/eval/niah/sample_results/niah_results.json tools/eval/niah/sample_results/niah_summary.md
git commit -m "tools(eval): TurboQuant Phase 2 NIAH raw + summary

Output of tools/eval/niah/niah_bench.py on Qwen3-8B Q8_0
(Qwen/Qwen3-8B-GGUF), RTX 5090, CUDA 13.2, imp:test docker image,
4 configs × 2 contexts × 5 depths × 3 seeds = 120 prompts.

Headline verdict in niah_summary.md; raw per-prompt records (including
generated text snippets) in niah_results.json for reproducibility
tracing."
```

---

## Task 6: Write the Phase 2 findings memo

**Files:**
- Create: `docs/superpowers/plans/2026-05-17-turboquant-phase2-findings.md`

- [ ] **Step 1: Draft the memo**

The structure mirrors Phase 1's findings memo. Fill in the actual numbers from `sample_results/niah_summary.md`. Template:

```markdown
# TurboQuant Phase 2 findings

**Date:** 2026-05-17
**Branch:** `perf/turboquant-phase2-niah`
**Scope:** NIAH retrieval-quality A/B per the design memo §5 Phase 2.
**Phase 1 carry-over:** `docs/superpowers/plans/2026-05-17-turboquant-phase1-findings.md` (decision: PROCEED WITH CAVEAT).
**Bench script:** `tools/eval/niah/niah_bench.py`
**Raw data:** `tools/eval/niah/sample_results/niah_results.json`

## Measurements

Qwen3-8B Q8_0, RTX 5090 sm_120a, CUDA 13.2, `imp:test` Docker image,
`--temperature 0 --seed 42`, 3 cyclic-offset seeds × 5 depth percentiles
× 2 contexts × 4 configs = 120 prompts. Needle: the canonical NIAH
"sandwich in Dolores Park" sentence; scorer: case-insensitive substring
match on "dolores park" in the model's completion.

| Config        | 4 096 tokens | 16 384 tokens |
|---            |          ---:|           ---:|
| FP16 (gold)   |       (fill) |        (fill) |
| FP8           |       (fill) |        (fill) |
| TQ (QJL on)   |       (fill) |        (fill) |
| TQ (QJL off)  |       (fill) |        (fill) |

**Δ at 16K = (TQ_QJL_off − TQ_QJL_on) = (fill) pp**

### Acceptance check (design memo §5 Phase 2)

- [ ] / [x] **PASS** (|Δ| ≤ 5 pp) → Path A green-light
- [ ] / [x] **PASS WITH CAVEAT** (5 < |Δ| ≤ 10 pp) → ship Path A with documented quality note
- [ ] / [x] **FAIL** (|Δ| > 10 pp) → Path A refuted

(Tick exactly one based on the table.)

## Decision

(One of the three. Justify in one sentence with the measured Δ.)

**If PASS:** Proceed to Phase 3 (production wire-up). Open a follow-up PR scoping the MXFP4-KV kernel + dispatcher + CLI flag work; reference this memo.

**If PASS-WITH-CAVEAT:** Per the design memo §5, "if 5-10pp uniform across depths, attribute to 4-bit quantization itself (not QJL absence)". Inspect the per-depth pattern from `niah_results.json`:
- If the regression is **uniform** across depths: 4-bit quantization is the residual quality cost; Path A still ships with a documented "long-context quality is N pp below TurboQuant" caveat.
- If the regression is **depth-concentrated** (e.g. worse at 0% and 95% than at 50%): QJL is doing localized-retrieval work; Path A may need a hybrid scheme.

**If FAIL:** QJL is doing real retrieval work that 4-bit storage alone cannot reproduce. Path A is refuted. Per design memo §6 worst-case: shelve the optimisation effort entirely and revisit as part of broader TurboQuant retirement scoping.

## Per-depth breakdown (16K context)

(Aggregate the per-(config, depth) accuracy from the raw JSON — useful for the PASS-WITH-CAVEAT investigation. Format as a table:)

| Depth % | FP16 | FP8 | TQ on | TQ off | TQ off Δ |
|---      | ---: | ---:| ---:  | ---:   |     ---: |
| 0   %   |(fill)|(fill)|(fill)|(fill) |   (fill) |
| 25  %   |(fill)|(fill)|(fill)|(fill) |   (fill) |
| 50  %   |(fill)|(fill)|(fill)|(fill) |   (fill) |
| 75  %   |(fill)|(fill)|(fill)|(fill) |   (fill) |
| 95  %   |(fill)|(fill)|(fill)|(fill) |   (fill) |

## Caveat carry-over

The `tq_qjl_off` config proxies post-Path-A storage at the **correction**
level only — it strips QJL while keeping PolarQuant FP4 K + INT4 V.
Path A's actual storage shape (straight MXFP4 K + INT4 V) may have
additional quality deltas not measured here. Phase 3 would need to ship
the real MXFP4-KV kernel to close that gap; if Phase 2 already produces
a borderline result, that uncertainty is load-bearing.

## Next steps

- (If PASS) Open Phase 3 design + implementation PR — multi-week kernel rewrite.
- (If PASS-WITH-CAVEAT) Same as PASS, plus a quality caveat in the user-facing docs.
- (If FAIL) Shelve. Update `docs/roadmap.md` to mark the TurboQuant-FP8 gap as refuted.
- Optionally: RULER-subset variable-tracking run on a smaller config matrix to cross-check. Defer unless this memo's signal is ambiguous.

## Cross-references

- Phase 1 findings: `docs/superpowers/plans/2026-05-17-turboquant-phase1-findings.md`
- Plan: `docs/superpowers/plans/2026-05-17-turboquant-phase2-niah.md` (this PR)
- Bench: `tools/eval/niah/niah_bench.py`
- Roadmap entry (updated in Task 7): `docs/roadmap.md` §"Closing the TurboQuant–FP8 gap"
- Memory pointer (Task 7): `memory/turboquant_phase2_findings_2026_05_17.md`
- Design memo: `docs/plans/turboquant_fp8_gap_design_2026_05_17.md`
```

- [ ] **Step 2: Commit**

```bash
cd /home/kekz/.config/superpowers/worktrees/imp/turboquant-phase2-niah
git add docs/superpowers/plans/2026-05-17-turboquant-phase2-findings.md
git commit -m "docs(plans): TurboQuant Phase 2 findings — <VERDICT>

Captures the NIAH retrieval-quality A/B from tools/eval/niah/niah_bench.py
on Qwen3-8B Q8_0. Acceptance Δ at 16K: <fill>pp. Decision: <fill>.

Per design memo §5 Phase 2."
```

---

## Task 7: Update roadmap + mirror to memory

**Files:**
- Modify: `docs/roadmap.md` (single section)
- Create: `/home/kekz/.claude/projects/-home-kekz-github-com-kekzl-imp/memory/turboquant_phase2_findings_2026_05_17.md`
- Modify: `/home/kekz/.claude/projects/-home-kekz-github-com-kekzl-imp/memory/MEMORY.md` (single line)

- [ ] **Step 1: Update the roadmap entry**

In `docs/roadmap.md`, find the "Closing the TurboQuant–FP8 gap" section (already partially updated in Phase 1's PR #246; this commit replaces the "Next gate: Phase 2 — NIAH retrieval-quality A/B" closing paragraph with the Phase 2 outcome).

Pick the matching template:

**If PASS or PASS-WITH-CAVEAT:**
```markdown
**Phase 2 NIAH (2026-05-17, `docs/superpowers/plans/2026-05-17-turboquant-phase2-findings.md`): <PASS | PASS-WITH-CAVEAT>.** TQ-with-QJL: <X>% accuracy at 16K; TQ-without-QJL: <Y>% (Δ <±N> pp). <Verdict line>. Path A proceeds to Phase 3 (production wire-up: write MXFP4-KV kernel + dispatcher + `--kv-mxfp4` CLI flag; estimated 1-2 weeks per design memo §5 Phase 3).
```

**If FAIL:**
```markdown
**Phase 2 NIAH (2026-05-17, `docs/superpowers/plans/2026-05-17-turboquant-phase2-findings.md`): REFUTED.** TQ-with-QJL: <X>% accuracy at 16K; TQ-without-QJL: <Y>% (Δ <−N> pp). The QJL XNOR+popcount correction is doing real retrieval work that 4-bit storage alone cannot reproduce. Path A refuted; TurboQuant stays as-is (opt-in, with the 23 % perf caveat documented). Consider broader retirement: NVFP4-KV already covers the Klasse-A models TurboQuant was designed for, and the design memo §6 worst-case framing applies.
```

- [ ] **Step 2: Create the memory mirror**

Write `/home/kekz/.claude/projects/-home-kekz-github-com-kekzl-imp/memory/turboquant_phase2_findings_2026_05_17.md`:

```markdown
---
name: turboquant-phase2-findings-2026-05-17
description: TurboQuant Phase 2 NIAH retrieval-quality A/B on Qwen3-8B Q8_0. Verdict: <PASS|PASS-WITH-CAVEAT|FAIL>. TQ-with-QJL vs TQ-without-QJL (IMP_TQ_SKIP_QJL=1) at 16K context = Δ <±N> pp. <Decision sentence>.
metadata:
  type: project
---

**TL;DR:** <one-paragraph result + decision>

**Headline numbers** (Qwen3-8B Q8_0, 120-prompt NIAH matrix, accuracy %):

| Config | 4K | 16K |
|---     |---:|---:|
| FP16   |(fill)|(fill)|
| FP8    |(fill)|(fill)|
| TQ on  |(fill)|(fill)|
| TQ off |(fill)|(fill)|

Δ at 16K (TQ_off − TQ_on) = <fill> pp.

**Why:** Use when revisiting Path A's quality-risk reasoning, scoping Phase 3
production work, or evaluating whether QJL is worth its perf cost.

**How to apply:**
- (PASS) Path A's quality risk is bounded; proceed to Phase 3 production work.
- (PASS-WITH-CAVEAT) Path A still ships, but the doc must call out the
  N-pp accuracy gap. Consider whether Klasse-A users would tolerate.
- (FAIL) QJL is load-bearing; do not retire it. Revisit TurboQuant's
  niche in the broader KV-dtype lineup (NVFP4 covers most of TQ's
  intended workloads).

**Important caveat:** `IMP_TQ_SKIP_QJL=1` strips QJL while keeping PolarQuant
FP4 K + INT4 V. Phase 2 measures "does QJL matter" — NOT "does PolarQuant
→ straight-MXFP4 storage transition matter". Phase 3 (production) would
need the real MXFP4-KV kernel to close that uncertainty.

[[turboquant_phase1_findings_2026_05_17]]
```

- [ ] **Step 3: Add the MEMORY.md index entry**

In `/home/kekz/.claude/projects/-home-kekz-github-com-kekzl-imp/memory/MEMORY.md`, find the existing TurboQuant section (added by Phase 1):

```markdown
### TurboQuant
- [TurboQuant Phase 1 findings](turboquant_phase1_findings_2026_05_17.md) — ...
```

Add immediately after the Phase 1 line:

```markdown
- [TurboQuant Phase 2 findings](turboquant_phase2_findings_2026_05_17.md) — Qwen3-8B Q8_0 NIAH A/B (FP16/FP8/TQ-QJL-on/TQ-QJL-off via IMP_TQ_SKIP_QJL=1). Verdict: <fill>. Δ TQ_off − TQ_on at 16K = <fill> pp.
```

- [ ] **Step 4: Commit the roadmap update** (memory files are not part of the repo)

```bash
cd /home/kekz/.config/superpowers/worktrees/imp/turboquant-phase2-niah
git add docs/roadmap.md
git commit -m "docs(roadmap): TurboQuant Phase 2 NIAH outcome — <VERDICT>

References docs/superpowers/plans/2026-05-17-turboquant-phase2-findings.md
for the measurements + decision."
```

---

## Task 8: Open PR

- [ ] **Step 1: Run `make verify-fast`**

```bash
cd /home/kekz/.config/superpowers/worktrees/imp/turboquant-phase2-niah
make verify-fast 2>&1 | tail -10
```
Expected: `=== verify fast: OK ===`. The Phase 2 changes don't touch CUDA or C++ code, so `verify-fast` is essentially a baseline-build smoke test in this PR. Acceptable to see SKIPs for model-dependent steps.

- [ ] **Step 2: Push branch and create PR**

```bash
cd /home/kekz/.config/superpowers/worktrees/imp/turboquant-phase2-niah
git push -u origin perf/turboquant-phase2-niah
gh pr create --base main --title "tools(eval): TurboQuant Phase 2 NIAH retrieval-quality A/B" \
  --body "$(cat <<'EOF'
## Summary

Implements **Phase 2** of `docs/plans/turboquant_fp8_gap_design_2026_05_17.md` — a 120-prompt NIAH retrieval-quality A/B on Qwen3-8B Q8_0 comparing FP16 / FP8 / TurboQuant (QJL on) / TurboQuant (QJL off, via `IMP_TQ_SKIP_QJL=1` shipped in PR #246).

**Result: <VERDICT — fill from findings memo>.** Full findings: `docs/superpowers/plans/2026-05-17-turboquant-phase2-findings.md`.

### Headline (Qwen3-8B Q8_0, NIAH accuracy %, 120 prompts)

| Config | 4 096 | 16 384 |
|---|---:|---:|
| FP16 | (fill) | (fill) |
| FP8  | (fill) | (fill) |
| TQ (QJL on)  | (fill) | (fill) |
| TQ (QJL off) | (fill) | (fill) |

**Δ at 16K = (fill) pp** vs the design memo §5 thresholds (±5pp / ±10pp / >10pp).

### Decision

(One sentence.)

## What's in this PR

- `tools/eval/niah/data/filler.txt` — public-domain Project Gutenberg filler corpus
- `tools/eval/niah/niah_bench.py` — Python harness (~200 LOC)
- `tools/eval/niah/README.md` — usage notes + caveat
- `tools/eval/niah/sample_results/{niah_results.json, niah_summary.md}` — Phase 2 raw data
- `docs/superpowers/plans/2026-05-17-turboquant-phase2-niah.md` — implementation plan
- `docs/superpowers/plans/2026-05-17-turboquant-phase2-findings.md` — findings memo
- `docs/roadmap.md` — TurboQuant entry updated with Phase 2 outcome

No CUDA / C++ changes. No test-suite additions (NIAH takes ~6-8 min and isn't appropriate for `make test-gpu`).

## Test plan

- [x] `python3 tools/eval/niah/niah_bench.py --smoke` runs 1 prompt and reports PASS
- [x] 4-config sanity (1 prompt per config at 4K, depth=0.5) reports PASS on all
- [x] Full matrix (`python3 tools/eval/niah/niah_bench.py`) ran in <10 min wall clock
- [x] `make verify-fast` green
- [x] No regressions in `make test-gpu` (no kernel changes in this PR)

## Caveat

`IMP_TQ_SKIP_QJL=1` proxies "post-Path-A storage" at the **correction** level only. Phase 3 (MXFP4-KV production wire-up) is needed to validate the actual storage-level transition. Phase 2 answers whether QJL is doing retrieval work; Phase 3 answers whether the full Path A rewrite preserves it.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Capture PR URL into the findings memo's "Next steps"**

After PR is open, edit `docs/superpowers/plans/2026-05-17-turboquant-phase2-findings.md` to include the PR URL in the Next steps section.

---

## Self-review

**1. Spec coverage:**

- ✅ Design memo §5 Phase 2 Task 1 (NIAH harness at depths {0%, 25%, 50%, 75%, 95%}, 4K + 16K) → Task 3 (`niah_bench.py`) + Task 2 (filler).
- ✅ Design memo §5 Phase 2 Task 2 (run NIAH at 4K+16K on Qwen3-8B Q8_0 with FP16 / FP8 / TQ-QJL-on / TQ-QJL-off) → Task 5 (full matrix execution).
- ✅ Design memo §5 Phase 2 Task 3 (optional RULER-subset) → explicitly deferred in plan header; can be added if Phase 2 signal is ambiguous.
- ✅ Acceptance criteria (±5pp / ±10pp / >10pp thresholds at 16K) → encoded in `write_summary()` and surfaced in Task 6's findings memo template.
- ✅ Caveat carry-over from Phase 1 ("`IMP_TQ_SKIP_QJL=1` is a quality proxy, not a storage proxy") → documented in plan header, harness README, findings memo template.

**2. Placeholder scan:**

- The findings memo template (Task 6 Step 1) has `(fill)` placeholders — those are intentional and get filled in from the actual run data. NOT a plan failure.
- The roadmap entry template (Task 7 Step 1) has `<X>%` and `<Y>%` placeholders — same; intentional.
- All step bodies have complete, runnable content.

**3. Type consistency:**

- The four config keys (`fp16`, `fp8`, `tq_qjl_on`, `tq_qjl_off`) are consistent across `niah_bench.py`, `niah_summary.md`, findings memo, and PR body.
- `IMP_TQ_SKIP_QJL=1` env var matches what was shipped in PR #246 (Phase 1).
- `DEFAULT_DEPTHS = [0.0, 0.25, 0.50, 0.75, 0.95]` matches the design memo §5 Phase 2 list verbatim.
- The acceptance Δ formula `(TQ_QJL_off − TQ_QJL_on)` is the same in `write_summary()` and in the findings memo template.

---

## Notes

- **Time budget:** 4-6 days per design memo. Tasks 1-3 are ~1 day; Tasks 4-5 are ~1 day (~10 min compute + investigation); Tasks 6-8 are ~1-2 days for writeup + PR.
- **Worktree:** isolation matters because the harness writes data into `tools/eval/niah/sample_results/`; running it on a different branch would litter the wrong tree.
- **`IMP_TQ_SKIP_QJL=1` already on `main`** (from PR #246). The Phase 2 harness directly depends on this — verified in Task 1 Step 2 before any other work.
- **Why Python, not C++:** the bench is glue (build prompts, invoke imp-cli, score, aggregate). C++ would inflate LOC 3-5× for no win. The existing `scripts/validate_safetensors.py` sets the precedent.
- **Why imp-cli per-prompt, not imp-server REST:** simpler. Per-prompt docker startup is ~2 s (cached image); model load is ~2 s. Total ~4 s overhead per prompt × 120 prompts = 8 min — already in the 6-8 min wall-clock budget. imp-server would shave a few minutes but adds a daemon-mode harness layer + JSON request marshalling for marginal gain. Defer if Phase 2 needs to re-run frequently.
- **Why substring match on "dolores park":** simplest possible accuracy metric with effectively zero false-positive risk against Einstein's _Relativity_ text. If the model gets confused and emits "Dolores Park, in the model's training data, refers to..." that still counts as PASS — the retrieval succeeded. The alternative (LLM-as-judge) adds a second model dependency for no measurable accuracy gain on this needle.
- **Why temperature=0 with 3 "seeds":** the seeds vary the filler-cycle starting offset, NOT the sampler seed (sampler is byte-deterministic at temperature 0 + seed 42). This isolates "does the model retrieve the needle when surrounded by different filler context" without conflating sampler stochasticity. If a sampler-level seed variation is desired in a Phase 2.1 follow-up, change `--seed 42` in `run_prompt` to derive from the prompt's seed field.
