#!/usr/bin/env python3
"""NIAH retrieval-quality bench for KV cache dtypes.

Runs KV-cache configs × contexts × depths × seeds on Qwen3-8B Q8_0,
comparing FP16 / FP8 / NVFP4 / MXFP4-KV. Reports per-(config, ctx) accuracy.

Per-prompt invocation: spawns `docker run imp:test imp-cli ... --prompt <P>
--max-tokens 64 --temperature 0 --seed 42`. Scores by substring match on
"dolores park" in the generated text (case-insensitive). The needle and
question are fixed.

Usage:
  tools/eval/niah/niah_bench.py [--model PATH] [--out DIR] [--smoke] \\
      [--config NAME ...] [--ctx N ...] [--depth PCT ...] [--seed N ...]

Note: TurboQuant (tq_qjl_on / tq_qjl_off) configs were removed in Phase 5
(2026-05-17) when TurboQuant was retired. Historical findings are documented in
docs/plans/turboquant_fp8_gap_design_2026_05_17.md §5.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
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
    "nvfp4":        (["--kv-nvfp4"],                         {}),
    "mxfp4_kv":     (["--kv-mxfp4"],                         {}),
    # tq_qjl_on / tq_qjl_off removed: TurboQuant retired Phase 5 (2026-05-17)
}
DEFAULT_CTX     = [4096, 16384]
DEFAULT_DEPTHS  = [0.0, 0.25, 0.50, 0.75, 0.95]
DEFAULT_SEEDS   = [0, 1, 2]
DEFAULT_MODEL   = "/m/Qwen3-8B-Q8_0.gguf"  # path inside container
DEFAULT_HOST_MODELS_DIR = str(Path.home() / "models")
DOCKER_IMAGE    = "imp:test"
CHARS_PER_TOKEN = 4  # Qwen3 BPE rough approximation; safe upper bound for English prose
# Qwen3-8B is a reasoning model — it emits <think>...</think> before the
# final answer, typically 100-200 tokens of "thinking" preamble. 384 gives
# enough headroom for the reasoning preamble plus a complete answer.
MAX_GEN_TOKENS  = 384
PER_PROMPT_TIMEOUT_S = 90


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
    budget = ctx_tokens * CHARS_PER_TOKEN - len(needle) - len(question)
    if budget <= 0:
        raise ValueError(f"ctx_tokens={ctx_tokens} too small for needle+question")
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
            errors="replace",  # MXFP4-KV degenerate output can emit invalid UTF-8 bytes
        )
        wall = time.time() - t0
        # imp-cli --prompt mode does NOT echo the prompt to stdout — only
        # the model's generation (with some interleaved log lines). The
        # needle "Dolores Park" appears in the prompt but never in stdout
        # unless the model retrieves it. Score on the full stdout: this
        # catches the needle whether the model surfaces it in <think>
        # reasoning or in the final answer.
        score = 1 if ANSWER_KEY in proc.stdout.lower() else 0
        # For the JSON record's human-readable generated field, keep the
        # tail of stdout (last ~2 KB) so reviewers can spot-check.
        gen = proc.stdout[-2048:].strip()
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

    def cell_pct(cfg: str, ctx: int) -> float:
        ss = by_cell.get((cfg, ctx), [])
        return 100 * sum(ss) / len(ss) if ss else float("nan")

    # Slice 3 re-run: with the real MXFP4-KV kernel shipped in PR #249, the
    # Key quality question: nvfp4 vs mxfp4_kv (does UE8M0 vs E4M3 scale
    # encoding affect retrieval quality?). TurboQuant configs removed Phase 5.
    if 16384 in ctxs and ("nvfp4", 16384) in by_cell and ("mxfp4_kv", 16384) in by_cell:
        nvfp4_pct    = cell_pct("nvfp4",    16384)
        mxfp4_kv_pct = cell_pct("mxfp4_kv", 16384)
        delta_pp = mxfp4_kv_pct - nvfp4_pct
        if abs(delta_pp) <= 5:
            verdict = "✅ **PASS** — Path A green-light (Δ within ±5 pp)"
        elif abs(delta_pp) <= 10:
            verdict = "🟡 **PASS WITH CAVEAT** — investigate per-depth pattern (Δ 5-10 pp)"
        else:
            verdict = "❌ **FAIL** — Path A refuted (Δ > 10 pp; UE8M0 scale encoding regresses retrieval)"
        lines += [
            "",
            "## Path A verdict (16K context, nvfp4 vs mxfp4_kv)",
            f"- nvfp4:    {nvfp4_pct:.1f}%",
            f"- mxfp4_kv: {mxfp4_kv_pct:.1f}%",
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
                    help="Run 1 prompt (mxfp4_kv, 4K, depth=0.5, seed=0)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    filler = build_filler_corpus(Path(args.filler), target_chars=200_000)

    if args.smoke:
        prompts = [
            Prompt(config="mxfp4_kv", ctx_tokens=4096, depth_pct=0.5, seed=0,
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
