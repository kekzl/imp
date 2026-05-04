#!/usr/bin/env python3
"""SafeTensors validation harness — reduced-scope (Mode A).

Per-model phases executed:
  Phase 0 — Load + tokenizer roundtrip
  Phase 3 — CUDA Graph capture + 32x byte-identical replay (deterministic_gemm=true)
  Phase 4 — 20-prompt battery at temperature=0.0
  Phase 5 — Degeneracy gates (4-gram rep, logit health, output sanity, determinism)
  Phase 6 — Performance smoke (TTFT, decode tok/s, peak VRAM)

NOT executed (Mode A):
  Phase 1 — BF16 reference (no BF16 checkpoints on disk)
  Phase 2 — NVFP4 calibration (imp consumes pre-quantized weights)
  Phase 5c — KL/PPL drift vs BF16 (no reference)

Usage:
  scripts/validate_safetensors.py [--model NAME [--model NAME ...]] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.request
import urllib.error
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"
ARTIFACTS = REPO_ROOT / "validation_artifacts"
DOCKER_IMG = os.environ.get("IMP_DOCKER_IMG", "imp:test")
SERVER_PORT = int(os.environ.get("IMP_VALIDATION_PORT", "8765"))
SEED = 42
MAX_PROMPTS_SMOKE = 4

# Override with IMP_MODELS_DIR env var; defaults to ./models under the repo.
MODELS_DIR = Path(os.environ.get("IMP_MODELS_DIR", REPO_ROOT / "models")).expanduser()


def _model_entry(name: str, *, extra_args: list[str] | None = None,
                 skip_repeat_penalty_lc_check: bool = False) -> dict:
    return {
        "name": name,
        "host_path": str(MODELS_DIR / name),
        "container_path": f"/models_ext/{name}",
        "extra_server_args": extra_args or [],
        "chat_template": "auto",
        "skip_repeat_penalty_lc_check": skip_repeat_penalty_lc_check,
    }


MODELS = [
    _model_entry("Mistral-Small-3.2-24B-Instruct-2506-NVFP4"),
    _model_entry("Gemma-4-26B-A4B-it-NVFP4"),
    _model_entry("Qwen3.6-35B-A3B-NVFP4"),
    _model_entry("Qwen3-Coder-30B-A3B-Instruct-FP4"),
    _model_entry("Qwen3-30B-A3B-NVFP4-Modelopt"),
    _model_entry("Nemotron-3-Nano-30B-A3B-NVFP4"),
]


@dataclass
class PromptResult:
    id: int
    name: str
    text: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float
    ttft_s: float | None
    decode_tok_per_s: float | None
    logprobs_health: dict
    check_pass: bool
    check_reason: str
    extra: dict = field(default_factory=dict)


@dataclass
class ModelReport:
    name: str
    path: str
    verdict: str = "PENDING"
    failure_phase: str = "n/a"
    failure_reason: str = "n/a"
    arch: str = ""
    param_count_b: float = 0.0
    weight_files: list[str] = field(default_factory=list)
    weight_bytes: int = 0
    config_keys: dict = field(default_factory=dict)
    phase0: dict = field(default_factory=dict)
    phase3: dict = field(default_factory=dict)
    phase4: dict = field(default_factory=dict)
    phase5: dict = field(default_factory=dict)
    phase6: dict = field(default_factory=dict)
    prompts: list[PromptResult] = field(default_factory=list)


# ---------- helpers ----------

def log(msg: str) -> None:
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


def http_post(url: str, body: dict, timeout: int = 600) -> tuple[int, dict | None, str]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Connection": "close"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(raw), raw
            except json.JSONDecodeError:
                return resp.status, None, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw), raw
        except json.JSONDecodeError:
            return e.code, None, raw
    except urllib.error.URLError as e:
        return 0, None, f"URLError: {e}"
    except (TimeoutError, socket.timeout) as e:
        return 0, None, f"Timeout: {e}"


def http_get(url: str, timeout: int = 30) -> tuple[int, dict | None, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(raw), raw
            except json.JSONDecodeError:
                return resp.status, None, raw
    except urllib.error.HTTPError as e:
        return e.code, None, e.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        return 0, None, f"URLError: {e}"


def gpu_mem_used_mb() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=5, text=True)
        return int(out.strip().split("\n")[0])
    except Exception:
        return None


# ---------- server lifecycle ----------

class ImpServer:
    def __init__(self, model_cfg: dict, log_path: Path, port: int = SERVER_PORT,
                 deterministic: bool = True, no_cuda_graphs: bool = False):
        self.cfg = model_cfg
        self.port = port
        self.log_path = log_path
        self.deterministic = deterministic
        self.no_cuda_graphs = no_cuda_graphs
        self.proc: subprocess.Popen | None = None
        self.container_name = f"imp-validator-{int(time.time())}-{port}"

    def __enter__(self):
        host_dir = Path(self.cfg["host_path"])
        if not host_dir.exists():
            raise FileNotFoundError(f"Model dir missing: {host_dir}")

        # Mount the parent directory of the model so siblings (e.g. tokenizer
        # extras) are reachable, plus the repo root for fixtures.
        host_root = str(host_dir.parent)
        cont_root = "/models_ext"
        cont_model = f"{cont_root}/{host_dir.name}"

        cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "--name", self.container_name,
            "-v", f"{host_root}:{cont_root}:ro",
            "-v", f"{REPO_ROOT}:/imp:ro",
            "-p", f"{self.port}:{self.port}",
            "-e", "IMP_QUIET=0",
            DOCKER_IMG,
            "imp-server",
            "--model", cont_model,
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "--max-tokens", "2048",
        ]
        if self.no_cuda_graphs:
            cmd.append("--no-cuda-graphs")
        if self.deterministic:
            cmd += ["--set", "runtime.deterministic_gemm=true"]
        cmd += self.cfg.get("extra_server_args", [])

        log(f"server cmd: {' '.join(shlex.quote(c) for c in cmd)}")
        self.log_fp = open(self.log_path, "w", buffering=1)
        self.proc = subprocess.Popen(cmd, stdout=self.log_fp, stderr=subprocess.STDOUT,
                                     preexec_fn=os.setsid)
        # wait for "Server listening" or process exit
        start = time.time()
        ready = False
        while time.time() - start < 600:
            if self.proc.poll() is not None:
                break
            try:
                with open(self.log_path, "r") as f:
                    if "Server listening" in f.read():
                        ready = True
                        break
            except FileNotFoundError:
                pass
            time.sleep(0.5)

        if not ready:
            self.__exit__(None, None, None)
            raise RuntimeError(f"server failed to come up; see {self.log_path}")

        # final readiness — /v1/models
        for _ in range(60):
            code, body, _ = http_get(f"http://127.0.0.1:{self.port}/v1/models", timeout=3)
            if code == 200 and body and body.get("data"):
                self.model_id = body["data"][0]["id"]
                log(f"server ready, model_id={self.model_id}")
                return self
            time.sleep(0.5)

        self.__exit__(None, None, None)
        raise RuntimeError("server up but /v1/models never returned a model")

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            subprocess.run(["docker", "kill", self.container_name], capture_output=True, timeout=10)
        except Exception:
            pass
        if self.proc is not None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except Exception:
                pass
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    pass
        if hasattr(self, "log_fp") and self.log_fp:
            self.log_fp.close()


# ---------- prompt execution ----------

def _build_doc_for_long_context(target_tokens: int, sentinel: str) -> str:
    """Build a plausible ~target_tokens passage with the sentinel in the middle."""
    para = (
        "The library was old, its shelves leaning under the weight of centuries. "
        "Each volume bore the dust of forgotten readers, their margin notes faded "
        "to whispers. The cataloguer worked through the night, sorting by century, "
        "by binding, by the smell of the leather. When she found a book without a "
        "title page, she set it apart on the long oak table by the window where the "
        "moon could read its spine. "
    )
    # tokens ≈ chars/4 for English
    target_chars = target_tokens * 4
    parts = []
    cur = 0
    while cur < target_chars // 2:
        parts.append(para)
        cur += len(para)
    parts.append(f"\n\n[CRITICAL_NOTE]: The unique sentinel is {sentinel}. "
                 f"Remember it precisely.\n\n")
    cur = sum(len(p) for p in parts)
    while cur < target_chars:
        parts.append(para)
        cur += len(para)
    return "".join(parts)


def _expand_messages(prompt: dict) -> list[dict]:
    if "messages_template" in prompt:
        target = prompt.get("doc_token_target", 1024)
        sentinel = prompt.get("sentinel", "SENTINEL-XXX")
        doc = _build_doc_for_long_context(target, sentinel)
        out = []
        for m in prompt["messages_template"]:
            out.append({"role": m["role"],
                        "content": m["content"].replace("{DOC}", doc)})
        return out
    return prompt["messages"]


def call_chat(model_id: str, prompt: dict, port: int,
              temperature: float = 0.0, seed: int = SEED,
              capture_logprobs: bool = True) -> dict:
    # Floor max_tokens at 256 so reasoning models (Gemma-4, Qwen3.x) have
    # space to finish a <think>...</think> block before emitting the actual
    # answer. The check helpers strip <think> blocks, so the first-N-token
    # checks still apply to the visible answer.
    cfg_max = prompt.get("max_tokens", 256)
    body = {
        "model": model_id,
        "messages": _expand_messages(prompt),
        "max_tokens": max(cfg_max, 256),
        "temperature": temperature,
        "top_p": 1.0,
        "seed": seed,
        "stream": False,
    }
    if capture_logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 20
    t0 = time.time()
    code, resp, raw = http_post(f"http://127.0.0.1:{port}/v1/chat/completions",
                                body, timeout=900)
    dt = time.time() - t0
    if code != 200 or resp is None or "choices" not in resp:
        log(f"  HTTP {code} after {dt:.2f}s body[:500]={raw[:500]!r}")
        return {"error": True, "code": code, "raw": raw[:500], "elapsed_s": dt}
    choice = resp["choices"][0]
    msg = choice.get("message", {})
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or ""
    # imp-server's deepseek-style reasoning parser routes <think>...</think>
    # into reasoning_content; some models (Gemma-4 NVFP4 here) emit tokens
    # entirely inside that block. We always evaluate the union so we test
    # the model's actual output, not the parser split.
    if reasoning and content:
        text = f"<think>{reasoning}</think>\n{content}"
    elif reasoning:
        text = f"<think>{reasoning}</think>"
    else:
        text = content
    finish = choice.get("finish_reason")
    usage = resp.get("usage", {}) or {}
    pt = int(usage.get("prompt_tokens", 0))
    ct = int(usage.get("completion_tokens", 0))
    lp_block = choice.get("logprobs") or {}
    lp_tokens = lp_block.get("content") or []
    return {
        "error": False,
        "text": text,
        "finish_reason": finish,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "elapsed_s": dt,
        "logprob_steps": lp_tokens,
    }


# ---------- check functions ----------

PRIMARY_COLOR_SETS = [
    {"red", "blue", "yellow"},                    # subtractive (paint)
    {"red", "green", "blue"},                     # additive (light)
    {"cyan", "magenta", "yellow"},                # printing (CMY)
]


def _norm_text(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()


def _first_n_tokens(s: str, n: int) -> str:
    # crude word/punct split; good enough for "first 5 tokens" type checks
    toks = re.findall(r"\w+|[^\w\s]", s)
    return " ".join(toks[:n])


def _strip_thinking(s: str) -> str:
    """Drop <think>...</think> blocks before checking output."""
    return re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()


def check_prompt(prompt: dict, text: str, all_runs: list[str] | None = None) -> tuple[bool, str]:
    chk = prompt.get("check", {})
    t = chk.get("type", "")
    body = _strip_thinking(_norm_text(text))
    body_lc = body.lower()

    if t == "contains":
        return (chk["needle"] in body, f"needle present={chk['needle'] in body}")
    if t == "contains_within":
        head = _first_n_tokens(body, chk["first_n_tokens"])
        return (chk["needle"] in head, f"first-{chk['first_n_tokens']}-tokens={head!r}")
    if t == "contains_any":
        for n in chk["needles"]:
            if n in body:
                return True, f"hit={n!r}"
        return False, f"none of {chk['needles']}"
    if t == "contains_any_lc":
        for n in chk["needles"]:
            if n.lower() in body_lc:
                return True, f"hit={n!r}"
        return False, f"none of {chk['needles']}"
    if t == "contains_all":
        missing = [n for n in chk["needles"] if n not in body]
        return (not missing, f"missing={missing}")
    if t == "regex":
        return (re.search(chk["pattern"], body) is not None, f"regex={chk['pattern']}")
    if t == "primary_colors_three_distinct":
        words = re.findall(r"[A-Za-z]+", body_lc)
        for s in PRIMARY_COLOR_SETS:
            hit = [w for w in words if w in s]
            if len(set(hit)) == 3:
                return True, f"matched={s} via {hit}"
        return False, f"no canonical 3-set in words={words[:12]}"
    if t == "json_with_keys":
        try:
            obj = json.loads(_extract_first_json(body))
            for k, v in chk["keys"].items():
                if obj.get(k) != v:
                    return False, f"key {k!r} expected {v!r} got {obj.get(k)!r}"
            return True, "json keys ok"
        except Exception as e:
            return False, f"json parse: {e}"
    if t == "sequence_one_to_n":
        nums = [int(x) for x in re.findall(r"\d+", body)]
        expect = list(range(1, chk["n"] + 1))
        return (nums[: chk["n"]] == expect, f"got={nums[:chk['n']]}")
    if t == "long_generation_health":
        # at least 200 words, no single 8-gram repeated 3+ times
        words = body.split()
        if len(words) < 200:
            return False, f"too short ({len(words)} words)"
        if _has_repeated_8gram(words, 3):
            return False, "8-gram repeated >=3x"
        if _max_consecutive_token_run(words) > 20:
            return False, "single token repeated >20x consecutively"
        return True, f"len={len(words)} words"
    if t == "no_garbage_utf8":
        if "�" in body:
            return False, "U+FFFD in output"
        if any(ord(c) < 0x20 and c not in "\n\t\r" for c in body):
            return False, "control chars present"
        return True, "utf-8 clean"
    if t == "all_runs_identical":
        runs = all_runs or []
        if not runs:
            return False, "no runs collected"
        ok = all(r == runs[0] for r in runs)
        return ok, f"{sum(1 for r in runs if r == runs[0])}/{len(runs)} identical"
    if t == "refusal_or_redirect":
        # positive signals — refusal phrasing
        refuse_signals = ["i can't", "i cannot", "i won't", "i will not",
                          "i'm not able", "i am not able", "i'm sorry",
                          "i am sorry", "unable to", "can't help",
                          "cannot help", "not appropriate", "won't provide",
                          "won't help", "can't assist", "cannot assist",
                          "decline", "against my", "i must decline"]
        # mojibake / NaN / empty are FAILs
        if not body.strip():
            return False, "empty refusal"
        if "�" in body:
            return False, "mojibake in refusal"
        if any(s in body_lc for s in refuse_signals):
            return True, "refusal phrasing present"
        return False, f"no refusal phrasing detected; got {body[:80]!r}"
    if t == "json_with_function_call":
        try:
            obj = json.loads(_extract_first_json(body))
        except Exception as e:
            return False, f"json parse: {e}"
        # accept either {name, arguments} or {function: {name, arguments}}
        cand = obj.get("function", obj)
        if cand.get("name") != chk["name"]:
            return False, f"name expected {chk['name']!r} got {cand.get('name')!r}"
        args = cand.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                return False, "arguments not parseable"
        for k, v in chk["args"].items():
            if args.get(k) != v:
                return False, f"args.{k} expected {v!r} got {args.get(k)!r}"
        return True, "function-call json ok"

    return False, f"unknown check type {t!r}"


def _extract_first_json(s: str) -> str:
    s = s.strip()
    # strip code fences
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # find first {...} balanced span
    depth = 0
    start = -1
    for i, c in enumerate(s):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start != -1:
                return s[start: i + 1]
    return s


def _has_repeated_8gram(words: list[str], min_count: int) -> bool:
    if len(words) < 8:
        return False
    counts = Counter(tuple(words[i:i+8]) for i in range(len(words) - 7))
    return any(c >= min_count for c in counts.values())


def _max_consecutive_token_run(words: list[str]) -> int:
    best = 0
    cur = 1
    for i in range(1, len(words)):
        if words[i] == words[i-1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return max(best, 1) if words else 0


def _ngram_repetition_rate(tokens: list[str], n: int = 4, window: int = 512) -> float:
    if len(tokens) < n:
        return 0.0
    tail = tokens[-window:] if len(tokens) > window else tokens
    grams = [tuple(tail[i:i+n]) for i in range(len(tail) - n + 1)]
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(grams)


# ---------- logprob health ----------

def logprob_health(steps: list[dict]) -> dict:
    if not steps:
        return {"steps": 0, "any_naninf": False, "min_top1": None,
                "max_top1": None, "zero_entropy_pct": 0.0, "ok": True}
    any_nan = False
    top1_vals = []
    zero_entropy = 0
    for st in steps:
        top = st.get("top_logprobs") or []
        if not top:
            continue
        try:
            lps = [float(t["logprob"]) for t in top]
        except (KeyError, TypeError, ValueError):
            any_nan = True
            continue
        if any(math.isnan(x) or math.isinf(x) for x in lps):
            any_nan = True
            continue
        # softmax over top-20 (already log-probs, just exp)
        probs = [math.exp(x) for x in lps]
        s = sum(probs)
        if s == 0:
            any_nan = True
            continue
        probs = [p / s for p in probs]
        top1 = max(probs)
        top1_vals.append(top1)
        ent = -sum(p * math.log(p) for p in probs if p > 0)
        if ent == 0.0:
            zero_entropy += 1
    if not top1_vals:
        return {"steps": len(steps), "any_naninf": any_nan, "min_top1": None,
                "max_top1": None, "zero_entropy_pct": 0.0,
                "ok": not any_nan}
    return {
        "steps": len(steps),
        "any_naninf": any_nan,
        "min_top1": min(top1_vals),
        "max_top1": max(top1_vals),
        "zero_entropy_pct": 100.0 * zero_entropy / len(steps),
        "ok": (not any_nan
               and min(top1_vals) >= 1e-6
               and max(top1_vals) <= 1.0),
    }


# ---------- Phase 0: load + tokenizer roundtrip ----------

def phase0(server: ImpServer, model_dir: Path, report: ModelReport) -> bool:
    # 0a: load already happened by virtue of server being up
    cfg_path = model_dir / "config.json"
    arch = ""
    n_params = 0.0
    try:
        cfg = json.loads(cfg_path.read_text())
        arch = ",".join(cfg.get("architectures", [])) or cfg.get("model_type", "")
        report.config_keys = {k: cfg[k] for k in
                              ("hidden_size", "num_hidden_layers", "num_attention_heads",
                               "num_key_value_heads", "vocab_size", "torch_dtype")
                              if k in cfg}
    except Exception as e:
        log(f"could not parse config.json: {e}")
    report.arch = arch

    # weights inventory
    sft = sorted(model_dir.glob("model*.safetensors"))
    report.weight_files = [f.name for f in sft]
    report.weight_bytes = sum(f.stat().st_size for f in sft)
    # crude param count from bytes/0.5 for FP4 → bytes*2 = parameters
    report.param_count_b = round(report.weight_bytes * 2 / 1e9, 1)

    # 0b: tokenizer roundtrip via the chat endpoint — we ask for an "echo" only on a few
    # short ASCII strings that the server supports. (Real tokenizer roundtrip would
    # need a /tokenize endpoint; imp-server doesn't expose one. We fall back to
    # behavioral check: the model can complete trivially without errors on these strings.)
    fixture = (FIXTURES / "tokenizer_roundtrip.txt").read_text().splitlines()
    sample = [s for s in fixture if s.strip()][:6]
    failures = []
    for s in sample:
        # We probe by asking the server to count tokens via /v1/embeddings if available,
        # but most paths just need confirmation that the prompt prefill doesn't crash.
        body = {
            "model": server.model_id,
            "messages": [{"role": "user", "content": s}],
            "max_tokens": 1,
            "temperature": 0.0,
            "seed": SEED,
        }
        code, _, raw = http_post(f"http://127.0.0.1:{server.port}/v1/chat/completions",
                                 body, timeout=120)
        if code != 200:
            failures.append((s, code, raw[:120]))
    report.phase0 = {
        "weight_files": len(sft),
        "weight_bytes": report.weight_bytes,
        "tokenizer_probe_strings": len(sample),
        "tokenizer_probe_failures": len(failures),
        "tokenizer_probe_failure_examples": failures[:3],
    }
    return not failures


# ---------- Phase 3: graph capture + 32x replay ----------

def phase3_graph_replay(server: ImpServer, report: ModelReport,
                        replays: int = 32) -> bool:
    # Short prompt, low max_tokens, T=0, seed=42, deterministic_gemm=true.
    # We do TWO warmup requests first (engine warmup forward primes cuBLAS,
    # but the first real request still allocates graph-capture resources +
    # the KV high-water mark on FP8/NVFP4 KV caches). The 32 replays must
    # then be byte-identical across the board.
    prompt = {"messages": [{"role": "user",
                             "content": "Reply with a single short sentence about the moon."}],
              "max_tokens": 24}
    warmup_outputs = []
    for _ in range(2):
        r = call_chat(server.model_id, prompt, server.port,
                      temperature=0.0, seed=SEED, capture_logprobs=False)
        if r.get("error"):
            report.phase3 = {"warmup_error": r}
            return False
        warmup_outputs.append(r["text"])
    runs = []
    elapsed = []
    for i in range(replays):
        r = call_chat(server.model_id, prompt, server.port,
                      temperature=0.0, seed=SEED,
                      capture_logprobs=False)
        if r.get("error"):
            report.phase3 = {"replays_attempted": i, "error": r,
                              "warmup_outputs": [w[:160] for w in warmup_outputs]}
            return False
        runs.append(r["text"])
        elapsed.append(r["elapsed_s"])
    identical = sum(1 for r in runs if r == runs[0])
    # Also score: is the first-request output (before warmup converged)
    # visibly degenerate? Heuristic: if it contains a 4+ digit run that
    # the second warmup output and the steady-state output don't, flag it.
    first_req_garbage = (
        bool(re.search(r"\d{4,}", warmup_outputs[0])) and
        not bool(re.search(r"\d{4,}", warmup_outputs[-1]))
    )
    report.phase3 = {
        "replays": replays,
        "identical_to_first": identical,
        "first_output_steady": runs[0][:200],
        "second_output_steady": runs[1][:200] if len(runs) > 1 else "",
        "warmup_request_1_output": warmup_outputs[0][:200],
        "warmup_request_2_output": warmup_outputs[1][:200] if len(warmup_outputs) > 1 else "",
        "first_request_visibly_degenerate": first_req_garbage,
        "median_elapsed_s": sorted(elapsed)[len(elapsed)//2],
    }
    return identical == replays


# ---------- Phase 4 + 5 + 6: battery ----------

def phase456_battery(server: ImpServer, report: ModelReport,
                      battery_path: Path, smoke: bool = False) -> bool:
    bat = json.loads(battery_path.read_text())
    prompts = bat["prompts"]
    if smoke:
        prompts = prompts[:MAX_PROMPTS_SMOKE]

    pass_count = 0
    fail_details = []
    overall_logit_ok = True
    long_gen_rep_rate = None
    ttft_samples: dict[str, float] = {}
    decode_samples: dict[str, float] = {}
    determinism_extra = {}
    server_dead = False
    server_dead_at = None

    vram_before = gpu_mem_used_mb()
    vram_peak = vram_before or 0

    for p in prompts:
        log(f"[{report.name}] prompt {p['id']} {p['name']}")
        repeat = p.get("repeat", 1)
        runs_text = []
        last = None
        if server_dead:
            last = {"error": True, "code": -1,
                    "raw": f"skipped: server crashed at prompt {server_dead_at}",
                    "elapsed_s": 0.0}
            runs_text.append("")
        else:
          for i in range(repeat):
            try:
                r = call_chat(server.model_id, p, server.port,
                              temperature=0.0, seed=SEED + (i if repeat > 1 else 0),
                              capture_logprobs=True)
            except Exception as e:
                log(f"  request raised {type(e).__name__}: {e}")
                r = {"error": True, "code": -2, "raw": f"{type(e).__name__}: {e}",
                     "elapsed_s": 0.0}
            if r.get("error"):
                last = r
                runs_text.append("")
                # if request raised a connection error AFTER having had successful
                # requests in this run, server is likely dead — short-circuit rest
                if r.get("code") in (-2, 0) and pass_count + len(fail_details) > 0:
                    server_dead = True
                    server_dead_at = p["id"]
                break
            runs_text.append(r["text"])
            last = r
            cur_vram = gpu_mem_used_mb()
            if cur_vram and cur_vram > vram_peak:
                vram_peak = cur_vram

        if last is None or last.get("error"):
            ok, why = False, f"server error code={last.get('code') if last else 'n/a'}"
            text = ""
            health = {"ok": False, "error": True}
            elapsed = 0.0
            ttft = None
            decode = None
            ct = 0
            pt = 0
        else:
            text = last["text"]
            ok, why = check_prompt(p, text, all_runs=runs_text)
            health = logprob_health(last["logprob_steps"])
            overall_logit_ok = overall_logit_ok and health["ok"]
            elapsed = last["elapsed_s"]
            ct = last["completion_tokens"]
            pt = last["prompt_tokens"]
            ttft = None
            decode = None
            if ct > 0 and elapsed > 0:
                # rough; without streaming we treat first-token≈prefill_share
                # of total. We split as (prefill / (prefill+decode)) * total.
                # without per-token timing we publish only end-to-end tok/s.
                decode = ct / elapsed
            # additional 5a checks on long generation
            if p.get("check", {}).get("type") == "long_generation_health":
                words = _strip_thinking(text).split()
                long_gen_rep_rate = _ngram_repetition_rate(words, n=4, window=512)
                if long_gen_rep_rate > 0.15:
                    ok, why = False, f"4-gram rep {long_gen_rep_rate:.1%} > 15%"

            # bucket perf samples
            if pt and ct:
                key = f"pt={pt}_ct={ct}"
                if pt not in ttft_samples:
                    ttft_samples[str(pt)] = elapsed  # coarse
                if str(pt) not in decode_samples:
                    decode_samples[str(pt)] = decode or 0

        report.prompts.append(PromptResult(
            id=p["id"], name=p["name"], text=text,
            finish_reason=(last.get("finish_reason") if last else None),
            prompt_tokens=pt, completion_tokens=ct,
            elapsed_s=elapsed, ttft_s=ttft, decode_tok_per_s=decode,
            logprobs_health=health, check_pass=ok, check_reason=why,
            extra={"runs_count": len(runs_text)},
        ))
        if ok:
            pass_count += 1
        else:
            fail_details.append({"id": p["id"], "name": p["name"], "why": why,
                                  "text_head": text[:200]})

    # Phase 5e — determinism: re-run prompt 1 three times with seed=42 → byte-identical
    log(f"[{report.name}] phase 5e: 3x determinism on prompt 1")
    p1 = bat["prompts"][0]
    detrun = []
    if server_dead:
        determinism_extra["det3_pass"] = False
        determinism_extra["det3_outputs"] = ["<server-dead>"] * 3
    else:
        for _ in range(3):
            try:
                r = call_chat(server.model_id, p1, server.port, temperature=0.0,
                              seed=SEED, capture_logprobs=False)
                detrun.append(r.get("text", "<err>") if not r.get("error") else "<err>")
            except Exception as e:
                detrun.append(f"<exc:{type(e).__name__}>")
        determinism_extra["det3_pass"] = (
            len(detrun) == 3 and all(t == detrun[0] for t in detrun)
        )
        determinism_extra["det3_outputs"] = detrun

    report.phase4 = {
        "prompts_total": len(prompts),
        "prompts_passed": pass_count,
        "fail_details": fail_details,
    }
    report.phase5 = {
        "long_gen_4gram_rep_rate": long_gen_rep_rate,
        "logit_health_ok": overall_logit_ok,
        "determinism_3x_byte_identical": determinism_extra["det3_pass"],
        "determinism_outputs_head": [s[:120] for s in determinism_extra["det3_outputs"]],
        "server_died_at_prompt": server_dead_at,
        "phase5c_drift_status": "INCOMPLETE — no BF16 reference (Mode A)",
        "phase5c_ppl_status": "INCOMPLETE — imp-server has no logprobs-of-arbitrary-text endpoint",
    }
    report.phase6 = {
        "vram_used_mb_before": vram_before,
        "vram_used_mb_peak": vram_peak,
        "ttft_s_by_prompt_tokens": ttft_samples,
        "decode_tok_per_s_by_prompt_tokens": decode_samples,
        "ttft_caveat": "non-streaming end-to-end latency; not strict TTFT",
    }
    return (pass_count == len(prompts) and overall_logit_ok
            and determinism_extra["det3_pass"])


# ---------- per-model orchestration ----------

def validate_model(cfg: dict, smoke: bool = False) -> ModelReport:
    rep = ModelReport(name=cfg["name"], path=cfg["host_path"])
    art = ARTIFACTS / cfg["name"]
    art.mkdir(parents=True, exist_ok=True)
    log_path = art / "server.log"

    model_dir = Path(cfg["host_path"])
    if not model_dir.exists():
        rep.verdict = "FAIL"
        rep.failure_phase = "discovery"
        rep.failure_reason = f"model dir does not exist: {model_dir}"
        return rep

    failed_phases = []
    try:
        with ImpServer(cfg, log_path, port=SERVER_PORT, deterministic=True) as srv:
            log(f"[{rep.name}] phase 0")
            if not phase0(srv, model_dir, rep):
                failed_phases.append("0")
            log(f"[{rep.name}] phase 3 (32 graph replays)")
            if not phase3_graph_replay(srv, rep, replays=4 if smoke else 32):
                failed_phases.append("3")
                # Continue into phase 4 even on phase-3 fail so we collect
                # full battery data for the report. The verdict is still FAIL.
            log(f"[{rep.name}] phase 4+5+6 (battery)")
            try:
                ok = phase456_battery(srv, rep,
                                       FIXTURES / "battery_prompts.json", smoke=smoke)
                if not ok:
                    failed_phases.append("4_or_5")
            except Exception as e:
                log(f"  phase 4/5 raised {type(e).__name__}: {e}")
                failed_phases.append("4_or_5_exception")
                rep.phase4 = rep.phase4 or {"error": f"{type(e).__name__}: {e}"}

            if failed_phases:
                rep.verdict = "FAIL"
                rep.failure_phase = "+".join(failed_phases)
                p4 = rep.phase4 or {}
                p5 = rep.phase5 or {}
                rep.failure_reason = (
                    f"battery passed {p4.get('prompts_passed', 0)}/{p4.get('prompts_total', 0)}; "
                    f"logit_health_ok={p5.get('logit_health_ok')}; "
                    f"det3={p5.get('determinism_3x_byte_identical')}; "
                    f"graph_replay={rep.phase3.get('identical_to_first', 0)}/{rep.phase3.get('replays', 0)}"
                )
            else:
                rep.verdict = "PASS"
            return rep
    except Exception as e:
        log(f"[{rep.name}] EXCEPTION: {e}")
        rep.verdict = "FAIL"
        rep.failure_phase = "infrastructure"
        rep.failure_reason = f"{type(e).__name__}: {e}"
        return rep


# ---------- report writer ----------

def write_report(reports: list[ModelReport]) -> None:
    md = REPO_ROOT / "MODEL_VALIDATION_REPORT.md"
    csv_p = REPO_ROOT / "MODEL_VALIDATION_SUMMARY.csv"

    lines = [
        "# Model Validation Report",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_  ",
        "_Mode: A (reduced scope — no BF16 reference, no NVFP4 calibration)_  ",
        "_Engine: imp (sm_120f), CUDA 13.2, deterministic_gemm=true, CUDA Graphs=on_  ",
        "",
        "Phase legend: 0=load+tokenizer, 3=graph replay (32x byte-identical), "
        "4=20-prompt battery (T=0 seed=42), 5=degeneracy gates, 6=perf smoke. "
        "Phases 1, 2, 5c are out of scope (see report header).",
        "",
        "## Verdicts",
        "",
        "| Model | Verdict | Failure phase | Failure reason |",
        "|---|---|---|---|",
    ]
    for r in reports:
        lines.append(f"| `{r.name}` | **{r.verdict}** | {r.failure_phase} | {r.failure_reason} |")
    lines += ["", "---", ""]

    for r in reports:
        lines.append(f"## {r.name}")
        lines.append(f"**Verdict:** {r.verdict}  ")
        lines.append(f"**Failure phase:** {r.failure_phase}  ")
        lines.append(f"**Failure reason:** {r.failure_reason}  ")
        lines.append("")
        lines.append("### Config")
        lines.append(f"- Path: `{r.path}`")
        lines.append(f"- Arch: `{r.arch}`")
        lines.append(f"- Param count (≈): {r.param_count_b}B")
        lines.append(f"- Weight files: {len(r.weight_files)} ({r.weight_bytes/1e9:.2f} GB)")
        lines.append(f"- Config keys: `{r.config_keys}`")
        lines.append("")
        lines.append("### Phase 0 (load + tokenizer probe)")
        lines.append(f"- {json.dumps(r.phase0, indent=2)}")
        lines.append("")
        lines.append("### Phase 3 (CUDA Graph 32x replay)")
        lines.append(f"```json\n{json.dumps(r.phase3, indent=2)}\n```")
        lines.append("")
        lines.append("### Phase 4 (battery)")
        lines.append(f"- Passed: {r.phase4.get('prompts_passed', 0)} / {r.phase4.get('prompts_total', 0)}")
        if r.phase4.get("fail_details"):
            lines.append("- Failures:")
            for f in r.phase4["fail_details"]:
                lines.append(f"  - prompt {f['id']} `{f['name']}`: {f['why']}")
                lines.append(f"    output head: `{f['text_head']!r}`")
        lines.append("")
        lines.append("### Phase 5 (degeneracy)")
        lines.append(f"```json\n{json.dumps(r.phase5, indent=2)}\n```")
        lines.append("")
        lines.append("### Phase 6 (perf smoke)")
        lines.append(f"```json\n{json.dumps(r.phase6, indent=2)}\n```")
        lines.append("")
        lines.append("### Per-prompt detail")
        lines.append("| # | name | check | tokens (in/out) | elapsed (s) | logits ok |")
        lines.append("|---|---|---|---|---|---|")
        for p in r.prompts:
            ok = "✅" if p.check_pass else "❌"
            lh = "✅" if p.logprobs_health.get("ok") else "❌"
            lines.append(
                f"| {p.id} | {p.name} | {ok} {p.check_reason} | "
                f"{p.prompt_tokens}/{p.completion_tokens} | {p.elapsed_s:.2f} | {lh} |"
            )
        lines.append("")
        lines.append("---")
        lines.append("")

    md.write_text("\n".join(lines))

    # CSV
    with csv_p.open("w") as f:
        f.write("model,verdict,failure_phase,failure_reason,arch,param_b,"
                "weight_gb,phase4_passed,phase4_total,det3_ok,logit_ok,"
                "graph_replay_identical,vram_peak_mb\n")
        for r in reports:
            f.write(",".join([
                r.name, r.verdict, r.failure_phase,
                f'"{r.failure_reason.replace(chr(34), chr(39))}"',
                r.arch, str(r.param_count_b),
                f"{r.weight_bytes/1e9:.2f}",
                str(r.phase4.get("prompts_passed", 0)),
                str(r.phase4.get("prompts_total", 0)),
                str(r.phase5.get("determinism_3x_byte_identical", False)),
                str(r.phase5.get("logit_health_ok", False)),
                f"{r.phase3.get('identical_to_first', 0)}/{r.phase3.get('replays', 0)}",
                str(r.phase6.get("vram_used_mb_peak", 0)),
            ]) + "\n")

    log(f"wrote {md}")
    log(f"wrote {csv_p}")


# ---------- entry ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=None,
                    help="Restrict to one or more model names (repeatable).")
    ap.add_argument("--smoke", action="store_true",
                    help="Run only first 4 prompts and 4 graph replays.")
    args = ap.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    selected = MODELS
    if args.model:
        wanted = set(args.model)
        selected = [m for m in MODELS if m["name"] in wanted]
        if not selected:
            print(f"No model matched {sorted(wanted)}; available: "
                  f"{[m['name'] for m in MODELS]}", file=sys.stderr)
            sys.exit(2)

    reports: list[ModelReport] = []
    for m in selected:
        log(f"==== {m['name']} ====")
        rep = validate_model(m, smoke=args.smoke)
        reports.append(rep)
        # write incrementally so we can see progress
        write_report(reports)
        # also dump per-model JSON
        out = ARTIFACTS / m["name"] / "report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(rep), indent=2, default=str))

    n_pass = sum(1 for r in reports if r.verdict == "PASS")
    log(f"DONE — {n_pass}/{len(reports)} PASS")


if __name__ == "__main__":
    main()
