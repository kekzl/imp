#!/usr/bin/env python3
# serving_kpi.py - serving KPI sweep against an OpenAI-compatible server
# (imp-server; vLLM accepts the same requests, its /metrics differ).
#
# Per concurrency level (closed loop: C workers, each sends its next request
# when the previous one finished):
#   latency    TTFT, TPOT (per request), ITL (per token), E2E, normalized
#              latency (E2E / output tokens), each p50 / p95 / p99
#   throughput req/s, output / input / total tok/s over the level's wall
#   goodput    requests meeting BOTH SLOs (TTFT <= --slo-ttft-ms and
#              TPOT <= --slo-tpot-ms), as req/s, tok/s and attainment %
#   server     /metrics deltas across the level: queue wait p50/p95/p99 from
#              imp_queue_time_seconds, decode rows per step, prefix-cache hit
#              rate, speculative acceptance, KV-pressure rejections,
#              StreamingLLM auto-enables, prefix-cache evictions; sampled
#              imp_kv_blocks_live / imp_kv_blocks_total and
#              imp_decode_batch_last_rows (mean, max)
#   power      nvidia-smi power.draw integrated over the level: mean W,
#              J per 1k output tokens, mean SM clock (the host-health check of
#              docs/internals/BENCHMARKING.md)
#
# Stdlib only, runs on the host like the other tools/analysis clients. Prompts
# are unique per request (a header token plus --prompt-tokens of filler), so
# the prefix cache does not turn the sweep into a cache benchmark.
#
# Usage:
#   serving_kpi.py --url http://127.0.0.1:8080 --levels 1,8,32 --max-tokens 300
#       [--requests-per-level N (default max(32, 2*C))] [--prompt-tokens 0]
#       [--slo-ttft-ms 500] [--slo-tpot-ms 50] [--ignore-eos] [--endpoint chat|completions]
#       [--no-power] [--json FILE] [--md-out FILE] [--tag x]
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

FILLER = [
    "A translation lookaside buffer caches recent virtual-to-physical page mappings so most loads skip the page walk.",
    "Set-associative caches index a set by address bits and compare the tag against every way in that set.",
    "Write-back caches mark a line dirty and defer the memory write until the line is evicted.",
    "A hardware prefetcher watches the stream of misses and issues loads for the lines it expects next.",
    "Out-of-order cores track dependencies with a reorder buffer and commit results in program order.",
    "Branch predictors keep global and local histories and combine them to guess the next instruction address.",
    "Cache coherence protocols such as MESI keep one writer or many readers per line across cores.",
    "Huge pages cut TLB pressure by mapping two megabytes or a gigabyte with one entry.",
]
QUESTIONS = [
    "Explain how a CPU cache works in three sentences.",
    "Describe the water cycle briefly.",
    "Summarize how photosynthesis works.",
    "Give a short definition of entropy in physics.",
    "Explain recursion with a tiny example.",
    "What is the capital of Japan and one fact about it?",
    "List five prime numbers and explain what makes them prime.",
    "Write a short paragraph about the ocean.",
]


# ---------------------------------------------------------------- statistics

def pct(xs, p):
    """Percentile with linear interpolation (numpy default). p in [0, 100]."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def hist_quantile(buckets, q):
    """Prometheus-style quantile from cumulative {le: count} buckets (deltas).

    Linear interpolation inside the bucket that crosses the rank, like
    histogram_quantile(); the +Inf bucket resolves to its lower bound.
    Returns nan when the histogram is empty.
    """
    items = sorted(((float(le), float(c)) for le, c in buckets.items()), key=lambda t: t[0])
    if not items or items[-1][1] <= 0:
        return float("nan")
    total = items[-1][1]
    rank = q * total
    prev_le, prev_c = 0.0, 0.0
    for le, c in items:
        if c >= rank:
            if math.isinf(le):
                return prev_le
            if c == prev_c:
                return le
            return prev_le + (le - prev_le) * (rank - prev_c) / (c - prev_c)
        prev_le, prev_c = le, c
    return items[-1][0]


def summarize_level(records, wall_s, slo_ttft_ms, slo_tpot_ms):
    """records: list of dicts {ok, t0, t_first, t_end, stamps, prompt_tokens,
    completion_tokens, cached_tokens}. Times in seconds (perf_counter)."""
    ok = [r for r in records if r["ok"]]
    err = len(records) - len(ok)
    ttft, tpot, itl, e2e, norm = [], [], [], [], []
    met, met_tokens, out_tokens, in_tokens, cached = 0, 0, 0, 0, 0
    for r in ok:
        n_out = r["completion_tokens"]
        out_tokens += n_out
        in_tokens += r["prompt_tokens"]
        cached += r.get("cached_tokens", 0)
        if r["t_first"] is None or n_out <= 0:
            continue
        t_ttft = (r["t_first"] - r["t0"]) * 1e3
        t_e2e = r["t_end"] - r["t0"]
        ttft.append(t_ttft)
        e2e.append(t_e2e)
        norm.append(t_e2e * 1e3 / n_out)
        st = r["stamps"]
        itl.extend((b - a) * 1e3 for a, b in zip(st, st[1:]))
        if n_out >= 2:
            t_tpot = (r["t_end"] - r["t_first"]) * 1e3 / (n_out - 1)
            tpot.append(t_tpot)
        else:
            t_tpot = 0.0
        if t_ttft <= slo_ttft_ms and t_tpot <= slo_tpot_ms:
            met += 1
            met_tokens += n_out

    def trio(xs):
        return {"p50": pct(xs, 50), "p95": pct(xs, 95), "p99": pct(xs, 99), "n": len(xs)}

    w = wall_s if wall_s > 0 else float("nan")
    return {
        "requests": len(records), "ok": len(ok), "err": err, "wall_s": wall_s,
        "req_s": len(ok) / w,
        "output_tok_s": out_tokens / w, "input_tok_s": in_tokens / w,
        "total_tok_s": (out_tokens + in_tokens) / w,
        "output_tokens": out_tokens, "input_tokens": in_tokens, "cached_tokens": cached,
        "ttft_ms": trio(ttft), "tpot_ms": trio(tpot), "itl_ms": trio(itl),
        "e2e_s": trio(e2e), "norm_ms_per_tok": trio(norm),
        "goodput_req_s": met / w, "goodput_tok_s": met_tokens / w,
        "slo_attainment_pct": (100.0 * met / len(ttft)) if ttft else float("nan"),
        "slo": {"ttft_ms": slo_ttft_ms, "tpot_ms": slo_tpot_ms},
    }


# ---------------------------------------------------------------- /metrics

_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(-?[0-9.eE+\-]+|NaN|\+Inf)$')


def parse_metrics(text):
    """{name: value} for unlabeled series, {name: {le: count}} for _bucket series."""
    plain, hists = {}, {}
    for line in text.splitlines():
        m = _LINE.match(line.strip())
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2), m.group(3)
        v = float("inf") if val == "+Inf" else float(val)
        if labels and name.endswith("_bucket"):
            le = re.search(r'le="([^"]+)"', labels)
            if le:
                hists.setdefault(name[:-len("_bucket")], {})[le.group(1)] = v
        elif not labels:
            plain[name] = v
    return plain, hists


def scrape(url):
    try:
        with urllib.request.urlopen(url + "/metrics", timeout=5) as r:
            return parse_metrics(r.read().decode("utf-8", "ignore"))
    except Exception as e:  # noqa: BLE001
        print(f"WARN /metrics: {e}", file=sys.stderr)
        return {}, {}


def hist_delta(before, after, name):
    a, b = after.get(name, {}), before.get(name, {})
    return {le: a[le] - b.get(le, 0.0) for le in a}


def delta(before, after, name):
    if name not in after:
        return float("nan")
    return after[name] - before.get(name, 0.0)


def server_summary(m0, h0, m1, h1, samples):
    q = hist_delta(h0, h1, "imp_queue_time_seconds")
    steps = delta(m0, m1, "imp_decode_batch_steps_total")
    rows = delta(m0, m1, "imp_decode_batch_rows_total")
    prompt = delta(m0, m1, "imp_tokens_prompt_total")
    cached = delta(m0, m1, "imp_tokens_cached_total")
    drafted = delta(m0, m1, "imp_spec_drafted_total")
    accepted = delta(m0, m1, "imp_spec_accepted_total")
    kv_util = [s["kv_util_pct"] for s in samples if s.get("kv_util_pct") is not None]
    active = [s["active"] for s in samples if s.get("active") is not None]
    return {
        "queue_ms": {"p50": hist_quantile(q, 0.5) * 1e3, "p95": hist_quantile(q, 0.95) * 1e3,
                     "p99": hist_quantile(q, 0.99) * 1e3, "n": q.get("+Inf", float("nan"))},
        "rows_per_step": rows / steps if steps and steps > 0 else float("nan"),
        "decode_steps": steps,
        "active_seqs_mean": (sum(active) / len(active)) if active else float("nan"),
        "active_seqs_max": max(active) if active else float("nan"),
        "kv_util_mean_pct": (sum(kv_util) / len(kv_util)) if kv_util else float("nan"),
        "kv_util_max_pct": max(kv_util) if kv_util else float("nan"),
        "prefix_hit_pct": (100.0 * cached / prompt) if prompt and prompt > 0 else float("nan"),
        "spec_accept_pct": (100.0 * accepted / drafted) if drafted and drafted > 0 else float("nan"),
        "spec_drafted": drafted,
        "kv_pressure_rejections": delta(m0, m1, "imp_kv_pressure_rejections_total"),
        "streaming_kv_auto_enables": delta(m0, m1, "imp_streaming_kv_auto_enables_total"),
        "prefix_cache_evictions": delta(m0, m1, "imp_prefix_cache_evictions_total"),
        "requests_rejected": delta(m0, m1, "imp_requests_rejected_total"),
        "requests_timed_out": delta(m0, m1, "imp_requests_timed_out_total"),
    }


class Sampler(threading.Thread):
    """Polls /metrics gauges and nvidia-smi power while a level runs."""

    def __init__(self, url, power, period=0.25):
        super().__init__(daemon=True)
        self.url, self.power, self.period = url, power, period
        self.samples, self.power_samples = [], []  # power: (t, watts, sm_mhz)
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            t = time.perf_counter()
            m, _ = scrape(self.url)
            tot, live = m.get("imp_kv_blocks_total"), m.get("imp_kv_blocks_live")
            self.samples.append({
                "t": t,
                "kv_util_pct": (100.0 * live / tot) if tot and live is not None else None,
                "active": m.get("imp_decode_batch_last_rows"),
            })
            if self.power:
                try:
                    out = subprocess.run(
                        ["nvidia-smi", "--query-gpu=power.draw,clocks.sm",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=2).stdout.strip().split(",")
                    self.power_samples.append((t, float(out[0]), float(out[1])))
                except Exception:  # noqa: BLE001
                    pass
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()
        self.join()

    def energy(self, t_lo, t_hi):
        pts = [p for p in self.power_samples if t_lo <= p[0] <= t_hi]
        if len(pts) < 2:
            return {"joules": float("nan"), "power_mean_w": float("nan"), "sm_mhz_mean": float("nan")}
        joules = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(pts, pts[1:]))
        return {"joules": joules,
                "power_mean_w": sum(p[1] for p in pts) / len(pts),
                "sm_mhz_mean": sum(p[2] for p in pts) / len(pts)}


# ---------------------------------------------------------------- client

def make_prompt(tag, level, i, prompt_tokens):
    head = f"[{tag}-c{level}-r{i}] "
    fill = ""
    if prompt_tokens > 0:
        n = max(1, prompt_tokens // 20)
        fill = " ".join(FILLER[(i + k) % len(FILLER)] for k in range(n)) + " "
    return head + fill + QUESTIONS[i % len(QUESTIONS)]


def one_request(args, level, i):
    prompt = make_prompt(args.tag, level, i, args.prompt_tokens)
    extra = {"ignore_eos": True} if args.ignore_eos else {}
    if args.endpoint == "chat":
        path = "/v1/chat/completions"
        body = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens, "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True}, **extra}
    else:
        path = "/v1/completions"
        body = {"model": args.model, "prompt": prompt, "max_tokens": args.max_tokens,
                "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
                **extra}
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = "Bearer " + args.api_key
    req = urllib.request.Request(args.url + path, data=json.dumps(body).encode(), headers=headers)
    rec = {"ok": False, "t0": time.perf_counter(), "t_first": None, "t_end": None, "stamps": [],
           "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "err": None}
    usage = {}
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    j = json.loads(payload)
                except ValueError:
                    continue
                if j.get("usage"):
                    usage = j["usage"]
                ch = j.get("choices") or []
                if not ch:
                    continue
                c0 = ch[0]
                d = c0.get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or c0.get("text"):
                    now = time.perf_counter()
                    if rec["t_first"] is None:
                        rec["t_first"] = now
                    rec["stamps"].append(now)
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001
        rec["err"] = str(e)
    rec["t_end"] = time.perf_counter()
    rec["prompt_tokens"] = int(usage.get("prompt_tokens", 0))
    rec["completion_tokens"] = int(usage.get("completion_tokens", len(rec["stamps"])))
    rec["cached_tokens"] = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
    return rec


def run_level(args, level, n_requests, sampler):
    records = [None] * n_requests
    next_i = [0]
    lock = threading.Lock()

    def worker():
        while True:
            with lock:
                i = next_i[0]
                next_i[0] += 1
            if i >= n_requests:
                return
            records[i] = one_request(args, level, i)

    m0, h0 = scrape(args.url)
    n_before = len(sampler.samples)
    t_lo = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(level)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t_hi = time.perf_counter()
    m1, h1 = scrape(args.url)
    client = summarize_level(records, t_hi - t_lo, args.slo_ttft_ms, args.slo_tpot_ms)
    server = server_summary(m0, h0, m1, h1, sampler.samples[n_before:])
    power = sampler.energy(t_lo, t_hi)
    out_tok = client["output_tokens"]
    power["j_per_1k_output_tok"] = (power["joules"] / (out_tok / 1000.0)) if out_tok else float("nan")
    errs = [r["err"] for r in records if r and r["err"]]
    return {"concurrency": level, "client": client, "server": server, "power": power,
            "sample_err": errs[0] if errs else None}


# ---------------------------------------------------------------- report

def fmt(v, nd=1):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def markdown(results, args):
    rows = []

    def add(label, fn, nd=1):
        rows.append((label, [fmt(fn(r), nd) for r in results]))

    def trio(sec, key, nd=1):
        return lambda r: " / ".join(fmt(r[sec][key][p], nd) for p in ("p50", "p95", "p99"))

    add("requests ok / err", lambda r: f"{r['client']['ok']} / {r['client']['err']}")
    add("wall s", lambda r: r["client"]["wall_s"], 2)
    add("req/s", lambda r: r["client"]["req_s"], 2)
    add("output tok/s", lambda r: r["client"]["output_tok_s"])
    add("input tok/s", lambda r: r["client"]["input_tok_s"])
    add("total tok/s", lambda r: r["client"]["total_tok_s"])
    add("TTFT p50 / p95 / p99 ms", trio("client", "ttft_ms", 0))
    add("TPOT p50 / p95 / p99 ms", trio("client", "tpot_ms", 1))
    add("ITL p50 / p95 / p99 ms", trio("client", "itl_ms", 1))
    add("E2E p50 / p95 / p99 s", trio("client", "e2e_s", 2))
    add("normalized p50 / p95 / p99 ms/tok", trio("client", "norm_ms_per_tok", 1))
    add(f"goodput req/s (TTFT<={args.slo_ttft_ms:.0f} ms, TPOT<={args.slo_tpot_ms:.0f} ms)",
        lambda r: r["client"]["goodput_req_s"], 2)
    add("goodput tok/s", lambda r: r["client"]["goodput_tok_s"])
    add("SLO attainment %", lambda r: r["client"]["slo_attainment_pct"])
    add("queue p50 / p95 / p99 ms (server)", trio("server", "queue_ms", 1))
    add("decode rows / step", lambda r: r["server"]["rows_per_step"], 2)
    add("active seqs mean / max (sampled)",
        lambda r: f"{fmt(r['server']['active_seqs_mean'])} / {fmt(r['server']['active_seqs_max'], 0)}")
    add("KV live mean / max %",
        lambda r: f"{fmt(r['server']['kv_util_mean_pct'])} / {fmt(r['server']['kv_util_max_pct'])}")
    add("prefix-cache hit %", lambda r: r["server"]["prefix_hit_pct"])
    add("spec accept % (drafted)",
        lambda r: f"{fmt(r['server']['spec_accept_pct'])} ({fmt(r['server']['spec_drafted'], 0)})")
    add("KV rejections / StreamingLLM auto / prefix evictions",
        lambda r: " / ".join(fmt(r["server"][k], 0) for k in
                             ("kv_pressure_rejections", "streaming_kv_auto_enables",
                              "prefix_cache_evictions")))
    add("power mean W", lambda r: r["power"]["power_mean_w"], 0)
    add("J per 1k output tok", lambda r: r["power"]["j_per_1k_output_tok"])
    add("SM clock mean MHz", lambda r: r["power"]["sm_mhz_mean"], 0)
    head = "| KPI | " + " | ".join(f"c={r['concurrency']}" for r in results) + " |"
    sep = "|---|" + "|".join("---" for _ in results) + "|"
    lines = [head, sep] + [f"| {label} | " + " | ".join(cells) + " |" for label, cells in rows]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--model", default=os.environ.get("MODEL_NAME", "x"))
    ap.add_argument("--endpoint", choices=("chat", "completions"), default="chat")
    ap.add_argument("--levels", default="1,8,32")
    ap.add_argument("--requests-per-level", type=int, default=0, help="default max(32, 2*C)")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--prompt-tokens", type=int, default=0, help="filler tokens per prompt")
    ap.add_argument("--ignore-eos", action="store_true", help="every request runs to --max-tokens")
    ap.add_argument("--slo-ttft-ms", type=float, default=500.0)
    ap.add_argument("--slo-tpot-ms", type=float, default=50.0)
    ap.add_argument("--warmup", type=int, default=1, help="warmup waves at the largest level (64 tokens)")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--no-power", action="store_true")
    ap.add_argument("--tag", default=f"kpi{int(time.time()) % 100000}")
    ap.add_argument("--json", default="")
    ap.add_argument("--md-out", default="", help="write the markdown table to this file")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x]
    power = not args.no_power and shutil.which("nvidia-smi") is not None
    sampler = Sampler(args.url, power)
    sampler.start()

    if args.warmup > 0:
        c = max(levels)
        saved = args.max_tokens
        args.max_tokens = 64
        for _ in range(args.warmup):
            run_level(args, c, c, sampler)
        args.max_tokens = saved
        print(f"warmup: {args.warmup} wave(s) at c={c}", file=sys.stderr)

    results = []
    for c in levels:
        n = args.requests_per_level or max(32, 2 * c)
        r = run_level(args, c, n, sampler)
        results.append(r)
        cl = r["client"]
        print(f"c={c}: {cl['ok']}/{cl['requests']} ok, {cl['output_tok_s']:.1f} out tok/s, "
              f"{cl['req_s']:.2f} req/s, TTFT p50/p95/p99 {fmt(cl['ttft_ms']['p50'], 0)}/"
              f"{fmt(cl['ttft_ms']['p95'], 0)}/{fmt(cl['ttft_ms']['p99'], 0)} ms, "
              f"TPOT p50/p99 {fmt(cl['tpot_ms']['p50'])}/{fmt(cl['tpot_ms']['p99'])} ms, "
              f"goodput {cl['goodput_req_s']:.2f} req/s ({fmt(cl['slo_attainment_pct'], 0)} %)",
              file=sys.stderr, flush=True)
        if r["sample_err"]:
            print(f"  sample error: {r['sample_err']}", file=sys.stderr)
        time.sleep(2)
    sampler.stop()

    md = markdown(results, args)
    print(md)
    if args.md_out:
        with open(args.md_out, "w") as f:
            f.write(md + "\n")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
