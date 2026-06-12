#!/usr/bin/env python3
"""Phase-0 VRAM-audit load driver.

Drives the fixed audit workload against a running imp-server:
  Qwen3-Coder-30B-A3B NVFP4, ctx=4096, continuous batching, 8 concurrent reqs.

The server is started separately with the VRAM-audit harness enabled
(`--max-batch 8 --set runtime.max_seq_len=4096 --set diagnostics.vram_audit=true`).
This script warms the GPU clocks (>1s ramp on the 5090) and then sustains 8
concurrent streaming completions with long prompts so the scheduler batches a
full 8-way decode and the KV pool fills toward ctx=4096 per sequence. The
server's MemAccount peak sampler captures the device-used peak across the run;
the per-component table is emitted to the server log + dump file on shutdown.

Usage:
  python3 vram_audit_load.py --url http://127.0.0.1:8080 --concurrency 8 \
      --ctx 4096 --rounds 3
"""
import argparse
import json
import threading
import time
import urllib.request

# A chunk of code-like filler (~1 token/word). Repeated to reach the target
# prompt length so each sequence's KV approaches ctx without needing a corpus.
FILLER = (
    "def transform(node, ctx): "
    "for child in node.children: "
    "ctx.visit(child); accumulate(child.value, ctx.state); "
    "return reduce(lambda a, b: a + b, ctx.state, init_value) "
)


def build_prompt(approx_tokens: int) -> str:
    reps = max(1, approx_tokens // 24)
    body = FILLER * reps
    return (
        "You are a code assistant. Carefully analyze the following program and "
        "then continue implementing it in detail, explaining each step:\n\n"
        + body
        + "\n\nNow continue the implementation step by step with full detail:"
    )


def get_model_id(url):
    """Strict API (PR #507): the request model name must equal the loaded
    model's name or the server returns 404. Fetch it from /v1/models."""
    try:
        with urllib.request.urlopen(url + "/v1/models", timeout=30) as resp:
            data = json.loads(resp.read())
            return data["data"][0]["id"]
    except Exception as e:  # noqa: BLE001
        print(f"  could not fetch model id ({e}); falling back to 'default'")
        return "default"


def one_request(url, prompt, max_tokens, idx, stats, model_id):
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions", data=data,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    n_tok = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if line.startswith("data: ") and "[DONE]" not in line:
                    n_tok += 1
    except Exception as e:  # noqa: BLE001
        stats["errors"] += 1
        print(f"  [req {idx}] error: {e}")
        return
    dt = time.time() - t0
    stats["tokens"] += n_tok
    stats["reqs"] += 1
    print(f"  [req {idx}] {n_tok} chunks in {dt:.1f}s ({n_tok / dt:.1f} tok/s)")


def fire_round(url, concurrency, prompt, max_tokens, stats, tag, model_id):
    print(f"[{tag}] firing {concurrency} concurrent requests...")
    threads = []
    for i in range(concurrency):
        t = threading.Thread(target=one_request,
                             args=(url, prompt, max_tokens, i, stats, model_id))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    # Prompt ~60% of ctx; generation fills the rest toward ctx per sequence.
    prompt = build_prompt(int(args.ctx * 0.6))
    gen = int(args.ctx * 0.4)

    model_id = get_model_id(args.url)
    print(f"model id: {model_id}")
    stats = {"tokens": 0, "reqs": 0, "errors": 0}

    # Warmup: ramp the 5090 clocks (>1s under load) with a discarded round.
    fire_round(args.url, args.concurrency, build_prompt(256), 128, stats, "warmup", model_id)
    time.sleep(1.0)
    stats = {"tokens": 0, "reqs": 0, "errors": 0}

    t0 = time.time()
    for r in range(args.rounds):
        fire_round(args.url, args.concurrency, prompt, gen, stats, f"round {r + 1}/{args.rounds}", model_id)
    dt = time.time() - t0

    print("\n=== load summary ===")
    print(f"requests: {stats['reqs']}  errors: {stats['errors']}")
    print(f"chunks:   {stats['tokens']}  wall: {dt:.1f}s")
    print(f"aggregate: {stats['tokens'] / dt:.1f} tok/s across {args.concurrency} streams")


if __name__ == "__main__":
    main()
