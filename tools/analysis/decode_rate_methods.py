#!/usr/bin/env python3
# Decode-rate definitions on one streaming run: same 6 prompts as seq_stream_client.py, per request
# Usage (imp:toolchain, --network host, -v tools/analysis:/t): decode_rate_methods.py PORT
# A all tokens / time from the first delta of any kind (seq_stream_client.py),
# B all tokens / time from the first `content` delta (answer only, thinking excluded from the clock),
# C content-only deltas / time from the first `content` delta.
import json, sys, time, urllib.request, statistics
import importlib.util
spec = importlib.util.spec_from_file_location("ssc", "/t/seq_stream_client.py")
PORT = int(sys.argv[1])
sys.argv = ["x", str(PORT), "m"]
ssc = importlib.util.module_from_spec(spec); spec.loader.exec_module(ssc)
rows = []
for name, prompt in ssc.PROMPTS:
    body = {"model": ssc.MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 400,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t_any = t_content = t_last = None; n_content = 0; usage = None; n_reason = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                now = time.perf_counter()
                if d.get("reasoning_content"):
                    t_any = t_any or now; t_last = now; n_reason += 1
                if d.get("content"):
                    t_any = t_any or now; t_content = t_content or now; t_last = now; n_content += 1
    n = usage["completion_tokens"]
    a = (n - 1) / (t_last - t_any)
    b = (n - 1) / (t_last - t_content) if t_content and t_last > t_content else float("nan")
    c = (n_content - 1) / (t_last - t_content) if t_content and t_last > t_content else float("nan")
    rows.append((a, b, c))
    print(f"{name:8s} tokens {n} reasoning_deltas {n_reason} content_deltas {n_content}  A {a:7.2f}  B {b:7.2f}  C {c:7.2f}", flush=True)
for i, k in enumerate("ABC"):
    print(f"MEDIAN {k} {statistics.median(r[i] for r in rows):.2f}  MEAN {k} {statistics.mean(r[i] for r in rows):.2f}")
