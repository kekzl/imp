#!/usr/bin/env python3
"""Differential probe: real imp-server vs the mock CI actually tests.

`Mock API contract` is the only CI job that touches the HTTP API, and it runs
`tests/api` against `tests/api/mock_server.py` with IMP_USE_MOCK=1 — a Python
reimplementation of the endpoints. `tools/imp-server/` is never executed there
(#1302). So every assertion in that suite describes the *mock's* behaviour, and
nothing checks that the shipping server agrees.

This sends the same table of edge-case requests to both and reports where they
diverge. A divergence is not automatically a bug in either one — it is a place
where a green CI run says nothing about the server.

Usage:
  api_diff.py --real http://127.0.0.1:8099 --model <id>      # starts the mock itself
"""
import argparse
import json
import sys
import threading

import httpx

sys.path.insert(0, 'tests/api')


def cases(model):
    """(name, method, path, body_or_raw) — body None means send raw text."""
    msg = [{"role": "user", "content": "Hi"}]
    base = {"model": model, "messages": msg, "max_tokens": 1}

    def b(**kw):
        d = dict(base)
        d.update(kw)
        return d

    out = [
        ("baseline", b()),
        ("max_tokens=0", b(max_tokens=0)),
        ("max_tokens=-1", b(max_tokens=-1)),
        ("max_tokens=10^9", b(max_tokens=10**9)),
        ("max_tokens=null", b(max_tokens=None)),
        ("max_tokens missing", {"model": model, "messages": msg}),
        ("max_tokens as string", b(max_tokens="10")),
        ("temperature=-0.1", b(temperature=-0.1)),
        ("temperature=0", b(temperature=0)),
        ("temperature=2", b(temperature=2.0)),
        ("temperature=2.5", b(temperature=2.5)),
        ("temperature as string", b(temperature="hot")),
        ("top_p=-0.1", b(top_p=-0.1)),
        ("top_p=0", b(top_p=0)),
        ("top_p=1.5", b(top_p=1.5)),
        ("top_k=0", b(top_k=0)),
        ("top_k=-1", b(top_k=-1)),
        ("n=2", b(n=2)),
        ("messages missing", {"model": model, "max_tokens": 1}),
        ("messages not array", {"model": model, "messages": "nope", "max_tokens": 1}),
        ("messages empty", {"model": model, "messages": [], "max_tokens": 1}),
        ("message empty content", b(messages=[{"role": "user", "content": ""}])),
        ("message whitespace content", b(messages=[{"role": "user", "content": "   "}])),
        ("message null content", b(messages=[{"role": "user", "content": None}])),
        ("message no role", b(messages=[{"content": "Hi"}])),
        ("message unknown role", b(messages=[{"role": "wizard", "content": "Hi"}])),
        ("system only", b(messages=[{"role": "system", "content": "Be terse"}])),
        ("two consecutive user turns",
         b(messages=[{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])),
        ("assistant first", b(messages=[{"role": "assistant", "content": "hi"}])),
        ("model missing", {"messages": msg, "max_tokens": 1}),
        ("model unknown", b(model="no-such-model")),
        ("unknown field", b(frobnicate=True)),
        ("stop empty string", b(stop="")),
        ("stop empty list", b(stop=[])),
        ("stop list of 10", b(stop=[f"s{i}" for i in range(10)])),
        ("stop appears in prompt", b(messages=[{"role": "user", "content": "say Hi"}], stop=["Hi"])),
        ("emoji content", b(messages=[{"role": "user", "content": "👩‍👩‍👧‍👦 x"}])),
        ("combining marks", b(messages=[{"role": "user", "content": "é́́"}])),
        ("RTL content", b(messages=[{"role": "user", "content": "مرحبا"}])),
        ("lone surrogate escape", b(messages=[{"role": "user", "content": "\\ud800"}])),
        ("very long content", b(messages=[{"role": "user", "content": "x" * 200000}])),
        ("deeply nested content", b(messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}])),
    ]
    return [(n, "/v1/chat/completions", body) for n, body in out]


RAW_CASES = [
    ("malformed json", "/v1/chat/completions", "not json{{{"),
    ("empty body", "/v1/chat/completions", ""),
    ("json array body", "/v1/chat/completions", "[1,2,3]"),
    ("json scalar body", "/v1/chat/completions", "42"),
]


def probe(client, path, body, raw=False):
    try:
        if raw:
            r = client.post(path, content=body, headers={"content-type": "application/json"},
                            timeout=60)
        else:
            r = client.post(path, json=body, timeout=120)
    except Exception as e:  # noqa: BLE001
        return {"status": "EXC", "type": type(e).__name__, "msg": str(e)[:60]}
    out = {"status": r.status_code}
    try:
        j = r.json()
    except Exception:  # noqa: BLE001
        out["type"] = "<non-json>"
        out["msg"] = r.text[:60]
        return out
    if isinstance(j, dict) and "error" in j and isinstance(j["error"], dict):
        out["type"] = j["error"].get("type", "?")
        out["msg"] = str(j["error"].get("message", ""))[:70]
    elif isinstance(j, dict) and "choices" in j:
        ch = (j.get("choices") or [{}])[0]
        out["type"] = "ok"
        out["msg"] = f"finish={ch.get('finish_reason')} usage={j.get('usage', {})}"
    else:
        out["type"] = "?"
        out["msg"] = json.dumps(j)[:70]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--mock-port', type=int, default=9099)
    args = ap.parse_args()

    from mock_server import MOCK_MODEL_ID, run_server
    srv = run_server(port=args.mock_port, latency_ms=1)
    threading.Event().wait(0.5)

    real = httpx.Client(base_url=args.real)
    mock = httpx.Client(base_url=f"http://127.0.0.1:{args.mock_port}")

    rows = []
    for name, path, body in cases(args.model):
        r = probe(real, path, body)
        m = probe(mock, path, dict(body, model=MOCK_MODEL_ID) if isinstance(body, dict) and 'model' in body else body)
        rows.append((name, r, m))
    for name, path, raw in RAW_CASES:
        rows.append((name, probe(real, path, raw, raw=True), probe(mock, path, raw, raw=True)))

    diverged = [(n, r, m) for n, r, m in rows if r["status"] != m["status"]]
    print(f"{len(rows)} cases, {len(diverged)} status divergences\n")
    print(f"{'case':<30} {'real':>6} {'mock':>6}  real-type / mock-type")
    print('-' * 100)
    for n, r, m in rows:
        flag = '  <<<' if r["status"] != m["status"] else ''
        print(f"{n:<30} {str(r['status']):>6} {str(m['status']):>6}  "
              f"{r.get('type', '')} / {m.get('type', '')}{flag}")
    print('\n--- divergence detail ---')
    for n, r, m in diverged:
        print(f"\n{n}\n  real: {r}\n  mock: {m}")
    if hasattr(srv, 'shutdown'):
        srv.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
