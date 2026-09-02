#!/usr/bin/env python3
"""MANUAL gate - part of the local server-test stage (`make test-server`).
Needs a running imp-server started with
    --set server.otlp_endpoint=http://host.docker.internal:4318/v1/traces
(scripts/test_server.sh does that). This script plays the collector: it binds
an OTLP/HTTP receiver on :4318, sends one streaming and one non-streaming chat
request carrying a W3C `traceparent`, and asserts that the exported spans
  * arrive as ExportTraceServiceRequest JSON (resourceSpans/scopeSpans/spans),
  * join the caller's trace (same traceId, parentSpanId = the caller's span),
  * carry the request ids, token counts and model as attributes,
  * split the streaming request into queue / prefill / decode children whose
    times lie inside the root span.
Run (stdlib only):  python3 tests/test_server_tracing.py
Env: IMP_BASE (default http://localhost:8080), IMP_OTLP_PORT (4318)
"""
import json
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

BASE = os.environ.get("IMP_BASE", "http://localhost:8080")
OTLP_PORT = int(os.environ.get("IMP_OTLP_PORT", "4318"))
received = []
lock = threading.Lock()


class Receiver(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n)
        with lock:
            received.append((self.path, json.loads(body)))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *a):  # quiet
        pass


def spans():
    out = []
    with lock:
        for _, body in received:
            for rs in body.get("resourceSpans", []):
                for ss in rs.get("scopeSpans", []):
                    out.extend(ss.get("spans", []))
    return out


def attrs(span):
    d = {}
    for a in span.get("attributes", []):
        v = a["value"]
        d[a["key"]] = v.get("stringValue", v.get("intValue", v.get("boolValue")))
    return d


def chat(stream, trace_id, parent):
    with urllib.request.urlopen(BASE + "/v1/models", timeout=30) as r:
        model = json.loads(r.read())["data"][0]["id"]
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "Say hello in five words."}],
                       "max_tokens": 16, "temperature": 0, "stream": stream}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "X-Request-Id": f"trace-test-{'s' if stream else 'n'}",
                                          "traceparent": f"00-{trace_id}-{parent}-01"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if not stream:
        return json.loads(data)["id"]
    ids = [json.loads(l[5:]).get("id") for l in data.decode().splitlines() if l.startswith("data:") and "[DONE]" not in l]
    return next(i for i in ids if i)


def main():
    srv = HTTPServer(("0.0.0.0", OTLP_PORT), Receiver)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    tid_s, par_s = "4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7"
    tid_n, par_n = "1234567890abcdef1234567890abcdef", "fedcba9876543210"
    id_s = chat(True, tid_s, par_s)
    id_n = chat(False, tid_n, par_n)
    deadline = time.time() + 15
    while time.time() < deadline:
        names = {s["traceId"] for s in spans()}
        if tid_s in names and tid_n in names:
            break
        time.sleep(0.5)
    all_spans = spans()
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    check(received and received[0][0].endswith("/v1/traces"), f"POST path: {received[0][0] if received else 'nothing received'}")
    by_trace = {}
    for s in all_spans:
        by_trace.setdefault(s["traceId"], []).append(s)
    for tid, par, rid, stream in ((tid_s, par_s, id_s, True), (tid_n, par_n, id_n, False)):
        ss = by_trace.get(tid, [])
        roots = [s for s in ss if s.get("parentSpanId") == par]
        check(len(roots) == 1, f"{tid}: expected one root span under the caller's span, got {len(roots)} of {len(ss)}")
        if not roots:
            continue
        root = roots[0]
        a = attrs(root)
        check(root["name"] == "/v1/chat/completions", f"root name {root['name']}")
        check(root["kind"] == 2, f"root kind {root['kind']} (want SERVER=2)")
        check(a.get("imp.request_id") == rid, f"imp.request_id {a.get('imp.request_id')} != {rid}")
        check(a.get("imp.client_request_id") == f"trace-test-{'s' if stream else 'n'}", f"client id {a.get('imp.client_request_id')}")
        check(int(a.get("gen_ai.usage.input_tokens", 0)) > 0, "input_tokens missing")
        check(int(a.get("gen_ai.usage.output_tokens", 0)) > 0, "output_tokens missing")
        check("gen_ai.request.model" in a, "model attribute missing")
        check(a.get("imp.stream") is stream, f"imp.stream {a.get('imp.stream')} != {stream}")
        t0, t1 = int(root["startTimeUnixNano"]), int(root["endTimeUnixNano"])
        check(t1 > t0, "root span has no duration")
        children = {s["name"]: s for s in ss if s.get("parentSpanId") == root["spanId"]}
        if stream:
            for name in ("prefill", "decode"):
                check(name in children, f"streaming request lacks '{name}' child span")
            for name, c in children.items():
                cs, ce = int(c["startTimeUnixNano"]), int(c["endTimeUnixNano"])
                check(t0 <= cs <= ce <= t1, f"child {name} [{cs},{ce}] outside root [{t0},{t1}]")
            if "prefill" in children and "decode" in children:
                check(int(children["prefill"]["endTimeUnixNano"]) == int(children["decode"]["startTimeUnixNano"]),
                      "prefill end != decode start")
        else:
            check("prefill" not in children, "non-stream request must not claim a prefill/decode split")
    srv.shutdown()
    if fails:
        print("FAIL:\n  " + "\n  ".join(fails))
        return 1
    print(f"PASS: {len(all_spans)} spans across {len(by_trace)} traces, both joined to their caller's trace")
    return 0


if __name__ == "__main__":
    sys.exit(main())
