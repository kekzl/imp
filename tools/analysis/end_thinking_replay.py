"""Replay a familiar session (Anthropic messages JSON) against imp, POST end_thinking after AFTER_S seconds,
and record every stream event with its time relative to the ack.

Usage: python3 end_thinking_replay.py URL MODEL SESSION.json OUT.jsonl [AFTER_S] [MAX_TOKENS]
Tools are rebuilt from the tool_use names in the session with open schemas (the session has no tool list).
"""
import json
import sys
import threading
import time
import urllib.request

url, model, session, out_path = sys.argv[1:5]
after_s = float(sys.argv[5]) if len(sys.argv) > 5 else 5.0
max_tokens = int(sys.argv[6]) if len(sys.argv) > 6 else 8000
msgs = json.load(open(session))["messages"]
names = sorted({p["name"] for m in msgs if isinstance(m.get("content"), list) for p in m["content"]
                if p.get("type") == "tool_use"})
tools = [{"name": n, "description": f"{n} tool", "input_schema": {"type": "object", "additionalProperties": True}}
         for n in names]
rid = f"endthink-replay-{int(time.time())}"
body = {"model": model, "max_tokens": max_tokens, "stream": True, "tools": tools,
        "thinking": {"type": "enabled", "budget_tokens": max_tokens - 1000}, "messages": msgs}
req = urllib.request.Request(url + "/v1/messages", json.dumps(body).encode(),
                             {"content-type": "application/json", "X-Request-Id": rid})
ack = {"t": None, "resp": None}


def end_later():
    time.sleep(after_s)
    r = urllib.request.Request(f"{url}/v1/requests/{rid}/end_thinking", b"", method="POST")
    try:
        with urllib.request.urlopen(r) as resp:
            ack["resp"] = (resp.status, resp.read().decode()[:200])
    except urllib.error.HTTPError as e:
        ack["resp"] = (e.code, e.read().decode()[:200])
    ack["t"] = time.time()


threading.Thread(target=end_later, daemon=True).start()
counts = {}
with urllib.request.urlopen(req) as resp, open(out_path, "w") as out:
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        ev = json.loads(line[5:].strip())
        phase = "after" if ack["t"] else "before"
        kind = ev.get("type")
        if kind == "content_block_delta":
            kind = ev["delta"].get("type")
        elif kind == "content_block_start":
            kind = "start:" + ev["content_block"].get("type", "")
        counts[f"{phase}:{kind}"] = counts.get(f"{phase}:{kind}", 0) + 1
        out.write(json.dumps({"phase": phase, "ev": ev}) + "\n")
print(json.dumps({"ack": ack["resp"], "tools": names, "counts": counts}))
