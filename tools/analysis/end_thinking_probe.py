"""end_thinking probe: stream a thinking request, POST /v1/requests/{id}/end_thinking after AFTER_S seconds,
report how many reasoning / text deltas arrive before and after the ack.

Usage: python3 end_thinking_probe.py URL MODEL [AFTER_S] [ROUTE] [OPENER_ID]   (ROUTE: messages | chat)
OPENER_ID (chat): logit_bias +50 on the think opener, so a model re-opens the block whenever it may.
"""
import json
import sys
import threading
import time
import urllib.request

url, model = sys.argv[1], sys.argv[2]
after_s = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
route = sys.argv[4] if len(sys.argv) > 4 else "messages"
opener = sys.argv[5] if len(sys.argv) > 5 else None
rid = f"endthink-probe-{int(time.time())}"
prompt = ("Prove carefully that there are infinitely many primes of the form 4k+3, then list the first "
          "twenty such primes and check each one. Think through every step in detail before answering.")
if route == "messages":
    path = "/v1/messages"
    body = {"model": model, "max_tokens": 6000, "stream": True,
            "thinking": {"type": "enabled", "budget_tokens": 5000},
            "messages": [{"role": "user", "content": prompt}]}
else:
    path = "/v1/chat/completions"
    body = {"model": model, "max_tokens": 6000, "stream": True,
            "messages": [{"role": "user", "content": prompt}]}
    if opener:
        body["logit_bias"] = {opener: 50}
req = urllib.request.Request(url + path, json.dumps(body).encode(),
                             {"content-type": "application/json", "X-Request-Id": rid})
state = {"ack": None, "ack_t": None}
counts = {"reason_before": 0, "reason_after": 0, "text_before": 0, "text_after": 0}
first_text_after = []


def end_later():
    time.sleep(after_s)
    r = urllib.request.Request(f"{url}/v1/requests/{rid}/end_thinking", b"", method="POST")
    try:
        with urllib.request.urlopen(r) as resp:
            state["ack"] = (resp.status, resp.read().decode()[:200])
    except urllib.error.HTTPError as e:
        state["ack"] = (e.code, e.read().decode()[:200])
    state["ack_t"] = time.time()


threading.Thread(target=end_later, daemon=True).start()
t0 = time.time()
with urllib.request.urlopen(req) as resp:
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        ev = json.loads(data)
        after = state["ack_t"] is not None
        reason = text = ""
        if route == "messages" and ev.get("type") == "content_block_delta":
            d = ev["delta"]
            reason, text = d.get("thinking", ""), d.get("text", "")
        elif route == "chat" and ev.get("choices"):
            d = ev["choices"][0].get("delta", {})
            reason, text = d.get("reasoning_content") or "", d.get("content") or ""
        if reason:
            counts["reason_after" if after else "reason_before"] += 1
        if text:
            counts["text_after" if after else "text_before"] += 1
            if after and sum(map(len, first_text_after)) < 300:
                first_text_after.append(text)
        if after and counts["reason_after"] > 400:
            break  # reasoning ran on 400 deltas past the ack: the bug
print(json.dumps({"route": route, "ack": state["ack"], "counts": counts, "text_after_head": "".join(first_text_after)[:300],
                  "wall_s": round(time.time() - t0, 1)}))
