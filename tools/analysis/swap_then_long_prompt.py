"""Swap away and back, then a long prompt as the first request on the fresh model (familiar on Flint: the KV
pool had grown to 5096 of its ceiling 2 s after the reload and admission cancelled an 83.7k prompt).

Usage: python3 swap_then_long_prompt.py URL MODEL_A MODEL_B CORPUS_JSON TOKENS [ROUNDS]
Prints per round: HTTP status, prompt_tokens, the error message if any.
"""
import json
import sys
import urllib.error
import urllib.request

url, model_a, model_b, corpus_path, tokens = sys.argv[1:6]
rounds = int(sys.argv[6]) if len(sys.argv) > 6 else 2
msgs = json.load(open(corpus_path))["messages"]
text = "\n\n".join(p.get("text") or p.get("thinking") or json.dumps(p.get("input") or p.get("content"))
                   for m in msgs if isinstance(m.get("content"), list) for p in m["content"])
need = int(int(tokens) * 3.2)
while len(text) < need:
    text += "\n\n" + text
text = text[:need]


def post(model, content, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.loads(r.read())
            return r.status, d.get("usage", {}).get("prompt_tokens"), ""
    except urllib.error.HTTPError as e:
        return e.code, None, e.read().decode()[:200]


for i in range(rounds):
    print("swap away:", post(model_b, "Say OK.", 4), flush=True)
    print(f"round {i} long prompt:", post(model_a, text + "\n\nSummarize the text above in one sentence.", 32),
          flush=True)
