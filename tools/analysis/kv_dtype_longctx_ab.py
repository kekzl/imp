"""KV dtype quality A/B at long context against one running imp-server (run once per arm, compare the JSON).

Haystack: real agent text (a familiar session JSON flattened to text), so attention is not looking at one
repeated sentence. Per target length: (1) three needles at depths 0.15 / 0.5 / 0.85, each asked for, hit or
miss; (2) the greedy answer to a question about the haystack (temperature 0, runtime.deterministic on the
server), saved for a byte comparison between arms.

Usage: python3 kv_dtype_longctx_ab.py URL MODEL SESSION.json OUT.json [LENGTHS]
LENGTHS: comma list of approximate prompt tokens (default 32000,64000).
"""
import json
import sys
import urllib.error
import urllib.request

url, model, session, out_path = sys.argv[1:5]
lengths = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "32000,64000").split(",")]
msgs = json.load(open(session))["messages"]
parts = []
for m in msgs:
    c = m.get("content")
    if isinstance(c, str):
        parts.append(c)
    elif isinstance(c, list):
        for p in c:
            for k in ("text", "thinking", "content"):
                if isinstance(p.get(k), str):
                    parts.append(p[k])
            if p.get("type") == "tool_use":
                parts.append(json.dumps(p.get("input"), ensure_ascii=False))
corpus = "\n\n".join(parts)
CHARS_PER_TOKEN = 3.2  # agent text with code and JSON


def haystack(n_tokens):
    need = int(n_tokens * CHARS_PER_TOKEN)
    text = corpus
    while len(text) < need:
        text += "\n\n" + corpus
    return text[:need]


def chat(content, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return "", {"error": f"{e.code} {e.read().decode()[:160]}"}
    return d["choices"][0]["message"].get("content") or "", d.get("usage", {})


result = {}
for n in lengths:
    hay = haystack(n)
    row = {"needles": []}
    for i, depth in enumerate((0.15, 0.5, 0.85)):
        code = f"KV-CHECK-{n}-{i}-{(n * 7919 + i * 104729) % 1000003}"
        cut = int(len(hay) * depth)
        doc = hay[:cut] + f"\n\nRemember this exact verification code: {code}.\n\n" + hay[cut:]
        ans, usage = chat(doc + "\n\nWhat is the exact verification code mentioned in the text above? "
                          "Reply with the code only.", 64)
        row["needles"].append({"depth": depth, "hit": code in ans, "answer": ans[:80],
                               "prompt_tokens": usage.get("prompt_tokens"), "error": usage.get("error", "")})
    ans, usage = chat(hay + "\n\nSummarize what the assistant in the text above was building and which "
                      "problems it ran into, in about 200 words.", 400)
    row["summary"] = ans
    row["summary_prompt_tokens"] = usage.get("prompt_tokens")
    result[str(n)] = row
    print(n, [x["hit"] for x in row["needles"]], usage.get("prompt_tokens"), usage.get("error", ""), flush=True)
    for x in row["needles"]:
        print("   ", x["depth"], repr(x["answer"][:60]), flush=True)
    json.dump(result, open(out_path, "w"), indent=1, ensure_ascii=False)
json.dump(result, open(out_path, "w"), indent=1, ensure_ascii=False)
