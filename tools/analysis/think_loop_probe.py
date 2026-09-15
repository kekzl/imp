#!/usr/bin/env python3
"""N concurrent two-turn chat sessions against a running imp-server, non-streaming with
logprobs (they cover reasoning AND content tokens; streaming logprobs cover content only).

Per session it prints: turn-1 token count, the first token where the history rendering of
the reply (reasoning.strip() + "\\n</think>\\n\\n" + content) re-tokenizes differently from
the generated tokens, and turn-2 cached_tokens. A diff inside the reply on a thinking model
is the batched-decode repeat (a row re-emits </think> + its answer); a diff at the think seam
is a tokenization seam; cached_tokens == turn-1 tokens + prompt - 1 is a transcript restore.

  python3 tools/analysis/think_loop_probe.py http://127.0.0.1:8080 8 [dump_dir]
  EXTRA_JSON='{"enable_thinking": false}' ...   # merged into every request body
  NOLP=1 ...   # no logprobs: the rows qualify for the decode pipeline (a logprobs row never
               # does), re-closes are counted as "</think>" text inside content, no re-tokenize
"""
import json
import os
import sys
import threading
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
NOLP = os.environ.get("NOLP", "0") == "1"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
DUMP = sys.argv[3] if len(sys.argv) > 3 else None
TOPICS = ["Explain how a paged KV cache shares blocks between sequences.",
          "Why does a recurrent state make prefix caching harder than plain attention?",
          "Give a short argument for measuring bandwidth instead of trusting cudaMalloc.",
          "What is the cost of one extra chunk boundary in a chunked prefill?"]


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())


def main():
    models = json.loads(urllib.request.urlopen(URL + "/v1/models", timeout=10).read())["data"]
    model = next((m["id"] for m in models if m.get("loaded")), models[0]["id"])
    res = [None] * N

    def one(i):
        msgs = [{"role": "system", "content": f"Session {i}. Answer briefly."},
                {"role": "user", "content": TOPICS[i % len(TOPICS)]}]
        base = {"model": model, "temperature": 0.0, "max_tokens": 1500, "enable_thinking": True,
                "stream": False}
        base.update(json.loads(os.environ.get("EXTRA_JSON", "{}")))
        r1 = post("/v1/chat/completions", dict(base, messages=msgs, logprobs=not NOLP))
        m = r1["choices"][0]["message"]
        content, reasoning = m.get("content") or "", m.get("reasoning_content") or ""
        lp = [] if NOLP else r1["choices"][0]["logprobs"]["content"]
        msgs.append({"role": "assistant", "content": content, "reasoning_content": reasoning})
        msgs.append({"role": "user", "content": "Summarise that in one sentence."})
        r2 = post("/v1/chat/completions", dict(base, messages=msgs, max_tokens=64))
        u2 = r2.get("usage", {})
        res[i] = (lp, reasoning, content, r1["choices"][0].get("finish_reason"), u2.get("prompt_tokens"),
                  (u2.get("prompt_tokens_details") or {}).get("cached_tokens"))

    threads = [threading.Thread(target=one, args=(i,)) for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    bad = 0
    for i in range(N):
        lp, reasoning, content, fin, p2, c2 = res[i]
        if NOLP:
            n_close = 1 + content.count("</think>") if reasoning else content.count("</think>")
            print(f"session {i}: turn1 content {len(content)} chars finish={fin} </think> x{n_close} "
                  f"(text count) | turn2 prompt {p2} cached {c2}")
            bad += n_close > 1
            continue
        gen = [t["token"] for t in lp]
        rendered = reasoning.strip() + "\n</think>\n\n" + content
        ids = post("/tokenize", {"prompt": rendered})["tokens"]
        pieces = [post("/detokenize", {"tokens": [x]})["content"] for x in ids]
        k = next((j for j in range(min(len(gen), len(pieces))) if gen[j] != pieces[j]), None)
        ends = [j for j, t in enumerate(gen) if t == "</think>"]
        print(f"session {i}: turn1 tokens {len(gen)} finish={fin} </think> x{len(ends)} at {ends[:6]} "
              f"retok {len(pieces)} first diff {k} | turn2 prompt {p2} cached {c2}")
        if k is not None:
            print("  gen  :", [repr(x) for x in gen[max(0, k - 4):k + 5]])
            print("  retok:", [repr(x) for x in pieces[max(0, k - 4):k + 5]])
        bad += len(ends) > 1
        if DUMP:
            with open(os.path.join(DUMP, f"sess{i}.json"), "w") as f:
                json.dump({"reasoning": reasoning, "content": content, "lp": lp}, f)
    print(f"{bad}/{N} sessions closed a think block more than once")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
