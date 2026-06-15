#!/usr/bin/env python3
"""MANUAL gate — part of the local server-test stage (`make test-server`, Stage 3).
Needs a running imp-server with a chat model (default Qwen3-8B-NVFP4-cortecs).

Asserts the OpenAI `logprobs` payload that handlers.cpp builds (handlers.cpp
~1621: `logprobs.content[].{token,logprob,bytes,top_logprobs[]}`). The coverage
driver only checks the endpoint does not 5xx; this asserts the actual numbers:

  * every per-position `logprob` is a real, non-positive float (it is log P);
  * `top_logprobs` is sorted DESCENDING by logprob (the #1 requirement the audit
    named — top-k ordering stability);
  * `len(top_logprobs)` honours the requested cap and is non-empty;
  * sum of exp(top_logprobs.logprob) is a sub-distribution (<= 1 + eps);
  * at temperature 0 (greedy) the emitted token IS the top-1 alternative — the
    chosen token equals top_logprobs[0].token and shares its logprob. This ties
    the sampler's choice to the reported distribution (a real correctness check,
    not just a shape check).

Run (stdlib only, no deps):
    python3 tests/test_server_logprobs.py
Env:
    IMP_BASE   (default http://localhost:8080)
    IMP_MODEL  (default: auto-detected from /v1/models)
    TOPK       (default 5)   top_logprobs requested

Exit code: 0 = all assertions hold (PASS), 1 = a regression (FAIL).
"""
import json, math, os, sys, urllib.request

BASE = os.environ.get("IMP_BASE", "http://localhost:8080").rstrip("/")
TOPK = int(os.environ.get("TOPK", "5"))
_fail = []


def _post(path, body, timeout=120):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def model_id():
    if os.environ.get("IMP_MODEL"):
        return os.environ["IMP_MODEL"]
    data = json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=10))
    return data["data"][0]["id"]


M = model_id()


def fail(msg):
    _fail.append(msg)
    print(f"  FAIL {msg}")


def check(cond, msg):
    if not cond:
        fail(msg)
    return cond


def main():
    print(f"logprobs gate: base={BASE} model={M} top_logprobs={TOPK}")
    r = _post("/v1/chat/completions", {
        "model": M,
        "messages": [{"role": "user", "content": "Count: one two three four"}],
        "max_tokens": 24, "temperature": 0.0,
        "logprobs": True, "top_logprobs": TOPK,
    })

    lp = r["choices"][0].get("logprobs")
    if not check(isinstance(lp, dict) and isinstance(lp.get("content"), list) and lp["content"],
                 "response has no logprobs.content array"):
        return _verdict()

    EPS = 1e-3
    for i, pos in enumerate(lp["content"]):
        tag = f"content[{i}]"
        # per-position chosen token
        check(isinstance(pos.get("token"), str), f"{tag}.token not a string")
        check(isinstance(pos.get("logprob"), (int, float)), f"{tag}.logprob not numeric")
        check(isinstance(pos.get("bytes"), list), f"{tag}.bytes not a list")
        check(pos.get("logprob", 1.0) <= EPS, f"{tag}.logprob {pos.get('logprob')} > 0 (not a log-prob)")

        top = pos.get("top_logprobs")
        if not check(isinstance(top, list) and top, f"{tag}.top_logprobs empty/missing"):
            continue
        check(len(top) <= TOPK, f"{tag} returned {len(top)} > requested {TOPK} top_logprobs")

        lps = [t.get("logprob") for t in top]
        check(all(isinstance(x, (int, float)) for x in lps), f"{tag} top_logprobs has non-numeric logprob")
        # THE ordering assertion: descending by logprob.
        check(all(lps[k] >= lps[k + 1] - EPS for k in range(len(lps) - 1)),
              f"{tag} top_logprobs NOT descending: {lps}")
        check(all(x <= EPS for x in lps), f"{tag} top_logprobs has positive logprob: {lps}")
        # sub-distribution: the top-k probabilities cannot exceed 1.
        psum = sum(math.exp(x) for x in lps)
        check(psum <= 1.0 + EPS, f"{tag} sum exp(top_logprobs) = {psum:.4f} > 1")

        # greedy tie-in: at temp 0 the chosen token is the top-1 alternative.
        if i == 0:
            check(pos["token"] == top[0].get("token"),
                  f"{tag} greedy chosen token {pos['token']!r} != top1 {top[0].get('token')!r}")
            check(abs(pos["logprob"] - top[0].get("logprob", -99)) <= EPS,
                  f"{tag} chosen logprob {pos['logprob']} != top1 {top[0].get('logprob')}")

    return _verdict()


def _verdict():
    if _fail:
        print(f"FAIL: {len(_fail)} logprob assertion(s) regressed")
        return 1
    print("PASS: logprobs well-formed, top-k descending, greedy token == top1")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
