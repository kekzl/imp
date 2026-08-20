#!/usr/bin/env python3
"""MANUAL tool — not wired into ctest/CI/verify.sh (TEST_AUDIT (retired) §7).
Needs a running imp-server on :8080 with an instruction-following model that
also serves /v1/embeddings (default Qwen3-8B-NVFP4-cortecs).

Regression battery for the imp "0 completion tokens" bug, fixed in PR #710 / v0.11.1. Under interleaved embeddings + chat the embeddings
handler used to stop() the batching engine, cancelling every in-flight
generation — chats came back empty (`finish_reason:"cancelled"`, the lone
reasoning token logged as "0 completion tokens"). The empty rate looked
"content-driven" because it only surfaced when an embed happened to overlap a
chat; under sustained mixed load it climbed toward 100% and "stuck" until
restart. The fix drains in-flight work (pause/resume) instead of cancelling.

This battery varies ONE thing at a time (content / size / temperature / trailing
cue / assistant prefill / sustained mixed load) and asserts that NO case wedges
into empty completions. It is both a diagnostic (prints the per-case empty rate)
and a gate (exits non-zero if any case regresses).

Run (stdlib only, no deps):
    python3 tests/test_server_0token_battery.py
Env:
    IMP_BASE         (default http://localhost:8080)
    IMP_MODEL        (default: auto-detected from /v1/models)
    N                (default 16)    samples per case
    LOAD             (default 150)   total mixed requests in the sustained-load lane
    FAIL_THRESHOLD   (default 0.10)  a temp>0 case fails above this empty rate;
                                     temp=0 (deterministic) cases must be exactly 0.

Exit code: 0 = all cases clean (PASS), 1 = a case regressed (FAIL).
"""
import json, os, sys, urllib.request

BASE = os.environ.get("IMP_BASE", "http://localhost:8080").rstrip("/")
N = int(os.environ.get("N", "16"))
LOAD = int(os.environ.get("LOAD", "150"))
FAIL_THRESHOLD = float(os.environ.get("FAIL_THRESHOLD", "0.10"))

_failures = []  # (label, rate, why)


def _post(path, body, timeout=120):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def model_id():
    if os.environ.get("IMP_MODEL"):
        return os.environ["IMP_MODEL"]
    data = json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=10))
    return data["data"][0]["id"]


M = model_id()


def chat(messages, temperature=0.7, max_tokens=64):
    r = _post("/v1/chat/completions", {"model": M, "messages": messages,
                                       "max_tokens": max_tokens, "temperature": temperature})
    ch = r["choices"][0]
    msg = ch["message"]
    # A completion counts as "empty" only if BOTH the visible content and any
    # reasoning_content are blank (a thinking-only turn is still a real answer).
    body = (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "").strip()
    return body, ch.get("finish_reason")


def embed(text):
    _post("/v1/embeddings", {"model": M, "input": text})


def run(label, messages, temperature=0.7, max_tokens=64, n=N):
    empty = 0
    for _ in range(n):
        try:
            body, _fr = chat(messages, temperature, max_tokens)
        except Exception as e:
            body = f"(http error: {e})"
        if body == "":
            empty += 1
    rate = empty / n
    bar = "#" * round(rate * 20)
    # Gate: temp=0 is deterministic (any empty = argmax EOS = real bug); temp>0
    # tolerates rare single-sample noise but fails on a genuine regression.
    failed = (empty > 0) if temperature == 0.0 else (rate > FAIL_THRESHOLD)
    flag = "  <-- FAIL" if failed else ""
    if failed:
        _failures.append((label, rate))
    print(f"  {label:<34} empty {empty:>2}/{n}  {rate*100:5.1f}%  {bar}{flag}")
    return rate


def U(text):
    return [{"role": "user", "content": text}]


ACTION_TAIL = ("\nFormat: one short reasoning line, then EXACTLY ONE tool action.\n"
               "Example:\nCheck a sum.\nTOOL use calc\n<<<\nprint(calc('2+2'))\n>>>\n\nNow take your action:")
TOOLS = "Tools: TOOL ls / TOOL read <f> / TOOL write <f> / TOOL python / TOOL use <name> / TOOL skill <name>."


def spark_prompt(repeat_goal=1, with_cue=True):
    p = "You are spark, an autonomous being in a Linux sandbox. " + TOOLS + "\n"
    p += "GOAL: understand yourself and make concrete progress. " * repeat_goal
    p += ACTION_TAIL if with_cue else "\nDecide what to do."
    return p


def main():
    print(f"imp 0-token battery — model={M} base={BASE} N={N} LOAD={LOAD} thr={FAIL_THRESHOLD:.0%}\n")

    print("[A] CONTROL — plain generation:")
    run("control: 'Say PONG'", U("Say PONG"))
    run("control: 'Say PONG' temp=0", U("Say PONG"), temperature=0.0)
    run("control: short Q&A", U("Name one Linux command. One word."))

    print("\n[B] CONTENT — action/tool-instruction style (the original suspected trigger):")
    run("action: tiny ('emit TOOL ls')", U("Reply with exactly one line: TOOL ls"))
    run("action: spark short", U(spark_prompt(1)))
    run("action: spark short temp=0", U(spark_prompt(1)), temperature=0.0)
    run("action: spark medium", U(spark_prompt(15)))
    run("action: spark large", U(spark_prompt(40)))

    print("\n[C] ISOLATE THE TRAILING CUE ('Now take your action:'):")
    run("action: WITH 'Now act' cue", U(spark_prompt(1, with_cue=True)))
    run("action: WITHOUT that cue", U(spark_prompt(1, with_cue=False)))

    print("\n[D] ASSISTANT PREFILL — continue mode:")
    run("action: + assistant prefill 'I will'",
        U(spark_prompt(1)) + [{"role": "assistant", "content": "I will"}])

    print("\n[E] max_tokens sensitivity:")
    run("action: max_tokens=16", U(spark_prompt(1)), max_tokens=16)
    run("action: max_tokens=240", U(spark_prompt(1)), max_tokens=240)

    print(f"\n[F] SUSTAINED LOAD — {LOAD} mixed embed+chat reqs, re-check the control every 50:")
    for done in range(0, LOAD, 50):
        for i in range(min(50, LOAD - done)):
            try:
                embed(f"situation {done + i}: explore the sandbox and make progress")
                chat(U(spark_prompt(40)), max_tokens=80)
            except Exception:
                pass
        run(f"control after {done + 50:>4} reqs", U("Say PONG"), n=6)

    print()
    if _failures:
        print(f"FAIL: {len(_failures)} case(s) regressed into empty completions:")
        for label, rate in _failures:
            print(f"  - {label}: {rate*100:.1f}% empty")
        return 1
    print("PASS: no case wedged into empty completions (interleaved embed+chat is robust)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
