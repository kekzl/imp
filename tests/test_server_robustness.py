#!/usr/bin/env python3
"""MANUAL tool — not wired into ctest/CI/verify.sh (TEST_AUDIT.md §7).
Needs a running imp-server on :8080 (default model Qwen3-8B-NVFP4-cortecs).

Robustness battery for the HTTP surface (DEBUG-500-on-bad-input.md). imp must
NEVER answer client-supplied bad input with a 5xx or a bare/opaque body: bad
input is a 4xx with an OpenAI-style `{"error":{"message":..,"type":..}}`
envelope. The original bug: invalid UTF-8 in the body made json::parse throw
parse_error, the error envelope echoed the offending bytes, and err.dump()
then threw json::type_error.316 (dump rejects ill-formed UTF-8) which escaped
the catch → bare HTTP 500. Fix: a global set_exception_handler + dump_safe()
(dump with error_handler_t::replace) on every response/SSE/error/log body.

This battery hits every body-taking endpoint with many one-variable-off bad
inputs (invalid UTF-8 via raw socket so the byte survives, malformed JSON,
empty body, non-object, wrong field types, missing required fields) and asserts:
  * status is 4xx (never 5xx, never a non-JSON/opaque body),
  * the body parses as JSON and carries a non-empty error.message,
and that valid control requests still return 2xx.

Exit code: 0 = all good (PASS), 1 = a case regressed (FAIL).
"""
import json, os, socket, sys, urllib.request, urllib.error

HOST = os.environ.get("IMP_HOST", "localhost")
PORT = int(os.environ.get("IMP_PORT", "8080"))
M = os.environ.get("IMP_MODEL", "Qwen3-8B-NVFP4-cortecs")
BASE = f"http://{HOST}:{PORT}"

_fail = []
_count = 0


def raw_post(path, body_bytes):
    """Send a request over a raw socket so invalid bytes survive (urllib/requests
    would re-encode or reject). Returns (status:int, body:bytes)."""
    req = (f"POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
           f"Content-Length: {len(body_bytes)}\r\nConnection: close\r\n\r\n").encode() + body_bytes
    s = socket.create_connection((HOST, PORT), timeout=60)
    s.sendall(req)
    buf = b""
    while True:
        d = s.recv(8192)
        if not d:
            break
        buf += d
    s.close()
    head, _, body = buf.partition(b"\r\n\r\n")
    status_line = head.split(b"\r\n", 1)[0].decode("latin1")
    status = int(status_line.split(" ")[1]) if len(status_line.split(" ")) > 1 else 0
    # de-chunk if needed (Connection: close usually gives identity; be lenient)
    return status, body


def check_bad(label, path, body_bytes):
    """A bad-input request must yield 4xx + a JSON error envelope, never 5xx/opaque."""
    global _count
    _count += 1
    try:
        status, body = raw_post(path, body_bytes)
    except Exception as e:
        _fail.append((label, f"transport error: {e}"))
        print(f"  FAIL {label}: transport error {e}")
        return
    ok = True
    reason = ""
    if status >= 500 or status < 400:
        ok = False
        reason = f"status {status} (want 4xx)"
    else:
        try:
            j = json.loads(body.decode("utf-8", "replace"))
            msg = (j.get("error") or {}).get("message")
            if not msg:
                ok = False
                reason = f"status {status} but no error.message (body={body[:120]!r})"
        except Exception:
            ok = False
            reason = f"status {status} but non-JSON body ({body[:120]!r})"
    if ok:
        print(f"  ok   {label}: {status} + error envelope")
    else:
        _fail.append((label, reason))
        print(f"  FAIL {label}: {reason}")


def check_no5xx(label, path, body_bytes):
    """A lenient case: imp may accept it (2xx) or reject it (4xx+envelope), but it
    must NEVER 5xx or return an opaque body. Used for wrong-type optional fields
    where strict-400 vs lenient-coerce is a design choice, not a robustness bug."""
    global _count
    _count += 1
    try:
        status, body = raw_post(path, body_bytes)
    except Exception as e:
        _fail.append((label, f"transport error: {e}"))
        print(f"  FAIL {label}: transport error {e}")
        return
    if status >= 500:
        _fail.append((label, f"status {status} (5xx on bad input)"))
        print(f"  FAIL {label}: status {status} (5xx)")
        return
    if status >= 400:
        try:
            j = json.loads(body.decode("utf-8", "replace"))
            if not (j.get("error") or {}).get("message"):
                _fail.append((label, f"4xx without error.message ({body[:120]!r})"))
                print(f"  FAIL {label}: 4xx without error.message")
                return
        except Exception:
            _fail.append((label, f"4xx non-JSON body ({body[:120]!r})"))
            print(f"  FAIL {label}: 4xx non-JSON body")
            return
    print(f"  ok   {label}: {status} (no 5xx)")


def check_valid(label, path, obj):
    """A valid request must return 2xx."""
    global _count
    _count += 1
    data = json.dumps(obj).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            status = r.status
            r.read()
    except urllib.error.HTTPError as e:
        status = e.code
    except Exception as e:
        _fail.append((label, f"transport error: {e}"))
        print(f"  FAIL {label}: transport error {e}")
        return
    if 200 <= status < 300:
        print(f"  ok   {label}: {status}")
    else:
        _fail.append((label, f"status {status} (want 2xx)"))
        print(f"  FAIL {label}: status {status} (want 2xx)")


def jb(obj):
    return json.dumps(obj).encode()


def invalid_utf8_struct(obj):
    """Valid JSON for obj, then corrupted: drop 3 trailing bytes + inject lone 0x80."""
    return jb(obj)[:-3] + b"\x80\"}]}"


def invalid_utf8_in_string(obj, placeholder="PONGMARK"):
    """obj must contain the placeholder in a string value; replace with an invalid byte."""
    return jb(obj).replace(placeholder.encode(), b"hi\x80there")


# endpoint -> a minimal valid body + a body that contains PONGMARK in a string field
ENDPOINTS = {
    "/v1/chat/completions": (
        {"model": M, "messages": [{"role": "user", "content": "ok"}], "max_tokens": 8},
        {"model": M, "messages": [{"role": "user", "content": "PONGMARK"}], "max_tokens": 8},
    ),
    "/v1/completions": (
        {"model": M, "prompt": "ok", "max_tokens": 8},
        {"model": M, "prompt": "PONGMARK", "max_tokens": 8},
    ),
    "/v1/embeddings": (
        {"model": M, "input": "ok"},
        {"model": M, "input": "PONGMARK"},
    ),
    "/v1/messages": (
        {"model": M, "messages": [{"role": "user", "content": "ok"}], "max_tokens": 8},
        {"model": M, "messages": [{"role": "user", "content": "PONGMARK"}], "max_tokens": 8},
    ),
    "/tokenize": ({"model": M, "content": "ok"}, {"model": M, "content": "PONGMARK"}),
    "/detokenize": ({"model": M, "tokens": [1, 2, 3]}, None),
}

# per-endpoint wrong-type / missing-field bad bodies (valid UTF-8, malformed semantics)
BAD_SEMANTIC = {
    "/v1/chat/completions": [
        ("missing messages", {"model": M, "max_tokens": 8}),
        ("messages not array", {"model": M, "messages": "hi", "max_tokens": 8}),
        ("message not object", {"model": M, "messages": [123]}),
        ("temperature wrong type", {"model": M, "messages": [{"role": "user", "content": "x"}], "temperature": "hot"}),
        ("max_tokens wrong type", {"model": M, "messages": [{"role": "user", "content": "x"}], "max_tokens": "lots"}),
        ("top_p wrong type", {"model": M, "messages": [{"role": "user", "content": "x"}], "top_p": "high"}),
        ("n wrong type", {"model": M, "messages": [{"role": "user", "content": "x"}], "n": "two"}),
    ],
    "/v1/completions": [
        ("missing prompt", {"model": M, "max_tokens": 8}),
        ("max_tokens wrong type", {"model": M, "prompt": "x", "max_tokens": "lots"}),
    ],
    "/v1/embeddings": [
        ("missing input", {"model": M}),
        ("input wrong type", {"model": M, "input": 123}),
        ("input array of non-strings", {"model": M, "input": [1, 2, 3]}),
    ],
    "/v1/messages": [
        ("missing messages", {"model": M, "max_tokens": 8}),
        ("messages not array", {"model": M, "messages": "hi", "max_tokens": 8}),
    ],
    "/tokenize": [
        ("missing content", {"model": M}),
        ("content wrong type", {"model": M, "content": 123}),
    ],
    "/detokenize": [
        ("missing tokens", {"model": M}),
        ("tokens not array", {"model": M, "tokens": "abc"}),
        ("tokens non-int elements", {"model": M, "tokens": ["a", "b"]}),
    ],
}

# generic malformed bodies applied to every endpoint
def generic_bad_bodies():
    return [
        ("malformed JSON", b'{"model": garbage'),
        ("empty body", b""),
        ("not an object (array)", b"[]"),
        ("not an object (string)", b'"hello"'),
        ("not an object (number)", b"123"),
        ("truncated", b'{"model":'),
    ]


def main():
    print(f"imp HTTP robustness battery — base={BASE} model={M}\n")

    print("[1] invalid UTF-8 (raw socket — the original 500 bug):")
    for path, (valid, marked) in ENDPOINTS.items():
        check_bad(f"{path}  invalid-UTF-8 (broken struct)", path, invalid_utf8_struct(valid))
        if marked is not None:
            check_bad(f"{path}  invalid-UTF-8 (in string)", path, invalid_utf8_in_string(marked))

    print("\n[2] generic malformed bodies on every endpoint:")
    for path in ENDPOINTS:
        for label, body in generic_bad_bodies():
            check_bad(f"{path}  {label}", path, body)

    print("\n[3] semantic errors — missing required field must 4xx; wrong-type may 4xx or 2xx but never 5xx:")
    for path, cases in BAD_SEMANTIC.items():
        for label, obj in cases:
            if label.startswith("missing"):
                check_bad(f"{path}  {label}", path, jb(obj))
            else:
                check_no5xx(f"{path}  {label}", path, jb(obj))

    print("\n[4] oversized field (1 MiB string):")
    big = "A" * (1024 * 1024)
    check_no5xx("/v1/chat/completions  huge non-numeric max_tokens",
                "/v1/chat/completions",
                jb({"model": M, "messages": [{"role": "user", "content": "x"}], "max_tokens": big}))

    print("\n[5] valid control requests (must be 2xx):")
    check_valid("/v1/chat/completions valid", "/v1/chat/completions",
                {"model": M, "messages": [{"role": "user", "content": "Say PONG"}], "max_tokens": 8})
    check_valid("/v1/completions valid", "/v1/completions", {"model": M, "prompt": "Say PONG", "max_tokens": 8})
    check_valid("/v1/embeddings valid", "/v1/embeddings", {"model": M, "input": "hello"})
    check_valid("/tokenize valid", "/tokenize", {"model": M, "content": "hello world"})
    check_valid("/detokenize valid", "/detokenize", {"model": M, "tokens": [9707, 1879]})

    print()
    if _fail:
        print(f"FAIL: {len(_fail)}/{_count} cases regressed:")
        for label, reason in _fail:
            print(f"  - {label}: {reason}")
        return 1
    print(f"PASS: all {_count} cases robust (bad input -> 4xx+envelope, valid -> 2xx)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
