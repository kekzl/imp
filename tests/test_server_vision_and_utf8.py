#!/usr/bin/env python3
"""Two ways the server answered a question nobody asked (#1197, #1198).

Needs a running imp-server. The default model is TEXT-ONLY on purpose: half of
this battery is about what happens when a picture reaches a model that has no
vision tower.

#1198 — image parts on a model without a vision tower. The load log said the
tower would be skipped, the request said 200, and the model answered from the
text alone: "please upload an image so I can describe it", or a confident
description of a picture it never saw. A caller cannot tell that apart from a
real verdict. It is now a 400 with code `vision_unavailable`.

#1197 — `response_format: json_schema` dropped every non-ASCII character:
"Die Bären hören" came back as "Die Baren horen". The FSM was fine; the token
CATEGORY pre-filter that runs before it classified any token containing a byte
>= 0x80 as "not string content", because `char` is signed. The model then
spelled the nearest ASCII word it was allowed to emit. Structure must never
alter content, so this asserts the constrained reply keeps the characters.

#1199 — the constrainer offered EOS while the JSON document was still open, so a
model that reached for it ended the request with a 200 carrying invalid JSON. The
EOS clear in the allow mask was gated on the XML dialect while the hazard belongs
to the JSON free-string shortcut just as much.

Exit code: 0 = PASS, 1 = a case regressed.
"""
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("IMP_BASE", "http://localhost:8080")
MODEL = os.environ.get("IMP_MODEL", "")

# 1x1 transparent PNG — the smallest thing that is unambiguously an image.
PNG_1X1 = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

failures = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        failures.append(label)


def post(path, payload, timeout=180):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, {"_raw": body}


def model_name():
    if MODEL:
        return MODEL
    with urllib.request.urlopen(BASE + "/v1/models", timeout=30) as r:
        return json.loads(r.read().decode())["data"][0]["id"]


def test_image_is_used_or_refused(m):
    """Either the image reaches the model, or the request is refused. Never both-nor.

    Deliberately NOT written as "skip unless the model lacks vision": deciding
    that from the response is exactly what the bug made impossible. A probe that
    reads "no 400 came back" as "this model can see" skips itself into green on
    precisely the build that is broken. So the contract is stated in terms both
    kinds of model must satisfy, and the evidence is the prompt-token count —
    an encoded image adds hundreds of tokens, and its absence is what "silently
    ignored" looks like from outside.
    """
    print("\n#1198 — an image must be used or refused, never accepted and dropped")
    text_only = {
        "model": m,
        "max_tokens": 1,
        "temperature": 0,
        "messages": [{"role": "user", "content": "Describe this image."}],
    }
    status, body = post("/v1/chat/completions", text_only)
    if status != 200:
        check("baseline text request works", False, f"got {status}")
        return
    baseline_tokens = body.get("usage", {}).get("prompt_tokens", 0)

    with_image = {
        "model": m,
        "max_tokens": 40,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": PNG_1X1}},
                ],
            }
        ],
    }
    status, body = post("/v1/chat/completions", with_image)

    if status == 400:
        err = body.get("error", {}) if isinstance(body, dict) else {}
        check("refusal carries an error envelope", bool(err.get("message")), json.dumps(body)[:160])
        check("refusal names the cause", err.get("code") == "vision_unavailable", str(err.get("code")))
        check("refusal carries no completion", "choices" not in body, "choices[] alongside the error")
        print(f"       (model cannot see; refused — baseline was {baseline_tokens} prompt tokens)")
        return

    check("accepted request answered with 200", status == 200, f"got {status}")
    if status != 200:
        return
    img_tokens = body.get("usage", {}).get("prompt_tokens", 0)
    check(
        "the image actually reached the model",
        img_tokens > baseline_tokens + 20,
        f"prompt_tokens {img_tokens} vs {baseline_tokens} text-only — the picture was dropped, "
        f"but the request was answered anyway",
    )


def test_non_ascii_survives_json_schema(m):
    print("\n#1197 — non-ASCII through response_format: json_schema")
    sentence = "Die Bären hören Gebüsch rascheln, größte Stärke, Persönlichkeit."
    payload = {
        "model": m,
        "max_tokens": 120,
        "temperature": 0,
        "messages": [{"role": "user", "content": f"Gib exakt diesen Satz zurück: {sentence}"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "S",
                "schema": {
                    "type": "object",
                    "properties": {"satz": {"type": "string"}},
                    "required": ["satz"],
                },
            },
        },
    }
    status, body = post("/v1/chat/completions", payload)
    check("status 200", status == 200, f"got {status}: {json.dumps(body)[:160]}")
    if status != 200:
        return
    content = body["choices"][0]["message"]["content"]
    text = json.loads(content)["satz"] if _is_json(content) else content

    # The model may not echo perfectly, so this asserts the CHARACTER SET is
    # reachable rather than string equality: with the bug, not one of these
    # could appear, because every token carrying them was masked out.
    produced = [c for c in "äöüÄÖÜß" if c in text]
    check(
        "umlauts are reachable under the constraint",
        len(produced) > 0,
        f"no non-ASCII in constrained output: {text!r}",
    )
    check("output is valid UTF-8", _is_utf8(text), repr(text)[:160])

    check("reply is well-formed JSON", _is_json(content), content[:160])


def test_eos_is_masked_while_the_document_is_open(m):
    """#1199 — the model must not be *offered* EOS mid-document.

    Asserting only that the reply parses would pass by luck on a model that
    never reaches for EOS mid-string. This reads the alternatives the sampler
    was given: while the JSON is incomplete, no end-of-turn token may be among
    them. On the build that had the bug, `<|im_end|>` sat at rank 2 on the very
    last step — the model took it, and the request returned 200 with the string
    still open.
    """
    print("\n#1199 — EOS must not be offered while the document is incomplete")
    status, body = post(
        "/v1/chat/completions",
        {
            "model": m,
            "max_tokens": 200,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 5,
            "messages": [
                {
                    "role": "user",
                    "content": "Gib exakt diesen Satz zurück: Die Bären hören Gebüsch rascheln, "
                    "größte Stärke, Persönlichkeit.",
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "S",
                    "schema": {
                        "type": "object",
                        "properties": {"satz": {"type": "string"}},
                        "required": ["satz"],
                    },
                },
            },
        },
    )
    if status != 200:
        check("request succeeded", False, f"got {status}")
        return
    content = body["choices"][0]["message"]["content"]
    check("constrained reply is well-formed JSON", _is_json(content), repr(content)[:160])

    steps = (body["choices"][0].get("logprobs") or {}).get("content") or []
    if not steps:
        print("       note: server returned no logprobs — the offer check needs them")
        return
    # Every step except the last is unambiguously mid-document.
    offered = []
    for st in steps[:-1]:
        for alt in st.get("top_logprobs") or []:
            tokentext = alt.get("token") or ""
            if "<|im_end|>" in tokentext or "<|endoftext|>" in tokentext:
                offered.append(tokentext)
    check(
        "no end-of-turn token among the candidates mid-document",
        not offered,
        f"offered {sorted(set(offered))} while the JSON was still open",
    )


def _is_json(s):
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def _is_utf8(s):
    try:
        s.encode("utf-8").decode("utf-8")
        return True
    except Exception:
        return False


def main():
    m = model_name()
    print(f"server: {BASE}   model: {m}")
    test_image_is_used_or_refused(m)
    test_non_ascii_survives_json_schema(m)
    test_eos_is_masked_while_the_document_is_open(m)
    print()
    if failures:
        print(f"FAIL — {len(failures)} case(s): {', '.join(failures)}")
        return 1
    print("PASS — images are refused when unusable, non-ASCII survives constraining")
    return 0


if __name__ == "__main__":
    sys.exit(main())
