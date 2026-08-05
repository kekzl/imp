#!/usr/bin/env python3
"""Does the vision tower actually see? — a two-minute answer.

The failure this exists for is the quiet one. A tower that loads partly, or
whose embeddings never reach the sequence, still returns fluent, plausible,
well-formed descriptions — so it reads as "this model is weak at vision"
rather than "this model is blind". #1246 cost ~20 iterations of a downstream
pipeline to that misreading.

Two checks, in order of how hard they are to fake:

  1. DISTINCTNESS — different pictures must produce different answers.
     A blind tower answers from the prompt alone, so its replies collapse to
     one text. This needs no ground truth and catches total blindness cold.

  2. COUNTING — N images with a known shape count, balanced across counts.
     Scored against the BEST CONSTANT ANSWER, never against raw accuracy:
     the report in #1246 showed 4/6 correct, which looked like partial
     competence but was exactly the score of always replying "1". A model
     that cannot see cannot beat the constant baseline; one that can, does.

Images are generated here (stdlib zlib only, no Pillow), so ground truth is
exact and nothing binary lands in the repo.

Usage:
  tools/analysis/vision_sight_check.py --url http://127.0.0.1:8080 --model NAME
  tools/analysis/vision_sight_check.py --model NAME --images 8 --save-dir /tmp/x
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import struct
import sys
import urllib.error
import urllib.request
import zlib

# Counts used by the counting battery. Balanced on purpose: with an equal
# number of images per count, the best constant answer scores 1/len(COUNTS),
# so "always say 1" cannot look like competence.
COUNTS = (1, 2, 3, 4)

BG = (245, 240, 225)
FG = (230, 126, 34)


def png(width: int, height: int, rgb_rows: list[bytearray]) -> bytes:
    raw = b"".join(b"\x00" + bytes(row) for row in rgb_rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 6)) +
            chunk(b"IEND", b""))


def make_image(n_circles: int, size: int = 448) -> bytes:
    """`n_circles` filled circles on a plain background, laid out on a row.

    Big and well separated: the question under test is "does any image signal
    reach the LM", not "how good is this model at small-object counting". A
    blind model fails this; a weak-but-sighted one should not.
    """
    rows = [bytearray(BG * size) for _ in range(size)]
    radius = size // 12
    for i in range(n_circles):
        cx = int(size * (i + 1) / (n_circles + 1))
        cy = size // 2
        for y in range(max(0, cy - radius), min(size, cy + radius + 1)):
            dy = y - cy
            half = int((radius * radius - dy * dy) ** 0.5)
            for x in range(max(0, cx - half), min(size, cx + half + 1)):
                rows[y][3 * x:3 * x + 3] = bytes(FG)
    return png(size, size, rows)


def ask(url: str, model: str, prompt: str, image: bytes, max_tokens: int,
        timeout: float) -> str:
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(image).decode()}},
            ],
        }],
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:400]
        raise SystemExit(f"server returned HTTP {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise SystemExit(f"cannot reach {url}: {e.reason}")
    if "error" in payload:
        raise SystemExit(f"server error: {payload['error']}")
    return (payload["choices"][0]["message"].get("content") or "").strip()


def first_int(text: str) -> int | None:
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "eins": 1, "zwei": 2, "drei": 3, "vier": 4, "fünf": 5}
    m = re.search(r"\d+", text)
    if m:
        return int(m.group())
    for word, value in words.items():
        if re.search(rf"\b{word}\b", text, re.IGNORECASE):
            return value
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", required=True, help="model id as /v1/models reports it")
    ap.add_argument("--images", type=int, default=8,
                    help="counting-battery size (rounded down to a multiple of "
                         f"{len(COUNTS)} so the counts stay balanced)")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--save-dir", help="also write the generated PNGs here")
    args = ap.parse_args()

    n = max(len(COUNTS), (args.images // len(COUNTS)) * len(COUNTS))
    truth = [COUNTS[i % len(COUNTS)] for i in range(n)]
    images = [make_image(t) for t in truth]

    if args.save_dir:
        import os
        os.makedirs(args.save_dir, exist_ok=True)
        for i, (t, img) in enumerate(zip(truth, images)):
            with open(f"{args.save_dir}/sight_{i:02d}_n{t}.png", "wb") as fh:
                fh.write(img)

    # ---- Check 1: distinctness ----
    print("== 1. distinctness (different pictures -> different answers) ==")
    describe = "Describe exactly what is in this image."
    probes = [images[truth.index(c)] for c in (COUNTS[0], COUNTS[-1])]
    answers = [ask(args.url, args.model, describe, img, 48, args.timeout) for img in probes]
    for count, answer in zip((COUNTS[0], COUNTS[-1]), answers):
        print(f"   {count} circle(s): {answer[:100]!r}")
    identical = answers[0] == answers[1]
    print(f"   -> {'IDENTICAL — the tower is not reaching the LM' if identical else 'distinct'}\n")

    # ---- Check 2: counting vs the constant baseline ----
    print(f"== 2. counting ({n} images, counts {list(COUNTS)}) ==")
    question = ("How many orange circles are in this image? "
                "Answer with a single digit and nothing else.")
    correct, guesses = 0, []
    for i, (t, img) in enumerate(zip(truth, images)):
        reply = ask(args.url, args.model, question, img, 8, args.timeout)
        got = first_int(reply)
        guesses.append(got)
        ok = got == t
        correct += ok
        print(f"   img{i:02d}  true {t}   model {got if got is not None else '?':>4}   "
              f"{'ok' if ok else 'WRONG'}")

    # The bar a blind model clears by luck: always answer the most common count.
    best_constant = max(sum(1 for t in truth if t == c) for c in COUNTS)
    print(f"\n   accuracy         {correct}/{n}")
    print(f"   constant-answer  {best_constant}/{n}   <- a blind model scores this")

    sees = (not identical) and correct > best_constant
    print("\n" + ("VERDICT: the tower SEES." if sees else
                  "VERDICT: BLIND (or the embeddings never reach the sequence)."))
    if not sees:
        print("  Check the load log for 'Vision: assigned X / Y tensors' (X<Y is fatal),")
        print("  and that the prompt token count rises by the image-token count when an")
        print("  image is attached — if it does not, the placeholders were never inserted.")
    return 0 if sees else 1


if __name__ == "__main__":
    sys.exit(main())
