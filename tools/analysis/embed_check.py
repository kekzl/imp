#!/usr/bin/env python3
"""/v1/embeddings against an HF golden (tests/refs/qwen3_embedding_ref.json): cos >= MIN_COS per text.

Usage: embed_check.py <server-url> <golden.json> [min_cos=0.9999]. Exit 1 on any text below the bar.
Stdlib only. Mean pooling on Qwen3-Embedding measured cos 0.23-0.57 here; last-token 0.999993+.
"""
import json
import math
import sys
import urllib.request

url, golden_path = sys.argv[1], sys.argv[2]
min_cos = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9999
golden = json.load(open(golden_path, encoding="utf-8"))
fails = 0
for row in golden["rows"]:
    req = urllib.request.Request(url + "/v1/embeddings", data=json.dumps({"input": row["text"]}).encode(),
                                 headers={"Content-Type": "application/json"})
    got = json.load(urllib.request.urlopen(req, timeout=120))["data"][0]["embedding"]
    ref = row["embedding"]
    cos = sum(a * b for a, b in zip(got, ref)) / (math.sqrt(sum(a * a for a in got)) * math.sqrt(sum(b * b for b in ref)))
    ok = len(got) == len(ref) and cos >= min_cos
    fails += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] cos={cos:.6f} dim={len(got)} {row['text'][:48]!r}")
print(f"embed_check: {fails} FAIL (bar cos >= {min_cos})")
sys.exit(1 if fails else 0)
