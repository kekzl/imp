#!/usr/bin/env python3
# Greedy divergence of speculation arms against a reference arm (MTP off): per context, the common prefix of
# the two texts in characters and words, out of the reference length. Texts from mtp_longctx_client.py.
# Usage: mtp_longctx_diverge.py DIR REF_TAG TAG [TAG ...]   (files DIR/<tag>/<tag>.<ctx>.<rep>.<stream>.txt)
import sys
from pathlib import Path

root, ref, tags = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
for f in sorted((root / ref).glob(f"{ref}.*.txt"), key=lambda p: int(p.name.split(".")[1])):
    suffix = f.name[len(ref):]
    a = f.read_text(encoding="utf-8")
    for tag in tags:
        g = root / tag / f"{tag}{suffix}"
        if not g.exists():
            continue
        b = g.read_text(encoding="utf-8")
        n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        words = len(a[:n].split())
        print(f"{tag} ctx {suffix.split('.')[1]} common_chars {n}/{len(a)} common_words {words}/{len(a.split())}"
              f"{' IDENTICAL' if a == b else ''}")
