#!/usr/bin/env python3
"""Token-based clone detector (read-only audit helper).

Normalizes identifiers/literals away, hashes sliding windows of N tokens,
reports cross-file duplicate blocks. Approximates jscpd/CPD.
"""
import hashlib
import os
import re
import sys
from collections import defaultdict

ROOTS = ["src", "tools"]
EXTS = (".cpp", ".cu", ".h", ".cuh", ".hpp")
WIN = 60  # tokens

TOK = re.compile(r"[A-Za-z_]\w*|\d+\.?\d*|[^\s\w]")
KEYWORDS = set("""if else for while do switch case break continue return const constexpr
static inline void int float double char bool auto struct class enum namespace template
typename using public private protected virtual override final new delete sizeof nullptr
true false throw try catch operator unsigned long short signed extern __global__ __device__
__host__ __shared__ __restrict__ half float2 float4 size_t uint32_t int32_t""".split())


def strip_comments(s):
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"//[^\n]*", " ", s)
    return s


def tokens(path):
    try:
        src = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return [], []
    src = strip_comments(src)
    out, lines = [], []
    line = 1
    pos = 0
    for m in TOK.finditer(src):
        line += src.count("\n", pos, m.start())
        pos = m.start()
        t = m.group(0)
        if t in KEYWORDS:
            out.append(t)
        elif t[0].isalpha() or t[0] == "_":
            out.append("ID")
        elif t[0].isdigit():
            out.append("N")
        else:
            out.append(t)
        lines.append(line)
    return out, lines


def main():
    files = []
    for r in ROOTS:
        for dp, _, fns in os.walk(r):
            for fn in fns:
                if fn.endswith(EXTS):
                    files.append(os.path.join(dp, fn))
    index = defaultdict(list)
    store = {}
    for f in files:
        tk, ln = tokens(f)
        store[f] = (tk, ln)
        for i in range(0, len(tk) - WIN, 10):
            h = hashlib.md5(" ".join(tk[i:i + WIN]).encode()).hexdigest()[:16]
            index[h].append((f, i))
    groups = []
    for h, occ in index.items():
        fs = {o[0] for o in occ}
        if len(occ) >= 2 and len(fs) >= 2:
            groups.append((h, occ))
    # merge by file-pair
    pair = defaultdict(int)
    pair_ex = {}
    for h, occ in groups:
        fs = sorted({o[0] for o in occ})
        for a in range(len(fs)):
            for b in range(a + 1, len(fs)):
                key = (fs[a], fs[b])
                pair[key] += 1
                if key not in pair_ex:
                    la = store[fs[a]][1]
                    lb = store[fs[b]][1]
                    ia = next(o[1] for o in occ if o[0] == fs[a])
                    ib = next(o[1] for o in occ if o[0] == fs[b])
                    pair_ex[key] = (la[ia] if ia < len(la) else 0,
                                    lb[ib] if ib < len(lb) else 0)
    for (a, b), n in sorted(pair.items(), key=lambda kv: -kv[1])[:70]:
        la, lb = pair_ex[(a, b)]
        print(f"{n:5d} windows  {a}:{la}  <->  {b}:{lb}")


main()
