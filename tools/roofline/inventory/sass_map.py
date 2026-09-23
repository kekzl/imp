"""Per-kernel SASS facts from `cuobjdump -sass`: tensor-core opcode variants, TMA use, local memory.
Usage: python sass_map.py <imp.sass> <sass_map.json>
Keys are mangled names (nsys `mangledName` / cuobjdump `Function :`).
"""

import collections
import json
import re
import sys

OPC = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)")
TC = re.compile(r"^(OMMA|QMMA|HMMA|IMMA|DMMA)")


def main():
    funcs = {}
    cur = None
    for line in open(sys.argv[1], errors="replace"):
        if "Function :" in line:
            cur = line.split("Function :", 1)[1].strip()
            funcs[cur] = collections.Counter()
            continue
        if cur is None:
            continue
        m = OPC.search(line)
        if not m:
            continue
        op = m.group(1)
        c = funcs[cur]
        c["_insts"] += 1
        if TC.match(op):
            c[op] += 1
        elif op.startswith(("UTMALDG", "UTMASTG", "UBLKCP", "UTMAPF")):
            c["_tma"] += 1
        elif op.startswith(("LDL", "STL")):
            c["_local"] += 1
    out = {}
    for f, c in funcs.items():
        tc = {k: v for k, v in c.items() if not k.startswith("_")}
        out[f] = {"insts": c["_insts"], "tc": tc, "tma": c["_tma"] > 0, "local_ops": c["_local"]}
    json.dump(out, open(sys.argv[2], "w"))
    n_tc = sum(1 for v in out.values() if v["tc"])
    print(f"{len(out)} functions, {n_tc} with tensor ops, {sum(v['tma'] for v in out.values())} with TMA")


if __name__ == "__main__":
    main()
