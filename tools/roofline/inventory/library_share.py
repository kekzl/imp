"""Library share per phase from inventory_extract.py JSONs (ROADMAP row 123).

Class: cublas = cuBLAS/cuBLASLt internals (nvjet_*, cutlass3x_*, *xmma*, gemv2*, splitKreduce),
cutlass = cutlass:: templates instantiated in src/, own = the rest. Share = class ns / phase kernel sum.
Usage: python library_share.py <out_dir> [top=3]
"""

import json
import pathlib
import re
import sys

CUBLAS = re.compile(r"nvjet|cutlass3x_|xmma|gemv2|gemvx|gemmk1|splitKreduce|cublas", re.I)
CUTLASS = re.compile(r"cutlass::")
PHASES = ("bench:pp", "bench:tg")


def classify(name):
    if CUBLAS.search(name):
        return "cublas"
    return "cutlass" if CUTLASS.search(name) else "own"


def main():
    out = pathlib.Path(sys.argv[1])
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    print("| cell | phase | kernel ms | cuBLAS/Lt ms | % | CUTLASS ms | % | top library kernels (us/call x calls) |")
    print("|---|---|---|---|---|---|---|---|")
    for path in sorted(out.glob("*.json")):
        doc = json.loads(path.read_text())
        for phase in PHASES:
            ph = doc.get("phases", {}).get(phase)
            if not ph:
                continue
            total = ph["kernel_sum_ns"] or 1
            ns = {"cublas": 0, "cutlass": 0, "own": 0}
            lib = []
            for k in ph["kernels"]:
                cls = classify(k["name"])
                ns[cls] += k["total_ns"]
                if cls != "own":
                    lib.append(k)
            lib.sort(key=lambda k: -k["total_ns"])
            tops = ", ".join(
                f"{cls_tag(k)} {k['short'][:48]} {k['total_ns'] / max(k['count'], 1) / 1e3:.1f}x{k['count']}"
                for k in lib[:top]
            )
            print(
                f"| {doc['cell']} | {phase} | {total / 1e6:.1f} | {ns['cublas'] / 1e6:.1f} | "
                f"{100 * ns['cublas'] / total:.1f} | {ns['cutlass'] / 1e6:.1f} | "
                f"{100 * ns['cutlass'] / total:.1f} | {tops or '-'} |"
            )


def cls_tag(k):
    return "[Lt]" if classify(k["name"]) == "cublas" else "[CT]"


if __name__ == "__main__":
    main()
