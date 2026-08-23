#!/usr/bin/env python3
"""Per-kernel register and local-frame usage on sm_120a, and the ratchet on it.

WHY THIS EXISTS (#1549): the build never asked ptxas for resource usage, so no
committed artifact carried a per-kernel register or spill number, while 82
hand-set `__launch_bounds__` in src/ steer register allocation by hand and
src/compute/CLAUDE.md says never to add one blind. A kernel edit that pushes a
hot kernel past the 255-register ceiling into a local frame is invisible to
every gate in the repo: verify-fast measures throughput at an 8 % threshold, and
one spilling kernel inside a 48-layer forward is well under that.

WHERE THE NUMBERS COME FROM: `cuobjdump -res-usage` on the built library. It
reads the compiled artifact, so it needs no GPU and no special build flags - the
gate runs against whatever the normal build produced.

  REG    registers per thread. 255 is the hardware ceiling; ptxas spills past it.
  STACK  per-thread local frame in bytes. Non-zero means the kernel keeps state
         in local memory - spilled registers, or an indexed local array.
  LOCAL  separately declared local memory.

WHAT IS PINNED: not all 823 kernels - a pin that moves on every unrelated edit
stops being read. Only the ones close enough to matter (see AT_RISK below), the
same shape as tools/alloc_allowlist.txt: a two-way ratchet, so a kernel that
starts spilling fails AND a pinned kernel that improved fails, and the list
cannot go stale in either direction.
"""

import argparse
import os
import re
import subprocess
import sys

BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_resource_baseline.txt")

# A kernel is worth pinning when it is within reach of the 255-register ceiling,
# or already keeps state in local memory. 240 is 15 registers of headroom: one
# more live value in a hot loop closes that, and the next step after the ceiling
# is a spill.
REG_AT_RISK = 240

_FUNC = re.compile(r"^\s*Function (\S+):\s*$")
_ARCH = re.compile(r"^arch = (\S+)\s*$")
_USE = re.compile(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)")


# Internal-linkage kernels carry an nvcc prefix that c++filt cannot read:
#   __nv_static_48__<hash>_26_gemm_capture_fp16_sm120_cu_<hash>__ZN3imp...
# The Itanium name starts at the _Z. Without stripping this, every kernel in an
# anonymous namespace stays mangled in the baseline, which is most of the GEMM
# ones - exactly the kernels the register ceiling matters for.
_NV_STATIC = re.compile(r"^__nv_static_\d+__.*?(_ZN)")


def demangle(names):
    """Batch-demangle through c++filt; fall back to the mangled name."""
    if not names:
        return {}
    stripped = [_NV_STATIC.sub(r"\1", n) for n in names]
    for tool in ("c++filt", "/usr/local/cuda/bin/cu++filt"):
        try:
            out = subprocess.run([tool], input="\n".join(stripped), capture_output=True,
                                 text=True, check=True).stdout.splitlines()
            if len(out) == len(names):
                return dict(zip(names, out))
        except (OSError, subprocess.CalledProcessError):
            continue
    return {n: n for n in names}


def strip_signature(name):
    """Drop the trailing parameter list, keeping template arguments.

    Not `split("(")`: an anonymous-namespace kernel demangles to
    `void imp::(anonymous namespace)::gemm_fp16_kernel<128, 2>(...)`, and cutting
    at the first paren leaves `void imp::` for every one of them.
    """
    if not name.endswith(")"):
        return name
    depth = 0
    for i in range(len(name) - 1, -1, -1):
        if name[i] == ")":
            depth += 1
        elif name[i] == "(":
            depth -= 1
            if depth == 0:
                return name[:i]
    return name


def parse(text):
    """cuobjdump -res-usage output -> {(kernel, arch): (reg, stack, shared, local)}."""
    recs = {}
    arch, fn = None, None
    for line in text.splitlines():
        m = _ARCH.match(line.strip())
        if m:
            arch = m.group(1)
            continue
        m = _FUNC.match(line)
        if m:
            fn = m.group(1)
            continue
        m = _USE.search(line)
        if m and fn:
            recs[(fn, arch or "unknown")] = tuple(int(x) for x in m.groups())
            fn = None
    return recs


def at_risk(recs):
    """The subset worth pinning, demangled, sorted by how close to the cliff."""
    hits = {k: v for k, v in recs.items() if v[0] >= REG_AT_RISK or v[1] > 0 or v[3] > 0}
    names = demangle(sorted({k[0] for k in hits}))
    rows = []
    for (fn, arch), (reg, stack, shared, local) in hits.items():
        # The signature makes the line unreadable and changes with a parameter
        # rename; the template arguments do not, and they are what distinguishes
        # two instantiations with different register pressure.
        pretty = strip_signature(names.get(fn, fn))
        rows.append((pretty, arch, reg, stack, local))
    rows.sort(key=lambda r: (-r[2], -r[3], r[0], r[1]))
    return rows


def render(rows):
    out = [
        "# Per-kernel register and local-frame usage on the shipped build (#1549).",
        "# Regenerate:  make kernel-resources",
        "#",
        "# Listed here: every kernel with REG >= %d, or a non-zero local frame." % REG_AT_RISK,
        "# 255 is the hardware register ceiling - ptxas spills to local memory past",
        "# it, and one spilling kernel inside a forward pass is far below what the",
        "# 8 % throughput gate can see.",
        "#",
        "# This is a TWO-WAY ratchet: a kernel that starts spilling fails the gate,",
        "# and so does a pinned kernel that improved. Re-pin deliberately and say in",
        "# the PR which way it moved.",
        "#",
        "# kernel<tab>arch<tab>reg<tab>stack<tab>local",
    ]
    for pretty, arch, reg, stack, local in rows:
        out.append("%s\t%s\t%d\t%d\t%d" % (pretty, arch, reg, stack, local))
    return "\n".join(out) + "\n"


def load_baseline(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 5:
                continue
            rows[(parts[0], parts[1])] = tuple(int(x) for x in parts[2:])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", nargs="?", default="-",
                    help="cuobjdump -res-usage output ('-' for stdin)")
    ap.add_argument("--update", action="store_true", help="rewrite the baseline")
    ap.add_argument("--stats", action="store_true", help="print totals and exit 0")
    args = ap.parse_args()

    text = sys.stdin.read() if args.dump == "-" else open(args.dump).read()
    recs = parse(text)
    if not recs:
        print("kernel-resources: no resource records in the input - was the artifact built "
              "with device code?", file=sys.stderr)
        return 2
    rows = at_risk(recs)

    if args.stats:
        ceiling = sum(1 for r in rows if r[2] >= 255)
        framed = sum(1 for r in rows if r[3] > 0 or r[4] > 0)
        print("kernels: %d total, %d at risk (%d at the 255 ceiling, %d with a local frame)"
              % (len(recs), len(rows), ceiling, framed))
        return 0

    if args.update:
        with open(BASELINE, "w") as fh:
            fh.write(render(rows))
        print("kernel-resources: baseline re-pinned (%d kernels)" % len(rows))
        return 0

    have = {(r[0], r[1]): (r[2], r[3], r[4]) for r in rows}
    want = load_baseline(BASELINE)
    if not want:
        print("kernel-resources: no baseline at %s - create it with "
              "`python3 tools/kernel_resources.py --update`" % BASELINE, file=sys.stderr)
        return 2

    new = sorted(set(have) - set(want))
    gone = sorted(set(want) - set(have))
    moved = sorted(k for k in set(have) & set(want) if have[k] != want[k])

    for k in new:
        reg, stack, local = have[k]
        print("NEW      %s [%s] reg=%d stack=%d local=%d" % (k[0], k[1], reg, stack, local))
    for k in gone:
        print("IMPROVED %s [%s] no longer at risk" % k)
    for k in moved:
        print("MOVED    %s [%s] reg %d->%d stack %d->%d local %d->%d"
              % (k[0], k[1], want[k][0], have[k][0], want[k][1], have[k][1], want[k][2],
                 have[k][2]))

    if new or gone or moved:
        print("\nFAIL: kernel resource usage drifted from the pin.\n"
              "A kernel that STARTED spilling is the failure this gate exists for; a kernel\n"
              "that improved is reported too, so the list cannot go stale. Re-pin with\n"
              "  make kernel-resources-update\n"
              "and say in the PR which way it moved and why.", file=sys.stderr)
        return 1

    print("kernel-resources: %d at-risk kernel(s) match the pin (of %d total)"
          % (len(rows), len(recs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
