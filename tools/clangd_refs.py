#!/usr/bin/env python3
"""Symbol references through clangd, backing `make refs`.

WHY THIS EXISTS
---------------
`rg` cannot tell a field read from a string literal or from a same-named member of another
type: for `Speculative::mtp_k` it returned 154 lines where 64 were reads of that field
("speculative.mtp_k=2" log strings, `p.mtp_k` of the request parser). clangd resolves the
symbol, also in .cu files and through aliases (`scfg.`, `cfg.speculative.`). Probe
2026-11-14 on 22789f23: 1024 TUs (371 .cu) indexed in 72 s; 6 symbols (constexpr, host
function in a .cu, template kernel behind pdl::launch, two config fields): every code
reference found, no false one. Key strings in scripts, docs and imp.conf stay `rg` work.

HOW
---
clangd cannot read nvcc commands, so the .cu entries of build-dev/compile_commands.json are
rewritten as clang CUDA commands (sm_120a, the image's CUDA, its -D/-I/-isystem kept) into
build-dev/clangd/compile_commands.json. clangd indexes in the background; the queries run
once the index reports done.

Usage (imp:lint, repo mounted at /src, after `make dev`):
    python3 tools/clangd_refs.py --sym gemm_nvfp4_batched --sym Speculative::mtp_k
    python3 tools/clangd_refs.py --at src/core/config/speculative.h:95:mtp_k
    python3 tools/clangd_refs.py --selftest
--sym takes a name or a scope-qualified suffix (`Speculative::mtp_k`); every symbol of that
name is reported on its own, with its container. Output per symbol: qualified name, kind,
location, reference count, then `path:line: source` per reference.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time

ROOT = pathlib.Path("/src")
SRC_CDB = ROOT / "build-dev" / "compile_commands.json"
OUT_DIR = ROOT / "build-dev" / "clangd"
CUDA_FLAGS = ["-x", "cuda", "--cuda-gpu-arch=sm_120a", "--cuda-path=/usr/local/cuda",
              "--no-cuda-version-check", "-Wno-unknown-cuda-version", "-std=c++23"]
KINDS = {5: "class", 6: "method", 8: "field", 9: "constructor", 10: "enum", 12: "function",
         13: "variable", 14: "constant", 22: "enum member", 23: "struct"}


def clang_cuda_args(command: str, file: str) -> list[str]:
    """nvcc command line -> clang CUDA arguments: -D/-I/-isystem kept, every nvcc flag dropped."""
    toks = shlex.split(command)
    out = ["clang++", *CUDA_FLAGS]
    i = 1
    while i < len(toks):
        t = toks[i]
        if t in ("-I", "-isystem", "-D") and i + 1 < len(toks):
            out += [t, toks[i + 1]]
            i += 2
            continue
        if t.startswith(("-I", "-D")):
            out.append(t)
        i += 1
    return out + ["-c", file]


def convert(src: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
    """Rewrite the .cu entries; reuse the output while it is newer than the build's database."""
    out = out_dir / "compile_commands.json"
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out
    entries = []
    for e in json.loads(src.read_text()):
        if e["file"].endswith((".cu", ".cuh")):
            cmd = e["command"] if "command" in e else shlex.join(e["arguments"])
            e = {"directory": e["directory"], "file": e["file"], "arguments": clang_cuda_args(cmd, e["file"])}
        entries.append(e)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries))
    return out


def symbol_matches(query: str, name: str, container: str) -> bool:
    """`mtp_k` matches every symbol named mtp_k; `Speculative::mtp_k` only one in a scope ending so."""
    scope, _, leaf = query.rpartition("::")
    if name != leaf:
        return False
    return not scope or container == scope or container.endswith("::" + scope)


class Clangd:
    def __init__(self, cdb_dir: pathlib.Path):
        self.proc = subprocess.Popen(
            ["clangd", f"--compile-commands-dir={cdb_dir}", "--background-index", f"-j={os.cpu_count()}",
             "--log=error", "--pch-storage=memory", "--limit-references=0", "--limit-results=0"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.next_id = 0
        self.index_state = None  # None, "begin", "end"
        self.opened: set[str] = set()

    def send(self, msg: dict) -> None:
        body = json.dumps(msg).encode()
        self.proc.stdin.write(b"Content-Length: %d\r\n\r\n" % len(body) + body)
        self.proc.stdin.flush()

    def read(self) -> dict:
        length = 0
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("clangd exited")
            if line in (b"\r\n", b"\n"):
                break
            if line.lower().startswith(b"content-length:"):
                length = int(line.split(b":")[1])
        return json.loads(self.proc.stdout.read(length))

    def handle(self, m: dict) -> None:
        if m.get("method") == "window/workDoneProgress/create":
            self.send({"jsonrpc": "2.0", "id": m["id"], "result": None})
        elif m.get("method") == "$/progress" and m["params"].get("token") == "backgroundIndexProgress":
            self.index_state = m["params"]["value"].get("kind", self.index_state)
            if self.index_state == "report":
                self.index_state = "begin"

    def request(self, method: str, params: dict):
        self.next_id += 1
        rid = self.next_id
        self.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        while True:
            m = self.read()
            if m.get("id") == rid and "method" not in m:
                return m.get("result")
            self.handle(m)

    def open(self, path: str) -> None:
        if path in self.opened:
            return
        self.opened.add(path)
        self.send({"jsonrpc": "2.0", "method": "textDocument/didOpen",
                   "params": {"textDocument": {"uri": "file://" + path, "languageId": "cpp", "version": 1,
                                               "text": pathlib.Path(path).read_text(errors="replace")}}})

    def start(self, first_file: str) -> float:
        t0 = time.time()
        self.request("initialize", {"processId": None, "rootUri": f"file://{ROOT}",
                                    "capabilities": {"window": {"workDoneProgress": True}}})
        self.send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
        self.open(first_file)  # loads the database, which starts the background index
        while self.index_state != "end":
            self.handle(self.read())
        return time.time() - t0

    def references(self, path: str, line0: int, col: int) -> list[tuple[str, int]]:
        self.open(path)
        res = self.request("textDocument/references",
                           {"textDocument": {"uri": "file://" + path}, "position": {"line": line0, "character": col},
                            "context": {"includeDeclaration": False}}) or []
        return sorted({(r["uri"][len("file://"):], r["range"]["start"]["line"] + 1) for r in res})

    def stop(self) -> None:
        self.proc.terminate()


def rel(path: str) -> str:
    return path[len(str(ROOT)) + 1:] if path.startswith(str(ROOT) + "/") else path


def source_line(path: str, line: int) -> str:
    try:
        return pathlib.Path(path).read_text(errors="replace").split("\n")[line - 1].strip()[:120]
    except (OSError, IndexError):
        return ""


def report(title: str, refs: list[tuple[str, int]]) -> None:
    print(f"{title}: {len(refs)} refs")
    for path, line in refs:
        print(f"  {rel(path)}:{line}: {source_line(path, line)}")


def selftest() -> int:
    nvcc = ('/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler -DIMP_X=\\"1\\" -I/src/src '
            '-isystem /usr/local/cuda/include --expt-relaxed-constexpr -Xcudafe --diag_suppress=177 '
            '--generate-code=arch=compute_120a,code=sm_120a -Xptxas -O3 -x cu -c /src/a.cu -o a.o')
    got = clang_cuda_args(nvcc, "/src/a.cu")
    cases = [
        ("defines and includes kept", '-DIMP_X="1"' in got and "-I/src/src" in got
         and got[got.index("-isystem") + 1] == "/usr/local/cuda/include"),
        ("nvcc flags dropped", not any(t in got for t in ("-Xptxas", "-Xcudafe", "cu", "-O3", "a.o"))),
        ("clang CUDA mode", got[:3] == ["clang++", "-x", "cuda"] and got[-2:] == ["-c", "/src/a.cu"]),
        ("bare name matches every scope", symbol_matches("mtp_k", "mtp_k", "imp::cfg::Speculative")
         and symbol_matches("mtp_k", "mtp_k", "SpecField")),
        ("scope suffix selects one", symbol_matches("Speculative::mtp_k", "mtp_k", "imp::cfg::Speculative")
         and not symbol_matches("Speculative::mtp_k", "mtp_k", "SpecField")
         and not symbol_matches("Speculative::mtp_k", "mtp_k", "imp::cfg::NotSpeculative")),
        ("other name rejected", not symbol_matches("mtp_k", "mtp_kv_window", "imp::cfg::Speculative")),
    ]
    bad = [n for n, ok in cases if not ok]
    for n in bad:
        print(f"selftest FAIL: {n}")
    print(f"selftest: {len(cases) - len(bad)}/{len(cases)} planted cases pass")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sym", action="append", default=[], help="name or Scope::name, repeatable")
    ap.add_argument("--at", action="append", default=[], help="path:line:token, repeatable")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.sym and not a.at:
        ap.error("give --sym or --at")
    if not SRC_CDB.exists():
        print(f"clangd_refs: {rel(str(SRC_CDB))} missing, run `make dev` first", file=sys.stderr)
        return 2
    cdb = convert(SRC_CDB, OUT_DIR)
    first = json.loads(cdb.read_text())[0]["file"]
    ats = []
    for at in a.at:
        path, line, token = at.rsplit(":", 2)
        ats.append((str(ROOT / path), int(line), token))
    c = Clangd(OUT_DIR)
    try:
        secs = c.start(ats[0][0] if ats else first)
        print(f"clangd_refs: index ready in {secs:.0f} s")
        for path, line, token in ats:
            text = pathlib.Path(path).read_text(errors="replace").split("\n")[line - 1]
            col = text.find(token)
            if col < 0:
                print(f"{rel(path)}:{line}: token '{token}' not on that line")
                continue
            report(f"{token} at {rel(path)}:{line}", c.references(path, line - 1, col))
        for sym in a.sym:
            leaf = sym.rpartition("::")[2]
            infos = c.request("workspace/symbol", {"query": sym}) or []
            seen = set()
            for s in infos:
                container = s.get("containerName") or ""
                if not symbol_matches(sym, s["name"], container) or (container, s["name"]) in seen:
                    continue
                seen.add((container, s["name"]))
                loc = s["location"]
                path, line0 = loc["uri"][len("file://"):], loc["range"]["start"]["line"]
                col = pathlib.Path(path).read_text(errors="replace").split("\n")[line0].find(leaf)
                col = col if col >= 0 else loc["range"]["start"]["character"]
                qual = f"{container}::{s['name']}" if container else s["name"]
                kind = KINDS.get(s.get("kind"), str(s.get("kind")))
                report(f"{qual} ({kind}, {rel(path)}:{line0 + 1})", c.references(path, line0, col))
            if not seen:
                print(f"{sym}: no symbol of that name in the index")
    finally:
        c.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
