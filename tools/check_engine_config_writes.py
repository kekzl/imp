#!/usr/bin/env python3
"""AUDIT_ARCH_2026_11_24 C-9 / B-129 gate: Engine::config_ keeps what the operator set.

Init resolves into EngineRuntimeState rt_ (src/runtime/engine_config.h); until 2026-11-27 it wrote
26 resolved values of 13 fields back into config_ (22 direct, 4 through KvInitCtx::ec). Fails on:
  1. a field write on config_ (=, op=, ++/--, .reset()/.emplace()) in an Engine source of src/runtime
  2. a non-const EngineConfig& or EngineConfig* in src/runtime outside ALLOW (the imp.conf bridge)

Usage:
    python3 tools/check_engine_config_writes.py
    python3 tools/check_engine_config_writes.py --selftest
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
ROOT = "src/runtime"
SUFFIXES = (".cpp", ".cu", ".h", ".cuh")
# (file, text on the line) -> reason. Only shrinks.
ALLOW = {
    ("src/runtime/engine_config.h", "bridge_runtime_config"): "fills unset optionals from imp.conf, the operator's",
    ("src/runtime/engine_config.cpp", "bridge_runtime_config"): "same function, its definition",
}

COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
STRING = re.compile(r'"(?:\\.|[^"\\\n])*"')
FIELD = r"(?<![\w.>])config_\.\w+(?:\.\w+(?:\(\))?)*"
WRITE = re.compile(
    rf"{FIELD}\s*(?:[-+*/|&^]|<<|>>)?=(?!=)|{FIELD}\s*(?:\+\+|--)|(?:\+\+|--)\s*{FIELD}|{FIELD}\.(?:reset|emplace)\s*\("
)
MUT_REF = re.compile(r"(?<![\w:])(?:imp::)?EngineConfig\s*(?:&(?!&)|\*)")
CONST_BEFORE = re.compile(r"\bconst\s+(?:imp::)?$")
ENGINE_DEF = re.compile(r"\bEngine::")


def strip(text: str) -> str:
    """Comments and string literals blanked, newlines kept (line numbers stay)."""
    blank = lambda m: re.sub(r"[^\n]", " ", m.group(0))  # noqa: E731
    return COMMENT.sub(blank, STRING.sub(blank, text))


def engine_source(path: str, code: str) -> bool:
    return pathlib.PurePosixPath(path).name.startswith("engine") or bool(ENGINE_DEF.search(code))


def check(files: dict[str, str]) -> list[str]:
    errors = []
    for path in sorted(files):
        raw_lines = files[path].splitlines()
        code = strip(files[path])
        lines = code.splitlines()
        if engine_source(path, code):
            for m in WRITE.finditer(code):
                n = code.count("\n", 0, m.start())
                errors.append(f"{path}:{n + 1}: writes config_ ({m.group(0).strip()}); resolve into rt_ (C-9)")
        for m in MUT_REF.finditer(code):
            n = code.count("\n", 0, m.start())
            line_start = code.rfind("\n", 0, m.start()) + 1
            if CONST_BEFORE.search(code[line_start:m.start()]):
                continue
            raw = raw_lines[n] if n < len(raw_lines) else lines[n]
            if any(p == path and key in raw for (p, key) in ALLOW):
                continue
            errors.append(f"{path}:{n + 1}: non-const EngineConfig reference or pointer; take const and write rt_")
    for (p, key) in ALLOW:
        if p in files and key not in files[p]:
            errors.append(f"ALLOW row ({p}, {key}) matches nothing; drop it")
    return errors


def repo_files() -> dict[str, str]:
    out = {}
    for path in sorted((REPO / ROOT).rglob("*")):
        if path.is_file() and path.suffix in SUFFIXES:
            out[path.relative_to(REPO).as_posix()] = path.read_text(encoding="utf-8", errors="replace")
    return out


def selftest() -> int:
    base = {
        "src/runtime/engine_config.h": "void bridge_runtime_config(EngineConfig& ec, const RuntimeConfig& rc);\n",
        "src/runtime/engine_config.cpp": "void bridge_runtime_config(EngineConfig& ec, const RuntimeConfig& rc) {}\n",
        "src/runtime/engine_x.cpp": "void Engine::f() {\n    rt_.use_fp8_prefill = false;\n"
        "    if (config_.use_fp8_prefill == true && config_.max_seq_len != 0 && config_.a <= 1) {}\n"
        "    // config_.kv_block_size = 16;\n    log(\"config_.x = 1\");\n"
        "    model_->config_.n_layers = 3;\n    runtime_config_.gemm.x = 1;\n"
        "    const EngineConfig& c = config_;\n    f(EngineConfig&& t);\n}\n",
        "src/runtime/cuda_graph.cu": "void CudaGraph::g() { config_.step_limit = 3; }\n",
    }
    cases = [
        ("clean tree", {}, 0),
        ("plain write", {"src/runtime/engine_x.cpp": "void Engine::f() { config_.use_fp8_prefill = false; }\n"}, 1),
        ("compound and increment", {"src/runtime/engine_y.cpp": "config_.a |= 1;\nconfig_.b++;\n--config_.c;\n"}, 3),
        ("optional reset and emplace", {"src/runtime/engine_y.cpp": "config_.a.reset();\nconfig_.b.emplace(3);\n"}, 2),
        ("nested field write", {"src/runtime/engine_y.cpp": "config_.mmproj_path.value() = x;\n"}, 1),
        ("write in a non-Engine file is not ours", {"src/runtime/other.cpp": "void G::h() { config_.x = 1; }\n"}, 0),
        ("Engine file without the prefix", {"src/runtime/other.cpp": "void Engine::h() { config_.x = 1; }\n"}, 1),
        ("non-const reference member", {"src/runtime/engine_y.cpp": "struct K { EngineConfig& ec; };\n"}, 1),
        ("non-const pointer", {"src/runtime/engine_y.cpp": "imp::EngineConfig* p = &config_;\n"}, 1),
        ("const reference and pointer", {"src/runtime/engine_y.cpp": "const EngineConfig& a; const imp::EngineConfig* b;\n"}, 0),
        ("stale allow row", {"src/runtime/engine_config.cpp": "int x;\n"}, 1),
    ]
    bad = 0
    for name, extra, want in cases:
        errors = check({**base, **extra})
        if len(errors) != want:
            print(f"selftest: {name}: want {want} error(s), got {len(errors)}: {errors}")
            bad += 1
    print(f"selftest: {len(cases) - bad}/{len(cases)} cases ok")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        return selftest()
    files = repo_files()
    errors = check(files)
    for e in errors:
        print(e)
    n_engine = sum(1 for p, t in files.items() if engine_source(p, strip(t)))
    print(f"engine config writes: {n_engine} Engine sources of {len(files)} in {ROOT}, {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
