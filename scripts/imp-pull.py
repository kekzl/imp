#!/usr/bin/env python3
"""imp-pull — Model manager for imp inference engine.

Reads presets.toml as source of truth for available models.
Downloads GGUF files from HuggingFace Hub.

Usage:
    python imp-pull.py              # List all models + status
    python imp-pull.py pull <id>    # Download model
    python imp-pull.py delete <id>  # Delete model (with confirmation)
"""

import os
import re
import sys
import shutil
from pathlib import Path

# Map preset IDs to HuggingFace repos and filenames.
# This is the download registry — presets.toml defines runtime config,
# this dict defines where to fetch the weights.
MODEL_REGISTRY = {
    # ── Top-5 flagship ──
    "qwen3-coder-next": {
        "repo": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        "file": "Qwen3-Coder-30B-A3B-Instruct-Q6_K.gguf",
    },
    "qwen3-32b": {
        "repo": "unsloth/Qwen3-32B-GGUF",
        "file": "Qwen3-32B-Q4_K_M.gguf",
    },
    "glm-4.7-flash": {
        "repo": "unsloth/GLM-4.7-Flash-GGUF",
        "file": "GLM-4.7-Flash-Q6_K.gguf",
    },
    "deepseek-r1-32b": {
        "repo": "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        "file": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
    },
    "mimo-v2-flash": {
        "repo": "unsloth/MiMo-7B-RL-GGUF",
        "file": "MiMo-7B-RL-Q8_0.gguf",
    },
    # ── Qwen3 family ──
    "qwen3-8b": {
        "repo": "unsloth/Qwen3-8B-GGUF",
        "file": "Qwen3-8B-Q8_0.gguf",
    },
    "qwen3-4b": {
        "repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        "file": "Qwen3-4B-Instruct-2507-Q8_0.gguf",
    },
    # ── Qwen3-Coder dense ──
    "qwen3-coder-32b": {
        "repo": "unsloth/Qwen3-Coder-32B-Instruct-GGUF",
        "file": "Qwen3-Coder-32B-Instruct-Q4_K_M.gguf",
    },
    "qwen3-coder-14b": {
        "repo": "unsloth/Qwen3-Coder-14B-Instruct-GGUF",
        "file": "Qwen3-Coder-14B-Instruct-Q8_0.gguf",
    },
    "qwen3-coder-7b": {
        "repo": "unsloth/Qwen3-Coder-7B-Instruct-GGUF",
        "file": "Qwen3-Coder-7B-Instruct-Q8_0.gguf",
    },
    # ── Gemma-3 ──
    "gemma3-27b": {
        "repo": "unsloth/gemma-3-27b-it-GGUF",
        "file": "gemma-3-27b-it-Q4_K_M.gguf",
    },
    "gemma3-12b": {
        "repo": "unsloth/gemma-3-12b-it-GGUF",
        "file": "gemma-3-12b-it-Q8_0.gguf",
    },
    "gemma3-4b": {
        "repo": "unsloth/gemma-3-4b-it-GGUF",
        "file": "gemma-3-4b-it-Q8_0.gguf",
    },
    # ── LLaMA 3 ──
    "llama3-70b": {
        "repo": "unsloth/Meta-Llama-3.1-70B-Instruct-GGUF",
        "file": "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
    },
    "llama3-8b": {
        "repo": "unsloth/Meta-Llama-3.1-8B-Instruct-GGUF",
        "file": "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
    },
    # ── Mistral / Mixtral ──
    "mistral-24b": {
        "repo": "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
        "file": "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf",
    },
    "mistral-7b": {
        "repo": "unsloth/mistral-7b-instruct-v0.3-GGUF",
        "file": "mistral-7b-instruct-v0.3-Q8_0.gguf",
    },
    "mixtral-8x7b": {
        "repo": "unsloth/Mixtral-8x7B-Instruct-v0.1-GGUF",
        "file": "Mixtral-8x7B-Instruct-v0.1-Q4_K_M.gguf",
    },
    # ── Coding ──
    "devstral": {
        "repo": "unsloth/Devstral-Small-2507-GGUF",
        "file": "Devstral-Small-2507-Q4_K_M.gguf",
    },
    "codestral-25b": {
        "repo": "unsloth/Codestral-2501-GGUF",
        "file": "Codestral-2501-Q4_K_M.gguf",
    },
    "qwen2.5-coder-32b": {
        "repo": "unsloth/Qwen2.5-Coder-32B-Instruct-GGUF",
        "file": "Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf",
    },
    "qwen2.5-coder-7b": {
        "repo": "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF",
        "file": "Qwen2.5-Coder-7B-Instruct-Q8_0.gguf",
    },
    "deepseek-coder": {
        "repo": "unsloth/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        "file": "DeepSeek-Coder-V2-Lite-Instruct-Q6_K.gguf",
    },
    # ── Other ──
    "nemotron-h": {
        "repo": "unsloth/Nemotron-3-Nano-30B-A3B-GGUF",
        "file": "Nemotron-3-Nano-30B-A3B-Q6_K.gguf",
    },
}

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models"))


def parse_presets_toml(path: Path) -> dict:
    """Parse presets.toml using imp's simple TOML format (not strict TOML).

    imp's C++ parser treats [section.with.dots] as a flat key, not nested tables.
    Python's tomllib would interpret dots as subtable separators, so we use a
    line-based parser matching the C++ behavior.
    """
    result = {}
    current_section = ""
    current_desc = ""

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Section header: [name]
            m = re.match(r"^\[([^\]]+)\]$", line)
            if m:
                if current_section and current_section != "defaults":
                    result[current_section] = current_desc or current_section
                current_section = m.group(1).strip()
                current_desc = ""
                continue

            # Key = value
            if "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                if key == "description" and val.startswith('"') and val.endswith('"'):
                    current_desc = val[1:-1]

    # Flush last section
    if current_section and current_section != "defaults":
        result[current_section] = current_desc or current_section

    return result


def load_presets() -> dict:
    """Load presets.toml and return {id: description} for known models."""
    search = [
        Path("/app/presets.toml"),
        Path(__file__).parent.parent / "presets.toml",
    ]
    for p in search:
        if p.is_file():
            return parse_presets_toml(p)
    print("Warning: presets.toml not found, using built-in registry", file=sys.stderr)
    return {k: k for k in MODEL_REGISTRY}


def fmt_size(size_bytes: int) -> str:
    if size_bytes >= 1_073_741_824:
        return f"{size_bytes / 1_073_741_824:.1f} GB"
    return f"{size_bytes / 1_048_576:.0f} MB"


def cmd_list():
    presets = load_presets()
    print(f"{'ID':<24} {'Description':<48} {'Status':<16}")
    print("-" * 88)
    for preset_id, desc in presets.items():
        reg = MODEL_REGISTRY.get(preset_id)
        if not reg:
            status = "no download"
        else:
            path = MODELS_DIR / reg["file"]
            if path.exists():
                status = f"ready ({fmt_size(path.stat().st_size)})"
            else:
                status = "not downloaded"
        print(f"{preset_id:<24} {desc:<48} {status:<16}")


def cmd_pull(model_id: str):
    if model_id not in MODEL_REGISTRY:
        print(f"Error: Unknown model '{model_id}'")
        print(f"Available: {', '.join(sorted(MODEL_REGISTRY.keys()))}")
        sys.exit(1)

    reg = MODEL_REGISTRY[model_id]
    dest = MODELS_DIR / reg["file"]

    if dest.exists():
        print(f"Model already exists: {dest} ({fmt_size(dest.stat().st_size)})")
        return

    # Check free space
    disk = shutil.disk_usage(MODELS_DIR)
    free_gb = disk.free / 1_073_741_824
    print(f"Free disk space: {free_gb:.1f} GB")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub not installed.")
        print("  pip install huggingface_hub")
        sys.exit(1)

    print(f"Downloading {reg['repo']} / {reg['file']} ...")
    path = hf_hub_download(
        repo_id=reg["repo"],
        filename=reg["file"],
        local_dir=str(MODELS_DIR),
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded: {path}")
    print()
    print("Add to .env:")
    print(f"  IMP_MODEL=/models/{reg['file']}")


def cmd_delete(model_id: str):
    if model_id not in MODEL_REGISTRY:
        print(f"Error: Unknown model '{model_id}'")
        sys.exit(1)

    reg = MODEL_REGISTRY[model_id]
    path = MODELS_DIR / reg["file"]

    if not path.exists():
        print(f"Model not found: {path}")
        sys.exit(1)

    size = fmt_size(path.stat().st_size)
    confirm = input(f"Delete {path} ({size})? [y/N] ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    path.unlink()
    print(f"Deleted: {path}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("list", "-l", "--list"):
        cmd_list()
    elif sys.argv[1] == "pull":
        if len(sys.argv) < 3:
            print("Usage: imp-pull.py pull <model-id>")
            sys.exit(1)
        cmd_pull(sys.argv[2])
    elif sys.argv[1] == "delete":
        if len(sys.argv) < 3:
            print("Usage: imp-pull.py delete <model-id>")
            sys.exit(1)
        cmd_delete(sys.argv[2])
    else:
        print(f"Unknown command: {sys.argv[1]}")
        print("Commands: list, pull <id>, delete <id>")
        sys.exit(1)


if __name__ == "__main__":
    main()
