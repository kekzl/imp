<!--
layer: L1
audience: operators
verified: 2026-11-01
commit: 216a779c
-->

# Codex CLI on imp

Codex talks `/v1/responses`. Tested: `codex-cli 0.163.0`, Qwen3.8-Flash-Next-NVFP4, RTX 5090.

## Provider

`~/.codex/config.toml`:

```toml
model = "Qwen3.8-Flash-Next-NVFP4"          # the id from GET /v1/models
model_provider = "imp"
model_catalog_json = "/home/me/.codex/imp-models.json"

[model_providers.imp]
name = "imp"
base_url = "http://localhost:8080/v1"
wire_api = "responses"
```

## Model catalog (needed)

| Without a catalog entry | Effect |
|---|---|
| Codex warns "Model metadata for ... not found" | it runs the model on fallback metadata |
| fallback metadata | no `apply_patch` tool, while Codex's instructions tell the model to use one |
| the model | calls tools that do not exist (`apply_patch`, `shell`, `exec`, `view`) and loops |

`imp-models.json`, one entry per model id:

```json
{
  "models": [
    {
      "slug": "Qwen3.8-Flash-Next-NVFP4",
      "display_name": "Qwen3.8-Flash-Next-NVFP4",
      "apply_patch_tool_type": "freeform",
      "context_window": 131072,
      "supported_reasoning_levels": [
        {"effort": "low", "description": "low"},
        {"effort": "medium", "description": "medium"},
        {"effort": "high", "description": "high"}
      ],
      "shell_type": "default",
      "visibility": "list",
      "supported_in_api": true,
      "priority": 1,
      "support_verbosity": false,
      "truncation_policy": {"mode": "tokens", "limit": 10000},
      "experimental_supported_tools": [],
      "base_instructions": "<Codex's base instructions, see below>"
    }
  ]
}
```

- `base_instructions` is required. Copy Codex's own text (the first `developer` input item of any Codex request, "You are a coding agent running in the Codex CLI ..."), or write your own.
- `context_window`: Flash-Next serves up to 131072 tokens; its KV pool grows on demand ([`BIG_MODELS.md`](BIG_MODELS.md#after-start)).

## Measured

`codex exec`, five tasks (script, bug fix + unittest, multi-file package, fix at `model_reasoning_effort=high`, `-i` screenshot), a fresh server per round, verdict from the files Codex leaves:

| Setup | Passed | Image task |
|---|---|---|
| v0.56.0, no catalog | 7/10 | 0/2 (loops, one host OOM kill) |
| system-message fix (#2837), no catalog | 9/10 | 1/2 |
| #2837 + catalog | 9/10 | 1/2 (an `apply_patch` call ended at EOS without `</tool_call>` shipped as text) |
| #2837 + EOS tool-call parse + catalog | 10/10 | 2/2 |

## Host RAM

Qwen3.8-Flash-Next holds 67.3-67.5 GiB of host RAM under this load (host experts). On a 78 GiB host another few-GB job triggered a kernel OOM kill of imp-server: keep the rest of the box light.
