# Tools + JSON-Schema coordination

Status: design approved 2026-05-07. Implementation pending.

## Problem

When an OpenAI-compatible request sets both `tools` and `response_format=json_schema`/`json_object`, imp currently drops `response_format` with a warning at `tools/imp-server/handlers.cpp:628-633`. The reason: the schema mask would block the `<` token that opens `<tool_call>` (ChatML/Hermes/Mistral), `<|tool_call>` (Gemma), and `<function=` (Llama3) — the model could not emit any tool call.

This is a workaround, not a fix. The fix is runtime coordination: the schema mask should apply only when the model is *not* in a tool call.

## Goal

Make schema-constrained decoding apply only to free-text JSON output. Tool-call output (token sequences enclosed by the model's tool-tag dialect) bypasses the schema FSM entirely. After the first tool call closes, the mask stays off for the remainder of the generation, supporting parallel tool calls.

In scope: `tool_choice: "auto"` (the only mode where coordination is needed). `"none"` keeps existing schema-only behavior; `"required"` and forced-function suppress schema (already work without coordination). Tool-argument schema enforcement (constraining the body of a tool call to its `parameters` schema) is out of scope.

## Architecture

The existing `imp::PreambleGate` (`src/compute/preamble_gate.h`) sits in front of every JSON/schema FSM. It already supports two exit conditions for free-form preambles: a `</think>` close-token (reasoning models) and a budget cap with `{`/`[` early-exit (markdown fences and short verbal openers). We extend it to a tri-state machine that is also tool-aware.

```
Request → Handler                          Engine
─────────────────────────────────────────────────────────────────────
parse tools + response_format              ConstraintManager::prepare(
        │                                     json_mode, json_schema,
  tools && schema?                            tokenizer,
        │                                     /* NEW */ has_tools,
   set req->has_tools = true                  /* NEW */ tpl_family)
        │                                          │
   keep req->json_schema                     SchemaConstrainer::set_preamble(
        │                                       think_close, budget,
   submit                                       /* NEW */ tool_open_tokens,
                                                /* NEW */ tool_close_tokens,
                                                /* NEW */ tool_open_prefix,
                                                /* NEW */ tool_close_suffix)
                                                  │
                                            PreambleGate (tri-state):
                                              ACTIVE → TOOL_BODY → OFF
```

### Tri-state PreambleGate

```
ACTIVE   ── tool-opener-token sampled ──────────→ TOOL_BODY
ACTIVE   ── tool-open-prefix matched (chars) ───→ TOOL_BODY
ACTIVE   ── '{' or '[' in token text ───────────→ OFF (preamble exit, FSM enforces)
ACTIVE   ── close_token (</think>) sampled ─────→ OFF (preamble exit, FSM enforces)
ACTIVE   ── budget exhausted ───────────────────→ OFF (preamble exit, FSM enforces)
TOOL_BODY── tool-close-token sampled ───────────→ TERMINAL_OFF (no FSM ever)
TOOL_BODY── tool-close-suffix matched (chars) ──→ TERMINAL_OFF (no FSM ever)
```

`OFF` from preamble exit means the FSM mask kicks in normally for the rest of the generation. `TERMINAL_OFF` from a tool close means no mask is applied for the rest of the generation — supporting EOS, parallel tool calls, and short trailing text.

The existing `absorb()` semantics are preserved: returns `true` if the gate consumed the token (FSM should not see it), `false` if forwarded.

## Components

| File | Change |
|---|---|
| `src/runtime/engine.h` (Request struct) | Add `bool has_tools = false` and `imp::ChatTemplateFamily tpl_family = CHATML`. |
| `src/compute/preamble_gate.h` | Tri-state: enum `Gate::State { ACTIVE, TOOL_BODY, OFF, TERMINAL_OFF }`. New `configure(...)` overload accepting `std::vector<int32_t> tool_open_tokens, tool_close_tokens` and `std::string tool_open_prefix, tool_close_suffix`. Internal small char-buffer (≤16 chars) for char-level prefix/suffix matching across token boundaries. |
| `src/runtime/constraint_manager.h/.cpp` | `prepare(...)` gets two new params: `bool has_tools, ChatTemplateFamily family`. Resolves dialect-specific tag tokens via `tokenizer->find_token(...)`. Bumps preamble budget to 64 tokens when `has_tools && (json_mode \|\| !schema.empty())`. |
| `src/compute/schema_constrain.h/.cu` | `set_preamble(...)` overload threading the new parameters down to the gate. `apply_mask` already early-returns when gate is non-OFF — no mask-logic change. New: `apply_mask` also early-returns on `TERMINAL_OFF`. |
| `src/compute/json_constrain.h/.cu` | Same overload. |
| `tools/imp-server/handlers.cpp:628-633` | Replace "drop response_format" branch with: `req->has_tools = has_tools; req->tpl_family = tpl_family;`. Drop the warning log. |
| `tests/test_preamble_gate.cpp` (new) | Unit tests for every transition × every dialect. |
| `tests/test_schema_constrain.cpp` | Add `has_tools=true` cases: synthetic `<tool_call>{...}</tool_call>` body bypasses FSM mask. |
| `tests/test_server_*.cpp` (existing integration test files) | E2e: tools+json_schema request, mock model emits known tool-call token sequence, response surfaces `tool_calls` (no 500, no dropped warning). |

### Dialect → tag resolution

Computed in `ConstraintManager::prepare(...)` using `tpl_family`:

| Family | Open string | Close string | Token resolution |
|---|---|---|---|
| ChatML / Hermes / Mistral | `<tool_call>` | `</tool_call>` | Single token in vocab; resolves via `find_token()` |
| Gemma | `<\|tool_call>` | `<tool_call\|>` | Single token in vocab; resolves via `find_token()` |
| Llama3 | `<function=` | `</function>` | Multi-token; `find_token()` returns -1, char-prefix/suffix used |

For each family, the strings always populate `tool_open_prefix` / `tool_close_suffix` as the cheap char-level fallback (≤10 chars). Token IDs populate the token sets only when they resolve. Both are checked in `absorb()` — token-set membership first (O(1) hash), then char-buffer prefix match.

### Char-level prefix/suffix matching

Inside the gate, a small ring buffer (16 chars) of recently seen decoded text is maintained. After each `absorb(token, text)`:

- In `ACTIVE` state: append `text` to buffer (truncate to last 16 chars). Check if `tool_open_prefix` is a substring. If yes → transition to `TOOL_BODY`, clear buffer.
- In `TOOL_BODY` state: append `text` to buffer. Check if `tool_close_suffix` is a substring. If yes → transition to `TERMINAL_OFF`.

Buffer cost: 16 bytes plus a `find()` per token. Negligible.

## Data flow

1. Handler parses body. If `has_tools && (json_mode || !json_schema.empty())`: set `req->has_tools = true`, `req->tpl_family = tpl_family`. Keep `json_mode` / `json_schema_str` set.
2. `engine.cpp:1835` calls `constraints_.prepare(json_mode, json_schema, tokenizer, has_tools, tpl_family)`.
3. ConstraintManager resolves dialect tags (token + char-prefix). Sets gate budget to 64 if `has_tools`, else 8 / 8192 unchanged.
4. Per decode step, `apply_mask`:
   - Gate `ACTIVE` or `TOOL_BODY` → no mask, all tokens pass.
   - Gate `OFF` (preamble exited via `{`/`[`/think-close/budget) → schema/json FSM mask kicks in.
   - Gate `TERMINAL_OFF` (tool close seen) → no mask for rest of generation.
5. Per decode step, `update(token)` advances both gate and FSM (if FSM is non-OFF).

## Edge cases

- **Reasoning model + tools + schema** (Qwen3.6, Gemma-4 thinking): `</think>` close-token does not directly transition out — gate stays ACTIVE with fresh budget=64 in the post-think window. Tool-opener detection runs in post-think state. If the model goes straight to a tool call without thinking, close-token never fires; tool-opener still detected.
- **Parallel tool calls**: After first close → `TERMINAL_OFF`. Subsequent `<tool_call>...` openers pass unmasked. Handler-side scanner already handles multiple calls.
- **Model emits free-text JSON instead of tool**: `ACTIVE` → `{` seen → FSM kicks in → schema enforced. Standard schema path.
- **Model emits short prose then tool**: `ACTIVE` → "I'll help" (≤64 tokens) → `<tool_call>` opener → `TOOL_BODY`. If prose exceeds 64 tokens, budget exhausts → FSM kicks in → schema masks `<` → tool cannot fire. Mitigation: bumpable via `IMP_TOOL_PREAMBLE_BUDGET` env var.
- **Tool body contains `</tool_call>` inside a string literal**: not handled; same vulnerability as handler-side scanner. Models trained on this format do not produce such bodies.
- **Empty tool body** (`<tool_call></tool_call>`): parsed as zero-arg call by handler — already supported.
- **EOS during `TOOL_BODY`**: generation ends, handler emits whatever was buffered. No new gate behavior required.
- **Tokenizer has no dialect tokens AND char-prefix never matches**: at `prepare()`, if `has_tools && tpl_family` resolves zero tag tokens and the family has no char-prefix string, log a warning and fall back to the current behavior (clear `json_mode`/`json_schema_str`). Strictly no-worse-than-today for unsupported tokenizers.
- **CUDA-graph capture**: gate is host-side state, not part of the captured device graph — no interaction.

## Testing

- Unit: `tests/test_preamble_gate.cpp` (new) — synthetic tokenizer with mock tag IDs, drive every transition × every dialect:
  - ACTIVE → TOOL_BODY via token, via char-prefix.
  - ACTIVE → OFF via `{`, via `[`, via `</think>`, via budget exhaust.
  - TOOL_BODY → TERMINAL_OFF via close-token, via char-suffix.
  - Parallel calls: TERMINAL_OFF stays terminal across multiple synthetic openers.
- Unit: `tests/test_schema_constrain.cpp` extension — schema FSM stays inactive across a synthetic `<tool_call>{...}</tool_call>` token sequence; activates correctly on free-text `{...}`.
- Integration: existing server tests (`tests/test_server_*.cpp`) — POST with both `tools` and `response_format`, deterministic-seed mock model emits known tool-call sequence, response contains `tool_calls`, no 500.
- Regression: ensure existing reasoning + JSON-schema tests still pass (preamble close-token path).

## Out of scope

- Tool-argument schema enforcement (`parameters` schema constraining the tool-call body during sampling). Continues to be post-hoc validation in handler.
- Streaming SSE handler-side scanner reorganisation. The handler keeps its existing CONTENT/TAG_SCANNING/TOOL_CALL_BODY state machine; engine-side gate is independent.
- Tool-choice modes other than `auto`. `none` and `required`/forced are unaffected by this design.
