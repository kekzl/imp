# Tools + JSON-Schema Coordination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `response_format=json_schema`/`json_object` apply only to free-text JSON output when `tools` is also set, so tool calls bypass the schema mask entirely. Stop unconditionally dropping `response_format` in the handler.

**Architecture:** Extend `imp::PreambleGate` to a tri-state FSM (`ACTIVE → TOOL_BODY → TERMINAL_OFF`) with token-set + char-prefix tool-tag detection. ChatML/Hermes/Mistral resolve `<tool_call>`/`</tool_call>` as single special tokens; Llama3 falls back to `<function=`/`</function>` char-prefix. Gemma uses `<|tool_call>`/`<tool_call|>` single-token. The gate's external API (`active()`, `absorb()`) stays binary; the four internal states differ only in transitions.

**Tech Stack:** C++20, CUDA 13.2, GTest, sm_120a target. Build via `make build` (Docker) or `cmake -B build && cmake --build build -j$(nproc)`. Tests: `make test-unit` (CPU) / `make test-gpu` (full).

**Spec:** `docs/superpowers/specs/2026-05-07-tools-schema-coordination-design.md` (commit `105dfbf`).

---

## File Map

- **Modify** `src/runtime/request.h` — add `has_tools` + `tpl_family` fields
- **Modify** `src/compute/preamble_gate.h` — tri-state FSM, tool-aware config
- **Modify** `src/compute/json_constrain.h` + `src/compute/json_constrain.cu` — `set_preamble` overload threading tool config
- **Modify** `src/compute/schema_constrain.h` + `src/compute/schema_constrain.cu` — same overload
- **Modify** `src/runtime/constraint_manager.h` + `src/runtime/constraint_manager.cpp` — `prepare(...)` accepts `has_tools` + `tpl_family`, resolves dialect tags
- **Modify** `src/runtime/engine.cpp:1835` — pass new params to ConstraintManager
- **Modify** `tools/imp-server/handlers.cpp:617-633` — set `req->has_tools` / `req->tpl_family`, drop the early return that nukes `json_mode`/`json_schema_str`
- **Modify** `tests/test_json_constrain.cu` — extend `PreambleGateTest` with tri-state cases
- **Add** `tests/api/test_tools_with_schema.py` (or extend `test_tools.py`) — integration test, marker-gated

---

## Task 1: Add Request fields for tool coordination

**Files:**
- Modify: `src/runtime/request.h:79`

- [ ] **Step 1: Add includes**

`src/runtime/request.h` already includes `<string>`. No new include needed (we use `imp::ChatTemplateFamily` which lives in `model/chat_template.h`).

Add this include near the top of the file, after the existing `#include <string>`:

```cpp
#include "model/chat_template.h"
```

- [ ] **Step 2: Add the two fields**

In `src/runtime/request.h`, after line 78 (`std::string json_schema;`), add:

```cpp
    // Tool-call coordination: when true and (json_mode || !json_schema.empty()),
    // the preamble gate enters tool-aware mode so the schema/JSON FSM mask
    // does not block the model's tool-tag opener (`<tool_call>`, `<|tool_call>`,
    // `<function=`).
    bool has_tools = false;
    ChatTemplateFamily tpl_family = ChatTemplateFamily::CHATML;
```

- [ ] **Step 3: Build (host)**

Run:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc) --target imp-core 2>&1 | tail -20
```
Expected: builds clean. If unknown target, just build everything: `cmake --build build -j$(nproc) 2>&1 | tail -20` — expect success.

- [ ] **Step 4: Commit**

```bash
git add src/runtime/request.h
git commit -m "$(cat <<'EOF'
feat(runtime): add has_tools + tpl_family to Request struct

Foundation for tools + JSON-schema coordination: the engine-side gate
needs to know both whether tools are active and which dialect's tool
tags to look for. Default is no-tools / CHATML, matching today's
behaviour for callers that don't set the new fields.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Failing tests for PreambleGate tool-opener token transition

**Files:**
- Modify: `tests/test_json_constrain.cu` (append new tests)

- [ ] **Step 1: Add test constants and tri-state token-opener tests**

Append to `tests/test_json_constrain.cu`, after the existing `PreambleGateTest` block (search for `TEST(PreambleGateTest, ResetReactivatesGate)` and add the new tests after that test's closing brace):

```cpp
// ===========================================================================
// Tool-aware tri-state tests
// ===========================================================================

constexpr int32_t TOK_TOOL_OPEN = 400;   // synthetic <tool_call>
constexpr int32_t TOK_TOOL_CLOSE = 401;  // synthetic </tool_call>

TEST(PreambleGateTest, ToolOpenerTokenTransitionsToToolBody) {
    PreambleGate g;
    g.configure_with_tools(TOK_THINK_CLOSE, /*budget=*/64,
                           /*open_tokens=*/{TOK_TOOL_OPEN},
                           /*close_tokens=*/{TOK_TOOL_CLOSE},
                           /*open_prefix=*/"",
                           /*close_suffix=*/"");
    EXPECT_TRUE(g.active());

    // Free-form preamble before tool: still absorbed, still active.
    EXPECT_TRUE(g.absorb(TOK_TEXT, "Sure! "));
    EXPECT_TRUE(g.active());

    // Opener token: absorbed, gate stays "not masking" but is now in TOOL_BODY.
    EXPECT_TRUE(g.absorb(TOK_TOOL_OPEN, "<tool_call>"));
    EXPECT_TRUE(g.active());  // active() still means "no mask"

    // Tool body content (including `{`!) does NOT trigger preamble exit
    // anymore — we are inside a tool body.
    EXPECT_TRUE(g.absorb(TOK_OPEN_BRACE, "{"));
    EXPECT_TRUE(g.active());
    EXPECT_TRUE(g.absorb(TOK_TEXT, "\"name\": \"x\"}"));
    EXPECT_TRUE(g.active());
}

TEST(PreambleGateTest, ToolCloseTokenTransitionsToTerminalOff) {
    PreambleGate g;
    g.configure_with_tools(TOK_THINK_CLOSE, 64,
                           {TOK_TOOL_OPEN}, {TOK_TOOL_CLOSE}, "", "");
    g.absorb(TOK_TOOL_OPEN, "<tool_call>");
    g.absorb(TOK_TEXT, "{...}");

    // Close token: absorbed, terminal OFF.
    EXPECT_TRUE(g.absorb(TOK_TOOL_CLOSE, "</tool_call>"));
    EXPECT_TRUE(g.active());  // TERMINAL_OFF still reads as "no mask"

    // Subsequent tokens — including `{` — are absorbed, FSM never re-engages.
    EXPECT_TRUE(g.absorb(TOK_OPEN_BRACE, "{"));
    EXPECT_TRUE(g.active());
    EXPECT_TRUE(g.absorb(TOK_TEXT, "free text after"));
    EXPECT_TRUE(g.active());
}

TEST(PreambleGateTest, ToolModeStillExitsOnJsonStartIfNoTool) {
    // Model emits free-text JSON instead of a tool call: gate exits to FSM
    // exactly like non-tool mode.
    PreambleGate g;
    g.configure_with_tools(TOK_THINK_CLOSE, 64,
                           {TOK_TOOL_OPEN}, {TOK_TOOL_CLOSE}, "", "");
    EXPECT_TRUE(g.active());
    EXPECT_FALSE(g.absorb(TOK_OPEN_BRACE, "{"));
    EXPECT_FALSE(g.active());  // OFF (preamble exit), FSM enforces
}

TEST(PreambleGateTest, ToolModeBudgetExhaustExitsToFsm) {
    // Long preamble without a tool opener: budget exhausts, FSM kicks in.
    PreambleGate g;
    g.configure_with_tools(/*close_token=*/-1, /*budget=*/3,
                           {TOK_TOOL_OPEN}, {TOK_TOOL_CLOSE}, "", "");
    EXPECT_TRUE(g.absorb(TOK_TEXT, "blah"));
    EXPECT_TRUE(g.active());
    EXPECT_TRUE(g.absorb(TOK_TEXT, "blah"));
    EXPECT_TRUE(g.active());
    EXPECT_TRUE(g.absorb(TOK_TEXT, "blah"));
    EXPECT_FALSE(g.active());  // budget exhausted → FSM kicks in
}

TEST(PreambleGateTest, ToolModeParallelCallsStayTerminalOff) {
    PreambleGate g;
    g.configure_with_tools(TOK_THINK_CLOSE, 64,
                           {TOK_TOOL_OPEN}, {TOK_TOOL_CLOSE}, "", "");
    g.absorb(TOK_TOOL_OPEN, "<tool_call>");
    g.absorb(TOK_TEXT, "{a}");
    g.absorb(TOK_TOOL_CLOSE, "</tool_call>");
    EXPECT_TRUE(g.active());

    // Second tool call — opener and body both absorbed in TERMINAL_OFF.
    EXPECT_TRUE(g.absorb(TOK_TOOL_OPEN, "<tool_call>"));
    EXPECT_TRUE(g.absorb(TOK_TEXT, "{b}"));
    EXPECT_TRUE(g.absorb(TOK_TOOL_CLOSE, "</tool_call>"));
    EXPECT_TRUE(g.active());
}

TEST(PreambleGateTest, ToolModeResetReturnsToActive) {
    PreambleGate g;
    g.configure_with_tools(TOK_THINK_CLOSE, 64,
                           {TOK_TOOL_OPEN}, {TOK_TOOL_CLOSE}, "", "");
    g.absorb(TOK_TOOL_OPEN, "<tool_call>");
    g.absorb(TOK_TOOL_CLOSE, "</tool_call>");
    EXPECT_TRUE(g.active());  // TERMINAL_OFF → active()=true

    g.reset();
    EXPECT_TRUE(g.active());
    // After reset, an opener token works fresh.
    EXPECT_TRUE(g.absorb(TOK_TEXT, "hi"));
    EXPECT_TRUE(g.absorb(TOK_TOOL_OPEN, "<tool_call>"));
    EXPECT_TRUE(g.active());
}
```

- [ ] **Step 2: Run tests to verify they fail to compile**

Run:
```bash
cmake --build build -j$(nproc) --target test_json_constrain 2>&1 | tail -10
```
Expected: compile FAIL — `configure_with_tools` is not a member of `PreambleGate`.

- [ ] **Step 3: Commit (failing tests)**

```bash
git add tests/test_json_constrain.cu
git commit -m "$(cat <<'EOF'
test(preamble_gate): tri-state tool-aware transitions (failing)

Six new tests covering:
- Tool-opener token transitions ACTIVE → TOOL_BODY (mask stays off, `{` no
  longer triggers preamble exit).
- Close token transitions TOOL_BODY → TERMINAL_OFF (no mask ever again).
- Free-text JSON path still exits gate to FSM when no tool fires.
- Budget exhaust still kicks in when neither tool nor JSON appears.
- Parallel tool calls stay TERMINAL_OFF.
- Reset restores ACTIVE.

Tests intentionally fail to compile until `configure_with_tools` lands.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Implement PreambleGate tri-state with token-set detection

**Files:**
- Modify: `src/compute/preamble_gate.h`

- [ ] **Step 1: Replace the class with the tri-state implementation**

Replace the entire class body of `PreambleGate` in `src/compute/preamble_gate.h` (lines 30-89 in the current file) with this. Keep the `#pragma once`, `#include`s, and namespace open/close as-is.

```cpp
class PreambleGate {
public:
    // Existing two-arg overload — preserved for non-tool callers.
    void configure(int32_t close_token, int max_tokens) {
        configure_with_tools(close_token, max_tokens,
                             /*open_tokens=*/{},
                             /*close_tokens=*/{},
                             /*open_prefix=*/"",
                             /*close_suffix=*/"");
    }

    // Tool-aware configure. open_tokens/close_tokens are token IDs of
    // tool-tag boundaries (single-token dialects: ChatML <tool_call>,
    // Gemma <|tool_call>). open_prefix/close_suffix are char-level
    // fallbacks for multi-token dialects (Llama3 <function=).
    //
    // Empty open_tokens AND empty open_prefix means "tool detection
    // disabled" — gate behaves exactly like the legacy two-arg configure.
    void configure_with_tools(int32_t close_token, int max_tokens,
                              std::vector<int32_t> open_tokens,
                              std::vector<int32_t> close_tokens,
                              std::string open_prefix,
                              std::string close_suffix) {
        max_tokens_ = max_tokens > 0 ? max_tokens : 0;
        close_token_ = close_token;
        open_tokens_ = std::move(open_tokens);
        close_tokens_ = std::move(close_tokens);
        open_prefix_ = std::move(open_prefix);
        close_suffix_ = std::move(close_suffix);
        configured_ = (close_token >= 0) || (max_tokens_ > 0);
        reset();
    }

    void reset() {
        state_ = configured_ ? State::ACTIVE : State::OFF;
        seen_ = 0;
        char_buf_.clear();
    }

    // active() returns true whenever the FSM mask should be skipped.
    // That's three of the four internal states: ACTIVE (preamble),
    // TOOL_BODY (inside a tool call), TERMINAL_OFF (after a tool call
    // closed). Only OFF — reached via {/[/think-close/budget — lets
    // the FSM mask through.
    bool active() const noexcept { return state_ != State::OFF; }

    // Returns true if the token was fully consumed by the gate (FSM should
    // NOT process it). Returns false only for the one transition where
    // the token must be forwarded to the FSM: ACTIVE → OFF via `{` or `[`.
    bool absorb(int32_t token, const std::string& text) {
        switch (state_) {
            case State::ACTIVE:
                return absorb_active(token, text);
            case State::TOOL_BODY:
                return absorb_tool_body(token, text);
            case State::TERMINAL_OFF:
                return true;  // permanently absorbing
            case State::OFF:
                return false;
        }
        return false;
    }

private:
    enum class State : uint8_t { ACTIVE, TOOL_BODY, TERMINAL_OFF, OFF };

    bool absorb_active(int32_t token, const std::string& text) {
        seen_++;

        // Close token (e.g. </think>) — consume and exit to FSM.
        if (close_token_ >= 0 && token == close_token_) {
            state_ = State::OFF;
            return true;
        }

        // Tool-opener detection (token-set fast-path).
        if (is_tool_open_token(token)) {
            state_ = State::TOOL_BODY;
            char_buf_.clear();
            return true;
        }

        // Tool-opener detection (char-prefix fallback).
        if (!open_prefix_.empty()) {
            append_char_buf(text);
            if (char_buf_.find(open_prefix_) != std::string::npos) {
                state_ = State::TOOL_BODY;
                char_buf_.clear();
                return true;
            }
        }

        // JSON start — exit to FSM and forward this token.
        for (char c : text) {
            if (c == '{' || c == '[') {
                state_ = State::OFF;
                return false;
            }
        }

        // Budget exhausted — give up on preamble.
        if (max_tokens_ > 0 && seen_ >= max_tokens_) {
            state_ = State::OFF;
            return true;
        }

        return true;
    }

    bool absorb_tool_body(int32_t token, const std::string& text) {
        // Close-token detection (token-set fast-path).
        if (is_tool_close_token(token)) {
            state_ = State::TERMINAL_OFF;
            char_buf_.clear();
            return true;
        }

        // Close-suffix detection (char-suffix fallback).
        if (!close_suffix_.empty()) {
            append_char_buf(text);
            if (char_buf_.find(close_suffix_) != std::string::npos) {
                state_ = State::TERMINAL_OFF;
                char_buf_.clear();
                return true;
            }
        }

        return true;  // body content is always absorbed
    }

    bool is_tool_open_token(int32_t token) const {
        if (token < 0)
            return false;
        for (int32_t t : open_tokens_) {
            if (t == token)
                return true;
        }
        return false;
    }

    bool is_tool_close_token(int32_t token) const {
        if (token < 0)
            return false;
        for (int32_t t : close_tokens_) {
            if (t == token)
                return true;
        }
        return false;
    }

    void append_char_buf(const std::string& text) {
        // Sliding window: keep up to 32 chars (covers any tag we care about).
        char_buf_ += text;
        if (char_buf_.size() > 32)
            char_buf_.erase(0, char_buf_.size() - 32);
    }

    bool configured_ = false;
    State state_ = State::OFF;
    int32_t close_token_ = -1;
    int max_tokens_ = 0;
    int seen_ = 0;

    std::vector<int32_t> open_tokens_;
    std::vector<int32_t> close_tokens_;
    std::string open_prefix_;
    std::string close_suffix_;
    std::string char_buf_;
};
```

Also: at the top of the file, add `#include <vector>` next to the existing includes.

- [ ] **Step 2: Run the new tests**

```bash
cmake --build build -j$(nproc) --target test_json_constrain 2>&1 | tail -10
./build/tests/test_json_constrain --gtest_filter='PreambleGateTest.*' 2>&1 | tail -25
```
Expected: all `PreambleGateTest.*` cases pass, including the six new tool-aware ones.

- [ ] **Step 3: Commit**

```bash
git add src/compute/preamble_gate.h
git commit -m "$(cat <<'EOF'
feat(preamble_gate): tri-state tool-aware transitions

Adds configure_with_tools() that hands the gate a set of tool-opener
and tool-close token IDs plus optional char-level fallbacks for
multi-token dialects (Llama3 <function=). Internal state machine:

  ACTIVE → TOOL_BODY on opener (token-set or char-prefix)
  ACTIVE → OFF       on '{'/'['/close-token/budget
  TOOL_BODY → TERMINAL_OFF on close-token or close-suffix
  TERMINAL_OFF → (terminal, no mask ever again)

External API stays binary: active() returns true for any state where
the FSM mask should be bypassed. The legacy configure() is preserved
unchanged for callers that don't want tool detection.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: PreambleGate char-level prefix/suffix tests + verification

**Files:**
- Modify: `tests/test_json_constrain.cu`

- [ ] **Step 1: Add char-level dialect tests**

Append to `tests/test_json_constrain.cu` after the previous tool-aware tests:

```cpp
TEST(PreambleGateTest, ToolModeCharPrefixFallback) {
    // Llama3 dialect: <function= is multi-token. Use char-prefix only;
    // open_tokens is empty.
    PreambleGate g;
    g.configure_with_tools(/*close_token=*/-1, /*budget=*/64,
                           /*open_tokens=*/{},
                           /*close_tokens=*/{},
                           /*open_prefix=*/"<function=",
                           /*close_suffix=*/"</function>");

    EXPECT_TRUE(g.active());
    EXPECT_TRUE(g.absorb(TOK_TEXT, "<"));
    EXPECT_TRUE(g.absorb(TOK_TEXT, "function"));
    EXPECT_TRUE(g.absorb(TOK_TEXT, "="));  // prefix complete here
    EXPECT_TRUE(g.active());

    // Body content with `{` is absorbed (TOOL_BODY).
    EXPECT_TRUE(g.absorb(TOK_OPEN_BRACE, "{"));
    EXPECT_TRUE(g.active());

    // Close suffix split across tokens.
    EXPECT_TRUE(g.absorb(TOK_TEXT, "</"));
    EXPECT_TRUE(g.absorb(TOK_TEXT, "function"));
    EXPECT_TRUE(g.absorb(TOK_TEXT, ">"));
    EXPECT_TRUE(g.active());  // TERMINAL_OFF
    EXPECT_TRUE(g.absorb(TOK_TEXT, "anything"));
    EXPECT_TRUE(g.active());
}

TEST(PreambleGateTest, LegacyConfigureKeepsBinaryBehavior) {
    // The two-arg configure() must not enable tool detection — protects
    // existing JsonConstrainer/SchemaConstrainer callers that don't know
    // about tools.
    PreambleGate g;
    g.configure(TOK_THINK_CLOSE, 8192);
    EXPECT_TRUE(g.active());

    // A token id that *would* be a tool-opener if registered must be
    // treated as ordinary text here.
    EXPECT_TRUE(g.absorb(TOK_TOOL_OPEN, "<tool_call>"));
    EXPECT_TRUE(g.active());  // still in ACTIVE (preamble), not TOOL_BODY

    // `{` still triggers preamble exit (legacy behaviour).
    EXPECT_FALSE(g.absorb(TOK_OPEN_BRACE, "{"));
    EXPECT_FALSE(g.active());
}
```

- [ ] **Step 2: Run tests**

```bash
cmake --build build -j$(nproc) --target test_json_constrain 2>&1 | tail -5
./build/tests/test_json_constrain --gtest_filter='PreambleGateTest.*' 2>&1 | tail -25
```
Expected: all PreambleGate tests pass (originals + tool-aware + char + legacy).

- [ ] **Step 3: Commit**

```bash
git add tests/test_json_constrain.cu
git commit -m "$(cat <<'EOF'
test(preamble_gate): char-prefix/suffix and legacy-config coverage

Two more cases:
- Llama3-style <function=...></function> split across multiple tokens —
  char-prefix and char-suffix matching across the sliding window.
- Legacy two-arg configure() must NOT enable tool detection (protects
  existing JsonConstrainer/SchemaConstrainer callers).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Add tool-aware overload to JsonConstrainer + SchemaConstrainer

**Files:**
- Modify: `src/compute/json_constrain.h`
- Modify: `src/compute/schema_constrain.h`

- [ ] **Step 1: JsonConstrainer overload**

In `src/compute/json_constrain.h`, replace the existing `set_preamble` (line 83) with both the existing signature and a new overload:

```cpp
    // Allow the model to emit a free-form preamble before strict JSON
    // enforcement starts. close_token>=0 enables close-token mode (reasoning
    // models with </think>); close_token<0 + max_tokens>0 enables budget-only
    // mode (markdown-fence preambles). Both modes also exit on the first
    // `{` / `[` seen. Pass close_token=-1 with max_tokens<=0 to fully disable.
    void set_preamble(int32_t close_token, int max_tokens = 8192) {
        preamble_.configure(close_token, max_tokens);
    }

    // Tool-aware preamble: when configured, the gate stays "no-mask" through
    // a tool-call body (delimited by open_tokens/close_tokens or the
    // open_prefix/close_suffix char fallback) and never re-enables the mask
    // after the tool closes. See PreambleGate::configure_with_tools.
    void set_preamble_with_tools(int32_t close_token, int max_tokens,
                                 std::vector<int32_t> open_tokens,
                                 std::vector<int32_t> close_tokens,
                                 std::string open_prefix,
                                 std::string close_suffix) {
        preamble_.configure_with_tools(close_token, max_tokens,
                                       std::move(open_tokens),
                                       std::move(close_tokens),
                                       std::move(open_prefix),
                                       std::move(close_suffix));
    }
```

Add `#include <vector>` and `#include <string>` near the top if not already present (`<vector>` is already there per inspection; `<string>` too).

- [ ] **Step 2: SchemaConstrainer overload**

In `src/compute/schema_constrain.h`, replace the existing `set_preamble` block (around lines 78-82) with the same pair:

```cpp
    // See JsonConstrainer::set_preamble for semantics — close-token mode for
    // reasoning models (</think>) or budget-only mode for markdown fences.
    void set_preamble(int32_t close_token, int max_tokens = 8192) {
        preamble_.configure(close_token, max_tokens);
    }

    // Tool-aware preamble. See PreambleGate::configure_with_tools for the
    // tri-state semantics — gate stays no-mask through tool-call bodies and
    // permanently disables the mask after the first tool close.
    void set_preamble_with_tools(int32_t close_token, int max_tokens,
                                 std::vector<int32_t> open_tokens,
                                 std::vector<int32_t> close_tokens,
                                 std::string open_prefix,
                                 std::string close_suffix) {
        preamble_.configure_with_tools(close_token, max_tokens,
                                       std::move(open_tokens),
                                       std::move(close_tokens),
                                       std::move(open_prefix),
                                       std::move(close_suffix));
    }
```

- [ ] **Step 3: Build and run all constrainer tests**

```bash
cmake --build build -j$(nproc) --target test_json_constrain 2>&1 | tail -5
./build/tests/test_json_constrain 2>&1 | tail -15
```
Expected: all tests pass — the new overload is unused so far, but the existing tests must keep passing.

- [ ] **Step 4: Commit**

```bash
git add src/compute/json_constrain.h src/compute/schema_constrain.h
git commit -m "$(cat <<'EOF'
feat(constrainers): set_preamble_with_tools overload

Pass-through to PreambleGate::configure_with_tools. Existing
set_preamble() callers are unchanged; the new overload is what
ConstraintManager will use when tools + json/schema are both set.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: ConstraintManager dialect resolution + has_tools wiring

**Files:**
- Modify: `src/runtime/constraint_manager.h`
- Modify: `src/runtime/constraint_manager.cpp`

- [ ] **Step 1: Update header signature**

In `src/runtime/constraint_manager.h`, add the include `#include "model/chat_template.h"` near the top, then update `prepare(...)` (around line 25) to:

```cpp
    // Prepare constraints for a request. Call before building InferenceState.
    // json_mode: enforce valid JSON syntax
    // json_schema: enforce JSON matching this schema string (empty = disabled)
    // tokenizer: needed for lazy init
    // has_tools: true if the request also has tools — gate enters tool-aware
    //   mode, schema/json mask only applies to free-text JSON, not tool bodies
    // tpl_family: chat-template family — selects which tool-tag dialect to look
    //   for (only consulted when has_tools is true)
    void prepare(bool json_mode, const std::string& json_schema, Tokenizer* tokenizer,
                 bool has_tools = false,
                 ChatTemplateFamily tpl_family = ChatTemplateFamily::CHATML);
```

- [ ] **Step 2: Implement dialect resolution + tool-aware gate config**

Replace the entire body of `src/runtime/constraint_manager.cpp` with:

```cpp
#include "runtime/constraint_manager.h"

namespace imp {

namespace {

int32_t detect_think_close(Tokenizer* tokenizer) {
    if (!tokenizer)
        return -1;
    int32_t close = tokenizer->find_token("</think>");
    if (close < 0)
        return -1;
    if (tokenizer->find_token("<think>") < 0)
        return -1;
    return close;
}

struct ToolDialect {
    std::vector<int32_t> open_tokens;
    std::vector<int32_t> close_tokens;
    std::string open_prefix;
    std::string close_suffix;

    bool empty() const {
        return open_tokens.empty() && close_tokens.empty() && open_prefix.empty() &&
               close_suffix.empty();
    }
};

// Resolves dialect-specific tool tags into token IDs (where the vocab has them
// as single special tokens) plus char-level prefix/suffix fallbacks.
//
// ChatML/Hermes/Mistral: <tool_call>...</tool_call>  (single special tokens)
// Gemma:                 <|tool_call>...<tool_call|> (single special tokens)
// Llama3:                <function=...></function>   (multi-token, char fallback)
// Other families fall through to ChatML defaults.
ToolDialect resolve_tool_dialect(Tokenizer* tokenizer, ChatTemplateFamily family) {
    ToolDialect d;
    if (!tokenizer)
        return d;

    auto add_token_if_present = [&](const std::string& s, std::vector<int32_t>& out) {
        int32_t id = tokenizer->find_token(s);
        if (id >= 0)
            out.push_back(id);
    };

    switch (family) {
        case ChatTemplateFamily::LLAMA3:
            // <function=NAME> has dynamic NAME — char-prefix is the only path.
            d.open_prefix = "<function=";
            d.close_suffix = "</function>";
            return d;

        case ChatTemplateFamily::GEMMA:
            d.open_prefix = "<|tool_call>";
            d.close_suffix = "<tool_call|>";
            add_token_if_present("<|tool_call>", d.open_tokens);
            add_token_if_present("<tool_call|>", d.close_tokens);
            return d;

        case ChatTemplateFamily::CHATML:
        case ChatTemplateFamily::MISTRAL_V3:
        case ChatTemplateFamily::DEEPSEEK_R1:
        case ChatTemplateFamily::PHI:
        case ChatTemplateFamily::NEMOTRON:
        case ChatTemplateFamily::LLAMA2:
        case ChatTemplateFamily::RAW:
        default:
            d.open_prefix = "<tool_call>";
            d.close_suffix = "</tool_call>";
            add_token_if_present("<tool_call>", d.open_tokens);
            add_token_if_present("</tool_call>", d.close_tokens);
            return d;
    }
}

}  // namespace

void ConstraintManager::prepare(bool json_mode, const std::string& json_schema, Tokenizer* tokenizer,
                                bool has_tools, ChatTemplateFamily tpl_family) {
    active_json_ = false;
    active_schema_ = false;

    const int32_t think_close = detect_think_close(tokenizer);

    // Reasoning models always get the large think-close budget. Otherwise:
    //   - has_tools: 64-token slack so short verbal preambles ("Sure! ")
    //     don't squeeze out the tool-tag opener.
    //   - no tools: 8-token slack for markdown fences, matches today's
    //     non-reasoning default.
    int preamble_budget;
    if (think_close >= 0) {
        preamble_budget = 8192;
    } else if (has_tools) {
        preamble_budget = 64;
    } else {
        preamble_budget = 8;
    }

    ToolDialect dialect;
    if (has_tools) {
        dialect = resolve_tool_dialect(tokenizer, tpl_family);
        if (dialect.empty()) {
            // Tokenizer surfaced none of the dialect tags AND the family had
            // no char fallback — degrade to current "drop schema" behaviour.
            IMP_LOG_INFO(
                "ConstraintManager: no tool-tag dialect for family %d, dropping schema/json_mode",
                static_cast<int>(tpl_family));
            return;
        }
    }

    auto configure_gate = [&](auto* constrainer) {
        if (has_tools) {
            constrainer->set_preamble_with_tools(think_close, preamble_budget,
                                                 dialect.open_tokens, dialect.close_tokens,
                                                 dialect.open_prefix, dialect.close_suffix);
        } else {
            constrainer->set_preamble(think_close, preamble_budget);
        }
    };

    if (!json_schema.empty()) {
        if (schema_constrainer_ && schema_constrainer_->is_initialized() &&
            json_schema == cached_schema_string_) {
            configure_gate(schema_constrainer_.get());
            schema_constrainer_->reset();
            active_schema_ = true;
        } else {
            auto schema = parse_json_schema(json_schema);
            if (schema) {
                schema_constrainer_ = std::make_unique<SchemaConstrainer>();
                if (tokenizer && schema_constrainer_->init(*tokenizer, std::move(schema))) {
                    cached_schema_string_ = json_schema;
                    configure_gate(schema_constrainer_.get());
                    schema_constrainer_->reset();
                    active_schema_ = true;
                } else {
                    IMP_LOG_ERROR("Failed to initialize schema constrainer");
                    schema_constrainer_.reset();
                    cached_schema_string_.clear();
                }
            } else {
                IMP_LOG_ERROR("Failed to parse JSON schema");
            }
        }
        return;
    }

    if (json_mode) {
        if (!json_constrainer_) {
            json_constrainer_ = std::make_unique<JsonConstrainer>();
            if (!tokenizer || !json_constrainer_->init(*tokenizer)) {
                IMP_LOG_ERROR("Failed to initialize JSON constrainer");
                json_constrainer_.reset();
                return;
            }
        }
        configure_gate(json_constrainer_.get());
        json_constrainer_->reset();
        active_json_ = true;
    }
}

void ConstraintManager::update(int32_t token) {
    if (active_schema_ && schema_constrainer_) {
        schema_constrainer_->update(token);
    } else if (active_json_ && json_constrainer_) {
        json_constrainer_->update(token);
    }
}

void ConstraintManager::reset() {
    if (active_schema_ && schema_constrainer_) {
        schema_constrainer_->reset();
    } else if (active_json_ && json_constrainer_) {
        json_constrainer_->reset();
    }
    active_json_ = false;
    active_schema_ = false;
}

}  // namespace imp
```

- [ ] **Step 3: Build (this is where C++ wiring errors surface)**

```bash
cmake --build build -j$(nproc) 2>&1 | tail -25
```
Expected: builds clean. If `IMP_LOG_INFO` is not found here, replace with the existing logging macro (the file already has `#include "core/logging.h"` via `constraint_manager.h`).

- [ ] **Step 4: Run constrainer + integration tests**

```bash
./build/tests/test_json_constrain 2>&1 | tail -10
```
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/runtime/constraint_manager.h src/runtime/constraint_manager.cpp
git commit -m "$(cat <<'EOF'
feat(constraint_manager): tool-aware preamble + dialect resolution

prepare() gains has_tools + tpl_family. Resolves per-dialect tool-tag
tokens via Tokenizer::find_token(); falls back to char-level prefix/suffix
where the tag is multi-token (Llama3 <function=NAME>). Bumps preamble
budget to 64 when tools are active. If a tokenizer/family combination
yields neither tokens nor prefix, logs and degrades to today's
"drop schema" behaviour — strictly no-worse-than-current.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Wire engine.cpp to pass new params + handler to set them

**Files:**
- Modify: `src/runtime/engine.cpp:1835`
- Modify: `tools/imp-server/handlers.cpp:617-633` and `:1002-1003`

- [ ] **Step 1: engine.cpp — pass new params**

In `src/runtime/engine.cpp`, change the call at line 1835. Before:

```cpp
    constraints_.prepare(req->json_mode, req->json_schema, model_->tokenizer());
```

After:

```cpp
    constraints_.prepare(req->json_mode, req->json_schema, model_->tokenizer(), req->has_tools,
                         req->tpl_family);
```

- [ ] **Step 2: handlers.cpp — set req fields, drop the early-return**

Open `tools/imp-server/handlers.cpp`. Locate the existing `has_tools` check at lines 619-633.

Replace lines 621-633 (the comment block + the `if (has_tools && ...) { drop }` body) with:

```cpp
    // tools + response_format=json_schema/json_object: the engine-side gate
    // stays "no-mask" through tool-call bodies (see ConstraintManager::prepare
    // and PreambleGate::configure_with_tools), so we keep both signals set
    // and the gate decides at runtime which path the model takes. Tool-call
    // dialect comes from tpl_family, captured below into the request.
```

(the warning log + the `json_mode = false; json_schema_str.clear();` go away.)

Then locate the request-builder block around line 1002-1003 (`req->json_mode = json_mode;`). Right after `req->json_schema = json_schema_str;`, add:

```cpp
        req->has_tools = has_tools;
        req->tpl_family = tpl_family;
```

`has_tools` and `tpl_family` are already in scope at that point (they're defined earlier in the same function — verify by reading lines 617-644 of the modified file).

- [ ] **Step 3: Build everything**

```bash
cmake --build build -j$(nproc) 2>&1 | tail -25
```
Expected: clean. If `req->has_tools` doesn't compile because the field wasn't added, go back to Task 1.

- [ ] **Step 4: Run unit tests**

```bash
./build/tests/test_json_constrain 2>&1 | tail -10
```
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/runtime/engine.cpp tools/imp-server/handlers.cpp
git commit -m "$(cat <<'EOF'
fix(server): stop dropping response_format when tools are also set

Plumbs has_tools + tpl_family from the request body through
Request → ConstraintManager so the engine-side gate enters tool-aware
mode. The schema/JSON FSM mask now applies only to free-text JSON
output; tool-call bodies bypass the mask entirely and TERMINAL_OFF
keeps subsequent (parallel) calls unmasked too.

Closes the runtime-coordination gap noted in the roadmap under
"Reasoning models + JSON schema — preamble pass-through".

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Integration test — tools + response_format end-to-end

**Files:**
- Modify: `tests/api/test_tools.py` (add a test method to `TestToolCalling`)

- [ ] **Step 1: Add the test**

Append to the existing `TestToolCalling` class in `tests/api/test_tools.py`:

```python
    def test_tools_plus_json_schema_passes_through(self, client, model):
        """Setting tools + response_format=json_schema must NOT drop the schema.

        The engine-side gate should let the model's tool-tag opener through
        unconditionally and only apply the schema mask if the model actually
        emits free-text JSON instead of a tool call. Either outcome (tool_call
        OR schema-shaped JSON) is acceptable here — what we're catching is the
        old failure mode where the request was rejected or response_format was
        silently dropped (which the server used to log).
        """
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
            "tools": [WEATHER_TOOL],
            "tool_choice": "auto",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "weather_or_text",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                        },
                        "required": ["answer"],
                    },
                },
            },
            "max_tokens": 128,
            "temperature": 0,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        choice = body["choices"][0]
        msg = choice["message"]

        # Either path is acceptable; both must be syntactically valid.
        if "tool_calls" in msg and msg["tool_calls"]:
            tc = msg["tool_calls"]
            assert tc[0]["function"]["name"] == "get_weather"
            import json
            args = json.loads(tc[0]["function"]["arguments"])
            assert isinstance(args, dict)
        else:
            # Free-text path: the schema must have been enforced.
            content = msg.get("content", "")
            import json
            payload = json.loads(content)
            assert "answer" in payload
            assert isinstance(payload["answer"], str)
```

- [ ] **Step 2: Run the test against a running server**

The integration tests need a running imp server. The repo's `tests/api/run_all_models.sh` is the canonical entry point; for a single test:

```bash
# In one terminal:
./build/tools/imp-server/imp-server --model models/Qwen3-4B-Instruct-2507-Q8_0.gguf --port 8080 &

# In another terminal (or same after a sleep):
cd tests/api
python -m pytest test_tools.py::TestToolCalling::test_tools_plus_json_schema_passes_through -v
```

Expected: PASS. If the model picks the tool path: `tool_calls` populated. If it picks free-text: schema-shaped JSON returned. Both are correct — the test catches a regression where neither would happen (request 500 or schema-mask blocking the `<` of `<tool_call>`).

- [ ] **Step 3: Run the existing tools tests too — make sure nothing regressed**

```bash
python -m pytest test_tools.py -v
```
Expected: all green (existing 3 tests + new one).

- [ ] **Step 4: Commit**

```bash
git add tests/api/test_tools.py
git commit -m "$(cat <<'EOF'
test(api): tools + response_format end-to-end coverage

Either path (tool_call or schema-shaped free-text JSON) is acceptable —
the assertion catches the regression where neither happens because the
schema mask blocked the `<` token that opens `<tool_call>`/`<function=`.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Roadmap update + verify gate

**Files:**
- Modify: `docs/roadmap.md` — strike the "preamble pass-through" / "tools + response_format" caveat in the JSON-schema section

- [ ] **Step 1: Edit roadmap.md**

In `docs/roadmap.md`, locate the paragraph in the "Reasoning models + JSON schema — preamble pass-through" section that begins:

> When a request sets both `tools` and `response_format=json_schema`/`json_object`, the schema mask would block the `<` of `<tool_call>`/`<function=` openers and prevent any tool call from being emitted. The server logs a warning and drops `response_format` in that case;

Replace that paragraph with:

> When a request sets both `tools` and `response_format=json_schema`/`json_object`, the engine-side `PreambleGate` enters tool-aware mode. It bypasses the schema mask through the entire tool-call body (delimited by single-token tags for ChatML/Hermes/Mistral/Gemma, or `<function=`/`</function>` char-prefix/suffix for Llama3) and stays unmasked for the rest of the generation, supporting parallel tool calls. If the model emits free-text JSON instead, the schema mask kicks in normally on the first `{`/`[`. Tool argument validation continues to flow through each tool's own `parameters` schema (post-hoc, not in-stream).

- [ ] **Step 2: Run verify-fast**

```bash
make verify-fast 2>&1 | tail -25
```
Expected: build + filtered tests + perf gate + smoke prompt all green. ~90s.

- [ ] **Step 3: Commit**

```bash
git add docs/roadmap.md
git commit -m "$(cat <<'EOF'
docs: roadmap reflects tools + JSON-schema coordination shipped

PreambleGate's tri-state tool-aware mode (this branch) replaces the
"server logs a warning and drops response_format" behaviour.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**

- Tri-state PreambleGate: Tasks 2 + 3 + 4 ✓
- Token-set + char-prefix dialect detection: Task 6 (resolve_tool_dialect) ✓
- ConstraintManager wiring + budget bump: Task 6 ✓
- SchemaConstrainer/JsonConstrainer overload: Task 5 ✓
- Request struct fields: Task 1 ✓
- Handler stops dropping response_format: Task 7 ✓
- Engine wiring: Task 7 ✓
- Failure-mode safeguard (no tokens + no prefix → degrade): Task 6 (the `dialect.empty()` early-return) ✓
- Tests: Tasks 2 + 4 (unit) + 8 (integration) ✓
- Edge cases (parallel tool calls, reasoning model, budget exhaust, free-text JSON path): Task 2 covers all four ✓
- Out-of-scope items not implemented: tool-arg schema enforcement (yes, not in plan), `tool_choice: required`/forced (yes, untouched — they already suppress schema via existing flow) ✓

**Placeholder scan:** no TBD/TODO/"add appropriate" found in any task. Every code step shows the actual code.

**Type consistency:** `configure_with_tools` signature appears consistently in Task 3 (definition) and Task 5 (calls in `set_preamble_with_tools`). `prepare(...)` signature in Task 6 matches the call site in Task 7 (engine.cpp). `Request::has_tools` / `Request::tpl_family` defined in Task 1 and read in Task 7 (engine.cpp) and written in Task 7 (handlers.cpp). `ToolDialect` is internal-anonymous-namespace in Task 6.

**Order of tasks:** 1 (Request) → 2-4 (gate + tests) → 5 (constrainer overloads) → 6 (manager) → 7 (engine + handler) → 8 (integration) → 9 (roadmap). Each task builds clean against the previous; no out-of-order references.
