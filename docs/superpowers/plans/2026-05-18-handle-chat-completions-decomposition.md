# `handle_chat_completions` Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Break the 1737-LOC `handle_chat_completions()` HTTP handler in `tools/imp-server/handlers.cpp` into a thin orchestrator plus 5 named helper functions, preserving exact behavior.

**Architecture:** Bundle the 30+ captures into two structs (`ChatRequestParams` for body-parsed input, `ChatStateSnapshot` for lock-acquired engine state) plus a `ChatRequestContext` wrapper holding both. Each phase function takes the context by ref. The streaming/non-streaming branches become named free functions invoked from the orchestrator. The streaming function's chunked-provider-lambda capture list collapses to `[ctx = std::move(ctx)]`.

**Tech Stack:** C++20, httplib, nlohmann/json, imp's internal types. No new files; just helper functions + structs in `tools/imp-server/handlers.cpp` anonymous namespace.

---

## Sub-concern map (verified by structural read of current `main`)

| Section | Lines | LOC | Target function |
|---|---|---|---|
| Request parse + param extraction | 534-784 | 250 | `parse_chat_request_params()` |
| State snapshot + tools setup + tokenize | 791-1018 | 227 | `snapshot_state_and_tokenize_()` |
| Vision blocking decode | 1067-1203 | 136 | `handle_vision_chat_blocking_()` |
| Submit to batching | 1205-1220 | 16 | stays inline |
| Streaming response (chunked SSE provider) | 1222-1997 | 775 | `stream_chat_response_()` |
| Non-streaming response | 1998-2259 | 262 | `nonstream_chat_response_()` |

## Context structs (anonymous namespace)

```cpp
// Bundles body-parsed input parameters (no lock needed to populate).
struct ChatRequestParams {
    // Sampling
    float temperature, top_p, min_p, typical_p, repetition_penalty;
    float frequency_penalty, presence_penalty, dry_multiplier, dry_base;
    float mirostat_tau, mirostat_eta, think_budget;
    int top_k, max_tokens, seed, repeat_last_n;
    int dry_allowed_length, dry_penalty_last_n, mirostat;
    int n_completions, top_logprobs;
    bool stream, json_mode, req_logprobs, include_usage;
    bool top_p_explicit, top_k_explicit, rep_pen_explicit;
    // Stop sequences
    std::vector<std::string> stop_sequences;
    size_t max_stop_len;
    // Logit bias / format
    std::vector<std::pair<int32_t, float>> logit_bias;
    std::string json_schema_str;
    // Tools
    json tools, tool_choice;
    bool has_tools;
    // Messages
    std::vector<imp::ChatMessage> chat_msgs;
    std::vector<uint8_t> image_data;
    std::string requested_model;
};

// Bundles lock-acquired engine state (populated under state.mtx).
struct ChatStateSnapshot {
    imp::Tokenizer* tok;
    imp::ChatTemplate chat_tpl;
    bool have_template;
    std::string model_name;
    bool is_think_model;
    int32_t think_start_id, think_end_id;
    int32_t channel_open_id, channel_close_id, channel_newline_id;
    int max_seq_len;
    bool has_vision_request;
    std::vector<int32_t> stop_token_ids;
    imp::ChatTemplateFamily tpl_family;
    std::vector<imp::ToolFunction> tool_defs;
    bool tools_via_jinja;
    bool enable_thinking, suppress_thinking;
    std::vector<int32_t> tokens;
    int n_prompt_tokens;
};

// Top-level context bundling params + snap + transients.
struct ChatRequestContext {
    ChatRequestParams params;
    ChatStateSnapshot snap;
    std::string req_id;
    std::string comp_id;
    int64_t created;
    std::chrono::high_resolution_clock::time_point t_start;
    std::chrono::system_clock::time_point t_log_start;
    std::string log_endpoint, log_client_ip, log_raw_body;
    bool log_skip;
    std::shared_ptr<imp::Request> imp_req;
    std::shared_ptr<ServerRequest> server_req;
};
```

---

## Tasks

### T1: Add struct definitions

**Files:** Modify `tools/imp-server/handlers.cpp` near top (after `using json = nlohmann::json;` or similar, in anonymous namespace if one exists, else create one).

- [ ] **Step 1:** Read `tools/imp-server/handlers.cpp` lines 1-50 to find a clean insertion point for the structs.
- [ ] **Step 2:** Insert the three struct definitions (from the section above) into anonymous namespace.
- [ ] **Step 3:** Build with `make build 2>&1 | tail -10`. Expected: clean (structs alone don't change behavior).
- [ ] **Step 4:** No commit. Bundled with T2.

### T2: Extract `parse_chat_request_params()` (250 LOC)

**Files:** Modify `tools/imp-server/handlers.cpp`.

Signature:
```cpp
// Returns true if params parsed OK; sets res with 400/error and returns false on failure.
// chat_msgs are built (with image fetch if present) using tpl_family from state.
static bool parse_chat_request_params(
    const httplib::Request& req,
    httplib::Response& res,
    ServerState& state,
    ChatRequestParams& out_params,
    imp::ChatTemplateFamily& out_tpl_family_hint);
```

Extracts lines 534-784 of the orchestrator. The tpl_family hint is needed because parsing `tool` role messages requires it (uses `format_tool_response(tpl_family, msg)`), which is set under lock — pass in current best-effort snapshot.

- [ ] **Step 1:** Add the function definition near the orchestrator (anonymous namespace).
- [ ] **Step 2:** Replace orchestrator lines 534-784 with `if (!parse_chat_request_params(req, res, state, ctx.params, ctx.snap.tpl_family)) return;`.
- [ ] **Step 3:** Build.
- [ ] **Step 4:** Smoke test (after T7 wires it all together) — for now, just build.
- [ ] **Step 5:** No commit. Bundled with T3.

### T3: Extract `snapshot_state_and_tokenize_()` (227 LOC)

Signature:
```cpp
// Acquires lock, snapshots all state fields, sets up tools/vision/thinking,
// tokenizes prompt with chat template, validates prompt length. Returns true
// if OK; sets res with 400/503/etc and returns false on failure.
static bool snapshot_state_and_tokenize_(
    const httplib::Request& req,
    httplib::Response& res,
    ServerState& state,
    ChatRequestContext& ctx);
```

Extracts lines 791-1018 of the orchestrator (state snapshot under lock, channel-model default adjustment, tool defs build, vision lock setup, thinking detection, tokenization, think prefix detection, prompt length validation, max_tokens clamping).

- [ ] **Step 1:** Add the function definition.
- [ ] **Step 2:** Replace orchestrator lines 791-1018 with `if (!snapshot_state_and_tokenize_(req, res, state, ctx)) return;`.
- [ ] **Step 3:** Build.
- [ ] **Step 4:** Commit T1+T2+T3 together: "refactor(handlers): introduce ChatRequestContext + extract param parse + state snapshot".

### T4: Extract `handle_vision_chat_blocking_()` (136 LOC)

Signature:
```cpp
// Handle the vision-request blocking path: prefill via C API + decode loop + response.
// Caller must hold no lock on entry. Returns after sending the response.
static void handle_vision_chat_blocking_(
    httplib::Response& res,
    ServerState& state,
    ChatRequestContext& ctx);
```

Extracts lines 1067-1203 of the orchestrator.

- [ ] **Step 1:** Add function definition.
- [ ] **Step 2:** Replace orchestrator's `if (has_vision_request) { ... }` block with:
   ```cpp
   if (ctx.snap.has_vision_request) {
       handle_vision_chat_blocking_(res, state, ctx);
       return;
   }
   ```
- [ ] **Step 3:** Build.
- [ ] **Step 4:** Smoke test non-streaming Q8_0 (vision path is unused on text models — regression check).
- [ ] **Step 5:** Commit "refactor(handlers): extract handle_vision_chat_blocking_".

### T5: Extract `stream_chat_response_()` (775 LOC — biggest)

Signature:
```cpp
// Set up SSE chunked content provider for streaming chat completion.
// The provider captures ctx by ref. Returns after res.set_chunked_content_provider returns.
static void stream_chat_response_(
    httplib::Response& res,
    ServerState& state,
    ChatRequestContext& ctx);
```

Extracts lines 1222-1997 of the orchestrator (the entire `res.set_chunked_content_provider(...)` call including the 775-LOC lambda body).

**Key transformation:** the inline lambda's 30-var capture list collapses to `[&state, &ctx]` since both are already in the function's scope. Internally, references to `comp_id`, `snap_tok`, etc. become `ctx.comp_id`, `ctx.snap.tok`, etc.

This is the highest-risk extraction. Use sonnet for the implementer.

- [ ] **Step 1:** Add function definition.
- [ ] **Step 2:** Replace orchestrator's `if (stream) { res.set_chunked_content_provider(...) }` block with:
   ```cpp
   if (ctx.params.stream) {
       stream_chat_response_(res, state, ctx);
   } else { ... non-stream branch stays inline for now ... }
   ```
- [ ] **Step 3:** Build.
- [ ] **Step 4:** Smoke test streaming on Qwen3-8B Q8_0:
   ```bash
   # Start server in background
   docker run -d --rm --name imp-test-stream --gpus all -v /home/kekz/models:/models -p 18080:18080 imp:test \
     imp-server --model /models/Qwen3-8B-Q8_0.gguf --port 18080
   sleep 8
   # Test streaming
   curl -s -X POST http://localhost:18080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"test","messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"stream":true}' \
     | head -20
   docker stop imp-test-stream
   ```
   Expected: SSE chunks with role:"assistant" then content tokens then [DONE].
- [ ] **Step 5:** Commit "refactor(handlers): extract stream_chat_response_ (775 LOC lambda → named function)".

### T6: Extract `nonstream_chat_response_()` (262 LOC)

Signature:
```cpp
static void nonstream_chat_response_(
    httplib::Response& res,
    ServerState& state,
    ChatRequestContext& ctx);
```

Extracts lines 1998-2259 of the orchestrator (the `else { ... }` non-streaming branch).

- [ ] **Step 1:** Add function definition.
- [ ] **Step 2:** Replace orchestrator's `else { ... }` (non-stream branch) with `nonstream_chat_response_(res, state, ctx);`.
- [ ] **Step 3:** Build.
- [ ] **Step 4:** Smoke test non-streaming Q8_0:
   ```bash
   docker run -d --rm --name imp-test-nonstream --gpus all -v /home/kekz/models:/models -p 18081:18081 imp:test \
     imp-server --model /models/Qwen3-8B-Q8_0.gguf --port 18081
   sleep 8
   curl -s -X POST http://localhost:18081/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"test","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}' \
     | head -10
   docker stop imp-test-nonstream
   ```
   Expected: JSON response with choices[0].message.content non-empty.
- [ ] **Step 5:** Commit "refactor(handlers): extract nonstream_chat_response_".

### T7: Final verify + PR

- [ ] **Step 1:** Inspect orchestrator size — should be ~30 LOC.
- [ ] **Step 2:** Run both streaming + non-streaming smoke once more on the final commit.
- [ ] **Step 3:** `make verify-fast`.
- [ ] **Step 4:** Push branch + open PR on main with summary table.

---

## Risks + mitigations

- **30-var capture list collapse:** the streaming lambda's `[&state, server_req, comp_id, ...]` becomes `[&state, &ctx]`. Each reference inside changes from `comp_id` → `ctx.comp_id`. Mechanical but high-volume. The implementer must use `sed` or a careful Find+Replace to rewrite all 30+ references. Test thoroughly after.
- **`ChatStateSnapshot` field-rename risk:** the original code uses `snap_tok`, `snap_chat_tpl`, etc. (with `snap_` prefix). In the struct, these become `snap.tok`, `snap.chat_tpl`. All ~50 references need updating. Use `sed` carefully — match `snap_X` → `ctx.snap.X` only where it's the var, not part of a longer identifier.
- **Vision/tools paths are rarely exercised:** smoke testing on Q8_0 covers ~80% of the code paths but not vision or tool calling. Accept the residual risk for now; the structural refactor is bit-identical per code-reading.
- **HTTP server smoke is more complex than CLI:** requires docker run + sleep + curl. The 8-second sleep may not be enough on slow model loads. Use `docker logs imp-test-stream` if the curl fails to check server readiness.

## Self-Review checklist

- [x] **Spec coverage:** all 6 sub-concerns from the structural map have a task.
- [x] **No placeholders:** every step has exact code or commands.
- [x] **Type consistency:** struct fields match the orchestrator's locals; helper signatures pass `ChatRequestContext&` consistently.
