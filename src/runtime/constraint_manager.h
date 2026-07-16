#pragma once

#include "compute/json_constrain.h"
#include "compute/schema_constrain.h"
#include "compute/json_schema.h"
#include "model/tokenizer.h"
#include "model/chat_template.h"
#include "core/logging.h"
#include <memory>
#include <string>
#include <cstdint>

namespace imp {

// Manages JSON and JSON-schema constrained decoding.
// Caches the schema constrainer across requests with identical schemas
// to avoid re-parsing and re-classifying ~151K tokens per request.
class ConstraintManager {
public:
    ConstraintManager() = default;

    // Prepare constraints for a request. Call before building InferenceState.
    // json_mode: enforce valid JSON syntax
    // json_schema: enforce JSON matching this schema string (empty = disabled)
    // tokenizer: needed for lazy init
    // has_tools: true if the request also has tools — gate enters tool-aware
    //   mode, schema/json mask only applies to free-text JSON, not tool bodies
    // tpl_family: chat-template family — selects which tool-tag dialect to look
    //   for (only consulted when has_tools is true)
    void prepare(bool json_mode, const std::string& json_schema, Tokenizer* tokenizer, bool has_tools = false,
                 ChatTemplateFamily tpl_family = ChatTemplateFamily::CHATML, bool thinking_open = true);

    // Enforced tool calling (#1002): constrain generation to a tool-call
    // envelope with a TOOL_CALL schema FSM (see build_tool_call_schema).
    // tools: (name, parameter-schema JSON) per callable tool.
    //
    // optional=false (tool_choice=required / forced function): the envelope is
    //   FORCED from token 1 (after any <think>) — a tool call is mandatory.
    // optional=true (OpenAI strict:true with a model-chosen call): the envelope
    //   is NOT forced — the tool-aware preamble gate lets free text/a plain
    //   answer pass, and only IF the model emits the opener does the body FSM
    //   enforce the arguments. `tpl_family` selects the tool-tag dialect the
    //   gate watches for (ChatML only for now — non-ChatML families decline).
    //
    // Returns false when the schemas are not enforceable — caller keeps the
    // prompt-hint behavior (optionally calling prepare() for json fallback).
    // parallel (strict optional mode only): true = the model may emit several
    // tool calls (each body enforced); false = at most one, then EOS.
    bool prepare_tool_call(const std::vector<std::pair<std::string, std::string>>& tools,
                           const std::string& envelope_open, const std::string& envelope_close,
                           Tokenizer* tokenizer, bool thinking_open, bool optional = false,
                           ChatTemplateFamily tpl_family = ChatTemplateFamily::CHATML, bool parallel = true);

    // Cache/pool key for a tool-call constraint — shared by the engine's
    // constraint pool lookup and the internal classified-table cache.
    static std::string tool_call_key(const std::vector<std::pair<std::string, std::string>>& tools,
                                     const std::string& envelope_open, const std::string& envelope_close) {
        std::string key = "tool-call:" + envelope_open + "\x1f" + envelope_close;
        for (auto& [name, params] : tools)
            key += "\x1f" + name + "\x1e" + params;
        return key;
    }

    // Get current constrainer pointers (nullptr if not active).
    JsonConstrainer* json_constrainer() const noexcept {
        return active_json_ ? json_constrainer_.get() : nullptr;
    }
    SchemaConstrainer* schema_constrainer() const noexcept {
        return active_schema_ ? schema_constrainer_.get() : nullptr;
    }

    // Update FSM state after sampling a token.
    void update(int32_t token);

    // Jump-ahead (#844), schema only (json_mode has no schema skeleton to
    // force): the characters every legal continuation must spell next (see
    // SchemaConstrainer::forced_text). Pure probe — never advances the FSM.
    // Returns 0 for json_mode.
    int forced_text(std::string& out, int max_chars) const;

    // Reset FSM state (call when request finishes).
    void reset();

    // Check if any constraint is active.
    bool is_active() const noexcept { return active_json_ || active_schema_; }
    bool has_json() const noexcept { return active_json_; }
    bool has_schema() const noexcept { return active_schema_; }

    // Schema string of the cached (initialized) schema constrainer — lets the
    // engine's manager pool prefer an instance that already classified this
    // schema over re-classifying the vocab.
    const std::string& cached_schema() const noexcept { return cached_schema_string_; }

private:
    std::unique_ptr<JsonConstrainer> json_constrainer_;
    std::unique_ptr<SchemaConstrainer> schema_constrainer_;
    std::string cached_schema_string_;
    bool active_json_ = false;
    bool active_schema_ = false;
};

}  // namespace imp
