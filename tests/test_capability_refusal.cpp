// C-6: a request for a feature the loaded forward does not implement gets 400, not 200 without it.
// extension_error is the one check every chat dialect passes (snapshot_state_and_tokenize_);
// /v1/messages and /v1/responses carry the generation-changing extension fields to it.

#include <gtest/gtest.h>

#include "anthropic.h"
#include "handlers_internal.h"
#include "responses.h"
#include "utils.h"

#include "runtime/engine_capabilities.h"

#include <nlohmann/json.hpp>
#include <httplib.h>

#include <string>

using json = nlohmann::ordered_json;

namespace {

const imp::EngineCapabilities kGeneric = imp::derive_capabilities({.model_loaded = true});
const imp::EngineCapabilities kInkling = imp::derive_capabilities(
    {.model_loaded = true, .layer_loop_replaced = true});

ChatRequestContext ctx_for(const imp::EngineCapabilities& caps) {
    ChatRequestContext ctx;
    ctx.snap.caps = caps;
    ctx.snap.model_name = "Inkling-2-Mini";
    ctx.snap.sae_width = 16;  // an SAE is loaded: only the capability can refuse
    ctx.params.chat_msgs.push_back({"system", "be terse"});
    return ctx;
}

// One request per gated feature, as the chat parser leaves it.
struct Ask {
    const char* field;
    void (*set)(ChatRequestParams&);
};
const Ask kAsks[] = {
    {"logit_lens", [](ChatRequestParams& p) { p.logit_lens = 3; }},
    {"sae_features", [](ChatRequestParams& p) { p.sae_features = 4; }},
    {"sae_clamp", [](ChatRequestParams& p) { p.sae_clamp.emplace_back(1, 2.0f); }},
    {"lora", [](ChatRequestParams& p) { p.lora_name = "style"; }},
    {"control_vector", [](ChatRequestParams& p) { p.cvec_name = "happy"; }},
    {"banned_strings", [](ChatRequestParams& p) { p.banned_strings = {"delve"}; }},
    {"cfg_scale", [](ChatRequestParams& p) { p.cfg_scale = 2.0f; }},
};

}  // namespace

TEST(CapabilityRefusal, InklingRefusesEveryLoopFeatureButGuidance) {
    for (const Ask& a : kAsks) {
        ChatRequestContext ctx = ctx_for(kInkling);
        a.set(ctx.params);
        const ExtensionError e = extension_error(ctx);
        if (std::string(a.field) == "cfg_scale") {
            EXPECT_TRUE(e.message.empty()) << e.message;
            continue;
        }
        ASSERT_FALSE(e.message.empty()) << a.field;
        EXPECT_STREQ(e.param, a.field);
        EXPECT_STREQ(e.code, "unsupported_feature");
        EXPECT_NE(e.message.find("Inkling-2-Mini"), std::string::npos) << e.message;
        EXPECT_NE(e.message.find(a.field), std::string::npos) << e.message;
    }
}

TEST(CapabilityRefusal, GenericLoadAcceptsEveryFeature) {
    for (const Ask& a : kAsks) {
        ChatRequestContext ctx = ctx_for(kGeneric);
        a.set(ctx.params);
        EXPECT_EQ(unsupported_feature_field(ctx.params, kGeneric), nullptr) << a.field;
        EXPECT_TRUE(extension_error(ctx).message.empty()) << a.field << ": " << extension_error(ctx).message;
    }
}

TEST(CapabilityRefusal, PlainRequestIsUntouchedOnInkling) {
    const ChatRequestContext ctx = ctx_for(kInkling);
    EXPECT_EQ(unsupported_feature_field(ctx.params, kInkling), nullptr);
    EXPECT_TRUE(extension_error(ctx).message.empty());
}

TEST(CapabilityRefusal, BannedRegexNamesItsOwnField) {
    ChatRequestParams p;
    p.banned_regex.push_back(nullptr);  // the check reads presence, not the pattern
    EXPECT_STREQ(unsupported_feature_field(p, kInkling), "banned_regex");
}

TEST(CapabilityRefusal, OpenAIEnvelopeCarriesParamAndCode) {
    ChatRequestContext ctx = ctx_for(kInkling);
    ctx.params.logit_lens = 2;
    const ExtensionError e = extension_error(ctx);
    httplib::Response res;
    send_json_error(res, 400, "invalid_request_error", e.message, e.param, e.code);
    EXPECT_EQ(res.status, 400);
    const json body = json::parse(res.body);
    EXPECT_EQ(body["error"]["type"], "invalid_request_error");
    EXPECT_EQ(body["error"]["param"], "logit_lens");
    EXPECT_EQ(body["error"]["code"], "unsupported_feature");
}

TEST(CapabilityRefusal, MessagesAndResponsesCarryTheGatedFields) {
    const json ext = {{"lora", "style"},
                      {"control_vector", "happy"},
                      {"banned_strings", json::array({"delve"})},
                      {"banned_regex", json::array({"de+lve"})},
                      {"cfg_scale", 2.0},
                      {"cfg_negative_prompt", "be verbose"},
                      {"sae_clamp", {{"1", 2.0}}}};
    const json user = {{"role", "user"}, {"content", "hi"}};
    json anth = {{"model", "m"}, {"max_tokens", 8}, {"messages", json::array({user})}};
    json rsp = {{"model", "m"}, {"input", "hi"}};
    for (const auto& [k, v] : ext.items()) {
        anth[k] = v;
        rsp[k] = v;
    }
    const json from_messages = imp_server::anthropic::anthropic_to_openai_body(anth);
    const json from_responses = imp_server::responses::responses_to_openai_body(rsp);
    for (const auto& [k, v] : ext.items()) {
        EXPECT_EQ(from_messages.value(k, json()), v) << "/v1/messages drops " << k;
        EXPECT_EQ(from_responses.value(k, json()), v) << "/v1/responses drops " << k;
    }
}
