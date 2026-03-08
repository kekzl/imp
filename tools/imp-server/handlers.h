#pragma once

#include "args.h"
#include "model/chat_template.h"
#include "model/tokenizer.h"

#include <imp/imp.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <mutex>
#include <string>

using json = nlohmann::json;

struct ServerState {
    ImpModel model = nullptr;
    ImpContext ctx = nullptr;
    imp::Tokenizer* tok = nullptr;
    imp::ChatTemplate chat_tpl;
    bool have_template = false;
    std::string model_name;
    std::timed_mutex mtx;
    int default_max_tokens = 16384;
    int max_seq_len = 0;
    std::atomic<int> next_id{0};
    std::atomic<int> next_tool_call_id{0};
    ServerArgs default_args;
    std::string models_dir;  // directory to scan for available .gguf files
    std::string api_key;     // if non-empty, require Bearer token auth
    bool is_think_model = false;  // model has <think> token (DeepSeek R1 etc.)
    int32_t think_start_id = -1;  // <think> token ID (-1 if not present)
    int32_t think_end_id = -1;    // </think> token ID (-1 if not present)
    bool model_loaded() const { return ctx != nullptr; }
};

// Graceful shutdown
extern std::atomic<httplib::Server*> g_server;

void signal_handler(int sig);
std::string make_completion_id(ServerState& state);
int64_t unix_timestamp();

std::vector<std::pair<std::string, std::string>> scan_gguf_files(const std::string& dir);
std::string find_model_path(const ServerState& state, const std::string& name);

ImpConfig build_config(const ServerArgs& args, const std::string& model_path = {},
                       const json& overrides = json::object());
std::string load_model_into_state(ServerState& state, const std::string& path,
                                  const json& config_overrides = json::object());

void handle_health(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_models(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_chat_completions(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_completions(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_tokenize(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_detokenize(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_load_model(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_unload_model(const httplib::Request& req, httplib::Response& res, ServerState& state);
