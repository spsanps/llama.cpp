#include <deque>

#include "common.h"
#include "llama.h"
#include "grammar-parser.h"

// no image support for autocomplete
/* #include "../llava/clip.h"

#include "stb_image.h" */

#ifndef NDEBUG
// crash the server in debug mode, otherwise send an http 500 error
#define CPPHTTPLIB_NO_EXCEPTIONS 1
#endif
// increase max payload length to allow use of larger context size
#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH 1048576
#include "httplib.h"
#include "json.hpp"

// auto generated files (update with ./deps.sh)
#include "index.html.hpp"
#include "index.js.hpp"
#include "completion.js.hpp"
#include "json-schema-to-grammar.mjs.hpp"

#include <cstddef>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <atomic>

#ifndef SERVER_VERBOSE
#define SERVER_VERBOSE 1
#endif

#define DEFAULT_OAICOMPAT_MODEL "gpt-3.5-turbo-0613"

using json = nlohmann::json;

template <typename T>
static T json_value(const json &body, const std::string &key, const T &default_value)
{
    // Fallback null to default value
    return body.contains(key) && !body.at(key).is_null()
               ? body.value(key, default_value)
               : default_value;
}

static bool server_verbose = false;

#if SERVER_VERBOSE != 1
#define LOG_VERBOSE(MSG, ...)
#else
#define LOG_VERBOSE(MSG, ...)                                            \
    do                                                                   \
    {                                                                    \
        if (server_verbose)                                              \
        {                                                                \
            server_log("VERBOSE", __func__, __LINE__, MSG, __VA_ARGS__); \
        }                                                                \
    } while (0)
#endif

#define LOG_ERROR(MSG, ...) server_log("ERROR", __func__, __LINE__, MSG, __VA_ARGS__)
#define LOG_WARNING(MSG, ...) server_log("WARNING", __func__, __LINE__, MSG, __VA_ARGS__)
#define LOG_INFO(MSG, ...) server_log("INFO", __func__, __LINE__, MSG, __VA_ARGS__)

static void server_log(const char *level, const char *function, int line,
                       const char *message, const nlohmann::ordered_json &extra)
{
    nlohmann::ordered_json log{
        {"timestamp", time(nullptr)},
        {"level", level},
        {"function", function},
        {"line", line},
        {"message", message},
    };

    if (!extra.empty())
    {
        log.merge_patch(extra);
    }

    const std::string str = log.dump(-1, ' ', false, json::error_handler_t::replace);
    printf("%.*s\n", (int)str.size(), str.data());
    fflush(stdout);
}

static void log_server_request(const httplib::Request &req, const httplib::Response &res)
{
    LOG_INFO("request", {
                            {"remote_addr", req.remote_addr},
                            {"remote_port", req.remote_port},
                            {"status", res.status},
                            {"method", req.method},
                            {"path", req.path},
                            {"params", req.params},
                        });

    LOG_VERBOSE("request", {
                               {"request", req.body},
                               {"response", res.body},
                           });
}

struct server_params
{
    std::string hostname = "127.0.0.1";
    std::vector<std::string> api_keys;
    std::string public_path = "examples/server/public";
    int32_t port = 8080;
    int32_t read_timeout = 600;
    int32_t write_timeout = 600;
};

// server states remain the same for autocomplete
enum server_state
{
    SERVER_STATE_LOADING_MODEL, // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,         // Server is ready and model is loaded
    SERVER_STATE_ERROR          // An error occurred, load_model failed
};

// For autocomplete there will only be a single slot
// should call it task instead to avoid confusion
enum task_state
{
    INACTIVE,
    CANCEL, // cancel the task
    IDLE,
    // PROCESSING,

    // Added for autocomplete: slot is tokenizing the prompt
    TOKENIZING,

    // Added for autocomplete: slot is comparing caches tokens to the prompt
    // After this point - if not cancelled, the cache will be changed
    // in this state we also compare if it is the same prompt as before
    // if so can skip to context processing
    CACHE_COMPARE,

    // Added for autocomplete: slot is preparing relevant cache for the prompt
    // Cache will be updated in this CACHE_PREPARE state
    CACHE_PREPARE,

    // Added for autocomplete: slot is catching up to the provided context
    // also updating the cache
    // done token by token so that it can be cancelled
    CONTEXT_PROCESSING,

    // Added for autocomplete: slot is generating suggestions up to the required suggestion length
    // done token by token so that it can be cancelled
    SUGGESTION_PROCESSING,

    // Added for autocomplete: slot is generating tokens past the required suggestion length for the next suggestion
    POST_SUGGESTION_PROCESSING,
};

struct llama_token_cell
{
    // equivalent to llama_kv_cell
    // does llama.cpp keep track of tokens?
    // if so I am reinventing the wheel here
    // I suppose it can't hurt

    // We need to ensure circular buffer properties

    llama_token token;

    llama_pos pos = -1;
    llama_pos delta = 0;

    std::set<llama_seq_id> seq_id;

    bool has_seq_id(const llama_seq_id &id) const
    {
        return seq_id.find(id) != seq_id.end();
    }
};

struct seq_info
{
    llama_seq_id seq_id;
    int start_ind;
    int end_ind;
    int length;
};

uint64_t compute_hash(
    const std::vector<llama_token> &token_vec,
    uint64_t a = 32003,
    uint64_t mod = 1e9 + 9)
{
    uint64_t hash_value = 0;
    uint64_t pow_a = 1; // a^0

    for (const auto &token : token_vec)
    {
        hash_value = (hash_value + static_cast<uint64_t>(token) * pow_a) % mod;
        pow_a = (pow_a * a) % mod;
    }

    return hash_value;
}

struct token_cache
{
    // equivalent to llama_kv_cache

    std::vector<llama_token_cell> cache;

    // we will have a vector of seq_info
    // we will use this vector to find the latest seq_id
    // and the oldest seq_id when we need to clean the cache
    std::vector<seq_info> seq_queue;

    std::vector<bool> searched;

    int next_ind(int curr_ind)
    {
        return (curr_ind + 1) % cache.size();
    }

    int next_ind_seq(int curr_ind, int seq_id)
    {
        int next_ind = curr_ind;
        for (int i = 0; i < cache.size(); i++)
        {
            next_ind = next_ind(next_ind);
            if (cache[next_ind].has_seq_id(seq_id))
            {
                // if next_ind is before curr_ind means we have looped around
                if (cache[next_ind].pos < cache[curr_ind].pos)
                {
                    return -1;
                }
                return next_ind;
            }
        }
        return -1;
    }

    void reset_searched()
    {
        for (int i = 0; i < searched.size(); i++)
        {
            searched[i] = false;
        }
    }

    void reset()
    {
        cache.clear();
        seq_queue.clear();
        searched.clear();
    }

    ~token_cache()
    {
        reset();
    }

    llama_seq_id get_latest_seq_id()
    {
        if (seq_queue.size() == 0)
        {
            return -1;
        }
        return seq_queue[seq_queue.size() - 1].seq_id;
    }

    llama_seq_id get_oldest_seq_id()
    {
        if (seq_queue.size() == 0)
        {
            return -1;
        }
        return seq_queue[0].seq_id;
    }

    seq_info get_seq_info(const llama_seq_id &seq_id)
    {
        for (int32_t i = 0; i < seq_queue.size(); i++)
        {
            if (seq_queue[i].seq_id == seq_id)
            {
                return seq_queue[i];
            }
        }
        return seq_info{-1, -1, -1};
    }

    // function to make a seq_id the latest seq_id
    // if seq_id is not in the cache, return false

    bool has_seq_id(const llama_seq_id &seq_id)
    {
        for (int32_t i = 0; i < seq_queue.size(); i++)
        {
            if (seq_queue[i].seq_id == seq_id)
            {
                return true;
            }
        }
        return false;
    }

    bool make_latest_seq_id(const llama_seq_id &seq_id)
    {
        // if seq_id is not in the cache, return false
        if (!has_seq_id(seq_id))
        {
            return false;
        }

        // if seq_id is already the latest seq_id, return true
        if (seq_id == get_latest_seq_id())
        {
            return true;
        }

        // find the seq_id in seq_queue
        int32_t ind = -1;
        for (int32_t i = 0; i < seq_queue.size(); i++)
        {
            if (seq_queue[i].seq_id == seq_id)
            {
                ind = i;
                break;
            }
        }

        // if seq_id is not found, return false
        if (ind == -1)
        {
            return false;
        }

        // move the seq_id to the end of seq_queue
        seq_queue.push_back(seq_queue[ind]);
        seq_queue.erase(seq_queue.begin() + ind);

        return true;
    }

    void delete_seq_id(const llama_seq_id &seq_id)
    {
        // if seq_id is not in the cache, return false
        if (!has_seq_id(seq_id))
        {
            return;
        }

        // find the seq_id in seq_queue
        int32_t ind = -1;
        for (int32_t i = 0; i < seq_queue.size(); i++)
        {
            if (seq_queue[i].seq_id == seq_id)
            {
                ind = i;
                break;
            }
        }

        // if seq_id is not found, return false
        if (ind == -1)
        {
            return;
        }

        // delete the seq_id from seq_queue
        seq_queue.erase(seq_queue.begin() + ind);

        // delete the seq_id from cache
        for (int32_t i = 0; i < cache.size(); i++)
        {
            if (cache[i].has_seq_id(seq_id))
            {
                cache[i].seq_id.erase(seq_id);
                // if cache[i] is empty, set pos to -1
                if (cache[i].seq_id.size() == 0)
                {
                    cache[i].pos = -1;
                }
            }
        }
    }

    bool match_pattern_seq_id_with_trim(
        const std::vector<llama_token> &pattern,
        const int32_t &trim, // how much of the pattern can be trimmed (from right)
        const llama_seq_id &seq_id,
        int &result_ind, // index of the first token of the pattern in the cache
        uint64_t a = 32003,
        uint64_t mod = 1e9 + 9)
    {
        // we will use rabin karp algorithm to match the pattern
        // we will use a rolling hash to match the pattern
        // if not found we will start removing tokens from the right
        // we will also mark the tokens as searched so it doesn't get searched again with another seq_id
        // should we store the hashes? - yes maybe

        seq_info info = get_seq_info(seq_id);

        std::vector<llama_token> trimmed_pattern(
            pattern.begin(),
            pattern.end() - trim);
        int pattern_length = trimmed_pattern.size();
        // compute hash of the trimmed pattern
        uint64_t pattern_hash = compute_hash(trimmed_pattern, a, mod);

        int seq_length = info.length;

        // if pattern is larger than cache, return empty vector
        if (pattern_length > seq_length)
        {
            return false;
        }

        // go to start_ind in cache
        int ind = info.start_ind;
        // compute hash of the first pattern_length tokens in cache
        // we will use "token" variable in each cell to compute the hash
        uint64_t seq_hash = 0;
        // vector to store the indices of the tokens in the cache for the sequence so we can do a rolling hash
        std::deque<int> seq_indices;

        uint64_t pow_a = 1; // a^0

        // compute hash of the first pattern_length tokens in cache
        for (int i = 0; i < pattern_length; i++)
        {
            if (ind == -1)
            {
                return false; // if we reach the end of the cache before we finish computing the hash, return false
            }

            seq_hash = (seq_hash + static_cast<uint64_t>(cache[ind].token) * pow_a) % mod;
            seq_indices.push_back(ind);
            pow_a = (pow_a * a) % mod;
            ind = next_ind_seq(ind, seq_id);
        }

        if (seq_hash == pattern_hash)
        {
            // check if the pattern matches
            bool match = true;
            for (int i = 0; i < pattern_length; i++)
            {
                if (cache[seq_indices[i]].token != trimmed_pattern[i])
                {
                    match = false;
                    break;
                }
            }
            if (match)
            {
                // if pattern matches, return true
                result_ind = seq_indices[0];
                return true;
            }
        }

        // do a rolling hash to find the pattern
        int i = 0;
        while (ind != -1 && i < cache.size())
        {
            // update the hash
            uint64_t first_term = static_cast<uint64_t>(cache[seq_indices.front()].token) * pow_a;

            seq_hash = (seq_hash + mod - first_term % mod) % mod; // Ensure non-negative result

            seq_hash = (seq_hash * a) % mod;

            seq_hash = (seq_hash + static_cast<uint64_t>(cache[ind].token)) % mod;
            seq_indices.pop_front(); // Efficient removal of the first element
            seq_indices.push_back(ind);

            if (seq_hash == pattern_hash)
            {
                // check if the pattern matches
                bool match = true;
                for (int i = 0; i < pattern_length; i++)
                {
                    if (cache[seq_indices[i]].token != trimmed_pattern[i])
                    {
                        match = false;
                        break;
                    }
                }
                if (match)
                {
                    // if pattern matches, return true
                    result_ind = seq_indices[0];
                    return true;
                }
            }
            // update ind
            ind = next_ind_seq(ind, seq_id);
        }
        return false;
    }

    void search_pattern(const std::vector<llama_token> &pattern, std::vector<llama_pos> &result)
    {
        // search for pattern in cache
        // return the positions of the pattern in the cache
        // if not found, return empty vector

        // reset searched
        reset_searched();

        // if pattern is empty, return empty vector
        if (pattern.size() == 0)
        {
            return;
        }

        // if pattern is larger than cache, return empty vector
        if (pattern.size() > cache.size())
        {
            return;
        }

        // find the latest seq_id
    }
};

struct llama_server_context
{
    /* unlike server.cpp we don't have slots as we won't process more than one suggestion in parallel */

    llama_model *model = nullptr;
    llama_context *ctx = nullptr;

    // clip_ctx *clp_ctx = nullptr;

    gpt_params params;
    llama_batch batch;

    // bool multimodal         = false;
    // bool clean_kv_cache = true;
    // bool all_slots_are_idle = false;
    bool add_bos_token = true;

    int32_t id_gen;
    // REVIEW: might have to change what this is for autocomplete
    int32_t n_ctx; // total context for all clients / slots

    // for autocomplete
    int32_t min_acceptable_ctx; // not sure if we need this
    int32_t max_ctx;            // per suggestion (ctx value till cache will be preserved)
    int32_t suggestion_length;  // length of the suggestion

    // system prompt
    bool system_need_update = false;

    // REVIEW: might have to change what this is for autocomplete
    std::string system_prompt;
    std::vector<llama_token> system_tokens;

    task_state state = INACTIVE;
    // mutex for the task state
    std::mutex state_mutex;
    // lock for the task state
    std::condition_variable state_lock;

    ~llama_server_context()
    {
        if (ctx)
        {
            llama_free(ctx);
            ctx = nullptr;
        }
        if (model)
        {
            llama_free_model(model);
            model = nullptr;
        }
    }

    bool load_model(const gpt_params &params_)
    {
        params = params_;

        std::tie(model, ctx) = llama_init_from_gpt_params(params);
        if (model == nullptr)
        {
            LOG_ERROR("unable to load model", {{"model", params.model}});
            return false;
        }

        // max context tokens acceptable.
        // sum of all contexts currently in use by all slots + suggested length
        // should be less than this value
        // When this value is reached, we will start to clean the kv cache
        // from the oldest slot
        n_ctx = llama_n_ctx(ctx);

        add_bos_token = llama_should_add_bos_token(model);

        return true;
    }

    void initialize()
    {
        // raise not implemented error
        throw std::runtime_error("Not implemented");

        task_state state = IDLE;
    }

    std::vector<llama_token> tokenize(
        const json &json_prompt,
        bool add_bos) const
    {
        // TODO: currently, we tokenize using special tokens by default
        //       this is not always correct (see https://github.com/ggerganov/llama.cpp/pull/4160#issuecomment-1824826216)
        //       but it's better compared to completely ignoring ChatML and other chat templates
        const bool TMP_FORCE_SPECIAL = true;

        // If `add_bos` is true, we only add BOS, when json_prompt is a string,
        // or the first element of the json_prompt array is a string.
        std::vector<llama_token> prompt_tokens;

        if (json_prompt.is_array())
        {
            bool first = true;
            for (const auto &p : json_prompt)
            {
                if (p.is_string())
                {
                    auto s = p.template get<std::string>();
                    std::vector<llama_token> p;
                    if (first)
                    {
                        p = ::llama_tokenize(ctx, s, add_bos, TMP_FORCE_SPECIAL);
                        first = false;
                    }
                    else
                    {
                        p = ::llama_tokenize(ctx, s, false, TMP_FORCE_SPECIAL);
                    }
                    prompt_tokens.insert(
                        prompt_tokens.end(), p.begin(), p.end());
                }
                else
                {
                    if (first)
                    {
                        first = false;
                    }
                    prompt_tokens.push_back(p.template get<llama_token>());
                }
            }
        }
        else
        {
            auto s = json_prompt.template get<std::string>();
            prompt_tokens = ::llama_tokenize(ctx,
                                             s,
                                             add_bos,
                                             TMP_FORCE_SPECIAL);
        }

        return prompt_tokens;
    }

    // main function for autocomplete
    // entry point for /completion
    void complete(const json &json_prompt,
                  // const json &json_context,
                  const json &json_options,
                  json &json_result){

    };

    void set_state(task_state new_state)
    {
        // guard the state change
        std::lock_guard<std::mutex> lock(state_mutex);
        state = state;
    }

    bool set_state_or_cancel(task_state new_state)
    {
        // guard the state change
        std::lock_guard<std::mutex> lock(state_mutex);
        if (state == CANCEL)
        {
            return false;
        }
        state = state;
        return true;
    }

    void request_cancel()
    {
        // if server is not ready, return
        LOG_INFO("cancel request", {});

        // guard the state change

        {
            std::lock_guard<std::mutex> lock(state_mutex);
            LOG_VERBOSE("Current state", {{"state", state}});

            if (state == INACTIVE || state == CANCEL || state == IDLE)
            {
                return;
            }

            state_lock.notify_all();
        }

        // wait for cancellation to complete (ie state to become not CANCEL)

        {
            std::unique_lock<std::mutex> lock(state_mutex);
            state_lock.wait(lock, [this]()
                            { return state != CANCEL; });
        }
    };

    void cache_compare()
    {
        if (!set_state_or_cancel(CACHE_COMPARE))
        {
            return;
        }

        // check if tokens are
    };
};

int32_t main(int32_t argc, char **argv)
{
#if SERVER_VERBOSE != 1
    log_disable();
#endif
    // own arguments required by this example
    gpt_params params;
    server_params sparams;

    // struct that contains llama context and inference
    // REVIEW
    llama_server_context llama;

    server_params_parse(argc, argv, sparams, params, llama);

    if (params.model_alias == "unknown")
    {
        params.model_alias = params.model;
    }

    llama_backend_init(params.numa);

    LOG_INFO("build info", {{"build", LLAMA_BUILD_NUMBER},
                            {"commit", LLAMA_COMMIT}});

    LOG_INFO("system info", {
                                {"n_threads", params.n_threads},
                                {"n_threads_batch", params.n_threads_batch},
                                {"total_threads", std::thread::hardware_concurrency()},
                                {"system_info", llama_print_system_info()},
                            });

    httplib::Server svr;

    std::atomic<server_state> state{SERVER_STATE_LOADING_MODEL};

    svr.set_default_headers({{"Server", "llama-autocomplete.cpp"}});

    // CORS preflight
    svr.Options(R"(.*)", [](const httplib::Request &req, httplib::Response &res)
                {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("Access-Control-Allow-Methods", "POST");
        res.set_header("Access-Control-Allow-Headers", "*"); });

    svr.Get("/health", [&](const httplib::Request &, httplib::Response &res)
            {
        server_state current_state = state.load();
        switch(current_state) {
            case SERVER_STATE_READY:
                res.set_content(R"({"status": "ok"})", "application/json");
                res.status = 200; // HTTP OK
                break;
            case SERVER_STATE_LOADING_MODEL:
                res.set_content(R"({"status": "loading model"})", "application/json");
                res.status = 503; // HTTP Service Unavailable
                break;
            case SERVER_STATE_ERROR:
                res.set_content(R"({"status": "error", "error": "Model failed to load"})", "application/json");
                res.status = 500; // HTTP Internal Server Error
                break;
        } });

    svr.set_logger(log_server_request);

    svr.set_exception_handler([](const httplib::Request &, httplib::Response &res, std::exception_ptr ep)
                              {
                const char fmt[] = "500 Internal Server Error\n%s";
                char buf[BUFSIZ];
                try
                {
                    std::rethrow_exception(std::move(ep));
                }
                catch (std::exception &e)
                {
                    snprintf(buf, sizeof(buf), fmt, e.what());
                }
                catch (...)
                {
                    snprintf(buf, sizeof(buf), fmt, "Unknown Exception");
                }
                res.set_content(buf, "text/plain; charset=utf-8");
                res.status = 500; });

    svr.set_error_handler([](const httplib::Request &, httplib::Response &res)
                          {
                if (res.status == 401)
                {
                    res.set_content("Unauthorized", "text/plain; charset=utf-8");
                }
                if (res.status == 400)
                {
                    res.set_content("Invalid request", "text/plain; charset=utf-8");
                }
                else if (res.status == 404)
                {
                    res.set_content("File Not Found", "text/plain; charset=utf-8");
                    res.status = 404;
                } });

    // set timeouts and change hostname and port
    svr.set_read_timeout(sparams.read_timeout);
    svr.set_write_timeout(sparams.write_timeout);

    if (!svr.bind_to_port(sparams.hostname, sparams.port))
    {
        fprintf(stderr, "\ncouldn't bind to server socket: hostname=%s port=%d\n\n", sparams.hostname.c_str(), sparams.port);
        return 1;
    }

    // Set the base directory for serving static files
    // svr.set_base_dir(sparams.public_path);

    // to make it ctrl+clickable:
    LOG_TEE("\nllama server listening at http://%s:%d\n\n", sparams.hostname.c_str(), sparams.port);

    std::unordered_map<std::string, std::string> log_data;
    log_data["hostname"] = sparams.hostname;
    log_data["port"] = std::to_string(sparams.port);

    LOG_INFO("HTTP server listening", log_data);
    // run the HTTP server in a thread - see comment below
    std::thread t([&]()
                  {
                      if (!svr.listen_after_bind())
                      {
                          state.store(SERVER_STATE_ERROR);
                          return 1;
                      }

                      return 0; });

    // load the model
    if (!llama.load_model(params))
    {
        state.store(SERVER_STATE_ERROR);
        return 1;
    }
    else
    {
        llama.initialize();
        state.store(SERVER_STATE_READY);
        LOG_INFO("model loaded", {});
    }

    svr.Post("/completion", [&llama, &validate_api_key](const httplib::Request &req, httplib::Response &res)
             {
        res.set_header(
            "Access-Control-Allow-Origin", 
            req.get_header_value("Origin"));

        json data = json::parse(req.body);
        const int task_id = llama.request_completion(data, false, false, -1);
        if (!json_value(data, "stream", false)) {
            std::string completion_text;
            task_result result = llama.next_result(task_id);
            if (!result.error && result.stop) {
                res.set_content(result.result_json.dump(-1, ' ', false, json::error_handler_t::replace), "application/json; charset=utf-8");
            }
            else
            {
                res.status = 404;
                res.set_content(result.result_json["content"], "text/plain; charset=utf-8");
                return;
            }
        } else {
            const auto chunked_content_provider = [task_id, &llama](size_t, httplib::DataSink & sink)
            {
                while (true)
                {
                    task_result result = llama.next_result(task_id);
                    if (!result.error) {
                        const std::string str =
                            "data: " +
                            result.result_json.dump(-1, ' ', false, json::error_handler_t::replace) +
                            "\n\n";
                        LOG_VERBOSE("data stream", {
                            { "to_send", str }
                        });
                        if (!sink.write(str.c_str(), str.size()))
                        {
                            return false;
                        }
                        if (result.stop) {
                            break;
                        }
                    } else {
                        const std::string str =
                            "error: " +
                            result.result_json.dump(-1, ' ', false, json::error_handler_t::replace) +
                            "\n\n";
                        LOG_VERBOSE("data stream", {
                            { "to_send", str }
                        });
                        if (!sink.write(str.c_str(), str.size()))
                        {
                            return false;
                        }
                        break;
                    }
                }
                sink.done();
                return true;
            };

            auto on_complete = [task_id, &llama] (bool)
            {
                // cancel
                llama.request_cancel(task_id);
            };

            res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
        } });

    svr.Options(R"(/.*)", [](const httplib::Request &, httplib::Response &res)
                { return res.set_content("", "application/json; charset=utf-8"); });

    // GG: if I put the main loop inside a thread, it crashes on the first request when build in Debug!?
    //     "Bus error: 10" - this is on macOS, it does not crash on Linux
    // std::thread t2([&]()
    {
        bool running = true;
        while (running)
        {
            running = llama.update_slots();
        }
    }
    //);

    t.join();

    llama_backend_free();
    return 0;
}