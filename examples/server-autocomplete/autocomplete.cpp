#include "common.h"
#include "llama.h"
#include "log.h"
#include "token_cache.h"
#include "search_info.h"
#include "task_management.h"

#ifndef NDEBUG
// crash the server in debug mode, otherwise send an http 500 error
#define CPPHTTPLIB_NO_EXCEPTIONS 1
#endif
// increase max payload length to allow use of larger context size
#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH 1048576

// SERVER_VERBOSE
#ifndef SERVER_VERBOSE
#define SERVER_VERBOSE 1
bool server_verbose = true;
#endif

#include "httplib.h"
#include "json.hpp"

#include <cstddef>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <atomic>

using json = nlohmann::json;

template <typename T>
static T json_value(const json &body,
                    const std::string &key,
                    const T &default_value)
{
    // Fallback null to default value
    return body.contains(key) && !body.at(key).is_null()
               ? body.value(key, default_value)
               : default_value;
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

static void log_server_request(const httplib::Request &req, const httplib::Response &res)
{
    //
    return;
}

struct llama_server_context
{
    /* unlike server.cpp we don't have slots as we won't process more than one suggestion in parallel */

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

    // system prompt
    bool system_need_update = false;

    // REVIEW: might have to change what this is for autocomplete
    // std::string system_prompt;
    // std::vector<llama_token> system_tokens;

    search_params s_params;

    task_management task;
    search_info search;

    // server_state state = SERVER_STATE_LOADING_MODEL;
    // SS: should be atomic
    std::atomic<server_state> state;

    // mutex for the task state
    std::mutex state_mutex;
    // lock for the task state
    std::condition_variable state_lock;

    token_cache cache;

    void clear()
    {
        // REVIEW: might have to change what this is for autocomplete
        // system_prompt.clear();
        // system_tokens.clear();
        // task.reset();
        cache.clear();
    }

    ~llama_server_context()
    {
        // ~cache();
        // delete cache;
    }

    void initialize()
    {
        search.clear();

        // load the model
        cache.init(256);
        state.store(SERVER_STATE_READY);
        task.set_state(IDLE);
    };

    bool handle_status(const int status)
    {

        switch (status)
        {
        case 0: // SUCCESS
            return true;
        case 1: // CANCELLED
            // response["n_decoded"] = search.decoded_tokens;
            // response["n_suggested"] = search.suggested_tokens;
            // response["suggested"] = search.suggested_string;
            return false;
        case 2: // WARNING
            // response["warning"] = "cache failed with warning";
            search.clear();
            task.set_state(IDLE);
            return false;
        case 3: // ERROR
            // response["error"] = "cache error";
            search.clear();
            cache.clear();
            task.set_state(IDLE);
            return false;
        }
        return false;
    }

    void complete(std::string prompt, json &response)
    {
        // LOG_INFO("Request received\n");
        LOG_VERBOSE("Prompt: %s\n", prompt.c_str());
        LOG_VERBOSE("Search String: %s\n", search.search_string.c_str());
        if (prompt == search.search_string)
        {
            // same prompt, no need to update
            response["n_decoded"] = search.decoded_tokens;
            response["n_suggested"] = search.suggested_tokens;
            response["suggested"] = search.suggested_string;
            return;
        }
        else
        {
            response["n_decoded"] = 0;
            response["n_suggested"] = 0;
            response["suggested"] = "";
        }

        // task.accept_if_cancel(); // if cancel request is pending, accept it
        task.request_cancel();   // WAIT FOR all threads to finish if not in IDLE state
        // search.clear();
        search.new_string = prompt;

        // CACHE_INIT
        task.set_state(CACHE_INIT);
        return;

        /* int status = cache.cache_init(search);

        if (!handle_status(status, response))
            return;
        if (task.accept_if_cancel())
            return;

        // CACHE_COMPARE
        if (!task.set_state(CACHE_COMPARE))
            return;

        status = cache.cache_compare(search, s_params);

        if (!handle_status(status, response))
            return;
        if (task.accept_if_cancel())
            return;

        // CACHE_PREPARE
        if (!task.set_state(CACHE_PREPARE))
            return;

        status = cache.cache_prepare(search);

        if (!handle_status(status, response))
            return;
        if (task.accept_if_cancel())
            return;

        // CACHE_UPDATE
        if (!task.set_state(CACHE_UPDATE))
            return;

        status = cache.cache_update(search, task);

        if (!handle_status(status, response))
            return;
        if (task.accept_if_cancel())
            return;

        // CACHE_SUGGEST
        if (!task.set_state(CACHE_SUGGEST))
            return;

        status = cache.cache_suggest(search, task);

        if (!handle_status(status, response))
            return;

        response["n_decoded"] = search.decoded_tokens;
        response["n_suggested"] = search.suggested_tokens;
        response["suggested"] = search.suggested_string;

        if (task.accept_if_cancel())
            return;

        task.set_state(IDLE);

        LOG_INFO("Request completed\n");

        cache.print_cache();

        return; */
    }
};

static void server_params_parse(int argc, char **argv, server_params &sparams,
                                gpt_params &params, llama_server_context &llama)
{
    gpt_params default_params;
    server_params default_sparams;
    std::string arg;
    bool invalid_param = false;

    for (int i = 1; i < argc; i++)
    {
        arg = argv[i];
        if (arg == "--port")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.port = std::stoi(argv[i]);
        }
        else if (arg == "--host")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.hostname = argv[i];
        }
        else if (arg == "--path")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.public_path = argv[i];
        }
        else if (arg == "--api-key")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.api_keys.push_back(argv[i]);
        }
        else if (arg == "--api-key-file")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            std::ifstream key_file(argv[i]);
            if (!key_file)
            {
                fprintf(stderr, "error: failed to open file '%s'\n", argv[i]);
                invalid_param = true;
                break;
            }
            std::string key;
            while (std::getline(key_file, key))
            {
                if (key.size() > 0)
                {
                    sparams.api_keys.push_back(key);
                }
            }
            key_file.close();
        }
        else if (arg == "--timeout" || arg == "-to")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.read_timeout = std::stoi(argv[i]);
            sparams.write_timeout = std::stoi(argv[i]);
        }
        else if (arg == "-m" || arg == "--model")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.model = argv[i];
        }
        else if (arg == "-a" || arg == "--alias")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.model_alias = argv[i];
        }
        else if (arg == "-h" || arg == "--help")
        {
            // server_print_usage(argv[0], default_params, default_sparams);
            exit(0);
        }
        else if (arg == "-c" || arg == "--ctx-size" || arg == "--ctx_size")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.n_ctx = std::stoi(argv[i]);
        }
        else if (arg == "--rope-scaling")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            std::string value(argv[i]);
            /**/ if (value == "none")
            {
                params.rope_scaling_type = LLAMA_ROPE_SCALING_NONE;
            }
            else if (value == "linear")
            {
                params.rope_scaling_type = LLAMA_ROPE_SCALING_LINEAR;
            }
            else if (value == "yarn")
            {
                params.rope_scaling_type = LLAMA_ROPE_SCALING_YARN;
            }
            else
            {
                invalid_param = true;
                break;
            }
        }
        else if (arg == "--rope-freq-base")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.rope_freq_base = std::stof(argv[i]);
        }
        else if (arg == "--rope-freq-scale")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.rope_freq_scale = std::stof(argv[i]);
        }
        else if (arg == "--yarn-ext-factor")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.yarn_ext_factor = std::stof(argv[i]);
        }
        else if (arg == "--yarn-attn-factor")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.yarn_attn_factor = std::stof(argv[i]);
        }
        else if (arg == "--yarn-beta-fast")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.yarn_beta_fast = std::stof(argv[i]);
        }
        else if (arg == "--yarn-beta-slow")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.yarn_beta_slow = std::stof(argv[i]);
        }
        else if (arg == "--threads" || arg == "-t")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.n_threads = std::stoi(argv[i]);
        }
        else if (arg == "--threads-batch" || arg == "-tb")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.n_threads_batch = std::stoi(argv[i]);
        }
        else if (arg == "-b" || arg == "--batch-size")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.n_batch = std::stoi(argv[i]);
            params.n_batch = std::min(512, params.n_batch);
        }
        else if (arg == "--gpu-layers" || arg == "-ngl" || arg == "--n-gpu-layers")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
#ifdef LLAMA_SUPPORTS_GPU_OFFLOAD
            params.n_gpu_layers = std::stoi(argv[i]);
#else
            // LOG_WARNING("Not compiled with GPU offload support, --n-gpu-layers option will be ignored. "
            //             "See main README.md for information on enabling GPU BLAS support",
            //             {{"n_gpu_layers", params.n_gpu_layers}});
#endif
        }
        else if (arg == "--tensor-split" || arg == "-ts")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
#ifdef GGML_USE_CUBLAS
            std::string arg_next = argv[i];

            // split string by , and /
            const std::regex regex{R"([,/]+)"};
            std::sregex_token_iterator it{arg_next.begin(), arg_next.end(), regex, -1};
            std::vector<std::string> split_arg{it, {}};
            GGML_ASSERT(split_arg.size() <= LLAMA_MAX_DEVICES);

            for (size_t i_device = 0; i_device < LLAMA_MAX_DEVICES; ++i_device)
            {
                if (i_device < split_arg.size())
                {
                    params.tensor_split[i_device] = std::stof(split_arg[i_device]);
                }
                else
                {
                    params.tensor_split[i_device] = 0.0f;
                }
            }
#else
            // LOG_WARNING("llama.cpp was compiled without cuBLAS. It is not possible to set a tensor split.\n", {});
#endif // GGML_USE_CUBLAS
        }
        else if (arg == "--no-mul-mat-q" || arg == "-nommq")
        {
#ifdef GGML_USE_CUBLAS
            params.mul_mat_q = false;
#else
            // LOG_WARNING("warning: llama.cpp was compiled without cuBLAS. .Disabling mul_mat_q kernels has no effect.\n", {});
#endif // GGML_USE_CUBLAS
        }
        else if (arg == "--main-gpu" || arg == "-mg")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
#ifdef GGML_USE_CUBLAS
            params.main_gpu = std::stoi(argv[i]);
#else
            // LOG_WARNING("llama.cpp was compiled without cuBLAS. It is not possible to set a main GPU.", {});
#endif
        }
        else if (arg == "--lora")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.lora_adapter.push_back(std::make_tuple(argv[i], 1.0f));
            params.use_mmap = false;
        }
        else if (arg == "--lora-scaled")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            const char *lora_adapter = argv[i];
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.lora_adapter.push_back(std::make_tuple(lora_adapter, std::stof(argv[i])));
            params.use_mmap = false;
        }
        else if (arg == "--lora-base")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.lora_base = argv[i];
        }
        else if (arg == "-v" || arg == "--verbose")
        {
#if SERVER_VERBOSE != 1
            // LOG_WARNING("server.cpp is not built with verbose logging.", {});
#else
            // server_verbose = true;
#endif
        }
        else if (arg == "--mlock")
        {
            params.use_mlock = true;
        }
        else if (arg == "--no-mmap")
        {
            params.use_mmap = false;
        }
        else if (arg == "--numa")
        {
            params.numa = true;
        }
        else if (arg == "--embedding")
        {
            params.embedding = true;
        }
        else if (arg == "-cb" || arg == "--cont-batching")
        {
            params.cont_batching = true;
        }
        else if (arg == "-np" || arg == "--parallel")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.n_parallel = std::stoi(argv[i]);
        }
        else if (arg == "-n" || arg == "--n-predict")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.n_predict = std::stoi(argv[i]);
        }
        else if (arg == "-spf" || arg == "--system-prompt-file")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            std::ifstream file(argv[i]);
            if (!file)
            {
                fprintf(stderr, "error: failed to open file '%s'\n", argv[i]);
                invalid_param = true;
                break;
            }
            std::string systm_content;
            std::copy(
                std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>(),
                std::back_inserter(systm_content));
            // llama.process_system_prompt_data(json::parse(systm_content));
        }
        else if (arg == "--mmproj")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.mmproj = argv[i];
        }
        else if (arg == "--log-disable")
        {
            log_set_target(stdout);
            // LOG_INFO("logging to file is disabled.", {});
        }
        else if (arg == "--override-kv")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            char *sep = strchr(argv[i], '=');
            if (sep == nullptr || sep - argv[i] >= 128)
            {
                fprintf(stderr, "error: Malformed KV override: %s\n", argv[i]);
                invalid_param = true;
                break;
            }
            struct llama_model_kv_override kvo;
            std::strncpy(kvo.key, argv[i], sep - argv[i]);
            kvo.key[sep - argv[i]] = 0;
            sep++;
            if (strncmp(sep, "int:", 4) == 0)
            {
                sep += 4;
                kvo.tag = LLAMA_KV_OVERRIDE_INT;
                kvo.int_value = std::atol(sep);
            }
            else if (strncmp(sep, "float:", 6) == 0)
            {
                sep += 6;
                kvo.tag = LLAMA_KV_OVERRIDE_FLOAT;
                kvo.float_value = std::atof(sep);
            }
            else if (strncmp(sep, "bool:", 5) == 0)
            {
                sep += 5;
                kvo.tag = LLAMA_KV_OVERRIDE_BOOL;
                if (std::strcmp(sep, "true") == 0)
                {
                    kvo.bool_value = true;
                }
                else if (std::strcmp(sep, "false") == 0)
                {
                    kvo.bool_value = false;
                }
                else
                {
                    fprintf(stderr, "error: Invalid boolean value for KV override: %s\n", argv[i]);
                    invalid_param = true;
                    break;
                }
            }
            else
            {
                fprintf(stderr, "error: Invalid type for KV override: %s\n", argv[i]);
                invalid_param = true;
                break;
            }
            params.kv_overrides.push_back(kvo);
        }
        else
        {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            // server_print_usage(argv[0], default_params, default_sparams);
            exit(1);
        }
    }
    if (!params.kv_overrides.empty())
    {
        params.kv_overrides.emplace_back(llama_model_kv_override());
        params.kv_overrides.back().key[0] = 0;
    }

    if (invalid_param)
    {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        // server_print_usage(argv[0], default_params, default_sparams);
        exit(1);
    }
}

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

    // if (params.model_alias == "unknown")
    //{
    //     params.model_alias = params.model;
    // }

    // llama_backend_init(params.numa);

    /*LOG_INFO("build info", {{"build", LLAMA_BUILD_NUMBER},
                            {"commit", LLAMA_COMMIT}});

    LOG_INFO("system info", {
                                {"n_threads", params.n_threads},
                                {"n_threads_batch", params.n_threads_batch},
                                {"total_threads", std::thread::hardware_concurrency()},
                                {"system_info", llama_print_system_info()},
                            }); */

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
    LOG_INFO("\nllama server listening at http://%s:%d\n\n", sparams.hostname.c_str(), sparams.port);

    std::unordered_map<std::string, std::string> log_data;
    log_data["hostname"] = sparams.hostname;
    log_data["port"] = std::to_string(sparams.port);

    LOG_INFO("HTTP server listening");
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
    llama.initialize();
    LOG_INFO("Model loaded.\n");
    state.store(SERVER_STATE_READY);

    svr.Post("/completion",
             [&llama](const httplib::Request &req, httplib::Response &res)
             {
                 res.set_header(
                     "Access-Control-Allow-Origin",
                     req.get_header_value("Origin"));

                 json data = json::parse(req.body);
                 std::string prompt = json_value(data, "prompt", std::string());
                 json response;

                 llama.complete(prompt, response);

                 res.set_content(response.dump(), "application/json");
             });

    svr.Options(R"(/.*)", [](const httplib::Request &, httplib::Response &res)
                { return res.set_content("", "application/json; charset=utf-8"); });

    // GG: if I put the main loop inside a thread, it crashes on the first request when build in Debug!?
    //     "Bus error: 10" - this is on macOS, it does not crash on Linux
    // std::thread t2([&]()
    token_cache &cache = llama.cache;
    search_info &search = llama.search;
    task_management &task = llama.task;

    {
        bool running = true;
        int status;

        while (running)
        {

            LOG_INFO("Waiting for request\n");

            task.wait_for_state(CACHE_INIT);

            LOG_INFO("Request received\n");

            status = cache.cache_init(search);

            LOG_INFO("CACHE_INIT status: %d\n", status);
            // log search.search_string
            LOG_VERBOSE("Search String: %s\n", search.search_string.c_str());

            if (!llama.handle_status(status))
                continue;
            if (task.accept_if_cancel())
                continue;

            // CACHE_COMPARE
            if (!task.set_state(CACHE_COMPARE))
                continue;

            status = cache.cache_compare(search, llama.s_params);

            if (!llama.handle_status(status))
                continue;
            if (task.accept_if_cancel())
                continue;

            // CACHE_PREPARE
            if (!task.set_state(CACHE_PREPARE))
                continue;

            status = cache.cache_prepare(search);

            if (!llama.handle_status(status))
                continue;
            if (task.accept_if_cancel())
                continue;

            // CACHE_UPDATE
            if (!task.set_state(CACHE_UPDATE))
                continue;

            status = cache.cache_update(search, task);

            if (!llama.handle_status(status))
                continue;
            if (task.accept_if_cancel())
                continue;

            // CACHE_SUGGEST
            if (!task.set_state(CACHE_SUGGEST))
                continue;

            status = cache.cache_suggest(search, task);

            if (!llama.handle_status(status))
                continue;

            // if (task.accept_if_cancel())
            //    continue;

            task.set_state(IDLE);

            LOG_INFO("Processing completed\n");
            cache.print_cache();
        }
    }
    //);

    t.join();

    llama_backend_free();
    return 0;
}