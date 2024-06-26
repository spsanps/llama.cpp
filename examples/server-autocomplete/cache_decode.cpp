#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "json.hpp"

using json = nlohmann::json;

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

std::string read_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL)
    {
        return "";
    }

    fseek(f, 0, SEEK_END);
    const size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::string data(size, 0);
    fread(&data[0], 1, size, f);
    fclose(f);

    return data;
}

int main(int argc, char **argv)
{
    gpt_params params;
    if (gpt_params_parse(argc, argv, params) == false)
    {
        return 1;
    }

    params.n_ctx = 2048;
    params.n_batch = 2048;

    llama_backend_init(params.numa);

    llama_model *model = NULL;

    llama_context *ctx = NULL;
    const char *model_path = "mist-7b-sft-gutenberg-50k.gguf";
    // set model (std:string type)
    params.model = model_path;

    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    // llama_load_model_from_file(model_path, params, model);

    LOG_INFO("build info", {{"build", LLAMA_BUILD_NUMBER},
                            {"commit", LLAMA_COMMIT}});

    LOG_INFO("system info", {
                                {"n_threads", params.n_threads},
                                {"n_threads_batch", params.n_threads_batch},
                                {"total_threads", std::thread::hardware_concurrency()},
                                {"system_info", llama_print_system_info()},
                            });

    // what is add bos, let's see
    const bool add_bos = llama_should_add_bos_token(model);
    LOG_INFO("add_bos: %d\n", add_bos);

    // declare prompt of type string
    std::string prompt;
    // read entire sample.txt file into prompt
    prompt = read_file("sample.txt");


    params.prompt = prompt.c_str(); 

    std::vector<llama_token> inp;
    inp = ::llama_tokenize(ctx, params.prompt, add_bos, true);

    // delete last element of inp
    // inp.pop_back();

    const int max_context_size = llama_n_ctx(ctx);
    const int max_tokens_list_size = max_context_size - 4; // i guess 4 is for some kind of extra 'buffer' to handle special tokens

    // print inp.size()
    fprintf(stderr, "inp.size(): %d\n", (int)inp.size());

    // print max_context_size
    fprintf(stderr, "max_context_size: %d\n", max_context_size);

    if ((int)inp.size() > max_tokens_list_size)
    {
        fprintf(stderr, "%s: error: prompt too long (%d tokens, max %d)\n", __func__, (int)inp.size(), max_tokens_list_size);
        return 1;
    }

    fprintf(stderr, "\n\n");

    //
    for (auto id : inp)
    {
        // fprintf(stderr, "%s", llama_token_to_piece(ctx, id).c_str());
        // regular print not error print
        fprintf(stderr, "%s", llama_token_to_piece(ctx, id).c_str());
    }

    fflush(stderr);

    const int n_input = inp.size();

    // print inputs as is
    /*fprintf(stderr, "\n\nInputs as is: \n");
    for (int i = 0; i < n_input; ++i)
    {
        fprintf(stderr, "%d ", inp[i]);
    }
    */
    llama_batch batch = llama_batch_init(n_input, 0, 1);

    for (int i = 0; i < n_input; ++i)
    {
        llama_batch_add(batch, inp[i], i, {0}, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    auto t_enc_start = std::chrono::high_resolution_clock::now();

    llama_decode(ctx, batch);

    auto t_enc_end = std::chrono::high_resolution_clock::now();
    auto enc_duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_enc_end - t_enc_start).count();

    // print time
    fprintf(stderr, "Encoding time: %lld ms\n", enc_duration);

    int n_predict = 128;

    int n_past = inp.size();

    // used to determine end of generation
    bool has_eos = false;

    // target model sampling context
    struct llama_sampling_context *ctx_sampling = llama_sampling_init(params.sparams);

    const auto t_dec_start = ggml_time_us();

    int i_dft = n_input;
    int s_keep = 0;

    // print
    fprintf(stderr, "Entering loop\n");


    for (int i = 0; i < n_predict; ++i)
    {
        /*fprintf(stderr, "i_dft: %d\n", i);
        llama_batch batch_view =
            {
                1,
                batch.token + i,
                nullptr,
                batch.pos + i,
                batch.n_seq_id,
                batch.seq_id + i,
                nullptr,
                0, 0, 0, // unused
            };
        */

        // const int ret = llama_decode(ctx, batch);

        // print i_dft
        // fprintf(stderr, "i_dft2: %d\n", i);

        llama_token id = 0;

        // sample from the target model
        if (i == 0) {
        id = llama_sampling_sample(ctx_sampling, ctx, NULL, n_input + i - 1);
        } else {
        id = llama_sampling_sample(ctx_sampling, ctx, NULL, 0);
        }

        // print token as is
        // fprintf(stderr, "Token as is: %d\n", id);

        // print
        // fprintf(stderr, "Sampling done\n");

        llama_sampling_accept(ctx_sampling, ctx, id, false);

        // print
        // fprintf(stderr, "Accept done\n");

        // LOG("last: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_tgt, ctx_sampling->prev).c_str());

        std::string token_str = llama_token_to_piece(ctx, id);

        // if (!params.use_color)
        // fprintf(stderr, "Token is: |%s|", token_str.c_str());
        fprintf(stderr, "%s", token_str.c_str());

        // print
        // fprintf(stderr, "\nToken done\n");

        // create new batch
        // llama_batch batch_new = llama_batch_get_one(&id, 1, n_input + i, 0);
        llama_batch_clear(batch);
        llama_batch_add(batch, id, n_input + i, {0}, true);

        // print
        // fprintf(stderr, "Batch_new done\n");

        // print batch.logits[batch.n_tokens - 1] for batch_new
        // batch_new.logits[batch_new.n_tokens - 1] = true;
        // print

        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch))
        {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        // print
        // fprintf(stderr, "Decode done\n");
    }

    auto t_dec_end = ggml_time_us();

    // print
    fprintf(stderr, "Decoding done\n");
    fprintf(stderr, "Decoding time: %lld ms\n", (t_dec_end - t_dec_start) / 1000);

    llama_print_timings(ctx);

    // lets delete some samples and go back -100 tokens
    int start_pos = -1;
    int end_pos = n_input + n_predict - 100;
    // using sequence 1
    llama_kv_cache_seq_cp(ctx, 0, 1, start_pos, end_pos);
    llama_sampling_reset(ctx_sampling);
    llama_set_rng_seed(ctx, 0);
    
    // get token from what we have
    // replace with actual token that was generated for the next position after -100
    llama_token id = 0;

    llama_sampling_accept(ctx_sampling, ctx, id, false);

    llama_batch_clear(batch);
    llama_batch_add(batch, id, n_input + n_predict - 100, {1}, true);

    llama_decode(ctx, batch);

    // accept

    fprintf(stderr, "Decoding again\n");

    // let's decode again
    for(int i = 0; i < 100; ++i) {
        llama_token id = 0;

        // sample from the target model

        id = llama_sampling_sample(ctx_sampling, ctx, NULL, 0);

        // print token as is
        // fprintf(stderr, "Token as is: %d\n", id);

        // print
        // fprintf(stderr, "Sampling done\n");

        llama_sampling_accept(ctx_sampling, ctx, id, false);

        // print
        // fprintf(stderr, "Accept done\n");

        // LOG("last: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_tgt, ctx_sampling->prev).c_str());

        std::string token_str = llama_token_to_piece(ctx, id);

        // if (!params.use_color)
        // fprintf(stderr, "Token is: |%s|", token_str.c_str());
        fprintf(stderr, "%s", token_str.c_str());

        // print
        // fprintf(stderr, "\nToken done\n");

        // create new batch
        // llama_batch batch_new = llama_batch_get_one(&id, 1, n_input + i, 0);
        llama_batch_clear(batch);
        llama_batch_add(batch, id, n_input + n_predict - 100 + i, {0}, true);

        // print
        // fprintf(stderr, "Batch_new done\n");

        // print batch.logits[batch.n_tokens - 1] for batch_new
        // batch_new.logits[batch_new.n_tokens - 1] = true;
        // print

        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch))
        {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        // print
        // fprintf(stderr, "Decode done\n");
    }
    



    return 0;

    

}