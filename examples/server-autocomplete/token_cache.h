#ifndef TOKEN_CACHE_H
#define TOKEN_CACHE_H

#include "common.h"
#include "llama.h"
#include "rolling.h"
#include "log.h"
#include "task_management.h"
#include "search_info.h"
#include "kmp.h"

#include <set>
#include <string>
#include <vector>
#include <deque>

#include <deque>
#include <cstddef>
#include <chrono>
#include <iostream>
#include <vector>
#include <list>
#include <set>

// SS:
// A token cache implementation to compare against new prompts
// similar to llama_kv_cache in llama.cpp
// does llama.cpp keep track of tokens?
// if so I am reinventing the wheel here
// I suppose it can't hurt
// or at least I can learn something

// -1 is used to denote empty cells
// it can be token, pos or seq_id
struct llama_token_cell
{
    // equivalent to llama_kv_cell
    llama_token token = -1;
    bool has_logits = false;
    llama_pos pos = -1;
    std::string piece; // string representation of the token
    // llama_pos delta = 0; // what is this for?

    std::set<llama_seq_id> seq_id;

    bool has_seq_id(const llama_seq_id &id) const
    {
        return seq_id.find(id) != seq_id.end();
    }

    void reset()
    {
        token = -1;
        pos = -1;
        seq_id.clear();
        has_logits = false;
        piece = "";
    }
};

struct seq_info
{
    llama_seq_id seq_id;
    int start_ind;
    int end_ind;
    int length;
};

struct token_cache
{
    // equivalent to llama_kv_cache

    int n_max_seq;

    int active_seq_id = -1;

    std::vector<llama_token_cell> cache;
    int max_cache_size = 4096;      // eg. 4096 (2 * max_sequence_length)
    int max_sequence_length = 2048; // eg. 2048 (max_cache_size / 2)
    int suggestion_length = 32;     // eg. 32
    int batch_size = 32;            // eg. 32

    int n_empty;

    // we will have a list of seq_info
    // we will use this vector to find the latest seq_id
    // and the oldest seq_id when we need to clean the cache
    std::list<seq_info> seq_queue;
    seq_info empty_seq_info = {-1, -1, -1, -1};

    // kv cache related stuff
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;

    // llama_batch batch;

    gpt_params params;

    void clear()
    {
        cache.clear();
        seq_queue.clear();
        llama_kv_cache_clear(ctx);
    }

    ~token_cache()
    {
        clear();
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

    void init(int max_size)
    {
        // initialize the cache
        cache.resize(max_size);
        max_cache_size = max_size;
        n_empty = max_size;

        active_seq_id = -1;

        // TODO: do better initialization
        const char *model_path = "mist-7b-sft-gutenberg-50k.gguf";
        // set model (std:string type)
        params.model = model_path;
        params.n_ctx = max_size;
        params.n_batch = max_size;

        std::tie(model, ctx) = llama_init_from_gpt_params(params);
    }

    void print_cache() const
    {

        std::cout << "Cache Contents:\n";
        for (int i = 0; i < max_cache_size; i++)
        {
            const auto &cell = cache[i];
            std::cout << "Index " << i << ": Token = " << cell.token
                      << ", Pos = " << cell.pos << ", Seq IDs = {";
            for (const auto &id : cell.seq_id)
            {
                std::cout << id << " ";
            }
            std::cout << "}\n";
        }

        std::cout << "\nSequence Queue:\n";
        for (const auto &info : seq_queue)
        {
            std::cout << "Seq ID = " << info.seq_id << ", Start Index = " << info.start_ind
                      << ", End Index = " << info.end_ind << ", Length = " << info.length << "\n";
        }
    }

    void print_seq_queue() const
    {
        std::cout << "\nSequence Queue:\n";
        for (const auto &info : seq_queue)
        {
            std::cout << "Seq ID = " << info.seq_id << ", Start Index = " << info.start_ind
                      << ", Length = " << info.length << "\n";
        }
    }
    // index related functions

    inline int get_next_ind(int curr_ind)
    {
        // get next ind in cache (circular)
        return (curr_ind + 1) % max_cache_size;
    }

    inline int get_prev_ind(int curr_ind)
    {
        // get prev ind in cache (circular)
        return (curr_ind - 1 + max_cache_size) % max_cache_size;
    }

    int get_next_ind_in_seq(int curr_ind, int seq_id)
    {
        // get next ind in cache (circular) belonging to a sequence

        /*if (seq_id == -1)
        {
            return get_next_ind_empty(curr_ind);
        }*/
        // SS: this is not needed as we do the check where needed

        int next_ind = curr_ind;
        for (int i = 0; i < max_cache_size; i++)
        {
            next_ind = get_next_ind(next_ind);
            if (cache[next_ind].has_seq_id(seq_id))
            {
                // if next_ind is before curr_ind means we have looped around
                // could be a bug if we have positions being overwritten
                if (cache[next_ind].pos < cache[curr_ind].pos)
                {
                    return -1;
                }
                return next_ind;
            }
        }
        return -1;
    }

    int get_prev_ind_in_seq(int curr_ind, int seq_id)
    {
        // get prev ind in cache (circular) belonging to a sequence

        /*if (seq_id == -1)
        {
            return get_next_ind_empty(curr_ind);
        }*/
        // SS: this is not needed as we do the check where needed

        int prev_ind = curr_ind;
        for (int i = 0; i < max_cache_size; i++)
        {
            prev_ind = get_prev_ind(prev_ind);
            if (cache[prev_ind].has_seq_id(seq_id))
            {
                // if prev_ind is after curr_ind means we have looped around
                // could be a bug if we have positions being overwritten
                if (cache[prev_ind].pos > cache[curr_ind].pos)
                {
                    return -1;
                }
                return prev_ind;
            }
        }
        return -1;
    }

    int get_next_ind_empty(int curr_ind)
    {
        // get next ind in cache (circular) that is empty
        int next_ind = curr_ind;
        for (int i = 0; i < max_cache_size; i++)
        {
            next_ind = get_next_ind(next_ind);
            if (cache[next_ind].token == -1)
            {
                return next_ind;
            }
        }
        return -1;
    }

    // cell related functions

    int find_first_empty_cell()
    {
        // find the first empty cell in the cache
        for (int i = 0; i < max_cache_size; i++)
        {
            if (cache[i].token == -1)
            {
                return i;
            }
        }
        return -1;
    }

    bool _is_token_at_index(const llama_token &token, const int &index)
    {
        return cache[index].token == token;
    }

    void _insert_token_at_cell(
        llama_token_cell &cell,
        llama_token token,
        llama_seq_id seq_id,
        int position,
        bool has_logits = false)
    {
        if (cell.token == -1)
        {
            n_empty--;
        }
        cell.token = token;
        cell.seq_id.insert(seq_id);
        cell.has_logits = has_logits;
        cell.pos = position;
        cell.piece = llama_token_to_piece(ctx, token);
    }

    void _delete_cell(llama_token_cell &cell)
    {
        if (cell.token != -1)
        {
            n_empty++;
            cell.reset();
        }
    }

    void _delete_cache_cell(const int index, const llama_seq_id seq_id)
    {
        // check if seq_id is present in the cell
        /*if (!cache[index].has_seq_id(seq_id))
        {
            return;
        }*/
        // SS: assume this check is done where needed

        llama_kv_cache_seq_rm(ctx, seq_id, index, index + 1);

        cache[index].seq_id.erase(seq_id);

        if (cache[index].seq_id.size() == 0)
        {
            _delete_cell(cache[index]);
        }
    }

    // seq_id related functions

    bool has_seq_id(const llama_seq_id &seq_id)
    {
        for (auto &info : seq_queue)
        {
            if (info.seq_id == seq_id)
            {
                return true;
            }
        }
        return false;
    }

    llama_seq_id get_latest_seq_id()
    {
        if (seq_queue.size() == 0)
        {
            return -1;
        }
        return seq_queue.front().seq_id;
    }

    llama_seq_id get_oldest_seq_id()
    {
        if (seq_queue.size() == 0)
        {
            return -1;
        }
        return seq_queue.back().seq_id;
    }

    seq_info &get_seq_info(const llama_seq_id &seq_id)
    {
        // get the seq_info of a seq_id
        for (auto &info : seq_queue)
        {
            if (info.seq_id == seq_id)
            {
                return info;
            }
        }
        return empty_seq_info;
    }

    bool get_empty_seq_id(llama_seq_id &seq_id)
    {
        // get an empty seq_id
        // if no empty seq_id, return false
        // if empty seq_id, return true and set seq_id to the empty seq_id
        for (int i = 0; i <= seq_queue.size(); i++)
        {
            if (!has_seq_id(i))
            {
                seq_id = i;
                return true;
            }
        }
        return false;
    }

    void _delete_seq_id(const llama_seq_id seq_id)
    {
        seq_queue.remove_if([&seq_id](const seq_info &si)
                            { return si.seq_id == seq_id; });
    }

    // sequence related functions

    void delete_seq(const llama_seq_id seq_id)
    {

        // delete all tokens belonging to a sequence
        for (auto &cell : cache)
        {
            cell.seq_id.erase(seq_id);
            if (cell.seq_id.size() == 0)
            {
                _delete_cell(cell);
            }
        }
        LOG_TEE("Deleted a sequence in cache\n");
        _delete_seq_id(seq_id);
        llama_kv_cache_seq_rm(ctx, seq_id, -1, -1); // delete entire sequence
    }

    bool delete_oldest_seq_but_not(const llama_seq_id seq_id)
    {
        // delete the oldest sequence in the cache
        // iterate backwards from end of seq_queue
        for (auto it = seq_queue.rbegin(); it != seq_queue.rend(); ++it)
        {
            if (it->seq_id != seq_id)
            {
                delete_seq(it->seq_id);
                return true;
            }
        }
        return false;
    }

    void copy_seq(
        const std::deque<int> &indices,
        const llama_seq_id &target_seq_id)
    {
        // insert target_seq_id at indices
        for (auto &ind : indices)
        {
            cache[ind].seq_id.insert(target_seq_id);
        }
    }

    // operations

    // return 0 - success
    // return 1 - cancelled
    // return 2 - warning
    // return 3 - error - fatal

    int cache_init(
        search_info &s_info) //,
    // task_management &task_manager)
    {
        // first operation with new sequence
        // we will tokenize and trim the sequence
        // setup s_info with the new sequence

        std::string &s_info_string = s_info.search_string;
        std::vector<llama_token> &seq = s_info.seq;
        std::string &new_string = s_info.new_string;

        s_info.clear(true); // partial clear

        int match = kmpMatchSuffixPrefix(s_info_string, new_string);

        LOG_INFO("Initializing cache for new sequence\n");

        if (match <= 0)
        {
            LOG_VERBOSE("Sequence is entirely new\n");
            s_info_string = new_string;
            seq = llama_tokenize(ctx, new_string.c_str(), false, false);
        }
        else
        {
            LOG_VERBOSE("Sequence is a continuation\n");
            // we only need to add new string
            // which is from match to end

            // extend the existing sequence
            std::vector<llama_token> new_seq =
                llama_tokenize(ctx,
                               new_string.substr(match).c_str(),
                               false,
                               false);

            seq.insert(seq.end(), new_seq.begin(), new_seq.end());

            s_info_string = new_string;
        }
        // print the tokens
        LOG_VERBOSE("Tokens: ");
        if (verbose_level > 0)
        {
            for (const auto &token : seq)
            {
                printf("%d ", token);
            }
        }
        LOG_VERBOSE("\n");

        // trim the sequence
        // we will trim from the left

        if (seq.size() <= 0)
        {
            LOG_WARNING("Empty sequence\n");
            return 2;
        }

        int required_length = max_sequence_length - suggestion_length;
        LOG_VERBOSE("Required Length: %d\n", required_length);
        if (required_length <= seq.size()) // = for 0 bos token
        {
            // if the sequence is too long, trim it
            // the resulting sequence will be of length required_length
            LOG_VERBOSE("Trimming sequence\n");
            seq.erase(seq.begin(),
                      seq.begin() + (seq.size() - required_length));

            if (match <= 0) // no need to add BOS token if sequence is a continuation
            {
                seq[0] = llama_token_bos(llama_get_model(ctx));
            }
        }
        else
        {
            // inefficient
            // TODO: maybe BOS can be part of the system prompt
            if (match <= 0)
            {
                seq.insert(seq.begin(), llama_token_bos(llama_get_model(ctx)));
            }
        }

        // log new size
        LOG_VERBOSE("New Size: %d\n", seq.size());

        return 0;
    }

    bool _match_pattern_seq_with_trim(
        const std::vector<llama_token> &pattern,
        const int32_t &ltrim, // how much of the pattern should be trimmed (from left)
        const int32_t &rtrim, // how much of the pattern should be trimmed (from right)
        const llama_seq_id &seq_id,
        const search_params &params,
        rolling_buffer &seq_indices)
    {
        // we will use rabin-karp algorithm to match the pattern
        // ie a rolling hash to match the pattern

        if (pattern.size() == 0)
        {
            return false;
        }
        if (rtrim >= pattern.size() || ltrim >= pattern.size())
        {
            return false;
        }

        if (seq_id == -1)
        {
            return false;
        }

        LOG_VERBOSE("Searching for pattern: ");

        seq_info info = get_seq_info(seq_id);
        int seq_length = info.length;

        LOG_VERBOSE("Seq ID: %d, Start Ind: %d, Length: %d\n", seq_id, info.start_ind, seq_length);

        int pattern_length = pattern.size() - rtrim - ltrim;
        LOG_VERBOSE("Pattern Length: %d\n", pattern_length);

        // if pattern is larger than cache, return empty vector
        if (pattern_length > seq_length)
        {
            return false;
        }

        // compute hash of the trimmed pattern
        rolling_hash pattern_rolling_hash;
        pattern_rolling_hash.init(params.a, params.mod);
        for (int i = 0; i < pattern_length; i++)
        {
            pattern_rolling_hash.add(static_cast<uint64_t>(pattern[i + ltrim]));
        }
        LOG_VERBOSE("Pattern Hash: %llu\n", pattern_rolling_hash.hash);

        // go to start_ind in cache
        int curr_ind = info.start_ind;

        // compute hash of the first pattern_length tokens in cache
        rolling_hash seq_rolling_hash;
        seq_rolling_hash.init(params.a, params.mod);

        // rolling_buffer to store the indices of the tokens in the cache for the sequence so we can do a rolling hash
        seq_indices.init(pattern_length);

        // compute hash of the first pattern_length tokens in cache
        for (int i = 0; i < pattern_length; i++)
        {
            seq_rolling_hash.add(static_cast<uint64_t>(cache[curr_ind].token));
            seq_indices.push_back(curr_ind);
            curr_ind = get_next_ind_in_seq(curr_ind, seq_id);
        }

        LOG_VERBOSE("Seq Hash: %llu\n", seq_rolling_hash.hash);

        int i = 0;
        int max_roll = seq_length - pattern_length + 1;
        // rolling hash
        do
        {
            if (seq_rolling_hash.hash == pattern_rolling_hash.hash)
            {
                // hashes match, check if tokens match
                bool match = true;
                for (int j = 0; j < pattern_length; j++)
                {
                    if (!_is_token_at_index(pattern[j + ltrim], seq_indices[j]))
                    {
                        match = false;
                        break;
                    }
                }
                if (match)
                {
                    // we have found a match
                    return true;
                }
            }

            if (curr_ind == -1)
            {
                // we have reached the end of the sequence
                break;
            }

            // remove the first token from the hash
            // and add the next token to the hash
            seq_rolling_hash.add_remove(
                static_cast<uint64_t>(cache[seq_indices.front()].token),
                static_cast<uint64_t>(cache[curr_ind].token));

            // update the seq_indices
            seq_indices.push_back(curr_ind);

            LOG_VERBOSE("Seq Hash: %llu\n", seq_rolling_hash.hash);

            // update curr_ind
            curr_ind = get_next_ind_in_seq(curr_ind, seq_id);

        } while (++i < max_roll);

        return false;
    }

    int cache_compare(
        search_info &s_info,
        // const std::vector<llama_token> &pattern,
        const search_params &params
        // llama_seq_id &match_seq_id
        // std::deque<int> &result_indices
        ) // indices of matches - if empty, no match
    {

        const std::vector<llama_token> &pattern = s_info.seq;
        int &new_tokens = s_info.new_tokens;
        llama_seq_id &match_seq_id = s_info.match_seq_id;
        std::deque<int> &result_indices = s_info.indices;

        bool &extend = s_info.extend;

        LOG_INFO("Comparing cache with pattern\n");

        if (pattern.size() > max_sequence_length || pattern.size() <= 1) // pattern will always have BOS tokens empty is 1
        {
            LOG_WARNING("Pattern too large or too small\n");
            return 2;
        }

        int pattern_start_size = params.pattern_start_size;
        int &ltrim = s_info.ltrim;
        int &rtrim = s_info.rtrim;
        if (pattern.size() <= pattern_start_size)
        {
            pattern_start_size = pattern.size() - 1; // -1 for BOS token
            ltrim = 1;                               // BOS token
        }
        else
        {
            ltrim = pattern.size() - pattern_start_size; // BOS token will be included
        }

        int min_overlap_requested = params.minimum_overlap_required;
        // using pattern_start_size as kind of a buffer for the minimum overlap
        // as pattern_start_size is kind of set to tell how many new tokens we can add at a time
        // this is to avoid the case when we are at "minimum_overlap_required" length and we have new tokens to add
        // we should still match even if we don't have the minimum overlap
        if (pattern.size() < min_overlap_requested + pattern_start_size)
        {
            min_overlap_requested = -1; // any overlap is okay
        }

        int trim_skip_amount = params.trim_skip_amount;
        if (pattern.size() < trim_skip_amount)
        {
            trim_skip_amount = params.trim_skip_small_amount;
        }

        int pattern_length = pattern.size() - rtrim - ltrim;
        rolling_buffer seq_indices;
        bool match = false;
        while (pattern_length > 0)
        {
            seq_indices.reset();
            // iterate over the sequence queue from newest to oldest
            for (auto &info : seq_queue)
            {
                // search the sequence
                if (_match_pattern_seq_with_trim(pattern,
                                                 ltrim,
                                                 rtrim,
                                                 info.seq_id,
                                                 params,
                                                 seq_indices))
                {
                    // we have found a match
                    // add the indices to the result_indices
                    for (int i = 0; i < pattern_length; i++)
                    {
                        result_indices.push_back(seq_indices[i]);
                    }
                    match = true;
                    match_seq_id = info.seq_id;
                    if (rtrim == 0)
                    {
                        extend = true;
                    }
                    else if (seq_indices.back() == info.end_ind)
                    {
                        extend = true;
                    }
                    break;
                }
            }

            if (match)
            {
                // we have found a match
                LOG_VERBOSE("Match found\n");
                break;
            }

            // trim the pattern from right
            rtrim += trim_skip_amount;
            pattern_length = pattern.size() - rtrim - ltrim;
        }

        if (!match)
        {
            // we have not found a match
            // return empty vector
            // no match is not an error
            // empty the result_indices
            result_indices.clear();
            match_seq_id = -1;
            new_tokens = pattern.size();
            ltrim = 0;
            rtrim = 0;
            extend = false;
            LOG_VERBOSE("No match found\n");
            return 0;
        }

        // we have found a match
        // we will now see if we can extend the match
        // we will now see if there are any matching tokens in the left and right of the match
        LOG_VERBOSE("Extending match\n");
        // tokens to the left
        int cache_ind = result_indices.front();
        for (int i = ltrim - 1; i >= 0; i--)
        {
            cache_ind = get_prev_ind_in_seq(cache_ind, match_seq_id);
            if (cache_ind == -1)
            {
                break;
            }
            if (_is_token_at_index(pattern[i], cache_ind))
            {
                result_indices.push_front(cache_ind);
                ltrim--;
            }
            else
            {
                break;
            }
        }

        LOG_VERBOSE("Left Trim: %d\n", ltrim);
        LOG_VERBOSE("Right Trim: %d\n", rtrim);

        // tokens to the right
        cache_ind = result_indices.back();
        for (int i = pattern.size() - rtrim; i < pattern.size(); i++)
        {

            cache_ind = get_next_ind_in_seq(cache_ind, match_seq_id);
            if (cache_ind == -1)
            {
                break;
            }
            LOG_VERBOSE("Cache Ind: %d\n", cache_ind);
            // token at cache_ind
            LOG_VERBOSE("Token: %d\n", cache[cache_ind].token);
            // pattern token ind
            LOG_VERBOSE("Pattern Token Ind: %d\n", i);
            // token in pattern
            LOG_VERBOSE("Pattern Token: %d\n", pattern[i]);

            if (_is_token_at_index(pattern[i], cache_ind))
            {

                result_indices.push_back(cache_ind);
                rtrim--;
            }
            else
            {
                break;
            }
        }

        if (verbose_level > 0)
        {
            // print all information
            // extend
            // match_seq_id
            // new_tokens
            // ltrim
            // rtrim
            // result_indices

            LOG_VERBOSE("Extend: %d\n", extend);
            LOG_VERBOSE("Match Seq ID: %d\n", match_seq_id);
            LOG_VERBOSE("New Tokens: %d\n", new_tokens);
            LOG_VERBOSE("Left Trim: %d\n", ltrim);
            LOG_VERBOSE("Right Trim: %d\n", rtrim);
            LOG_VERBOSE("Result Indices: ");
            for (const auto &ind : result_indices)
            {
                LOG_VERBOSE("%d ", ind);
            }
            LOG_VERBOSE("\n");
        }

        if (min_overlap_requested != -1 && result_indices.size() < min_overlap_requested)
        {
            // we don't have the minimum overlap required
            // return empty vector
            // empty the result_indices
            result_indices.clear();
            match_seq_id = -1;
            new_tokens = pattern.size();
            ltrim = 0;
            rtrim = 0;
            extend = false;
            LOG_VERBOSE("Minimum overlap not found\n");
            return 0;
        }

        // we have the minimum overlap required
        // new_tokens = pattern.size() - result_indices.size();
        // we ignore tokens on the left, ie will be rejected
        new_tokens = rtrim;
        return 0;
    }

    int cache_prepare(
        search_info &s_info)
    {
        // prepare the cache for a new sequence
        // this doesn't involve decoding
        // but instead prepares the cache for the new sequence
        // by creating space for the new sequence
        // and copying the sequence from the cache if needed
        // and copying the suggested tokens from the cache if present

        const std::vector<llama_token> &seq = s_info.seq;
        const int &new_tokens = s_info.new_tokens;
        const llama_seq_id &seq_id = s_info.match_seq_id;
        std::deque<int> &indices = s_info.indices;
        const int &ltrim = s_info.ltrim;
        const int &rtrim = s_info.rtrim;
        const bool &extend = s_info.extend;

        llama_seq_id &target_seq_id = s_info.target_seq_id;

        LOG_INFO("Preparing cache for new sequence\n");

        if (seq.size() > max_sequence_length || seq.size() == 0)
        {
            LOG_WARNING("Sequence too large or too small\n");
            return 2;
        }
        if (new_tokens > seq.size())
        {
            // SS: kind of an unnecessary check
            // but for now we will keep it
            LOG_WARNING("New tokens too large\n");
            return 2;
        }

        // int extend_existing_size = 0;
        // A. extend cases:
        // 1.   a a a a
        //        a a a
        // 2.   a a a a
        //        a a a b b
        // 3.   a a a a
        //        a a
        // B. not extend
        // 1.   a a a a ...
        //        a a b ...
        // (a used to represent elements of 1 seq, b another)
        // (the one represented below is the new one and above is in cache)

        /*if (indices.size() != 0) // following a sequence
        {
            // int last_ind = indices.back();
            seq_info &follow_info = get_seq_info(seq_id);

            // int last_follow_pos = cache[follow_info.end_ind].pos; // ind is circular so use pos
            // if (last_follow_pos >= cache[last_ind].pos)
            // {
            // we can extend the sequence from the cache
            //     extend = true;
            //     extend_size = last_follow_pos - cache[last_ind].pos;
            // }

            int last_match_ind = indices.back();
            int next_ind = get_next_ind_in_seq(last_match_ind, seq_id);
            if (next_ind == -1) // A.1, A.2
            {
                // we can extend the sequence from the cache
                extend = true;
                extend_existing_size = 0;
            }
            else if (rtrim == 0) // A.3 match sequence continues past pattern
            {
                extend = true;
                // size is how much of the sequence is left
                int last_follow_seq_ind = follow_info.end_ind; // end of matched sequence
                int last_follow_pos = cache[last_follow_seq_ind].pos;
                int last_pat_pos = cache[last_match_ind].pos; // end of match

                extend_existing_size = last_follow_pos - last_pat_pos;
            }

            if (
                extend &&
                (new_tokens + follow_info.length > max_sequence_length - suggestion_length))
            {
                // ensure that if we are adding new tokens to the sequence
                // we don't exceed the max sequence length
                // remove tokens from the left of the sequence
                // this shouldn't affect matching indices
                // if it did it meant pattern was too long

                // delete tokens from the left
                int overflow = new_tokens + follow_info.length - max_sequence_length;
                int &left_ind = follow_info.start_ind;
                for (int i = 0; i < overflow; i++)
                {
                    _delete_cache_cell(left_ind, seq_id);
                    left_ind = get_next_ind_in_seq(left_ind, seq_id);
                }
                follow_info.length -= overflow;
            }
        }*/

        if (extend && new_tokens > 0) // see if we have to shorten the existing sequence
        {
            seq_info &follow_info = get_seq_info(seq_id);
            int overflow = new_tokens + follow_info.length - max_sequence_length - suggestion_length;
            if (overflow > 0)
            {
                // ensure that if we are adding new tokens to the sequence
                // we don't exceed the max sequence length
                // remove tokens from the left of the sequence
                // this shouldn't affect matching indices
                // if it did it meant pattern was too long

                // delete tokens from the left
                int &left_ind = follow_info.start_ind;
                for (int i = 0; i < overflow; i++)
                {
                    _delete_cache_cell(left_ind, seq_id);
                    left_ind = get_next_ind_in_seq(left_ind, seq_id);
                }
                follow_info.length -= overflow;
            }
        }

        /*while (!extend && seq_queue.size() >= n_max_seq)
        {
            // if we have reached the max number of sequences, delete the oldest sequence
            // because we will be creating a new sequence
            delete_seq(get_oldest_seq_id());
        }*/

        // create space for the sequence in seq_queue if we are not extending
        while (!extend && seq_queue.size() >= n_max_seq)
        {
            // if we have reached the max number of sequences, delete the oldest sequence
            // because we will be creating a new sequence
            if (!delete_oldest_seq_but_not(seq_id))
            {
                LOG_WARNING("Failed to create space for new sequence\n");
                return 2;
            }
        }

        // create space for new tokens if needed
        while (n_empty < new_tokens + suggestion_length)
        {
            // if we don't have enough empty cells, delete the oldest sequence
            if (!delete_oldest_seq_but_not(seq_id))
            {
                LOG_WARNING("Failed to create space for new tokens\n");
                return 2;
            }
            // NOTE: since we need cache to be 2x the size of the sequence
            // we will always have enough empty cells even if don't delete the matching sequence
        }

        // create a new seq_id
        if (extend)
        {
            target_seq_id = seq_id;
        }
        else if (!get_empty_seq_id(target_seq_id))
        {
            // if we don't have an empty seq_id, return false
            LOG_WARNING("Failed to get an empty seq_id\n");
            return 2;
        }

        // extending
        if (extend)
        {
            seq_info info = get_seq_info(target_seq_id);
            // we also want to delete it from queue to add it to the front
            _delete_seq_id(target_seq_id);

            // add the info to seq_queue (top of queue as it is last used)
            // using top = front = newest, bottom = back = oldest
            seq_queue.push_front(info);

            // we also have to add the suggested tokens from the cache
            // if (rtrim <= 0) // if pattern is not different // won't be extending if pattern is different
            {
                int &n_suggested = s_info.suggested_tokens;
                std::string &suggested_string = s_info.suggested_string;
                int end_pat_ind = indices.back();
                int cache_ind = get_next_ind_in_seq(end_pat_ind, seq_id);
                while (cache_ind != -1)
                {
                    n_suggested++;
                    suggested_string += cache[cache_ind].piece;
                    indices.push_back(cache_ind);
                    cache_ind = get_next_ind_in_seq(cache_ind, seq_id);
                }
            }

            LOG_VERBOSE("Extended sequence\n");
            return 0;
        }

        // not following a sequence
        if (seq_id == -1) // && !extend
        {
            int start_ind = find_first_empty_cell();
            int end_ind = -1; // have to be careful about this
            seq_info info = {target_seq_id, start_ind, end_ind, 0};

            // add the info to seq_queue (top of queue as it is last used)
            // using top = front = newest, bottom = back = oldest
            seq_queue.push_front(info);

            LOG_VERBOSE("Created new sequence\n");
            return 0;
        }

        // else if (!extend && seq_id != -1)
        // remaining case is not extending and we have a sequence
        // ie partially following a sequence
        // the more complicated case
        int start_ind = indices.front();
        llama_pos start_pos = cache[start_ind].pos;

        int end_ind = indices.back();
        llama_pos end_pos = cache[end_ind].pos + 1; // llama_..._cp is exclusive

        // copy the sequence from the cache
        llama_kv_cache_seq_cp(ctx, seq_id, target_seq_id, start_pos, end_pos);
        copy_seq(indices, target_seq_id);

        end_ind = indices.back();

        int length = indices.size();

        seq_info info = {target_seq_id, start_ind, end_ind, length};

        // add the info to seq_queue (top of queue as it is last used)
        // using top = front = newest, bottom = back = oldest
        seq_queue.push_front(info);

        LOG_VERBOSE("Copied sequence\n");
        return 0;
    }

    inline bool _decode(
        llama_batch &batch)
    {
        if (llama_decode(ctx, batch))
        {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return false;
        }
        return true;
    }

    int cache_update(
        search_info &s_info,
        /*      const std::vector<llama_token> &seq,
                const int new_tokens,
                const llama_seq_id &seq_id,
                const std::deque<int> &indices,
                const llama_seq_id &target_seq_id */
        task_management &task_manager)
    {
        // update the cache for a new sequence
        // this involves decoding
        // deleting the last token of the sequence
        // and decoding with logits for the last token

        const std::vector<llama_token> &seq = s_info.seq;
        const int &new_tokens = s_info.new_tokens;
        const llama_seq_id &seq_id = s_info.match_seq_id;
        const std::deque<int> &indices = s_info.indices;
        const llama_seq_id &target_seq_id = s_info.target_seq_id;

        const int ltrim = s_info.ltrim;

        seq_info &target_info = get_seq_info(target_seq_id);
        int &n_decoded = s_info.decoded_tokens;

        LOG_INFO("Updating Cache\n");

        if (target_info.seq_id == -1)
        {
            LOG_WARNING("Invalid seq_id\n");
            return 2;
        }

        int cache_ind = target_info.start_ind - 1; // -1 because we will call get_next_ind_empty() later

        int seq_start_ind = indices.size() + ltrim;
        int start_pos = 0;
        if (seq_id != -1)
        {
            cache_ind = indices.back();
            start_pos = cache[cache_ind].pos + 1;
        }

        int total_tokens_processed = 0;
        while (seq_start_ind < seq.size())
        {
            int remaining_tokens = seq.size() - seq_start_ind;
            int decode_size = std::min(batch_size, remaining_tokens);

            LOG_VERBOSE("Decoding with size: %d\n", decode_size);

            llama_batch batch = llama_batch_init(decode_size, 0, 1);

            int seq_ind, pos;
            for (pos = start_pos, seq_ind = seq_start_ind;
                 seq_ind < seq_start_ind + decode_size && seq_ind < seq.size();
                 seq_ind++, pos++)
            {
                LOG_VERBOSE("Adding token %d at index %d\n", seq[seq_ind], seq_ind);
                llama_token id = seq[seq_ind];

                bool last = false; // seq_ind == seq.size() - 1;
                // SS: weird situation where this causes token 28705 to be inferred as last token

                llama_batch_add(batch, id, pos, {target_seq_id}, last);

                // also add the token to the cache
                cache_ind = get_next_ind_empty(cache_ind);
                if (cache_ind == -1)
                {
                    LOG_WARNING("No more space in cache\n");
                    return 2;
                }
                _insert_token_at_cell(cache[cache_ind],
                                      seq[seq_ind],
                                      target_seq_id,
                                      pos,
                                      last);

                LOG_VERBOSE("Inserted token at index %d\n", cache_ind);
            }

            if (!_decode(batch))
            {
                LOG_ERROR("Failed to decode\n"); // this is a problem because now we will have a cache miss-match
                return 3;
            }
            int tokens_in_batch = pos - start_pos;

            total_tokens_processed += tokens_in_batch;
            LOG_VERBOSE("Decoded %d tokens in this batch\n", tokens_in_batch);

            target_info.end_ind = cache_ind;       // Update end index
            target_info.length += tokens_in_batch; // Update length

            seq_start_ind += decode_size;
            start_pos += tokens_in_batch;

            n_decoded += tokens_in_batch;

            if (task_manager.accept_if_cancel())
            {
                LOG_INFO("Cache Update Cancelled\n");
                return 1;
            }
        }

        LOG_VERBOSE("Total Decoded: %d tokens\n", total_tokens_processed);

        // there might be cases where copied the cache from middle of a sequence
        // then we have to re compute the last token with logits
        int last_ind = target_info.end_ind;
        if (!cache[last_ind].has_logits)
        {

            // delete the last token so we can decode with logits

            llama_pos last_pos = cache[cache_ind].pos;

            LOG_VERBOSE("Deleting last token\n");
            // pos and token
            LOG_VERBOSE("Last Token: %d\n", cache[last_ind].token);
            LOG_VERBOSE("Last Pos: %d\n", last_pos);

            llama_kv_cache_seq_rm(ctx, target_seq_id, last_pos, -1);

            LOG_VERBOSE("Deleted last token\n");

            llama_batch batch = llama_batch_init(1, 0, 1);
            llama_batch_add(batch,
                            cache[cache_ind].token,
                            last_pos,
                            {target_seq_id},
                            true); // setup for logits

            // decode with logits
            _decode(batch);

            LOG_VERBOSE("Decoded with logits\n");

            return 0;
        }
    }

    int cache_suggest(search_info &s_info,
                      task_management &task_manager)
    {

        const llama_seq_id target_seq_id = s_info.target_seq_id;

        int &n_suggested = s_info.suggested_tokens;
        std::string &suggested_string = s_info.suggested_string;

        LOG_INFO("Suggesting tokens\n");

        seq_info &info = get_seq_info(target_seq_id);

        struct llama_sampling_context *ctx_sampling =
            llama_sampling_init(params.sparams);

        llama_token id = 0;

        int cache_ind = info.end_ind;
        int pos = cache[cache_ind].pos;

        llama_batch batch = llama_batch_init(1, 0, 1);
        while (n_suggested < suggestion_length)
        {
            id = llama_sampling_sample(ctx_sampling, ctx, NULL, 0);
            llama_sampling_accept(ctx_sampling, ctx, id, false);

            pos++;
            cache_ind = get_next_ind_empty(cache_ind);
            if (cache_ind == -1)
            {
                LOG_WARNING("No more space in cache\n");
                return 2;
            }

            llama_batch_clear(batch);
            llama_batch_add(batch,
                            id,
                            pos,
                            {target_seq_id},
                            true);

            if (!_decode(batch))
            {
                LOG_ERROR("Failed to decode\n");
                return 2; // it's okay we don't have to return an error
                // we can just try again
            }
            _insert_token_at_cell(cache[cache_ind],
                                  id,
                                  target_seq_id,
                                  pos,
                                  true);

            suggested_string += cache[cache_ind].piece;
            n_suggested++;

            info.end_ind = cache_ind;
            info.length++;

            LOG_VERBOSE("Suggested %d tokens\n", n_suggested);
            // show token id, piece, and pos
            LOG_VERBOSE("Token: %d\n", id);
            LOG_VERBOSE("Piece: %s\n", cache[cache_ind].piece.c_str());
            LOG_VERBOSE("Pos: %d\n", pos);

            if (task_manager.accept_if_cancel())
            {
                LOG_INFO("Cache Suggest Cancelled\n");
                return 1;
            }
        }

        return 0;
    }

    /*    bool _add_new_seq(
            const std::vector<llama_token> &seq,
            const seq_info &info,
            const int &start_pos,
            const int &start_ind,
            int follow_seq_id = -1)
        {
            // Add the info to seq_queue (top of queue as it is last used)
            // using top = front = newest, bottom = back = oldest
            seq_queue.push_front(info);

            int curr_ind = start_ind - 1; // -1 because of how the loop is written
            int next_index;

            int count = 0;
            while (count < seq.size())
            {
                if (follow_seq_id != -1) // following a sequence
                {
                    next_index = get_next_ind_in_seq(curr_ind, follow_seq_id);
                    // Check for divergence: if the token doesn't match, find the next empty cell

                    if (next_index == -1 || cache[next_index].token != seq[count])
                    {
                        follow_seq_id = -1;                        // Stop following the sequence
                        next_index = get_next_ind_empty(curr_ind); // Find next empty cell
                    }
                }
                else
                {
                    next_index = get_next_ind_empty(curr_ind);
                }

                if (next_index == -1)
                {

                    // No more space in cache
                    return false;
                    // SS: can I just do with this check? Do we need to really pre check the required cache size?
                }

                curr_ind = next_index;

                // Add the token to the cache at curr_ind
                _insert_token_at_cell(cache[curr_ind],
                                      seq[count],
                                      info.seq_id,
                                      start_pos + count);

                count++;
            }
            return true;
        }

        bool add_new_seq(
            const std::vector<llama_token> &seq,
            const int &start_pos,
            const int &start_ind,
            const int &new_tokens,
            const llama_seq_id follow_seq_id = -1)
        {
            // if it (at least) partly matches another sequence, we will add it to that sequence

            if (seq.size() > max_sequence_length || seq.size() == 0)
            {
                return false;
            }
            if (new_tokens > seq.size())
            {
                return false;
            }
            // create a new sequence
            // if seq_id already exists, return false
            // if seq_id doesn't exist, create a new seq_id and return true
            while (seq_queue.size() >= n_max_seq)
            {
                // if we have reached the max number of sequences, delete the oldest sequence
                delete_seq(get_oldest_seq_id());
            }

            // if provided start_ind is not empty, return false
            // or if provided start_ind doesn't match the seq_id (if -1) return false
            if (follow_seq_id != -1)
            {
                if (!cache[start_ind].has_seq_id(follow_seq_id))
                {
                    return false;
                }
                if (cache[start_ind].token != seq[0] || cache[start_ind].pos != start_pos)
                {
                    return false; // it doesn't match the sequence
                }
            }
            else
            {
                if (cache[start_ind].token != -1)
                {
                    return false;
                }
            }

            int required_space = new_tokens;
            // we will start at start_ind and
            // check how many empty cells we need
            // if the token is already in the cache, we don't need to add it

            int n_deleted = 0;
            while (n_empty < required_space)
            {
                // if we don't have enough space, delete the oldest sequence
                delete_seq(get_oldest_seq_id());
                n_deleted++;

                if (n_deleted > n_max_seq)
                {
                    // if we have deleted all sequences, return false
                    return false;
                }
            }

            // create info
            // create a new seq_id by iterating over all seq_ids and finding an empty one
            llama_seq_id seq_id;
            if (!get_empty_seq_id(seq_id))
            {
                // if we don't have an empty seq_id, return false
                return false;
            }

            int size = seq.size();

            // info = {seq_id, start_ind, size};
            seq_info info = {seq_id, start_ind, size};

            // we have enough space, add the sequence
            return _add_new_seq(seq, info, start_pos, start_ind, follow_seq_id);
        } */
};

#endif // TOKEN_CACHE_H