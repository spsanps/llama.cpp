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
    int string_length;

    void print() const
    {
        std::cout << "Seq ID: " << seq_id << ", Start Ind: " << start_ind
                  << ", End Ind: " << end_ind << ", Length: " << length << ", String Length: " << string_length << "\n";
    }
};

struct token_cache
{
    // equivalent to llama_kv_cache

    int n_max_seq = 16;

    std::vector<llama_token_cell> cache;
    int max_cache_size = 1024;     // eg. 4096 (2 * max_sequence_length)
    int max_sequence_length = 512; // eg. 2048 ( should be < max_cache_size / 2)
    int max_string_length = 1024;  // eg. 2048 (will ensure that sequence is not too long)
    int suggestion_length = 32;    // eg. 32
    int batch_size = 32;           // eg. 32

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

    void init()
    {
        // initialize the cache
        cache.resize(max_cache_size);
        int max_size = max_cache_size;
        n_empty = max_cache_size;

        // TODO: do better initialization
        const char *model_path = "Meta-Llama-3-8B.Q8_0.gguf";
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
                      << ", Pos = " << cell.pos << ", Piece = " << cell.piece << ", Seq IDs = {";
            for (const auto &id : cell.seq_id)
            {
                std::cout << id << " ";
            }
            std::cout << "}\n";
        }

        print_seq_queue();
    }

    void print_seq_queue() const
    {
        std::cout << "\nSequence Queue:\n";
        for (const auto &info : seq_queue)
        {
            info.print();
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

    void update_to_prev_piece_ind(int &token_ind, int &piece_ind, int seq_id)
    {
        // update piece_ind to the previous piece ind
        piece_ind--;
        if (piece_ind < 0)
        {
            token_ind = get_prev_ind_in_seq(token_ind, seq_id);
            if (token_ind == -1)
            {
                piece_ind = -1;
                return;
            }
            piece_ind = cache[token_ind].piece.size() - 1;
        }
        return;
    }

    void update_to_next_piece_ind(int &token_ind, int &piece_ind, int seq_id)
    {
        // update piece_ind to the next piece ind
        piece_ind++;
        if (piece_ind == cache[token_ind].piece.size())
        {
            token_ind = get_next_ind_in_seq(token_ind, seq_id);
            if (token_ind == -1)
            {
                piece_ind = -1;
                return;
            }
            piece_ind = 0;
        }
        return;
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
        const llama_token token,
        const llama_seq_id seq_id,
        const int position,
        const bool has_logits = false)
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

    void _delete_cache_cell(const int index, seq_info &info)
    {
        // check if seq_id is present in the cell
        /*if (!cache[index].has_seq_id(seq_id))
        {
            return;
        }*/
        // SS: assume this check is done where needed

        llama_kv_cache_seq_rm(ctx, info.seq_id, index, index + 1);

        info.length--;
        info.string_length -= cache[index].piece.size();

        cache[index].seq_id.erase(info.seq_id);

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
        LOG_VERBOSE("Deleted a sequence in cache\n");
        _delete_seq_id(seq_id);
        llama_kv_cache_seq_rm(ctx, seq_id, -1, -1); // delete entire sequence
    }

    bool delete_oldest_seq_but_not(const llama_seq_id seq_id)
    {
        // delete the oldest sequence in the cache
        for (auto it = seq_queue.rbegin(); it != seq_queue.rend(); ++it)
        {
            if (it->seq_id != seq_id)
            {
                delete_seq(it->seq_id);
                LOG_INFO("Deleted oldest sequence in cache\n");
                return true;
            }
        }
        return false;
    }

    int copy_seq(
        const std::deque<int> &indices,
        const llama_seq_id &target_seq_id)
    {
        // insert target_seq_id at indices
        int string_length = 0;
        for (auto &ind : indices)
        {
            cache[ind].seq_id.insert(target_seq_id);
            string_length += cache[ind].piece.size();
        }
        return string_length;
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

        // new string may change and we want to be safe
        s_info.search_string = s_info.new_string; // copy the new string

        std::string &pattern_string = s_info.search_string;
        // std::vector<llama_token> &seq = s_info.tokenized_string;
        // std::string &new_string = s_info.new_string;

        LOG_VERBOSE("Search String: %s\n", pattern_string.c_str());
        // LOG_VERBOSE("New String: %s\n", new_string.c_str());
        s_info.clear(true); // partial clear

        LOG_INFO("Initializing cache for new sequence\n");

        // pattern_string = new_string;

        // print the tokens

        int required_length = max_string_length;
        LOG_VERBOSE("Required Length: %d\n", required_length);
        if (required_length <= pattern_string.size())
        {
            // if the sequence is too long, trim it (from the beginning)
            // the resulting sequence will be of length required_length
            pattern_string = pattern_string.substr(pattern_string.size() - required_length, pattern_string.size());
        }

        // log new size
        LOG_VERBOSE("New Size: %d\n", pattern_string.size());

        return 0;
    }

    bool _match_string_with_trim(
        const std::string &pattern_string,
        // const std::vector<llama_token> &pattern,
        const int32_t &ltrim, // how much of the pattern should be trimmed (from left)
        const int32_t &rtrim, // how much of the pattern should be trimmed (from right)
        const llama_seq_id &seq_id,
        const search_params &params,
        int &seq_token_ind,
        int &seq_piece_ind) //,
    // std::deque<int> &result_indices)
    {
        // we will use rabin-karp algorithm to match the pattern
        // ie a rolling hash to match the pattern

        if (pattern_string.size() == 0)
        {
            return false;
        }
        if (rtrim >= pattern_string.size() || ltrim >= pattern_string.size())
        {
            return false;
        }

        if (seq_id == -1)
        {
            return false;
        }

        LOG_VERBOSE("Searching for pattern: ");

        seq_info info = get_seq_info(seq_id);
        int seq_token_length = info.length;
        int seq_string_length = info.string_length;

        if (verbose_level > 0)
            info.print();

        int pattern_string_length = pattern_string.size() - rtrim - ltrim;
        LOG_VERBOSE("Pattern Length: %d\n", pattern_string_length);

        // if pattern is larger than the string of the sequence
        if (pattern_string_length > seq_string_length)
        {
            return false;
        }

        // compute hash of the trimmed pattern
        rolling_hash pattern_hash;
        pattern_hash.init(params.a, params.mod, pattern_string_length);
        for (int i = 0; i < pattern_string_length; i++)
        {
            const uint64_t piece_int =
                static_cast<uint64_t>(pattern_string[i + ltrim]);
            pattern_hash.add(piece_int);

            if (verbose_level > 0)
            {
                std::cout << "(Pat) Piece: " << pattern_string[i + ltrim] << "\n";
            }
        }

        if (verbose_level > 0)
        {
            std::cout << "Pattern Hash: ";
            pattern_hash.print();
        }
        // go to start_ind in cache
        seq_token_ind = info.start_ind;
        seq_piece_ind = -1; // will be updated to 0 in the first iteration

        // compute hash of the first pattern_length tokens in cache
        rolling_hash seq_hash;
        seq_hash.init(params.a, params.mod, pattern_string_length);

        // compute hash of the first pattern_length tokens in cache
        // assume 1 token is at least 1 piece
        // seq is guaranteed to be at least pattern_length
        for (int i = 0; i < pattern_string_length; i++)
        {
            update_to_next_piece_ind(seq_token_ind, seq_piece_ind, seq_id);

            const uint64_t piece_integer =
                static_cast<uint64_t>(
                    cache[seq_token_ind].piece[seq_piece_ind]);

            if (verbose_level > 0)
            {
                std::cout << "Piece: " << cache[seq_token_ind].piece[seq_piece_ind] << "\n";
            }

            seq_hash.add(piece_integer);
        }

        if (verbose_level > 0)
        {
            std::cout << "Seq Hash: ";
            seq_hash.print();
        }

        if (seq_hash == pattern_hash)
        {
            return true;
        }
        // else we will search the rest of the sequence
        update_to_next_piece_ind(seq_token_ind, seq_piece_ind, seq_id);

        for (int i = 0; i < seq_string_length && seq_token_ind != -1; i++)
        {
            const uint64_t piece_int =
                static_cast<uint64_t>(
                    cache[seq_token_ind].piece[seq_piece_ind]);

            seq_hash.add_remove(piece_int);

            if (verbose_level > 0)
            {
                // piece
                std::cout << "Piece: " << cache[seq_token_ind].piece[seq_piece_ind] << "\n";
                std::cout << "Seq Hash: ";
                seq_hash.print();
            }

            if (seq_hash == pattern_hash)
            {
                return true; // seq_*_ind will be pointing correctly to last token, piece
            }

            update_to_next_piece_ind(seq_token_ind, seq_piece_ind, seq_id);
        }

        return false;
    }

    int cache_compare(
        search_info &s_info,
        const search_params &params)
    {

        std::vector<llama_token> &pattern_tokens = s_info.tokenized_string;
        const std::string &pattern_string = s_info.search_string;
        int &new_token_count = s_info.new_tokens;
        llama_seq_id &match_seq_id = s_info.match_seq_id;
        std::deque<int> &match_inds = s_info.match_indices;
        bool &extend = s_info.extend;
        std::string &suggested_string = s_info.suggested_string;

        LOG_INFO("Comparing cache with pattern\n");

        if (pattern_string.size() > max_string_length)
        {
            LOG_WARNING("Pattern too large or too small\n");
            return 2;
        }

        int pattern_start_size = params.pattern_start_size;
        int ltrim_string = 0;
        int rtrim_string = 0;
        if (pattern_string.size() <= pattern_start_size)
        {
            pattern_start_size = pattern_string.size(); // if pattern is too small
        }
        else
        {
            ltrim_string = pattern_string.size() - pattern_start_size;
        }

        int min_overlap_requested = params.minimum_overlap_required;
        // using pattern_start_size as kind of a buffer for the minimum overlap
        // as pattern_start_size is kind of set to tell how many new tokens we can add at a time
        // this is to avoid the case when we are at "minimum_overlap_required" length and we have new tokens to add
        // we should still match even if we don't have the minimum overlap
        if (pattern_string.size() < min_overlap_requested + pattern_start_size)
        {
            min_overlap_requested = -1; // any overlap is okay
        }

        int trim_skip_amount = params.trim_skip_amount;
        if (pattern_string.size() < trim_skip_amount)
        {
            trim_skip_amount = params.trim_skip_small_amount;
        }

        int pattern_length = pattern_string.size() - rtrim_string - ltrim_string;
        bool match = false;
        int seq_cache_ind = -1;
        int seq_piece_ind = -1;

        int end_ind_of_match = -1;

        while (pattern_length > 0)
        {
            // match_inds.clear();
            // iterate over the sequence queue from newest to oldest
            for (auto &info : seq_queue)
            {
                // search the sequence
                if (_match_string_with_trim(pattern_string,
                                            ltrim_string,
                                            rtrim_string,
                                            info.seq_id,
                                            params,
                                            seq_cache_ind,
                                            seq_piece_ind))
                {
                    // we have found a match
                    match = true;
                    match_seq_id = info.seq_id;
                    end_ind_of_match = info.end_ind;
                    break;
                }
            }

            if (match)
            {
                // we have found a match
                LOG_INFO("Match found\n");
                break;
            }

            // no match update the pattern
            // trim the pattern from right
            rtrim_string += trim_skip_amount;
            pattern_length =
                pattern_string.size() - rtrim_string - ltrim_string;
        }

        if (!match)
        {
            // we have not found a match
            pattern_tokens = llama_tokenize(ctx,
                                            pattern_string.c_str(),
                                            false,
                                            true);

            // return empty vector
            // no match is not an error
            // empty the result_indices
            match_inds.clear();
            match_seq_id = -1;
            new_token_count = pattern_tokens.size();
            ltrim_string = 0;
            rtrim_string = 0;
            extend = false;
            LOG_VERBOSE("No match found\n");
            return 0;
        }

        // else found a match!
        // check if there are any matching tokens in the left and right of the match
        LOG_VERBOSE("Left Trim: %d\n", ltrim_string);
        LOG_VERBOSE("Right Trim: %d\n", rtrim_string);

        // pieces to the left
        int start_seq_cache_ind = seq_cache_ind;
        int start_seq_piece_ind = seq_piece_ind;
        // let's update match indices simultaneously
        match_inds.push_front(start_seq_cache_ind);
        // we will count backwards until start of match
        for (int i = 0; i < pattern_length; i++)
        {
            if (start_seq_cache_ind != match_inds.front()) // tokens will repeat as many pieces are in a token
            {
                match_inds.push_front(start_seq_cache_ind); // it won't have the last token if it is not complete
            }
            update_to_prev_piece_ind(start_seq_cache_ind,
                                     start_seq_piece_ind,
                                     match_seq_id);
        }
        // now start_seq_cache_ind and start_seq_piece_ind are pointing to
        // one piece before the start of the match

        // point to the last character of the left trim
        int left_string_ind = ltrim_string - 1;
        while (start_seq_cache_ind != -1 && left_string_ind >= 0)
        {

            const char piece_char = cache[start_seq_cache_ind].piece[start_seq_piece_ind];

            if (pattern_string[left_string_ind] != piece_char)
            {
                break;
            }
            // == case below

            if (start_seq_cache_ind != match_inds.front())
            {
                match_inds.push_front(start_seq_cache_ind);
            }

            left_string_ind--;
            update_to_prev_piece_ind(start_seq_cache_ind,
                                     start_seq_piece_ind,
                                     match_seq_id);
        }

        ltrim_string = left_string_ind + 1;

        // tokens to the right
        update_to_next_piece_ind(seq_cache_ind, seq_piece_ind, match_seq_id);
        // point to the first character of the right trim
        int right_string_ind = pattern_string.size() - rtrim_string;

        extend = true; // as of now extend is possible until any mismatch
        // extend = true signifies we can reuse the entire previous matched sequence in cache
        while (seq_cache_ind != -1 && right_string_ind < pattern_string.size())
        {
            const char piece_char = cache[seq_cache_ind].piece[seq_piece_ind];

            if (pattern_string[right_string_ind] != piece_char)
            {
                extend = false; // if there is a mismatch, we can't extend
                break;
            }
            // == case below

            if (seq_piece_ind == 0)
            {
                match_inds.push_back(seq_cache_ind); // it won't have the last token if it is not complete
            }

            right_string_ind++;
            update_to_next_piece_ind(seq_cache_ind,
                                     seq_piece_ind,
                                     match_seq_id);
        }
        rtrim_string = pattern_string.size() - right_string_ind;

        LOG_VERBOSE("Extend Done\n");

        if (verbose_level > 0)
        {
            // print all information
            LOG_VERBOSE("Extend: %d\n", extend);
            LOG_VERBOSE("Match Seq ID: %d\n", match_seq_id);
            LOG_VERBOSE("New Tokens: %d\n", new_token_count);
            LOG_VERBOSE("Left Trim: %d\n", ltrim_string);
            LOG_VERBOSE("Right Trim: %d\n", rtrim_string);
            LOG_VERBOSE("Result Indices: ");
            for (const auto &ind : match_inds)
            {
                LOG_VERBOSE("%d ", ind);
            }
            LOG_VERBOSE("\n");
        }

        const int string_match_size = pattern_string.size() - ltrim_string - rtrim_string;
        if (min_overlap_requested != -1 && string_match_size < min_overlap_requested)
        {
            // we don't have the minimum overlap required
            pattern_tokens = llama_tokenize(ctx, pattern_string.c_str(), false, true);
            // return empty vector
            // empty the result_indices
            match_inds.clear();
            match_seq_id = -1;
            new_token_count = pattern_tokens.size();
            extend = false;
            LOG_VERBOSE("Minimum overlap not found\n");
            return 0;
        }

        // in case where we have only partially used the last token
        // either remove it or add it
        // so that downstream can deal with a complete token
        if (extend)
        {
            // add the rest of the token to the suggestion
            while (seq_cache_ind != -1 && seq_piece_ind != 0)
            {
                suggested_string += cache[seq_cache_ind].piece[seq_piece_ind];
                update_to_next_piece_ind(seq_cache_ind, seq_piece_ind, match_seq_id);

                LOG_VERBOSE("Adding last token\n");
            }
        }
        else
        {
            // remove it from the match
            if (seq_piece_ind > 0) // not 0 or -1
            {
                right_string_ind -= seq_piece_ind;
                match_inds.pop_back();
                LOG_VERBOSE("Removing last token\n");
            }
        }
        // LOG_VERBOSE("Suggested String: %s\n", suggested_string.c_str());

        // now string to tokenize is from right_string_ind to pattern_string.size()

        LOG_VERBOSE("right_string_ind: %d\n", right_string_ind);

        if (right_string_ind < 0 || right_string_ind > pattern_string.size())
        {
            LOG_WARNING("Right String Ind out of bounds\n");
            return 2;
        }

        if (right_string_ind == pattern_string.size())
        {
            // no new tokens to add
            new_token_count = 0;
            return 0;
        }

        const std::string string_to_tokenize = pattern_string.substr(right_string_ind, pattern_string.size());

        LOG_VERBOSE("String prepared for tokenization\n");

        pattern_tokens = llama_tokenize(ctx, string_to_tokenize.c_str(), false, false);

        new_token_count = pattern_tokens.size();

        LOG_VERBOSE("New Tokens: %d\n", new_token_count);
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

        const std::vector<llama_token> &new_tokens = s_info.tokenized_string;
        // const int &new_tokens = s_info.new_tokens;
        const llama_seq_id &seq_id = s_info.match_seq_id;
        std::deque<int> &match_indices = s_info.match_indices;
        const int &ltrim = s_info.ltrim;
        const int &rtrim = s_info.rtrim;
        const bool &extend = s_info.extend;

        llama_seq_id &target_seq_id = s_info.target_seq_id;

        LOG_INFO("Preparing cache for new sequence\n");

        // check if we have to shorten the sequence
        // to fit it in the cache
        if (extend && new_tokens.size() > 0)
        {
            seq_info &follow_info = get_seq_info(seq_id);
            int overflow = new_tokens.size() + suggestion_length +
                           follow_info.length - max_sequence_length; // Account for suggestion length
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
                    _delete_cache_cell(left_ind, follow_info);
                    left_ind = get_next_ind_in_seq(left_ind, seq_id);
                }
            }
        }

        // create space for the sequence in seq_queue if we are not extending
        while (!extend && seq_queue.size() >= n_max_seq)
        {
            // if we have reached the max number of sequences
            // delete the oldest sequence
            // because we will be creating a new sequence
            if (!delete_oldest_seq_but_not(seq_id))
            {
                LOG_WARNING("Failed to create space for new sequence\n");
                return 2;
            }
        }

        // create space for new tokens if needed
        while (n_empty < new_tokens.size() + suggestion_length)
        {
            // if we don't have enough empty cells, delete the oldest sequence
            if (!delete_oldest_seq_but_not(seq_id))
            {
                LOG_WARNING("Failed to create space for new tokens\n");
                // print empty cells, number of sequences active, new tokens.size() and suggestion_length
                LOG_WARNING("Empty Cells: %d, Active Sequences: %d, New Tokens: %d, Suggestion Length: %d\n",
                            n_empty, seq_queue.size(), new_tokens.size(), suggestion_length);
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
            {
                int &n_suggested = s_info.suggested_tokens;
                std::string &suggested_string = s_info.suggested_string;
                int end_pat_ind = match_indices.back();
                int cache_ind = get_next_ind_in_seq(end_pat_ind, seq_id);
                while (cache_ind != -1)
                {
                    n_suggested++;
                    suggested_string += cache[cache_ind].piece;
                    match_indices.push_back(cache_ind);
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
            seq_info info = {target_seq_id, start_ind, end_ind, 0, 0};

            // add the info to seq_queue (top of queue as it is last used)
            // using top = front = newest, bottom = back = oldest
            seq_queue.push_front(info);

            LOG_VERBOSE("Created new sequence\n");
            return 0;
        }

        // else if (!extend && seq_id != -1)
        // remaining case is not extending and we have a sequence
        // ie partially following a sequence

        int start_ind = match_indices.front();
        llama_pos start_pos = cache[start_ind].pos;

        int end_ind = match_indices.back();
        llama_pos end_pos = cache[end_ind].pos + 1; // llama_..._cp is exclusive

        // copy the sequence from the cache
        llama_kv_cache_seq_cp(ctx, seq_id, target_seq_id, start_pos, end_pos);
        int string_length = copy_seq(match_indices, target_seq_id);

        end_ind = match_indices.back();

        int length = match_indices.size();

        seq_info info = {target_seq_id,
                         start_ind,
                         end_ind,
                         length,
                         string_length};

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

        const std::vector<llama_token> &new_tokens = s_info.tokenized_string;
        const llama_seq_id &match_seq_id = s_info.match_seq_id;
        const std::deque<int> &match_indices = s_info.match_indices;
        const llama_seq_id &target_seq_id = s_info.target_seq_id;

        // const int ltrim = s_info.ltrim;

        seq_info &target_info = get_seq_info(target_seq_id);
        int &n_decoded = s_info.decoded_tokens;

        LOG_INFO("Updating Cache\n");

        if (target_info.seq_id == -1)
        {
            LOG_WARNING("Invalid seq_id\n");
            return 2;
        }

        int cache_ind = target_info.start_ind - 1; // -1 because we will call get_next_ind_empty() later

        int token_ind = 0;
        int start_pos = 0;
        if (match_seq_id != -1)
        {
            cache_ind = match_indices.back();
            start_pos = cache[cache_ind].pos + 1;
        }

        int total_tokens_processed = 0;
        while (token_ind < new_tokens.size())
        {
            int remaining_tokens = new_tokens.size() - token_ind;
            int decode_size = std::min(batch_size, remaining_tokens);

            LOG_VERBOSE("Decoding with size: %d\n", decode_size);

            llama_batch batch = llama_batch_init(decode_size, 0, 1);

            int seq_ind, pos;
            for (pos = start_pos, seq_ind = token_ind;
                 seq_ind < token_ind + decode_size && seq_ind < new_tokens.size();
                 seq_ind++, pos++)
            {
                LOG_VERBOSE("Adding token %d at index %d\n", new_tokens[seq_ind], seq_ind);
                llama_token id = new_tokens[seq_ind];

                llama_batch_add(batch, id, pos, {target_seq_id}, false);

                // also add the token to the cache
                cache_ind = get_next_ind_empty(cache_ind);
                if (cache_ind == -1)
                {
                    // get rid of this check as we have multiple checks for empty cells
                    LOG_WARNING("No more space in cache\n");
                    return 2;
                }
                _insert_token_at_cell(cache[cache_ind],
                                      new_tokens[seq_ind],
                                      target_seq_id,
                                      pos,
                                      false);

                target_info.string_length += cache[cache_ind].piece.size();

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

            token_ind += decode_size;
            start_pos += tokens_in_batch;

            n_decoded += tokens_in_batch;

            if (task_manager.accept_if_cancel())
            {
                LOG_INFO("Cache Update Cancelled\n");
                return 1;
            }
        }

        LOG_VERBOSE("Total Decoded: %d tokens\n", total_tokens_processed);

        // there might be cases where we have to re compute the last token with logits
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
        }
        return 0;
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
            info.string_length += cache[cache_ind].piece.size();

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
};

#endif // TOKEN_CACHE_H