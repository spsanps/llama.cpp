// search_info.h
#ifndef SEARCH_INFO_H
#define SEARCH_INFO_H

#include "llama.h"
#include <deque>
#include <string>
#include <vector>

struct search_params
{
    int minimum_overlap_required = 32; // minimum overlap required to match a pattern (in tokens), if pattern is smaller then any overlap is okay

    int pattern_start_size = 32; // how many tokens to start with when searching for a pattern, if pattern is smaller then the pattern size is used

    int trim_skip_amount = 4; // how many tokens to skip when trimming the pattern in each iteration

    int trim_skip_small_amount = 1; // how many tokens to skip when trimming the pattern in each iteration if the pattern is smaller than trim_skip_amount

    // rolling hash parameters
    uint64_t a = 32003;
    uint64_t mod = 1e9 + 9;
};

struct search_info
{
    // let's have a string to store the original search string
    std::string search_string;
    std::string new_string;

    std::vector<llama_token> seq; // cleaned, truncated, tokenized search string

    llama_seq_id match_seq_id;  // matching sequence id
    llama_seq_id target_seq_id; // new sequence id

    std::deque<int> indices; // indices of the matching tokens in the cache

    int new_tokens; // new tokens - determined after searching in cache

    int ltrim, rtrim; // how much of the pattern should be trimmed (from left and right) for matching

    bool extend; // if we can extend a sequence from the cache

    int decoded_tokens;   // how many tokens have been decoded
    int suggested_tokens; // how many tokens have been suggested
    std::string suggested_string;

    void clear(bool partial = false)
    {
        if (!partial)
        {
            search_string = "";
            seq.clear();
            new_string = "";
        }
        new_tokens = 0;
        match_seq_id = -1;
        target_seq_id = -1;
        indices.clear();
        ltrim = 0;
        rtrim = 0;
        extend = false;
        decoded_tokens = 0;
        suggested_tokens = 0;
        suggested_string = "";
    }
    void print_string() const;
    void print_tokens() const;
    void print_indices() const;
};

#endif // SEARCH_INFO_H
