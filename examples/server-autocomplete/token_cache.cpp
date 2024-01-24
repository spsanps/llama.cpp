#include "common.h"
#include "llama.h"

#include <deque>
#include <cstddef>
#include <thread>
#include <mutex>
#include <chrono>
#include <iostream>
#include <vector>
#include <condition_variable>
#include <atomic>
#include <list>
#include <algorithm>
// set
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

int verbose = 0;
#define LOG_VERBOSE(...)         \
    do                           \
    {                            \
        if (verbose)             \
        {                        \
            printf(__VA_ARGS__); \
            fflush(stdout);      \
        }                        \
    } while (0)

// hash helper functions

struct rolling_buffer
{
    // a rolling buffer to store n elements
    int n, curr_ind;
    std::vector<int> buffer;

    void init(int new_n)
    {
        n = new_n;
        curr_ind = 0;
        buffer.resize(n);
    }

    void reset()
    {
        curr_ind = 0;
        buffer.clear();
    }

    void push_back(int token)
    {
        buffer[curr_ind] = token;
        curr_ind = (curr_ind + 1) % n; // curr_ind will always be the index of the oldest token
    }

    int at(int ind)
    {
        return buffer[(curr_ind + ind) % n];
    }

    int begin()
    {
        return buffer[curr_ind];
    }

    int end()
    {
        return buffer[(curr_ind + n - 1) % n];
    }

    // overload [] operator
    int operator[](int ind)
    {
        return buffer[(curr_ind + ind) % n];
    }

    // overload for const [] operator
    int operator[](int ind) const
    {
        return buffer[(curr_ind + ind) % n];
    }

    void print() const
    {
        std::cout << "Buffer: ";
        for (int i = 0; i < n; i++)
        {
            std::cout << buffer[(curr_ind + i) % n] << " ";
        }
        std::cout << "\n";
    }

    void log_verbose() const
    {
        if (verbose)
        {
            print();
        }
    }
};

struct rolling_hash
{
    uint64_t hash;
    uint64_t a;
    uint64_t pow_a;
    uint64_t mod;

    void init(const uint64_t new_a, const uint64_t new_mod)
    {
        a = new_a;
        mod = new_mod;
        hash = 0;
        pow_a = 1;
    }

    void add(const uint64_t token)
    {
        hash = (a * hash + token) % mod;
        pow_a = (pow_a * a) % mod;
    }

    void add_remove(const uint64_t token_remove, const uint64_t token_add)
    {
        // add first
        // (because pow_a is +1 of leading term that needs to be removed)
        hash = (a * hash + token_add) % mod;
        // remove
        uint64_t token_hash = (token_remove * pow_a) % mod;
        hash = (hash + mod - token_hash) % mod;
    }
};

// -----

struct llama_token_cell
{
    // equivalent to llama_kv_cell
    llama_token token = -1;

    llama_pos pos = -1;
    llama_pos delta = 0; // what is this for?

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
    // int end_ind;
    int length;
};

struct search_params
{
    int minimum_overlap_required = 128; // minimum overlap required to match a pattern (in tokens), if pattern is smaller then any overlap is okay

    int pattern_start_size = 32; // how many tokens to start with when searching for a pattern, if pattern is smaller then the pattern size is used

    int trim_skip_amount = 4; // how many tokens to skip when trimming the pattern in each iteration

    int trim_skip_small_amount = 1; // how many tokens to skip when trimming the pattern in each iteration if the pattern is smaller than trim_skip_amount

    // rolling hash parameters
    uint64_t a = 32003;
    uint64_t mod = 1e9 + 9;
};

struct token_cache
{
    // equivalent to llama_kv_cache

    int n_max_seq;

    std::vector<llama_token_cell> cache;
    int max_cache_size;
    int n_empty;

    // we will have a list of seq_info
    // we will use this vector to find the latest seq_id
    // and the oldest seq_id when we need to clean the cache
    std::list<seq_info> seq_queue;

    std::vector<bool> searched;

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

    void init(int max_size)
    {
        // initialize the cache
        cache.resize(max_size);
        max_cache_size = max_size;
        n_empty = max_size;
        searched.resize(max_size);
        reset_searched();
    }

    void print_cache(int verb = verbose) const
    {
        if (!verb)
        {
            return;
        }
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
                      << ", Length = " << info.length << "\n";
        }
    }

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

    seq_info get_seq_info(const llama_seq_id &seq_id)
    {
        // get the seq_info of a seq_id
        for (auto &info : seq_queue)
        {
            if (info.seq_id == seq_id)
            {
                return info;
            }
        }
        return {-1, -1, -1};
    }

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

    void delete_seq_id(const llama_seq_id &seq_id)
    {
        seq_queue.remove_if([&seq_id](const seq_info &si)
                            { return si.seq_id == seq_id; });
    }

    void delete_seq(const llama_seq_id &seq_id)
    {
        // delete all tokens belonging to a sequence
        for (auto &cell : cache)
        {
            cell.seq_id.erase(seq_id);
            if (cell.seq_id.size() == 0)
            {
                if (cell.token != -1)
                {
                    n_empty++;
                    cell.token = -1;
                    cell.pos = -1;
                }
            }
        }
        LOG_TEE("Deleted a sequence in cache\n");
        delete_seq_id(seq_id);
    }

    bool upsize(int new_size)
    {
        // resize the cache
        // if new_size is smaller than current size, return false

        if (new_size < max_cache_size || new_size < max_cache_size)
        {
            return false;
        }

        // we will add empty cells with pos = -1
        cache.resize(new_size);
        max_cache_size = new_size;
    }

    int calculate_empty_count()
    {
        // get the number of empty cells in the cache
        int count = 0;
        for (auto &cell : cache)
        {
            if (cell.token == -1)
            {
                count++;
            }
        }
        return count;
    }
    int find_first_empty()
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

    void update_cache_with_token(
        llama_token_cell &cell,
        llama_token token,
        llama_seq_id seq_id,
        int position)
    {
        if (cell.token == -1)
        {
            n_empty--;
        }
        cell.token = token;
        cell.seq_id.insert(seq_id);
        cell.pos = position;
    }

    bool _add_new_seq(
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
            update_cache_with_token(cache[curr_ind],
                                    seq[count],
                                    info.seq_id,
                                    start_pos + count);

            count++;
        }
        return true;
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

    bool add_new_seq(
        const std::vector<llama_token> &seq,
        const int &start_pos,
        const int &start_ind,
        const int &new_tokens,
        const llama_seq_id follow_seq_id = -1)
    {
        // if it (at least) partly matches another sequence, we will add it to that sequence

        if (seq.size() > max_cache_size || seq.size() == 0)
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
    }

    bool is_token_at_index(const llama_token &token, const int &index)
    {
        return cache[index].token == token;
    }

    bool match_pattern_seq_with_trim(
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
        // print the seq_indices
        LOG_VERBOSE("Seq Indices: ");
        seq_indices.log_verbose();

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
                    if (!is_token_at_index(pattern[j + ltrim], seq_indices[j]))
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
                static_cast<uint64_t>(cache[seq_indices.begin()].token),
                static_cast<uint64_t>(cache[curr_ind].token));

            // update the seq_indices
            seq_indices.push_back(curr_ind);

            LOG_VERBOSE("Seq Hash: %llu\n", seq_rolling_hash.hash);
            // print the seq_indices
            LOG_VERBOSE("Seq Indices: ");
            seq_indices.log_verbose();

            // update curr_ind
            curr_ind = get_next_ind_in_seq(curr_ind, seq_id);

        } while (++i < max_roll);

        return false;
    }

    bool find_pattern(
        const std::vector<llama_token> &pattern,
        const search_params &params,
        std::deque<int> &result_indices) // indices of matches - if empty, no match
    {

        if (pattern.size() > max_cache_size || pattern.size() == 0)
        {
            return false;
        }

        int pattern_start_size = params.pattern_start_size;
        int ltrim;
        int rtrim = 0;
        if (pattern.size() < pattern_start_size)
        {
            pattern_start_size = pattern.size();
            ltrim = 0;
        }
        else
        {
            ltrim = pattern.size() - pattern_start_size;
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
        llama_seq_id match_seq_id;
        while (pattern_length > 0)
        {
            seq_indices.reset();
            // iterate over the sequence queue from newest to oldest
            for (auto &info : seq_queue)
            {
                // search the sequence
                if (match_pattern_seq_with_trim(pattern, ltrim, rtrim, info.seq_id, params, seq_indices))
                {
                    // we have found a match
                    // add the indices to the result_indices
                    for (int i = 0; i < pattern_length; i++)
                    {
                        result_indices.push_back(seq_indices[i]);
                    }
                    match = true;
                    match_seq_id = info.seq_id;
                    break;
                }
            }

            if (match)
            {
                // we have found a match
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
            return false;
        }

        // we have found a match
        // we will now see if we can extend the match
        // we will now see if there are any matching tokens in the left and right of the match

        // tokens to the left
        int cache_ind = result_indices.front();
        for (int i = ltrim - 1; i >= 0; i--)
        {
            cache_ind = get_prev_ind_in_seq(cache_ind, match_seq_id);
            if (cache_ind == -1)
            {
                break;
            }
            if (is_token_at_index(pattern[i], cache_ind))
            {
                result_indices.push_front(cache_ind);
            }
            else
            {
                break;
            }
        }

        // tokens to the right
        cache_ind = result_indices.back();
        for (int i = pattern.size() - rtrim; i < pattern.size(); i++)
        {

            cache_ind = get_next_ind_in_seq(cache_ind, match_seq_id);
            if (cache_ind == -1)
            {
                break;
            }
            if (is_token_at_index(pattern[i], cache_ind))
            {
                result_indices.push_back(cache_ind);
            }
            else
            {
                break;
            }
        }
        if (min_overlap_requested != -1 && result_indices.size() < min_overlap_requested)
        {
            // we don't have the minimum overlap required
            // return empty vector
            // empty the result_indices
            result_indices.clear();
            return false;
        }
    }
};

void printTestResult(bool result, const std::string &testName)
{
    std::cout << "Test \"" << testName << "\": "
              << (result ? "PASSED" : "FAILED") << std::endl;
}

std::vector<llama_token> generateSequence(int length)
{
    std::vector<llama_token> sequence;
    sequence.reserve(length);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<llama_token> dist(1, 1000); // Assuming tokens are integers between 1 and 1000

    for (int i = 0; i < length; ++i)
    {
        sequence.push_back(dist(rng));
    }

    return sequence;
}

bool testCreateSequence(token_cache &cache, const std::vector<llama_token> &seq)
{
    int start_pos = 0;
    int start_ind = cache.find_first_empty();
    int new_tokens = seq.size();

    return cache.add_new_seq(seq, start_pos, start_ind, new_tokens);
}

bool testSearchSequence(token_cache &cache, const std::vector<llama_token> &pattern)
{
    rolling_buffer seq_indices;
    search_params params;
    bool result = cache.match_pattern_seq_with_trim(pattern, 0, 0, cache.get_latest_seq_id(), params, seq_indices);
    std::cout << "Seq Indices: ";
    seq_indices.print();
    return result;
}

int main()
{
    token_cache cache;
    cache.init(4096); // Larger size to accommodate longer sequences and their overlaps
    cache.n_max_seq = 10;

    // Test 1: Create a Long Sequence
    std::cout << "Test 1: Create a Long Sequence\n";
    auto seq = generateSequence(2048);
    auto start = std::chrono::high_resolution_clock::now();
    bool result = testCreateSequence(cache, seq);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printTestResult(result, "Create Long Sequence");
    cache.print_cache();
    std::cout << "Time taken: " << elapsed.count() << " ms\n\n";

    // Test 1.1: Add a Partially Overlapping Sequence
    std::cout << "Test 1.1: Add a Partially Overlapping Sequence\n";
    std::vector<llama_token> seq2 = generateSequence(1024); // New sequence
    int overlap = 100;                                      // Overlap size
    int start_pos = 2048 - overlap;
    int start_ind = 2048 - overlap;
    int new_tokens = 2048 - overlap;
    llama_seq_id follow_seq_id = cache.get_latest_seq_id();
    std::cout << "Follow Seq ID: " << follow_seq_id << "\n";
    start = std::chrono::high_resolution_clock::now();
    result = cache.add_new_seq(seq2, start_pos, start_ind, new_tokens, follow_seq_id);
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printTestResult(result, "Add Partially Overlapping Sequence");
    cache.print_cache();
    std::cout << "Time taken: " << elapsed.count() << " ms\n\n";

    // Test 1.2: Add a Sequence that Overflows the Cache
    std::cout << "Test 1.2: Add a Sequence that Overflows the Cache\n";
    std::vector<llama_token> seq3 = generateSequence(2048);
    start_pos = 0;     // Start from the beginning for the new sequence
    start_ind = 0;     // Assuming the new sequence starts from a new index
    new_tokens = 2048; // Entire new sequence
    follow_seq_id = cache.get_latest_seq_id();
    std::cout << "Follow Seq ID: " << follow_seq_id << "\n";
    start = std::chrono::high_resolution_clock::now();
    result = cache.add_new_seq(seq3, start_pos, start_ind, new_tokens, follow_seq_id);
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printTestResult(result, "Add Sequence Causing Overflow");
    cache.print_cache();
    std::cout << "Time taken: " << elapsed.count() << " ms\n\n";

    // Test 2: Search for a Random Sequence
    std::cout << "Test 2: Search for a Random Sequence\n";
    std::vector<llama_token> randomSeq = generateSequence(500); // Smaller random sequence
    start = std::chrono::high_resolution_clock::now();
    result = testSearchSequence(cache, randomSeq);
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printTestResult(result, "Search Random Sequence");
    cache.print_cache();
    std::cout << "Time taken: " << elapsed.count() << " ms\n\n";

    // Test 3: Search for a Subsequence
    std::cout << "Test 3: Search for a Subsequence\n";
    std::vector<llama_token> subSeq = generateSequence(300); // Subsequence
    start = std::chrono::high_resolution_clock::now();
    result = testSearchSequence(cache, subSeq);
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printTestResult(result, "Search Subsequence");
    cache.print_cache();
    std::cout << "Time taken: " << elapsed.count() << " ms\n\n";

    // Test 4: Find Pattern
    std::cout << "Test 4: Find Pattern\n";
    std::vector<llama_token> pattern = generateSequence(400); // Pattern sequence
    std::deque<int> result_indices;
    search_params params;
    start = std::chrono::high_resolution_clock::now();
    result = cache.find_pattern(pattern, params, result_indices);
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printTestResult(result, "Find Pattern");
    std::cout << "Result Indices: ";
    for (const auto &ind : result_indices)
    {
        std::cout << ind << " ";
    }
    std::cout << "\n";
    std::cout << "Time taken: " << elapsed.count() << " ms\n\n";

    return 0;
}

/*

int main()
{
    token_cache cache;
    cache.init(10); // Initialize with a size of 10 for example
    cache.n_max_seq = 3;

    // Sequence to add
    std::vector<llama_token> seq = {1, 2, 3, 4, 5};

    // Test 1: Create a Sequence
    std::cout << "Test 1: Create a Sequence\n";
    auto start = std::chrono::high_resolution_clock::now();
    bool result = testCreateSequence(cache, seq);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printTestResult(result, "Create Sequence");
    cache.print_cache();
    std::cout << "Time taken: " << elapsed.count() << " ms\n\n\n";

    // Test 1.1: Add a Sequence to a Sequence (Partially Overlapping)
    std::cout << "Test 1.1: Add a Sequence to a Sequence (Partially Overlapping)\n";
    std::vector<llama_token> seq2 = {4, 5, 6, 7};
    int start_pos = 3;
    int start_ind = 3; // same as 4 in seq
    int new_tokens = 2;
    llama_seq_id follow_seq_id = cache.get_latest_seq_id();
    std::cout << "Follow Seq ID: " << follow_seq_id << "\n";
    start = std::chrono::high_resolution_clock::now();
    result = cache.add_new_seq(seq2, start_pos, start_ind, new_tokens, follow_seq_id);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    printTestResult(result, "Add Sequence to Sequence");
    cache.print_cache();
    std::cout << "Time taken: " << elapsed.count() << " ms\n\n\n";

    // Test 1.2: Add a Sequence to a Sequence (Partially Overlapping which overflows the cache)
    std::cout << "Test 1.2: Add a Sequence to a Sequence (Partially Overlapping which overflows the cache)\n";
    std::vector<llama_token> seq3 = {5, 6, 8, 9, 10, 11};
    start_pos = 4;
    start_ind = 4; // same as 5 in seq
    new_tokens = 4;
    follow_seq_id = 1;
    std::cout << "Follow Seq ID: " << follow_seq_id << "\n";
    start = std::chrono::high_resolution_clock::now();
    result = cache.add_new_seq(seq3, start_pos, start_ind, new_tokens, follow_seq_id);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    printTestResult(result, "Add Sequence to Sequence (Overflow)");
    cache.print_cache();
    std::cout << "Time taken: " << elapsed.count() << " ms\n\n\n";

    // Test 2: Search for a Random Sequence
    std::cout << "Test 2: Search for a Random Sequence\n";
    std::vector<llama_token> randomSeq = {6, 7, 8};
    start = std::chrono::high_resolution_clock::now();
    result = testSearchSequence(cache, randomSeq);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    printTestResult(result, "Search Random Sequence");
    cache.print_cache();
    std::cout << "Time taken: " << elapsed.count() << " ms\n\n\n";

    // Test 3: Search for a Subsequence
    std::cout << "Test 3: Search for a Subsequence\n";
    std::vector<llama_token> subSeq = {6, 8, 9};
    start = std::chrono::high_resolution_clock::now();
    result = testSearchSequence(cache, subSeq);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    printTestResult(result, "Search Subsequence");
    cache.print_cache();
    std::cout << "Time taken: " << elapsed.count() << " ms\n\n\n";



    // Test 4: Find Pattern
    std::cout << "Test 4: Find Pattern\n";
    std::vector<llama_token> pattern = {5, 6, 7, 8, 20, 21, 22, 34, 33, 34};
    std::deque<int> result_indices;
    search_params params;
    start = std::chrono::high_resolution_clock::now();
    result = cache.find_pattern(pattern, params, result_indices);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    printTestResult(result, "Find Pattern");
    std::cout << "Result Indices: ";
    for (const auto &ind : result_indices)
    {
        std::cout << ind << " ";
    }
    std::cout << "\n";
    std::cout << "Time taken: " << elapsed.count() << " ms\n\n\n";

    return 0;

}

*/