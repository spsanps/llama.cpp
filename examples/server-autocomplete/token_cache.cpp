#include <iostream>
#include <chrono>
#include "token_cache.h"
#include "search_info.h"

void printTestResult(bool result, const std::string &testName)
{
    std::cout << "Test \"" << testName << "\": "
              << (result ? "PASSED" : "FAILED") << std::endl;
}

int test_all(token_cache &cache, const std::string &test_string)
{
    search_info s_info;
    s_info.clear();
    search_params params;

    // Test 1:
    // initialize the cache

    // search string is test_string repeated n times
    s_info.search_string = "";
    for (int i = 0; i < 1; i++)
    {
        s_info.search_string += test_string;
    }

    std::cout << "Test 1: Initialize the Cache\n";
    auto start = std::chrono::high_resolution_clock::now();
    bool result = cache.cache_init(s_info);
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printTestResult(result, "Initialize Cache");

    std::cout << "Time taken: " << elapsed.count() << " ms\n";

    // s_info.print_string();
    // s_info.print_tokens();

    std::cout << "\n\n";

    // Test 2:
    // compare the search string with the cache

    std::cout << "Test 2: Compare the Search String with the Cache\n";
    start = std::chrono::high_resolution_clock::now();
    result = cache.cache_compare(s_info, params);
    end = std::chrono::high_resolution_clock::now();

    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printTestResult(result, "Compare Search String with Cache");

    std::cout << "Time taken: " << elapsed.count() << " ms\n";

    s_info.print_indices();
    // print match_seq_id
    // print new_tokens
    std::cout << "Match Seq ID: " << s_info.match_seq_id << "\n";
    std::cout << "New Tokens: " << s_info.new_tokens << "\n";

    std::cout << "\n\n";

    // Test 3:
    // prepare the cache for the search string

    std::cout << "Test 3: Prepare the Cache for the Search String\n";
    start = std::chrono::high_resolution_clock::now();
    result = cache.cache_prepare(s_info);
    end = std::chrono::high_resolution_clock::now();

    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printTestResult(result, "Prepare Cache for Search String");

    std::cout << "Time taken: " << elapsed.count() << " ms\n";

    // cache.print_cache();
    // cache.print_seq_queue();

    // print target_seq_id
    std::cout << "Target Seq ID: " << s_info.target_seq_id << "\n\n";

    // print seq_queue

    // print cache
    // cache.print_cache();

    // Test 4:
    // update the cache for the search string

    std::cout << "Test 4: Update the Cache for the Search String\n";
    start = std::chrono::high_resolution_clock::now();
    task_management task;
    task.set_state(CACHE_UPDATE);
    result = cache.cache_update(s_info, task);
    end = std::chrono::high_resolution_clock::now();

    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printTestResult(result, "Update Cache for Search String");

    std::cout << "Time taken: " << elapsed.count() << " ms\n";

    // print token_cache
    // cache.print_cache();

    // Test 5:
    // suggest tokens for the search string

    std::cout << "Test 5: Suggest Tokens for the Search String\n";
    start = std::chrono::high_resolution_clock::now();
    task.set_state(CACHE_SUGGEST);
    result = cache.cache_suggest(s_info, task);
    end = std::chrono::high_resolution_clock::now();

    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printTestResult(result, "Suggest Tokens for Search String");

    std::cout << "Time taken: " << elapsed.count() << " ms\n";

    // print suggested_string
    std::cout << "Suggested String: " << s_info.suggested_string << "\n";

    // print cache
    cache.print_cache();

    return 0;
}

int main()
{
    token_cache cache;
    cache.init(64); // Larger size to accommodate longer sequences and their overlaps
    cache.n_max_seq = 10;

    std::string test_string = "What is your";

    test_all(cache, test_string);

    // another string

    test_string = "What is your name?";

    test_all(cache, test_string);

    // completely different string

    test_string = "The sun rose in the east";

    test_all(cache, test_string);
}

/*


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
    int start_ind = cache.find_first_empty_cell();
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
*/

/*
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

*/
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