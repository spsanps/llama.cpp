#include "search_info.h"

#include <iostream>

void search_info::clear(bool partial = false)
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

void search_info::print_string() const
{
    std::cout << "Search String: " << search_string << "\n";
}

void search_info::print_tokens() const
{
    std::cout << "Tokens: ";
    for (const auto &token : seq)
    {
        std::cout << token << " ";
    }
    std::cout << "\n";
}

void search_info::print_indices() const
{
    std::cout << "Indices: ";
    for (const auto &ind : indices)
    {
        std::cout << ind << " ";
    }
    std::cout << "\n";
}
