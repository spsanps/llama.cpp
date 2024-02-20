#include "search_info.h"

#include <iostream>

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
