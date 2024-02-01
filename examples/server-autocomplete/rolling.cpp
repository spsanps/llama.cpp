#include "rolling.h"
#include <iostream>

void rolling_buffer::init(int new_n)
{
    n = new_n;
    curr_ind = 0;
    buffer.resize(n, -1);
}

void rolling_buffer::reset()
{
    curr_ind = 0;
    std::fill(buffer.begin(), buffer.end(), -1);
}

void rolling_buffer::push_back(int token)
{
    buffer[curr_ind] = token;
    curr_ind = (curr_ind + 1) % n;
}

int rolling_buffer::at(int ind)
{
    return buffer[(curr_ind + ind) % n];
}

int rolling_buffer::front()
{
    return buffer[curr_ind];
}

int rolling_buffer::back()
{
    return buffer[(curr_ind + n - 1) % n];
}

int rolling_buffer::operator[](int ind)
{
    return buffer[(curr_ind + ind) % n];
}

int rolling_buffer::operator[](int ind) const
{
    return buffer[(curr_ind + ind) % n];
}

void rolling_buffer::print() const
{
    std::cout << "Buffer: ";
    for (int i = 0; i < n; i++)
    {
        std::cout << buffer[(curr_ind + i) % n] << " ";
    }
    std::cout << "\n";
}

void rolling_hash::init(const uint64_t new_a, const uint64_t new_mod)
{
    a = new_a;
    mod = new_mod;
    hash = 0;
    pow_a = 1;
}

void rolling_hash::add(const uint64_t token)
{
    hash = (a * hash + token) % mod;
    pow_a = (pow_a * a) % mod;
}

void rolling_hash::add_remove(const uint64_t token_remove, const uint64_t token_add)
{
    // add first
    // (because pow_a is +1 of leading term that needs to be removed)
    hash = (a * hash + token_add) % mod;
    // remove
    uint64_t token_hash = (token_remove * pow_a) % mod;
    hash = (hash + mod - token_hash) % mod;
}