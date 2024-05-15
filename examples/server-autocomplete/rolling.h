#ifndef ROLLING_H
#define ROLLING_H

#include <vector>
#include <cstdint>
#include <iostream>

struct rolling_buffer
{
    int length, curr_ind;
    std::vector<int> buffer;

    void init(int new_length)
    {
        length = new_length;
        curr_ind = 0;
        buffer.resize(length, -1);
    }

    void reset()
    {
        curr_ind = 0;
        std::fill(buffer.begin(), buffer.end(), -1);
    }

    void push_back(int token)
    {
        buffer[curr_ind] = token;
        curr_ind = (curr_ind + 1) % length; // next empty token
        // curr_ind will be the start of the buffer when it is full
    }

    int at(int ind)
    {
        return buffer[(curr_ind + ind) % length];
    }

    int front()
    {
        return buffer[curr_ind];
    }

    int back()
    {
        return buffer[(curr_ind + length - 1) % length];
    }

    int operator[](int ind)
    {
        return buffer[(curr_ind + ind) % length];
    }

    inline int value_at(int ind) const
    {
        return buffer[(curr_ind + ind) % length];
    }

    int operator[](int ind) const
    {
        return buffer[(curr_ind + ind) % length];
    }

    void print() const
    {
        std::cout << "Buffer: ";
        for (int i = 0; i < length; i++)
        {
            std::cout << buffer[(curr_ind + i) % length] << " ";
        }
        std::cout << "\n";
    }

    // overload the equality operator
    bool operator==(const rolling_buffer &other) const
    {
        // compare each element of the buffer
        // we don't check if length is the same. Assume that the length is the same
        for (int i = 0; i < length; i++)
        {
            if (value_at(i) != other.value_at(i))
            {
                return false;
            }
        }
        return true;
    }

    // overload the inequality operator
    bool operator!=(const rolling_buffer &other) const
    {
        // This can simply return the negation of the equality check
        return !(*this == other);
    }

};

struct rolling_hash
{
    uint64_t hash;
    uint64_t a;
    uint64_t pow_a;
    uint64_t mod;
    rolling_buffer buffer;

    void init(const uint64_t new_a,
              const uint64_t new_mod,
              const int length)
    {
        a = new_a;
        mod = new_mod;
        hash = 0;
        pow_a = 1;
        buffer.init(length);
    }

    void add(const uint64_t token)
    {
        hash = (a * hash + token) % mod;
        pow_a = (pow_a * a) % mod;
        buffer.push_back(token);
    }

    void add_remove(const uint64_t token_add)
    {
        const uint64_t token_remove = buffer.front(); // cast from int to uint64_t
        // add first
        // (because pow_a is +1 of leading term that needs to be removed)
        hash = (a * hash + token_add) % mod;
        // remove
        uint64_t token_hash = (token_remove * pow_a) % mod;
        hash = (hash + mod - token_hash) % mod;
        // No checks for buffer. It is assumed that the buffer is full at this point
        buffer.push_back(token_add); // will be cast to int
    }

    // Overload the equality operator
    bool operator==(const rolling_hash &other) const
    {
        return hash == other.hash && buffer == other.buffer; // compare the hash and the buffer
    }

    // Overload the inequality operator
    bool operator!=(const rolling_hash &other) const
    {
        // This can simply return the negation of the equality check
        return !(*this == other);
    }

    // Overload the [] operator
    int operator[](int ind)
    {
        return buffer[ind];
    }

    void print() const
    {
        std::cout << "Hash: " << hash << " ";
        buffer.print();
    }
};

#endif // ROLLING_H
