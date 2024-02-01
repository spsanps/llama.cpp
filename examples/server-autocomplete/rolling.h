#ifndef ROLLING_H
#define ROLLING_H

#include <vector>
#include <cstdint>

struct rolling_buffer
{
    int n, curr_ind;
    std::vector<int> buffer;

    void init(int new_n);
    void reset();
    void push_back(int token);
    int at(int ind);
    int front();
    int back();
    int operator[](int ind);
    int operator[](int ind) const;
    void print() const;
};

struct rolling_hash {
    void init(const uint64_t new_a, const uint64_t new_mod);
    void add(const uint64_t token);
    void add_remove(const uint64_t token_remove, const uint64_t token_add);

    uint64_t hash;
    uint64_t a;
    uint64_t pow_a;
    uint64_t mod;
};

#endif // ROLLING_H
