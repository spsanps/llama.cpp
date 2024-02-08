#include <iostream>
#include <vector>
#include <string>

std::vector<int> computeFailureFunction(const std::string& s) {
    int n = s.length();
    std::vector<int> failure(n, 0);
    int j = 0;

    for (int i = 1; i < n; ++i) {
        while (j > 0 && s[i] != s[j]) {
            j = failure[j - 1];
        }

        if (s[i] == s[j]) {
            ++j;
        }

        failure[i] = j;
    }

    return failure;
}

int kmpMatchSuffixPrefix(const std::string& s1, const std::string& s2) {
    std::vector<int> failure = computeFailureFunction(s2);

    int j = 0; // Index for s2
    for (int i = 0; i < s1.length(); ++i) { // Iterate over s1
        while (j > 0 && s1[i] != s2[j]) {
            j = failure[j - 1]; // Use failure function to find new position in s2
        }

        if (s1[i] == s2[j]) {
            ++j; // Characters match, move to the next character in s2
        }

        if (j == s2.length()) {
            break; // Found a complete match
        }
    }

    return j; // Length of the longest prefix of s2 that is a suffix of s1
}