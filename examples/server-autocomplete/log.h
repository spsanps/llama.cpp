#ifndef LOG_H
#define LOG_H

#include <iostream>

// LOG_VERBOSE

const int verbose_level = 3;

#define LOG_VERBOSE(...)         \
    do                           \
    {                            \
        if (verbose_level > 0)   \
        {                        \
            printf("VERBOSE: "); \
            printf(__VA_ARGS__); \
            fflush(stdout);      \
        }                        \
    } while (0)

// LOG_INFO

#define LOG_INFO(...)        \
    do                       \
    {                        \
        printf("INFO: ");    \
        printf(__VA_ARGS__); \
        fflush(stdout);      \
    } while (0)

// LOG_WARNING

#define LOG_WARNING(...)     \
    do                       \
    {                        \
        printf("WARNING: "); \
        printf(__VA_ARGS__); \
        fflush(stdout);      \
    } while (0)

// LOG_ERROR

#define LOG_ERROR(...)       \
    do                       \
    {                        \
        printf("ERROR: ");   \
        printf(__VA_ARGS__); \
        fflush(stdout);      \
    } while (0)

#endif // LOG_H