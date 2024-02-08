#ifndef TASK_MANAGEMENT_H
#define TASK_MANAGEMENT_H

#include "log.h"

#include <atomic>
#include <mutex>
#include <condition_variable>

/*
// should call it task instead to avoid confusion
enum task_state
{
    INACTIVE,
    CANCEL, // cancel the task
    IDLE,
    // PROCESSING,

    // Added for autocomplete: slot is tokenizing the prompt
    TOKENIZING,

    // Added for autocomplete: slot is comparing caches tokens to the prompt
    // After this point - if not cancelled, the cache will be changed
    // in this state we also compare if it is the same prompt as before
    // if so can skip to context processing
    CACHE_COMPARE,

    // Added for autocomplete: slot is preparing relevant cache for the prompt
    // Cache will be updated in this CACHE_PREPARE state
    CACHE_PREPARE,

    // Added for autocomplete: slot is catching up to the provided context
    // also updating the cache
    // done token by token so that it can be cancelled
    CONTEXT_PROCESSING,

    // Added for autocomplete: slot is generating suggestions up to the required suggestion length
    // done token by token so that it can be cancelled
    SUGGESTION_PROCESSING,

    // Added for autocomplete: slot is generating tokens past the required suggestion length for the next suggestion
    POST_SUGGESTION_PROCESSING,
};
*/

enum task_state
{
    INACTIVE,
    CANCEL,
    IDLE,
    CACHE_INIT,
    CACHE_COMPARE,
    CACHE_PREPARE,
    CACHE_UPDATE,
    CACHE_SUGGEST,
    POST_SUGGESTION_PROCESSING,
};

struct task_management
{
    task_management() : state(INACTIVE) {}

    void request_cancel()
    {
        LOG_INFO("Requesting cancel\n");

        {
            std::lock_guard<std::mutex> lock(state_mutex);
            LOG_VERBOSE("Cancel Request in state = %d\n", state.load());

            if (state == INACTIVE || state == CANCEL || state == IDLE)
            {
                LOG_INFO("Cancel Request not required\n");
                return;
            }

            state.store(CANCEL); // Set state to CANCEL to initiate cancellation
            state_changed.notify_all();
        }

        {
            std::unique_lock<std::mutex> lock(state_mutex);
            state_changed.wait(lock, [this]
                               { return state.load() != CANCEL; });
            LOG_INFO("Cancel Request completed\n");
        }
    }

    bool set_state(task_state new_state)
    {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (state.load() == CANCEL && new_state != IDLE)
        {
            return false;
        }
        state.store(new_state);
        return true;
    }

    bool accept_if_cancel()
    {
        LOG_VERBOSE("Checking cancel request\n");

        std::lock_guard<std::mutex> lock(state_mutex);
        if (state.load() == CANCEL)
        {
            state.store(IDLE);
            state_changed.notify_all();
            LOG_INFO("Cancel Request accepted\n");
            return true;
        }
        LOG_VERBOSE("No cancel request\n");
        return false;
    }

    bool wait_for_state(task_state expected_state)
    {
        std::unique_lock<std::mutex> lock(state_mutex);
        state_changed.wait(lock, [this, expected_state]
                           { return state.load() == expected_state; });
        return state.load() == expected_state;
    }

private:
    std::atomic<task_state> state;
    std::mutex state_mutex;
    std::condition_variable state_changed;
};

#endif // TASK_MANAGEMENT_H
