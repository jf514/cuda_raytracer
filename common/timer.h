#ifndef COMMON_TIMER_H
#define COMMON_TIMER_H
#pragma once

#include <chrono>
#include <ctime>

/// For measuring how long an operation takes.
/// C++ chrono unfortunately makes the whole type system very complicated.
struct Timer {
    std::chrono::time_point<std::chrono::system_clock> last;
};

inline float tick(Timer &timer) {
    std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = now - timer.last;
    float ret = elapsed.count();
    timer.last = now;
    return ret;
}

#endif //COMMON_TIMER_H