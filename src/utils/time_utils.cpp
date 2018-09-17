#include <chrono>
#include <iostream>
#include "time_utils.h"

using namespace std;

float getElapse(struct timeval *tv1,struct timeval *tv2)
{
    float t = 0.0f;
    if (tv1->tv_sec == tv2->tv_sec)
        t = (tv2->tv_usec - tv1->tv_usec)/1000.0f;
    else
        t = ((tv2->tv_sec - tv1->tv_sec) * 1000 * 1000 + tv2->tv_usec - tv1->tv_usec)/1000.0f;
    return t;
}

std::string get_current_time() {
    char buffer[20];
    char full[24];

    std::chrono::time_point<std::chrono::system_clock> timePoint = std::chrono::system_clock::now();
    time_t now = std::chrono::system_clock::to_time_t(timePoint);
    struct tm *timeinfo = localtime(&now);
    long long millisec = std::chrono::duration_cast<std::chrono::milliseconds>(timePoint.time_since_epoch()).count();
    millisec %= 1000;

    strftime(buffer, 20, "%Y-%m-%d.%H-%M-%S", timeinfo);
    sprintf(full, "%s.%03lld", buffer, millisec);

    std::string output = full;

    return output;
}
