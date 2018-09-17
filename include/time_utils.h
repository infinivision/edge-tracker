#ifndef __TIME_UTILS_H__
#define __TIME_UTILS_H__

#include <ctime>
#include <string>

std::string get_current_time();

float getElapse(struct timeval *tv1,struct timeval *tv2);

#endif