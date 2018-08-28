#ifndef __UTILS_H__
#define __UTILS_H__

#include <sys/time.h>
#include <string>
#include <vector>

using namespace std;

std::string get_current_time();

float getElapse(struct timeval *tv1,struct timeval *tv2);

int trave_dir(std::string& path, std::vector<std::string>& file_list);

const vector<string> split(const string& s, const char& c);

#endif
