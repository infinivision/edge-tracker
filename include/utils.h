#ifndef __UTILS_H__
#define __UTILS_H__

#include <sys/time.h>
#include <string>
#include <vector>

#include <curl/curl.h>
#include <stdio.h>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

float getElapse(struct timeval *tv1,struct timeval *tv2);

int trave_dir(std::string& path, std::vector<std::string>& file_list);

size_t WriteCallback(char *contents, size_t size, size_t nmemb, void *userp);

#endif