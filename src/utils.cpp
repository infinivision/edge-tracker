#include <iostream>
#include "utils.h"
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include <vector>

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
    char buffer[26];
    char full[30];
    int millisec;
    struct tm* tm_info;
    struct timeval tv;

    gettimeofday(&tv, NULL);

    millisec = tv.tv_usec % 1000;
    tm_info = localtime(&tv.tv_sec);

    strftime(buffer, 26, "%Y-%m-%d.%H-%M-%S", tm_info);
    sprintf(full, "%s.%03d", buffer, millisec);
    std::string output = full;

    return output;
}

int trave_dir(std::string& path, std::vector<std::string>& file_list)
{
    DIR *d; //声明一个句柄
    struct dirent* dt; //readdir函数的返回值就存放在这个结构体中
    struct stat sb;    
    
    if(!(d = opendir(path.c_str())))
    {
        std::cout << "Error opendir: " << path << std::endl;
        return -1;
    }

    while((dt = readdir(d)) != NULL)
    {
        // std::cout << "file name: " << dt->d_name << std::endl;
        std::string file_name = dt->d_name;
        if(file_name[0] == '.') {
            continue;
        }
        // if(strncmp(dt->d_name, ".", 1) == 0 || strncmp(dt->d_name, "..", 2) == 0) {
        //     continue;
        // }

        std::string file_path = path + "/" + dt->d_name;
        // std::cout << "file path: " << file_path << std::endl;

        if(stat(file_path.c_str(), &sb) < 0) {
            std::cout << "Error stat file: " << file_path << std::endl;
            return -1;
        }

        if (S_ISDIR(sb.st_mode)) {
            // is a directory
            // std::cout << "directory: " << file_path << std::endl;
            trave_dir(file_path, file_list);
        } else {
            // is a regular file
            // std::cout << "file: " << file_path << std::endl;            
            file_list.push_back(file_path); 
        }
    }

    // close directory
    closedir(d);
    return 0;
}

const vector<string> split(const string& s, const char& c) {
    string buff = "";
    vector<string> v;

    for (auto n:s) {
        if (n != c) {
            buff += n;
        } else {
            if (n == c && buff != "") {
                v.push_back(buff);
                buff = "";
            }
        }
    }
    if (buff != "") {
        v.push_back(buff);
    }

    return v;
}

// int main() {
//     std::string time = get_current_time();

//     std::cout << time << std::endl;

//     return 0;
// }
