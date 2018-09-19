#ifndef __FACE_SAMPLE__
#define __FACE_SAMPLE__

#include <clickhouse/client.h>
using namespace clickhouse;
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <stdexcept>

class dbHandle {
    public:
        dbHandle(std::string db_server, std::string camera_ip);
        void insert(long frameCounter, long faceId, unsigned tracker_index, \
                    bool front_side, float score, int age, int reid, cv::Mat & coordinate);
    private:
        Client client;
        std::string camera_id;
        dbHandle(const dbHandle & );
        dbHandle(dbHandle && );
        dbHandle & operator=(const dbHandle & );
        dbHandle & operator=(dbHandle && );
};

#endif