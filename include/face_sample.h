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
        dbHandle(std::string db_ip, int db_port, std::string camera_ip);
        void insert_coordinate(long frameCounter, long faceId, unsigned tracker_index, \
                               int face_pose_type, float score, int age, cv::Mat & coordinate);
        void insert_reid(long faceId, int reid );
    private:
        Client client;
        std::string camera_id;
        dbHandle(const dbHandle & );
        dbHandle(dbHandle && );
        dbHandle & operator=(const dbHandle & );
        dbHandle & operator=(dbHandle && );
};

#endif