#ifndef __CAMERA_H__
#define __CAMERA_H__

#include "cpptoml.h"
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <vector>

class CameraConfig {

public:
    int index;
    std::string ip;
    std::string username;
    std::string password;
    int resize_rows, resize_cols;
    int detection_period;

    std::string tracker;

    // add for pose estimate
    std::string mat_file;
    std::string dist_file;
    cv::Mat matrix;
    cv::Mat dist_coeff;
    std::string r_file;
    std::string t_file;
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat rmtx;

    CameraConfig(){};

    // return ip, or index if no ip is given
    std::string identity() const;

    // Get video capture from a camera index (0, 1) or an ip (192.168.1.15)
    cv::VideoCapture GetCapture() const;
};

std::vector<CameraConfig> LoadCameraConfig(std::string config_path);

#endif