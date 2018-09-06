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
    int detection_period;

    CameraConfig(){};

    // return ip, or index if no ip is given
    std::string identity() const;

    /*
     * Update Attribute
     * content is of format "key=value"
     */
    void updateAttribute(const std::string content);

    // Get video capture from a camera index (0, 1) or an ip (192.168.1.15)
    cv::VideoCapture GetCapture() const;
};

std::vector<CameraConfig> LoadCameraConfig(std::string config_path);

#endif
