#ifndef __CAMERA_H__
#define __CAMERA_H__

#include <opencv2/opencv.hpp>

class CameraConfig {

public:
    int index;
    std::string ip;
    std::string username;
    std::string password;
    int resize_rows, resize_cols;

    CameraConfig() {};
    CameraConfig(int index, int resize_rows, int resize_cols): index(index), resize_rows(resize_rows), resize_cols(resize_cols) {};
    CameraConfig(std::string ip, std::string username, std::string password, int resize_rows, int resize_cols): ip(ip), username(username), password(password), resize_rows(resize_rows), resize_cols(resize_cols) {};

    // return ip, or index if no ip is given
    inline std::string identity() {
        if (ip.empty()) {
        	return std::to_string(index);
        } else {
        	return ip;
        }
    }

    /*
     * Get video capture from a camera index (0, 1) or an ip (192.168.1.15)
     */
    inline cv::VideoCapture GetCapture() const {
        if ( this->ip.empty()) {
            // use camera index
            std::cout << "camera index: " << this->index << std::endl;
            cv::VideoCapture capture(this->index);
            return capture;
        } else {
            std::cout << "camera ip: " << this->ip << std::endl;
            std::string camera_stream = "rtsp://" + this->username +  ":" + this->password + "@" + this->ip + ":554//Streaming/Channels/1";
            cv::VideoCapture capture(camera_stream);
            return capture;
        }
    }
};

#endif