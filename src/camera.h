#ifndef __CAMERA_H__
#define __CAMERA_H__

#include "cpptoml.h"
#include <opencv2/opencv.hpp>
#include <glog/logging.h>

class CameraConfig {

public:
    int index;
    std::string ip;
    std::string username;
    std::string password;
    int resize_rows, resize_cols;
    int detection_period;

    CameraConfig(){};

    CameraConfig(std::string config_path) {
    	try {
        	std::shared_ptr<cpptoml::table> g = cpptoml::parse_file(config_path);

            auto array = g->get_table_array("camera")->get();
            auto camera_ptr = array[0];
            auto index_ptr = camera_ptr->get_as<int>("index");
            auto ip = camera_ptr->get_as<std::string>("ip").value_or("");
            auto username_ptr = camera_ptr->get_as<std::string>("username");
            auto password_ptr = camera_ptr->get_as<std::string>("password");
            auto resize_rows = camera_ptr->get_as<int>("resize_rows").value_or(0);
            auto resize_cols = camera_ptr->get_as<int>("resize_cols").value_or(0);
            auto detection_period = camera_ptr->get_as<int>("detection_period").value_or(1);

            this->resize_rows = resize_rows;
            this->resize_cols = resize_cols;
            this->detection_period = detection_period;

            if (ip.empty()) {
            	this->index = *index_ptr;
            } else {
            	this->ip = ip;
            	this->username = *username_ptr;
            	this->password = *password_ptr;
            }
        }
        catch (const cpptoml::parse_exception& e) {
            std::cerr << "Failed to parse " << config_path << ": " << e.what() << std::endl;
            exit(1);
        }
    }

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
            LOG(INFO) << "camera index: " << this->index;
            cv::VideoCapture capture(this->index);
            return capture;
        } else {
            LOG(INFO) << "camera ip: " << this->ip << std::endl;
            std::string camera_stream = "rtsp://" + this->username +  ":" + this->password + "@" + this->ip + ":554//Streaming/Channels/1";
            cv::VideoCapture capture(camera_stream);
            return capture;
        }
    }
};

#endif