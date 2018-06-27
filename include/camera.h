#ifndef __CAMERA_H__
#define __CAMERA_H__

class CameraConfig {

public:
    int index;
    std::string ip;
    std::string username;
    std::string password;

    CameraConfig() {};
    CameraConfig(int index): index(index) {};
    CameraConfig(std::string ip, std::string username, std::string password): ip(ip), username(username), password(password) {};

    // return ip, or index if no ip is given
    inline std::string identity() {
        if (ip.empty()) {
        	return std::to_string(index);
        } else {
        	return ip;
        }
    } 
};

#endif