#include <camera.h>
#include <vector>
#include "utils.h"

using namespace std;

std::string CameraConfig::identity() const {
	if (ip.empty()) {
        return std::to_string(index);
    } else {
        return ip;
    }
}

/*
 * Update Attribute
 * content is of format "key=value"
 */
void CameraConfig::updateAttribute(const std::string content) {
    vector<string> values = split(content, '=');
    if (values.size() > 1)
    {
        if (values[0] == "username")
        {
            this->username = values[1];
        } else if (values[0] == "password") {
            this->password = values[1];
        }
    }
}

/*
 * Get video capture from a camera index (0, 1) or an ip (192.168.1.15)
 */
cv::VideoCapture CameraConfig::GetCapture() const {
    if ( this->ip.empty()) {
        // use camera index
        LOG(INFO) << "camera index: " << this->index;
        cv::VideoCapture capture(this->index);
        return capture;
    } else {
        LOG(INFO) << "camera ip: " << this->ip << std::endl;
        std::string camera_stream = "rtsp://" + this->username +  ":" + this->password + "@" + this->ip + ":554//Streaming/Channels/1";
        cv::VideoCapture capture(camera_stream, cv::CAP_FFMPEG);
        return capture;
    }
}

/*
 * parse toml configuration using cpptoml library [https://github.com/skystrife/cpptoml]
 */
std::vector<CameraConfig> LoadCameraConfig(std::string config_path) {

	std::vector<CameraConfig> cameras;

	try {
        std::shared_ptr<cpptoml::table> g = cpptoml::parse_file(config_path);

        auto array = g->get_table_array("camera");
        for (const auto &table : *array) {
            auto type_table = table->get_table("Type");
            auto type = type_table->get_as<string>("Title").value_or("");

            if (type == "Camera") {
                CameraConfig camera;

                auto ip = table->get_as<string>("IP").value_or("");
                camera.ip = ip;
                auto meta = table->get_as<std::string>("Meta").value_or("");
                vector<string> metas = split(meta, ',');
                for (auto content: metas) {
                    camera.updateAttribute(content);
                }
                cameras.push_back(camera);
            }
        }
    }
    catch (const cpptoml::parse_exception& e) {
        std::cerr << "Failed to parse " << config_path << ": " << e.what() << std::endl;
        exit(1);
    }

    return cameras;
}
