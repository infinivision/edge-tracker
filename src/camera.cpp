#include <camera.h>

std::string CameraConfig::identity() const {
	if (ip.empty()) {
        return std::to_string(index);
    } else {
        return ip;
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
        	CameraConfig camera;
        	auto index_ptr = table->get_as<int>("index");
            auto ip = table->get_as<std::string>("ip").value_or("");
            auto username_ptr = table->get_as<std::string>("username");
            auto password_ptr = table->get_as<std::string>("password");
            auto resize_rows = table->get_as<int>("resize_rows").value_or(0);
            auto resize_cols = table->get_as<int>("resize_cols").value_or(0);
            auto detection_period = table->get_as<int>("detection_period").value_or(1);

            camera.resize_rows = resize_rows;
            camera.resize_cols = resize_cols;
            camera.detection_period = detection_period;

            if (ip.empty()) {
                camera.index = *index_ptr;
            } else {
                camera.ip = ip;
                camera.username = *username_ptr;
                camera.password = *password_ptr;
            }

            cameras.push_back(camera);
        }
    }
    catch (const cpptoml::parse_exception& e) {
        std::cerr << "Failed to parse " << config_path << ": " << e.what() << std::endl;
        exit(1);
    }

    return cameras;
}