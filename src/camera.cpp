#include <camera.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

int min_score = 10;
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
        cv::VideoCapture capture(camera_stream);
        return capture;
    }
}

cv::Mat read_csv2d(std::string file, int row, int col) {
    char separator = ' ';
    cv::Mat result(row,col,CV_64FC1);
    std::string line, item;
    std::ifstream in( file );
    if(!in.is_open()){
        std::cout << "open csv file: " << file << " fail! " << std::endl;
        exit(1);
    }
    int i = 0, j = 0;
    while(1) {
        std::getline( in, line );
        if(!in.eof()){
            if(i > row-1) {
                std::cout<< "csv file[" << file << "] format wrong, too many row" << std::endl;
                exit(1);
            }
            std::stringstream ss( line );
            j = 0;
            while(1){
                getline ( ss, item, separator );
                if(!ss.eof()){
                    if(j>col-1){
                        std::cout<< "csv file[" << file << "] format wrong, too many col" << std::endl;
                        exit(1);
                    }
                    result.at<double>(i,j) = atof(item.c_str());
                    j++;                    
                }
                else {
                    if(j != col-1 ){
                        std::cout<< "csv file[" << file << "] format wrong, col is not enough" << std::endl;
                        exit(1);
                    }
                    result.at<double>(i,j) = atof(item.c_str());
                    break;
                }
            }
            i++;
        } else {
            if(i != row ){
                std::cout<< "csv file[" << file << "] format wrong, row is not enough" << std::endl;
                exit(1);
            }
            break;
        }
   }
   return result;
}

/*
 * parse toml configuration using cpptoml library [https://github.com/skystrife/cpptoml]
 */
std::vector<CameraConfig> LoadCameraConfig(std::string config_path) {

	std::vector<CameraConfig> cameras;

	try {
        std::shared_ptr<cpptoml::table> g = cpptoml::parse_file(config_path);
        min_score = g->get_qualified_as<int>("global.min_score").value_or(10);
        std::cout << "min score: " << min_score << std::endl;
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

            camera.tracker = table->get_as<std::string>("tracker").value_or("staple");

            camera.mat_file = table->get_as<std::string>("matrix").value_or("");
            if(camera.mat_file != "")
                camera.matrix = read_csv2d(camera.mat_file,3,3);
            camera.dist_file = table->get_as<std::string>("dist_coeff").value_or("");
            if(camera.dist_file!="")
                camera.dist_coeff = read_csv2d(camera.dist_file,1,5);
            camera.r_file = table->get_as<std::string>("rvector").value_or("");
            if(camera.r_file != ""){
                camera.rvec = read_csv2d(camera.r_file,3,1);
                cv::Rodrigues(camera.rvec,camera.rmtx);
            }
            camera.t_file = table->get_as<std::string>("tvector").value_or("");
            if(camera.t_file != "")
                camera.tvec = read_csv2d(camera.t_file,3,1);

            camera.euler_alpha = table->get_as<double>("euler_alpha").value_or(10.0);
            camera.euler_beta  = table->get_as<double>("euler_beta").value_or(35.0);
            
            cameras.push_back(camera);
        }
    }
    catch (const cpptoml::parse_exception& e) {
        std::cerr << "Failed to parse " << config_path << ": " << e.what() << std::endl;
        exit(1);
    }

    return cameras;
}