#include <camera.h>

#include <iostream>
#include <fstream>
//#include <sstream>
#include <vector>

#include <dirent.h>
#include <string>
#include "utils.h"

std::string CameraConfig::identity() const {
	if (source_type==1) {
        return ip + "_" + std::to_string(NO);
    } else if(source_type==2){
        return  "local_camera_" + std::to_string(NO);
    } else if(source_type==3){
        return "video_file_" + std::to_string(NO);
    }
    return "";
}

/*
 * Get video capture from a camera index (0, 1) or an ip (192.168.1.15)
 */
cv::VideoCapture CameraConfig::GetCapture() const {
    if(source_type == 1){
        LOG(INFO) << "camera ip: " << this->ip << std::endl;
        std::string camera_stream = "rtsp://" + this->username +  ":" + this->password + "@" + this->ip + ":554//Streaming/Channels/1";
        cv::VideoCapture capture(camera_stream);
        return capture;
    } else if (source_type == 2) {
        LOG(INFO) << "camera index: " << this->idx;
        cv::VideoCapture capture(this->idx);
        return capture;
    } else if(source_type == 3){
        // use camera index
        LOG(INFO) << "video file: " << this->video_file;
        cv::VideoCapture capture(this->video_file);
        return capture;
    }
    return cv::VideoCapture();
}

cv::Mat json2Mat(json & j_array, int row, int col, std::string param_name ) {

    if(!j_array.is_array()){
        std::cout << param_name << " not a json array\n";
        exit(-1);
    }
    if(j_array.size()!=(row*col)){
        std::cout << param_name << " array size is wrong \n";
        exit(-1);
    }

    cv::Mat result(row,col,CV_64FC1);
    for(int i=0; i<row; i++)
        for(int j=0; j<col; j++){
            result.at<double>(i,j) = j_array[i*col+j];
        }
            

    return result;
}

/*
 * parse toml configuration using cpptoml library [https://github.com/skystrife/cpptoml]
 */
std::vector<CameraConfig> LoadCameraConfig(std::string config_path) {
    std::vector<std::string> meta_files;
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (config_path.c_str())) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {
            std::string file(ent->d_name);
            std::string postfix = ".meta";
            if(file.rfind(postfix)==(file.length()-postfix.length())){
                meta_files.push_back(std::string(ent->d_name));
            }                
        }
        closedir(dir);
    } else {
        /* could not open directory */
        perror ("");
        exit(-1);
    }

	std::vector<CameraConfig> cameras;

    int no=0;
    for(auto & file: meta_files){
        no++;
        try {
            std::ifstream meta_file(config_path + "/" + file);
            json meta;
            meta_file >> meta;

            CameraConfig camera;
            camera.NO = no;

            camera.source_type = meta["source_type"];

            if(camera.source_type==1)
                camera.ip = meta["ip"];
            else if(camera.source_type==2)
                camera.idx = meta["idx"];
            else if(camera.source_type==3)
                camera.video_file  = meta["video_file"];
            else{
                std::cout << "unknown camera source type!\n";
                exit(-1);
            }

            camera.username    = meta["username"];
            camera.password    = meta["password"];
//            camera.euler_alpha = meta["euler_alpha"];
//            camera.euler_beta  = meta["euler_beta"];

            camera.default_intrinsic = meta["default_intrinsic"];
            if(!camera.default_intrinsic){
                json mtx = meta["matrix"];
                camera.matrix = json2Mat(mtx,3,3,camera.ip+"intrinsic matrix");

                json dist_coeff = meta["dist_coeff"];
                camera.dist_coeff = json2Mat(dist_coeff,1,5,camera.ip+"intrinsic dist_coeff");
            }

            camera.default_extrinsic = meta["default_extrinsic"];
            if(!camera.default_extrinsic){
                json rvecor = meta["rvector"];
                camera.rvec = json2Mat(rvecor,3,1,camera.ip+"extrinsic rotation vector");
                cv::Rodrigues(camera.rvec,camera.rmtx);

                json tvecor = meta["tvector"];
                camera.tvec = json2Mat(tvecor,3,1,camera.ip+"extrinsic translation vector");
            }

            cameras.push_back(camera);

        }
        catch(json::exception & e ){
            LOG(WARNING) <<"failed to parse meta file[" << file << "]: " << e.what() << "\n";
            exit(-1);
        }
        catch(std::exception & e){
            LOG(WARNING) << "failed to process meta file[" << file << "]: " << e.what() << "\n";
            exit(-1);            
        }
    }

    return cameras;
}