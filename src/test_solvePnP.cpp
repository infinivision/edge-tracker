#include <iostream>
#include <opencv2/opencv.hpp>
#include "camera.h"
#include "mtcnn.h"
#include <string.h>
#include <chrono>
#include <cstdlib>
#include "utils.h"
#include <glog/logging.h>
#include <stdlib.h> 

#define QUIT_KEY 'q'

using namespace std;
using namespace cv;

std::vector<cv::Point3d> model_points;
bool  use_default_intrinsic = false;
int   pnp_algo = cv::SOLVEPNP_EPNP;
float male_weight;
float female_weight;
float child_weight;

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

void read3D_conf(){
    model_points.clear();
    char * pnpConfPath = getenv("pnpPath");
    if(pnpConfPath == nullptr){
        pnpConfPath = "3dpnp.toml";
    }
    try {
        auto g = cpptoml::parse_file(pnpConfPath);
        auto eyeCornerLength = g->get_qualified_as<int>("faceModel.eyeCornerLength").value_or(105);
        male_weight = g->get_qualified_as<double>("faceModel.male").value_or(1.05);
        female_weight = g->get_qualified_as<double>("faceModel.female").value_or(0.95);
        child_weight = g->get_qualified_as<double>("faceModel.child").value_or(0.66);
        auto array = g->get_table_array("face3dpoint");
        for (const auto &table : *array) {
        	auto x = table->get_as<double>("x");
            auto y = table->get_as<double>("y");
            auto z = table->get_as<double>("z");
            cv::Point3d point(*x,*y,*z);
            point = point / 450 * eyeCornerLength;
            model_points.push_back(point);
        }
        
        use_default_intrinsic = g->get_qualified_as<bool>("pnp.use_default_intrinsic").value_or("false");
        cout <<"use_default_intrinsic: " << use_default_intrinsic << endl;
        std::string algo = g->get_qualified_as<std::string>("pnp.pnp_algo").value_or("EPNP");
        if( algo == "EPNP")
            pnp_algo = cv::SOLVEPNP_EPNP;
        else if (algo == "UPNP"){
            pnp_algo = cv::SOLVEPNP_UPNP;
        }
        else if (algo == "DLS"){
            pnp_algo = cv::SOLVEPNP_DLS;
        }
    }
    catch (const cpptoml::parse_exception& e) {
        std::cerr << "Failed to parse 3dpnp.toml: " << e.what() << std::endl;
        exit(1);
    }
}

int main(int argc, char* argv[]) {

    google::InitGoogleLogging("testPnP");
    FLAGS_log_dir = "./";
//    FLAGS_logtostderr = true;
    read3D_conf();
    const String keys =
        "{help h usage ? |                         | print this message   }"
        "{model        |models/ncnn                | path to mtcnn model  }"
        "{config       |config.toml                | camera config        }"
        "{image        |                           | input image  file     }"
        "{points       |                           | input points file     }"
    ;

    CommandLineParser parser(argc, argv, keys);
    parser.about("camera face detector");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    String config_path = parser.get<String>("config");
    LOG(INFO) << "config path: " << config_path;

    String model_path = parser.get<String>("model");
    String image_file = parser.get<String>("image");
    String points_file = parser.get<String>("points");
    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    vector<CameraConfig> cameras = LoadCameraConfig(config_path);

    // LOG(INFO) << "detection period: " << camera.detection_period;

    CameraConfig camera = cameras[cameras.size()-1];
    cout << "processing camera: " << camera.identity() << endl;
    read3D_conf();
    for(size_t i =0;i< model_points.size();i++)
        model_points[i] = model_points[i] * male_weight;
    
    LOG(INFO) << "mtx:\n "   << camera.matrix;
    LOG(INFO) << "dist:\n "  << camera.dist_coeff;

    MTCNN mm(model_path);

    VideoCapture cap = camera.GetCapture();
    if (!cap.isOpened()) {
        cerr << "failed to open camera" << endl;
        return -1;
    }
    if(image_file!=""){
        cv::Mat frame = imread(image_file);
        int total = 0;
        std::vector<cv::Point2d> image_points;
        vector<Bbox> detected_bounding_boxes;
        ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
        mm.detect(ncnn_img, detected_bounding_boxes);

        for(vector<Bbox>::iterator it=detected_bounding_boxes.begin(); it!=detected_bounding_boxes.end();it++) {
            if((*it).exist) {
                total++;
                // get face bounding box
                Bbox box = *it;
                image_points.clear();
                for(int i =0;i<5;i++){
                    cv::Point2d point(box.ppoint[i],box.ppoint[i+5]);
                    image_points.push_back(point);
                }
                string points_str;
                for(auto point: image_points){
                    points_str += to_string(point.x) + "," + to_string(point.y) + ";";
                }
                LOG(INFO) << "ip[" << camera.ip << "] image point: " << points_str;
                cv::Mat rotation_vector;
                cv::Mat translation_vector;
                cv::solvePnP(model_points, image_points, camera.matrix, camera.dist_coeff,    \
                            rotation_vector, translation_vector, false, cv::SOLVEPNP_EPNP);
                LOG(INFO) << "ip[" << camera.ip << "] tvec: "<< translation_vector.t()/1000;
            }
        }
        LOG(INFO) << "detected " << total << " Persons";
    }

    if(points_file!=""){
        cv::Mat  points_mat = read_csv2d(points_file, 5, 2);

        cv::Mat rotation_vector;
        cv::Mat translation_vector;
        for(int i=0;i<100;i++){

            std::vector<cv::Point2d> image_points;
            image_points.clear();
            for(size_t i=0;i<5;i++){
                cv::Point2d p;
                p.x = points_mat.at<double>(i,0) + (double)(rand()%20-10) /10;
                p.y = points_mat.at<double>(i,1) + (double)(rand()%20-10) /10;
                image_points.push_back(p);
            }
            string points_str;
            for(auto point: image_points) {
                points_str += to_string(point.x) + "," + to_string(point.y) + ";";
            }
            LOG(INFO) << "ip[" << camera.ip << "] image point: " << points_str;
            cv::solvePnP(model_points, image_points, camera.matrix, camera.dist_coeff,    \
                        rotation_vector, translation_vector, false, cv::SOLVEPNP_EPNP);
            LOG(INFO) << "ip[" << camera.ip << "] tvec: "<< translation_vector.t()/1000;
            cout << translation_vector.t()/1000 << endl;
        }
    }
    return 0;
}
