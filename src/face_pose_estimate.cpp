#include <opencv2/opencv.hpp>
#include "cpptoml.h"
#include "camera.h"
#include <string>

using namespace std;
using namespace cv;

std::vector<cv::Point3d> model_points;
bool  use_default_intrinsic = false;
int   pnp_algo = cv::SOLVEPNP_EPNP;
float male_weight;
float female_weight;
float child_weight;

bool read3D_conf(){
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
        auto array = g->get_table_array("face3d");
        for (const auto &table : *array) {
        	auto x = table->get_as<double>("x");
            auto y = table->get_as<double>("y");
            auto z = table->get_as<double>("z");
            cv::Point3d point(*x,*y,*z);
            point = point / 450 * eyeCornerLength;
            model_points.push_back(point);
        }
        use_default_intrinsic = g->get_qualified_as<bool>("pnp.use_default_intrinsic").value_or(false);
        std::string algo = g->get_qualified_as<std::string>("pnp.pnp_algo").value_or("");
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
    return true;
}

void compute_coordinate( const cv::Mat im, const std::vector<cv::Point2d> & image_points, const CameraConfig & camera, int age, int sex) {
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    double focal_length = im.cols; // Approximate focal length.
    Point2d center = cv::Point2d(im.cols/2,im.rows/2);
    if(use_default_intrinsic) {
        // Camera internals
        camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
        dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type); // Assuming no lens distortion
    } else {
        if(camera.mat_file!="")
            camera_matrix =  camera.matrix;
        else
            camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
        if(camera.dist_file!="")
            dist_coeffs   =  camera.dist_coeff;
        else
            dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type);
    }
    
    cv::Mat rotation_vector; // Rotation in axis-angle form
    cv::Mat translation_vector;

    if(age > 18){
        if(sex == 1)
            for(size_t i =0;i< model_points.size();i++)
                model_points[i] = model_points[i] * male_weight;
        else
            for(size_t i =0;i< model_points.size();i++)
                model_points[i] = model_points[i] * female_weight;
    } else if(age >= 6){
            for(size_t i =0;i< model_points.size();i++)
                model_points[i] = model_points[i] * (child_weight * ((age- 6) / 12.0f + 1));
    } else {
        LOG(INFO) << "can't compute for child age less than six!";
        return;
    }

    // Solve for pose
    cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs,    \
                    rotation_vector, translation_vector, false, pnp_algo);
 
    // Project a 3D point (0, 0, 1000.0) onto the image plane.
    // We use this to draw a line sticking out of the nose
    cv::Mat world_coordinate;
    if(camera.r_file!="" && camera.t_file!=""){
        world_coordinate = (translation_vector - camera.tvec) * camera.rmtx.t();
        LOG(INFO) << "world_coordinate: " << world_coordinate ; 
    } else
        LOG(INFO) << "can't compute coordinate without camera rotation matrix of the world coordinate" ;

}