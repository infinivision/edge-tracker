#include "face_pose_estimate.h"

using namespace std;
using namespace cv;

std::vector<cv::Point3d> model_points;
int   pnp_algo = cv::SOLVEPNP_EPNP;
float child_weight;
int   child_age_min = 6;

void read3D_conf(string pnpConfFile){
    model_points.clear();
    try {
        auto g = cpptoml::parse_file(pnpConfFile);
        auto eyeCornerLength = g->get_qualified_as<int>("faceModel.eyeCornerLength").value_or(105);
        child_weight = g->get_qualified_as<double>("faceModel.child").value_or(0.66);
        child_age_min = g->get_qualified_as<int>("faceModel.child_age_min").value_or(6);
        auto array = g->get_table_array("face3dpoint");
        for (const auto &table : *array) {
        	auto x = table->get_as<double>("x");
            auto y = table->get_as<double>("y");
            auto z = table->get_as<double>("z");
            cv::Point3d point(*x,*y,*z);
            point = point / 450 * eyeCornerLength;
            model_points.push_back(point);
        }
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

bool isRotationMatrix(Mat &R)
{
    Mat Rt;
    transpose(R, Rt);
    Mat shouldBeIdentity = Rt * R;
    Mat I = Mat::eye(3,3, shouldBeIdentity.type());
    return  norm(I, shouldBeIdentity) < 1e-6; 
}

Vec3f rotationMatrixToEulerAngles(Mat &R)
{
    assert(isRotationMatrix(R));
    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
    bool singular = sy < 1e-6; // If
    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return Vec3f(x, y, z); 
}

#ifdef BENCH_EDGE
long  sum_t_pnp     = 0;
long  pnp_count = 0;
#endif

bool compute_coordinate( const cv::Mat im, const std::vector<cv::Point2d> & image_points, const CameraConfig & camera, \
                         cv::Mat & world_coordinate, int age, int frameCount,int faceId) {
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    double focal_length = im.cols; // Approximate focal length.
    Point2d center = cv::Point2d(im.cols/2,im.rows/2);
    if(camera.default_intrinsic) {
        // Camera internals
        camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
        dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type); // Assuming no lens distortion
    } else {
        camera_matrix =  camera.matrix;
        dist_coeffs   =  camera.dist_coeff;
    }
    
    cv::Mat rotation_vector; // Rotation in axis-angle form
    cv::Mat translation_vector;
    std::vector<cv::Point3d> model_points_clone;
    Point3d point;
    if(age >= child_age_min && age < 18) {
        for(size_t i =0;i< model_points.size();i++){
            point = model_points[i];
            point *= child_weight + (1-child_weight) / (18-child_age_min) * (age-child_age_min);
            model_points_clone.push_back(point);
        }
    } else if(age>=18)
        model_points_clone = model_points;
    else
        return false;

#ifdef BENCH_EDGE
    struct timeval  tv;
    gettimeofday(&tv,NULL);
    long t_ms1 = tv.tv_sec * 1000 * 1000 + tv.tv_usec;
#endif
    // Solve for pose
    cv::solvePnP(model_points_clone, image_points, camera_matrix, dist_coeffs,    \
                    rotation_vector, translation_vector, false, pnp_algo);

#ifdef BENCH_EDGE
    gettimeofday(&tv,NULL);
    long t_ms2 = tv.tv_sec * 1000 * 1000 + tv.tv_usec;
    sum_t_pnp += t_ms2-t_ms1;
    pnp_count++;
    LOG(INFO) << "PnP performance: [" << (sum_t_pnp/1000.0 ) / pnp_count << "] mili second latency per time";
#endif
    /*
    string points_str;
    for(auto point: image_points){
        points_str += to_string(point.x) + "," + to_string(point.y) + ";";
    }
    LOG(INFO) << "ip[" << camera.ip << "] image point: " << points_str;
    
    std::cout << "camera_matrix\n" << camera_matrix << endl;
    std::cout << "model_points\n" << model_points << endl;
    std::cout << "image_points\n" << image_points << endl;
    */

    // Project a 3D point (0, 0, 1000.0) onto the image plane.
    // We use this to draw a line sticking out of the nose
    // LOG(INFO) << camera.ip << " tvec: "<< translation_vector.t()/1000;

    if(!camera.default_extrinsic) {
        world_coordinate = camera.rmtx.t() * (translation_vector - camera.tvec);
        LOG(INFO) << camera.ip << " frame["<< frameCount << "]faceId[" << faceId 
                  << "] w coordinate: " << world_coordinate.t() / 1000;
    } else 
        world_coordinate = translation_vector;

    cv::Mat r_mat;
    cv::Rodrigues(rotation_vector,r_mat);

    auto euler_angle = rotationMatrixToEulerAngles(r_mat);
    euler_angle = euler_angle / M_PI*180;

    LOG(INFO) << camera.ip << " frame["<< frameCount << "]faceId[" << faceId
              << "] euler_angle: " << euler_angle;

    auto x = abs(euler_angle[0]);
    if(x>90)
        x = 180 - x;
    auto y = abs(euler_angle[1]);
    auto z = abs(euler_angle[2]);
    if(z>90)
        z = 180 - z;
    if(x>camera.euler_alpha || z>camera.euler_alpha)
        return false;
    if(y>camera.euler_beta)
        return false;
    return true;
}
