#include "face_pose_estimate.h"
#include <math.h>
#include <assert.h>

#define PI 3.14159265

using namespace std;
using namespace cv;

std::vector<cv::Point3d> model_points;
int   pnp_algo = cv::SOLVEPNP_EPNP;
float child_weight;
int   min_child_age = 6;
std::vector<float> face_pose_threshold;
float min_face_size = 0.0002;
void read3D_conf(string pnpConfFile){
    model_points.clear();
    try {
        auto g = cpptoml::parse_file(pnpConfFile);
        auto eyeCornerLength = g->get_qualified_as<int>("faceModel.eyeCornerLength").value_or(105);
        child_weight = g->get_qualified_as<double>("faceModel.child").value_or(0.66);
        min_child_age = g->get_qualified_as<int>("faceModel.min_child_age").value_or(6);
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

        auto threshold_array = g->get_qualified_array_of<double>("facePoseType.threshold");
        for (const auto& element : *threshold_array){
            face_pose_threshold.push_back(element);
        }

        min_face_size = g->get_qualified_as<double>("facePoseType.min_face_size").value_or(0.0002);

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

void compute_coordinate( const cv::Mat im, const std::vector<cv::Point2d> & image_points, const CameraConfig & camera, \
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
    if(age >= min_child_age && age < 18) {
        for(size_t i =0;i< model_points.size();i++){
            point = model_points[i];
            point *= child_weight + (1-child_weight) / (18-min_child_age) * (age-min_child_age);
            model_points_clone.push_back(point);
        }
    } else if(age>=18)
        model_points_clone = model_points;
    else
        return;

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

    if(!camera.default_extrinsic) {
        world_coordinate = camera.rmtx.t() * (translation_vector - camera.tvec);
        LOG(INFO) << "camera["<< camera.NO << "]" << " frame["<< frameCount << "]faceId[" << faceId 
                  << "] w coordinate: " << world_coordinate.t() / 1000;
    } else {
        world_coordinate = translation_vector;
        LOG(INFO) << "camera["<< camera.NO << "]" << " frame["<< frameCount << "]faceId[" << faceId 
                  << "] w coordinate: " << world_coordinate.t() / 1000;
    }
    
    /*
    cv::Mat r_mat;
    cv::Rodrigues(rotation_vector,r_mat);

    auto euler_angle = rotationMatrixToEulerAngles(r_mat);
    euler_angle = euler_angle / M_PI*180;

    LOG(INFO) << "camera["<< camera.NO << "]" << " frame["<< frameCount << "]faceId[" << faceId
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
    */
}


float get_theta(Point2d base, Point2d x, Point2d y) {

    Point2d vx = x-base;
    Point2d vy = y-base;
    vx.y *= -1;
    vy.y *= -1;

    float dx = atan2(vx.y,vx.x) * 180 / PI;
    float dy = atan2(vy.y,vy.x) * 180 / PI;
    float d = dy - dx;

    if(d<-180.0)
        d+=360.0;
    else if (d>180.0)
        d -= 360.0;

    return d; 

}

int check_large_pose(std::vector<Point2d> & landmark, Rect2d & box ){
    assert(landmark.size()==5);
    float theta1 = get_theta(landmark[0], landmark[3], landmark[2]);
    float theta2 = get_theta(landmark[1], landmark[2], landmark[4]);
    float theta3 = get_theta(landmark[0], landmark[2], landmark[1]);
    float theta4 = get_theta(landmark[1], landmark[0], landmark[2]);
    float theta5 = get_theta(landmark[3], landmark[4], landmark[2]);
    float theta6 = get_theta(landmark[4], landmark[2], landmark[3]);
    float theta7 = get_theta(landmark[3], landmark[2], landmark[0]);
    float theta8 = get_theta(landmark[4], landmark[1], landmark[2]);

    float left_score = 0.0;
    float right_score = 0.0;
    float up_score = 0.0;
    float down_score = 0.0;

    if(theta1<=0.0)
        left_score = 10.0;
    else if(theta2<=0.0)
        right_score = 10.0;
    else{
        left_score = theta2/theta1;
        right_score = theta1/theta2;
    }

    if(theta3<=10.0 || theta4<=10.0)
        up_score = 10.0;
    else
        up_score = max(theta1/theta3, theta2/theta4);
    
    if(theta5<=10.0 || theta6<=10.0)
        down_score = 10.0;
    else
        down_score = max(theta7/theta5, theta8/theta6);
    float mleft = (landmark[0].x+landmark[3].x)/2;
    float mright = (landmark[1].x+landmark[4].x)/2;

    Point2d box_center = (box.br() + box.tl())*0.5;

    int ret = 0;
    if(left_score>=face_pose_threshold[0])
        ret = 3;
    if(ret==0 && left_score>=face_pose_threshold[1])
        if(mright<=box_center.x)
            ret = 3;
    if(ret==0 && right_score>=face_pose_threshold[0])
        ret = 4;
    if(ret==0 && right_score>=face_pose_threshold[1])
        if(mleft>=box_center.x)
            ret = 4;
    if(ret==0 && up_score>=2.0)
        ret = 5;
    if(ret==0 && down_score>=5.0)
        ret = 6;
    if(ret==0 && left_score>face_pose_threshold[2])
        ret = 1;
    if(ret==0 && right_score>face_pose_threshold[2])
        ret = 2;

    return ret;
}

bool check_face_coordinate(Mat & frame, vector<cv::Point2d> & image_points, double * face_size){
    assert(image_points.size()==5);
    assert(frame.data);
    double dx = image_points[0].x - image_points[1].x;
    double dy = image_points[0].y - image_points[1].y;
    double dz = dx*dx+dy*dy;

    *face_size = dz / (frame.cols*frame.rows);
    if(*face_size>min_face_size)
        return true;
    else{
        LOG(INFO) << "face size is too small: " << *face_size;
        return false;
    }
}
