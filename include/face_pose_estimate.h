#ifndef _FACE_POSE_ESTIMATE_H_
#define _FACE_POSE_ESTIMATE_H_

#include <opencv2/opencv.hpp>
#include "cpptoml.h"
#include "camera.h"
#include <string>
#include <math.h>
#include <stdlib.h> 
#include <utils.h>

extern std::vector<cv::Point3d> model_points;
extern bool  use_default_intrinsic;
extern int   pnp_algo;
extern float child_weight;
extern int   min_child_age;

void read3D_conf(std::string pnpConfFile);
void compute_coordinate( const cv::Mat im, const std::vector<cv::Point2d> & image_points, const CameraConfig & camera, \
                         cv::Mat & world_coordinate, int age, int frameCount,int faceId);
int check_large_pose(std::vector<cv::Point2d> & landmark, cv::Rect2d & box );
bool check_face_coordinate(cv::Mat & frame, std::vector<cv::Point2d> & image_points, double * face_size);

#endif