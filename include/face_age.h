#ifndef __FACE_AGE__
#define __FACE_AGE__
#include "face_tracker.h"
#include "camera.h"

#include "mxnet/c_predict_api.h"
#include <vector>

void LoadAgeConf(std::string mx_model_conf);
int  proc_age(cv::Mat & face, vector<mx_float> & face_vec, face_tracker & target);

#endif