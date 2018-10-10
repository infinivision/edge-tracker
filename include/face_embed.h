#ifndef __FACE_EMBED__
#define __FACE_EMBED__
#include "face_tracker.h"
#include "camera.h"

#include "mxnet/c_predict_api.h"
#include <vector>

void LoadEmbedConf(std::string mx_model_conf);
int proc_embeding(cv::Mat & face, std::vector<mx_float> & face_vec,
                 const CameraConfig & camera, int frameCounter, int thisFace);

extern int n_sample_count_type1;
extern int n_sample_count_type2;

#endif