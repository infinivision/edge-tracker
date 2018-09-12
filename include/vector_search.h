#ifndef __VEC_SEARCH__
#define __VEC_SEARCH__

#include "camera.h"

void LoadVecSearchConf(std::string mx_model_conf);

int proc_embd_vec(std::vector<float> &data, const CameraConfig & camera,int frameCount,int faceId);


#endif