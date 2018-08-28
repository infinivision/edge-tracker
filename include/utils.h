#ifndef __UTILS_H__
#define __UTILS_H__

#include "camera.h"
#include "mtcnn.h"
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <string>
#include <vector>

using namespace std;

std::string get_current_time();

float getElapse(struct timeval *tv1,struct timeval *tv2);

int trave_dir(std::string& path, std::vector<std::string>& file_list);

const vector<string> split(const string& s, const char& c);

bool overlap(const cv::Rect2d &box1, const cv::Rect2d &box2);

void prepare_output_folder(const CameraConfig &camera, string &output_folder);

void saveFace(const cv::Mat &frame, const Bbox &box, long faceId, string outputFolder);

#endif
