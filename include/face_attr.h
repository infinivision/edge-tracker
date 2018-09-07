#ifndef __FACEATTR_H__
#define __FACEATTR_H__
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

double GetVarianceOfLaplacianSharpness(const cv::Mat& image);

#endif

