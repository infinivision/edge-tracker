#ifndef __IMAGE_QUALITY_H__
#define __IMAGE_QUALITY_H__

#include <opencv2/opencv.hpp>

static int magic = 35000;    // for measuring blur

//double GetImageQuality(IplImage* img, int left, int top, int right, int bottom);
double GetNormOfDerivativesBlurriness(const cv::Mat& image);
double GetVarianceOfLaplacianSharpness(const cv::Mat& image);

#endif
