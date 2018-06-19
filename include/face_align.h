#ifndef __FACEALIGN_H__
#define __FACEALIGN_H__
#include <iostream>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class FaceAlign { 
  private:
    std::vector<cv::Point2d> align_src_;
    cv::Size size;
  public:
    FaceAlign();
    cv::Mat Align(cv::Mat& input, const std::vector<cv::Point2d>& align_dst);
};

#endif

