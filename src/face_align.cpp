#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <face_align.h>

using namespace std;

FaceAlign::FaceAlign() {
  align_src_.push_back(cv::Point2f(38.2946, 51.6963));
  align_src_.push_back(cv::Point2f(73.5318, 51.5014));
  align_src_.push_back(cv::Point2f(56.0252, 71.7366));
  align_src_.push_back(cv::Point2f(41.5493, 92.3655));
  align_src_.push_back(cv::Point2f(70.7299, 92.2041));

  size = cv::Size(112,112);
}

/*
 * Align face
 */
cv::Mat FaceAlign::Align(const cv::Mat& input, const std::vector<cv::Point2f>& align_dst) {

  cv::Mat warped;

  // for (int i = 0; i < align_dst.size(); ++i){
  //   cout << align_dst[i] << endl;
  // }

  // cout << "\testimate rigid transform ...";
  cv::Mat R = cv::estimateRigidTransform(align_dst,align_src_,false);
  // cout << " end, R size: " << R.size() << endl;

  if (R.empty()) {
    cout << "\tskip face alignment" << endl;
    return warped;
  }

  // cout << R << endl;

  // R is now a matrix of size 2 * 3
  cv::Mat H = cv::Mat(2,3,R.type());
  H.at<double>(0,0) = R.at<double>(0,0);
  H.at<double>(0,1) = R.at<double>(0,1);
  H.at<double>(0,2) = R.at<double>(0,2);

  H.at<double>(1,0) = R.at<double>(1,0);
  H.at<double>(1,1) = R.at<double>(1,1);
  H.at<double>(1,2) = R.at<double>(1,2);

  // H.at<double>(2,0) = 0.0;
  // H.at<double>(2,1) = 0.0;
  // H.at<double>(2,2) = 1.0;

  // cout << "\twarp affine ...";
  cv::warpAffine(input,warped,H,size);
  // cv::warpPerspective(input,warped,H,size);
  // cout << " end" << endl;

  return warped;
}
