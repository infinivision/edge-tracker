#ifndef __FACEATTR_H__
#define __FACEATTR_H__
#include <iostream>                                                                                                                                                                                               
#include <dlib/opencv.h>                                                                                                                                                                                          
#include <opencv2/highgui/highgui.hpp>                                                                                                                                                                            
#include <opencv2/calib3d/calib3d.hpp>                                                                                                                                                                            
#include <opencv2/imgproc/imgproc.hpp>                                                                                                                                                                            
#include <dlib/image_processing/frontal_face_detector.h>                                                                                                                                                          
#include <dlib/image_processing/render_face_detections.h>                                                                                                                                                         
#include <dlib/image_processing.h>

class FaceAttr {                                                                                                                                                                                              
  private:                                                                                                                                                                                                        
    //dlib::frontal_face_detector detector_;                                                                                                                                                                        
    dlib::shape_predictor predictor_;                                                                                                                                                                             
    std::vector<cv::Point3d> object_pts_;                                                                                                                                                                         
    std::vector<cv::Point3d> reprojectsrc_;                                                                                                                                                                       
  public:                                                                                                                                                                                                         
    FaceAttr();
    void Load(const std::string& path="shape_predictor_68_face_landmarks.dat");
    std::vector<double> GetQuality(dlib::cv_image<dlib::bgr_pixel>& cimg, int left, int top, int right, int bottom);
};

#endif

