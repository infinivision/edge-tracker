#include <iostream>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <face_attr.h>

struct profiler
{
    std::string name;
    std::chrono::high_resolution_clock::time_point p;
    profiler(std::string const &n) :
        name(n), p(std::chrono::high_resolution_clock::now()) { }
    ~profiler()
    {
        using dura = std::chrono::duration<double>;
        auto d = std::chrono::high_resolution_clock::now() - p;
        std::cout << name << ": "
            << std::chrono::duration_cast<dura>(d).count()
            << std::endl;
    }
};

#define PROFILE_BLOCK(pbn) profiler _pfinstance(pbn)

FaceAttr::FaceAttr() {
  //detector_ = dlib::get_frontal_face_detector();


  //fill in 3D ref points(world coordinates), model referenced from http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
  std::vector<cv::Point3d>& object_pts = object_pts_;
  object_pts.push_back(cv::Point3d(6.825897, 6.760612, 4.402142));     //#33 left brow left corner
  object_pts.push_back(cv::Point3d(1.330353, 7.122144, 6.903745));     //#29 left brow right corner
  object_pts.push_back(cv::Point3d(-1.330353, 7.122144, 6.903745));    //#34 right brow left corner
  object_pts.push_back(cv::Point3d(-6.825897, 6.760612, 4.402142));    //#38 right brow right corner
  object_pts.push_back(cv::Point3d(5.311432, 5.485328, 3.987654));     //#13 left eye left corner
  object_pts.push_back(cv::Point3d(1.789930, 5.393625, 4.413414));     //#17 left eye right corner
  object_pts.push_back(cv::Point3d(-1.789930, 5.393625, 4.413414));    //#25 right eye left corner
  object_pts.push_back(cv::Point3d(-5.311432, 5.485328, 3.987654));    //#21 right eye right corner
  object_pts.push_back(cv::Point3d(2.005628, 1.409845, 6.165652));     //#55 nose left corner
  object_pts.push_back(cv::Point3d(-2.005628, 1.409845, 6.165652));    //#49 nose right corner
  object_pts.push_back(cv::Point3d(2.774015, -2.080775, 5.048531));    //#43 mouth left corner
  object_pts.push_back(cv::Point3d(-2.774015, -2.080775, 5.048531));   //#39 mouth right corner
  object_pts.push_back(cv::Point3d(0.000000, -3.116408, 6.097667));    //#45 mouth central bottom corner
  object_pts.push_back(cv::Point3d(0.000000, -7.415691, 4.070434));    //#6 chin corner



  //reproject 3D points world coordinate axis to verify result pose
  std::vector<cv::Point3d>& reprojectsrc = reprojectsrc_;
  reprojectsrc.push_back(cv::Point3d(10.0, 10.0, 10.0));
  reprojectsrc.push_back(cv::Point3d(10.0, 10.0, -10.0));
  reprojectsrc.push_back(cv::Point3d(10.0, -10.0, -10.0));
  reprojectsrc.push_back(cv::Point3d(10.0, -10.0, 10.0));
  reprojectsrc.push_back(cv::Point3d(-10.0, 10.0, 10.0));
  reprojectsrc.push_back(cv::Point3d(-10.0, 10.0, -10.0));
  reprojectsrc.push_back(cv::Point3d(-10.0, -10.0, -10.0));
  reprojectsrc.push_back(cv::Point3d(-10.0, -10.0, 10.0));

}

void FaceAttr::Load(const std::string& path) {
  dlib::deserialize(path) >> predictor_;
}

std::vector<double> FaceAttr::GetPoseQuality(dlib::cv_image<dlib::bgr_pixel>& cimg, int left, int top, int right, int bottom) {
  //PROFILE_BLOCK("all");
  dlib::rectangle face(left, top, right, bottom);
  std::vector<double> ret;
  //std::vector<dlib::rectangle> faces;
  //{
  //  PROFILE_BLOCK("det");
  //  faces = detector_(cimg);
  //}
  double K[9] = { 6.5308391993466671e+002, 0.0, 3.1950000000000000e+002, 0.0, 6.5308391993466671e+002, 2.3950000000000000e+002, 0.0, 0.0, 1.0 };
  static double D[5] = { 7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000 };
  static int L[14] = {17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8};
  K[2] = (face.left()+face.right())/2;
  K[5] = (face.top()+face.bottom())/2;
  dlib::full_object_detection shape;
  {
    //PROFILE_BLOCK("landmark");
    shape = predictor_(cimg, face);
  }

  std::vector<cv::Point2d> image_pts;
  for(int m:L) {
    int x = shape.part(m).x();
    int y = shape.part(m).y();
    image_pts.push_back(cv::Point2d(x, y));
  }
  cv::Mat cam_matrix = cv::Mat(3, 3, CV_64FC1, K);
  cv::Mat dist_coeffs = cv::Mat(5, 1, CV_64FC1, D);
  cv::Mat rotation_vec;
  cv::Mat rotation_mat;
  cv::Mat translation_vec;
  cv::Mat pose_mat = cv::Mat(3, 4, CV_64FC1);
  cv::Mat euler_angle = cv::Mat(3, 1, CV_64FC1);
  std::vector<cv::Point2d> reprojectdst;
  reprojectdst.resize(8);
  cv::Mat out_intrinsics = cv::Mat(3, 3, CV_64FC1);
  cv::Mat out_rotation = cv::Mat(3, 3, CV_64FC1);
  cv::Mat out_translation = cv::Mat(3, 1, CV_64FC1);
  //cv::solvePnP(object_pts_, image_pts, cam_matrix, dist_coeffs, rotation_vec, translation_vec);
  cv::solvePnP(object_pts_, image_pts, cam_matrix, cv::Mat(), rotation_vec, translation_vec);
  cv::projectPoints(reprojectsrc_, rotation_vec, translation_vec, cam_matrix, dist_coeffs, reprojectdst);
  cv::Rodrigues(rotation_vec, rotation_mat);
  cv::hconcat(rotation_mat, translation_vec, pose_mat);
  cv::decomposeProjectionMatrix(pose_mat, out_intrinsics, out_rotation, out_translation, cv::noArray(), cv::noArray(), cv::noArray(), euler_angle);
  //std::cout<<"("<<faces[f].left()<<","<<faces[f].top()<<")("<<faces[f].right()<<","<<faces[f].bottom()<<")"<<std::endl;
  for(int i=0;i<3;i++) {
    double v = euler_angle.at<double>(i);
    ret.push_back(v);
    //std::cout<<i<<":"<<v<<std::endl;
  }
  return ret;
}

double FaceAttr::GetImageQuality(IplImage* img, int left, int top, int right, int bottom) {
    double temp = 0;                                                                                                                                                                                              
    double DR = 0;
    int i,j;                                                                                                                                                                                                      
    int height=bottom-top;
    int width=right-left;                                                                                                                                                                                         
    int step=img->widthStep/sizeof(uchar);
    uchar *data=(uchar*)img->imageData;                                                                                                                                                                           
    double num = width*height;                                                                                                                                                                                                                                                                                                                                                                                                      
    for(i=top;i<bottom;i++) {                                                                                                                                                                                                             
        for(j=left;j<right;j++) {                                                                                                                                                                                                         
            temp += sqrt((pow((double)(data[(i+1)*step+j]-data[i*step+j]),2) + pow((double)(data[i*step+j+1]-data[i*step+j]),2)));
            temp += abs(data[(i+1)*step+j]-data[i*step+j])+abs(data[i*step+j+1]-data[i*step+j]);                                                                                                                  
        }
    }                                                                                                                                                                                                             
    DR = temp/num;                                                                                                                                                                                                
    return DR;                                                                                                                                                                                                    
}