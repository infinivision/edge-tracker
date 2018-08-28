#include <iostream>
#include <opencv2/opencv.hpp>
#include "face_attr.h"
#include <image_quality.h>
#include "utils.h"

using namespace std;

int test_picture(string imagepath) {

    cv::Mat image = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    if (image.empty()) {
        std::cerr << "cv::Imread failed. File Path: " << imagepath << std::endl;
        return -1;
    }

    // cout << "image rows: " << image.rows << ", cols: " << image.cols << endl;

    dlib::cv_image<dlib::bgr_pixel> cimg(image);
    cv::Mat gray;
    cv::cvtColor(image, gray, CV_BGR2GRAY);
    IplImage limg = gray;
    IplImage *plimg = &limg;
    FaceAttr fa;
    fa.Load();

    struct timeval  tv1,tv2;
    struct timezone tz1,tz2;

    gettimeofday(&tv1,&tz1);
    // detect faces
    std::vector<double> poses = fa.GetPoseQuality(cimg, 0, 0, image.rows-1, image.cols-1);
    gettimeofday(&tv2,&tz2);
    std::cout << "time eclipsed: " <<  getElapse(&tv1, &tv2) << " ms" << std::endl;


    gettimeofday(&tv1,&tz1);
    double quality = GetImageQuality(plimg, 0, 0, image.rows-1, image.cols-1);
    gettimeofday(&tv2,&tz2);
    std::cout << "Image quality: " << quality << ", time eclipsed: " <<  getElapse(&tv1, &tv2) << " ms" << std::endl;

    gettimeofday(&tv1,&tz1);
    double blur = GetVarianceOfLaplacianSharpness(gray);
    gettimeofday(&tv2,&tz2);
    std::cout << "Sharpness (variance of laplacian): " << blur << ", time eclipsed: " <<  getElapse(&tv1, &tv2) << " ms" << std::endl;

    return 0;
}

int main(int argc, char* argv[]) {

    if(argc != 2) {
        std::cout << "usage: test_face_quality <image>" << std::endl;
        exit(1); 
    }

    std::string imagepath = argv[1];

    test_picture(imagepath);
}