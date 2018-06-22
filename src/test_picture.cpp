#include <iostream>
#include <opencv2/opencv.hpp>
#include "face_align.h"
#include "mtcnn.h"
#include "utils.h"

using namespace std;

int test_picture(int argc, char** argv) {
	if(argc != 4) {
		std::cout << "usage: test_picture <model_path> <image> <output_folder>" << std::endl;
		exit(1); 
	}

    std::string model_path = argv[1];
    std::string imagepath = argv[2];
    std::string output_folder = argv[3];

    cv::Mat cv_img = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    if (cv_img.empty())
    {
        std::cerr << "cv::Imread failed. File Path: " << imagepath << std::endl;
        return -1;
    }
    cv::Mat image = cv_img.clone();

    cv::namedWindow("face_detection", cv::WINDOW_AUTOSIZE);
    std::vector<Bbox> finalBbox;
    MTCNN mm(model_path);

    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(cv_img.data, ncnn::Mat::PIXEL_BGR2RGB, cv_img.cols, cv_img.rows);
    struct timeval  tv1,tv2;
    struct timezone tz1,tz2;

    gettimeofday(&tv1,&tz1);
    // detect faces
    mm.detect(ncnn_img, finalBbox);
    gettimeofday(&tv2,&tz2);

    FaceAlign faceAlign = FaceAlign();

    int total = 0;
    for(vector<Bbox>::iterator it=finalBbox.begin(); it!=finalBbox.end();it++) {
    	if((*it).exist) {
            total++;
            cout << "detected face #" << total << endl;
            Bbox box = *it;
            std::vector<cv::Point2f> points;
            Rect2d roi(Point(box.x1, box.y1),Point(box.x2, box.y2));

            cv::rectangle(image, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), cv::Scalar(0,0,255), 2,8,0);
            for(int num=0;num<5;num++) {
                cv::Point2f point(box.ppoint[num], box.ppoint[num+5]);
                points.push_back(point);
                circle(image, point, 3, cv::Scalar(0,255,255), -1);
            }

            // imshow("face_detection", image);
            // cv::waitKey(0);

            cv::Mat aligned = faceAlign.Align(cv_img, points);

            if (!aligned.empty()) {
                std::string outpath_origin = output_folder + "/" + std::to_string(total) + ".jpg";
                std::string outpath = output_folder + "/" + std::to_string(total) + "-align.jpg";
                Mat cropped(cv_img, roi);
                cv::imwrite(outpath_origin, cropped);
                cout << "\twrite original image to " << outpath_origin << endl;
                cv::imwrite(outpath, aligned);
                cout << "\twrite aligned image to " << outpath << endl;
            }
        }
    }

    std::cout << "detected " << total << " Persons. time eclipsed: " <<  getElapse(&tv1, &tv2) << " ms" << std::endl;

    return 0;
}

int main(int argc, char* argv[]) {
    test_picture(argc, argv);
}
