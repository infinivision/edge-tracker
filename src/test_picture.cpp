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

    std::cout << "reading image ..." << std::endl;
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

    std::cout << "convert image to ncnn Mat ..." << std::endl;
    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(cv_img.data, ncnn::Mat::PIXEL_BGR2RGB, cv_img.cols, cv_img.rows);
    struct timeval  tv1,tv2;
    struct timezone tz1,tz2;

    gettimeofday(&tv1,&tz1);
    // detect faces
    cout << "detecting faces ...";
    mm.detect(ncnn_img, finalBbox);
    cout << " end" << endl;
    gettimeofday(&tv2,&tz2);

    auto faceAlign = FaceAlign();

    int total = 0;
    cout << "begin loop，size: " << finalBbox.size() << endl;
    for(vector<Bbox>::iterator it=finalBbox.begin(); it!=finalBbox.end();it++) {
    	if((*it).exist) {
            total++;
            cout << total << endl;
            Bbox box = *it;
            cv::Rect2d roi(cv::Point2d(box.x1, box.y1), cv::Point2d(box.x2, box.y2));
            cv::Mat cropped(cv_img, roi);
            std::vector<cv::Point2d> points;

            cout << "after initialization" << endl;

            cv::rectangle(image, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), cv::Scalar(0,0,255), 2,8,0);
            for(int num=0;num<5;num++) {
                cv::Point2d point((int)box.ppoint[num], (int)box.ppoint[num+5]);
                points.push_back(point);
                circle(image, point, 3, cv::Scalar(0,255,255), -1);
            }

            imshow("face_detection", image);
            cv::waitKey(0);

            cout << "before align, points: " << points.size() << endl;
            cv::Mat aligned = faceAlign.Align(cropped, points);
            cout << "end align" << endl;

            std::string outpath = output_folder + "/" + std::to_string(total) + ".jpg";
            cv::imwrite(outpath, aligned);

        }
    }

    std::cout << "detected " << total << " Persons. time eclipsed: " <<  getElapse(&tv1, &tv2) << " ms" << std::endl;

    return 0;
}

int main(int argc, char* argv[]) {
    test_picture(argc, argv);
}
