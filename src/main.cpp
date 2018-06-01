#include <iostream>
#include <opencv2/opencv.hpp>
//#include <opencv2/tracking.hpp>
#include "mtcnn.h"
#include <string.h>
#include "tracker.hpp" // use optimised tracker instead of OpenCV version of KCF tracker
#include "utils.h"

#define QUIT_KEY 'q'

cv::VideoCapture getCaptureFromIndexOrIp(const char *str) {
	 if (strcmp(str, "0") == 0 || strcmp(str, "1") == 0) {
        // use camera index
        int camera_id = atoi(str);
        std::cout << "camera index: " << camera_id << std::endl;
        cv::VideoCapture camera(camera_id);
		return camera;
    } else {
        std::string camera_ip = str;
        std::cout << "camera ip: " << camera_ip << std::endl;
        std::string camera_stream = "rtsp://admin:Mcdonalds@" + camera_ip + ":554//Streaming/Channels/1";
        cv::VideoCapture camera(camera_stream);
		return camera;
    }	
}

void getLastFrame(cv::VideoCapture& video, cv::Mat& frame) {

    //Get total number of frames in the video
    //Won't work on live video capture
    const int frames = video.get(CV_CAP_PROP_FRAME_COUNT);

    //Seek video to last frame
    video.set(CV_CAP_PROP_POS_FRAMES,frames-1);

    //Capture the last frame
    video>>frame;

    //Rewind video
    video.set(CV_CAP_PROP_POS_FRAMES,0);
}

void test_video(int argc, char* argv[]) {
	if(argc != 3) {
		std::cout << "usage: main $model_path $camera_ip" << std::endl;
		exit(1); 
	}

	std::string model_path = argv[1];
	MTCNN mm(model_path);

	cv::VideoCapture camera = getCaptureFromIndexOrIp(argv[2]);
    if (!camera.isOpened()) {
        std::cerr << "failed to open camera" << std::endl;
        return;
    }

    int counter = 0;
    struct timeval  tv1,tv2;
    struct timezone tz1,tz2;

	std::vector<Bbox> finalBbox;
    std::vector<cv::Ptr<cv::Tracker>> trackers;
	cv::Rect2d roi;
	cv::Mat frame;

	cv::namedWindow("face_detection", cv::WINDOW_NORMAL);

   	do {
		finalBbox.clear();
        //camera >> frame;
		getLastFrame(camera, frame);
        if (!frame.data) {
            std::cerr << "Capture video failed" << std::endl;
            continue;
        }

		if (counter % 25 == 0) {
			// renew trackers
			trackers.clear();

			gettimeofday(&tv1,&tz1);
            ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
            mm.detect(ncnn_img, finalBbox);
            gettimeofday(&tv2,&tz2);
            int total = 0;
            for(vector<Bbox>::iterator it=finalBbox.begin(); it!=finalBbox.end();it++) {
                if((*it).exist) {
                    total++;
					// draw rectangle
                    //cv::rectangle(frame, cv::Point((*it).x1, (*it).y1), cv::Point((*it).x2, (*it).y2), cv::Scalar(0,0,255), 2,8,0);
                    //for(int num=0;num<5;num++) {
						// draw 5 landmarks
                        //circle(frame, cv::Point((int)*(it->ppoint+num), (int)*(it->ppoint+num+5)), 3, cv::Scalar(0,255,255), -1);
                    //}
					
					// create tracker
					auto box = *it;
					cv::Rect2d roi(cv::Point(box.x1, box.y1),cv::Point(box.x2, box.y2));
	
					cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
					tracker->init(frame,roi);
					trackers.push_back(tracker);
                }
            }

            std::cout << "detected " << total << " Persons. time eclipsed: " <<  getElapse(&tv1, &tv2) << " ms" << std::endl;
		}

		for (auto it = trackers.begin(); it != trackers.end(); it++) {
			auto loss = (*it)->update(frame,roi);
        	if (!loss) {
            	std::cout << "Stop the tracking process" << std::endl;
            	// break;
 				continue;
        	}
			cv::rectangle( frame, roi, cv::Scalar( 255, 0, 0 ), 2, 1 );
        }

		imshow("face_detection", frame);

		counter++;

    } while (QUIT_KEY != cv::waitKey(1));
}

int main(int argc, char* argv[]) {
    test_video(argc, argv);
}
