#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include "utils.h"
#include "mtcnn.h"

#define QUIT_KEY 'q'

void test_video(int argc, char* argv[]) {
	if(argc != 3) {
		std::cout << "usage: test_video $model_path $camera_ip" << std::endl;
		exit(1); 
	}

	std::string model_path = argv[1];
	//int camera_id = atoi(argv[2]);
    std::string camera_ip = argv[2];
    std::string camera_stream = "rtsp://admin:Mcdonalds@" + camera_ip + ":554//Streaming/Channels/1";
	MTCNN mm(model_path);

    int counter = 0;
    struct timeval  tv1,tv2;
    struct timezone tz1,tz2;

    //cv::VideoCapture camera(camera_id);
    cv::VideoCapture camera(camera_stream);

    if (!camera.isOpened()) {
        std::cerr << "failed to open camera" << std::endl;
        return;
    }

	std::vector<Bbox> finalBbox;
    std::vector<cv::Ptr<cv::Tracker>> trackers;
	cv::Rect2d roi;
	cv::Mat frame;

   	do {
		finalBbox.clear();
        camera >> frame;
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
