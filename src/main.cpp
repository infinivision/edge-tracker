#include <iostream>
#include <opencv2/opencv.hpp>
//#include <opencv2/tracking.hpp>
#include "mtcnn.h"
#include <string.h>
#include <cstdlib>
#include "tracker.hpp" // use optimised tracker instead of OpenCV version of KCF tracker
#include "utils.h"
#include "face_attr.h"
#include "face_align.h"

#define QUIT_KEY 'q'

using namespace std;
using namespace cv;

FaceAlign faceAlign = FaceAlign();

/*
 * Get video capture from a camera index (0, 1) or an ip (192.168.1.15)
 */
VideoCapture getCaptureFromIndexOrIp(const char *str) {
    if (strcmp(str, "0") == 0 || strcmp(str, "1") == 0) {
        // use camera index
        int camera_id = atoi(str);
        cout << "camera index: " << camera_id << endl;
        VideoCapture camera(camera_id);
        return camera;
    } else {
        string camera_ip = str;
        cout << "camera ip: " << camera_ip << endl;
        string camera_stream = "rtsp://admin:mdc12345678@" + camera_ip + ":554//Streaming/Channels/1";
        VideoCapture camera(camera_stream);
        return camera;
    }
}

/*
 * Decide whether the detected face is same as the tracking one
 * 
 * return true when:
 *   center point of one box is inside the other
 */
bool isSameFace(Rect2d &box1, Rect2d &box2) {
    int x1 = box1.x + box1.width/2;
    int y1 = box1.y + box1.height/2;
    int x2 = box2.x + box2.width/2;
    int y2 = box2.y + box2.height/2;

    if ( x1 > box2.x && x1 < box2.x + box2.width &&
         y1 > box2.y && y1 < box2.y + box2.height &&
         x2 > box1.x && x2 < box1.x + box1.width &&
         y2 > box1.y && y2 < box1.y + box1.height ) {
        return true;
    }

    return false;
}

/*
 * write face to the output folder
 */
void saveFace(Mat &frame, Bbox &box, long faceId, string outputFolder) {

    Rect2d roi(Point(box.x1, box.y1),Point(box.x2, box.y2));
    Mat cropped(frame, roi);
    string output = outputFolder + "/" + to_string(faceId) + ".jpg";
    imwrite(output, cropped);

    std::vector<cv::Point2f> points;
    for(int num=0;num<5;num++) {
        Point2f point(box.ppoint[num], box.ppoint[num+5]);
        points.push_back(point);
    }

    Mat image = faceAlign.Align(frame, points);

    if (image.empty()) {
        /* empty image means unable to align face */
        return;
    }

    output = outputFolder + "/" + to_string(faceId) + "-align.jpg";
    if ( imwrite(output, image) ) {
        cout << "\tsave face #" << faceId << " to " << output << endl;
    } else {
        cout << "\tfail to save face #" << faceId << endl;
    }
}

void test_video(int argc, char* argv[]) {
    
    int detectionFrameInterval = 25; // nb of frames
    string outputFolder; // folder to save face

    // parsing arguments
    if(argc != 3 && argc != 4 && argc != 5) {
        cout << "usage: main <model_path> <camera_ip> <frames>" << endl;
        exit(1); 
    }

    if (argc >= 4) {
    	detectionFrameInterval = atoi(argv[3]);

        if (argc == 5) {
            outputFolder = argv[4];
        }
    }

    string model_path = argv[1];
    MTCNN mm(model_path);
    FaceAttr fa;
    fa.Load();
    FaceAlign align;

    VideoCapture camera = getCaptureFromIndexOrIp(argv[2]);
    if (!camera.isOpened()) {
        cerr << "failed to open camera" << endl;
        return;
    }

    int frameCounter = 0;
    long faceId = 0;
    struct timeval  tv1,tv2;
    struct timezone tz1,tz2;

    vector<Bbox> finalBbox;
    Rect2d roi;
    vector<Ptr<Tracker>> trackers;
    vector<Rect2d> boxes;
    Mat frame;

    FileStorage fs;
    fs.open("kcf.yaml", FileStorage::READ);
    TrackerKCF::Params kcf_param;
    kcf_param.read(fs.root());

    namedWindow("face_detection", WINDOW_NORMAL);
    // resizeWindow("face_detection", 800, 600);

    do {
        finalBbox.clear();
        camera >> frame;
        if (!frame.data) {
            cerr << "Capture video failed" << endl;
            continue;
        }
        dlib::cv_image<dlib::bgr_pixel> cimg(frame);

        if (frameCounter % detectionFrameInterval == 0) {
            // start face detection
            gettimeofday(&tv1,&tz1);
            ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
            mm.detect(ncnn_img, finalBbox);
            gettimeofday(&tv2,&tz2);
            int total = 0;

            // update trackers' bounding boxes and create tracker for a new face
            for(vector<Bbox>::iterator it=finalBbox.begin(); it!=finalBbox.end();it++) {
                if((*it).exist) {
                    total++;

                    // get face bounding box
                    Bbox box = *it;
                    std::vector<double> qualities = fa.GetQuality(cimg, box.x1, box.y1, box.x2, box.y2);
                    Rect2d detectedFace(Point(box.x1, box.y1),Point(box.x2, box.y2));

                    // test whether is a new face
                    bool newFace = true;
                    unsigned i;
                    for (i=0;i<boxes.size();i++) {
                        Rect2d trackedFace = boxes[i];
                        if (isSameFace(detectedFace, trackedFace)) {
                            newFace = false;
                            break;
                        }
                    }

                    if (newFace) {
                        // create a tracker if a new face is detected
                        Ptr<Tracker> tracker = TrackerKCF::create(kcf_param);
                        tracker->init(frame, detectedFace);
                        tracker->id = faceId;
                        trackers.push_back(tracker);
                        boxes.push_back(detectedFace);
                        cout << "frame " << frameCounter << ": start tracking face #" << tracker->id << endl;

                        if (!outputFolder.empty()) {
                            // save face
                            saveFace(frame, box, faceId, outputFolder);
                        }
                        
                        faceId++;
                    } else {
                        // update tracker's bounding box
                        trackers[i]->reset(frame, detectedFace);
                    }
                }
            }

            cout << "frame " << frameCounter << ": detected " << total << " Persons. time eclipsed: " <<  getElapse(&tv1, &tv2) << " ms" << endl;

            // clean up trackers if the tracker doesn't follow a face
            for (unsigned i=0; i < trackers.size(); i++) {
                Ptr<Tracker> tracker = trackers[i];
                Rect2d trackedFace = boxes[i];

                bool isFace = false;
                for (vector<Bbox>::iterator it=finalBbox.begin(); it!=finalBbox.end();it++) {
                    Bbox box = *it;
                    if ((*it).exist) {
                        Bbox box = *it;
                        Rect2d detectedFace(Point(box.x1, box.y1),Point(box.x2, box.y2));
                        if (isSameFace(detectedFace, trackedFace)) {
                            isFace = true;
                            break;
                        }
                    } 
                }

                if (!isFace) {
                    /* clean up tracker */
                    cout << "frame " << frameCounter << ": stop tracking face #" << tracker->id << endl;
                    trackers.erase(trackers.begin() + i);
                    boxes.erase(boxes.begin() + i);
                    i--;
                }
            }
        }

        // update trackers
        for (int i = 0; i < trackers.size(); i++) {
            Ptr<Tracker> tracker = trackers[i];

            bool tracked = tracker->update(frame, boxes[i]);
            if (!tracked) {
                // delete tracker
                cout << "frame " << frameCounter << ": stop tracking face #" << tracker->id << endl;
                trackers.erase(trackers.begin() + i);
                boxes.erase(boxes.begin() + i);
                i--;
                continue;
            }

            Rect2d box = boxes[i];
            // draw tracked face
            rectangle( frame, box, Scalar( 255, 0, 0 ), 2, 1 );
            // show face id
            Point middleHighPoint = Point(box.x+box.width/2, box.y);
            putText(frame, to_string(tracker->id), middleHighPoint, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        }

        imshow("face_detection", frame);

        frameCounter++;

    } while (QUIT_KEY != waitKey(1));
}

int main(int argc, char* argv[]) {
    test_video(argc, argv);
}
