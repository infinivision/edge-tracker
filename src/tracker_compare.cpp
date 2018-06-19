#include <iostream>
#include <opencv2/opencv.hpp>
#include "mtcnn.h"
#include <string.h>
#include <cstdlib>
#include "tracker.hpp" // use optimised tracker instead of OpenCV version of KCF tracker
#include "staple_tracker.hpp"
#include "utils.h"
#include "face_attr.h"
#include "face_align.h"

#define QUIT_KEY 'q'

using namespace std;
using namespace cv;

/*
 * Get video capture from a camera index (0, 1) or an ip (192.168.1.15)
 */
VideoCapture getCaptureFromIndexOrIpOrFile(int source_type, const char *str) {
    if (source_type == 1) {
        // use camera index
        int camera_id = atoi(str);
        cout << "camera index: " << camera_id << endl;
        VideoCapture camera(camera_id);
        return camera;
    } else if(source_type == 2) {
        string camera_ip = str;
        cout << "camera ip: " << camera_ip << endl;
        string camera_stream = "rtsp://admin:Mcdonalds@" + camera_ip + ":554//Streaming/Channels/1";
        VideoCapture camera(camera_stream);
        return camera;
    } else {
        string file = str;
        VideoCapture camera(file);
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
void saveFace(Mat &frame, Rect2d &roi, long faceId, string outputFolder) {
    Mat image = frame(roi);
    string output = outputFolder + to_string(faceId) + ".jpg";
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
    if(argc != 4 && argc != 5 && argc != 6) {
        cout << "usage: main <model_path> <video source type> <camera_index/camera_ip/video_file> <frames>" << endl;
        exit(1); 
    }

    int source_type = atoi(argv[2]);
    if(source_type<1 || source_type>3) {
        cout << "video source type error!" << endl;
        exit(1);
    }

    if (argc >= 5) {
    	detectionFrameInterval = atoi(argv[4]);

        if (argc == 6) {
            outputFolder = argv[5];
        }
    }

    string model_path = argv[1];
    MTCNN mm(model_path);
    FaceAttr fa;
    fa.Load();
    FaceAlign align;

    VideoCapture camera = getCaptureFromIndexOrIpOrFile(source_type,argv[3]);
    if (!camera.isOpened()) {
        cerr << "failed to open camera" << endl;
        return;
    }

    int frameCounter = 0;
    long faceIdKCF = 0;
    long faceIdStaple = 0;
    struct timeval  tv1,tv2;
    struct timezone tz1,tz2;

    vector<Bbox> finalBbox;
    // Rect2d roi;
    vector<Ptr<Tracker>> trackersKCF;
    vector<STAPLE_TRACKER *>  trackersStaple;
    vector<Rect2d> boxesKCF;
    vector<Rect2d> boxesStaple;
    Mat frame;

    FileStorage fs;
    fs.open("kcf.yaml", FileStorage::READ);
    TrackerKCF::Params kcf_param;
    kcf_param.read(fs.root());
    fs.open("staple.yaml", FileStorage::READ);
    staple_cfg staple_cfg;
    staple_cfg.read(fs.root());

    namedWindow("face_detection", WINDOW_NORMAL);

    do {
        finalBbox.clear();
        camera >> frame;
        if (!frame.data) {
            cerr << "Capture video failed" << endl;
            if(source_type==3) exit(0);
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
                    auto box = *it;
                    std::vector<double> qualities = fa.GetQuality(cimg, box.x1, box.y1, box.x2, box.y2);
                    Rect2d detectedFace(Point(box.x1, box.y1),Point(box.x2, box.y2));

                    // test whether is a new face
                    bool newFace = true;
                    unsigned i=0;
                    for (i=0;i<boxesKCF.size();i++) {
                        Rect2d trackedFace = boxesKCF[i];
                        if (isSameFace(detectedFace, trackedFace)) {
                            newFace = false;
                            break;
                        }
                    }

                    if (newFace) {
                        // create a tracker if a new face is detected
                        Ptr<Tracker> tracker = TrackerKCF::create(kcf_param);
                        tracker->init(frame, detectedFace);
                        tracker->id = faceIdKCF;
                        trackersKCF.push_back(tracker);
                        boxesKCF.push_back(detectedFace);
                        cout << "frame " << frameCounter << ":KCF start tracking face #" << tracker->id << endl;

                        if (!outputFolder.empty()) {
                            // save face
                            saveFace(frame, detectedFace, faceIdKCF, outputFolder);
                        }
                        
                        faceIdKCF++;
                    } else {
                        // update tracker's bounding box
                        trackersKCF[i]->reset(frame, detectedFace);
                    }

                    // add staple
                    newFace = true;
                    i = 0;
                    for (i=0;i<boxesStaple.size();i++) {
                        Rect2d trackedFace = boxesStaple[i];
                        if (isSameFace(detectedFace, trackedFace)) {
                            newFace = false;
                            break;
                        }
                    }

                    if (newFace) {
                        // create a tracker if a new face is detected
                        STAPLE_TRACKER * tracker2 = new STAPLE_TRACKER(staple_cfg);
                        tracker2->tracker_staple_initialize(frame,detectedFace);
                        tracker2->tracker_staple_train(frame,true);
                        tracker2->id = faceIdStaple;
                        trackersStaple.push_back(tracker2);
                        boxesStaple.push_back(detectedFace);
                        cout << "frame " << frameCounter << ":Staple start tracking face #" << tracker2->id << endl;

                        if (!outputFolder.empty()) {
                            // save face
                            saveFace(frame, detectedFace, faceIdStaple, outputFolder);
                        }
                        
                        faceIdStaple++;
                    } else {
                        // update tracker's bounding box
                        STAPLE_TRACKER * tracker2 = trackersStaple[i];
                        long id_ = tracker2->id;
                        delete tracker2;
                        tracker2 = new STAPLE_TRACKER(staple_cfg);
                        tracker2->id = id_;
                        tracker2->tracker_staple_initialize(frame,detectedFace);
                        tracker2->tracker_staple_train(frame,true);
                        trackersStaple[i] = tracker2;
                    }
                }
            }

            cout << "frame " << frameCounter << ": detected " << total << " Persons. time eclipsed: " <<  getElapse(&tv1, &tv2) << " ms" << endl;

            // clean up trackers if the tracker doesn't follow a face
            for (unsigned i=0; i < trackersKCF.size(); i++) {
                Ptr<Tracker> tracker = trackersKCF[i];
                Rect2d trackedFace = boxesKCF[i];

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
                    cout << "frame " << frameCounter << ":KCF stop tracking face #" << tracker->id << endl;
                    trackersKCF.erase(trackersKCF.begin() + i);
                    boxesKCF.erase(boxesKCF.begin() + i);
                    i--;
                }
            }
            // add staple
            for (unsigned i=0; i < trackersStaple.size(); i++) {
                STAPLE_TRACKER *tracker2 = trackersStaple[i];
                Rect2d trackedFace = boxesStaple[i];

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
                    cout << "frame " << frameCounter << ":Staple stop tracking face #" << tracker2->id << endl;
                    delete tracker2;
                    trackersStaple.erase(trackersStaple.begin() + i);
                    boxesStaple.erase(boxesStaple.begin() + i);
                    i--;
                }
            }

        }

        // update trackers
        for (int i = 0; i < trackersKCF.size(); i++) {
            Ptr<Tracker> tracker = trackersKCF[i];

            bool tracked = tracker->update(frame, boxesKCF[i]);
            if (!tracked) {
                // delete tracker
                cout << "frame " << frameCounter << ": stop tracking face #" << tracker->id << endl;
                trackersKCF.erase(trackersKCF.begin() + i);
                boxesKCF.erase(boxesKCF.begin() + i);
                i--;
                continue;
            }

            Rect2d box = boxesKCF[i];
            // draw tracked face
            rectangle( frame, box, Scalar( 255, 0, 0 ), 1, 8 );
            // show face id
            Point middleHighPoint = Point(box.x+box.width/2-10, box.y);
            putText(frame, to_string(tracker->id), middleHighPoint, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
        }
        // add staple
        for (int i = 0; i < trackersStaple.size(); i++) {
            STAPLE_TRACKER *tracker2 = trackersStaple[i];

            boxesStaple[i] = tracker2->tracker_staple_update(frame);
            tracker2->tracker_staple_train(frame,false);
            Rect2d box = boxesStaple[i];
            // draw tracked face
            rectangle( frame, box, Scalar( 0, 255, 0 ), 1, 1 );
            // show face id
            Point middleHighPoint = Point(box.x+box.width/2+10, box.y);
            putText(frame, to_string(tracker2->id), middleHighPoint, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        }

        imshow("face_detection", frame);

        frameCounter++;

    } while (QUIT_KEY != waitKey(1));
}

int main(int argc, char* argv[]) {
    test_video(argc, argv);
}
