#include <iostream>
#include <opencv2/opencv.hpp>
//#include <opencv2/tracking.hpp>
#include "camera.h"
#include "cpptoml.h"
#include "mtcnn.h"
#include <string.h>
#include <chrono>
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
VideoCapture getCaptureFromIndexOrIp(const CameraConfig &camera) {
    if ( camera.ip.empty()) {
        // use camera index
        cout << "camera index: " << camera.index << endl;
        VideoCapture capture(camera.index);
        return capture;
    } else {
        cout << "camera ip: " << camera.ip << endl;
        string camera_stream = "rtsp://" + camera.username +  ":" + camera.password + "@" + camera.ip + ":554//Streaming/Channels/1";
        VideoCapture capture(camera_stream);
        return capture;
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
    string output = outputFolder + "/original/" + to_string(faceId) + ".jpg";
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

    output = outputFolder + "/" + to_string(faceId) + ".jpg";
    if ( imwrite(output, image) ) {
        cout << "\tsave face #" << faceId << " to " << output << endl;
    } else {
        cout << "\tfail to save face #" << faceId << endl;
    }
}

/*
 * scale bounding box for resized frames
 */
void scaleBox(Bbox &box, float factor_x, float factor_y) {
    box.x1 = round(box.x1 * factor_x);
    box.x2 = round(box.x2 * factor_x);
    box.y1 = round(box.y1 * factor_y);
    box.y2 = round(box.y2 * factor_y);

    for (int i=0; i<5; i++) {
        box.ppoint[i] *= factor_x;
        box.ppoint[i+5] *= factor_y;
    }
}

void test_video(const string model_path, const CameraConfig &camera, int detectionFrameInterval, string outputFolder) {
    
    MTCNN mm(model_path);
    FaceAttr fa;
    fa.Load();
    FaceAlign align;

    VideoCapture cap = getCaptureFromIndexOrIp(camera);
    if (!cap.isOpened()) {
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
        cap >> frame;
        if (!frame.data) {
            cerr << "Capture video failed" << endl;
            continue;
        }

        dlib::cv_image<dlib::bgr_pixel> cimg(frame);

        if (frameCounter % detectionFrameInterval == 0) {
            // start face detection
            auto timenow = chrono::system_clock::to_time_t(chrono::system_clock::now());
            cout << "frame #" << frameCounter << ", " << ctime(&timenow);
            Mat resized_image;
            bool resized = false;
            float resize_factor_x, resize_factor_y = 1;

            if (frame.cols > 1280) {
                // resize to 720p
                gettimeofday(&tv1,&tz1);
                resize(frame, resized_image, Size(1280, 720), 0, 0, INTER_NEAREST);
                gettimeofday(&tv2,&tz2);
                cout << "\tresize to 720p, time eclipsed: " << getElapse(&tv1, &tv2) << " ms" << endl;
                resized = true;
                resize_factor_x = frame.cols / 1280.0;
                resize_factor_y = frame.rows / 720.0;
            } else {
                resized_image = frame;
            }

            //ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
            ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(resized_image.data, ncnn::Mat::PIXEL_BGR2RGB, resized_image.cols, resized_image.rows);
            //ncnn::Mat ncnn_imag_resized;
            //bool resized = false;
            //float resize_factor_x, resize_factor_y = 1;
/*
            if (frame.cols > 1280) {
                // resize to 720p
                gettimeofday(&tv1,&tz1);
                ncnn::resize_bilinear(ncnn_img, ncnn_img_resized, 1280, 720);
                gettimeofday(&tv2,&tz2);
                cout << "\tresize to 720p, time eclipsed: " << getElapse(&tv1, &tv2) << " ms" << endl;
                resized = true;
                resize_factor_x = frame.cols / 1280.0;
                resize_factor_y = frame.rows / 720.0;
            } else {
                ncnn_img_resized = ncnn_img;
            }
*/

            gettimeofday(&tv1,&tz1);
            mm.detect(ncnn_img, finalBbox);
            gettimeofday(&tv2,&tz2);
            int total = 0;

            // update trackers' bounding boxes and create tracker for a new face
            for(vector<Bbox>::iterator it=finalBbox.begin(); it!=finalBbox.end();it++) {
                if((*it).exist) {
                    total++;

                    // get face bounding box
                    Bbox box = *it;
                    if (resized) {
                        scaleBox(box, resize_factor_x, resize_factor_y);
                    }

                    //std::vector<double> qualities = fa.GetQuality(cimg, box.x1, box.y1, box.x2, box.y2);
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

            cout << "\tdetected " << total << " Persons. time eclipsed: " <<  getElapse(&tv1, &tv2) << " ms" << endl;

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

    string config_path = "config.toml";
    string model_path = "models/ncnn";
    string output_folder = "~/Pictures/faces/";
    int detection_interval = 50;
    CameraConfig camera;

    int res;

    while ((res = getopt(argc,argv,"c:m:o:h")) != -1) {
        switch (res) {
            case 'c':
                config_path = std::string(optarg);
                break;
            case 'm':
                model_path = std::string(optarg);
                break;
            case 'o':
                output_folder = std::string(optarg);
                break;
            case 'h':
                cout << "usage: main -m <model_path> -c <config_file> -o <face_folder>" << endl;
                exit(1);
            default:
                break;
        }
    }


    try {
        std::shared_ptr<cpptoml::table> g = cpptoml::parse_file(config_path);

        auto detection_interval_ptr = g->get_qualified_as<int>("global.detection_interval");
        detection_interval = *detection_interval_ptr;

        auto array = g->get_table_array("camera")->get();
        auto camera_ptr = array[0];
        auto index_ptr = camera_ptr->get_as<int>("index");
        auto ip_ptr = camera_ptr->get_as<std::string>("ip");
        auto username_ptr = camera_ptr->get_as<std::string>("username");
        auto password_ptr = camera_ptr->get_as<std::string>("password");

        if (ip_ptr) {
            camera = CameraConfig(*ip_ptr, *username_ptr, *password_ptr);
        } else {
            camera = CameraConfig(*index_ptr);
        }
    }
    catch (const cpptoml::parse_exception& e) {
        std::cerr << "Failed to parse " << "config.toml" << ": " << e.what() << std::endl;
        return 1;
    }

    output_folder += "/" + camera.identity();

    string cmd = "mkdir -p " + output_folder + "/original";
    const int dir_err = system(cmd.c_str());
    if (-1 == dir_err) {
        printf("Error creating directory!n");
        exit(1);
    }

    cmd = "rm -f " + output_folder + "/*";
    system(cmd.c_str());
    cmd = "rm -f " + output_folder + "/original/*";
    system(cmd.c_str());

    // start processing video
    test_video(model_path, camera, detection_interval, output_folder);
}
