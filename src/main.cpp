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
bool overlap(Rect2d &box1, Rect2d &box2) {
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
void saveFace(const Mat &frame, const Bbox &box, long faceId, string outputFolder) {

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

// void test_video(const string model_path, const CameraConfig &camera, int detectionFrameInterval, string outputFolder) {
void test_video(const string model_path, const CameraConfig &camera, string outputFolder) {
    
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

    vector<Bbox> detected_bounding_boxes;
    Rect2d roi;
    vector<Ptr<Tracker>> trackers;
    vector<Rect2d> tracker_boxes;
    // selected_faces[i] on frame[i] is a selected face
    vector<Mat> selected_frames;
    vector<Bbox> selected_faces;
    vector<double> scores; // scores[i] is the face score of selected_faces[i]
    Mat frame;

    FileStorage fs;
    fs.open("kcf.yaml", FileStorage::READ);
    TrackerKCF::Params kcf_param;
    kcf_param.read(fs.root());

    namedWindow("face_detection", WINDOW_NORMAL);

    do {
        detected_bounding_boxes.clear();
        cap >> frame;
        if (!frame.data) {
            cerr << "Capture video failed" << endl;
            continue;
        }

        // dlib::cv_image<dlib::bgr_pixel> cimg(frame);

        // update trackers
        for (int i = 0; i < trackers.size(); i++) {
            trackers[i]->update(frame, tracker_boxes[i]);
        }

        // if (frameCounter % detectionFrameInterval == 0) {
        {
            // start face detection
            auto timenow = chrono::system_clock::to_time_t(chrono::system_clock::now());
            cout << "frame #" << frameCounter << ", " << ctime(&timenow);
            Mat small_frame;
            bool resized = false;
            float resize_factor_x, resize_factor_y = 1;

            if (frame.rows > 480) {
                // resize to 480p
                gettimeofday(&tv1,&tz1);
                resize(frame, small_frame, Size(848, 480), 0, 0, INTER_NEAREST);
                gettimeofday(&tv2,&tz2);
                cout << "\tresize to 480p, time eclipsed: " << getElapse(&tv1, &tv2) << " ms" << endl;
                resized = true;
                resize_factor_x = frame.cols / 848.0;
                resize_factor_y = frame.rows / 480.0;
            } else {
                small_frame = frame;
            }

            ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(small_frame.data, ncnn::Mat::PIXEL_BGR2RGB, small_frame.cols, small_frame.rows);

            gettimeofday(&tv1,&tz1);
            mm.detect(ncnn_img, detected_bounding_boxes);
            gettimeofday(&tv2,&tz2);
            int total = 0;

            // update trackers' bounding boxes and create tracker for a new face
            for(vector<Bbox>::iterator it=detected_bounding_boxes.begin(); it!=detected_bounding_boxes.end();it++) {
                if((*it).exist) {
                    total++;

                    // get face bounding box
                    Bbox box = *it;
                    if (resized) {
                        scaleBox(box, resize_factor_x, resize_factor_y);
                    }

                    //std::vector<double> qualities = fa.GetQuality(cimg, box.x1, box.y1, box.x2, box.y2);
                    Rect2d detected_face(Point(box.x1, box.y1),Point(box.x2, box.y2));

                    // test whether is a new face
                    bool newFace = true;
                    unsigned i;
                    for (i=0;i<tracker_boxes.size();i++) {
                        if (overlap(detected_face, tracker_boxes[i])) {
                            newFace = false;
                            break;
                        }
                    }

                    if (newFace) {
                        // create a new tracker if a new face is detected
                        Ptr<Tracker> tracker = TrackerKCF::create(kcf_param);
                        tracker->init(frame, detected_face);
                        tracker->id = faceId;
                        trackers.push_back(tracker);
                        tracker_boxes.push_back(detected_face);
                        selected_faces.push_back(box);
                        selected_frames.push_back(frame);
                        // calculate score of the selected face
                        Mat face(frame, detected_face);
                        double score = fa.GetVarianceOfLaplacianSharpness(face);
                        scores.push_back(score);
                        cout << "frame " << frameCounter << ": start tracking face #" << tracker->id << endl;

                        // save face now when tracker is lost
                        // saveFace(frame, box, faceId, outputFolder);
                        
                        faceId++;
                    } else {
                        // update tracker's bounding box
                        trackers[i]->reset(frame, detected_face);
                        tracker_boxes[i] = detected_face;
                    }
                }
            }

            cout << "\tdetected " << total << " Persons. time eclipsed: " <<  getElapse(&tv1, &tv2) << " ms" << endl;

            // clean up trackers if the tracker doesn't follow a face
            for (unsigned i=0; i < trackers.size(); i++) {
                Ptr<Tracker> tracker = trackers[i];
                Rect2d tracker_box = tracker_boxes[i];

                bool isFace = false;
                for (vector<Bbox>::iterator it=detected_bounding_boxes.begin(); it!=detected_bounding_boxes.end();it++) {
                    Bbox box = *it;
                    if ((*it).exist) {
                        Bbox box = *it;
                        Rect2d detected_face(Point(box.x1, box.y1),Point(box.x2, box.y2));
                        if (overlap(detected_face, tracker_boxes[i])) {
                            isFace = true;
                            // update face score
                            Mat face(frame, tracker_boxes[i]);
                            double score = fa.GetVarianceOfLaplacianSharpness(face);
                            if (score > scores[i]) {
                                // select a better face
                                selected_frames[i] = frame;
                                selected_faces[i] = box;
                                scores[i] = score;
                            }
                            break;
                        }
                    } 
                }

                if (!isFace) {
                    /* clean up tracker */
                    cout << "frame " << frameCounter << ": stop tracking face #" << tracker->id << endl;
                    saveFace(selected_frames[i], selected_faces[i], tracker->id, outputFolder);

                    trackers.erase(trackers.begin() + i);
                    tracker_boxes.erase(tracker_boxes.begin() + i);
                    selected_faces.erase(selected_faces.begin() + i);
                    selected_frames.erase(selected_frames.begin() + i);
                    scores.erase(scores.begin() + i);
                    i--;
                }

                // draw tracked face
                rectangle( frame, tracker_box, Scalar( 255, 0, 0 ), 2, 1 );
                // show face id
                Point middleHighPoint = Point(tracker_box.x+tracker_box.width/2, tracker_box.y);
                putText(frame, to_string(tracker->id), middleHighPoint, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
            }
        }

        imshow("face_detection", frame);

        frameCounter++;

    } while (QUIT_KEY != waitKey(1));
}

int main(int argc, char* argv[]) {

    string config_path = "config.toml";
    string model_path = "models/ncnn";
    string output_folder = "~/Pictures/faces/";
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
    test_video(model_path, camera, output_folder);
}
