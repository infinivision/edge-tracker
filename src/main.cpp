#include <iostream>
#include <opencv2/opencv.hpp>
//#include <opencv2/tracking.hpp>
#include "camera.h"
#include "mtcnn.h"
#include <string.h>
#include <chrono>
#include <cstdlib>
#include "tracker.hpp" // use optimised tracker instead of OpenCV version of KCF tracker
#include "utils.h"
#include <face_attr.h>
#include <face_align.h>
#include <image_quality.h>
#include <glog/logging.h>
#include <thread>

#define QUIT_KEY 'q'

using namespace std;
using namespace std::chrono_literals;
using namespace cv;

FaceAlign faceAlign = FaceAlign();

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

    string current_time = get_current_time();

    Rect2d roi(Point(box.x1, box.y1),Point(box.x2, box.y2));
    Mat cropped(frame, roi);
    string output = outputFolder + "/original/" + current_time + ".jpg";
    if ( imwrite(output, cropped) ) {
        LOG(INFO) << "\tsave face #" << faceId << " to " << output;
        cout << "save face #" << faceId << " to " << output << endl;
    } else {
        LOG(ERROR) << "\tfail to save face #" << faceId << " to " << output;
    }

    //namedWindow("output", WINDOW_NORMAL);

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

    output = outputFolder + "/" + current_time + ".jpg";
    if ( imwrite(output, image) ) {
        LOG(INFO) << "\tsave face #" << faceId << " to " << output;
        cout << "save face #" << faceId << " to " << output << endl;
        LOG(INFO) << "\tmtcnn score: " << box.score;
    } else {
        LOG(ERROR) << "\tfail to save face #" << faceId << " to " << output;
    }

    output = outputFolder + "/aligned/" + current_time + ".jpg";
    imwrite(output, image);

    //imshow("output", image);
}

// prepare (clean output folder), output_folder argument will be changed!
void prepare_output_folder(const CameraConfig &camera, string &output_folder) {
    output_folder += "/" + camera.identity();

    string cmd = "mkdir -p " + output_folder + "/original";
    int dir_err = system(cmd.c_str());
    if (-1 == dir_err) {
        LOG(ERROR) << "Error creating directory";
        exit(1);
    }

    cmd = "mkdir -p " + output_folder + "/aligned";
    dir_err = system(cmd.c_str());
    if (-1 == dir_err) {
        LOG(ERROR) << "Error creating directory";
        exit(1);
    }

    // cmd = "rm -f " + output_folder + "/*";
    // system(cmd.c_str());
    // cmd = "rm -f " + output_folder + "/original/*";
    // system(cmd.c_str());
}

void process_camera(const string &model_path, const CameraConfig &camera, string output_folder, const FaceAttr &fa) {

    cout << "processing camera: " << camera.identity() << endl;

    prepare_output_folder(camera, output_folder);

    VideoCapture cap = camera.GetCapture();
    if (!cap.isOpened()) {
        LOG(ERROR) << "failed to open camera: " << camera.identity();
        return;
    }

    int frameCounter = 0;
    long faceId = 0;
    struct timeval  tv1,tv2;
    struct timezone tz1,tz2;

    MTCNN mm(model_path);
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

    // namedWindow("window", WINDOW_NORMAL);

    do {
        detected_bounding_boxes.clear();
        bool enable_detection = false;
        cap >> frame;
        if (!frame.data) {
            LOG(ERROR) << "Capture video failed: " << camera.identity() << ", opened: " << cap.isOpened();
            cap.release();

            LOG(ERROR) << "sleep for 5 seconds ...";
            std::this_thread::sleep_for(5s);

            cap = camera.GetCapture();
            if (!cap.isOpened()) {
                LOG(ERROR) << "failed to open camera: " << camera.identity();
                return;
            }
            LOG(INFO) << "reopen camera: " << camera.identity();
            continue;
        }

        Mat show_frame = frame.clone();

        string log = "frame #" + to_string(frameCounter) + ", tracking faces: ";
        // update trackers
        for (int i = 0; i < trackers.size(); i++) {
            trackers[i]->update(frame, tracker_boxes[i]);
            log += "#" + to_string(trackers[i]->id) + " ";
        }

        if (frameCounter % camera.detection_period == 0)
        {
            LOG(INFO) << log;
            enable_detection = true;
            Mat small_frame;
            bool resized = false;
            float resize_factor_x, resize_factor_y = 1;

            if ( camera.resize_rows > 0 && frame.rows > camera.resize_rows) {
                // resize
                gettimeofday(&tv1,&tz1);
                resize(frame, small_frame, Size(camera.resize_cols, camera.resize_rows), 0, 0, INTER_NEAREST);
                gettimeofday(&tv2,&tz2);
                LOG(INFO) << "\tresize to (" << camera.resize_cols << " * " << camera.resize_rows << "), time eclipsed: " << getElapse(&tv1, &tv2) << " ms";
                resized = true;
                resize_factor_x = (float)frame.cols / camera.resize_cols;
                resize_factor_y = (float)frame.rows / camera.resize_rows;
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
                        box.scale(resize_factor_x, resize_factor_y);
                        *it = box;
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
                        Mat cloned_frame = frame.clone();
                        selected_frames.push_back(cloned_frame);
                        // calculate score of the selected face
                        Mat face(frame, detected_face);
                        double score = GetVarianceOfLaplacianSharpness(face);
                        scores.push_back(score);
                        LOG(INFO) << "\tstart tracking face #" << tracker->id << ", score: " << score;
                        
                        faceId++;
                    } else {
                        // update tracker's bounding box
                        trackers[i]->reset(frame, detected_face);
                        tracker_boxes[i] = detected_face;
                    }
                }
            }

            LOG(INFO) << "\tdetected " << total << " Persons. time eclipsed: " <<  getElapse(&tv1, &tv2) << " ms";
        }

        // clean up trackers if the tracker doesn't follow a face
        for (unsigned i=0; i < trackers.size(); i++) {
            Ptr<Tracker> tracker = trackers[i];
            Rect2d tracker_box = tracker_boxes[i];

            if (enable_detection)
            {
                bool isFace = false;
                for (vector<Bbox>::iterator it=detected_bounding_boxes.begin(); it!=detected_bounding_boxes.end();it++) {
                    if ((*it).exist) {
                        Bbox box = *it;

                        Rect2d detected_face(Point(box.x1, box.y1),Point(box.x2, box.y2));
                        if (overlap(detected_face, tracker_boxes[i])) {
                            isFace = true;
                            // update face score
                            Mat face(frame, tracker_boxes[i]);
                            double score = GetVarianceOfLaplacianSharpness(face);
                            if (score > scores[i]) {
                                // select a better face
                                LOG(INFO) << "\tupdate selected face, new score: " << score;
                                Mat cloned_frame = frame.clone();
                                selected_frames[i] = cloned_frame;
                                selected_faces[i] = box;
                                scores[i] = score;
                            }
                            break;
                        }
                    } 
                }

                if (!isFace) {
                    /* clean up tracker */
                    LOG(INFO) << "\tstop tracking face #" << tracker->id << ", final score: " << scores[i];
                    saveFace(selected_frames[i], selected_faces[i], tracker->id, output_folder);

                    trackers.erase(trackers.begin() + i);
                    tracker_boxes.erase(tracker_boxes.begin() + i);
                    selected_faces.erase(selected_faces.begin() + i);
                    selected_frames.erase(selected_frames.begin() + i);
                    scores.erase(scores.begin() + i);
                    i--;
                    continue;
                }
            }

            // draw tracked face
            // rectangle( show_frame, tracker_box, Scalar( 255, 0, 0 ), 2, 1 );
            // show face id
            // Point middleHighPoint = Point(tracker_box.x+tracker_box.width/2, tracker_box.y);
            // putText(show_frame, to_string(tracker->id), middleHighPoint, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        }

        // imshow("window", show_frame);

        frameCounter++;

        google::FlushLogFiles(google::GLOG_INFO);

    } while (true);
    // } while (QUIT_KEY != waitKey(1));
}

int main(int argc, char* argv[]) {

    google::InitGoogleLogging(argv[0]);

    const String keys =
        "{help h usage ? |                         | print this message   }"
        "{model        |models/ncnn                | path to mtcnn model  }"
        "{config       |config.toml                | camera config        }"
        "{output       |/Users/moon/Pictures/faces | output folder        }"
    ;

    CommandLineParser parser(argc, argv, keys);
    parser.about("camera face detector");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    String config_path = parser.get<String>("config");
    LOG(INFO) << "config path: " << config_path;

    String model_path = parser.get<String>("model");
    String output_folder = parser.get<String>("output");
    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    FaceAttr fa;
    fa.Load();
    vector<CameraConfig> cameras = LoadCameraConfig(config_path);

    // LOG(INFO) << "detection period: " << camera.detection_period;

    for (CameraConfig camera: cameras) {
        // start processing video
        thread t {process_camera, model_path, camera, output_folder, fa};
        t.detach();
    }

    while(true) {
        std::this_thread::sleep_for(chrono::seconds(1));
    }

}
