#include "camera.h"
#include <chrono>
#include <cstdlib>
#include <face_attr.h>
#include <glog/logging.h>
#include <image_quality.h>
#include <iostream>
#include <kcf/tracker.hpp>
#include "mtcnn.h"
#include <opencv2/opencv.hpp>
#include <string.h>
#include <thread>
#include "utils.h"

#define QUIT_KEY 'q'

using namespace std;
using namespace cv;

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
            std::this_thread::sleep_for(std::chrono::seconds(5));

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

            ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);

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
                            // Mat face(frame, tracker_boxes[i]);
                            Mat face(frame, detected_face);
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
        "{config       |/opt/dev_keeper/keeper.toml| camera config        }"
        "{output       |/opt/dev_keeper/faces      | output folder        }"
    ;

    CommandLineParser parser(argc, argv, keys);
    parser.about("camera face detector");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    String config_path = parser.get<String>("config");
    cout << "config path: " << config_path << endl;

    String model_path = parser.get<String>("model");
    String output_folder = parser.get<String>("output");
    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    FaceAttr fa;
    fa.Load();
    vector<CameraConfig> cameras = LoadCameraConfig(config_path);

    CameraConfig main_camera = cameras[cameras.size()-1];
    cameras.pop_back();

    for (CameraConfig camera: cameras) {
        // start processing video
        thread t {process_camera, model_path, camera, output_folder, fa};
        t.detach();
    }

    process_camera(model_path, main_camera, output_folder, fa);
}
