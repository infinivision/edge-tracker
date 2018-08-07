#include <iostream>
#include <opencv2/opencv.hpp>
//#include <opencv2/tracking.hpp>
#include "camera.h"
#include "mtcnn.h"
#include <string.h>
#include <chrono>
#include <cstdlib>
#include "tracker.hpp" // use optimised tracker instead of OpenCV version of KCF tracker
#include "staple_tracker.hpp" // staple trakcer
#include "utils.h"
#include "face_attr.h"
#include "face_align.h"
#include "face_predict.h"
#include <glog/logging.h>
#include <thread>

#define QUIT_KEY 'q'

using namespace std;
using namespace cv;
extern int min_score;
void compute_coordinate( const cv::Mat im, const std::vector<cv::Point2d> & image_points, const CameraConfig & camera, int age, int sex);
void read3D_conf();

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

// prepare (clean output folder), output_folder argument will be changed!
void prepare_output_folder(const CameraConfig &camera, string &output_folder) {
    output_folder += "/" + camera.identity();

    string cmd = "mkdir -p " + output_folder + "/original";
    int dir_err = system(cmd.c_str());
    if (-1 == dir_err) {
        LOG(ERROR) << "Error creating directory";
        exit(1);
    }

    cmd = "rm -f " + output_folder + "/*";
    system(cmd.c_str());
    cmd = "rm -f " + output_folder + "/original/*";
    system(cmd.c_str());
}

void process_camera(const string model_path, const CameraConfig &camera, string output_folder, bool mainThread) {

    cout << "processing camera: " << camera.identity() << endl;

    prepare_output_folder(camera, output_folder);
    
    MTCNN mm(model_path);
    FaceAttr fa;
    fa.Load();
    FaceAlign align;

    VideoCapture cap = camera.GetCapture();
    if (!cap.isOpened()) {
        cerr << "failed to open camera" << endl;
        return;
    }

    int frameCounter = 0;
    long faceId = 0;
    long thisFace = 0;
    struct timeval  tv1,tv2;
    struct timezone tz1,tz2;

    vector<Bbox> detected_bounding_boxes;
    Rect2d roi;
    vector<STAPLE_TRACKER *>  trackers;
    vector<Rect2d> tracker_boxes;
    // selected_faces[i] on frame[i] is a selected face
    vector<Mat> selected_frames;
    vector<Bbox> selected_faces;
    vector<double> scores; // scores[i] is the face score of selected_faces[i]
    Mat frame;

    FileStorage fs;
    fs.open("staple.yaml", FileStorage::READ);
    staple_cfg staple_cfg;
    staple_cfg.read(fs.root());

    // namedWindow("window", WINDOW_NORMAL);
    std::vector<cv::Point2d> image_points;
    do {
        detected_bounding_boxes.clear();
        bool enable_detection = false;
        cap >> frame;
        if (!frame.data) {
            cerr << "Capture video failed" << endl;
            continue;
        }

        Mat show_frame = frame.clone();

        string log = "frame #" + to_string(frameCounter) + ", tracking faces: ";
        // update trackers
        for (int i = 0; i < trackers.size(); i++) {
            STAPLE_TRACKER *tracker = trackers[i];
            tracker_boxes[i] = tracker->tracker_staple_update(frame);
            tracker->tracker_staple_train(frame,false);
            log += "#" + to_string(trackers[i]->id) + " ";
        }
        LOG(INFO) << log;
        
        if (frameCounter % camera.detection_period == 0)
        {
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
                    Rect2d detected_face(Point(box.x1, box.y1),Point(box.x2, box.y2));

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
                        STAPLE_TRACKER * tracker = new STAPLE_TRACKER(staple_cfg);
                        tracker->tracker_staple_initialize(frame,detected_face);
                        tracker->tracker_staple_train(frame,true);
                        tracker->id = faceId;
                        trackers.push_back(tracker);
                        tracker_boxes.push_back(detected_face);
                        LOG(INFO) << "start tracking face #" << tracker->id;
                        thisFace = faceId;
                        faceId++;
                    } else {
                        // update tracker's bounding box
                        STAPLE_TRACKER * tracker = trackers[i];
                        long id_ = tracker->id;
                        delete tracker;
                        tracker = new STAPLE_TRACKER(staple_cfg);
                        tracker->id = id_;
                        thisFace = id_;
                        tracker->tracker_staple_initialize(frame,detected_face);
                        tracker->tracker_staple_train(frame,true);
                        trackers[i] = tracker;
                    }

                    // calculate score of face
                    Mat face(frame, detected_face);                    
                    double score = fa.GetVarianceOfLaplacianSharpness(face);
                    LOG(INFO) << "ip[" << camera.ip <<"] dlib score: " << score 
                                       << ", frame " << frameCounter << ",faceId: " << thisFace;
                    std::vector<mx_float> face_vec;
                    std::vector<float> face_embed_vec;
                    imgFormConvert(face,face_vec);
                    Infer(embd_hd,face_vec,face_embed_vec);
                    PrintOutputResult(face_embed_vec);
                    if(score>min_score){
                        image_points.clear();
                        for(int i =0;i<5;i++){
                            cv::Point2d point(box.ppoint[i],box.ppoint[i+5]);
                            image_points.push_back(point);
                        }
                        compute_coordinate(frame, image_points, camera, 20, 1);
                    }

                    // draw detected face
                    if(mainThread){
                        rectangle( show_frame, detected_face, Scalar( 255, 0, 0 ), 2, 1 );
                        // show face id
                        Point middleHighPoint = Point(detected_face.x+detected_face.width/2, detected_face.y);
                        putText(show_frame, to_string(thisFace), middleHighPoint, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
                        for(auto point: image_points){
                            drawMarker(show_frame, point,  cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 10, 1);
                        }
                    }
                    string output = output_folder + "/" + to_string(thisFace) + "_" + to_string(frameCounter) + ".jpg";             
                    imwrite(output,show_frame);
                    output = output_folder + "/original/"+to_string(thisFace) + "_" + to_string(frameCounter) + ".jpg";
                    imwrite(output,frame);
                }
            }
            LOG(INFO) << "trackers size after decect: " << trackers.size();
            LOG(INFO) << "detected " << total << " Persons. time eclipsed: " <<  getElapse(&tv1, &tv2) << " ms";
            if(total == 0) {
                string output = output_folder + "/original/"+ "none" + "_" + to_string(frameCounter) + ".jpg";             
                imwrite(output,frame);
            }

            // clean up trackers if the tracker doesn't follow a face
            for (unsigned i=0; i < trackers.size(); i++) {
                STAPLE_TRACKER *tracker = trackers[i];
                Rect2d tracker_box = tracker_boxes[i];
                bool isFace = false;
                for (vector<Bbox>::iterator it=detected_bounding_boxes.begin(); it!=detected_bounding_boxes.end();it++) {
                    if ((*it).exist) {
                        Bbox box = *it;

                        Rect2d detected_face(Point(box.x1, box.y1),Point(box.x2, box.y2));
                        if (overlap(detected_face, tracker_boxes[i])) {
                            isFace = true;                                                          
                            break;
                        }
                    }
                }
                if (!isFace) 
                {
                    LOG(INFO) << "stop tracking face #" << tracker->id;
                    delete tracker;
                    trackers.erase(trackers.begin() + i);
                    tracker_boxes.erase(tracker_boxes.begin() + i);
                    i--;
                }
            }
        }

        frameCounter++;

        if(enable_detection && mainThread) {
            Mat small_show_frame;
            if(show_frame.cols>1500)
                resize(show_frame,small_show_frame,frame.size()/2);
            else
                small_show_frame = show_frame;
            
            LOG(INFO) << "camera ip: "<< camera.ip;
            imshow("window" + camera.ip , small_show_frame);
            if(QUIT_KEY == waitKey(1)) break;
        }

        google::FlushLogFiles(google::GLOG_INFO);

    } while (true);
    // } while (QUIT_KEY != waitKey(1));
}

int main(int argc, char* argv[]) {

    google::InitGoogleLogging("multi-camera-trakcing");
    FLAGS_log_dir = "./";
//    FLAGS_logtostderr = true;
    read3D_conf();
    LoadMxModelConf();
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

    vector<CameraConfig> cameras = LoadCameraConfig(config_path);

    // LOG(INFO) << "detection period: " << camera.detection_period;

    CameraConfig demo = cameras[cameras.size()-1];
    cameras.pop_back();

    for (CameraConfig camera: cameras) {
        // start processing video
        thread t {process_camera, model_path, camera, output_folder,false};
        t.detach();
 
    }

    process_camera(model_path,demo,output_folder,true);
/*
    while(true) {
        std::this_thread::sleep_for(chrono::seconds(1));
    }
*/
}
