#include <iostream>
#include <opencv2/opencv.hpp>
#include "camera.h"
#include "mtcnn.h"
#include <string.h>
#include <chrono>
#include <cstdlib>
#include "utils.h"
#include "face_attr.h"
#include "face_predict.h"
#include <glog/logging.h>
#include <thread>

#define QUIT_KEY 'q'

using namespace std;
using namespace cv;

#ifdef BENCH_EDGE
long  sum_t_infer_embed  = 0;
long  infer_count_embed  = 0;
long  sum_t_infer_age  = 0;
long  infer_count_age  = 0;
#endif

extern int min_score;
extern int child_age_min;
bool compute_coordinate( const cv::Mat im, const std::vector<cv::Point2d> & image_points, const CameraConfig & camera, \
                         cv::Mat & world_coordinate, int age, int frameCount,int faceId);
void read3D_conf();

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
    string cmd = "rm -rf " + output_folder + "";
    system(cmd.c_str());

    cmd = "mkdir -p " + output_folder + "/original";
    int dir_err = system(cmd.c_str());
    if (-1 == dir_err) {
        LOG(ERROR) << "Error creating directory";
        exit(1);
    }
    cmd = "mkdir -p " + output_folder + "/tracker";
    dir_err = system(cmd.c_str());
    if (-1 == dir_err) {
        LOG(ERROR) << "Error creating directory";
        exit(1);
    }    
}

void process_camera(const string model_path, const CameraConfig &camera, string output_folder, const String video_file, bool mainThread) {

    cout << "processing camera: " << camera.identity() << endl;

    prepare_output_folder(camera, output_folder);
    
    MTCNN mm(model_path);

    VideoCapture cap = camera.GetCapture();

    if(video_file!="none")
        cap = VideoCapture(video_file);
    
    if (!cap.isOpened()) {
        cerr << "failed to open camera" << endl;
        return;
    }

    int frameCounter = 0;
    long faceId = 0;
    long thisFace = 0;

    vector<Bbox> detected_bounding_boxes;
    vector<cv::Point2d> image_points;
    Mat frame;

    do {
        detected_bounding_boxes.clear();
        cap >> frame;
        if (!frame.data) {
            if(video_file!="none"){
                std::cout << "video file read over!\n" ;
                exit(0);
            }

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

        if (frameCounter % camera.detection_period == 0)
        {
            ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);

            mm.detect(ncnn_img, detected_bounding_boxes);

            int total = 0;
            // update trackers' bounding boxes and create tracker for a new face
            for(vector<Bbox>::iterator it=detected_bounding_boxes.begin(); it!=detected_bounding_boxes.end();it++) {
                if((*it).exist) {
                    total++;
                    // get face bounding box
                    Bbox box = *it;
                    Rect2d detected_face(Point(box.x1, box.y1),Point(box.x2, box.y2));
                    // calculate score of face
                    Mat face(frame, detected_face);
                    bool front_side =false;
                    double score = GetVarianceOfLaplacianSharpness(face);
                    LOG(INFO) << camera.ip <<" frame[" << frameCounter << "]faceId[" << thisFace 
                              << "], LaplacianSharpness score: " << score;
                    if(score>min_score){
                        image_points.clear();
                        for(int j =0;j<5;j++){
                            cv::Point2d point(box.ppoint[j],box.ppoint[j+5]);
                            image_points.push_back(point);
                        }
                        vector<mx_float> face_vec;
                        imgFormConvert(face,face_vec);
                        int age = 0;
                        if(age_enable){
                            std::vector<float> age_vec;
                            Infer(age_hd,face_vec, age_vec);
                            for(size_t j = 2; j<age_vec.size()-1; j+=2){
                                if(age_vec[j]<age_vec[j+1])
                                    age++;
                            }
                            LOG(INFO) << "target age: " << age;

                        } else 
                            age = 20;
                        
                        cv::Mat world_coordinate;
                        if (age >= child_age_min) {
                            
                            front_side = compute_coordinate(frame, image_points, camera, world_coordinate, age, frameCounter, thisFace);
                            if(front_side) {
                                vector<float> face_embed_vec;
                                Infer(embd_hd,face_vec,face_embed_vec);

                                int new_id;
                                new_id = proc_embd_vec(face_embed_vec, camera, frameCounter, thisFace);

                            }
                            if(front_side){
                                // to do: push the coordinate reid timestamp info into the time series database
                                // (world_coordinate, reid[i], ts_ms)
                            }
                            else {
                                LOG(INFO) << camera.ip <<" frame[" << frameCounter << "]faceId[" << thisFace
                                        << "], pose is skew, don't make face embedding";
                            }
                        } else {
                                LOG(INFO) << camera.ip <<" frame[" << frameCounter << "]faceId[" << thisFace
                                        << "], can't compute coordinate for child age less than " << child_age_min;
                        }
                    } else {
                        LOG(WARNING) << camera.ip <<" frame[" << frameCounter << "]faceId[" << thisFace 
                                     << "], video frame is blur";
                    }
                }
            }
        }

        frameCounter++;
        google::FlushLogFiles(google::GLOG_INFO);

    } while (true);
    // } while (QUIT_KEY != waitKey(1));
}

int main(int argc, char* argv[]) {

    google::InitGoogleLogging("detect_and_infer");
    FLAGS_log_dir = "./";
//    FLAGS_logtostderr = true;
    read3D_conf();
    LoadMxModelConf();
    const String keys =
        "{help h usage ? |                         | print this message   }"
        "{model        |../models/ncnn             | path to mtcnn model  }"
        "{config       |config.toml                | camera config        }"
        "{output       |output                     | output folder        }"
        "{video-file   |none                       | use video file instead of camera stream }"
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
    String video_file = parser.get<String>("video-file");
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
        thread t {process_camera, model_path, camera, output_folder, video_file, false};
        t.detach();
 
    }

    process_camera(model_path,demo,output_folder,video_file,true);
/*
    while(true) {
        std::this_thread::sleep_for(chrono::seconds(1));
    }
*/
}
